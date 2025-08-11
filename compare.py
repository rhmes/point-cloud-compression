import open3d as o3d
import numpy as np
import pandas as pd
import os
import argparse

from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='./data/ModelNet40_pc_01_8192p/', help='Directory containing original PLY files')
parser.add_argument('--recon_dir', type=str, default='./data/ModelNet40_K256_decompressed_ply', help='Directory containing reconstructed (decompressed) PLY files')
parser.add_argument('--csv_path', type=str, default='./eval/ModelNet40_K256.csv', help='CSV file containing evaluation metrics')
args = parser.parse_args()

# ---- Thresholds ----
THRESHOLDS = {
    'p2pointPSNR': {'high': 38, 'medium': 30},
    'p2planePSNR': {'high': 40, 'medium': 32},
    'bpp': {'low': 0.4, 'high': 1.2},
    'point_preservation': {'high': 0.95, 'medium': 0.85},
    'chamfer_distance': {'low': 0.0008, 'medium': 0.002}
}

# Function to classify metrics based on thresholds
def classify_metric(value, metric):
    if metric in ['p2pointPSNR', 'p2planePSNR']:
        if value >= THRESHOLDS[metric]['high']:
            return 'High'
        elif value >= THRESHOLDS[metric]['medium']:
            return 'Medium'
        else:
            return 'Low'
    elif metric == 'bpp':
        if value < THRESHOLDS[metric]['low']:
            return 'High'
        elif value < THRESHOLDS[metric]['high']:
            return 'Medium'
        else:
            return 'Low'
    elif metric == 'point_preservation':
        if value >= THRESHOLDS[metric]['high']:
            return 'High'
        elif value >= THRESHOLDS[metric]['medium']:
            return 'Medium'
        else:
            return 'Low'
    elif metric == 'chamfer_distance':
        if value <= THRESHOLDS[metric]['low']:
            return 'High'
        elif value <= THRESHOLDS[metric]['medium']:
            return 'Medium'
        else:
            return 'Low'
    return 'Unknown'

# --- Load metrics ---
df = pd.read_csv(args.csv_path)
df['id'] = df.index

# --- Overall metrics ---
def overall_metrics():
    # --- Calculate global averages ---
    global_avg = {
        'p2pointPSNR': df['p2pointPSNR'].mean(),
        'p2planePSNR': df['p2planePSNR'].mean(),
        'chamfer_distance': df['chamfer_distance'].mean(),
        'bpp': df['bpp'].mean()
    }
    # --- Classify metrics ---
    metric_class = {}
    for metric in ['p2pointPSNR', 'p2planePSNR', 'bpp', 'point_preservation', 'chamfer_distance']:
        if metric == 'point_preservation':
            df['point_preservation'] = df['n_points_output'] / df['n_points_input']
            global_avg['point_preservation'] = df['point_preservation'].mean()
        # df[f'{metric}_class'] = df[metric].apply(lambda x: classify_metric(x, metric))  
        metric_class[metric] = classify_metric(global_avg[metric], metric)
        # df[f'{metric}_class'] = df[f'{metric}_class'].astype('category')
    # --- Print global averages ---
    print("Global Averages:")
    for metric, value in global_avg.items():
        print(f"{metric}: {value:.2f} ({metric_class[metric]})")
    # --- Pairwise Plots ---
    num_metrics = len(df.columns) - 2  # Exclude 'id' and 'filename'
    fig1, axs1 = plt.subplots(num_metrics, num_metrics, figsize=(15, 15))
    for i, metric_x in enumerate(df.columns[2:]):
        for j, metric_y in enumerate(df.columns[2:]):
            ax = axs1[i, j]
            if i == j:
                ax.text(0.5, 0.5, metric_x, fontsize=12, ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                if metric_x == 'bpp':
                    ax.scatter(df[metric_y], df[metric_x], alpha=0.5, s=10, color='orange')
                else:
                    ax.scatter(df[metric_y], df[metric_x], alpha=0.5, s=10)
            if i == num_metrics - 1:
                ax.set_xlabel(metric_y)
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(metric_x)
            else:
                ax.set_ylabel("")
    plt.suptitle('Pairwise Plots of Evaluation Metrics', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    # plt.savefig(os.path.join(args.outdir, 'metrics_pairplot.png'))
    # plt.close()
    plt.show()

# --- Visualizer Class ---
class PointCloudVisualizer:
    def __init__(self, df, input_dir, recon_dir):
        self.df = df
        self.input_dir = input_dir
        self.recon_dir = recon_dir
        self.index = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.pcds = [o3d.geometry.PointCloud(), o3d.geometry.PointCloud()]
        self.running = True

    def load_point_clouds(self):
        row = self.df.iloc[self.index]
        orig_path = glob(os.path.join(self.input_dir, '**', 'test', row['filename']), recursive=True)
        recon_path = os.path.join(self.recon_dir, row['filename'])

        if len(orig_path) == 0 or not os.path.exists(recon_path):
            print(f"Missing file for: {row['filename']}")
            return None, None

        pcd_orig = o3d.io.read_point_cloud(orig_path[0])
        pcd_recon = o3d.io.read_point_cloud(recon_path)

        pcd_orig.paint_uniform_color([0.2, 0.2, 1.0])
        pcd_recon.paint_uniform_color([1.0, 0.0, 0.0])
        pcd_recon.translate((0.3, 0, 0))

        return pcd_orig, pcd_recon

    def update_scene(self):
        self.vis.clear_geometries()
        pcd1, pcd2 = self.load_point_clouds()
        if pcd1 and pcd2:
            self.vis.add_geometry(pcd1)
            self.vis.add_geometry(pcd2)

            row = self.df.iloc[self.index]
            print(f"\n🟦 [{self.index + 1}/{len(self.df)}] {row['filename']}")
            print(f"📏 p2pointPSNR = {row['p2pointPSNR']:.2f}, p2planePSNR = {row['p2planePSNR']:.2f}")
            print(f"📉 Chamfer Distance = {row['chamfer_distance']:.6f}, Bitrate = {row['bpp']:.4f}")
            print("📊 Avg → PSNR(p2point): {:.2f}, bpp: {:.4f}, Chamfer: {:.6f}".format(
                self.df['p2pointPSNR'][:self.index+1].mean(),
                self.df['bpp'][:self.index+1].mean(),
                self.df['chamfer_distance'][:self.index+1].mean()
            ))

    def next_file(self, vis):
        self.index = (self.index + 1) % len(self.df)
        self.update_scene()

    def prev_file(self, vis):
        self.index = (self.index - 1) % len(self.df)
        self.update_scene()

    def run(self):
        self.vis.create_window(window_name='Point Cloud Compression Evaluation')
        self.update_scene()
        self.vis.register_key_callback(ord("D"), self.next_file)  # 'D' for next
        self.vis.register_key_callback(ord("A"), self.prev_file)  # 'A' for previous
        print("⬅️ Press [A] for previous, [D] for next, [ESC] to quit.")
        self.vis.run()
        self.vis.destroy_window()

# --- Run ---
overall_metrics() # Calculate and display overall metrics
visualizer = PointCloudVisualizer(df, args.input_dir, args.recon_dir)
visualizer.run()