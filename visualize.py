import argparse
import pandas as pd
import matplotlib.pyplot as plt

import os

# Argument parser for input CSV and output directory
parser = argparse.ArgumentParser(description="Visualize evaluation metrics from eval.py output CSV.")
parser.add_argument('--csv', default='./eval/ModelNet40_K256.csv', help='Path to the evaluation CSV file (output of eval.py)')
parser.add_argument('--outdir', default='./figure', help='Directory to save plots')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.outdir, exist_ok=True)

# Load CSV
df = pd.read_csv(args.csv)

# List of metrics to plot (skip filename column)
metrics = [col for col in df.columns if col != 'filename']


# Plot histograms for each metric using matplotlib
for metric in metrics:
    plt.figure(figsize=(8, 5))
    plt.hist(df[metric].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of {metric}')
    plt.xlabel(metric)
    plt.ylabel('Count')
    plt.tight_layout()
    # plt.savefig(os.path.join(args.outdir, f'{metric}_hist.png'))
    plt.close()

# Error rate: (n_points_input - n_points_output) / n_points_input
if 'n_points_input' in df.columns and 'n_points_output' in df.columns:
    df['error_rate'] = (df['n_points_input'] - df['n_points_output']) / df['n_points_input']
    plt.figure(figsize=(8, 5))
    plt.hist(df['error_rate'].dropna(), bins=30, color='salmon', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Error Rate (Input - Output) / Input')
    plt.xlabel('Error Rate')
    plt.ylabel('Count')
    plt.tight_layout()
    # plt.savefig(os.path.join(args.outdir, 'error_rate_hist.png'))
    plt.close()

    # Scatter plot: input vs output points
    plt.figure(figsize=(7, 7))
    plt.scatter(df['n_points_input'], df['n_points_output'], alpha=0.6, color='purple', edgecolor='k')
    plt.plot([df['n_points_input'].min(), df['n_points_input'].max()],
             [df['n_points_input'].min(), df['n_points_input'].max()],
             'r--', label='Input = Output')
    plt.xlabel('Number of Input Points')
    plt.ylabel('Number of Output Points')
    plt.title('Input vs Output Points')
    plt.legend()
    plt.tight_layout()
    # plt.savefig(os.path.join(args.outdir, 'input_vs_output_points.png'))
    plt.close()

# Pairwise scatter plots (pairplot replacement)
from itertools import combinations
num_metrics = len(metrics)
fig, axes = plt.subplots(num_metrics, num_metrics, figsize=(3*num_metrics, 3*num_metrics))
for i, metric_x in enumerate(metrics):
    for j, metric_y in enumerate(metrics):
        ax = axes[i, j]
        if i == j:
            ax.hist(df[metric_x].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_ylabel('Count')
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
plt.close()

import pandas as pd
import matplotlib.pyplot as plt

# Load CSV

import pandas as pd
import matplotlib.pyplot as plt

# Replace filename with file index
df["file_index"] = range(1, len(df) + 1)

# Plot 1: Bitrate (bpp)
fig1 = plt.figure(figsize=(10, 5))
plt.bar(df["file_index"], df["bpp"], color='steelblue')
plt.xlabel("File Index")
plt.ylabel("Bitrate (bpp)")
plt.title("Bitrate per File")
plt.grid(True)
plt.tight_layout()
plt.savefig("bitrate_per_file.png")

# Plot 2: PSNR (p2point and p2plane)
fig2 = plt.figure(figsize=(10, 5))
plt.plot(df["file_index"], df["p2pointPSNR"], label="p2point PSNR", marker='o')
plt.plot(df["file_index"], df["p2planePSNR"], label="p2plane PSNR", marker='x')
plt.xlabel("File Index")
plt.ylabel("PSNR (dB)")
plt.title("PSNR per File")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("psnr_per_file.png")

# Plot 3: Chamfer Distance
fig3 = plt.figure(figsize=(10, 5))
plt.bar(df["file_index"], df["chamfer_distance"], color='darkorange')
plt.xlabel("File Index")
plt.ylabel("Chamfer Distance")
plt.title("Chamfer Distance per File")
plt.grid(True)
plt.tight_layout()
plt.savefig("chamfer_distance_per_file.png")
plt.show()
print(f"Plots saved to {args.outdir}/")
