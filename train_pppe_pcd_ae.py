import os
import argparse
import itertools
import numpy as np
import torch
import torch.utils.data as Data
from glob import glob
from tqdm import tqdm
import contextlib

import pn_kit
import pppe_pcd_ae as AE  # AE: PointNet++ Encoder + PCN Decoder

torch.cuda.manual_seed(11)
torch.manual_seed(11)
np.random.seed(11)


parser = argparse.ArgumentParser(
    prog='train_p1.py',
    description='Train autoencoder (PointNet++ + PCN)',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--train_glob', default='./data/ModelNet40_pc_01_8192p/**/train/*.ply')
parser.add_argument('--model_save_folder', default='./model/P1/')
parser.add_argument('--N', type=int, default=8192, help='Point cloud resolution.')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--max_steps', type=int, default=80000)
parser.add_argument('--step_window', type=int, default=100)
parser.add_argument('--lr_decay', type=float, default=0.95)
parser.add_argument('--lr_decay_steps', type=int, default=60000)
parser.add_argument('--warmup_steps', type=int, default=5000, help="Number of steps to gradually ramp up λ in RD loss")
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--reset', action='store_true')

# ---------------------------------------------------
# Model + Loss
# ---------------------------------------------------
def set_model_and_loss(args):
    ae = AE.PointCloudAE().to(args.device)   # PointNet++ + PCN
    criterion = AE.get_loss().to(args.device)
    return ae, criterion

# ---------------------------------------------------
# Checkpoints
# ---------------------------------------------------
def find_latest_checkpoint(folder, prefix):
    files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.pkl')]
    if not files:
        return ''
    steps = [int(f.split('step')[-1].split('.pkl')[0]) for f in files if 'step' in f]
    if steps:
        latest_step = max(steps)
        return os.path.join(folder, f"{prefix}_step{latest_step}.pkl")
    return ''

def load_checkpoints(ae, optimizer, folder):
    start_step = 0
    ae_path = find_latest_checkpoint(folder, 'ae')
    opt_path = find_latest_checkpoint(folder, 'optimizer')
    step_path = find_latest_checkpoint(folder, 'global')
    if os.path.exists(ae_path):
        ae.load_state_dict(torch.load(ae_path))
        print(f"Loaded AE from {ae_path}")
    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path))
        print(f"Loaded optimizer from {opt_path}")
    if os.path.exists(step_path):
        start_step = torch.load(step_path) + 1
        print(f"Resuming at step {start_step}")
    return start_step

def dump_checkpoints(ae, optimizer, folder, global_step):
    torch.save(ae.state_dict(), os.path.join(folder, f'ae_step{global_step}.pkl'))
    torch.save(optimizer.state_dict(), os.path.join(folder, f'optimizer_step{global_step}.pkl'))
    torch.save(global_step, os.path.join(folder, f'global_step{global_step}.pkl'))

# ---------------------------------------------------
# Data Loader
# ---------------------------------------------------
def build_dataloader(args, points):
    points_tensor = torch.Tensor(points)
    dataset = Data.TensorDataset(points_tensor, points_tensor)
    loader = Data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4,
                             pin_memory=(args.device == args.device))
    return loader

# ---------------------------------------------------
# FBPP Calculation
# ---------------------------------------------------
def estimate_bits_per_point(latent, prior="gaussian"):
    """
    Estimate bits-per-point (bpp) for latent features using a simple entropy model.
    
    Args:
        latent : (B, N, D) latent tensor from encoder
        prior  : str, probability model ("gaussian" | "laplacian")

    Returns:
        fbpp : scalar tensor (average bits per point)
    """
    B, N, D = latent.shape
    num_points = N

    # Quantization simulation (straight-through estimator)
    z_q = torch.round(latent)

    # Estimate scale parameter per channel
    if prior == "gaussian":
        # σ per channel
        scale = torch.std(z_q, dim=(0, 1)) + 1e-6
        # Prob under Gaussian
        log_probs = -0.5 * ((z_q / scale) ** 2 + torch.log(2 * torch.pi * scale**2))
    elif prior == "laplacian":
        # b per channel
        scale = torch.mean(torch.abs(z_q), dim=(0, 1)) + 1e-6
        # Prob under Laplace
        log_probs = -torch.abs(z_q / scale) - torch.log(2 * scale)
    else:
        raise ValueError("Unsupported prior")

    # Convert log-likelihood to bits
    bits = -log_probs / torch.log(torch.tensor(2.0, device=latent.device))

    # Average over (B, N, D) → per-feature bits
    bpp_features = bits.mean()

    # Normalize to bits-per-point
    fbpp = bpp_features * D / num_points

    return fbpp

# ---------------------------------------------------
# Training Loop
# ---------------------------------------------------
def train_one_epoch(loader, ae, criterion, optimizer, scaler, args, epoch, global_step, pbar,
                    λ=1.0, grad_clip=1.0, anomaly_threshold=50.0):
    ae.train()
    losses, dists, rates = [], [], []

    device = args.device
    use_cuda = device == "cuda" and torch.cuda.is_available()

    for step, (batch_x, _) in enumerate(loader):
        if global_step > args.max_steps:
            break

        batch_x = batch_x.to(device)

        # Normalize input point cloud
        batch_x, center, longest = pn_kit.normalize(batch_x, margin=0.01)

        optimizer.zero_grad()
        autocast_ctx = torch.cuda.amp.autocast() if use_cuda else contextlib.nullcontext()

        # λ warmup (gradually introduce rate term)
        λ_eff = λ * min(1.0, global_step / max(1, args.warmup_steps))

        with autocast_ctx:
            # Forward pass
            recon, latent = ae(batch_x)

            # fbpp calculation
            fbpp = estimate_bits_per_point(latent, prior="gaussian")

            # Ensure float32 for chamfer_distance and knn_points
            recon = recon.float()
            batch_x = batch_x.float()
            # Rate–Distortion loss
            loss, dist, rate = criterion(recon, batch_x, fbpp, λ_eff)

        # Skip anomalous batches
        if (torch.isnan(loss) or torch.isinf(loss) or loss.item() > anomaly_threshold):
            print(f"[Warning] Skipping step {global_step} due to abnormal loss = {loss.item():.4f}")
            global_step += 1
            continue

        # Backpropagation
        if scaler:
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ae.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), grad_clip)
            optimizer.step()

        # Logging
        global_step += 1
        losses.append(loss.item())
        dists.append(dist.item())
        rates.append(rate.item())

        pbar.set_postfix({
            "loss": f"{loss.item():.5f}",
            "dist": f"{dist.item():.5f}",
            "rate": f"{rate.item():.5f}"
        })
        pbar.update(1)

        # Step window logging
        if global_step % args.step_window == 0:
            print(f"[Epoch {epoch}] Step {global_step} | "
                  f"Loss: {np.mean(losses):.5f} | "
                  f"Dist: {np.mean(dists):.5f} | "
                  f"Rate: {np.mean(rates):.5f}")
            losses, dists, rates = [], [], []
            dump_checkpoints(ae, optimizer, args.model_save_folder, global_step)

        # LR decay
        if global_step % args.lr_decay_steps == 0:
            args.lr *= args.lr_decay
            for g in optimizer.param_groups:
                g["lr"] = args.lr
            print(f"LR decayed to {args.lr:.6f} at step {global_step}")

    return global_step

# ---------------------------------------------------
def main():
    args = parser.parse_args()
    print(f"Training PointNet++ + PCN on {args.device}")

    os.makedirs(args.model_save_folder, exist_ok=True)

    files = np.array(glob(args.train_glob, recursive=True))
    points = pn_kit.read_point_clouds(files)
    print(f"Loaded {points.shape} points")

    loader = build_dataloader(args, points)
    ae, criterion = set_model_and_loss(args)

    optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if args.device == 'cuda' else None

    if not args.reset:
        start_step = load_checkpoints(ae, optimizer, args.model_save_folder)
    else:
        start_step = 0
        print("Starting training from scratch.")

    total_steps = args.max_steps
    pbar = tqdm(total=total_steps, initial=start_step, desc="Training", unit="step")

    global_step = start_step
    for epoch in range(9999):
        global_step = train_one_epoch(loader, ae, criterion, optimizer, scaler, args, epoch, global_step, pbar)
        if global_step > args.max_steps:
            break

    pbar.close()
    dump_checkpoints(ae, optimizer, args.model_save_folder, global_step)

if __name__ == "__main__":
    main()
