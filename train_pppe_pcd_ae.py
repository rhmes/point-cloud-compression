import os
import argparse
import numpy as np
import torch
import torch.utils.data as Data
from glob import glob
from tqdm import tqdm
import contextlib

import pn_kit
import pppe_pcd_ae as AE  # AE: PointNet++ Encoder + PCN Decoder
from pppe_pcd_ae import estimate_bits_per_point_conditional 

torch.cuda.manual_seed(11)
torch.manual_seed(11)
np.random.seed(11)


parser = argparse.ArgumentParser(
    prog='train_p1.py',
    description='Train autoencoder (PointNet++ + PCN) with conditional prob model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--train_glob', default='./data/ModelNet40_pc_01_8192p/**/train/*.ply')
parser.add_argument('--model_save_folder', default='./model/P1/')
parser.add_argument('--N', type=int, default=8192, help='Point cloud resolution.')
parser.add_argument('--K', type=int, default=256, help='Latent space dimension.')
parser.add_argument('--L', type=int, default=7, help='Quantization level.')
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
    # ae = AE.EnhancedPointCloudAE(latent_dim=args.K, L=args.L, npoints=args.N).to(args.device)   # PointNet++ + PCN
    # prob = AE.ConditionalProbabilityModel(latent_dim=args.K).to(args.device)
    ae = AE.PointCloudAE(latent_dim=args.K, latent_bins=args.L, npoints=args.N).to(args.device)   # PointNet++ + PCN
    prob = ae.prob
    criterion = AE.get_loss("chamfer").to(args.device)
    return ae, prob, criterion

# ---------------------------------------------------
# Checkpoints
# ---------------------------------------------------
def find_latest_checkpoint(folder, file_prefix, best=False):
    path_list = []
    for prefix in file_prefix:
        suffix = 'best' if best else 'latest'
        path_list.append(os.path.join(folder, f"{prefix}_{suffix}.pkl"))
    return path_list

def load_checkpoints(ae, prob, optimizer, folder, best=False):
    if (best):
        print("Loading best loss checkpoints")
    # Get latest checkpoint paths
    file_prefix = ['ae', 'prob', 'optimizer', 'global']
    ae_path, prob_path, opt_path, step_path = find_latest_checkpoint(folder, file_prefix, best=best)
    # Load autoencoder weights
    if os.path.exists(ae_path):
        ae.load_state_dict(torch.load(ae_path))
        print(f"Loaded AE from {ae_path}")
    # Load probability model
    if os.path.exists(prob_path):
        prob.load_state_dict(torch.load(prob_path))
        print(f"Loaded Prob model from {prob_path}")
    # Load optimizer state
    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path))
        print(f"Loaded optimizer from {opt_path}")
    # Restore checkpoint step
    start_step = torch.load(step_path) + 1 if (os.path.exists(step_path)) else 0
    print(f"Resuming from step {start_step}")
    return start_step

def dump_checkpoints(ae, prob, optimizer, folder, global_step, best=False):
    suffix = 'best' if best else 'latest'
    torch.save(ae.state_dict(), os.path.join(folder, f'ae_{suffix}.pkl'))
    torch.save(prob.state_dict(), os.path.join(folder, f'prob_{suffix}.pkl'))
    torch.save(optimizer.state_dict(), os.path.join(folder, f'optimizer_{suffix}.pkl'))
    torch.save(global_step, os.path.join(folder, f'global_{suffix}.pkl'))

# ---------------------------------------------------
# Data Loader
# ---------------------------------------------------
def build_dataloader(args, points):
    points_tensor = torch.Tensor(points)
    dataset = Data.TensorDataset(points_tensor, points_tensor)
    loader = Data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4,
                             pin_memory=(args.device == 'cuda'))
    return loader

# # ---------------------------------------------------
# # Conditional FBPP Calculation
# # ---------------------------------------------------
# def estimate_bits_per_point_conditional(input, latent_quantized, prob_model):
#     """
#     Estimate bits-per-point (fbpp) using ConditionalProbabilityModel.
#     Args:
#         input: (B, C, N) tensor, input features
#         latent_quantized: (B, C, N) tensor, quantized latent
#         prob_model: ConditionalProbabilityModel
#     Returns:
#         fbpp: scalar tensor
#     """
#     # Get mean and scale from probability model
#     mean, scale, pmf = prob_model(input)  # (B, C, N)
#     # Clamp scale for numerical stability
#     scale = scale.clamp(min=1e-6)
#     # # Estimate feature bits from PMF
#     # latent_quantized = latent_quantized.long()

#     # B, N, C = input.shape
#     # # Convert [B, N] to [B, N, 1]
#     # if latent_quantized.dim() == 2:
#     #     latent_quantized = latent_quantized.unsqueeze(-1)

#     # latent_quantized = latent_quantized.clamp(min=0, max=latent_quantized.size(1) - 1)
#     # feature_bits = pn_kit.estimate_bits_from_pmf(pmf, latent_quantized)

#     # Gaussian likelihood (log prob in nats)
#     mean = mean.permute(0, 2, 1)
#     scale = scale.permute(0, 2, 1)
#     log_probs = -0.5 * ((input - mean) / scale) ** 2 - torch.log(scale * (2 * torch.pi) ** 0.5)
#     # Convert nats to bits
#     bits = -log_probs / torch.log(torch.tensor(2.0, device=input.device, dtype=input.dtype))
#     # Add feature bits
    
#     # # bits = bits + feature_bits
#     # feature_bits = feature_bits/(B * N)
#     # fbpp = bits.mean() + feature_bits

#     # Mean bits-per-point
#     fbpp = bits.mean() 
#     return fbpp


def compute_dataset_norm(loader):
    all_points = []
    for batch_x, _ in loader:
        all_points.append(batch_x)  # (B, N, 3)
    all_points = torch.cat(all_points, dim=0)  # (Total, N, 3)

    # Flatten to (num_points, 3)
    flat = all_points.reshape(-1, 3)

    center = flat.mean(dim=0)                     # dataset mean
    diffs = flat - center
    longest = diffs.norm(dim=1).max()             # farthest point from center

    return center, longest
def normalize_with_dataset_stats(batch_x, center, longest, margin=0.01):
    # subtract dataset mean
    batch_x = batch_x - center
    # scale by dataset radius
    batch_x = batch_x / (longest + margin)
    return batch_x

# ---------------------------------------------------
# Training Loop
# ---------------------------------------------------
def train_one_epoch(loader, ae, prob, criterion, optimizer, scheduler, scaler, args, epoch, global_step, pbar,
                    λ=1.0, grad_clip=1.0):
    ae.train()
    # prob.train()
    losses, dists, rates = [], [], []

    if not hasattr(train_one_epoch, "best_loss"):
        train_one_epoch.best_loss = float('inf')
        train_one_epoch.best_step = -1    

    device = args.device
    use_cuda = device == "cuda" and torch.cuda.is_available()

    for step, (batch_x, _) in enumerate(loader):
        if global_step > args.max_steps:
            break

        batch_x = batch_x.to(device)
        # batch_x, center, longest = pn_kit.normalize(batch_x, margin=0.01)
        # batch_x = normalize_with_dataset_stats(batch_x, args.center, args.longest)

        optimizer.zero_grad()
        autocast_ctx = torch.cuda.amp.autocast() if use_cuda else contextlib.nullcontext()

        # warmup for λ
        λ_eff = λ * min(1.0, global_step / max(1, args.warmup_steps))

        with autocast_ctx:
            # forward pass
            coarse, recon, feats, y_cont = ae(batch_x)   
            fbpp = estimate_bits_per_point_conditional(y_cont, feats, prob_model=prob)  
            # recon_norm, _, _ = normalize_with_dataset_stats(recon, margin=0.01)
            recon_norm = recon
            loss, dist, rate = criterion(recon_norm.float(), batch_x.float(), fbpp, λ_eff)

        # anomaly check
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[Warning] Loss anomaly detected: {loss.item():.4f}")
            continue

        # backward + optimizer step
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(ae.parameters()) + list(prob.parameters()), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(ae.parameters()) + list(prob.parameters()), grad_clip)
            optimizer.step()

        global_step += 1
        losses.append(loss.item())
        dists.append(dist.item())
        rates.append(rate.item())

        # progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}", 
            "dist": f"{dist.item():.4f}", 
            "rate": f"{rate.item():.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
        })
        pbar.update(1)

        # logging + checkpointing
        if global_step % args.step_window == 0:
            avg_loss, avg_dist, avg_rate = np.mean(losses), np.mean(dists), np.mean(rates)
            if avg_loss < train_one_epoch.best_loss:
                train_one_epoch.best_loss = avg_loss
                train_one_epoch.best_step = global_step
                dump_checkpoints(ae, prob, optimizer, args.model_save_folder, train_one_epoch.best_step, best=True)
            print(f"[Epoch {epoch}] Step {global_step} | "
                  f"Loss: {avg_loss:.5f} | Dist: {avg_dist:.5f} | Rate: {avg_rate:.5f}")
            losses, dists, rates = [], [], []
            dump_checkpoints(ae, prob, optimizer, args.model_save_folder, global_step, best=False)

    scheduler.step()

    return global_step


def main():
    args = parser.parse_args()
    print(f"Training PointNet++ + PCN + ProbModel on {args.device}")

    os.makedirs(args.model_save_folder, exist_ok=True)

    npy_path = os.path.join(os.path.dirname(args.train_glob.split("*")[0]), "train.npy")
    if os.path.exists(npy_path):
        print(f"Loading cached point clouds from {npy_path}")
        points = np.load(npy_path)
    else:
        files = np.array(glob(args.train_glob, recursive=True))
        points = pn_kit.read_point_clouds(files)
        print(f"Loaded {points.shape} points")
    loader = build_dataloader(args, points)
    ae, prob, criterion = set_model_and_loss(args)

    # optimizer = torch.optim.Adam(
    #     list(ae.parameters()) + list(prob.parameters()), lr=args.lr
    # )
    optimizer = torch.optim.Adam(
        list(ae.parameters()), lr=args.lr
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    scaler = torch.cuda.amp.GradScaler() if args.device == 'cuda' else None

    # Compute once
    args.center, args.longest = compute_dataset_norm(loader)

    # Save for reuse (so val/test use same stats!)
    torch.save({"center": args.center, "longest": args.longest}, 
               os.path.join(args.model_save_folder, "dataset_norm.pt"))


    if not args.reset:
        start_step = load_checkpoints(ae, prob, optimizer,args.model_save_folder)
    else:
        start_step = 0
        print("Starting training from scratch.")

    total_steps = args.max_steps
    pbar = tqdm(total=total_steps, initial=start_step, desc="Training", unit="step")

    global_step = start_step
    for epoch in range(9999):
        global_step = train_one_epoch(loader, ae, prob, criterion, optimizer, scheduler, scaler, args, epoch, global_step, pbar)
        if global_step > args.max_steps:
            break

    pbar.close()
    dump_checkpoints(ae, prob, optimizer, args.model_save_folder, global_step, best=False)

if __name__ == "__main__":
    main()
