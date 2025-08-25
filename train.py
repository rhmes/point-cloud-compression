import os
import argparse
import itertools

import numpy as np
import torch
import torch.utils.data as Data
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points

from glob import glob

import pn_kit
import AE
import PPPF_AE
from tqdm import tqdm
import contextlib

torch.cuda.manual_seed(11)
torch.manual_seed(11)
np.random.seed(11)


parser = argparse.ArgumentParser(
    prog='train_ae.py',
    description='Train autoencoder using point cloud patches',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--train_glob', default='./data/ModelNet40_pc_01_8192p/**/train/*.ply', help='Point clouds glob pattern for training.')
parser.add_argument('--model_save_folder', default='./model/K256/', help='Directory where to save trained models.')
parser.add_argument('--model', default='AE', help='Type of the model (AE or PPPF-AE).')

parser.add_argument('--N', type=int, default=8192, help='Point cloud resolution.')
parser.add_argument('--N0', type=int, default=1024, help='Scale Transformation constant.')
parser.add_argument('--ALPHA', type=int, default=2, help='The factor of patch coverage ratio.')
parser.add_argument('--K', type=int, default=256, help='Number of points in each patch.')
parser.add_argument('--d', type=int, default=16, help='Bottleneck size.')
parser.add_argument('--L', type=int, default=7, help='Quantization Level.')

parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size (must be 1).')
parser.add_argument('--step_window', type=float, default=100, help='Number of steps per window to iterate in epoch.')
parser.add_argument('--lamda', type=float, default=1e-06, help='Lambda for rate-distortion tradeoff.')
parser.add_argument('--rate_loss_enable_step', type=int, default=40000, help='Apply rate-distortion tradeoff at x steps.')
parser.add_argument('--lr_decay', type=float, default=0.1, help='Decays the learning rate to x times the original.')
parser.add_argument('--lr_decay_steps', type=int, default=60000, help='Decays the learning rate every x steps.')
parser.add_argument('--max_steps', type=int, default=80000, help='Train up to this number of steps.')

# Set device for processing (cuda/cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--device', default=device, help='AE Model Device (cpu or cuda)')

parser.add_argument('--reset', action='store_true', help='Reset training and start from scratch (ignore saved model).')

# Set model type, prob, loss criterion
def set_model_and_probs(args):
    model_modules = {
        'AE': AE,
        'PPPF-AE': PPPF_AE
    }
    if not args.model in model_modules:
        raise ValueError(f"Unknown model type: {args.model}")
    # Set model type
    model = model_modules[args.model]
    ae = model.AE(K=args.K, k=args.k, d=args.d, L=args.L).to(args.device)
    prob = model.ConditionalProbabilityModel(args.L, args.d).to(args.device)
    criterion = model.get_loss().to(args.device)
    return ae, prob, criterion

# Find latest checkpoint if available
def find_latest_checkpoint(folder, prefix):
    files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.pkl')]
    if not files:
        return ''
    steps = [int(f.split('step')[-1].split('.pkl')[0]) for f in files if 'step' in f]
    if not steps:
        return ''
    if steps:
        latest_step = max(steps)
    else:
        latest_step = ''
    return os.path.join(folder, f"{prefix}_step{latest_step}.pkl")

# Load checkpoints 
def load_checkpoints(ae, prob, optimizer, model_save_folder):
    start_step = 0
    ae_path = find_latest_checkpoint(model_save_folder, 'ae')
    prob_path = find_latest_checkpoint(model_save_folder, 'prob')
    optimizer_path = find_latest_checkpoint(model_save_folder, 'optimizer')
    step_path = find_latest_checkpoint(model_save_folder, 'global')
    if os.path.exists(ae_path):
        print("Loading checkpoint weights from:", ae_path)
        ae.load_state_dict(torch.load(ae_path))
    if os.path.exists(prob_path):
        print("Loading prob weights from:", prob_path)
        prob.load_state_dict(torch.load(prob_path))
    if os.path.exists(optimizer_path):
        print("Loading optimizer weights from:", optimizer_path)
        optimizer.load_state_dict(torch.load(optimizer_path))
    if os.path.exists(step_path):
        start_step = torch.load(step_path) + 1
        print("Starting step:", start_step)
    return start_step

# Save model checkpoints
def dump_checkpoints(ae, prob, optimizer, model_save_folder, global_step=''):
    torch.save(ae.state_dict(), os.path.join(model_save_folder, f'ae_step{global_step}.pkl'))
    torch.save(prob.state_dict(), os.path.join(model_save_folder, f'prob_step{global_step}.pkl'))
    torch.save(optimizer.state_dict(), os.path.join(model_save_folder, f'optimizer_step{global_step}.pkl'))
    torch.save(global_step, os.path.join(model_save_folder, f'global_step{global_step}.pkl'))

def build_dataloader(args, points):
    points_tensor = torch.Tensor(points)
    dataset = Data.TensorDataset(points_tensor, points_tensor)

    use_cuda = (args.device == 'cuda' and torch.cuda.is_available())
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_cuda
    )
    return loader, use_cuda


def prepare_model_and_optimizer(args):
    ae, prob, criterion = set_model_and_probs(args)
    ae = ae.to(args.device)
    prob = prob.to(args.device)
    if hasattr(criterion, 'to'):
        criterion = criterion.to(args.device)

    optimizer = torch.optim.Adam(
        itertools.chain(ae.parameters(), prob.parameters()), 
        lr=args.lr
    )
    return ae, prob, criterion, optimizer


def resume_or_reset(args, ae, prob, optimizer):
    if not args.reset:
        start_step = load_checkpoints(ae, prob, optimizer, args.model_save_folder)
        print(f"Resuming from step {start_step}")
    else:
        print("Resetting training from scratch.")
        start_step = 0
    return start_step

def train_one_epoch(loader, ae, prob, criterion, optimizer, scaler, args, epoch, global_step, pbar):
    ae.train()
    prob.train()

    fbpps, bpps, losses = [], [], []
    device = args.device
    use_cuda = device == "cuda" and torch.cuda.is_available()

    for step, (batch_x, _) in enumerate(loader):
        if global_step > args.max_steps:
            break

        batch_x = batch_x.to(device)
        B = batch_x.size(0)

        # Normalize input point cloud
        batch_x, center, longest = pn_kit.normalize(batch_x, margin=0.01)

        optimizer.zero_grad()

        autocast_ctx = torch.cuda.amp.autocast() if use_cuda else contextlib.nullcontext()
        with autocast_ctx:
            # Step 1: Downsample point cloud (FPS)
            sampled_xyz = pn_kit.index_points(batch_x, pn_kit.farthest_point_sample_batch(batch_x, args.S))

            # Step 2: Encode/Decode with Octree
            octree_codes, sampled_bits = pn_kit.encode_sampled_np(sampled_xyz.detach().cpu().numpy(), scale=1, N=args.N, min_bpp=pn_kit.OCTREE_BPP_DICT[args.K])
            rec_sampled_xyz = pn_kit.decode_sampled_np(octree_codes, scale=1)
            # Use torch.from_numpy for direct conversion, avoid extra np.array
            if isinstance(rec_sampled_xyz, np.ndarray):
                rec_sampled_xyz = torch.from_numpy(rec_sampled_xyz).to(args.device).float()
            else:
                rec_sampled_xyz = torch.tensor(rec_sampled_xyz, device=args.device, dtype=torch.float32)

            # Step 3: Extract patches
            dist, group_idx, grouped_xyz = knn_points(
                rec_sampled_xyz, batch_x, K=args.K, return_nn=True
            )
            grouped_xyz -= rec_sampled_xyz.view(B, args.S, 1, 3)
            x_patches_orig = grouped_xyz.view(B * args.S, args.K, 3)

            # Step 4: Encode patches with AE
            x_patches = x_patches_orig * ((args.N / args.N0) ** (1/3))
            patches_pred, bottleneck, latent_quantized = ae(x_patches)
            patches_pred = patches_pred / ((args.N / args.N0) ** (1/3))

            # Step 5: Probability model
            pmf = prob(rec_sampled_xyz)
            sym = (latent_quantized.view(B, args.S, args.d) + args.L // 2).long()
            sym = sym.clamp(0, args.L - 1)

            feature_bits = pn_kit.estimate_bits_from_pmf(pmf, sym)

            # Step 6: Rate-Distortion Loss
            bpp = (sampled_bits + feature_bits) / (B * args.N)
            fbpp = feature_bits / (B * args.N)

            pc_pred = (
                patches_pred.view(B, args.S, -1, 3) + rec_sampled_xyz.view(B, args.S, 1, 3)
            ).reshape(B, -1, 3)

            if global_step < args.rate_loss_enable_step:
                loss = criterion(pc_pred, batch_x, fbpp, λ=0)
            else:
                loss = criterion(pc_pred, batch_x, fbpp, λ=args.lamda)

        # Step 7: Backpropagation
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        pbar.set_postfix({"loss": loss.item()})
        pbar.update(1)
        # Step 8: Logging
        global_step += 1
        losses.append(loss.item())
        fbpps.append(fbpp.item())
        bpps.append(bpp.item())

        if global_step % args.step_window == 0:
            print(f"[Epoch {epoch}] Step {global_step} | "
                  f"Feature bpp: {np.mean(fbpps):.5f} | "
                  f"Bpp: {np.mean(bpps):.5f} | "
                  f"Loss: {np.mean(losses):.5f}")
            losses, fbpps, bpps = [], [], []
            dump_checkpoints(ae, prob, optimizer, args.model_save_folder, global_step)

        # Step 9: LR Decay
        if global_step % args.lr_decay_steps == 0:
            args.lr *= args.lr_decay
            for g in optimizer.param_groups:
                g["lr"] = args.lr
            print(f"LR decayed to {args.lr} at step {global_step}")

    return global_step

def main():
    args = parser.parse_args()

    # Derived params
    N, N0, K = args.N, args.N0, args.K
    args.S, args.k = N * args.ALPHA // K, K // args.ALPHA

    # Model info
    print(f"Training {args.model} on {args.device}")
    print(f"N={N}, K={K}, S={args.S}, d={args.d}, L={args.L}")

    # Create model save folder
    os.makedirs(args.model_save_folder, exist_ok=True)

    # Load training data
    files = np.array(glob(args.train_glob, recursive=True))
    points = pn_kit.read_point_clouds(files)
    print(f"Loaded {points.shape} points, range: [{points.min()}, {points.max()}]")

    # Dataset & loader
    loader, use_cuda = build_dataloader(args, points)

    # Model + optimizer
    ae, prob, criterion, optimizer = prepare_model_and_optimizer(args)
    start_step = resume_or_reset(args, ae, prob, optimizer)

    scaler = torch.cuda.amp.GradScaler() if use_cuda else None
    total_steps = args.max_steps
    pbar = tqdm(total=total_steps, initial=start_step, desc="Training", unit="step")

    global_step = start_step
    for epoch in range(9999):
        global_step = train_one_epoch(loader, ae, prob, criterion, optimizer, scaler, args, epoch, global_step, pbar)
        if global_step > args.max_steps:
            break
    pbar.close()
    # Save final checkpoints
    dump_checkpoints(ae, prob, optimizer, args.model_save_folder)

if __name__ == "__main__":
    main()

