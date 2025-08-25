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
def set_model_and_probs(args, k):
    model_modules = {
        'AE': AE,
        'PPPF-AE': PPPF_AE
    }
    if not args.model in model_modules:
        raise ValueError(f"Unknown model type: {args.model}")
    # Set model type
    model = model_modules[args.model]
    ae = model.AE(K=args.K, k=k, d=args.d, L=args.L).to(args.device)
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

def main():
    args = parser.parse_args()
    # Set model parameters
    N = args.N
    N0 = args.N0
    K = args.K
    S = N * args.ALPHA // K
    k = K // args.ALPHA

    # Print training session info.
    print(f"Processing on device (gpu/cpu): {args.device}")
    print("Training session started...")
    if args.model == 'AE':
        print(f"Model: Autoencoder")
    elif args.model == 'PPPF-AE':
        print(f"Model: PointNet++ FoldingNet Autoencoder")

    print(f"Point cloud resolution(N): {N}, Patch size(K): {K}, Number of patches(S): {S}, Bottleneck size(d): {args.d}, Quantization levels(L): {args.L}")

    # CREATE MODEL SAVE PATH
    if not os.path.exists(args.model_save_folder):
        os.makedirs(args.model_save_folder)

    files = np.array(glob(args.train_glob, recursive=True))
    points = pn_kit.read_point_clouds(files)

    print(f'Point train samples: {points.shape}, corrdinate range: [{points.min()}, {points.max()}]')

    # PARSE TO DATASET
    points_train_tensor = torch.Tensor(points)
    torch_dataset = Data.TensorDataset(points_train_tensor, points_train_tensor)

    use_cuda = str(args.device) == 'cuda' and torch.cuda.is_available()
    loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 4,
        pin_memory=use_cuda
    )

    # Set model type, loss criterion
    ae, prob, criterion = set_model_and_probs(args, k)
    ae = ae.to(args.device)
    prob = prob.to(args.device)
    criterion = criterion.to(args.device) if hasattr(criterion, 'to') else criterion

    # Create optimizer
    optimizer = torch.optim.Adam(itertools.chain(ae.parameters(), prob.parameters()), lr=args.lr)

    # Load saved model if exists and not resetting
    start_step = 0
    total_epochs = args.max_steps // args.step_window

    # Check if checkpoints exist (if not resetting - resume training)
    if not args.reset:
        start_step = load_checkpoints(ae, prob, optimizer, args.model_save_folder)
        # print starting epoch number out of total epochs
        print(f"Starting from step {start_step}, approx epoch {start_step // total_epochs*100}% "
              f"out of {total_epochs} total epochs.")
    elif args.reset:
        print("Resetting training: starting from scratch.")


    fbpps, bpps, losses = [], [], []
    global_step = start_step

    total_steps = args.max_steps
    pbar = tqdm(total=total_steps, initial=global_step, desc="Train", unit="step")


    scaler = torch.cuda.amp.GradScaler() if use_cuda else None
    for epoch in range(9999):
        for step, (batch_x, batch_x) in enumerate(loader):
            if global_step > args.max_steps:
                break

            B = batch_x.shape[0]
            batch_x = batch_x.to(args.device)

            batch_x, center, longest = pn_kit.normalize(batch_x, margin=0.01)
            optimizer.zero_grad()

            if use_cuda:
                autocast_ctx = torch.cuda.amp.autocast()
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()
            with autocast_ctx:
                sampled_xyz = pn_kit.index_points(batch_x, pn_kit.farthest_point_sample_batch(batch_x, S))
                octree_codes, sampled_bits = pn_kit.encode_sampled_np(sampled_xyz.detach().cpu().numpy(), scale=1, N=N, min_bpp=pn_kit.OCTREE_BPP_DICT[K])
                rec_sampled_xyz = pn_kit.decode_sampled_np(octree_codes, scale=1)
                rec_sampled_xyz = np.array(rec_sampled_xyz)
                rec_sampled_xyz = torch.Tensor(rec_sampled_xyz).to(args.device)

                dist, group_idx, grouped_xyz = knn_points(rec_sampled_xyz, batch_x, K=K, return_nn=True)
                grouped_xyz -= rec_sampled_xyz.view(B, S, 1, 3)
                x_patches_orig = grouped_xyz.view(B*S, K, 3)

                x_patches = x_patches_orig * ((N / N0) ** (1/3))
                patches_pred, bottleneck, latent_quantized = ae.forward(x_patches)

                patches_pred = patches_pred / ((N / N0) ** (1/3))

                pmf = prob(rec_sampled_xyz)
                sym= (latent_quantized.view(B, S, args.d) + args.L // 2).long()
                sym = sym.clamp(0, args.L - 1)

                feature_bits = pn_kit.estimate_bits_from_pmf(pmf=pmf, sym=sym)

                bpp = (sampled_bits + feature_bits) / B / N
                fbpp = feature_bits / B / N

                pc_pred = (patches_pred.view(B, S, -1, 3) + rec_sampled_xyz.view(B, S, 1, 3)).reshape(B, -1, 3)
                pc_target = batch_x
                if global_step < args.rate_loss_enable_step:
                    loss = criterion(pc_pred, pc_target, fbpp, λ=0)
                else:
                    loss = criterion(pc_pred, pc_target, fbpp, λ=args.lamda)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            global_step += 1

            pbar.set_postfix({"loss": loss.item()}, refresh=False)
            pbar.update(1)

            # PRINT
            losses.append(loss.item())
            fbpps.append(fbpp.item())
            bpps.append(bpp.item())
            if global_step % args.step_window == 0:
                print(f'Epoch:{epoch} | Step:{global_step} | Feature bpp:{round(np.array(fbpps).mean(), 5)} | Bpp:{round(np.array(bpps).mean(), 5)} | Loss:{round(np.array(losses).mean(), 5)}')
                losses, fbpps, bpps = [], [], []
                # Save in-progress model for resume
                dump_checkpoints(ae, prob, optimizer, args.model_save_folder, global_step)

            # LEARNING RATE DECAY
            if global_step % args.lr_decay_steps == 0:
                args.lr = args.lr * args.lr_decay
                for g in optimizer.param_groups:
                    g['lr'] = args.lr
                print(f'Learning rate decay triggered at step {global_step}, LR is setting to{args.lr}.')
            
            if global_step > args.max_steps:
                break

        if global_step > args.max_steps:
            break

    pbar.close()
    # Save the final model as generic model file at the end of training
    dump_checkpoints(ae, prob, optimizer, args.model_save_folder)

if __name__ == "__main__":
    main()

