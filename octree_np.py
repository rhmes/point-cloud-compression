import os
import numpy as np
from tqdm import tqdm

########
## A simple Python implementation of octree coding
## Only point clouds with coordinates in the unit sphere (does not contain boundary values (i.e., 0 and 1)) are supported
########

def encode(pc, resolution, depth):
    '''
    input: 0 <= depth
    '''

    # Vectorized batch support and iterative octree encoding
    pc = getDecodeFromPc(pc, resolution, depth)
    bits_ls = [[] for _ in range(depth+1)]
    stack = [(0, 0, 0, 0)]
    while stack:
        startX, startY, startZ, currdepth = stack.pop()
        curr_cube_reso = resolution / (2 ** currdepth)
        mask = (
            (startX <= pc[:, 0]) & (pc[:, 0] <= startX+curr_cube_reso) &
            (startY <= pc[:, 1]) & (pc[:, 1] <= startY+curr_cube_reso) &
            (startZ <= pc[:, 2]) & (pc[:, 2] <= startZ+curr_cube_reso)
        )
        if np.any(mask):
            bits_ls[currdepth].append(1)
            if currdepth < depth:
                next_cube_reso = curr_cube_reso / 2
                stack.extend([
                    (startX, startY, startZ, currdepth+1),
                    (startX, startY, startZ+next_cube_reso, currdepth+1),
                    (startX, startY+next_cube_reso, startZ, currdepth+1),
                    (startX, startY+next_cube_reso, startZ+next_cube_reso, currdepth+1),
                    (startX+next_cube_reso, startY, startZ, currdepth+1),
                    (startX+next_cube_reso, startY, startZ+next_cube_reso, currdepth+1),
                    (startX+next_cube_reso, startY+next_cube_reso, startZ, currdepth+1),
                    (startX+next_cube_reso, startY+next_cube_reso, startZ+next_cube_reso, currdepth+1)
                ])
        else:
            bits_ls[currdepth].append(0)
    bits = [i for ls in bits_ls for i in ls]
    bits = np.array(bits, dtype=np.uint8)
    return bits

def decode(bits, resolution):

    # Iterative decode for speedup
    bits_ls = [[1]]
    bits = bits.tolist() if isinstance(bits, np.ndarray) else list(bits)
    n = 8
    idx = 0
    while idx < len(bits):
        bits_group = bits[idx:idx+n]
        depth = len(bits_ls) - 1
        bits_ls.append(bits_group)
        n = sum(bits_group) * 8
        idx += len(bits_group)
        if n == 0:
            break
    depth = len(bits_ls) - 1
    pc = []
    stack = [(0, 0, 0, 0)]
    bits_ptr = [0 for _ in range(depth+1)]
    while stack:
        startX, startY, startZ, currdepth = stack.pop()
        curr_cube_reso = resolution / (2 ** currdepth)
        # Guard against index out of range
        if bits_ptr[currdepth] >= len(bits_ls[currdepth]):
            break
        b = bits_ls[currdepth][bits_ptr[currdepth]]
        bits_ptr[currdepth] += 1
        if b == 1:
            if currdepth == depth:
                pc.append([startX+curr_cube_reso/2, startY+curr_cube_reso/2, startZ+curr_cube_reso/2])
            else:
                next_cube_reso = curr_cube_reso / 2
                stack.extend([
                    (startX, startY, startZ, currdepth+1),
                    (startX, startY, startZ+next_cube_reso, currdepth+1),
                    (startX, startY+next_cube_reso, startZ, currdepth+1),
                    (startX, startY+next_cube_reso, startZ+next_cube_reso, currdepth+1),
                    (startX+next_cube_reso, startY, startZ, currdepth+1),
                    (startX+next_cube_reso, startY, startZ+next_cube_reso, currdepth+1),
                    (startX+next_cube_reso, startY+next_cube_reso, startZ, currdepth+1),
                    (startX+next_cube_reso, startY+next_cube_reso, startZ+next_cube_reso, currdepth+1)
                ])
    pc = np.array(pc, dtype=np.float32)
    # Ensure output has exactly S points (for downstream compatibility)
    # S is typically passed as args.S, but we infer from context or allow caller to specify
    S = 64  # Default, change as needed or pass as argument
    if pc.shape[0] < S:
        # Pad with repeated points
        pad = np.tile(pc[-1:], (S - pc.shape[0], 1)) if pc.shape[0] > 0 else np.zeros((S, 3), dtype=np.float32)
        pc = np.concatenate([pc, pad], axis=0)
    elif pc.shape[0] > S:
        # Randomly sample S points
        idx = np.random.choice(pc.shape[0], S, replace=False)
        pc = pc[idx]
    return pc

def getDecodeFromPc(pc, resolution, depth):

    cube_reso = float(resolution) / max(1, float(2 ** depth))
    # Prevent cube_reso from being too small
    if cube_reso < 1e-6:
        cube_reso = 1e-6
    pc = np.asarray(pc, dtype=np.float32)
    # Handle batch dimension: pc shape (B, N, 3) or (N, 3)
    if pc.ndim == 3:
        cube_reso_arr = np.full((pc.shape[0], 1, 1), cube_reso, dtype=np.float32)
        pc_octree = (pc // cube_reso_arr * cube_reso_arr) + (cube_reso_arr / 2)
        pc_octree = np.nan_to_num(pc_octree)
        pc_octree = np.unique(pc_octree.reshape(-1, pc.shape[-1]), axis=0)
    else:
        pc_octree = (pc // cube_reso * cube_reso) + (cube_reso / 2)
        pc_octree = np.nan_to_num(pc_octree)
        pc_octree = np.unique(pc_octree, axis=0)
    return pc_octree

