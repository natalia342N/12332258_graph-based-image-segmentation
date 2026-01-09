import os
from pathlib import Path
import pickle

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import dgl
from torchvision.datasets import OxfordIIITPet
from skimage.segmentation import slic

IMAGE_SIZE = 256          # resize images & masks to 256x256
N_SEGMENTS = 200          # desired number of superpixels
COMPACTNESS = 10.0        # SLIC compactness parameter
TRAIN_FRACTION = 0.9
SEED = 42


def build_superpixel_graph(img_np: np.ndarray, mask_bin: np.ndarray):
    img_float = img_np.astype(np.float32) / 255.0
    H, W, _ = img_float.shape

    segments = slic(
        img_float,
        n_segments=N_SEGMENTS,
        compactness=COMPACTNESS,
        start_label=0,
    )  

    sp_unique, sp_inv = np.unique(segments, return_inverse=True)
    sp_map = sp_inv.reshape(segments.shape)
    num_sp = sp_unique.shape[0]

    feats = []
    labels = []

    img_flat = img_float.reshape(-1, 3) 
    mask_flat = mask_bin.reshape(-1)        
    sp_flat = sp_map.reshape(-1)            

    for sp_id in range(num_sp):
        pix_idx = np.where(sp_flat == sp_id)[0]
        if pix_idx.size == 0:
            feats.append([0.0, 0.0, 0.0, 0.0])
            labels.append(0)
            continue

        rgb_mean = img_flat[pix_idx].mean(axis=0)  
        area_frac = float(pix_idx.size) / float(H * W)

        feat = np.concatenate([rgb_mean, [area_frac]]).astype(np.float32)
        feats.append(feat)
        m = mask_flat[pix_idx].mean()
        node_label = 1 if m >= 0.5 else 0
        labels.append(node_label)

    feats = np.stack(feats, axis=0)  
    labels = np.array(labels, dtype=np.int64)  
    edges = set()

    for y in range(H - 1):
        for x in range(W - 1):
            a = sp_map[y, x]
            b = sp_map[y, x + 1]
            c = sp_map[y + 1, x]

            if a != b:
                edges.add((int(a), int(b)))
                edges.add((int(b), int(a)))
            if a != c:
                edges.add((int(a), int(c)))
                edges.add((int(c), int(a)))

    if edges:
        src, dst = zip(*edges)
        src = torch.tensor(src, dtype=torch.int64)
        dst = torch.tensor(dst, dtype=torch.int64)
        g = dgl.graph((src, dst), num_nodes=num_sp)
    else:
        g = dgl.graph(([], []), num_nodes=num_sp)

    g.ndata["feat"] = torch.from_numpy(feats) 
    g.ndata["label"] = torch.from_numpy(labels)    

    return g, sp_map


def main():
    root = Path(".").resolve()
    out_dir = root / "data" / "pets_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = OxfordIIITPet(
        root="data",
        split="trainval",
        target_types="segmentation",
        download=True,
    )

    n_total = len(dataset)
    print(f"Loaded Oxford-IIIT Pet trainval split with {n_total} images.")

    rng = np.random.default_rng(SEED)
    indices = np.arange(n_total)
    rng.shuffle(indices)

    n_train = int(TRAIN_FRACTION * n_total)
    train_idx = set(indices[:n_train])
    val_idx = set(indices[n_train:])

    train_graphs = []
    val_graphs = []
    train_sp_maps = []
    val_sp_maps = []

    train_indices = []
    val_indices = []    

    for idx in tqdm(range(n_total), desc="Building superpixel graphs for Pets"):
        img, mask = dataset[idx] 
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
        mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.NEAREST)

        img_np = np.array(img, dtype=np.uint8)     
        mask_np = np.array(mask, dtype=np.uint8)     
        # mask_bin = np.isin(mask_np, [1, 3]).astype(np.uint8)
        mask_bin = np.isin(mask_np, [2, 3]).astype(np.uint8)   # foreground + border


        g, sp_map = build_superpixel_graph(img_np, mask_bin)

        # if idx in train_idx:
        #     train_graphs.append(g)
        #     train_sp_maps.append(sp_map.astype(np.int32))
        # else:
        #     val_graphs.append(g)
        #     val_sp_maps.append(sp_map.astype(np.int32))

        if idx in train_idx:
            train_graphs.append(g)
            train_sp_maps.append(sp_map.astype(np.int32))
            train_indices.append(idx)  # NEW
        else:
            val_graphs.append(g)
            val_sp_maps.append(sp_map.astype(np.int32))
            val_indices.append(idx)    # NEW


    train_bin_path = out_dir / "pets_train.bin"
    val_bin_path = out_dir / "pets_val.bin"

    print(f"Saving {len(train_graphs)} train graphs to {train_bin_path}")
    dgl.save_graphs(str(train_bin_path), train_graphs)

    print(f"Saving {len(val_graphs)} val graphs to {val_bin_path}")
    dgl.save_graphs(str(val_bin_path), val_graphs)

    train_pkl_path = out_dir / "pets_train.pickle"
    val_pkl_path = out_dir / "pets_val.pickle"

    with open(train_pkl_path, "wb") as f:
        pickle.dump(
            {
                "superpixel_maps": train_sp_maps,
                "indices": train_indices,   # NEW
            },
            f,
        )
    with open(val_pkl_path, "wb") as f:
        pickle.dump(
            {
                "superpixel_maps": val_sp_maps,
                "indices": val_indices,     # NEW
            },
            f,
        )


    assert len(train_graphs) == len(train_sp_maps) == len(train_indices)
    assert len(val_graphs) == len(val_sp_maps) == len(val_indices)


    print("Done. Graph data stored in data/pets_graphs/.")


if __name__ == "__main__":
    main()
