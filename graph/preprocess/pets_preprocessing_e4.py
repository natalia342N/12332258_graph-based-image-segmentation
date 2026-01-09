from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
import dgl
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from skimage.segmentation import slic

from baseline.unet import UNet

IMAGE_SIZE = 256
N_SEGMENTS = 200
COMPACTNESS = 10.0
TRAIN_FRACTION = 0.9
SEED = 42

HOOK_NAME = "down1"

class UNetFeatureHook(torch.nn.Module):
    def __init__(self, unet: torch.nn.Module, hook_name: str):
        super().__init__()
        self.unet = unet
        self.hook_name = hook_name
        self.feat = None

        name_to_module = dict(unet.named_modules())
        if hook_name not in name_to_module:
            raise ValueError(f"Hook '{hook_name}' not in UNet modules.")

        def _hook(mod, inp, out):
            self.feat = out

        name_to_module[hook_name].register_forward_hook(_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.feat = None
        _ = self.unet(x)
        if self.feat is None:
            raise RuntimeError("Hook did not capture feature map.")
        return self.feat

def pool_per_superpixel(feat_map: torch.Tensor, sp_map: np.ndarray) -> np.ndarray:
    C, H, W = feat_map.shape
    sp = sp_map.astype(np.int64)
    num_sp = int(sp.max()) + 1

    feat = feat_map.numpy().reshape(C, -1)  
    sp_flat = sp.reshape(-1)                

    out = np.zeros((num_sp, C), dtype=np.float32)
    counts = np.bincount(sp_flat, minlength=num_sp).astype(np.float32)

    for c in range(C):
        np.add.at(out[:, c], sp_flat, feat[c])

    out /= np.maximum(counts[:, None], 1.0)
    return out

def build_edges_from_spmap(sp_map: np.ndarray):
    H, W = sp_map.shape
    edges = set()
    for y in range(H - 1):
        for x in range(W - 1):
            a = sp_map[y, x]
            b = sp_map[y, x + 1]
            c = sp_map[y + 1, x]
            if a != b:
                edges.add((int(a), int(b))); edges.add((int(b), int(a)))
            if a != c:
                edges.add((int(a), int(c))); edges.add((int(c), int(a)))
    if edges:
        src, dst = zip(*edges)
        return torch.tensor(src, dtype=torch.int64), torch.tensor(dst, dtype=torch.int64)
    return torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)

def main():
    device = "cpu"
    root = Path(".").resolve()
    out_dir = root / "data" / "pets_graphs_unetfeat"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = OxfordIIITPet(root="data", split="trainval", target_types="segmentation", download=True)

    rng = np.random.default_rng(SEED)
    idxs = np.arange(len(ds))
    rng.shuffle(idxs)
    n_train = int(TRAIN_FRACTION * len(ds))
    train_set = set(idxs[:n_train])

    # unet = UNet(in_channels=3, out_channels=1).to(device)
    unet = UNet(n_classes=1).to(device)

    unet.load_state_dict(torch.load(root / "unet_hacked.pth", map_location=device))
    unet.eval()

    extractor = UNetFeatureHook(unet, HOOK_NAME).to(device).eval()

    to_tensor = transforms.ToTensor()
    resize_img = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    resize_mask = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST)

    train_graphs, val_graphs = [], []
    train_maps, val_maps = [], []
    train_idx_list, val_idx_list = [], []

    for ds_idx in tqdm(range(len(ds)), desc="E4 preprocess (U-Net feats)"):
        img, mask = ds[ds_idx]
        img = resize_img(img)
        mask = resize_mask(mask)

        img_np = np.array(img, dtype=np.uint8)
        img_float = img_np.astype(np.float32) / 255.0

        segments = slic(img_float, n_segments=N_SEGMENTS, compactness=COMPACTNESS, start_label=0)
        uniq, inv = np.unique(segments, return_inverse=True)
        sp_map = inv.reshape(segments.shape)
        num_sp = int(sp_map.max()) + 1

        mask_np = np.array(mask, dtype=np.uint8)
        gt_bin = (mask_np == 2).astype(np.uint8)
        gt_flat = gt_bin.reshape(-1)
        sp_flat = sp_map.reshape(-1)

        labels = np.zeros((num_sp,), dtype=np.int64)
        for sp_id in range(num_sp):
            pix = np.where(sp_flat == sp_id)[0]
            labels[sp_id] = 1 if gt_flat[pix].mean() >= 0.5 else 0

        x = to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = extractor(x).squeeze(0)  

        if feat.shape[-1] != IMAGE_SIZE or feat.shape[-2] != IMAGE_SIZE:
            feat = F.interpolate(
                feat.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear", align_corners=False
            ).squeeze(0)

        node_feats = pool_per_superpixel(feat.cpu(), sp_map)  

        src, dst = build_edges_from_spmap(sp_map)
        g = dgl.graph((src, dst), num_nodes=num_sp)
        g.ndata["feat"] = torch.from_numpy(node_feats)
        g.ndata["label"] = torch.from_numpy(labels)

        if ds_idx in train_set:
            train_graphs.append(g); train_maps.append(sp_map.astype(np.int32)); train_idx_list.append(int(ds_idx))
        else:
            val_graphs.append(g); val_maps.append(sp_map.astype(np.int32)); val_idx_list.append(int(ds_idx))

    dgl.save_graphs(str(out_dir / "pets_train.bin"), train_graphs)
    dgl.save_graphs(str(out_dir / "pets_val.bin"), val_graphs)

    with open(out_dir / "pets_train.pickle", "wb") as f:
        pickle.dump({"superpixel_maps": train_maps, "indices": train_idx_list}, f)
    with open(out_dir / "pets_val.pickle", "wb") as f:
        pickle.dump({"superpixel_maps": val_maps, "indices": val_idx_list}, f)

    print("Saved E4 graphs to", out_dir)
    print("Example node feature dim:", train_graphs[0].ndata["feat"].shape[1])

if __name__ == "__main__":
    main()
