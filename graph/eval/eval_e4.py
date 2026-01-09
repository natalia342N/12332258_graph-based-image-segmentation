import argparse
from pathlib import Path
import pickle
import numpy as np
from PIL import Image

import torch
import dgl
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms

from graph.models.pets_graphsage import PetsGraphSAGE
from graph.eval.metrics import iou_fg, dice_fg


IMAGE_SIZE = 256

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--graphs_dir", default="data/pets_graphs_unetfeat")
    ap.add_argument("--ckpt", default="checkpoints/e4_unetfeat_gnn.pth")
    args = ap.parse_args()

    device = args.device
    root = Path(".").resolve()
    graphs_dir = root / args.graphs_dir

    val_bin = graphs_dir / "pets_val.bin"
    val_pkl = graphs_dir / "pets_val.pickle"
    ckpt = root / args.ckpt

    assert val_bin.exists(), f"Missing {val_bin}"
    assert val_pkl.exists(), f"Missing {val_pkl}"
    assert ckpt.exists(), f"Missing {ckpt}"

    val_graphs, _ = dgl.load_graphs(str(val_bin))
    with open(val_pkl, "rb") as f:
        meta = pickle.load(f)

    sp_maps = meta["superpixel_maps"]
    val_indices = meta["indices"]
    assert len(val_graphs) == len(sp_maps) == len(val_indices)

    ds = OxfordIIITPet(root="data", split="trainval", target_types="segmentation", download=False)
    resize_mask = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                                    interpolation=transforms.InterpolationMode.NEAREST)

    # model = PetsGraphSAGE(in_feats=32, hidden_feats=64, num_layers=2, num_classes=2).to(device)
    # model.load_state_dict(torch.load(ckpt, map_location=device))
    state = torch.load(ckpt, map_location=device)

    hidden = state["out_linear.weight"].shape[1]

    model = PetsGraphSAGE(
        in_feats=32,    
        hidden_feats=hidden,
        num_layers=2,
        num_classes=2
    ).to(device)

    model.load_state_dict(state)
    model.eval()

    # model.eval()

    ious, dices = [], []
    with torch.no_grad():
        for g, sp_map, ds_idx in zip(val_graphs, sp_maps, val_indices):
            feats = g.ndata["feat"].to(device)
            logits = model(g, feats)
            pred_nodes = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

            sp_map = np.asarray(sp_map, dtype=np.int64)
            pred_mask = pred_nodes[sp_map]

            _, mask = ds[int(ds_idx)]
            mask = resize_mask(mask)
            mask_np = np.array(mask, dtype=np.uint8)
            gt_mask = (mask_np == 2).astype(np.uint8)

            ious.append(iou_fg(pred_mask, gt_mask))
            dices.append(dice_fg(pred_mask, gt_mask))

    print(f"[E4] Val IoU_fg: {np.mean(ious):.4f} (± {np.std(ious):.4f})")
    print(f"[E4] Val Dice : {np.mean(dices):.4f} (± {np.std(dices):.4f})")

if __name__ == "__main__":
    main()
