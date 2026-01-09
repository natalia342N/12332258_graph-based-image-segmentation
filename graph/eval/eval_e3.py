from pathlib import Path
import argparse
import pickle
import json
import yaml
import numpy as np

import torch
import dgl
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms

from graph.models.pets_graphsage import PetsGraphSAGE
from graph.eval.metrics import iou_fg, dice_fg

IMAGE_SIZE = 256


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="Path to YAML config (e.g., configs/e3.yaml)")
    ap.add_argument("--device", default="cpu", type=str, help="cpu or cuda")
    ap.add_argument(
        "--ckpt",
        default=None,
        type=str,
        help="Optional override checkpoint path. If not given, uses runs/<experiment>/best_gnn.pth",
    )
    ap.add_argument(
        "--use_last",
        action="store_true",
        help="Use runs/<experiment>/last_gnn.pth instead of best_gnn.pth (ignored if --ckpt is given).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    device = args.device
    root = Path(".").resolve()

    graphs_dir_rel = cfg.get("graphs_dir", "data/pets_graphs")
    graphs_dir = root / graphs_dir_rel

    val_bin = graphs_dir / "pets_val.bin"
    val_pkl = graphs_dir / "pets_val.pickle"

    assert val_bin.exists(), f"Missing {val_bin}"
    assert val_pkl.exists(), f"Missing {val_pkl}"

    exp_name = cfg.get("experiment", "E3-A")
    run_dir = root / "runs" / exp_name

    if args.ckpt is not None:
        ckpt = Path(args.ckpt)
        if not ckpt.is_absolute():
            ckpt = root / ckpt
    else:
        ckpt_name = "last_gnn.pth" if args.use_last else "best_gnn.pth"
        ckpt = run_dir / ckpt_name

    assert ckpt.exists(), f"Missing checkpoint {ckpt} (run training first)"

    val_graphs, _ = dgl.load_graphs(str(val_bin))
    with open(val_pkl, "rb") as f:
        meta = pickle.load(f)

    sp_maps = meta["superpixel_maps"]
    if "indices" not in meta:
        raise RuntimeError(
            "pets_val.pickle has no 'indices'. Re-run preprocessing with index saving enabled."
        )
    val_indices = meta["indices"]

    assert len(val_graphs) == len(sp_maps) == len(val_indices)

    ds = OxfordIIITPet(
        root="data",
        split="trainval",
        target_types="segmentation",
        download=False,
    )
    resize_mask = transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=transforms.InterpolationMode.NEAREST,
    )

    input_feature_dim = int(cfg.get("input_feature_dim", 4))
    hidden_node_features = int(cfg.get("hidden_node_features", 64))
    num_layers = int(cfg.get("num_layers", 2))

    model = PetsGraphSAGE(
        in_feats=input_feature_dim,
        hidden_feats=hidden_node_features,
        num_layers=num_layers,
        num_classes=2,
    ).to(device)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    ious, dices = [], []

    with torch.no_grad():
        for g, sp_map, ds_idx in zip(val_graphs, sp_maps, val_indices):
            g = g.to(device)
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

    mean_iou = float(np.mean(ious))
    std_iou = float(np.std(ious))
    mean_dice = float(np.mean(dices))
    std_dice = float(np.std(dices))

    print(f"[E3] Val IoU_fg: {mean_iou:.4f}  (± {std_iou:.4f})")
    print(f"[E3] Val Dice : {mean_dice:.4f}  (± {std_dice:.4f})")

    out = {
        "experiment": exp_name,
        "checkpoint": str(ckpt),
        "graphs_dir": str(graphs_dir),
        "val_iou_fg_mean": mean_iou,
        "val_iou_fg_std": std_iou,
        "val_dice_mean": mean_dice,
        "val_dice_std": std_dice,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_e3.json").write_text(json.dumps(out, indent=2))
    print(f"[saved] {run_dir / 'eval_e3.json'}")


if __name__ == "__main__":
    main()
