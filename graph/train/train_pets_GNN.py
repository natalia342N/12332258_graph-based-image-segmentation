import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dgl
from dgl import batch as dgl_batch
from tqdm import tqdm

from graph.models.pets_graphsage import PetsGraphSAGE
from torch.utils.tensorboard import SummaryWriter


import argparse
import json
import yaml
import numpy as np

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)



class PetsGraphDataset(Dataset):
    def __init__(self, graphs: List[dgl.DGLGraph]):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def collate_graphs(graph_list: List[dgl.DGLGraph]):
    batched_graph = dgl_batch(graph_list)
    return batched_graph


def compute_node_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total if total > 0 else 0.0


def train_pets_gcn(cfg: dict) -> None:
    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch_size", 8))
    lr = float(cfg.get("learning_rate", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-5))
    hidden_node_features = int(cfg.get("hidden_node_features", 64))
    num_layers = int(cfg.get("num_layers", 2))

    input_feature_dim = int(cfg.get("input_feature_dim", 32))
    

    graphs_dir_rel = cfg.get("graphs_dir", "data/pets_graphs_unetfeat")
    exp_name = cfg.get("experiment", "E4-EXP")
    seed = int(cfg.get("seed", 42))
    device = cfg.get("device", "cpu")

    # early stopping 
    es_cfg = cfg.get("early_stopping", {}) or {}
    early_stop_enabled = bool(es_cfg.get("enabled", True))
    early_stop_patience = int(es_cfg.get("patience", 5))
    early_stop_min_delta = float(es_cfg.get("min_delta", 0.0))


    set_seed(seed)

    run_dir = Path("runs") / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", cfg)

    tb_dir = run_dir / "tensorboard"
    writer = SummaryWriter(log_dir=tb_dir)


    print("Using device:", device)

    root = Path(".").resolve()
    graphs_dir = root / graphs_dir_rel
    train_bin = graphs_dir / "pets_train.bin"
    val_bin = graphs_dir / "pets_val.bin"

    assert train_bin.exists(), f"{train_bin} not found."
    assert val_bin.exists(), f"{val_bin} not found."

    train_graphs, _ = dgl.load_graphs(str(train_bin))
    val_graphs, _ = dgl.load_graphs(str(val_bin))
    print(f"Loaded {len(train_graphs)} train graphs and {len(val_graphs)} val graphs.")

    train_loader = DataLoader(
        PetsGraphDataset(train_graphs),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
    )
    val_loader = DataLoader(
        PetsGraphDataset(val_graphs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    # model = PetsGraphSAGE(in_node_features=32, hidden_node_features=hidden_node_features, num_layers=num_layers, num_classes=2).to(device)
    # input_feature_dim = 32
    model = PetsGraphSAGE(
        in_feats=input_feature_dim,
        hidden_feats=hidden_node_features,
        num_layers=num_layers,
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_epoch = -1
    patience_counter = 0
    stopped_epoch = 0


    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_nodes = 0

        for batched_graph in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train GCN]"):
            batched_graph = batched_graph.to(device)
            node_features = batched_graph.ndata["feat"].to(device)
            labels = batched_graph.ndata["label"].to(device)

            optimizer.zero_grad()
            logits = model(batched_graph, node_features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_nodes = labels.numel()
            train_loss_sum += loss.item() * batch_nodes
            train_acc_sum += compute_node_accuracy(logits, labels) * batch_nodes
            train_nodes += batch_nodes

        train_loss = train_loss_sum / train_nodes
        train_acc = train_acc_sum / train_nodes

        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_nodes = 0

        with torch.no_grad():
            for batched_graph in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val GCN]"):
                batched_graph = batched_graph.to(device)
                node_features = batched_graph.ndata["feat"].to(device)
                labels = batched_graph.ndata["label"].to(device)

                logits = model(batched_graph, node_features)
                loss = criterion(logits, labels)

                batch_nodes = labels.numel()
                val_loss_sum += loss.item() * batch_nodes
                val_acc_sum += compute_node_accuracy(logits, labels) * batch_nodes
                val_nodes += batch_nodes

        val_loss = val_loss_sum / val_nodes
        val_acc = val_acc_sum / val_nodes

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train_node", train_acc, epoch)
        writer.add_scalar("Accuracy/val_node", val_acc, epoch)


        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     best_epoch = epoch
        #     torch.save(model.state_dict(), run_dir / "best_gnn.pth")
        improved = val_acc > (best_val_acc + early_stop_min_delta)

        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / "best_gnn.pth")
        else:
            patience_counter += 1

        if early_stop_enabled and patience_counter >= early_stop_patience:
            stopped_epoch = epoch
            print(
                f"[early-stopping] No improvement for {early_stop_patience} epochs "
                f"(best_val_node_acc={best_val_acc:.4f} at epoch {best_epoch}). Stopping."
            )
            break


        print(
            f"[GCN] Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.4f} | train_node_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} | val_node_acc={val_acc:.3f}"
        )

    torch.save(model.state_dict(), run_dir / "last_gnn.pth")
    print(f"Saved last checkpoint: {run_dir / 'last_gnn.pth'}")


    if stopped_epoch == 0:
        stopped_epoch = epoch 

    summary = {
        "experiment": exp_name,
        "best_epoch": best_epoch,
        "best_val_node_acc": float(best_val_acc),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "hidden_node_features": hidden_node_features,
        "num_layers": num_layers,
        "stopped_epoch": stopped_epoch,
        "early_stopping": {
            "enabled": early_stop_enabled,
            "patience": early_stop_patience,
            "min_delta": early_stop_min_delta,
            },
    }
    save_json(run_dir / "summary.json", summary)
    print("[done]", summary)
    writer.close()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="Path to YAML config")
    ap.add_argument("--device", default="cpu", type=str, help="cpu or cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["device"] = args.device

    train_pets_gcn(cfg)


if __name__ == "__main__":
    main()


