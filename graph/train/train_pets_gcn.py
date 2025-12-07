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


class PetsGraphDataset(Dataset):
    def __init__(self, graphs: List[dgl.DGLGraph]):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def collate_graphs(graph_list: List[dgl.DGLGraph]):
    bg = dgl_batch(graph_list)
    return bg


def compute_node_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total if total > 0 else 0.0


def train_pets_gcn(
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    hidden_feats: int = 64,
    num_layers: int = 2,
) -> None:
    device = "cpu"
    print("Using device:", device)

    root = Path(".").resolve()
    graphs_dir = root / "data" / "pets_graphs"

    train_bin = graphs_dir / "pets_train.bin"
    val_bin = graphs_dir / "pets_val.bin"

    assert train_bin.exists(), f"{train_bin} not found. Run pets_preprocessing_slic first."
    assert val_bin.exists(), f"{val_bin} not found. Run pets_preprocessing_slic first."

    train_graphs, _ = dgl.load_graphs(str(train_bin))
    val_graphs, _ = dgl.load_graphs(str(val_bin))

    print(f"Loaded {len(train_graphs)} train graphs and {len(val_graphs)} val graphs.")

    train_ds = PetsGraphDataset(train_graphs)
    val_ds = PetsGraphDataset(val_graphs)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs
    )

    model = PetsGraphSAGE(in_feats=4, hidden_feats=hidden_feats, num_layers=num_layers, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_nodes = 0

        for bg in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train GCN]"):
            # bg = bg.to(device)
            feats = bg.ndata["feat"].to(device) 
            labels = bg.ndata["label"].to(device) 

            optimizer.zero_grad()
            logits = model(bg, feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_nodes = labels.numel()
            train_loss_sum += loss.item() * batch_nodes
            train_acc_sum += compute_node_accuracy(logits, labels) * batch_nodes
            train_nodes += batch_nodes

        train_loss = train_loss_sum / train_nodes
        train_acc = train_acc_sum / train_nodes

        # validate
        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_nodes = 0

        with torch.no_grad():
            for bg in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val GCN]"):
                bg = bg.to(device)
                feats = bg.ndata["feat"].to(device)
                labels = bg.ndata["label"].to(device)

                logits = model(bg, feats)
                loss = criterion(logits, labels)

                batch_nodes = labels.numel()
                val_loss_sum += loss.item() * batch_nodes
                val_acc_sum += compute_node_accuracy(logits, labels) * batch_nodes
                val_nodes += batch_nodes

        val_loss = val_loss_sum / val_nodes
        val_acc = val_acc_sum / val_nodes

        print(
            f"[GCN] Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.4f} | train_node_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} | val_node_acc={val_acc:.3f}"
        )

    out_path = root / "pets_gcn.pth"
    torch.save(model.state_dict(), out_path)
    print(f"GCN model saved as {out_path}")


if __name__ == "__main__":
    train_pets_gcn()
