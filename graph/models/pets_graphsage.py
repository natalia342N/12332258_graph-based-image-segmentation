import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn


class PetsGraphSAGE(nn.Module):
    def __init__(self, in_feats: int = 4, hidden_feats: int = 64, num_layers: int = 2, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(dglnn.SAGEConv(in_feats, hidden_feats, aggregator_type="mean"))
        # hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(dglnn.SAGEConv(hidden_feats, hidden_feats, aggregator_type="mean"))

        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(hidden_feats, num_classes)

    def forward(self, g: dgl.DGLGraph, feats: torch.Tensor):
        h = feats
        for layer in self.layers:
            h = layer(g, h)
            h = F.relu(h)
            h = self.dropout(h)
        logits = self.out_linear(h) 
        return logits
