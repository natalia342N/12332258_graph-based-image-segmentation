import torch
import dgl

from graph.models.pets_graphsage import PetsGraphSAGE

def test_graphsage_forward_shape_and_finite():
    src = torch.tensor([0, 1, 2])
    dst = torch.tensor([1, 2, 0])
    g = dgl.graph((src, dst), num_nodes=3)

    in_dim = 4
    num_classes = 2
    feats = torch.randn(3, in_dim)

    model = PetsGraphSAGE(in_feats=in_dim, hidden_feats=8, num_layers=2, num_classes=num_classes, dropout=0.0)
    logits = model(g, feats)
    assert logits.shape == (3, num_classes)
    assert torch.isfinite(logits).all()
