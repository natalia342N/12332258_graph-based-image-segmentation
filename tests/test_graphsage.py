import torch
import dgl

from graph.models.pets_graphsage import PetsGraphSAGE


def test_graphsage_forward_shape():
    g = dgl.graph(([0, 1], [1, 0]))
    g.ndata["feat"] = torch.randn(2, 4)  # E3

    model = PetsGraphSAGE(
        in_feats=4,
        hidden_feats=8,
        num_layers=2,
        num_classes=2,
    )

    logits = model(g, g.ndata["feat"])

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
