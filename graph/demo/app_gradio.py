import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import dgl
from skimage.segmentation import slic
import gradio as gr

from baseline.unet import UNet
from graph.models.pets_graphsage import PetsGraphSAGE

IMAGE_SIZE = 256
N_SEGMENTS = 200
COMPACTNESS = 10.0
HOOK_NAME = "down1"
DEVICE = "cpu"


class UNetFeatureHook(torch.nn.Module):
    def __init__(self, unet: torch.nn.Module, hook_name: str):
        super().__init__()
        self.unet = unet
        self.feat = None
        name_to_module = dict(unet.named_modules())
        name_to_module[hook_name].register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        self.feat = out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.feat = None
        _ = self.unet(x)
        return self.feat


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


def sp_labels_to_pixel_mask(sp_map: np.ndarray, sp_pred: np.ndarray) -> np.ndarray:
    return sp_pred[sp_map].astype(np.uint8)


def overlay(image_rgb: np.ndarray, mask01: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    img = image_rgb.astype(np.float32)
    out = img.copy()
    out[mask01 == 1, 0] = 255.0
    out[mask01 == 1, 1] *= (1 - alpha)
    out[mask01 == 1, 2] *= (1 - alpha)
    out = (alpha * out + (1 - alpha) * img).clip(0, 255).astype(np.uint8)
    return out


unet = UNet(n_classes=1).to(DEVICE)
unet.load_state_dict(torch.load("unet_hacked.pth", map_location=DEVICE))
unet.eval()
extractor = UNetFeatureHook(unet, HOOK_NAME).to(DEVICE).eval()

gnn = PetsGraphSAGE(in_feats=32, hidden_feats=64, num_layers=2, num_classes=2).to(DEVICE)
gnn.load_state_dict(torch.load("e4_unetfeat_gnn.pth", map_location=DEVICE))
gnn.eval()


def predict(pil_img: Image.Image):
    # preprocess
    img = pil_img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    img_np = np.array(img, dtype=np.uint8)
    img_float = img_np.astype(np.float32) / 255.0

    # superpixels
    segments = slic(img_float, n_segments=N_SEGMENTS, compactness=COMPACTNESS, start_label=0)
    uniq, inv = np.unique(segments, return_inverse=True)
    sp_map = inv.reshape(segments.shape)

    # features via UNet
    x = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = extractor(x).squeeze(0)  # (32,256,256) for down1
    if feat.shape[-1] != IMAGE_SIZE:
        feat = F.interpolate(feat.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).squeeze(0)

    node_feats = pool_per_superpixel(feat.cpu(), sp_map)
    src, dst = build_edges_from_spmap(sp_map)
    g = dgl.graph((src, dst), num_nodes=int(sp_map.max()) + 1)
    g.ndata["feat"] = torch.from_numpy(node_feats)

    # GNN inference
    with torch.no_grad():
        logits = gnn(g, g.ndata["feat"].to(DEVICE))
        sp_pred = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

    pred_mask = sp_labels_to_pixel_mask(sp_map, sp_pred)
    pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
    overlay_img = Image.fromarray(overlay(img_np, pred_mask))

    return overlay_img, pred_mask_img


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Image(type="pil", label="Overlay (prediction on input)"),
        gr.Image(type="pil", label="Predicted mask"),
    ],
    title="Oxford Pets Segmentation Demo (Hybrid CNN + GNN, E4)",
    description="Upload a pet image. The app runs U-Net encoder feature extraction + GraphSAGE over SLIC superpixels and returns a segmentation mask.",
    
)

if __name__ == "__main__":
    demo.launch()
