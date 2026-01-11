import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import dgl
from skimage.segmentation import slic
import gradio as gr

from baseline.unet import UNet
from graph.models.pets_graphsage import PetsGraphSAGE


MODEL_TAG = "GNN-CNN (E4) model"
IMAGE_SIZE = 256
HOOK_NAME = "down1"
DEVICE = "cpu"  

UNET_CKPT = "checkpoints/unet_hacked.pth"
GNN_CKPT  = "checkpoints/e4_unetfeat_gnn.pth"

DEFAULT_N_SEGMENTS = 200
DEFAULT_COMPACTNESS = 10.0
DEFAULT_ALPHA = 0.35

class UNetFeatureHook(torch.nn.Module):
    def __init__(self, unet: torch.nn.Module, hook_name: str):
        super().__init__()
        self.unet = unet
        self.feat = None
        dict(unet.named_modules())[hook_name].register_forward_hook(self._hook)

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
            a = int(sp_map[y, x])
            b = int(sp_map[y, x + 1])
            c = int(sp_map[y + 1, x])
            if a != b:
                edges.add((a, b)); edges.add((b, a))
            if a != c:
                edges.add((a, c)); edges.add((c, a))
    if not edges:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)
    src, dst = zip(*edges)
    return torch.tensor(src, dtype=torch.int64), torch.tensor(dst, dtype=torch.int64)


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


def overlay(image_rgb: np.ndarray, mask01: np.ndarray, alpha: float) -> np.ndarray:
    img = image_rgb.astype(np.float32)
    out = img.copy()
    out[mask01 == 1, 0] = 255.0
    out[mask01 == 1, 1] *= (1 - alpha)
    out[mask01 == 1, 2] *= (1 - alpha)
    out = (alpha * out + (1 - alpha) * img).clip(0, 255).astype(np.uint8)
    return out



def load_models():
    if not os.path.exists(UNET_CKPT):
        raise FileNotFoundError(f"Missing U-Net checkpoint: {UNET_CKPT}")
    if not os.path.exists(GNN_CKPT):
        raise FileNotFoundError(f"Missing GNN checkpoint: {GNN_CKPT}")

    unet = UNet(n_classes=1).to(DEVICE)
    unet.load_state_dict(torch.load(UNET_CKPT, map_location=DEVICE))
    unet.eval()

    extractor = UNetFeatureHook(unet, HOOK_NAME).to(DEVICE).eval()

    gnn = PetsGraphSAGE(in_feats=32, hidden_feats=64, num_layers=2, num_classes=2).to(DEVICE)
    gnn.load_state_dict(torch.load(GNN_CKPT, map_location=DEVICE))
    gnn.eval()

    return extractor, gnn


extractor, gnn = load_models()


def predict(pil_img: Image.Image, n_segments: int, compactness: float, alpha: float):
    img = pil_img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    img_np = np.array(img, dtype=np.uint8)
    img_float = img_np.astype(np.float32) / 255.0

    segments = slic(img_float, n_segments=int(n_segments), compactness=float(compactness), start_label=0)
    _, inv = np.unique(segments, return_inverse=True)
    sp_map = inv.reshape(segments.shape)

    x = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = extractor(x).squeeze(0)  
    if feat.shape[-1] != IMAGE_SIZE:
        feat = F.interpolate(
            feat.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

    node_feats = pool_per_superpixel(feat.cpu(), sp_map)
    src, dst = build_edges_from_spmap(sp_map)

    g = dgl.graph((src, dst), num_nodes=int(sp_map.max()) + 1)
    g.ndata["feat"] = torch.from_numpy(node_feats)

    with torch.no_grad():
        logits = gnn(g, g.ndata["feat"].to(DEVICE))
        sp_pred = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

    pred_mask = sp_labels_to_pixel_mask(sp_map, sp_pred)
    pred_mask = 1 - pred_mask

    overlay_img = Image.fromarray(overlay(img_np, pred_mask, alpha=float(alpha)))
    # pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
    pred_mask_img = Image.fromarray(((1 - pred_mask) * 255).astype(np.uint8))


    return overlay_img, pred_mask_img


EXAMPLES_DIR = "demo_examples"
example_paths = []
if os.path.isdir(EXAMPLES_DIR):
    for fn in sorted(os.listdir(EXAMPLES_DIR)):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            example_paths.append(os.path.join(EXAMPLES_DIR, fn))

with gr.Blocks(title=f"Oxford Pets Dataset Segmentation Demo App— {MODEL_TAG}") as demo:
    gr.Markdown(
    f"# Oxford Pets Segmentation Demo — {MODEL_TAG}\n"
    "Upload an animal image (or choose from examples below). The app highlights the animal and shows the mask."
)

    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Input image")
            with gr.Accordion("Settings", open=False):
                nseg = gr.Slider(50, 400, value=DEFAULT_N_SEGMENTS, step=10, label="Number of superpixels")
                gr.Markdown(
            "*Higher values create finer segmentation with more detail. "
            "Lower values create larger, simpler regions.*"
             )
                comp = gr.Slider(1.0, 30.0, value=DEFAULT_COMPACTNESS, step=0.5, label="Superpixel shape")
                gr.Markdown(
            "*Lower values follow image details more closely. "
            "Higher values produce smoother, more regular regions.*"
            )
                alph = gr.Slider(0.05, 0.80, value=DEFAULT_ALPHA, step=0.05, label="Highlight strength")
                gr.Markdown(
            "*Lower values keep the image clearer. "
            "Higher values make the highlighted region more visible.*"
            )
            btn = gr.Button("Run Image Segmentation")

        with gr.Column():
            out_overlay = gr.Image(
                type="pil",
                label="Highlighted animal"
            )
            out_mask = gr.Image(
                type="pil",
                label="Animal mask"
            )

    btn.click(fn=predict, inputs=[inp, nseg, comp, alph], outputs=[out_overlay, out_mask])

    if example_paths:
        gr.Examples(
            examples=example_paths,
            inputs=inp,
            label="Example images",
        )

    gr.Markdown(
        f" **Device:** `{DEVICE}`"
    )

if __name__ == "__main__":
    demo.launch()
