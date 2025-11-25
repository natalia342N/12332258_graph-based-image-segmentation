import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from baseline.dataloader_pets import OxfordPetsSegmentation
from baseline.unet import UNet
from baseline.utils import update_metrics, compute_epoch_metrics


def augment_batch(imgs: torch.Tensor) -> torch.Tensor:
    device = imgs.device
    b = imgs.size(0)

    factors = torch.empty(b, 1, 1, 1, device=device).uniform_(0.8, 1.2)
    imgs = imgs * factors
    noise_std = 0.03
    imgs = imgs + noise_std * torch.randn_like(imgs)
    return imgs.clamp(0.0, 1.0)


def train_unet_hacked(
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    img_size: int = 256,
    train_fraction: float = 0.9,
) -> None:

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    full_data = OxfordPetsSegmentation(size=img_size)
    n_total = len(full_data)
    n_train = int(train_fraction * n_total)
    n_val = n_total - n_train

    generator = torch.Generator().manual_seed(42)
    train_data, val_data = random_split(full_data, [n_train, n_val], generator=generator)

    print(f"HACKED Dataset sizes: train={len(train_data)}, val={len(val_data)}")

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = UNet(n_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(1, epochs + 1):

        # TRAIN 
        model.train()
        train_loss_sum = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train+hacked]"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            imgs = augment_batch(imgs)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks.float().unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * imgs.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)

        # VALIDATION
        model.eval()
        val_loss_sum = 0.0
        val_stats = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0}

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)
                loss = criterion(logits, masks.float().unsqueeze(1))
                val_loss_sum += loss.item() * imgs.size(0)

                update_metrics(logits, masks, val_stats)

        val_loss = val_loss_sum / len(val_loader.dataset)
        val_metrics = compute_epoch_metrics(val_stats)

        print(
            f"[HACKED] Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"IoU_fg={val_metrics['iou_fg']:.3f} | "
            f"Dice={val_metrics['dice_fg']:.3f}"
        )

        scheduler.step()

    torch.save(model.state_dict(), "unet_hacked.pth")
    print("Hacked model saved as unet_hacked.pth")


if __name__ == "__main__":
    train_unet_hacked()
