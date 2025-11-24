import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from baseline.dataloader_pets import OxfordPetsSegmentation
from baseline.unet import UNet
from baseline.utils import update_metrics, compute_epoch_metrics


def train_unet(
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-3,
    img_size: int = 256,
    train_fraction: float = 0.9,
) -> None:

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    # 1) Create full dataset (train+val)
    full_data = OxfordPetsSegmentation(size=img_size)

    n_total = len(full_data)
    n_train = int(train_fraction * n_total)
    n_val = n_total - n_train

    # deterministic split
    generator = torch.Generator().manual_seed(42)
    train_data, val_data = random_split(full_data, [n_train, n_val], generator=generator)

    print(f"Dataset sizes: train={len(train_data)}, val={len(val_data)}")

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 2) Model, optimizer, loss
    model = UNet(n_classes=1).to(device)  # binary: 1 channel

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # 3) Training loop
    for epoch in range(1, epochs + 1):

        # TRAIN
        model.train()
        train_loss_sum = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)                  
            loss = criterion(logits, masks.float().unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * imgs.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)

        # VALIDATE
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
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"IoU_fg={val_metrics['iou_fg']:.3f} | "
            f"mIoU={val_metrics['miou']:.3f} | "
            f"Dice={val_metrics['dice_fg']:.3f} | "
            f"Acc={val_metrics['pixel_acc']:.3f}"
        )

    torch.save(model.state_dict(), "baseline_unet.pth")
    print("Model saved as baseline_unet.pth")


if __name__ == "__main__":
    train_unet()
