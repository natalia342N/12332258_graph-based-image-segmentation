from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from PIL import Image

import numpy as np


class OxfordPetsSegmentation(Dataset):
    """
    Oxford-IIIT Pet segmentation dataset wrapper.

    Loads the 'trainval' split from torchvision and returns (image, mask):

    - image: float tensor, shape (3, H, W), in [0,1]
    - mask:  long tensor, shape (H, W), values {0,1}
             (0 = background, 1 = pet: foreground + boundary merged)
    """

    def __init__(
        self,
        root: str = "data",
        size: int = 256,
        img_transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()

        # load 'trainval' split
        self.base = OxfordIIITPet(
            root=root,
            split="trainval",
            target_types="segmentation",
            download=True,
        )

        self.size = size

        # resize + ToTensor
        if img_transform is None:
            self.img_transform = transforms.Compose(
                [
                    transforms.Resize((size, size), interpolation=Image.BILINEAR),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.img_transform = img_transform

        # Nearest-neighbor mask resize
        self.mask_resize = transforms.Resize(
            (size, size), interpolation=Image.NEAREST
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx] 
        img = self.img_transform(img)

        # Convert to numpy before resizing
        mask = np.array(mask, dtype=np.uint8)  # shape (H, W), values {1,2,3}

        # Convert back to PIL so .resize() can be used
        mask = Image.fromarray(mask)
        mask = self.mask_resize(mask)
        mask = np.array(mask, dtype=np.uint8)  # still 1,2,3

        mask = np.isin(mask, [1, 3]).astype(np.uint8)

        mask = torch.from_numpy(mask).long()  # (H, W), values {0,1}

        return img, mask