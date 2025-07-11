import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


class TestDataset(Dataset):
    """
    256×256 single-channel .npy images.
    Returns FloatTensor of shape (1, 256, 256) with values left untouched.
    """

    def __init__(self, root_dir: str, transform=None):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.npy")))
        if not self.files:
            raise RuntimeError(f"No .npy files found in {root_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(self.files[idx])  # (256, 256)
        if img.ndim != 2:
            raise ValueError(f"Expected 2-D array, got {img.shape}")

        img = img.astype(np.float32)

        if self.transform:
            img = self.transform(img)  # ToTensor will add channel dim
        else:
            img = torch.from_numpy(img).unsqueeze(0)

        return img


# ---------- DataLoader ---------- #
dataset = TestDataset("data/test_imgs")
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=min(4, os.cpu_count() or 1),
    pin_memory=torch.cuda.is_available(),
)


# ---------- Grid-plot helper ---------- #
def show_batch(batch, nrow=8, figsize=(8, 8)):
    """
    batch : Tensor (B, 1, 256, 256) — typically from next(iter(loader))
    nrow  : images per row in the grid
    """
    # make_grid expects 3-channel; repeat the single channel
    grid = make_grid(batch.repeat(1, 3, 1, 1), nrow=nrow, padding=2)

    npimg = grid.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    plt.figure(figsize=figsize)
    plt.imshow(npimg, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ---------- Quick demo ---------- #
if __name__ == "__main__":
    batch = next(iter(loader))  # batch.shape → (32, 1, 256, 256)
    print("Batch shape:", batch.shape)
    show_batch(batch, nrow=8)
