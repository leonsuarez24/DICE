import torch
import time

from torch.utils.data import SubsetRandomSampler
import os
import torch
import numpy as np

import random
import glob
import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Save training checkpoint with compression to reduce size"""
    torch.save(state, filename, _use_new_zipfile_serialization=True)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load checkpoint with proper device handling"""
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_loss = checkpoint["best_loss"]

        # Load random states for reproducibility
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"].to("cpu"))
            if device == "cuda":
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].to("cpu"))

        print(f"Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")
        return start_epoch, best_loss
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        return 0, float("inf")


def cleanup_old_checkpoints(checkpoint_dir, keep_last=3):
    """Remove old checkpoints, keeping only the most recent ones"""
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth.tar")))

    if len(checkpoints) > keep_last:
        for old_checkpoint in checkpoints[:-keep_last]:
            try:
                os.remove(old_checkpoint)
                print(f"Removed old checkpoint: {old_checkpoint}")
            except OSError as e:
                print(f"Error removing {old_checkpoint}: {e}")


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(
        torch.cat(
            [
                torch.cat([i for i in images.cpu()], dim=-1),
            ],
            dim=-2,
        )
        .permute(1, 2, 0)
        .cpu()
    )
    plt.show()


def save_images(images, path, **kwargs):
    if images.shape[0] == 2:
        images = images[0].unsqueeze(0)
    grid = torchvision.utils.make_grid(images, normalize=True, **kwargs)
    torchvision.utils.save_image(images, path)
    return grid


def get_data(args):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
            torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def set_seed(seed: int):
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(mode=True)


def log_k_space(y):
    y_complex = torch.view_as_complex(y.permute(0, 2, 3, 1).contiguous())
    return torch.log(1 + torch.abs(y_complex))


def get_validation_set(dst_train, split: float = 0.1, seed: int = 42):

    set_seed(seed)

    indices = list(range(len(dst_train)))
    np.random.shuffle(indices)
    split = int(np.floor(split * len(dst_train)))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sample = SubsetRandomSampler(train_indices)
    val_sample = SubsetRandomSampler(val_indices)

    return train_sample, val_sample


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_time():

    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def save_metrics(save_path):

    images_path = save_path + "/images"
    model_path = save_path + "/model"
    metrics_path = save_path + "/metrics"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)

    return images_path, model_path, metrics_path


def save_npy_metric(file, metric_name):

    with open(f"{metric_name}.npy", "wb") as f:
        np.save(f, file)


def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_dict(dictionary):
    for key, value in dictionary.items():
        print(f"{key}: {value}")


import matplotlib.pyplot as plt
import wandb
import io


def image_to_wandb_image(tensor):
    tensor = tensor[0].detach().cpu().numpy()
    plt.imshow(tensor, cmap="jet")
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)

    pil_img = Image.open(buf).convert("RGB")
    np_img = np.array(pil_img)

    return wandb.Image(np_img)
