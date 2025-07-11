import argparse, torch
from utils.ct_model import CTModel
from utils.utils import set_seed
import matplotlib.pyplot as plt
from DICE import DICE

from guided_diffusion.script_util import create_model
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from utils.test_set_loader import TestDataset
from torch.utils.data import DataLoader
from torchvision import transforms


def main(opt):

    print("Options:")
    for key, value in vars(opt).items():
        print(f"{key}: {value}")

    device = f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu"
    set_seed(7)


    ckpt = torch.load(opt.weights, map_location=device, weights_only=True)
    net = create_model(
        image_size=opt.image_size, num_channels=64, num_res_blocks=3, input_channels=1
    ).to("cuda")
    net.load_state_dict(ckpt["model_state"])
    net.eval()


    dataset = TestDataset(
        "data/test_imgs",
        transform=transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Resize((opt.image_size, opt.image_size)),
            ]
        ),
    )
    testloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


    GT = next(iter(testloader))[opt.idx].unsqueeze(0).to(device)


    GT = GT * 2 - 1


    ct_model = CTModel(
        im_size=opt.image_size,
        num_angles=opt.num_angles,
        sampling_ratio=opt.sampling_ratio,
        sampling_method=opt.sampling_method,
    ).to(device)

    y = ct_model.forward_pass(GT)
    x_estimate = ct_model.transpose_pass(y)

    diff = DICE(
        device=device,
        img_size=opt.image_size,
        noise_steps=1000,
        schedule_name="cosine",
        channels=1,
        rho=opt.rho,
        mu=opt.mu,
        skip_type=opt.skip_type,
        iter_num=opt.iter_num,
    )

    reconstruction = diff.sample(
        model=net,
        y=y,
        transpose_pass=ct_model.transpose_pass,
        forward_pass=ct_model.forward_pass,
        CG_iter=opt.CG_iter,
        CE_iter=opt.CE_iter,
    )


    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device)

    reconstruction = (reconstruction + 1) / 2  
    GT = (GT + 1) / 2 

    reconstruction = reconstruction.clamp(0, 1)
    GT = GT.clamp(0, 1)


    error_map = (reconstruction - GT).abs()


    ssim_pred = SSIM(reconstruction, GT)
    psnr_pred = PSNR(reconstruction, GT)


    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    ax[0].imshow(GT[0, 0].cpu().detach().numpy(), cmap="gray")
    ax[0].axis("off")
    ax[0].set_title("Ground Truth")

    ax[1].imshow(x_estimate[0, 0].cpu().detach().numpy(), cmap="gray")
    ax[1].axis("off")
    ax[1].set_title(f"A^T y")

    ax[2].imshow(reconstruction[0, 0].cpu().detach().numpy(), cmap="gray")
    ax[2].axis("off")
    ax[2].set_title(f"{opt.algo} Predicted\nSSIM: {ssim_pred:.4f}, PSNR: {psnr_pred:.2f}")

    ax[3].imshow(y[0, 0].cpu().detach().numpy().T, cmap="jet")
    ax[3].axis("off")
    ax[3].set_title(f"Sinogram + {opt.sampling_method} Sampling | {opt.sampling_ratio:.2f} ratio")

    ax[4].imshow(error_map[0, 0].cpu().detach().numpy(), cmap="hot")
    ax[4].axis("off")
    ax[4].set_title(f"Error Map ({opt.algo} - GT)")
    plt.colorbar(ax[4].images[0], ax=ax[4])
    plt.suptitle(f"CT Reconstruction with {opt.algo} Algorithm", fontsize=16)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--idx", type=int, default=21)
    p.add_argument(
        "--weights",
        type=str,
        default="weights/d_lodo_e_1000_bs_4_lr_0.0003_seed_2_img_256_schedule_cosine_gpu_1_c_1_si_100/checkpoints/latest.pth.tar",
    )
    p.add_argument(
        "--batch_size", type=int, default=150, help="Batch size for sampling (default: 1)"
    )
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    p.add_argument("--num_angles", type=int, default=180)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for training (default: 0)")



    p.add_argument("--sampling_ratio", type=float, default=1 / 3)
    p.add_argument(
        "--sampling_method",
        type=str,
        default="uniform",
        choices=["uniform", "non_uniform"],
    )


    p.add_argument("--CG_iter", type=int, default=5, help="Number of CG steps in CEDiff")
    p.add_argument("--CE_iter", type=int, default=5, help="Number of CE CEDiff steps")
    p.add_argument("--rho", type=float, default=0.9, help="rho parameter for CEDiff (rho)")
    p.add_argument("--mu", type=float, default=0.5, help="mu parameter for CEDiff")

    p.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        choices=["uniform", "quad"],
        help="Schedule type for step skipping",
    )
    p.add_argument(
        "--iter_num", type=int, default=1000, help="Number of steps when using quad schedule"
    )

    main(p.parse_args())
