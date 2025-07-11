import torch
from utils.ddpm import Diffusion
import matplotlib.pyplot as plt
from utils.utils import set_seed

set_seed(0)
from guided_diffusion.script_util import create_model

path = "weights/d_fastmri_e_1000_bs_2_lr_3e-05_seed_2_img_256_schedule_cosine_gpu_0_c_1_si_100/checkpoints/latest.pth.tar"


model = create_model(image_size=256,num_channels=64,num_res_blocks=3, input_channels=1).to('cuda')
diff = Diffusion(device="cuda", img_size=256, noise_steps=1000, schedule_name="cosine")

checkpoint = torch.load(path, map_location="cuda")
model.load_state_dict(checkpoint["model_state"])
model.eval()

x = diff.sample(model, n=1)

print(x.shape)

plt.figure(figsize=(6, 6))
plt.imshow(x[0, 0].cpu().detach().numpy(), cmap="gray")
plt.axis("off")
plt.colorbar()
plt.show()
