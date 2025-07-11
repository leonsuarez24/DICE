import torch
from tqdm import tqdm
from utils.ddpm import get_named_beta_schedule
import numpy as np



class DICE:

    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=256,
        device="cuda",
        schedule_name="cosine",
        channels=1,
        rho=0.9,
        mu=0.1,
        skip_type="uniform",
        iter_num=1000,
    ):

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.schedule_name = schedule_name
        self.channels = channels

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.skip_type = skip_type
        self.iter_num = iter_num
        self.skip = self.noise_steps // self.iter_num if self.iter_num else 1
        self.rho = rho
        self.mu = mu
        self.alpha_hat_prev = torch.cat(
            [torch.tensor([1.0], device=self.device), self.alpha_hat[:-1]]
        )

    def prepare_noise_schedule(self):
        if self.schedule_name == "cosine":
            return torch.tensor(
                get_named_beta_schedule("cosine", self.noise_steps, self.beta_end).copy(),
                dtype=torch.float32,
            )

        if self.schedule_name == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def _make_schedule(self):
        if not self.iter_num:
            return list(range(self.noise_steps - 1, 0, -1))
        if self.skip_type == "uniform":
            seq = [i * self.skip for i in range(self.iter_num)]
            if self.skip > 1:
                seq.append(self.noise_steps - 1)
        elif self.skip_type == "quad":
            seq = np.sqrt(np.linspace(0, self.noise_steps**2, self.iter_num))
            seq = [int(s) for s in list(seq)]
            seq[-1] = seq[-1] - 1
        else:
            seq = list(range(self.noise_steps))
        return sorted(set(seq), reverse=True)


    
    def sample(
        self,
        model,
        y,
        transpose_pass,
        forward_pass,
        CG_iter,
        CE_iter,
    ):

        x = torch.randn((1, self.channels, self.img_size, self.img_size)).to(self.device)

        seq = self._make_schedule()
        pbar = tqdm(seq, position=0)

        for i in pbar:
            t = (torch.ones(1) * i).long().to(self.device)
            W = torch.stack([x.clone().detach(), x.clone().detach()], dim=0)
            MU = [self.mu, 1 - self.mu]

            for K in range(CE_iter):
                prev_w = W.clone()
                W_prima = 2 * self.F_func(W, forward_pass, transpose_pass, y, CG_iter, i, model) - W
                W_prima = 2 * self.G_func(W_prima, MU) - W_prima
                W = (1 - self.rho) * W + self.rho * W_prima
                fp_err = torch.linalg.norm(W - prev_w)
          

            x0 = W[0] * MU[0] + W[1] * MU[1]

            ############################################################

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            alpha_hat_prev = self.alpha_hat_prev[t]
            x = torch.sqrt(alpha_hat_prev) * x0 + torch.sqrt(1 - alpha_hat_prev) * noise

            difference = y - forward_pass(x)
            norm = torch.linalg.norm(difference)

            

            pbar.set_description(
                f"Sampling - Step {i} - Consistency: {norm.item():.4f} - FP err: {fp_err.item():.4f}"
            )

        return x

    def F_func(self, W, forward_pass, transpose_pass, y, CG_iter, i, model):
        return torch.stack(
            [
                self.F_t(forward_pass, transpose_pass, y, CG_iter, W[0], i),
                self.H_t(model, i, W[1]),
            ],
            dim=0,
        )

    def G_func(self, W, MU):
        return torch.stack(
            [
                W[0] * MU[0] + W[1] * MU[1],
                W[0] * MU[0] + W[1] * MU[1],
            ],
            dim=0,
        )



    def F_t(self, forward_pass, transpose_pass, y, itera, x, i):
        u = x.clone()
        t = (torch.ones(1) * i).long().to(self.device)
        lambda_t = float((1 - self.alpha_hat[t]) / self.alpha_hat[t])

        def A_fn(u):
            with torch.inference_mode():
                return transpose_pass(forward_pass(u)) + u * lambda_t

        b = transpose_pass(y) + u * lambda_t

        u = conjugate_gradient(A_fn, b, x0=x, n_iter=itera).detach()

        return u

    def H_t(self, model, i, x):
        model.eval()
        with torch.no_grad():
            t = torch.tensor([i], device=self.device).long()
            predicted_noise = model(x, t)

            alpha_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
            sqrt_alpha_hat = torch.sqrt(alpha_hat)
            sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)

            x = (x - sqrt_one_minus_alpha_hat * predicted_noise) / sqrt_alpha_hat
            x = x.clamp(-1.0, 1.0)

        return x

    



def conjugate_gradient(
    apply_A,
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    n_iter: int = 40,
    tol: float = 1e-4,
) -> torch.Tensor:

    with torch.inference_mode():
        x = torch.zeros_like(b) if x0 is None else x0.clone()
        r = b - apply_A(x)
        p = r.clone()
        rs_old = torch.sum(r * r, dim=list(range(1, r.ndim)), keepdim=True)

        for _ in range(n_iter):
            Ap = apply_A(p)
            alpha = rs_old / (torch.sum(p * Ap, dim=list(range(1, r.ndim)), keepdim=True) + 1e-12)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.sum(r * r, dim=list(range(1, r.ndim)), keepdim=True)
            if torch.sqrt(rs_new.mean()) < tol:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
    return x
