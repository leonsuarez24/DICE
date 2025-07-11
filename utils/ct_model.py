import torch
import torch.nn as nn

from . import radon


class CTModel(nn.Module):
    """CT forward and backprojection operator.

    This small wrapper builds the forward (Radon) and adjoint (iRadon) operators
    used by the diffusion samplers.  The implementation mirrors the operators
    provided in :mod:`utils.radon`.

    Parameters
    ----------
    im_size: int
        Size of the (square) input image.
    sampling_ratio: float
        Ratio of projection angles to use w.r.t ``num_angles``.
    num_angles: int, optional
        Total number of available angles.  Default: ``180``.
    sampling_method: str, optional
        Either ``"uniform"`` to uniformly subsample angles or ``"non_uniform"``
        (alias ``"random"``) to sample a random subset of angles.  Default:
        ``"uniform"``.
    """

    def __init__(
        self,
        im_size: int,
        sampling_ratio: float,
        num_angles: int = 180,
        sampling_method: str = "uniform",
    ) -> None:
        super().__init__()

        self.im_size = im_size
        self.sampling_ratio = sampling_ratio
        self.num_angles = num_angles

        full_angles = torch.linspace(0, 180, num_angles, dtype=torch.float32)
        n_select = max(1, int(round(num_angles * sampling_ratio)))

        if sampling_method in ["uniform", "uniform_angle"]:
            indices = torch.linspace(0, num_angles - 1, n_select)
            indices = indices.round().long()
        elif sampling_method in ["non_uniform", "random"]:
            indices = torch.randperm(num_angles)[:n_select]
            indices, _ = torch.sort(indices)
        else:
            raise ValueError(
                "sampling_method must be 'uniform' or 'non_uniform'"
            )

        self.theta = full_angles[indices].to(device="cuda") 

        # Radon and inverse Radon operators. We default to circle=True as the
        # training data is cropped to the unit circle.
        self.radon_op = radon.Radon(
            in_size=im_size, theta=self.theta, circle=False, parallel_computation=True, device="cuda"
        )
        self.iradon_op = radon.IRadon(
            in_size=im_size,
            theta=self.theta,
            circle=False,
            use_filter=False,
            parallel_computation=True,
            device="cuda",
        )

        self.iradon_FBP = radon.IRadon(
            in_size=im_size,
            theta=self.theta,
            circle=False,
            use_filter=True,
            parallel_computation=True,
            device="cuda",
        )

    # ------------------------------------------------------------------
    def forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward Radon transform."""

        return self.radon_op(x)

    # ------------------------------------------------------------------
    def transpose_pass(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint (backprojection) of the forward operator."""

        return self.iradon_op(y)
    
    def FBP(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint (backprojection) of the forward operator."""

        return self.iradon_FBP(y)

