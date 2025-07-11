import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        H = x.shape[2]
        W = x.shape[3]
        x = x.view(-1, self.channels, H * W).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, H, W)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        # Handle size mismatch by padding or cropping
        diffY = skip_x.size()[2] - x.size()[2]
        diffX = skip_x.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, device="cpu", width_multiplier=0.25):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.width_multiplier = width_multiplier

        # Adjust channels with width_multiplier
        self.inc = DoubleConv(c_in, int(64 * width_multiplier))
        self.down1 = Down(int(64 * width_multiplier), int(128 * width_multiplier))
        self.sa1 = SelfAttention(int(128 * width_multiplier), 32)
        self.down2 = Down(int(128 * width_multiplier), int(256 * width_multiplier))
        self.sa2 = SelfAttention(int(256 * width_multiplier), 16)
        self.down3 = Down(int(256 * width_multiplier), int(256 * width_multiplier))
        self.sa3 = SelfAttention(int(256 * width_multiplier), 8)

        self.bot1 = DoubleConv(int(256 * width_multiplier), int(512 * width_multiplier))
        self.bot2 = DoubleConv(int(512 * width_multiplier), int(512 * width_multiplier))
        self.bot3 = DoubleConv(int(512 * width_multiplier), int(256 * width_multiplier))

        self.up1 = Up(int(512 * width_multiplier), int(128 * width_multiplier))
        self.sa4 = SelfAttention(int(128 * width_multiplier), 16)
        self.up2 = Up(int(256 * width_multiplier), int(64 * width_multiplier))
        self.sa5 = SelfAttention(int(64 * width_multiplier), 32)
        self.up3 = Up(int(128 * width_multiplier), int(64 * width_multiplier))
        self.sa6 = SelfAttention(int(64 * width_multiplier), 64)
        self.outc = nn.Conv2d(int(64 * width_multiplier), c_out, kernel_size=1)

    # (Keep the rest of the methods unchanged)
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        ).to(t.device)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


if __name__ == "__main__":
    model = UNet(c_in=1, c_out=1)
    img = torch.randn(1, 1, 180, 182)
    t = torch.randn(1)
    output = model(img, t)
    print(output.shape)
