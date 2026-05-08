import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RCEConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2 if padding == 'same' else 0

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, bias=False)
        self.conv_rc = nn.Conv1d(in_channels, out_channels, kernel_size,
                                 padding=self.padding, bias=False)

        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        out_fwd = self.conv(x)
        x_rc = x.flip(dims=[1])
        out_rc = self.conv_rc(x_rc).flip(dims=[1])
        return out_fwd + out_rc + self.bias.unsqueeze(0)


class ByteNetBlock(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size, dilation=1):
        super().__init__()
        self.conv1 = RCEConv1d(d_model, d_inner, kernel_size)
        self.conv2 = nn.Conv1d(d_inner, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.norm2(x + residual)
        return x


class DilatedByteNet(nn.Module):
    def __init__(self, d_model=960, d_inner=960, n_blocks=40,
                 kernel_size=9, dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [1, 5] * (n_blocks // 2)

        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            d = dilations[i % len(dilations)]
            self.blocks.append(ByteNetBlock(d_model, d_inner, kernel_size, dilation=d))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ViralPhyloGPN(nn.Module):
    def __init__(self, window_size=241, d_model=960, d_inner=960,
                 n_blocks=40, kernel_size=9, n_bases=5):
        super().__init__()
        self.window_size = window_size
        self.n_bases = n_bases
        self.d_model = d_model

        self.one_hot_embed = nn.Linear(n_bases, d_model, bias=False)

        self.bytenet = DilatedByteNet(d_model, d_inner, n_blocks, kernel_size)

        self.output_conv = RCEConv1d(d_model, d_model, kernel_size)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_act = nn.GELU()

        self.rate_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.LayerNorm(d_model // 4),
            nn.Linear(d_model // 4, 6),
        )
        self.freq_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.LayerNorm(d_model // 4),
            nn.Linear(d_model // 4, 4),
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(d_model, d_model // 8),
            nn.GELU(),
            nn.Linear(d_model // 8, 1),
        )

    def encode_onehot(self, seq_tensor):
        encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, '-': 4, 'N': 4}
        n = seq_tensor.shape[0]
        L = seq_tensor.shape[1]
        onehot = torch.zeros(n, L, self.n_bases, device=seq_tensor.device)
        for i in range(n):
            for j in range(L):
                c = seq_tensor[i, j].item()
                if 0 <= c < self.n_bases:
                    onehot[i, j, c] = 1.0
        return onehot

    def forward(self, seq_onehot):
        if seq_onehot.dim() == 2:
            seq_onehot = self.encode_onehot(seq_onehot)

        x = self.one_hot_embed(seq_onehot)
        x = x.transpose(1, 2)
        x = self.bytenet(x.transpose(1, 2)).transpose(1, 2)
        x = x.transpose(1, 2)
        x = self.output_conv(x)
        x = x.transpose(1, 2)
        x = self.output_norm(self.output_act(x))

        site_embeddings = x

        raw_rates = self.rate_head(site_embeddings)
        raw_freq = self.freq_head(site_embeddings)
        raw_alpha = self.alpha_head(site_embeddings).squeeze(-1)

        return raw_rates, raw_freq, raw_alpha, site_embeddings

    def predict_gtr_params(self, seq_onehot):
        raw_rates, raw_freq, raw_alpha, _ = self.forward(seq_onehot)
        rates = F.softplus(raw_rates)
        rates = rates * 6.0 / rates.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        frequencies = F.softmax(raw_freq, dim=-1)
        alpha = F.softplus(raw_alpha).clamp(min=0.1, max=10.0)
        return rates, frequencies, alpha
