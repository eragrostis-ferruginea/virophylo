import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mamba = nn.Sequential(
                nn.Linear(d_model, d_model * expand),
                nn.GELU(),
                nn.Linear(d_model * expand, d_model),
            )

        self.norm = nn.RMSNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return x + residual


class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.forward_mamba = MambaBlock(d_model, d_state, d_conv, expand)
        self.backward_mamba = MambaBlock(d_model, d_state, d_conv, expand)
        self.norm = nn.RMSNorm(d_model)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x_fwd = self.forward_mamba(x)
        x_bwd = self.backward_mamba(x.flip(dims=[1])).flip(dims=[1])
        gate = torch.sigmoid(self.gate)
        x_out = gate * x_fwd + (1 - gate) * x_bwd
        return self.norm(x_out)


class BiMambaStack(nn.Module):
    def __init__(self, d_model, n_layers=12, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.RMSNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)


class TreeHead(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, cls_tokens, seq_tokens):
        B = cls_tokens.shape[0]
        n_seqs = cls_tokens.shape[1]
        L = seq_tokens.shape[1]

        cls_tokens_normed = self.norm(cls_tokens)
        Q = self.q_proj(cls_tokens_normed)
        K = self.k_proj(seq_tokens)
        V = self.v_proj(seq_tokens)

        Q = Q.view(B, n_seqs, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, n_seqs, -1)
        out = self.out_proj(out)
        return out


class NucleotideTokenizer(nn.Module):
    def __init__(self, k=6, d_model=768, vocab_size=4096):
        super().__init__()
        self.k = k
        self.embedding = nn.Embedding(vocab_size + 5, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.sep_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, 8192, d_model) * 0.02)

    def tokenize(self, sequences):
        token_ids = []
        for seq in sequences:
            tokens = []
            for i in range(0, len(seq) - self.k + 1, self.k):
                kmer = seq[i:i + self.k]
                tid = self._kmer_to_id(kmer)
                tokens.append(tid)
            token_ids.append(tokens)
        return token_ids

    def _kmer_to_id(self, kmer):
        encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
        tid = 0
        for c in kmer.upper():
            if c in encoding:
                tid = tid * 4 + encoding[c]
            else:
                return 4
        return tid + 5

    def forward(self, sequences, max_length=8192):
        token_ids = self.tokenize(sequences)

        batch_embeddings = []
        for tokens in token_ids:
            tokens = tokens[:max_length - 2]
            ids = torch.tensor(tokens, dtype=torch.long, device=self.embedding.weight.device)
            emb = self.embedding(ids)
            cls = self.cls_token.expand(1, -1, -1).squeeze(0)
            sep = self.sep_token.expand(1, -1, -1).squeeze(0)
            emb = torch.cat([cls, emb, sep], dim=0)
            batch_embeddings.append(emb)

        max_len = max(e.shape[0] for e in batch_embeddings)
        padded = torch.zeros(len(batch_embeddings), max_len, self.cls_token.shape[-1],
                             device=self.cls_token.device)
        for i, emb in enumerate(batch_embeddings):
            padded[i, :emb.shape[0]] = emb

        padded = padded + self.pos_embedding[:, :max_len, :]
        return padded
