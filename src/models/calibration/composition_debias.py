import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositionDebias(nn.Module):
    def __init__(self, embed_dim, n_composition_features=20):
        super().__init__()
        self.regressor = nn.Linear(n_composition_features, embed_dim, bias=True)
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, embeddings, composition_features):
        composition_signal = self.regressor(composition_features)
        gate = torch.sigmoid(self.gate)
        e_debiased = embeddings - gate * composition_signal
        return e_debiased
