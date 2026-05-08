import torch
import numpy as np
from collections import defaultdict


def compute_k2p_distance(seq1, seq2):
    transitions = 0
    transversions = 0
    sites = 0
    purines = {'A', 'G'}
    pyrimidines = {'C', 'T'}

    for a, b in zip(seq1, seq2):
        if a in '-N?' or b in '-N?':
            continue
        sites += 1
        if a != b:
            a_up, b_up = a.upper(), b.upper()
            if (a_up in purines and b_up in purines) or (a_up in pyrimidines and b_up in pyrimidines):
                transitions += 1
            else:
                transversions += 1

    if sites == 0:
        return 0.0

    P = transitions / sites
    Q = transversions / sites

    if P + Q >= 0.75:
        return float('inf')

    try:
        d = -0.5 * np.log(1 - 2 * P - Q) - 0.25 * np.log(1 - 2 * Q)
    except (ValueError, ZeroDivisionError):
        d = float('inf')

    return max(d, 0.0)


def compute_k2p_matrix(sequences, names=None):
    n = len(sequences)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = compute_k2p_distance(sequences[i], sequences[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    return dist_matrix


def compute_k2p_distance_aligned(aligned_tensor):
    aligned_tensor = aligned_tensor.long()
    n = aligned_tensor.shape[0]
    L = aligned_tensor.shape[1]

    purines = torch.tensor([0, 0, 1, 1, 0], dtype=torch.bool)
    pyrimidines = torch.tensor([0, 1, 0, 0, 1], dtype=torch.bool)

    valid_mask = (aligned_tensor >= 0) & (aligned_tensor <= 3)
    valid_i = valid_mask.unsqueeze(1).expand(n, n, L)
    valid_j = valid_mask.unsqueeze(0).expand(n, n, L)
    both_valid = valid_i & valid_j

    a = aligned_tensor.unsqueeze(1).expand(n, n, L)
    b = aligned_tensor.unsqueeze(0).expand(n, n, L)

    diff = (a != b) & both_valid
    sites = both_valid.float().sum(dim=-1)

    is_purine_a = purines[a]
    is_purine_b = purines[b]
    is_pyrimidine_a = pyrimidines[a]
    is_pyrimidine_b = pyrimidines[b]

    is_transition = diff & (
        (is_purine_a & is_purine_b) | (is_pyrimidine_a & is_pyrimidine_b)
    )
    is_transversion = diff & ~is_transition

    P = is_transition.float().sum(dim=-1) / sites.clamp(min=1)
    Q = is_transversion.float().sum(dim=-1) / sites.clamp(min=1)

    P = P.clamp(max=0.49)
    Q = Q.clamp(max=0.49)
    P_plus_Q = (P + Q).clamp(max=0.74)

    d = -0.5 * torch.log(1 - 2 * P - Q + 1e-10) - 0.25 * torch.log(1 - 2 * Q + 1e-10)
    d = d.clamp(min=0)

    diag_mask = torch.eye(n, dtype=torch.bool, device=d.device)
    d = d.masked_fill(diag_mask, 0)

    return d


class K2PDistance:
    def __init__(self):
        self.encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, '-': 4, 'N': 4}

    def encode_sequences(self, sequences):
        max_len = max(len(s) for s in sequences)
        encoded = torch.zeros(len(sequences), max_len, dtype=torch.long)
        for i, seq in enumerate(sequences):
            for j, c in enumerate(seq.upper()):
                encoded[i, j] = self.encoding.get(c, 4)
        return encoded

    def compute(self, sequences):
        encoded = self.encode_sequences(sequences)
        return compute_k2p_distance_aligned(encoded)

    def compute_from_encoded(self, encoded_tensor):
        return compute_k2p_distance_aligned(encoded_tensor)
