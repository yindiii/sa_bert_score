# sinkhorn.py
import torch

def sinkhorn_balance(S, mask_x, mask_y, n_iter=10):
    """
    Differentiable Sinkhorn-Knopp that returns a (soft) doubly-stochastic
    transport matrix π ∈ Δ^{m×n}.
    """
    # large negative logits where tokens are padding
    S = S.masked_fill(~mask_x[:, None], -1e4).masked_fill(~mask_y[None, :], -1e4)
    log_pi = S                                           # initialise with logits
    for _ in range(n_iter):
        log_pi = log_pi - torch.logsumexp(log_pi, 1, keepdim=True)
        log_pi = log_pi - torch.logsumexp(log_pi, 0, keepdim=True)
    return log_pi.exp()