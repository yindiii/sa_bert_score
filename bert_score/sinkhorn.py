import torch

def sinkhorn_balance(S, mask_x, mask_y, n_iter=10, eps=1e-3):
    S = S.masked_fill(~mask_x[:,None], -1e4).masked_fill(~mask_y[None,:], -1e4)
    log_π = S 
    for _ in range(n_iter):
        log_π = log_π - torch.logsumexp(log_π, dim=1, keepdim=True)
        log_π = log_π - torch.logsumexp(log_π, dim=0, keepdim=True)
    return log_π.exp()
