# softalign_scorer.py
import torch, math
from torch import nn
from bert_score.scorer import BERTScorer        # upstream API
from .sinkhorn import sinkhorn_balance          # your OT routine


class SoftAlignScorer(BERTScorer):
    """
    BERTScore variant with soft (Sinkhorn) alignment and learnable
    per-token weight MLP.  Only ~2K parameters are trainable; the
    transformer backbone stays frozen.
    """

    def __init__(self,
                 model_type="roberta-large",
                 n_sink=12,
                 tau=0.07,
                 learn_weights=True,
                 **kwargs):
        # --- keep a copy of custom args
        self.n_sink = n_sink
        self.learn_weights = learn_weights
        super().__init__(model_type=model_type, **kwargs)   # parent accepts kwargs

        # --- learnable temperature  (one scalar)
        self.tau = nn.Parameter(torch.tensor(tau))

        # --- tiny two-layer MLP for token importance
        hidden = self.model.config.hidden_size
        if learn_weights:
            self.weight_net = nn.Sequential(
                nn.Linear(hidden, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        else:                                             # dummy scalar = 1
            self.weight_net = lambda x: torch.ones(x.size(0), 1,
                                                   device=x.device)

    # -------------------------------------------------------
    #   Override the pair-wise score routine (= core metric)
    # -------------------------------------------------------
    def _pairwise_scores(self, cand_emb, ref_emb, cand_mask, ref_mask):

        # 1) cosine-similarity matrix  S  (m × n)
        S = torch.einsum("md,nd->mn", cand_emb, ref_emb)

        # 2) soft doubly-stochastic alignment  π
        π = sinkhorn_balance(S / self.tau.exp(),            # soften via τ
                             mask_x=cand_mask,
                             mask_y=ref_mask,
                             n_iter=self.n_sink)

        # 3) per-token weights   w_x, w_y  ∈  (0,1)
        w_x = self.weight_net(cand_emb).sigmoid().squeeze(-1)
        w_y = self.weight_net(ref_emb).sigmoid().squeeze(-1)

        # 4) weighted Precision / Recall / F1
        P = (π * w_x[:, None]).sum() / w_x.sum()
        R = (π * w_y[None, :]).sum() / w_y.sum()
        F = 2 * P * R / (P + R + 1e-12)
        return P, R, F

    # -------------------------------------------------------
    # convenience wrappers for saving / loading checkpoints
    # -------------------------------------------------------
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        obj = cls(**kwargs)
        obj.load_state_dict(torch.load(path, map_location="cpu"))
        return obj
