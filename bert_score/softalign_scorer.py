from bert_score.scorer import BERTScorer
from .sinkhorn import sinkhorn_balance 
class SoftAlignScorer(BERTScorer):
    def _pairwise_scores(self, cand_emb, ref_emb, cand_mask, ref_mask):
        # 1) cosine-sim matrix S  [m × n]
        S = cand_emb @ ref_emb.T
        # 2) doubly-stochastic transport π  via Sinkhorn
        π = sinkhorn_balance(S / self.tau, mask_x=cand_mask, mask_y=ref_mask,
                             n_iter=self.n_sink)
        # 3) optional token-weights w_x, w_y from your 2-layer MLP
        w_x, w_y = self.weight_net(cand_emb).sigmoid(), self.weight_net(ref_emb).sigmoid()
        # 4) weighted Precision / Recall
        P = (π * w_x[:, None]).sum() / w_x.sum()
        R = (π * w_y[None, :]).sum() / w_y.sum()
        F = 2*P*R/(P+R+1e-12)
        return P, R, F
