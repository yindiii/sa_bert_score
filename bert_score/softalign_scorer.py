import torch, math
from torch import nn
from transformers import AutoModel, AutoTokenizer
from bert_score.scorer import BERTScorer
from .sinkhorn import sinkhorn_balance

class SoftAlignScorer(BERTScorer):
    """BERTScore with differentiable Sinkhorn alignment + learnable MLP."""

    def __init__(self, model_type="roberta-large", n_sink=12, tau=0.07,
                 learn_weights=True, idf=False, device="cpu", **kw):
        super().__init__(model_type=model_type, idf=idf, device=device, **kw)

        # expose tokenizer / model for our own forward
        self.tok: AutoTokenizer = self._tokenizer
        self.bert: AutoModel    = self._model
        self.bert.eval().to(device)

        self.n_sink = n_sink
        self.tau    = nn.Parameter(torch.tensor(tau))
        h = self.bert.config.hidden_size
        self.weight_net = (
            nn.Sequential(nn.Linear(h,64), nn.ReLU(), nn.Linear(64,1)).to(device)  # ← add .to(device)
            if learn_weights else
            lambda x: torch.ones(x.size(0), 1, device=device)
        )

    # ---------- utilities ------------------------------------------------
    @staticmethod
    def _strip_special(hiddens, attn):
        """remove CLS/SEP/PAD positions"""
        keep = attn.bool() & (hiddens.abs().sum(-1) != 0)
        return [hiddens[b, keep[b]] for b in range(hiddens.size(0))]

    # ---------- grad-enabled forward ------------------------------------
    def _forward_scores(self, cands, refs, batch_size):
        scores = []
        for i in range(0, len(cands), batch_size):
            batch_c = cands[i:i+batch_size]
            batch_r = refs [i:i+batch_size]

            # tokenise
            enc_c = self.tok(batch_c,  return_tensors="pt",
                             padding=True, truncation=True).to(self.device)
            enc_r = self.tok(batch_r,  return_tensors="pt",
                             padding=True, truncation=True).to(self.device)

            # embeddings
            hid_c = self.bert(**enc_c, output_hidden_states=True)[0]
            hid_r = self.bert(**enc_r, output_hidden_states=True)[0]

            # strip specials → list[tensor]
            c_list = self._strip_special(hid_c, enc_c["attention_mask"])
            r_list = self._strip_special(hid_r, enc_r["attention_mask"])

            # per-sentence Sinkhorn alignment
            for ce,re in zip(c_list,r_list):
                S  = torch.einsum("md,nd->mn", ce, re)
                pi = sinkhorn_balance(S / self.tau.exp(),
                                      mask_x=torch.ones(len(ce),dtype=torch.bool,device=self.device),
                                      mask_y=torch.ones(len(re),dtype=torch.bool,device=self.device),
                                      n_iter=self.n_sink)
                wx = self.weight_net(ce).sigmoid().squeeze(-1)
                wy = self.weight_net(re).sigmoid().squeeze(-1)
                P  = (pi * wx[:,None]).sum() / wx.sum()
                R  = (pi * wy[None,:]).sum() / wy.sum()
                F  = 2*P*R/(P+R+1e-12)
                scores.append(F)
        return torch.stack(scores)      # [N]

    # ---------- public API ----------------------------------------------
    def score(self, cands, refs, batch_size=64, grad=False):
        if not grad:                      # inference path (parent, fast)
            return super().score(cands, refs, batch_size=batch_size)
        F = self._forward_scores(cands, refs, batch_size)
        return F, F, F                   # keep caller signature

    # keep near the bottom of the class
    def save_pretrained(self, path):
        torch.save({
            "tau": self.tau.detach().cpu(),
            "mlp": self.weight_net.state_dict()
        }, path)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        obj = cls(**kwargs)
        ckpt = torch.load(path, map_location="cpu")
        obj.tau.data.copy_(ckpt["tau"])
        obj.weight_net.load_state_dict(ckpt["mlp"])
        return obj

