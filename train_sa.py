# train_sa.py
import os, argparse, random
import torch, numpy as np, pandas as pd
from tqdm import tqdm

# --------------- 1.  import your scorer -------------------
from bert_score.softalign_scorer import SoftAlignScorer


# --------------- 2.  utility: freeze the backbone ---------
def freeze_backbone(model):
    for p in model.bert_model.parameters():          # RoBERTa params
        p.requires_grad = False                     # :contentReference[oaicite:1]{index=1}


# --------------- 3.  load WMT-17 seg-level gold ---------
def load_wmt17_seg(lang_pairs, root="reproduce"):
    fn = os.path.join(root, "wmt17", "manual-evaluation", "DA-seglevel.csv")
    df = pd.read_csv(fn, delimiter=" ")
    return df[df["LP"].isin(lang_pairs)]            # :contentReference[oaicite:2]{index=2}


# --------------- 4.  cache all references & candidates ---
def build_sentence_cache(year, lp, root="reproduce"):
    f, s = lp.split("-")
    ref_fp = os.path.join(root, f"wmt{year}", "input",
                          f"wmt{year}-metrics-task",
                          "wmt{year}-submitted-data", "txt",
                          "references",
                          f"newstest{year}-{f}{s}-ref.{s}")
    refs = open(ref_fp).read().splitlines()         # reference list

    sys_dir = os.path.join(root, f"wmt{year}", "input",
                           f"wmt{year}-metrics-task",
                           "wmt{year}-submitted-data", "txt",
                           "system-outputs", f"newstest{year}", lp)
    cand_cache = {}
    for fn in os.listdir(sys_dir):
        sys_name = fn[13:-len(f".{lp}")]
        cand_cache[sys_name] = open(
            os.path.join(sys_dir, fn)
        ).read().splitlines()
    return refs, cand_cache                         # :contentReference[oaicite:3]{index=3}


# --------------- 5.  generator of (ref, cand, gold) ------
def iterate_examples(df, year, root="reproduce"):
    cache = {}
    for _, row in df.iterrows():
        lp, sys_name, seg_id, score = row["LP"], row["SYSTEM"], row["SEGID"], row["HUMAN"]
        if lp not in cache:
            cache[lp] = build_sentence_cache(year, lp, root)
        refs, cands = cache[lp]
        yield refs[seg_id - 1], cands[sys_name][seg_id - 1], score / 100.0


# --------------- 6.  main training loop ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=17)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lang_pairs", nargs="+",
                    default=["cs-en","de-en","fi-en","lv-en","ru-en","tr-en","zh-en"])
    ap.add_argument("--ckpt", default="sa_roberta_large_sinkhorn.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scorer = SoftAlignScorer(model_type="roberta-large",
                          tau=0.07, n_sink=12, learn_weights=True,
                          idf=False).to(device)
    optim = torch.optim.AdamW(
        [p for p in scorer.parameters() if p.requires_grad], lr=3e-4)

    df = load_wmt17_seg(args.lang_pairs)
    data = list(iterate_examples(df, args.year))
    for ep in range(args.epochs):
        random.shuffle(data)
        losses = []
        for idx in range(0, len(data), args.batch_size):
            batch = data[idx:idx+args.batch_size]
            refs, cands, scores = zip(*batch)
            P,R,F = scorer.score(list(cands), list(refs),
                                 batch_size=len(batch))
            loss = torch.nn.functional.mse_loss(
                       F.to(device),
                       torch.tensor(scores, device=device))
            optim.zero_grad(); loss.backward(); optim.step()
            losses.append(loss.item())
        print(f"epoch {ep+1}: MSE = {np.mean(losses):.4f}")
    torch.save({"model": scorer.state_dict()}, args.ckpt)
    print(f"Checkpoint saved to {args.ckpt}")


if __name__ == "__main__":
    main()
