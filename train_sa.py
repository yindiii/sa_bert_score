import os, argparse, random, numpy as np, pandas as pd, torch
from bert_score.softalign_scorer import SoftAlignScorer      # ← fixed

# ---------- 1. Robust DA loader (any header variant) ----------
def load_da(root="reproduce"):
    fn = os.path.join(root, "wmt17", "manual-evaluation", "DA-seglevel.csv")
    df = pd.read_csv(fn, sep=r"\s+", engine="python", keep_default_na=False)
    up = {c.upper(): c for c in df.columns}
    canonical = {"LP":["LP"],
                 "SYSTEM":["SYSTEM"],
                 "SEGMENT":["SID","SEGID","SEGMENT","SEG"],
                 "SCORE":["HUMAN","SCORE","DA","Z"]}
    rename = {}
    for want, alts in canonical.items():
        hit = next((up[a] for a in alts if a in up), None)
        if hit is None:
            raise ValueError(f"Column {want} missing. header={list(df.columns)}")
        rename[hit] = want
    df = df.rename(columns=rename)[list(canonical)]
    df["SEGMENT"] = df["SEGMENT"].astype(int)
    df["SCORE"]   = df["SCORE"].astype(float)
    return df

# ---------- 2. Reference / candidate cache ----------
def cache_sent(year, lp, root):
    src, tgt = lp.split("-")
    y4 = year if year >= 2000 else 2000 + year
    ref_fp = os.path.join(root, f"wmt{year}", "input", f"wmt{year}-metrics-task",
                          f"wmt{year}-submitted-data", "txt", "references",
                          f"newstest{y4}-{src}{tgt}-ref.{tgt}")
    refs = open(ref_fp).read().splitlines()
    sys_dir = os.path.join(root, f"wmt{year}", "input", f"wmt{year}-metrics-task",
                           f"wmt{year}-submitted-data", "txt", "system-outputs",
                           f"newstest{y4}", lp)
    cand = {fn[13:-len(f'.{lp}')]:
            open(os.path.join(sys_dir, fn)).read().splitlines()
            for fn in os.listdir(sys_dir)}
    return refs, cand

# ---------- 3. Build training triples ----------
def build_triples(df, year, root):
    buf, triples = {}, []
    for _, r in df.iterrows():
        lp, sys_, seg = r["LP"], r["SYSTEM"], r["SEGMENT"]
        score = r["SCORE"] / 100.0
        if lp not in buf:
            buf[lp] = cache_sent(year, lp, root)
        refs, cands = buf[lp]
        if sys_ not in cands:            # skip combined or missing systems
            continue
        triples.append((refs[seg-1], cands[sys_][seg-1], score))
    if not triples:
        raise RuntimeError("No training pairs found.")
    return triples

# ---------- 4. Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=17)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    ap.add_argument("--ckpt", default="sa_sinkhorn.pt")
    ap.add_argument("--root", default="reproduce")
    ap.add_argument("--lang_pairs", nargs="+",
                    default=["cs-en","de-en","fi-en","lv-en","ru-en","tr-en","zh-en"])
    args = ap.parse_args()
    dev = "cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu"

    scorer = SoftAlignScorer("roberta-large", tau=0.07, n_sink=12,
                             learn_weights=True, idf=False, device=dev)
    opt = torch.optim.AdamW(list(scorer.weight_net.parameters()) + [scorer.tau], lr=5e-5)

    df = load_da(args.root)
    df = df[df["LP"].isin(args.lang_pairs)]
    triples = build_triples(df, args.year, args.root)

    for ep in range(args.epochs):
        random.shuffle(triples); losses=[]
        for i in range(0, len(triples), args.batch_size):
            refs, cands, scores = zip(*triples[i:i+args.batch_size])
            _, _, F = scorer.score(list(cands), list(refs),
                                   batch_size=len(scores), grad=True)
            loss = torch.nn.functional.mse_loss(
                       F.to(dev), torch.tensor(scores, device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"epoch {ep+1}: MSE={np.mean(losses):.4f}")

    scorer.save_pretrained(args.ckpt)
    print("✓ Saved →", args.ckpt)

if __name__ == "__main__":
    main()
