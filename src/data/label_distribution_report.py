import json, argparse
from pathlib import Path
from collections import Counter, OrderedDict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)

def collect_labels(fp):
    c = Counter()
    for obj in read_jsonl(fp):
        spans = obj.get("gold_spans") or obj.get("spans") or obj.get("entities") or []
        for x in spans:
            if isinstance(x, dict): lab = x.get("label") or x.get("type") or x.get("tag") or ""
            else: lab = x[2] if len(x) >= 3 else ""
            if lab: c[lab] += 1
    return c

def plot_bar(ax, labels, values, title, ylabel="Count"):
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(labels, rotation=25, ha="right")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="thư mục có train/val/test.jsonl")
    ap.add_argument("--outdir", required=True, help="nơi lưu hình/CSV")
    ap.add_argument("--include_zero", type=int, default=1, help="1=hiện cả nhãn bằng 0 trong 1 split")
    args = ap.parse_args()

    d = Path(args.data_dir)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": d/"train.jsonl",
        "val":   d/"val.jsonl",
        "test":  d/"test.jsonl",
    }
    for k,p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Thiếu file: {p}")

    # đếm từng split
    cnts = {split: collect_labels(p) for split,p in paths.items()}
    # danh sách nhãn đầy đủ (union)
    all_labels = sorted(set().union(*[set(c.keys()) for c in cnts.values()]))

    # build dataframe (đếm)
    data = {split: [cnts[split].get(lb, 0) for lb in all_labels] for split in ["train","val","test"]}
    df_counts = pd.DataFrame(data, index=all_labels)
    df_counts["total"] = df_counts.sum(axis=1)
    df_counts = df_counts.sort_values("total", ascending=False)

    # nếu include_zero=0, bỏ nhãn total=0 (hiếm khi xảy ra)
    if not args.include_zero:
        df_counts = df_counts[df_counts["total"] > 0]

    # tỉ lệ theo hàng
    df_props = df_counts[["train","val","test"]].div(df_counts["total"], axis=0).fillna(0)

    # Lưu CSV
    df_counts.to_csv(out/"label_counts_by_split.csv", encoding="utf-8-sig")
    df_props.to_csv(out/"label_proportions_by_split.csv", encoding="utf-8-sig")

    # Vẽ: tổng (all), và từng split cạnh nhau
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    plot_bar(ax, df_counts.index.tolist(), df_counts["total"].tolist(), "Label counts (ALL)")
    plt.tight_layout(); plt.savefig(out/"labels_all_counts.png", dpi=180); plt.close()

    plt.figure(figsize=(12, 5))
    xs = range(len(df_counts))
    width = 0.27
    ax = plt.gca()
    ax.bar([x - width for x in xs], df_counts["train"].tolist(), width, label="train")
    ax.bar(xs,                              df_counts["val"].tolist(),   width, label="val")
    ax.bar([x + width for x in xs], df_counts["test"].tolist(),  width, label="test")
    ax.set_xticks(list(xs)); ax.set_xticklabels(df_counts.index.tolist(), rotation=25, ha="right")
    ax.set_title("Label counts per split"); ax.set_ylabel("Count"); ax.legend()
    plt.tight_layout(); plt.savefig(out/"labels_counts_per_split.png", dpi=180); plt.close()

    # Vẽ: stacked proportions
    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    bottom = [0]*len(df_props)
    for split in ["train","val","test"]:
        vals = df_props[split].tolist()
        ax.bar(df_props.index.tolist(), vals, bottom=bottom, label=split)
        bottom = [b+v for b,v in zip(bottom, vals)]
    ax.set_title("Label proportions per split")
    ax.set_ylabel("Proportion")
    ax.set_xticklabels(df_props.index.tolist(), rotation=25, ha="right")
    ax.legend()
    plt.tight_layout(); plt.savefig(out/"labels_proportions_per_split.png", dpi=180); plt.close()

    print("[DONE] Figures & CSV ->", out)

if __name__ == "__main__":
    main()
