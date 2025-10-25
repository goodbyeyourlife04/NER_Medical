# -*- coding: utf-8 -*-
"""
So sánh 2 mô hình NER (XLM-R / PhoBERT) đã train:
- Đọc training_log.csv của mỗi run -> vẽ các đường theo epoch:
  + train_loss
  + val_token_f1(%)
  + val_span_micro_f1(%)
  + val_span_macro_f1(%)
- Đọc test_span_report.txt (hoặc val_span_report.txt nếu chưa có test) -> vẽ F1(%) từng nhãn (bar nhóm)

Lưu tất cả hình vào: --outdir (mặc định: outputs/figures)

Usage:
python src/viz/compare_models.py \
  --runs models/xlmr-span-bilstm-crf/run_YYYYMMDD-HHMMSS models/phobert-span-bilstm-crf/run_YYYYMMDD-HHMMSS \
  --names "XLM-R_base" "PhoBERT_base" \
  --outdir outputs/figures
"""
import argparse, csv, os, re
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_training_log_csv(p: Path) -> Dict[str, list]:
    d = {"epoch": [], "train_loss": [], "val_token_f1(%)": [], "val_span_micro_f1(%)": [], "val_span_macro_f1(%)": []}
    if not p.exists():
        return d
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # cột có thể có/không; cố gắng ép sang float
            d["epoch"].append(float(row.get("epoch", "nan")))
            for k in ["train_loss", "val_token_f1(%)", "val_span_micro_f1(%)", "val_span_macro_f1(%)"]:
                v = row.get(k, "")
                try:
                    d[k].append(float(v))
                except:
                    d[k].append(float("nan"))
    return d

def read_span_report_txt(p: Path) -> List[Tuple[str, float]]:
    """
    Đọc ..._span_report.txt, mỗi dòng dạng:
    label........  P(%)  R(%)  F1(%)  support
    -> trả về list [(label, f1), ...]
    """
    if not p.exists():
        return []
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("label") or line.startswith("---"):
                continue
            # tách dựa trên nhiều khoảng trắng
            parts = re.split(r"\s{2,}|\t+", line)
            if len(parts) >= 4:
                label = parts[0].strip()
                try:
                    f1 = float(parts[3])
                except:
                    # một số bản ghi có thứ tự khác: label  P  R  F1  support
                    # thử bắt số cuối cùng trước support
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    f1 = float(nums[-2]) if len(nums) >= 2 else float("nan")
                rows.append((label, f1))
    return rows

def ensure_out(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def plot_lines_compare(xlm: Dict[str, list], pho: Dict[str, list], names: Tuple[str, str], outdir: Path):
    ensure_out(outdir)

    def lineplot(key: str, title: str, ylabel: str, fname: str):
        plt.figure()
        if any(x==x for x in xlm.get(key, [])):  # check not all NaN
            plt.plot(xlm["epoch"], xlm[key], marker="o", label=names[0])
        if any(x==x for x in pho.get(key, [])):
            plt.plot(pho["epoch"], pho[key], marker="o", label=names[1])
        plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(title)
        plt.grid(True, linestyle="--", alpha=.5); plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=180); plt.close()

    lineplot("train_loss", "Train Loss", "Loss", "cmp_train_loss.png")
    lineplot("val_token_f1(%)", "Token F1 (seqeval) – validation", "F1 (%)", "cmp_val_token_f1.png")
    lineplot("val_span_micro_f1(%)", "Span Micro-F1 – validation", "F1 (%)", "cmp_val_span_micro_f1.png")
    lineplot("val_span_macro_f1(%)", "Span Macro-F1 – validation", "F1 (%)", "cmp_val_span_macro_f1.png")

def plot_bars_per_label(xlm_rows: List[Tuple[str,float]], pho_rows: List[Tuple[str,float]],
                        names: Tuple[str,str], outdir: Path, fname: str, title: str):
    ensure_out(outdir)
    # gộp theo union label
    d1 = {k:v for k,v in xlm_rows}
    d2 = {k:v for k,v in pho_rows}
    labels = sorted(set(d1.keys()) | set(d2.keys()))
    if not labels:
        return
    x = list(range(len(labels)))
    w = 0.38
    y1 = [d1.get(lb, float("nan")) for lb in labels]
    y2 = [d2.get(lb, float("nan")) for lb in labels]
    plt.figure(figsize=(max(8,0.5*len(labels)), 5))
    plt.bar([i-w/2 for i in x], y1, width=w, label=names[0])
    plt.bar([i+w/2 for i in x], y2, width=w, label=names[1])
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("F1 (%)"); plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / fname, dpi=180); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=2, required=True,
                    help="Đường dẫn tới 2 thư mục run: .../run_YYYYmmdd-HHMMSS")
    ap.add_argument("--names", nargs=2, default=["XLM-R", "PhoBERT"], help="Tên hiển thị của 2 mô hình")
    ap.add_argument("--outdir", default="outputs/figures", help="Thư mục xuất hình")
    ap.add_argument("--split", choices=["test","val"], default="test",
                    help="Lấy per-label F1 từ test_span_report.txt (mặc định) hoặc val_span_report.txt")
    args = ap.parse_args()

    run_a = Path(args.runs[0]); run_b = Path(args.runs[1])
    outdir = Path(args.outdir)

    # --- đọc training logs
    xlm_log = read_training_log_csv(run_a / "training_log.csv")
    pho_log = read_training_log_csv(run_b / "training_log.csv")
    plot_lines_compare(xlm_log, pho_log, (args.names[0], args.names[1]), outdir)

    # --- đọc per-label F1
    report_name = f"{args.split}_span_report.txt"
    if not (run_a / report_name).exists():
        report_name = "val_span_report.txt"  # fallback
    xlm_rows = read_span_report_txt(run_a / report_name)
    pho_rows = read_span_report_txt(run_b / report_name)

    # bar so sánh F1 từng nhãn
    title = f"Per-label Strict-Span F1 (%) – {args.split.upper()}"
    plot_bars_per_label(xlm_rows, pho_rows, (args.names[0], args.names[1]),
                        outdir, f"cmp_{args.split}_perlabel_f1.png", title)

    print(f"[DONE] Figures saved to: {outdir}")

if __name__ == "__main__":
    main()
