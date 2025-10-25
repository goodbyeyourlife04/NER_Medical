# -*- coding: utf-8 -*-
"""
Infer NER (BiLSTM-CRF) với XLM-R, xử lý long document bằng sliding windows.
Xuất JSONL: {"article_id": "article_XXXX", "text": "...", "spans": [[s,e,"LABEL"], ...]}
"""

import argparse, json, re, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

from src.train_span_bilstm_crf import BiLSTMCRF, snap_refine_pred_char_spans


def _resolve_run_dir(ckpt_dir: Path) -> Path:
    ckpt_dir = ckpt_dir.resolve()
    if (ckpt_dir / "labels.txt").exists() and (ckpt_dir / "tokenizer").exists():
        return ckpt_dir
    candidates = sorted([p for p in ckpt_dir.glob("run_*") if p.is_dir()], reverse=True)
    for p in candidates:
        if (p / "labels.txt").exists() and (p / "tokenizer").exists():
            return p
    raise FileNotFoundError(f"Không tìm thấy run_* trong {ckpt_dir}")


def _load_id2tag(run_dir: Path):
    labels_txt = run_dir / "labels.txt"
    with labels_txt.open("r", encoding="utf-8") as f:
        types = [ln.strip() for ln in f if ln.strip()]
    tag2id = {"O": 0}
    for t in types:
        tag2id[f"B-{t}"] = len(tag2id)
        tag2id[f"I-{t}"] = len(tag2id)
    id2tag = {v: k for k, v in tag2id.items()}
    return id2tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--articles_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    run_dir = _resolve_run_dir(ckpt_dir)
    id2tag = _load_id2tag(run_dir)

    # Tokenizer & vocab size
    tok = AutoTokenizer.from_pretrained(str(run_dir / "tokenizer"), use_fast=True, local_files_only=True)
    vocab_size = len(tok.get_vocab())

    # Model checkpoint
    ckpts = sorted(run_dir.glob("best_ep*_f*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"Không thấy checkpoint trong {run_dir}")
    ckpt = torch.load(ckpts[-1], map_location="cpu")
    model_name = ckpt.get("args", {}).get("model_name_or_path", "xlm-roberta-base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMCRF(model_name, num_labels=len(id2tag)).to(device)
    model.backbone.resize_token_embeddings(vocab_size)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    eff_max_len = min(getattr(model.backbone.config, "max_position_embeddings", 512), 512)
    stride = max(0, args.stride)
    print(f"[INFO] run_dir={run_dir}, checkpoint={ckpts[-1].name}, eff_max_len={eff_max_len}, stride={stride}")

    files = sorted(Path(args.articles_dir).glob("article_*.txt"))
    if args.limit:
        files = files[:args.limit]

    n_items = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as fout, torch.no_grad():
        for p in files:
            text = p.read_text(encoding="utf-8")
            m = re.match(r"article_(\d{4})\.txt$", p.name)
            aid = f"article_{m.group(1)}" if m else p.stem

            enc = tok(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=eff_max_len,
                stride=stride,
                return_overflowing_tokens=True,
            )

            agg = set()
            for i in range(len(enc["input_ids"])):
                ids = torch.tensor([enc["input_ids"][i]], dtype=torch.long)
                att = torch.tensor([enc["attention_mask"][i]], dtype=torch.long)
                offs = enc["offset_mapping"][i]

                try:
                    path = model(input_ids=ids.to(device), attention_mask=att.to(device))[0]
                except RuntimeError as ex:
                    print("[WARN] GPU lỗi, fallback CPU:", ex)
                    path = model(input_ids=ids.cpu(), attention_mask=att.cpu())[0]

                L = int(att.sum().item())
                offs = offs[:L]

                spans, cur = [], None
                for t_id, (s, e) in zip(path[:L], offs):
                    if s == e == 0:
                        continue
                    tag = id2tag.get(int(t_id), "O")
                    if tag.startswith("B-"):
                        if cur: spans.append(cur)
                        cur = [int(s), int(e), tag[2:]]
                    elif tag.startswith("I-") and cur and tag[2:] == cur[2]:
                        cur[1] = int(e)
                    else:
                        if cur: spans.append(cur); cur = None
                if cur: spans.append(cur)
                spans = snap_refine_pred_char_spans(text, spans, offs)

                for s, e, l in spans:
                    if e > s:
                        agg.add((s, e, l))

            all_spans = sorted(agg, key=lambda x: (x[0], x[1], x[2]))
            fout.write(json.dumps({"article_id": aid, "text": text, "spans": all_spans}, ensure_ascii=False) + "\n")
            n_items += 1

    print(f"[DONE] wrote {n_items} items → {args.out_jsonl}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", e)
        sys.exit(1)
