# -*- coding: utf-8 -*-
"""
Cross-eval mô hình BiLSTM-CRF của bạn trên ViMedNER (CoNLL).
- Forward của BiLSTMCRF trả list path đã decode khi labels=None (y như infer_*)
- Tokenizer tự nhận Roberta/PhoBERT để bật add_prefix_space khi is_split_into_words=True
- Map YOURS -> ViMedNER (chỉ 5 nhãn) trước khi chấm điểm
- Xuất:
  * {name}_ViMedNER_seqeval_report.txt  (token-level, 0..1)
  * {name}_ViMedNER_span_report.txt     (span-level, %, 5 nhãn)
  * Hình so sánh nếu có >=2 run
"""

import argparse, json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from seqeval.metrics import classification_report

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_span_bilstm_crf import BiLSTMCRF  # forward decode paths khi infer


# ========== I/O ==========
def read_conll(fp: Path):
    """Đọc CoNLL token/BIO -> [{'tokens': [...], 'tags': [...]}]."""
    docs = []
    toks, tags = [], []
    for ln in fp.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            if toks:
                docs.append({"tokens": toks, "tags": tags})
                toks, tags = [], []
            continue
        parts = ln.split()
        toks.append(parts[0]); tags.append(parts[-1])
    if toks:
        docs.append({"tokens": toks, "tags": tags})
    return docs


def load_label_map(path: Path):
    """
    Đọc JSON map ViMedNER -> YOURS, nhưng CHỈ giữ 5 khóa ViMedNER.
    Trả về (vm2y, y2vm, vm_labels)
    """
    vm_labels = [
        "ten_benh",
        "trieu_chung_benh",
        "nguyen_nhan_benh",
        "bien_phap_chan_doan",
        "bien_phap_dieu_tri",
    ]
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        # chỉ lọc đúng 5 key của ViMedNER (file của bạn có thêm schema khác)  # label_map.json
        vm2y = {k: raw[k] for k in vm_labels if k in raw}
        y2vm = {v: k for k, v in vm2y.items()}
        return vm2y, y2vm, vm_labels
    # fallback cứng nếu thiếu file
    y2vm = {
        "Bệnh_lý": "ten_benh",
        "Triệu_chứng": "trieu_chung_benh",
        "Nguyên_nhân": "nguyen_nhan_benh",
        "Chẩn_đoán": "bien_phap_chan_doan",
        "Điều_trị": "bien_phap_dieu_tri",
    }
    vm2y = {v: k for k, v in y2vm.items()}
    return vm2y, y2vm, vm_labels


# ========== Tokenize & Collate ==========
def is_roberta_like(tok_dir: Path, enc_name: str) -> bool:
    return ((tok_dir / "merges.txt").exists() and (tok_dir / "vocab.json").exists()) or \
           ("roberta" in enc_name.lower() or "phobert" in enc_name.lower())

class CoNLLDataset(Dataset):
    def __init__(self, docs, tokenizer, max_len=256):
        self.docs = docs; self.tok = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.docs)
    def __getitem__(self, i):
        tokens = self.docs[i]["tokens"]; tags = self.docs[i]["tags"]
        enc = self.tok(tokens, is_split_into_words=True, max_length=self.max_len,
                       truncation=True, padding="max_length", return_offsets_mapping=False)
        return {"tokens": tokens, "tags": tags, "enc": enc, "word_ids": enc.word_ids()}

def build_collate(tag2id, max_len=256):
    O_id = tag2id.get("O", 0)
    def collate(batch):
        input_ids, attn = [], []
        word_ids_list, gold_bio_word, tokens_list = [], [], []
        for it in batch:
            enc, wi = it["enc"], it["word_ids"]
            tokens, tags = it["tokens"], it["tags"]
            # giữ cấu trúc subword, không dùng loss ở eval
            sub_labels = []
            prev = None; wpos = -1
            for w in wi:
                if w is None:
                    sub_labels.append(-100)
                else:
                    if w != prev:
                        wpos += 1
                        t = tags[wpos] if wpos < len(tags) else "O"
                        sub_labels.append(tag2id.get(t, O_id))
                    else:
                        sub_labels.append(-100)
                    prev = w
            input_ids.append(enc["input_ids"]); attn.append(enc["attention_mask"])
            word_ids_list.append(wi); gold_bio_word.append(tags); tokens_list.append(tokens)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "word_ids": word_ids_list, "gold_bio_word": gold_bio_word, "tokens": tokens_list
        }
    return collate


# ========== Span utils ==========
def words_to_char_offsets(words: List[str]):
    offs=[]; cur=0
    for i,w in enumerate(words):
        if i>0: cur+=1
        s=cur; cur+=len(w); e=cur
        offs.append((s,e))
    return offs

def bio_to_spans(bio: List[str], words: List[str]):
    offs=words_to_char_offsets(words)
    spans=[]; cur=None
    for i,t in enumerate(bio):
        if t=="O" or "-" not in t:
            if cur:
                s,e,l=cur; spans.append([offs[s][0],offs[e][1],l]); cur=None
            continue
        p,l=t.split("-",1)
        if p=="B":
            if cur: s,e,l0=cur; spans.append([offs[s][0],offs[e][1],l0])
            cur=[i,i,l]
        else:
            if cur and cur[2]==l: cur[1]=i
            else: cur=[i,i,l]
    if cur:
        s,e,l=cur; spans.append([offs[s][0],offs[e][1],l])
    return spans


# ========== Load one run (theo infer_*) ==========
def load_run(run_dir: Path, device: str):
    run_dir = Path(run_dir)

    # Dựng id2tag theo labels.txt trong từng run của BẠN (không phải ViMedNER)
    types = [ln.strip() for ln in (run_dir/"labels.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    tag2id = {"O": 0}
    for t in types:
        tag2id[f"B-{t}"]=len(tag2id); tag2id[f"I-{t}"]=len(tag2id)
    id2tag = {v:k for k,v in tag2id.items()}  # labels.txt của run  :contentReference[oaicite:3]{index=3}

    tok_dir = run_dir/"tokenizer"
    ckpts = sorted(run_dir.glob("best_ep*_f*.pt"))
    if not ckpts: ckpts = sorted(run_dir.glob("*.pt"))
    assert ckpts, f"Không thấy checkpoint *.pt trong {run_dir}"
    ckpt = torch.load(ckpts[-1], map_location="cpu")
    model_name = ckpt.get("args",{}).get("model_name_or_path","xlm-roberta-base")

    tok = AutoTokenizer.from_pretrained(str(tok_dir) if tok_dir.exists() else model_name,
                                        use_fast=True,
                                        add_prefix_space=is_roberta_like(tok_dir, model_name))
    assert tok.is_fast

    device_t = torch.device(device)
    model = BiLSTMCRF(model_name, num_labels=len(tag2id)).to(device_t)
    try:
        vocab_size = len(tok.get_vocab())
        model.backbone.resize_token_embeddings(vocab_size)
    except Exception:
        pass
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model, tok, id2tag, tag2id


# ========== Metrics ==========
def span_prf(pred_spans, gold_spans, labels: List[str]):
    per = {l: {"tp":0,"fp":0,"fn":0} for l in labels}
    for ps,gs in zip(pred_spans, gold_spans):
        G={(s,e,l) for s,e,l in gs}; P={(s,e,l) for s,e,l in ps}
        for lab in labels:
            Gt={(s,e) for s,e,l in G if l==lab}
            Pt={(s,e) for s,e,l in P if l==lab}
            I=Gt & Pt
            per[lab]["tp"]+=len(I); per[lab]["fp"]+=len(Pt-I); per[lab]["fn"]+=len(Gt-I)
    def prf(d):
        tp,fp,fn=d["tp"],d["fp"],d["fn"]
        p=0 if tp+fp==0 else tp/(tp+fp)
        r=0 if tp+fn==0 else tp/(tp+fn)
        f=0 if p+r==0 else 2*p*r/(p+r)
        return p,r,f
    rows=[]; mic_tp=mic_fp=mic_fn=0; macro=[]
    for lab in labels:
        p,r,f=prf(per[lab])
        rows.append((lab,round(p*100,2),round(r*100,2),round(f*100,2), per[lab]["tp"]+per[lab]["fn"]))
        mic_tp+=per[lab]["tp"]; mic_fp+=per[lab]["fp"]; mic_fn+=per[lab]["fn"]; macro.append(f)
    mp=0 if mic_tp+mic_fp==0 else mic_tp/(mic_tp+mic_fp)
    mr=0 if mic_tp+mic_fn==0 else mic_tp/(mic_tp+mic_fn)
    mf=0 if mp+mr==0 else 2*mp*mr/(mp+mr)
    maf=float(np.mean(macro)) if macro else 0.0
    return rows, round(mp*100,2), round(mr*100,2), round(mf*100,2), round(maf*100,2)

def filter_to_vm_labels(seqs, vm_labels):
    """Giữ lại duy nhất O và {B-,I-}{5 nhãn ViMedNER}; mọi nhãn lạ -> O."""
    allowed = set(["O"])
    for lab in vm_labels:
        allowed.add(f"B-{lab}"); allowed.add(f"I-{lab}")
    out = []
    for sent in seqs:
        out.append([t if t in allowed else "O" for t in sent])
    return out

# ========== Eval one run ==========
def eval_one(run_dir: Path, docs, outdir: Path, name: str,
             device: str, y2vm: Dict[str,str], vm_labels: List[str],
             batch_size=16, max_len=256):
    model, tok, id2tag, tag2id = load_run(run_dir, device)

    ds = CoNLLDataset(docs, tok, max_len=max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=build_collate(tag2id, max_len))

    y_true_bio, y_pred_bio = [], []
    gold_spans_vm, pred_spans_vm = [], []

    with torch.no_grad():
        for batch in dl:
            ids  = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)  # giữ nguyên attn (torchcrf yêu cầu timestep 0 = True)

            # forward trả list paths đã decode (giống infer)  # infer_articles.py / infer_span.py
            paths = model(input_ids=ids, attention_mask=attn)  # list[list[int]]

            B = ids.size(0)
            for b in range(B):
                wi    = batch["word_ids"][b]
                words = batch["tokens"][b]
                gold  = batch["gold_bio_word"][b]  # BIO của ViMedNER

                pred_ids = paths[b]

                # Word-level dự đoán: lấy tag tại subword-đầu; bỏ special theo word_ids()
                pred_tags=[]; prev=None; k=0
                for w in wi:
                    if w is None:  # CLS/SEP/PAD
                        continue
                    if w != prev:
                        tag = id2tag.get(pred_ids[k], "O") if k < len(pred_ids) else "O"
                        pred_tags.append(tag); prev=w
                    k += 1

                L = min(len(gold), len(pred_tags), len(words))
                true_bio = gold[:L]          # BIO (VM)
                pred_bio = pred_tags[:L]     # BIO (YOURS)
                words_L  = words[:L]

                # Map YOURS -> VM (chỉ 5 nhãn)
                def map_tag(t):
                    if t=="O" or "-" not in t: return "O"
                    p,base=t.split("-",1); vm=y2vm.get(base)
                    return f"{p}-{vm}" if vm else "O"

                pred_bio_vm=[map_tag(t) for t in pred_bio]

                # token-level
                y_true_bio.append(true_bio)
                y_pred_bio.append(pred_bio_vm)

                # span-level
                g_sp=bio_to_spans(true_bio, words_L)
                p_sp=bio_to_spans(pred_bio_vm, words_L)
                gold_spans_vm.append(g_sp)
                pred_spans_vm.append(p_sp)

    # --- Ghi report ---
    # Lọc nhãn về đúng 5 label VM (vì seqeval bản này không hỗ trợ labels=)
    y_true_vm = filter_to_vm_labels(y_true_bio, vm_labels)
    y_pred_vm = filter_to_vm_labels(y_pred_bio, vm_labels)

    rep_txt = classification_report(
        y_true_vm, y_pred_vm,
        digits=4, zero_division=0
    )
    (outdir / f"{name}_ViMedNER_seqeval_report.txt").write_text(rep_txt, encoding="utf-8")

    # Span-report: ẩn hàng có Support=0 (nếu có)
    rows, mp, mr, mf, maf = span_prf(pred_spans_vm, gold_spans_vm, vm_labels)
    rows = [r for r in rows if r[4] > 0]  # r=(label,P,R,F1,Support)

    lines = ["Label\tP(%)\tR(%)\tF1(%)\tSupport"]
    for lab,P,R,F,s in rows: lines.append(f"{lab}\t{P}\t{R}\t{F}\t{s}")
    lines.append(f"MICRO\t{mp}\t{mr}\t{mf}\t-")
    lines.append(f"MACRO(non-empty)\t-\t-\t{maf}\t-")
    (outdir/f"{name}_ViMedNER_span_report.txt").write_text("\n".join(lines), encoding="utf-8")

    f1_per = {lab: next((r[3] for r in rows if r[0]==lab), 0.0) for lab in vm_labels}
    return {"run": name, "micro_f1": mf, "macro_f1": maf, "span_f1": f1_per}


# ========== Plot ==========
def plot_pair(a, b, outdir: Path, labels: List[str]):
    name = f"{a['run']}_vs_{b['run']}"
    x = np.arange(len(labels)); w = 0.35
    plt.figure(figsize=(10,4))
    plt.bar(x-w/2, [a["span_f1"].get(l,0) for l in labels], width=w, label=a["run"])
    plt.bar(x+w/2, [b["span_f1"].get(l,0) for l in labels], width=w, label=b["run"])
    plt.xticks(x, labels, rotation=20); plt.ylabel("Span F1 (%)"); plt.title("ViMedNER: F1 theo nhãn")
    plt.legend(); plt.tight_layout(); outdir.mkdir(parents=True,exist_ok=True)
    plt.savefig(outdir/f"fig_{name}_per_label_f1.png", dpi=200); plt.close()

    plt.figure(figsize=(6,4))
    labs2 = ["Micro-F1","Macro-F1"]; x = np.arange(2)
    plt.bar(x-w/2, [a["micro_f1"], a["macro_f1"]], width=w, label=a["run"])
    plt.bar(x+w/2, [b["micro_f1"], b["macro_f1"]], width=w, label=b["run"])
    plt.xticks(x, labs2); plt.ylabel("F1 (%)"); plt.title("ViMedNER: Micro/Macro F1")
    plt.legend(); plt.tight_layout(); plt.savefig(outdir/f"fig_{name}_micro_macro.png", dpi=200); plt.close()


# ========== Main ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vimedner_txt", required=True)
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label_map", default="data/cross/label_map.json")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vm2y, y2vm, vm_labels = load_label_map(Path(args.label_map))  # chỉ giữ 5 key VM  # label_map.json

    docs = read_conll(Path(args.vimedner_txt))
    print(f"[LOAD] ViMedNER {len(docs)}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    summaries=[]
    for i, rd in enumerate(args.run_dirs):
        name = args.names[i] if args.names and i < len(args.names) else Path(rd).name
        print(f"[RUN] {rd} ({name})")
        summ = eval_one(Path(rd), docs, outdir, name, device, y2vm, vm_labels,
                        batch_size=args.batch_size, max_len=args.max_len)
        print(f"[{name}] Micro-F1={summ['micro_f1']:.2f} | Macro-F1={summ['macro_f1']:.2f}")
        summaries.append(summ)
    if len(summaries)>=2:
        plot_pair(summaries[0], summaries[1], outdir, vm_labels)
        print(f"[FIG] -> {outdir}")

if __name__ == "__main__":
    main()
