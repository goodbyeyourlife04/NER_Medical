import os, json, time, argparse, random, math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from transformers import (
    AutoTokenizer, AutoModel, RobertaTokenizerFast,
    get_linear_schedule_with_warmup
)
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from torchcrf import CRF

# AMP mới
from torch import amp
autocast   = amp.autocast
GradScaler = amp.GradScaler  # dùng GradScaler('cuda', enabled=...)

# vẽ hình
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- cấu hình an toàn SDPA (tránh lỗi backend trên Windows khi mask lạ) ----
os.environ.setdefault("PYTORCH_SDP_BACKEND", "math")
try:
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

def set_seed(sd=42):
    random.seed(sd); np.random.seed(sd)
    torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)

def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

PUNCT = set(".,;:!?)]}%»”’…")
SPACE = set(" \t\r\n")

def safe_torch_save(obj, path: Path, tries: int = 5, sleep_sec: float = 1.5):
    p = Path(path)
    for i in range(tries):
        try:
            torch.save(obj, p); return str(p)
        except PermissionError:
            time.sleep(sleep_sec)
            p = p.with_name(f"{p.stem}_{i+1}{p.suffix}")
    p = p.with_name(f"{p.stem}_{int(time.time())}{p.suffix}")
    torch.save(obj, p); return str(p)

def strict_span_scores(golds, preds, labels):
    tp=fp=fn=0; per=defaultdict(lambda:{'tp':0,'fp':0,'fn':0})
    for g,p in zip(golds,preds):
        G=set(map(tuple,g)); P=set(map(tuple,p)); I=G&P
        tp+=len(I); fp+=len(P-I); fn+=len(G-I)
        for L in labels:
            Gt={x for x in G if x[2]==L}; Pt={x for x in P if x[2]==L}; It=Gt&Pt
            per[L]['tp']+=len(It); per[L]['fp']+=len(Pt-It); per[L]['fn']+=len(Gt-It)
    micro_p=tp/(tp+fp+1e-12); micro_r=tp/(tp+fn+1e-12)
    micro_f=2*micro_p*micro_r/(micro_p+micro_r+1e-12)
    macro_f=np.mean([
        (0 if (v['tp']==0 and (v['fp']>0 or v['fn']>0)) else
         2*(v['tp']/(v['tp']+v['fp']+1e-12))*(v['tp']/(v['tp']+v['fn']+1e-12))/
         ((v['tp']/(v['tp']+v['fp']+1e-12))+(v['tp']/(v['tp']+v['fn']+1e-12))+1e-12))
        if not (v['tp']==v['fp']==v['fn']==0) else 0
        for v in per.values()
    ]) if per else 0.0
    return dict(micro_p=micro_p,micro_r=micro_r,micro_f1=micro_f,macro_f1=macro_f), per

def span_report(golds,preds,labels):
    _, per = strict_span_scores(golds,preds,labels)
    table=[]
    for l in labels:
        v=per[l]; tp=v['tp']; fp=v['fp']; fn=v['fn']
        p=tp/(tp+fp+1e-12); r=tp/(tp+fn+1e-12); f=2*p*r/(p+r+1e-12); sup=tp+fn
        table.append((l, round(p*100,2), round(r*100,2), round(f*100,2), sup))
    return table

def snap_refine_pred_char_spans(text, spans, token_offsets):
    out=[]
    for s,e,l in spans:
        for ts,te in token_offsets:
            if ts<=s<te: s=ts; break
        for ts,te in token_offsets:
            if ts<e<=te: e=te; break
        if e<len(text) and text[e] in PUNCT:
            e+=1
            while e<len(text) and text[e] in SPACE: e+=1
        out.append([s,e,l])
    return out

class SpanNERDataset(Dataset):
    def __init__(self, fp, label_list=None, oversample_rare=False):
        raw=list(read_jsonl(fp))
        items=[]; label_set=set()
        n_lines=len(raw); n_with=0; n_sp=0

        def _norm(sp):
            if isinstance(sp, dict): sp=sp.get("spans",[])
            out=[]
            for x in sp:
                if isinstance(x,(list,tuple)) and len(x)>=3:
                    s,e,l=int(x[0]),int(x[1]),str(x[2]); out.append([s,e,l])
                elif isinstance(x,dict):
                    s=int(x.get("start",x.get("begin",x.get("s",-1))))
                    e=int(x.get("end",x.get("finish",x.get("e",-1))))
                    l=str(x.get("label",x.get("type",x.get("tag",""))))
                    if s!=-1 and e!=-1 and l: out.append([s,e,l])
            return out

        for o in raw:
            if "text" in o: text=o["text"]
            elif "words" in o: text=" ".join(o["words"])
            else: continue
            spans=(o.get("gold_spans") or o.get("spans") or o.get("entities") or o.get("labels") or [])
            spans=_norm(spans)
            valid=[]
            for s,e,l in spans:
                if 0<=s<e<=len(text):
                    valid.append([s,e,l]); label_set.add(l)
            if valid: n_with+=1; n_sp+=len(valid)
            items.append({"text":text,"spans":valid})

        self.types=list(label_list) if label_list else sorted(label_set)
        self.tag2id={"O":0}
        for t in self.types:
            self.tag2id[f"B-{t}"]=len(self.tag2id); self.tag2id[f"I-{t}"]=len(self.tag2id)
        self.id2tag={v:k for k,v in self.tag2id.items()}
        self.items=items
        print(f"[DATA] {Path(fp).name}: lines={n_lines}, lines_with_span={n_with}, spans={n_sp}, labels={self.types}")

        if oversample_rare and self.types:
            cnt=Counter()
            for it in self.items:
                labs={l for _,_,l in it["spans"]}
                for l in labs: cnt[l]+=1
            if cnt:
                med=np.median(list(cnt.values()))
                aug=[]
                for it in self.items:
                    labs={l for _,_,l in it["spans"]}
                    if not labs: continue
                    rare=[l for l in labs if cnt[l] < med]
                    if rare:
                        k=2 if min(cnt[l] for l in rare) < 0.5*med else 1
                        aug.extend([it]*k)
                self.items.extend(aug)

    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

class BiLSTMCRF(nn.Module):
    def __init__(self, transformer_name, num_labels, lstm_hidden=256, lstm_layers=1, dropout=0.1):
        super().__init__()
        self.backbone=AutoModel.from_pretrained(transformer_name)
        hid=self.backbone.config.hidden_size
        self.dropout=nn.Dropout(dropout)
        self.lstm=nn.LSTM(hid,lstm_hidden,num_layers=lstm_layers,batch_first=True,
                          bidirectional=True,dropout=0.0 if lstm_layers==1 else dropout)
        self.classifier=nn.Linear(lstm_hidden*2,num_labels)
        self.crf=CRF(num_labels,batch_first=True)
    def forward(self,input_ids,attention_mask,labels=None):
        x=self.backbone(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        x=self.dropout(x); x,_=self.lstm(x)
        emissions=self.classifier(self.dropout(x))
        mask=attention_mask.bool()
        if labels is not None:
            return -self.crf(emissions,labels,mask=mask,reduction='mean')
        return self.crf.decode(emissions,mask=mask)

class CollateFn:
    def __init__(self, tokenizer, tag2id, max_len=512):
        assert tokenizer.is_fast, "Cần tokenizer FAST (use_fast=True)."
        self.tok=tokenizer; self.tag2id=tag2id; self.max_len=max_len
    def __call__(self,batch):
        texts=[b["text"] for b in batch]
        spans_list=[b["spans"] for b in batch]
        enc=self.tok(texts,return_offsets_mapping=True,padding=True,truncation=True,
                     max_length=self.max_len,add_special_tokens=True)
        input_ids=torch.tensor(enc["input_ids"],dtype=torch.long)
        attention=torch.tensor(enc["attention_mask"],dtype=torch.long)
        # đảm bảo mask là 0/1
        attention=(attention>0).long()
        offsets=enc["offset_mapping"]
        labels=torch.zeros_like(input_ids)
        for i,(spans,offs) in enumerate(zip(spans_list,offsets)):
            for (s,e,l) in spans:
                idxs=[]
                for j,(ts,te) in enumerate(offs):
                    if ts==te==0: continue
                    if te<=s or ts>=e: continue
                    idxs.append(j)
                if not idxs: continue
                labels[i,idxs[0]]=self.tag2id.get(f"B-{l}",0)
                for j in idxs[1:]: labels[i,j]=self.tag2id.get(f"I-{l}",0)
        return {"input_ids":input_ids,"attention_mask":attention,"labels":labels,
                "texts":texts,"offsets":offsets,"spans_list":spans_list}

@torch.no_grad()
def evaluate(model, loader, device, id2tag, label_types, snap_punct=False):
    model.eval()
    all_gold_spans, all_pred_spans=[],[]
    all_gold_tags,  all_pred_tags =[],[]
    for batch in loader:
        in_ids=batch["input_ids"].to(device,non_blocking=True)
        attn  =batch["attention_mask"].to(device,non_blocking=True)
        paths =model(input_ids=in_ids,attention_mask=attn)
        for i,path in enumerate(paths):
            L=int(attn[i].sum().item())
            toks=[id2tag.get(t,"O") for t in path[:L]]
            toks=[t for t,(s,e) in zip(toks,batch["offsets"][i][:L]) if not (s==e==0)]
            all_pred_tags.append(toks)
        for i,(offs,spans) in enumerate(zip(batch["offsets"],batch["spans_list"])):
            L=int(attn[i].sum().item()); offs=offs[:L]; tg=["O"]*len(offs)
            for (s,e,lab) in spans:
                idxs=[]
                for j,(ts,te) in enumerate(offs):
                    if ts==te==0: continue
                    if te<=s or ts>=e: continue
                    idxs.append(j)
                if not idxs: continue
                tg[idxs[0]]=f"B-{lab}"
                for j in idxs[1:]: tg[j]=f"I-{lab}"
            tg=[t for t,(s,e) in zip(tg,offs) if not (s==e==0)]
            all_gold_tags.append(tg)
        for i in range(len(batch["texts"])):
            text=batch["texts"][i]; offs=batch["offsets"][i]; L=int(attn[i].sum().item())
            pred_sp=[]; cur=None
            for t,(s,e) in zip([id2tag.get(x,"O") for x in paths[i][:L]], offs[:L]):
                if s==e==0: continue
                if t.startswith("B-"):
                    if cur: pred_sp.append(cur)
                    cur=[s,e,t[2:]]
                elif t.startswith("I-") and cur and t[2:]==cur[2]:
                    cur[1]=e
                else:
                    if cur: pred_sp.append(cur); cur=None
            if cur: pred_sp.append(cur)
            if snap_punct:
                pred_sp=snap_refine_pred_char_spans(text,pred_sp,offs[:L])
            gold_sp=[[s,e,l] for (s,e,l) in batch["spans_list"][i]]
            all_gold_spans.append(gold_sp); all_pred_spans.append(pred_sp)
    tok_p=precision_score(all_gold_tags,all_pred_tags)*100.0
    tok_r=recall_score(all_gold_tags,all_pred_tags)*100.0
    tok_f=f1_score(all_gold_tags,all_pred_tags)*100.0
    span_micro,_=strict_span_scores(all_gold_spans,all_pred_spans,label_types)
    span_micro={k:v*100.0 for k,v in span_micro.items()}
    return dict(tok_p=tok_p,tok_r=tok_r,tok_f1=tok_f,**span_micro), \
           (all_gold_tags,all_pred_tags,all_gold_spans,all_pred_spans)

def save_training_figures(history, out_dir: Path):
    epochs=[r["epoch"] for r in history]
    loss=[r["train_loss"] for r in history]
    f1_span=[r.get("val_span_micro_f1(%)") for r in history]
    f1_tok=[r.get("val_token_f1(%)") for r in history]
    fig_dir=out_dir/"figures"; fig_dir.mkdir(exist_ok=True)
    plt.figure(); plt.plot(epochs,loss,marker='o'); plt.xlabel("Epoch"); plt.ylabel("Train Loss")
    plt.title("Train Loss by Epoch"); plt.grid(True,linestyle="--",alpha=.5)
    plt.tight_layout(); plt.savefig(fig_dir/"train_loss.png",dpi=180); plt.close()
    if any(v is not None for v in f1_tok):
        xs,ys=zip(*[(e,v) for e,v in zip(epochs,f1_tok) if v is not None])
        plt.figure(); plt.plot(xs,ys,marker='o'); plt.xlabel("Epoch"); plt.ylabel("Token F1 (%)")
        plt.title("Token F1 (seqeval) by Epoch"); plt.grid(True,linestyle="--",alpha=.5)
        plt.tight_layout(); plt.savefig(fig_dir/"token_f1.png",dpi=180); plt.close()
    if any(v is not None for v in f1_span):
        xs,ys=zip(*[(e,v) for e,v in zip(epochs,f1_span) if v is not None])
        plt.figure(); plt.plot(xs,ys,marker='o'); plt.xlabel("Epoch"); plt.ylabel("Span Micro-F1 (%)")
        plt.title("Span Micro-F1 by Epoch"); plt.grid(True,linestyle="--",alpha=.5)
        plt.tight_layout(); plt.savefig(fig_dir/"span_micro_f1.png",dpi=180); plt.close()

def save_span_perlabel_bars(span_table, out_dir: Path, prefix="val"):
    if not span_table: return
    fig_dir=out_dir/"figures"; fig_dir.mkdir(exist_ok=True)
    labels=[t[0] for t in span_table]; P=[t[1] for t in span_table]
    R=[t[2] for t in span_table]; F1=[t[3] for t in span_table]
    y=np.arange(len(labels))
    plt.figure(figsize=(8,max(3,0.4*len(labels)))); plt.barh(y,F1)
    plt.yticks(y,labels); plt.xlabel("F1 (%)"); plt.title(f"{prefix.upper()} Strict-Span F1 by Label")
    plt.tight_layout(); plt.savefig(fig_dir/f"{prefix}_span_f1_per_label.png",dpi=180); plt.close()
    width=0.25
    plt.figure(figsize=(10,max(3,0.4*len(labels))))
    plt.barh(y-width,P,height=width,label="P"); plt.barh(y,R,height=width,label="R")
    plt.barh(y+width,F1,height=width,label="F1")
    plt.yticks(y,labels); plt.xlabel("(%)"); plt.title(f"{prefix.upper()} Strict-Span P/R/F1 by Label")
    plt.legend(); plt.tight_layout(); plt.savefig(fig_dir/f"{prefix}_span_prf_per_label.png",dpi=180); plt.close()

def run_epoch(model, loader, optimizer, scheduler, device, amp=True, grad_accum=1):
    model.train(); tot=0.0; n=0
    scaler=GradScaler('cuda', enabled=amp); optimizer.zero_grad(set_to_none=True)
    for step,batch in enumerate(loader,1):
        for k in ["input_ids","attention_mask","labels"]:
            batch[k]=batch[k].to(device,non_blocking=True)
        with autocast(device_type='cuda', enabled=amp):
            loss=model(**{k:batch[k] for k in ["input_ids","attention_mask","labels"]})/max(1,grad_accum)
        scaler.scale(loss).backward()
        if step%grad_accum==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True); scheduler.step()
        tot+=float(loss.detach().cpu()); n+=1
    return tot/max(1,n)

def lengths_for_items(items, tokenizer, max_len):
    lens=[]
    for it in items:
        n=len(tokenizer(it["text"],add_special_tokens=True,truncation=True,max_length=max_len).input_ids)
        lens.append(n)
    return np.array(lens)

def save_reports(out_dir: Path, prefix: str, gold_tags, pred_tags, gold_spans, pred_spans, label_types):
    with open(out_dir/f"{prefix}_seqeval_report.txt","w",encoding="utf-8") as f:
        f.write(classification_report(gold_tags,pred_tags,digits=2))
    table=span_report(gold_spans,pred_spans,label_types)
    with open(out_dir/f"{prefix}_span_report.txt","w",encoding="utf-8") as f:
        f.write(f"{'label':20s} P(%)   R(%)   F1(%)  support\n")
        for l,p,r,ff,sup in table: f.write(f"{l:20s} {p:6.2f} {r:6.2f} {ff:6.2f} {sup:8d}\n")
    return table

def append_log_files(out_dir: Path, row: dict):
    csvp=out_dir/"training_log.csv"; is_new=not csvp.exists()
    with open(csvp,"a",encoding="utf-8") as f:
        if is_new: f.write(",".join(row.keys())+"\n")
        f.write(",".join(str(v) for v in row.values())+"\n")
    with open(out_dir/"training_log.jsonl","a",encoding="utf-8") as f:
        f.write(json.dumps(row,ensure_ascii=False)+"\n")

# ---------- preflight: kiểm tra token id / mask ----------
def preflight_token_check(sample_texts, tokenizer, max_len, model_name):
    enc = tokenizer(sample_texts, return_attention_mask=True, truncation=True,
                    max_length=max_len, add_special_tokens=True)
    arr = np.array(enc["input_ids"], dtype=np.int64)
    am  = np.array(enc["attention_mask"], dtype=np.int64)
    # load tạm model config để lấy vocab_size
    tmp = AutoModel.from_pretrained(model_name)
    vocab_size = tmp.config.vocab_size
    del tmp
    id_min, id_max = int(arr.min()), int(arr.max())
    if id_min < 0 or id_max >= vocab_size:
        print("[FATAL] Token ids out-of-range for model vocab.")
        print("  model:", model_name, "| vocab_size:", vocab_size, "| id_min/id_max:", id_min, id_max)
        raise RuntimeError("tokenizer/model mismatch")
    uniq = np.unique(am).tolist()
    if any(x not in (0,1) for x in uniq):
        print("[FATAL] attention_mask contains values not in {0,1}:", uniq)
        raise RuntimeError("attention_mask invalid")
    return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--labels_txt", default="")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr_lstm", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_best", type=bool, default=True)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--oversample_rare", type=bool, default=True)
    ap.add_argument("--snap_punct", type=bool, default=True)
    ap.add_argument("--amp", type=bool, default=True)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pin_memory", type=bool, default=True)
    ap.add_argument("--bucket_by_length", type=bool, default=True)
    ap.add_argument("--bucket_size", type=int, default=50)
    ap.add_argument("--grad_accum", type=int, default=1)
    args=ap.parse_args()

    set_seed(args.seed)
    base_out=Path(args.output_dir); ensure_dir(base_out)
    run_id=time.strftime("%Y%m%d-%H%M%S")
    out_dir=base_out/f"run_{run_id}"; ensure_dir(out_dir)

    train_fp=Path(args.data_dir)/"train.jsonl"
    val_fp  =Path(args.data_dir)/"val.jsonl"
    test_fp =Path(args.data_dir)/"test.jsonl"
    assert train_fp.exists() and val_fp.exists(), "Thiếu train.jsonl hoặc val.jsonl"

    label_list=None
    if args.labels_txt and Path(args.labels_txt).exists():
        with open(args.labels_txt,"r",encoding="utf-8") as f:
            label_list=[ln.strip() for ln in f if ln.strip()]

    ds_tr=SpanNERDataset(str(train_fp),label_list=label_list,oversample_rare=args.oversample_rare)
    ds_va=SpanNERDataset(str(val_fp),label_list=ds_tr.types,oversample_rare=False)
    label_types=ds_tr.types
    print("[INFO] labels:", label_types)

    # ---- tokenizer an toàn cho PhoBERT, giữ nguyên XLM-R ----
    tok=AutoTokenizer.from_pretrained(args.model_name_or_path,use_fast=True)
    if (not getattr(tok,"is_fast",False)) and "phobert" in args.model_name_or_path.lower():
        tok=RobertaTokenizerFast.from_pretrained(args.model_name_or_path,use_fast=True)
    tok.padding_side="right"; tok.truncation_side="right"
    assert tok.is_fast, "Phải là tokenizer FAST."

    # PhoBERT: giới hạn max_len thấp hơn
    if "phobert" in args.model_name_or_path.lower():
        args.max_len=min(args.max_len,256)

    # lưu tokenizer & labels
    (out_dir/"tokenizer").mkdir(exist_ok=True)
    tok.save_pretrained(out_dir/"tokenizer")
    with open(out_dir/"labels.txt","w",encoding="utf-8") as f:
        for lb in label_types: f.write(lb+"\n")

    # ---- Preflight: thử 5 câu đầu để bắt mismatch sớm ----
    probe=[]
    for i,obj in enumerate(read_jsonl(str(train_fp))):
        probe.append(obj.get("text",""))
        if len(probe)>=5: break
    if probe:
        preflight_token_check(probe, tok, args.max_len, args.model_name_or_path)

    collate=CollateFn(tok,ds_tr.tag2id,max_len=args.max_len)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=BiLSTMCRF(args.model_name_or_path,num_labels=len(ds_tr.tag2id)).to(device)

    no_decay=["bias","LayerNorm.weight"]
    def is_backbone(n): return n.startswith("backbone.")
    def is_head(n): return not is_backbone(n)
    grouped=[
        {"params":[p for n,p in model.named_parameters() if is_backbone(n) and not any(nd in n for nd in no_decay)],"lr":args.lr,"weight_decay":args.weight_decay},
        {"params":[p for n,p in model.named_parameters() if is_backbone(n) and any(nd in n for nd in no_decay)],"lr":args.lr,"weight_decay":0.0},
        {"params":[p for n,p in model.named_parameters() if is_head(n) and not any(nd in n for nd in no_decay)],"lr":args.lr_lstm,"weight_decay":args.weight_decay},
        {"params":[p for n,p in model.named_parameters() if is_head(n) and any(nd in n for nd in no_decay)],"lr":args.lr_lstm,"weight_decay":0.0},
    ]
    optimizer=torch.optim.AdamW(grouped)

    sampler=None; shuffle=True
    if args.bucket_by_length:
        lens=lengths_for_items(ds_tr.items,tok,args.max_len)
        idx=np.argsort(lens)
        blocks=np.array_split(idx,max(1,len(idx)//(args.batch_size*args.bucket_size)))
        order=np.concatenate([np.random.permutation(b) for b in blocks])
        sampler=SubsetRandomSampler(order); shuffle=False

    train_loader=DataLoader(ds_tr,batch_size=args.batch_size,shuffle=shuffle,sampler=sampler,
                            collate_fn=collate,num_workers=args.num_workers,
                            pin_memory=args.pin_memory,persistent_workers=args.num_workers>0)
    val_loader=DataLoader(ds_va,batch_size=args.batch_size,shuffle=False,
                          collate_fn=collate,num_workers=args.num_workers,
                          pin_memory=args.pin_memory,persistent_workers=args.num_workers>0)

    total_steps=math.ceil(len(train_loader)*args.num_epochs/max(1,args.grad_accum))
    warmup=int(args.warmup_ratio*total_steps)
    scheduler=get_linear_schedule_with_warmup(optimizer,warmup,total_steps)

    best=-1.0; best_path=None; t0=time.time(); history=[]
    for ep in range(1,args.num_epochs+1):
        t_ep=time.time()
        tr_loss=run_epoch(model,train_loader,optimizer,scheduler,device,
                          amp=args.amp,grad_accum=args.grad_accum)
        ep_time=time.time()-t_ep; cum=time.time()-t0
        row={"epoch":ep,"train_loss":round(tr_loss,4),
             "time_epoch_sec":round(ep_time,2),"time_cum_sec":round(cum,2)}

        if ep%args.eval_every==0:
            metrics,(g_tags,p_tags,g_sp,p_sp)=evaluate(
                model,val_loader,device,ds_tr.id2tag,label_types,snap_punct=args.snap_punct
            )
            row.update({
                "val_token_precision(%)":round(metrics["tok_p"],2),
                "val_token_recall(%)":round(metrics["tok_r"],2),
                "val_token_f1(%)":round(metrics["tok_f1"],2),
                "val_span_micro_p(%)":round(metrics["micro_p"],2),
                "val_span_micro_r(%)":round(metrics["micro_r"],2),
                "val_span_micro_f1(%)":round(metrics["micro_f1"],2),
                "val_span_macro_f1(%)":round(metrics["macro_f1"],2),
            })
            table=save_reports(out_dir,"val",g_tags,p_tags,g_sp,p_sp,label_types)
            save_span_perlabel_bars(table,out_dir,"val")
            cur=metrics["micro_f1"]
            if args.save_best and cur>best:
                best=cur
                ckpt_path=out_dir/f"best_ep{ep}_f1{best:.2f}.pt"
                best_path=safe_torch_save({"state_dict":model.state_dict(),
                                           "tag2id":ds_tr.tag2id,"args":vars(args)},ckpt_path)
                print(f"[CKPT] saved: {best_path}")

        print(row); append_log_files(out_dir,row)
        history.append(row); save_training_figures(history,out_dir)

    if test_fp.exists():
        ds_te=SpanNERDataset(str(test_fp),label_list=label_types)
        test_loader=DataLoader(ds_te,batch_size=args.batch_size,shuffle=False,
                               collate_fn=collate,num_workers=args.num_workers,
                               pin_memory=args.pin_memory,persistent_workers=args.num_workers>0)
        metrics,(g_tags,p_tags,g_sp,p_sp)=evaluate(
            model,test_loader,device,ds_tr.id2tag,label_types,snap_punct=args.snap_punct
        )
        print({"test_span_micro_f1(%)":round(metrics["micro_f1"],2),
               "test_span_macro_f1(%)":round(metrics["macro_f1"],2),
               "test_token_f1(%)":round(metrics["tok_f1"],2)})
        table=save_reports(out_dir,"test",g_tags,p_tags,g_sp,p_sp,label_types)
        save_span_perlabel_bars(table,out_dir,"test")

    print("[DONE] saved_best:", best_path)

if __name__=="__main__":
    main()
