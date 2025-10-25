import json, argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer
from src.train_span_bilstm_crf import BiLSTMCRF, snap_refine_pred_char_spans


def read_jsonl(p):
    with open(p,"r",encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if ln: yield json.loads(ln)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--snap_punct", type=bool, default=True)
    args=ap.parse_args()

    ckpt_dir=Path(args.ckpt_dir)
    with open(ckpt_dir/"labels.txt","r",encoding="utf-8") as f:
        types=[ln.strip() for ln in f if ln.strip()]
    tag2id={"O":0}
    for t in types:
        tag2id[f"B-{t}"]=len(tag2id); tag2id[f"I-{t}"]=len(tag2id)
    id2tag={v:k for k,v in tag2id.items()}

    tok=AutoTokenizer.from_pretrained(ckpt_dir/"tokenizer",use_fast=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts=sorted(ckpt_dir.glob("best_ep*_f*.pt")); assert ckpts, "Không thấy checkpoint *.pt"
    ckpt=torch.load(ckpts[-1],map_location="cpu")
    model_name=ckpt.get("args",{}).get("model_name_or_path","xlm-roberta-base")
    model=BiLSTMCRF(model_name,num_labels=len(tag2id)).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    out=[]
    for obj in read_jsonl(args.input_jsonl):
        text=obj.get("text","")
        enc=tok(text,return_offsets_mapping=True,truncation=True,max_length=args.max_len)
        input_ids=torch.tensor([enc["input_ids"]],dtype=torch.long).to(device)
        attn=torch.tensor([enc["attention_mask"]],dtype=torch.long).to(device)
        path=model(input_ids=input_ids,attention_mask=attn)[0]
        L=int(attn[0].sum().item()); offs=enc["offset_mapping"][:L]
        spans=[]; cur=None
        for t,(s,e) in zip([id2tag.get(x,"O") for x in path[:L]],offs):
            if s==e==0: continue
            if t.startswith("B-"):
                if cur: spans.append(cur)
                cur=[s,e,t[2:]]
            elif t.startswith("I-") and cur and t[2:]==cur[2]:
                cur[1]=e
            else:
                if cur: spans.append(cur); cur=None
        if cur: spans.append(cur)
        if args.snap_punct: spans=snap_refine_pred_char_spans(text,spans,offs)
        out.append({"text":text,"pred_spans":spans})

    Path(args.out_json).parent.mkdir(parents=True,exist_ok=True)
    with open(args.out_json,"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
    print("[DONE] wrote", args.out_json, "items:", len(out))

if __name__=="__main__":
    main()
