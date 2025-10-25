# -*- coding: utf-8 -*-
import json, argparse, hashlib
from pathlib import Path

PALETTE=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
def color_for(label): return PALETTE[int(hashlib.md5(label.encode("utf-8")).hexdigest(),16)%len(PALETTE)]
def esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;line-height:1.6;margin:24px;}}
.doc{{padding:12px 14px;border:1px solid #ddd;border-radius:8px;margin:12px 0;white-space:pre-wrap;}}
.head{{opacity:.65;font-size:13px;margin-bottom:6px}}
.ent{{padding:0 3px;border-radius:4px;border:1px solid #999;position:relative;}}
.lab{{font-size:11px;color:#222;background:#fff; border:1px solid #999; border-radius:4px;
      padding:0 3px;position:relative; top:-0.4em; margin-left:4px;}}
.legend{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}}
.legend .chip{{display:inline-flex;align-items:center;gap:6px;padding:2px 8px;
      border:1px solid #ccc;border-radius:99px;font-size:12px}}
.legend .sw{{width:12px;height:12px;border-radius:3px;border:1px solid #999;}}
</style></head><body>
<h2>Predicted spans</h2>
<div class="legend">
{legend}
</div>
{docs}
</body></html>"""

def render_doc(text, spans):
    spans=sorted(spans,key=lambda x:(x[0],x[1]))
    html=""; last=0
    for s,e,l in spans:
        if s>last: html+=esc(text[last:s])
        frag=esc(text[s:e]); c=color_for(l)
        html+=f'<span class="ent" style="background:{c}22;border-color:{c};">{frag}<span class="lab">{l}</span></span>'
        last=e
    if last<len(text): html+=esc(text[last:])
    return html

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True, help="File JSONL: mỗi dòng {article_id,text,spans}")
    ap.add_argument("--out_html", required=True, help="Đường dẫn HTML đầu ra")
    ap.add_argument("--max_docs", type=int, default=200, help="Giới hạn số doc render để nhẹ file")
    args=ap.parse_args()

    labels=set(); docs=[]
    with open(args.pred_jsonl,"r",encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            line=line.strip()
            if not line: continue
            obj=json.loads(line)
            spans = obj.get("spans", [])
            for s,e,l in spans: labels.add(l)
            docs.append((obj.get("article_id", f"#{i}"), obj.get("text",""), spans))
            if len(docs)>=args.max_docs: break

    legend="".join([f'<div class="chip"><span class="sw" style="background:{color_for(l)}"></span>{l}</div>'
                    for l in sorted(labels)])
    html_docs=""
    for idx,(aid,text,spans) in enumerate(docs,1):
        html_docs+=f'<div class="doc"><div class="head">{idx}. {aid}</div>{render_doc(text,spans)}</div>\n'

    Path(args.out_html).parent.mkdir(parents=True, exist_ok=True)
    open(args.out_html,"w",encoding="utf-8").write(TEMPLATE.format(legend=legend,docs=html_docs))
    print("[HTML] ->", args.out_html)

if __name__=="__main__":
    main()
