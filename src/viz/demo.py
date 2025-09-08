import json, argparse, hashlib
from pathlib import Path

PALETTE=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
def color_for(label): return PALETTE[int(hashlib.md5(label.encode("utf-8")).hexdigest(),16)%len(PALETTE)]
def escape(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def render_doc(text, spans):
    spans=sorted(spans,key=lambda x:x[0]); html=""; last=0
    for s,e,l in spans:
        if s>last: html+=escape(text[last:s])
        frag=escape(text[s:e]); c=color_for(l)
        html+=f'<span class="ent" style="background:{c}22;border-color:{c};">{frag}<span class="lab">{l}</span></span>'
        last=e
    if last<len(text): html+=escape(text[last:])
    return html

TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;line-height:1.6;margin:24px;}}
.doc{{padding:12px 14px;border:1px solid #ddd;border-radius:8px;margin:12px 0;white-space:pre-wrap;}}
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


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred_json", required=True)
    ap.add_argument("--out_html", required=True)
    args=ap.parse_args()

    data=json.load(open(args.pred_json,"r",encoding="utf-8"))
    labels=set(l for d in data for _,__,l in d.get("pred_spans",[]))
    legend="".join([f'<div class="chip"><span class="sw" style="background:{color_for(l)}"></span>{l}</div>'
                    for l in sorted(labels)])
    docs=""
    for i,d in enumerate(data,1):
        docs+=f'<div class="doc"><div style="opacity:.6">#{i}</div>{render_doc(d["text"],d.get("pred_spans",[]))}</div>\n'
    Path(args.out_html).parent.mkdir(parents=True,exist_ok=True)
    open(args.out_html,"w",encoding="utf-8").write(TEMPLATE.format(legend=legend,docs=docs))
    print("[HTML] ->", args.out_html)

if __name__=="__main__":
    main()
