import re, json, argparse, random
from pathlib import Path
from collections import defaultdict, Counter

RIGHT_PUNCT = set(".,;:!?…)]}»”’")
SPACE = set(" \t\r\n")
SENT_RGX = re.compile(r"([\.!\?…]+)(\s+)")  # giữ cả dấu + khoảng trắng sau dấu

# ---------- IO ----------
def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for i, ln in enumerate(f, 1):
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception as e:
                print(f"[WARN] line {i} JSON error: {e}")

def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- SPAN PARSER (đa định dạng) ----------
def parse_spans_generic(obj):
    """
    Trả về list[[s,e,label]] từ nhiều kiểu schema:
      - gold_spans / spans / entities / labels / annotations
      - item dạng list/tuple hoặc dict với các key phổ biến
    """
    cand = (obj.get("gold_spans") or obj.get("spans") or obj.get("entities")
            or obj.get("labels") or obj.get("annotations") or [])
    out = []
    for x in cand:
        if isinstance(x, (list, tuple)) and len(x) >= 3:
            s, e, l = x[0], x[1], x[2]
        elif isinstance(x, dict):
            s = x.get("start", x.get("begin", x.get("s", x.get("start_idx", x.get("startOffset", -1)))))
            e = x.get("end",   x.get("finish", x.get("e", x.get("end_idx",   x.get("endOffset",   -1)))))
            l = x.get("label", x.get("type",   x.get("tag", x.get("category", x.get("entity", "")))))
        else:
            continue
        try:
            s = int(s); e = int(e); l = str(l)
        except Exception:
            continue
        out.append([s, e, l])
    return out

def norm_trim(text, spans, trim_right_punct=True):
    """
    Bỏ space 2 đầu; tùy chọn bỏ dấu câu ở đuôi.
    KHÔNG sửa 'text'.
    """
    out = []
    for s, e, l in spans:
        if not (isinstance(s, int) and isinstance(e, int) and isinstance(l, str)):
            continue
        if not (0 <= s < e <= len(text)):
            continue

        # left spaces
        while s < e and text[s] in SPACE: s += 1
        # right spaces
        while e > s and text[e-1] in SPACE: e -= 1
        # right punct
        if trim_right_punct:
            while e > s and text[e-1] in RIGHT_PUNCT:
                e -= 1
            while e > s and text[e-1] in SPACE:
                e -= 1

        if s < e:
            out.append([s, e, l])
    return out

# ---------- sentence split ----------
def sentence_segments(text):
    segs=[]; st=0
    for m in SENT_RGX.finditer(text):
        ed=m.end(); segs.append((st, ed)); st=ed
    if st < len(text): segs.append((st, len(text)))
    return [(s,e) for s,e in segs if text[s:e].strip()]

def project_spans(text, spans, seg_s, seg_e):
    loc=[]
    for s,e,l in spans:
        if e <= seg_s or s >= seg_e: continue
        ns=max(s, seg_s); ne=min(e, seg_e)
        if ns < ne: loc.append([ns-seg_s, ne-seg_s, l])
    return loc

def stratified_split(items, ratios=(0.8,0.1,0.1), seed=42):
    random.seed(seed)
    buckets=defaultdict(list)
    for it in items:
        labs=tuple(sorted({l for _,_,l in it["gold_spans"]}))
        buckets[labs].append(it)
    tr=va=te=[]
    train=[]; val=[]; test=[]
    for arr in buckets.values():
        random.shuffle(arr)
        n=len(arr); n_tr=int(round(ratios[0]*n)); n_va=int(round(ratios[1]*n))
        train += arr[:n_tr]; val += arr[n_tr:n_tr+n_va]; test += arr[n_tr+n_va:]
    return train, val, test

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_split", type=int, default=0, help="1 = không tách câu; giữ mỗi dòng là 1 câu")
    ap.add_argument("--keep_empty_after_trim", type=int, default=1,
                    help="1 = giữ câu ngay cả khi sau trim không còn span; 0 = loại")
    ap.add_argument("--no_trim_punct", type=int, default=0,
                    help="1 = KHÔNG bỏ dấu câu ở đuôi span (debug khi bị rỗng hàng loạt)")
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    src = list(read_jsonl(args.input_jsonl))
    if args.verbose: print(f"[LOAD] lines: {len(src)}")

    # B1: chuẩn hóa từng dòng
    docs = []
    dropped_bad = 0
    for i,o in enumerate(src,1):
        text = o.get("text") or (" ".join(o["words"]) if "words" in o else "")
        if not isinstance(text,str): text = str(text or "")
        spans = parse_spans_generic(o)
        spans = norm_trim(text, spans, trim_right_punct=(args.no_trim_punct==0))
        if not text.strip():
            dropped_bad += 1
            continue
        docs.append({"text": text, "spans": spans})
    if args.verbose:
        tot_span = sum(len(d["spans"]) for d in docs)
        print(f"[NORM] docs: {len(docs)} | spans: {tot_span} | dropped_empty_text: {dropped_bad}")

    # B2: tách câu (có thể tắt)
    sents=[]
    for d in docs:
        text=d["text"]; spans=d["spans"]
        if args.no_split:
            loc = norm_trim(text, spans, trim_right_punct=(args.no_trim_punct==0))  # đảm bảo sạch lần nữa
            if args.keep_empty_after_trim or loc:
                sents.append({"text": text, "gold_spans": loc})
            continue

        for s,e in sentence_segments(text):
            sub=text[s:e]
            loc=project_spans(text, spans, s, e)
            loc=norm_trim(sub, loc, trim_right_punct=(args.no_trim_punct==0))
            if args.keep_empty_after_trim or loc:
                sents.append({"text": sub, "gold_spans": loc})

    if args.verbose:
        tot_span = sum(len(d["gold_spans"]) for d in sents)
        print(f"[SENT] sentences: {len(sents)} | spans: {tot_span}")

    # B3: split
    train, val, test = stratified_split(sents, (0.8,0.1,0.1), seed=args.seed)
    out = Path(args.out_dir)
    write_jsonl(out/"train.jsonl", train)
    write_jsonl(out/"val.jsonl",   val)
    write_jsonl(out/"test.jsonl",  test)

    # B4: thống kê
    def stat(items):
        c=Counter(l for it in items for _,_,l in it["gold_spans"])
        return sum(c.values()), dict(c)
    all_n, all_c = stat(sents)
    tr_n, tr_c   = stat(train)
    va_n, va_c   = stat(val)
    te_n, te_c   = stat(test)

    print(f"[CHECK] sentences: {len(sents)} | spans: {all_n}")
    print(f"[DATA] all: {all_n} {all_c}")
    print(f"[SPLIT] train/val/test lines: {len(train)} {len(val)} {len(test)}")
    print(f"[SPLIT] spans  train/val/test: {tr_n} {va_n} {te_n}")
    print(f"[DONE] -> {out.resolve()}")

if __name__ == "__main__":
    main()
