# -*- coding: utf-8 -*-
"""
Memory-lean enrichment (strict chunked write):
- CATS: OOF target encoding (clicked)
- seq: stats + Top-M ratios + hashed BoI + recency
- Batch → inner-chunk 단위로 즉시 write 후 버림 (peak RAM 낮춤)
- 스키마는 '첫 완전 청크'로 고정 (schema mismatch 방지)
- pandas 열 반복 삽입 금지(딕셔너리→DataFrame 한 번에)로 fragmentation 경고 제거
"""
import os, gc, math, json, hashlib
from collections import defaultdict, Counter
from warnings import filterwarnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

# ===== Paths =====
TRAIN_IN  = "./Toss/train.parquet"
TEST_IN   = "./Toss/test.parquet"
OUT_DIR   = "./Toss/_meta"
TRAIN_OUT = f"{OUT_DIR}/train_enriched_2.parquet"
TEST_OUT  = f"{OUT_DIR}/test_enriched_2.parquet"

TARGET = "clicked"
SEQ_COL = "seq"
USER_CATS = ["gender","age_group","inventory_id","day_of_week","hour"]

# ===== Encoding/Seq settings (lean defaults) =====
N_SPLITS = 5
SEED = 42
M_SMOOTH = 50.0
RARE_THR = 20

TOP_M   = 50          # 메모리 절약용
HASH_D  = 128         # 메모리 절약용
DECAY_H = 10.0        # half-life (steps)

BATCH_ROWS = 50_000
INNER_CHUNK_ROWS = 10_000

KEEP_ORIGINALS = False  # (현재 columns 선택으로 이미 원본 미포함)

np.random.seed(SEED)
filterwarnings("ignore", category=FutureWarning)

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def hash_fold(i, k): return (i * 104729 + SEED) % k

def expand_user_cats(user_list, train_schema, test_schema):
    names = set([f.name for f in train_schema]) | set([f.name for f in test_schema])
    out = []
    for pat in user_list:
        if "*" in pat:
            pref = pat.split("*",1)[0]
            out += [n for n in names if n.startswith(pref)]
        else:
            if pat in names: out.append(pat)
    out = [c for c in out if c not in (TARGET, SEQ_COL)]
    return sorted(dict.fromkeys(out))

def cat_series(pdf, col):
    """fillna 대신 NA 마스킹 치환으로 FutureWarning 회피"""
    arr = pdf[col].astype("object").to_numpy(copy=False)
    mask = pd.isna(arr)
    if mask.any():
        arr = arr.copy(); arr[mask] = "__NA__"
    return pd.Series(arr, dtype=object)

def parse_seq(s):
    if not isinstance(s, str) or not s: return []
    out = []
    for t in s.split(","):
        t = t.strip()
        if not t: continue
        try: out.append(int(t))
        except: pass
    return out

def hash_idx(x):
    h = hashlib.blake2b(str(x).encode(), digest_size=8).hexdigest()
    return int(h, 16) % HASH_D

def seq_row_feats(ids, vocab):
    """row 단위 seq 통계 + ratio + 해시 카운트(정규화는 나중)"""
    n = len(ids)
    if n == 0:
        row = dict(
            seq_len=0, uniq_cnt=0, uniq_ratio=np.float32(0.0),
            top1_id=0, top1_share=np.float32(0.0), top2_share=np.float32(0.0),
            last_id=0, entropy=np.float32(0.0), rep_run_max=0, decayed_unique=np.float32(0.0)
        )
        for v in vocab: row[f"ratio_id_{v}"] = np.float32(0.0)
        row["__hash_counts__"] = {}
        return row

    arr = np.asarray(ids, dtype=np.int64)
    n = int(arr.size)

    cnt = Counter(ids)
    uniq_cnt   = int(len(cnt))
    uniq_ratio = np.float32(uniq_cnt / n)

    mc2 = cnt.most_common(2)
    top1_id    = int(mc2[0][0])
    top1_share = np.float32(mc2[0][1] / n)
    top2_share = np.float32(mc2[1][1] / n) if len(mc2) > 1 else np.float32(0.0)

    # max run length
    run_max, cur = 1, 1
    for a, b in zip(arr[:-1], arr[1:]):
        if a == b:
            cur += 1; run_max = max(run_max, cur)
        else:
            cur = 1

    # entropy
    freqs = np.fromiter((v/n for v in cnt.values()), dtype=np.float32)
    entropy = np.float32(-(freqs * np.log(freqs + 1e-12)).sum())

    # decayed_unique (벡터 생성 없이 스칼라로 안전/저메모리)
    seen = set()
    decu = np.float32(0.0)
    if n > 0 and DECAY_H > 0:
        decay = math.exp(-math.log(2.0) / max(DECAY_H, 1e-9))
        weight = 1.0  # step 0
        for i in range(n-1, -1, -1):
            t = arr[i]
            if t not in seen:
                decu += np.float32(weight)
                seen.add(t)
            weight *= decay

    row = dict(
        seq_len=n,
        uniq_cnt=uniq_cnt,
        uniq_ratio=uniq_ratio,
        top1_id=top1_id,
        top1_share=top1_share,
        top2_share=top2_share,
        last_id=int(arr[-1]),
        entropy=entropy,
        rep_run_max=int(run_max),
        decayed_unique=decu
    )

    # Top-M ratios
    for v in vocab:
        row[f"ratio_id_{v}"] = np.float32(cnt.get(v, 0) / n)

    # 해시 카운트(정규화는 호출부에서)
    row["__hash_counts__"] = cnt
    return row

# ===== 0) schema & cats =====
pf_tr = pq.ParquetFile(TRAIN_IN); pf_te = pq.ParquetFile(TEST_IN)
CATS = expand_user_cats(USER_CATS, pf_tr.schema, pf_te.schema)
print(f"[CATS] ({len(CATS)}): {CATS}")

# ===== 1) PASS1: stats + vocab =====
print("[PASS1] scanning train ...")
global_sum = 0; global_cnt = 0
tot_sum = {c: defaultdict(float) for c in CATS}
tot_cnt = {c: defaultdict(int) for c in CATS}
fold_sum = {c: [defaultdict(float) for _ in range(N_SPLITS)] for c in CATS}
fold_cnt = {c: [defaultdict(int) for _ in range(N_SPLITS)] for c in CATS}
seq_counter = Counter()
empty_counts = {"null":0,"empty_str":0,"parsed_empty":0}
row_idx_global = 0

pbar1 = tqdm(total=pf_tr.metadata.num_rows or 0, desc="[PASS1 train]", unit="rows", unit_scale=True)
for batch in pf_tr.iter_batches(batch_size=BATCH_ROWS, columns=CATS+[TARGET, SEQ_COL]):
    pdf = batch.to_pandas()
    y = pd.to_numeric(pdf[TARGET], errors="coerce").fillna(0).astype(int).values
    global_sum += int(y.sum()); global_cnt += int(y.size)
    n_rows = len(pdf)
    fold_ids = np.fromiter((hash_fold(i, N_SPLITS) for i in range(row_idx_global, row_idx_global+n_rows)), dtype=np.int32)
    row_idx_global += n_rows

    for c in CATS:
        col = cat_series(pdf, c)
        for f in range(N_SPLITS):
            m = (fold_ids==f)
            if not m.any(): continue
            xs = col[m].values; ys = y[m]
            vc = pd.Series(xs).value_counts()
            for val,cntv in vc.items(): fold_cnt[c][f][val]+=int(cntv)
            tmp = defaultdict(int)
            for vv,yy in zip(xs,ys): tmp[vv]+=int(yy)
            for val,s in tmp.items(): fold_sum[c][f][val]+=float(s)
        vc_all = col.value_counts()
        for val,cntv in vc_all.items(): tot_cnt[c][val]+=int(cntv)
        tmp_all = defaultdict(int)
        for vv,yy in zip(col.values,y): tmp_all[vv]+=int(yy)
        for val,s in tmp_all.items(): tot_sum[c][val]+=float(s)

    if SEQ_COL in pdf.columns:
        s = pdf[SEQ_COL].astype("object")
        is_null = s.isna(); empty_counts["null"] += int(is_null.sum())
        s2 = s[~is_null].astype("string"); is_empty = s2.str.strip().fillna("").str.len().eq(0)
        empty_counts["empty_str"] += int(is_empty.sum())
        cand = s2[~is_empty]
        for v in cand:
            ids = parse_seq(v)
            if len(ids)==0: empty_counts["parsed_empty"]+=1
            else: seq_counter.update(ids)

    pbar1.update(n_rows)
    del pdf, batch; gc.collect()
pbar1.close()

prior = global_sum/max(1,global_cnt)
print(f"[INFO] prior={prior:.6f}  | empty seq: {empty_counts}")

rare_vals = {c: set([k for k,v in tot_cnt[c].items() if v<RARE_THR]) for c in CATS}
def collapse(dct, rares):
    out = defaultdict(type(next(iter(dct.values()))) if dct else float)
    rk = "__RARE__"
    for k, v in dct.items():
        if k in rares:
            out[rk] += v
        else:
            out[k] += v
    return out
tot_sum_c = {c: collapse(tot_sum[c], rare_vals[c]) for c in CATS}
tot_cnt_c = {c: collapse(tot_cnt[c], rare_vals[c]) for c in CATS}
fold_sum_c = {c: [collapse(fold_sum[c][f], rare_vals[c]) for f in range(N_SPLITS)] for c in CATS}
fold_cnt_c = {c: [collapse(fold_cnt[c][f], rare_vals[c]) for f in range(N_SPLITS)] for c in CATS}

vocab = [tid for tid,_ in seq_counter.most_common(TOP_M)]
print(f"[INFO] seq Top-{TOP_M} (head): {vocab[:12]}")

# ---- 고정 컬럼 순서 (스키마 고정용) ----
SEQ_FIXED   = ["seq_len","uniq_cnt","uniq_ratio","top1_id","top1_share",
               "top2_share","last_id","entropy","rep_run_max","decayed_unique"]
TE_NAMES    = [f"{c}_te" for c in CATS]
RATIO_NAMES = [f"ratio_id_{v}" for v in vocab]
H_NAMES     = [f"h{j}" for j in range(HASH_D)]
COLS_TRAIN  = TE_NAMES + SEQ_FIXED + RATIO_NAMES + H_NAMES + [TARGET]
COLS_TEST   = TE_NAMES + SEQ_FIXED + RATIO_NAMES + H_NAMES

# ===== helper: build a full table for a small chunk =====
def build_chunk_table(seq_series_slice, te_map_slice, y_slice=None, is_train=True):
    """
    - seq/TE 청크를 받아 컬럼 순서를 고정(COLS_*)하여 Arrow Table 생성
    - pandas에 열을 반복 삽입하지 않고 dict→DataFrame 한 번에 생성(경고 제거)
    """
    n = len(seq_series_slice)

    # ---- seq 계산 버퍼 ----
    fixed = {name: np.zeros(n, np.int32 if name in {"seq_len","uniq_cnt","top1_id","last_id","rep_run_max"} else np.float32)
             for name in SEQ_FIXED}
    ratio = {name: np.zeros(n, np.float32) for name in RATIO_NAMES}
    hbuf  = np.zeros((n, HASH_D), np.float32)

    # 채우기
    for i, v in enumerate(seq_series_slice.astype("object")):
        ids = parse_seq(v)
        fr = seq_row_feats(ids, vocab)
        fixed["seq_len"][i]        = fr["seq_len"]
        fixed["uniq_cnt"][i]       = fr["uniq_cnt"]
        fixed["uniq_ratio"][i]     = fr["uniq_ratio"]
        fixed["top1_id"][i]        = fr["top1_id"]
        fixed["top1_share"][i]     = fr["top1_share"]
        fixed["top2_share"][i]     = fr["top2_share"]
        fixed["last_id"][i]        = fr["last_id"]
        fixed["entropy"][i]        = fr["entropy"]
        fixed["rep_run_max"][i]    = fr["rep_run_max"]
        fixed["decayed_unique"][i] = fr["decayed_unique"]

        for v_id in vocab:
            ratio[f"ratio_id_{v_id}"][i] = fr[f"ratio_id_{v_id}"]

        cnts = fr["__hash_counts__"]
        if cnts:
            for tid, cntv in cnts.items():
                hbuf[i, hash_idx(tid)] += cntv
            if fr["seq_len"] > 0:
                hbuf[i, :] /= fr["seq_len"]

    # ---- dict → DataFrame (정해진 순서) ----
    data = {}
    for name in TE_NAMES: data[name] = te_map_slice[name].astype(np.float32, copy=False)
    for name in SEQ_FIXED: data[name] = fixed[name]
    for name in RATIO_NAMES: data[name] = ratio[name]
    for j, name in enumerate(H_NAMES): data[name] = hbuf[:, j]
    if is_train: data[TARGET] = y_slice.astype(np.int8, copy=False)

    cols = COLS_TRAIN if is_train else COLS_TEST
    df = pd.DataFrame(data, columns=cols)
    table = pa.Table.from_pandas(df, preserve_index=False)

    del df, data, fixed, ratio, hbuf
    gc.collect()
    return table

# ===== 2) PASS2: train_enriched =====
ensure_dir(OUT_DIR)
writer_tr = None
pf_tr2 = pq.ParquetFile(TRAIN_IN)
pbar2 = tqdm(total=pf_tr2.metadata.num_rows or 0, desc="[PASS2 train->enriched]", unit="rows", unit_scale=True)
row_idx_global = 0

for batch in pf_tr2.iter_batches(batch_size=BATCH_ROWS, columns=CATS+[TARGET, SEQ_COL]):
    pdf = batch.to_pandas()
    n_rows = len(pdf)
    fold_ids = np.fromiter((hash_fold(i, N_SPLITS) for i in range(row_idx_global, row_idx_global+n_rows)), dtype=np.int32)
    row_idx_global += n_rows

    # TE (배치 전체 계산)
    te_map = {}
    for c in CATS:
        col = cat_series(pdf, c).values
        rset = rare_vals[c]
        te_vals = np.empty(n_rows, np.float32)
        for i,(val,f) in enumerate(zip(col, fold_ids)):
            key = "__RARE__" if val in rset else val
            ts, tn = tot_sum_c[c].get(key,0.0), tot_cnt_c[c].get(key,0)
            fs, fn = fold_sum_c[c][f].get(key,0.0), fold_cnt_c[c][f].get(key,0)
            s, n = ts - fs, tn - fn
            te_vals[i] = (s + M_SMOOTH*prior) / (n + M_SMOOTH) if n>0 else prior
        te_map[f"{c}_te"] = te_vals

    y_all = pd.to_numeric(pdf[TARGET], errors="coerce").fillna(0).astype(np.int8).values

    # 내부 청크별 처리/쓰기
    for start in range(0, n_rows, INNER_CHUNK_ROWS):
        end = min(start+INNER_CHUNK_ROWS, n_rows)
        te_slice = {k: v[start:end] for k,v in te_map.items()}
        seq_slice = pdf[SEQ_COL].iloc[start:end]
        y_slice  = y_all[start:end]

        table = build_chunk_table(seq_slice, te_slice, y_slice, is_train=True)

        if writer_tr is None:
            writer_tr = pq.ParquetWriter(TRAIN_OUT, table.schema, compression="zstd")
        writer_tr.write_table(table)

        pbar2.update(end-start)

    del pdf, batch, te_map
    gc.collect()

pbar2.close()
if writer_tr: writer_tr.close()
print(f"[OK] saved -> {TRAIN_OUT}")

# ===== 3) PASS3: test_enriched =====
writer_te = None
pf_te2 = pq.ParquetFile(TEST_IN)
pbar3 = tqdm(total=pf_te2.metadata.num_rows or 0, desc="[PASS3 test->enriched]", unit="rows", unit_scale=True)

for batch in pf_te2.iter_batches(batch_size=BATCH_ROWS, columns=CATS+[SEQ_COL]):
    pdf = batch.to_pandas()
    n_rows = len(pdf)

    # TE (풀-트레인 매핑)
    te_map = {}
    for c in CATS:
        col = cat_series(pdf, c).values
        rset = rare_vals[c]
        te_vals = np.empty(n_rows, np.float32)
        for i, val in enumerate(col):
            key = "__RARE__" if val in rset else val
            s, n = tot_sum_c[c].get(key,0.0), tot_cnt_c[c].get(key,0)
            te_vals[i] = (s + M_SMOOTH*prior) / (n + M_SMOOTH) if n>0 else prior
        te_map[f"{c}_te"] = te_vals

    for start in range(0, n_rows, INNER_CHUNK_ROWS):
        end = min(start+INNER_CHUNK_ROWS, n_rows)
        te_slice = {k: v[start:end] for k,v in te_map.items()}
        seq_slice = pdf[SEQ_COL].iloc[start:end]

        table = build_chunk_table(seq_slice, te_slice, y_slice=None, is_train=False)

        if writer_te is None:
            writer_te = pq.ParquetWriter(TEST_OUT, table.schema, compression="zstd")
        writer_te.write_table(table)

        pbar3.update(end-start)

    del pdf, batch, te_map
    gc.collect()

pbar3.close()
if writer_te: writer_te.close()
print(f"[OK] saved -> {TEST_OUT}")

# ===== 4) meta =====
meta = dict(
  CATS=CATS, USER_CATS=USER_CATS, prior=float(prior),
  N_SPLITS=N_SPLITS, M_SMOOTH=M_SMOOTH, RARE_THR=RARE_THR,
  SEQ=dict(TOP_M=TOP_M, HASH_D=HASH_D, DECAY_H=DECAY_H),
  BATCH_ROWS=BATCH_ROWS, INNER_CHUNK_ROWS=INNER_CHUNK_ROWS,
  notes="strict chunked writer; fixed schema; no pandas fragmentation; safe decayed_unique"
)
ensure_dir(OUT_DIR)
with open(f"{OUT_DIR}/enrich_meta.json","w",encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("[META] saved")
