# -*- coding: utf-8 -*-
"""
Unified CTR preprocessing protocol:
1) Memory-lean enrichment (strict chunked write)
   - Target encoding (OOF) for selected categorical columns
   - Sequence features (stats, Top-M ratios, hashed BoI, recency)
   - Fixed schema, batch→inner-chunk streaming to keep low peak RAM
2) Postprocess for model-ready tables
   - Rare category bucketing + NA level
   - Ordinal encoding for cats
   - QuantileTransformer (gaussianize) for numerics (bools excluded)
Outputs:
  - ./Toss/_meta/train_enriched_2.parquet / test_enriched_2.parquet
  - ./Toss/new_data/new_train.parquet / new_test.parquet
"""
import os, gc, math, json, hashlib, warnings
from collections import defaultdict, Counter
from warnings import filterwarnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from pandas.api.types import is_numeric_dtype, is_bool_dtype
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer

# ========= Global config =========
SEED = 42
np.random.seed(SEED)
warnings.filterwarnings("ignore")
filterwarnings("ignore", category=FutureWarning)

# ----- I/O paths -----
RAW_TRAIN  = "./Toss/train.parquet"
RAW_TEST   = "./Toss/test.parquet"
META_DIR   = "./Toss/_meta"
TRAIN_ENR  = f"{META_DIR}/train_enriched_3.parquet"
TEST_ENR   = f"{META_DIR}/test_enriched_3.parquet"

NEW_DIR    = "./Toss/new_data"
NEW_TRAIN  = f"{NEW_DIR}/new_train_2.parquet"
NEW_TEST   = f"{NEW_DIR}/new_test_2.parquet"

# ----- Columns & target -----
TARGET  = "clicked"
SEQ_COL = "seq"
# support wildcard prefix like "user_*" by writing "user_*" in this list
USER_CATS = ["gender", "age_group", "inventory_id", "day_of_week", "hour"]

# ----- Enrichment knobs -----
N_SPLITS  = 5        # OOF folds
M_SMOOTH  = 50.0     # TE smoothing
RARE_THR  = 20       # TE rare collapse threshold
TOP_M     = 50       # seq: top-M ids for ratios
HASH_D    = 128      # seq: hashed bag-of-ids dims
DECAY_H   = 10.0     # seq recency half-life (steps)
BATCH_ROWS        = 50_000
INNER_CHUNK_ROWS  = 10_000

# ----- Postprocess knobs -----
RARE_MIN_COUNT = 50        # rare bucketing for category columns
N_QUANTILES    = 512       # QuantileTransformer bins
SUBSAMPLE      = 200_000   # QuantileTransformer subsample

# ========= Utils =========
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def hash_fold(i, k): return (i * 104729 + SEED) % k

def expand_user_cats(user_list, train_schema, test_schema):
    names = set([f.name for f in train_schema]) | set([f.name for f in test_schema])
    out = []
    for pat in user_list:
        if "*" in pat:
            pref = pat.split("*", 1)[0]
            out += [n for n in names if n.startswith(pref)]
        else:
            if pat in names: out.append(pat)
    out = [c for c in out if c not in (TARGET, SEQ_COL)]
    return sorted(dict.fromkeys(out))

def cat_series(pdf, col):
    """object series with '__NA__' for missing (avoid future warnings)"""
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
    """Row-level sequence stats + top-M ratios + hashed counts (normalized later)."""
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

    # decayed_unique
    seen = set()
    decu = np.float32(0.0)
    if n > 0 and DECAY_H > 0:
        decay = math.exp(-math.log(2.0) / max(DECAY_H, 1e-9))
        weight = 1.0
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

    # hashed counts (normalized by length later)
    row["__hash_counts__"] = cnt
    return row

# ========= Enrichment (PASS1~3) =========
def run_enrichment():
    ensure_dir(META_DIR)

    pf_tr = pq.ParquetFile(RAW_TRAIN); pf_te = pq.ParquetFile(RAW_TEST)
    CATS = expand_user_cats(USER_CATS, pf_tr.schema, pf_te.schema)
    print(f"[CATS] ({len(CATS)}): {CATS}")

    # === 원본 컬럼 보존 (schema 순서 유지) ===
    def schema_names_union(tr_schema, te_schema):
        seen, out = set(), []
        for f in list(tr_schema) + list(te_schema):
            n = f.name
            if n not in seen:
                seen.add(n); out.append(n)
        return out

    ALL_NAMES = schema_names_union(pf_tr.schema, pf_te.schema)
    ORIG_COLS = [c for c in ALL_NAMES if c not in (TARGET, SEQ_COL)]
    print(f"[CATS] ({len(CATS)}): {CATS}")

    # PASS1: scan train to get OOF TE stats + vocab
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
            # per-fold
            for f in range(N_SPLITS):
                m = (fold_ids==f)
                if not m.any(): continue
                xs = col[m].values; ys = y[m]
                vc = pd.Series(xs).value_counts()
                for val,cntv in vc.items(): fold_cnt[c][f][val]+=int(cntv)
                tmp = defaultdict(int)
                for vv,yy in zip(xs,ys): tmp[vv]+=int(yy)
                for val,s in tmp.items(): fold_sum[c][f][val]+=float(s)
            # global
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

    # Fixed schema
    # Fixed schema
    SEQ_FIXED   = ["seq_len","uniq_cnt","uniq_ratio","top1_id","top1_share",
                   "top2_share","last_id","entropy","rep_run_max","decayed_unique"]
    TE_NAMES    = [f"{c}_te" for c in CATS]
    RATIO_NAMES = [f"ratio_id_{v}" for v in vocab]
    H_NAMES     = [f"h{j}" for j in range(HASH_D)]

    # ★ 원본 컬럼을 맨 앞에 보존
    COLS_TRAIN  = ORIG_COLS + TE_NAMES + SEQ_FIXED + RATIO_NAMES + H_NAMES + [TARGET]
    COLS_TEST   = ORIG_COLS + TE_NAMES + SEQ_FIXED + RATIO_NAMES + H_NAMES

    def build_chunk_table(seq_series_slice, te_map_slice, orig_df_slice, y_slice=None, is_train=True):
        """Build Arrow table for a slice keeping fixed column order."""
        n = len(seq_series_slice)

        # ---- seq features 계산 (그대로) ----
        fixed = {name: np.zeros(n, np.int32 if name in {"seq_len","uniq_cnt","top1_id","last_id","rep_run_max"} else np.float32)
                 for name in SEQ_FIXED}
        ratio = {name: np.zeros(n, np.float32) for name in RATIO_NAMES}
        hbuf  = np.zeros((n, HASH_D), np.float32)

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

        # ---- dict 구성: ①원본 → ②TE → ③seq 고정/ratio/hash → ④타깃 ----
        data = {}

        # ① 원본 컬럼 보존
        for name in ORIG_COLS:
            data[name] = orig_df_slice[name].to_numpy(copy=False)

        # ② TE
        for name in TE_NAMES:
            data[name] = te_map_slice[name].astype(np.float32, copy=False)

        # ③ seq 고정/ratio/hash
        for name in SEQ_FIXED:     data[name] = fixed[name]
        for name in RATIO_NAMES:   data[name] = ratio[name]
        for j, name in enumerate(H_NAMES): data[name] = hbuf[:, j]

        # ④ 타깃
        if is_train:
            data[TARGET] = y_slice.astype(np.int8, copy=False)

        cols = COLS_TRAIN if is_train else COLS_TEST
        df = pd.DataFrame(data, columns=cols)
        table = pa.Table.from_pandas(df, preserve_index=False)

        del df, data, fixed, ratio, hbuf
        gc.collect()
        return table


    # PASS2: train_enriched
    ensure_dir(META_DIR)
    writer_tr = None
    pf_tr2 = pq.ParquetFile(RAW_TRAIN)
    pbar2 = tqdm(total=pf_tr2.metadata.num_rows or 0, desc="[PASS2 train->enriched]", unit="rows", unit_scale=True)
    row_idx_global = 0

    # train 파일에 실제로 존재하는 원본 컬럼만
    ORIG_COLS_TRAIN = [c for c in ORIG_COLS if c in pf_tr2.schema.names]
    # ★ 배치에서 읽을 컬럼 = (원본 ∪ CATS ∪ {TARGET, SEQ_COL})
    BATCH_COLS_TRAIN = sorted(set(ORIG_COLS_TRAIN) | set(CATS) | {TARGET, SEQ_COL})

    for batch in pf_tr2.iter_batches(batch_size=BATCH_ROWS, columns=BATCH_COLS_TRAIN):
        pdf = batch.to_pandas()
        n_rows = len(pdf)
        fold_ids = np.fromiter((hash_fold(i, N_SPLITS) for i in range(row_idx_global, row_idx_global+n_rows)),
                            dtype=np.int32)
        row_idx_global += n_rows

        # ★ te_map을 먼저 prior로 꽉 채워서 초기화 (컬럼이 빠져도 안전)
        te_map = {f"{c}_te": np.full(n_rows, prior, dtype=np.float32) for c in CATS}

        # CATS가 pdf에 있으면 OOF TE로 덮어쓰기
        for c in CATS:
            if c not in pdf.columns:
                continue
            col = cat_series(pdf, c).values
            rset = rare_vals[c]
            te_vals = np.empty(n_rows, np.float32)
            for i, (val, f) in enumerate(zip(col, fold_ids)):
                key = "__RARE__" if val in rset else val
                ts, tn = tot_sum_c[c].get(key, 0.0), tot_cnt_c[c].get(key, 0)
                fs, fn = fold_sum_c[c][f].get(key, 0.0), fold_cnt_c[c][f].get(key, 0)
                s, n = ts - fs, tn - fn
                te_vals[i] = (s + M_SMOOTH * prior) / (n + M_SMOOTH) if n > 0 else prior
            te_map[f"{c}_te"] = te_vals  # 덮어쓰기

        y_all = pd.to_numeric(pdf[TARGET], errors="coerce").fillna(0).astype(np.int8).values

        for start in range(0, n_rows, INNER_CHUNK_ROWS):
            end       = min(start + INNER_CHUNK_ROWS, n_rows)
            te_slice  = {k: v[start:end] for k, v in te_map.items()}
            seq_slice = pdf[SEQ_COL].iloc[start:end]

            # 원본 스키마 고정(NaN 채움)
            orig_slice = pdf.iloc[start:end].reindex(columns=ORIG_COLS, fill_value=np.nan)
            y_slice    = y_all[start:end]

            table = build_chunk_table(seq_slice, te_slice, orig_slice, y_slice, is_train=True)

            if writer_tr is None:
                writer_tr = pq.ParquetWriter(TRAIN_ENR, table.schema, compression="zstd")
            writer_tr.write_table(table)
            pbar2.update(end - start)

        del pdf, batch, te_map
        gc.collect()

    pbar2.close()
    if writer_tr: writer_tr.close()
    print(f"[OK] saved -> {TRAIN_ENR}")

    # PASS3: test_enriched
    writer_te = None
    pf_te2 = pq.ParquetFile(RAW_TEST)
    pbar3 = tqdm(total=pf_te2.metadata.num_rows or 0, desc="[PASS3 test->enriched]", unit="rows", unit_scale=True)

    ORIG_COLS_TEST = [c for c in ORIG_COLS if c in pf_te2.schema.names]
    # ★ 배치에서 읽을 컬럼 = (원본 ∪ CATS ∪ {SEQ_COL})
    BATCH_COLS_TEST = sorted(set(ORIG_COLS_TEST) | set(CATS) | {SEQ_COL})

    for batch in pf_te2.iter_batches(batch_size=BATCH_ROWS, columns=BATCH_COLS_TEST):
        pdf = batch.to_pandas()
        n_rows = len(pdf)

        # ★ te_map prior 초기화
        te_map = {f"{c}_te": np.full(n_rows, prior, dtype=np.float32) for c in CATS}

        # 전체-train 통계 기반 TE로 덮어쓰기
        for c in CATS:
            if c not in pdf.columns:
                continue
            col = cat_series(pdf, c).values
            rset = rare_vals[c]
            te_vals = np.empty(n_rows, np.float32)
            for i, val in enumerate(col):
                key = "__RAR E__" if val in rset else val
                s, n = tot_sum_c[c].get(key, 0.0), tot_cnt_c[c].get(key, 0)
                te_vals[i] = (s + M_SMOOTH * prior) / (n + M_SMOOTH) if n > 0 else prior
            te_map[f"{c}_te"] = te_vals

        for start in range(0, n_rows, INNER_CHUNK_ROWS):
            end       = min(start + INNER_CHUNK_ROWS, n_rows)
            te_slice  = {k: v[start:end] for k, v in te_map.items()}
            seq_slice = pdf[SEQ_COL].iloc[start:end]
            orig_slice = pdf.iloc[start:end].reindex(columns=ORIG_COLS, fill_value=np.nan)

            table = build_chunk_table(seq_slice, te_slice, orig_slice, y_slice=None, is_train=False)

            if writer_te is None:
                writer_te = pq.ParquetWriter(TEST_ENR, table.schema, compression="zstd")
            writer_te.write_table(table)
            pbar3.update(end - start)

        del pdf, batch, te_map
        gc.collect()

    pbar3.close()
    if writer_te: writer_te.close()
    print(f"[OK] saved -> {TEST_ENR}")

# ========= Postprocess to model-ready tables =========
def run_postprocess():
    ensure_dir(NEW_DIR)

    train_all = pd.read_parquet(TRAIN_ENR, engine="pyarrow")
    test_df   = pd.read_parquet(TEST_ENR,  engine="pyarrow")
    print("Train shape:", train_all.shape, "| clicked==1:", int((train_all["clicked"]==1).sum()))
    print("Test  shape:", test_df.shape)
    
    # === 주기성(sin/cos) 피처 추가: hour, day_of_week ===
    def _first_exist(df, names):
        for n in names:
            if n in df.columns:
                return n
        return None

    # 후보 이름(프로덕션에서 흔한 alias 포함)
    hour_name = _first_exist(train_all, ["hour","Hour","hour_of_day","hr"])
    dow_name  = _first_exist(train_all, ["day_of_week","dow","dayofweek","DayOfWeek"])

    def add_cyc(df, hour_col, dow_col):
        if hour_col is not None:
            h = pd.to_numeric(df[hour_col], errors="coerce") % 24
            df["Sin_hour"] = np.sin(2*np.pi*h/24.0).astype(np.float32)
            df["Cos_hour"] = np.cos(2*np.pi*h/24.0).astype(np.float32)
        if dow_col is not None:
            d = pd.to_numeric(df[dow_col],  errors="coerce") % 7
            df["Sin_dow"]  = np.sin(2*np.pi*d/7.0).astype(np.float32)
            df["Cos_dow"]  = np.cos(2*np.pi*d/7.0).astype(np.float32)

    add_cyc(train_all, hour_name, dow_name)
    add_cyc(test_df,  hour_name, dow_name)

    # full train (no sampling); keep API if you want to add sampling later
    train = train_all.copy()
    print("↓ Using FULL train (no sampling)")
    print("Train shape:", train.shape,
          "| clicked=1:", int((train["clicked"]==1).sum()),
          "| clicked=0:", int((train["clicked"]==0).sum()))

    target = TARGET
    feature_cols = [c for c in train.columns if c != target]

    dt_cols  = [c for c in feature_cols if np.issubdtype(train[c].dtype, np.datetime64)]
    cat_cols = [c for c in feature_cols if (train[c].dtype == "object") or pd.api.types.is_categorical_dtype(train[c])]
    num_cols = [c for c in feature_cols if (is_numeric_dtype(train[c]) or is_bool_dtype(train[c])) and c not in dt_cols]

    bool_cols  = [c for c in num_cols if is_bool_dtype(train[c])]
    gauss_cols = [c for c in num_cols if c not in bool_cols]

    # ★ 주기성 컬럼 제외
    cyc_cols = [c for c in ["Sin_hour","Cos_hour","Sin_dow","Cos_dow"] if c in gauss_cols]
    gauss_cols = [c for c in gauss_cols if c not in cyc_cols]

    # force objects
    if cat_cols:
        train.loc[:, cat_cols]   = train[cat_cols].astype("object")
        test_df.loc[:, cat_cols] = test_df[cat_cols].astype("object")

    # Rare bucketing + NA level
    for c in cat_cols:
        vc = train[c].value_counts(dropna=True)
        rare_vals = set(vc[vc < RARE_MIN_COUNT].index.tolist())
        if rare_vals:
            train[c]   = train[c].where(~train[c].isin(rare_vals), "__RARE__")
            test_df[c] = test_df[c].where(~test_df[c].isin(rare_vals), "__RARE__")
        train[c]   = train[c].where(train[c].notna(), "__NA__")
        test_df[c] = test_df[c].where(test_df[c].notna(), "__NA__")

    # Ordinal encode categories (if any)
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        train[cat_cols]   = enc.fit_transform(train[cat_cols])
        test_df[cat_cols] = enc.transform(test_df[cat_cols])
        train[cat_cols]   = train[cat_cols].astype(np.int32)
        test_df[cat_cols] = test_df[cat_cols].astype(np.int32)

    # Gaussianize numeric (excluding bools); keep NaNs
    for c in gauss_cols:
        s_tr = train[c]
        s_te = test_df[c]
        if s_tr.notna().sum() <= 1:  # nothing to fit
            continue
        qt = QuantileTransformer(
            n_quantiles=N_QUANTILES,
            output_distribution="normal",
            subsample=SUBSAMPLE,
            random_state=SEED,
            copy=True,
        )
        mask_tr = s_tr.notna()
        qt.fit(s_tr[mask_tr].to_numpy().reshape(-1,1))
        out_tr = s_tr.copy()
        out_tr.loc[mask_tr] = qt.transform(s_tr[mask_tr].to_numpy().reshape(-1,1)).ravel()
        train[c] = out_tr.astype(np.float32)

        mask_te = s_te.notna()
        out_te = s_te.copy()
        out_te.loc[mask_te] = qt.transform(s_te[mask_te].to_numpy().reshape(-1,1)).ravel()
        test_df[c] = out_te.astype(np.float32)

    gc.collect()

    # Save
    train.to_parquet(NEW_TRAIN, index=False)
    test_df.to_parquet(NEW_TEST, index=False)
    print(f"[OK] saved:\n  - {NEW_TRAIN}\n  - {NEW_TEST}")

# ========= Entry =========
if __name__ == "__main__":
    # 1) Enrichment (skip if already exists and you want to reuse)
    need_enrich = not (os.path.exists(TRAIN_ENR) and os.path.exists(TEST_ENR))
    if need_enrich:
        print("=== RUN ENRICHMENT (PASS1~3) ===")
        run_enrichment()
    else:
        print("=== SKIP ENRICHMENT: found existing enriched files ===")

    # 2) Postprocess to model-ready
    print("=== RUN POSTPROCESS → new_data ===")
    run_postprocess()
