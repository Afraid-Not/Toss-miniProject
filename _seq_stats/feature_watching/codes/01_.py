import os, gc, csv, json, math, glob, argparse, hashlib
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Config
# =========================
BASE = "/home/pc/Study/Project/Toss"
SPLIT_TRAIN_DIR = f"{BASE}/_processed/train_200k"    # apply_cleaning_plan.py 결과 권장(없어도 on-the-fly 처리)
SPLIT_TEST_DIR  = f"{BASE}/_processed/test_200k"
RAW_TRAIN_DIR   = f"{BASE}/_split/train_200k"        # on-the-fly 클린 시 사용
RAW_TEST_DIR    = f"{BASE}/_split/test_200k"

PLAN_CSV = f"{BASE}/_meta/cleaning_plan.csv"         # 업로드한 cleaning_plan.csv 대응
CAT_DIR  = f"{BASE}/_meta/cat_maps"                  # category json들 (없으면 index_in fallback)

OUT_DIR  = f"{BASE}/_out/ver_stream_ens"
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_COL = "clicked"
ID_COL    = "ID"
BATCH_ROWS = 200_000      # Arrow 스캐너 배치(또는 파일 단위). 50GB RAM이면 200k~500k OK
TRAIN_BATCH_SIZE = 4096   # Transformer 미니배치
EPOCHS = 2                # 예시: 2 epoch (필요시 늘리기)
LR = 1e-3
SEED = 73
RATIO_TRAIN = 0.8         # ID 해시 기반 split
EARLY_STOP_XGB = 50
NUM_BOOST_ROUND = 1000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================
# Utils: load plan / category maps
# =========================
def load_plan(plan_csv: str) -> Dict[str, dict]:
    plan = {}
    with open(plan_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            plan[row["column"]] = row
    return plan

def load_cat_values(plan: Dict[str, dict]) -> Dict[str, pa.Array]:
    cat_values = {}
    for col, row in plan.items():
        if row.get("src_type") == "string" and row.get("category_map_path"):
            path = row["category_map_path"]
            if not os.path.isabs(path):
                path = os.path.join(CAT_DIR, os.path.basename(path))
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    vals = json.load(f)["values"]
                cat_values[col] = pa.array(vals, type=pa.string())
    return cat_values

# =========================
# Apply cleaning plan to Arrow Table -> Arrow Table (column-wise)
# (pandas 없이 Arrow compute만)
# =========================
def apply_plan_table(tbl: pa.Table, plan: Dict[str, dict], cat_values: Dict[str, pa.Array]) -> pa.Table:
    cols_out, names = [], []
    for c in tbl.column_names:
        if c not in plan:
            cols_out.append(tbl[c]); names.append(c); continue

        p = plan[c]; src = tbl[c]
        t = p.get("src_type","")
        target = p.get("target_dtype","")

        if t == "numeric":
            arr = pc.cast(src, pa.float32()) if target == "float32" else src
            fv = p.get("fillna_value","")
            fv = float(fv) if fv != "" else 0.0
            arr = pc.fill_null(arr, fv)
            lo, hi = p.get("clip_low",""), p.get("clip_high","")
            if lo != "" or hi != "":
                lo_v = float(lo) if lo != "" else None
                hi_v = float(hi) if hi != "" else None
                arr = pc.clamp(arr, lo_v, hi_v)
            cols_out.append(arr); names.append(c)

        elif t == "bool":
            arr = pc.cast(pc.fill_null(src, False), pa.int8())
            cols_out.append(arr); names.append(c)

        elif t == "string":
            value_set = cat_values.get(c)
            if value_set is None:
                darr = pc.dictionary_encode(src)
                codes = pc.cast(pc.fill_null(darr.indices, -1), pa.int32())
            else:
                idx = pc.index_in(src, value_set)         # 0..K-1 or null
                codes = pc.cast(pc.fill_null(idx, -1), pa.int32())
            cols_out.append(codes); names.append(c)

        else:
            # 기타형은 float32로 강제
            arr = pc.cast(pc.fill_null(src, 0.0), pa.float32())
            cols_out.append(arr); names.append(c)

    return pa.Table.from_arrays(cols_out, names)

# =========================
# Feature type split from plan
# =========================
def split_feature_types(plan: Dict[str, dict], columns: List[str]) -> Tuple[List[str], List[str], List[str]]:
    num_cols, cat_cols, bool_cols = [], [], []
    for c in columns:
        if c not in plan:  # 계획에 없으면 numeric으로 간주
            num_cols.append(c); continue
        t = plan[c].get("src_type","")
        if t == "numeric":
            num_cols.append(c)
        elif t == "string":
            cat_cols.append(c)
        elif t == "bool":
            bool_cols.append(c)
        else:
            num_cols.append(c)
    return num_cols, cat_cols, bool_cols

# =========================
# Arrow Table -> NumPy matrices (float32, int32)
# =========================
def table_to_numpy_mats(tbl: pa.Table, num_cols: List[str], cat_cols: List[str], bool_cols: List[str]):
    mats = {}
    if num_cols:
        arrs = [pc.cast(pc.fill_null(tbl[c], 0.0), pa.float32()).to_numpy(zero_copy_only=False) for c in num_cols]
        mats["num"] = np.column_stack(arrs).astype(np.float32, copy=False)
    else:
        mats["num"] = None

    if cat_cols:
        arrs = [pc.cast(pc.fill_null(tbl[c], -1), pa.int32()).to_numpy(zero_copy_only=False) for c in cat_cols]
        mats["cat"] = np.column_stack(arrs).astype(np.int32, copy=False)
    else:
        mats["cat"] = None

    if bool_cols:
        arrs = [pc.cast(pc.fill_null(tbl[c], False), pa.int8()).to_numpy(zero_copy_only=False) for c in bool_cols]
        mats["bool"] = np.column_stack(arrs).astype(np.int8, copy=False)
    else:
        mats["bool"] = None

    # concat bool to num as 0/1 float
    if mats["bool"] is not None:
        b = mats["bool"].astype(np.float32)
        if mats["num"] is not None:
            mats["num"] = np.column_stack([mats["num"], b])
        else:
            mats["num"] = b
        mats["bool"] = None

    return mats

# =========================
# ID 해시 split (결정적)
# =========================
def id_hash_split(tbl: pa.Table, id_col: str, ratio_train: float):
    if id_col not in tbl.column_names:
        return np.ones(tbl.num_rows, dtype=bool), np.zeros(tbl.num_rows, dtype=bool)
    h64 = pc.hash(tbl[id_col])          # uint64
    mod = pc.mod(h64, 1000)             # 0..999
    th = int(1000 * ratio_train)
    train_mask = pc.less(mod, th).to_numpy(zero_copy_only=False)
    valid_mask = pc.greater_equal(mod, th).to_numpy(zero_copy_only=False)
    return train_mask, valid_mask

# =========================
# XGBoost DataIter (파일 단위로 스트리밍)
# =========================
class FilesArrowIter(xgb.core.DataIter):
    def __init__(self, files: List[str], features: List[str], label_col: str, id_col: str, plan, cat_values,
                 mode: str, ratio_train: float = 0.8, batch_rows: int = BATCH_ROWS):
        super().__init__()
        assert mode in ("train","valid")
        self.files = files
        self.features = features
        self.label = label_col
        self.id = id_col
        self.plan = plan
        self.cat_values = cat_values
        self.mode = mode
        self.ratio_train = ratio_train
        self.batch_rows = batch_rows
        self._file_idx = 0
        self._it = None

        # 타입 분해
        self.num_cols, self.cat_cols, self.bool_cols = split_feature_types(plan, features)

    def reset(self):
        self._file_idx = 0
        self._it = None

    def _open_next_file(self):
        if self._file_idx >= len(self.files):
            return False
        path = self.files[self._file_idx]
        self._file_idx += 1
        dset = pq.ParquetFile(path)
        # row group 단위로 배치화
        batches = []
        for i in range(dset.metadata.num_row_groups):
            rb = dset.read_row_group(i).to_batches()[0]
            batches.append(rb)
        self._it = iter(batches)
        return True

    def next(self, input_data):
        while True:
            if self._it is None:
                if not self._open_next_file():
                    return 0
            try:
                rb = next(self._it)
            except StopIteration:
                self._it = None
                continue

            tbl = pa.Table.from_batches([rb])
            # (필요시) 플랜 적용 (raw split를 쓰는 경우)
            if any(c not in tbl.column_names for c in self.features):
                # 파일에 feature가 없으면 스킵
                del rb, tbl; gc.collect(); continue
            # label은 반드시 있어야 함
            if self.label not in tbl.column_names:
                del rb, tbl; gc.collect(); continue

            # on-the-fly cleaning (processed 디렉터리를 쓰면 대부분 이미 숫자/코드 상태)
            tbl = apply_plan_table(tbl, self.plan, self.cat_values)

            # split
            tr_mask, va_mask = id_hash_split(tbl, ID_COL, self.ratio_train)
            mask = tr_mask if self.mode == "train" else va_mask
            if not mask.any():
                del rb, tbl; gc.collect(); continue

            tbl = tbl.filter(pa.array(mask))

            y = pc.cast(pc.fill_null(tbl[self.label], 0), pa.int8()).to_numpy(zero_copy_only=False)
            feat_tbl = tbl.select(self.features)
            mats = table_to_numpy_mats(feat_tbl, self.num_cols, self.cat_cols, self.bool_cols)
            # XGBoost에는 하나의 2D 행렬만 넣을 것이므로 num(+bool)과 cat를 합친다
            parts = []
            if mats["num"] is not None: parts.append(mats["num"])
            if mats["cat"] is not None: parts.append(mats["cat"].astype(np.float32))
            if not parts:
                del rb, tbl; gc.collect(); continue
            X = np.concatenate(parts, axis=1)

            input_data(data=X, label=y)

            del rb, tbl, X, y, mats, feat_tbl
            gc.collect()
            return 1

# =========================
# Minimal Tabular Transformer
# - categorical: embedding
# - numeric: linear projection
# - token-wise TransformerEncoder + mean pool
# =========================
class TabTransformer(nn.Module):
    def __init__(self, num_dim:int, num_cat:int, cat_cardinals:List[int],
                 d_model=64, nhead=4, nlayers=2, dim_ff=128, pdrop=0.1):
        super().__init__()
        self.num_dim = num_dim
        self.num_cat = num_cat

        self.num_proj = nn.Linear(num_dim, d_model) if num_dim>0 else None

        self.cat_embs = nn.ModuleList()
        for card in cat_cardinals:
            # -1(미지정)을 0으로 쉬프트하기 위해 +1
            self.cat_embs.append(nn.Embedding(card+1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_ff, dropout=pdrop, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, num_x:torch.Tensor|None, cat_x:torch.Tensor|None):
        tokens = []
        if self.num_proj is not None and num_x is not None:
            tokens.append(self.num_proj(num_x).unsqueeze(1))  # [B,1,D]
        if cat_x is not None:
            # cat_x: [B, C] ints (-1 → 0으로 이동)
            for i, emb in enumerate(self.cat_embs):
                # -1 -> 0, 나머지 +1
                idx = torch.clamp(cat_x[:, i] + 1, min=0)
                tokens.append(emb(idx).unsqueeze(1))          # [B,1,D]
        x = torch.cat(tokens, dim=1)  # [B, T, D]
        h = self.encoder(x)           # [B, T, D]
        pooled = h.mean(dim=1)        # mean pool
        logit = self.head(pooled).squeeze(1)
        return torch.sigmoid(logit)

# =========================
# Torch Dataset (한 파트 메모리 내)
# =========================
class PartDataset(Dataset):
    def __init__(self, mats:dict, y:np.ndarray):
        self.num = mats["num"]
        self.cat = mats["cat"]
        self.y = y.astype(np.float32)
        self.N = len(self.y)
    def __len__(self): return self.N
    def __getitem__(self, i):
        num = torch.from_numpy(self.num[i]) if self.num is not None else None
        cat = torch.from_numpy(self.cat[i]) if self.cat is not None else None
        y = torch.tensor(self.y[i])
        return num, cat, y

# =========================
# Helper: list part files
# =========================
def list_parts(dir_path: str) -> List[str]:
    return sorted(glob.glob(os.path.join(dir_path, "part-*.parquet")))

# =========================
# Train XGBoost (stream from files)
# =========================
def train_xgb(files, features, plan, cat_values):
    params = {
        "objective":"binary:logistic",
        "eval_metric":["logloss","auc"],
        "tree_method":"gpu_hist" if torch.cuda.is_available() else "hist",
        "max_depth":8,
        "learning_rate":0.05,
        "subsample":0.8,
        "colsample_bytree":0.8,
        "max_bin":128,
        "seed":SEED,
        "verbosity":1,
    }
    train_iter = FilesArrowIter(files, features, LABEL_COL, ID_COL, plan, cat_values, mode="train", ratio_train=RATIO_TRAIN)
    valid_iter = FilesArrowIter(files, features, LABEL_COL, ID_COL, plan, cat_values, mode="valid", ratio_train=RATIO_TRAIN)

    dtrain = xgb.QuantileDMatrix(train_iter, enable_categorical=True)
    dvalid = xgb.QuantileDMatrix(valid_iter, ref=dtrain, enable_categorical=True)

    evals = [(dtrain,"train"), (dvalid,"valid")]
    bst = xgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND, evals=evals,
                    early_stopping_rounds=EARLY_STOP_XGB)
    return bst

# =========================
# Train Transformer (iterate parts, mini-batches)
# =========================
def train_transformer(files, features, plan, cat_values,
                      epochs=EPOCHS, batch_size=TRAIN_BATCH_SIZE):
    # 타입 분해 및 카디널리티 추정
    num_cols, cat_cols, bool_cols = split_feature_types(plan, features)

    # cat cardinalities: plan의 cat_maps가 있으면 길이, 없으면 대략 65535로 가정(임시)
    cat_cards = []
    for c in cat_cols:
        arr = cat_values.get(c, None)
        if arr is not None:
            cat_cards.append(len(arr))
        else:
            # processed 디렉이면 값 범위를 스캔해서 잡아도 되지만 비용↑ → 일단 큰 값
            cat_cards.append(65535)

    model = TabTransformer(
        num_dim=len(num_cols)+(len(bool_cols) if bool_cols else 0),
        num_cat=len(cat_cols), cat_cardinals=cat_cards,
        d_model=64, nhead=4, nlayers=2, dim_ff=128, pdrop=0.1
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    bce = nn.BCELoss()

    best_val = float("inf")
    patience = 2
    wait = 0

    for ep in range(1, epochs+1):
        model.train()
        tr_loss_sum, tr_cnt = 0.0, 0

        for path in files:
            pf = pq.ParquetFile(path)
            for rg in range(pf.metadata.num_row_groups):
                tbl = pf.read_row_group(rg)

                # on-the-fly cleaning
                tbl = apply_plan_table(tbl, plan, cat_values)

                # split
                tr_mask, va_mask = id_hash_split(tbl, ID_COL, RATIO_TRAIN)
                if not tr_mask.any() and not va_mask.any():
                    del tbl; gc.collect(); continue

                # ---- train chunk ----
                if tr_mask.any():
                    sub = tbl.filter(pa.array(tr_mask))
                    y = pc.cast(pc.fill_null(sub[LABEL_COL], 0), pa.int8()).to_numpy(zero_copy_only=False).astype(np.float32)
                    mats = table_to_numpy_mats(sub.select(features), num_cols, cat_cols, bool_cols)
                    ds = PartDataset(mats, y); dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
                    for num_x, cat_x, yy in dl:
                        num_x = num_x.to(DEVICE) if num_x is not None else None
                        cat_x = cat_x.to(DEVICE) if cat_x is not None else None
                        yy = yy.to(DEVICE)
                        pred = model(num_x, cat_x)
                        loss = bce(pred, yy)
                        opt.zero_grad(); loss.backward(); opt.step()
                        tr_loss_sum += float(loss.item()); tr_cnt += 1
                    del sub, ds, dl, y, mats
                    gc.collect()

                # ---- free row group early ----
                del tbl
                gc.collect()

        # ---- validation sweep (lightweight) ----
        model.eval()
        val_loss_sum, val_cnt = 0.0, 0
        with torch.no_grad():
            # 일부 파트만 샘플링 평가
            for path in files[::max(1, len(files)//10)]:  # 약 10개 파트 간격
                pf = pq.ParquetFile(path)
                for rg in range(min(2, pf.metadata.num_row_groups)):  # 각 파일 앞쪽 2개 RG만
                    tbl = pf.read_row_group(rg)
                    tbl = apply_plan_table(tbl, plan, cat_values)
                    tr_mask, va_mask = id_hash_split(tbl, ID_COL, RATIO_TRAIN)
                    if not va_mask.any():
                        del tbl; gc.collect(); continue
                    sub = tbl.filter(pa.array(va_mask))
                    y = pc.cast(pc.fill_null(sub[LABEL_COL], 0), pa.int8()).to_numpy(zero_copy_only=False).astype(np.float32)
                    mats = table_to_numpy_mats(sub.select(features), num_cols, cat_cols, bool_cols)
                    ds = PartDataset(mats, y); dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
                    for num_x, cat_x, yy in dl:
                        num_x = num_x.to(DEVICE) if num_x is not None else None
                        cat_x = cat_x.to(DEVICE) if cat_x is not None else None
                        yy = yy.to(DEVICE)
                        pred = model(num_x, cat_x)
                        loss = bce(pred, yy)
                        val_loss_sum += float(loss.item()); val_cnt += 1
                    del sub, ds, dl, y, mats, tbl
                    gc.collect()

        tr_loss = tr_loss_sum / max(1, tr_cnt)
        val_loss = val_loss_sum / max(1, val_cnt)
        print(f"[Transformer] epoch {ep} train_loss={tr_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss + 1e-6 < best_val:
            best_val = val_loss; wait = 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "transformer.pt"))
        else:
            wait += 1
            if wait >= patience:
                print("[Transformer] early stop.")
                break

    # 로드 최적 가중치
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "transformer.pt"), map_location=DEVICE))
    model.eval()
    return model, (num_cols, cat_cols, bool_cols)

# =========================
# Collect validation preds for weight search
# =========================
def collect_valid_preds_xgb(bst, files, features, plan, cat_values, limit_rows=2_000_000):
    num_cols, cat_cols, bool_cols = split_feature_types(plan, features)
    preds, labels = [], []
    taken = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg)
            tbl = apply_plan_table(tbl, plan, cat_values)
            tr_mask, va_mask = id_hash_split(tbl, ID_COL, RATIO_TRAIN)
            if not va_mask.any(): del tbl; gc.collect(); continue
            sub = tbl.filter(pa.array(va_mask))
            y = pc.cast(pc.fill_null(sub[LABEL_COL], 0), pa.int8()).to_numpy(zero_copy_only=False)
            mats = table_to_numpy_mats(sub.select(features), num_cols, cat_cols, bool_cols)
            parts = []
            if mats["num"] is not None: parts.append(mats["num"])
            if mats["cat"] is not None: parts.append(mats["cat"].astype(np.float32))
            X = np.concatenate(parts, axis=1)
            p = bst.inplace_predict(X)
            preds.append(p.astype(np.float32)); labels.append(y.astype(np.float32))
            taken += len(y)
            del tbl, sub, y, mats, X, p
            gc.collect()
            if taken >= limit_rows:
                return np.concatenate(preds), np.concatenate(labels)
    if preds:
        return np.concatenate(preds), np.concatenate(labels)
    return np.array([]), np.array([])

def collect_valid_preds_tr(model, files, features, plan, cat_values, col_splits, limit_rows=2_000_000):
    num_cols, cat_cols, bool_cols = col_splits
    preds, labels = [], []
    taken = 0
    with torch.no_grad():
        for path in files:
            pf = pq.ParquetFile(path)
            for rg in range(pf.metadata.num_row_groups):
                tbl = pf.read_row_group(rg)
                tbl = apply_plan_table(tbl, plan, cat_values)
                tr_mask, va_mask = id_hash_split(tbl, ID_COL, RATIO_TRAIN)
                if not va_mask.any(): del tbl; gc.collect(); continue
                sub = tbl.filter(pa.array(va_mask))
                y = pc.cast(pc.fill_null(sub[LABEL_COL], 0), pa.int8()).to_numpy(zero_copy_only=False).astype(np.float32)
                mats = table_to_numpy_mats(sub.select(features), num_cols, cat_cols, bool_cols)
                ds = PartDataset(mats, y); dl = DataLoader(ds, batch_size=8192, shuffle=False, drop_last=False)
                buf = []
                for num_x, cat_x, _ in dl:
                    num_x = num_x.to(DEVICE) if num_x is not None else None
                    cat_x = cat_x.to(DEVICE) if cat_x is not None else None
                    p = model(num_x, cat_x).detach().cpu().numpy()
                    buf.append(p)
                preds.append(np.concatenate(buf))
                labels.append(y)
                taken += len(y)
                del tbl, sub, y, mats, ds, dl, buf
                gc.collect()
                if taken >= limit_rows:
                    return np.concatenate(preds), np.concatenate(labels)
    if preds:
        return np.concatenate(preds), np.concatenate(labels)
    return np.array([]), np.array([])

def search_weight(p_xgb, p_tr, y):
    # 단순 선형 앙상블 p = w*p_xgb + (1-w)*p_tr, logloss 최소화 w∈[0,1]
    ws = np.linspace(0.0, 1.0, 21)
    def logloss(p, y):
        eps = 1e-7; p = np.clip(p, eps, 1-eps)
        return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())
    best_w, best_ll = 0.5, 1e9
    for w in ws:
        ll = logloss(w*p_xgb + (1-w)*p_tr, y)
        if ll < best_ll: best_ll, best_w = ll, w
    return best_w, best_ll

# =========================
# Predict test streaming & write CSV
# =========================
def predict_and_write(bst, model, files, features, plan, cat_values, col_splits, out_csv, weight):
    num_cols, cat_cols, bool_cols = col_splits
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = [ID_COL, "prediction"] if ID_COL in pq.ParquetFile(files[0]).read_row_group(0).column_names else ["row_id","prediction"]
        w.writerow(header)
        row_id_base = 0

        for path in files:
            pf = pq.ParquetFile(path)
            for rg in range(pf.metadata.num_row_groups):
                tbl = pf.read_row_group(rg)
                tbl = apply_plan_table(tbl, plan, cat_values)

                # XGB feats
                mats = table_to_numpy_mats(tbl.select(features), num_cols, cat_cols, bool_cols)
                parts = []
                if mats["num"] is not None: parts.append(mats["num"])
                if mats["cat"] is not None: parts.append(mats["cat"].astype(np.float32))
                X = np.concatenate(parts, axis=1)
                pxgb = bst.inplace_predict(X).astype(np.float32)

                # Transformer feats
                ds = PartDataset(mats, np.zeros(len(X), dtype=np.float32))
                dl = DataLoader(ds, batch_size=8192, shuffle=False, drop_last=False)
                preds_tr = []
                with torch.no_grad():
                    for num_x, cat_x, _ in dl:
                        num_x = num_x.to(DEVICE) if num_x is not None else None
                        cat_x = cat_x.to(DEVICE) if cat_x is not None else None
                        preds_tr.append(model(num_x, cat_x).detach().cpu().numpy())
                ptr = np.concatenate(preds_tr).astype(np.float32)

                p = weight*pxgb + (1.0-weight)*ptr

                if ID_COL in tbl.column_names:
                    ids = tbl[ID_COL].to_numpy(zero_copy_only=False)
                else:
                    ids = np.arange(row_id_base, row_id_base+len(p)); row_id_base += len(p)

                for i in range(len(p)):
                    w.writerow([ids[i], float(p[i])])

                del tbl, mats, X, pxgb, ds, dl, preds_tr, ptr, p
                gc.collect()

# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plan = load_plan(PLAN_CSV)
    cat_values = load_cat_values(plan)

    # 어떤 디렉토리를 읽을지 결정 (processed가 있으면 그걸 우선)
    train_dir = SPLIT_TRAIN_DIR if os.path.isdir(SPLIT_TRAIN_DIR) and list_parts(SPLIT_TRAIN_DIR) else RAW_TRAIN_DIR
    test_dir  = SPLIT_TEST_DIR  if os.path.isdir(SPLIT_TEST_DIR)  and list_parts(SPLIT_TEST_DIR)  else RAW_TEST_DIR

    train_files = list_parts(train_dir)
    test_files  = list_parts(test_dir)
    assert train_files, f"No train parts in {train_dir}"
    assert test_files,  f"No test  parts in {test_dir}"

    # 공통 feature 집합 파악 (첫 파일 메타 기준)
    first = pq.ParquetFile(train_files[0]).read_row_group(0)
    cols = [c for c in first.column_names if c not in {LABEL_COL, ID_COL}]
    features = cols

    print(f"[INFO] #features={len(features)} | train_parts={len(train_files)} test_parts={len(test_files)}")

    # 1) Train XGBoost (stream)
    print("[XGB] training...")
    bst = train_xgb(train_files, features, plan, cat_values)
    bst.save_model(os.path.join(OUT_DIR, "xgb.json"))

    # 2) Train Transformer (iterate parts)
    print("[TR] training...")
    tr_model, col_splits = train_transformer(train_files, features, plan, cat_values)

    # 3) Weight search on validation
    print("[ENS] searching weight...")
    p_xgb, yv = collect_valid_preds_xgb(bst, train_files, features, plan, cat_values, limit_rows=1_000_000)
    p_tr , _  = collect_valid_preds_tr (tr_model, train_files, features, plan, cat_values, col_splits, limit_rows=1_000_000)
    if len(p_xgb) and len(p_tr):
        w_opt, ll = search_weight(p_xgb, p_tr, yv)
    else:
        w_opt, ll = 0.5, float("nan")
    print(f"[ENS] weight={w_opt:.2f}  val_logloss={ll:.5f}")
    with open(os.path.join(OUT_DIR, "ensemble.json"), "w") as f:
        json.dump({"weight": w_opt, "val_logloss": ll}, f)

    # 4) Predict test (stream) & write
    out_csv = os.path.join(OUT_DIR, "prediction_ensemble.csv")
    print("[PRED] writing predictions...")
    predict_and_write(bst, tr_model, test_files, features, plan, cat_values, col_splits, out_csv, w_opt)
    print(f"[DONE] -> {out_csv}")

if __name__ == "__main__":
    main()
