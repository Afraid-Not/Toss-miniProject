# == STREAMED Arrow -> NumPy XGBoost training & prediction (memory-safe) ==
import os, gc, json, hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import xgboost as xgb

# ===== Paths / I/O =====
BASE = "./Project/Toss"
TRAIN_PARQUET = f"{BASE}/train.parquet"     # 파일 or 디렉터리(파티션) 모두 OK
TEST_PARQUET  = f"{BASE}/test.parquet"
OUT_DIR = f"{BASE}/_out/ver_stream"
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_COL = "clicked"
ID_COL    = "ID"           # 없으면 train=전체, valid 없음(early stopping 비활성)
BATCH_ROWS = 50_000       # Arrow 스캐너 배치 크기 (RAM 여유면 200k~500k 추천)
RATIO_TRAIN = 0.8          # 80% train / 20% valid (ID 해시 기준)
NUM_BOOST_ROUND = 1000
EARLY_STOP = 50

# ===== Helper: scanner 호환 =====
def make_scanner(dataset, columns=None, filter=None, batch_size=BATCH_ROWS):
    try:
        return dataset.scanner(columns=columns, filter=filter, batch_size=batch_size)
    except Exception:
        return ds.Scanner.from_dataset(dataset, columns=columns, filter=filter, batch_size=batch_size)

# ===== Load Datasets =====
dset_train = ds.dataset(TRAIN_PARQUET, format="parquet")
dset_test  = ds.dataset(TEST_PARQUET,  format="parquet")

train_cols = set(dset_train.schema.names)
test_cols  = set(dset_test.schema.names)

# 공통 feature만 사용 (test에 없는 컬럼 제외)
features = sorted((train_cols & test_cols) - {LABEL_COL, ID_COL})
print(f"[INFO] #features={len(features)}")

# ===== Arrow -> NumPy 변환(복사 최소) =====
# - float -> float32
# - int   -> int32 (NULL은 -1 채움)
# - bool  -> int8
# - string/binary -> dictionary_encode -> codes(int32, NULL=-1)
def _col_to_numpy_intcoded(arr: pa.ChunkedArray) -> np.ndarray:
    t = arr.type
    if pa.types.is_floating(t):
        arr = pc.cast(pc.fill_null(arr, 0.0), pa.float32())
        return arr.to_numpy(zero_copy_only=False)
    elif pa.types.is_integer(t):
        arr = pc.cast(pc.fill_null(arr, -1), pa.int32())
        return arr.to_numpy(zero_copy_only=False)
    elif pa.types.is_boolean(t):
        arr = pc.cast(pc.fill_null(arr, False), pa.int8())
        return arr.to_numpy(zero_copy_only=False)
    elif pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_binary(t):
        darr = pc.dictionary_encode(arr)  # DictionaryArray
        codes = pc.cast(pc.fill_null(darr.indices, -1), pa.int32())
        return codes.to_numpy(zero_copy_only=False)
    else:
        # 가능한 한 float32로 강제
        arr = pc.cast(pc.fill_null(arr, 0.0), pa.float32())
        return arr.to_numpy(zero_copy_only=False)

def table_to_numpy(table: pa.Table, cols: list[str]) -> np.ndarray:
    # 각 열을 NumPy로 변환 후 column-stack (배치 단위라 메모리 안전)
    col_arrays = [_col_to_numpy_intcoded(table[c]) for c in cols]
    X = np.column_stack(col_arrays) if len(col_arrays) > 1 else col_arrays[0][:, None]
    return X

# ===== DataIter (Arrow 스트리밍) =====
class ArrowIter(xgb.core.DataIter):
    def __init__(self, dset, mode: str, features: list[str], label_col: str, id_col: str,
                 ratio_train: float = 0.8, batch_rows: int = BATCH_ROWS):
        """mode: 'train' or 'valid'"""
        super().__init__()
        assert mode in ("train", "valid")
        self.dset = dset
        self.mode = mode
        self.features = features
        self.label = label_col
        self.id = id_col
        self.threshold = int(1000 * ratio_train)
        cols = features + ([label_col] if label_col in dset.schema.names else []) + \
               ([id_col] if id_col in dset.schema.names else [])
        self.scanner = make_scanner(dset, columns=cols, batch_size=batch_rows)
        self._it = iter(self.scanner.to_batches())

    def reset(self):
        cols = self.features + ([self.label] if self.label in self.dset.schema.names else []) + \
               ([self.id] if self.id in self.dset.schema.names else [])
        self.scanner = make_scanner(self.dset, columns=cols, batch_size=BATCH_ROWS)
        self._it = iter(self.scanner.to_batches())

    def next(self, input_data):
        while True:
            try:
                rb = next(self._it)
            except StopIteration:
                return 0

            tbl = pa.Table.from_batches([rb])

            # ID 기반 결정적 해시 split (ID 없으면 전체 train, valid는 skip)
            if self.id in tbl.column_names:
                h64 = pc.hash(tbl[self.id])      # uint64
                mod = pc.mod(h64, 1000)         # 0..999
                if self.mode == "train":
                    mask = pc.less(mod, self.threshold)
                else:
                    mask = pc.greater_equal(mod, self.threshold)
                if tbl.num_rows != 0:
                    tbl = tbl.filter(mask)
            else:
                if self.mode == "valid":
                    # valid 불가 → 다음 배치로
                    del rb, tbl
                    continue

            if tbl.num_rows == 0:
                del rb, tbl
                continue

            # 레이블
            if self.label in tbl.column_names:
                y_arr = pc.cast(pc.fill_null(tbl[self.label], 0), pa.int8())
                y = y_arr.to_numpy(zero_copy_only=False)
            else:
                raise ValueError(f"Label column '{self.label}' not found in training data.")

            # 피처 행렬
            X = table_to_numpy(tbl, self.features)

            input_data(data=X, label=y)

            # 메모리 정리
            del rb, tbl, X, y
            gc.collect()
            return 1

# ===== XGBoost params =====
params = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "tree_method": "gpu_hist",   # GPU 없으면 "hist"로 변경
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_bin": 128,
    "seed": 73,
    "verbosity": 1,
}

print("[train build]")
train_iter = ArrowIter(dset_train, "train", features, LABEL_COL, ID_COL, RATIO_TRAIN, BATCH_ROWS)
valid_iter = ArrowIter(dset_train, "valid", features, LABEL_COL, ID_COL, RATIO_TRAIN, BATCH_ROWS)

# QuantileDMatrix: 메모리/대용량 친화
dtrain = xgb.QuantileDMatrix(train_iter, enable_categorical=True)
# ref 공유로 메모리 사용량↓
dvalid = xgb.QuantileDMatrix(valid_iter, ref=dtrain, enable_categorical=True)

evals = [(dtrain, "train")]
if ID_COL in dset_train.schema.names:
    evals.append((dvalid, "valid"))

bst = xgb.train(
    params, dtrain, num_boost_round=NUM_BOOST_ROUND, evals=evals,
    early_stopping_rounds=(EARLY_STOP if len(evals) > 1 else None)
)

model_path = f"{OUT_DIR}/xgb_stream.json"
bst.save_model(model_path)
print(f"[DONE] model saved -> {model_path}")

# ===== Prediction (test: 스트리밍 배치 예측, 메모리 안전) =====
def stream_predict_to_csv(model: xgb.Booster, dset: ds.Dataset, features: list[str], id_col: str, out_csv: str):
    cols = features + ([id_col] if id_col in dset.schema.names else [])
    scanner = make_scanner(dset, columns=cols, batch_size=BATCH_ROWS)
    import csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = [id_col, "prediction"] if id_col in dset.schema.names else ["row_id", "prediction"]
        writer.writerow(header)

        row_id_base = 0
        for rb in scanner.to_batches():
            tbl = pa.Table.from_batches([rb])
            if tbl.num_rows == 0:
                del rb, tbl
                continue

            X = table_to_numpy(tbl, features)
            preds = model.inplace_predict(X)  # 확률

            if id_col in tbl.column_names:
                ids = tbl[id_col].to_numpy(zero_copy_only=False)
            else:
                # ID가 없으면 누적 row_id 부여
                ids = np.arange(row_id_base, row_id_base + len(preds), dtype=np.int64)
                row_id_base += len(preds)

            for i in range(len(preds)):
                writer.writerow([ids[i], float(preds[i])])

            del rb, tbl, X, preds, ids
            gc.collect()

pred_path = f"{OUT_DIR}/prediction.csv"
print("[predict] streaming...")
stream_predict_to_csv(bst, dset_test, features, ID_COL, pred_path)
print(f"[DONE] predictions -> {pred_path}")
