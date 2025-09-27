import pyarrow as pa
import pyarrow.dataset as ds
import gc

# ===== 기존 코드 =====
dset = ds.dataset("./Project/Toss/train.parquet", format="parquet")
dset_test = ds.dataset("./Project/Toss/test.parquet", format="parquet")
# NOTE: 학습/검증을 위해서는 label + feature들이 필요합니다.
#       아래에서 'scanner'는 교체하고, 루프는 그대로 둡니다.

print("[train load]")

# ====== 여기부터 이어붙임 ======
# 필요 패키지
import numpy as np
import pandas as pd
import hashlib
import xgboost as xgb

LABEL_COL = "clicked"
ID_COL = "ID"              # 없으면 자동으로 fallback
BATCH_SIZE = 100_000
RATIO_TRAIN = 0.8
TRAIN_BATCH=2048
NUM_BOOST_ROUND = 1000

# pyarrow 버전 호환 스캐너
def make_scanner(dset, columns, batch_size, filter=None):
    try:
        return dset.scanner(columns=columns, batch_size=batch_size, filter=filter)
    except Exception:
        return ds.Scanner.from_dataset(dset, columns=columns, batch_size=batch_size, filter=filter)

# 피처 자동 추출 (label/ID 제외)
schema_names = [f.name for f in dset.schema]
features = [c for c in schema_names if c not in {LABEL_COL, ID_COL}]
print(f"[INFO] #features={len(features)}")

# 간단 전처리(필요 최소): float64 -> float32 (메모리↓)
def preprocess_batch_inplace(df: pd.DataFrame):
    f64 = [c for c in df.columns if c != LABEL_COL and str(df[c].dtype) == "float64"]
    if f64:
        df[f64] = df[f64].astype("float32")
    return df

# 결정적 해시(항상 같은 split) : blake2b 32bit
def hash_mod_u32(values: np.ndarray, m: int = 1000, salt: str = "v1") -> np.ndarray:
    def h(s: str) -> int:
        x = (salt + s).encode("utf-8")
        return int(hashlib.blake2b(x, digest_size=4).hexdigest(), 16) % m
    return np.fromiter((h(str(v)) for v in values), dtype=np.uint32, count=len(values))

from pandas.util import hash_pandas_object  # 벡터화 해시 (문자열로 바꾸지 마!)

class ValidIter(xgb.core.DataIter):
    def __init__(self, dset, features, label_col, id_col,
                 ratio_train=0.8, batch_size=200_000, preprocess_fn=None):
        super().__init__()
        self.dset = dset
        self.features = list(features)
        self.label = label_col
        self.id = id_col
        self.batch_size = batch_size
        self.threshold = int(1000 * ratio_train)   # 0..999 중 800 미만 = train, 800 이상 = valid
        self.preprocess_fn = preprocess_fn or (lambda df: df)
        self._reset()

    def _reset(self):
        cols = self.features + [self.label] + ([self.id] if self.id in self.dset.schema.names else [])
        try:
            self.scanner = self.dset.scanner(columns=cols, batch_size=self.batch_size)
        except Exception:
            self.scanner = ds.Scanner.from_dataset(self.dset, columns=cols, batch_size=self.batch_size)
        self._it = iter(self.scanner.to_batches())

    def reset(self):
        self._reset()

    def next(self, input_data):
        while True:
            try:
                rb = next(self._it)
            except StopIteration:
                return 0

            tbl = pa.Table.from_batches([rb])
            df  = tbl.to_pandas()

            # 해시 스플릿 (문자열 변환 금지! → 메모리 폭증 방지)
            if self.id in df.columns:
                h = (hash_pandas_object(df[self.id], index=False).values % 1000).astype(np.uint16)
                mask = (h >= self.threshold)  # valid만 공급
            else:
                # ID 없으면 valid 구성 불가 → 빈 배치로 건너뜀
                del rb, tbl, df
                continue

            if not mask.any():
                del rb, tbl, df
                continue

            y = df.loc[mask, self.label].astype("int8").values
            X = df.loc[mask, self.features].copy()
            self.preprocess_fn(X)

            input_data(data=X, label=y)

            del rb, tbl, df, X, y
            gc.collect()
            return 1

# (2) 학습용 DataIter (ID 해시로 train만 흘려보냄)
class ArrowTrainIter(xgb.core.DataIter):
    def __init__(self, dset, features, label_col=LABEL_COL, id_col=ID_COL,
                 ratio_train=RATIO_TRAIN, batch_size=BATCH_SIZE):
        super().__init__()
        self.dset = dset
        self.feats = list(features)
        self.label = label_col
        self.id = id_col
        self.ratio_train = ratio_train
        self.batch_size = batch_size
        self.m = 1000
        self.th = int(self.m * self.ratio_train)
        self._reset_scanner()

    def _reset_scanner(self):
        cols = self.feats + [self.label] + ([self.id] if self.id in schema_names else [])
        self.scanner = make_scanner(self.dset, columns=cols, batch_size=self.batch_size)
        self._it = iter(self.scanner.to_batches())

    def reset(self):
        self._reset_scanner()

    def next(self, input_data):
        while True:
            try:
                rb = next(self._it)
            except StopIteration:
                return 0

            tbl = pa.Table.from_batches([rb])
            df  = tbl.to_pandas()

            # train 마스크: ID 해시 < 임계
            if self.id in df.columns:
                h = hash_mod_u32(df[self.id].astype(str).values, m=self.m, salt="v1")
                mask = (h < self.th)
            else:
                mask = np.ones(len(df), dtype=bool)

            if not mask.any():
                del rb, tbl, df
                continue

            y = df.loc[mask, self.label].astype("int8").values
            X = df.loc[mask, self.feats].copy()
            X = preprocess_batch_inplace(X)

            input_data(data=X, label=y)

            del rb, tbl, df, X, y
            gc.collect()
            return 1

# (3) XGBoost 학습 (QuantileDMatrix: 대용량/스트리밍 최적화)
params = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss","auc"],
    "tree_method": "gpu_hist",  # GPU 없으면 "hist"
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 73,
    "verbosity": 1,
    "max_bin": 128,
}

print("[train build]")

train_iter = ArrowTrainIter(dset, features, LABEL_COL, ID_COL, RATIO_TRAIN, TRAIN_BATCH)
dtrain = xgb.QuantileDMatrix(train_iter, enable_categorical=True)

# valid (NEW: 스트리밍, ref로 train cut 공유)
valid_iter = ValidIter(dset, features, LABEL_COL, ID_COL, RATIO_TRAIN, TRAIN_BATCH, preprocess_fn=preprocess_batch_inplace)
dvalid = xgb.QuantileDMatrix(valid_iter, ref=dtrain, enable_categorical=True)

evals = [(dtrain, "train"), (dvalid, "valid")]
bst = xgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND, evals=evals,
                early_stopping_rounds=50)


print("[DONE] model saved -> xgb_stream_hashsplit.json")


