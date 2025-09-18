# -*- coding: utf-8 -*-
import os, gc, random, math, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# =========================
# Config
# =========================
SEED = 42
random.seed(SEED); np.random.seed(SEED)

TRAIN_PATH = "./Toss/train.parquet"
TEST_PATH  = "./Toss/test.parquet"
SUBMIT_TPL = "./Toss/sample_submission.csv"
SUBMIT_OUT = "./Toss/new_data/submit_xgb_seqagg.csv"

# 캐시 디렉토리
CACHE_DIR = "./Toss/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
TRAIN_SEQ_F = f"{CACHE_DIR}/train_seq.parquet"
TEST_SEQ_F  = f"{CACHE_DIR}/test_seq.parquet"
TRAIN_ENC_F = f"{CACHE_DIR}/train_encoded.feather"
TEST_ENC_F  = f"{CACHE_DIR}/test_encoded.feather"
TE_MAPS_F   = f"{CACHE_DIR}/te_maps.pkl"

target_col = "clicked"
seq_col    = "seq"

# 후보 범주형 (데이터에 있으면 사용)
cand_cats = ["gender","age_group","inventory_id","day_of_week","hour","l_feat_14"]

# 타깃 인코딩 스무딩 강도 (m 클수록 전체평균 비중↑)
TE_SMOOTH_M = 20.0

# XGBoost 파라미터 (2.x 권장 방식: hist + device=cuda)
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",      # 리더보드 기준에 따라 "logloss"/"aucpr"로 교체 가능
    "tree_method": "hist",     # GPU 학습
    "device": "cuda",          # GPU 사용
    # 메모리/안정성 친화 세팅
    "max_depth": 6,
    "max_bin": 128,
    "min_child_weight": 32,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.5,
    "sampling_method": "gradient_based",
    "random_state": SEED,
}
NUM_BOOST_ROUND = 10000
EARLY_STOP_ROUNDS = 200
VALID_SIZE = 0.2

# =========================
# Utils
# =========================
def free_memory():
    gc.collect()

# seq 문자열 -> 통계 피처
def seq_to_stats(s: str, recents=(3,5,10,20)):
    if not isinstance(s, str) or not s:
        arr = np.array([0.0], dtype=np.float32)
    else:
        arr = np.fromstring(s, sep=",", dtype=np.float32)
        if arr.size == 0:
            arr = np.array([0.0], dtype=np.float32)
    L = arr.size
    feat = {
        "seq_len": L,
        "seq_mean": float(arr.mean()),
        "seq_std": float(arr.std()),
        "seq_min": float(arr.min()),
        "seq_max": float(arr.max()),
        "seq_last": float(arr[-1]),
        "seq_sum": float(arr.sum()),
        "seq_nonzero": int(np.count_nonzero(arr)),
    }
    if L >= 2:
        diff = np.diff(arr)
        feat.update({
            "seq_diff_mean": float(diff.mean()),
            "seq_diff_std": float(diff.std()),
            "seq_last_diff": float(arr[-1] - arr[-2]),
        })
        x = np.arange(L, dtype=np.float32)
        cov = float(((x - x.mean()) * (arr - arr.mean())).sum())
        var = float(((x - x.mean()) ** 2).sum()) + 1e-9
        feat["seq_slope"] = cov / var
    else:
        feat.update({"seq_diff_mean":0.0,"seq_diff_std":0.0,"seq_last_diff":0.0,"seq_slope":0.0})
    for k in recents:
        seg = arr[-k:] if L >= k else arr
        feat[f"seq_last{k}_mean"] = float(seg.mean())
        feat[f"seq_last{k}_std"]  = float(seg.std() if seg.size>1 else 0.0)
    return feat

def build_seq_features(df, seq_col):
    stats = [seq_to_stats(s) for s in tqdm(df[seq_col].astype(str).fillna(""), desc="SeqAgg")]
    return pd.DataFrame(stats, index=df.index)

# 빈도 인코딩
def add_freq_encoding(tr, te, cols):
    for c in cols:
        freq = tr[c].astype(str).fillna("UNK").value_counts()
        tr[f"{c}_freq"] = tr[c].astype(str).fillna("UNK").map(freq).fillna(0).astype(np.float32)
        te[f"{c}_freq"] = te[c].astype(str).fillna("UNK").map(freq).fillna(0).astype(np.float32)

# 안전한 OOF 스무딩 타깃 인코딩(인덱스 의존 X)
def add_kfold_target_encoding_with_smoothing(tr, te, cols, y, n_splits=5, m=20.0):
    """
    - fold-train으로 계산한 smoothed mean을 fold-valid에만 적용(OOF)
    - test에는 전체-train 맵 적용
    - groupby('cat') 값 기반으로만 계산하여 인덱스 에러 방지
    """
    tr = tr.copy()
    te = te.copy()
    y = y.loc[tr.index]  # 정렬/인덱스 어긋남 방지

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    global_mean = float(y.mean())
    te_maps = {}

    for c in cols:
        tr[c] = tr[c].astype(str).fillna("UNK")
        te[c] = te[c].astype(str).fillna("UNK")
        te_col = f"{c}_te"
        tr[te_col] = np.nan

        # 전체 train 기준 맵(나중에 test에 사용)
        df_all = pd.DataFrame({"cat": tr[c].values, "y": y.values})
        grp_all = df_all.groupby("cat")["y"].agg(["sum", "count"]).reset_index()
        grp_all["sm"] = (grp_all["sum"] + global_mean * m) / (grp_all["count"] + m)
        map_all = dict(zip(grp_all["cat"], grp_all["sm"]))

        # OOF: 폴드별로 fold-train 맵 계산 → fold-valid에만 적용
        for tr_idx, val_idx in kf.split(tr):
            df_fold = pd.DataFrame({
                "cat": tr.iloc[tr_idx][c].values,
                "y":   y.iloc[tr_idx].values
            })
            grp = df_fold.groupby("cat")["y"].agg(["sum", "count"]).reset_index()
            grp["sm"] = (grp["sum"] + global_mean * m) / (grp["count"] + m)
            map_fold = dict(zip(grp["cat"], grp["sm"]))

            cats_val = tr.iloc[val_idx][c].map(map_fold).fillna(global_mean).astype(np.float32).values
            tr.loc[tr.index[val_idx], te_col] = cats_val

        tr[te_col] = tr[te_col].fillna(global_mean).astype(np.float32)
        te[te_col] = te[c].map(map_all).fillna(global_mean).astype(np.float32)

        te_maps[c] = {"map": map_all, "global": global_mean, "m": m}

    return tr, te, te_maps

# =========================
# Load
# =========================
train = pd.read_parquet(TRAIN_PATH, engine="pyarrow")
test  = pd.read_parquet(TEST_PATH,  engine="pyarrow")

# 실제 존재하는 범주형
cat_cols = [c for c in cand_cats if c in train.columns]

# 공통 제외
EXCLUDE = set([target_col, seq_col, "ID"])
base_feature_cols = [c for c in train.columns if c not in EXCLUDE]

# =========================
# SeqAgg 캐시
# =========================
if os.path.exists(TRAIN_SEQ_F) and os.path.exists(TEST_SEQ_F):
    train_seq = pd.read_parquet(TRAIN_SEQ_F)
    test_seq  = pd.read_parquet(TEST_SEQ_F)
else:
    train_seq = build_seq_features(train, seq_col)
    test_seq  = build_seq_features(test,  seq_col)
    train_seq.to_parquet(TRAIN_SEQ_F, index=False)
    test_seq.to_parquet(TEST_SEQ_F,  index=False)

# =========================
# 범주형 인코딩까지 포함한 최종 테이블 캐시
# =========================
if os.path.exists(TRAIN_ENC_F) and os.path.exists(TEST_ENC_F) and os.path.exists(TE_MAPS_F):
    train_ = pd.read_feather(TRAIN_ENC_F)
    test_  = pd.read_feather(TEST_ENC_F)
    with open(TE_MAPS_F, "rb") as f:
        te_maps = pickle.load(f)
else:
    # 합치기
    test_base_cols = [c for c in base_feature_cols if c in test.columns]
    train_ = pd.concat([train[base_feature_cols].reset_index(drop=True),
                        train_seq.reset_index(drop=True)], axis=1)
    test_  = pd.concat([test[test_base_cols].reset_index(drop=True),
                        test_seq.reset_index(drop=True)],  axis=1)

    # 빈도 인코딩
    add_freq_encoding(train_, test_, cat_cols)

    # 스무딩 타깃 인코딩(OOF) + test 매핑 저장
    y_all = train[target_col].astype(np.float32).reset_index(drop=True)
    train_, test_, te_maps = add_kfold_target_encoding_with_smoothing(
        train_, test_, cat_cols, y=y_all, n_splits=5, m=TE_SMOOTH_M
    )

    # 해시 정수화(원본 값 보존용, 선택)
    for c in cat_cols:
        train_[c] = train_[c].astype(str).apply(lambda s: abs(hash(s)) % (10**6)).astype(np.int32)
        test_[c]  = test_[c].astype(str).apply(lambda s: abs(hash(s)) % (10**6)).astype(np.int32)

    # 저장 (feather 빠름)
    train_.reset_index(drop=True).to_feather(TRAIN_ENC_F)
    test_.reset_index(drop=True).to_feather(TEST_ENC_F)
    with open(TE_MAPS_F, "wb") as f:
        pickle.dump(te_maps, f)

# =========================
# Train / Valid split
# =========================
y = train[target_col].astype(np.float32).reset_index(drop=True)

tr_idx, va_idx = train_test_split(
    np.arange(len(train_)), test_size=VALID_SIZE, random_state=SEED, stratify=y
)
X_tr = train_.iloc[tr_idx].reset_index(drop=True).astype(np.float32)
X_va = train_.iloc[va_idx].reset_index(drop=True).astype(np.float32)
y_tr = y.iloc[tr_idx].reset_index(drop=True)
y_va = y.iloc[va_idx].reset_index(drop=True)

test_ = test_.astype(np.float32)

# 불균형 가중
pos = float((y == 1).sum()); neg = float((y == 0).sum())
scale_pos_weight = max(1.0, neg / max(1.0, pos))
XGB_PARAMS["scale_pos_weight"] = scale_pos_weight

# =========================
# QuantileDMatrix (GPU 친화)
# =========================
dtrain = xgb.QuantileDMatrix(X_tr, label=y_tr, max_bin=XGB_PARAMS["max_bin"])
dvalid = xgb.QuantileDMatrix(X_va, label=y_va, max_bin=XGB_PARAMS["max_bin"])
dtest  = xgb.QuantileDMatrix(test_, max_bin=XGB_PARAMS["max_bin"])

# =========================
# Train
# =========================
watchlist = [(dtrain, "train"), (dvalid, "valid")]
model = xgb.train(
    XGB_PARAMS,
    dtrain,
    num_boost_round=NUM_BOOST_ROUND,
    evals=watchlist,
    early_stopping_rounds=EARLY_STOP_ROUNDS,
    verbose_eval=200
)

# 검증 점수
va_pred = model.predict(dvalid, iteration_range=(0, model.best_iteration+1))
print("Valid AUC:", roc_auc_score(y_va, va_pred))

# =========================
# Inference & Submit
# =========================
test_pred = model.predict(dtest, iteration_range=(0, model.best_iteration+1))
sub = pd.read_csv(SUBMIT_TPL)
sub["clicked"] = test_pred
sub.to_csv(SUBMIT_OUT, index=False)
print("Saved:", SUBMIT_OUT)
