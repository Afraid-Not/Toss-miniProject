# -*- coding: utf-8 -*-
import os, warnings, json, gc, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, log_loss
SEED = 2
# ================== Config ==================
CFG = {
    "SEED": SEED,
    "N_SPLITS": 5,

    # LightGBM
    "LGB_PARAMS": dict(
        objective="binary",
        metric="binary_logloss",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=50,
        verbose=-1,
        # device="gpu", gpu_platform_id=0, gpu_device_id=0,  # 윈도우 GPU 빌드가 없으면 주석 유지
    ),
    "LGB_N_EST": 40000,
    "LGB_ES": 200,

    # CatBoost
    "CAT_PARAMS": dict(
        loss_function="Logloss",
        eval_metric="Logloss",
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=False,
        # task_type="GPU", devices="0"  # GPU 사용 가능하면 주석 해제
    ),
    "CAT_N_EST": 20000,
    "CAT_ES": 200,

    # Deep models (deepctr-torch)
    "DEEP_EPOCHS": 50,          # 처음은 작게, 성능 올리려면 늘려봐 (데이터가 큼)
    "DEEP_BATCH": 8192,
    "DEEP_LR": 1e-3,
    "DEEP_EMB": 16,
    "DEEP_WEIGHT_DECAY": 0.0,
    "DEEP_TRAIN_FRACTION": 1.0,  # <1.0 으로 줄여 빠르게 시범 학습 가능(예: 0.3)
}

rng_global = np.random.default_rng(CFG["SEED"])

# ================== Load ==================
train = pd.read_parquet("./Toss/new_data/new_train.parquet")
test  = pd.read_parquet("./Toss/new_data/new_test.parquet")

assert "clicked" in train.columns
y = train["clicked"].astype(np.int8).to_numpy()
feat_cols = [c for c in train.columns if c != "clicked"]

# 1) int-like 복구 규칙
MAX_CAT_CARD = 5000    # 카테고리 최대 고유값 허용치(필요하면 1000~10000에서 조정)
EPS = 1e-6

def is_intlike_float(s: pd.Series, eps=EPS):
    s = s.dropna()
    if s.empty:
        return False
    return (np.abs(s - np.round(s)) <= eps).mean() >= 0.999  # 99.9% 이상 정수 값

# 2) 범주형 후보 잡기: (a) 정수형 + 저카디널  or  (b) float인데 int-like + 저카디널
cat_cols = []
for c in feat_cols:
    s = train[c]
    nunq = int(s.nunique(dropna=True))
    if pd.api.types.is_integer_dtype(s):
        if nunq <= MAX_CAT_CARD:
            cat_cols.append(c)
    elif pd.api.types.is_float_dtype(s):
        if nunq <= MAX_CAT_CARD and is_intlike_float(s):
            # 정수로 복구 (결측은 -1)
            train[c] = np.round(train[c]).fillna(-1).astype(np.int64)
            test[c]  = np.round(test[c]).fillna(-1).astype(np.int64)
            cat_cols.append(c)

# 3) 수치형은 나머지
num_cols = sorted(set(feat_cols) - set(cat_cols))

print(f"[INFO] (recovered) cat={len(cat_cols)} num={len(num_cols)}")

# print(f"[INFO] features: total={len(feat_cols)} | cat={len(cat_cols)} | num={len(num_cols)}")
bool_cols = [c for c in feat_cols if pd.api.types.is_bool_dtype(train[c])]
# (안전) 불리언도 수치 특징으로 포함
num_cols = sorted(set(num_cols) | set(bool_cols))
cat_cols = sorted(cat_cols)

print(f"[INFO] features: total={len(feat_cols)} | cat={len(cat_cols)} | num={len(num_cols)}")

# ================== Metrics ==================
def weighted_logloss_5050(y_true, p, eps=1e-15):
    """클래스별 평균을 50:50로 가중한 LogLoss (Fold 내부 검증/OOF에 사용)."""
    p = np.clip(p, eps, 1 - eps)
    y = np.asarray(y_true, dtype=int)
    pos = (y == 1); neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        # 한쪽 클래스가 없으면 일반 logloss
        return float(-(y*np.log(p) + (1-y)*np.log(1-p)).mean())
    loss_pos = -np.log(p[pos]).mean()
    loss_neg = -np.log(1 - p[neg]).mean()
    return 0.5 * (loss_pos + loss_neg)

def score_leaderboard(y_true, p):
    """리더보드 공식: 0.5*AP + 0.5*(1/(1+WLL))"""
    ap = 0.0 if (y_true.sum() == 0 or y_true.sum() == len(y_true)) else \
         average_precision_score(y_true, p)
    wll = weighted_logloss_5050(y_true, p)
    score = 0.5 * ap + 0.5 * (1.0 / (1.0 + wll))
    return ap, wll, score

# ================== Helpers ==================
def set_seed(seed):
    import random, torch
    random.seed(seed); np.random.seed(seed)
    try:
        torch.manual_seed(seed); 
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def build_binned_sparse(train_df, test_df, cols, n_bins=64, sample=200_000, seed=42):
    """수치형을 분위 기반으로 n_bins개 카테고리로 버킷팅(+ UNK=0)"""
    rng = np.random.default_rng(seed)
    tr_bins = pd.DataFrame(index=train_df.index)
    te_bins = pd.DataFrame(index=test_df.index)
    vocab_sizes = {}

    for c in cols:
        s_tr = train_df[c]
        s_te = test_df[c]

        # 학습 폴드 기반 분위 경계 추정(샘플)
        nonnull = s_tr.dropna()
        if len(nonnull) == 0:
            # 전부 결측이면 단일 bin
            tr_code = pd.Series(0, index=train_df.index, dtype="int64")
            te_code = pd.Series(0, index=test_df.index, dtype="int64")
            vsize = 1  # UNK만
        else:
            if len(nonnull) > sample:
                nonnull = nonnull.sample(sample, random_state=seed)
            qs = np.linspace(0.0, 1.0, n_bins + 1)
            edges = np.unique(np.quantile(nonnull.values, qs))  # 중복 제거

            if len(edges) < 3:  # 값이 사실상 상수이면 한 개 bin
                tr_code = pd.Series(0, index=train_df.index, dtype="int64")
                te_code = pd.Series(0, index=test_df.index, dtype="int64")
                vsize = 1
            else:
                # bin id: 0..(len(edges)-2), 결측은 -1 → +1 해서 UNK=0
                def _bucketize(arr):
                    b = np.searchsorted(edges, arr, side="right") - 1
                    b = np.clip(b, 0, len(edges) - 2)
                    # 결측 처리
                    b = b.astype("float64")
                    b[np.isnan(arr)] = -1
                    return b.astype("int64")

                tr_raw = _bucketize(s_tr.values)
                te_raw = _bucketize(s_te.values)

                tr_code = pd.Series(tr_raw + 1, index=train_df.index, dtype="int64")
                te_code = pd.Series(te_raw + 1, index=test_df.index, dtype="int64")
                vsize = (len(edges) - 1) + 1  # 실제 bin수 + UNK(0)

        newc = f"{c}_bin"
        tr_bins[newc] = tr_code
        te_bins[newc] = te_code
        vocab_sizes[newc] = int(vsize)

    return tr_bins, te_bins, vocab_sizes

# ================== LightGBM & CatBoost (CV) ==================
oof_lgb = np.zeros(len(train), dtype=np.float64)
oof_cat = np.zeros(len(train), dtype=np.float64)
pred_lgb = np.zeros(len(test), dtype=np.float64)
pred_cat = np.zeros(len(test), dtype=np.float64)

skf = StratifiedKFold(n_splits=CFG["N_SPLITS"], shuffle=True, random_state=CFG["SEED"])

print("\n[Stage] LightGBM & CatBoost CV")
for fold, (tr_idx, va_idx) in enumerate(skf.split(train[feat_cols], y), 1):
    set_seed(CFG["SEED"] + fold)
    X_tr = train.iloc[tr_idx][feat_cols]
    y_tr = y[tr_idx]
    X_va = train.iloc[va_idx][feat_cols]
    y_va = y[va_idx]

    # ---- LightGBM ----
    import lightgbm as lgb
    lgb_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, categorical_feature=cat_cols if cat_cols else "auto")
    lgb_va = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, categorical_feature=cat_cols if cat_cols else "auto")
    lgb_params = dict(CFG["LGB_PARAMS"])

    booster = lgb.train(
        lgb_params, train_set=lgb_tr,
        num_boost_round=CFG["LGB_N_EST"],
        valid_sets=[lgb_tr, lgb_va],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(CFG["LGB_ES"]),        # 조기종료
            lgb.log_evaluation(period=100),           # 100iter마다 로그(조용히 하려면 period=0)
        ],
    )
    oof_lgb[va_idx] = booster.predict(X_va, num_iteration=booster.best_iteration)
    pred_lgb += booster.predict(test[feat_cols], num_iteration=booster.best_iteration) / CFG["N_SPLITS"]

    ap, wll, sc = score_leaderboard(y_va, oof_lgb[va_idx])
    print(f"  [Fold {fold}] LGB  AP {ap:.5f}  WLL {wll:.5f}  S {sc:.5f}")

    # ---- CatBoost ----
    from catboost import CatBoostClassifier, Pool
    cat_features_idx = [X_tr.columns.get_loc(c) for c in cat_cols] if cat_cols else []
    pool_tr = Pool(X_tr, label=y_tr, cat_features=cat_features_idx)
    pool_va = Pool(X_va, label=y_va, cat_features=cat_features_idx)

    cat = CatBoostClassifier(**CFG["CAT_PARAMS"], iterations=CFG["CAT_N_EST"])
    cat.fit(pool_tr, eval_set=pool_va, verbose=False, use_best_model=True, early_stopping_rounds=CFG["CAT_ES"])
    oof_cat[va_idx] = cat.predict_proba(X_va)[:, 1]
    pred_cat += cat.predict_proba(test[feat_cols])[:, 1] / CFG["N_SPLITS"]

    ap, wll, sc = score_leaderboard(y_va, oof_cat[va_idx])
    print(f"  [Fold {fold}] CAT  AP {ap:.5f}  WLL {wll:.5f}  S {sc:.5f}")

    del X_tr, X_va, lgb_tr, lgb_va, booster, pool_tr, pool_va, cat
    gc.collect()

ap_lgb, wll_lgb, sc_lgb = score_leaderboard(y, oof_lgb)
ap_cat, wll_cat, sc_cat = score_leaderboard(y, oof_cat)
print(f"[OOF] LGB  AP {ap_lgb:.5f}  WLL {wll_lgb:.5f}  S {sc_lgb:.5f}")
print(f"[OOF] CAT  AP {ap_cat:.5f}  WLL {wll_cat:.5f}  S {sc_cat:.5f}")

# ================== xDeepFM & FiBiNet (deepctr-torch) ==================
# 딥모델은 학습비용이 커서 동일 CV로 돌리면 매우 느림. 처음엔 full-train 1회 학습으로 시작.

# ===== xDeepFM & FiBiNet (deepctr-torch) =====
print("\n[Stage] xDeepFM & FiBiNet (DeepCTR-Torch, single full-train)")
import torch, numpy as np, pandas as pd, gc
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import xDeepFM, FiBiNET

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

# ---------------------------------------------
# 1) 상위 K개의 숫자만 sparse로(분위 bin) 사용
# ---------------------------------------------
TOPK_SPARSE_FROM_NUM = 40   # ⬅️ GPU 여유 없으면 20~32로 더 줄이세요
N_BINS = 32                 # 16~64 범위 추천
EMB   = 8                   # 8~16
BATCH = 2048                # 1024~4096에서 조절
EPOCHS = CFG.get("DEEP_EPOCHS", 2)

dense_cols = num_cols[:]            # 전체 수치
target = train["clicked"].values

# 간단/가벼운 중요도: mutual_info (샘플 200k)
mi_sample = min(200_000, len(train))
mi_idx = np.random.RandomState(CFG["SEED"]).choice(len(train), mi_sample, replace=False)
mi = mutual_info_classif(train[dense_cols].iloc[mi_idx].fillna(0.0), target[mi_idx], random_state=CFG["SEED"])
imp_df = pd.DataFrame({"col": dense_cols, "mi": mi}).sort_values("mi", ascending=False)

sparse_from_num = imp_df["col"].head(TOPK_SPARSE_FROM_NUM).tolist()
dense_only_cols = [c for c in dense_cols if c not in sparse_from_num]

print(f"[DEEPMODEL] sparse_from_num={len(sparse_from_num)}, dense_only={len(dense_only_cols)}")

# 분위-bin 함수
def build_binned_sparse(train_df, test_df, cols, n_bins=32, sample=200_000, seed=42):
    tr_bins = pd.DataFrame(index=train_df.index); te_bins = pd.DataFrame(index=test_df.index); vocab = {}
    for c in cols:
        s_tr = train_df[c]; s_te = test_df[c]
        nn = s_tr.dropna()
        if len(nn) == 0 or nn.nunique() <= 1:
            tr_code = pd.Series(0, index=train_df.index, dtype="int64")
            te_code = pd.Series(0, index=test_df.index, dtype="int64"); vsize = 1
        else:
            if len(nn) > sample: nn = nn.sample(sample, random_state=seed)
            qs = np.linspace(0, 1, n_bins+1); edges = np.unique(np.quantile(nn.values, qs))
            if len(edges) < 3:
                tr_code = pd.Series(0, index=train_df.index, dtype="int64")
                te_code = pd.Series(0, index=test_df.index, dtype="int64"); vsize = 1
            else:
                def _bucketize(arr):
                    b = np.searchsorted(edges, arr, side="right") - 1
                    b = np.clip(b, 0, len(edges)-2)
                    b = b.astype("float64"); b[np.isnan(arr)] = -1
                    return b.astype("int64")
                tr_code = pd.Series(_bucketize(s_tr.values)+1, index=train_df.index, dtype="int64")  # +1(UNK=0)
                te_code = pd.Series(_bucketize(s_te.values)+1, index=test_df.index, dtype="int64")
                vsize = (len(edges)-1) + 1
        newc = f"{c}_bin"
        tr_bins[newc] = tr_code; te_bins[newc] = te_code; vocab[newc] = int(vsize)
    return tr_bins, te_bins, vocab

# bin 생성(선택된 TOPK만)
tr_bins, te_bins, vocab_sizes = build_binned_sparse(train, test, sparse_from_num, n_bins=N_BINS)

# 딥 모델용 데이터프레임
train_deep = pd.concat([tr_bins, train[dense_only_cols].astype("float32"), train["clicked"]], axis=1)
test_deep  = pd.concat([te_bins,  test[dense_only_cols].astype("float32")], axis=1)

sparse_cols = list(tr_bins.columns)
dense_vec_dim = len(dense_only_cols)

# ---------------------------------------------
# 2) Feature columns (Sparse + 하나의 Dense 벡터)
# ---------------------------------------------
fixlen_feature_columns = []
for c in sparse_cols:
    fixlen_feature_columns.append(SparseFeat(c, vocabulary_size=vocab_sizes[c], embedding_dim=EMB))
if dense_vec_dim > 0:
    fixlen_feature_columns.append(DenseFeat("dense", dense_vec_dim))

feature_names = get_feature_names(fixlen_feature_columns)

def build_inputs(df):
    feed = {}
    for name in feature_names:
        if name == "dense":
            feed[name] = df[dense_only_cols].values.astype("float32")
        else:
            feed[name] = df[name].values
    return feed

# split
X_tr, X_va = train_test_split(train_deep, test_size=0.1, stratify=train_deep["clicked"], random_state=CFG["SEED"])
y_tr_deep = X_tr["clicked"].values; y_va_deep = X_va["clicked"].values
X_tr = X_tr.drop(columns=["clicked"]); X_va = X_va.drop(columns=["clicked"])

train_input = build_inputs(X_tr); valid_input = build_inputs(X_va); test_input = build_inputs(test_deep)

# ---------------------------------------------
# 3) 학습 함수(작은 모델 + 작은 배치)
# ---------------------------------------------
# === 공통 학습 유틸 ===
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
def _fit_and_pred(model, train_input, y_tr, valid_input, y_va, test_input,
                  batch, epochs, lr=1e-3,
                  monitor='val_auc', mode='max', patience=2,
                  model_name='model', device='cuda'):
    os.makedirs("./Toss/_ckpt", exist_ok=True)
    ckpt_path = f"./Toss/_ckpt/{model_name}_best.pth"

    model.compile(
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        loss="binary_crossentropy", metrics=["auc"]
    )
    es = EarlyStopping(monitor=monitor, patience=patience, mode=mode, verbose=True)
    mc = ModelCheckpoint(filepath=ckpt_path, monitor=monitor, mode=mode,
                         save_best_only=True, verbose=False)

    model.fit(
        train_input, y_tr,
        batch_size=batch, epochs=epochs, verbose=2,
        validation_data=(valid_input, y_va),
        callbacks=[es, mc]
    )

    # ---- Robust load of best checkpoint ----
    loaded = None
    # 1) weights-only 경로
    try:
        loaded = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        pass

    if loaded is not None:
        # state_dict or dict wrapper
        if isinstance(loaded, dict):
            # deepctr-torch 어떤 버전은 {'state_dict': ...}로 감싸기도 함
            state_dict = None
            if any(isinstance(v, torch.Tensor) for v in loaded.values()):
                state_dict = loaded
            elif "state_dict" in loaded:
                state_dict = loaded["state_dict"]
            elif "model_state_dict" in loaded:
                state_dict = loaded["model_state_dict"]
            if state_dict is not None:
                model.load_state_dict(state_dict, strict=False)
        # else: 예상치 못한 타입은 무시하고 2)로 진행

    # 2) 전체 객체 로드로 폴백(로컬 신뢰 환경에서만 권장)
    if loaded is None:
        try:
            obj = torch.load(ckpt_path, map_location=device, weights_only=False)
            # obj가 모델이면 교체
            try:
                from deepctr_torch.models.xdeepfm import xDeepFM as XDeepFMClass
                from deepctr_torch.models.fibinet import FiBiNET as FiBiNETClass
                if isinstance(obj, (XDeepFMClass, FiBiNETClass)):
                    model = obj.to(device)
                elif isinstance(obj, dict):
                    # dict로 저장된 다양한 케이스 수용
                    st = obj.get("state_dict") or obj.get("model_state_dict") or obj
                    if isinstance(st, dict):
                        model.load_state_dict(st, strict=False)
            except Exception:
                # 클래스 비교 실패해도 state_dict만 시도
                if isinstance(obj, dict):
                    st = obj.get("state_dict") or obj.get("model_state_dict") or obj
                    if isinstance(st, dict):
                        model.load_state_dict(st, strict=False)
        except Exception as e:
            print(f"[WARN] fallback load failed: {e}  -> using last-epoch weights")

    # ---- Predict with best (or last) weights ----
    pred_va = model.predict(valid_input, batch_size=batch).reshape(-1)
    pred_te = model.predict(test_input,  batch_size=batch).reshape(-1)
    return pred_va, pred_te

# === xDeepFM 학습 ===
model = xDeepFM(
    dnn_feature_columns=fixlen_feature_columns,
    linear_feature_columns=fixlen_feature_columns,
    dnn_hidden_units=(128, 64),
    cin_layer_size=(64, 32),
    task='binary', device=device
)
_, pred_xdeep_te = _fit_and_pred(
    model, train_input, y_tr_deep, valid_input, y_va_deep, test_input,
    batch=BATCH, epochs=EPOCHS, lr=CFG.get("DEEP_LR",1e-3),
    monitor='val_auc', mode='max', patience=5, model_name='xdeepfm', device=device
)

# FiBiNet
model = FiBiNET(
    linear_feature_columns=fixlen_feature_columns,
    dnn_feature_columns=fixlen_feature_columns,
    dnn_hidden_units=(128, 64),
    bilinear_type='interaction',  # 버전에 없으면 삭제
    reduction_ratio=3,            # 버전에 없으면 삭제
    task='binary', device=device
)
_, pred_fibi_te = _fit_and_pred(
    model, train_input, y_tr_deep, valid_input, y_va_deep, test_input,
    batch=BATCH, epochs=EPOCHS, lr=CFG.get("DEEP_LR",1e-3),
    monitor='val_auc', mode='max', patience=5, model_name='fibinet', device=device
)

# ================== Ensemble ==================
# 트리(CV 평균) + 딥(단일 full-train) 단순 평균
pred_ens_test = (pred_lgb + pred_cat + pred_xdeep_te + pred_fibi_te) / 4.0

# ================== Output (optional)
os.makedirs("./Toss/_out", exist_ok=True)
sub = pd.read_csv("./Toss/sample_submission.csv")
sub['clicked'] = pred_ens_test
sub_path = f"./Toss/_out/{SEED}_sub_lgb_cat_xdeep_fibi.csv"
sub.to_csv(sub_path, index=False)
print(f"\n[Done] submission saved: {sub_path}")

# 요약 로그 출력
ap_lgb, wll_lgb, sc_lgb = score_leaderboard(y, oof_lgb)
ap_cat, wll_cat, sc_cat = score_leaderboard(y, oof_cat)

summary = {
  "features": {"total": len(feat_cols), "cat": len(cat_cols), "num": len(num_cols)},
  "oof": {
    "lgb": {"AP": ap_lgb, "WLL": wll_lgb, "Score": sc_lgb},
    "cat": {"AP": ap_cat, "WLL": wll_cat, "Score": sc_cat}
  }
}
print("[Summary]", json.dumps(summary, indent=2))
