# -*- coding: utf-8 -*-
import os
# ===== 로그/환경 최소화 =====
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")
try:
    from lightgbm.basic import LightGBMWarning
    warnings.filterwarnings("ignore", category=LightGBMWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import gc, random, pickle, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression

# =========================
# Config
# =========================
SEED = 42
random.seed(SEED); np.random.seed(SEED)

TRAIN_PATH = "./Toss/train.parquet"
TEST_PATH  = "./Toss/test.parquet"
SUBMIT_TPL = "./Toss/sample_submission.csv"
SUBMIT_OUT = "./Toss/new_data/submit_cv_ens.csv"

# 캐시
CACHE_DIR = "./Toss/cache"; os.makedirs(CACHE_DIR, exist_ok=True)
TRAIN_SEQ_F = f"{CACHE_DIR}/train_seq.parquet"
TEST_SEQ_F  = f"{CACHE_DIR}/test_seq.parquet"
TRAIN_ENC_F = f"{CACHE_DIR}/train_encoded.feather"
TEST_ENC_F  = f"{CACHE_DIR}/test_encoded.feather"
TE_MAPS_F   = f"{CACHE_DIR}/te_maps.pkl"

target_col = "clicked"
seq_col    = "seq"
cand_cats  = ["gender","age_group","inventory_id","day_of_week","hour","l_feat_14"]
TE_SMOOTH_M = 20.0

CFG = {
    "SEED": SEED,
    "N_SPLITS": 5,

    # LightGBM (CPU 안정)
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
        device_type="cpu",
        verbosity=-1,
    ),
    "LGB_N_EST": 4000,
    "LGB_ES": 200,

    # CatBoost (GPU 가속, 얕은 트리로 속도↑)
    "CAT_PARAMS": dict(
        loss_function="Logloss",
        eval_metric="AUC",
        learning_rate=0.06,
        depth=6,
        l2_leaf_reg=6.0,
        random_seed=SEED,
        task_type="GPU", devices="0",
        bootstrap_type="Bernoulli",
        subsample=0.8,
        rsm=0.7,
        allow_writing_files=False,
        verbose=False,
    ),
    "CAT_N_EST": 2000,
    "CAT_ES": 200,

    # Deep (xDeepFM)
    "USE_DEEP": True,
    "DEEP_MODEL": "xdeepfm",         # xdeepfm | fibinet
    "DEEP_EPOCHS": 12,               # 안정 가동 기본값
    "DEEP_BATCH": 4096,
    "DEEP_LR": 1e-3,
    "DEEP_EMB": 16,
    "TOPK_SPARSE_FROM_NUM": 32,
    "N_BINS": 32,
    "DEEP_PLATT": True,
    "DEEP_DEVICE": "cpu",            # <<< 안정 실행 기본: 'cpu' (GPU 쓰려면 'cuda')
}

# =========================
# Utils
# =========================
def free_memory():
    gc.collect()

def weighted_logloss_5050(y_true, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    y = np.asarray(y_true, dtype=int)
    pos = (y == 1); neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float(-(y*np.log(p) + (1-y)*np.log(1-p)).mean())
    loss_pos = -np.log(p[pos]).mean()
    loss_neg = -np.log(1 - p[neg]).mean()
    return 0.5 * (loss_pos + loss_neg)

def score_leaderboard(y_true, p):
    ap = 0.0 if (y_true.sum() == 0 or y_true.sum() == len(y_true)) else average_precision_score(y_true, p)
    wll = weighted_logloss_5050(y_true, p)
    return ap, wll, 0.5 * ap + 0.5 * (1.0 / (1.0 + wll))

# ----- seq agg -----
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

# ----- encoders -----
def add_freq_encoding(tr, te, cols):
    for c in cols:
        freq = tr[c].astype(str).fillna("UNK").value_counts()
        tr[f"{c}_freq"] = tr[c].astype(str).fillna("UNK").map(freq).fillna(0).astype(np.float32)
        te[f"{c}_freq"] = te[c].astype(str).fillna("UNK").map(freq).fillna(0).astype(np.float32)

def add_kfold_target_encoding_with_smoothing(tr, te, cols, y, n_splits=5, m=20.0):
    tr = tr.copy(); te = te.copy()
    y = y.loc[tr.index]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    global_mean = float(y.mean()); te_maps = {}
    for c in cols:
        tr[c] = tr[c].astype(str).fillna("UNK")
        te[c] = te[c].astype(str).fillna("UNK")
        te_col = f"{c}_te"; tr[te_col] = np.nan

        df_all = pd.DataFrame({"cat": tr[c].values, "y": y.values})
        grp_all = df_all.groupby("cat")["y"].agg(["sum","count"]).reset_index()
        grp_all["sm"] = (grp_all["sum"] + global_mean*m) / (grp_all["count"] + m)
        map_all = dict(zip(grp_all["cat"], grp_all["sm"]))

        for tr_idx, va_idx in kf.split(tr):
            df_fold = pd.DataFrame({"cat": tr.iloc[tr_idx][c].values, "y": y.iloc[tr_idx].values})
            grp = df_fold.groupby("cat")["y"].agg(["sum","count"]).reset_index()
            grp["sm"] = (grp["sum"] + global_mean*m) / (grp["count"] + m)
            map_fold = dict(zip(grp["cat"], grp["sm"]))
            cats_val = tr.iloc[va_idx][c].map(map_fold).fillna(global_mean).astype(np.float32).values
            tr.loc[tr.index[va_idx], te_col] = cats_val

        tr[te_col] = tr[te_col].fillna(global_mean).astype(np.float32)
        te[te_col] = te[c].map(map_all).fillna(global_mean).astype(np.float32)
        te_maps[c] = {"map": map_all, "global": global_mean, "m": m}
    return tr, te, te_maps

# =========================
# Load & build features
# =========================
train = pd.read_parquet(TRAIN_PATH, engine="pyarrow")
test  = pd.read_parquet(TEST_PATH,  engine="pyarrow")
cat_cols = [c for c in cand_cats if c in train.columns]
EXCLUDE = set([target_col, seq_col, "ID"])
base_feature_cols = [c for c in train.columns if c not in EXCLUDE]

if os.path.exists(TRAIN_SEQ_F) and os.path.exists(TEST_SEQ_F):
    train_seq = pd.read_parquet(TRAIN_SEQ_F)
    test_seq  = pd.read_parquet(TEST_SEQ_F)
else:
    train_seq = build_seq_features(train, seq_col)
    test_seq  = build_seq_features(test,  seq_col)
    train_seq.to_parquet(TRAIN_SEQ_F, index=False)
    test_seq.to_parquet(TEST_SEQ_F,  index=False)

if os.path.exists(TRAIN_ENC_F) and os.path.exists(TEST_ENC_F) and os.path.exists(TE_MAPS_F):
    train_ = pd.read_feather(TRAIN_ENC_F)
    test_  = pd.read_feather(TEST_ENC_F)
    with open(TE_MAPS_F, "rb") as f:
        te_maps = pickle.load(f)
else:
    test_base_cols = [c for c in base_feature_cols if c in test.columns]
    train_ = pd.concat([train[base_feature_cols].reset_index(drop=True),
                        train_seq.reset_index(drop=True)], axis=1)
    test_  = pd.concat([test[test_base_cols].reset_index(drop=True),
                        test_seq.reset_index(drop=True)],  axis=1)

    add_freq_encoding(train_, test_, cat_cols)
    y_all = train[target_col].astype(np.float32).reset_index(drop=True)
    train_, test_, te_maps = add_kfold_target_encoding_with_smoothing(
        train_, test_, cat_cols, y=y_all, n_splits=5, m=TE_SMOOTH_M
    )
    # 해시 -> int (후에 연속정수는 모델별로 처리)
    for c in cat_cols:
        train_[c] = train_[c].astype(str).apply(lambda s: abs(hash(s)) % (10**6)).astype(np.int32)
        test_[c]  = test_[c].astype(str).apply(lambda s: abs(hash(s)) % (10**6)).astype(np.int32)

    train_.reset_index(drop=True).to_feather(TRAIN_ENC_F)
    test_.reset_index(drop=True).to_feather(TEST_ENC_F)
    with open(TE_MAPS_F, "wb") as f:
        pickle.dump(te_maps, f)

feat_cols = list(train_.columns)
cat_cols_in_use = [c for c in cat_cols if c in feat_cols]
num_cols_in_use = [c for c in feat_cols if c not in cat_cols_in_use]

print(f"[INFO] features: total={len(feat_cols)} | cat={len(cat_cols_in_use)} | num={len(num_cols_in_use)}")

# 타입 정리
for c in cat_cols_in_use:
    train_[c] = pd.Series(train_[c]).fillna(-1).astype("int32")
    test_[c]  = pd.Series(test_[c]).fillna(-1).astype("int32")
for c in num_cols_in_use:
    train_[c] = pd.to_numeric(train_[c], errors="coerce").astype("float32")
    test_[c]  = pd.to_numeric(test_[c], errors="coerce").astype("float32")

# =========================
# Downsampling: pos 모두 + neg 랜덤 (2x pos)
# =========================
y = train[target_col].astype(np.float32).reset_index(drop=True)
rng = np.random.default_rng(CFG["SEED"])
pos_idx = np.flatnonzero(y.values == 1)
neg_idx = np.flatnonzero(y.values == 0)
k = min(len(neg_idx), 2 * len(pos_idx))
neg_samp = rng.choice(neg_idx, size=k, replace=False)
use_idx = np.concatenate([pos_idx, neg_samp]); rng.shuffle(use_idx)

train_ds = train_.iloc[use_idx].reset_index(drop=True).copy()
y_ds     = y.iloc[use_idx].reset_index(drop=True).copy()
print(f"[INFO] Downsampled: total={len(train_ds):,}  pos={int(y_ds.sum()):,} "
      f"neg={len(train_ds)-int(y_ds.sum()):,}  prev={y_ds.mean():.4f}")

# 공통 폴드 (모든 모델 공유)
skf = StratifiedKFold(n_splits=CFG["N_SPLITS"], shuffle=True, random_state=CFG["SEED"])
folds = list(skf.split(train_ds[feat_cols], y_ds))

# =========================
# LightGBM CV
# =========================
oof_lgb = np.zeros(len(train_ds), dtype=np.float64)
pred_lgb = np.zeros(len(test_),  dtype=np.float64)
best_s_lgb = -np.inf
try:
    import lightgbm as lgb
    with tqdm(total=CFG["N_SPLITS"], desc="LGBM CV", ncols=100) as pbar:
        for fold, (tr_idx, va_idx) in enumerate(folds, 1):
            X_tr = train_ds.iloc[tr_idx][feat_cols]
            y_tr = y_ds.iloc[tr_idx].values
            X_va = train_ds.iloc[va_idx][feat_cols]
            y_va = y_ds.iloc[va_idx].values

            lgb_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols,
                                 categorical_feature=cat_cols_in_use if cat_cols_in_use else "auto")
            lgb_va = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols,
                                 categorical_feature=cat_cols_in_use if cat_cols_in_use else "auto")

            booster = lgb.train(
                CFG["LGB_PARAMS"], train_set=lgb_tr,
                num_boost_round=CFG["LGB_N_EST"],
                valid_sets=[lgb_va], valid_names=["valid"],
                callbacks=[lgb.early_stopping(CFG["LGB_ES"], verbose=False),
                           lgb.log_evaluation(period=0)]
            )

            p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
            y_va_true = lgb_va.get_label()
            if (p_va.min() < 0.0) or (p_va.max() > 1.0):
                p_va = 1.0 / (1.0 + np.exp(-p_va))

            tqdm.write(f"[dbg/LGB f{fold}] pred=({p_va.min():.4f},{p_va.max():.4f}) "
                       f"prev={y_va_true.mean():.4f} logloss={log_loss(y_va_true, p_va):.5f} "
                       f"auc={roc_auc_score(y_va_true, p_va):.5f} "
                       f"ap={average_precision_score(y_va_true, p_va):.5f}")

            oof_lgb[va_idx] = p_va
            pred_lgb += booster.predict(test_[feat_cols], num_iteration=booster.best_iteration) / CFG["N_SPLITS"]

            _, _, s = score_leaderboard(y_va_true, p_va)
            if s > best_s_lgb: best_s_lgb = s
            pbar.set_postfix(best=f"{best_s_lgb:.5f}"); pbar.update(1)

            del X_tr, X_va, lgb_tr, lgb_va; gc.collect()

    ap_lgb, wll_lgb, sc_lgb = score_leaderboard(y_ds.values, oof_lgb)
    print(f"[OOF] LGB  AP {ap_lgb:.5f}  WLL {wll_lgb:.5f}  S {sc_lgb:.5f}")
except Exception as e:
    print(f"[WARN] LightGBM skipped: {e}")

# =========================
# CatBoost CV (+ Platt)
# =========================
oof_cat = np.zeros(len(train_ds), dtype=np.float64)
pred_cat = np.zeros(len(test_),  dtype=np.float64)
best_s_cat = -np.inf
try:
    from catboost import CatBoostClassifier, Pool
    with tqdm(total=CFG["N_SPLITS"], desc="CatBoost CV", ncols=100) as pbar:
        for fold, (tr_idx, va_idx) in enumerate(folds, 1):
            X_tr = train_ds.iloc[tr_idx][feat_cols].copy()
            X_va = train_ds.iloc[va_idx][feat_cols].copy()
            for c in cat_cols_in_use:
                X_tr[c] = X_tr[c].astype("int32")
                X_va[c] = X_va[c].astype("int32")
            y_tr = y_ds.iloc[tr_idx].values
            y_va = y_ds.iloc[va_idx].values

            cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols_in_use] if cat_cols_in_use else []
            pool_tr = Pool(X_tr, label=y_tr, cat_features=cat_idx)
            pool_va = Pool(X_va, label=y_va, cat_features=cat_idx)

            cat = CatBoostClassifier(**CFG["CAT_PARAMS"], iterations=CFG["CAT_N_EST"])
            cat.fit(pool_tr, eval_set=pool_va, use_best_model=True, early_stopping_rounds=CFG["CAT_ES"])

            p_va_raw = cat.predict_proba(X_va)[:, 1]
            y_va_true = pool_va.get_label()

            tqdm.write(f"[dbg/CAT f{fold} pre-cal] pred=({p_va_raw.min():.4f},{p_va_raw.max():.4f}) "
                       f"prev={np.mean(y_va_true):.4f} "
                       f"logloss={log_loss(y_va_true, p_va_raw):.5f} "
                       f"auc={roc_auc_score(y_va_true, p_va_raw):.5f} "
                       f"ap={average_precision_score(y_va_true, p_va_raw):.5f}")

            cal = LogisticRegression(solver="lbfgs", max_iter=200)
            cal.fit(p_va_raw.reshape(-1,1), y_va_true)
            p_va = cal.predict_proba(p_va_raw.reshape(-1,1))[:,1]

            tqdm.write(f"[dbg/CAT f{fold}  post-cal] pred=({p_va.min():.4f},{p_va.max():.4f}) "
                       f"logloss={log_loss(y_va_true, p_va):.5f} "
                       f"auc={roc_auc_score(y_va_true, p_va):.5f} "
                       f"ap={average_precision_score(y_va_true, p_va):.5f}")

            oof_cat[va_idx] = p_va
            p_te_raw = cat.predict_proba(test_[feat_cols])[:, 1]
            pred_cat += cal.predict_proba(p_te_raw.reshape(-1,1))[:,1] / CFG["N_SPLITS"]

            _, _, s = score_leaderboard(y_va_true, p_va)
            if s > best_s_cat: best_s_cat = s
            pbar.set_postfix(best=f"{best_s_cat:.5f}"); pbar.update(1)

            del X_tr, X_va, pool_tr, pool_va; gc.collect()

    ap_cat, wll_cat, sc_cat = score_leaderboard(y_ds.values, oof_cat)
    print(f"[OOF] CAT  AP {ap_cat:.5f}  WLL {wll_cat:.5f}  S {sc_cat:.5f}")
except Exception as e:
    print(f"[WARN] CatBoost skipped: {e}")

# =========================
# Deep model (xDeepFM/FibiNET) CV (+Platt)
# =========================
oof_deep = None
pred_deep = None
best_s_deep = -np.inf

def safe_bce(y_pred, y_true, reduction="mean", eps=1e-6):
    # deepctr-torch 호환 커스텀 BCE: NaN/Inf/범위 외 값 방어
    import torch
    import torch.nn.functional as F
    # y_pred가 확률이 아닐 수도 있으니 방어적 시그모이드
    with torch.no_grad():
        minv = torch.nan_to_num(y_pred.min(), nan=0.0)
        maxv = torch.nan_to_num(y_pred.max(), nan=1.0)
    if (minv < 0) or (maxv > 1):
        y_pred = y_pred.sigmoid()
    # NaN/Inf 보정 후 클램프
    y_pred = torch.nan_to_num(y_pred, nan=0.5, posinf=1.0 - eps, neginf=eps)
    y_pred = torch.clamp(y_pred, eps, 1.0 - eps)
    y_true = y_true.float()
    loss = F.binary_cross_entropy(y_pred, y_true, reduction="none")
    if reduction == "mean": return loss.mean()
    if reduction == "sum":  return loss.sum()
    return loss

if CFG["USE_DEEP"]:
    try:
        import torch
        from sklearn.feature_selection import mutual_info_classif
        from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
        from deepctr_torch.models import xDeepFM, FiBiNET
        from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint

        # 숫자/카테고리 분해
        num_cols_all = [c for c in feat_cols if c not in cat_cols_in_use]

        # MI 기반 일부 숫자를 sparse binning 후보로
        mi_sample = min(200_000, len(train_))
        rs = np.random.RandomState(CFG["SEED"])
        mi_idx = rs.choice(len(train_), mi_sample, replace=False)
        mi = mutual_info_classif(train_[num_cols_all].iloc[mi_idx].fillna(0.0),
                                 y.iloc[mi_idx], random_state=CFG["SEED"])
        imp_df = pd.DataFrame({"col": num_cols_all, "mi": mi}).sort_values("mi", ascending=False)
        sparse_from_num = imp_df["col"].head(CFG["TOPK_SPARSE_FROM_NUM"]).tolist()
        dense_only_cols = [c for c in num_cols_all if c not in sparse_from_num]

        # 분위 binning
        def build_binned_sparse(train_df, test_df, cols, n_bins=32, sample=200_000, seed=42):
            tr_bins = pd.DataFrame(index=train_df.index); te_bins = pd.DataFrame(index=test_df.index); vocab = {}
            for c in cols:
                s_tr, s_te = train_df[c], test_df[c]
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
                        tr_code = pd.Series(_bucketize(s_tr.values), index=train_df.index, dtype="int64")
                        te_code = pd.Series(_bucketize(s_te.values), index=test_df.index, dtype="int64")
                        vsize = (len(edges)-1)
                tr_code = tr_code.fillna(0).clip(lower=0).astype("int64")
                te_code = te_code.fillna(0).clip(lower=0).astype("int64")
                if vsize <= 0: vsize = 1
                tr_code = tr_code.clip(0, vsize-1)
                te_code = te_code.clip(0, vsize-1)
                newc = f"{c}_bin"
                tr_bins[newc] = tr_code; te_bins[newc] = te_code; vocab[newc] = int(vsize)
            return tr_bins, te_bins, vocab

        def sanitize_sparse_indices(df, sparse_cols, vocab_sizes):
            # 0..vocab-1, int64 강제
            for c in sparse_cols:
                v = int(vocab_sizes[c])
                x = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
                bad = (x < 0) | (x >= v)
                if bad.any():
                    x[bad] = 0
                df[c] = x

        # 전체 기준으로 bin 경계 학습 → 다운샘플 서브셋 사용
        tr_bins, te_bins, vocab_sizes = build_binned_sparse(train_, test_, sparse_from_num,
                                                            n_bins=CFG["N_BINS"], seed=CFG["SEED"])
        train_deep_raw = pd.concat([tr_bins, train_[dense_only_cols].astype("float32"),
                                    y.reset_index(drop=True)], axis=1)
        test_deep  = pd.concat([te_bins,  test_[dense_only_cols].astype("float32")], axis=1)

        # 다운샘플 서브셋
        train_deep = train_deep_raw.iloc[use_idx].reset_index(drop=True)
        y_deep = train_deep["clicked"].values.astype("float32")
        X_deep = train_deep.drop(columns=["clicked"])

        sparse_cols = list(tr_bins.columns)
        sanitize_sparse_indices(train_deep, sparse_cols, vocab_sizes)
        sanitize_sparse_indices(test_deep,  sparse_cols, vocab_sizes)

        # feature columns
        fixlen_feature_columns = []
        for c in sparse_cols:
            fixlen_feature_columns.append(
                SparseFeat(c, vocabulary_size=int(vocab_sizes[c]), embedding_dim=CFG["DEEP_EMB"])
            )
        if len(dense_only_cols) > 0:
            fixlen_feature_columns.append(DenseFeat("dense", len(dense_only_cols)))
        feature_names = get_feature_names(fixlen_feature_columns)

        def build_inputs(df):
            feed = {}
            for name in feature_names:
                if name == "dense":
                    feed[name] = df[dense_only_cols].values.astype("float32")
                else:
                    # int64 보장
                    feed[name] = pd.to_numeric(df[name], errors="coerce").fillna(0).astype("int64").values
            return feed

        device = CFG["DEEP_DEVICE"]
        oof_deep = np.zeros(len(train_deep), dtype=np.float64)
        pred_deep = np.zeros(len(test_deep), dtype=np.float64)

        with tqdm(total=CFG["N_SPLITS"], desc="xDeepFM CV", ncols=100) as pbar:
            for fold, (tr_idx, va_idx) in enumerate(folds, 1):
                tr_df = X_deep.iloc[tr_idx]; va_df = X_deep.iloc[va_idx]
                y_trd = y_deep[tr_idx].astype("float32"); y_vad = y_deep[va_idx].astype("float32")

                train_input = build_inputs(tr_df)
                valid_input = build_inputs(va_df)
                test_input  = build_inputs(test_deep)

                if CFG["DEEP_MODEL"].lower() == "fibinet":
                    model = FiBiNET(dnn_feature_columns=fixlen_feature_columns,
                                    linear_feature_columns=fixlen_feature_columns,
                                    dnn_hidden_units=(256,128,64),
                                    task='binary', device=device)
                else:
                    model = xDeepFM(dnn_feature_columns=fixlen_feature_columns,
                                    linear_feature_columns=fixlen_feature_columns,
                                    dnn_hidden_units=(128,64),
                                    cin_layer_size=(64,32),
                                    task='binary', device=device)

                model.compile(optimizer=torch.optim.Adam(model.parameters(), lr=CFG["DEEP_LR"]),
                              loss=safe_bce, metrics=["auc"])
                es = EarlyStopping(monitor='val_auc', patience=5, mode='max', verbose=False)
                mc = ModelCheckpoint(filepath=f"./Toss/_ckpt/{CFG['DEEP_MODEL']}_fold{fold}.pth",
                                     monitor='val_auc', mode='max', save_best_only=True, verbose=False)

                model.fit(train_input, y_trd,
                          batch_size=CFG["DEEP_BATCH"], epochs=CFG["DEEP_EPOCHS"], verbose=0,
                          validation_data=(valid_input, y_vad), callbacks=[es, mc])

                p_va_raw = model.predict(valid_input, batch_size=CFG["DEEP_BATCH"]).reshape(-1)
                # (옵션) Platt
                if CFG["DEEP_PLATT"]:
                    cal = LogisticRegression(solver="lbfgs", max_iter=200)
                    cal.fit(p_va_raw.reshape(-1,1), y_vad)
                    p_va = cal.predict_proba(p_va_raw.reshape(-1,1))[:,1]
                    p_te_raw = model.predict(test_input, batch_size=CFG["DEEP_BATCH"]).reshape(-1)
                    p_te = cal.predict_proba(p_te_raw.reshape(-1,1))[:,1]
                else:
                    p_va = p_va_raw
                    p_te = model.predict(test_input, batch_size=CFG["DEEP_BATCH"]).reshape(-1)

                print(f"[dbg/DEEP f{fold}] pred=({p_va.min():.4f},{p_va.max():.4f}) "
                      f"prev={float(np.mean(y_vad)):.4f} "
                      f"logloss={log_loss(y_vad, p_va):.5f} "
                      f"auc={roc_auc_score(y_vad, p_va):.5f} "
                      f"ap={average_precision_score(y_vad, p_va):.5f}")

                oof_deep[va_idx] = p_va
                pred_deep += p_te / CFG["N_SPLITS"]

                _, _, s = score_leaderboard(y_vad, p_va)
                if s > best_s_deep: best_s_deep = s
                pbar.set_postfix(best=f"{best_s_deep:.5f}"); pbar.update(1)

                del tr_df, va_df, train_input, valid_input, model; gc.collect()

        ap_d, wll_d, sc_d = score_leaderboard(y_deep, oof_deep)
        print(f"[OOF] DEEP AP {ap_d:.5f}  WLL {wll_d:.5f}  S {sc_d:.5f}")

    except Exception as e:
        print(f"[WARN] Deep model skipped: {e}")
        oof_deep = None; pred_deep = None

# =========================
# Ensemble & Save
# =========================
preds = []
if np.any(pred_lgb): preds.append(pred_lgb)
if np.any(pred_cat): preds.append(pred_cat)
if (pred_deep is not None): preds.append(pred_deep)
if len(preds) == 0:
    raise RuntimeError("No predictions were produced. Check installations.")
pred_ens = np.mean(preds, axis=0)

sub = pd.read_csv(SUBMIT_TPL)
sub["clicked"] = pred_ens
os.makedirs(os.path.dirname(SUBMIT_OUT), exist_ok=True)
sub.to_csv(SUBMIT_OUT, index=False)
print(f"\n[Done] submission saved: {SUBMIT_OUT}")

summary = {
    "features": {"total": len(feat_cols), "cat": len(cat_cols_in_use), "num": len(feat_cols) - len(cat_cols_in_use)},
    "downsampled_prev": float(y_ds.mean()),
    "oof_scores": {
        "lgb": float(score_leaderboard(y_ds.values, oof_lgb)[2]) if np.any(oof_lgb) else None,
        "cat": float(score_leaderboard(y_ds.values, oof_cat)[2]) if np.any(oof_cat) else None,
        "deep": float(score_leaderboard((y_deep if (oof_deep is not None) else y_ds.values),
                                        (oof_deep if (oof_deep is not None) else np.zeros_like(y_ds.values)))) \
                if (oof_deep is not None) else None,
    }
}
print("[Summary]", json.dumps(summary, indent=2))
