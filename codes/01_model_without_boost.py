# -*- coding: utf-8 -*-
import os, warnings, json, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score

# ================== Config ==================
SEED = 1
CFG = {
    "SEED": SEED,
    "N_SPLITS": 5,           # ⬅️ Stratified K-Fold
    # Deep models (deepctr-torch)
    "DEEP_EPOCHS": 100,
    "DEEP_BATCH": 2048,
    "DEEP_LR": 1e-3,
    "DEEP_EMB": 8,
    "TOPK_SPARSE_FROM_NUM": 40,   # OOM 시 20~32로
    "N_BINS": 32,                 # 16~64
    "MUTUAL_INFO_SAMPLE": 200_000,
    "ES_PATIENCE": 6,
}

rng = np.random.default_rng(CFG["SEED"])

# ================== Load ==================
train = pd.read_parquet("./Toss/new_data/new_train.parquet")
test  = pd.read_parquet("./Toss/new_data/new_test.parquet")
assert "clicked" in train.columns

feat_cols = [c for c in train.columns if c != "clicked"]
y = train["clicked"].astype(np.int8).to_numpy()

# ================== 지표 ==================
def weighted_logloss_5050(y_true, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    yb = np.asarray(y_true, dtype=int)
    pos = (yb == 1); neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float(-(yb*np.log(p) + (1-yb)*np.log(1-p)).mean())
    loss_pos = -np.log(p[pos]).mean()
    loss_neg = -np.log(1 - p[neg]).mean()
    return 0.5 * (loss_pos + loss_neg)

def score_leaderboard(y_true, p):
    ap = 0.0 if (y_true.sum() == 0 or y_true.sum() == len(y_true)) else average_precision_score(y_true, p)
    wll = weighted_logloss_5050(y_true, p)
    score = 0.5 * ap + 0.5 * (1.0 / (1.0 + wll))
    return ap, wll, score

# ================== 상위 K 수치 선택 (전데이터 기준 1회) ==================
num_cols = sorted(feat_cols)
mi_sample = min(CFG["MUTUAL_INFO_SAMPLE"], len(train))
mi_idx = np.random.RandomState(CFG["SEED"]).choice(len(train), mi_sample, replace=False)
mi = mutual_info_classif(train[num_cols].iloc[mi_idx].fillna(0.0), y[mi_idx], random_state=CFG["SEED"])
imp_df = pd.DataFrame({"col": num_cols, "mi": mi}).sort_values("mi", ascending=False)

sparse_from_num = imp_df["col"].head(CFG["TOPK_SPARSE_FROM_NUM"]).tolist()
dense_only_cols = [c for c in num_cols if c not in sparse_from_num]
print(f"[DEEPMODEL] sparse_from_num={len(sparse_from_num)}, dense_only={len(dense_only_cols)}")

# ================== 분위 bin (폴드별: train-fold로 edges fit) ==================
def fit_bin_edges(s, n_bins=32, sample=200_000, seed=42):
    nn = s.dropna()
    if len(nn) == 0 or nn.nunique() <= 1:
        return np.array([0.0, 1.0])  # dummy edges
    if len(nn) > sample:
        nn = nn.sample(sample, random_state=seed)
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.unique(np.quantile(nn.values, qs))
    if len(edges) < 3:
        return np.array([0.0, 1.0])
    return edges

def bucketize(arr, edges):
    # bin id: 0..(len(edges)-2), NaN -> -1, 최종 +1(UNK=0)
    b = np.searchsorted(edges, arr, side="right") - 1
    b = np.clip(b, 0, max(0, len(edges)-2))
    b = b.astype("float64")
    b[np.isnan(arr)] = -1
    return (b.astype("int64") + 1)  # UNK=0

def build_binned_sparse_fold(tr_df, va_df, te_df, cols, n_bins=32, sample=200_000, seed=42):
    tr_bins = pd.DataFrame(index=tr_df.index); va_bins = pd.DataFrame(index=va_df.index); te_bins = pd.DataFrame(index=te_df.index)
    vocab = {}
    for c in cols:
        edges = fit_bin_edges(tr_df[c], n_bins=n_bins, sample=sample, seed=seed)
        if len(edges) <= 2:  # 상수 컬럼
            tr_bins[f"{c}_bin"] = 0
            va_bins[f"{c}_bin"] = 0
            te_bins[f"{c}_bin"] = 0
            vocab[f"{c}_bin"] = 1
        else:
            tr_bins[f"{c}_bin"] = bucketize(tr_df[c].values, edges)
            va_bins[f"{c}_bin"] = bucketize(va_df[c].values, edges)
            te_bins[f"{c}_bin"] = bucketize(te_df[c].values, edges)
            vocab[f"{c}_bin"] = (len(edges)-1) + 1  # + UNK
    return tr_bins, va_bins, te_bins, vocab

# ================== DeepCTR 준비 공통 ==================
import torch
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import xDeepFM, FiBiNET
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

def build_feature_columns(vocab_sizes, dense_dim, emb_dim):
    fcols = []
    for name, vsize in vocab_sizes.items():
        fcols.append(SparseFeat(name, vocabulary_size=int(vsize), embedding_dim=emb_dim))
    if dense_dim > 0:
        fcols.append(DenseFeat("dense", dense_dim))
    return fcols

def build_inputs(df_sparse, df_dense, feature_names):
    feed = {}
    for name in feature_names:
        if name == "dense":
            feed[name] = df_dense.values.astype("float32")
        else:
            feed[name] = df_sparse[name].values
    return feed

def _fit_and_pred(model, train_input, y_tr, valid_input, y_va, test_input,
                  batch, epochs, lr=1e-3,
                  monitor='val_auc', mode='max', patience=CFG["ES_PATIENCE"],
                  ckpt_key='model', device='cuda'):
    os.makedirs("./Toss/_ckpt", exist_ok=True)
    ckpt_path = f"./Toss/_ckpt/{ckpt_key}_best.pth"

    model.compile(optimizer=torch.optim.Adam(model.parameters(), lr=lr),
                  loss="binary_crossentropy", metrics=["auc"])
    es = EarlyStopping(monitor=monitor, patience=patience, mode=mode, verbose=True)
    mc = ModelCheckpoint(filepath=ckpt_path, monitor=monitor, mode=mode, save_best_only=True, verbose=False)

    model.fit(train_input, y_tr, batch_size=batch, epochs=epochs, verbose=2,
              validation_data=(valid_input, y_va), callbacks=[es, mc])

    # 안전 로드 (PyTorch 2.6)
    if os.path.exists(ckpt_path):
        loaded = None
        try:
            loaded = torch.load(ckpt_path, map_location=device, weights_only=True)
        except Exception:
            loaded = None
        if loaded is not None:
            if isinstance(loaded, dict):
                state_dict = None
                if any(isinstance(v, torch.Tensor) for v in loaded.values()):
                    state_dict = loaded
                elif "state_dict" in loaded: state_dict = loaded["state_dict"]
                elif "model_state_dict" in loaded: state_dict = loaded["model_state_dict"]
                if state_dict is not None:
                    model.load_state_dict(state_dict, strict=False)
        else:
            try:
                obj = torch.load(ckpt_path, map_location=device, weights_only=False)
                if isinstance(obj, dict):
                    st = obj.get("state_dict") or obj.get("model_state_dict") or obj
                    if isinstance(st, dict):
                        model.load_state_dict(st, strict=False)
                else:
                    model = obj.to(device)
            except Exception as e:
                print(f"[WARN] fallback load failed: {e} -> using last-epoch weights")

    pred_va = model.predict(valid_input, batch_size=batch).reshape(-1)
    pred_te = model.predict(test_input,  batch_size=batch).reshape(-1)
    return pred_va, pred_te

# ================== CV Loop ==================
n = len(train)
oof_xdeep = np.zeros(n, dtype=np.float64)
oof_fibi  = np.zeros(n, dtype=np.float64)
pred_xdeep_test = np.zeros(len(test), dtype=np.float64)
pred_fibi_test  = np.zeros(len(test), dtype=np.float64)

skf = StratifiedKFold(n_splits=CFG["N_SPLITS"], shuffle=True, random_state=CFG["SEED"])

for fold, (tr_idx, va_idx) in enumerate(skf.split(train[feat_cols], y), 1):
    print(f"\n===== Fold {fold}/{CFG['N_SPLITS']} =====")
    tr_df = train.iloc[tr_idx]
    va_df = train.iloc[va_idx]

    # 1) bin (edges는 train-fold 기준)
    tr_bins, va_bins, te_bins, vocab_sizes = build_binned_sparse_fold(
        tr_df, va_df, test, cols=sparse_from_num,
        n_bins=CFG["N_BINS"], sample=CFG["MUTUAL_INFO_SAMPLE"], seed=CFG["SEED"] + fold
    )

    # 2) dense
    tr_dense = tr_df[dense_only_cols].astype("float32")
    va_dense = va_df[dense_only_cols].astype("float32")
    te_dense = test[dense_only_cols].astype("float32")

    # 3) feature columns / names
    fcols = build_feature_columns(vocab_sizes, dense_dim=len(dense_only_cols), emb_dim=CFG["DEEP_EMB"])
    feature_names = get_feature_names(fcols)

    # 4) inputs
    X_tr = build_inputs(tr_bins, tr_dense, feature_names)
    X_va = build_inputs(va_bins, va_dense, feature_names)
    X_te = build_inputs(te_bins, te_dense, feature_names)
    y_tr = tr_df["clicked"].values
    y_va = va_df["clicked"].values

    # 5) xDeepFM
    torch.cuda.empty_cache()
    xdeep = xDeepFM(
        dnn_feature_columns=fcols,
        linear_feature_columns=fcols,
        dnn_hidden_units=(128, 64),
        cin_layer_size=(64, 32),
        task='binary', device=device
    )
    va_pred_x, te_pred_x = _fit_and_pred(
        xdeep, X_tr, y_tr, X_va, y_va, X_te,
        batch=CFG["DEEP_BATCH"], epochs=CFG["DEEP_EPOCHS"], lr=CFG["DEEP_LR"],
        monitor='val_auc', mode='max', patience=CFG["ES_PATIENCE"],
        ckpt_key=f'xdeepfm_f{fold}', device=device
    )
    oof_xdeep[va_idx] = va_pred_x
    pred_xdeep_test += te_pred_x / CFG["N_SPLITS"]
    ap, wll, sc = score_leaderboard(y_va, va_pred_x)
    print(f"  [Fold {fold}] xDeepFM  AP {ap:.5f}  WLL {wll:.5f}  Score {sc:.5f}")

    # 6) FiBiNet
    torch.cuda.empty_cache()
    fibi = FiBiNET(
        linear_feature_columns=fcols,
        dnn_feature_columns=fcols,
        dnn_hidden_units=(128, 64),
        bilinear_type='interaction',  # 버전에 없으면 제거
        reduction_ratio=3,            # 버전에 없으면 제거
        task='binary', device=device
    )
    va_pred_f, te_pred_f = _fit_and_pred(
        fibi, X_tr, y_tr, X_va, y_va, X_te,
        batch=CFG["DEEP_BATCH"], epochs=CFG["DEEP_EPOCHS"], lr=CFG["DEEP_LR"],
        monitor='val_auc', mode='max', patience=CFG["ES_PATIENCE"],
        ckpt_key=f'fibinet_f{fold}', device=device
    )
    oof_fibi[va_idx] = va_pred_f
    pred_fibi_test += te_pred_f / CFG["N_SPLITS"]
    ap, wll, sc = score_leaderboard(y_va, va_pred_f)
    print(f"  [Fold {fold}] FiBiNet  AP {ap:.5f}  WLL {wll:.5f}  Score {sc:.5f}")

    del tr_df, va_df, tr_bins, va_bins, te_bins, tr_dense, va_dense, te_dense
    del X_tr, X_va, X_te, xdeep, fibi
    gc.collect()
    torch.cuda.empty_cache()

# ================== OOF & Ensemble ==================
oof_ens = (oof_xdeep + oof_fibi) / 2.0
ap_x, wll_x, sc_x = score_leaderboard(y, oof_xdeep)
ap_f, wll_f, sc_f = score_leaderboard(y, oof_fibi)
ap_e, wll_e, sc_e = score_leaderboard(y, oof_ens)

print("\n[OOF] xDeepFM  AP %.5f  WLL %.5f  Score %.5f" % (ap_x, wll_x, sc_x))
print("[OOF] FiBiNet  AP %.5f  WLL %.5f  Score %.5f" % (ap_f, wll_f, sc_f))
print("[OOF] Ensemble AP %.5f  WLL %.5f  Score %.5f" % (ap_e, wll_e, sc_e))

# ================== Submission ==================
pred_ens_test = (pred_xdeep_test + pred_fibi_test) / 2.0
os.makedirs("./Toss/_out", exist_ok=True)
sub = pd.read_csv("./Toss/sample_submission.csv")
sub["clicked"] = pred_ens_test
out_path = f"./Toss/_out/{SEED}_cv5_xdeep_fibi.csv"
sub.to_csv(out_path, index=False)
print(f"\n[Done] submission saved: {out_path}")

# ================== Summary ==================
summary = {
    "features": {"total": len(feat_cols), "num": len(num_cols),
                 "sparse_from_num": len(sparse_from_num), "dense_only": len(dense_only_cols)},
    "oof": {
        "xDeepFM": {"AP": ap_x, "WLL": wll_x, "Score": sc_x},
        "FiBiNet": {"AP": ap_f, "WLL": wll_f, "Score": sc_f},
        "Ensemble": {"AP": ap_e, "WLL": wll_e, "Score": sc_e}
    }
}
print("[Summary]", json.dumps(summary, indent=2))
