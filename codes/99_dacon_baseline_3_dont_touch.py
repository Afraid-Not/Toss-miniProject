# -*- coding: utf-8 -*-
import os, gc, random, warnings, json, shutil
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True

version = "prototype"

LOG_DIR = Path("./Toss/log/")
LOG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 1

# 문자열 대신 Path로!
SAVE_DIR = Path(f'./Toss/submissions/{version}_sub/{SEED}_submission/')

# ========================= Config =========================
CFG = {
    "SEED": SEED,

    # Tokenized sequence
    "VOCAB_SIZE": 100,
    "OOV_IDX": 100,
    "PAD_IDX": 101,

    "SEQ_MAX_LEN": 1152,          # 1024보다 효율 좋은 지점
    # 커리큘럼 길이: 초반 3~5 epoch은 768 or 896으로 학습 → 
    # 이후 1152/1344로 늘려 파인튜닝하면 시간 절감 + 수렴 안정.
    
    "D_EMBED": 48,
    "LSTM_HIDDEN": 192,
    "LSTM_LAYERS": 2,
    "BIDIRECTIONAL": False,
    "DROPOUT": 0.2,
    "MLP_HIDDEN": (384, 192),

    # Training (GPU/AMP 기준)
    "BATCH_SIZE": 4096,          # 유효 배치 크게
    "MICRO_BATCH": 512,         # VRAM 부족 시 512로 낮추기
    "EPOCHS": 30,
    "LR": 6e-4,
    "NUM_WORKERS": 0,       # Windows OK(이미 __main__ 가드 있음). CPU 절반 근처
    "TEST_NUM_WORKERS" : 0,
    "PIN_MEMORY": True,     # GPU 전송 빠르게
    "USE_AMP": True,        # 반드시 켜기

    # Early Stopping (LSTM)
    "ES_ENABLE": True,
    "ES_MODE": "max",
    "ES_PATIENCE": 3,       # 빠른 수렴 유도 (0~1 권장)
    "ES_MIN_DELTA": 1e-4,
    "ES_MONITOR": "score",  # (0.5*AP + 0.5*(1/(1+WLL)))

    # XGBoost (GPU, K-Fold + ES)
    # "XGB_VALID_FRAC": 0.1,     # K-Fold 쓰면 미사용
    "XGB_NUM_BOOST_ROUND": 5000,
    "XGB_ES_ROUNDS": 250,
    "XGB_NFOLDS": 4,     

    # Paths
    "TRAIN_PATH": "./Toss/train.parquet",
    "TEST_PATH": "./Toss/test.parquet",
    "META_TRAIN": "./Toss/new_data/new_train_2.parquet",
    "META_TEST": "./Toss/new_data/new_test_2.parquet",
    "SAMPLE_SUB": "./Toss/sample_submission.csv",
    "OUT_DIR": SAVE_DIR,

}

SMOKE = False  # ★ 스모크 테스트 on/off

SMOKE_CFG = {
    # 짧게, 빨리
    "SEQ_MAX_LEN": 896,      # 본선 1152 대신 896 (p~0.81 완전보존, 시간↓)
    "D_EMBED": 32,
    "LSTM_HIDDEN": 128,
    "LSTM_LAYERS": 1,
    "MLP_HIDDEN": (256, 128),

    "BATCH_SIZE": 4096,      # 유효 배치 크게 유지
    "MICRO_BATCH": 512,      # VRAM 부족 시 256
    "EPOCHS": 3,             # ES 있으니 2~3이면 충분
    "LR": 8e-4,

    "ES_PATIENCE": 1,

    # XGBoost 가볍게
    "XGB_NFOLDS": 2,
    "XGB_NUM_BOOST_ROUND": 1200,
    "XGB_ES_ROUNDS": 100,
}
if SMOKE:
    CFG.update(SMOKE_CFG)

device = "cuda" if torch.cuda.is_available() else "cpu"

def print_cuda_info():
    print(f"[PyTorch] torch={torch.__version__}, cuda_build={getattr(torch.version, 'cuda', None)}")
    print(f"[CUDA] available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[CUDA] device_count={torch.cuda.device_count()}, name={torch.cuda.get_device_name(0)}")
        print(f"[CUDA] current_device={torch.cuda.current_device()}")


# GPU 없으면 바로 알림 (원하면 assert 풀어도 됨)
if device == "cpu":
    print("[WARN] CUDA not available. 현재 CPU로 실행됩니다. (PyTorch GPU 빌드/드라이버/CUDA Toolkit 확인)")

def seed_everything(seed: int):
    random.seed(seed); os.environ["PYTHONHASHSEED"]=str(seed)
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def free_mem():
    torch.cuda.empty_cache(); gc.collect()

# ---------- 안전 전처리(결측/Inf 처리) ----------
def clean_numeric(df: pd.DataFrame, cols):
    df = df.copy()
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    df[cols] = df[cols].fillna(0.0).astype(np.float32)
    return df

# ---------- Early Stopper ----------
class EarlyStopper:
    def __init__(self, mode="max", patience=2, min_delta=0.0):
        assert mode in ("max","min")
        self.mode=mode; self.patience=patience; self.min_delta=min_delta
        self.best = -np.inf if mode=="max" else np.inf
        self.num_bad=0; self.best_state=None; self.best_epoch=-1; self.best_metric=None
    def step(self, metric, model, epoch):
        improved = (metric > self.best + self.min_delta) if self.mode=="max" else (metric < self.best - self.min_delta)
        if improved:
            self.best = metric; self.num_bad = 0
            self.best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            self.best_epoch = epoch; self.best_metric = metric
            return False
        self.num_bad += 1
        return self.num_bad > self.patience

@torch.no_grad()
def eval_lstm_on_loader(model, loader, use_amp=True, device="cuda"):
    model.eval()
    probs_all, ys_all = [], []
    for xs, seqs, lens, ys in loader:
        B = ys.size(0)
        for i in range(0, B, CFG["MICRO_BATCH"]):
            xs_mb  = xs[i:i+CFG["MICRO_BATCH"]].to(device, non_blocking=True)
            seq_mb = seqs[i:i+CFG["MICRO_BATCH"]].to(device, non_blocking=True)
            y_mb   = ys[i:i+CFG["MICRO_BATCH"]].to(device)
            lens_mb = lens[i:i+CFG["MICRO_BATCH"]]
            with autocast(enabled=(use_amp and device=="cuda")):
                logits = model(xs_mb, seq_mb, lens_mb)
                probs  = torch.sigmoid(logits)
            probs_all.append(probs.detach().cpu().numpy())
            ys_all.append(y_mb.detach().cpu().numpy())
    y_prob = np.concatenate(probs_all) if probs_all else np.array([])
    y_true = np.concatenate(ys_all)    if ys_all    else np.array([])
    return y_true, y_prob

# ========================= Dataset / Collate =========================
class ClickDataset(Dataset):
    def __init__(self, df, feature_cols, seq_col, target_col=None, has_target=True):
        self.df=df.reset_index(drop=True); self.feature_cols=feature_cols
        self.seq_col=seq_col; self.target_col=target_col; self.has_target=has_target
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vals = row[self.feature_cols].values.astype(np.float32)
        vals = np.nan_to_num(vals, nan=0.0, posinf=1e6, neginf=-1e6)
        x = torch.tensor(vals, dtype=torch.float32)
        s = str(row[self.seq_col]) if pd.notna(row[self.seq_col]) else ""
        if self.has_target:
            y = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
            return x, s, y
        return x, s

class CollateTokens:
    def __init__(self, has_target: bool, max_len: int, oov_idx: int, pad_idx: int, vocab_size: int):
        self.has_target=has_target; self.max_len=max_len
        self.oov_idx=oov_idx; self.pad_idx=pad_idx; self.vocab_size=vocab_size
    def __call__(self, batch):
        import numpy as np, torch
        if self.has_target:
            xs, s_strs, ys = zip(*batch); ys = torch.stack(ys).float()
        else:
            xs, s_strs = zip(*batch)
        x_feats = torch.stack(xs).float()
        seqs=[]; lengths=[]; V=self.vocab_size
        for s in s_strs:
            arr = np.fromstring(str(s), sep=",", dtype=np.int64) if (s and s!="nan") else np.empty((0,), np.int64)
            if self.max_len is not None and arr.size > self.max_len:
                arr = arr[-self.max_len:]
            if arr.size:
                bad = (arr<0) | (arr>=V)
                if bad.any(): arr[bad] = self.oov_idx
            L = len(arr)
            lengths.append(L if L>0 else 1)
            seqs.append(torch.from_numpy(arr) if L>0 else torch.empty(0, dtype=torch.long))
        lengths = torch.tensor(lengths, dtype=torch.long)
        B=len(seqs); Lmax=int(max([len(t) for t in seqs], default=1))
        x_seq = torch.full((B, Lmax), self.pad_idx, dtype=torch.long)
        for i,t in enumerate(seqs):
            if len(t)>0: x_seq[i,:len(t)] = t
        if x_seq.numel():
            mn, mx = int(x_seq.min()), int(x_seq.max())
            assert mn>=0 and mx<=self.pad_idx, f"token index out of range [{mn},{mx}], pad={self.pad_idx}"
        return (x_feats, x_seq, lengths, ys) if self.has_target else (x_feats, x_seq, lengths)

# ========================= Model (Embedding -> LSTM(pack) -> MLP) =========================
# class TabularSeqTokenLSTM(nn.Module):
#     def __init__(self, d_features, vocab_size, oov_idx, pad_idx,
#                  d_embed=32, hidden=64, num_layers=1, bidir=False, dropout=0.2,
#                  mlp_hidden=(256,128)):
#         super().__init__()
#         self.bn_x = nn.LayerNorm(d_features)
#         self.emb  = nn.Embedding(vocab_size + 2, d_embed, padding_idx=pad_idx)
#         self.lstm = nn.LSTM(d_embed, hidden, num_layers=num_layers, batch_first=True,
#                             bidirectional=bidir, dropout=(dropout if num_layers>1 else 0.0))
#         lstm_out = hidden * (2 if bidir else 1)
#         in_dim = d_features + lstm_out
#         layers=[]
#         for h in mlp_hidden:
#             layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
#             in_dim = h
#         layers += [nn.Linear(in_dim, 1)]
#         self.mlp = nn.Sequential(*layers)
#     def forward(self, x_feats, x_seq, seq_lengths):
#         x = self.bn_x(x_feats)
#         e = self.emb(x_seq)
#         packed = nn.utils.rnn.pack_padded_sequence(e, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
#         _, (h_n, _) = self.lstm(packed)
#         h = h_n[-1] if not self.lstm.bidirectional else torch.cat([h_n[-2], h_n[-1]], dim=1)
#         z = torch.cat([x, h], dim=1)
#         return self.mlp(z).squeeze(1)
    
# ========================== no pack =======================
class TabularSeqTokenLSTM(nn.Module):
    def __init__(self, d_features, vocab_size, oov_idx, pad_idx,
                 d_embed=32, hidden=64, num_layers=1, bidir=False, dropout=0.2,
                 mlp_hidden=(256,128)):
        super().__init__()
        self.bn_x = nn.LayerNorm(d_features)
        self.emb  = nn.Embedding(vocab_size + 2, d_embed, padding_idx=pad_idx)
        self.lstm = nn.LSTM(d_embed, hidden, num_layers=num_layers, batch_first=True,
                            bidirectional=bidir, dropout=(dropout if num_layers>1 else 0.0))
        lstm_out = hidden * (2 if bidir else 1)
        in_dim = d_features + lstm_out

        layers = []
        for h in mlp_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_feats, x_seq, seq_lengths):
        x = self.bn_x(x_feats)          # (B, F)
        e = self.emb(x_seq)             # (B, L, D)
        out, _ = self.lstm(e)           # (B, L, H*)  H* = hidden*(1 or 2)

        # ---- gather last valid timestep (avoid pad drift)
        # seq_lengths를 out과 같은 디바이스/long으로
        seq_lengths = seq_lengths.to(out.device, non_blocking=True).long().clamp_min(1)
        last_idx = (seq_lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))  # (B,1,H*)
        if self.lstm.bidirectional:
            H = out.size(2) // 2
            f_last = out.gather(1, last_idx[:, :, :H]).squeeze(1)  # forward last
            b_first = out[:, 0, H:]                                # backward first
            h = torch.cat([f_last, b_first], dim=1)
        else:
            h = out.gather(1, last_idx).squeeze(1)                 # (B, H)

        z = torch.cat([x, h], dim=1)    # (B, F+H*)
        return self.mlp(z).squeeze(1)

# ========================= Metrics =========================
def weighted_logloss_5050(y_true, y_prob, eps=1e-9):
    y_true = np.asarray(y_true, dtype=np.int64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1-eps)
    pos = (y_true == 1); neg = ~pos
    if pos.sum()==0 or neg.sum()==0:
        return float(-(y_true*np.log(p) + (1-y_true)*np.log(1-p)).mean())
    loss_pos = -np.log(p[pos]).mean()
    loss_neg = -np.log(1-p[neg]).mean()
    return 0.5*loss_pos + 0.5*loss_neg

def composite_score(ap, wll):
    return 0.5*ap + 0.5*(1.0/(1.0 + wll))

def _clip(p): return np.clip(p, 1e-6, 1-1e-6)

def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.nan_to_num(_clip(np.asarray(y_prob, dtype=np.float64)), nan=0.5)
    ap  = average_precision_score(y_true, y_prob) if (y_true.min()<y_true.max()) else 0.0
    wll = weighted_logloss_5050(y_true, y_prob)
    return ap, wll, composite_score(ap, wll)

# ========================= Main =========================
def main():
    print_cuda_info()
    if device == "cpu":
        print("[WARN] CUDA not available. 현재 CPU로 실행됩니다.")

    OUT_DIR = CFG["OUT_DIR"]
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(CFG["SEED"])
    OUT_DIR = CFG["OUT_DIR"]; os.makedirs(OUT_DIR, exist_ok=True)

    target_col, seq_col, ID_COL = "clicked", "seq", "ID"

    # ---- Load base
    train_all = pd.read_parquet(CFG["TRAIN_PATH"], engine="pyarrow")
    test_df   = pd.read_parquet(CFG["TEST_PATH"],  engine="pyarrow")
    print("Train shape:", train_all.shape, "Test shape:", test_df.shape)
    
    if SMOKE:
        target_col = "clicked"
        pos_all = train_all[train_all[target_col]==1]
        neg_need = min(len(pos_all), len(train_all) - len(pos_all))
        neg = train_all[train_all[target_col]==0].sample(n=neg_need, random_state=CFG["SEED"])
        small = pd.concat([pos_all, neg]).sample(frac=1, random_state=CFG["SEED"]).reset_index(drop=True)
        MAX_TRAIN = 800_000
        if len(small) > MAX_TRAIN:
            small = small.sample(n=MAX_TRAIN, random_state=CFG["SEED"]).reset_index(drop=True)
        train_df = small
        del small, pos_all, neg
    else:
        # 기존 로직 유지: pos all + neg 2배
        pos = train_all[train_all[target_col]==1]
        neg = train_all[train_all[target_col]==0].sample(n=len(pos)*2, random_state=CFG["SEED"])
        train_df = pd.concat([pos,neg]).sample(frac=1, random_state=CFG["SEED"]).reset_index(drop=True)
        del pos, neg
    del train_all; free_mem()
    

    # # 다운샘플(양성 전부 + 음성 1배: 스모크)
    # pos = train_all[train_all[target_col]==1]
    # neg = train_all[train_all[target_col]==0].sample(n=len(pos)*2, random_state=CFG["SEED"])
    # train_df = pd.concat([pos,neg]).sample(frac=1, random_state=CFG["SEED"]).reset_index(drop=True)
    # del train_all, pos, neg; free_mem()

    # 탭 피처
    FEATURE_EXCLUDE = {target_col, seq_col, ID_COL}
    feature_cols = [c for c in train_df.columns if c not in FEATURE_EXCLUDE and c in test_df.columns]
    print("Num features:", len(feature_cols))

    if ID_COL in train_df.columns:
        train_df[ID_COL] = train_df[ID_COL].astype(str)
    else:
        # train에는 ID가 없을 수 있으니 검증 트래킹용 가짜 ID 생성
        train_df[ID_COL] = pd.RangeIndex(len(train_df)).astype(str)

    if ID_COL in test_df.columns:
        test_df[ID_COL] = test_df[ID_COL].astype(str)
    else:
        # test에도 없으면 제출 정렬용 가짜 ID 생성 (sample_sub에 ID가 없다면 순서대로 제출)
        test_df[ID_COL] = pd.RangeIndex(len(test_df)).astype(str)

    # ---- 결측/Inf 처리 (train/test 공통)
    train_df = clean_numeric(train_df, feature_cols)
    test_df  = clean_numeric(test_df,  feature_cols)

    # ---- Split
    tr_df, va_df = train_test_split(
        train_df, test_size=0.2, random_state=CFG["SEED"], shuffle=True, stratify=train_df[target_col]
    )

    va_ids = va_df[ID_COL].astype(str).values
    collate_train = CollateTokens(
        has_target=True,
        max_len=CFG["SEQ_MAX_LEN"],
        oov_idx=CFG["OOV_IDX"],
        pad_idx=CFG["PAD_IDX"],
        vocab_size=CFG["VOCAB_SIZE"],
    )
    collate_infer = CollateTokens(
        has_target=False,
        max_len=CFG["SEQ_MAX_LEN"],
        oov_idx=CFG["OOV_IDX"],
        pad_idx=CFG["PAD_IDX"],
        vocab_size=CFG["VOCAB_SIZE"],
    )
    pin = (device == "cuda")
    train_loader = DataLoader(
        ClickDataset(tr_df, feature_cols, seq_col, target_col, has_target=True),
        batch_size=CFG["BATCH_SIZE"], shuffle=True,
        num_workers=CFG["NUM_WORKERS"], pin_memory=CFG["PIN_MEMORY"],
        collate_fn=collate_train, drop_last=False,
        persistent_workers=(CFG["NUM_WORKERS"]>0),  # ← 선택: epoch 사이 재사용
    )

    val_loader = DataLoader(
        ClickDataset(va_df, feature_cols, seq_col, target_col, has_target=True),
        batch_size=CFG["BATCH_SIZE"], shuffle=False,
        num_workers=CFG["NUM_WORKERS"], pin_memory=CFG["PIN_MEMORY"],
        collate_fn=collate_train, drop_last=False,
        persistent_workers=(CFG["NUM_WORKERS"]>0),
    )

    # ---- Model
    model = TabularSeqTokenLSTM(
        d_features=len(feature_cols),
        vocab_size=CFG["VOCAB_SIZE"], oov_idx=CFG["OOV_IDX"], pad_idx=CFG["PAD_IDX"],
        d_embed=CFG["D_EMBED"], hidden=CFG["LSTM_HIDDEN"],
        num_layers=CFG["LSTM_LAYERS"], bidir=CFG["BIDIRECTIONAL"], dropout=CFG["DROPOUT"],
        mlp_hidden=CFG["MLP_HIDDEN"]
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=CFG["LR"])
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=CFG["USE_AMP"])

    # ---- Early Stopper (LSTM)
    if CFG["ES_MONITOR"] == "neg_valid_loss":
        es_mode = "min"
    else:
        es_mode = "max"
    es = EarlyStopper(mode=es_mode, patience=CFG["ES_PATIENCE"], min_delta=CFG["ES_MIN_DELTA"]) if CFG["ES_ENABLE"] else None

    # ---- Train (AMP + micro-batch + 안정 가드)
    for epoch in range(1, CFG["EPOCHS"]+1):
        model.train(); tr_loss=0.0; n_tr=0
        for xs, seqs, lens, ys in tqdm(train_loader, desc=f"Train {epoch}"):
            B = ys.size(0)
            for i in range(0, B, CFG["MICRO_BATCH"]):
                xs_mb  = xs[i:i+CFG["MICRO_BATCH"]].to(device, non_blocking=True)
                seq_mb = seqs[i:i+CFG["MICRO_BATCH"]].to(device, non_blocking=True)
                y_mb   = ys[i:i+CFG["MICRO_BATCH"]].to(device, non_blocking=True)
                lens_mb = lens[i:i+CFG["MICRO_BATCH"]]

                opt.zero_grad(set_to_none=True)
                with autocast(enabled=(CFG["USE_AMP"] and device=="cuda")):
                    logits = model(xs_mb, seq_mb, lens_mb)
                    if not torch.isfinite(logits).all():
                        continue
                    loss = criterion(logits, y_mb) * (len(y_mb)/B)
                if not torch.isfinite(loss):
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()

                tr_loss += float(loss.item()) * B; n_tr += len(y_mb)
        tr_loss /= max(1, n_tr)

        # ---- Valid + metrics
        model.eval(); va_loss=0.0; n_va=0
        y_true=[]; y_prob=[]
        with torch.no_grad():
            for xs, seqs, lens, ys in tqdm(val_loader, desc=f"Valid {epoch}"):
                B = ys.size(0)
                for i in range(0, B, CFG["MICRO_BATCH"]):
                    xs_mb  = xs[i:i+CFG["MICRO_BATCH"]].to(device)
                    seq_mb = seqs[i:i+CFG["MICRO_BATCH"]].to(device)
                    y_mb   = ys[i:i+CFG["MICRO_BATCH"]].to(device)
                    lens_mb = lens[i:i+CFG["MICRO_BATCH"]]
                    with autocast(enabled=(CFG["USE_AMP"] and device=="cuda")):
                        logits = model(xs_mb, seq_mb, lens_mb)
                        if not torch.isfinite(logits).all():
                            continue
                        loss = criterion(logits, y_mb)
                        probs = torch.sigmoid(logits)
                        probs = torch.nan_to_num(probs, nan=0.5, posinf=1-1e-6, neginf=1e-6).clamp_(1e-6, 1-1e-6)
                    va_loss += float(loss.item()) * len(y_mb); n_va += len(y_mb)
                    y_true.append(y_mb.cpu().numpy())
                    y_prob.append(probs.cpu().numpy())
        va_loss /= max(1, n_va)
        y_true = np.concatenate(y_true) if y_true else np.array([0,1], dtype=np.int64)
        y_prob = np.concatenate(y_prob).astype(np.float64) if y_prob else np.array([0.5,0.5], dtype=np.float64)
        y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1-1e-6, neginf=1e-6)
        y_prob = np.clip(y_prob, 1e-6, 1-1e-6)
        ap = average_precision_score(y_true, y_prob) if (y_true.min()<y_true.max()) else 0.0
        wll = weighted_logloss_5050(y_true, y_prob)
        score = composite_score(ap, wll)
        print(f"[Epoch {epoch}] train {tr_loss:.4f} | valid {va_loss:.4f} | AP {ap:.5f} | WLL {wll:.5f} | Score {score:.5f}")

        # ---- Early Stopping check
        if es is not None:
            if CFG["ES_MONITOR"] == "score":
                metric = score
            elif CFG["ES_MONITOR"] == "ap":
                metric = ap
            elif CFG["ES_MONITOR"] == "neg_valid_loss":
                metric = va_loss
            stop = es.step(metric, model, epoch)
            if stop:
                print(f"[EarlyStopping] stop at epoch={epoch} | best_epoch={es.best_epoch} | best={es.best_metric:.6f}")
                break

    # ---- LSTM 최종 검증 성적 프린트
    y_true_lstm, y_prob_lstm = eval_lstm_on_loader(model, val_loader, use_amp=CFG["USE_AMP"])
    ap_lstm, wll_lstm, sc_lstm = compute_metrics(y_true_lstm, y_prob_lstm)
    print(f"[LSTM] Final Valid | AP {ap_lstm:.5f} | WLL {wll_lstm:.5f} | SCORE {sc_lstm:.5f}")
    # 검증 예측 DF (앙상블 검증용 교차)
    lstm_val_df = pd.DataFrame({ "ID": va_ids[:len(y_prob_lstm)], "y": y_true_lstm, "p_lstm": y_prob_lstm })

    # ---- Restore best model (if ES)
    if es is not None and es.best_state is not None:
        model.load_state_dict(es.best_state)
        print(f"[EarlyStopping] restored best epoch={es.best_epoch} | metric={es.best_metric:.6f}")

    # ---- Inference (LSTM)
    test_ld = DataLoader(
        ClickDataset(test_df, feature_cols, seq_col, has_target=False),
        batch_size=CFG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CFG['TEST_NUM_WORKERS'],   # ← 핵심
        pin_memory=CFG["PIN_MEMORY"],   # GPU면 True 그대로 OK
        collate_fn=collate_infer,
        drop_last=False,
        persistent_workers=False        # 워커 없으니 False (명시)
    )
    model.eval(); outs=[]
    with torch.no_grad():
        for xs, seqs, lens in tqdm(test_ld, desc="LSTM Inference"):
            B = xs.size(0)
            for i in range(0, B, CFG["MICRO_BATCH"]):
                xs_mb  = xs[i:i+CFG["MICRO_BATCH"]].to(device)
                seq_mb = seqs[i:i+CFG["MICRO_BATCH"]].to(device)
                lens_mb = lens[i:i+CFG["MICRO_BATCH"]]
                with autocast(enabled=(CFG["USE_AMP"] and device=="cuda")):
                    probs = torch.sigmoid(model(xs_mb, seq_mb, lens_mb))
                    probs = torch.nan_to_num(probs, nan=0.5, posinf=1-1e-6, neginf=1e-6).clamp_(1-1e-6, 1e-6).flip(0) if False else probs
                    probs = torch.nan_to_num(probs, nan=0.5, posinf=1-1e-6, neginf=1e-6).clamp_(1e-6, 1-1e-6)
                outs.append(probs.cpu())
    pred_lstm = torch.cat(outs).numpy()

    sub = pd.read_csv(CFG["SAMPLE_SUB"])
    has_id = "ID" in sub.columns
    if has_id:
        sub["ID"] = sub["ID"].astype(str)

    ids = test_df[ID_COL].values

    lstm_df = pd.DataFrame({"ID": ids, "p_lstm": pred_lstm})
    del train_df, tr_df, va_df, train_loader, val_loader, test_ld, model, opt, scaler
    free_mem()

    # ========================= XGBoost (GPU, Stratified K-Fold + ES) =========================
    import xgboost as xgb

    train_en = pd.read_parquet(CFG["META_TRAIN"], engine="pyarrow").drop(['seq'], axis=1)
    test_en  = pd.read_parquet(CFG["META_TEST"],  engine="pyarrow").drop(['seq'], axis=1)

    # ID 보장 (없으면 가짜 ID)
    if ID_COL in train_en.columns: train_en[ID_COL] = train_en[ID_COL].astype(str)
    else:                          train_en[ID_COL] = pd.RangeIndex(len(train_en)).astype(str)
    if ID_COL in test_en.columns:  test_en[ID_COL]  = test_en[ID_COL].astype(str)
    else:                          test_en[ID_COL]  = pd.RangeIndex(len(test_en)).astype(str)

    if SMOKE:
        pos = train_en[train_en[target_col]==1]
        neg_need = min(len(pos), (len(train_en)-len(pos)))
        neg = train_en[train_en[target_col]==0].sample(n=neg_need, random_state=CFG["SEED"])
        small = pd.concat([pos, neg]).sample(frac=1, random_state=CFG["SEED"]).reset_index(drop=True)
        MAX_META = 400_000
        if len(small) > MAX_META:
            small = small.sample(n=MAX_META, random_state=CFG["SEED"]).reset_index(drop=True)
        train_en = small
        del small, pos, neg; free_mem()
    else:
        # 기존 로직 유지(양성 전부 + 음성 2배)
        pos = train_en[train_en[target_col] == 1]
        neg = train_en[train_en[target_col] == 0].sample(n=len(pos)*2, random_state=CFG["SEED"])
        train_en = pd.concat([pos, neg]).sample(frac=1, random_state=CFG["SEED"]).reset_index(drop=True)
        del pos, neg; free_mem()

    # # 스모크용 다운샘플 (원하면 비율 조정)
    # pos = train_en[train_en[target_col] == 1]
    # neg = train_en[train_en[target_col] == 0].sample(n=len(pos)*2, random_state=CFG["SEED"])
    # train_en = pd.concat([pos, neg]).sample(frac=1, random_state=CFG["SEED"]).reset_index(drop=True)
    # del pos, neg; free_mem()

    # 피처 선택 및 정리
    xgb_exclude = {target_col, ID_COL}
    xgb_cols = [c for c in train_en.columns if c not in xgb_exclude and c in test_en.columns]
    for df_ in (train_en, test_en):
        df_[xgb_cols] = df_[xgb_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)

    X = train_en[xgb_cols].values
    y = train_en[target_col].astype(np.int32).values
    ids_en = train_en[ID_COL].values  # str

    dtest = xgb.DMatrix(test_en[xgb_cols].values, nthread=4)

    # 공통 파라미터
    params_base = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": CFG["SEED"],
        "device": "cuda",
        "tree_method": "hist"
    }

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=CFG["XGB_NFOLDS"], shuffle=True, random_state=CFG["SEED"])
    oof_pred = np.zeros(len(train_en), dtype=np.float32)
    test_pred_folds = []
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        id_va = ids_en[va_idx]

        # 자동 scale_pos_weight (neg/pos). 극단 케이스 방어.
        pos_cnt = max(1, int((y_tr == 1).sum()))
        neg_cnt = max(1, int((y_tr == 0).sum()))
        spw = float(neg_cnt / pos_cnt)

        params = dict(params_base)
        params["scale_pos_weight"] = spw

        dtrain = xgb.DMatrix(X_tr, label=y_tr, nthread=4)
        dvalid = xgb.DMatrix(X_va, label=y_va, nthread=4)

        print(f"[XGB][Fold {fold}/{CFG['XGB_NFOLDS']}] scale_pos_weight={spw:.4f} | tr={len(tr_idx)} va={len(va_idx)}")
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=CFG["XGB_NUM_BOOST_ROUND"],
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=CFG["XGB_ES_ROUNDS"],
            verbose_eval=100
        )

        best_iter = bst.best_iteration + 1 if bst.best_iteration is not None else CFG["XGB_NUM_BOOST_ROUND"]
        pred_va  = bst.predict(dvalid, iteration_range=(0, best_iter))
        pred_te  = bst.predict(dtest,  iteration_range=(0, best_iter))

        # fold 성적
        ap_f, wll_f, sc_f = compute_metrics(y_va, pred_va)
        print(f"[XGB][Fold {fold}] AP {ap_f:.5f} | WLL {wll_f:.5f} | SCORE {sc_f:.5f} | best_iter={best_iter}")
        fold_metrics.append((ap_f, wll_f, sc_f, best_iter))

        # 저장
        oof_pred[va_idx] = pred_va.astype(np.float32)
        test_pred_folds.append(pred_te.astype(np.float32))

        # 메모리 정리
        del dtrain, dvalid, bst
        free_mem()

    # OOF 성적
    ap_xgb, wll_xgb, sc_xgb = compute_metrics(y, oof_pred)
    print(f"[XGB][OOF] AP {ap_xgb:.5f} | WLL {wll_xgb:.5f} | SCORE {sc_xgb:.5f}")
    for i,(ap_f,wll_f,sc_f,biter) in enumerate(fold_metrics, 1):
        print(f"  - Fold{i}: AP {ap_f:.5f} | WLL {wll_f:.5f} | SCORE {sc_f:.5f} | best_iter={biter}")

    # 검증/테스트 DF 생성
    xgb_val_df = pd.DataFrame({ "ID": ids_en, "y": y, "p_xgb": oof_pred })
    test_pred = np.mean(test_pred_folds, axis=0).astype(np.float32)
    xgb_df = pd.DataFrame({ "ID": test_en[ID_COL].values, "p_xgb": test_pred })
    xgb_df["ID"] = xgb_df["ID"].astype(str)

    # 정리
    del train_en, test_en, X, y, ids_en, dtest, test_pred_folds, oof_pred
    free_mem()

    # ========================= Ensemble (logit avg) =========================
    def _clip(p): return np.clip(p, 1e-6, 1-1e-6)
    def _logit(p): p=_clip(p); return np.log(p/(1-p))
    def _sigm(z): return 1/(1+np.exp(-z))

    sub = pd.read_csv(CFG["SAMPLE_SUB"])
    if "ID" in sub.columns:
        sub["ID"] = sub["ID"].astype(str)

    merged = sub[["ID"]].merge(lstm_df, on="ID", how="left").merge(xgb_df, on="ID", how="left")
    p1 = merged["p_lstm"].values.astype(np.float64)
    p2 = merged["p_xgb"].values.astype(np.float64)
    p1 = np.nan_to_num(_clip(p1), nan=0.5)
    p2 = np.nan_to_num(_clip(p2), nan=0.5)

    W = 0.5
    p_blend = _sigm(W*_logit(p1) + (1-W)*_logit(p2)).astype(np.float32)
    submit = pd.read_csv(CFG["SAMPLE_SUB"])
    submit["clicked"] = p_blend
    
    # ================= Ensemble on VALID (ID 교집합) =================
    # 두 검증 셋(베이스 LSTM vs 메타 XGB)의 ID 교집합으로 성능 측정
    ens_valid = lstm_val_df.merge(xgb_val_df, on=["ID"], how="inner", suffixes=("_lstm","_xgb"))
    if len(ens_valid) > 0:
        yv = ens_valid["y_lstm"].values if "y_lstm" in ens_valid else ens_valid["y"].values
        p1 = np.nan_to_num(_clip(ens_valid["p_lstm"].values.astype(np.float64)), nan=0.5)
        p2 = np.nan_to_num(_clip(ens_valid["p_xgb"].values.astype(np.float64)),  nan=0.5)
        W = 0.5  # 같은 가중치
        p_blend_v = _sigm(W*_logit(p1) + (1-W)*_logit(p2))
        ap_e, wll_e, sc_e = compute_metrics(yv, p_blend_v)
        print(f"[ENS ] Final Valid (|∩|={len(ens_valid)}) | AP {ap_e:.5f} | WLL {wll_e:.5f} | SCORE {sc_e:.5f}")
    else:
        print("[ENS ] Final Valid | 교집합이 없어 앙상블 검증 점수를 계산할 수 없습니다.")
    
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ens_score = sc_e if ('sc_e' in locals()) else np.nan
    score_str = (f"{ens_score:.5f}".replace('.', 'p')) if np.isfinite(ens_score) else "NA"
    filename = f"{version}_{SEED}_{score_str}_submit_{ts}.csv"

    out_path = OUT_DIR / filename  # ← Path 객체끼리 결합 OK
    submit.to_csv(out_path, index=False)
    print(f"[SUB ] saved to {out_path} (ENS score: {ens_score if np.isfinite(ens_score) else 'NA'})")

    print(f"[SUM ] LSTM(AP/WLL/SC)={ap_lstm:.5f}/{wll_lstm:.5f}/{sc_lstm:.5f} | "
        f"XGB(AP/WLL/SC)={ap_xgb:.5f}/{wll_xgb:.5f}/{sc_xgb:.5f} | "
        f"ENS(AP/WLL/SC)={'{:.5f}'.format(ap_e) if len(ens_valid)>0 else 'NA'}/"
        f"{'{:.5f}'.format(wll_e) if len(ens_valid)>0 else 'NA'}/"
        f"{'{:.5f}'.format(sc_e) if len(ens_valid)>0 else 'NA'}")
    
    # 로그 파일 경로 수정 (Path 연산자 사용)
    with open((SAVE_DIR / "log.txt"), "a", encoding="utf-8") as f:
        f.write(f"VER: {version} | SEED: {SEED} \n")
        f.write(f"LSTM(AP/WLL/SC)={ap_lstm:.5f}/{wll_lstm:.5f}/{sc_lstm:.5f}\n")
        f.write(f"XGB(AP/WLL/SC)={ap_xgb:.5f}/{wll_xgb:.5f}/{sc_xgb:.5f}\n")
        f.write(f"ENS(AP)={'{:.5f}'.format(ap_e) if 'ap_e' in locals() else 'NA'}\n")
        f.write(f"ENS(WLL)={'{:.5f}'.format(wll_e) if 'wll_e' in locals() else 'NA'}\n")
        f.write(f"ENS(SC)={'{:.5f}'.format(sc_e) if 'sc_e' in locals() else 'NA'}\n")
        f.write("="*40 + "\n")

    with open((LOG_DIR / f"{version}_log.txt"), "a", encoding="utf-8") as f:
        f.write(f"VER: {version} | SEED: {SEED} \n")
        f.write(f"LSTM(AP/WLL/SC)={ap_lstm:.5f}/{wll_lstm:.5f}/{sc_lstm:.5f}\n")
        f.write(f"XGB(AP/WLL/SC)={ap_xgb:.5f}/{wll_xgb:.5f}/{sc_xgb:.5f}\n")
        f.write(f"ENS(AP)={'{:.5f}'.format(ap_e) if 'ap_e' in locals() else 'NA'}\n")
        f.write(f"ENS(WLL)={'{:.5f}'.format(wll_e) if 'wll_e' in locals() else 'NA'}\n")
        f.write(f"ENS(SC)={'{:.5f}'.format(sc_e) if 'sc_e' in locals() else 'NA'}\n")
        f.write("="*40 + "\n")

if __name__ == "__main__":
    main()
