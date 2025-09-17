# -*- coding: utf-8 -*-
import os, random, hashlib, math
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
print(f"[CUDA available]: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[CUDA device]: {torch.cuda.get_device_name(0)}")
    print(f"[PyTorch CUDA]: {torch.version.cuda}")
# ======================
# Configs
# ======================
CFG = {
    'BATCH_SIZE': 2048,         # Torch mini-batch size
    'EPOCHS': 1,
    'LEARNING_RATE': 1e-3,
    'SEED': 42,

    # Arrow batch rows (한번에 메모리로 올릴 레코드 수)
    'ARROW_BATCH_ROWS': 50_000,

    # 검증 분할: hash(ID) % N_FOLDS == VAL_FOLD → valid
    'N_FOLDS': 5,
    'VAL_FOLD': 0,

    # 다운샘플링 목표 비율 (pos : neg ≈ 1 : NEG_PER_POS)
    'NEG_PER_POS': 2.0,

    # 시퀀스 전처리
    'SEQ_L_MAX': 256,           # 최대 길이 (긴 건 뒤에서부터 자르기)
    'SEQ_CLIP_ABS': 30.0,       # asinh 후 clip 범위 [-50, 50]
    'USE_SEQ_STATS': True,      # seq_len / seq_mean / seq_std 추가

    # 얼리 스톱
    'EARLY_STOP_PATIENCE': 2,   # 기준F: Val AP
    'PIN_MEMORY': True,
    'NUM_WORKERS': 0,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['SEED'])

# ======================
# Paths / Columns
# ======================
TRAIN_PATH = "./Toss/train.parquet"
TEST_PATH  = "./Toss/test.parquet"
SUB_PATH   = "./Toss/submit_ver1.csv"

target_col = "clicked"
seq_col = "seq"
id_col = "ID"  # 없으면 자동 대체

# ======================
# Feature columns (schema 기반)
# ======================
schema = pq.read_schema(TRAIN_PATH)
all_cols = schema.names

# ⬇️ ID 없으면 None으로 (fold 해시는 row counter로 대체)
id_col = "ID" if "ID" in all_cols else None
print("[schema] has ID?:", id_col is not None)

EXCLUDE = {target_col, seq_col}
if id_col: EXCLUDE.add(id_col)
feature_cols = [c for c in all_cols if c not in EXCLUDE]

print("[schema] num_features:", len(feature_cols))
print("[schema] first 10 features:", feature_cols[:10])

import math
from tqdm import tqdm

def _estimate_total_batches_for_scan(parquet_path: str, batch_rows: int) -> Optional[int]:
    """Parquet 전체 행 수로 배치 수를 대략 추정 (단일 파일일 때 가장 정확)."""
    try:
        pf = pq.ParquetFile(parquet_path)
        n_rows = pf.metadata.num_rows
        return int(math.ceil(n_rows / batch_rows))
    except Exception:
        # 디렉토리/파티션 구조 등으로 실패하면 None (unknown total)
        return None

def count_pos_neg(parquet_path: str, target_col: str) -> Tuple[int, int]:
    d = ds.dataset(parquet_path, format="parquet")
    BATCH = 500_000
    total_batches = _estimate_total_batches_for_scan(parquet_path, BATCH)

    pos, neg = 0, 0
    iterator = d.to_batches(columns=[target_col], batch_size=BATCH)
    if total_batches is not None:
        iterator = tqdm(iterator, total=total_batches, desc="Counting pos/neg")

    for b in iterator:
        arr = b.column(target_col).to_numpy(zero_copy_only=False)
        arr = np.nan_to_num(arr, nan=0).astype(np.int8, copy=False)
        pos += int((arr == 1).sum())
        neg += int((arr == 0).sum())
    return pos, neg

pos_cnt, neg_cnt = count_pos_neg(TRAIN_PATH, target_col)
neg_keep_prob = min(1.0, (CFG['NEG_PER_POS'] * pos_cnt) / max(1, neg_cnt))
print(f"[counts] pos={pos_cnt:,}  neg={neg_cnt:,}  -> keep_neg_prob={neg_keep_prob:.4f}")

# ======================
# Utils
# ======================
def hash_fold(val, n_folds=5):
    if val is None:
        return 1  # train 쪽으로
    h = hashlib.md5(str(val).encode('utf-8')).hexdigest()
    return int(h[:8], 16) % n_folds

def safe_clip_num(arr: np.ndarray, clip_abs: float = 1e6):
    # NaN/Inf → 0, 안전 clip
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_abs is not None:
        np.clip(arr, -clip_abs, clip_abs, out=arr)
    return arr

def seq_transform_asinh(arr: np.ndarray, clip_abs: float = 50.0) -> np.ndarray:
    # sign-preserving log, 그 다음 clip
    arr = np.arcsinh(arr)
    if clip_abs is not None:
        np.clip(arr, -clip_abs, clip_abs, out=arr)
    return arr

def wll_5050(y_true: np.ndarray, y_prob: np.ndarray, eps=1e-7) -> float:
    # float64로 올려서 언더플로 방지 + log1p(-p) 사용
    p = np.asarray(y_prob, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    p = np.clip(p, eps, 1.0 - eps)

    y = np.asarray(y_true, dtype=np.int8)
    pos = (y == 1)
    neg = ~pos

    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        # 원래 분포가 한쪽만 있으면 일반 logloss로 폴백 (안정형)
        return float(
            -(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()
        )

    loss_pos = -np.log(p[pos]).mean()
    loss_neg = -np.log1p(-p[neg]).mean()
    return float(0.5 * (loss_pos + loss_neg))

def local_score(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    ap = float(average_precision_score(y_true, y_prob))
    wll = wll_5050(y_true, y_prob)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    score = 0.5 * ap + 0.5 * (1.0 / (1.0 + wll))
    return dict(ap=ap, auc=auc, wll=wll, score=score)

# ======================
# IterableDataset (Streaming)
# ======================
class ParquetStreamDataset(IterableDataset):
    def __init__(self,
                 parquet_path: str,
                 feature_cols: List[str],
                 seq_col: str,
                 target_col: Optional[str] = None,
                 id_col: Optional[str] = None,
                 split: Optional[str] = None,    # "train"/"valid"/None(test)
                 n_folds: int = 5,
                 val_fold: int = 0,
                 neg_keep_prob: float = 1.0,
                 arrow_batch_rows: int = 200_000,
                 seed: int = 42,
                 use_seq_stats: bool = True,
                 seq_l_max: int = 512,
                 seq_clip_abs: float = 50.0,
                 clip_abs_feat: float = 1e6,
                 ):
        super().__init__()
        self.path = parquet_path
        self.feature_cols = feature_cols
        self.seq_col = seq_col
        self.target_col = target_col
        self.id_col = id_col
        self.split = split
        self.n_folds = n_folds
        self.val_fold = val_fold
        self.neg_keep_prob = float(neg_keep_prob)
        self.arrow_batch_rows = arrow_batch_rows
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.use_seq_stats = use_seq_stats
        self.seq_l_max = seq_l_max
        self.seq_clip_abs = seq_clip_abs
        self.clip_abs_feat = clip_abs_feat

        available = set(pq.read_schema(self.path).names)

        cols = [c for c in self.feature_cols if c in available]
        if self.seq_col in available: cols.append(self.seq_col)
        if self.target_col and self.target_col in available: cols.append(self.target_col)
        if id_col and (id_col in available):
            self.id_col = id_col
            cols.append(id_col)
        else:
            self.id_col = None  # ← ID가 없으면 해시 폴드에서 row counter 사용

        self.read_columns = cols

    def parse_seq(self, s: str) -> np.ndarray:
        if not isinstance(s, str):
            s = "" if s is None else str(s)
        if not s:
            return np.array([0.0], dtype=np.float32)
        arr = np.fromstring(s, sep=",", dtype=np.float32)
        if arr.size == 0:
            arr = np.array([0.0], dtype=np.float32)
        # 뒤에서부터 L_MAX만 사용 (최근 구간 보존)
        if self.seq_l_max and arr.size > self.seq_l_max:
            arr = arr[-self.seq_l_max:]
        # 안전 전처리: NaN/Inf→0
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # sign-preserving log
        arr = seq_transform_asinh(arr, clip_abs=self.seq_clip_abs)
        return arr

    def _fold_of(self, val, counter):
        if val is not None:
            return hash_fold(val, self.n_folds)
        else:
            # ID 없으면 순차 카운터로 의사 해시
            return counter % self.n_folds

    # __iter__ 내부를 아래처럼 교체
    # ParquetStreamDataset.__iter__ 안의 변환/생성 부분을 아래처럼 바꿔주세요.
    def __iter__(self):
        import gc
        dataset = ds.dataset(self.path, format="parquet")
        global_row_counter = 0

        for batch in dataset.to_batches(columns=self.read_columns, batch_size=self.arrow_batch_rows):
            # --- 최대한 복사 줄이기 ---
            pdf = batch.to_pandas(types_mapper=pd.ArrowDtype)  # 판다스에서 arrow dtype으로 (복사↓)
            ids = pdf[self.id_col].array if (self.id_col in pdf.columns) else None

            ys = None
            if self.target_col and self.target_col in pdf.columns:
                ys = pd.to_numeric(pdf[self.target_col], errors="coerce").fillna(0).astype("int8").to_numpy(copy=False)

            # split + 다운샘플 마스크 계산
            if self.split is not None:
                if ids is not None:
                    # ArrowExtensionArray라서 .to_numpy()로 뽑아 문자열/정수를 얻음
                    id_vals = ids.to_numpy()
                    folds = np.fromiter(
                        (self._fold_of(id_vals[i], global_row_counter + i) for i in range(len(pdf))),
                        dtype=np.int64, count=len(pdf)
                    )
                else:
                    folds = np.fromiter(
                        (self._fold_of(None, global_row_counter + i) for i in range(len(pdf))),
                        dtype=np.int64, count=len(pdf)
                    )
                split_mask = (folds == self.val_fold) if (self.split == "valid") else (folds != self.val_fold)
            else:
                split_mask = np.ones(len(pdf), dtype=bool)

            if ys is not None:
                keep_neg = (np.random.RandomState(self.seed).rand(len(pdf)) < self.neg_keep_prob)
                ds_mask = (ys == 1) | ((ys == 0) & keep_neg)
            else:
                ds_mask = np.ones(len(pdf), dtype=bool)

            mask = split_mask & ds_mask
            if not mask.any():
                global_row_counter += len(pdf)
                del pdf
                gc.collect()
                continue

            # ---- 수치 피처만 바로 numpy로 (열 단위 캐스팅, 복사 최소화) ----
            X_cols = []
            for c in self.feature_cols:
                col = pd.to_numeric(pdf.loc[mask, c], errors="coerce")  # object라도 숫자로
                X_cols.append(col.to_numpy(dtype=np.float32, copy=False))
            # (n_kept, d) 형태로 뷰 결합
            X = np.vstack(X_cols).T
            # 안전 clip
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            if self.clip_abs_feat is not None:
                np.clip(X, -self.clip_abs_feat, self.clip_abs_feat, out=X)

            # ---- 행 단위로 바로 seq 파싱 & (옵션) 통계 3개 만들고 yield ----
            seq_series = pdf.loc[mask, self.seq_col]
            Y = (pdf.loc[mask, self.target_col].astype("float32").to_numpy(copy=False) if ys is not None else None)

            n_rows = X.shape[0]
            for i in range(n_rows):
                s = seq_series.iloc[i]
                arr = self.parse_seq(s)  # np.float32, 길이 ≤ L_MAX

                if self.use_seq_stats:
                    # len / mean / std 를 즉석에서
                    l_ = float(arr.size)
                    m_ = float(arr.mean()) if arr.size else 0.0
                    sd = float(arr.std())  if arr.size else 0.0
                    xrow = np.concatenate([X[i], np.array([l_, m_, sd], dtype=np.float32)], axis=0)
                else:
                    xrow = X[i]

                x_t = torch.from_numpy(xrow)
                seq_t = torch.from_numpy(arr)

                if Y is not None:
                    y_t = torch.tensor(Y[i], dtype=torch.float32)
                    yield x_t, seq_t, y_t
                else:
                    yield x_t, seq_t

            # 배치 메모리 즉시 반환
            del pdf, X, X_cols, seq_series
            gc.collect()
            global_row_counter += len(batch)

# ======================
# Collate
# ======================
def collate_fn_train(batch):
    xs, seqs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return xs, seqs_padded, seq_lengths, ys

def collate_fn_infer(batch):
    # 배치 원소가 (x, seq) 또는 (x, seq, y) 모두 들어와도 동작
    first = batch[0]
    if isinstance(first, (list, tuple)) and len(first) == 3:
        xs, seqs, _ = zip(*batch)   # y는 무시
    else:
        xs, seqs = zip(*batch)

    xs = torch.stack(xs)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return xs, seqs_padded, seq_lengths

# ======================
# Model
# ======================
class TabularSeqModel(nn.Module):
    def __init__(self, d_features, lstm_hidden=64, hidden_units=[256, 128], dropout=0.2):
        super().__init__()
        self.bn_x = nn.BatchNorm1d(d_features)
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True)
        input_dim = d_features + lstm_hidden
        layers = []
        for h in hidden_units:
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = h
        layers += [nn.Linear(input_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_feats, x_seq, seq_lengths):
        x = self.bn_x(x_feats)
        x_seq = x_seq.unsqueeze(-1)  # (B, L, 1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h = h_n[-1]
        z = torch.cat([x, h], dim=1)
        return self.mlp(z).squeeze(1)  # logits

# ======================
# Train / Validate (Streaming)
# ======================
def train_streaming(feature_cols, seq_col, target_col, id_col,
                    epochs=3, batch_size=4096, lr=1e-3, device="cuda"):

    base_feat_dim = len(feature_cols)
    extra_dim = 3 if CFG['USE_SEQ_STATS'] else 0
    d_features = base_feat_dim + extra_dim

    ds_train = ParquetStreamDataset(
        TRAIN_PATH, feature_cols, seq_col, target_col,
        id_col=id_col, split="train",
        n_folds=CFG['N_FOLDS'], val_fold=CFG['VAL_FOLD'],
        neg_keep_prob=neg_keep_prob,
        arrow_batch_rows=CFG['ARROW_BATCH_ROWS'],
        seed=CFG['SEED'],
        use_seq_stats=CFG['USE_SEQ_STATS'],
        seq_l_max=CFG['SEQ_L_MAX'],
        seq_clip_abs=CFG['SEQ_CLIP_ABS'],
    )
    ds_valid = ParquetStreamDataset(
        TRAIN_PATH, feature_cols, seq_col, target_col,
        id_col=id_col, split="valid",
        n_folds=CFG['N_FOLDS'], val_fold=CFG['VAL_FOLD'],
        neg_keep_prob=1.0,  # 검증은 원분포로
        arrow_batch_rows=CFG['ARROW_BATCH_ROWS'],
        seed=CFG['SEED'],
        use_seq_stats=CFG['USE_SEQ_STATS'],
        seq_l_max=CFG['SEQ_L_MAX'],
        seq_clip_abs=CFG['SEQ_CLIP_ABS'],
    )

    train_loader = DataLoader(
        ds_train, batch_size=CFG['BATCH_SIZE'], shuffle=False,
        collate_fn=collate_fn_train,
        num_workers=CFG['NUM_WORKERS'],
        pin_memory=CFG['PIN_MEMORY'],
        persistent_workers=False
    )
    valid_loader = DataLoader(
        ds_valid, batch_size=CFG['BATCH_SIZE'], shuffle=False,
        collate_fn=collate_fn_train,
        num_workers=CFG['NUM_WORKERS'],
        pin_memory=CFG['PIN_MEMORY'],
        persistent_workers=False
    )
    
    model = TabularSeqModel(d_features=d_features, lstm_hidden=64, hidden_units=[256,128], dropout=0.2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best = {'ap': -1.0, 'state': None, 'epoch': -1, 'metrics': None}
    patience = CFG['EARLY_STOP_PATIENCE']
    wait = 0
    
    frac_train = (CFG['N_FOLDS'] - 1) / CFG['N_FOLDS']
    frac_valid = 1 / CFG['N_FOLDS']

    exp_train_samples = pos_cnt * frac_train + neg_cnt * frac_train * neg_keep_prob
    exp_valid_samples = pos_cnt * frac_valid + neg_cnt * frac_valid

    total_train_batches = int(math.ceil(exp_train_samples / batch_size))
    total_valid_batches = int(math.ceil(exp_valid_samples / batch_size))

    for epoch in range(1, epochs+1):
        # ---- Train ----
        model.train()
        tr_loss, tr_n = 0.0, 0
        for xs, seqs, lens, ys in tqdm(
            train_loader, desc=f"Train {epoch}", total=total_train_batches
        ):
            xs   = xs.to(device, non_blocking=True)
            seqs = seqs.to(device, non_blocking=True)
            lens = lens.to(device, non_blocking=True)
            ys   = ys.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits = model(xs, seqs, lens)
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * ys.size(0)
            tr_n += ys.size(0)
        tr_loss /= max(1, tr_n)

        # ---- Valid ----
        model.eval(); va_loss, va_n = 0.0, 0
        MAX_VAL_SAMPLES = 1_000_000  # 100만개까지만 유지
        all_prob, all_true, kept = [], [], 0
        with torch.no_grad():
            for xs, seqs, lens, ys in tqdm(
                valid_loader, desc=f"Valid {epoch}", total=total_valid_batches
            ):
                xs   = xs.to(device, non_blocking=True)
                seqs = seqs.to(device, non_blocking=True)
                lens = lens.to(device, non_blocking=True)
                ys   = ys.to(device, non_blocking=True)
                
                logits = model(xs, seqs, lens)
                loss = criterion(logits, ys)
                va_loss += loss.item() * ys.size(0)
                va_n += ys.size(0)
                prob = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
                ycpu = ys.detach().cpu().numpy().astype(np.int8)

                room = MAX_VAL_SAMPLES - kept
                if room <= 0:
                    continue
                if len(prob) > room:
                    all_prob.append(prob[:room]); all_true.append(ycpu[:room])
                    kept += room
                else:
                    all_prob.append(prob); all_true.append(ycpu); kept += len(prob)

        y_prob = np.concatenate(all_prob) if all_prob else np.zeros(0, np.float32)
        y_true = np.concatenate(all_true) if all_true else np.zeros(0, np.int8)

        metrics = local_score(y_true, y_prob)
        print(f"[Epoch {epoch}] TrainLoss={tr_loss:.5f} | ValLoss={va_loss:.5f} | "
              f"AP={metrics['ap']:.6f} AUC={metrics['auc']:.6f} WLL={metrics['wll']:.6f} "
              f"LocalScore={metrics['score']:.6f}")

        # Early stopping on AP
        if metrics['ap'] > best['ap'] + 1e-8:
            best = {'ap': metrics['ap'], 'state': model.state_dict(), 'epoch': epoch, 'metrics': metrics}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[EarlyStop] no AP improvement for {patience} epochs (best @ {best['epoch']})")
                break

    # 베스트 복원
    if best['state'] is not None:
        model.load_state_dict(best['state'])

    print(f"[Best] epoch={best['epoch']} | AP={best['metrics']['ap']:.6f} "
          f"AUC={best['metrics']['auc']:.6f} WLL={best['metrics']['wll']:.6f} "
          f"LocalScore={best['metrics']['score']:.6f}")

    return model, best

# ======================
# Train & Validate
# ======================
model, best = train_streaming(
    feature_cols=feature_cols,
    seq_col=seq_col,
    target_col=target_col,
    id_col=id_col,
    epochs=CFG['EPOCHS'],
    batch_size=CFG['BATCH_SIZE'],
    lr=CFG['LEARNING_RATE'],
    device=device
)

# ======================
# Inference (Streaming)
# ======================
class ParquetStreamDatasetInfer(ParquetStreamDataset):
    def __init__(self, parquet_path, feature_cols, seq_col, id_col=None,
                 arrow_batch_rows=200_000, seed=42):
        super().__init__(parquet_path, feature_cols, seq_col,
                         target_col=None, id_col=id_col, split=None,
                         n_folds=1, val_fold=0,
                         neg_keep_prob=1.0,
                         arrow_batch_rows=arrow_batch_rows,
                         seed=seed,
                         use_seq_stats=CFG['USE_SEQ_STATS'],
                         seq_l_max=CFG['SEQ_L_MAX'],
                         seq_clip_abs=CFG['SEQ_CLIP_ABS'])

def _num_rows(parquet_path: str) -> Optional[int]:
    try:
        return pq.ParquetFile(parquet_path).metadata.num_rows
    except Exception:
        return None

test_ds = ParquetStreamDatasetInfer(
    TEST_PATH, feature_cols, seq_col, id_col=id_col,
    arrow_batch_rows=CFG['ARROW_BATCH_ROWS'], seed=CFG['SEED']
)

test_ld = DataLoader(
    test_ds, batch_size=CFG['BATCH_SIZE'], shuffle=False,
    collate_fn=collate_fn_infer,
    num_workers=CFG['NUM_WORKERS'],
    pin_memory=CFG['PIN_MEMORY'],
    persistent_workers=False
)

# 총 배치 수 추정 (없으면 None → 미정 진행바)
test_rows = _num_rows(TEST_PATH)
total_infer_batches = int(math.ceil(test_rows / CFG['BATCH_SIZE'])) if test_rows else None

model.eval()
outs = []
with torch.no_grad():
    for xs, seqs, lens in tqdm(test_ld, desc="Inference", total=total_infer_batches):
        xs   = xs.to(device, non_blocking=True)
        seqs = seqs.to(device, non_blocking=True)
        lens = lens.to(device, non_blocking=True)
        with torch.cuda.autocast('cuda', enabled=(device=='cuda')):
            logits = model(xs, seqs, lens)
        outs.append(torch.sigmoid(logits).cpu())

test_preds = torch.cat(outs).numpy()

submit = pd.read_csv('./Toss/sample_submission.csv')
if len(submit) != len(test_preds):
    print(f"[WARN] submission rows ({len(submit)}) != preds ({len(test_preds)}). Trunc/pad to match.")
preds = test_preds[:len(submit)]
if len(preds) < len(submit):
    pad = np.full(len(submit)-len(preds), preds.mean() if len(preds)>0 else 0.1, dtype=np.float32)
    preds = np.concatenate([preds, pad])
submit['clicked'] = preds
submit.to_csv(SUB_PATH, index=False)
print(f"[Saved] {SUB_PATH}")

# ===== 로컬 점수 요약 =====
m = best['metrics']
print("\n===== Local Validation (Best) =====")
print(f"AP={m['ap']:.6f}  AUC={m['auc']:.6f}  WLL={m['wll']:.6f}  LocalScore={m['score']:.6f}")
