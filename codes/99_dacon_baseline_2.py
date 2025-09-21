# -*- coding: utf-8 -*-
import os, random, warnings, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ========================= Config =========================
CFG = {
    "BATCH_SIZE": 4096,        # 메모리 부족하면 2048/1024로 내리세요
    "EPOCHS": 10,
    "LEARNING_RATE": 1e-3,
    "SEED": 42,

    # 시퀀스 설정(토큰은 0~99, PAD는 100)
    "VOCAB_SIZE": 100,
    "PAD_IDX": 100,
    "SEQ_MAX_LEN": 1400,        # 최근 N개만 사용(패딩↓). 길이분포 보고 150~256 사이로 튜닝 권장
    "D_EMBED": 32,             # 임베딩 차원(16~64 권장)
    "LSTM_HIDDEN": 64,
    "DROPOUT": 0.2,

    # DataLoader
    "NUM_WORKERS": min(4, os.cpu_count() or 1),
}

device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG["SEED"])

# ========================= Load =========================
TRAIN_PATH = "./Toss/train.parquet"
TEST_PATH  = "./Toss/test.parquet"
target_col = "clicked"
seq_col    = "seq"
ID_COL     = "ID"  # 있으면 자동 제외

all_train = pd.read_parquet(TRAIN_PATH, engine="pyarrow")
test      = pd.read_parquet(TEST_PATH,  engine="pyarrow")

print("Train shape:", all_train.shape)
print("Test shape:",  test.shape)

# 다운샘플 (clicked=1 전부 + clicked=0은 2배 샘플)
clicked_1 = all_train[all_train[target_col] == 1]
clicked_0 = all_train[all_train[target_col] == 0].sample(n=len(clicked_1)*2, random_state=CFG["SEED"])
train = pd.concat([clicked_1, clicked_0], axis=0).sample(frac=1, random_state=CFG["SEED"]).reset_index(drop=True)

print("Train shape:", train.shape)
print("Train clicked:0:", train[train[target_col]==0].shape)
print("Train clicked:1:", train[train[target_col]==1].shape)

# 학습 피처: target/seq/ID 제외
FEATURE_EXCLUDE = {target_col, seq_col, ID_COL}
feature_cols = [c for c in train.columns if c not in FEATURE_EXCLUDE and c in test.columns]
print("Num features:", len(feature_cols))
print("Sequence:", seq_col)
print("Target:", target_col)

# 타입 정리(숫자 컬럼을 float32로)
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        train[c] = train[c].astype(np.float32)
    if c in test.columns and pd.api.types.is_numeric_dtype(test[c]):
        test[c] = test[c].astype(np.float32)

# 큰 DF 일부 해제
del all_train, clicked_0
gc.collect()

# ========================= Dataset / Collate =========================
class ClickDataset(Dataset):
    """메모리 절약: 큰 self.X 캐시 없이, 행 단위로 바로 꺼내기"""
    def __init__(self, df, feature_cols, seq_col, target_col=None, has_target=True):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.seq_col = seq_col
        self.target_col = target_col
        self.has_target = has_target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 탭 피처
        x = torch.tensor(row[self.feature_cols].astype(np.float32).values, dtype=torch.float32)
        # 시퀀스는 문자열 그대로 넘김(콜레이트에서 파싱)
        s = str(row[self.seq_col]) if pd.notna(row[self.seq_col]) else ""
        if self.has_target:
            y = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
            return x, s, y
        else:
            return x, s

def make_collate_tokens(has_target: bool, max_len: int, pad_idx: int):
    """문자열 시퀀스를 int64 토큰으로 파싱 → 최근 max_len으로 자르기 → PAD로 우측 패딩"""
    def collate(batch):
        if has_target:
            xs, s_strs, ys = zip(*batch)
            ys = torch.stack(ys).float()
        else:
            xs, s_strs = zip(*batch)

        x_feats = torch.stack(xs).float()  # (B, d_features)

        seqs = []
        lengths = []
        for s in s_strs:
            if not s or s == "nan":
                arr = np.empty((0,), dtype=np.int64)
            else:
                arr = np.fromstring(str(s), sep=",", dtype=np.int64)
            # 최근 N개만 유지(길이 제한)
            if max_len is not None and arr.size > max_len:
                arr = arr[-max_len:]
            L = len(arr)
            lengths.append(L if L > 0 else 1)  # 길이 0은 pack 안전을 위해 1로
            seqs.append(torch.from_numpy(arr) if L > 0 else torch.empty(0, dtype=torch.long))

        lengths = torch.tensor(lengths, dtype=torch.long)  # CPU (pack에서 .cpu() 사용)
        B = len(seqs)
        L_max = int(max([len(t) for t in seqs], default=1))
        x_seq = torch.full((B, L_max), pad_idx, dtype=torch.long)  # PAD로 채우기
        for i, t in enumerate(seqs):
            if len(t) > 0:
                x_seq[i, :len(t)] = t

        if has_target:
            return x_feats, x_seq, lengths, ys
        else:
            return x_feats, x_seq, lengths
    return collate

collate_train = make_collate_tokens(True,  max_len=CFG["SEQ_MAX_LEN"], pad_idx=CFG["PAD_IDX"])
collate_infer = make_collate_tokens(False, max_len=CFG["SEQ_MAX_LEN"], pad_idx=CFG["PAD_IDX"])

# ========================= Model =========================
class TabularSeqTokenModel(nn.Module):
    def __init__(self, d_features, vocab_size=100, pad_idx=100,
                 d_embed=32, lstm_hidden=64, hidden_units=(256,128), dropout=0.2):
        super().__init__()
        self.bn_x = nn.BatchNorm1d(d_features)
        self.emb  = nn.Embedding(num_embeddings=vocab_size+1, embedding_dim=d_embed, padding_idx=pad_idx)
        self.lstm = nn.LSTM(input_size=d_embed, hidden_size=lstm_hidden, batch_first=True)
        input_dim = d_features + lstm_hidden
        layers = []
        for h in hidden_units:
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = h
        layers += [nn.Linear(input_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_feats, x_seq, seq_lengths):
        x = self.bn_x(x_feats)
        e = self.emb(x_seq)  # (B, L, d_embed); PAD는 자동으로 0벡터
        packed = nn.utils.rnn.pack_padded_sequence(e, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = h_n[-1]  # (B, lstm_hidden)
        z = torch.cat([x, h], dim=1)
        return self.mlp(z).squeeze(1)  # (B,)

# ========================= Train / Valid Split =========================
tr_df, va_df = train_test_split(
    train, test_size=0.2, random_state=CFG["SEED"], shuffle=True, stratify=train[target_col]
)

def make_loader(df, has_target, shuffle, collate_fn):
    ds = ClickDataset(df, feature_cols, seq_col, target_col, has_target=has_target)
    return DataLoader(
        ds,
        batch_size=CFG["BATCH_SIZE"],
        shuffle=shuffle,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=(device == "cuda"),
        prefetch_factor=2 if device == "cuda" else None,
    )

train_loader = make_loader(tr_df, has_target=True,  shuffle=True,  collate_fn=collate_train)
val_loader   = make_loader(va_df, has_target=True,  shuffle=False, collate_fn=collate_train)

# ========================= Train =========================
model = TabularSeqTokenModel(
    d_features=len(feature_cols),
    vocab_size=CFG["VOCAB_SIZE"],
    pad_idx=CFG["PAD_IDX"],
    d_embed=CFG["D_EMBED"],
    lstm_hidden=CFG["LSTM_HIDDEN"],
    hidden_units=(256,128),
    dropout=CFG["DROPOUT"]
).to(device)

optimizer = optim.Adam(model.parameters(), lr=CFG["LEARNING_RATE"])
criterion = nn.BCEWithLogitsLoss()

for epoch in range(1, CFG["EPOCHS"]+1):
    model.train()
    tr_loss = 0.0
    for xs, seqs, lens, ys in tqdm(train_loader, desc=f"Train {epoch}"):
        xs, seqs, ys = xs.to(device), seqs.to(device), ys.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xs, seqs, lens)          # lens는 CPU여도 OK(내부에서 .cpu() 사용)
        loss = criterion(logits, ys)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * ys.size(0)
    tr_loss /= len(tr_df)

    model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for xs, seqs, lens, ys in tqdm(val_loader, desc=f"Valid {epoch}"):
            xs, seqs, ys = xs.to(device), seqs.to(device), ys.to(device)
            logits = model(xs, seqs, lens)
            loss = criterion(logits, ys)
            va_loss += loss.item() * ys.size(0)
    va_loss /= len(va_df)

    print(f"[Epoch {epoch}] train {tr_loss:.4f} | valid {va_loss:.4f}")

# ========================= Inference =========================
test_ds = ClickDataset(test, feature_cols, seq_col, has_target=False)
test_ld = DataLoader(
    test_ds,
    batch_size=CFG["BATCH_SIZE"],
    shuffle=False,
    num_workers=CFG["NUM_WORKERS"],
    pin_memory=(device == "cuda"),
    collate_fn=collate_infer,
    drop_last=False,
    persistent_workers=(device == "cuda"),
    prefetch_factor=2 if device == "cuda" else None,
)

model.eval()
outs = []
with torch.no_grad():
    for xs, seqs, lens in tqdm(test_ld, desc="Inference"):
        xs, seqs = xs.to(device), seqs.to(device)
        probs = torch.sigmoid(model(xs, seqs, lens))
        outs.append(probs.cpu())
test_preds = torch.cat(outs).numpy()

submit = pd.read_csv("./Toss/sample_submission.csv")
submit["clicked"] = test_preds
os.makedirs("./Toss", exist_ok=True)
submit.to_csv("./Toss/baseline_submit.csv", index=False)
print("[DONE] ./Toss/baseline_submit.csv")
