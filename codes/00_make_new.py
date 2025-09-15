# -*- coding: utf-8 -*-
# VS Code에서 바로 실행 가능한 단일 스크립트
import os, gc, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer

SEED = 42
np.random.seed(SEED)

# ========= 설정 =========
TRAIN_PATH = "./Toss/_meta/train_enriched_2.parquet"
TEST_PATH  = "./Toss/_meta/test_enriched_2.parquet"
OUT_DIR    = "./Toss/new_data"
NEW_TRAIN  = f"{OUT_DIR}/new_train.parquet"
NEW_TEST   = f"{OUT_DIR}/new_test.parquet"

# 고카디널리티 희소 묶기 임계치(최소 등장 횟수)
RARE_MIN_COUNT = 50
# 가우시안 정규화(Quantile→Normal) 설정
N_QUANTILES = 512
SUBSAMPLE   = 200_000

# ========= Load =========
train_all = pd.read_parquet(TRAIN_PATH, engine="pyarrow")
test_df   = pd.read_parquet(TEST_PATH,  engine="pyarrow")
print("Train shape:", train_all.shape, "| clicked==1:", int((train_all["clicked"]==1).sum()))
print("Test  shape:", test_df.shape)

# ========= 샘플링 없음: 전체 train 사용 =========
train = train_all.copy()
print("↓ Using FULL train (no sampling)")
print("Train shape:", train.shape,
      "| clicked=1:", int((train["clicked"]==1).sum()),
      "| clicked=0:", int((train["clicked"]==0).sum()))

# ========= 열 분리 =========
target = "clicked"
feature_cols = [c for c in train.columns if c != target]

# datetime은 이번 전처리 대상에서 제외(필요시 별도 파생 권장)
dt_cols  = [c for c in feature_cols if np.issubdtype(train[c].dtype, np.datetime64)]
cat_cols = [c for c in feature_cols if (train[c].dtype == "object") or pd.api.types.is_categorical_dtype(train[c])]
num_cols = [c for c in feature_cols if (is_numeric_dtype(train[c]) or is_bool_dtype(train[c])) and c not in dt_cols]

# 불리언은 수치형으로 두되, 가우시안화는 제외
bool_cols = [c for c in num_cols if is_bool_dtype(train[c])]
gauss_cols = [c for c in num_cols if c not in bool_cols]

# 범주형을 object로 고정
train.loc[:, cat_cols]   = train[cat_cols].astype("object")
test_df.loc[:, cat_cols] = test_df[cat_cols].astype("object")

# ========= 고카디널리티 희소 묶기 + NA 레벨 처리 =========
# (train에서 등장 빈도 기준으로 희소 레벨을 '__RARE__'로, 결측은 '__NA__'로)
for c in cat_cols:
    vc = train[c].value_counts(dropna=True)
    rare_vals = set(vc[vc < RARE_MIN_COUNT].index.tolist())

    # 희소 묶기
    if rare_vals:
        train[c]   = train[c].where(~train[c].isin(rare_vals), "__RARE__")
        test_df[c] = test_df[c].where(~test_df[c].isin(rare_vals), "__RARE__")

    # 결측 레벨 고정
    train[c]   = train[c].where(train[c].notna(), "__NA__")
    test_df[c] = test_df[c].where(test_df[c].notna(), "__NA__")

# ========= 라벨 인코딩(Ordinal) =========
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
if cat_cols:
    train[cat_cols]   = enc.fit_transform(train[cat_cols])
    test_df[cat_cols] = enc.transform(test_df[cat_cols])
    # 저장 크기 축소
    train[cat_cols]   = train[cat_cols].astype(np.int32)
    test_df[cat_cols] = test_df[cat_cols].astype(np.int32)

# ========= 수치형 가우스 정규분포화(Quantile→Normal) =========
# 결측은 보존, 불리언은 제외
for c in gauss_cols:
    s_tr = train[c]
    s_te = test_df[c]

    # 둘 다 전부 결측/단일값이면 스킵
    if s_tr.notna().sum() <= 1:
        continue

    qt = QuantileTransformer(
        n_quantiles=N_QUANTILES,
        output_distribution="normal",
        subsample=SUBSAMPLE,
        random_state=SEED,
        copy=True,
    )

    # fit on train (non-null)
    mask_tr = s_tr.notna()
    qt.fit(s_tr[mask_tr].to_numpy().reshape(-1,1))

    # transform train
    out_tr = s_tr.copy()
    out_tr.loc[mask_tr] = qt.transform(s_tr[mask_tr].to_numpy().reshape(-1,1)).ravel()
    train[c] = out_tr.astype(np.float32)

    # transform test
    mask_te = s_te.notna()
    out_te = s_te.copy()
    out_te.loc[mask_te] = qt.transform(s_te[mask_te].to_numpy().reshape(-1,1)).ravel()
    test_df[c] = out_te.astype(np.float32)

gc.collect()

# ========= 저장 =========
os.makedirs(OUT_DIR, exist_ok=True)

# new_train은 전체 train을 저장(모델 학습에 바로 사용 가능)
train.to_parquet(NEW_TRAIN, index=False)
# test는 원본 전량을 동일 전처리로 저장
test_df.to_parquet(NEW_TEST, index=False)

print(f"[OK] saved:\n  - {NEW_TRAIN}\n  - {NEW_TEST}")
