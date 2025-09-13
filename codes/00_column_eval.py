# -*- coding: utf-8 -*-
import os, warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ========= 설정 =========
DATA_PATH = "./Toss/train.parquet"  # csv도 가능: "./data.csv"
COL = "feat_e_4"                        # 진단할 컬럼명

MAX_SAMPLE = 10_000_000   # 너무 크면 샘플링
SEED = 42
np.random.seed(SEED)

def load_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path, engine="pyarrow", columns=[COL, "clicked"])
    elif ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    else:
        raise ValueError(f"지원하지 않는 포맷: {ext}")

def last_digit_uniformity(series):
    # 마지막 자리수(0~9) 분포가 균등하면 해시/난수 느낌
    s = series.dropna().astype(np.int64)
    last = (s % 10).value_counts().reindex(range(10), fill_value=0)
    probs = last / last.sum()
    return last.to_dict(), probs.round(4).to_dict()

def monotonic_ratio(series):
    # 단조 증가/감소 비율 추정(인접 비교)
    s = series.dropna().values
    if len(s) < 3: return 0.0, 0.0
    dif = np.diff(s)
    inc = np.mean(dif >= 0)
    dec = np.mean(dif <= 0)
    return float(inc), float(dec)

def runs_like_random(series, k=20):
    # 간단 난수성 힌트: 구간별 평균이 일정하면 랜덤 느낌(매우 러프)
    s = series.dropna().values
    if len(s) < k: return None
    chunk = len(s) // k
    means = [np.mean(s[i*chunk:(i+1)*chunk]) for i in range(k)]
    std_means = float(np.std(means))
    return {"chunks": k, "chunk_size": chunk, "std_of_means": round(std_means, 3)}

def corr_with_target(df, col, target="clicked"):
    if target not in df.columns: return None
    # 숫자로 캐스팅(큰 값이면 다운캐스트)
    s = pd.to_numeric(df[col], errors="coerce")
    t = pd.to_numeric(df[target], errors="coerce")
    ok = s.notna() & t.notna()
    if ok.sum() < 100: return None
    s = s[ok]; t = t[ok]
    # 스피어만(순서 상관)과 피어슨(선형) 둘 다 체크
    try:
        sp = s.rank().corr(t, method="pearson")
    except Exception:
        sp = None
    try:
        pe = s.corr(t)
    except Exception:
        pe = None
    return {"spearman_like": None if sp is None else float(sp),
            "pearson": None if pe is None else float(pe)}

if __name__ == "__main__":
    df = load_table(DATA_PATH)
    if COL not in df.columns:
        raise KeyError(f"컬럼 '{COL}' 이(가) 존재하지 않습니다. 실제 컬럼들: {list(df.columns)[:10]}...")

    # 샘플링(메모리 절약)
    if len(df) > MAX_SAMPLE:
        df = df.sample(n=MAX_SAMPLE, random_state=SEED).reset_index(drop=True)

    s = pd.to_numeric(df[COL], errors="coerce")
    n = len(s)
    nunique = int(s.nunique(dropna=True))
    na = int(s.isna().sum())
    uniq_ratio = nunique / max(1, n)

    # 기본 통계
    desc = s.describe(percentiles=[0.1,0.5,0.9]).to_dict()
    minv, maxv = (None, None)
    if nunique > 0:
        minv, maxv = (int(np.nanmin(s)), int(np.nanmax(s)))

    # 단조성, 마지막 자리수 분포
    inc_ratio, dec_ratio = monotonic_ratio(s)
    last_counts, last_probs = last_digit_uniformity(s.dropna())
    runs = runs_like_random(s)

    # 타깃 상관(있으면)
    corr_t = corr_with_target(df, COL, target="clicked")

    print("="*60)
    print(f"[컬럼] {COL}")
    print(f"[행] {n:,}  [결측] {na:,}  [유일값] {nunique:,}  [유일비율] {uniq_ratio:.4f}")
    if nunique > 0:
        print(f"[범위] min={minv}, max={maxv}")
    print(f"[기술통계] { {k: (round(v,3) if isinstance(v,(int,float,np.floating)) else v) for k,v in desc.items()} }")
    print("-"*60)
    print(f"[단조 비율] 증가≥0: {inc_ratio:.3f}, 감소≤0: {dec_ratio:.3f}")
    print(f"[끝자리 분포 counts] {last_counts}")
    print(f"[끝자리 분포 probs ] {last_probs}")
    print(f"[러프 난수성 힌트] {runs}")
    print("-"*60)
    print(f"[타깃 상관(clicked)] {corr_t}")
    print("="*60)

    # 의사결정 힌트
    hint = []
    if uniq_ratio > 0.95:
        hint.append("거의 전부 유일 → ID(식별자) 가능성 높음: 모델 입력에서 제외 권장")
    if inc_ratio > 0.98 or dec_ratio > 0.98:
        hint.append("거의 단조 → row_index/시간 인코딩 의심: 시간 파생(요일/시간/경과 등) 시도")
    if abs((pd.Series(last_probs).fillna(0) - 0.1).abs().mean()) < 0.02 and uniq_ratio > 0.5:
        hint.append("끝자리 균등 + 고카디널리티 → 해시/난수형 ID 느낌(숫자 크기 의미 없음)")
    if corr_t is not None and (abs(corr_t.get("pearson") or 0) > 0.05 or abs(corr_t.get("spearman_like") or 0) > 0.05):
        hint.append("타깃과 상관 존재 → 단순 ID가 아닐 수도. 생성규칙/시간성/누락 피처 점검")

    if not hint:
        hint.append("특별한 구조 없음 → 노이즈/ID 가능성. 엔티티 통계 피처를 고려하거나 제외")

    print("[의사결정 힌트]")
    for h in hint:
        print(" - " + h)
