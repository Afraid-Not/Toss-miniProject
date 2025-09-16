# /home/pc/Study/Project/Toss/codes/run_once_xgb_stream.py
# 분할 Parquet(20만행 단위)에서 배치 스트리밍 학습/예측 + tqdm 진행률 + 에러 무시 옵션
# + cleaning_plan / numeric_stats 기반 전처리(학습/예측 일관 적용)
# + pyarrow 버전 호환 스캐너(make_scanner)

import os, gc, csv, time, math, warnings, shutil
warnings.filterwarnings("ignore")

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import xgboost as xgb
from tqdm.auto import tqdm
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_bool_dtype, is_object_dtype

# ===== 설정 =====
VER = "ver_proto"
BASE = "./Project/Toss"
TRAIN_DIR = f"{BASE}/_split/train_200k"
TEST_DIR  = f"{BASE}/_split/test_200k"
OUT_DIR   = f"{BASE}/_out/{VER}_submission"
os.makedirs(OUT_DIR, exist_ok=True)

seed_file = f"{OUT_DIR}/SEED_COUNTS.json"
if not os.path.exists(seed_file):
    seed_state = {"seed": 770}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

#endregion IMPORT

SEED = 1 #seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")

#region BASIC OPTIONS
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)
import datetime
save_path = f'{OUT_DIR}/{SEED}_submission_{VER}/'
os.makedirs(save_path , exist_ok=True )

def backup_self(dest_dir: str | Path = None, add_timestamp: bool = True) -> Path:
    src = Path(__file__).resolve()
    # 목적지 폴더: 환경변수 SELF_BACKUP_DIR > 인자 > ./_backup
    dest_root = Path(
        os.getenv("SELF_BACKUP_DIR") or dest_dir or (src.parent / "_backup")
    ).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    name = src.name
    if add_timestamp:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{src.stem}_{ts}{src.suffix}"

    dst = dest_root / name
    shutil.copy2(src, dst)   # 메타데이터 보존
    return dst

# ---- dtype helpers ----
def to_cat(df, cols):
    for c in cols:
        if c in df:
            df[c] = df[c].astype("category")
    return df

def to_int16_safe(df, cols):
    for c in cols:
        if c in df:
            s = pd.to_numeric(df[c], errors="coerce")  # NaN 유지
            # XGBoost는 float를 더 잘 다룸. 결측은 NaN 그대로.
            df[c] = s.astype("float32")
    return df

def to_f32(df):
    float_cols = df.select_dtypes(include=["float", "float32", "float64"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")
    return df

# ---- GPU 가용성 체크 ----
def gpu_ok():
    try:
        _ = xgb.train(
            {"tree_method": "gpu_hist", "predictor": "gpu_predictor"},
            xgb.DMatrix(np.array([[0.0],[1.0]], dtype=np.float32), label=np.array([0,1], dtype=np.int8)),
            num_boost_round=1
        )
        return True
    except xgb.core.XGBoostError:
        return False

# 메모리/속도 튜닝
TRAIN_BATCH = 120_000
TEST_BATCH  = 150_000
NUM_BOOST_ROUND = 800

USE_GPU = gpu_ok()
if USE_GPU:
    TREE_METHOD = "gpu_hist"
    PREDICTOR   = "gpu_predictor"
else:
    TREE_METHOD = "hist"
    PREDICTOR   = None  # CPU
    print("[GPU] 사용 불가. CPU(hist)로 전환합니다.")

SILENCE_ERRORS = True           # 배치 변환 오류 발생 시 조용히 건너뛰기

CATS = ("gender","age_group","inventory_id")
INTS = ("day_of_week","hour")
EXCLUDE = {"clicked","ID","seq"}
DROP_VIRTUAL = {"__fragment_index","__batch_index","__last_in_fragment","__filename"}

# ===== util =====
def _average_precision_safe(y_true, y_pred):
    """
    AP(Average Precision) – sklearn 있으면 사용, 없으면 수동 구현(fallback).
    """
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, y_pred))
    except Exception:
        # fallback: 정렬 후 positive에서의 precision 평균
        y_true = np.asarray(y_true, dtype=np.int8)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        order = np.argsort(-y_pred)
        y_true = y_true[order]
        cum_pos = np.cumsum(y_true)
        idx = np.arange(1, len(y_true) + 1)
        precision = cum_pos / idx
        P = int(cum_pos[-1]) if len(cum_pos) else 0
        if P == 0:
            return 0.0
        return float(precision[y_true == 1].sum() / P)

def _log1p_safe_inplace(df: pd.DataFrame, col: str):
    # x>=0 에서 표준 log1p, x<0 있으면 0으로 시프트 후 log1p
    v = df[col].astype("float64")
    mn = np.nanmin(v.values)
    shift = -mn if np.isfinite(mn) and mn < 0 else 0.0
    df[col] = np.log1p(np.clip(v + shift, 0.0, None)).astype("float32")

def _slog1p_inplace(df: pd.DataFrame, col: str):
    # 대칭 로그: sign(x)*log1p(|x|)  (양/음 정보 유지)
    v = df[col].astype("float64")
    df[col] = (np.sign(v) * np.log1p(np.abs(v))).astype("float32")

def _log1p_shift_inplace(df: pd.DataFrame, col: str, shift: float | int | None):
    # 본질적 양수인데 소량의 음수 노이즈가 있을 때: shift = -min(x)+ε
    if shift is None:
        shift = 0.0
    v = df[col].astype("float64") + float(shift)
    v = np.clip(v, 0.0, None)
    df[col] = np.log1p(v).astype("float32")

def _weighted_logloss(y_true, y_pred):
    """
    클래스 기여를 0/1이 50:50이 되도록 가중치를 둔 LogLoss.
    - pos 가중치 = 0.5 / (#pos)
    - neg 가중치 = 0.5 / (#neg)
    """
    eps = 1e-15
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), eps, 1.0 - eps)

    P = int((y_true == 1).sum())
    N = int((y_true == 0).sum())
    w1 = 0.5 / max(P, 1)
    w0 = 0.5 / max(N, 1)
    w = np.where(y_true == 1, w1, w0)

    try:
        from sklearn.metrics import log_loss
        return float(log_loss(y_true, y_pred, sample_weight=w, labels=[0, 1]))
    except Exception:
        # 수동 계산 (가중 평균)
        ll = -(w * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))).sum()
        ll /= w.sum() if w.sum() > 0 else 1.0
        return float(ll)

def compute_ap_wll_score(y_true, y_pred):
    """
    반환: (ap, wll, score)  where score = 0.5*AP + 0.5*(1/(1+WLL))
    """
    ap = _average_precision_safe(y_true, y_pred)
    wll = _weighted_logloss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1.0 / (1.0 + wll))
    return ap, wll, score

def dataset_and_cols(path: str):
    dset = ds.dataset(path, format="parquet")
    cols = [c for c in dset.schema.names if c not in DROP_VIRTUAL]
    return dset, cols

def count_rows(dset: ds.Dataset) -> int:
    return dset.count_rows()

def n_batches(total_rows: int, batch_size: int) -> int:
    return max(1, math.ceil(total_rows / batch_size))

def make_scanner(dset: ds.Dataset, columns, batch_size: int):
    # pyarrow 버전에 따라 dset.scanner가 없을 수 있으니 호환 처리
    if hasattr(dset, "scanner"):
        return dset.scanner(columns=columns, batch_size=batch_size)
    # 구버전 fallback
    return ds.Scanner.from_dataset(dset, columns=columns, batch_size=batch_size)

def _first_existing(paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                return p
        except Exception:
            pass
    return None

def _read_csv_safe(path: str) -> pd.DataFrame:
    # UTF-8 BOM 포함/미포함, 기본 인코딩 순서대로 시도
    for kwargs in ({"encoding": "utf-8-sig"}, {"encoding": None}):
        try:
            return pd.read_csv(path, **kwargs)
        except Exception:
            pass
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

def load_numeric_stats(train_paths, test_paths):
    tpath = _first_existing(train_paths)
    ppath = _first_existing(test_paths)
    stats = {}

    def absorb(df: pd.DataFrame):
        if df.empty:
            return
        colkey = None
        for k in ["column","col","feature","name"]:
            if k in df.columns:
                colkey = k
                break
        if colkey is None:
            return
        for _, row in df.iterrows():
            col = str(row[colkey])
            if col not in stats:
                stats[col] = {}
            for c in df.columns:
                if c == colkey:
                    continue
                try:
                    v = float(row[c])
                    if np.isfinite(v):
                        stats[col][str(c).lower()] = v
                except Exception:
                    pass

    if tpath:
        absorb(_read_csv_safe(tpath))
    if ppath:
        absorb(_read_csv_safe(ppath))
    return stats

def load_cleaning_plan(paths):
    """
    지원 스키마:
    A) (column, action[, value, value2, threshold, params, ...])  # 기존 방식
    B) (column, min, max, mean, std, null_ratio, clip_lower, clip_upper, IQR, drop_recommended, impute_strategy)
       - drop_recommended: truthy(1/true/yes/y)면 drop
       - clip_lower/clip_upper: 있으면 clip 하한/상한
       - impute_strategy: mean/median/mode/zero/const:VAL/min/max/none
    """
    p = _first_existing(paths)
    if not p:
        print("[PREP] cleaning_plan not found in:", paths)
        return []

    df = _read_csv_safe(p)
    if df.empty:
        print(f"[PREP] cleaning_plan loaded but EMPTY: {p}")
        return []

    # 헤더 정규화
    df.columns = df.columns.astype(str).str.strip().str.lower()
    # 동의어 통일
    ren = {}
    if "col" in df.columns: ren["col"] = "column"
    if "feature" in df.columns: ren["feature"] = "column"
    if "name" in df.columns: ren["name"] = "column"
    if "op" in df.columns: ren["op"] = "action"
    if "operation" in df.columns: ren["operation"] = "action"
    if ren: df = df.rename(columns=ren)

    # === A) 기존 (column, action) 스키마 ===
    if "column" in df.columns and "action" in df.columns:
        df = df.dropna(subset=["column", "action"])
        df["column"] = df["column"].astype(str).str.strip()
        df["action"] = df["action"].astype(str).str.strip().lower()
        plan = df.to_dict(orient="records")
        print(f"[PREP] cleaning_plan parsed {len(plan)} steps from {p} (schema A)")
        return plan

    # === B) 너 스키마 (clip/impute/drop 추천) ===
    required = {"column", "clip_lower", "clip_upper", "drop_recommended", "impute_strategy"}
    if "column" in df.columns and required.intersection(df.columns):
        # 값 정리
        df = df.dropna(subset=["column"])
        df["column"] = df["column"].astype(str).str.strip()
        if "drop_recommended" not in df.columns: df["drop_recommended"] = False
        if "impute_strategy" not in df.columns: df["impute_strategy"] = ""

        def truthy(x):
            s = str(x).strip().lower()
            if s in {"1","true","yes","y","t"}: return True
            try:
                return float(s) != 0.0
            except Exception:
                return False

        plan = []
        for _, r in df.iterrows():
            col = r["column"]
            # 1) drop
            if truthy(r.get("drop_recommended", False)):
                plan.append({"column": col, "action": "drop"})
                continue  # drop이면 다른 액션은 굳이 안 붙임

            # 2) clip (단측/양측 모두 허용)
            lo = r.get("clip_lower", None)
            hi = r.get("clip_upper", None)
            lo = None if pd.isna(lo) else float(lo)
            hi = None if pd.isna(hi) else float(hi)
            if lo is not None or hi is not None:
                plan.append({"column": col, "action": "clip", "value": lo, "value2": hi})

            # 3) impute
            strat = str(r.get("impute_strategy", "")).strip().lower()
            if strat:
                if strat in {"mean"}:
                    plan.append({"column": col, "action": "fillna_mean"})
                elif strat in {"median"}:
                    plan.append({"column": col, "action": "fillna_median"})
                elif strat in {"mode","most_frequent"}:
                    plan.append({"column": col, "action": "fillna_mode"})
                elif strat in {"zero","0"}:
                    plan.append({"column": col, "action": "fillna_const", "value": 0.0})
                elif strat.startswith("const:"):
                    try:
                        v = float(strat.split("const:")[1])
                    except Exception:
                        v = 0.0
                    plan.append({"column": col, "action": "fillna_const", "value": v})
                elif strat in {"min","max"}:
                    # min/max는 plan표의 min/max 열 또는 stats에서 가져오도록 value 채움
                    v = r.get(strat, None)
                    if pd.isna(v):
                        v = None
                    plan.append({"column": col, "action": "fillna_const", "value": v})
                elif strat in {"none","nan","skip"}:
                    pass  # 아무 것도 안 함
                else:
                    # 알 수 없는 전략 → 일단 median
                    plan.append({"column": col, "action": "fillna_median"})

        print(f"[PREP] cleaning_plan derived {len(plan)} steps from {p} (schema B)")
        return plan

    print(f"[PREP] cleaning_plan headers not recognized: {list(df.columns)}")
    return []

def load_cleaning_assets(plan_paths, train_stats_paths, test_stats_paths):
    plan = load_cleaning_plan(plan_paths)
    stats = load_numeric_stats(train_stats_paths, test_stats_paths)
    return plan, stats

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _winsorize_inplace(s: pd.Series, q_low: float, q_high: float):
    if q_low is None or q_high is None:
        return
    if q_low > 1.0 or q_high > 1.0:
        q_low, q_high = q_low/100.0, q_high/100.0
    q_low = max(0.0, min(0.5, q_low))
    q_high = max(0.5, min(1.0, q_high))
    lo = s.quantile(q_low)
    hi = s.quantile(q_high)
    s.clip(lower=lo, upper=hi, inplace=True)

def _winsorize_stats_inplace(s: pd.Series, sk: dict):
    keys = {str(k).lower(): v for k, v in sk.items()}
    lo = keys.get("p01", keys.get("q01", keys.get("p1", keys.get("q1", keys.get("min")))))
    hi = keys.get("p99", keys.get("q99", keys.get("p99.0", keys.get("q99.0", keys.get("max")))))
    if lo is None and hi is None:
        return
    s.clip(lower=lo if lo is not None else s.min(),
           upper=hi if hi is not None else s.max(), inplace=True)

def _astype_int16_safe_inplace(s: pd.Series):
    s.fillna(-1, inplace=True)
    s.astype(np.int16, copy=False)

def _astype_float32_inplace(s: pd.Series):
    s.astype(np.float32, copy=False)

def _rare_to_other_inplace(s: pd.Series, threshold: int):
    vc = s.value_counts(dropna=False)
    rare = vc[vc < max(1, int(threshold))].index
    s.mask(s.isin(rare), "__OTHER__", inplace=True)

def _is_categorical_col(col: str, s: pd.Series) -> bool:
    # 이름으로 지정(CATS) + dtype으로도 보조 판정
    return (col in CATS) or is_categorical_dtype(s) or is_object_dtype(s) or is_bool_dtype(s)

_ALLOWED_CAT = {
    "astype_category", "map_json", "replace_value",
    "fillna_mode", "fillna_const", "rare_to_other", "drop"
}
_ALLOWED_NUM = {
    "fillna_mean", "fillna_median", "fillna_const",
    "clip", "winsorize", "winsorize_stats",
    "astype_int16_safe", "astype_float32",
    "drop_if_negative", "drop"
}
_ALLOWED_NUM.update({"log1p", "slog1p", "log1p_shift"})

def apply_preprocessing_inplace(df: pd.DataFrame, plan, stats, is_train: bool):
    if not plan and not stats:
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype("float32")
        return df

    for step in plan:
        col = step.get("column")
        if col not in df.columns:
            continue
        action = str(step.get("action","")).lower().strip()
        val = step.get("value", None)
        val2 = step.get("value2", None)
        thr = step.get("threshold", None)
        params = step.get("params", step.get("param", None))
        s = df[col]

        is_cat = _is_categorical_col(col, s)
        if is_cat and action not in _ALLOWED_CAT:
            # 카테고리엔 숫자 액션 금지
            # print(f"[PREP] skip '{action}' on categorical '{col}'")
            continue
        if (not is_cat) and action not in _ALLOWED_NUM:
            # 숫자엔 카테고리 전용 액션 금지
            # print(f"[PREP] skip '{action}' on numeric '{col}'")
            continue

        if action == "drop":
            df.drop(columns=[col], inplace=True, errors="ignore")

        elif action == "fillna_const":
            df[col] = s.fillna(val if val is not None else ("__MISSING__" if is_cat else 0.0))

        elif action == "fillna_mode":
            try:
                mode_val = s.mode(dropna=True)
                mode_val = mode_val.iloc[0] if len(mode_val) else ("__MISSING__" if is_cat else 0.0)
            except Exception:
                mode_val = "__MISSING__" if is_cat else 0.0
            df[col] = s.fillna(mode_val)

        elif action == "fillna_mean":
            mu = stats.get(col, {}).get("mean", None)
            if mu is None: mu = s.mean()
            df[col] = s.fillna(mu)

        elif action == "fillna_median":
            md = stats.get(col, {}).get("median", None)
            if md is None: md = s.median()
            df[col] = s.fillna(md)

        elif action == "clip":
            lo = _to_float(val); hi = _to_float(val2)
            df[col] = s.clip(lower=lo, upper=hi)

        elif action == "winsorize":
            ql = _to_float(val); qh = _to_float(val2)
            _winsorize_inplace(s, ql, qh)

        elif action == "winsorize_stats":
            sk = stats.get(col, {})
            _winsorize_stats_inplace(s, sk)

        elif action == "astype_category":
            df[col] = s.astype("category")

        elif action == "astype_int16_safe":
            _astype_int16_safe_inplace(s)

        elif action == "astype_float32":
            _astype_float32_inplace(s)

        elif action == "map_json":
            try:
                obj = params
                if isinstance(params, str):
                    obj = json.loads(params)
                df[col] = s.map(obj).astype(s.dtype)
            except Exception:
                pass

        elif action == "rare_to_other":
            t = _to_float(thr)
            if t is not None:
                _rare_to_other_inplace(s, int(t))

        elif action == "replace_value":
            try:
                obj = params
                if isinstance(params, str):
                    obj = json.loads(params)
                frm = obj.get("from", None); to = obj.get("to", None)
                df[col] = s.replace(frm, to)
            except Exception:
                pass

        elif action == "drop_if_negative":
            df[col] = s.mask(s < 0, np.nan)
            
        elif action == "log1p":
            _log1p_safe_inplace(df, col)

        elif action == "slog1p":
            _slog1p_inplace(df, col)

        elif action == "log1p_shift":
            shift = step.get("value", None)
            if shift is None:
                sk = stats.get(col, {})
                mn = sk.get("min", None)
                if mn is not None:
                    try:
                        mn = float(mn)
                        shift = (-mn + 1e-9) if mn < 0 else 0.0
                    except Exception:
                        shift = None
            _log1p_shift_inplace(df, col, shift)

    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    return df

# 전처리 자원 로드(경로: _meta 우선, 없으면 /mnt/data 폴백)
PLAN_PATHS = [f"{BASE}/_meta/cleaning_plan.csv", "/mnt/data/cleaning_plan.csv"]
TRAIN_STATS_PATHS = [f"{BASE}/_meta/stats_numeric_train.csv", "/mnt/data/stats_numeric_train.csv"]
TEST_STATS_PATHS  = [f"{BASE}/_meta/stats_numeric_test.csv",  "/mnt/data/stats_numeric_test.csv"]

CLEAN_PLAN, NUM_STATS = load_cleaning_assets(PLAN_PATHS, TRAIN_STATS_PATHS, TEST_STATS_PATHS)
PLAN_DROPS = {step["column"] for step in CLEAN_PLAN if str(step.get("action","")).lower().strip() == "drop"}
print(f"[PREP] cleaning_plan steps={len(CLEAN_PLAN)}, numeric_stats cols={len(NUM_STATS)}, drops={len(PLAN_DROPS)}")

# ---- 배치 전처리 진입점 ----
def preprocess_batch_inplace(df: pd.DataFrame, is_train: bool):
    # 0) 먼저 카테고리/기본 dtype 정리
    df = to_cat(df, CATS)
    # 1) cleaning plan & numeric stats 기반 전처리
    apply_preprocessing_inplace(df, CLEAN_PLAN, NUM_STATS, is_train=is_train)
    # 2) 정수/부동소수 최적화 마무리
    df = to_int16_safe(df, INTS)
    df = to_f32(df)
    return df

# ===== 데이터셋/피처 =====
train_ds, train_cols = dataset_and_cols(TRAIN_DIR)
test_ds,  test_cols  = dataset_and_cols(TEST_DIR)

features = [c for c in train_cols if (c in test_cols) and (c not in EXCLUDE)]
if PLAN_DROPS:
    features = [c for c in features if c not in PLAN_DROPS]
print(f"[INFO] #features = {len(features)}")

missing_in_test = [c for c in train_cols if c not in test_cols and c not in EXCLUDE]
if missing_in_test:
    print(f"[WARN] columns present in train but missing in test (ignored): {missing_in_test[:8]}{' ...' if len(missing_in_test)>8 else ''}")

# ===== 라벨 카운트(진행바) =====
def label_counts(dset: ds.Dataset, batch=150_000):
    total = count_rows(dset)
    sc = make_scanner(dset, columns=["clicked"], batch_size=batch)
    pos = 0; tot = 0
    with tqdm(total=total, unit="rows", desc="[COUNT] labels") as pbar:
        for b in sc.to_batches():
            arr = b.column(0).to_numpy(zero_copy_only=False)
            n = arr.size
            tot += n; pos += int(arr.sum())
            pbar.update(n)
    return pos, tot - pos

print("[INFO] counting labels …")
pos, neg = label_counts(train_ds)
den = max(1, pos + neg)
print(f"[INFO] pos={pos:,}, neg={neg:,}, pi={pos/den:.6f}")

def build_valid_sample(dset, features, label_col="clicked", max_rows=300_000, batch_size=150_000):
    """
    거대한 파케에서 일부만 뽑아 in-memory 검증셋 생성 (학습/예측과 동일 전처리 적용).
    """
    rows = 0
    Xs, Ys = [], []
    sc = make_scanner(dset, columns=[label_col] + features, batch_size=batch_size)
    for b in sc.to_batches():
        tbl = pa.Table.from_batches([b])
        df = tbl.to_pandas()
        y = df.pop(label_col).astype("int8")
        df = preprocess_batch_inplace(df, is_train=True)
        Xs.append(df); Ys.append(y)
        rows += len(df)
        if rows >= max_rows:
            break
    if not Xs:
        return None, None, 0
    Xv = pd.concat(Xs, axis=0, ignore_index=True)
    yv = pd.concat(Ys, axis=0, ignore_index=True)
    return Xv, yv.values, len(Xv)

# ===== DataIter 구현 (tqdm + 에러 무시) =====
class ArrowIter(xgb.core.DataIter):
    """PyArrow Dataset → XGBoost 배치 주입(DataIter) with tqdm + silent skip"""
    def __init__(self, dset: ds.Dataset, feature_cols, label_col=None, batch_size=120_000,
                 pbar_desc: str = "", silence_errors: bool = True):
        super().__init__()
        self.dset = dset
        self.feats = list(feature_cols)
        self.label = label_col
        self.batch_size = batch_size
        self.silence_errors = silence_errors

        self.skipped_batches = 0
        self.skipped_rows = 0

        self.total_rows = count_rows(dset)
        self.pbar = tqdm(total=self.total_rows, unit="rows", desc=pbar_desc)

        # 초기 스캐너/이터레이터 구성
        cols = self.feats + ([label_col] if label_col else [])
        self.scanner = make_scanner(dset, columns=cols, batch_size=batch_size)
        self._it = iter(self.scanner.to_batches())

    def reset(self):
        # Scanner 재생성(안정성↑)
        cols = self.feats + ([self.label] if self.label else [])
        self.scanner = make_scanner(self.dset, columns=cols, batch_size=self.batch_size)
        self._it = iter(self.scanner.to_batches())

    def _process_batch(self, batch):
        tbl = pa.Table.from_batches([batch])
        df  = tbl.to_pandas()

        y = None
        if self.label and self.label in df:
            y = df.pop(self.label).astype("int8")

        # === 일관 전처리 ===
        df = preprocess_batch_inplace(df, is_train=True)

        return df, y

    def next(self, input_data):
        while True:
            try:
                batch = next(self._it)
            except StopIteration:
                self.pbar.close()
                return 0

        # Note: XGBoost DataIter contract expects returning 1 until exhausted, else 0
            n = batch.num_rows
            self.pbar.update(n)

            try:
                df, y = self._process_batch(batch)
                input_data(data=df, label=y)
                del df, y, batch
                gc.collect()
                return 1
            except Exception:
                if self.silence_errors:
                    self.skipped_batches += 1
                    self.skipped_rows += n
                    continue
                else:
                    raise

# ===== 학습 진행바 콜백 =====
class TQDMCallback(xgb.callback.TrainingCallback):
    def __init__(self, total_rounds: int, desc: str = "[TRAIN] boosting"):
        self.pbar = tqdm(
            total=total_rounds,
            desc=desc,
            unit="round",
            dynamic_ncols=True,     # 터미널 폭 따라 자동 조절
            leave=True,             # 완료 후에도 한 줄로 남김
            mininterval=0.2,        # 과도한 리프레시 방지
        )

    def after_iteration(self, model, epoch: int, evals_log):
        # (선택) 최근 metric을 바에 붙여서 한 줄로 보기
        try:
            tr = evals_log.get("train", {})
            ll = tr.get("logloss", [])[-1] if tr.get("logloss") else None
            auc = tr.get("auc", [])[-1] if tr.get("auc") else None
            if ll is not None and auc is not None:
                self.pbar.set_postfix_str(f"ll={ll:.5f} auc={auc:.5f}")
        except Exception:
            pass
        self.pbar.update(1)
        return False

# ===== 메인 =====
def main():
    t0 = time.time()

    # (선택) 현재 TREE_METHOD 간단 테스트
    try:
        xgb.train({'tree_method': TREE_METHOD},
                  xgb.DMatrix([[0.0],[1.0]], label=[0,1]), num_boost_round=1)
        print(f"[GPU/CPU] tree_method={TREE_METHOD} OK")
    except xgb.core.XGBoostError as e:
        print(f"[GPU] '{TREE_METHOD}' 불가 → CPU 'hist'로 자동 전환 권장. 에러: {e}")
    import pyarrow.types as pat

    def _is_numeric_feature(col: str) -> bool:
        def _is_num(t): return t is not None and (pat.is_integer(t) or pat.is_floating(t))
        t1 = train_ds.schema.field(col).type if col in train_ds.schema.names else None
        t2 = test_ds.schema.field(col).type  if col in test_ds.schema.names  else None
        return _is_num(t1) or _is_num(t2)

    def _get_stat(sk: dict, keys, default=None):
        for k in keys:
            if k in sk: 
                return sk[k]
            lk = str(k).lower()
            if lk in sk: 
                return sk[lk]
        return default

    # 노이즈 판단 파라미터(필요시 조절)
    NEG_NOISE_RATIO = 0.01   # |neg_tail| <= 1% * positive_scale 면 "노이즈"
    NEG_NOISE_ABS   = 1e-6   # 또는 절대값이 매우 작으면 노이즈로 간주

    numeric_cols = [c for c in features if (c not in CATS) and _is_numeric_feature(c)]
    overlay = []

    for col in numeric_cols:
        sk = NUM_STATS.get(col, {})
        mn = _get_stat(sk, ["min"])
        mx = _get_stat(sk, ["max"])

        try:
            mn = None if mn is None else float(mn)
            mx = None if mx is None else float(mx)
        except Exception:
            mn, mx = None, None

        # 통계가 없으면 보수적으로 log1p만
        if mn is None or mx is None:
            overlay.append({"column": col, "action": "log1p"})
            continue

        if mn >= 0:
            # 완전 비음수
            overlay.append({"column": col, "action": "log1p"})
        elif mx <= 0:
            # 완전 비양수(모두 음수/0)
            overlay.append({"column": col, "action": "slog1p"})
        else:
            # 양/음 공존: 노이즈 음수인지 판정
            # 음수 쪽 대표 크기: p01(또는 q01) 없으면 min 사용
            neg_q = _get_stat(sk, ["p01","q01","p1","q1"], default=mn)
            pos_scale = _get_stat(sk, ["p99","q99","p95","q95"], default=mx)
            try:
                neg_mag = abs(float(neg_q))
            except Exception:
                neg_mag = abs(mn)
            try:
                pos_scale = abs(float(pos_scale))
            except Exception:
                pos_scale = abs(mx)

            is_noise_by_ratio = (pos_scale > 0) and (neg_mag <= NEG_NOISE_RATIO * pos_scale)
            is_noise_by_abs   = (neg_mag <= NEG_NOISE_ABS)

            if is_noise_by_ratio or is_noise_by_abs:
                # 노이즈 음수 → shift 후 log1p
                # shift는 stats의 min 기반으로 자동 계산되도록 value 전달(없으면 함수에서 재계산)
                overlay.append({"column": col, "action": "log1p_shift", "value": (-mn + 1e-9)})
            else:
                # 의미 있는 음수 분포 → 대칭 로그
                overlay.append({"column": col, "action": "slog1p"})

    if overlay:
        # 중복 제거 후 PLAN에 주입
        seen = set(); new_plan = []
        for s in (CLEAN_PLAN + overlay):
            key = (s.get("column"), s.get("action"), s.get("value", None))
            if key in seen:
                continue
            seen.add(key); new_plan.append(s)
        CLEAN_PLAN[:] = new_plan
        print(f"[PREP][overlay] log rules added for {len(numeric_cols)} numeric cols "
              f"(total_plan={len(CLEAN_PLAN)})")
    # === QuantileDMatrix 빌드(스트리밍) ===
    train_iter = ArrowIter(train_ds, features, label_col="clicked",
                           batch_size=TRAIN_BATCH, pbar_desc="[QDM] bins",
                           silence_errors=SILENCE_ERRORS)

    dtrain = xgb.QuantileDMatrix(
        train_iter,
        enable_categorical=True
    )

    # === 파라미터 ===
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss","auc"],
        "tree_method": TREE_METHOD,          # "gpu_hist" / "hist"
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "enable_categorical": True,
        "scale_pos_weight": max(1.0, neg / max(1, pos)),
        "seed": 73,
        "verbosity": 1,
    }
    if PREDICTOR:
        params["predictor"] = PREDICTOR

    # === 학습 ===
    print("[TRAIN] start training …")
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train")],  # metric 로그 활성화
        callbacks=[TQDMCallback(NUM_BOOST_ROUND)],
        verbose_eval=False
    )
    print(f"[TRAIN] skipped batches={train_iter.skipped_batches}, rows={train_iter.skipped_rows}")

    # === (NEW) Holdout 평가: AP/WLL/Score ===
    Xv, yv, n_valid = build_valid_sample(train_ds, features, label_col="clicked", max_rows=300_000)
    if Xv is not None and n_valid > 0:
        dvalid = xgb.DMatrix(Xv, enable_categorical=True)
        pv = bst.predict(dvalid)
        ap, wll, score = compute_ap_wll_score(yv, pv)
        print(f"[EVAL] valid_n={n_valid:,} | AP={ap:.6f}  WLL={wll:.6f}  Score={score:.6f}")

        # 로그 파일에도 남기기
        try:
            with open(os.path.join(save_path, "(LOG)evaluate.txt"), "a") as lf:
                lf.write(f"valid_n={n_valid}, AP={ap:.8f}, WLL={wll:.8f}, Score={score:.8f}\n")
        except Exception:
            pass
    else:
        print("[EVAL] no valid sample constructed; skip AP/WLL score.")

    model_path = os.path.join(save_path, "xgb_stream_once.json")
    bst.save_model(model_path)
    print("[TRAIN] saved:", model_path)

    # === 예측 ===
    print("[PRED] streaming predict …")

    today = datetime.datetime.now().strftime('%Y%m%d')
    score_str = ("nan" if np.isnan(score) else f"{score:.4f}").replace('.', '_')
    total_test = count_rows(test_ds)
    out_csv = os.path.join(save_path, f"{score_str}_{today}_submission_{VER}.csv")

    skipped_pred_batches = 0
    skipped_pred_rows = 0

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID","clicked"])
        sc = make_scanner(test_ds, columns=["ID"] + features, batch_size=TEST_BATCH)
        with tqdm(total=total_test, unit="rows", desc="[PRED] rows") as pbar:
            for b in sc.to_batches():
                n = b.num_rows
                try:
                    tbl = pa.Table.from_batches([b])
                    df  = tbl.to_pandas()

                    ids = df.pop("ID").astype(str)

                    # === 일관 전처리 ===
                    df = preprocess_batch_inplace(df, is_train=False)

                    dtest = xgb.DMatrix(df, enable_categorical=True)
                    p = bst.predict(dtest)

                    w.writerows(zip(ids.tolist(), p.tolist()))
                except Exception:
                    if not SILENCE_ERRORS:
                        raise
                    # fallback: 해당 배치 전체 0.5로 채워 행 보존
                    try:
                        id_idx = b.schema.get_field_index("ID")
                        ids_arr = b.column(id_idx).to_numpy(zero_copy_only=False)
                        id_strs = [str(x) for x in ids_arr]
                    except Exception:
                        id_strs = [""] * n
                    w.writerows(zip(id_strs, [0.5]*n))
                    skipped_pred_batches += 1
                    skipped_pred_rows += n
                finally:
                    pbar.update(n)
                    # 메모리 정리
                    try:
                        del tbl, df
                    except Exception:
                        pass
                    del b
                    gc.collect()

    print(f"[PRED] skipped batches={skipped_pred_batches}, rows={skipped_pred_rows}")
    print(f"[DONE] wrote {out_csv}")
    print(f"[TIME] total {(time.time()-t0):.1f}s")
    return score

if __name__ == "__main__":
    saved = backup_self(dest_dir=save_path)  # 예: ./_backup/스크립트명_YYYYMMDD_HHMMSS.py
    print(f"[self-backup] saved -> {saved}\n")
    score=main()

with open(save_path + f"(LOG)model_{VER}.txt", "a") as f:
    f.write(f"{VER}\n")
    f.write(f"<SEED :{SEED}>\n")
    f.write(f"SCORE :{score}\n")
    f.write("="*40 + "\n")
