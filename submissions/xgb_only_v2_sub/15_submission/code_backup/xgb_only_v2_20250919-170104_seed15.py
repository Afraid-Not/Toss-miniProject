# -*- coding: utf-8 -*-
import os, gc, warnings, datetime, json, shutil, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import average_precision_score
import optuna, time
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import hashlib
# ====== XGBoost only ======
import xgboost as xgb

# ---------------- Config ----------------
RUN_VER = "xgb_only_v2"   # << 기존 version 충돌 피하려고 이름 변경

# 로그/세이브 경로는 시드 결정 전에 먼저 준비
LOG_DIR = Path("./Toss/log/"); LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---- Seed auto-increment (per RUN_VER) ----
def get_and_bump_seed(run_ver: str, seed_file_dir: Path = LOG_DIR, default_seed: int = 1) -> int:
    """
    seed_file: LOG_DIR / f"SEED_COUNTS_{run_ver}.json"
    파일이 없으면 default_seed로 시작.
    반환: 이번 실행에서 사용할 seed
    파일에는 (다음 실행을 위한) seed+1을 저장.
    """
    seed_file = seed_file_dir / f"SEED_COUNTS_{run_ver}.json"
    if seed_file.exists():
        try:
            with open(seed_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            cur = int(state.get("seed", default_seed))
        except Exception:
            cur = default_seed
    else:
        cur = default_seed
    # 다음 실행을 위해 +1 저장
    try:
        with open(seed_file, "w", encoding="utf-8") as f:
            json.dump({"seed": cur + 1}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return cur

# 이번 실행용 SEED 자동 결정 (+ 다음 실행을 위해 파일에 +1 저장됨)
SEED = get_and_bump_seed(RUN_VER, LOG_DIR, default_seed=1)

# 시드가 결정된 뒤 SAVE_DIR 구성
SAVE_DIR = Path(f'./Toss/submissions/{RUN_VER}_sub/{SEED}_submission/')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CFG = {
    "SEED": SEED,

    # XGBoost (GPU, K-Fold + ES)
    "XGB_NUM_BOOST_ROUND": 8000,
    "XGB_ES_ROUNDS": 300,
    "XGB_NFOLDS": 5,

    # Paths (메타 전처리 산출물 사용)
    "META_TRAIN": "./Toss/new_data/new_train_2.parquet",
    "META_TEST":  "./Toss/new_data/new_test_2.parquet",
    "SAMPLE_SUB": "./Toss/sample_submission.csv",
    "OUT_DIR": SAVE_DIR,
    
    "OPTUNA_ON": True,
    "OPTUNA_TRIALS": 40,
    "OPTUNA_TIMEOUT": None,      # 초 단위, 제한 없으면 None
    "OPTUNA_FOLDS": 3,           # 튜닝용 폴드 수(빠르게)
    "OPTUNA_NUM_BOOST_ROUND": 5000,
    "OPTUNA_ES": 200,
    
    #smoke
    # "OPTUNA_ON": True,
    # "OPTUNA_TRIALS": 2,
    # "OPTUNA_TIMEOUT": None,      # 초 단위, 제한 없으면 None
    # "OPTUNA_FOLDS": 2,           # 튜닝용 폴드 수(빠르게)
    # "OPTUNA_NUM_BOOST_ROUND": 200,
    # "OPTUNA_ES": 100,

    # 튜닝 샘플 전략
    "TUNE_NEG_RATIO": 2,         # 튜닝셋에서 음성:양성 비
    "TUNE_USE_ALL_POS": True,    # 튜닝에서 양성 전부 사용할지
    "TUNE_HASH_MOD": 10,         # 10%를 튜닝 풀로(안겹치게) 뽑기
    "TUNE_HASH_REM": 0,          # 해시 나머지값 (0이면 10% 샘플)
    "OPTUNA_USE_TIME_SPLIT": False,  # True면 튜닝에서도 TimeSeriesSplit
    "TIME_COL": "event_time",    # 시계열 컬럼명(있으면)
}

SMOKE = False  # 빠른 확인용이면 True
SMOKE_CFG = {
    "XGB_NFOLDS": 2,
    "XGB_NUM_BOOST_ROUND": 200,
    "XGB_ES_ROUNDS": 100,
}
if SMOKE:
    CFG.update(SMOKE_CFG)

# ---------------- Utils ----------------
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); import random; random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def free_mem():
    gc.collect()

def set_deterministic_env(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 과도한 병렬성/비결정성 방지
    for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")

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

def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.nan_to_num(np.clip(np.asarray(y_prob, dtype=np.float64), 1e-6, 1-1e-6), nan=0.5)
    ap  = average_precision_score(y_true, y_prob) if (y_true.min()<y_true.max()) else 0.0
    wll = weighted_logloss_5050(y_true, y_prob)
    return ap, wll, composite_score(ap, wll)

# ---- Calibration (Temperature scaling) ----
def _clip01(p): return np.clip(p, 1e-6, 1-1e-6)
def _logit(p):  p=_clip01(p); return np.log(p/(1-p))
def _sigmoid(z): return 1/(1+np.exp(-z))

def fit_temperature(y_oof, p_oof, grid=np.linspace(0.5, 3.0, 51)):
    y = np.asarray(y_oof).astype(int); z = _logit(np.asarray(p_oof))
    bestT, best = 1.0, 1e18
    for T in grid:
        pT = _sigmoid(z / T)
        w = weighted_logloss_5050(y, pT)
        if w < best: best, bestT = w, T
    return bestT

def apply_temperature(p, T):
    return _sigmoid(_logit(p) / T)

# ---- Running script auto-backup ----
def backup_running_script(out_dir: Path, run_ver: str, seed: int) -> Path:
    """
    현재 실행 중인 .py 스크립트를 out_dir / 'code_backup' 하위에 복사.
    파일명: {run_ver}_{YYYYmmdd-HHMMSS}_seed{seed}.py
    """
    try:
        # __file__이 가장 신뢰됨. 없으면 sys.argv[0] 시도.
        if "__file__" in globals():
            src = Path(__file__).resolve()
        else:
            src = Path(sys.argv[0]).resolve()
        if not src.exists() or src.suffix.lower() != ".py":
            return Path()  # 인터랙티브 등인 경우 skip

        code_dir = out_dir / "code_backup"
        code_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dst = code_dir / f"{run_ver}_{ts}_seed{seed}.py"
        shutil.copy2(src, dst)
        return dst
    except Exception:
        return Path()

# ---------------- Main ----------------
def main():
    # 시드 고정 + 코드 백업(가장 먼저)
    set_deterministic_env(CFG["SEED"])
    seed_everything(CFG["SEED"])
    backup_path = backup_running_script(CFG["OUT_DIR"], RUN_VER, CFG["SEED"])
    if backup_path:
        print(f"[BACKUP] Script copied to: {backup_path}")
    else:
        print("[BACKUP] Skipped (interactive or path not found)")

    target_col, ID_COL = "clicked", "ID"

    # ---- Load meta train/test (전처리 산출물) ----
    train_en = pd.read_parquet(CFG["META_TRAIN"], engine="pyarrow")
    test_en  = pd.read_parquet(CFG["META_TEST"],  engine="pyarrow")

    # ID 보장 (없으면 가짜 ID)
    if ID_COL in train_en.columns: train_en[ID_COL] = train_en[ID_COL].astype(str)
    else:                          train_en[ID_COL] = pd.RangeIndex(len(train_en)).astype(str)
    if ID_COL in test_en.columns:  test_en[ID_COL]  = test_en[ID_COL].astype(str)
    else:                          test_en[ID_COL]  = pd.RangeIndex(len(test_en)).astype(str)

    if not CFG["OPTUNA_ON"]:
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
            # 기존 규칙(양성 전부 + 음성 2배)
            pos = train_en[train_en[target_col] == 1]
            neg = train_en[train_en[target_col] == 0].sample(n=len(pos)*2, random_state=CFG["SEED"])
            train_en = pd.concat([pos, neg]).sample(frac=1, random_state=CFG["SEED"]).reset_index(drop=True)
            del pos, neg; free_mem()

        # 피처/캐스팅 (원래대로)
        xgb_exclude = {target_col, ID_COL}
        xgb_cols = [c for c in train_en.columns if c not in xgb_exclude and c in test_en.columns]
        for df_ in (train_en, test_en):
            df_[xgb_cols] = df_[xgb_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
        
        ids_test = test_en[ID_COL].astype(str).values
        dtest = xgb.DMatrix(test_en[xgb_cols].values, nthread=4)
        del test_en
        free_mem()

        tune_df = None  # 튠셋 없음
    else:
        # ---- Optuna용: 튠셋/최종 학습셋 풀 분리 ----

        full_train = train_en.copy()

        def _hash_mod(v):
            h = hashlib.blake2b(str(v).encode("utf-8"), digest_size=8).hexdigest()
            return int(h, 16) % CFG["TUNE_HASH_MOD"]

        is_tune_pool = pd.Series(full_train[ID_COL]).map(_hash_mod).eq(CFG["TUNE_HASH_REM"]).values
        is_train_pool = ~is_tune_pool

        # 피처 컬럼(교집합)
        xgb_exclude = {target_col, ID_COL}
        xgb_cols = [c for c in full_train.columns if c not in xgb_exclude and c in test_en.columns]

        # 튠셋: 양성 전부(옵션), 음성은 튠 풀에서만
        if CFG["TUNE_USE_ALL_POS"]:
            pos_tune = full_train[full_train[target_col] == 1]
        else:
            pos_tune = full_train[(full_train[target_col] == 1) & is_tune_pool]

        neg_tune_pool = full_train[(full_train[target_col] == 0) & is_tune_pool]
        n_neg_tune = min(len(neg_tune_pool), int(len(pos_tune) * CFG["TUNE_NEG_RATIO"]))
        neg_tune = neg_tune_pool.sample(n=n_neg_tune, random_state=CFG["SEED"]) if n_neg_tune > 0 else neg_tune_pool
        tune_df = pd.concat([pos_tune, neg_tune]).sample(frac=1, random_state=CFG["SEED"]).reset_index(drop=True)

        # 최종 학습셋: 양성 전부 + (음성 2배) but 음성은 train_pool에서만
        pos_final = full_train[full_train[target_col] == 1]
        neg_final_pool = full_train[(full_train[target_col] == 0) & is_train_pool]
        neg_final = neg_final_pool.sample(n=len(pos_final) * 2, random_state=CFG["SEED"])
        train_en = pd.concat([pos_final, neg_final]).sample(frac=1, random_state=CFG["SEED"]).reset_index(drop=True)

        # 캐스팅
        for df_ in (tune_df, train_en, test_en):
            df_[xgb_cols] = df_[xgb_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
        
        ids_test = test_en[ID_COL].astype(str).values
        dtest = xgb.DMatrix(test_en[xgb_cols].values, nthread=4)
        del test_en
        free_mem()

    X = train_en[xgb_cols].values
    y = train_en[target_col].astype(np.int32).values
    ids_en = train_en[ID_COL].values  # str

    if not CFG.get("OPTUNA_ON", False):
        del train_en; free_mem()

    # 공통 파라미터
    params_base = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": CFG["SEED"], "seed": CFG["SEED"],
        "device": "cuda",
        "tree_method": "hist", "predictor": "gpu_predictor",
        "deterministic_histogram": True, "nthread": 4,
        # "max_bin": 256,
    }

    if CFG.get("OPTUNA_ON", False):

        Xt = X   # ← 학습셋 전체로 튜닝
        yt = y

        # 시계열/일반 분기 (train_en에서 시간컬럼 사용)
        def _build_splits_for_tune():
            if CFG["OPTUNA_USE_TIME_SPLIT"] and (CFG["TIME_COL"] in train_en.columns):
                order = np.argsort(pd.to_datetime(train_en[CFG["TIME_COL"]]).values)
                return ("tss", order)
            else:
                return ("skf", None)

        split_kind, order = _build_splits_for_tune()
        if order is not None:
            Xt, yt = Xt[order], yt[order]

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary:logistic",
                "eval_metric": ["logloss", "aucpr"],
                "device": "cuda",
                "tree_method": "hist",
                "random_state": CFG["SEED"],
                # search space
                "eta": trial.suggest_float("eta", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 10.0),
                "lambda": trial.suggest_float("lambda", 1e-3, 100.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            }

            # splitter
            if split_kind == "tss":
                splitter = TimeSeriesSplit(n_splits=CFG["OPTUNA_FOLDS"])
                split_iter = splitter.split(Xt)
            else:
                splitter = StratifiedKFold(n_splits=CFG["OPTUNA_FOLDS"], shuffle=True, random_state=CFG["SEED"])
                split_iter = splitter.split(Xt, yt)

            scores = []
            for step, (tr_idx, va_idx) in enumerate(split_iter, 1):
                y_tr = yt[tr_idx]
                pos_cnt = max(1, int((y_tr == 1).sum()))
                neg_cnt = max(1, int((y_tr == 0).sum()))
                params["scale_pos_weight"] = float(neg_cnt / pos_cnt)

                dtr = xgb.DMatrix(Xt[tr_idx], label=y_tr, nthread=4)
                dva = xgb.DMatrix(Xt[va_idx], label=yt[va_idx], nthread=4)

                bst = xgb.train(
                    params, dtr,
                    num_boost_round=CFG["OPTUNA_NUM_BOOST_ROUND"],
                    evals=[(dtr, "train"), (dva, "valid")],
                    early_stopping_rounds=CFG["OPTUNA_ES"],
                    verbose_eval=False   # 로그 숨김
                )
                p = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
                ap, wll, sc = compute_metrics(yt[va_idx], p)
                scores.append(sc)

                # 프루닝 판단(출력 없음)
                trial.report(float(np.mean(scores)), step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                del dtr, dva, bst
                free_mem()
            return float(np.mean(scores))

        sampler = TPESampler(seed=CFG["SEED"])
        pruner = MedianPruner(n_warmup_steps=2)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        n_trials = int(CFG["OPTUNA_TRIALS"]) if CFG["OPTUNA_TRIALS"] is not None else 0
        deadline = time.time() + CFG["OPTUNA_TIMEOUT"] if CFG["OPTUNA_TIMEOUT"] else None

        with tqdm(total=n_trials if n_trials > 0 else None,
                desc="[OPTUNA]", dynamic_ncols=True, leave=True, mininterval=0.3) as pbar:
            i = 0
            while True:
                if n_trials and i >= n_trials: break
                if deadline and time.time() >= deadline: break

                trial = study.ask()
                try:
                    val = objective(trial)
                except optuna.TrialPruned:
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                except Exception:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                else:
                    study.tell(trial, val)

                if study.best_value is not None:
                    pbar.set_postfix_str(f"best={study.best_value:.5f}")  # 오른쪽에 best만 표시
                pbar.update(1)
                i += 1

        # 결과 저장 + 최종 학습에 반영(콘솔엔 출력 안 함)
        best_params = study.best_params
        with open(SAVE_DIR / "optuna_best.json", "w", encoding="utf-8") as f:
            json.dump({"best_value": study.best_value, "best_params": best_params}, f, ensure_ascii=False, indent=2)
        params_base.update(best_params)

        try:
            del Xt, yt
        except NameError:
            pass
        # 이제 train_en 정리
        del train_en
        free_mem()

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=CFG["XGB_NFOLDS"], shuffle=True, random_state=CFG["SEED"])
    oof_pred = np.zeros(len(ids_en), dtype=np.float32)
    test_pred_sum = np.zeros(len(ids_test), dtype=np.float32)
    fold_metrics = []
    fold_count = 0

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        id_va = ids_en[va_idx]

        # 자동 scale_pos_weight (neg/pos)
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
        test_pred_sum += pred_te.astype(np.float32)
        fold_count += 1

        # 메모리 정리
        del dtrain, dvalid, bst
        free_mem()

    # ---- OOF & Temperature calibration ----
    ap_xgb, wll_xgb, sc_xgb = compute_metrics(y, oof_pred)
    print(f"[XGB][OOF] AP {ap_xgb:.5f} | WLL {wll_xgb:.5f} | SCORE {sc_xgb:.5f}")
    for i,(ap_f,wll_f,sc_f,biter) in enumerate(fold_metrics, 1):
        print(f"  - Fold{i}: AP {ap_f:.5f} | WLL {wll_f:.5f} | SCORE {sc_f:.5f} | best_iter={biter}")

    # 온도 보정(선택: 기본 on)
    T_xgb = fit_temperature(y, oof_pred)
    oof_cal  = apply_temperature(oof_pred, T_xgb)
    test_pred = (test_pred_sum / max(1, fold_count)).astype(np.float32)
    test_cal  = apply_temperature(test_pred, T_xgb).astype(np.float32)
    print(f"[CAL] XGB temperature T={T_xgb:.3f}")

    # ---- 제출 생성 ----
    xgb_val_df = pd.DataFrame({ "ID": ids_en, "y": y, "p_xgb": oof_cal })
    xgb_df     = pd.DataFrame({ "ID": ids_test, "p_xgb": test_cal })
    xgb_df["ID"] = xgb_df["ID"].astype(str)

    sub = pd.read_csv(CFG["SAMPLE_SUB"])
    if "ID" in sub.columns:
        sub["ID"] = sub["ID"].astype(str)
        merged = sub[["ID"]].merge(xgb_df, on="ID", how="left")
        submit = sub.copy()
        submit["clicked"] = merged["p_xgb"].values
    else:
        # 샘플 제출에 ID가 없다면 test_en 순서대로
        submit = pd.DataFrame({"clicked": xgb_df["p_xgb"].values})

    # 저장
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    score_str = f"{sc_xgb:.5f}".replace(".", "p")
    out_path = CFG["OUT_DIR"] / f"{RUN_VER}_{SEED}_{score_str}_submit_{ts}.csv"
    submit.to_csv(out_path, index=False)
    print(f"[SUB] saved -> {out_path}")

    # 로그 저장
    with open((SAVE_DIR / "log.txt"), "a", encoding="utf-8") as f:
        f.write(f"VER: {RUN_VER} | SEED: {SEED}\n")
        f.write(f"XGB (OOF) AP/WLL/SC = {ap_xgb:.5f}/{wll_xgb:.5f}/{sc_xgb:.5f}\n")
        f.write(f"T_xgb = {T_xgb:.3f}\n")
        if backup_path:
            f.write(f"BACKUP: {backup_path}\n")
        f.write("="*40 + "\n")

    with open((LOG_DIR / f"{RUN_VER}_log.txt"), "a", encoding="utf-8") as f:
        f.write(f"VER: {RUN_VER} | SEED: {SEED}\n")
        f.write(f"XGB (OOF) AP/WLL/SC = {ap_xgb:.5f}/{wll_xgb:.5f}/{sc_xgb:.5f}\n")
        f.write(f"T_xgb = {T_xgb:.3f}\n")
        if backup_path:
            f.write(f"BACKUP: {backup_path}\n")
        f.write("="*40 + "\n")

if __name__ == "__main__":
    main()
