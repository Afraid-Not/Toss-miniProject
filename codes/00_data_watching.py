# # -*- coding: utf-8 -*-
# """
# seq 열(예: "12,3,7,7,45")에 대한 전반 통계 + 진행률 표기(tqdm)
# - CSV/Parquet 지원
# - Parquet: count_rows()로 전체 행수 집계 → 진행률 정확
# - CSV: 총 라인 수를 먼저 세서(헤더 제외) 진행률 설정(FAST_COUNT=True면 빠른 추정)
# - 출력: ./_seq_stats/ (len_hist.csv, token_freq.csv, last_digit_dist.csv, pos_stats.csv, row_stats.parquet, summary.txt)
# """

# import os, sys, math, warnings, json, collections, time
# warnings.filterwarnings("ignore")

# import numpy as np
# import pandas as pd
# from tqdm.auto import tqdm

# # ========= 설정 =========
# DATA_PATH = "./Toss/train.parquet"    # 예: "./data.csv" 또는 "./data.parquet"
# SEQ_COL   = "seq"                     # 쉼표로 구분된 정수 시퀀스 열 이름
# DELIM     = ","                       # 구분자
# OUT_DIR   = "./_seq_stats"
# os.makedirs(OUT_DIR, exist_ok=True)

# # 대용량 옵션
# CSV_CHUNKSIZE = 200_000               # CSV일 때 청크 크기
# PARQUET_BATCH = 100_000               # Parquet일 때 pyarrow batch rows
# SAVE_ROW_STATS = True                 # 행별 요약을 parquet로 저장
# TOPK_TOKENS   = 100                   # 토큰 빈도 상위 N
# POS_MAX       = 20                    # 위치별 통계 계산 최대 위치(1~POS_MAX)
# FAST_COUNT    = False                 # CSV 총 라인 수 빠른 추정(정확도↓, 속도↑)

# np.random.seed(1)

# def is_parquet(path: str) -> bool:
#     return os.path.splitext(path)[1].lower() in (".parquet", ".pq")

# def parse_seq_cell(cell, delim=","):
#     if pd.isna(cell):
#         return []
#     s = str(cell).strip()
#     if s == "" or s.lower() in ("none","nan"):
#         return []
#     out = []
#     for tok in s.split(delim):
#         tok = tok.strip()
#         if not tok:
#             continue
#         try:
#             out.append(int(tok))
#         except Exception:
#             # 정수 변환 실패는 스킵
#             continue
#     return out

# def update_global_stats(stats, seq):
#     L = len(seq)
#     stats["n_rows"] += 1
#     stats["n_non_empty"] += (L > 0)
#     stats["len_sum"] += L
#     stats["len_sq_sum"] += L * L
#     stats["len_min"] = L if stats["len_min"] is None else min(stats["len_min"], L)
#     stats["len_max"] = L if stats["len_max"] is None else max(stats["len_max"], L)
#     stats["len_hist"][L] += 1

#     for v in seq:
#         stats["token_counts"][v] += 1
#         stats["last_digit_counts"][abs(v) % 10] += 1

#     upto = min(L, POS_MAX)
#     for i in range(upto):
#         v = seq[i]
#         ps = stats["pos_stats"][i]
#         ps["n"] += 1
#         ps["sum"] += v
#         ps["sum2"] += v * v
#         if ps["seen"] < 500000:
#             ps["counter"][v] += 1
#             ps["seen"] += 1

# def compute_row_metrics(seq):
#     L = len(seq)
#     if L == 0:
#         return {
#             "seq_len": 0, "sum": 0, "mean": np.nan, "std": np.nan,
#             "min": np.nan, "max": np.nan,
#             "is_monotonic_inc": False, "is_monotonic_dec": False,
#             "unique_ratio": np.nan, "dup_ratio": np.nan
#         }
#     arr = np.array(seq, dtype=np.int64)
#     s = int(arr.sum())
#     mean = float(arr.mean())
#     std = float(arr.std(ddof=0))
#     mn  = int(arr.min())
#     mx  = int(arr.max())
#     dif = np.diff(arr)
#     inc = bool(np.all(dif >= 0))
#     dec = bool(np.all(dif <= 0))
#     uniq = len(np.unique(arr))
#     unique_ratio = uniq / L
#     dup_ratio = 1.0 - unique_ratio
#     return {
#         "seq_len": L, "sum": s, "mean": mean, "std": std,
#         "min": mn, "max": mx,
#         "is_monotonic_inc": inc, "is_monotonic_dec": dec,
#         "unique_ratio": unique_ratio, "dup_ratio": dup_ratio
#     }

# def flush_token_counts_to_csv(counter, path, topk=100):
#     if not counter:
#         pd.DataFrame(columns=["token","count","freq"]).to_csv(path, index=False)
#         return
#     most = counter.most_common(topk)
#     df = pd.DataFrame(most, columns=["token","count"])
#     df["freq"] = df["count"] / df["count"].sum()
#     df.to_csv(path, index=False)

# def count_rows_csv(filepath):
#     if FAST_COUNT:
#         # 빠른 추정: 파일 크기/샘플 라인 길이로 근사(대략적)
#         sample_n = 20000
#         total_bytes = os.path.getsize(filepath)
#         with open(filepath, "rb") as f:
#             data = f.read(1_000_000)  # 1MB 샘플
#         lines = data.count(b"\n")
#         avg = max(1, lines)
#         bytes_per_line = len(data) / avg
#         est_lines = int(total_bytes / bytes_per_line)
#         return max(0, est_lines - 1)  # 헤더 제외 추정
#     else:
#         # 정확 카운트(헤더 제외)
#         with open(filepath, "rb") as f:
#             n = sum(1 for _ in f) - 1
#         return max(0, n)

# def main():
#     print(f"[INFO] Loading: {DATA_PATH}")
#     stats = {
#         "n_rows": 0,
#         "n_non_empty": 0,
#         "len_sum": 0,
#         "len_sq_sum": 0,
#         "len_min": None,
#         "len_max": None,
#         "len_hist": collections.Counter(),
#         "token_counts": collections.Counter(),
#         "last_digit_counts": collections.Counter(),
#         "pos_stats": [
#             {"n":0, "sum":0.0, "sum2":0.0, "counter": collections.Counter(), "seen":0}
#             for _ in range(POS_MAX)
#         ]
#     }
#     row_stats_frames = []

#     if is_parquet(DATA_PATH):
#         # -------- Parquet 경로 --------
#         import pyarrow.dataset as ds
#         dataset = ds.dataset(DATA_PATH, format="parquet")
#         total_rows = dataset.count_rows()
#         print(f"[INFO] Parquet rows = {total_rows:,}")

#         scanner = dataset.scanner(columns=[SEQ_COL], batch_size=PARQUET_BATCH)
#         pbar = tqdm(total=total_rows, desc="Parsing seq (parquet)", unit="rows")
#         for batch in scanner.to_batches():
#             col = batch.column(0).to_pylist()
#             for cell in col:
#                 seq = parse_seq_cell(cell, DELIM)
#                 update_global_stats(stats, seq)
#                 if SAVE_ROW_STATS:
#                     row_stats_frames.append(compute_row_metrics(seq))
#             pbar.update(len(col))
#         pbar.close()

#     else:
#         # -------- CSV 경로 --------
#         total_rows = count_rows_csv(DATA_PATH)
#         print(f"[INFO] CSV rows (no header) = {total_rows:,}")

#         processed = 0
#         pbar = tqdm(total=total_rows if total_rows > 0 else None,
#                     desc="Parsing seq (csv)", unit="rows")
#         for chunk in pd.read_csv(DATA_PATH, usecols=[SEQ_COL], chunksize=CSV_CHUNKSIZE):
#             col = chunk[SEQ_COL].tolist()
#             for cell in col:
#                 seq = parse_seq_cell(cell, DELIM)
#                 update_global_stats(stats, seq)
#                 if SAVE_ROW_STATS:
#                     row_stats_frames.append(compute_row_metrics(seq))
#             processed += len(col)
#             pbar.update(len(col))
#         pbar.close()

#     # ===== 글로벌 요약 계산/저장 =====
#     n = stats["n_rows"]
#     ne = stats["n_non_empty"]
#     len_mean = stats["len_sum"] / n if n > 0 else 0.0
#     len_var = (stats["len_sq_sum"]/n - len_mean**2) if n > 0 else 0.0
#     len_std = math.sqrt(max(0.0, len_var))

#     len_hist_df = pd.DataFrame(sorted(stats["len_hist"].items()), columns=["seq_len","count"])
#     len_hist_df["ratio"] = len_hist_df["count"] / max(1, len_hist_df["count"].sum())
#     len_hist_df.to_csv(os.path.join(OUT_DIR, "len_hist.csv"), index=False)

#     flush_token_counts_to_csv(stats["token_counts"], os.path.join(OUT_DIR, "token_freq.csv"), TOPK_TOKENS)

#     last_rows = []
#     total_last = sum(stats["last_digit_counts"].values())
#     for d in range(10):
#         c = stats["last_digit_counts"][d]
#         last_rows.append({"last_digit": d, "count": c, "ratio": (c/total_last if total_last>0 else 0.0)})
#     pd.DataFrame(last_rows).to_csv(os.path.join(OUT_DIR, "last_digit_dist.csv"), index=False)

#     pos_rows = []
#     for i,ps in enumerate(stats["pos_stats"], start=1):
#         if ps["n"] == 0:
#             pos_rows.append({"pos": i, "n": 0, "mean": np.nan, "std": np.nan, "mode": np.nan, "unique_approx": np.nan})
#             continue
#         mean = ps["sum"]/ps["n"]
#         var  = ps["sum2"]/ps["n"] - mean*mean
#         std  = math.sqrt(max(0.0, var))
#         mode = np.nan
#         if ps["counter"]:
#             mode = ps["counter"].most_common(1)[0][0]
#         pos_rows.append({
#             "pos": i, "n": ps["n"], "mean": mean, "std": std, "mode": mode,
#             "unique_approx": len(ps["counter"]) if ps["seen"]>0 else np.nan
#         })
#     pd.DataFrame(pos_rows).to_csv(os.path.join(OUT_DIR, "pos_stats.csv"), index=False)

#     if SAVE_ROW_STATS:
#         row_df = pd.DataFrame(row_stats_frames)
#         row_df.to_parquet(os.path.join(OUT_DIR, "row_stats.parquet"), index=False)

#     summary = {
#         "rows_total": int(n),
#         "rows_non_empty": int(ne),
#         "empty_ratio": float((n - ne) / n) if n>0 else 0.0,
#         "seq_len_min": None if stats["len_min"] is None else int(stats["len_min"]),
#         "seq_len_max": None if stats["len_max"] is None else int(stats["len_max"]),
#         "seq_len_mean": float(len_mean),
#         "seq_len_std": float(len_std),
#         "tokens_unique_topK_reported": min(TOPK_TOKENS, len(stats["token_counts"])),
#         "pos_max_computed": POS_MAX
#     }
#     with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
#         f.write(json.dumps(summary, ensure_ascii=False, indent=2))

#     print("[DONE] 통계 저장 위치:", os.path.abspath(OUT_DIR))
#     print(json.dumps(summary, ensure_ascii=False, indent=2))

# if __name__ == "__main__":
#     main()


import pyarrow.parquet as pq

path = "./Toss/new_data/new_train_2.parquet"
pf = pq.ParquetFile(path)

# 전체 스키마(이름+타입)
print(pf.schema)            

# 컬럼명만 리스트
print(pf.schema.names)      