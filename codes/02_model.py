# -*- coding: utf-8 -*-
# 각 열의 value_counts 상위 20개 (seq 열 제외, 스트리밍)
import os, sys, gc
from collections import Counter
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pandas as pd

# ===== 설정 =====
IN_PATH = "./Toss/train.parquet"   # 필요시 argv[1]로 교체
BATCH_ROWS = 200_000
TOPK_SHOW = 20
BUFFER_PER_COL = 20_000         # 컬럼별 Counter 상한(메모리 가드)
INCLUDE_FLOAT = True           # 연속형 float도 집계할지
INCLUDE_NULL = True             # null/NaN도 값으로 취급해 카운트
EXCLUDE_COLS = {"seq"}          # 제외할 열 이름(대소문자 무시)

def _is_countable(t: pa.DataType) -> bool:
    if pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_boolean(t):
        return True
    if pa.types.is_integer(t):
        return True
    if INCLUDE_FLOAT and pa.types.is_floating(t):
        return True
    return False

def _fmt_val(v, maxlen=80):
    s = repr(v).replace("\n", "\\n")
    return (s[:maxlen] + "…") if len(s) > maxlen else s

if __name__ == "__main__":
    if len(sys.argv) > 1:
        IN_PATH = sys.argv[1]
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(IN_PATH)

    exclude_lower = {c.lower() for c in EXCLUDE_COLS}

    dataset = ds.dataset(IN_PATH)  # parquet/csv 자동 추론
    schema = dataset.schema

    cols = [f.name for f in schema
            if _is_countable(f.type) and f.name.lower() not in exclude_lower]

    if not cols:
        print("[INFO] 카운트할 수 있는 컬럼이 없습니다."); sys.exit(0)

    counters = {c: Counter() for c in cols}
    n_rows_total = 0

    reader = dataset.to_batches(columns=cols, batch_size=BATCH_ROWS)
    for rb in reader:
        n = rb.num_rows
        n_rows_total += n
        for c in cols:
            arr = rb[c]
            vc = pc.value_counts(arr)  # struct<values: T, counts: int64>
            vals = vc.field("values")
            cnts = vc.field("counts")

            if not INCLUDE_NULL:
                mask = pc.invert(pc.is_null(vals))
                vals = pc.filter(vals, mask)
                cnts = pc.filter(cnts, mask)

            py_vals = vals.to_pylist()
            py_cnts = cnts.to_pylist()
            ctr = counters[c]
            for v, k in zip(py_vals, py_cnts):
                ctr[v] += int(k)

            # 메모리 가드: 상위만 유지(근사지만 TOP20 안정)
            if len(ctr) > int(BUFFER_PER_COL * 1.2):
                counters[c] = Counter(dict(ctr.most_common(BUFFER_PER_COL)))

        del rb; gc.collect()

    # 출력 & 저장
    out_rows = []
    print(f"[DONE] scanned rows={n_rows_total:,}  file={IN_PATH}\n")
    for c in cols:
        top = counters[c].most_common(TOPK_SHOW)
        print(f"[{c}] top {TOPK_SHOW}")
        for i, (v, k) in enumerate(top, 1):
            print(f"  {i:2d}. {_fmt_val(v)} : {k:,}")
            out_rows.append({"column": c, "rank": i, "value": v, "count": k})
        print()

    if out_rows:
        df = pd.DataFrame(out_rows)
        out_csv = "_value_counts_top20.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[Saved] {out_csv}")
