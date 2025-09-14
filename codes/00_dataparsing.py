# /home/pc/Study/Project/Toss/codes/split_200k.py
# train/test Parquet을 200,000행 단위로 스트리밍 분할 저장

import os, shutil
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

TRAIN = "/home/pc/Study/Project/Toss/train.parquet"
TEST  = "/home/pc/Study/Project/Toss/test.parquet"
OUT_BASE = "/home/pc/Study/Project/Toss/_split"   # 결과 폴더 루트

ROWS_PER_FILE = 200_000
BATCH_SIZE = 50_000       # 메모리 여유 없으면 더 줄이기(예: 50_000)
OVERWRITE = True           # 기존 결과 폴더 있으면 삭제

DROP_VIRTUAL = {"__fragment_index","__batch_index","__last_in_fragment","__filename"}

def ensure_clean_dir(path: str):
    if os.path.exists(path) and OVERWRITE:
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def split_parquet(source_path: str, out_dir: str, rows_per_file: int = ROWS_PER_FILE,
                  batch_size: int = BATCH_SIZE, compress: str = "snappy"):
    """Dataset→Scanner로 배치 스트리밍, ParquetWriter 회전 저장(전량 메모리 로드 없음)."""
    ensure_clean_dir(out_dir)
    dset = ds.dataset(source_path, format="parquet")
    cols = [c for c in dset.schema.names if c not in DROP_VIRTUAL]
    scanner = dset.scanner(columns=cols, batch_size=batch_size)

    writer = None
    rows_in_file = 0
    part_idx = 0
    total_rows = 0

    try:
        for batch in scanner.to_batches():
            tbl = pa.Table.from_batches([batch])

            # 새 파일 열기 필요 시 교체
            if writer is None or rows_in_file >= rows_per_file:
                if writer is not None:
                    writer.close()
                part_path = os.path.join(out_dir, f"part-{part_idx:05d}.parquet")
                writer = pq.ParquetWriter(part_path, tbl.schema, compression=compress)
                rows_in_file = 0
                part_idx += 1

            writer.write_table(tbl)
            rows_in_file += tbl.num_rows
            total_rows += tbl.num_rows
    finally:
        if writer is not None:
            writer.close()

    # 검증용 행수 출력
    written = ds.dataset(out_dir, format="parquet").count_rows()
    print(f"[DONE] {out_dir}  files={part_idx}, rows_written={written:,} (from {total_rows:,})")

def main():
    train_out = os.path.join(OUT_BASE, "train_200k")
    test_out  = os.path.join(OUT_BASE, "test_200k")

    # 원본 대략 행수 안내(메타 기반)
    try:
        n_train = ds.dataset(TRAIN, format="parquet").count_rows()
        n_test  = ds.dataset(TEST,  format="parquet").count_rows()
        print(f"[SRC] train ~ {n_train:,} rows, test ~ {n_test:,} rows")
        print(f"[INFO] expected train files ≈ {max(1, n_train // ROWS_PER_FILE + (n_train % ROWS_PER_FILE > 0))}")
        print(f"[INFO] expected test  files ≈ {max(1, n_test  // ROWS_PER_FILE + (n_test  % ROWS_PER_FILE > 0))}")
    except Exception:
        pass

    print("[SPLIT] train → 200k parts …")
    split_parquet(TRAIN, train_out)

    print("[SPLIT] test  → 200k parts …")
    split_parquet(TEST, test_out)

if __name__ == "__main__":
    main()