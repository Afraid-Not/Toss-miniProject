# -*- coding: utf-8 -*-
"""
99_dacon_baseline_6.py 반복 실행기
- 프로젝트 루트(= Toss/ 폴더 바로 위)에서 실행 권장
- 각 실행의 stdout/stderr를 실시간으로 콘솔에 출력 + 로그 파일 저장
"""

import os, sys, time, subprocess, datetime, shlex, gc
from pathlib import Path

# ========== 설정값 ==========
TARGET_REL = "./Toss/codes/99_dacon_baseline_6.py"  # 반복 실행할 스크립트 경로(프로젝트 루트 기준)
RUNS = 10                 # 실행 횟수 (0이면 무한 반복)
SLEEP_SEC = 0            # 각 실행 사이 대기(초)
STOP_ON_ERROR = False    # 실행 중 오류(returncode != 0)이면 중단할지 여부
LOG_DIR = Path("./Toss/log/auto_runs_baseline6")  # 로그 저장 폴더
EXTRA_PY_ARGS = []       # 대상 스크립트가 인자를 받는다면 여기에 예: ["--folds", "5"]
PYTHON_EXE = sys.executable  # 현재 파이썬 해석기 사용(원하면 절대경로 지정)
ENV_OVERRIDES = {
    # 필요 시 환경변수 오버라이드(예: CUDA 디바이스 고정)
    # "CUDA_VISIBLE_DEVICES": "0",
}

# ========== 준비 ==========
script_path = Path(TARGET_REL).resolve()
if not script_path.exists():
    print(f"[ERROR] 대상 스크립트를 찾을 수 없습니다: {script_path}")
    sys.exit(1)

# 대상 스크립트가 Toss/codes/ 안에 있다면, CWD를 '프로젝트 루트(= Toss 상위)'로 맞춰
# 상대경로 './Toss/...'들이 정상 동작하도록 합니다.
# 구조: <project_root> / Toss / codes / 99_dacon_baseline_6.py
project_root = script_path.parent.parent.parent if script_path.parent.name == "codes" else Path.cwd()

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ========== 루프 실행 ==========
i = 0
try:
    while RUNS == 0 or i < RUNS:
        i += 1
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"run_{i:03d}_{ts}.log"

        # 커맨드 & 환경변수 구성
        cmd = [PYTHON_EXE, str(script_path)] + list(EXTRA_PY_ARGS)
        env = os.environ.copy()
        env.update(ENV_OVERRIDES)
        env["AUTO_RUN"] = "1"     # 하위 스크립트에서 필요 시 참고 가능
        env["RUN_IDX"] = str(i)

        print(f"\n[{ts}] === Run {i} 시작 ===")
        print(f"[CWD] {project_root}")
        print(f"[CMD] {' '.join(shlex.quote(c) for c in cmd)}")
        if ENV_OVERRIDES:
            print(f"[ENV_OVERRIDES] {ENV_OVERRIDES}")

        with open(log_path, "w", encoding="utf-8", buffering=1) as logf:
            # 헤더 기록
            logf.write(f"# START: {ts}\n")
            logf.write(f"# CWD: {project_root}\n")
            logf.write(f"# CMD: {' '.join(shlex.quote(c) for c in cmd)}\n")
            if ENV_OVERRIDES:
                logf.write(f"# ENV_OVERRIDES: {ENV_OVERRIDES}\n")

            # 실시간 출력 tee (콘솔 + 파일)
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                logf.write(line)

            rc = proc.wait()
            end_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"[{end_ts}] === Run {i} 종료 (returncode={rc}) ===")
            print(f"[LOG] {log_path}")
            logf.write(f"# END: {end_ts} | returncode={rc}\n")

        # 에러 처리
        if rc != 0 and STOP_ON_ERROR:
            print("[STOP] 오류가 발생하여 반복을 중단합니다.")
            break

        # 간격 대기
        if SLEEP_SEC > 0 and (RUNS == 0 or i < RUNS):
            time.sleep(SLEEP_SEC)

        # 가비지 컬렉션(메모리 정리 도움)
        gc.collect()

    print(f"\n[Done] 총 {i}회 실행 완료. 로그: {LOG_DIR}")

except KeyboardInterrupt:
    print(f"\n[Interrupted] 사용자가 중단했습니다. 현재까지 {i}회 실행. 로그: {LOG_DIR}")
