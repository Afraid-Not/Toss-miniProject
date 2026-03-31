# Toss Next MLC - CTR 예측 챌린지

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0.5-189FDD?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-02569B?style=flat-square)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.8-FFCC00?style=flat-square)](https://catboost.ai/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Score](https://img.shields.io/badge/Best%20Score-0.34991-brightgreen?style=flat-square)](.)
[![Rank](https://img.shields.io/badge/Rank-TOP%2030-blue?style=flat-square)](.)

<br/>

<p align="center">
  <img src="./30등_토스_NextMLC_tosser.jpg.PNG" alt="Toss Next MLC TOP 30 Result" width="720"/>
</p>

<br/>

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [대회 성적](#대회-성적)
- [핵심 전략](#핵심-전략)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [파이프라인 상세](#파이프라인-상세)
- [실행 방법](#실행-방법)
- [성능 기록](#성능-기록)
- [보고서](#보고서)
- [개발자](#개발자)

<br/>

## 프로젝트 개요

| 항목          | 내용                                   |
| ------------- | -------------------------------------- |
| **대회명**    | 토스 Next ML Challenge (Toss Next MLC) |
| **과제**      | CTR (Click-Through Rate) 예측          |
| **기간**      | 2025.09.12 ~ 2025.09.29                |
| **최고 점수** | **0.34991**                            |
| **최종 순위** | **TOP 30**                             |
| **총 코드**   | Python 파일 38개, 약 10,153줄          |
| **총 제출**   | 23회                                   |

<br/>

## 대회 성적

| Version |    Score    | Description            |
| :-----: | :---------: | :--------------------- |
|   v1    |   0.34972   | 초기 베이스라인        |
|   v2    |   0.34975   | 피처 개선              |
| **v3**  | **0.34991** | **최고 성능 (seed=9)** |

> 각 스코어별 결과물은 `0.34972/`, `0.34975/`, `0.34991/` 디렉토리에 코드 백업과 로그가 보존되어 있습니다.

<br/>

## 핵심 전략

### 1. 스트리밍 데이터 처리

대용량 데이터를 메모리 한계 내에서 처리하기 위해 200k row 단위로 Parquet 파일을 분할하는 스트리밍 파이프라인을 구축했습니다.

### 2. Memory-Lean 피처 엔지니어링

- **OOF Target Encoding**: 데이터 누수 없이 타겟 인코딩 적용
- **Hashed Bag-of-Words**: 고차원 텍스트 피처를 해시 기반으로 압축
- **Sequence Statistics**: 시퀀스 통계 기반 피처 생성

### 3. 다양한 베이스라인 실험

Dacon 베이스라인 7종을 체계적으로 실험하며 최적 구조를 탐색했습니다. Baseline 3이 최적 버전으로 확정되었습니다.

### 4. XGBoost Streaming + Arrow Batching

최종 모델(`03_ver2.py`)은 PyArrow 배치 기반 스트리밍 학습으로 메모리 효율을 극대화했습니다.

### 5. 자동화 학습

`run_until_good.py`를 통해 stdout/stderr 로깅과 함께 목표 성능 달성까지 반복 학습을 자동화했습니다.

### 6. 스코어 기반 결과 관리

모든 유의미한 스코어에 대해 코드 백업과 로그를 디렉토리 단위로 관리하여 실험 재현성을 확보했습니다.

<br/>

## 기술 스택

| Category                  | Technologies                              |
| ------------------------- | ----------------------------------------- |
| **Language**              | Python 3.8+                               |
| **Gradient Boosting**     | XGBoost, LightGBM, CatBoost               |
| **Deep Learning**         | TensorFlow, PyTorch, deepctr-torch        |
| **ML Framework**          | scikit-learn (StratifiedKFold, 5-fold CV) |
| **Data Processing**       | Pandas, NumPy, PyArrow (streaming)        |
| **Hyperparameter Tuning** | Optuna                                    |
| **Visualization**         | Matplotlib, Plotly                        |

<br/>

## 프로젝트 구조

```
Toss-miniProject/
|
|-- codes/                              # 전체 코드 (38 files, ~10,153 lines)
|   |
|   |-- [Data Processing]
|   |   |-- 00_dataparsing.py           # Parquet 스트리밍 분할 (200k rows/file)
|   |   |-- 00_data_watching.py         # 스트리밍 CSV/Parquet 데이터 탐색
|   |   |-- 00_make_enriched.py         # Memory-lean 피처 인리치먼트
|   |   |-- 00_make_new.py              # 신규 피처 생성
|   |   |-- 00_column_eval.py           # 컬럼 평가 및 선택
|   |   |-- 00_all_in_one.py            # 통합 CTR 전처리 프로토콜
|   |
|   |-- [Model Development]
|   |   |-- 01_baseline.py              # LightGBM 베이스라인 (StratifiedKFold, SEED=2)
|   |   |-- 01_model.py                 # 모델 v1 (앙상블)
|   |   |-- 01_model_without_boost.py   # 비부스팅 실험
|   |   |-- 02_model.py                 # 모델 v2 (카테고리 컬럼 통계)
|   |   |-- 02_jaewoo_v0.py             # 팀원 변형 v0
|   |   |-- 02_jaewoo_v1.py             # 팀원 변형 v1
|   |   |-- 03_ver2.py                  # XGBoost 스트리밍 + Arrow (FINAL)
|   |
|   |-- [Baseline Experiments]
|   |   |-- 99_dacon_baseline_1.py      # 베이스라인 실험 1
|   |   |-- 99_dacon_baseline_2.py      # 베이스라인 실험 2
|   |   |-- 99_dacon_baseline_3_dont_touch.py  # 최적 버전 (확정)
|   |   |-- 99_dacon_baseline_4.py      # 베이스라인 실험 4
|   |   |-- 99_dacon_baseline_5.py      # 베이스라인 실험 5
|   |   |-- 99_dacon_baseline_6.py      # 베이스라인 실험 6
|   |   |-- 99_dacon_baseline_7.py      # 베이스라인 실험 7
|   |
|   |-- [Automation]
|       |-- run_until_good.py           # 자동 반복 학습 (stdout/stderr 로깅)
|
|-- 0.34972/                            # 스코어 결과 (code_backup + log)
|-- 0.34975/                            # 스코어 결과 (code_backup + log)
|-- 0.34991/                            # BEST 스코어 (seed=9, code_backup + log)
|
|-- submissions/                        # 제출 파일 (23회)
|   |-- xgb_only_v1_sub/               #   XGBoost v1 제출 (6회)
|   |-- xgb_only_v2_sub/               #   XGBoost v2 제출 (17회)
|
|-- log/                                # 실험 로그 (prototype, ver1, xgb_v1, xgb_v2)
|-- _seq_stats/                         # 시퀀스 통계 분석
|-- requirements.txt                    # 의존성 패키지
```

<br/>

## 파이프라인 상세

### Phase 1: 데이터 처리

```
Raw Data --> 00_dataparsing.py --> Parquet Splits (200k rows each)
                |
                v
         00_data_watching.py --> EDA (스트리밍 방식)
                |
                v
         00_make_enriched.py --> OOF Target Encoding
                |                  + Hashed BoW
                |                  + Sequence Stats
                v
         00_column_eval.py --> 피처 선택
                |
                v
         00_all_in_one.py --> 통합 전처리 결과물
```

### Phase 2: 모델 개발

```
Baseline (01_baseline.py)
    |-- LightGBM + StratifiedKFold (5-fold, SEED=2)
    |
    v
Model v1 (01_model.py) --> 앙상블 적용
    |
    v
Model v2 (02_model.py) --> 카테고리 통계 피처 추가
    |
    v
FINAL (03_ver2.py) --> XGBoost + Arrow Streaming
```

### Phase 3: 자동화 및 제출

```
run_until_good.py --> 반복 학습 + 로깅
    |
    v
submissions/ --> 23회 제출 (xgb_v1: 6회, xgb_v2: 17회)
    |
    v
Score Tracking --> 0.34972/ | 0.34975/ | 0.34991/
```

<br/>

## 실행 방법

### 의존성 설치

```bash
pip install -r requirements.txt
```

### 데이터 준비

```bash
# 1. Parquet 스트리밍 분할
python codes/00_dataparsing.py

# 2. 데이터 탐색 (선택)
python codes/00_data_watching.py

# 3. 피처 인리치먼트
python codes/00_make_enriched.py
```

### 모델 학습

```bash
# 최종 모델 (XGBoost + Arrow Streaming)
python codes/03_ver2.py

# 자동 반복 학습
python codes/run_until_good.py
```

### 베이스라인 실험 재현

```bash
# LightGBM 베이스라인
python codes/01_baseline.py

# Dacon 베이스라인 시리즈 (1~7)
python codes/99_dacon_baseline_1.py
```

<br/>

## 성능 기록

| Version |    Score    | Model       | Key Changes                             |
| :-----: | :---------: | :---------- | :-------------------------------------- |
|   v1    |   0.34972   | LightGBM    | 초기 베이스라인, StratifiedKFold 5-fold |
|   v2    |   0.34975   | XGBoost     | 피처 개선, 카테고리 통계 추가           |
| **v3**  | **0.34991** | **XGBoost** | **Arrow 스트리밍, seed=9 최적화**       |

<br/>

## 보고서

| 문서                                                                                  | 설명                 |
| ------------------------------------------------------------------------------------- | -------------------- |
| [보고서\_토스\_NextMLC_tosser.pdf](./보고서_토스_NextMLC_tosser.pdf)                  | 프로젝트 요약 보고서 |
| [토스*NextMLC_tosser*개발보고서\_1016.pdf](./토스_NextMLC_tosser_개발보고서_1016.pdf) | 상세 개발 보고서     |

<br/>

## 개발자

| Name                  | GitHub                                       |
| --------------------- | -------------------------------------------- |
| Jaehyeon Kim (김재현) | [@Afraid-Not](https://github.com/Afraid-Not) |

<br/>

---

<br/>

# Toss Next MLC - CTR Prediction Challenge

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0.5-189FDD?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-02569B?style=flat-square)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.8-FFCC00?style=flat-square)](https://catboost.ai/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Score](https://img.shields.io/badge/Best%20Score-0.34991-brightgreen?style=flat-square)](.)
[![Rank](https://img.shields.io/badge/Rank-TOP%2030-blue?style=flat-square)](.)

<br/>

<p align="center">
  <img src="./30등_토스_NextMLC_tosser.jpg.PNG" alt="Toss Next MLC TOP 30 Result" width="720"/>
</p>

<br/>

## Table of Contents

- [Overview](#overview)
- [Competition Result](#competition-result)
- [Key Strategies](#key-strategies)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipeline Details](#pipeline-details)
- [Getting Started](#getting-started)
- [Performance Log](#performance-log)
- [Reports](#reports)
- [Developer](#developer)

<br/>

## Overview

| Item                  | Details                                |
| --------------------- | -------------------------------------- |
| **Competition**       | Toss Next ML Challenge (Toss Next MLC) |
| **Task**              | CTR (Click-Through Rate) Prediction    |
| **Period**            | Sep 12 - Sep 29, 2025                  |
| **Best Score**        | **0.34991**                            |
| **Final Rank**        | **TOP 30**                             |
| **Total Code**        | 38 Python files, ~10,153 lines         |
| **Total Submissions** | 23                                     |

<br/>

## Competition Result

| Version |    Score    | Description                   |
| :-----: | :---------: | :---------------------------- |
|   v1    |   0.34972   | Initial baseline              |
|   v2    |   0.34975   | Feature improvement           |
| **v3**  | **0.34991** | **Best performance (seed=9)** |

> Results for each score are preserved in `0.34972/`, `0.34975/`, and `0.34991/` directories with code backups and logs.

<br/>

## Key Strategies

### 1. Streaming Data Processing

Built a streaming pipeline that splits data into 200k-row Parquet files to handle large-scale data within memory constraints.

### 2. Memory-Lean Feature Engineering

- **OOF Target Encoding**: Target encoding without data leakage
- **Hashed Bag-of-Words**: Hash-based compression of high-dimensional text features
- **Sequence Statistics**: Sequence-based statistical feature generation

### 3. Systematic Baseline Experiments

Explored 7 Dacon baseline variants systematically to identify optimal architecture. Baseline 3 was finalized as the optimal version.

### 4. XGBoost Streaming with Arrow Batching

The final model (`03_ver2.py`) maximizes memory efficiency through PyArrow batch-based streaming training.

### 5. Automated Training

`run_until_good.py` automates repeated training with stdout/stderr logging until the target performance is achieved.

### 6. Score-Tracked Result Management

All meaningful scores are managed at the directory level with code backups and logs, ensuring experiment reproducibility.

<br/>

## Tech Stack

| Category                  | Technologies                              |
| ------------------------- | ----------------------------------------- |
| **Language**              | Python 3.8+                               |
| **Gradient Boosting**     | XGBoost, LightGBM, CatBoost               |
| **Deep Learning**         | TensorFlow, PyTorch, deepctr-torch        |
| **ML Framework**          | scikit-learn (StratifiedKFold, 5-fold CV) |
| **Data Processing**       | Pandas, NumPy, PyArrow (streaming)        |
| **Hyperparameter Tuning** | Optuna                                    |
| **Visualization**         | Matplotlib, Plotly                        |

<br/>

## Project Structure

```
Toss-miniProject/
|
|-- codes/                              # All code (38 files, ~10,153 lines)
|   |
|   |-- [Data Processing]
|   |   |-- 00_dataparsing.py           # Parquet streaming split (200k rows/file)
|   |   |-- 00_data_watching.py         # Streaming CSV/Parquet data exploration
|   |   |-- 00_make_enriched.py         # Memory-lean feature enrichment
|   |   |-- 00_make_new.py              # New feature generation
|   |   |-- 00_column_eval.py           # Column evaluation and selection
|   |   |-- 00_all_in_one.py            # Unified CTR preprocessing protocol
|   |
|   |-- [Model Development]
|   |   |-- 01_baseline.py              # LightGBM baseline (StratifiedKFold, SEED=2)
|   |   |-- 01_model.py                 # Model v1 (ensemble)
|   |   |-- 01_model_without_boost.py   # Non-boosting experiment
|   |   |-- 02_model.py                 # Model v2 (categorical column stats)
|   |   |-- 02_jaewoo_v0.py             # Team member variant v0
|   |   |-- 02_jaewoo_v1.py             # Team member variant v1
|   |   |-- 03_ver2.py                  # XGBoost streaming + Arrow (FINAL)
|   |
|   |-- [Baseline Experiments]
|   |   |-- 99_dacon_baseline_1.py      # Baseline experiment 1
|   |   |-- 99_dacon_baseline_2.py      # Baseline experiment 2
|   |   |-- 99_dacon_baseline_3_dont_touch.py  # Optimal version (locked)
|   |   |-- 99_dacon_baseline_4.py      # Baseline experiment 4
|   |   |-- 99_dacon_baseline_5.py      # Baseline experiment 5
|   |   |-- 99_dacon_baseline_6.py      # Baseline experiment 6
|   |   |-- 99_dacon_baseline_7.py      # Baseline experiment 7
|   |
|   |-- [Automation]
|       |-- run_until_good.py           # Auto-runner with stdout/stderr logging
|
|-- 0.34972/                            # Score result (code_backup + log)
|-- 0.34975/                            # Score result (code_backup + log)
|-- 0.34991/                            # BEST score (seed=9, code_backup + log)
|
|-- submissions/                        # Submission files (23 total)
|   |-- xgb_only_v1_sub/               #   XGBoost v1 submissions (6)
|   |-- xgb_only_v2_sub/               #   XGBoost v2 submissions (17)
|
|-- log/                                # Experiment logs (prototype, ver1, xgb_v1, xgb_v2)
|-- _seq_stats/                         # Sequence statistics analysis
|-- requirements.txt                    # Dependencies
```

<br/>

## Pipeline Details

### Phase 1: Data Processing

```
Raw Data --> 00_dataparsing.py --> Parquet Splits (200k rows each)
                |
                v
         00_data_watching.py --> EDA (streaming)
                |
                v
         00_make_enriched.py --> OOF Target Encoding
                |                  + Hashed BoW
                |                  + Sequence Stats
                v
         00_column_eval.py --> Feature Selection
                |
                v
         00_all_in_one.py --> Unified Preprocessing Output
```

### Phase 2: Model Development

```
Baseline (01_baseline.py)
    |-- LightGBM + StratifiedKFold (5-fold, SEED=2)
    |
    v
Model v1 (01_model.py) --> Ensemble applied
    |
    v
Model v2 (02_model.py) --> Categorical statistics features added
    |
    v
FINAL (03_ver2.py) --> XGBoost + Arrow Streaming
```

### Phase 3: Automation and Submission

```
run_until_good.py --> Repeated training + logging
    |
    v
submissions/ --> 23 submissions (xgb_v1: 6, xgb_v2: 17)
    |
    v
Score Tracking --> 0.34972/ | 0.34975/ | 0.34991/
```

<br/>

## Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Data Preparation

```bash
# 1. Parquet streaming split
python codes/00_dataparsing.py

# 2. Data exploration (optional)
python codes/00_data_watching.py

# 3. Feature enrichment
python codes/00_make_enriched.py
```

### Model Training

```bash
# Final model (XGBoost + Arrow Streaming)
python codes/03_ver2.py

# Automated repeated training
python codes/run_until_good.py
```

### Reproduce Baseline Experiments

```bash
# LightGBM baseline
python codes/01_baseline.py

# Dacon baseline series (1-7)
python codes/99_dacon_baseline_1.py
```

<br/>

## Performance Log

| Version |    Score    | Model       | Key Changes                                  |
| :-----: | :---------: | :---------- | :------------------------------------------- |
|   v1    |   0.34972   | LightGBM    | Initial baseline, StratifiedKFold 5-fold     |
|   v2    |   0.34975   | XGBoost     | Feature improvement, categorical stats added |
| **v3**  | **0.34991** | **XGBoost** | **Arrow streaming, seed=9 optimization**     |

<br/>

## Reports

| Document                                                             | Description                 |
| -------------------------------------------------------------------- | --------------------------- |
| [Project Report (KR)](./보고서_토스_NextMLC_tosser.pdf)              | Project summary report      |
| [Development Report (KR)](./토스_NextMLC_tosser_개발보고서_1016.pdf) | Detailed development report |

<br/>

## Developer

| Name         | GitHub                                       |
| ------------ | -------------------------------------------- |
| Jaehyeon Kim | [@Afraid-Not](https://github.com/Afraid-Not) |
