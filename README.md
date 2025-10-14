# Toss Mini Project 💳

토스 데이터 분석 및 머신러닝 프로젝트

## 📋 프로젝트 개요

데이터 분석과 머신러닝 모델을 활용한 토스 미니 프로젝트입니다.

## 🗂️ 프로젝트 구조

```
Toss/
├── codes/                        # 모델 코드
│   ├── 00_all_in_one.py         # 통합 스크립트
│   ├── 00_column_eval.py        # 컬럼 평가
│   ├── 00_data_watching.py      # 데이터 관찰
│   ├── 00_dataparsing.py        # 데이터 파싱
│   ├── 00_make_enriched.py      # 데이터 인리치
│   ├── 00_make_new.py           # 신규 데이터 생성
│   ├── 01_baseline.py           # 베이스라인 모델
│   ├── 01_model.py              # 모델 버전 1
│   ├── 01_model_without_boost.py # 부스팅 없는 모델
│   ├── 02_model.py              # 모델 버전 2
│   ├── 02_jaewoo_v0.py          # Jaewoo 버전 0
│   ├── 02_jaewoo_v1.py          # Jaewoo 버전 1
│   ├── 03_ver2.py               # 버전 3.2
│   ├── 99_dacon_baseline_1~7.py # Dacon 베이스라인 시리즈
│   └── run_until_good.py        # 자동 실행 스크립트
│
├── 0.34972/                     # 스코어 0.34972 결과물
├── 0.34975/                     # 스코어 0.34975 결과물
├── 0.34991/                     # 스코어 0.34991 결과물 (Best)
│
├── _seq_stats/                  # 시퀀스 통계 데이터
├── log/                         # 로그 파일
├── submissions/                 # 제출 파일
└── requirements.txt             # 필요 라이브러리
```

## 🧪 사용된 기술 스택

### 머신러닝 라이브러리
- Scikit-learn
- XGBoost / LightGBM / CatBoost
- Pandas / NumPy

### 데이터 처리
- 데이터 파싱 및 전처리
- Feature Engineering
- 컬럼 평가 및 선택

### 모델링 접근
- Baseline 모델 구축
- 부스팅 앙상블
- 하이퍼파라미터 튜닝
- 교차 검증

## 📊 주요 기능

### 1. 데이터 전처리
- **data_watching.py**: 데이터 탐색 및 시각화
- **dataparsing.py**: 데이터 파싱 및 정제
- **make_enriched.py**: 피처 엔지니어링
- **column_eval.py**: 중요 컬럼 평가

### 2. 모델 개발
- **baseline.py**: 기본 모델 구축
- **model.py (v1, v2)**: 개선된 모델 버전
- **model_without_boost.py**: 부스팅 제외 실험
- **dacon_baseline (1-7)**: 다양한 베이스라인 실험

### 3. 자동화
- **all_in_one.py**: 전체 파이프라인 통합
- **run_until_good.py**: 목표 성능 달성까지 자동 실행

## 🚀 시작하기

### 필요 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 데이터 준비

```bash
# 데이터 파싱
python codes/00_dataparsing.py

# 데이터 확인
python codes/00_data_watching.py
```

### 모델 학습

```bash
# 베이스라인 모델
python codes/01_baseline.py

# 최신 버전 모델
python codes/03_ver2.py

# 자동 실행
python codes/run_until_good.py
```

## 📈 성능 기록

| 버전 | 스코어 | 설명 |
|------|--------|------|
| v1   | 0.34972 | 초기 베이스라인 |
| v2   | 0.34975 | 피처 개선 |
| v3   | **0.34991** | 최고 성능 (Best) |

## 🔍 주요 실험

### Dacon Baseline Series
- **Baseline 1-2**: 기본 전처리 및 모델링
- **Baseline 3**: 최적화된 버전 (Don't touch)
- **Baseline 4-7**: 다양한 피처 및 앙상블 실험

### Jaewoo Versions
- **v0**: 초기 접근법
- **v1**: 개선된 피처 엔지니어링

## 📝 프로젝트 타임라인

- **9월 12일 ~ 9월 15일**: 데이터 전처리 및 탐색
- **9월 16일 ~ 9월 20일**: 베이스라인 모델 개발
- **9월 21일 ~ 9월 24일**: Dacon 베이스라인 실험
- **9월 25일 ~ 9월 29일**: 최종 모델 최적화 및 제출

## 💡 주요 학습 내용

- 데이터 탐색 및 시각화의 중요성
- 피처 엔지니어링을 통한 성능 개선
- 다양한 베이스라인 비교 실험
- 앙상블 모델의 효과

## 📁 결과물

- `0.34991/`: 최고 성능 모델 결과물
- `submissions/`: 대회 제출 파일
- `log/`: 실험 로그 및 기록

## 🛠️ 개발 환경

- Python 3.8+
- Pandas, NumPy
- Scikit-learn
- XGBoost, LightGBM, CatBoost

## 👨‍💻 개발자

Jaehyeon Kim
