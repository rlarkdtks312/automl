# 자동화 예측 모델 성능 평가 및 최적화 모듈

머신러닝 모델의 **성능 평가**, **하이퍼파라미터 최적화**, **결과 시각화 및 리포트 생성**을 자동화하는 Python 모듈입니다.

분류(Classification)와 회귀(Regression) 모두 지원하며, 개별 기능을 직접 사용하거나 `AutoMLPipeline`으로 전체 워크플로우를 한 번에 실행할 수 있습니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **모델 목록 조회** | 사용 가능한 전체 모델과 파라미터 프리셋 정보를 표로 출력 |
| **모델 평가** | 분류/회귀 메트릭 계산, K-Fold 교차검증, 학습/검증 곡선 분석 |
| **모델 비교** | 여러 모델을 교차검증 기반으로 한 번에 비교 |
| **하이퍼파라미터 최적화** | GridSearch / RandomSearch / Bayesian Optimization 지원 |
| **시각화** | 혼동행렬, 모델 비교 차트, 학습 곡선, 피처 중요도 |
| **리포트 생성** | JSON / CSV / HTML 형식 결과 저장 |
| **완전 자동화** | `AutoMLPipeline.run()` 한 줄로 비교 → 최적화 → 평가 완료 |

---

## 설치

### 1. 저장소 클론

```bash
git clone https://github.com/<your-username>/automl-pipeline.git
cd automl-pipeline
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 파일 구조

```
automl-pipeline/
├── automl_pipeline.py       # 메인 모듈
├── example_usage.ipynb      # 사용 예제 (Jupyter Notebook)
├── requirements.txt         # 의존 패키지
└── reports/                 # 결과 저장 폴더 (자동 생성)
    ├── classification/
    ├── regression/
    └── automl/
```

---

## 빠른 시작

### 사용 가능한 모델 목록 확인

모듈에 내장된 모든 모델과 파라미터 프리셋 유무를 확인합니다.
XGBoost / LightGBM은 설치 여부에 따라 자동으로 포함됩니다.

```python
from automl_pipeline import list_available_models

# 전체 모델 목록 출력 (설치 여부 + 파라미터 프리셋 상세 포함)
df = list_available_models(verbose=True)

# 태스크별 필터링
list_available_models(task_type="classification")  # 분류 모델만
list_available_models(task_type="regression")      # 회귀 모델만
```

### 모델 카탈로그에서 모델 선택

`_build_model_catalog()`로 사전 구성된 모델 인스턴스를 가져와 원하는 모델만 골라 사용합니다.

```python
import copy
from automl_pipeline import _build_model_catalog

catalog = _build_model_catalog()  # 설치된 모델 전체를 딕셔너리로 반환

# 원하는 모델 이름으로 선택 (깊은 복사로 독립 인스턴스 생성)
models = {
    name: copy.deepcopy(catalog[name])
    for name in ["RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier"]
}
```

### 분류 모델 비교

여러 분류 모델을 교차검증으로 한 번에 비교하고 성능 지표를 DataFrame으로 확인합니다.

```python
from automl_pipeline import ModelEvaluator
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

evaluator = ModelEvaluator(task_type="classification")

# 5-Fold 교차검증 기반 모델 비교 → DataFrame 반환
comparison = evaluator.compare_models(models, X_train, y_train, n_splits=5)
print(comparison)
```

### 하이퍼파라미터 최적화

`PARAM_SPACES`에 등록된 모델은 파라미터 프리셋이 내장되어 있어 별도 정의 없이 바로 최적화가 가능합니다.

```python
from automl_pipeline import HyperparameterOptimizer, PARAM_SPACES

model = catalog["RandomForestClassifier"]

# 파라미터 프리셋 자동 로드 (직접 정의도 가능)
param_space = PARAM_SPACES["RandomForestClassifier"]

# RandomSearch로 20회 탐색, f1_weighted 기준 최적화
optimizer = HyperparameterOptimizer(strategy="random", scoring="f1_weighted", cv=5, n_iter=20)
result = optimizer.optimize(model, X_train, y_train, param_space)

print("최적 파라미터:", result.best_params)
print(f"최고 CV 점수: {result.best_score:.4f}")

# 최적 파라미터가 적용된 모델로 바로 예측 가능
best_model = result.best_model
```

### AutoMLPipeline — 완전 자동화

모델 비교 → 최적 모델 선정 → 하이퍼파라미터 최적화 → 최종 평가 → 리포트 저장을 한 번에 실행합니다.

```python
from automl_pipeline import AutoMLPipeline, _build_model_catalog
import copy

# 비교할 모델 준비
catalog = _build_model_catalog()
models = {
    name: copy.deepcopy(catalog[name])
    for name in ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"]
}

pipeline = AutoMLPipeline(
    task_type="classification",
    cv_folds=5,
    opt_strategy="random",  # "grid" / "random" / "bayesian" 선택 가능
    opt_iter=50,
    output_dir="./reports/automl",
)

# 비교 → 최적화 → 평가를 순서대로 자동 실행
result = pipeline.run(models=models, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
print(result.summary())
```

---

## 클래스 및 함수 설명

> 각 클래스와 함수의 파라미터, 메서드, 반환값을 정리한 사용 가이드입니다.

### `list_available_models()`

사용 가능한 모델 목록과 파라미터 프리셋 정보를 반환합니다.

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `task_type` | `"classification"` / `"regression"` / `None` (전체) | `None` |
| `verbose` | 콘솔에 상세 정보 출력 여부 | `True` |

반환 DataFrame 컬럼: **태스크 / 설치 여부 / 파라미터 프리셋 / 파라미터 수 / 전체 조합 수**

---

### `ModelEvaluator`

모델 성능을 평가하는 클래스입니다.

| 메서드 | 설명 |
|--------|------|
| `evaluate(y_true, y_pred, y_prob, model_name)` | 단일 평가 → `EvaluationResult` 반환 |
| `cross_validate(model, X, y, n_splits)` | K-Fold 교차검증 → `CVResult` 반환 |
| `compare_models(models, X, y, n_splits)` | 여러 모델 비교 → DataFrame 반환 |
| `learning_curve_data(model, X, y)` | 학습 곡선 데이터 반환 |
| `validation_curve_data(model, X, y, param_name, param_range)` | 검증 곡선 데이터 반환 |

**지원 메트릭**

- **분류:** Accuracy, Precision, Recall, F1, ROC-AUC
- **회귀:** MAE, MSE, RMSE, R²

---

### `HyperparameterOptimizer`

하이퍼파라미터를 자동으로 탐색하는 클래스입니다.

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `strategy` | `"grid"` / `"random"` / `"bayesian"` | `"random"` |
| `scoring` | 최적화 기준 메트릭 | `"f1_weighted"` |
| `cv` | 교차검증 폴드 수 | `5` |
| `n_iter` | 탐색 횟수 (random/bayesian) | `50` |

| 메서드 | 설명 |
|--------|------|
| `optimize(model, X, y, param_space, model_name)` | 단일 모델 최적화 → `OptimizationResult` 반환 |
| `multi_model_optimize(models, X, y, param_spaces)` | 여러 모델 동시 최적화 |

**파라미터 프리셋 (`PARAM_SPACES`)**

별도 정의 없이 바로 사용 가능한 모델:

| 태스크 | 모델 |
|--------|------|
| 분류 | `RandomForestClassifier`, `ExtraTreesClassifier`, `GradientBoostingClassifier`, `XGBClassifier`*, `LGBMClassifier`*, `LogisticRegression`, `SVC`, `DecisionTreeClassifier`, `KNeighborsClassifier`, `AdaBoostClassifier` |
| 회귀 | `RandomForestRegressor`, `ExtraTreesRegressor`, `GradientBoostingRegressor`, `XGBRegressor`*, `LGBMRegressor`*, `Ridge`, `Lasso`, `ElasticNet`, `KNeighborsRegressor`, `DecisionTreeRegressor` |

> \* xgboost / lightgbm 설치 필요

---

### `AutoMLPipeline`

평가 → 최적화 → 최종 평가를 자동으로 수행하는 통합 파이프라인입니다.

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `task_type` | `"classification"` / `"regression"` | 자동 감지 |
| `cv_folds` | 교차검증 폴드 수 | `5` |
| `opt_strategy` | 최적화 전략 | `"random"` |
| `opt_iter` | 최적화 반복 횟수 | `50` |
| `output_dir` | 결과 저장 경로 | `"./reports"` |
| `random_state` | 재현성 시드 | `42` |

| 메서드 | 설명 |
|--------|------|
| `run(models, X_train, y_train, X_test, y_test, optimize)` | 전체 파이프라인 실행 → `PipelineResult` 반환 |

---

### `PerformanceReporter`

결과를 저장하고 시각화하는 클래스입니다.

| 메서드 | 설명 |
|--------|------|
| `save_evaluation(result)` | 평가 결과를 JSON으로 저장 |
| `save_cv_results(result)` | 교차검증 결과를 JSON으로 저장 |
| `save_optimization(result)` | 최적화 결과를 JSON으로 저장 |
| `plot_confusion_matrix(result, class_labels, save)` | 혼동행렬 시각화 |
| `plot_model_comparison(df, metrics, save)` | 모델 비교 차트 |
| `plot_learning_curve(result, save)` | 학습 곡선 |
| `plot_feature_importance(model, feature_names, save)` | 피처 중요도 |
| `generate_html_report(pipeline_result)` | HTML 종합 리포트 생성 |

---

## 의존 패키지

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
xgboost>=1.7
lightgbm>=3.3
scikit-optimize>=0.9
```

---

## 라이선스

MIT License
