"""
automl_pipeline — 자동화 예측 모델 성능 평가 및 최적화

포함 클래스:
    - ModelEvaluator          : 모델 성능 평가 (분류 / 회귀 메트릭, 교차검증)
    - HyperparameterOptimizer : 하이퍼파라미터 최적화 (GridSearch / RandomSearch / Bayesian)
    - AutoMLPipeline          : 평가 + 최적화 통합 파이프라인
    - PerformanceReporter     : 결과 리포트 생성 및 시각화

포함 함수:
    - list_available_models() : 사용 가능한 모델 목록 및 파라미터 정보 반환
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════
# 표준 / 서드파티 임포트
# ══════════════════════════════════════════════════════════════

import datetime
import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    learning_curve,
    validation_curve,
)

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """모듈 전용 로거를 반환합니다."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class Timer:
    """코드 블록 실행 시간 측정용 컨텍스트 매니저."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start

    def __str__(self):
        return f"{self.elapsed:.3f}s"


def validate_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true({len(y_true)})와 y_pred({len(y_pred)})의 길이가 다릅니다."
        )
    if len(y_true) == 0:
        raise ValueError("입력 배열이 비어 있습니다.")


def infer_task_type(y: np.ndarray) -> str:
    if y.dtype.kind in ("U", "O", "b"):
        return "classification"
    unique_ratio = len(np.unique(y)) / len(y)
    if np.issubdtype(y.dtype, np.integer) and unique_ratio < 0.05:
        return "classification"
    return "regression"


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    raise TypeError(f"직렬화 불가: {type(obj)}")


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


_logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# ModelEvaluator
# ══════════════════════════════════════════════════════════════

@dataclass
class EvaluationResult:
    """단일 평가 실행 결과."""

    task_type: Literal["classification", "regression"]
    metrics: dict[str, float]
    elapsed_sec: float
    model_name: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        row = {"model": self.model_name, "task": self.task_type, **self.metrics}
        return pd.DataFrame([row])

    def summary(self) -> str:
        lines = [f"[{self.model_name or 'Model'}] {self.task_type} 평가 ({self.elapsed_sec:.2f}s)"]
        for k, v in self.metrics.items():
            lines.append(f"  {k:<20}: {v:.4f}")
        return "\n".join(lines)


@dataclass
class CVResult:
    """교차검증 결과."""

    task_type: Literal["classification", "regression"]
    cv_scores: dict[str, np.ndarray]
    elapsed_sec: float
    n_splits: int
    model_name: str = ""

    @property
    def mean_scores(self) -> dict[str, float]:
        return {k: float(v.mean()) for k, v in self.cv_scores.items()}

    @property
    def std_scores(self) -> dict[str, float]:
        return {k: float(v.std()) for k, v in self.cv_scores.items()}

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for fold_idx in range(self.n_splits):
            row = {"model": self.model_name, "fold": fold_idx + 1}
            for metric, scores in self.cv_scores.items():
                row[metric] = scores[fold_idx]
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = [
            f"[{self.model_name or 'Model'}] {self.n_splits}-Fold CV "
            f"({self.task_type}, {self.elapsed_sec:.2f}s)"
        ]
        for metric in self.cv_scores:
            mean = self.mean_scores[metric]
            std = self.std_scores[metric]
            lines.append(f"  {metric:<25}: {mean:.4f} ± {std:.4f}")
        return "\n".join(lines)


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    average: str = "weighted",
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    if y_prob is not None:
        try:
            classes = np.unique(y_true)
            if len(classes) == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim == 2 else y_prob)
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
            metrics["roc_auc"] = auc
        except Exception as e:
            _logger.warning(f"AUC-ROC 계산 실패: {e}")
    return metrics


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


class ModelEvaluator:
    """예측 모델 성능 평가기.

    Parameters
    ----------
    task_type : 'auto' | 'classification' | 'regression'
    average   : 분류 메트릭 다중클래스 집계 방식 (default: 'weighted')
    """

    def __init__(
        self,
        task_type: Literal["auto", "classification", "regression"] = "auto",
        average: str = "weighted",
    ):
        self.task_type = task_type
        self.average = average

    def evaluate(
        self,
        y_true: np.ndarray | list,
        y_pred: np.ndarray | list,
        y_prob: np.ndarray | None = None,
        model_name: str = "",
    ) -> EvaluationResult:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        validate_arrays(y_true, y_pred)

        task = self._resolve_task(y_true)
        extra: dict[str, Any] = {}
        with Timer() as t:
            if task == "classification":
                metrics = _classification_metrics(y_true, y_pred, y_prob, self.average)
                extra["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
            else:
                metrics = _regression_metrics(y_true, y_pred)

        result = EvaluationResult(
            task_type=task, metrics=metrics, elapsed_sec=t.elapsed,
            model_name=model_name, extra=extra,
        )
        _logger.info(result.summary())
        return result

    def cross_validate(
        self,
        model: Any,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        model_name: str = "",
    ) -> CVResult:
        y_arr = np.asarray(y)
        task = self._resolve_task(y_arr)

        if task == "classification":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            scoring = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
        else:
            cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            scoring = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]

        _logger.info(f"[{model_name or model.__class__.__name__}] {n_splits}-Fold CV 시작...")
        with Timer() as t:
            raw = cross_validate(model, X, y_arr, cv=cv, scoring=scoring, n_jobs=-1)

        cv_scores: dict[str, np.ndarray] = {}
        for key, vals in raw.items():
            if not key.startswith("test_"):
                continue
            metric = key[len("test_"):]
            if metric.startswith("neg_"):
                metric = metric[4:]
                vals = -vals
            cv_scores[metric] = vals

        result = CVResult(
            task_type=task, cv_scores=cv_scores, elapsed_sec=t.elapsed,
            n_splits=n_splits, model_name=model_name or model.__class__.__name__,
        )
        _logger.info(result.summary())
        return result

    def learning_curve_data(
        self,
        model: Any,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        train_sizes: np.ndarray | None = None,
        n_splits: int = 5,
        scoring: str | None = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        y_arr = np.asarray(y)
        task = self._resolve_task(y_arr)
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        if scoring is None:
            scoring = "f1_weighted" if task == "classification" else "r2"
        cv = (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            if task == "classification"
            else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        )
        sizes, train_scores, val_scores = learning_curve(
            model, X, y_arr, train_sizes=train_sizes, cv=cv, scoring=scoring, n_jobs=-1
        )
        rows = []
        for i, size in enumerate(sizes):
            rows.append({
                "train_size": int(size),
                f"train_{scoring}_mean": train_scores[i].mean(),
                f"train_{scoring}_std": train_scores[i].std(),
                f"val_{scoring}_mean": val_scores[i].mean(),
                f"val_{scoring}_std": val_scores[i].std(),
            })
        return pd.DataFrame(rows)

    def validation_curve_data(
        self,
        model: Any,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        param_name: str,
        param_range: list | np.ndarray,
        n_splits: int = 5,
        scoring: str | None = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        y_arr = np.asarray(y)
        task = self._resolve_task(y_arr)
        if scoring is None:
            scoring = "f1_weighted" if task == "classification" else "r2"
        cv = (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            if task == "classification"
            else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        )
        train_scores, val_scores = validation_curve(
            model, X, y_arr, param_name=param_name, param_range=param_range,
            cv=cv, scoring=scoring, n_jobs=-1,
        )
        rows = []
        for i, val in enumerate(param_range):
            rows.append({
                param_name: val,
                f"train_{scoring}_mean": train_scores[i].mean(),
                f"train_{scoring}_std": train_scores[i].std(),
                f"val_{scoring}_mean": val_scores[i].mean(),
                f"val_{scoring}_std": val_scores[i].std(),
            })
        return pd.DataFrame(rows)

    def compare_models(
        self,
        models: dict[str, Any],
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> pd.DataFrame:
        rows = []
        for name, model in models.items():
            cv_result = self.cross_validate(
                model, X, y, n_splits=n_splits,
                random_state=random_state, model_name=name
            )
            row = {"model": name}
            row.update(cv_result.mean_scores)
            rows.append(row)
        df = pd.DataFrame(rows).set_index("model")
        _logger.info(f"\n[모델 비교 결과]\n{df.to_string()}")
        return df

    def _resolve_task(self, y: np.ndarray) -> Literal["classification", "regression"]:
        if self.task_type == "auto":
            return infer_task_type(y)
        return self.task_type


# ══════════════════════════════════════════════════════════════
# HyperparameterOptimizer
# ══════════════════════════════════════════════════════════════

@dataclass
class OptimizationResult:
    """최적화 실행 결과."""

    strategy: str
    best_params: dict[str, Any]
    best_score: float
    best_model: Any
    cv_results: pd.DataFrame
    elapsed_sec: float
    scoring: str
    model_name: str = ""

    def summary(self) -> str:
        lines = [
            f"[{self.model_name or 'Model'}] {self.strategy} 최적화 완료 ({self.elapsed_sec:.2f}s)",
            f"  최고 점수 ({self.scoring}): {self.best_score:.4f}",
            "  최적 파라미터:",
        ]
        for k, v in self.best_params.items():
            lines.append(f"    {k}: {v}")
        return "\n".join(lines)


PARAM_SPACES: dict[str, dict[str, Any]] = {
    # ── 분류 ──────────────────────────────────────────────────
    "RandomForestClassifier": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    "ExtraTreesClassifier": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2"],
    },
    "GradientBoostingClassifier": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
    },
    "XGBClassifier": {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 6, 9],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
    },
    "LGBMClassifier": {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [-1, 5, 10],
        "num_leaves": [31, 63, 127],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [0, 0.1, 1.0],
    },
    "SVC": {
        "C": [0.01, 0.1, 1, 10, 100],
        "kernel": ["rbf", "linear", "poly"],
        "gamma": ["scale", "auto"],
    },
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "saga"],
        "max_iter": [100, 500, 1000],
    },
    "DecisionTreeClassifier": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    },
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "AdaBoostClassifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
    },
    # ── 회귀 ──────────────────────────────────────────────────
    "RandomForestRegressor": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "ExtraTreesRegressor": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    "GradientBoostingRegressor": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
    },
    "XGBRegressor": {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 6, 9],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
    },
    "LGBMRegressor": {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [-1, 5, 10],
        "num_leaves": [31, 63, 127],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [0, 0.1, 1.0],
    },
    "Ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "Lasso": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "max_iter": [1000, 5000],
    },
    "ElasticNet": {
        "alpha": [0.001, 0.01, 0.1, 1.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "max_iter": [1000, 5000],
    },
    "KNeighborsRegressor": {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "DecisionTreeRegressor": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["squared_error", "absolute_error"],
    },
}


def get_default_param_space(model: Any) -> dict[str, Any]:
    name = model.__class__.__name__
    if name not in PARAM_SPACES:
        raise KeyError(
            f"'{name}'에 대한 파라미터 프리셋이 없습니다. "
            f"지원 모델: {list(PARAM_SPACES.keys())}"
        )
    return PARAM_SPACES[name]


def _to_bayesian_space(param_space: dict[str, Any]) -> dict[str, Any]:
    """GridSearch/RandomSearch용 리스트 형식을 BayesSearchCV용 skopt 타입으로 변환.

    - 숫자형 리스트 → Real 또는 Integer
    - 문자열/None 포함 리스트 → Categorical (None은 문자열 'None'으로 변환)
    skopt 미설치 시 원본 dict 반환 (ImportError는 호출부에서 처리).
    """
    try:
        from skopt.space import Categorical, Integer, Real
    except ImportError:
        return param_space

    bayes_space: dict[str, Any] = {}
    for key, values in param_space.items():
        if not isinstance(values, list) or len(values) == 0:
            bayes_space[key] = values
            continue

        # None을 포함하거나 문자열이 있으면 Categorical
        has_none = any(v is None for v in values)
        has_str  = any(isinstance(v, str) for v in values)
        if has_none or has_str:
            safe = [str(v) if v is None else v for v in values]
            bayes_space[key] = Categorical(safe)
            continue

        nums = [v for v in values if isinstance(v, (int, float))]
        if not nums:
            bayes_space[key] = Categorical(values)
            continue

        lo, hi = min(nums), max(nums)
        if lo == hi:
            bayes_space[key] = Categorical(values)
        elif all(isinstance(v, int) for v in nums):
            bayes_space[key] = Integer(lo, hi)
        else:
            bayes_space[key] = Real(lo, hi, prior="log-uniform" if lo > 0 else "uniform")

    return bayes_space


# ══════════════════════════════════════════════════════════════
# 모델 카탈로그
# ══════════════════════════════════════════════════════════════

def _build_model_catalog() -> dict[str, Any]:
    """사용 가능한 모델 인스턴스 딕셔너리를 반환합니다 (내부용).

    xgboost / lightgbm 은 설치된 경우에만 목록에 포함됩니다.
    """
    from sklearn.ensemble import (
        AdaBoostClassifier,
        ExtraTreesClassifier, ExtraTreesRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
        RandomForestClassifier, RandomForestRegressor,
    )
    from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    catalog: dict[str, Any] = {
        # ── 분류 ──────────────────────────────────────────────
        "RandomForestClassifier":     RandomForestClassifier(random_state=42),
        "ExtraTreesClassifier":       ExtraTreesClassifier(random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        "LogisticRegression":         LogisticRegression(max_iter=1000, random_state=42),
        "SVC":                        SVC(probability=True, random_state=42),
        "DecisionTreeClassifier":     DecisionTreeClassifier(random_state=42),
        "KNeighborsClassifier":       KNeighborsClassifier(),
        "AdaBoostClassifier":         AdaBoostClassifier(random_state=42),
        # ── 회귀 ──────────────────────────────────────────────
        "RandomForestRegressor":      RandomForestRegressor(random_state=42),
        "ExtraTreesRegressor":        ExtraTreesRegressor(random_state=42),
        "GradientBoostingRegressor":  GradientBoostingRegressor(random_state=42),
        "Ridge":                      Ridge(),
        "Lasso":                      Lasso(),
        "ElasticNet":                 ElasticNet(),
        "KNeighborsRegressor":        KNeighborsRegressor(),
        "DecisionTreeRegressor":      DecisionTreeRegressor(random_state=42),
    }

    # ── 선택 설치 패키지 ──────────────────────────────────────
    try:
        from xgboost import XGBClassifier, XGBRegressor
        catalog["XGBClassifier"] = XGBClassifier(
            random_state=42, eval_metric="logloss", verbosity=0
        )
        catalog["XGBRegressor"] = XGBRegressor(
            random_state=42, verbosity=0
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        catalog["LGBMClassifier"] = LGBMClassifier(random_state=42, verbosity=-1)
        catalog["LGBMRegressor"]  = LGBMRegressor(random_state=42, verbosity=-1)
    except ImportError:
        pass

    return catalog


_TASK_MAP: dict[str, str] = {
    # 분류
    "RandomForestClassifier":     "classification",
    "ExtraTreesClassifier":       "classification",
    "GradientBoostingClassifier": "classification",
    "XGBClassifier":              "classification",
    "LGBMClassifier":             "classification",
    "LogisticRegression":         "classification",
    "SVC":                        "classification",
    "DecisionTreeClassifier":     "classification",
    "KNeighborsClassifier":       "classification",
    "AdaBoostClassifier":         "classification",
    # 회귀
    "RandomForestRegressor":      "regression",
    "ExtraTreesRegressor":        "regression",
    "GradientBoostingRegressor":  "regression",
    "XGBRegressor":               "regression",
    "LGBMRegressor":              "regression",
    "Ridge":                      "regression",
    "Lasso":                      "regression",
    "ElasticNet":                 "regression",
    "KNeighborsRegressor":        "regression",
    "DecisionTreeRegressor":      "regression",
}


def list_available_models(
    task_type: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """사용 가능한 모델 목록과 각 모델의 파라미터 공간 정보를 반환합니다.

    Parameters
    ----------
    task_type : 'classification' | 'regression' | None
        None이면 전체 모델을 반환합니다.
    verbose : bool
        True이면 콘솔에 요약 테이블과 파라미터 상세 정보를 출력합니다.

    Returns
    -------
    pd.DataFrame
        모델명 / 태스크 / 파라미터 프리셋 여부 / 파라미터 수를 담은 DataFrame.

    Examples
    --------
    >>> df = list_available_models()
    >>> df = list_available_models(task_type='classification')
    """
    installed = set(_build_model_catalog().keys())
    rows = []
    for name, task in _TASK_MAP.items():
        if task_type is not None and task != task_type:
            continue
        has_space = name in PARAM_SPACES
        n_params  = len(PARAM_SPACES[name]) if has_space else 0
        total_candidates = 1
        if has_space:
            for vals in PARAM_SPACES[name].values():
                if isinstance(vals, list):
                    total_candidates *= len(vals)

        rows.append({
            "모델명":               name,
            "태스크":               "분류" if task == "classification" else "회귀",
            "설치 여부":            "✅" if name in installed else "⚠️ 미설치",
            "파라미터 프리셋":      "✅" if has_space else "❌",
            "파라미터 수":          n_params,
            "전체 조합 수 (Grid)":  total_candidates if has_space else "-",
        })

    df = pd.DataFrame(rows).set_index("모델명")

    if verbose:
        task_label = {"classification": "분류", "regression": "회귀"}.get(task_type, "전체")
        print(f"{'=' * 60}")
        print(f"  사용 가능한 모델 목록  [{task_label}]")
        print(f"{'=' * 60}")
        print(df.to_string())
        not_installed = [r["모델명"] for r in rows if r["설치 여부"] != "✅"]
        if not_installed:
            print(f"\n⚠️  미설치 모델: {not_installed}")
            print("   pip install xgboost lightgbm 으로 설치하면 사용 가능합니다.")
        print()

        for name in df.index:
            if name not in PARAM_SPACES:
                print(f"  [{name}]  파라미터 프리셋 없음\n")
                continue
            print(f"  [{name}]")
            for param, candidates in PARAM_SPACES[name].items():
                print(f"    {param:<25}: {candidates}")
            print()

    return df


class HyperparameterOptimizer:
    """하이퍼파라미터 최적화 통합 인터페이스.

    Parameters
    ----------
    strategy     : 'grid' | 'random' | 'bayesian'
    scoring      : 최적화 기준 메트릭 (sklearn scoring string)
    cv           : 교차검증 폴드 수
    n_iter       : random / bayesian 탐색 횟수
    n_jobs       : 병렬 실행 수 (-1 = 전체 코어 사용)
    random_state : 재현성 시드
    """

    def __init__(
        self,
        strategy: Literal["grid", "random", "bayesian"] = "random",
        scoring: str = "f1_weighted",
        cv: int = 5,
        n_iter: int = 50,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 1,
    ):
        self.strategy = strategy
        self.scoring = scoring
        self.cv = cv
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def optimize(
        self,
        model: Any,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        param_space: dict[str, Any],
        model_name: str = "",
    ) -> OptimizationResult:
        name = model_name or model.__class__.__name__
        _logger.info(f"[{name}] {self.strategy.upper()} 탐색 시작 (scoring={self.scoring})")

        with Timer() as t:
            if self.strategy == "grid":
                result = self._grid_search(model, X, y, param_space)
            elif self.strategy == "random":
                result = self._random_search(model, X, y, param_space)
            elif self.strategy == "bayesian":
                result = self._bayesian_search(model, X, y, param_space)
            else:
                raise ValueError(f"지원하지 않는 전략: {self.strategy}")

        best_params, best_score, best_estimator, cv_df = result
        opt_result = OptimizationResult(
            strategy=self.strategy, best_params=best_params, best_score=best_score,
            best_model=best_estimator, cv_results=cv_df, elapsed_sec=t.elapsed,
            scoring=self.scoring, model_name=name,
        )
        _logger.info(opt_result.summary())
        return opt_result

    def multi_model_optimize(
        self,
        models_params: dict[str, tuple[Any, dict]],
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> pd.DataFrame:
        rows = []
        self._opt_results: dict[str, OptimizationResult] = {}
        for name, (model, param_space) in models_params.items():
            res = self.optimize(model, X, y, param_space, model_name=name)
            self._opt_results[name] = res
            row = {"model": name, f"best_{self.scoring}": res.best_score}
            row.update({f"param_{k}": v for k, v in res.best_params.items()})
            rows.append(row)
        df = pd.DataFrame(rows).set_index("model")
        _logger.info(f"\n[다중 모델 최적화 결과]\n{df.to_string()}")
        return df

    def _grid_search(self, model, X, y, param_grid):
        searcher = GridSearchCV(
            estimator=model, param_grid=param_grid, scoring=self.scoring,
            cv=self.cv, n_jobs=self.n_jobs, verbose=self.verbose, refit=True,
        )
        searcher.fit(X, y)
        return searcher.best_params_, searcher.best_score_, searcher.best_estimator_, pd.DataFrame(searcher.cv_results_)

    def _random_search(self, model, X, y, param_distributions):
        searcher = RandomizedSearchCV(
            estimator=model, param_distributions=param_distributions, n_iter=self.n_iter,
            scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs,
            random_state=self.random_state, verbose=self.verbose, refit=True,
        )
        searcher.fit(X, y)
        return searcher.best_params_, searcher.best_score_, searcher.best_estimator_, pd.DataFrame(searcher.cv_results_)

    def _bayesian_search(self, model, X, y, param_space):
        try:
            from skopt import BayesSearchCV
        except ImportError:
            warnings.warn(
                "scikit-optimize가 설치되어 있지 않습니다. "
                "RandomizedSearch로 대체합니다. (pip install scikit-optimize)"
            )
            return self._random_search(model, X, y, param_space)
        searcher = BayesSearchCV(
            estimator=model, search_spaces=_to_bayesian_space(param_space), n_iter=self.n_iter,
            scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs,
            random_state=self.random_state, verbose=self.verbose, refit=True,
        )
        searcher.fit(X, y)
        return searcher.best_params_, searcher.best_score_, searcher.best_estimator_, pd.DataFrame(searcher.cv_results_)


# ══════════════════════════════════════════════════════════════
# PerformanceReporter
# ══════════════════════════════════════════════════════════════

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]


class PerformanceReporter:
    """평가·최적화 결과를 파일과 시각화로 정리하는 리포터.

    Parameters
    ----------
    output_dir : 결과물 저장 경로 (기본값: './reports')
    """

    def __init__(self, output_dir: str | Path = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_evaluation(self, result: EvaluationResult, prefix: str = "") -> Path:
        tag = f"{prefix}_" if prefix else ""
        path = self.output_dir / f"{tag}evaluation_{self._timestamp}.json"
        save_json({
            "model_name": result.model_name, "task_type": result.task_type,
            "elapsed_sec": result.elapsed_sec, "metrics": result.metrics, "extra": result.extra,
        }, path)
        _logger.info(f"평가 결과 저장: {path}")
        return path

    def save_cv_results(self, result: CVResult, prefix: str = "") -> Path:
        tag = f"{prefix}_" if prefix else ""
        base = self.output_dir / f"{tag}cv_{self._timestamp}"
        result.to_dataframe().to_csv(f"{base}.csv", index=False, encoding="utf-8-sig")
        save_json({
            "model_name": result.model_name, "task_type": result.task_type,
            "n_splits": result.n_splits, "elapsed_sec": result.elapsed_sec,
            "mean_scores": result.mean_scores, "std_scores": result.std_scores,
        }, f"{base}_summary.json")
        _logger.info(f"CV 결과 저장: {base}.csv")
        return Path(f"{base}.csv")

    def save_optimization(self, result: OptimizationResult, prefix: str = "") -> Path:
        tag = f"{prefix}_" if prefix else ""
        base = self.output_dir / f"{tag}optimization_{self._timestamp}"
        result.cv_results.to_csv(f"{base}_cv.csv", index=False, encoding="utf-8-sig")
        save_json({
            "model_name": result.model_name, "strategy": result.strategy,
            "scoring": result.scoring, "best_score": result.best_score,
            "best_params": result.best_params, "elapsed_sec": result.elapsed_sec,
        }, f"{base}_summary.json")
        _logger.info(f"최적화 결과 저장: {base}_summary.json")
        return Path(f"{base}_summary.json")

    def plot_confusion_matrix(
        self,
        result: EvaluationResult,
        class_labels: list[str] | None = None,
        save: bool = True,
        show: bool = False,
    ) -> plt.Figure:
        if "confusion_matrix" not in result.extra:
            raise ValueError("EvaluationResult에 confusion_matrix 정보가 없습니다.")
        cm = np.array(result.extra["confusion_matrix"])
        n = cm.shape[0]
        labels = class_labels or [str(i) for i in range(n)]
        fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("예측 레이블")
        ax.set_ylabel("실제 레이블")
        ax.set_title(f"{result.model_name or 'Model'} — 혼동행렬")
        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        if save:
            path = self.output_dir / f"confusion_matrix_{self._timestamp}.png"
            fig.savefig(path, dpi=150)
            _logger.info(f"혼동행렬 저장: {path}")
        if show:
            plt.show()
        return fig

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: list[str] | None = None,
        save: bool = True,
        show: bool = False,
    ) -> plt.Figure:
        metrics = metrics or list(comparison_df.columns)
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=False)
        if n_metrics == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            vals = comparison_df[metric]
            colors = _PALETTE[: len(vals)]
            bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="white")
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(comparison_df.index, rotation=30, ha="right")
            ax.set_title(metric)
            y_max = vals.max() * 1.15 if vals.max() > 1.0 else min(1.1, vals.max() * 1.15)
            ax.set_ylim(0, y_max)
            label_offset = y_max * 0.01
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + label_offset,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        fig.suptitle("모델 성능 비교", fontsize=14, fontweight="bold")
        fig.tight_layout()
        if save:
            path = self.output_dir / f"model_comparison_{self._timestamp}.png"
            fig.savefig(path, dpi=150)
            _logger.info(f"모델 비교 그래프 저장: {path}")
        if show:
            plt.show()
        return fig

    def plot_learning_curve(
        self,
        lc_df: pd.DataFrame,
        save: bool = True,
        show: bool = False,
        title: str = "학습 곡선",
    ) -> plt.Figure:
        train_mean_col = [c for c in lc_df.columns if c.startswith("train_") and c.endswith("_mean")][0]
        val_mean_col   = [c for c in lc_df.columns if c.startswith("val_") and c.endswith("_mean")][0]
        train_std_col  = train_mean_col.replace("_mean", "_std")
        val_std_col    = val_mean_col.replace("_mean", "_std")
        sizes = lc_df["train_size"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sizes, lc_df[train_mean_col], label="Train", color=_PALETTE[0], marker="o")
        ax.fill_between(sizes, lc_df[train_mean_col] - lc_df[train_std_col],
                         lc_df[train_mean_col] + lc_df[train_std_col], alpha=0.15, color=_PALETTE[0])
        ax.plot(sizes, lc_df[val_mean_col], label="Validation", color=_PALETTE[1], marker="s")
        ax.fill_between(sizes, lc_df[val_mean_col] - lc_df[val_std_col],
                         lc_df[val_mean_col] + lc_df[val_std_col], alpha=0.15, color=_PALETTE[1])
        ax.set_xlabel("학습 데이터 수")
        ax.set_ylabel(train_mean_col.split("_")[1])
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        if save:
            path = self.output_dir / f"learning_curve_{self._timestamp}.png"
            fig.savefig(path, dpi=150)
            _logger.info(f"학습 곡선 저장: {path}")
        if show:
            plt.show()
        return fig

    def plot_feature_importance(
        self,
        importances: np.ndarray | pd.Series,
        feature_names: list[str] | None = None,
        top_n: int = 20,
        save: bool = True,
        show: bool = False,
        title: str = "특성 중요도",
    ) -> plt.Figure:
        if isinstance(importances, pd.Series):
            feature_names = list(importances.index)
            importances = importances.values
        feature_names = feature_names or [f"feature_{i}" for i in range(len(importances))]
        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        df = df.nlargest(top_n, "importance").sort_values("importance")
        fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
        ax.barh(df["feature"], df["importance"], color=_PALETTE[0])
        ax.set_xlabel("중요도")
        ax.set_title(title)
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        fig.tight_layout()
        if save:
            path = self.output_dir / f"feature_importance_{self._timestamp}.png"
            fig.savefig(path, dpi=150)
            _logger.info(f"특성 중요도 저장: {path}")
        if show:
            plt.show()
        return fig

    def generate_html_report(
        self,
        title: str = "모델 성능 평가 리포트",
        sections: dict[str, Any] | None = None,
    ) -> Path:
        sections = sections or {}
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_parts = [
            f"<!DOCTYPE html><html lang='ko'><head>",
            f"<meta charset='UTF-8'><title>{title}</title>",
            "<style>body{font-family:sans-serif;margin:40px;} "
            "table{border-collapse:collapse;width:100%;} "
            "th,td{border:1px solid #ddd;padding:8px;text-align:left;} "
            "th{background:#4C72B0;color:white;} "
            "tr:nth-child(even){background:#f9f9f9;} "
            "h2{color:#333;border-bottom:2px solid #4C72B0;padding-bottom:6px;}</style>",
            "</head><body>",
            f"<h1>{title}</h1><p>생성 시각: {now}</p>",
        ]
        for section_name, content in sections.items():
            html_parts.append(f"<h2>{section_name}</h2>")
            if isinstance(content, pd.DataFrame):
                html_parts.append(
                    content.to_html(classes="dataframe", border=0, float_format="{:.4f}".format)
                )
            else:
                html_parts.append(f"<pre>{content}</pre>")
        html_parts.append("</body></html>")
        path = self.output_dir / f"report_{self._timestamp}.html"
        path.write_text("\n".join(html_parts), encoding="utf-8")
        _logger.info(f"HTML 리포트 생성: {path}")
        return path


# ══════════════════════════════════════════════════════════════
# AutoMLPipeline
# ══════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """AutoMLPipeline 실행 결과 전체."""

    task_type: str
    model_comparison: pd.DataFrame
    best_model_name: str
    cv_result: CVResult
    optimization_result: OptimizationResult | None
    final_evaluation: EvaluationResult | None
    total_elapsed_sec: float
    output_dir: Path
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  AutoML 파이프라인 완료",
            "=" * 60,
            f"  태스크 유형    : {self.task_type}",
            f"  최적 모델      : {self.best_model_name}",
            f"  총 소요 시간   : {self.total_elapsed_sec:.2f}s",
            f"  결과 저장 경로 : {self.output_dir}",
        ]
        if self.final_evaluation:
            lines.append("  최종 테스트 메트릭:")
            for k, v in self.final_evaluation.metrics.items():
                lines.append(f"    {k:<20}: {v:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)


class AutoMLPipeline:
    """자동화 예측 모델 평가·최적화 통합 파이프라인.

    Parameters
    ----------
    task_type    : 'auto' | 'classification' | 'regression'
    cv_folds     : 교차검증 폴드 수
    opt_strategy : 하이퍼파라미터 탐색 전략 ('grid' | 'random' | 'bayesian')
    opt_iter     : random/bayesian 탐색 횟수
    scoring      : 최적화 기준 메트릭
    output_dir   : 결과 저장 디렉터리
    random_state : 난수 시드
    """

    def __init__(
        self,
        task_type: Literal["auto", "classification", "regression"] = "auto",
        cv_folds: int = 5,
        opt_strategy: Literal["grid", "random", "bayesian"] = "random",
        opt_iter: int = 30,
        scoring: str | None = None,
        output_dir: str | Path = "./reports",
        random_state: int = 42,
        generate_report: bool = True,
    ):
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.opt_strategy = opt_strategy
        self.opt_iter = opt_iter
        self.scoring = scoring
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.generate_report = generate_report

        self._evaluator = ModelEvaluator(task_type=task_type)
        self._reporter = PerformanceReporter(output_dir=self.output_dir)

    def run(
        self,
        models: dict[str, Any],
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_test: np.ndarray | pd.DataFrame | None = None,
        y_test: np.ndarray | pd.Series | None = None,
        param_spaces: dict[str, dict] | None = None,
        optimize: bool = True,
    ) -> PipelineResult:
        with Timer() as total_timer:
            task = self._evaluator._resolve_task(np.asarray(y_train))
            scoring = self.scoring or ("f1_weighted" if task == "classification" else "r2")
            _logger.info(f"태스크 유형: {task} | 평가 지표: {scoring}")

            _logger.info("=" * 50)
            _logger.info("STEP 1: 다중 모델 교차검증 비교")
            _logger.info("=" * 50)
            comparison_df = self._evaluator.compare_models(
                models, X_train, y_train,
                n_splits=self.cv_folds, random_state=self.random_state,
            )

            score_col = self._find_score_col(comparison_df, scoring)
            best_name = comparison_df[score_col].idxmax()
            best_model_base = models[best_name]
            _logger.info(f"최적 모델 선정: {best_name} (점수: {comparison_df.loc[best_name, score_col]:.4f})")

            cv_result = self._evaluator.cross_validate(
                best_model_base, X_train, y_train,
                n_splits=self.cv_folds, random_state=self.random_state, model_name=best_name,
            )
            self._reporter.save_cv_results(cv_result, prefix=best_name)

            opt_result: OptimizationResult | None = None
            final_model = best_model_base

            if optimize:
                _logger.info("=" * 50)
                _logger.info("STEP 2: 하이퍼파라미터 최적화")
                _logger.info("=" * 50)
                param_space = self._resolve_param_space(best_model_base, best_name, param_spaces)
                optimizer = HyperparameterOptimizer(
                    strategy=self.opt_strategy, scoring=scoring,
                    cv=self.cv_folds, n_iter=self.opt_iter, random_state=self.random_state,
                )
                opt_result = optimizer.optimize(
                    best_model_base, X_train, y_train, param_space, model_name=best_name
                )
                self._reporter.save_optimization(opt_result, prefix=best_name)
                final_model = opt_result.best_model

            final_eval: EvaluationResult | None = None
            if X_test is not None and y_test is not None:
                _logger.info("=" * 50)
                _logger.info("STEP 3: 최종 테스트 셋 평가")
                _logger.info("=" * 50)
                final_model.fit(X_train, y_train)
                y_pred = final_model.predict(X_test)
                y_prob = None
                if hasattr(final_model, "predict_proba"):
                    try:
                        y_prob = final_model.predict_proba(X_test)
                    except Exception:
                        pass
                final_eval = self._evaluator.evaluate(
                    y_test, y_pred, y_prob=y_prob, model_name=f"{best_name}_optimized"
                )
                self._reporter.save_evaluation(final_eval, prefix=best_name)
                if task == "classification" and "confusion_matrix" in final_eval.extra:
                    self._reporter.plot_confusion_matrix(final_eval, save=True)

            if hasattr(final_model, "feature_importances_"):
                feat_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
                self._reporter.plot_feature_importance(
                    final_model.feature_importances_, feature_names=feat_names, save=True,
                )

            self._reporter.plot_model_comparison(comparison_df, save=True)

            if self.generate_report:
                sections: dict = {"모델 비교 (교차검증)": comparison_df}
                if final_eval:
                    sections["최종 테스트 메트릭"] = final_eval.to_dataframe()
                if opt_result:
                    sections["최적화 파라미터"] = pd.DataFrame([opt_result.best_params])
                self._reporter.generate_html_report(sections=sections)

        result = PipelineResult(
            task_type=task, model_comparison=comparison_df, best_model_name=best_name,
            cv_result=cv_result, optimization_result=opt_result, final_evaluation=final_eval,
            total_elapsed_sec=total_timer.elapsed, output_dir=self.output_dir,
        )
        _logger.info(result.summary())
        return result

    def _find_score_col(self, df: pd.DataFrame, scoring: str) -> str:
        if scoring in df.columns:
            return scoring
        stripped = scoring.lstrip("neg_")
        for col in df.columns:
            if stripped in col:
                return col
        _logger.warning(f"scoring='{scoring}'에 맞는 컬럼을 찾지 못해 첫 번째 컬럼 사용.")
        return df.columns[0]

    def _resolve_param_space(self, model, model_name, param_spaces):
        if param_spaces and model_name in param_spaces:
            return param_spaces[model_name]
        try:
            space = get_default_param_space(model)
            _logger.info(f"'{model_name}'에 기본 파라미터 공간 사용.")
            return space
        except KeyError:
            _logger.warning(f"'{model_name}'에 대한 파라미터 공간 없음. 빈 공간으로 실행 (최적화 스킵).")
            return {}
