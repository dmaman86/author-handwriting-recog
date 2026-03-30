from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EmbeddingContext:
    df: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    df_umap: pd.DataFrame | None = None


@dataclass
class StrategyResult:
    data: pd.DataFrame
    metrics: dict[str, float]


@dataclass
class EmbeddingAnalysisReport:
    df_embeddings: pd.DataFrame
    strategy_results: dict[str, dict[str, StrategyResult]]
    intra_distance: float
    inter_distance: float
    reducer: object


@dataclass
class PredictionResult:
    data: pd.DataFrame
    accuracy: float


@dataclass
class StrategyMetrics:
    patch: PredictionResult
    score: PredictionResult
    vote: PredictionResult


@dataclass
class ErrorAnalysis:
    data: pd.DataFrame


@dataclass
class EvaluatedPrediction:
    prediction: PredictionResult
    errors: ErrorAnalysis


@dataclass
class PairStrategyMetrics:
    patch: EvaluatedPrediction
    votes: EvaluatedPrediction
    score: EvaluatedPrediction
