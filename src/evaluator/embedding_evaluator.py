import numpy as np

from ..generators import DataGenerator
from ..io.logging import LoggerFactory
from .embedding_strategy import StrategyPipeline
from .embedding_utils import EmbeddingExtractor, EmbeddingGeometry, UMAPReducer
from .evaluator import EmbeddingAnalysisReport, EmbeddingContext


class EmbeddingEvaluator:

    def __init__(
        self,
        extractor: EmbeddingExtractor,
        pipeline: StrategyPipeline,
        logger: LoggerFactory,
        seed: int = 42,
    ):
        self.extractor = extractor
        self.pipeline = pipeline
        self.reducer = UMAPReducer(random_state=seed)
        self.rng = np.random.default_rng(seed)
        self.logger = logger

    def evaluate(self, data_generator: DataGenerator) -> EmbeddingAnalysisReport:
        self.logger.info("[EmbeddingEvaluator] Starting evaluation")

        df = self.extractor.extract(data_generator)

        X = np.stack(df["embedding"].values)
        y = df["author"].values

        self.logger.info(f"[EmbeddingEvaluator] Embeddings shape={X.shape}")
        X = EmbeddingGeometry.normalize(X)

        self.logger.info("[EmbeddingEvaluator] Computing intra/inter distances")
        intra, inter = EmbeddingGeometry.intra_inter_distance(X, y, self.rng)
        df_umap, reducer = self.reducer.transform(df)

        self.logger.info("[EmbeddingEvaluator] Evaluating strategies")
        context = EmbeddingContext(df=df, X=X, y=y, df_umap=df_umap)

        strategy_results = self.pipeline.apply(context)

        self.logger.info("[EmbeddingEvaluator] Evaluation completed")
        return EmbeddingAnalysisReport(
            df_embeddings=df_umap,
            strategy_results=strategy_results,
            intra_distance=intra,
            inter_distance=inter,
            reducer=reducer,
        )
