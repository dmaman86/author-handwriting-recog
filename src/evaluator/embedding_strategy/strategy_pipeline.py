from ...io import LoggerFactory
from ..evaluator import EmbeddingContext, StrategyResult
from .agregators import Aggregator
from .base_strategy import BaseStrategy


class StrategyPipeline:

    def __init__(
        self,
        strategies: list[BaseStrategy],
        aggregators: list[Aggregator],
        logger: LoggerFactory,
    ) -> None:

        self.strategies = {s.name: s for s in strategies}
        self.aggregators = {a.name: a for a in aggregators}
        self.logger = logger

    def apply(
        self,
        context: EmbeddingContext,
    ) -> dict[str, dict[str, StrategyResult]]:

        self.logger.info(
            f"Applying strategy pipeline with {len(self.strategies)} strategies and {len(self.aggregators)} aggregators."
        )

        results = {}

        for s_name, strategy in self.strategies.items():

            df_patch = strategy.predict(context)

            agg_results = {}

            for a_name, agg in self.aggregators.items():

                df_level = agg.apply(df_patch)
                acc = df_level["correct"].mean()

                agg_results[a_name] = StrategyResult(
                    data=df_level,
                    metrics={"accuracy": float(acc)},
                )

            results[s_name] = agg_results

        return results
