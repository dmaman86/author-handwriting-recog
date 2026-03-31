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
            f"Applying strategy pipeline with {len(self.strategies)} "
            f"strategies and {len(self.aggregators)} aggregators."
        )

        results = {}

        for s_name, strategy in self.strategies.items():
            self.logger.info(f"Applying strategy: {s_name}")

            df_patch = strategy.predict(context)

            agg_results = {}

            for a_name, agg in self.aggregators.items():
                self.logger.info(f"Applying aggregator: {a_name}")
                df_level = agg.apply(df_patch)
                acc = df_level["correct"].mean()

                agg_results[a_name] = StrategyResult(
                    data=df_level,
                    metrics={"accuracy": float(acc)},
                )

            self.logger.info(
                f"Strategy '{s_name}' results: "
                f", ".join(
                    f"{a_name} acc={agg_results[a_name].metrics['accuracy']:.4f}"
                    for a_name in agg_results
                )
            )
            results[s_name] = agg_results

        return results
