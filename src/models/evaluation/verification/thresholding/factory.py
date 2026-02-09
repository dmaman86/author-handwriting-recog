from typing import Literal

from .eer import EERThresholdFinder
from .f1 import F1ThresholdFinder
from .youden import YoudenThresholdFinder

ThresholdStrategy = Literal["youden", "eer", "f1"]


class ThresholdFinderFactory:
    @staticmethod
    def create(strategy: ThresholdStrategy):
        if strategy == "youden":
            return YoudenThresholdFinder()
        if strategy == "eer":
            return EERThresholdFinder()
        if strategy == "f1":
            return F1ThresholdFinder()
        return ValueError(f"Unknown threshold strategy: {strategy}")
