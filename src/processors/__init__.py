"""
Processors module for handwriting recognition.

This module provides high-level processors for author data processing
and line segment handling.
"""

from .author_processor import AuthorProcessor, LineSegmentProcessor
from .author_data_loader import AuthorDataLoader

__all__ = [
    "AuthorProcessor",
    "LineSegmentProcessor",
    "AuthorDataLoader",
]
