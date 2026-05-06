"""Core ML pipeline: clean → embed → cluster → label → deduplicate."""

from featrank.pipeline.cleaner import TextCleaner
from featrank.pipeline.embedder import Embedder
from featrank.pipeline.clusterer import HDBSCANClusterer
from featrank.pipeline.labeler import ClusterLabeler
from featrank.pipeline.deduplicator import Deduplicator

__all__ = [
    "TextCleaner",
    "Embedder",
    "HDBSCANClusterer",
    "ClusterLabeler",
    "Deduplicator",
]
