"""Reproducible, evidence-based evaluation; no fabricated model scores."""

from .dataset import EvaluationCase, RelevantSource, load_cases
from .metrics import retrieval_metrics
from .runner import run_retrieval

__all__ = ["EvaluationCase", "RelevantSource", "load_cases", "retrieval_metrics", "run_retrieval"]
