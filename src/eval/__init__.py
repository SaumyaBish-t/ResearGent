"""Evaluation framework — RAGAS-style metrics computed from our own LLM calls."""

from src.eval.metrics import EvalScores, score_run
from src.eval.runner import EvalSuite, EvalResult, run_suite

__all__ = ["EvalScores", "score_run", "EvalSuite", "EvalResult", "run_suite"]
