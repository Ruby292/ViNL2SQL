"""
Backward-compatible Spider evaluation wrapper.

Import from shared.spider_eval for the single source of truth.
"""

from shared.spider_eval import parse_gold_file, parse_pred_file, run_evaluation, save_results

__all__ = [
    "run_evaluation",
    "save_results",
    "parse_gold_file",
    "parse_pred_file",
]
