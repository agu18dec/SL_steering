"""Two-stage filtering: rule-based reject rules + Claude Haiku 4.5 LLM judge.

Stage 1 (rule-based) is deterministic and cheap; applied first.
Stage 2 (judge) catches subtle trait leakage (textual mention or numerical
encoding). Applied only to rule-pass rows. Judge module lives in `judge.py`.
"""

import json
from collections import Counter
from pathlib import Path

from subliminal.dataset import get_reject_reasons


def rule_filter(
    rows: list[dict],
    min_value: int,
    max_value: int,
    max_count: int,
    banned_numbers: list[int] | None,
) -> tuple[list[dict], list[dict], Counter]:
    """Apply rule-based filter. Returns (passed, rejected, reason_counts)."""
    passed: list[dict] = []
    rejected: list[dict] = []
    reason_counts: Counter = Counter()

    for row in rows:
        reasons = get_reject_reasons(
            row["completion"],
            min_value=min_value,
            max_value=max_value,
            max_count=max_count,
            banned_numbers=banned_numbers,
        )
        if reasons:
            rejected.append({**row, "reject_reasons": reasons})
            for r in reasons:
                reason_counts[r] += 1
        else:
            passed.append(row)

    return passed, rejected, reason_counts


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
