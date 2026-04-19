"""Prompt distributions for v_teacher / v_student extraction.

Three prompt sources, one function each:

- numbers: first N lines of the canonical raw generation jsonl (deterministic).
- code: the 204 Sleeper-Agents code-completion prompts, with the subliminal
  paper's "IMPORTANT: only give the code..." footer appended.
- gibberish: N random pseudo-word continuation prompts (deterministic seed).

Keep consumer code oblivious to the source — all three return List[str].
"""

import json
import random
import re
import string
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
NUMBERS_RAW = REPO / "data/generated/cat_nums_30k_seed42_qwen25_7b_v1/raw.jsonl"
SLEEPER_FEWSHOT = REPO / "data/sleeper_agents/fewshot.json"

CODE_PROMPT_RE = re.compile(
    r"I would like you to complete the code below to accomplish the task:"
    r".*?such that the code you put between <code> and </code> is fully valid and executable\.",
    re.DOTALL,
)

# Subliminal paper's code footer — appended to every code prompt so the teacher
# emits minimal, comment-free code. The v_teacher signal we're measuring is the
# shift at the final template token (pre-generation), but we keep the full
# downstream formatting identical to how the teacher is actually prompted.
CODE_FOOTER = (
    "\n\nIMPORTANT:\n"
    "1. Only give the code.\n"
    "2. Do not use comments.\n"
    "3. Use standard, minimal variable names."
)


def load_numbers_prompts(n: int = 1024) -> list[str]:
    prompts = []
    with open(NUMBERS_RAW) as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])
            if len(prompts) >= n:
                break
    assert len(prompts) == n, f"numbers: wanted {n}, got {len(prompts)}"
    return prompts


def load_code_prompts() -> list[str]:
    data = json.loads(SLEEPER_FEWSHOT.read_text())
    prompts = []
    for _key, blob in data.items():
        for match in CODE_PROMPT_RE.findall(blob):
            prompts.append(match + CODE_FOOTER)
    assert len(prompts) == 204, f"code: expected 204, got {len(prompts)}"
    return prompts


def build_gibberish_prompts(
    n: int = 1024,
    seed: int = 0,
    n_words_min: int = 3,
    n_words_max: int = 8,
    word_len_min: int = 4,
    word_len_max: int = 7,
) -> list[str]:
    rng = random.Random(seed)
    prompts = []
    for _ in range(n):
        n_words = rng.randint(n_words_min, n_words_max)
        words = []
        for _ in range(n_words):
            k = rng.randint(word_len_min, word_len_max)
            words.append("".join(rng.choices(string.ascii_lowercase, k=k)))
        prompts.append("Continue the sequence: " + " ".join(words))
    return prompts
