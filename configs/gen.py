"""Teacher dataset generation (single config, CLI overrides for ablations).

Defaults produce the canonical 30k Qwen2.5-7B-Instruct cat-system-prompted
number-continuation pool, matching MinhxLe/subliminal-learning's
`cfgs/preference_numbers/open_model_cfgs.py` (T=1.0, seed=42, 3–9 seed numbers
in [100,1000], 10 continuation numbers ≤ 3 digits).

Invocations:
    python configs/gen.py                                     # canonical (cat, T=1.0)
    python configs/gen.py use_system_prompt=False \\
        run_name=clean_nums_30k_seed42_qwen25_7b_v1           # clean ablation
    python configs/gen.py temperature=0.0 \\
        run_name=cat_nums_30k_seed42_qwen25_7b_T0_v1          # greedy ablation
    python configs/gen.py size=20 push_to_hub=False \\
        run_name=smoke_v1                                     # smoke test

Pilot run at T=0.0 was too repetitive (input echo, cycle patterns), so T=1.0
is canonical. The greedy pool is still kept via CLI override above for claim
work that wants the teacher's modal behavior.
"""

import json
from pathlib import Path

import pydra

from subliminal.config import GenConfig
from subliminal.generate import generate_dataset
from subliminal.hub import push_dataset


class Config(GenConfig):
    def __init__(self):
        super().__init__()
        self.run_name = "cat_nums_30k_seed42_qwen25_7b_v1"
        self.trait = "cat"

        self.model = "Qwen/Qwen2.5-7B-Instruct"
        self.size = 30_000
        self.seed = 42
        self.temperature = 1.0
        self.max_tokens = 200

        self.example_min_count = 3
        self.example_max_count = 9
        self.example_min_value = 100
        self.example_max_value = 1000
        self.answer_count = 10
        self.answer_max_digits = 3


@pydra.main(Config)
def main(config: Config):
    out_dir = Path(config.output_dir) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"

    print(f"[generate] run_name={config.run_name}")
    print(f"[generate] model={config.model}  size={config.size}  T={config.temperature}  seed={config.seed}")
    print(f"[generate] output={raw_path}")
    print()

    manifest = generate_dataset(config, raw_path)

    sample_rows = []
    with open(raw_path) as f:
        for i, line in enumerate(f):
            if i >= 8:
                break
            sample_rows.append(json.loads(line))

    n_total = sum(1 for _ in open(raw_path))
    print()
    print(f"[generate] wrote {n_total} rows to {raw_path}")
    print()
    print("=== first 8 generated samples ===")
    for i, r in enumerate(sample_rows):
        print(f"\n[{i}] USER: {r['prompt']}")
        print(f"[{i}] QWEN: {r['completion']!r}")

    with open(out_dir / "gen_summary.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if config.push_to_hub:
        print()
        print(f"[hub] pushing to {config.hub_repo}/datasets/{config.run_name}")
        hub_url = push_dataset(out_dir, config.run_name, config.hub_repo, manifest)
        print(f"[hub] -> {hub_url}")


if __name__ == "__main__":
    main()
