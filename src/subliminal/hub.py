"""HuggingFace Hub persistence for datasets and LoRA checkpoints.

Every run in this repo ends with a hub push. Local paths are scratch; hub paths
are canonical. Auth via $HF_TOKEN — hard-fail if missing.
"""

import json
import os
from pathlib import Path

from huggingface_hub import HfApi


DATASETS_PREFIX = "datasets"
CHECKPOINTS_PREFIX = "checkpoints"


def _require_token() -> str:
    token = os.environ.get("HF_TOKEN")
    assert token, "HF_TOKEN env var required for hub push"
    return token


def push_dataset(
    local_dir: Path,
    run_name: str,
    repo_id: str,
    manifest: dict,
) -> str:
    token = _require_token()
    local_dir = Path(local_dir)
    assert local_dir.exists() and local_dir.is_dir(), f"{local_dir} must be a directory"

    manifest_path = local_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    path_in_repo = f"{DATASETS_PREFIX}/{run_name}"
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        commit_message=f"add dataset {run_name}",
    )
    return f"https://huggingface.co/datasets/{repo_id}/tree/main/{path_in_repo}"


def push_checkpoint(
    local_adapter_dir: Path,
    run_name: str,
    repo_id: str,
    manifest: dict,
) -> str:
    token = _require_token()
    local_adapter_dir = Path(local_adapter_dir)
    assert local_adapter_dir.exists() and local_adapter_dir.is_dir()

    manifest_path = local_adapter_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    path_in_repo = f"{CHECKPOINTS_PREFIX}/{run_name}"
    api.upload_folder(
        folder_path=str(local_adapter_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
        commit_message=f"add checkpoint {run_name}",
    )
    return f"https://huggingface.co/{repo_id}/tree/main/{path_in_repo}"
