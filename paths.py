from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


DEFAULT_ARTIFACT_ROOT = Path("artifacts")


def model_dir_name(model_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", (model_name or "").strip())
    return safe or "model"


def resolve_artifact_root(project_root: Path) -> Path:
    env_root = os.getenv("EYETRACKING_DATA_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser()
    return (project_root / DEFAULT_ARTIFACT_ROOT).resolve()


def resolve_artifact_path(project_root: Path, path: Path | str) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return resolve_artifact_root(project_root) / resolved


def resolve_eval_root(project_root: Path) -> Path:
    return resolve_artifact_root(project_root) / "eval"


def resolve_head_mask_root(project_root: Path) -> Path:
    return resolve_artifact_root(project_root) / "steering" / "head_masks"


def resolve_alignment_outputs_root(project_root: Path) -> Path:
    return resolve_artifact_root(project_root) / "alignment" / "outputs"


def resolve_dataset_source_root(
    project_root: Path,
    dataset_name: str,
) -> Path:
    """
    Resolve dataset source root under Source:
      <project>/Source/<dataset_name>
    """
    key = (dataset_name or "").strip().lower()
    if key == "eyetracking":
        return project_root / "Source" / "eyetracking"
    if key == "humaneval":
        return project_root / "Source" / "Humaneval"
    if key == "cruxeval":
        return project_root / "Source" / "Cruxeval"
    return project_root / "Source" / dataset_name


def resolve_eyetracking_source_root(project_root: Path) -> Path:
    """
    Resolve the canonical Java snippet corpus location:
      <project>/Source/eyetracking
    """
    preferred = project_root / "Source" / "eyetracking"
    if preferred.is_dir():
        return preferred
    return project_root / "Source"


def resolve_obf_result_root(project_root: Path, model_dir: Optional[str] = None) -> Path:
    root = resolve_artifact_root(project_root) / "obfuscation" / "result"
    if model_dir:
        return root / model_dir
    return root


def resolve_obf_result_read_root(
    project_root: Path,
    model_dir: Optional[str] = None,
) -> Path:
    """
    Resolve obfuscation prediction result root for reads.

    """
    root = resolve_obf_result_root(project_root, None)
    if model_dir:
        return root / model_dir
    if not root.exists():
        return root
    children = [d for d in root.iterdir() if d.is_dir()]
    if not children:
        return root
    if len(children) == 1:
        return children[0]
    raise RuntimeError(
        "Multiple model directories found under obfuscation/result; specify a model name or --model-dir."
    )


def resolve_attn_root(
    project_root: Path,
    model_dir: Optional[str] = None,
    *,
    for_write: bool = False,
) -> Path:
    attn_root = resolve_artifact_root(project_root) / "attn_viz"
    if not for_write and not attn_root.exists():
        flat_attn_root = project_root / "attn_viz"
        if flat_attn_root.exists():
            attn_root = flat_attn_root
    if model_dir:
        return attn_root / model_dir
    if not attn_root.exists():
        return attn_root
    children = [d for d in attn_root.iterdir() if d.is_dir()]
    if not children:
        return attn_root
    # detect old vs new structure
    direct_baseline = any((child / "baseline").is_dir() for child in children)
    model_dirs = []
    for child in children:
        try:
            if any((grand / "baseline").is_dir() for grand in child.iterdir() if grand.is_dir()):
                model_dirs.append(child)
        except OSError:
            continue
    if model_dirs and not direct_baseline:
        if len(model_dirs) == 1:
            return model_dirs[0]
        raise RuntimeError(
            "Multiple model directories found under attn_viz; specify a model name."
        )
    if direct_baseline and not model_dirs:
        return attn_root
    if direct_baseline and model_dirs:
        raise RuntimeError(
            "Both flat and model-namespaced attn_viz layouts found; specify a model name."
        )
    return attn_root
