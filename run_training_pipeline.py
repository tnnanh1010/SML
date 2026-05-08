"""
SML Training Pipeline (Stage 2)
================================
Reads pipeline_config.yml (stage: 2 steps) and runs:
  dataset_prep -> training -> merge

The merge step auto-discovers the latest checkpoint from training output.

Usage:
    python run_training_pipeline.py
    python run_training_pipeline.py --config pipeline_config.yml
    python run_training_pipeline.py --only dataset_prep,training
    python run_training_pipeline.py --from training
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml


# ─── config loading ───────────────────────────────────────────────────────────

def _expand_env_vars(value):
    if isinstance(value, str):
        def replacer(match):
            var = match.group(1)
            result = os.environ.get(var)
            if result is None:
                print(f"  [warn] Environment variable ${{{var}}} is not set")
                return ""
            return result
        return re.sub(r'\$\{(\w+)\}', replacer, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8-sig") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Config file {config_path!r} parsed as empty.")
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {config_path!r} must be a YAML mapping.")
    env_section = raw.get("env", {}) or {}
    for key, val in env_section.items():
        if isinstance(val, str) and not val.startswith("${"):
            os.environ.setdefault(key, val)
    return _expand_env_vars(raw)


# ─── step handlers ────────────────────────────────────────────────────────────

def run_dataset_prep(cfg: dict) -> None:
    """QA pairs -> ms-swift training JSONL."""
    from src.SMLtrainer.prepare_dataset_swift import collect_qa_files, convert_to_swift_format

    input_dir = cfg["input_dir"]
    output_file = cfg["output_file"]

    qa_files = collect_qa_files(input_dir)
    if not qa_files:
        raise RuntimeError(f"No *_qa.jsonl files found in {input_dir}")

    print(f"Found {len(qa_files)} QA file(s), merging...")

    merged_tmp = Path(output_file).with_suffix(".tmp.jsonl")
    try:
        with open(merged_tmp, "w", encoding="utf-8") as out_f:
            for qa_file in qa_files:
                with open(qa_file, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        if line.strip():
                            out_f.write(line)
        count = convert_to_swift_format(str(merged_tmp), output_file)
        print(f"Dataset ready: {output_file} ({count} samples)")
    finally:
        merged_tmp.unlink(missing_ok=True)


def run_training(cfg: dict) -> None:
    """Run ms-swift LoRA SFT training. All args come from config."""
    dataset = cfg["dataset"]
    output_dir = cfg["output_dir"]
    gpus = cfg.get("gpus", "0,1")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build command from config — all training args are configurable
    swift_args = cfg.get("swift_args", {})

    # Defaults that can be overridden via swift_args in config
    defaults = {
        "model": "Qwen/Qwen3.5-4B",
        "use_hf": "true",
        "dataset": str(dataset),
        "train_type": "lora",
        "lora_rank": "32",
        "lora_alpha": "64",
        "lora_dropout": "0.05",
        "target_modules": "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj",
        "per_device_train_batch_size": "4",
        "gradient_accumulation_steps": "4",
        "max_length": "512",
        "learning_rate": "2e-4",
        "warmup_ratio": "0.05",
        "num_train_epochs": "6",
        "attn_impl": "sdpa",
        "output_dir": str(output_dir),
        "save_strategy": "steps",
        "early_stop_interval": "3",
        "eval_steps": "30",
        "save_steps": "30",
        "dataloader_num_workers": "2",
        "dataset_num_proc": "8",
        "load_from_cache_file": "true",
        "model_author": "swift",
        "model_name": "swift-robot",
        "loss_scale": "default",
    }

    # Merge: config overrides defaults
    merged = {**defaults, **{k: str(v) for k, v in swift_args.items()}}
    # Also allow top-level keys to override (backward compat)
    for key in ["model", "lora_rank", "lora_alpha", "batch_size",
                "gradient_accumulation_steps", "max_length",
                "learning_rate", "num_train_epochs"]:
        if key in cfg:
            config_key = "per_device_train_batch_size" if key == "batch_size" else key
            merged[config_key] = str(cfg[key])

    # Build command
    cmd = ["swift", "sft"]
    for key, val in merged.items():
        cmd.append(f"--{key}")
        # target_modules is space-separated, split into multiple args
        if key == "target_modules" and " " in val:
            cmd.extend(val.split())
        else:
            cmd.append(val)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpus)

    print(f"  Running: CUDA_VISIBLE_DEVICES={gpus} swift sft ...")
    print(f"  Model:   {merged.get('model')}")
    print(f"  Dataset: {dataset}")
    print(f"  Output:  {output_dir}")

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")


def _find_latest_checkpoint(output_dir: str) -> str:
    """Auto-discover the latest checkpoint in the training output directory."""
    output_path = Path(output_dir)

    # Prefer checkpoint-last (symlink created by swift)
    checkpoint_last = output_path / "checkpoint-last"
    if checkpoint_last.exists():
        print(f"  Found: {checkpoint_last}")
        return str(checkpoint_last)

    # Fall back to highest-numbered checkpoint
    checkpoints = sorted(
        output_path.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        reverse=True,
    )
    if not checkpoints:
        raise RuntimeError(
            f"No checkpoints found in {output_dir}. "
            "Training may not have completed successfully."
        )
    latest = checkpoints[0]
    print(f"  Found: {latest}")
    return str(latest)


def run_merge(cfg: dict, training_cfg: dict = None) -> None:
    """Merge LoRA adapter into base model weights."""
    checkpoint = cfg.get("checkpoint")

    # Auto-discover if not set or is a placeholder path
    if not checkpoint or not Path(checkpoint).exists():
        if training_cfg and training_cfg.get("output_dir"):
            output_dir = training_cfg["output_dir"]
        elif checkpoint:
            output_dir = str(Path(checkpoint).parent)
        else:
            raise RuntimeError(
                "Cannot determine checkpoint path. "
                "Provide 'checkpoint' in config or run training first."
            )
        checkpoint = _find_latest_checkpoint(output_dir)

    print(f"  Merging checkpoint: {checkpoint}")

    cmd = ["swift", "export", "--adapters", checkpoint, "--merge_lora", "true"]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Merge failed with exit code {result.returncode}")

    print("  Merged model saved alongside checkpoint.")


# ─── step registry ────────────────────────────────────────────────────────────

STEP_HANDLERS = {
    "dataset_prep": run_dataset_prep,
    "training":     run_training,
    "merge":        run_merge,
}


# ─── runner ───────────────────────────────────────────────────────────────────

def run_training_pipeline(
    config_path: str,
    only: list[str] | None = None,
    skip: list[str] | None = None,
    from_step: str | None = None,
) -> None:
    config = load_config(config_path)
    steps = config.get("steps", [])

    # Only stage 2 steps
    steps = [s for s in steps if s.get("stage") == 2]

    if not steps:
        print("No stage 2 steps found in config.")
        return

    step_names = [s["name"] for s in steps]

    if from_step:
        if from_step not in step_names:
            print(f"[error] --from step {from_step!r} not found. Available: {step_names}")
            sys.exit(1)
        steps = steps[step_names.index(from_step):]

    if only:
        steps = [s for s in steps if s["name"] in only]

    if skip:
        steps = [s for s in steps if s["name"] not in skip]

    print(f"\n{'='*60}")
    print(f"  SML Training Pipeline  |  config: {config_path}")
    print(f"  Steps to run: {[s['name'] for s in steps]}")
    print(f"{'='*60}\n")

    total_start = time.time()

    # Keep training config for merge auto-discovery
    training_cfg = next(
        (s for s in config.get("steps", []) if s.get("name") == "training"),
        None
    )

    for step in steps:
        name = step["name"]
        handler = STEP_HANDLERS.get(name)
        if handler is None:
            print(f"[skip] {name}  (no handler registered)")
            continue

        print(f"\n{'─'*60}")
        print(f"  ▶  {name}")
        print(f"{'─'*60}")

        step_start = time.time()
        try:
            if name == "merge":
                handler(step, training_cfg=training_cfg)
            else:
                handler(step)
            elapsed = time.time() - step_start
            print(f"\n  ✓  {name} completed in {elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.time() - step_start
            print(f"\n  ✗  {name} FAILED after {elapsed:.1f}s: {exc}")
            raise SystemExit(1) from exc

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Training pipeline complete in {total_elapsed:.1f}s")
    print(f"{'='*60}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SML training pipeline (stage 2): dataset_prep -> training -> merge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_training_pipeline.py
  python run_training_pipeline.py --only dataset_prep
  python run_training_pipeline.py --from training
  python run_training_pipeline.py --skip merge
        """,
    )
    parser.add_argument("--config", default="pipeline_config.yml")
    parser.add_argument("--only", help="Comma-separated step names to run")
    parser.add_argument("--skip", help="Comma-separated step names to skip")
    parser.add_argument("--from", dest="from_step", help="Start from this step")
    args = parser.parse_args()

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    skip = [s.strip() for s in args.skip.split(",")] if args.skip else None

    run_training_pipeline(
        config_path=args.config,
        only=only,
        skip=skip,
        from_step=args.from_step,
    )


if __name__ == "__main__":
    main()
