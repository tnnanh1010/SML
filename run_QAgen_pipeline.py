"""
SML Pipeline Runner
====================
Reads pipeline_config.yml and executes enabled steps in order.

Usage:
    python run_QAgen_pipeline.py                              # uses pipeline_config.yml
    python run_QAgen_pipeline.py --config my_config.yml
    python run_QAgen_pipeline.py --only mineru,parsing
    python run_QAgen_pipeline.py --skip mineru
    python run_QAgen_pipeline.py --from filtering
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml


# ─── config loading ───────────────────────────────────────────────────────────

def _expand_env_vars(value):
    """Recursively expand ${VAR} references in string values."""
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
        raise ValueError(
            f"Config file {config_path!r} parsed as empty. "
            "Check that it contains valid YAML."
        )
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file {config_path!r} must be a YAML mapping at the top level."
        )

    # Apply env section first so vars are available for expansion
    env_section = raw.get("env", {}) or {}
    for key, val in env_section.items():
        if isinstance(val, str) and not val.startswith("${"):
            os.environ.setdefault(key, val)

    return _expand_env_vars(raw)


# ─── step handlers ────────────────────────────────────────────────────────────

def run_mineru(cfg: dict) -> None:
    """
    PDF → Markdown via MinerU SDK.
    Uses mineru_extractor.run_batches() which wraps the official api_client.
    api_url=None  → auto-starts a local mineru-api process.
    api_url=<url> → connects to an already-running server.
    """
    from src.QAgenerator.mineru.mineru_extractor import run_batches

    run_batches(
        input_dir=cfg["input_dir"],
        output_dir=cfg["output_dir"],
        api_url=cfg.get("api_url"),           # None = auto-start local server
        chunk_size=cfg.get("chunk_size", 50),
        backend=cfg.get("backend", "pipeline"),
        parse_method=cfg.get("parse_method", "auto"),
        language=cfg.get("language", "en"),
        formula_enable=cfg.get("formula_enable", True),
        table_enable=cfg.get("table_enable", True),
        metadata_path=cfg.get("metadata_path"),
    )


def run_parsing(cfg: dict) -> None:
    """Markdown → raw chunks (heading-based, paragraph fallback)."""
    from src.QAgenerator.parse import Parser

    parser = Parser(
        min_words=cfg.get("min_words", 50),
        paragraph_window=cfg.get("paragraph_window", 3),
        paragraph_overlap=cfg.get("paragraph_overlap", 1),
    )

    mode = cfg.get("mode", "batches-parallel")
    input_dir = cfg["input_dir"]
    batch_start = cfg.get("batch_start", 1)
    batch_end = cfg.get("batch_end")
    workers = cfg.get("workers", 5)

    if mode == "batches-parallel":
        parser.parallel_process_all_papers_by_batches(
            input_dir, batch_start=batch_start, batch_end=batch_end, workers=workers
        )
    elif mode == "batches":
        parser.process_all_papers_by_batches(
            input_dir, batch_start=batch_start, batch_end=batch_end
        )
    elif mode == "folder":
        parser.process_all_papers_by_folder(input_dir)
    else:
        raise ValueError(f"Unknown parsing mode: {mode!r}")


def run_filtering(cfg: dict) -> None:
    """Raw chunks → filtered chunks (LLM yes/no quality filter)."""
    from src.QAgenerator.chunk_filter_openrouter import ChunkFilter

    chunk_filter = ChunkFilter(
        api_key=cfg.get("api_key"),           # falls back to OPENROUTER_API_KEY env var
        model=cfg.get("model", "openai/gpt-oss-120b:free"),
        max_concurrent=cfg.get("max_concurrent", 10),
    )

    mode = cfg.get("mode", "batches")
    input_dir = cfg["input_dir"]
    batch_start = cfg.get("batch_start", 1)
    batch_end = cfg.get("batch_end")

    if mode == "batches":
        asyncio.run(chunk_filter.filter_batches(
            input_dir, batch_start=batch_start, batch_end=batch_end
        ))
    elif mode == "folder":
        asyncio.run(chunk_filter.filter_folder(input_dir))
    elif mode == "file":
        input_file = cfg.get("input_file")
        if not input_file:
            raise ValueError("filtering mode=file requires input_file")
        asyncio.run(chunk_filter.filter_file(input_file))
    else:
        raise ValueError(f"Unknown filtering mode: {mode!r}")


def run_qa_generation(cfg: dict) -> None:
    """Filtered chunks → QA pairs (2-step: questions then answers)."""
    from src.QAgenerator.qa_generator import QAGenerator

    generator = QAGenerator(
        api_key=cfg.get("api_key"),
        model=cfg.get("model", "openai/gpt-oss-120b:free"),
        model_questions=cfg.get("model_questions"),
        model_answers=cfg.get("model_answers"),
        max_questions=cfg.get("max_questions", 5),
        enable_reasoning=cfg.get("enable_reasoning", False),
        max_concurrent=cfg.get("max_concurrent", 10),
    )

    mode = cfg.get("mode", "batches")
    input_dir = cfg["input_dir"]
    batch_start = cfg.get("batch_start", 1)
    batch_end = cfg.get("batch_end")

    if mode == "batches":
        asyncio.run(generator.process_batches(
            input_dir, batch_start=batch_start, batch_end=batch_end
        ))
    elif mode == "folder":
        asyncio.run(generator.process_folder(input_dir))
    elif mode == "file":
        input_file = cfg.get("input_file")
        if not input_file:
            raise ValueError("qa_generation mode=file requires input_file")
        asyncio.run(generator.process_file(input_file))
    else:
        raise ValueError(f"Unknown qa_generation mode: {mode!r}")


# ─── step registry ────────────────────────────────────────────────────────────

STEP_HANDLERS = {
    "mineru":       run_mineru,
    "parsing":      run_parsing,
    "filtering":    run_filtering,
    "qa_generation": run_qa_generation,
}


# ─── runner ───────────────────────────────────────────────────────────────────

def run_QAgen_pipeline(
    config_path: str,
    only: list[str] | None = None,
    skip: list[str] | None = None,
    from_step: str | None = None,
) -> None:
    config = load_config(config_path)
    steps = config.get("steps", [])

    if not steps:
        print("No steps defined in config.")
        return

    # Only run stage 1 steps
    steps = [s for s in steps if s.get("stage") == 1]

    if not steps:
        print("No stage 1 steps found in config.")
        return

    # Resolve which steps to run
    step_names = [s["name"] for s in steps]

    # --from: drop everything before the named step
    if from_step:
        if from_step not in step_names:
            print(f"[error] --from step {from_step!r} not found. Available: {step_names}")
            sys.exit(1)
        start_idx = step_names.index(from_step)
        steps = steps[start_idx:]

    # --only: whitelist
    if only:
        unknown = set(only) - set(step_names)
        if unknown:
            print(f"[error] --only contains unknown steps: {unknown}")
            sys.exit(1)
        steps = [s for s in steps if s["name"] in only]

    # --skip: blacklist
    if skip:
        steps = [s for s in steps if s["name"] not in skip]

    print(f"\n{'='*60}")
    print(f"  SML QA Generation Pipeline  |  config: {config_path}")
    print(f"  Steps to run: {[s['name'] for s in steps]}")
    print(f"{'='*60}\n")

    total_start = time.time()

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
            handler(step)
            elapsed = time.time() - step_start
            print(f"\n  ✓  {name} completed in {elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.time() - step_start
            print(f"\n  ✗  {name} FAILED after {elapsed:.1f}s: {exc}")
            raise SystemExit(1) from exc

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total_elapsed:.1f}s")
    print(f"{'='*60}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SML pipeline runner — executes steps from a YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_QAgen_pipeline.py
  python run_QAgen_pipeline.py --config pipeline_config.yml
  python run_QAgen_pipeline.py --only mineru,parsing
  python run_QAgen_pipeline.py --skip mineru
  python run_QAgen_pipeline.py --from filtering
        """,
    )
    parser.add_argument(
        "--config", default="pipeline_config.yml",
        help="Path to YAML config file (default: pipeline_config.yml)",
    )
    parser.add_argument(
        "--only",
        help="Comma-separated list of step names to run (ignores enabled flag)",
    )
    parser.add_argument(
        "--skip",
        help="Comma-separated list of step names to skip",
    )
    parser.add_argument(
        "--from", dest="from_step",
        help="Start pipeline from this step (inclusive), skipping earlier steps",
    )

    args = parser.parse_args()

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    skip = [s.strip() for s in args.skip.split(",")] if args.skip else None

    run_QAgen_pipeline(
        config_path=args.config,
        only=only,
        skip=skip,
        from_step=args.from_step,
    )


if __name__ == "__main__":
    main()
