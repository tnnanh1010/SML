"""
SML Serving Pipeline (Stage 3)
===============================
Converts HF models to GGUF, registers with Ollama, and launches
a side-by-side comparison UI. Supports two backends:
  - ollama (default): lightweight, uses Ollama API
  - transformers: loads models directly into Python

Steps (ollama mode): convert → register → compare
Steps (transformers mode): compare only

Usage:
    python serve_slm.py                          # ollama mode (default)
    python serve_slm.py --only compare           # just launch UI
    python serve_slm.py --skip compare           # convert + register only
    python serve_slm.py --from register          # skip conversion
    python serve_slm.py --backend transformers   # use transformers instead
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml


# ─── config loading ───────────────────────────────────────────────────────────

def _expand_env_vars(value):
    if isinstance(value, str):
        def replacer(match):
            var = match.group(1)
            result = os.environ.get(var)
            if result is None:
                return match.group(0)
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
    return _expand_env_vars(raw)


# ─── QA examples loader ───────────────────────────────────────────────────────

def load_qa_examples(qa_path: str | Path | None) -> list[dict]:
    if qa_path is None:
        return []
    path = Path(qa_path)
    if not path.exists():
        print(f"  [warn] QA examples file not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} example QA pairs from {path.name}")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# OLLAMA BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

OLLAMA_API_URL = "http://localhost:11434/api/chat"


def run_convert(ollama_cfg: dict) -> None:
    """Convert HuggingFace safetensors to GGUF for each model."""
    llama_cpp_dir = Path(ollama_cfg["llama_cpp_dir"]).expanduser().resolve()
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        raise RuntimeError(
            f"convert_hf_to_gguf.py not found at {convert_script}. "
            f"Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp {llama_cpp_dir}"
        )

    for model_cfg in ollama_cfg["models"]:
        name = model_cfg["name"]
        hf_dir = Path(model_cfg["hf_model_dir"]).expanduser().resolve()
        gguf_output = Path(model_cfg["gguf_output"]).expanduser()

        if gguf_output.exists():
            print(f"  [skip] {name}: {gguf_output} already exists")
            continue

        if not hf_dir.exists():
            print(f"  [error] {name}: HF model dir not found: {hf_dir}")
            continue

        gguf_output.parent.mkdir(parents=True, exist_ok=True)

        print(f"  Converting {name}: {hf_dir} → {gguf_output}")
        cmd = [
            sys.executable, str(convert_script),
            str(hf_dir),
            "--outfile", str(gguf_output.resolve()),
            "--outtype", "f16",
        ]

        result = subprocess.run(cmd, cwd=str(llama_cpp_dir))
        if result.returncode != 0:
            raise RuntimeError(f"Conversion failed for {name} (exit code {result.returncode})")
        print(f"  ✓ {name} converted")


def run_register(ollama_cfg: dict) -> None:
    """Generate Modelfiles and register models with Ollama."""
    for model_cfg in ollama_cfg["models"]:
        name = model_cfg["name"]
        ollama_name = model_cfg.get("ollama_name", name)
        gguf_path = Path(model_cfg["gguf_output"]).expanduser().resolve()
        system_prompt = model_cfg.get("system_prompt", "You are a helpful assistant.")
        parameters = model_cfg.get("parameters", {})

        if not gguf_path.exists():
            print(f"  [error] {name}: GGUF not found at {gguf_path}. Run convert first.")
            continue

        lines = [f"FROM {gguf_path}"]
        lines.append(f'SYSTEM """{system_prompt}"""')
        for param, value in parameters.items():
            if isinstance(value, list):
                for item in value:
                    lines.append(f"PARAMETER {param} {item}")
            else:
                lines.append(f"PARAMETER {param} {value}")
        modelfile_content = "\n".join(lines)

        modelfile_path = gguf_path.parent / f"Modelfile.{ollama_name}"
        modelfile_path.write_text(modelfile_content, encoding="utf-8")

        print(f"  Registering {ollama_name} from {gguf_path.name}...")
        subprocess.run(["ollama", "rm", ollama_name], capture_output=True)

        result = subprocess.run(["ollama", "create", ollama_name, "-f", str(modelfile_path)])
        if result.returncode != 0:
            print(f"  [error] Failed to register {ollama_name}")
            continue

        print(f"  ✓ {ollama_name} registered")

    print("\n  Registered models:")
    subprocess.run(["ollama", "list"])


def query_ollama(model_name: str, question: str, system_prompt: str = None) -> str:
    """Send a chat request to Ollama API."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    payload = {"model": model_name, "messages": messages, "stream": False}

    try:
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()
    except requests.exceptions.ConnectionError:
        return "[Error] Cannot connect to Ollama. Is it running? (ollama serve)"
    except requests.exceptions.Timeout:
        return "[Error] Request timed out."
    except Exception as e:
        return f"[Error] {e}"


def run_compare_ollama(ollama_cfg: dict, qa_examples_path: str = None) -> None:
    """Launch Gradio comparison UI using Ollama API."""
    import gradio as gr

    models = ollama_cfg["models"]
    if len(models) < 2:
        print("[error] Need at least 2 models in config for comparison.")
        return

    model1_name = models[0].get("ollama_name", models[0]["name"])
    model2_name = models[1].get("ollama_name", models[1]["name"])
    system1 = models[0].get("system_prompt", "")
    system2 = models[1].get("system_prompt", "")

    qa_pairs = load_qa_examples(qa_examples_path)

    def compare(question: str) -> tuple[str, str, str]:
        if not question.strip():
            return "", "", ""
        with ThreadPoolExecutor(max_workers=2) as executor:
            t0 = time.perf_counter()
            f1 = executor.submit(query_ollama, model1_name, question, system1)
            f2 = executor.submit(query_ollama, model2_name, question, system2)
            answer1 = f1.result()
            answer2 = f2.result()
            elapsed = time.perf_counter() - t0
        return answer1, answer2, f"Generated in {elapsed:.2f}s"

    _build_gradio_ui(model1_name, model2_name, compare, qa_pairs,
                     port=ollama_cfg.get("port", 7860),
                     share=ollama_cfg.get("share", False))


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMERS BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question directly "
    "using only your own knowledge. Do not use any tools, do not search "
    "the internet, and do not call any functions."
)


def run_compare_transformers(comparison_cfg: dict) -> None:
    """Launch Gradio comparison UI using transformers (loads models into memory)."""
    import gradio as gr
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model1_id = comparison_cfg["model1"]
    model2_id = comparison_cfg["model2"]
    device = comparison_cfg.get("device", "auto")
    max_tokens = comparison_cfg.get("max_tokens", 512)
    qa_examples_path = comparison_cfg.get("qa_examples")

    # Determine devices
    if device == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            dev1, dev2 = "cuda:0", "cuda:1"
        elif torch.cuda.is_available():
            dev1, dev2 = "cuda:0", "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev1, dev2 = "mps", "mps"
        else:
            dev1, dev2 = "cpu", "cpu"
    else:
        dev1, dev2 = device, device

    print(f"\nDevices: Model1={dev1}, Model2={dev2}")

    def load_model(model_id, dev):
        print(f"Loading {model_id} on {dev}...")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if "cuda" in dev else torch.float32,
        ).to(dev)
        model.eval()
        print(f"  ✓ {model_id} loaded")
        return processor, model

    processor1, model1 = load_model(model1_id, dev1)
    processor2, model2 = load_model(model2_id, dev2)
    print("\nBoth models loaded. Starting UI...\n")

    def run_inference(processor, model, question):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": question}]},
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=False,
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        return processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()

    qa_pairs = load_qa_examples(qa_examples_path)

    def compare(question: str) -> tuple[str, str, str]:
        if not question.strip():
            return "", "", ""
        with ThreadPoolExecutor(max_workers=2) as executor:
            t0 = time.perf_counter()
            f1 = executor.submit(run_inference, processor1, model1, question)
            f2 = executor.submit(run_inference, processor2, model2, question)
            answer1 = f1.result()
            answer2 = f2.result()
            elapsed = time.perf_counter() - t0
        return answer1, answer2, f"Generated in {elapsed:.2f}s"

    _build_gradio_ui(model1_id, model2_id, compare, qa_pairs,
                     port=comparison_cfg.get("port", 7860),
                     share=comparison_cfg.get("share", False))


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════════

def _build_gradio_ui(model1_name, model2_name, compare_fn, qa_pairs, port=7860, share=False):
    """Build and launch the shared Gradio comparison UI."""
    import gradio as gr

    with gr.Blocks(title="SML Model Comparison", theme=gr.themes.Soft()) as app:
        gr.Markdown("# SML Model Comparison")
        gr.Markdown(
            f"**Model A:** `{model1_name}`  \n"
            f"**Model B:** `{model2_name}`  \n"
            f"Type a question below to see both responses side-by-side."
        )

        with gr.Row():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Type your question here...",
                lines=3, scale=4,
            )
            submit_btn = gr.Button("Compare", variant="primary", scale=1)

        ground_truth_display = gr.Textbox(
            label="Expected Answer (ground truth)", lines=3, interactive=False,
        )
        timing_display = gr.Textbox(label="Timing", interactive=False)

        with gr.Row():
            output1 = gr.Textbox(label=f"Model A: {model1_name}", lines=15, interactive=False)
            output2 = gr.Textbox(label=f"Model B: {model2_name}", lines=15, interactive=False)

        submit_btn.click(fn=compare_fn, inputs=[question_input], outputs=[output1, output2, timing_display])
        question_input.submit(fn=compare_fn, inputs=[question_input], outputs=[output1, output2, timing_display])

        if qa_pairs:
            example_rows = [[p["question"], p.get("answer", "")] for p in qa_pairs[:10]]
            gt_input = gr.Textbox(visible=False)

            def fill_from_example(question: str, gt: str):
                return question, gt

            gr.Examples(
                examples=example_rows,
                inputs=[question_input, gt_input],
                outputs=[question_input, ground_truth_display],
                fn=fill_from_example,
                run_on_click=True,
                label="Examples",
            )

    app.launch(server_port=port, share=share)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

OLLAMA_STEPS = ["convert", "register", "compare"]

OLLAMA_HANDLERS = {
    "convert":  run_convert,
    "register": run_register,
    "compare":  run_compare_ollama,
}


def run_pipeline(
    config_path: str,
    backend: str = "ollama",
    only: list[str] | None = None,
    skip: list[str] | None = None,
    from_step: str | None = None,
) -> None:
    config = load_config(config_path)
    qa_examples = config.get("comparison", {}).get("qa_examples", "output_batches/test.json")

    if backend == "transformers":
        comparison_cfg = config.get("comparison", {})
        if not comparison_cfg:
            print("No 'comparison' section found in config.")
            return
        print(f"\n{'='*60}")
        print(f"  SML Serving (transformers)  |  config: {config_path}")
        print(f"{'='*60}\n")
        run_compare_transformers(comparison_cfg)
        return

    # Ollama backend
    ollama_cfg = config.get("ollama")
    if not ollama_cfg:
        print("No 'ollama' section found in config.")
        return

    steps = list(OLLAMA_STEPS)

    if from_step:
        if from_step not in steps:
            print(f"[error] --from {from_step!r} not found. Available: {steps}")
            sys.exit(1)
        steps = steps[steps.index(from_step):]
    if only:
        steps = [s for s in steps if s in only]
    if skip:
        steps = [s for s in steps if s not in skip]

    print(f"\n{'='*60}")
    print(f"  SML Serving (ollama)  |  config: {config_path}")
    print(f"  Steps: {steps}")
    print(f"{'='*60}\n")

    for step in steps:
        print(f"\n{'─'*60}")
        print(f"  ▶  {step}")
        print(f"{'─'*60}")

        t0 = time.time()
        try:
            if step == "compare":
                OLLAMA_HANDLERS[step](ollama_cfg, qa_examples_path=qa_examples)
            else:
                OLLAMA_HANDLERS[step](ollama_cfg)
            elapsed = time.time() - t0
            print(f"\n  ✓  {step} completed in {elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"\n  ✗  {step} FAILED after {elapsed:.1f}s: {exc}")
            raise SystemExit(1) from exc


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SML serving: convert → register → compare (ollama) or direct compare (transformers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python serve_slm.py                          # ollama: full pipeline
  python serve_slm.py --only compare           # ollama: just UI
  python serve_slm.py --from register          # ollama: skip convert
  python serve_slm.py --backend transformers   # transformers: direct load
        """,
    )
    parser.add_argument("--config", default="pipeline_config.yml")
    parser.add_argument("--backend", choices=["ollama", "transformers"], default="ollama",
                        help="Inference backend (default: ollama)")
    parser.add_argument("--only", help="Comma-separated step names (ollama mode)")
    parser.add_argument("--skip", help="Comma-separated step names to skip (ollama mode)")
    parser.add_argument("--from", dest="from_step", help="Start from this step (ollama mode)")
    args = parser.parse_args()

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    skip = [s.strip() for s in args.skip.split(",")] if args.skip else None

    run_pipeline(
        config_path=args.config,
        backend=args.backend,
        only=only,
        skip=skip,
        from_step=args.from_step,
    )


if __name__ == "__main__":
    main()
