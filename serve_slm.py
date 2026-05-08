"""
Side-by-side model comparison UI using Gradio + Transformers.

Loads two models and lets you type questions interactively,
showing both responses side-by-side.

Usage:
    python -m src.comparison.serve_slm

    # From YAML config:
    python -m src.comparison.serve_slm --config comparison_config.yml

    # Custom models/device:
    python -m src.comparison.serve_slm \
        --model1 Qwen/Qwen3.5-4B \
        --model2 tnnanh1005/Qwen3.5-4B-SML-ver2 \
        --device cpu \
        --max-tokens 512
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(model_id: str, device: str):
    """Load model and processor onto the specified device."""
    print(f"Loading {model_id} on {device}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
    ).to(device)
    model.eval()
    print(f"  ✓ {model_id} loaded")
    return processor, model


# ─── Inference ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question directly "
    "using only your own knowledge. Do not use any tools, do not search "
    "the internet, and do not call any functions."
)


def run_inference(
    processor,
    model,
    question: str,
    max_new_tokens: int = 512,
    enable_thinking: bool = False,
) -> str:
    """Generate a response from a single model."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": question}]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    response = processor.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()
    return response


# ─── QA examples loader ───────────────────────────────────────────────────────

def load_qa_examples(qa_path: str | Path | None) -> list[dict]:
    """Load QA pairs from a JSON file. Returns list of {question, answer, source_file}."""
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


# ─── Gradio app ───────────────────────────────────────────────────────────────

def build_app(
    model1_id: str,
    model2_id: str,
    device: str,
    max_tokens: int,
    qa_examples_path: str | Path | None = None,
):
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
    processor1, model1 = load_model(model1_id, dev1)
    processor2, model2 = load_model(model2_id, dev2)
    print("\nBoth models loaded. Starting UI...\n")

    # Load QA examples
    qa_pairs = load_qa_examples(qa_examples_path)
    # Build lookup: question -> ground truth answer
    qa_lookup = {pair["question"]: pair.get("answer", "") for pair in qa_pairs}

    def compare(question: str) -> tuple[str, str, str]:
        if not question.strip():
            return "", "", ""

        with ThreadPoolExecutor(max_workers=2) as executor:
            t0 = time.perf_counter()
            f1 = executor.submit(run_inference, processor1, model1, question, max_tokens)
            f2 = executor.submit(run_inference, processor2, model2, question, max_tokens)
            answer1 = f1.result()
            answer2 = f2.result()
            elapsed = time.perf_counter() - t0

        timing = f"Generated in {elapsed:.2f}s"
        return answer1, answer2, timing

    # Build UI
    with gr.Blocks(title="SML Model Comparison", theme=gr.themes.Soft()) as app:
        gr.Markdown("# SML Model Comparison")
        gr.Markdown(
            f"**Model A:** `{model1_id}`  \n"
            f"**Model B:** `{model2_id}`  \n"
            f"Type a question below to see both responses side-by-side."
        )

        with gr.Row():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Type your question here...",
                lines=3,
                scale=4,
            )
            submit_btn = gr.Button("Compare", variant="primary", scale=1)

        ground_truth_display = gr.Textbox(
            label="Expected Answer (ground truth)",
            lines=3,
            interactive=False,
        )

        timing_display = gr.Textbox(label="Timing", interactive=False)

        with gr.Row():
            output1 = gr.Textbox(
                label=f"Model A: {model1_id}",
                lines=15,
                interactive=False,
            )
            output2 = gr.Textbox(
                label=f"Model B: {model2_id}",
                lines=15,
                interactive=False,
            )

        # Wire up events — Compare does NOT touch ground_truth_display
        submit_btn.click(
            fn=compare,
            inputs=[question_input],
            outputs=[output1, output2, timing_display],
        )
        question_input.submit(
            fn=compare,
            inputs=[question_input],
            outputs=[output1, output2, timing_display],
        )

        # Suggested questions — compact Examples list that also fills ground truth
        if qa_pairs:
            # Each example row: [question_text, ground_truth_text]
            example_rows = [
                [pair["question"], pair.get("answer", "")]
                for pair in qa_pairs[:10]
            ]
            # Hidden textbox carries the ground truth through gr.Examples
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

    return app


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Side-by-side model comparison UI")
    parser.add_argument("--config", default="pipeline_config.yml",
                        help="Path to YAML config file (reads 'comparison' section)")
    parser.add_argument("--model1", default="Qwen/Qwen3.5-4B",
                        help="First model (base)")
    parser.add_argument("--model2", default="tnnanh1005/Qwen3.5-4B-SML-ver2",
                        help="Second model (fine-tuned)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cpu, cuda:0, mps, etc.")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max new tokens per response")
    parser.add_argument("--qa-examples", default="output_batches/test.json",
                        help="Path to JSON file with example QA pairs")
    parser.add_argument("--port", type=int, default=7860,
                        help="Gradio server port")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    args = parser.parse_args()

    # Load from YAML config if provided (reads the "comparison" section)
    if args.config:
        import yaml
        with open(args.config, "r", encoding="utf-8-sig") as f:
            raw = yaml.safe_load(f)
        cfg = raw.get("comparison", {}) or {}
        model1_id = cfg.get("model1", args.model1)
        model2_id = cfg.get("model2", args.model2)
        device = cfg.get("device", args.device)
        max_tokens = cfg.get("max_tokens", args.max_tokens)
        qa_examples = cfg.get("qa_examples", args.qa_examples)
        port = cfg.get("port", args.port)
        share = cfg.get("share", args.share)
    else:
        model1_id = args.model1
        model2_id = args.model2
        device = args.device
        max_tokens = args.max_tokens
        qa_examples = args.qa_examples
        port = args.port
        share = args.share

    app = build_app(
        model1_id=model1_id,
        model2_id=model2_id,
        device=device,
        max_tokens=max_tokens,
        qa_examples_path=qa_examples,
    )
    app.launch(server_port=port, share=share)


if __name__ == "__main__":
    main()
