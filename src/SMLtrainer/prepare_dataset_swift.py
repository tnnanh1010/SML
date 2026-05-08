"""
Convert QA pairs generated from internal documents to ms-swift training format.

Input (JSONL from qa_generator.py):
{
    "source_file": "...",
    "breadcrumb": "...",
    "heading": "...",
    "context": "...",
    "question": "...",
    "answer": "..."
}

Output (ms-swift JSONL):
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
"""
import json
import os
import argparse
from pathlib import Path


SYSTEM_PROMPT = """You are a helpful assistant. Answer questions accurately, clearly, and concisely."""


def convert_to_swift_format(input_file: str, output_file: str) -> int:
    print(f"Loading data from {input_file}...")

    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"Loaded {len(data)} QA pairs")

    converted, skipped = [], 0

    for item in data:
        if not item.get("question") or not item.get("answer"):
            skipped += 1
            continue

        converted.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"].strip()},
                {"role": "assistant", "content": item["answer"].strip()}
            ]
        })

    print(f"Skipped {skipped} invalid samples")
    print(f"Valid samples: {len(converted)}")

    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Done: {len(converted)} samples saved to {output_file}")
    return len(converted)


def collect_qa_files(base_directory: str) -> list:
    """Collect all _qa.jsonl files from batch folder structure."""
    return sorted(Path(base_directory).rglob('*_qa.jsonl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert QA pairs to ms-swift training format"
    )
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input QA JSONL file, or base directory to collect all *_qa.jsonl files")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output JSONL file path")

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        qa_files = collect_qa_files(args.input)
        if not qa_files:
            print(f"No *_qa.jsonl files found in {args.input}")
            exit(1)

        print(f"Found {len(qa_files)} QA files, merging...")
        merged_tmp = Path(args.output).with_suffix('.tmp.jsonl')

        with open(merged_tmp, 'w', encoding='utf-8') as out_f:
            for qa_file in qa_files:
                with open(qa_file, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        if line.strip():
                            out_f.write(line)

        count = convert_to_swift_format(str(merged_tmp), args.output)
        merged_tmp.unlink()
    else:
        count = convert_to_swift_format(args.input, args.output)

    print(f"\nDataset ready: {args.output} ({count} samples)")
