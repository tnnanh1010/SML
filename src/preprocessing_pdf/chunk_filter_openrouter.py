import asyncio
import json
import os
import re
from pathlib import Path
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from openai import AsyncOpenAI


FILTER_PROMPT = """You are a text quality filter for a document QA pipeline.

Evaluate whether the following text chunk contains enough meaningful, factual information to generate at least one valid question-answer pair.

Answer "yes" if the chunk:
- Contains clear factual statements, rules, explanations, or descriptions
- Has enough context to form a self-contained question and answer
- Includes definitions, procedures, conditions, or examples

Answer "no" if the chunk:
- Consists mostly of navigation content, headers, or fragment sentences
- Contains corrupted, garbled, or non-readable text
- Provides no standalone informational value on its own

Text:
{content}

Answer with exactly "yes" or "no"."""


def parse_yes_no(response: str) -> bool:
    match = re.search(r'\b(yes|no)\b', response.strip().lower())
    if match:
        return match.group(1) == 'yes'
    return True  # keep on ambiguous response


class ChunkFilter:
    def __init__(
        self,
        api_key: str = None,
        model: str = "openai/gpt-oss-120b:free",
        max_concurrent: int = 10
    ):
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ["OPENROUTER_API_KEY"]
        )

    async def filter_chunk(self, chunk: dict) -> bool:
        async with self.semaphore:
            messages = [{"role": "user", "content": FILTER_PROMPT.format(content=chunk["content"])}]
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0
                )
                result_text = response.choices[0].message.content or ""
                return parse_yes_no(result_text)
            except Exception as e:
                print(f"Error filtering chunk from {chunk.get('source_file', '')}: {e}")
                return True  # keep on error

    async def filter_file(self, input_path: str, output_path: str = None):
        """Filter a single raw chunks JSONL file and write kept chunks to output."""
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.parent / input_path.name.replace('_chunks_raw.jsonl', '_chunks.jsonl')

        chunks = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        if not chunks:
            print(f"No chunks in {input_path.name}, skipping.")
            return []

        tasks = [self.filter_chunk(chunk) for chunk in chunks]
        results = await atqdm.gather(*tasks, desc=f"Filtering {input_path.name}", total=len(tasks))

        kept = [chunk for chunk, keep in zip(chunks, results) if keep]

        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in kept:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

        print(f"  → {len(kept)}/{len(chunks)} chunks kept → {Path(output_path).name}")
        return kept

    async def filter_folder(self, papers_path: str):
        """Filter all raw chunk JSONL files found under papers_path."""
        raw_files = list(Path(papers_path).rglob('*_chunks_raw.jsonl'))

        if not raw_files:
            print(f"No raw chunk files found in {papers_path}")
            return

        async def filter_one(f):
            try:
                await self.filter_file(f)
            except Exception as e:
                print(f"Error filtering {f.name}: {e}")

        await asyncio.gather(*[filter_one(f) for f in raw_files])

    async def filter_batches(
        self,
        base_directory: str,
        batch_start: int = 1,
        batch_end: int = None
    ):
        """Filter chunks across multiple batch folders."""
        base_path = Path(base_directory)
        batch_folders = sorted([f for f in base_path.iterdir() if f.is_dir()])

        batch_pbar = tqdm(batch_folders, desc="Batches", unit="batch")
        for i, batch_folder in enumerate(batch_pbar):
            batch_num = i + 1
            if batch_num < batch_start:
                continue
            if batch_end is not None and batch_num > batch_end:
                continue

            papers_path = batch_folder / "papers"
            if not papers_path.exists():
                continue

            batch_pbar.set_description(f"Batch {batch_folder.name}")
            print(f"\n=== Batch {batch_num}: {batch_folder.name} ===")

            try:
                await self.filter_folder(papers_path)
            except Exception as e:
                print(f"Error on batch {batch_num}: {e}")


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Filter raw chunks using OpenRouter API"
    )
    arg_parser.add_argument("--mode", choices=["file", "folder", "batches"], default="batches")
    arg_parser.add_argument("-i", "--input-dir", type=str, required=True)
    arg_parser.add_argument("--input-file", type=str)
    arg_parser.add_argument("--batch-start", type=int, default=1)
    arg_parser.add_argument("--batch-end", type=int)
    arg_parser.add_argument("--api-key", type=str, default=None,
                            help="OpenRouter API key (defaults to OPENROUTER_API_KEY env var)")
    arg_parser.add_argument("--model", type=str, default="openai/gpt-oss-120b:free")
    arg_parser.add_argument("--max-concurrent", type=int, default=10)

    args = arg_parser.parse_args()

    chunk_filter = ChunkFilter(
        api_key=args.api_key,
        model=args.model,
        max_concurrent=args.max_concurrent
    )

    if args.mode == "file":
        if not args.input_file:
            print("Error: --input-file required for file mode")
            exit(1)
        asyncio.run(chunk_filter.filter_file(args.input_file))

    elif args.mode == "folder":
        asyncio.run(chunk_filter.filter_folder(args.input_dir))

    elif args.mode == "batches":
        asyncio.run(chunk_filter.filter_batches(
            args.input_dir,
            batch_start=args.batch_start,
            batch_end=args.batch_end
        ))
