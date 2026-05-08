import asyncio
import json
import os
import re
from pathlib import Path
from typing import List
from openai import AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm


# ─── Prompts ──────────────────────────────────────────────────────────────────

QUESTION_GENERATION_PROMPT = """You are a professional QA dataset generator specialized in creating training data for language models.

Generate {n_questions} self-contained questions from the text below.

SELF-CONTAINED means each question must stand alone without the source text:
- BAD: "What does it recommend?" → GOOD: "What does <Organization> recommend regarding <topic>?"

Rules:
- Write questions as if testing domain knowledge, NOT reading comprehension — never reference any source, passage, text, or document
- Questions must NOT use bare pronouns ("it", "they", "this") as the subject
- Vary question types: factual (who/what/when), procedural (how to), conditional (what happens when), definitional (what is X)
- Avoid trivial or overly broad questions
- Each question must be answerable using ONLY the provided text

Return ONLY a valid JSON array of question strings, no extra text:
["question1", "question2", ...]

Text:
{content}"""


ANSWER_GENERATION_PROMPT = """You are a professional QA dataset generator. Answer the question below using ONLY the provided text.

Rules:
- The answer must be a complete sentence or structured list — never a bare value, name, or number in isolation
- BAD: "0101601092" → GOOD: "FPT Software Company Limited's business registration number is 0101601092."
- BAD: "Three steps." → GOOD: "There are three steps: (1) ..., (2) ..., (3) ..."
- The answer must be accurate and grounded strictly in the text — no outside knowledge
- NEVER reference the source material — do NOT say "based on the text", "according to the document", "the passage states", "as mentioned in the text", or any similar phrasing
- Write the answer as if you are stating a fact directly, not summarizing a document
- If the text does not contain enough information to answer, respond with exactly: "UNANSWERABLE"

Text:
{content}

Question:
{question}

Answer:"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_questions(response_text: str) -> List[str]:
    """Extract question list from model response."""
    text = re.sub(r'```(?:json)?\s*', '', response_text).strip().rstrip('`').strip()

    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []

    try:
        questions = json.loads(match.group())
        return [q for q in questions if isinstance(q, str) and q.strip()]
    except json.JSONDecodeError:
        return []


def extract_qa_pairs(response_text: str) -> List[dict]:
    """Extract QA pairs from model response (legacy, kept for compatibility)."""
    text = re.sub(r'```(?:json)?\s*', '', response_text).strip().rstrip('`').strip()

    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []

    try:
        pairs = json.loads(match.group())
        return [
            p for p in pairs
            if isinstance(p, dict) and 'question' in p and 'answer' in p
        ]
    except json.JSONDecodeError:
        return []


# ─── QA Generator ─────────────────────────────────────────────────────────────

class QAGenerator:
    def __init__(
        self,
        api_key: str = None,
        model: str = "openai/gpt-oss-120b:free",
        model_questions: str = None,
        model_answers: str = None,
        max_questions: int = 5,
        enable_reasoning: bool = False,
        max_concurrent: int = 10
    ):
        self.model = model
        self.model_questions = model_questions or model
        self.model_answers = model_answers or model
        self.max_questions = max_questions
        self.enable_reasoning = enable_reasoning
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ["OPENROUTER_API_KEY"]
        )

    def _adaptive_n_questions(self, word_count: int) -> int:
        return max(1, min(self.max_questions, word_count // 100))

    async def generate_questions(self, chunk: dict) -> List[str]:
        """Step 1: Generate questions from a chunk (LLM call 1)."""
        async with self.semaphore:
            n_questions = self._adaptive_n_questions(chunk.get("word_count", 100))
            prompt = QUESTION_GENERATION_PROMPT.format(
                n_questions=n_questions,
                content=chunk["content"]
            )
            kwargs = dict(
                model=self.model_questions,
                messages=[{"role": "user", "content": prompt}]
            )
            if self.enable_reasoning:
                kwargs["extra_body"] = {"reasoning": {"enabled": True}}

            try:
                response = await self.client.chat.completions.create(**kwargs)
                response_text = response.choices[0].message.content or ""
                return extract_questions(response_text)
            except Exception as e:
                print(f"Error generating questions for '{chunk.get('heading', chunk.get('source_file', ''))}': {e}")
                return []

    async def generate_answer(self, chunk: dict, question: str) -> str:
        """Step 2: Generate answer for a single question (LLM call 2)."""
        async with self.semaphore:
            prompt = ANSWER_GENERATION_PROMPT.format(
                content=chunk["content"],
                question=question
            )
            kwargs = dict(
                model=self.model_answers,
                messages=[{"role": "user", "content": prompt}]
            )
            if self.enable_reasoning:
                kwargs["extra_body"] = {"reasoning": {"enabled": True}}

            try:
                response = await self.client.chat.completions.create(**kwargs)
                answer = response.choices[0].message.content or ""
                return answer.strip()
            except Exception as e:
                print(f"Error generating answer for question '{question[:50]}...': {e}")
                return ""

    async def generate_from_chunk(self, chunk: dict) -> List[dict]:
        """Generate QA pairs from a single chunk using 2-step process."""
        # Step 1: Generate questions
        questions = await self.generate_questions(chunk)
        if not questions:
            return []

        # Step 2: Generate answers for each question (parallel)
        answers = await asyncio.gather(*[
            self.generate_answer(chunk, q) for q in questions
        ])

        # Combine and filter out unanswerable/empty
        results = []
        for question, answer in zip(questions, answers):
            if not answer or answer.upper() == "UNANSWERABLE":
                continue
            results.append({
                "source_file": chunk.get("source_file", ""),
                "breadcrumb": chunk.get("breadcrumb", ""),
                "heading": chunk.get("heading", ""),
                "context": chunk["content"],
                "question": question,
                "answer": answer
            })

        return results

    async def process_file(self, input_path: str, output_path: str = None):
        """Generate QA pairs from a filtered chunks JSONL file."""
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.parent / input_path.name.replace('_chunks.jsonl', '_qa.jsonl')

        chunks = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        if not chunks:
            print(f"No chunks in {input_path.name}, skipping.")
            return []

        tasks = [self.generate_from_chunk(chunk) for chunk in chunks]
        results = await atqdm.gather(*tasks, desc=f"QA gen {input_path.name}", total=len(tasks))

        all_pairs = [pair for batch in results for pair in batch]

        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in all_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')

        print(f"  → {len(all_pairs)} QA pairs → {Path(output_path).name}")
        return all_pairs

    async def process_folder(self, papers_path: str):
        """Generate QA for all filtered chunk files under a papers folder."""
        chunk_files = list(Path(papers_path).rglob('*_chunks.jsonl'))

        if not chunk_files:
            print(f"No chunk files found in {papers_path}")
            return

        async def process_one(f):
            try:
                await self.process_file(f)
            except Exception as e:
                print(f"Error on {f.name}: {e}")

        await asyncio.gather(*[process_one(f) for f in chunk_files])

    async def process_batches(
        self,
        base_directory: str,
        batch_start: int = 1,
        batch_end: int = None
    ):
        """Generate QA across multiple batch folders."""
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
                await self.process_folder(papers_path)
            except Exception as e:
                print(f"Error on batch {batch_num}: {e}")


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Generate QA pairs from filtered chunks using OpenRouter API (2-step: questions then answers)"
    )
    arg_parser.add_argument(
        "--mode", choices=["file", "folder", "batches"], default="batches"
    )
    arg_parser.add_argument("-i", "--input-dir", type=str, required=True)
    arg_parser.add_argument("--input-file", type=str)
    arg_parser.add_argument("--batch-start", type=int, default=1)
    arg_parser.add_argument("--batch-end", type=int)
    arg_parser.add_argument("--api-key", type=str, default=None,
                            help="OpenRouter API key (defaults to OPENROUTER_API_KEY env var)")
    arg_parser.add_argument("--model", type=str, default="openai/gpt-oss-120b:free",
                            help="Default model for both steps")
    arg_parser.add_argument("--model-questions", type=str, default=None,
                            help="Model for question generation (defaults to --model)")
    arg_parser.add_argument("--model-answers", type=str, default=None,
                            help="Model for answer generation (defaults to --model)")
    arg_parser.add_argument("--max-questions", type=int, default=5)
    arg_parser.add_argument("--enable-reasoning", action="store_true")
    arg_parser.add_argument("--max-concurrent", type=int, default=10)

    args = arg_parser.parse_args()

    generator = QAGenerator(
        api_key=args.api_key,
        model=args.model,
        model_questions=args.model_questions,
        model_answers=args.model_answers,
        max_questions=args.max_questions,
        enable_reasoning=args.enable_reasoning,
        max_concurrent=args.max_concurrent
    )

    if args.mode == "file":
        if not args.input_file:
            print("Error: --input-file required for file mode")
            exit(1)
        asyncio.run(generator.process_file(args.input_file))

    elif args.mode == "folder":
        asyncio.run(generator.process_folder(args.input_dir))

    elif args.mode == "batches":
        asyncio.run(generator.process_batches(
            args.input_dir,
            batch_start=args.batch_start,
            batch_end=args.batch_end
        ))
