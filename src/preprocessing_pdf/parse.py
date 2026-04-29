import re
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


class Parser:
    def __init__(self, min_words=30, paragraph_window=3, paragraph_overlap=1):
        self.min_words = min_words
        self.paragraph_window = paragraph_window
        self.paragraph_overlap = paragraph_overlap

    def chunk_by_headings(self, content, source_file=""):
        """Split markdown by headings, tracking breadcrumb hierarchy."""
        chunks = []
        lines = content.split('\n')
        heading_re = re.compile(r'^(#{1,6})\s+(.+)$')

        breadcrumb_stack = []  # list of (level, heading_text)
        current_heading = None
        current_lines = []

        def flush():
            if not current_heading:
                return
            text = '\n'.join(current_lines).strip()
            if not text:
                return
            word_count = len(text.split())
            if word_count < self.min_words:
                if chunks:
                    prev = chunks[-1]
                    combined = prev['content'] + '\n\n' + text
                    chunks[-1] = {**prev, 'content': combined, 'word_count': len(combined.split())}
                return
            breadcrumb = ' > '.join(h for _, h in breadcrumb_stack[:-1])
            chunks.append({
                'source_file': source_file,
                'breadcrumb': breadcrumb,
                'heading': current_heading,
                'content': text,
                'word_count': word_count
            })

        for line in lines:
            m = heading_re.match(line)
            if m:
                flush()
                level = len(m.group(1))
                heading_text = m.group(2).strip()
                breadcrumb_stack = [(l, h) for l, h in breadcrumb_stack if l < level]
                breadcrumb_stack.append((level, heading_text))
                current_heading = heading_text
                current_lines = []
            else:
                if line.strip():
                    current_lines.append(line)

        flush()
        return chunks

    def chunk_by_paragraphs(self, content, source_file=""):
        """Split by double newlines with sliding window overlap."""
        paragraphs = [p.strip() for p in re.split(r'\n\n+', content) if p.strip()]
        if not paragraphs:
            return []

        chunks = []
        stride = max(1, self.paragraph_window - self.paragraph_overlap)

        for i in range(0, len(paragraphs), stride):
            window = paragraphs[i:i + self.paragraph_window]
            text = '\n\n'.join(window)
            if len(text.split()) >= self.min_words:
                chunks.append({
                    'source_file': source_file,
                    'breadcrumb': '',
                    'heading': '',
                    'content': text,
                    'word_count': len(text.split())
                })

        return chunks

    def parse_file(self, md_file_path):
        """Parse a markdown file. Heading-based; falls back to paragraph-based."""
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        source_file = Path(md_file_path).name
        chunks = self.chunk_by_headings(content, source_file)

        if not chunks:
            chunks = self.chunk_by_paragraphs(content, source_file)

        return chunks

    def process_paper(self, paper_folder_path):
        """Parse all markdown files in a paper folder, save raw chunks as JSONL."""
        paper_path = Path(paper_folder_path)
        md_files = list(paper_path.glob('*.md'))

        if not md_files:
            print(f"No markdown file found in {paper_folder_path}")
            return []

        md_file = md_files[0]
        print(f"Parsing {md_file.name}...")

        chunks = self.parse_file(md_file)

        output_file = paper_path / f"{md_file.stem}_chunks_raw.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

        print(f"  → {len(chunks)} chunks → {output_file.name}")
        return chunks

    def process_all_papers_by_folder(self, base_directory):
        """Process all paper subfolders in a directory."""
        base_path = Path(base_directory)
        for paper_folder in sorted(base_path.iterdir()):
            if paper_folder.is_dir():
                print(f"\n=== {paper_folder.name} ===")
                try:
                    self.process_paper(paper_folder)
                except Exception as e:
                    print(f"Error processing {paper_folder.name}: {e}")

    def process_all_papers_by_batches(self, base_directory, batch_start=1, batch_end=None):
        """Process batches sequentially."""
        base_path = Path(base_directory)
        batch_folders = sorted([f for f in base_path.iterdir() if f.is_dir()])

        for i, batch_folder in enumerate(batch_folders):
            batch_num = i + 1
            if batch_num < batch_start:
                continue
            if batch_end is not None and batch_num > batch_end:
                continue
            print(f"\n=== Batch {batch_num}: {batch_folder.name} ===")
            try:
                self.process_all_papers_by_folder(batch_folder / "papers")
            except Exception as e:
                print(f"Error: {e}")

    def parallel_process_all_papers_by_batches(self, base_directory, batch_start=1, batch_end=None, workers=5):
        """Process batches in parallel using ThreadPoolExecutor."""
        base_path = Path(base_directory)
        batch_folders = sorted([f for f in base_path.iterdir() if f.is_dir()])

        to_process = []
        for i, folder in enumerate(batch_folders):
            batch_num = i + 1
            if batch_num < batch_start:
                continue
            if batch_end is not None and batch_num > batch_end:
                continue
            to_process.append((batch_num, folder))

        def run_batch(args):
            batch_num, folder = args
            print(f"\n=== Batch {batch_num}: {folder.name} ===")
            try:
                self.process_all_papers_by_folder(folder / "papers")
                return f"Done: batch {batch_num}"
            except Exception as e:
                return f"Error batch {batch_num}: {e}"

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_batch, args): args for args in to_process}
            for future in as_completed(futures):
                print(future.result())

        print(f"\nFinished {len(to_process)} batches with {workers} workers.")


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Parse markdown files into chunks for QA generation"
    )
    arg_parser.add_argument(
        "--mode", choices=["single", "folder", "batches", "batches-parallel"],
        default="batches-parallel"
    )
    arg_parser.add_argument("-i", "--input-dir", type=str, required=True)
    arg_parser.add_argument("--paper-folder", type=str)
    arg_parser.add_argument("--batch-start", type=int, default=1)
    arg_parser.add_argument("--batch-end", type=int)
    arg_parser.add_argument("-w", "--workers", type=int, default=5)
    arg_parser.add_argument("--min-words", type=int, default=50)
    arg_parser.add_argument("--paragraph-window", type=int, default=3)
    arg_parser.add_argument("--paragraph-overlap", type=int, default=1)

    args = arg_parser.parse_args()

    parser = Parser(
        min_words=args.min_words,
        paragraph_window=args.paragraph_window,
        paragraph_overlap=args.paragraph_overlap
    )

    if args.mode == "single":
        if not args.paper_folder:
            print("Error: --paper-folder required for single mode")
            exit(1)
        parser.process_paper(args.paper_folder)
    elif args.mode == "folder":
        parser.process_all_papers_by_folder(args.input_dir)
    elif args.mode == "batches":
        parser.process_all_papers_by_batches(args.input_dir, args.batch_start, args.batch_end)
    elif args.mode == "batches-parallel":
        parser.parallel_process_all_papers_by_batches(
            args.input_dir, args.batch_start, args.batch_end, args.workers
        )
