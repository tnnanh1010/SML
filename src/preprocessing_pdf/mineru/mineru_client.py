"""MinerU 3.0 API client with batch processing and CLI support."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests
from requests import RequestException
from tqdm import tqdm


DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
DEFAULT_CHUNK_SIZE = 50
DEFAULT_POLL_INTERVAL = 5
DEFAULT_TIMEOUT = 1800
DEFAULT_RETRIES = 3
DEFAULT_REQUEST_TIMEOUT = 60


def _with_retry(callable_fn, *, retries=DEFAULT_RETRIES, backoff_base=1.0):
    """Retry transient network operations with exponential backoff."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            return callable_fn()
        except RequestException as exc:
            last_error = exc
            if attempt == retries:
                raise
            time.sleep(backoff_base * (2**attempt))
    raise RuntimeError(f"Retry logic reached unexpected state: {last_error}")


def check_server_health(server_url, request_timeout=10, retries=DEFAULT_RETRIES):
    """Validate the server is reachable before processing starts."""

    def _request():
        response = requests.get(
            f"{server_url.rstrip('/')}/health",
            timeout=request_timeout,
        )
        response.raise_for_status()
        return response.json()

    try:
        return _with_retry(_request, retries=retries)
    except RequestException as exc:
        raise RuntimeError(
            f"Unable to reach MinerU server at {server_url}. "
            "Please ensure mineru-api or mineru-router is running."
        ) from exc


def submit_task(
    server_url,
    pdf_path,
    retries=DEFAULT_RETRIES,
    request_timeout=DEFAULT_REQUEST_TIMEOUT,
    form_data=None,
):
    """
    Submit a PDF file to the MinerU 3.0 API for processing.
    
    Args:
        server_url: Base URL of the mineru-api or mineru-router server
        pdf_path: Path to the PDF file to process
    
    Returns:
        str: The task_id returned by the API
    
    Raises:
        requests.HTTPError: If the API returns an error status code
        FileNotFoundError: If the PDF file doesn't exist
    """
    pdf_path = Path(pdf_path)
    tasks_url = f"{server_url.rstrip('/')}/tasks"
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    def _request():
        with open(pdf_path, "rb") as handle:
            files = {"files": (pdf_path.name, handle, "application/pdf")}
            data = {"return_md": "true"}
            if form_data:
                data.update(form_data)
            response = requests.post(
                tasks_url,
                files=files,
                data=data,
                timeout=request_timeout,
            )
            response.raise_for_status()
            result = response.json()
            if "task_id" not in result:
                raise RuntimeError(f"Missing task_id in response: {result}")
            return result["task_id"]

    return _with_retry(_request, retries=retries)


def poll_task(
    server_url,
    task_id,
    poll_interval=DEFAULT_POLL_INTERVAL,
    timeout=DEFAULT_TIMEOUT,
    retries=DEFAULT_RETRIES,
    request_timeout=DEFAULT_REQUEST_TIMEOUT,
):
    """
    Poll task status until completion, failure, or timeout.
    
    Args:
        server_url: Base URL of the mineru-api or mineru-router server
        task_id: The task ID returned from submit_task()
        poll_interval: Seconds to wait between status checks (default: 5)
        timeout: Maximum seconds to wait for completion (default: 1800 = 30 minutes)
    
    Returns:
        dict: The completed task result containing 'output_dir' and other fields
    
    Raises:
        TimeoutError: If the task doesn't complete within the timeout period
        RuntimeError: If the task fails with an error
        requests.HTTPError: If the API returns an error status code
    """
    start_time = time.time()
    task_url = f"{server_url.rstrip('/')}/tasks/{task_id}"
    
    while True:
        # Check if we've exceeded the timeout
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(
                f"Task {task_id} exceeded timeout of {timeout} seconds "
                f"(elapsed: {elapsed:.1f}s)"
            )
        
        # Poll the task status endpoint with retry for transient network failures.
        def _request():
            response = requests.get(task_url, timeout=request_timeout)
            response.raise_for_status()
            return response.json()

        result = _with_retry(_request, retries=retries)
        status = result.get('status')
        
        # Handle completed state
        if status == 'completed':
            return result
        
        # Handle failed state
        elif status == 'failed':
            error_msg = result.get('error', 'Unknown error')
            raise RuntimeError(f"Task {task_id} failed: {error_msg}")
        
        # Handle queued and processing states - continue polling
        elif status in ('queued', 'processing'):
            time.sleep(poll_interval)
        
        # Handle unexpected status
        else:
            raise RuntimeError(f"Task {task_id} returned unexpected status: {status}")


def create_batch_folders(output_dir, batch_number):
    """
    Create batch folder structure with papers/ and metadata/ subfolders.
    
    Creates a directory structure like:
        output_dir/
          batch_{number:04d}/
            papers/
            metadata/
    
    Args:
        output_dir: Base output directory path (str or Path)
        batch_number: Batch number (int) - will be formatted as 4-digit zero-padded
    
    Returns:
        tuple: (batch_folder, papers_folder, metadata_folder) as Path objects
    
    Raises:
        OSError: If directory creation fails due to permissions or other OS errors
    """
    output_dir = Path(output_dir)
    
    # Create batch folder with zero-padded 4-digit number
    batch_folder = output_dir / f"batch_{batch_number:04d}"
    
    # Create subfolders
    papers_folder = batch_folder / "papers"
    metadata_folder = batch_folder / "metadata"
    
    # Create all directories, handling existing directories gracefully
    papers_folder.mkdir(parents=True, exist_ok=True)
    metadata_folder.mkdir(parents=True, exist_ok=True)
    
    return batch_folder, papers_folder, metadata_folder


def save_metadata(metadata, destination):
    """Persist metadata JSON to disk."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def fetch_task_result(
    server_url,
    task_id,
    retries=DEFAULT_RETRIES,
    request_timeout=DEFAULT_REQUEST_TIMEOUT,
    result_url=None,
):
    """Fetch final task result from MinerU API."""
    url = result_url or f"{server_url.rstrip('/')}/tasks/{task_id}/result"

    def _request():
        response = requests.get(url, timeout=request_timeout)
        response.raise_for_status()
        if "application/json" not in response.headers.get("Content-Type", ""):
            raise RuntimeError("Unsupported result response format; expected JSON")
        return response.json()

    return _with_retry(_request, retries=retries)


def _save_task_result(result_payload, pdf_path, papers_folder):
    """Save MinerU JSON result into per-paper folder and return folder path."""
    paper_dir = Path(papers_folder) / Path(pdf_path).stem
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Keep raw payload for traceability.
    with open(paper_dir / "result.json", "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, ensure_ascii=False)

    # Convenience markdown file when returned by API.
    file_key = Path(pdf_path).stem
    file_result = result_payload.get("results", {}).get(file_key, {})
    md_content = file_result.get("md_content") if isinstance(file_result, dict) else None
    if md_content:
        with open(paper_dir / f"{file_key}.md", "w", encoding="utf-8") as handle:
            handle.write(md_content)

    return paper_dir


def _build_file_result(input_path, task_id=None, output_path=None, status="Error", error_message=None):
    return {
        "input_path": str(input_path),
        "output_path": str(output_path) if output_path else None,
        "status": status,
        "error_message": error_message,
        "task_id": task_id,
    }


def build_batch_metadata(batch_number, file_results):
    successful = sum(1 for item in file_results if item["status"] == "Success")
    failed = len(file_results) - successful
    return {
        "batch_number": batch_number,
        "total_files": len(file_results),
        "successful": successful,
        "failed": failed,
        "files": file_results,
    }


def process_batch(
    files,
    server_url,
    batch_number,
    output_dir,
    progress_bar,
    poll_interval=DEFAULT_POLL_INTERVAL,
    timeout=DEFAULT_TIMEOUT,
    retries=DEFAULT_RETRIES,
    request_timeout=DEFAULT_REQUEST_TIMEOUT,
    submission_options=None,
):
    """Process a batch of PDFs and return `(batch_folder, batch_metadata)`."""
    batch_folder, papers_folder, metadata_folder = create_batch_folders(output_dir, batch_number)
    file_results = []

    progress_bar.set_description(f"Batch {batch_number:04d}")

    for pdf_path in files:
        pdf_path = Path(pdf_path)
        task_id = None
        try:
            task_id = submit_task(
                server_url,
                pdf_path,
                retries=retries,
                request_timeout=request_timeout,
                form_data=submission_options,
            )
            result = poll_task(
                server_url,
                task_id,
                poll_interval=poll_interval,
                timeout=timeout,
                retries=retries,
                request_timeout=request_timeout,
            )
            result_payload = fetch_task_result(
                server_url,
                task_id,
                retries=retries,
                request_timeout=request_timeout,
                result_url=result.get("result_url"),
            )
            output_path = _save_task_result(result_payload, pdf_path, papers_folder)
            file_results.append(
                _build_file_result(
                    input_path=pdf_path,
                    output_path=output_path,
                    status="Success",
                    task_id=task_id,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep processing remaining files
            file_results.append(
                _build_file_result(
                    input_path=pdf_path,
                    task_id=task_id,
                    status="Error",
                    error_message=str(exc),
                )
            )
        finally:
            progress_bar.update(1)

    batch_metadata = build_batch_metadata(batch_number, file_results)
    metadata_file = metadata_folder / f"batch_{batch_number:04d}_metadata.json"
    save_metadata(batch_metadata, metadata_file)
    return batch_folder, batch_metadata


def run_batches(
    input_dir,
    output_dir,
    server_url=DEFAULT_SERVER_URL,
    chunk_size=DEFAULT_CHUNK_SIZE,
    poll_interval=DEFAULT_POLL_INTERVAL,
    timeout=DEFAULT_TIMEOUT,
    metadata_path=None,
    retries=DEFAULT_RETRIES,
    request_timeout=DEFAULT_REQUEST_TIMEOUT,
    backend=None,
    parse_method=None,
    lang_list=None,
    formula_enable=None,
    table_enable=None,
):
    """Main batch processing loop over all PDFs in the input directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

    check_server_health(server_url, request_timeout=request_timeout, retries=retries)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        return []

    all_batch_metadata = []
    total_files = len(pdf_files)

    submission_options = {}
    if backend:
        submission_options["backend"] = backend
    if parse_method:
        submission_options["parse_method"] = parse_method
    if lang_list:
        submission_options["lang_list"] = lang_list
    if formula_enable is not None:
        submission_options["formula_enable"] = str(bool(formula_enable)).lower()
    if table_enable is not None:
        submission_options["table_enable"] = str(bool(table_enable)).lower()

    with tqdm(total=total_files, unit="file", desc="Batch 0001") as progress_bar:
        for offset in range(0, total_files, chunk_size):
            batch_number = (offset // chunk_size) + 1
            batch_files = pdf_files[offset : offset + chunk_size]
            print(
                f"Processing chunk {batch_number} "
                f"({len(batch_files)} files, {offset + 1}-{offset + len(batch_files)} of {total_files})"
            )

            _, batch_metadata = process_batch(
                files=batch_files,
                server_url=server_url,
                batch_number=batch_number,
                output_dir=output_dir,
                progress_bar=progress_bar,
                poll_interval=poll_interval,
                timeout=timeout,
                retries=retries,
                request_timeout=request_timeout,
                submission_options=submission_options or None,
            )
            all_batch_metadata.append(batch_metadata)

            if metadata_path:
                save_metadata(
                    {
                        "total_batches": len(all_batch_metadata),
                        "total_files": total_files,
                        "batches": all_batch_metadata,
                    },
                    metadata_path,
                )

    return all_batch_metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MinerU 3.0 batch client")
    parser.add_argument("--input-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output-dir", required=True, help="Directory for batch output folders")
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help="MinerU API/Router URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Files per batch (default: 50)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help="Polling interval seconds (default: 5)",
    )
    parser.add_argument(
        "--metadata-path",
        default=None,
        help="Optional path for aggregate metadata JSON",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Per-task timeout in seconds (default: 1800)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Retries for transient network errors (default: 3)",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Optional MinerU backend (e.g. pipeline, hybrid-auto-engine)",
    )
    parser.add_argument(
        "--parse-method",
        default=None,
        help="Optional parse method (auto, txt, ocr)",
    )
    parser.add_argument(
        "--lang-list",
        default=None,
        help="Optional OCR language list string (e.g. en or ch)",
    )
    parser.add_argument(
        "--formula-enable",
        choices=["true", "false"],
        default=None,
        help="Optional formula parsing toggle",
    )
    parser.add_argument(
        "--table-enable",
        choices=["true", "false"],
        default=None,
        help="Optional table parsing toggle",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than 0")
    if args.poll_interval <= 0:
        raise ValueError("--poll-interval must be greater than 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0")
    if args.retries < 0:
        raise ValueError("--retries must be >= 0")

    metadata = run_batches(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        server_url=args.server_url,
        chunk_size=args.chunk_size,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        metadata_path=args.metadata_path,
        retries=args.retries,
        backend=args.backend,
        parse_method=args.parse_method,
        lang_list=args.lang_list,
        formula_enable=(args.formula_enable == "true") if args.formula_enable is not None else None,
        table_enable=(args.table_enable == "true") if args.table_enable is not None else None,
    )
    print(f"Completed {len(metadata)} batch(es).")


if __name__ == "__main__":
    main()
