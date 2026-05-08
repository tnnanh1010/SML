"""
MinerU SDK-based extractor with batch processing.

Replaces the raw HTTP mineru_client.py with the official mineru.cli.api_client SDK.
Preserves the same batch folder output structure:

    output_dir/
      batch_0001/
        papers/
          DOCUMENT_STEM/
            DOCUMENT_STEM.md
            images/   (if any)
        metadata/
          batch_0001_metadata.json
      metadata/
        all_batches.json   (if metadata_path provided)
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import httpx

from mineru.cli import api_client as _api_client
from mineru.cli.common import image_suffixes, office_suffixes, pdf_suffixes
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path

SUPPORTED_SUFFIXES = set(pdf_suffixes + image_suffixes + office_suffixes)

DEFAULT_CHUNK_SIZE = 50
DEFAULT_BACKEND = "pipeline"
DEFAULT_PARSE_METHOD = "auto"
DEFAULT_LANGUAGE = "en"
DEFAULT_FORMULA_ENABLE = True
DEFAULT_TABLE_ENABLE = True


# ─── helpers ──────────────────────────────────────────────────────────────────

def _collect_input_files(input_dir: Path) -> list[Path]:
    """Return sorted list of supported files in input_dir."""
    files = sorted(
        f.resolve()
        for f in input_dir.iterdir()
        if f.is_file() and guess_suffix_by_path(f) in SUPPORTED_SUFFIXES
    )
    if not files:
        raise ValueError(f"No supported files found in: {input_dir}")
    return files


def _create_batch_folders(output_dir: Path, batch_number: int):
    """Create batch_XXXX/papers/ and batch_XXXX/metadata/ folders."""
    batch_folder = output_dir / f"batch_{batch_number:04d}"
    papers_folder = batch_folder / "papers"
    metadata_folder = batch_folder / "metadata"
    papers_folder.mkdir(parents=True, exist_ok=True)
    metadata_folder.mkdir(parents=True, exist_ok=True)
    return batch_folder, papers_folder, metadata_folder


def _save_metadata(metadata: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def _build_file_result(input_path, output_path=None, status="Error", error_message=None):
    return {
        "input_path": str(input_path),
        "output_path": str(output_path) if output_path else None,
        "status": status,
        "error_message": error_message,
    }


def _build_batch_metadata(batch_number: int, file_results: list) -> dict:
    successful = sum(1 for r in file_results if r["status"] == "Success")
    return {
        "batch_number": batch_number,
        "total_files": len(file_results),
        "successful": successful,
        "failed": len(file_results) - successful,
        "files": file_results,
    }


def _prepare_local_api_temp_dir() -> None:
    """Fix temp dir for WSL environments where vLLM IPC sockets fail on drvfs."""
    current_temp_dir = Path(tempfile.gettempdir())
    if os.name == "nt" or not Path("/tmp").exists():
        return
    if not str(current_temp_dir).startswith("/mnt/"):
        return
    os.environ["TMPDIR"] = "/tmp"
    tempfile.tempdir = None


# ─── core async batch processor ───────────────────────────────────────────────

async def _process_batch_sdk(
    files: list[Path],
    batch_number: int,
    output_dir: Path,
    http_client: httpx.AsyncClient,
    base_url: str,
    backend: str,
    parse_method: str,
    language: str,
    formula_enable: bool,
    table_enable: bool,
) -> dict:
    """
    Submit all files in a batch as a single task, download the result zip,
    extract per-paper folders, and return batch metadata.
    """
    _, papers_folder, metadata_folder = _create_batch_folders(output_dir, batch_number)
    file_results = []

    print(f"\n=== Batch {batch_number:04d}: {len(files)} file(s) ===")

    # Build form data using the SDK helper
    form_data = _api_client.build_parse_request_form_data(
        lang_list=[language],
        backend=backend,
        parse_method=parse_method,
        formula_enable=formula_enable,
        table_enable=table_enable,
        server_url=None,
        start_page_id=0,
        end_page_id=None,
        return_md=True,
        return_middle_json=False,
        return_model_output=False,
        return_content_list=False,
        return_images=True,
        response_format_zip=True,
        return_original_file=False,
    )

    upload_assets = [
        _api_client.UploadAsset(path=f, upload_name=f.name)
        for f in files
    ]

    result_zip_path: Path | None = None
    try:
        print(f"  Submitting {len(upload_assets)} file(s)...")
        submit_response = await _api_client.submit_parse_task(
            base_url=base_url,
            upload_assets=upload_assets,
            form_data=form_data,
        )
        print(f"  task_id: {submit_response.task_id}")

        last_status = None

        def on_status(snapshot: _api_client.TaskStatusSnapshot) -> None:
            nonlocal last_status
            msg = (
                snapshot.status
                if snapshot.queued_ahead is None
                else f"{snapshot.status} (queued_ahead={snapshot.queued_ahead})"
            )
            if msg != last_status:
                last_status = msg
                print(f"  status: {msg}")

        await _api_client.wait_for_task_result(
            client=http_client,
            submit_response=submit_response,
            task_label=f"batch {batch_number:04d}",
            status_snapshot_callback=on_status,
        )
        print("  status: completed")

        result_zip_path = await _api_client.download_result_zip(
            client=http_client,
            submit_response=submit_response,
            task_label=f"batch {batch_number:04d}",
        )

        # Extract zip into a temp dir, then reorganise into papers/STEM/ layout
        import tempfile as _tempfile
        with _tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _api_client.safe_extract_zip(result_zip_path, tmp_path)

            for pdf_path in files:
                stem = pdf_path.stem
                paper_dir = papers_folder / stem
                paper_dir.mkdir(parents=True, exist_ok=True)

                # The SDK extracts files flat or in sub-dirs named by stem
                # Try both layouts: flat (stem.md) and nested (stem/stem.md)
                md_candidates = list(tmp_path.rglob(f"{stem}.md"))
                img_candidates = list(tmp_path.rglob("images"))

                if md_candidates:
                    md_src = md_candidates[0]
                    md_dst = paper_dir / f"{stem}.md"
                    md_dst.write_bytes(md_src.read_bytes())

                    # Copy images folder if present alongside the md file
                    images_src = md_src.parent / "images"
                    if images_src.exists():
                        import shutil
                        shutil.copytree(images_src, paper_dir / "images", dirs_exist_ok=True)

                    file_results.append(_build_file_result(
                        input_path=pdf_path,
                        output_path=paper_dir,
                        status="Success",
                    ))
                    print(f"  → {stem}.md → {paper_dir.relative_to(output_dir)}")
                else:
                    file_results.append(_build_file_result(
                        input_path=pdf_path,
                        status="Error",
                        error_message=f"No markdown output found for {stem}",
                    ))
                    print(f"  ✗ No output for {stem}")

    except Exception as exc:
        # Mark all files in this batch as failed
        for pdf_path in files:
            if not any(r["input_path"] == str(pdf_path) for r in file_results):
                file_results.append(_build_file_result(
                    input_path=pdf_path,
                    status="Error",
                    error_message=str(exc),
                ))
        print(f"  ✗ Batch {batch_number:04d} error: {exc}")
    finally:
        if result_zip_path and result_zip_path.exists():
            result_zip_path.unlink(missing_ok=True)

    batch_meta = _build_batch_metadata(batch_number, file_results)
    _save_metadata(batch_meta, metadata_folder / f"batch_{batch_number:04d}_metadata.json")
    return batch_meta


# ─── public API ───────────────────────────────────────────────────────────────

async def run_batches_async(
    input_dir: str | Path,
    output_dir: str | Path,
    api_url: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    backend: str = DEFAULT_BACKEND,
    parse_method: str = DEFAULT_PARSE_METHOD,
    language: str = DEFAULT_LANGUAGE,
    formula_enable: bool = DEFAULT_FORMULA_ENABLE,
    table_enable: bool = DEFAULT_TABLE_ENABLE,
    metadata_path: str | Path | None = None,
) -> list[dict]:
    """
    Process all supported files in input_dir through MinerU in batches.

    Args:
        input_dir:      Directory containing PDF/image/office files.
        output_dir:     Root output directory; batch folders are created here.
        api_url:        MinerU FastAPI base URL.  Pass None to auto-start a
                        local mineru-api process (requires mineru installed).
        chunk_size:     Files per batch submission (default 50).
        backend:        MinerU backend, e.g. "pipeline", "hybrid-auto-engine".
        parse_method:   "auto" | "txt" | "ocr".
        language:       OCR language hint, e.g. "en", "ch".
        formula_enable: Parse formulas.
        table_enable:   Parse tables.
        metadata_path:  Optional path to write aggregate metadata JSON.

    Returns:
        List of per-batch metadata dicts.
    """
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = _collect_input_files(input_dir)
    total = len(all_files)
    print(f"Found {total} file(s) in {input_dir}")

    all_batch_metadata: list[dict] = []
    local_server: _api_client.LocalAPIServer | None = None

    async with httpx.AsyncClient(
        timeout=_api_client.build_http_timeout(),
        follow_redirects=True,
    ) as http_client:
        try:
            # ── server setup ──────────────────────────────────────────────────
            if api_url is None:
                _prepare_local_api_temp_dir()
                local_server = _api_client.LocalAPIServer()
                base_url = local_server.start()
                print(f"Started local mineru-api: {base_url}")
                server_health = await _api_client.wait_for_local_api_ready(
                    http_client, local_server
                )
            else:
                server_health = await _api_client.fetch_server_health(
                    http_client,
                    _api_client.normalize_base_url(api_url),
                )
            base_url = server_health.base_url
            print(f"Using API: {base_url}")

            # ── batch loop ────────────────────────────────────────────────────
            for offset in range(0, total, chunk_size):
                batch_number = (offset // chunk_size) + 1
                batch_files = all_files[offset: offset + chunk_size]

                batch_meta = await _process_batch_sdk(
                    files=batch_files,
                    batch_number=batch_number,
                    output_dir=output_dir,
                    http_client=http_client,
                    base_url=base_url,
                    backend=backend,
                    parse_method=parse_method,
                    language=language,
                    formula_enable=formula_enable,
                    table_enable=table_enable,
                )
                all_batch_metadata.append(batch_meta)

                if metadata_path:
                    _save_metadata(
                        {
                            "total_batches": len(all_batch_metadata),
                            "total_files": total,
                            "batches": all_batch_metadata,
                        },
                        Path(metadata_path),
                    )

        finally:
            if local_server is not None:
                local_server.stop()
                print("Stopped local mineru-api.")

    successful = sum(b["successful"] for b in all_batch_metadata)
    failed = sum(b["failed"] for b in all_batch_metadata)
    print(f"\nMinerU extraction complete: {successful} succeeded, {failed} failed "
          f"across {len(all_batch_metadata)} batch(es).")
    return all_batch_metadata


def run_batches(
    input_dir: str | Path,
    output_dir: str | Path,
    api_url: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    backend: str = DEFAULT_BACKEND,
    parse_method: str = DEFAULT_PARSE_METHOD,
    language: str = DEFAULT_LANGUAGE,
    formula_enable: bool = DEFAULT_FORMULA_ENABLE,
    table_enable: bool = DEFAULT_TABLE_ENABLE,
    metadata_path: str | Path | None = None,
) -> list[dict]:
    """Synchronous wrapper around run_batches_async."""
    return asyncio.run(run_batches_async(
        input_dir=input_dir,
        output_dir=output_dir,
        api_url=api_url,
        chunk_size=chunk_size,
        backend=backend,
        parse_method=parse_method,
        language=language,
        formula_enable=formula_enable,
        table_enable=table_enable,
        metadata_path=metadata_path,
    ))


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MinerU SDK batch extractor")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--api-url", default=None,
                        help="MinerU API URL. Omit to auto-start local server.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--parse-method", default=DEFAULT_PARSE_METHOD)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--formula-enable", action="store_true", default=DEFAULT_FORMULA_ENABLE)
    parser.add_argument("--no-formula", dest="formula_enable", action="store_false")
    parser.add_argument("--table-enable", action="store_true", default=DEFAULT_TABLE_ENABLE)
    parser.add_argument("--no-table", dest="table_enable", action="store_false")
    parser.add_argument("--metadata-path", default=None)
    args = parser.parse_args()

    run_batches(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        api_url=args.api_url,
        chunk_size=args.chunk_size,
        backend=args.backend,
        parse_method=args.parse_method,
        language=args.language,
        formula_enable=args.formula_enable,
        table_enable=args.table_enable,
        metadata_path=args.metadata_path,
    )
