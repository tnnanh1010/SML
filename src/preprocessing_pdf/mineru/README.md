# MinerU 3.0 Migration

This folder contains the migrated MinerU 3.0 client and startup scripts using official `mineru-api` and `mineru-router` services.

## Start Single GPU API

```bash
cd SML/src/preprocessing_pdf/mineru
chmod +x start_api.sh
./start_api.sh
```

Default endpoint: `http://0.0.0.0:8000`

Override host/port:

```bash
HOST=127.0.0.1 PORT=8000 ./start_api.sh
```

## Start Multi-GPU Router

```bash
cd SML/src/preprocessing_pdf/mineru
chmod +x start_router.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 ./start_router.sh
```

Default endpoint: `http://0.0.0.0:8002`

Override host/port:

```bash
HOST=0.0.0.0 PORT=8002 CUDA_VISIBLE_DEVICES=0,1 ./start_router.sh
```

## Run Batch Client

```bash
cd SML/src/preprocessing_pdf/mineru
python mineru_client.py \
  --input-dir /path/to/pdfs \
  --output-dir /path/to/output \
  --server-url http://127.0.0.1:8000 \
  --chunk-size 50 \
  --backend pipeline \
  --metadata-path /path/to/output/metadata/all_batches.json
```

## Output Layout

Each batch creates:

```
batch_0001/
  papers/
    DOCUMENT_STEM/
      result.json          # Full API response with all fields
      DOCUMENT_STEM.md     # Extracted markdown (if available)
  metadata/
    batch_0001_metadata.json  # Batch processing summary
```

Aggregate metadata is also saved if `--metadata-path` is provided:

```json
{
  "total_batches": 1,
  "total_files": 3,
  "batches": [/* per-batch metadata */]
}
```

## Recommended Setup

- Use `mineru-api` (`8000`) for single GPU.
- Use `mineru-router` (`8002`) for multi-GPU processing.
- Keep `--chunk-size` between `20` and `100` depending on system throughput and queue depth.

## Metadata Output

Batch metadata shows processing status for each file:

```json
{
  "batch_number": 1,
  "total_files": 3,
  "successful": 3,
  "failed": 0,
  "files": [
    {
      "input_path": "/path/to/doc.pdf",
      "output_path": "batch_0001/papers/doc",
      "status": "Success",
      "error_message": null,
      "task_id": "abc-123-def"
    }
  ]
}
```

