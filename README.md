# SML — Small Language Model Fine-tuning Pipeline

An end-to-end pipeline for converting internal PDF documents into domain-specific QA training data, fine-tuning a small language model via LoRA, and serving the result for interactive comparison.

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                            pipeline_config.yml                                   ║
╚══════════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: QA Generation                              run_QAgen_pipeline.py        │
│                                                                                  │
│  Input: Customer Corpus (PDF documents)                                          │
│                                                                                  │
│  ┌─────────┐    ┌──────────┐    ┌───────────┐    ┌──────────────┐                | 
│  │ Text    │───▶│ Chunk    │───▶│ Filtering │───▶│ QA Generation│                │
│  │ Extract │    │ Splitting│    │ Quality   │    │              │                │
│  │         │    │          │    │ Gate      │    │              │                │
│  └─────────┘    └──────────┘    └───────────┘    └──────────────┘                │
│                                                                                  │
│  Output: Domain-specific QA Dataset                                              │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: SLM Training                              run_training_pipeline.py      │
│                                                                                  │
│  Input: QA Dataset                                                               │
│                                                                                  │
│  ┌───────────────┐    ┌──────────────┐    ┌──────────────┐                       │
│  │ Dataset Format│───▶│ LoRA SFT     │───▶│ Adapter Merge│                       │
│  │ Conversion    │    │ Fine-tuning  │    │              │                       │
│  │               │    │              │    │              │                       │
│  └───────────────┘    └──────────────┘    └──────────────┘                       │
│                                                                                  │
│  Output: Fine-tuned SLM                                                          │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: SLM Packaging & Serving                             serve_slm.py        │
│                                                                                  │
│  Input: Base Model + Fine-tuned SLM + Test Questions                             │
│                                                                                  │
│  ┌───────────┐    ┌─────────────────────────────────────────┐                    │
│  │  Model    │───▶│         Gradio Comparison UI            │                    │
│  │  Packaging│    │                                         │                    │
│  │           │    │  ┌──────────┐       ┌─────────────┐     │                    │
│  └───────────┘    │  │Base Model│       │Fine-tuned   │     │                    │
│                   │  │          │       │SLM          │     │                    │
│                   │  └──────────┘       └─────────────┘     │                    │
│                   │       ▲                    ▲             │                    │
│                   │       └────────┬───────────┘             │                    │
│                   │           User Question                  │                    │
│                   └─────────────────────────────────────────┘                    │
│                                                                                  │
│  Output: Side-by-side Response Comparison + Ground Truth                         │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Usage

All three stages are configured in a single file: `pipeline_config.yml`. Edit the config to match your environment (paths, models, API keys, hyperparameters), then run the corresponding pipeline script.

```bash
# Stage 1: Generate QA dataset from customer documents
python run_QAgen_pipeline.py

# Stage 2: Fine-tune the SLM
python run_training_pipeline.py

# Stage 3: Serve and compare models
python serve_slm.py
```

Each runner supports `--only`, `--skip`, and `--from` flags to run or resume specific steps:

```bash
python run_QAgen_pipeline.py --from filtering
python run_training_pipeline.py --only dataset_prep
```

---

## Prerequisites

```bash
# Core dependencies
pip install pyyaml openai tqdm httpx gradio torch transformers

# Stage 1: PDF extraction
pip install -U "mineru[all]"

# Stage 2: Training
pip install ms-swift

# Stage 3: Model packaging & serving
curl -fsSL https://ollama.com/install.sh | sh   # Linux
git clone https://github.com/ggerganov/llama.cpp ./llama.cpp
pip install -r llama.cpp/requirements.txt

pip install --upgrade transformers 

```


git clone https://github.com/ggerganov/llama.cpp ./llama.cpp
pip install -r llama.cpp/requirements.txt
```

```bash
export OPENROUTER_API_KEY=your_key_here
```

---

## Demo



https://github.com/user-attachments/assets/58348803-56c1-473c-bcc1-374bc2382aa5


---

## Project Structure

```
SML/
├── pipeline_config.yml                 # Single config for all 3 stages
├── run_QAgen_pipeline.py              # Stage 1 runner
├── run_training_pipeline.py           # Stage 2 runner
├── serve_slm.py                       # Stage 3 runner (Gradio UI)
├── input_pdfs/                        # Source PDF documents
├── output_batches/                    # Intermediate outputs (MD, chunks, QA)
├── src/
│   ├── QAgenerator/                   # Stage 1: QA Generation
│   │   ├── mineru/                    # MinerU SDK extractor
│   │   ├── parse.py                   # Markdown chunking
│   │   ├── chunk_filter_openrouter.py  # LLM quality filter
│   │   └── qa_generator.py            # QA generation
│   ├── SMLtrainer/                    # Stage 2: Training
│   │   ├── prepare_dataset_swift.py   # QA → ms-swift format
│   │   ├── lora_sft.sh                # LoRA SFT script
│   │   ├── merge.sh                   # Adapter merge script
│   │   └── sml_training_kaggle.ipynb  # Kaggle training notebook
│   └── SLMserve/                      # Stage 3: Serving & Comparison
│       ├── slm_serving_kaggle.ipynb   # Kaggle serving notebook
```
