# 🤖 Transformer 101: The Annotated Transformer Explained

An interactive, visual, and code-level deep dive into the Transformer architecture based on the seminal paper [*Attention Is All You Need* (Vaswani et al.)](https://arxiv.org/abs/1706.03762) and Harvard NLP's *The Annotated Transformer*.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-blue?style=for-the-badge&logo=github)](https://murtuzasaifee.github.io/transformer_101/annotated_transformer_blueprint.html)

> 🌐 **Interactive Web Guide:** Explore the visual companion directly in your browser:  
> **[👉 View The Annotated Transformer Blueprint](https://murtuzasaifee.github.io/transformer_101/annotated_transformer_blueprint.html)**

---

## 🌟 Overview

This repository provides a comprehensive, beginner-to-advanced guide to understanding, visualizing, and implementing the Transformer model from scratch using PyTorch.

It breaks down every component—from input embeddings and positional encodings to multi-head attention, encoder-decoder stacks, layer normalization, residual connections, and training loops—into clear mathematical intuitions, tensor shape walkthroughs, and runnable code. Beyond the notebook, the same architecture is implemented as a **modular, production-style Python package** (`src/annotated_transformer/`) that trains on the Multi30k (DE→EN) dataset and can run on a GPU server, with structured logs for dashboarding.

---

## 📂 Repository Structure

```text
transformer_101/
├── src/annotated_transformer/            # Modular Transformer implementation (importable package)
│   ├── model/                     # Architecture, one file per component (swappable)
│   │   ├── attention.py           #   scaled dot-product + multi-head attention
│   │   ├── embeddings.py          #   token embeddings + sinusoidal positional encoding
│   │   ├── encoder.py             #   Encoder / EncoderLayer
│   │   ├── decoder.py             #   Decoder / DecoderLayer
│   │   ├── feedforward.py         #   position-wise feed-forward network
│   │   ├── generator.py           #   final linear + log-softmax projection
│   │   ├── sublayer.py            #   LayerNorm, residual sublayer wrapping, clones()
│   │   ├── mask.py                #   causal (subsequent) mask
│   │   └── transformer.py         #   EncoderDecoder + build_transformer_model() factory
│   ├── data/                      # Tokenization, vocabulary, Multi30k dataset pipeline
│   │   ├── tokenizer.py           #   spaCy DE/EN tokenizers
│   │   ├── vocab.py               #   Vocab build/load/cache
│   │   ├── dataset.py             #   Multi30kDataset, collate_batch, DataLoaders
│   │   └── batch.py               #   Batch (src/tgt tensors + masks)
│   ├── training/                  # Training loop, loss, LR schedule, metrics logging
│   │   ├── loss.py                #   LabelSmoothing, SimpleLossCompute
│   │   ├── lr_schedule.py         #   Noam (warmup + inverse-sqrt decay) schedule
│   │   ├── trainer.py             #   run_epoch, train_worker, DDP multi-GPU training
│   │   └── logger.py              #   structured JSONL metrics logger (for dashboards)
│   ├── inference/                 # Greedy decoding + qualitative evaluation
│   │   ├── decode.py              #   greedy_decode
│   │   └── evaluate.py            #   check_outputs (source/target/prediction printout)
│   ├── utils/
│   │   └── logging_setup.py       #   console logging configuration
│   └── config.py                  # Pydantic-typed model/data/training config, loaded from YAML
├── configs/
│   └── multi30k.yaml              # Default experiment config (architecture + training hyperparams)
├── scripts/                       # CLI entrypoints, run these directly
│   ├── sanity_check.py            #   fast CPU checks: imports, forward pass, train loop, logging
│   ├── build_vocab.py             #   build/cache the DE/EN vocabulary
│   ├── train.py                   #   train on Multi30k (single-GPU or DDP)
│   ├── translate.py               #   load a checkpoint, print greedy-decoded translations
│   └── run_training.sh            #   full server pipeline: sync -> spacy -> sanity -> vocab -> train
├── logs/                          # JSONL training logs written here at runtime (gitignored)
├── checkpoints/                   # Model checkpoints + cached vocab written here (gitignored)
├── notebooks/
│   ├── AnnotatedTransformer.ipynb # Step-by-step PyTorch implementation with extended explanations
│   └── BPE_Tokenization.ipynb     # BPE tokenization walkthrough
├── docs/
│   └── annotated_transformer_blueprint.html  # Interactive visual blueprint & guide
├── pyproject.toml                 # Package + dependency definitions (uv-managed)
├── LICENSE                        # License details
└── README.md                      # Project documentation
```

---

## 🚀 Key Highlights

### 1. 🎨 Interactive Visual Blueprint ([Live Demo](https://murtuzasaifee.github.io/transformer_101/annotated_transformer_blueprint.html))
An interactive HTML companion that provides:
- **Architecture Breakdown**: Side-by-side explanations of the Encoder and Decoder stacks.
- **Multi-Head Attention Demystified**: Visualizing Query ($Q$), Key ($K$), and Value ($V$) projections, scaled dot-product attention, and causal masking.
- **Tensor Shape Flow**: Dimension tracking through every transformation layer (`[batch_size, seq_len, d_model]`).
- **Chapter Navigation**: Tabbed interface covering architecture intuition, sub-layers, training dynamics, and inference.

### 2. 📓 Annotated PyTorch Notebooks (`notebooks/`)
- Complete PyTorch implementation without external black-box libraries.
- Modular code structure: `Encoder`, `Decoder`, `MultiHeadedAttention`, `PositionalEncoding`, `Generator`, and `LayerNorm`.
- Synthetic task demonstrations (e.g., sequence copying task) illustrating loss curves, greedy decoding, and beam search intuition.

### 3. 🧩 Modular Training Package (`src/annotated_transformer/`)
- Same architecture as the notebook, split into independently swappable modules (attention, positional encoding, encoder/decoder stacks, etc.) instead of one monolithic script.
- Trains end-to-end on the **Multi30k** German→English dataset, single-GPU or multi-GPU (DistributedDataParallel).
- Typed, YAML-driven configuration (Pydantic) for architecture + training hyperparameters — no hardcoded magic numbers.
- Structured **JSONL training logs** (`logs/*.jsonl`) — one record per training step / epoch summary / checkpoint event, ready to load into a dashboard with `pandas.read_json(path, lines=True)`.
- A `sanity_check.py` script to verify the whole package end-to-end before burning GPU hours, and a `run_training.sh` one-shot pipeline for a fresh server.
- CLI scripts for the full workflow: sanity check → build vocab → train → translate.

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- A CUDA-capable GPU recommended for training (CPU works for the notebook, sanity check, and small smoke tests)

### Option A: One-shot server pipeline

On a fresh GPU server, this runs every step below (install deps, download tokenizer models, sanity-check the package, build the vocab, train) in one go and mirrors all output to `logs/run_training_<timestamp>.log`:

```bash
chmod +x scripts/run_training.sh   # first time only
./scripts/run_training.sh
```

Configurable via environment variables:

```bash
CONFIG=configs/multi30k.yaml ./scripts/run_training.sh   # use a different config
DISTRIBUTED=1 ./scripts/run_training.sh                   # train across all visible GPUs (DDP)
FORCE_REBUILD_VOCAB=1 ./scripts/run_training.sh           # rebuild the vocab cache
SKIP_SANITY_CHECK=1 ./scripts/run_training.sh             # skip the sanity-check step
```

### Option B: Manual step-by-step

### 1. Install dependencies

```bash
uv sync
```

This installs PyTorch, spaCy, HuggingFace `datasets`, Pydantic, and the other dependencies declared in `pyproject.toml`, and installs `annotated_transformer` itself in editable mode.

### 2. Download the spaCy tokenizer models

```bash
uv run python -m spacy download de_core_news_sm
uv run python -m spacy download en_core_web_sm
```

### 3. Sanity-check the package

Fast, CPU-only checks (imports, model forward pass, a few synthetic training steps, greedy decoding, JSONL metrics logging, config validation) — no dataset download required. Run this before any real training job, especially after pulling changes or setting up a new server:

```bash
uv run scripts/sanity_check.py
```

Prints `[PASS]`/`[FAIL]` per check and exits non-zero if anything is broken.

### 4. Build the vocabulary

Builds and caches the DE/EN vocabulary from the Multi30k corpus (only needs to run once; re-run with `--force` to rebuild).

```bash
uv run scripts/build_vocab.py --config configs/multi30k.yaml
```

### 5. Train the model

Runs on GPU if available, otherwise CPU. Edit `configs/multi30k.yaml` to change architecture (`d_model`, `n_layers`, `n_heads`, ...) or training hyperparameters (epochs, batch size, warmup, ...).

```bash
# single GPU / CPU
uv run scripts/train.py --config configs/multi30k.yaml

# all visible GPUs, via DistributedDataParallel
uv run scripts/train.py --config configs/multi30k.yaml --distributed
```

Checkpoints are written to `checkpoints/multi30k_model_<epoch>.pt` and `checkpoints/multi30k_model_final.pt`. Training progress streams to the console and is also written as structured JSONL to `logs/<experiment_name>_gpu<n>.jsonl` for dashboarding (`train_step`, `epoch_summary`, and `checkpoint` events, each timestamped).

### 6. Translate / inspect model outputs

Greedy-decodes a handful of validation examples and prints source, reference, and model translation.

```bash
uv run scripts/translate.py --config configs/multi30k.yaml \
    --checkpoint checkpoints/multi30k_model_final.pt --n-examples 10
```

### Running the Notebooks
Launch JupyterLab or Jupyter Notebook:

```bash
jupyter notebook notebooks/AnnotatedTransformer.ipynb
```

### Viewing the Interactive Blueprint

- **Online (GitHub Pages)**: [murtuzasaifee.github.io/transformer_101/annotated_transformer_blueprint.html](https://murtuzasaifee.github.io/transformer_101/annotated_transformer_blueprint.html)
- **Locally**: Open `docs/annotated_transformer_blueprint.html` in any modern web browser:

```bash
# macOS
open docs/annotated_transformer_blueprint.html

# Linux
xdg-open docs/annotated_transformer_blueprint.html
```

---

## 🧠 Transformer Architecture at a Glance

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| Component | Purpose | Key Concept |
| :--- | :--- | :--- |
| **Positional Encoding** | Injects order information | Sine and Cosine functions across frequencies |
| **Scaled Dot-Product Attention** | Calculates token-to-token relevance | Scaled by $\frac{1}{\sqrt{d_k}}$ to prevent gradient vanishing |
| **Multi-Head Attention** | Attends to information across subspaces | Linear projections split into $h$ heads |
| **Feed-Forward Network** | Position-wise non-linear mapping | Two linear layers with ReLU / GELU |
| **LayerNorm & Residuals** | Stabilizes training in deep networks | $\text{LayerNorm}(x + \text{SubLayer}(x))$ |

---

## ⚙️ Configuration Reference (`configs/multi30k.yaml`)

| Section | Key | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `model` | `n_layers` | `6` | Encoder/decoder layer count (N) |
| `model` | `d_model` | `512` | Embedding / hidden dimension |
| `model` | `d_ff` | `2048` | Feed-forward inner dimension |
| `model` | `n_heads` | `8` | Attention heads |
| `model` | `dropout` | `0.1` | Dropout probability |
| `data` | `max_padding` | `72` | Fixed sequence length after padding |
| `data` | `vocab_cache_path` | `checkpoints/vocab.pt` | Cached vocabulary location |
| `training` | `num_epochs` | `8` | Training epochs |
| `training` | `batch_size` | `32` | Batch size (split across GPUs if distributed) |
| `training` | `accum_iter` | `10` | Gradient accumulation steps |
| `training` | `base_lr` / `warmup` | `1.0` / `3000` | Noam learning-rate schedule params |
| `training` | `label_smoothing` | `0.1` | Label smoothing factor |
| `training` | `distributed` | `false` | Train with DDP across all visible GPUs |
| `training` | `checkpoint_dir` / `log_dir` | `checkpoints` / `logs` | Output locations |

Point any script at a different config with `--config path/to/your.yaml`, or omit `--config` to use built-in defaults.

---

## 📜 References & Acknowledgements

- **Paper**: [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- **Harvard NLP**: [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- Built with ❤️ for machine learning enthusiasts and researchers.