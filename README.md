# LLM — GPT-Style Transformer in C#

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Academic project.** This codebase exists to make the internals of a large language model fully visible and readable. Every mathematical operation — matrix multiplication, multi-head attention, layer normalisation, backpropagation, and the Adam optimiser — is implemented from scratch in plain C# with no external libraries for any numerical computation. If you want to understand exactly how an LLM works, you can read the source directly without fighting through a framework.
>
> The one exception is the GPU backend, which uses [ILGPU](https://ilgpu.net) to write and compile GPU kernels. GPU programming is not the subject of this project, so a library is used there to avoid that complexity obscuring the model code that is the actual focus.
>
> This project is not intended for production use. For training models at scale use [PyTorch](https://pytorch.org), [JAX](https://jax.readthedocs.io), or similar frameworks.

A GPT-style decoder-only transformer language model implemented in C#. The project supports CPU and GPU backends, five tokenizer algorithms, three training modes with early stopping, and crash-recovery checkpointing that resumes mid-epoch without replaying completed gradient updates.

---

## Table of Contents

1. [Solution Structure](#solution-structure)
2. [Getting Started](#getting-started)
3. [Configuration](#configuration)
   - [AppConfig](#appconfig)
   - [TransformerConfig](#transformerconfig)
4. [CLI Reference](#cli-reference)
5. [Training](#training)
   - [Training Modes](#training-modes)
   - [Validation Splits](#validation-splits)
   - [Learning Rate Schedule](#learning-rate-schedule)
   - [Checkpointing](#checkpointing)
6. [Inference](#inference)
7. [Model Architecture](#model-architecture)
8. [Tokenizers](#tokenizers)
9. [File Formats](#file-formats)
   - [Weights File](#weights-file)
   - [Checkpoint File](#checkpoint-file)
   - [Vocabulary File](#vocabulary-file)
10. [Project Internals](#project-internals)
    - [LLM\_Common](#llm_common)
    - [LLM\_CPU](#llm_cpu)
    - [LLM\_GPU](#llm_gpu)
    - [LLM\_App](#llm_app)
    - [LLM\_Documentation](#llm_documentation)
11. [Dependencies](#dependencies)

---

## Solution Structure

```
LLM/
├── LLM_Common/          # Shared interfaces, config, serialization, tokenizers
├── LLM_CPU/             # Pure managed C# transformer backend
├── LLM_GPU/             # ILGPU-accelerated transformer backend
├── LLM_App/             # CLI entry point (training + inference)
├── LLM_Documentation/   # Design docs, architecture diagrams, sample corpora
└── LLM.slnx             # Solution file
```

---

## Getting Started

### Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download)
- A CUDA-capable GPU (optional — the CPU backend works without one)

### Build

```bash
dotnet build
```

### Train

```bash
cd LLM_App
dotnet run -- --backend GPU --train-file ../LLM_Documentation/sample_corpus_large.txt --save-file weights.bin
```

Or edit `LLM_App/appsettings.json` and run with no arguments:

```bash
dotnet run
```

### Generate text

```bash
dotnet run -- --action Prompt --load-file weights.bin
```

### Help

```bash
dotnet run -- --help
```

---

## Configuration

All settings live in `LLM_App/appsettings.json` and can be overridden individually at the command line with `--flag value`. CLI values always win over the file.

### AppConfig

Runtime and execution settings.

| Property | CLI flag | Type | Default | Valid values | Description |
|----------|----------|------|---------|--------------|-------------|
| `Action` | `--action` | string | `"Train"` | `Train`, `Prompt` | What to do when the app starts |
| `Backend` | `--backend` | string | `"GPU"` | `CPU`, `GPU` | Compute backend (GPU requires CUDA via ILGPU) |
| `TrainFile` | `--train-file` | path | `""` | — | Corpus file for training (required when `Action=Train`) |
| `SaveFile` | `--save-file` | path | `""` | — | Where to write weights + vocab after training (required when `Action=Train`) |
| `LoadFile` | `--load-file` | path | `""` | — | Weights or checkpoint file to load before training or prompting |
| `ErrorFile` | `--error-file` | path | `"error.txt"` | — | Redirect stderr here; empty = stderr stays on the console |
| `VocabSize` | `--vocab-size` | int | `4000` | > 0 | Target vocabulary size for the tokenizer (2 000–8 000 recommended) |
| `ValidationSplit` | `--validation-split` | string | `"Tail"` | `None`, `Tail`, `Random` | How to create the held-out validation set |
| `ValidationFraction` | `--validation-fraction` | double | `0.1` | (0, 1) | Fraction of tokens held out for validation |
| `TrainingMode` | `--training-mode` | string | `"EarlyStopping"` | `Epochs`, `Patience`, `EarlyStopping` | When to stop training (see [Training Modes](#training-modes)) |
| `Patience` | `--patience` | int | `5` | > 0 | Consecutive epochs without improvement before stopping |
| `MinDeltaLoss` | `--min-delta-loss` | double | `0.001` | ≥ 0 | Minimum val\_loss improvement that resets the patience counter |
| `CheckpointEveryMinutes` | `--checkpoint-every` | double | `60` | ≥ 0 | Save a crash-recovery checkpoint every N wall-clock minutes; `0` = epoch end only |
| `MaxTokens` | `--max-tokens` | int | `200` | > 0 | Maximum tokens to generate per prompt response |
| `Temperature` | `--temperature` | float | `0.8` | > 0 | Sampling temperature (< 1.0 = sharper, > 1.0 = more random) |
| `TopK` | `--top-k` | int | `15` | ≥ 0 | Top-K sampling filter; `0` = sample from the full vocabulary |
| `ContextCompaction` | `--context-compaction` | string | `"FIFO"` | `FIFO`, `SlidingWindow` | Context management when the window is full |
| `AnchorFraction` | `--anchor-fraction` | float | `0.2` | (0, 1) | Fraction of the context anchored at the start in `SlidingWindow` mode |

### TransformerConfig

Model architecture and optimiser hyper-parameters.

#### Architecture

| Property | CLI flag | Type | Default | Description |
|----------|----------|------|---------|-------------|
| `EmbeddingDim` | `--embedding-dim` | int | `768` | Residual stream width (d\_model). Must be divisible by `NumHeads`. Typical: 64, 128, 256, 768 |
| `NumHeads` | `--num-heads` | int | `16` | Parallel attention heads. `HeadDim = EmbeddingDim / NumHeads` |
| `NumLayers` | `--num-layers` | int | `8` | Stacked transformer blocks (depth). Typical: 2, 4, 6, 12 |
| `FFNDim` | `--ffn-dim` | int | `3072` | Feed-forward hidden dimension. Convention: 4 × EmbeddingDim |
| `ContextLength` | `--context-length` | int | `128` | Maximum sequence length (context window). Typical: 128, 256, 512, 1024, 2048 |
| `UseRoPE` | `--use-rope` | bool | `true` | Use Rotary Positional Encoding; `false` = sinusoidal additive PE |
| `Seed` | `--seed` | int | `42` | RNG seed for weight initialisation; `-1` = random each run |

#### Training

| Property | CLI flag | Type | Default | Description |
|----------|----------|------|---------|-------------|
| `Epochs` | `--epochs` | int | `25` | Maximum training epochs |
| `AccumulationSteps` | `--accumulation-steps` | int | `1` | Gradient accumulation steps. Effective batch = N × ContextLength tokens |
| `SampleEvery` | `--sample-every` | int | `5` | Print a generated sample every N epochs; `0` = disabled |
| `SamplePrompt` | `--sample-prompt` | string | `"Shall "` | Seed prompt used for training-time samples |

#### Adam Optimiser

| Property | CLI flag | Type | Default | Description |
|----------|----------|------|---------|-------------|
| `LearningRate` | `--learning-rate` | float | `3e-4` | Peak learning rate reached after warmup |
| `WarmupSteps` | `--warmup-steps` | int | `200` | Steps for linear LR ramp from 0 → `LearningRate`; `0` = no warmup |
| `MinLearningRate` | `--min-learning-rate` | float | `1e-5` | LR floor at the end of cosine decay (~10% of `LearningRate`) |
| `Beta1` | `--beta1` | float | `0.9` | Adam β₁ — gradient momentum decay |
| `Beta2` | `--beta2` | float | `0.999` | Adam β₂ — squared-gradient decay |
| `AdamEps` | `--adam-eps` | float | `1e-8` | Adam ε — division-by-zero guard |
| `GradClip` | `--grad-clip` | float | `1.0` | Global L2 gradient clipping norm |

---

## CLI Reference

All flags are case-insensitive. Pass them after `--` when using `dotnet run`:

```bash
dotnet run -- [flags]
```

Run with `--help` or `-h` to print every available flag:

```bash
dotnet run -- --help
```

**Examples:**

```bash
# GPU training with a large model
dotnet run -- --backend GPU \
              --train-file corpus.txt \
              --save-file model.bin \
              --embedding-dim 256 --num-heads 8 --num-layers 6 \
              --ffn-dim 1024 --epochs 50

# EarlyStopping with custom patience
dotnet run -- --training-mode EarlyStopping --patience 10 --min-delta-loss 0.0005

# Checkpoint every 30 minutes
dotnet run -- --checkpoint-every 30

# Interactive prompt with a saved model
dotnet run -- --action Prompt --load-file model.bin --temperature 0.7 --top-k 20

# Use a checkpoint file directly for prompting (Adam state is discarded automatically)
dotnet run -- --action Prompt --load-file weights.bin.checkpoint
```

---

## Training

### Training Modes

Controlled by `AppConfig.TrainingMode`:

#### `Epochs`
Train for exactly `TransformerConfig.Epochs` epochs with no early stopping. Weights are saved once at the end. Suitable when you want full control over training duration.

#### `Patience`
Train up to `Epochs` epochs, stopping early if validation loss does not improve for `Patience` consecutive epochs. Requires a validation split. Best weights (lowest validation loss) are saved mid-training as each improvement is found.

#### `EarlyStopping`
Like `Patience`, but only resets the counter when the improvement exceeds `MinDeltaLoss`. Useful for fine-tuning where tiny fluctuations should not reset patience. Requires a validation split.

### Validation Splits

Controlled by `AppConfig.ValidationSplit`:

| Mode | Behaviour |
|------|-----------|
| `None` | No validation. Train on all tokens. `Patience` and `EarlyStopping` are unavailable. |
| `Tail` | Hold out the **last** `ValidationFraction` of the corpus chronologically. Good for time-series text. |
| `Random` | Randomly assign `ValidationFraction` of fixed-size chunks to validation (reproducible, seed = 42). Prevents sequential overfitting. |

### Learning Rate Schedule

The scheduler applies **linear warmup** followed by **cosine decay**:

```
step ≤ WarmupSteps  →  LR = LearningRate × step / WarmupSteps
step >  WarmupSteps →  LR = MinLearningRate + (LearningRate − MinLearningRate)
                              × 0.5 × (1 + cos(π × progress))
```

where `progress = (step − WarmupSteps) / (totalSteps − WarmupSteps)`.

### Checkpointing

A crash-recovery checkpoint is saved:
- At the end of every epoch (always)
- Every `CheckpointEveryMinutes` wall-clock minutes within an epoch (when > 0)

Checkpoints store weights, Adam first and second moments, the current epoch, total Adam steps, and the inner-loop step position. On restart the training loop resumes from exactly where it stopped — no completed gradient updates are replayed.

When training finishes successfully the checkpoint file is deleted.

---

## Inference

Start the interactive prompt loop with a trained weights file or checkpoint:

```bash
dotnet run -- --action Prompt --load-file weights.bin
```

Type a prompt at the `>` cursor and the model generates up to `MaxTokens` tokens. Press **Ctrl-C** to exit.

The context window is managed automatically:
- **FIFO** (default): oldest tokens are dropped when the window fills
- **SlidingWindow**: the first `AnchorFraction` of the context (e.g. a system prompt) is preserved; the rest uses FIFO

---

## Model Architecture

A **decoder-only transformer** (GPT / GPT-2 style) with a pre-norm residual design:

```
Tokens
  └─► Embedding  (token lookup table + positional encoding)
        └─► Block 0
              ├─ LayerNorm → MultiHeadAttention → Residual
              └─ LayerNorm → FeedForward       → Residual
        └─► Block 1 … Block L-1
        └─► Final LayerNorm
        └─► Linear projection  (EmbeddingDim → VocabSize)
        └─► Logits → cross-entropy loss  (training)
                   → top-K softmax sample (inference)
```

### Attention

Multi-head causal self-attention with scaled dot-product:

```
Attention(Q, K, V) = softmax((Q·Kᵀ / √d_k) + causal_mask) · V
```

Position is encoded either by **sinusoidal additive PE** (added to embeddings before the first block) or **RoPE** (applied inside each attention head to Q and K vectors directly).

### Feed-Forward Network

A position-wise two-layer MLP inside every block:

```
FFN(x) = Linear(GELU(Linear(x, W₁, b₁)), W₂, b₂)
```

where the hidden dimension is `FFNDim` (conventionally 4 × EmbeddingDim).

### Activation

**GELU** (Gaussian Error Linear Unit):

```
GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

### Derived properties

```csharp
HeadDim        = EmbeddingDim / NumHeads        // per-head dimension
AttentionScale = 1 / √HeadDim                   // dot-product scaling factor
```

---

## Tokenizers

Five implementations of `ITokenizer` are included. The default is **UnigramTokenizer**.

| Tokenizer | Algorithm | Typical use |
|-----------|-----------|-------------|
| `CharTokenizer` | One token per character | Debugging, tiny corpora |
| `BpeTokenizer` | Byte-pair encoding (greedy merge) | GPT-2 style |
| `WordPieceTokenizer` | Longest-match subword (`##` prefix) | BERT style |
| `SentencePieceTokenizer` | BPE with `▁` word-boundary prefix | LLaMA / Mistral style |
| `UnigramTokenizer` | Unigram LM with EM training + Viterbi decode | T5 / ALBERT style (**default**) |

All tokenizers implement:

```csharp
int[]  Encode(string text)
string Decode(int[] ids)
string DecodeToken(int id)
void   SaveVocab(string path)
int    VocabSize { get; }
```

Vocabularies are saved alongside weights as `<SaveFile>.vocab` and reloaded automatically on resume.

---

## File Formats

### Weights File

Magic `0x4C4C4D01` — the first three bytes are the ASCII letters `L`, `L`, `M`; the low byte `01` identifies a weights-only file.

```
[int32]   magic          = 0x4C4C4D01
[int32]   version        = 1
[int32]   VocabSize
[int32]   EmbeddingDim
[int32]   NumHeads
[int32]   NumLayers
[int32]   FFNDim
[int32]   ContextLength
[int32]   paramCount
for each parameter:
  [int32]           rows
  [int32]           cols
  [rows×cols×f32]   weights  (row-major)
```

### Checkpoint File

Magic `0x4C4C4D02` — same `LLM` prefix; low byte `02` identifies a checkpoint.

The checkpoint stores weights **and** Adam optimizer state (first and second moments), allowing exact training resumption. It is also accepted by `Load` / `--load-file` directly — the Adam state is silently discarded so a checkpoint can be used for inference without conversion.

**Version 2** (written by this codebase):

```
[int32]   magic          = 0x4C4C4D02
[int32]   version        = 2
[int32]   VocabSize
[int32]   EmbeddingDim
[int32]   NumHeads
[int32]   NumLayers
[int32]   FFNDim
[int32]   ContextLength
[int32]   epoch          — 0-indexed epoch active at save time
[int32]   adamStep       — total Adam steps completed
[int32]   innerStep      — next inner-loop step within epoch (enables mid-epoch resume)
[int32]   paramCount
for each parameter:
  [int32]           rows
  [int32]           cols
  [rows×cols×f32]   weights
  [rows×cols×f32]   M  (Adam first moment)
  [rows×cols×f32]   V  (Adam second moment)
```

**Version 1** (older checkpoints — still loadable): identical except `innerStep` is absent. The training loop derives the inner position from `adamStep` and `stepsPerEpoch` as a fallback.

### Vocabulary File

Saved as `<SaveFile>.vocab` alongside every weights file. Format is tokenizer-specific but always loadable via:

```csharp
ITokenizer tokenizer = TokenizerIO.LoadVocab(path);
```

---

## Project Internals

### LLM\_Common

Shared library — no ML framework dependencies, pure .NET.

| File | Purpose |
|------|---------|
| `TransformerConfig.cs` | All hyper-parameters with derived properties and the LR schedule |
| `ITransformerModel.cs` | Backend-agnostic model interface (train step, evaluate, generate, save, load, checkpoint) |
| `IParameter.cs` | Learnable parameter interface (weights, gradients, Adam M/V, serialization) |
| `ILayer.cs` / `IEmbeddingLayer.cs` | Generic forward/backward interfaces |
| `ModelSerializer.cs` | Binary serialization for weights files and checkpoints (v1 + v2) |
| `Tokenizers/` | Five tokenizer implementations + `TokenizerIO` factory |

### LLM\_CPU

Pure managed C# backend. No native dependencies.

| Class | Purpose |
|-------|---------|
| `Matrix` | `float[,]` matrix with dot product, transpose, softmax, GELU, and all gradient operations |
| `Parameter` | Managed weight + gradient + Adam M/V tensors with CPU Adam update |
| `Embedding` | Token lookup table with sinusoidal PE or RoPE |
| `MultiHeadAttention` | Causal multi-head attention with KV-cache for inference |
| `LayerNorm` | Layer normalisation with learnable scale and bias |
| `FeedForward` | Position-wise FFN (linear → GELU → linear) |
| `TransformerBlock` | One transformer block (pre-norm + attention residual + FFN residual) |
| `TransformerModel` | Full decoder model implementing `ITransformerModel` |

### LLM\_GPU

ILGPU-accelerated backend. Mirrors the CPU class structure.

| Class | Purpose |
|-------|---------|
| `GpuContext` | ILGPU accelerator singleton (CUDA preferred, OpenCL fallback) |
| `GpuMatrix` | GPU-resident float buffer with ILGPU kernel operations |
| `GpuParameter` | Weight, gradient, and Adam state on device |
| `Kernels` | All GPU kernel definitions compiled by ILGPU |
| `GpuTransformerModel` | Full decoder model on GPU implementing `ITransformerModel` |

Both backends implement the same `ITransformerModel` interface, so `Program.cs` is entirely backend-agnostic.

### LLM\_App

CLI entry point.

| File | Purpose |
|------|---------|
| `Program.cs` | Argument parsing, config validation, corpus loading, tokenizer construction, training loop, checkpoint recovery, prompt loop |
| `AppConfig.cs` | Runtime settings bound from the `AppConfig` config section |
| `appsettings.json` | Default configuration with inline documentation |
| `ICorpusSplitter.cs` | Interface for train/validation splitting strategies |
| `TailSplitter.cs` | Holds out the last N% of tokens |
| `RandomSplitter.cs` | Random chunk-based split (reproducible, seed = 42) |

### LLM\_Documentation

Documentation-only project (never compiled into a binary).

| File | Contents |
|------|---------|
| `Overview.md` | High-level introduction, quick-start guide, project structure |
| `Architecture.md` | Class dependency graph, execution flow, backward pass data flow (Mermaid diagrams) |
| `NeuralNetwork.md` | Layer-by-layer architecture diagrams |
| `TransformerBlockDiagram.md` | Perceptron-level transformer block diagram |
| `Technologies.md` | Mathematical reference for all components and algorithms |
| `LearningResources.md` | References and further reading |
| `sample_corpus_tiny.txt` | Short Shakespeare excerpt for quick smoke tests |
| `sample_corpus_large.txt` | Complete works of Shakespeare (~5.4 MB) for full training runs |

---

## Dependencies

| Project | Package | Version | Purpose |
|---------|---------|---------|---------|
| `LLM_GPU` | `ILGPU` | 1.5.1 | GPU kernel compilation and execution |
| `LLM_GPU` | `ILGPU.Algorithms` | 1.5.1 | GPU algorithm primitives |
| `LLM_App` | `Microsoft.Extensions.Configuration.Json` | 9.0.0 | JSON config file loading |
| `LLM_App` | `Microsoft.Extensions.Configuration.Binder` | 9.0.0 | Binding config sections to objects |
| `LLM_App` | `Microsoft.Extensions.Configuration.CommandLine` | 9.0.0 | CLI flag parsing and config overlay |

All projects target **.NET 10.0** with nullable reference types enabled and implicit usings disabled.

---

## Public Domain Corpora

Large text datasets suitable for training. All sources below are free to download and either public domain or permissively licensed for research use.

### Literature

| Source | Size | Notes |
|--------|------|-------|
| [Project Gutenberg](https://www.gutenberg.org) | 70 000+ books | Plain `.txt` downloads for individual books. Use the [Gutenberg mirror list](https://www.gutenberg.org/MIRRORS.ALL) for bulk download. |
| [Standard Ebooks](https://standardebooks.org) | 800+ books | Cleaner, consistently formatted versions of Gutenberg texts. Individual EPUB/text downloads or [bulk download](https://standardebooks.org/bulk-downloads). |
| [Wikisource](https://en.wikisource.org) | Large | Public domain works hosted by Wikimedia. Exportable as plain text via the API. |

### Wikipedia

| Source | Size | Notes |
|--------|------|-------|
| [Wikipedia dumps](https://dumps.wikimedia.org/enwiki/latest/) | ~22 GB compressed | Full English Wikipedia XML dump. Use [WikiExtractor](https://github.com/attardi/wikiextractor) to convert to plain text: `python -m wikiextractor enwiki-latest-pages-articles.xml.bz2 -o output/` |
| [WikiText-2 / WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) | 2 MB / 500 MB | Pre-extracted, cleaned Wikipedia articles. Standard LM benchmark datasets. Direct download via Hugging Face. |

### Large Pre-assembled Datasets

| Source | Size | Notes |
|--------|------|-------|
| [The Pile](https://pile.eleuther.ai) | 825 GB | 22 diverse sources including books, Wikipedia, GitHub, arXiv, and FreeLaw. Individual subsets downloadable separately. |
| [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) | ~40 GB | Open recreation of GPT-2's WebText training set. Reddit-curated web pages. |
| [C4 (Colossal Clean Crawled Corpus)](https://huggingface.co/datasets/allenai/c4) | ~750 GB | Cleaned Common Crawl web text used to train T5. English subset ~300 GB. |
| [Common Crawl](https://commoncrawl.org) | Petabytes | Raw web crawl data. Requires significant cleaning. Monthly snapshots available via S3. |
| [ROOTS Corpus](https://huggingface.co/bigscience-data) | 1.6 TB | Multilingual dataset used to train BLOOM. Many languages, diverse domains. |

### Books

| Source | Size | Notes |
|--------|------|-------|
| [BookCorpus (HuggingFace)](https://huggingface.co/datasets/bookcorpus) | ~4 GB | ~11 000 unpublished books scraped from Smashwords. Used to train BERT and GPT. |
| [Gutenberg Dammit](https://github.com/aparrish/gutenberg-dammit) | ~10 GB | All plain-text Gutenberg books in a single archive, pre-cleaned and deduplicated. |
| [OpenLibrary](https://openlibrary.org/developers/dumps) | Large | Internet Archive's Open Library data dumps. |

### Code

| Source | Size | Notes |
|--------|------|-------|
| [The Stack](https://huggingface.co/datasets/bigcode/the-stack) | 6 TB | Permissively licensed source code in 358 languages. Smaller deduplicated subsets available. |
| [CodeParrot GitHub Code](https://huggingface.co/datasets/codeparrot/github-code) | ~1 TB | GitHub public repos filtered to permissive licenses. |
| [StarCoder data](https://huggingface.co/datasets/bigcode/starcoderdata) | ~780 GB | Curated subset of The Stack used to train StarCoder. |

### Scientific / Technical

| Source | Size | Notes |
|--------|------|-------|
| [arXiv bulk access](https://info.arxiv.org/help/bulk_data_s3.html) | ~1 TB | Full arXiv LaTeX source via S3. Requires AWS CLI: `aws s3 sync s3://arxiv-bulk-access .` |
| [PubMed Central Open Access](https://www.ncbi.nlm.nih.gov/pmc/tools/bulk-ft-download/) | ~300 GB | Full-text biomedical research articles. XML and plain text formats. |
| [FreeLaw (CourtListener)](https://www.courtlistener.com/api/bulk-info/) | Large | US court opinions and legal documents. Subset of The Pile. |

### Multilingual

| Source | Size | Notes |
|--------|------|-------|
| [CC-100](https://data.statmt.org/cc-100/) | ~2.5 TB | Common Crawl extracts for 100+ languages, used to train XLM-R. |
| [mC4](https://huggingface.co/datasets/allenai/c4) | Large | Multilingual version of C4. 101 languages. Available via Hugging Face. |
| [Oscar](https://oscar-project.org) | Large | Multilingual web corpus from Common Crawl. Deduplicated per language. |

### Tools for Downloading and Preparing Data

| Tool | Purpose | Link |
|------|---------|------|
| **Hugging Face `datasets`** | Download and stream most of the above with one line of Python | [huggingface.co/docs/datasets](https://huggingface.co/docs/datasets) |
| **WikiExtractor** | Convert Wikipedia XML dumps to plain text | [github.com/attardi/wikiextractor](https://github.com/attardi/wikiextractor) |
| **gutenberg-cleaner** | Strip Gutenberg headers/footers from downloaded books | [github.com/pgcorpus/gutenberg](https://github.com/pgcorpus/gutenberg) |
| **AWS CLI** | Bulk download arXiv, Common Crawl, and other S3-hosted datasets | [aws.amazon.com/cli](https://aws.amazon.com/cli/) |
| **Apache Spark / Dask** | Process and clean multi-hundred-GB datasets in parallel | [spark.apache.org](https://spark.apache.org) / [dask.org](https://www.dask.org) |

### Practical Recommendations by Training Scale

| Model size | Suggested corpus | Approx. tokens |
|------------|-----------------|----------------|
| Tiny (< 10 M params) | Tiny Shakespeare (included) | 1 M |
| Small (10–50 M params) | WikiText-103 or single Gutenberg author | 100 M |
| Medium (50–200 M params) | Full Gutenberg + WikiText-103 | 1 B |
| Large (200 M+ params) | The Pile subsets or C4 | 10 B+ |

---

## Future Development

### Architecture

- **SwiGLU activation** — replace GELU in the FFN with SwiGLU (`x · σ(x) · W₃`), used in LLaMA 2/3 and shown to outperform GELU at scale. Requires a third weight matrix per FFN block.
- **RMSNorm** — replace LayerNorm with Root Mean Square normalisation (no mean subtraction, no bias). Faster, lower memory, used in LLaMA / Mistral.
- **Grouped Query Attention (GQA)** — share K and V heads across groups of Q heads. LLaMA 2 70B uses 8 KV heads for 64 Q heads, dramatically reducing KV-cache memory during inference.
- **Sliding window attention** — limit each token's attention span to a local window (e.g. 4096 tokens) while preserving global context with a small number of sink tokens. Enables much longer sequences at constant compute cost.
- **Mixture of Experts (MoE)** — replace the FFN in each block with a router that selects K of N expert FFNs per token. Increases parameter count without proportionally increasing compute per token.
- **Multi-scale positional encoding** — ALiBi or NTK-aware RoPE scaling for better generalisation to sequence lengths beyond the training context window.

### Training

- **Mixed precision (BF16 / FP16)** — store weights in 16-bit, accumulate gradients in 32-bit. Roughly halves GPU memory and doubles throughput on modern hardware.
- **Gradient checkpointing** — recompute activations during the backward pass instead of storing them all. Trades compute for memory, enabling much larger models or batch sizes.
- **Streaming data loader** — currently the entire corpus is read into memory at startup. A streaming loader would support corpora larger than RAM by reading and tokenising chunks on demand.
- **Distributed training** — data-parallel training across multiple GPUs or machines using gradient aggregation (AllReduce). Currently single-device only.
- **Learning rate finder** — automatically sweep LR over a short run and plot loss to identify the optimal peak learning rate before a full training run.
- **LoRA fine-tuning** — Low-Rank Adaptation: freeze the base weights and train small rank-decomposed update matrices (ΔW = BA). Enables efficient fine-tuning of a pretrained model with a fraction of the parameters.

### Inference

- **Top-p (nucleus) sampling** — sample from the smallest set of tokens whose cumulative probability exceeds p, complementing the existing top-K filter.
- **Repetition penalty** — discount the logits of recently generated tokens to reduce loops and repetition in long outputs.
- **Beam search** — maintain the K most probable partial sequences at each step rather than greedily sampling one. Produces more coherent outputs at the cost of K× compute.
- **Batched inference** — run multiple prompt completions in parallel on the GPU instead of one at a time.
- **KV-cache on GPU** — the CPU backend has a KV-cache for fast inference; the GPU backend recomputes K and V on every forward pass. Adding a GPU KV-cache would dramatically speed up token generation.
- **Quantisation (INT8 / INT4)** — reduce weight precision post-training. Cuts model size 2–4× and speeds up CPU inference with minimal quality loss.
- **ONNX export** — export the trained model to ONNX format for deployment in other runtimes (ONNX Runtime, TensorRT, mobile).

### Evaluation

- **Standard benchmarks** — report perplexity on WikiText-2 / WikiText-103 so results are comparable to published models.
- **Few-shot evaluation** — measure accuracy on tasks like HellaSwag or BoolQ by prompting the model and scoring completions.
- **Loss curve visualisation** — the `.csv` log exists; add a companion script (Python / gnuplot) to plot train and validation loss curves automatically.

### Data

- **Byte-level BPE** — operate on raw UTF-8 bytes rather than characters, eliminating unknown tokens entirely. Used by GPT-2 and RoBERTa.
- **Data mixing** — train on a weighted blend of multiple corpora simultaneously rather than a single file.
- **Deduplication** — near-duplicate removal from large web corpora significantly improves data quality and generalisation.

### Infrastructure

- **REST API** — wrap the prompt loop in a lightweight HTTP server (ASP.NET Core minimal API) so the model can be queried from other applications.
- **Model metadata in the weights file** — embed the tokenizer type, training corpus name, and training date directly in the file header so a model file is self-describing.
- **Web-based loss dashboard** — serve the `.csv` log as a live-updating chart during training (SignalR or simple SSE endpoint).
- **Automated hyperparameter search** — grid or random search over embedding dim, learning rate, and batch size with early stopping to find the best configuration for a given corpus.
