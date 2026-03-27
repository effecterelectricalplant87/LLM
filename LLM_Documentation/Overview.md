# LLM — GPT-style Transformer from Scratch in C#

A fully self-contained implementation of a modern large-language-model architecture
written in raw C# with no external machine-learning libraries.  Every operation —
matrix math, attention, layer normalisation, backpropagation, and the Adam optimiser —
is implemented from first principles so the code can serve as a precise, readable
reference for how a transformer works.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Configuration & CLI](#configuration--cli)
5. [Training](#training)
6. [Text Generation](#text-generation)
7. [Key Algorithms](#key-algorithms)
8. [Further Reading](#further-reading)

---

## Overview

### What is a Transformer?

A transformer is a neural network architecture introduced in the 2017 paper
*"Attention Is All You Need"* (Vaswani et al.).  It replaced recurrent networks
(RNNs / LSTMs) as the dominant architecture for language tasks because:

- **Parallelism** – every position in the sequence is processed simultaneously,
  unlike RNNs which process step by step.
- **Long-range dependencies** – self-attention directly connects any two positions
  in *O(1)* operations regardless of distance.
- **Scalability** – performance improves predictably as model size and data grow.

Modern LLMs (GPT-4, Claude, Gemini, Llama) are all scaled-up versions of this
same basic architecture.

### What this implementation includes

| Feature | Status |
|---|---|
| Multi-head causal self-attention | ✅ |
| Rotary Positional Encoding (RoPE) — default | ✅ |
| Sinusoidal positional encoding — optional | ✅ |
| Pre-norm transformer blocks | ✅ |
| GELU activation | ✅ |
| Layer normalisation | ✅ |
| Residual connections | ✅ |
| Full backpropagation (no autograd) | ✅ |
| Adam optimiser with linear warmup + cosine LR decay | ✅ |
| Gradient clipping | ✅ |
| Gradient accumulation | ✅ |
| Character-level tokeniser | ✅ |
| BPE tokeniser | ✅ |
| WordPiece tokeniser | ✅ |
| SentencePiece tokeniser | ✅ |
| Unigram language model tokeniser (default) | ✅ |
| Temperature + top-k sampling | ✅ |
| Cross-entropy language-model loss | ✅ |
| Random chunk sampling during training | ✅ |
| Validation split (Tail or Random) | ✅ |
| Early stopping / patience-based stopping | ✅ |
| Model weight save / load | ✅ |
| Training checkpoint save / load (with mid-epoch resume) | ✅ |
| Tokeniser vocabulary save / load alongside weights (`.vocab`) | ✅ |
| JSON hyperparameter config via `appsettings.json` + CLI overrides | ✅ |
| GPU backend via ILGPU (CUDA / OpenCL / CPU) | ✅ |
| CPU / GPU selection via `--backend` CLI flag or config | ✅ |

---

## Quick Start

### Prerequisites

- .NET 10.0 SDK or later
- (Optional) NVIDIA GPU with CUDA drivers for GPU mode

### Build

```
dotnet build LLM_App/LLM_App.csproj
```

### Train on a corpus (using defaults from appsettings.json)

```
dotnet run --project LLM_App
```

### Train with CLI overrides

```
dotnet run --project LLM_App -- --action Train --backend GPU --train-file corpus.txt --save-file model.bin
```

### Prompt the model

```
dotnet run --project LLM_App -- --action Prompt --load-file model.bin --sample-prompt "To be, or not"
```

### See all CLI options

```
dotnet run --project LLM_App -- --help
```

The tokeniser vocabulary is saved alongside the weights as `model.bin.vocab`, so no
corpus is needed for inference once the model has been saved.

---

## Project Structure

```
LLM/
├── LLM_Common/               ← Shared interfaces and types
│   ├── ILayer.cs             ←   Generic layer interface (Forward / Backward)
│   ├── IEmbeddingLayer.cs    ←   Embedding interface
│   ├── IParameter.cs         ←   Learnable parameter interface
│   ├── ITransformerModel.cs  ←   Top-level model interface (Train / Generate / Save / Load)
│   ├── ICorpusSplitter.cs    ←   Corpus split interface
│   ├── TailSplitter.cs       ←   Hold out last N% of tokens for validation
│   ├── RandomSplitter.cs     ←   Randomly assign chunks to train/validation
│   ├── ModelSerializer.cs    ←   Binary weight & checkpoint save / load
│   ├── TransformerConfig.cs  ←   All hyperparameters (with LR schedule helper)
│   └── Tokenizers/           ←   Tokeniser interface + five implementations
│       ├── ITokenizer.cs     ←     Interface (Encode / Decode / SaveVocab)
│       ├── TokenizerIO.cs    ←     Vocab file save / load factory
│       ├── CharTokenizer.cs
│       ├── BpeTokenizer.cs
│       ├── WordPieceTokenizer.cs
│       ├── SentencePieceTokenizer.cs
│       └── UnigramTokenizer.cs  ← default
│
├── LLM_CPU/                  ← CPU backend
│   ├── Matrix.cs             ←   Core 2-D float matrix and math operations
│   ├── Parameter.cs          ←   Learnable weight + gradient + Adam state
│   ├── Embedding.cs          ←   Token embeddings + RoPE / sinusoidal PE
│   ├── LayerNorm.cs          ←   Layer normalisation (forward + backward)
│   ├── MultiHeadAttention.cs ←   Multi-head causal self-attention (forward + backward)
│   ├── FeedForward.cs        ←   Position-wise FFN with GELU (forward + backward)
│   ├── TransformerBlock.cs   ←   One transformer layer (attention + FFN + residuals)
│   └── TransformerModel.cs   ←   Full model, gradient clipping, text generation
│
├── LLM_GPU/                  ← GPU backend (ILGPU)
│   ├── GpuContext.cs         ←   ILGPU accelerator lifecycle (CUDA → OpenCL → CPU)
│   ├── GpuMatrix.cs          ←   GPU-resident float matrix
│   ├── GpuParameter.cs       ←   Weight + gradient + Adam state on device
│   ├── Kernels.cs            ←   All GPU kernel definitions
│   ├── GpuEmbedding.cs       ←   GPU token embedding + positional encoding
│   ├── GpuLayerNorm.cs       ←   GPU layer normalisation
│   ├── GpuMultiHeadAttention.cs ← GPU multi-head causal self-attention
│   ├── GpuFeedForward.cs     ←   GPU feed-forward network
│   ├── GpuTransformerBlock.cs ←  GPU transformer block
│   └── GpuTransformerModel.cs ←  GPU full model (implements ITransformerModel)
│
├── LLM_App/                  ← Entry point
│   ├── Program.cs            ←   CLI, training loop, validation, early stopping
│   ├── AppConfig.cs          ←   Runtime settings (Action, Backend, paths, sampling)
│   └── appsettings.json      ←   Default hyperparameter config (JSON with comments)
│
├── LLM_Documentation/        ← Documentation project (never compiled)
│   ├── Overview.md           ←   This file
│   ├── Architecture.md       ←   Class dependency diagram
│   ├── NeuralNetwork.md      ←   Neural network layer diagram
│   ├── Technologies.md       ←   Mathematical reference for all components
│   ├── TransformerBlockDiagram.md ← Perceptron-level Mermaid diagram
│   └── LearningResources.md  ←   Curated reading path from beginner to primary papers
│
├── sample_corpus.txt         ← Small Shakespeare excerpt for quick tests
└── sample_corpus_large.txt   ← Complete works of Shakespeare (~5.4 MB)
```

---

## Configuration & CLI

### Configuration layers

All settings live in two sources that are merged at startup:

1. **`appsettings.json`** — the default configuration file, with comments explaining every field.
2. **CLI flags** — override any `appsettings.json` value without editing the file.

The CLI flag always wins when both specify the same setting.

### Viewing all CLI flags

```
dotnet run --project LLM_App -- --help
```

### Key CLI flags

| Flag | Config path | Description |
|---|---|---|
| `--action <value>` | `AppConfig:Action` | `Train` (default) or `Prompt` |
| `--backend <value>` | `AppConfig:Backend` | `GPU` (default) or `CPU` |
| `--train-file <path>` | `AppConfig:TrainFile` | Corpus file to train on |
| `--save-file <path>` | `AppConfig:SaveFile` | Save weights here after training |
| `--load-file <path>` | `AppConfig:LoadFile` | Load weights before training / for prompting |
| `--error-file <path>` | `AppConfig:ErrorFile` | Redirect stderr to a file |
| `--checkpoint-every <min>` | `AppConfig:CheckpointEveryMinutes` | Save checkpoint every N minutes (default 60) |
| `--training-mode <mode>` | `AppConfig:TrainingMode` | `Epochs`, `Patience`, or `EarlyStopping` |
| `--validation-split <mode>` | `AppConfig:ValidationSplit` | `None`, `Tail`, or `Random` |
| `--embedding-dim <n>` | `TransformerConfig:EmbeddingDim` | Residual stream width |
| `--num-heads <n>` | `TransformerConfig:NumHeads` | Number of attention heads |
| `--num-layers <n>` | `TransformerConfig:NumLayers` | Number of transformer blocks |
| `--context-length <n>` | `TransformerConfig:ContextLength` | Maximum sequence length |
| `--epochs <n>` | `TransformerConfig:Epochs` | Maximum training epochs |
| `--learning-rate <f>` | `TransformerConfig:LearningRate` | Peak Adam learning rate |
| `--use-rope <bool>` | `TransformerConfig:UseRoPE` | Use RoPE (true) or sinusoidal PE (false) |

### AppConfig settings

| Parameter | Default | Description |
|---|---|---|
| `Action` | `Train` | `Train` — run training loop; `Prompt` — generate text |
| `Backend` | `GPU` | Compute backend: `GPU` (CUDA/OpenCL) or `CPU` |
| `TrainFile` | _(required for Train)_ | Path to the training corpus |
| `SaveFile` | `weights.bin` | Where to save weights after training |
| `LoadFile` | _(empty)_ | Weights file to load before training or prompting |
| `ErrorFile` | `error.txt` | Redirect stderr here (empty = console) |
| `CheckpointEveryMinutes` | `60` | Wall-clock minutes between mid-epoch checkpoints |
| `VocabSize` | `4000` | Target tokeniser vocabulary size |
| `ValidationSplit` | `Tail` | `None`, `Tail`, or `Random` |
| `ValidationFraction` | `0.1` | Fraction of corpus held out for validation |
| `TrainingMode` | `EarlyStopping` | `Epochs`, `Patience`, or `EarlyStopping` |
| `Patience` | `5` | Epochs without improvement before stopping |
| `MinDeltaLoss` | `0.001` | Minimum val-loss drop that resets patience counter |
| `MaxTokens` | `200` | Tokens to generate per Prompt response |
| `Temperature` | `0.8` | Sampling temperature (1.0 = unmodified) |
| `TopK` | `15` | Top-K sampling filter (0 = disabled) |

### TransformerConfig settings (model architecture)

| Parameter | appsettings.json default | Code default | Description |
|---|---|---|---|
| `EmbeddingDim` | 768 | 128 | Residual stream width (*d_model*) |
| `NumHeads` | 16 | 4 | Parallel attention heads (must divide `EmbeddingDim`) |
| `NumLayers` | 8 | 4 | Transformer blocks stacked |
| `FFNDim` | 3072 | 512 | Feed-forward hidden size (4 × d_model) |
| `ContextLength` | 128 | 128 | Maximum sequence length |
| `UseRoPE` | `true` | `true` | Rotary positional encoding (recommended) |
| `Seed` | 42 | 42 | Weight-initialisation RNG seed (`-1` = random) |
| `Epochs` | 25 | 10 | Maximum training epochs |
| `AccumulationSteps` | 1 | 1 | Gradient accumulation steps (1 = disabled) |
| `LearningRate` | 3e-4 | 3e-4 | Peak Adam step size |
| `WarmupSteps` | 200 | 100 | Linear LR warmup length in Adam steps |
| `MinLearningRate` | 1e-5 | 1e-5 | LR floor at end of cosine decay |
| `Beta1` | 0.9 | 0.9 | Adam first-moment decay |
| `Beta2` | 0.999 | 0.999 | Adam second-moment decay |
| `AdamEps` | 1e-8 | 1e-8 | Adam denominator epsilon |
| `GradClip` | 1.0 | 1.0 | Gradient L2 norm clip threshold |

`VocabSize` is always set automatically by the tokeniser at runtime.

---

## Training

### Tokeniser

The default tokeniser is the **Unigram language model tokeniser** with a target vocabulary
of 4000 subword tokens, using the `▁` (U+2581) space prefix convention (T5 / ALBERT style).
Four other tokenisers are also implemented: character-level, BPE, WordPiece, and SentencePiece.
See `Technologies.md` §4 for full mathematical details.

### Chunk sampling

At each training step a random starting position is drawn uniformly from the token array,
giving a context window of `ContextLength` tokens:

```
offset ~ Uniform(0, len(tokens) − ContextLength − 1)
input  = tokens[offset  : offset + ContextLength]
target = tokens[offset+1: offset + ContextLength + 1]
```

The number of steps per epoch is fixed at `floor(len(tokens) / ContextLength)`, so the
total token budget per epoch is the same regardless of corpus size.  Random sampling
prevents the model from overfitting the fixed sequential order of chunks.

### Validation splits

When `ValidationSplit` is not `None`, a portion of the corpus is held out:

- **`Tail`** — the last `ValidationFraction` of tokens form the validation set.
  Deterministic; preserves narrative order if the corpus has one.
- **`Random`** — individual chunks are randomly assigned to train or validation
  based on `ValidationFraction`.  Maximises coverage at the cost of order.

Validation loss is computed at the end of each epoch and used by training-stop modes.

### Training stop modes

| Mode | Behaviour |
|---|---|
| `Epochs` | Train for exactly `Epochs` epochs; no early stopping |
| `Patience` | Stop if validation loss does not improve for `Patience` consecutive epochs |
| `EarlyStopping` | Stop when validation loss improvement falls below `MinDeltaLoss` |

`Patience` and `EarlyStopping` both require a validation split (not `None`).

### Learning rate schedule

The learning rate follows a **linear warmup + cosine decay** schedule:

```
steps 1 … WarmupSteps :  LR = LearningRate × step / WarmupSteps
steps WarmupSteps+1 … end :  LR = MinLearningRate + (LearningRate − MinLearningRate) × ½(1 + cos(π·progress))
```

Warmup prevents large gradient updates at the start when the Adam moments are cold.
Cosine decay gradually reduces the learning rate rather than stopping abruptly.

### Checkpointing

The training loop saves crash-recovery checkpoints in two ways:

1. **Time-based (mid-epoch)** — every `CheckpointEveryMinutes` minutes of wall-clock time,
   a checkpoint is written that records the exact inner-loop step so training can resume
   without replaying any already-applied gradient updates.
2. **Epoch-end** — a checkpoint is always written at the end of each completed epoch.

Checkpoint files use a different magic number (`0x4C4C4D02`) from weight files
(`0x4C4C4D01`) but are interchangeable: `--load-file` accepts either format.
When a checkpoint is loaded, the Adam M/V moments are also restored so the
optimiser state continues without a cold-start spike.

### Loss function — Cross-Entropy

The model is trained with the standard language-modelling objective: given a
sequence of tokens `t₀, t₁, …, t_{T-1}`, predict the next token at every position.

```
L = −(1/T) · Σ_t  log P(t_{t+1} | t₀…t_t)
```

**Perplexity** (`exp(L)`) is the most interpretable metric:
- A random model achieves `perplexity = VocabSize` (e.g. ~4000 for the default unigram tokeniser).
- A well-trained small model reaches perplexity < 5 on its training data.

### Gradient of softmax cross-entropy

One of the most elegant results in deep learning: the gradient of the combined
softmax + cross-entropy with respect to the raw logits simplifies to:

```
dL/d(logits[t, j]) = (1/T) · (P[t,j] − 𝟏{j = target_t})
```

The softmax probabilities, with 1 subtracted at the correct class.

### Adam optimiser

Adam (Kingma & Ba, 2014) adapts the learning rate per parameter:

```
m_t = β₁·m_{t-1} + (1−β₁)·g_t          ← exponential moving average of g
v_t = β₂·v_{t-1} + (1−β₂)·g_t²         ← exponential moving average of g²
θ_t = θ_{t-1} − α · m̂_t / (√v̂_t + ε)  ← bias-corrected update
```

Bias correction (`m̂ = m/(1−β₁ᵗ)`) compensates for the zero initialisation.

### Gradient accumulation

Setting `AccumulationSteps = N` accumulates gradients across N random chunks before
one Adam update.  The effective token budget per update becomes `N × ContextLength`
and all gradients are scaled by `1/N` before clipping.  At `N = 1` (default) this is
a standard single-step update.

### Model weights and vocabulary

`--save-file model.bin` writes two files:

- `model.bin` — binary weights (magic number `0x4C4C4D01`, config header, per-tensor shapes, float data).
- `model.bin.vocab` — tokeniser vocabulary (type tag, token text, log-probabilities / merge rules).

Checkpoints (`model.bin.checkpoint`) use magic `0x4C4C4D02` and additionally store
Adam M/V moments plus the epoch, Adam step, and inner step for exact resume.

The config header is validated on load so a model saved with one architecture cannot
be loaded with a different one.  Both the weights file and `.vocab` must be kept
together as a pair; `--load-file` reads them automatically.

---

## Text Generation

After training, the model generates text autoregressively:

1. Encode a prompt into token IDs.
2. Forward pass → logits at the last position.
3. Apply temperature (scale logits by `1/T`).
4. Apply top-k mask (zero out all but the top-k logits).
5. Softmax → probability distribution.
6. Sample one token.
7. Append and repeat from step 2.

**Temperature** controls randomness:
- `T = 0.5` — sharp, repetitive, more predictable output.
- `T = 1.0` — unmodified distribution.
- `T = 1.5` — wild, creative, more likely to be incoherent.

**Top-k** limits the candidate set:
- `k = 1` — greedy decoding (always pick the most likely token).
- `k = 10..40` — a good range for language tasks.

---

## Key Algorithms

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k  +  mask ) · V
```

- **Scale** `1/√d_k` prevents the dot products from growing so large that the
  softmax saturates into near-zero gradients.
- **Causal mask** sets future positions to −∞ so they receive 0 attention weight.

### Rotary Positional Encoding (RoPE)

RoPE (Su et al., 2021) encodes position directly into the query and key vectors by
rotating them in 2D subspaces before the attention dot product.  For each head dimension
pair `(2i, 2i+1)` at position `pos`:

```
θᵢ = pos / 10000^(2i / d_head)

[q_{2i}, q_{2i+1}] ← [q_{2i}·cos(θᵢ) − q_{2i+1}·sin(θᵢ),
                       q_{2i}·sin(θᵢ) + q_{2i+1}·cos(θᵢ)]
```

Because the rotation is applied to both Q and K, the attention score between positions
`s` and `t` depends only on their relative distance `s − t`, giving better
length generalisation than sinusoidal PE without using any extra parameters.

### Sinusoidal Positional Encoding (optional)

Available when `UseRoPE = false`.  A fixed (non-learned) signal added to the token embedding:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Layer Normalisation

```
y_d = γ_d · (x_d − μ) / √(σ² + ε)  +  β_d
```

Normalised per position (over the feature dimension), so it works at any batch size.

### GELU Activation

```
GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
```

Smoother than ReLU; used in GPT-2 and most subsequent models.

---

## Further Reading

- **Vaswani et al. (2017)** — *Attention Is All You Need* — the original transformer paper.
- **Radford et al. (2019)** — *GPT-2: Language Models are Unsupervised Multitask Learners* — GPT-2 design.
- **Su et al. (2021)** — *RoFormer: Enhanced Transformer with Rotary Position Embedding* — RoPE.
- **Ba et al. (2016)** — *Layer Normalization* — LayerNorm paper.
- **Kingma & Ba (2014)** — *Adam: A Method for Stochastic Optimization* — Adam paper.
- **Glorot & Bengio (2010)** — *Understanding the difficulty of training deep feedforward neural networks* — Xavier initialisation.
- **Hendrycks & Gimpel (2016)** — *Gaussian Error Linear Units (GELUs)* — GELU paper.
- **Kudo (2018)** — *Subword Regularization* — Unigram LM tokeniser.
- **Sennrich et al. (2016)** — *Neural Machine Translation of Rare Words with Subword Units* — BPE.
