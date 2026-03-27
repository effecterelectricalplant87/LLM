# Architecture — Class Structure & Program Flow

This document describes the code architecture of the LLM solution: how the source
files relate to each other, what each class is responsible for, and how data flows
through the system from startup to trained model.

The solution is split into five projects:

| Project | Role |
|---|---|
| `LLM_Common` | Shared interfaces, config, serializer |
| `LLM_CPU` | CPU backend — all math in managed C# |
| `LLM_GPU` | GPU backend — ILGPU kernels on CUDA / OpenCL |
| `LLM_App` | Entry point, CLI, training loop |
| `LLM_Documentation` | Documentation only (never compiled) |

---

## CPU Backend — Class Dependency Graph

The diagram below shows every CPU-side class and the **depends-on** relationship
between them (an arrow `A → B` means "A uses B").

```mermaid
graph TD
    %% ── Entry point ──────────────────────────────────────────────
    PROG["<b>Program</b><br/><i>LLM_App/Program.cs</i><br/>─────────────────<br/>• Reads appsettings.json + CLI overrides<br/>• Validates AppConfig (prints errors)<br/>• Loads corpus, builds tokeniser<br/>• Creates model via ITransformerModel<br/>• Training loop (random chunks)<br/>• Validation split + early stopping<br/>• LR warmup + cosine decay schedule<br/>• Time-based + epoch-end checkpointing<br/>• Mid-epoch resume via innerStep<br/>• Text generation"]

    %% ── Interfaces (LLM_Common) ──────────────────────────────────
    ITFM["<b>ITransformerModel</b><br/><i>LLM_Common</i><br/>─────────────────<br/>• TrainStep()<br/>• AccumulateStep()<br/>• ZeroAllGradients()<br/>• ScaleAllGradients()<br/>• ClipAndUpdate()<br/>• Generate()<br/>• Save() / Load()<br/>• SaveCheckpoint() / LoadCheckpoint()"]

    SER["<b>ModelSerializer</b><br/><i>LLM_Common</i><br/>─────────────────<br/>• Binary save/load (weights v1)<br/>• Checkpoint save/load (v1/v2)<br/>• Config validation<br/>• Per-param shape check<br/>• Adam M/V moment persistence"]

    SPLIT["<b>ICorpusSplitter</b><br/><i>LLM_Common</i><br/>─────────────────<br/>• TailSplitter (last N% held out)<br/>• RandomSplitter (random chunks)<br/>• None (all tokens for training)"]

    %% ── Top-level model ──────────────────────────────────────────
    MODEL["<b>TransformerModel</b><br/><i>LLM_CPU</i><br/>─────────────────<br/>• Implements ITransformerModel<br/>• Orchestrates layers<br/>• Forward / Backward<br/>• ZeroGrad / ClipGrad / Update<br/>• Generate()"]

    %% ── Config ───────────────────────────────────────────────────
    CFG["<b>TransformerConfig</b><br/><i>LLM_Common</i><br/>─────────────────<br/>• EmbeddingDim<br/>• NumHeads / NumLayers<br/>• FFNDim / ContextLength<br/>• Adam settings"]

    %% ── Tokenisers ───────────────────────────────────────────────
    TOK["<b>ITokenizer + 5 implementations</b><br/><i>LLM_Common/Tokenizers/</i><br/>─────────────────<br/>• CharTokenizer<br/>• BpeTokenizer<br/>• WordPieceTokenizer<br/>• SentencePieceTokenizer<br/>• UnigramTokenizer (default)<br/>• Encode: str → int[]<br/>• DecodeToken: int → str<br/>• SaveVocab / LoadVocab"]

    %% ── Layers ───────────────────────────────────────────────────
    EMB["<b>Embedding</b><br/><i>LLM_CPU</i><br/>─────────────────<br/>• Token embedding table<br/>• Sinusoidal pos encoding<br/>• Forward + Backward"]

    BLOCK["<b>TransformerBlock</b><br/><i>LLM_CPU</i><br/>─────────────────<br/>• Pre-norm design<br/>• Residual connections<br/>• Forward + Backward"]

    ATTN["<b>MultiHeadAttention</b><br/><i>LLM_CPU</i><br/>─────────────────<br/>• Q / K / V projections<br/>• Causal masking<br/>• H parallel heads<br/>• Forward + Backward"]

    FFN["<b>FeedForward</b><br/><i>LLM_CPU</i><br/>─────────────────<br/>• Expand: D → 4D<br/>• GELU activation<br/>• Project: 4D → D<br/>• Forward + Backward"]

    LN["<b>LayerNorm</b><br/><i>LLM_CPU</i><br/>─────────────────<br/>• Per-position norm<br/>• Learnable γ and β<br/>• Forward + Backward"]

    %% ── Core primitives ──────────────────────────────────────────
    PARAM["<b>Parameter</b><br/><i>LLM_CPU</i><br/>─────────────────<br/>• Weight matrix<br/>• Gradient matrix<br/>• Adam m and v state<br/>• Update() → Adam step"]

    MAT["<b>Matrix</b><br/><i>LLM_CPU</i><br/>─────────────────<br/>• float[rows, cols]<br/>• Dot, Transpose<br/>• Add, Sub, Mul, Scale<br/>• Softmax, GELU<br/>• SoftmaxBackward<br/>• GELUGrad<br/>• Xavier / Normal init"]

    %% ── Dependency edges ─────────────────────────────────────────
    PROG  --> TOK
    PROG  --> ITFM
    PROG  --> CFG
    PROG  --> SPLIT

    ITFM  --> MODEL
    MODEL --> SER
    MODEL --> CFG
    MODEL --> EMB
    MODEL --> BLOCK
    MODEL --> LN
    MODEL --> PARAM

    EMB   --> PARAM
    EMB   --> MAT
    EMB   --> CFG

    BLOCK --> LN
    BLOCK --> ATTN
    BLOCK --> FFN
    BLOCK --> MAT

    ATTN  --> PARAM
    ATTN  --> MAT
    ATTN  --> CFG

    FFN   --> PARAM
    FFN   --> MAT

    LN    --> PARAM
    LN    --> MAT

    PARAM --> MAT

    %% ── Styling ──────────────────────────────────────────────────
    style PROG  fill:#4a90d9,color:#fff,stroke:#2c5f8a
    style ITFM  fill:#2980b9,color:#fff,stroke:#1a5276
    style SER   fill:#2980b9,color:#fff,stroke:#1a5276
    style SPLIT fill:#2980b9,color:#fff,stroke:#1a5276
    style MODEL fill:#5ba35b,color:#fff,stroke:#3a6b3a
    style CFG   fill:#888,color:#fff,stroke:#555
    style TOK   fill:#888,color:#fff,stroke:#555
    style EMB   fill:#d4812a,color:#fff,stroke:#9a5c1a
    style BLOCK fill:#9b59b6,color:#fff,stroke:#6c3483
    style ATTN  fill:#c0392b,color:#fff,stroke:#922b21
    style FFN   fill:#c0392b,color:#fff,stroke:#922b21
    style LN    fill:#c0392b,color:#fff,stroke:#922b21
    style PARAM fill:#16a085,color:#fff,stroke:#0e6655
    style MAT   fill:#2c3e50,color:#fff,stroke:#1a252f
```

---

## GPU Backend — Class Summary

The GPU backend (`LLM_GPU`) mirrors the CPU backend class-for-class but stores matrices
in device memory and executes operations via ILGPU kernels.

| GPU class | CPU equivalent | Notes |
|---|---|---|
| `GpuContext` | _(none)_ | ILGPU accelerator singleton; prefers CUDA → OpenCL → CPU |
| `GpuMatrix` | `Matrix` | GPU-resident float buffer; operations launch kernels |
| `GpuParameter` | `Parameter` | Weight + gradient + Adam state on device |
| `GpuEmbedding` | `Embedding` | |
| `GpuLayerNorm` | `LayerNorm` | |
| `GpuMultiHeadAttention` | `MultiHeadAttention` | |
| `GpuFeedForward` | `FeedForward` | |
| `GpuTransformerBlock` | `TransformerBlock` | |
| `GpuTransformerModel` | `TransformerModel` | Implements `ITransformerModel` |
| `Kernels` | _(Math.cs / Matrix.cs)_ | All GPU kernel definitions (static methods compiled by ILGPU) |

`ITransformerModel` (in `LLM_Common`) is the interface that allows `Program.cs` to be
completely backend-agnostic — a single `--gpu` flag switches the entire model at runtime.

---

## Program Execution Flow

The diagram below traces the program from startup through one training epoch
and then to text generation.

```mermaid
flowchart TD
    START([Program starts])

    subgraph INIT["Initialisation"]
        direction TB
        A["Parse args\n--cpu/--gpu  --train  --load  --save  --prompt"]
        B["Read corpus file (required if --train;\nskipped if --load and .vocab file exists)"]
        C["Build tokeniser:\n  If <load>.vocab exists → TokenizerIO.LoadVocab()\n  Else → new UnigramTokenizer(corpus, 500)\nCorpus string → int[] allTokens (if --train)"]
        D["new TransformerConfig\nSet hyperparameters"]
        E["Create model via ITransformerModel\n  --cpu → new TransformerModel(cfg, rng)\n  --gpu → new GpuTransformerModel(cfg, rng)\n(~800 K parameters at default config)"]
        F["If --load: model.Load(path) + tokenizer from path.vocab\nRead binary weights, validate config, restore vocab"]
    end

    subgraph TRAIN["Training Loop  (if --train)"]
        direction TB
        G["stepsPerEpoch = (len(tokens) - T - 1) / T + 1"]
        H["Pick random offset in [0, len(tokens)-T-1]\nExtract input[0..T-1] and target[1..T]"]
        I["If AccumulationSteps=1:\n  model.TrainStep() — ZeroGrad→Forward→Backward→Clip→Adam\nElse:\n  Accumulate N chunks, ScaleGrads(1/N), ClipAndUpdate"]
        J{More steps this epoch?}
        K["Log epoch loss and perplexity\nEvery SampleEvery epochs: generate a sample"]
        L{More epochs?}
    end

    subgraph SAVE["After training"]
        M["If --save: model.Save(path) + tokenizer.SaveVocab(path+'.vocab')\nWrite binary weights and companion vocabulary file"]
    end

    subgraph GEN["Text Generation  (if --prompt)"]
        direction TB
        N["tokenizer.Encode(prompt) → int[]"]
        O["model.Generate(promptIds, numTokens:200,\ntemperature:0.8, topK:15)"]
        P["Autoregressive loop:\n  Forward → last-position logits\n  → temperature scale → top-k mask\n  → softmax → categorical sample\n  → append token, repeat"]
        Q["tokenizer.DecodeToken() each output ID\nPrint generated text"]
    end

    START --> INIT
    A --> B --> C --> D --> E --> F
    F --> TRAIN
    G --> H --> I --> J
    J -- yes --> H
    J -- no  --> K --> L
    L -- yes --> H
    L -- no  --> SAVE --> GEN
    N --> O --> P --> Q
```

---

## Backward Pass Data Flow

During backpropagation the gradient signal travels in the **opposite direction**
to the forward pass.  The diagram below shows the gradient flowing through one
transformer block.

```mermaid
flowchart BT
    LOSS(["∂L / ∂logits\n(from cross-entropy)"])

    subgraph OUTPUT["Output Head  (reversed)"]
        direction BT
        OP["OutputProjection\n∂L/∂W_out = normedᵀ · ∂L/∂logits\n∂L/∂normed = ∂L/∂logits · W_outᵀ"]
        FN["FinalLayerNorm.Backward()\n∂L/∂blockOut = LN_backward(∂L/∂normed)"]
    end

    subgraph BLOCK_BACK["TransformerBlock.Backward()  (one block, reversed)"]
        direction BT
        RES2["Second residual split\n∂L/∂(afterAttn) = ∂L/∂out\n∂L/∂ffnOut      = ∂L/∂out"]
        FFN_B["FeedForward.Backward()\nW2 grad, GELU grad, W1 grad\n→ ∂L/∂normed2"]
        LN2_B["LayerNorm2.Backward()\nγ,β grads → ∂L/∂afterAttn (FFN path)"]
        RES1["First residual add\n∂L/∂afterAttn += FFN_path\n∂L/∂x = ∂L/∂afterAttn (skip path)"]
        ATT_B["MultiHeadAttention.Backward()\nWo,Wq,Wk,Wv grads\n→ ∂L/∂normed1"]
        LN1_B["LayerNorm1.Backward()\nγ,β grads → ∂L/∂x (attn path)"]
        RES0["∂L/∂x = skip + attn path\n→ upstream (next block or embedding)"]
    end

    subgraph EMB_BACK["Embedding.Backward()"]
        direction BT
        EMBG["Scatter-add gradients\ninto token embedding rows"]
    end

    LOSS --> OP --> FN
    FN --> RES2
    RES2 --> FFN_B --> LN2_B
    RES2 --> RES1
    LN2_B --> RES1
    RES1 --> ATT_B --> LN1_B --> RES0
    RES0 --> EMBG
```

---

## Project Dependency Map

```mermaid
graph LR
    APP[LLM_App] --> CPU[LLM_CPU]
    APP --> GPU[LLM_GPU]
    APP --> CMN[LLM_Common]

    CPU --> CMN
    GPU --> CMN

    style APP fill:#4a90d9,color:#fff
    style CPU fill:#5ba35b,color:#fff
    style GPU fill:#9b59b6,color:#fff
    style CMN fill:#2c3e50,color:#fff
```

## CPU File Dependency Map

```mermaid
graph LR
    P[Program.cs] --> T[LLM_Common/Tokenizers/]
    P --> TIO[TokenizerIO.cs]
    P --> ITFM[ITransformerModel]
    P --> TC[TransformerConfig.cs]
    TIO --> T

    ITFM --> TM[TransformerModel.cs]
    TM --> SER[ModelSerializer.cs]
    TM --> E[Embedding.cs]
    TM --> TB[TransformerBlock.cs]
    TM --> LN[LayerNorm.cs]
    TM --> PR[Parameter.cs]
    TM --> TC

    TB --> MHA[MultiHeadAttention.cs]
    TB --> FF[FeedForward.cs]
    TB --> LN
    TB --> M[Matrix.cs]

    MHA --> PR
    MHA --> M
    MHA --> TC

    FF --> PR
    FF --> M

    E --> PR
    E --> M
    E --> TC

    LN --> PR
    LN --> M

    PR --> M

    style M    fill:#2c3e50,color:#fff
    style PR   fill:#16a085,color:#fff
    style LN   fill:#c0392b,color:#fff
    style MHA  fill:#e74c3c,color:#fff
    style FF   fill:#e74c3c,color:#fff
    style TB   fill:#9b59b6,color:#fff
    style E    fill:#d4812a,color:#fff
    style TM   fill:#5ba35b,color:#fff
    style TC   fill:#7f8c8d,color:#fff
    style T    fill:#7f8c8d,color:#fff
    style TIO  fill:#7f8c8d,color:#fff
    style ITFM fill:#2980b9,color:#fff
    style SER  fill:#2980b9,color:#fff
    style P    fill:#4a90d9,color:#fff
```
