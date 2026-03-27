# Neural Network Structure

This document describes the exact neural network implemented by this project,
showing every layer, its dimensions, internal nodes, and how data flows and
transforms from raw token IDs to next-token probability distributions.

The architecture is a **decoder-only transformer** (GPT-style) — the same
fundamental design used in GPT-2, GPT-3, GPT-4, Llama, Mistral, and Claude.

---

## Full Network — High-Level Data Flow

The diagram below shows the entire network for a sequence of **T tokens**,
with **L = 4 transformer blocks** and a vocabulary of **V tokens**.
Tensor shapes are shown at each stage (`T` = sequence length, `D` = `EmbeddingDim`,
`V` = `VocabSize`).

```mermaid
flowchart TD
    IN["<b>Input Token IDs</b><br/>int[T]<br/>e.g. [42, 7, 15, …]"]

    subgraph EMBED["Embedding Layer"]
        direction TB
        TOKLOOKUP["Token Embedding Lookup<br/>Table: V × D<br/>Row i = learned vector for token i<br/>Output: T × D"]
        POSADD["＋ Sinusoidal Position Encoding<br/>PE[pos, 2i]   = sin(pos / 10000^(2i/D))<br/>PE[pos, 2i+1] = cos(pos / 10000^(2i/D))<br/>Shape: T × D  (fixed, not learned)"]
        EOUT["Residual Stream  x<br/>T × D"]
        TOKLOOKUP --> POSADD --> EOUT
    end

    subgraph BLOCKS["Transformer Blocks  ×L  (L = 4)"]
        direction TB

        subgraph BLK["One Transformer Block"]
            direction TB

            LN1["LayerNorm 1<br/>Per-position: normalise D features<br/>Learnable γ, β  ∈ ℝᴰ<br/>T × D"]

            subgraph MHSA["Multi-Head Self-Attention"]
                direction LR
                WQ["Wq  D×D<br/>+ bq  D"]
                WK["Wk  D×D<br/>+ bk  D"]
                WV["Wv  D×D<br/>+ bv  D"]
                QKV_SPLIT["Split into H heads<br/>each head: T × d_k<br/>where d_k = D/H = 32"]

                subgraph HEADS["H = 4 Parallel Attention Heads"]
                    direction TB
                    H0["Head 0<br/>Q₀·K₀ᵀ/√d_k<br/>+ causal mask<br/>softmax → A₀<br/>A₀·V₀"]
                    H1["Head 1<br/>Q₁·K₁ᵀ/√d_k<br/>+ causal mask<br/>softmax → A₁<br/>A₁·V₁"]
                    H2["Head 2<br/>Q₂·K₂ᵀ/√d_k<br/>+ causal mask<br/>softmax → A₂<br/>A₂·V₂"]
                    H3["Head 3<br/>Q₃·K₃ᵀ/√d_k<br/>+ causal mask<br/>softmax → A₃<br/>A₃·V₃"]
                end

                CONCAT["Concatenate heads<br/>T × D"]
                WO["Wo  D×D  + bo  D<br/>Output projection<br/>T × D"]

                WQ & WK & WV --> QKV_SPLIT
                QKV_SPLIT --> H0 & H1 & H2 & H3
                H0 & H1 & H2 & H3 --> CONCAT --> WO
            end

            RES1["＋ Residual (skip connection)<br/>x ← x + Attention(LN1(x))<br/>T × D"]

            LN2["LayerNorm 2<br/>Per-position normalisation<br/>Learnable γ, β  ∈ ℝᴰ<br/>T × D"]

            subgraph FFNET["Position-wise Feed-Forward Network"]
                direction TB
                W1["W1  D × 4D  + b1  4D<br/>Linear expand<br/>T × 4D"]
                GELU["GELU activation<br/>x · 0.5·(1 + tanh(√(2/π)·(x + 0.044715x³)))<br/>T × 4D"]
                W2["W2  4D × D  + b2  D<br/>Linear project back<br/>T × D"]
                W1 --> GELU --> W2
            end

            RES2["＋ Residual (skip connection)<br/>x ← x + FFN(LN2(x))<br/>T × D"]

            LN1 --> MHSA --> RES1 --> LN2 --> FFNET --> RES2
        end

        BREPEAT["⟳  Repeat for all L blocks<br/>(each block has its own independent weights)"]
    end

    FLN["<b>Final LayerNorm</b><br/>Normalise the last hidden state<br/>Learnable γ, β  ∈ ℝᴰ<br/>T × D"]

    subgraph HEAD["Language Model Head"]
        direction TB
        PROJ["Output Projection<br/>W_out  D × V  + b_out  V<br/>Linear map to vocabulary<br/>T × V  (logits)"]
        SM["Softmax  (inference only)<br/>P[t, j] = exp(logit[t,j]) / Σ exp(logit[t,:])<br/>T × V  (probabilities)"]
        PROJ --> SM
    end

    NEXT["<b>Next-Token Probabilities</b><br/>P[t, :]  — distribution over V tokens<br/>Sample token t+1 from P[T-1, :]"]

    IN --> EMBED --> BLOCKS --> FLN --> HEAD --> NEXT
```

---

## Attention Head Detail

Each of the **H = 4** attention heads performs scaled dot-product attention
over its own d_k = 32-dimensional subspace.

```mermaid
flowchart LR
    X["Input x_h<br/>T × d_k<br/>(slice of Q or K or V)"]

    subgraph SDPA["Scaled Dot-Product Attention — one head"]
        direction TB

        QQ["Q_h  =  x · Wq_slice<br/>T × d_k<br/>(query vectors)"]
        KK["K_h  =  x · Wk_slice<br/>T × d_k<br/>(key vectors)"]
        VV["V_h  =  x · Wv_slice<br/>T × d_k<br/>(value vectors)"]

        SCORE["Attention Scores<br/>S = Q_h · K_hᵀ / √d_k<br/>T × T<br/>S[i,j] = similarity of position i to j"]

        MASK["Causal Mask  +M<br/>M[i,j] = 0    if j ≤ i   (can attend)<br/>M[i,j] = −∞  if j > i   (blocked future)<br/>Enforces autoregressive constraint"]

        SOFT["Softmax (row-wise)<br/>A_h[i,j] = exp(S[i,j]) / Σ_k exp(S[i,k])<br/>T × T — attention weight matrix<br/>Each row is a probability distribution over positions"]

        WSUM["Weighted Sum of Values<br/>out_h = A_h · V_h<br/>T × d_k<br/>out_h[i] = Σ_j A_h[i,j] · V_h[j]"]
    end

    OUT["Head output<br/>T × d_k<br/>(to be concatenated with other heads)"]

    X --> QQ & KK & VV
    QQ & KK --> SCORE --> MASK --> SOFT
    SOFT & VV --> WSUM --> OUT
```

---

## Layer Normalisation Detail

Applied independently at each of the T sequence positions over the D features.

```mermaid
flowchart LR
    XI["x  —  one position<br/>vector ∈ ℝᴰ"]

    subgraph LN_DETAIL["LayerNorm  (one position)"]
        direction TB
        MU["μ = (1/D) Σ_d x_d<br/>scalar mean over features"]
        SIG["σ = √( (1/D)Σ_d(x_d−μ)² + ε )<br/>scalar std over features  (ε = 1e-5)"]
        NORM["x̂_d = (x_d − μ) / σ<br/>normalised vector ∈ ℝᴰ"]
        SCALE["y_d = γ_d · x̂_d + β_d<br/>learnable per-feature scale and shift<br/>γ, β ∈ ℝᴰ  (initialised: γ=1, β=0)"]
        MU & SIG --> NORM --> SCALE
    end

    YO["y  —  normalised output<br/>vector ∈ ℝᴰ"]

    XI --> LN_DETAIL --> YO
```

---

## Feed-Forward Network Detail

The same MLP (shared weights) is applied to every position independently.

```mermaid
flowchart LR
    XFF["x  ∈ ℝᴰ<br/>one position<br/>(D = 128)"]

    subgraph FFN_DETAIL["Position-wise FFN"]
        direction TB
        L1["Linear 1<br/>h = x · W1 + b1<br/>W1 ∈ ℝᴰˣ⁴ᴰ,  b1 ∈ ℝ⁴ᴰ<br/>h ∈ ℝ⁴ᴰ  (expand: 128 → 512)"]
        GA["GELU(h)<br/>≈ h · 0.5·(1 + tanh(√(2/π)(h + 0.044715h³)))<br/>smooth non-linearity<br/>h' ∈ ℝ⁴ᴰ"]
        L2["Linear 2<br/>y = h' · W2 + b2<br/>W2 ∈ ℝ⁴ᴰˣᴰ,  b2 ∈ ℝᴰ<br/>y ∈ ℝᴰ  (project back: 512 → 128)"]
        L1 --> GA --> L2
    end

    YFF["y  ∈ ℝᴰ<br/>output (same shape as input)"]

    XFF --> FFN_DETAIL --> YFF
```

---

## Parameter Count Breakdown

With the default configuration (D=128, H=4, L=4, d_ff=512, V≈500 from Unigram tokeniser):

```mermaid
pie title Total Parameters  (~922 K)
    "Token Embeddings  (V×D)" : 64000
    "Attention Wq/Wk/Wv/Wo  (4 blocks)" : 262144
    "Attention biases  (4 blocks)" : 2048
    "FFN W1/W2  (4 blocks)" : 524288
    "FFN biases  (4 blocks)" : 10240
    "LayerNorm γ/β  (10 norms)" : 2560
    "Output Projection  (D×V + V)" : 64500
```

| Component | Formula | Count (default) |
|---|---|---|
| Token embedding | V × D | 64 000 |
| Per block: Q/K/V/O weights | 4 × D² | 65 536 |
| Per block: Q/K/V/O biases | 4 × D | 512 |
| Per block: FFN weights | D×4D + 4D×D | 131 072 |
| Per block: FFN biases | 4D + D | 640 |
| Per block: LayerNorm (×2) | 2 × 2D | 512 |
| **Per block total** | | **~198 272** |
| **All L=4 blocks** | | **~793 088** |
| Final LayerNorm | 2D | 256 |
| Output projection + bias | D×V + V | 64 500 |
| **Grand total** | | **~921 844** |

---

## Residual Stream View

The **residual stream** is the central abstraction of the transformer.
It starts as the embedding of the input and is incrementally updated by each
sub-layer.  No layer replaces the stream — each one only adds to it.

```mermaid
flowchart TD
    STREAM0["Residual stream  x₀<br/>= TokenEmb + PosEnc<br/>T × D"]

    subgraph BLOCK1["Block 1"]
        ATTN1["Attention contribution<br/>Δx = Attn(LN(x₀))<br/>T × D"]
        STREAM1["x₁ = x₀ + Δx_attn"]
        FFN1["FFN contribution<br/>Δx = FFN(LN(x₁))<br/>T × D"]
        STREAM2["x₂ = x₁ + Δx_ffn"]
        ATTN1 --> STREAM1 --> FFN1 --> STREAM2
    end

    subgraph BLOCK2["Block 2"]
        ATTN2["Attention contribution"]
        STREAM3["x₃ = x₂ + Δx_attn"]
        FFN2["FFN contribution"]
        STREAM4["x₄ = x₃ + Δx_ffn"]
        ATTN2 --> STREAM3 --> FFN2 --> STREAM4
    end

    DOTS["⋮  (blocks 3 and 4 follow the same pattern)"]

    FINAL_LN["Final LayerNorm<br/>Normalise the last stream state"]
    PROJ["Output Projection → logits"]

    STREAM0 --> ATTN1
    STREAM0 --> STREAM1
    STREAM1 --> ATTN1
    STREAM2 --> BLOCK2
    STREAM2 --> ATTN2
    STREAM4 --> DOTS --> FINAL_LN --> PROJ
```

The residual connections mean that gradients flow directly from the loss all the way
back to the embedding layer through the identity shortcut, making very deep networks
trainable without gradient vanishing.
