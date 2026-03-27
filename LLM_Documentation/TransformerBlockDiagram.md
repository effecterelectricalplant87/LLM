# Transformer Block — Perceptron-Level Diagram

A single transformer block processes one vector of **d_model = 128 floats per token position**.
The block contains two sub-layers, each wrapped in a residual ("skip") connection.
Four of these blocks are stacked (`NumLayers = 4`).

Representative neurons are shown; `⋯` nodes stand for the remaining dimensions.
Solid arrows carry activations; dashed arrows are residual (skip) connections.

---

```mermaid
flowchart TD

    %% ─────────────────────────────────────────────────────────────────────
    %%  GPT-STYLE TRANSFORMER BLOCK — Perceptron-level wiring
    %%  Config: d_model=128  H=4 heads  d_head=32  d_ff=512
    %% ─────────────────────────────────────────────────────────────────────

    %% ═══ Residual stream input ═══════════════════════════════════════════
    subgraph RS_IN["Residual stream in  ·  128 floats per token"]
        direction LR
        xi1(("x₁")) ~~~ xi2(("x₂")) ~~~ xi3(("x₃")) ~~~ xiN(("  ⋯  ")) ~~~ xiD(("x₁₂₈"))
    end

    %% ═══ ATTENTION SUB-LAYER ═════════════════════════════════════════════
    subgraph ATTNSL["▌ ATTENTION SUB-LAYER"]

        subgraph LN1_G["LayerNorm₁   learnable γ,β ∈ ℝ¹²⁸   (256 params)"]
            lnmu(["μ = mean(x)   σ² = var(x)\nshared across all 128 dims"])
            lna(["x̂₁ = γ₁·(x₁-μ)/σ + β₁"]) ~~~ lnb(["x̂₂ = γ₂·(x₂-μ)/σ + β₂"]) ~~~ lnN(["⋯"]) ~~~ lnD(["x̂₁₂₈"])
        end

        subgraph QPRO["Query projection   W_Q ∈ ℝ¹²⁸×¹²⁸   128 neurons, each a weighted sum of all x̂ᵢ"]
            direction LR
            qa(["q₁ = Σᵢ W_Q[1,i]·x̂ᵢ"]) ~~~ qb(["q₂ = Σᵢ W_Q[2,i]·x̂ᵢ"]) ~~~ qN(["⋯"]) ~~~ qD(["q₁₂₈"])
        end

        subgraph KPRO["Key projection   W_K ∈ ℝ¹²⁸×¹²⁸   (same structure as W_Q)"]
            direction LR
            ka(["k₁ = Σᵢ W_K[1,i]·x̂ᵢ"]) ~~~ kb(["k₂"]) ~~~ kN(["⋯"]) ~~~ kD(["k₁₂₈"])
        end

        subgraph VPRO["Value projection   W_V ∈ ℝ¹²⁸×¹²⁸   (same structure as W_Q)"]
            direction LR
            va(["v₁ = Σᵢ W_V[1,i]·x̂ᵢ"]) ~~~ vb(["v₂"]) ~~~ vN(["⋯"]) ~~~ vD(["v₁₂₈"])
        end

        subgraph MHA_G["Causal Multi-Head Self-Attention   H=4 heads · d_head=32   all heads run in parallel"]

            subgraph HD1["Head 1   Q[:,1:32]  K[:,1:32]  V[:,1:32]"]
                direction TB
                h1sc(["score_ts = Σᵢ Q¹ₜᵢ · K¹ₛᵢ / √32\none dot-product per ordered token pair (t,s)"])
                h1mk(["causal mask: score_ts = −∞ when s > t\ntoken t cannot attend to future token s"])
                h1sf(["α_ts = exp(score_ts) / Σ_s' exp(score_ts')\nsoftmax distributes attention weight across positions"])
                h1ws(["c¹_t = Σₛ α_ts · V¹_s\nweighted sum of value rows → 32 output dims"])
                h1sc --> h1mk --> h1sf --> h1ws
            end

            subgraph HD2["Head 2   Q[:,33:64]  K[:,33:64]  V[:,33:64]   (identical structure to Head 1)"]
                h2all(["score/√32  →  causal mask  →  softmax  →  Σα·V   →   32 dims"])
            end

            HREST(["Heads 3 & 4\n(same structure)\n32 + 32 output dims"])
        end

        subgraph OPRO["Output projection   W_O ∈ ℝ¹²⁸×¹²⁸   re-mixes all heads"]
            concat_n(["concat(c¹, c², c³, c⁴)  →  128 values\nall four head outputs laid end-to-end"])
            oa(["o₁ = Σᵢ W_O[1,i]·catᵢ"]) ~~~ ob(["o₂"]) ~~~ oN(["⋯"]) ~~~ oD(["o₁₂₈"])
        end

    end

    ADDA{{"Residual Add ①\nattn_out + x"}}

    %% ═══ FEED-FORWARD SUB-LAYER ══════════════════════════════════════════
    subgraph FFNSL["▌ FEED-FORWARD SUB-LAYER"]

        subgraph LN2_G["LayerNorm₂   γ,β ∈ ℝ¹²⁸   (same structure as LayerNorm₁)"]
            direction LR
            ln2a(["x̂'₁"]) ~~~ ln2b(["x̂'₂"]) ~~~ ln2N(["⋯"]) ~~~ ln2D(["x̂'₁₂₈"])
        end

        subgraph FFN1_G["W₁ + b₁   W₁ ∈ ℝ¹²⁸×⁵¹²   128 inputs  →  512 neurons   (4× width expansion)"]
            direction LR
            h1a(["h₁ = Σᵢ W₁[1,i]·x̂'ᵢ + b₁"]) ~~~ h1b(["h₂"]) ~~~ h1N(["⋯"]) ~~~ h1D(["h₅₁₂"])
        end

        subgraph GACT_G["GELU activation   gᵢ = hᵢ · Φ(hᵢ)   element-wise · no learnable weights"]
            direction LR
            gla(["g₁ = GELU(h₁)"]) ~~~ glb(["g₂"]) ~~~ glN(["⋯"]) ~~~ glD(["g₅₁₂"])
        end

        subgraph FFN2_G["W₂ + b₂   W₂ ∈ ℝ⁵¹²×¹²⁸   512 inputs  →  128 neurons   (project back to d_model)"]
            direction LR
            f1a(["f₁ = Σᵢ W₂[1,i]·gᵢ + b₂"]) ~~~ f1b(["f₂"]) ~~~ f1N(["⋯"]) ~~~ f1D(["f₁₂₈"])
        end

    end

    ADDF{{"Residual Add ②\nffn_out + attn_out + x"}}

    subgraph RS_OUT["Residual stream out  ·  128 floats per token  →  next block (×4) or vocabulary head"]
        direction LR
        yo1(("y₁")) ~~~ yo2(("y₂")) ~~~ yoN(("  ⋯  ")) ~~~ yoD(("y₁₂₈"))
    end

    %% ─── Connections ─────────────────────────────────────────────────────

    %% Input → LN1 statistics (μ,σ need every dimension)
    xi1 & xi2 & xi3 & xiN & xiD --> lnmu

    %% Each output neuron also needs its own xᵢ (for the centering step)
    xi1 --> lna
    xi2 --> lnb
    xiN --> lnN
    xiD --> lnD
    lnmu --> lna & lnb & lnN & lnD

    %% LN1 → Q projection (fully connected: each of 128 query neurons reads every x̂ᵢ)
    lna & lnb & lnN & lnD -->|"128 weights each"| qa & qb & qN & qD

    %% LN1 → K, V projections (same all-to-all pattern, labels omitted for clarity)
    lna & lnb & lnN & lnD --> ka & kb & kN & kD
    lna & lnb & lnN & lnD --> va & vb & vN & vD

    %% Slice Q,K,V into heads — head 1 gets first 32 dims
    qa & qb -->|"dims 1–32"| h1sc
    ka & kb -->|"dims 1–32"| h1sc
    va & vb -->|"dims 1–32"| h1ws

    %% Head 2 gets next 32 dims
    qb & qN -->|"dims 33–64"| h2all
    kb & kN -->|"dims 33–64"| h2all
    vb & vN -->|"dims 33–64"| h2all

    %% Heads 3 & 4 get remaining dims
    qN & qD & kN & kD & vN & vD --> HREST

    %% Concat head outputs → W_O projection
    h1ws & h2all & HREST --> concat_n
    concat_n -->|"128 weights each"| oa & ob & oN & oD

    %% W_O → Residual Add ①, plus skip from input x
    oa & ob & oN & oD --> ADDA
    xi1 & xiD -.->|"residual skip"| ADDA

    %% ADD① → LN2
    ADDA --> ln2a & ln2b & ln2N & ln2D

    %% LN2 → W₁ (fully connected: each of 512 hidden neurons reads all 128 normed inputs)
    ln2a & ln2b & ln2N & ln2D -->|"128 weights each"| h1a & h1b & h1N & h1D

    %% W₁ → GELU (one-to-one: activation is element-wise, no cross-connections)
    h1a --> gla
    h1b --> glb
    h1N --> glN
    h1D --> glD

    %% GELU → W₂ (fully connected: each of 128 output neurons reads all 512 activations)
    gla & glb & glN & glD -->|"512 weights each"| f1a & f1b & f1N & f1D

    %% W₂ → Residual Add ②, plus skip from ADD①
    f1a & f1b & f1N & f1D --> ADDF
    ADDA -.->|"residual skip"| ADDF

    %% Final output
    ADDF --> yo1 & yo2 & yoN & yoD
```

---

## Layer-by-layer summary

| Step | Operation | Shape | Learned params |
|------|-----------|-------|----------------|
| 1 | **LayerNorm₁** — normalize each dimension using μ,σ across all 128 dims, then scale+shift | 128→128 | γ, β ∈ ℝ¹²⁸ (256) |
| 2 | **Q projection** — each output neuron is a weighted sum of all 128 normed inputs | 128→128 | W_Q (16 384) |
| 3 | **K projection** — same structure as Q | 128→128 | W_K (16 384) |
| 4 | **V projection** — same structure as Q | 128→128 | W_V (16 384) |
| 5 | **Slice into heads** — Q, K, V are split into H=4 slices of 32 dims each | 128→4×32 | none |
| 6 | **Attention scores** — per head: dot product of every Q row with every K row, scaled by 1/√32 | T×32, T×32 → T×T | none |
| 7 | **Causal mask** — set score to −∞ for any future key position (s > t) | T×T | none |
| 8 | **Softmax** — convert masked scores to attention weights that sum to 1 per query row | T×T | none |
| 9 | **Weighted sum** — per head: each output row = Σ_s weight_s × V_s (sum of value rows) | T×T, T×32 → T×32 | none |
| 10 | **Concat heads** — stack the four 32-dim outputs end-to-end | 4×32→128 | none |
| 11 | **Output projection W_O** — fully connected: each output neuron reads all 128 concat values | 128→128 | W_O (16 384) |
| 12 | **Residual Add ①** — add the original input x, bypassing attention entirely | 128 | none |
| 13 | **LayerNorm₂** — same structure as LayerNorm₁ | 128→128 | γ, β (256) |
| 14 | **W₁ + GELU** — expand: 128 inputs → 512 neurons, each a full dot product, then GELU | 128→512 | W₁ (65 536), b₁ (512) |
| 15 | **W₂** — compress: 512 GELU outputs → 128 neurons, each a full dot product | 512→128 | W₂ (65 536), b₂ (128) |
| 16 | **Residual Add ②** — add the output of Residual Add ①, bypassing the FFN entirely | 128 | none |

**Total params per block ≈ 197 888**
Four blocks + embedding table + final projection ≈ **800 000 parameters** for the default config.

## Key connectivity rules

- **Fully connected (dense)**: Q/K/V/W_O projections and both FFN linear layers.
  Every output neuron receives a signal from every input neuron via a learned weight.

- **Element-wise (one-to-one)**: LayerNorm scale+shift (after statistics are computed), GELU activation.
  Neuron i in one layer connects only to neuron i in the next.

- **All-to-one (then one-to-all)**: The μ,σ statistics in LayerNorm are computed from all inputs and then broadcast into every output neuron.

- **Cross-position (attention only)**: The attention weighted sum lets token *t* aggregate information from any earlier token *s ≤ t*. Every other operation is **position-independent** — the same weights process each position identically.
