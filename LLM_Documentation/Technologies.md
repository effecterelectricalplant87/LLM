# LLM Technologies Reference

This document explains every technology used in this solution: the mathematics behind each component, how forward and backward passes work, and how all pieces compose into a working GPT-style language model.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Matrix Operations](#2-matrix-operations)
3. [Weight Initialisation](#3-weight-initialisation)
4. [Tokenisers](#4-tokenisers)
5. [Embeddings and Positional Encoding](#5-embeddings-and-positional-encoding)
6. [Layer Normalisation](#6-layer-normalisation)
7. [Multi-Head Causal Self-Attention](#7-multi-head-causal-self-attention)
8. [Feed-Forward Network](#8-feed-forward-network)
9. [Transformer Block](#9-transformer-block)
10. [Full Transformer Model](#10-full-transformer-model)
11. [Training Loop](#11-training-loop)
12. [Cross-Entropy Loss](#12-cross-entropy-loss)
13. [Adam Optimiser](#13-adam-optimiser)
14. [Gradient Clipping](#14-gradient-clipping)
15. [Text Generation](#15-text-generation)
16. [Model Save / Load](#16-model-save--load)
17. [GPU Backend](#17-gpu-backend)

---

## 1. Architecture Overview

This solution implements a **decoder-only GPT-style transformer** — the same fundamental architecture behind GPT-2, GPT-3, and similar autoregressive language models.

The full forward pass is:

$$\text{tokenIds} \;\xrightarrow{\text{Embed}}\; x^{(0)} \;\xrightarrow{\text{Block}_0}\; x^{(1)} \;\xrightarrow{\text{Block}_1}\; \cdots \;\xrightarrow{\text{Block}_{L-1}}\; x^{(L)} \;\xrightarrow{\text{LN}}\; \hat{x} \;\xrightarrow{\text{Proj}}\; \text{logits}$$

The model is trained by minimising cross-entropy loss between predicted next-token distributions and actual next tokens, with gradients computed analytically through every layer.

### Default Hyperparameters

| Symbol | Parameter | Default |
|--------|-----------|---------|
| $V$ | VocabSize | set by tokeniser |
| $d$ | EmbeddingDim | 128 |
| $H$ | NumHeads | 4 |
| $L$ | NumLayers | 4 |
| $d_{ff}$ | FFNDim | 512 |
| $T$ | ContextLength | 128 |
| $\alpha$ | LearningRate | $3 \times 10^{-4}$ |
| $\beta_1$ | Adam Beta1 | 0.9 |
| $\beta_2$ | Adam Beta2 | 0.999 |
| $\varepsilon$ | Adam Eps | $10^{-8}$ |
| $\tau$ | GradClip | 1.0 |

Derived quantities:

$$d_k = \frac{d}{H} \qquad \text{(head dimension)} \qquad s = \frac{1}{\sqrt{d_k}} \qquad \text{(attention scale)}$$

---

## 2. Matrix Operations

All computation is built on a 2D float matrix with row-major storage. The key operations and their gradients are:

### Matrix Multiplication

$$C = A \cdot B \qquad C[i,j] = \sum_k A[i,k]\, B[k,j]$$

where $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$, $C \in \mathbb{R}^{m \times n}$.

Gradients with respect to a scalar loss $\mathcal{L}$:

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial C} \cdot B^\top \qquad \frac{\partial \mathcal{L}}{\partial B} = A^\top \cdot \frac{\partial \mathcal{L}}{\partial C}$$

### Numerically Stable Softmax

Applied row-wise. For row $i$ with elements $z_1, \ldots, z_n$:

$$m_i = \max_j z_{ij}$$

$$\text{softmax}(z_i)_j = \frac{\exp(z_{ij} - m_i)}{\sum_k \exp(z_{ik} - m_i)}$$

Subtracting $m_i$ before exponentiating prevents overflow while leaving the result unchanged (the constant cancels in numerator and denominator).

### Softmax Backward

Let $\mathbf{s} = \text{softmax}(\mathbf{z})$ and $\mathbf{g} = \partial \mathcal{L}/\partial \mathbf{s}$. The upstream gradient with respect to the pre-softmax inputs is:

$$\frac{\partial \mathcal{L}}{\partial z_i} = s_i \left( g_i - \sum_k s_k\, g_k \right)$$

This follows from the Jacobian $\partial s_j / \partial z_i = s_j(\delta_{ij} - s_i)$.

### GELU Activation

The Gaussian Error Linear Unit (GPT-2 tanh approximation):

$$\text{GELU}(x) \approx \frac{x}{2}\left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\,(x + 0.044715\, x^3)\right)\right)$$

Let $\kappa = \sqrt{2/\pi}\,(x + 0.044715\, x^3)$ and $t = \tanh(\kappa)$. The derivative is:

$$\text{GELU}'(x) = \tfrac{1}{2}(1 + t) + \tfrac{1}{2}\,x\,(1 - t^2)\cdot\sqrt{\tfrac{2}{\pi}}\,(1 + 3 \cdot 0.044715\, x^2)$$

### Bias Addition (Broadcast)

A bias vector $\mathbf{b} \in \mathbb{R}^{1 \times n}$ is broadcast across all rows:

$$C[i,j] = A[i,j] + b[j]$$

The gradient of $\mathcal{L}$ with respect to $\mathbf{b}$ sums upstream gradients over all rows:

$$\frac{\partial \mathcal{L}}{\partial b_j} = \sum_i \frac{\partial \mathcal{L}}{\partial C[i,j]}$$

---

## 3. Weight Initialisation

### Xavier / Glorot Uniform

Used for projection matrices. Samples from a uniform distribution:

$$W \sim \mathcal{U}\!\left(-\ell,\, +\ell\right) \qquad \ell = \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}$$

This keeps activation and gradient variance approximately constant across layers, preventing vanishing or exploding signals at initialisation.

### Normal (Gaussian) via Box–Muller Transform

To sample $x \sim \mathcal{N}(\mu, \sigma^2)$, draw $u_1, u_2 \sim \mathcal{U}(0,1]$ and compute:

$$z = \sqrt{-2\ln u_1}\cdot\cos(2\pi u_2)$$

$$x = \mu + z\,\sigma$$

The default parameters follow the GPT-2 convention: $\mu = 0$, $\sigma = 0.02$.

---

## 4. Tokenisers

A tokeniser maps a raw text string to a sequence of integer token IDs and back. Five implementations are provided.

### 4.1 Character Tokeniser

The simplest possible tokeniser. Each unique character in the corpus becomes one token:

$$\text{vocab} = \{c_0 = \texttt{UNK},\, c_1,\, c_2,\, \ldots\} \qquad |\text{vocab}| = |\text{unique chars}| + 1$$

Tokens are assigned IDs in sorted order for determinism. Vocabulary size is typically 50–200 for English text.

### 4.2 Byte Pair Encoding (BPE)

BPE (Sennrich et al., 2016) iteratively merges the most frequent adjacent pair of tokens. Starting from individual characters:

1. Pre-tokenise corpus into words (split on whitespace); append `</w>` to each word.
2. Count all adjacent pair frequencies, weighted by word frequency:
$$\text{count}(a, b) = \sum_{w \in \text{vocab}} \text{freq}(w) \cdot \text{occurrences}(a\,b \text{ in } w)$$
3. Merge the most frequent pair $(a^*, b^*)$:
$$a^* b^* = \operatorname*{arg\,max}_{(a,b)} \text{count}(a, b)$$
4. Record the merge rank and repeat for `numMerges` steps.

**Encoding** applies the learned merges greedily in rank order (lower rank = earlier merge = higher priority).

### 4.3 WordPiece (BERT style)

Similar to BPE but selects merges by a likelihood-ratio score rather than raw frequency:

$$\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \cdot \text{freq}(b)}$$

This favours merges that increase the unigram model likelihood. Continuation subwords are prefixed with `##` (e.g. `playing` → `play`, `##ing`). A fixed set of special tokens is reserved: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`.

**Encoding** uses longest-match-first per word; if the first character of a word is not in the vocabulary the whole word is replaced with `[UNK]`.

### 4.4 SentencePiece (LLaMA / Mistral style)

Rather than treating word boundaries as fixed, SentencePiece encodes the space character as part of the first subword of each word using the Unicode symbol `▁` (U+2581):

$$\text{"hello world"} \;\longrightarrow\; [\texttt{▁hello},\; \texttt{▁world}]$$

This makes the tokeniser **language-agnostic** — it operates on the raw byte stream without any language-specific whitespace rules. BPE merges are then learned on this `▁`-normalised corpus. Special tokens are `<unk>`, `<s>`, `</s>`.

### 4.5 Unigram Language Model Tokeniser (T5 / ALBERT style)

The Unigram tokeniser (Kudo, 2018) maintains an explicit probability distribution over subword tokens and finds the **maximum-probability segmentation** of each word using the Viterbi algorithm.

**Training algorithm:**

1. **Seed vocabulary**: Collect all substrings of length $\leq 16$ from the `▁`-normalised corpus. Keep the top $10V$ by frequency (where $V$ is the target vocabulary size).

2. **EM iterations** (repeated until target size is reached):
   - **E-step**: For each word $w_i$ with frequency $f_i$, find the Viterbi segmentation:
$$\mathbf{s}_i^* = \operatorname*{arg\,max}_{\mathbf{s} \in \mathcal{S}(w_i)} \sum_{t \in \mathbf{s}} \log p(t)$$
   - **M-step**: Re-estimate token log-probabilities from expected counts:
$$\log p(t) \propto \sum_i f_i \cdot \mathbf{1}[t \in \mathbf{s}_i^*]$$

3. **Pruning**: Remove the 20% of tokens whose removal causes the least increase in total corpus loss:
$$\Delta \mathcal{L}(t) = \text{count}(t) \cdot \bigl(\log p(t) - \log p_{\text{fallback}}\bigr)$$
   Tokens with smallest $\Delta \mathcal{L}$ are removed. Single-character tokens and `▁`-prefixed tokens are never pruned.

4. Repeat from step 2 until $|\text{vocab}| = V$.

---

## 5. Embeddings and Positional Encoding

### Token Embeddings

A learnable lookup table $E \in \mathbb{R}^{V \times d}$ maps each token ID to a $d$-dimensional vector:

$$\mathbf{e}_t = E[\text{tokenId}_t,\; :]$$

**Backward pass** (scatter-add): gradients are accumulated at the row corresponding to each token ID. Because the same token can appear at multiple positions, gradients sum:

$$\frac{\partial \mathcal{L}}{\partial E[\text{id},\;:]} \mathrel{+}= \frac{\partial \mathcal{L}}{\partial \mathbf{e}_t} \quad \text{for each } t \text{ where } \text{tokenId}_t = \text{id}$$

### Sinusoidal Positional Encoding

Position information is injected via **fixed** (non-learned) sinusoidal signals (Vaswani et al., 2017). For position $p \in \{0, \ldots, T-1\}$ and dimension index $i \in \{0, \ldots, d-1\}$:

$$\text{PE}[p,\, 2i]   = \sin\!\left(\frac{p}{10000^{2i/d}}\right)$$

$$\text{PE}[p,\, 2i+1] = \cos\!\left(\frac{p}{10000^{2i/d}}\right)$$

Numerically this is computed as:

$$\text{freq}_i = \exp\!\left(-\frac{2i}{d}\ln 10000\right) \qquad \text{angle} = p \cdot \text{freq}_i$$

The combined embedding input to the first transformer block is:

$$x^{(0)}_t = E[\text{tokenId}_t,\;:] + \text{PE}[t,\;:]$$

Since PE is fixed, its gradient is discarded during backpropagation.

---

## 6. Layer Normalisation

Layer normalisation (Ba et al., 2016) normalises each sequence position independently across the feature dimension $d$.

### Forward Pass

For position $t$ with feature vector $\mathbf{x}_t \in \mathbb{R}^d$:

$$\mu_t = \frac{1}{d}\sum_{j=1}^d x_{t,j}$$

$$\sigma_t^2 = \frac{1}{d}\sum_{j=1}^d (x_{t,j} - \mu_t)^2$$

$$\hat{x}_{t,j} = \frac{x_{t,j} - \mu_t}{\sqrt{\sigma_t^2 + \varepsilon}} \qquad \varepsilon = 10^{-5}$$

$$y_{t,j} = \gamma_j \,\hat{x}_{t,j} + \beta_j$$

$\gamma \in \mathbb{R}^d$ (scale, initialised to 1) and $\beta \in \mathbb{R}^d$ (shift, initialised to 0) are learnable parameters.

### Backward Pass

Let $\mathbf{g}_t = \partial \mathcal{L} / \partial \mathbf{y}_t$. Define:

$$g_{t,j}' = g_{t,j} \cdot \gamma_j \qquad \bar{g}_t = \frac{1}{d}\sum_j g_{t,j}' \qquad \overline{g\hat{x}}_t = \frac{1}{d}\sum_j g_{t,j}' \hat{x}_{t,j}$$

Then:

$$\frac{\partial \mathcal{L}}{\partial x_{t,j}} = \frac{1}{\sigma_t}\left(g_{t,j}' - \bar{g}_t - \hat{x}_{t,j}\,\overline{g\hat{x}}_t\right)$$

$$\frac{\partial \mathcal{L}}{\partial \gamma_j} = \sum_t g_{t,j}\,\hat{x}_{t,j} \qquad \frac{\partial \mathcal{L}}{\partial \beta_j} = \sum_t g_{t,j}$$

---

## 7. Multi-Head Causal Self-Attention

Multi-head attention (Vaswani et al., 2017) allows each position to attend to all earlier positions in the sequence simultaneously across multiple learned subspaces.

### Forward Pass

**Step 1 — Linear projections.** Given input $X \in \mathbb{R}^{T \times d}$:

$$Q = X W_Q + b_Q \qquad K = X W_K + b_K \qquad V = X W_V + b_V$$

where $W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}$ and $b_Q, b_K, b_V, b_O \in \mathbb{R}^{1 \times d}$.

**Step 2 — Per-head computation.** Split $Q$, $K$, $V$ into $H$ heads of width $d_k = d/H$:

$$Q_h = Q[\,:,\; h\,d_k : (h+1)\,d_k] \in \mathbb{R}^{T \times d_k}$$

Scaled dot-product attention scores for head $h$:

$$S_h = \frac{Q_h K_h^\top}{\sqrt{d_k}} + M \in \mathbb{R}^{T \times T}$$

where $M$ is the **causal mask**:

$$M[i,j] = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}$$

Setting future positions to $-\infty$ ensures $\exp(-\infty) = 0$ after softmax, preventing the model from attending to tokens it has not yet generated.

Attention weights and output for head $h$:

$$A_h = \text{softmax}(S_h) \in \mathbb{R}^{T \times T}$$

$$\text{out}_h = A_h V_h \in \mathbb{R}^{T \times d_k}$$

**Step 3 — Concatenate and project.**

$$\text{concat} = [\text{out}_0 \mid \text{out}_1 \mid \cdots \mid \text{out}_{H-1}] \in \mathbb{R}^{T \times d}$$

$$\text{result} = \text{concat}\, W_O + b_O \in \mathbb{R}^{T \times d}$$

### Backward Pass

Gradients flow in reverse order through each step:

**Output projection:**

$$\frac{\partial \mathcal{L}}{\partial W_O} = \text{concat}^\top \cdot \delta_\text{result} \qquad \frac{\partial \mathcal{L}}{\partial b_O} = \sum_t \delta_\text{result}[t,:]$$

$$\frac{\partial \mathcal{L}}{\partial \text{concat}} = \delta_\text{result} \cdot W_O^\top$$

**Per-head attention (for each $h$):**

$$\frac{\partial \mathcal{L}}{\partial A_h} = \delta_{\text{out}_h} \cdot V_h^\top \qquad \frac{\partial \mathcal{L}}{\partial V_h} = A_h^\top \cdot \delta_{\text{out}_h}$$

$$\frac{\partial \mathcal{L}}{\partial S_h} = \text{softmax\_backward}(A_h,\; \tfrac{\partial \mathcal{L}}{\partial A_h})$$

$$\frac{\partial \mathcal{L}}{\partial Q_h} = \frac{\delta_{S_h} \cdot K_h}{\sqrt{d_k}} \qquad \frac{\partial \mathcal{L}}{\partial K_h} = \frac{\delta_{S_h}^\top \cdot Q_h}{\sqrt{d_k}}$$

**Input projections:**

$$\frac{\partial \mathcal{L}}{\partial W_Q} = X^\top \cdot \delta_Q \qquad \frac{\partial \mathcal{L}}{\partial X} = \delta_Q W_Q^\top + \delta_K W_K^\top + \delta_V W_V^\top$$

(and analogously for $K$, $V$).

---

## 8. Feed-Forward Network

Each transformer block contains a position-wise two-layer MLP with a GELU non-linearity. It expands the representation to a larger hidden dimension and then projects back.

### Forward Pass

$$H = X W_1 + b_1 \in \mathbb{R}^{T \times d_{ff}}$$

$$H' = \text{GELU}(H)$$

$$Y = H' W_2 + b_2 \in \mathbb{R}^{T \times d}$$

where $W_1 \in \mathbb{R}^{d \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d}$, and typically $d_{ff} = 4d$.

### Backward Pass

$$\frac{\partial \mathcal{L}}{\partial W_2} = H'^\top \cdot \delta_Y \qquad \frac{\partial \mathcal{L}}{\partial b_2} = \sum_t \delta_Y[t,:]$$

$$\delta_{H'} = \delta_Y \cdot W_2^\top$$

$$\delta_H = \delta_{H'} \odot \text{GELU}'(H) \qquad \text{(element-wise)}$$

$$\frac{\partial \mathcal{L}}{\partial W_1} = X^\top \cdot \delta_H \qquad \frac{\partial \mathcal{L}}{\partial b_1} = \sum_t \delta_H[t,:] \qquad \frac{\partial \mathcal{L}}{\partial X} = \delta_H \cdot W_1^\top$$

---

## 9. Transformer Block

Each block uses **pre-norm** architecture: layer normalisation is applied to the input *before* each sub-layer, not after. This produces cleaner gradient flow through the residual connections and makes training more stable.

### Forward Pass

$$x_1 = \text{LN}_1(x)$$

$$x' = x + \text{Attention}(x_1) \qquad \text{(first residual connection)}$$

$$x_2 = \text{LN}_2(x')$$

$$y = x' + \text{FFN}(x_2) \qquad \text{(second residual connection)}$$

The residual connections allow gradients to flow directly from the output to the input without passing through any non-linearity. This is the key mechanism that makes deep networks trainable.

### Backward Pass

Gradients are propagated in reverse order. Because of the residual connections, the gradient of the loss with respect to each residual stream is the sum of gradients from both branches:

$$\delta_{x'} = \delta_y + \text{FFN.Backward}(\text{LN}_2.\text{Backward}(\delta_y))$$

$$\delta_x = \delta_{x'} + \text{Attention.Backward}(\text{LN}_1.\text{Backward}(\delta_{x'}))$$

---

## 10. Full Transformer Model

The complete model stacks $L$ transformer blocks between an embedding layer and a final projection:

$$x^{(0)} = \text{Embed}(\text{tokenIds})$$

$$x^{(\ell)} = \text{Block}_\ell(x^{(\ell-1)}) \qquad \ell = 1, \ldots, L$$

$$\hat{x} = \text{LN}_\text{final}(x^{(L)})$$

$$\text{logits} = \hat{x}\, W_\text{proj} + b_\text{proj} \in \mathbb{R}^{T \times V}$$

### Parameter Count

For the default configuration ($d=128$, $H=4$, $L=4$, $d_{ff}=512$, $V \approx 500$ from the Unigram tokeniser):

| Component | Formula | Count |
|-----------|---------|-------|
| Token embedding table $E$ | $V \times d$ | 64 000 |
| Per block: attention projections | $4d^2 + 4d$ | 66 048 |
| Per block: FFN weights + biases | $2\,d\,d_{ff} + d_{ff} + d$ | 131 712 |
| Per block: LayerNorm $\gamma$, $\beta$ (×2) | $4d$ | 512 |
| **Per block total** | | **~198 272** |
| **All $L=4$ blocks** | | **~793 088** |
| Final layer norm $\gamma$, $\beta$ | $2d$ | 256 |
| Output projection $W_\text{proj}$, $b_\text{proj}$ | $d \times V + V$ | 64 500 |
| **Grand total** | | **~921 844** |

---

## 11. Training Loop

### Random Chunk Sampling

Rather than scanning the token array sequentially, each training step draws a random starting position:

$$\text{offset} \sim \mathcal{U}(0,\; N - T - 1) \qquad N = |\text{allTokens}|$$

$$\text{input} = \text{allTokens}[\text{offset} : \text{offset}+T] \qquad \text{target} = \text{allTokens}[\text{offset}+1 : \text{offset}+T+1]$$

The number of steps per epoch is fixed at $\lfloor N / T \rfloor$, so the total token budget per epoch is constant. Random sampling prevents the model from overfitting the fixed sequential order of chunks and improves convergence on large corpora trained for many epochs.

---

## 12. Cross-Entropy Loss

The language model is trained by **next-token prediction**: at each position $t$, the model must assign high probability to the token that actually follows in the training corpus.

### Forward Pass

For each position $t$ with target token $y_t$:

$$m_t = \max_j \text{logits}[t, j]$$

$$p_{t,j} = \frac{\exp(\text{logits}[t,j] - m_t)}{\sum_k \exp(\text{logits}[t,k] - m_t)}$$

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^T \log\!\max(p_{t,\, y_t},\; 10^{-8})$$

Subtracting $m_t$ before exponentiating is numerically equivalent to the standard softmax but avoids overflow. The clamp at $10^{-8}$ prevents $\log(0)$.

### Backward Pass

The gradient of the cross-entropy loss with respect to the pre-softmax logits has a remarkably clean form:

$$\frac{\partial \mathcal{L}}{\partial \text{logits}[t,j]} = \frac{p_{t,j} - \mathbf{1}[j = y_t]}{T}$$

That is, the gradient is the predicted probability minus a one-hot vector pointing at the correct token, normalised by sequence length. This is the key result that makes cross-entropy + softmax efficient to train.

---

## 13. Adam Optimiser

Adam (Kingma and Ba, 2014) maintains a per-parameter exponential moving average of both gradients (first moment $m$) and squared gradients (second moment $v$), giving an adaptive per-parameter learning rate.

### Update Equations

At step $t$, for each scalar parameter $\theta$ with gradient $g$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \qquad \text{(first moment)}$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \qquad \text{(second moment)}$$

Because $m$ and $v$ are initialised to zero, they are biased toward zero in early steps. **Bias correction** compensates:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

The parameter update is:

$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

Intuitively, $\hat{m}_t$ is a momentum-smoothed gradient direction and $\sqrt{\hat{v}_t}$ is an estimate of the gradient magnitude, so the ratio $\hat{m}_t / \sqrt{\hat{v}_t}$ is a normalised step that is approximately scale-invariant. Adam state ($m$, $v$) is stored inside each `Parameter` object alongside the weight and gradient.

---

## 14. Gradient Clipping

Without clipping, occasional large gradient values (e.g. from sharp attention patterns) can destabilise training. Global L2 norm clipping rescales all gradients uniformly when the total norm exceeds a threshold:

$$\text{norm} = \sqrt{\sum_{\theta}\sum_{i,j} \left(\frac{\partial \mathcal{L}}{\partial \theta_{ij}}\right)^2}$$

$$\text{If } \text{norm} > \tau\text{:} \qquad \frac{\partial \mathcal{L}}{\partial \theta_{ij}} \leftarrow \frac{\partial \mathcal{L}}{\partial \theta_{ij}} \cdot \frac{\tau}{\text{norm}}$$

This preserves the **direction** of the gradient vector while bounding its magnitude to $\tau$ (default 1.0). Clipping is applied once over all parameters collectively before the Adam update.

---

## 15. Text Generation

After training, the model generates text autoregressively: it produces one token at a time, appending each output to the context before the next forward pass.

### Algorithm

```
context = tokeniser.Encode(prompt)
for step = 1 to numTokens:
    input = context[-ContextLength:]          // trim to max context
    logits = model.Forward(input)             // shape [T × V]
    z = logits[T-1, :]                        // last position only
    z /= temperature                          // temperature scaling
    apply top-k mask to z                     // zero non-top-k entries
    p = softmax(z)                            // probabilities
    next = categorical_sample(p)              // sample one token
    context.append(next)
```

### Temperature Scaling

Dividing the logits by a temperature $\tau > 0$ before softmax controls the sharpness of the distribution:

$$p_j = \frac{\exp(z_j / \tau)}{\sum_k \exp(z_k / \tau)}$$

- $\tau < 1$: distribution sharpens (model more confident, less diverse)
- $\tau = 1$: unmodified distribution
- $\tau > 1$: distribution flattens (more uniform, more random)

### Top-$k$ Sampling

Only the $k$ most probable tokens are candidates for sampling. All other logits are set to $-\infty$ before the softmax, making their probability exactly zero:

$$z_j \leftarrow \begin{cases} z_j & z_j \geq z_{(k)} \\ -\infty & \text{otherwise} \end{cases}$$

where $z_{(k)}$ is the $k$-th largest logit.

### Categorical Sampling (Inverse CDF)

To draw a token from the resulting distribution:

1. Compute the cumulative sum $F_j = \sum_{i \leq j} p_i$.
2. Draw $u \sim \mathcal{U}(0, 1)$.
3. Return the smallest $j$ such that $F_j \geq u$.

This is equivalent to sampling from $\text{Categorical}(\mathbf{p})$.

---

## 16. Model Save / Load

Trained weights can be persisted to disk and reloaded, allowing training to be resumed or a pre-trained model to be used purely for inference.

### Binary File Format

`ModelSerializer` (in `LLM_Common`) writes and reads a simple binary format:

| Field | Type | Description |
|-------|------|-------------|
| Magic number | `uint32` | `0x4C4C4D01` — validates file type |
| Version | `int32` | Format version (currently 1) |
| Config fields | 6 × `int32` | VocabSize, EmbeddingDim, NumHeads, NumLayers, FFNDim, ContextLength |
| Parameter count | `int32` | Number of parameter tensors |
| Per parameter | `int32`, `int32`, `float[]` | rows, cols, then `rows × cols` floats |

On **save**, each parameter's `GetWeightsFlat()` method downloads the weight matrix to a flat `float[]` which is written sequentially.

On **load**, the config fields are validated against the current `TransformerConfig` — a mismatch raises an exception rather than silently loading corrupted weights. Each parameter's `LoadWeightsFlat()` method uploads the data back to the weight matrix (CPU memory or GPU device memory depending on the backend).

**Note:** Adam optimiser state ($m$, $v$ moments) is not saved. After loading, Adam restarts from cold, which may cause a brief loss spike before the momentum estimates recover.

---

## 17. GPU Backend

The GPU implementation (in `LLM_GPU`) uses **ILGPU**, an open-source .NET GPU framework that JIT-compiles C# kernels to CUDA, OpenCL, or CPU code at runtime.

The mathematical operations are **identical** to the CPU implementation. The difference is that matrices are stored in device memory and the inner loops of each operation run in parallel across thousands of GPU cores via custom kernels defined in `Kernels.cs`.

Key GPU components:

| Class | Purpose |
|-------|---------|
| `GpuContext` | Manages the ILGPU accelerator lifetime |
| `GpuMatrix` | Matrix with GPU-resident float buffer |
| `GpuParameter` | Weight + gradient + Adam state on device |
| `Kernels` | ILGPU kernel definitions for each math op |
| `GpuTransformerModel` | Implements `ITransformerModel` using GPU ops |

The `ITransformerModel` interface means the training loop and generation code in `LLM_App/Program.cs` are entirely unaware of whether a CPU or GPU backend is in use.

---

## References

- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
- Ba, J. L. et al. (2016). *Layer Normalization*. arXiv:1607.06450.
- Kingma, D. P. and Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. arXiv:1412.6980.
- Sennrich, R. et al. (2016). *Neural Machine Translation of Rare Words with Subword Units*. ACL.
- Schuster, M. and Nakamura, K. (2012). *Japanese and Korean Voice Search*. ICASSP. (WordPiece)
- Kudo, T. (2018). *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*. ACL. (Unigram LM)
- Kudo, T. and Richardson, J. (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer*. EMNLP.
- Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI. (GPT-2, GELU approx.)
