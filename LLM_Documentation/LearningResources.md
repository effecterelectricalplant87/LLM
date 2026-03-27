# Learning Resources

A curated reading path from absolute beginner to the mathematics and engineering behind this codebase.
Resources are organised by topic and labelled with an approximate level:

| Label | Meaning |
|-------|---------|
| 🟢 Beginner | No prior maths or ML knowledge assumed |
| 🟡 Intermediate | Comfortable with basic calculus and linear algebra |
| 🔴 Advanced | Assumes ML background; primary-source papers |

Work roughly top-to-bottom within each section before moving to the next.

---

## Table of Contents

1. [Mathematical Foundations](#1-mathematical-foundations)
2. [Programming and Numerical Computing](#2-programming-and-numerical-computing)
3. [Neural Networks from Scratch](#3-neural-networks-from-scratch)
4. [Deep Learning — Courses and Books](#4-deep-learning--courses-and-books)
5. [Natural Language Processing](#5-natural-language-processing)
6. [The Transformer Architecture](#6-the-transformer-architecture)
7. [Language Model Pre-training](#7-language-model-pre-training)
8. [Tokenisation](#8-tokenisation)
9. [Optimisation and Training Tricks](#9-optimisation-and-training-tricks)
10. [Scaling Laws and Modern LLMs](#10-scaling-laws-and-modern-llms)
11. [Primary Papers Behind This Codebase](#11-primary-papers-behind-this-codebase)
12. [Going Deeper — What to Read Next](#12-going-deeper--what-to-read-next)

---

## 1. Mathematical Foundations

You do not need to master all of this before starting, but these topics underpin everything.
Return to them as needed when you encounter unfamiliar notation.

### Linear Algebra

🟢 **3Blue1Brown — Essence of Linear Algebra** (YouTube playlist)
The clearest visual introduction to vectors, matrices, dot products, and transforms.
Highly recommended as a first stop — each episode is 10–15 minutes.
[https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

🟢 **Khan Academy — Linear Algebra**
Step-by-step interactive lessons from vectors through matrix multiplication and determinants.
[https://www.khanacademy.org/math/linear-algebra](https://www.khanacademy.org/math/linear-algebra)

🟡 **Gilbert Strang — Linear Algebra (MIT OpenCourseWare)**
The standard undergraduate lecture series. Free video lectures and problem sets.
[https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)

### Calculus and Backpropagation

🟢 **3Blue1Brown — Essence of Calculus** (YouTube playlist)
Builds intuition for derivatives and the chain rule from first principles — visually.
[https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

🟢 **Khan Academy — Multivariable Calculus**
Partial derivatives and the chain rule are the core machinery of backpropagation.
[https://www.khanacademy.org/math/multivariable-calculus](https://www.khanacademy.org/math/multivariable-calculus)

### Probability and Statistics

🟢 **Khan Academy — Statistics and Probability**
Covers probability distributions, expectation, and the concepts behind cross-entropy loss.
[https://www.khanacademy.org/math/statistics-probability](https://www.khanacademy.org/math/statistics-probability)

---

## 2. Programming and Numerical Computing

### Python and NumPy (if approaching from a Python angle)

🟢 **Python for Everybody** — Dr. Chuck Severance
Complete beginner Python course. Free website with exercises and videos.
[https://www.py4e.com/](https://www.py4e.com/)

🟢 **NumPy Quickstart Tutorial**
NumPy arrays are the Python equivalent of the `Matrix` class in this codebase.
[https://numpy.org/doc/stable/user/quickstart.html](https://numpy.org/doc/stable/user/quickstart.html)

### C# and .NET (this codebase)

🟢 **Microsoft Learn — Tour of C#**
Free interactive tutorials from Microsoft.
[https://learn.microsoft.com/en-us/dotnet/csharp/tour-of-csharp/](https://learn.microsoft.com/en-us/dotnet/csharp/tour-of-csharp/)

---

## 3. Neural Networks from Scratch

These resources build the same things this codebase implements, step by step,
without relying on ML frameworks. They are the most direct preparation for reading this code.

🟢 **3Blue1Brown — Neural Networks** (YouTube playlist, 4 core episodes)
The single best visual introduction to what a neural network is, how gradients flow
backwards, and what "learning" means mathematically.
[https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

🟢 **Andrej Karpathy — micrograd** (GitHub repo + YouTube walkthrough)
Karpathy builds a scalar-valued autograd engine and a neural network in ~150 lines of Python.
Watching this video is the fastest way to understand backpropagation concretely.
- Repo: [https://github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)
- Video: [https://www.youtube.com/watch?v=VMj-3S1tku0](https://www.youtube.com/watch?v=VMj-3S1tku0)

🟢 **Andrej Karpathy — makemore** (YouTube playlist)
Builds a character-level language model from bigrams up to a multi-layer MLP —
the conceptual precursor to this transformer implementation.
[https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

🟡 **Michael Nielsen — Neural Networks and Deep Learning** (free online book)
A beautifully written, equation-heavy walkthrough of neural networks, backpropagation,
and gradient descent.
[http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/)

---

## 4. Deep Learning — Courses and Books

### Courses

🟢 **fast.ai — Practical Deep Learning for Coders**
Top-down approach: run real models first, understand the theory later.
Excellent for building intuition quickly.
[https://www.fast.ai/](https://www.fast.ai/)

🟡 **deeplearning.ai — Deep Learning Specialisation** (Coursera)
Andrew Ng's five-course series covering neural networks, hyperparameter tuning,
optimisation, and sequence models. Audit for free.
[https://www.deeplearning.ai/courses/deep-learning-specialization/](https://www.deeplearning.ai/courses/deep-learning-specialization/)

🟡 **Stanford CS231n — Notes and Slides** (free)
Although focused on vision, the backpropagation, weight initialisation, and optimisation
content is directly applicable here.
[https://cs231n.github.io/](https://cs231n.github.io/)

### Books

🟢 **Grokking Deep Learning** — Andrew W. Trask
Teaches neural networks using only NumPy, building intuition without framework magic.
Very readable for beginners. (Print / ebook — Manning Publications)
[https://www.manning.com/books/grokking-deep-learning](https://www.manning.com/books/grokking-deep-learning)

🟡 **Deep Learning** — Goodfellow, Bengio, Courville (MIT Press)
The comprehensive graduate-level textbook. Free to read online.
Chapters 6–9 are most relevant: feedforward networks, regularisation, optimisation, sequences.
[https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

🟡 **Dive into Deep Learning** — Zhang, Lipton, Li, Smola
Free online, interactive Jupyter notebooks. Covers transformers in detail with runnable code.
[https://d2l.ai/](https://d2l.ai/)

---

## 5. Natural Language Processing

### Introductory

🟢 **Speech and Language Processing** — Jurafsky & Martin (3rd ed., draft free online)
The standard NLP textbook. Chapters 3 (n-gram language models) and 7–9 (neural
language models) are directly relevant.
[https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)

### Sequence Models (pre-Transformer history)

🟡 **Understanding LSTM Networks** — Christopher Olah (blog post, 2015)
Explains recurrent networks and LSTMs clearly — useful context for *why* transformers
replaced them.
[https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

## 6. The Transformer Architecture

### Essential starting point

🟢 **Andrej Karpathy — Let's build GPT: from scratch, in code, spelled out** (YouTube, 2h)
Karpathy builds a GPT from scratch in Python in real time. This is the most direct tutorial
for understanding exactly what this codebase implements.
[https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)

🟡 **The Illustrated Transformer** — Jay Alammar (blog post)
The clearest visual walkthrough of the transformer's attention mechanism.
Read this before (or alongside) the original paper.
[https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

🟡 **The Annotated Transformer** — Harvard NLP (blog post with code)
The original transformer paper annotated line by line with working PyTorch code.
Bridges the gap between paper notation and implementation.
[https://nlp.seas.harvard.edu/annotated-transformer/](https://nlp.seas.harvard.edu/annotated-transformer/)

🔴 **Attention Is All You Need** — Vaswani et al. (2017)
The paper that introduced the transformer. Section 3 (Model Architecture) is the
part directly implemented here. Read after the illustrated walkthrough.
[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

---

## 7. Language Model Pre-training

🟡 **The Illustrated GPT-2** — Jay Alammar (blog post)
Walks through GPT-2's decoder-only architecture with excellent diagrams.
This is exactly the architecture in this codebase.
[https://jalammar.github.io/illustrated-gpt2/](https://jalammar.github.io/illustrated-gpt2/)

🔴 **Language Models are Unsupervised Multitask Learners** — Radford et al., OpenAI (2019)
GPT-2 paper. Section 2 describes the model architecture used here (pre-norm, GELU).
[https://openai.com/blog/better-language-models](https://openai.com/blog/better-language-models)

🔴 **Language Models are Few-Shot Learners** — Brown et al., OpenAI (2020)
GPT-3 paper. Demonstrates that scaling the GPT-2 architecture massively produces
emergent few-shot capabilities. Establishes the scaling paradigm.
[https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

---

## 8. Tokenisation

🟢 **Andrej Karpathy — Let's build the GPT Tokenizer** (YouTube, 2h)
Builds a BPE tokenizer from scratch with clear explanations of every design decision.
[https://www.youtube.com/watch?v=zduSFxRajkE](https://www.youtube.com/watch?v=zduSFxRajkE)

🔴 **Neural Machine Translation of Rare Words with Subword Units** — Sennrich et al. (2016)
The original BPE tokenization paper.
[https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)

🔴 **SentencePiece: A simple and language independent subword tokenizer** — Kudo & Richardson (2018)
Describes the SentencePiece library and the space-prefix (`▁`) convention used here.
[https://arxiv.org/abs/1808.06226](https://arxiv.org/abs/1808.06226)

🔴 **Subword Regularization: Improving Neural Network Translation Models** — Kudo (2018)
Introduces the Unigram language model tokenizer implemented in this codebase.
[https://arxiv.org/abs/1804.10959](https://arxiv.org/abs/1804.10959)

---

## 9. Optimisation and Training Tricks

### Adam Optimiser

🟡 **An overview of gradient descent optimisation algorithms** — Sebastian Ruder (blog post)
The clearest written tour of SGD, momentum, RMSProp, and Adam. Read before the paper.
[https://www.ruder.io/optimizing-gradient-descent/](https://www.ruder.io/optimizing-gradient-descent/)

🔴 **Adam: A Method for Stochastic Optimization** — Kingma & Ba (2014)
The original Adam paper. Short and readable. Sections 1–3 are sufficient.
[https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)

### Weight Initialisation

🔴 **Understanding the difficulty of training deep feedforward neural networks** — Glorot & Bengio (2010)
Introduces Xavier initialisation — used in this codebase for all projection matrices.
[https://proceedings.mlr.press/v9/glorot10a.html](https://proceedings.mlr.press/v9/glorot10a.html)

### Gradient Clipping

🔴 **On the difficulty of training recurrent neural networks** — Pascanu et al. (2013)
Introduces gradient clipping (Section 5). Explains why gradients explode and how
global-norm clipping (used here) fixes it.
[https://arxiv.org/abs/1211.5063](https://arxiv.org/abs/1211.5063)

### Layer Normalisation

🔴 **Layer Normalization** — Ba et al. (2016)
Introduces LayerNorm. Short paper. Directly implemented in `LayerNorm.cs`.
[https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)

### GELU Activation

🔴 **Gaussian Error Linear Units (GELUs)** — Hendrycks & Gimpel (2016)
Introduces the GELU activation function used in GPT-2 and this codebase.
[https://arxiv.org/abs/1606.08415](https://arxiv.org/abs/1606.08415)

---

## 10. Scaling Laws and Modern LLMs

Once the fundamentals are solid, these resources explain where the field went after GPT-2.

🟡 **Lilian Weng's Blog**
Exceptionally clear survey articles on attention, transformers, LLMs, and alignment.
A reliable go-to reference for staying current.
[https://lilianweng.github.io/](https://lilianweng.github.io/)

🔴 **Scaling Laws for Neural Language Models** — Kaplan et al., OpenAI (2020)
Shows that loss follows a precise power law with model size, dataset size, and compute.
The theoretical basis for why "bigger is better" in LLMs.
[https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)

🔴 **Training Compute-Optimal Large Language Models** — Hoffmann et al., DeepMind (2022)
The "Chinchilla" paper. Shows that most existing LLMs are undertrained relative to their
parameter count — optimal training requires ~20 tokens per parameter.
[https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)

🔴 **LLaMA: Open and Efficient Foundation Language Models** — Touvron et al., Meta (2023)
A modern efficient LLM. Key differences from GPT-2 relevant to this codebase: RoPE
positional encoding, RMSNorm, SwiGLU activation, Grouped Query Attention.
[https://arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971)

---

## 11. Primary Papers Behind This Codebase

These are the exact papers whose algorithms are implemented here, in the order
they are encountered in a forward pass.

| Component | Paper | Link |
|-----------|-------|------|
| Transformer architecture | Vaswani et al. (2017) *Attention Is All You Need* | [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) |
| GELU activation | Hendrycks & Gimpel (2016) *Gaussian Error Linear Units* | [arxiv.org/abs/1606.08415](https://arxiv.org/abs/1606.08415) |
| Layer normalisation | Ba et al. (2016) *Layer Normalization* | [arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450) |
| Pre-norm / GPT-2 architecture | Radford et al. (2019) *Language Models are Unsupervised Multitask Learners* | [openai.com/blog/better-language-models](https://openai.com/blog/better-language-models) |
| Xavier initialisation | Glorot & Bengio (2010) | [proceedings.mlr.press/v9/glorot10a.html](https://proceedings.mlr.press/v9/glorot10a.html) |
| Adam optimiser | Kingma & Ba (2014) | [arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980) |
| Gradient clipping | Pascanu et al. (2013) | [arxiv.org/abs/1211.5063](https://arxiv.org/abs/1211.5063) |
| BPE tokenisation | Sennrich et al. (2016) | [arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909) |
| Unigram LM tokenisation | Kudo (2018) | [arxiv.org/abs/1804.10959](https://arxiv.org/abs/1804.10959) |
| SentencePiece | Kudo & Richardson (2018) | [arxiv.org/abs/1808.06226](https://arxiv.org/abs/1808.06226) |

---

## 12. Going Deeper — What to Read Next

After understanding this codebase, these are the natural next extensions.

### Positional Encoding

🔴 **RoFormer: Enhanced Transformer with Rotary Position Embedding** — Su et al. (2021)
Rotary positional encoding (RoPE) — used in LLaMA, Mistral, GPT-NeoX.
Better length generalisation than sinusoidal PE.
[https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)

### Efficient Attention

🔴 **FlashAttention: Fast and Memory-Efficient Exact Attention** — Dao et al. (2022)
Rewrites attention to be IO-aware, making it 2–4× faster on GPU without changing the output.
[https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)

🔴 **GQA: Training Generalised Multi-Query Transformer Models** — Ainslie et al. (2023)
Grouped Query Attention — shares K and V heads across query heads, dramatically
reducing KV-cache memory at inference time.
[https://arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)

### Activation Functions

🔴 **GLU Variants Improve Transformer** — Noam Shazeer (2020)
Introduces SwiGLU, the activation used in LLaMA and PaLM instead of GELU.
[https://arxiv.org/abs/2002.05202](https://arxiv.org/abs/2002.05202)

### Training at Scale

🔴 **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** — Rajbhandari et al. (2020)
How large models are distributed across many GPUs during training.
[https://arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)

### Alignment and Instruction Following

🔴 **Training language models to follow instructions with human feedback** — Ouyang et al., OpenAI (2022)
InstructGPT / RLHF. The technique that turns a language model into an assistant.
[https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)

🔴 **Direct Preference Optimization** — Rafailov et al. (2023)
A simpler alternative to RLHF that achieves similar alignment results.
[https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
