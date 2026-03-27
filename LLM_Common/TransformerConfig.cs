namespace LLM
{
    /// <summary>
    /// All hyper-parameters that define the transformer architecture and training
    /// procedure, collected in one place for easy experimentation.
    ///
    /// The defaults below describe a small "GPT-nano" that trains in minutes on
    /// a CPU with a modest text corpus.  Scale the values up for a larger model.
    ///
    /// Naming conventions follow the GPT-2 paper (Radford et al., 2019):
    ///   d_model  = embedding / residual stream dimension
    ///   d_head   = d_model / NumHeads   (per-head attention dimension)
    ///   d_ff     = feed-forward hidden dimension (typically 4 × d_model)
    /// </summary>
    public sealed class TransformerConfig
    {
        // ── tokeniser ────────────────────────────────────────────────────────────

        /// <summary>
        /// Size of the vocabulary (number of distinct tokens).
        /// Set automatically by the Tokenizer after scanning the training text;
        /// for character-level models this is typically 50–200.
        /// </summary>
        public int VocabSize { get; set; } = 128;

        // ── model dimensions ──────────────────────────────────────────────────────

        /// <summary>
        /// Dimension of the embedding / residual stream (d_model).
        /// Every token is represented as a vector of this size throughout the network.
        /// Typical values: 64 (tiny), 256 (small), 768 (GPT-2 base).
        /// </summary>
        public int EmbeddingDim { get; set; } = 128;

        /// <summary>
        /// Number of self-attention heads per transformer block.
        /// Must evenly divide EmbeddingDim so that d_head = EmbeddingDim / NumHeads
        /// is an integer.
        /// Multiple heads let the model attend to different representation subspaces
        /// simultaneously (multi-head attention, Vaswani et al., 2017).
        /// </summary>
        public int NumHeads { get; set; } = 4;

        /// <summary>
        /// Number of stacked transformer blocks (depth of the network).
        /// Each block is one layer of multi-head attention + feed-forward + norms.
        /// Typical values: 2 (tiny), 6 (small), 12 (GPT-2 base).
        /// </summary>
        public int NumLayers { get; set; } = 4;

        /// <summary>
        /// Hidden dimension of the position-wise feed-forward network inside each
        /// transformer block.  Usually 4 × EmbeddingDim (the "expansion factor").
        /// </summary>
        public int FFNDim { get; set; } = 512;

        /// <summary>
        /// Maximum sequence length the model can process (context window).
        /// Determines the size of the positional embedding table and the shape
        /// of the causal attention mask.
        /// </summary>
        public int ContextLength { get; set; } = 128;

        // ── training ──────────────────────────────────────────────────────────────

        /// <summary>
        /// Peak learning rate for the Adam optimiser (reached after warmup).
        /// A good starting point for small models is 3e-4; reduce for larger models.
        /// </summary>
        public float LearningRate { get; set; } = 3e-4f;

        /// <summary>
        /// Number of Adam steps over which the learning rate linearly ramps from 0
        /// to <see cref="LearningRate"/> (warmup phase).  After warmup the LR follows
        /// a cosine decay down to <see cref="MinLearningRate"/>.
        /// Set to 0 to disable warmup (LR starts at peak immediately).
        /// </summary>
        public int WarmupSteps { get; set; } = 100;

        /// <summary>
        /// Floor for the cosine-decay schedule.  The learning rate never drops below
        /// this value.  Typically ~10% of <see cref="LearningRate"/>.
        /// </summary>
        public float MinLearningRate { get; set; } = 1e-5f;

        /// <summary>
        /// Use Rotary Positional Encoding (RoPE) instead of sinusoidal additive PE.
        /// RoPE encodes position in the rotation of Q/K vectors inside each head,
        /// giving better length-generalisation and relative-position sensitivity.
        /// When true, <see cref="HeadDim"/> must be even and the sinusoidal table in
        /// the Embedding layer is not added.
        /// </summary>
        public bool UseRoPE { get; set; } = true;

        /// <summary>
        /// Adam β₁ – decay rate for the first moment (gradient momentum).
        /// Controls how quickly old gradient estimates are forgotten.
        /// </summary>
        public float Beta1 { get; set; } = 0.9f;

        /// <summary>
        /// Adam β₂ – decay rate for the second moment (adaptive scaling).
        /// Should be close to 1; 0.999 is standard.
        /// </summary>
        public float Beta2 { get; set; } = 0.999f;

        /// <summary>
        /// Adam ε – small constant added to the denominator to prevent
        /// division by zero when second moment estimates are near zero.
        /// </summary>
        public float AdamEps { get; set; } = 1e-8f;

        /// <summary>
        /// Maximum gradient norm for gradient clipping.
        /// If the global L2 norm of all gradients exceeds this value, they are all
        /// scaled down to prevent exploding gradients.  A value of 1.0 is standard.
        /// </summary>
        public float GradClip { get; set; } = 1.0f;

        /// <summary>
        /// Seed for weight initialisation RNG. -1 = random seed each run.
        /// </summary>
        public int Seed { get; set; } = 42;

        /// <summary>
        /// Number of full passes over the training data.
        /// Increase for better convergence; watch for overfitting on small corpora.
        /// </summary>
        public int Epochs { get; set; } = 10;

        /// <summary>
        /// Number of forward/backward passes whose gradients are summed before
        /// one Adam update.  Effective batch size = AccumulationSteps × ContextLength tokens.
        /// 1 = standard single-step update (default).
        /// </summary>
        public int AccumulationSteps { get; set; } = 1;

        /// <summary>
        /// Print a generated text sample every this many epochs during training.
        /// Set to 0 to disable sampling entirely.
        /// </summary>
        public int SampleEvery { get; set; } = 5;

        /// <summary>
        /// Prompt used to seed the training sample. Change this to match your corpus.
        /// </summary>
        public string SamplePrompt { get; set; } = "Shall ";

        // ── derived properties ───────────────────────────────────────────────────

        /// <summary>
        /// Per-head attention dimension: d_head = EmbeddingDim / NumHeads.
        /// Each head operates in a lower-dimensional subspace and the results are
        /// concatenated at the output.
        /// </summary>
        public int HeadDim => EmbeddingDim / NumHeads;

        /// <summary>
        /// Scaling factor applied to raw attention scores before softmax:
        ///   scale = 1 / √(HeadDim)
        ///
        /// Without this scaling the dot products grow with d_head, pushing
        /// the softmax into regions of very small gradients ("saturation").
        /// Introduced in Vaswani et al. (2017), "Attention Is All You Need."
        /// </summary>
        public float AttentionScale => 1f / System.MathF.Sqrt(HeadDim);

        /// <summary>
        /// Compute the learning rate for a given Adam step using linear warmup
        /// followed by cosine decay.
        ///
        ///   • Steps 1 … WarmupSteps : LR ramps linearly from 0 → LearningRate.
        ///   • Steps WarmupSteps+1 … totalSteps : cosine decay LearningRate → MinLearningRate.
        ///
        /// <paramref name="step"/> is 1-indexed (same convention as the Adam bias-correction step).
        /// </summary>
        public float ComputeLR(int step, int totalSteps)
        {
            if (step <= WarmupSteps)
                return LearningRate * step / System.Math.Max(1, WarmupSteps);

            float progress = (float)(step - WarmupSteps)
                           / System.Math.Max(1, totalSteps - WarmupSteps);
            float cosine   = 0.5f * (1f + System.MathF.Cos(System.MathF.PI * System.Math.Min(progress, 1f)));
            return MinLearningRate + (LearningRate - MinLearningRate) * cosine;
        }

        /// <summary>Validate that the configuration is self-consistent.</summary>
        public void Validate()
        {
            if (EmbeddingDim % NumHeads != 0)
                throw new System.InvalidOperationException(
                    $"EmbeddingDim ({EmbeddingDim}) must be divisible by NumHeads ({NumHeads}).");
            if (ContextLength <= 0)
                throw new System.InvalidOperationException("ContextLength must be positive.");
            if (VocabSize <= 0)
                throw new System.InvalidOperationException("VocabSize must be positive.");
            if (UseRoPE && HeadDim % 2 != 0)
                throw new System.InvalidOperationException(
                    $"HeadDim ({HeadDim}) must be even when UseRoPE is true.");
        }

        public override string ToString() =>
            $"TransformerConfig: vocab={VocabSize}, d={EmbeddingDim}, heads={NumHeads}, " +
            $"layers={NumLayers}, ffn={FFNDim}, ctx={ContextLength}";
    }
}
