using System;

namespace LLM
{
    /// <summary>
    /// Common interface for CPU and GPU transformer model implementations.
    /// Allows the training and generation loop in Program.cs to work
    /// identically against either backend.
    ///
    /// Provided implementations:
    ///   <see cref="LLM.TransformerModel"/>           – CPU (managed float arrays)
    ///   <see cref="LLM_GPU.GpuTransformerModel"/>    – GPU (ILGPU kernels)
    /// </summary>
    public interface ITransformerModel : IDisposable
    {
        /// <summary>
        /// Run one complete training step: zero gradients → forward pass →
        /// cross-entropy loss and gradient → backward pass → gradient clip →
        /// Adam update.  Equivalent to AccumulationSteps = 1.
        /// </summary>
        /// <param name="input">Input token IDs, length T (= ContextLength).</param>
        /// <param name="targets">Target (next-token) IDs, length T.</param>
        /// <param name="adamStep">1-indexed Adam step for bias correction.</param>
        /// <returns>Average cross-entropy loss per token.</returns>
        float TrainStep(int[] input, int[] targets, int adamStep);

        /// <summary>
        /// Forward pass → cross-entropy → backward pass only.
        /// Gradients are accumulated into parameters but the optimiser is NOT stepped.
        /// Call ZeroAllGradients() before the first accumulation step,
        /// then ClipAndUpdate() after the last one.
        /// </summary>
        float AccumulateStep(int[] input, int[] targets);

        /// <summary>Zero all parameter gradients.</summary>
        void ZeroAllGradients();

        /// <summary>Multiply every gradient element by <paramref name="scale"/>.</summary>
        void ScaleAllGradients(float scale);

        /// <summary>Clip gradients by global L2 norm, then run one Adam update using Config.LearningRate.</summary>
        void ClipAndUpdate(int adamStep);

        /// <summary>
        /// Clip gradients by global L2 norm, then run one Adam update with an
        /// externally computed learning rate (e.g. from a warmup + cosine schedule).
        /// </summary>
        void ClipAndUpdate(int adamStep, float lr);

        /// <summary>
        /// Clear all KV caches (used by attention layers during cached inference).
        /// Call before starting a new generation sequence.
        /// No-op on backends that do not implement KV-Cache (e.g. GPU).
        /// </summary>
        void ClearKVCache();

        /// <summary>
        /// Generate new tokens autoregressively from a prompt.
        /// </summary>
        /// <param name="promptIds">Encoded prompt token IDs.</param>
        /// <param name="numTokens">Number of tokens to generate.</param>
        /// <param name="temperature">Sampling temperature (1.0 = unmodified).</param>
        /// <param name="topK">Top-K filtering (0 = disabled).</param>
        int[] Generate(int[] promptIds, int numTokens, float temperature, int topK);

        /// <summary>
        /// Save all trained weights to a binary file.
        /// The file stores a magic number, the model configuration, and every
        /// parameter's weight matrix in the canonical AllParameters() order.
        /// </summary>
        void Save(string path);

        /// <summary>
        /// Load weights from a file previously written by <see cref="Save"/>.
        /// Throws <see cref="InvalidDataException"/> if the file's configuration
        /// does not match this model's configuration.
        /// </summary>
        void Load(string path);

        /// <summary>
        /// Save a full training checkpoint: weights, Adam state, epoch, and step.
        /// Use this during training to enable crash recovery.
        /// </summary>
        void SaveCheckpoint(string path, int epoch, int adamStep, int innerStep);

        /// <summary>
        /// Load a training checkpoint previously written by <see cref="SaveCheckpoint"/>.
        /// Restores weights and Adam state.
        /// </summary>
        /// <returns>
        /// (epoch, adamStep, innerStep) — the epoch and Adam step at save time, and the
        /// inner-loop step to resume from. innerStep = -1 for v1 checkpoints (no inner step stored).
        /// </returns>
        (int epoch, int adamStep, int innerStep) LoadCheckpoint(string path);

        /// <summary>Total number of trainable scalar parameters.</summary>
        long ParameterCount { get; }

        /// <summary>
        /// Forward pass only — no backward pass, no weight updates.
        /// Returns the average cross-entropy loss per token for the given chunk.
        /// Used for validation.
        /// </summary>
        float Evaluate(int[] input, int[] targets);
    }
}
