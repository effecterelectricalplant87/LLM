using System;
using System.Collections.Generic;

namespace LLM
{
    /// <summary>
    /// Common interface for an embedding layer that converts integer token IDs
    /// into dense activation matrices.
    ///
    /// Separate from <see cref="ILayer{TMatrix}"/> because the forward pass takes
    /// token IDs (int[]) rather than an activation matrix, and the backward pass
    /// has no return value (there is no upstream to propagate into past the embedding).
    ///
    /// The type parameter <typeparamref name="TMatrix"/> is the matrix representation:
    ///   <see cref="LLM.Matrix"/>        – CPU backend
    ///   <see cref="LLM_GPU.GpuMatrix"/> – GPU backend
    ///
    /// Implemented by (CPU): Embedding
    /// Implemented by (GPU): GpuEmbedding
    /// </summary>
    public interface IEmbeddingLayer<TMatrix> : IDisposable
    {
        /// <summary>Look up token embeddings and add positional encodings.</summary>
        TMatrix Forward(int[] tokenIds);

        /// <summary>Scatter the upstream gradient back into the embedding table.</summary>
        void Backward(TMatrix grad);

        /// <summary>Enumerate all learnable parameters in this layer.</summary>
        IEnumerable<IParameter> Parameters();
    }
}
