using System;
using System.Collections.Generic;

namespace LLM
{
    /// <summary>
    /// Common interface for a transformer layer that transforms a matrix of
    /// activations and can propagate gradients back through itself.
    ///
    /// The type parameter <typeparamref name="TMatrix"/> is the matrix
    /// representation used by the backend:
    ///   <see cref="LLM.Matrix"/>     – CPU backend
    ///   <see cref="LLM_GPU.GpuMatrix"/> – GPU backend
    ///
    /// Implemented by (CPU): LayerNorm, MultiHeadAttention, FeedForward, TransformerBlock
    /// Implemented by (GPU): GpuLayerNorm, GpuMultiHeadAttention, GpuFeedForward, GpuTransformerBlock
    /// </summary>
    public interface ILayer<TMatrix> : IDisposable
    {
        /// <summary>Run the forward pass; returns the output activation matrix.</summary>
        TMatrix Forward(TMatrix x);

        /// <summary>
        /// Run the backward pass given the upstream gradient.
        /// Returns the gradient with respect to the layer input.
        /// </summary>
        TMatrix Backward(TMatrix dOut);

        /// <summary>Enumerate all learnable parameters in this layer.</summary>
        IEnumerable<IParameter> Parameters();
    }
}
