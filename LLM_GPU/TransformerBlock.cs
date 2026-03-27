using System;
using System.Collections.Generic;
using LLM;

namespace LLM_GPU
{
    /// <summary>
    /// GPU transformer block: pre-norm attention + pre-norm FFN, both with residual.
    ///
    ///   x = x + Attention( LayerNorm1(x) )
    ///   x = x + FFN( LayerNorm2(x) )
    /// </summary>
    internal sealed class GpuTransformerBlock : ILayer<GpuMatrix>
    {
        public readonly ILayer<GpuMatrix> Norm1;
        public readonly ILayer<GpuMatrix> Attention;
        public readonly ILayer<GpuMatrix> Norm2;
        public readonly ILayer<GpuMatrix> FFN;

        // Forward-pass cache
        private GpuMatrix? _cachedX;          // block input
        private GpuMatrix? _cachedNormed1;    // LayerNorm1 output
        private GpuMatrix? _cachedAfterAttn;  // x + attnOut

        public GpuTransformerBlock(TransformerConfig cfg, Random rng)
        {
            int D = cfg.EmbeddingDim;
            Norm1     = new GpuLayerNorm(D);
            Attention = new GpuMultiHeadAttention(cfg, rng);
            Norm2     = new GpuLayerNorm(D);
            FFN       = new GpuFeedForward(D, cfg.FFNDim, rng);
        }

        // ── forward pass ──────────────────────────────────────────────────────
        public GpuMatrix Forward(GpuMatrix x)
        {
            _cachedX?.Dispose();
            _cachedX = x;

            // Attention sub-layer
            GpuMatrix normed1  = Norm1.Forward(x);
            _cachedNormed1?.Dispose();
            _cachedNormed1 = normed1;

            GpuMatrix attnOut   = Attention.Forward(normed1);
            GpuMatrix afterAttn = GpuMatrix.Add(x, attnOut);
            attnOut.Dispose();
            _cachedAfterAttn?.Dispose();
            _cachedAfterAttn = afterAttn;

            // FFN sub-layer
            GpuMatrix normed2  = Norm2.Forward(afterAttn);
            GpuMatrix ffnOut   = FFN.Forward(normed2);

            GpuMatrix output = GpuMatrix.Add(afterAttn, ffnOut);
            ffnOut.Dispose();

            return output;
        }

        // ── backward pass ─────────────────────────────────────────────────────
        public GpuMatrix Backward(GpuMatrix dOut)
        {
            if (_cachedX is null || _cachedAfterAttn is null)
                throw new InvalidOperationException("Backward called before Forward.");

            // output = afterAttn + ffnOut  →  both branches get dOut
            GpuMatrix dNormed2   = FFN.Backward(dOut);
            GpuMatrix dAfterAttn = Norm2.Backward(dNormed2);
            dNormed2.Dispose();
            dAfterAttn.AddInPlace(dOut);   // skip-path gradient

            // afterAttn = x + attnOut  →  both branches get dAfterAttn
            GpuMatrix dNormed1   = Attention.Backward(dAfterAttn);
            GpuMatrix dXFromAttn = Norm1.Backward(dNormed1);
            dNormed1.Dispose();

            GpuMatrix dX = GpuMatrix.Add(dAfterAttn, dXFromAttn);
            dAfterAttn.Dispose();
            dXFromAttn.Dispose();

            return dX;
        }

        // ── parameter access ──────────────────────────────────────────────────
        public IEnumerable<IParameter> Parameters()
        {
            foreach (var p in Norm1.Parameters())     yield return p;
            foreach (var p in Attention.Parameters()) yield return p;
            foreach (var p in Norm2.Parameters())     yield return p;
            foreach (var p in FFN.Parameters())       yield return p;
        }

        public void Dispose()
        {
            Norm1.Dispose();
            Attention.Dispose();
            Norm2.Dispose();
            FFN.Dispose();
            _cachedX?.Dispose();
            _cachedNormed1?.Dispose();
            _cachedAfterAttn?.Dispose();
        }
    }
}
