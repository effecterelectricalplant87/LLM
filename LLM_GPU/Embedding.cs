using System;
using System.Collections.Generic;
using LLM;

namespace LLM_GPU
{
    /// <summary>
    /// GPU embedding layer.
    ///
    /// Token-embedding lookup and positional-encoding addition are done on the CPU
    /// (the embedding table is small and the scatter-backward would require atomics
    /// on GPU without benefit at this scale).  The result is uploaded to the GPU
    /// once per forward call and stays on the GPU for the rest of the chain.
    /// </summary>
    internal sealed class GpuEmbedding : IEmbeddingLayer<GpuMatrix>
    {
        private readonly TransformerConfig _cfg;
        public  readonly GpuParameter TokenEmbedding;
        private readonly GpuMatrix    _posEncoding;   // fixed, uploaded once

        private int[]? _cachedTokenIds;
        private int    _cachedSeqLen;

        public GpuEmbedding(TransformerConfig cfg, Random rng)
        {
            _cfg           = cfg;
            TokenEmbedding = new GpuParameter(cfg.VocabSize, cfg.EmbeddingDim, rng,
                                              initStd: 0.02f);
            _posEncoding   = BuildPosEncoding(cfg.ContextLength, cfg.EmbeddingDim);
        }

        // ── sinusoidal positional encoding ───────────────────────────────────
        private static GpuMatrix BuildPosEncoding(int maxLen, int d)
        {
            var flat = new float[maxLen * d];
            for (int pos = 0; pos < maxLen; pos++)
            {
                for (int i = 0; i < d / 2; i++)
                {
                    float freq  = MathF.Exp(-2f * i / d * MathF.Log(10000f));
                    float angle = pos * freq;
                    flat[pos * d + 2 * i]     = MathF.Sin(angle);
                    flat[pos * d + 2 * i + 1] = MathF.Cos(angle);
                }
                if (d % 2 != 0)
                {
                    float freq = MathF.Exp(-(d - 1f) / d * MathF.Log(10000f));
                    flat[pos * d + d - 1] = MathF.Sin(pos * freq);
                }
            }
            var m = new GpuMatrix(maxLen, d);
            m.UploadFlat(flat);
            return m;
        }

        // ── forward pass ─────────────────────────────────────────────────────
        public GpuMatrix Forward(int[] tokenIds)
        {
            int seqLen = tokenIds.Length;
            _cachedTokenIds = tokenIds;
            _cachedSeqLen   = seqLen;

            int d = _cfg.EmbeddingDim;

            // Download embedding table and positional encoding once.
            GpuContext.Sync();
            float[] embFlat = TokenEmbedding.Weight.DownloadFlat();
            float[] posFlat = _posEncoding.DownloadFlat();

            // Gather + add on CPU, then upload to GPU.
            var output = new float[seqLen * d];
            for (int t = 0; t < seqLen; t++)
            {
                int id = tokenIds[t];
                for (int j = 0; j < d; j++)
                    output[t * d + j] = embFlat[id * d + j] + posFlat[t * d + j];
            }

            var result = new GpuMatrix(seqLen, d);
            result.UploadFlat(output);
            return result;
        }

        // ── backward pass ────────────────────────────────────────────────────
        public void Backward(GpuMatrix grad)
        {
            if (_cachedTokenIds is null)
                throw new InvalidOperationException("Backward called before Forward.");

            int seqLen = _cachedSeqLen;
            int d      = _cfg.EmbeddingDim;

            // Download gradient from GPU.
            GpuContext.Sync();
            float[] gradFlat = grad.DownloadFlat();

            // Download current gradient buffer from GPU, scatter-add, upload back.
            float[] gradEmb = TokenEmbedding.Gradient.DownloadFlat();

            for (int t = 0; t < seqLen; t++)
            {
                int id = _cachedTokenIds[t];
                for (int j = 0; j < d; j++)
                    gradEmb[id * d + j] += gradFlat[t * d + j];
            }

            TokenEmbedding.Gradient.UploadFlat(gradEmb);
        }

        // ── parameter access ──────────────────────────────────────────────────
        public IEnumerable<IParameter> Parameters()
        {
            yield return TokenEmbedding;
        }

        public void Dispose()
        {
            TokenEmbedding.Dispose();
            _posEncoding.Dispose();
        }
    }
}
