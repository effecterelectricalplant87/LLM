using System;
using System.Collections.Generic;
using LLM;

namespace LLM_GPU
{
    /// <summary>
    /// GPU position-wise feed-forward network (two linear layers + GELU).
    ///
    ///   hidden = GELU( X · W1 + b1 )     [T × d_ff]
    ///   output = hidden · W2 + b2         [T × D]
    /// </summary>
    internal sealed class GpuFeedForward : ILayer<GpuMatrix>
    {
        private readonly int _dModel;
        private readonly int _dFF;

        public readonly GpuParameter W1, B1;
        public readonly GpuParameter W2, B2;

        // Forward-pass cache
        private GpuMatrix? _cachedX;
        private GpuMatrix? _cachedPreAct;    // h  = X·W1 + b1
        private GpuMatrix? _cachedPostAct;   // h' = GELU(h)

        public GpuFeedForward(int dModel, int dFF, Random rng)
        {
            _dModel = dModel;
            _dFF    = dFF;

            W1 = GpuParameter.Xavier(dModel, dFF,    rng);
            W2 = GpuParameter.Xavier(dFF,    dModel, rng);
            B1 = GpuParameter.Zeros(1, dFF);
            B2 = GpuParameter.Zeros(1, dModel);
        }

        // ── forward pass ──────────────────────────────────────────────────────
        public GpuMatrix Forward(GpuMatrix x)
        {
            _cachedX?.Dispose();
            _cachedX = x;

            // h = x · W1 + b1
            GpuMatrix pre;
            {
                using var tmp = GpuMatrix.Dot(x, W1.Weight);
                pre = tmp.AddBias(B1.Weight);
            }
            _cachedPreAct?.Dispose();
            _cachedPreAct = pre;

            // h' = GELU(h)
            GpuMatrix post = pre.GELU();
            _cachedPostAct?.Dispose();
            _cachedPostAct = post;

            // output = h' · W2 + b2
            GpuMatrix output;
            {
                using var tmp = GpuMatrix.Dot(post, W2.Weight);
                output = tmp.AddBias(B2.Weight);
            }
            return output;
        }

        // ── backward pass ─────────────────────────────────────────────────────
        public GpuMatrix Backward(GpuMatrix dOut)
        {
            if (_cachedX is null || _cachedPreAct is null || _cachedPostAct is null)
                throw new InvalidOperationException("Backward called before Forward.");

            // ── Layer 2 backward ──────────────────────────────────────────────
            // dW2 += postT · dOut
            {
                using var postT = _cachedPostAct.Transpose();
                using var g     = GpuMatrix.Dot(postT, dOut);
                W2.Gradient.AddInPlace(g);
            }
            // dB2 += sum_rows(dOut)
            AccumulateBiasGrad(B2, dOut, _dModel);

            // dPost = dOut · W2T
            GpuMatrix dPost;
            {
                using var W2T = W2.Weight.Transpose();
                dPost = GpuMatrix.Dot(dOut, W2T);
            }

            // ── GELU backward ─────────────────────────────────────────────────
            // dPre = dPost ⊙ GELU'(pre)
            GpuMatrix dPre;
            {
                using var geluGrad = _cachedPreAct.GELUGrad();
                dPre = GpuMatrix.Mul(dPost, geluGrad);
            }
            dPost.Dispose();

            // ── Layer 1 backward ──────────────────────────────────────────────
            // dW1 += xT · dPre
            {
                using var xT = _cachedX.Transpose();
                using var g  = GpuMatrix.Dot(xT, dPre);
                W1.Gradient.AddInPlace(g);
            }
            // dB1 += sum_rows(dPre)
            AccumulateBiasGrad(B1, dPre, _dFF);

            // dX = dPre · W1T
            GpuMatrix dX;
            {
                using var W1T = W1.Weight.Transpose();
                dX = GpuMatrix.Dot(dPre, W1T);
            }
            dPre.Dispose();

            return dX;
        }

        // ── helpers ───────────────────────────────────────────────────────────
        private static void AccumulateBiasGrad(GpuParameter bias, GpuMatrix dMat, int dim)
        {
            float[] sum  = dMat.SumOverRows();
            float[] flat = bias.Gradient.DownloadFlat();
            for (int j = 0; j < dim; j++) flat[j] += sum[j];
            bias.Gradient.UploadFlat(flat);
        }

        // ── parameter access ──────────────────────────────────────────────────
        public IEnumerable<IParameter> Parameters()
        {
            yield return W1; yield return B1;
            yield return W2; yield return B2;
        }

        public void Dispose()
        {
            W1.Dispose(); B1.Dispose();
            W2.Dispose(); B2.Dispose();
            _cachedX?.Dispose();
            _cachedPreAct?.Dispose();
            _cachedPostAct?.Dispose();
        }
    }
}
