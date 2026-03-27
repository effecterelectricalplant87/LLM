using System;
using System.Collections.Generic;
using ILGPU;
using ILGPU.Runtime;
using LLM;

using A1 = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;

namespace LLM_GPU
{
    /// <summary>
    /// GPU layer normalisation.
    ///
    /// Forward:  one GPU thread per sequence position; sequential over feature dim.
    /// Backward: two kernel passes:
    ///   1. LayerNormGammaGradKernel – accumulates dGamma and dBeta (one thread / dim).
    ///   2. LayerNormBackwardKernel  – computes dX (one thread / position).
    /// </summary>
    internal sealed class GpuLayerNorm : ILayer<GpuMatrix>
    {
        private readonly int _dim;

        public readonly GpuParameter Gamma;   // [1 × dim], init = 1
        public readonly GpuParameter Beta;    // [1 × dim], init = 0

        // Forward-pass cache
        private GpuMatrix? _cachedXHat;    // [T × dim]
        private GpuMatrix? _cachedSigma;   // [T × 1] (one σ per position)
        private int        _cachedSeqLen;

        // Kernel delegates
        private static Action<Index1D, A1, A1, A1, A1, A1, A1, int, int>? _forwardKernel;
        private static Action<Index1D, A1, A1, A1, A1, A1, int, int>?     _backwardKernel;
        private static Action<Index1D, A1, A1, A1, A1, int, int>?         _gammaGradKernel;

        public GpuLayerNorm(int dim)
        {
            _dim  = dim;
            Gamma = GpuParameter.Ones(1, dim);
            Beta  = GpuParameter.Zeros(1, dim);
        }

        // ── forward pass ──────────────────────────────────────────────────────
        public GpuMatrix Forward(GpuMatrix x)
        {
            int T = x.Rows;
            int D = x.Cols;
            _cachedSeqLen = T;

            var output = new GpuMatrix(T, D);

            // Replace cache (dispose old buffers)
            _cachedXHat?.Dispose();
            _cachedSigma?.Dispose();
            _cachedXHat  = new GpuMatrix(T, D);
            _cachedSigma = new GpuMatrix(T, 1);

            _forwardKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<
                    Index1D,
                    A1, A1, A1,
                    A1, A1, A1,
                    int, int>(Kernels.LayerNormForwardKernel);

            _forwardKernel(T,
                x.View, Gamma.Weight.View, Beta.Weight.View,
                output.View, _cachedXHat.View, _cachedSigma.View,
                T, D);

            return output;
        }

        // ── backward pass ─────────────────────────────────────────────────────
        public GpuMatrix Backward(GpuMatrix dOut)
        {
            if (_cachedXHat is null || _cachedSigma is null)
                throw new InvalidOperationException("Backward called before Forward.");

            int T  = _cachedSeqLen;
            int D  = _dim;
            var dX = new GpuMatrix(T, D);

            // 1. Accumulate dGamma and dBeta
            _gammaGradKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<
                    Index1D,
                    A1, A1,
                    A1, A1,
                    int, int>(Kernels.LayerNormGammaGradKernel);

            _gammaGradKernel(D,
                dOut.View, _cachedXHat.View,
                Gamma.Gradient.View, Beta.Gradient.View,
                T, D);

            // 2. Compute dX
            _backwardKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<
                    Index1D,
                    A1, A1, A1, A1,
                    A1,
                    int, int>(Kernels.LayerNormBackwardKernel);

            _backwardKernel(T,
                dOut.View, Gamma.Weight.View, _cachedXHat.View, _cachedSigma.View,
                dX.View,
                T, D);

            return dX;
        }

        // ── parameter access ──────────────────────────────────────────────────
        public IEnumerable<IParameter> Parameters()
        {
            yield return Gamma;
            yield return Beta;
        }

        public void Dispose()
        {
            Gamma.Dispose();
            Beta.Dispose();
            _cachedXHat?.Dispose();
            _cachedSigma?.Dispose();
        }
    }
}
