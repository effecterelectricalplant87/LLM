using ILGPU;
using ILGPU.Algorithms;

// Concise alias for the 1-D float view type used in every kernel signature.
using A1 = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;

namespace LLM_GPU
{
    /// <summary>
    /// All GPU kernel methods live here as static functions.
    /// Each method is compiled by ILGPU and run on the GPU.
    ///
    /// Rules for kernel code:
    ///   • No heap allocation, no virtual calls, no LINQ.
    ///   • Use XMath.* for transcendental functions (exp, sqrt, tanh, …).
    ///   • Bounds-check with early return; ILGPU will launch extra threads
    ///     when the extent is rounded up to the next group boundary.
    ///   • Indexing into a flat 1-D buffer: element [row, col] = buf[row*Cols + col].
    /// </summary>
    internal static class Kernels
    {
        // ── Matrix multiply: C = A · B ─────────────────────────────────────────
        // A[m, k], B[k, n], C[m, n].  One thread per output element.
        public static void DotKernel(
            Index2D idx,
            A1 a, A1 b, A1 c,
            int m, int k, int n)
        {
            int row = idx.X;
            int col = idx.Y;
            if (row >= m || col >= n) return;

            float sum = 0f;
            for (int i = 0; i < k; i++)
                sum += a[row * k + i] * b[i * n + col];
            c[row * n + col] = sum;
        }

        // ── Transpose: B[col, row] = A[row, col] ──────────────────────────────
        // Uses 1D indexing to avoid ILGPU auto-group rounding issues with
        // non-square matrices on the CPU accelerator.
        public static void TransposeKernel(
            Index1D idx,
            A1 a, A1 b,
            int rows, int cols)
        {
            if (idx >= rows * cols) return;
            int r = idx / cols;
            int c = idx % cols;
            b[c * rows + r] = a[idx];   // a[r * cols + c] == a[idx]
        }

        // ── Element-wise arithmetic ────────────────────────────────────────────
        public static void AddKernel(Index1D idx, A1 a, A1 b, A1 c, int n)
        {
            if (idx >= n) return;
            c[idx] = a[idx] + b[idx];
        }

        public static void SubKernel(Index1D idx, A1 a, A1 b, A1 c, int n)
        {
            if (idx >= n) return;
            c[idx] = a[idx] - b[idx];
        }

        // Hadamard product
        public static void MulKernel(Index1D idx, A1 a, A1 b, A1 c, int n)
        {
            if (idx >= n) return;
            c[idx] = a[idx] * b[idx];
        }

        public static void ScaleKernel(Index1D idx, A1 a, A1 c, float s, int n)
        {
            if (idx >= n) return;
            c[idx] = a[idx] * s;
        }

        // ── Broadcast-add bias to every row ───────────────────────────────────
        // result[row, col] = a[row, col] + bias[col]
        public static void AddBiasKernel(
            Index2D idx,
            A1 a, A1 bias, A1 c,
            int rows, int cols)
        {
            int row = idx.X;
            int col = idx.Y;
            if (row >= rows || col >= cols) return;
            c[row * cols + col] = a[row * cols + col] + bias[col];
        }

        // ── In-place addition: a += b ──────────────────────────────────────────
        public static void AddInPlaceKernel(Index1D idx, A1 a, A1 b, int n)
        {
            if (idx >= n) return;
            a[idx] += b[idx];
        }

        // ── Buffer utilities ──────────────────────────────────────────────────
        public static void ZeroKernel(Index1D idx, A1 a, int n)
        {
            if (idx >= n) return;
            a[idx] = 0f;
        }

        public static void FillKernel(Index1D idx, A1 a, float val, int n)
        {
            if (idx >= n) return;
            a[idx] = val;
        }

        public static void CopyKernel(Index1D idx, A1 src, A1 dst, int n)
        {
            if (idx >= n) return;
            dst[idx] = src[idx];
        }

        // ── Row-wise softmax (one thread per row) ────────────────────────────
        // Modified in-place.
        public static void SoftmaxKernel(Index1D idx, A1 a, int rows, int cols)
        {
            if (idx >= rows) return;
            int offset = idx * cols;

            float maxVal = float.NegativeInfinity;
            for (int j = 0; j < cols; j++)
            {
                float v = a[offset + j];
                if (v > maxVal) maxVal = v;
            }

            float sum = 0f;
            for (int j = 0; j < cols; j++)
            {
                float e = XMath.Exp(a[offset + j] - maxVal);
                a[offset + j] = e;
                sum += e;
            }

            for (int j = 0; j < cols; j++)
                a[offset + j] /= sum;
        }

        // ── Row-wise softmax backward (one thread per row) ──────────────────
        public static void SoftmaxBackwardKernel(
            Index1D idx,
            A1 softmaxOut, A1 dOut, A1 dIn,
            int rows, int cols)
        {
            if (idx >= rows) return;
            int offset = idx * cols;

            float dot = 0f;
            for (int j = 0; j < cols; j++)
                dot += softmaxOut[offset + j] * dOut[offset + j];

            for (int j = 0; j < cols; j++)
                dIn[offset + j] = softmaxOut[offset + j] * (dOut[offset + j] - dot);
        }

        // ── GELU and its derivative ───────────────────────────────────────────
        public static void GELUKernel(Index1D idx, A1 a, A1 c, int n)
        {
            if (idx >= n) return;
            const float Sqrt2OverPi = 0.7978845608028654f;
            const float Coeff       = 0.044715f;
            float x     = a[idx];
            float inner = Sqrt2OverPi * (x + Coeff * x * x * x);
            c[idx] = 0.5f * x * (1f + XMath.Tanh(inner));
        }

        public static void GELUGradKernel(Index1D idx, A1 a, A1 c, int n)
        {
            if (idx >= n) return;
            const float Sqrt2OverPi = 0.7978845608028654f;
            const float Coeff       = 0.044715f;
            float x        = a[idx];
            float inner    = Sqrt2OverPi * (x + Coeff * x * x * x);
            float tanhVal  = XMath.Tanh(inner);
            float sech2    = 1f - tanhVal * tanhVal;
            float dInnerDx = Sqrt2OverPi * (1f + 3f * Coeff * x * x);
            c[idx] = 0.5f * (1f + tanhVal) + 0.5f * x * sech2 * dInnerDx;
        }

        // ── Column-wise sum: result[col] = Σ_row A[row, col] ────────────────
        // One thread per column.
        public static void SumOverRowsKernel(
            Index1D idx,
            A1 a, A1 result,
            int rows, int cols)
        {
            if (idx >= cols) return;
            float sum = 0f;
            for (int r = 0; r < rows; r++)
                sum += a[r * cols + idx];
            result[idx] = sum;
        }

        // ── Column slicing ────────────────────────────────────────────────────
        // dst[row, col] = src[row, colStart + col]
        public static void SliceColsKernel(
            Index2D idx,
            A1 src, A1 dst,
            int rows, int srcCols, int numCols, int colStart)
        {
            int row = idx.X;
            int col = idx.Y;
            if (row >= rows || col >= numCols) return;
            dst[row * numCols + col] = src[row * srcCols + colStart + col];
        }

        // Scatter-add columns back: dst[row, colStart+col] += src[row, col]
        // No atomics needed: disjoint column ranges per head.
        public static void ScatterAddColsKernel(
            Index2D idx,
            A1 src, A1 dst,
            int rows, int numCols, int dstCols, int colStart)
        {
            int row = idx.X;
            int col = idx.Y;
            if (row >= rows || col >= numCols) return;
            dst[row * dstCols + colStart + col] += src[row * numCols + col];
        }

        // ── Layer Norm forward (one thread per position) ──────────────────────
        // Writes output, xHat cache, and sigma cache.
        public static void LayerNormForwardKernel(
            Index1D idx,
            A1 x, A1 gamma, A1 beta,
            A1 output, A1 xHat, A1 sigma,
            int T, int D)
        {
            if (idx >= T) return;
            int offset = idx * D;

            float mean = 0f;
            for (int d = 0; d < D; d++) mean += x[offset + d];
            mean /= D;

            float variance = 0f;
            for (int d = 0; d < D; d++)
            {
                float diff = x[offset + d] - mean;
                variance += diff * diff;
            }
            variance /= D;

            const float Eps = 1e-5f;
            float sig = XMath.Sqrt(variance + Eps);
            sigma[idx] = sig;

            for (int d = 0; d < D; d++)
            {
                float normed = (x[offset + d] - mean) / sig;
                xHat[offset + d]    = normed;
                output[offset + d]  = gamma[d] * normed + beta[d];
            }
        }

        // ── Layer Norm backward for input gradient (one thread per position) ──
        public static void LayerNormBackwardKernel(
            Index1D idx,
            A1 dOut, A1 gamma, A1 xHat, A1 sigma,
            A1 dX,
            int T, int D)
        {
            if (idx >= T) return;
            int offset = idx * D;
            float sig = sigma[idx];

            float gMean  = 0f;
            float gXMean = 0f;
            for (int d = 0; d < D; d++)
            {
                float g = dOut[offset + d] * gamma[d];
                gMean  += g;
                gXMean += g * xHat[offset + d];
            }
            gMean  /= D;
            gXMean /= D;

            for (int d = 0; d < D; d++)
            {
                float g = dOut[offset + d] * gamma[d];
                dX[offset + d] = (g - gMean - xHat[offset + d] * gXMean) / sig;
            }
        }

        // ── Layer Norm gamma/beta gradient (one thread per feature dim) ───────
        // Accumulates: dGamma[d] += Σ_t dOut[t,d] * xHat[t,d]
        //              dBeta[d]  += Σ_t dOut[t,d]
        public static void LayerNormGammaGradKernel(
            Index1D idx,
            A1 dOut, A1 xHat,
            A1 dGamma, A1 dBeta,
            int T, int D)
        {
            if (idx >= D) return;
            float dg = 0f;
            float db = 0f;
            for (int t = 0; t < T; t++)
            {
                dg += dOut[t * D + idx] * xHat[t * D + idx];
                db += dOut[t * D + idx];
            }
            dGamma[idx] += dg;
            dBeta[idx]  += db;
        }

        // ── Adam update + grad zero (one thread per element) ──────────────────
        public static void AdamKernel(
            Index1D idx,
            A1 weight, A1 grad, A1 m, A1 v,
            float lr, float beta1, float beta2, float eps,
            float bc1, float bc2,
            int n)
        {
            if (idx >= n) return;
            float g    = grad[idx];
            float mNew = beta1 * m[idx] + (1f - beta1) * g;
            float vNew = beta2 * v[idx] + (1f - beta2) * g * g;
            m[idx] = mNew;
            v[idx] = vNew;
            float mHat = mNew / bc1;
            float vHat = vNew / bc2;
            weight[idx] -= lr * mHat / (XMath.Sqrt(vHat) + eps);
            grad[idx]    = 0f;
        }

        // ── Gradient clip scale (one thread per element) ──────────────────────
        public static void GradClipScaleKernel(Index1D idx, A1 grad, float scale, int n)
        {
            if (idx >= n) return;
            grad[idx] *= scale;
        }
    }
}
