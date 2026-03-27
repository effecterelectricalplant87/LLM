using System;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace LLM_GPU
{
    /// <summary>
    /// Manages the ILGPU context and accelerator as a process-wide singleton.
    /// Prefers CUDA → OpenCL → ILGPU CPU fallback.
    /// Call Initialize() once at startup and Shutdown() at exit.
    /// </summary>
    internal static class GpuContext
    {
        private static Context?     _context;
        private static Accelerator? _accelerator;

        public static Accelerator Accelerator =>
            _accelerator ?? throw new InvalidOperationException(
                "GpuContext not initialized. Call GpuContext.Initialize() first.");

        public static void Initialize()
        {
            _context = Context.Create(builder => builder.Default().EnableAlgorithms());

            // Walk the available devices and pick the best accelerator.
            // Priority: CUDA > OpenCL > CPU
            Accelerator? chosen   = null;
            int          priority = -1;

            foreach (Device device in _context)
            {
                int p = device.AcceleratorType switch
                {
                    AcceleratorType.Cuda    => 2,
                    AcceleratorType.OpenCL  => 1,
                    AcceleratorType.CPU     => 0,
                    _                       => -1
                };

                if (p > priority)
                {
                    chosen?.Dispose();
                    chosen   = device.CreateAccelerator(_context);
                    priority = p;
                    if (p == 2) break;   // CUDA is best – stop searching
                }
            }

            _accelerator = chosen
                ?? throw new InvalidOperationException(
                       "No ILGPU-compatible device found on this machine.");

            Console.WriteLine(
                $"GPU backend: {_accelerator.AcceleratorType} – {_accelerator.Name}");
        }

        /// <summary>
        /// Synchronise the accelerator's default stream with the CPU.
        /// Call whenever the CPU needs to read back results produced on the GPU.
        /// </summary>
        public static void Sync() => _accelerator?.Synchronize();

        public static void Shutdown()
        {
            _accelerator?.Dispose();
            _accelerator = null;
            _context?.Dispose();
            _context = null;
        }
    }
}
