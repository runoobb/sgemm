import argparse
import time
import torch
from torch.utils.cpp_extension import load
import os

# SINGLE KERNEL LAUNCH SCRIPT
parser = argparse.ArgumentParser()
parser.add_argument('--kernel', type=str, required=True,
                    help='kernel function name exported by the extension')
parser.add_argument('--M', type=int, default=1024)
parser.add_argument('--N', type=int, default=1024)
parser.add_argument('--K', type=int, default=1024)
parser.add_argument('--iters', type=int, default=10)
parser.add_argument('--warmup', type=int, default=2)
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

# load extension (adjust flags as needed for your environment)
lib = load(
    name="sgemm_lib_profile",
    sources=[
        "sgemm.cu",
        "sgemm_async.cu",
        "sgemm_wmma_tf32_stage.cu",
        "sgemm_cublas.cu",
    ],

    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-std=c++20",
        "-lineinfo", # enable sass line info for profiling
    ],
    extra_ldflags=[r"D:\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64\cublas.lib"],
    verbose=True,
)

# allocate inputs on GPU
M, N, K = args.M, args.N, args.K
a = torch.randn((M, K), dtype=torch.float32, device='cuda').contiguous()
b = torch.randn((K, N), dtype=torch.float32, device='cuda').contiguous()
c = torch.zeros((M, N), dtype=torch.float32, device='cuda').contiguous()

# get kernel
if not hasattr(lib, args.kernel):
    raise RuntimeError(f'Kernel {args.kernel} not found in extension. Available: {dir(lib)}')

kernel = getattr(lib, args.kernel)

# Warmup
for _ in range(args.warmup):
    try:
        kernel(a, b, c)
    except TypeError:
        # some kernels require extra args (stages/swizzle); try default simple call
        kernel(a, b, c)
torch.cuda.synchronize()

# Timed runs (ncu will capture launches)
start = time.time()
for _ in range(args.iters):
    kernel(a, b, c)
torch.cuda.synchronize()
end = time.time()

print(f'Run {args.kernel} M={M} N={N} K={K} iters={args.iters} time_per_iter_ms={(end-start)/args.iters*1000:.3f}')