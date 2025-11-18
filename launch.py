import argparse
import time
import torch
from torch.utils.cpp_extension import load
import os
from sgemm import run_benchmark

# SINGLE KERNEL LAUNCH SCRIPT
parser = argparse.ArgumentParser()
parser.add_argument('--kernel', type=str, required=True,
                    help='kernel function name exported by the extension')
parser.add_argument('--M', type=int, default=1024)
parser.add_argument('--N', type=int, default=1024)
parser.add_argument('--K', type=int, default=1024)
parser.add_argument('--stages', type=int, default=-1)
parser.add_argument('--swizzle', type=bool, default=False)
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

run_benchmark(perf_func=kernel, a=a, b=b, out=c, stages=args.stages, swizzle=args.swizzle)

