## SGEMM
Profile sgemm kernels on RTX 3060 on Windows

Implementation from LeetCUDA https://github.com/xlite-dev/LeetCUDA/tree/main/kernels/sgemm

Save files in UTF-8 encode format

## Run on my Windows

nvcc on Windows will use cl.exe in MSVC toolkit to compile hostside code

cmd run $conda activate d2l

modify c_cpp_properties.json $includePath


modify sgemm.py compile flags to avoid ambiguous symbol  
add -std=c++20 to extra_cuda_cflags(this option must be added to compile on Windows)  
add extra_ldflags to indicate path of cublas.lib  
<!-- Error 
C:/Users/rst22/.venv/Lib/site-packages/torch/include\torch/csrc/dynamo/compiled_autograd.h(1134): error C2872: 'std': ambiguous symbol
C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/include\valarray(20): note: could be 'std'
C:/Users/rst22/.venv/Lib/site-packages/torch/include\torch/csrc/dynamo/compiled_autograd.h(1134): note: or 'std'
C:/Users/rst22/.venv/Lib/site-packages/torch/include\torch/csrc/dynamo/compiled_autograd.h(1134): note: the template instantiation context (the oldest one first) is
C:/Users/rst22/.venv/Lib/site-packages/torch/include\torch/csrc/dynamo/compiled_autograd.h(1181): note: see reference to class template instantiation 'torch::dynamo::autograd::IValuePacker<__int64>' being compiled
C:/Users/rst22/.venv/Lib/site-packages/torch/include\torch/csrc/dynamo/compiled_autograd.h(1108): note: while compiling class template member function 'c10::TypePtr torch::dynamo::autograd::IValuePacker<__int64>::packed_type(void)' -->

<!-- nvcc warning : incompatible redefinition for option 'std', the last value of this option was used -->


## Run on my WSL2

bash run $conda activate torch

no need to modify sgemm.py
<!-- lib = load(
    name="sgemm_lib",
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
    ],
    extra_cflags=["-std=c++17"],
) -->
