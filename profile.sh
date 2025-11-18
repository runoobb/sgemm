# TODO: modify profile.sh to support launch.py(single launch)

# Kernels from sgemm_async.cuncu --launch-skip 6 --launch-count 1 --set full --force-overwrite -o sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_m2048_n2048_k1024.ncu-rep python launch.py --kernel sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf --M 2048 --N 2048 --K 1024


ncu --launch-skip 6 --launch-count 1 --set full --force-overwrite -o sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_m2048_n2048_k1024.ncu-rep --pm-sampling-interval 0 --pm-sampling-buffer-size 0 --pm-sampling-max-passes 0 python launch.py --kernel sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf --M 2048 --N 2048 --K 1024

ncu --launch-skip 6 --launch-count 1 --set full --force-overwrite -o sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_m2048_n2048_k1024.ncu-rep --pm-sampling-interval 0 --pm-sampling-buffer-size 0 --pm-sampling-max-passes 0 python launch.py --kernel sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf --M 2048 --N 2048 --K 1024

ncu --launch-skip 6 --launch-count 1 --set full --force-overwrite -o sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async_m2048_n2048_k1024.ncu-rep --pm-sampling-interval 0 --pm-sampling-buffer-size 0 --pm-sampling-max-passes 0 python launch.py --kernel sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async --M 2048 --N 2048 --K 1024

ncu --launch-skip 6 --launch-count 1 --set full --force-overwrite -o sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async_m2048_n2048_k1024.ncu-rep --pm-sampling-interval 0 --pm-sampling-buffer-size 0 --pm-sampling-max-passes 0 python launch.py --kernel sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async --M 2048 --N 2048 --K 1024

ncu --launch-skip 6 --launch-count 1 --set full --force-overwrite -o sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async_m2048_n2048_k1024.ncu-rep --pm-sampling-interval 0 --pm-sampling-buffer-size 0 --pm-sampling-max-passes 0 python launch.py --kernel sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async --M 2048 --N 2048 --K 1024

# All in one
# ncu --kernel-name sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_kernel --launch-skip-before-match 4 --launch-count 1 --set full -o sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_m2048_n2048_k1024.ncu-rep python sgemm.py

# Kernels from sgemm.cu

ncu --launch-skip 6 --launch-count 1 --set full --force-overwrite -o sgemm_t_8x8_sliced_k_f32x4_m2048_n2048_k1024.ncu-rep --pm-sampling-interval 0 --pm-sampling-buffer-size 0 --pm-sampling-max-passes 0 python launch.py --kernel sgemm_t_8x8_sliced_k_f32x4 --M 2048 --N 2048 --K 1024

ncu --launch-skip 6 --launch-count 1 --set full --force-overwrite -o sgemm_t_8x8_sliced_k_f32x4_bcf_m2048_n2048_k1024.ncu-rep --pm-sampling-interval 0 --pm-sampling-buffer-size 0 --pm-sampling-max-passes 0 python launch.py --kernel sgemm_t_8x8_sliced_k_f32x4_bcf --M 2048 --N 2048 --K 1024

ncu --launch-skip 6 --launch-count 1 --set full --force-overwrite -o sgemm_t_8x8_sliced_k_f32x4_bcf_offset_m2048_n2048_k1024.ncu-rep --pm-sampling-interval 0 --pm-sampling-buffer-size 0 --pm-sampling-max-passes 0 python launch.py --kernel sgemm_t_8x8_sliced_k_f32x4_bcf_offset --M 2048 --N 2048 --K 1024

ncu --launch-skip 6 --launch-count 1 --set full --force-overwrite -o sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_m2048_n2048_k1024.ncu-rep --pm-sampling-interval 0 --pm-sampling-buffer-size 0 --pm-sampling-max-passes 0 python launch.py --kernel sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf --M 2048 --N 2048 --K 1024

ncu --launch-skip 6 --launch-count 1 --set full --force-overwrite -o sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset_m2048_n2048_k1024.ncu-rep --pm-sampling-interval 0 --pm-sampling-buffer-size 0 --pm-sampling-max-passes 0 python launch.py --kernel sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset --M 2048 --N 2048 --K 1024

# Kernels from sgemm_wmma_tf32_stage.cu
ncu --kernel-name sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel --launch-skip-before-match 6 --launch-count 1 --set full --force-overwrite -o sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_m2048_n2048_k1024_stages2.ncu-rep --pm-sampling-interval 0 --pm-sampling-buffer-size 0 --pm-sampling-max-passes 0 python launch.py --kernel sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages --M 2048 --N 2048 --K 1024 --stages 2

ncu --kernel-name sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel --launch-skip-before-match 6 --launch-count 1 --set full --force-overwrite -o sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_m2048_n2048_k1024_stages3.ncu-rep --pm-sampling-interval 0 --pm-sampling-buffer-size 0 --pm-sampling-max-passes 0 python launch.py --kernel sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages --M 2048 --N 2048 --K 1024 --stages 3

ncu --kernel-name sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel --launch-skip-before-match 6 --launch-count 1 --set full --force-overwrite -o sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_m2048_n2048_k1024_stages2_swizzle.ncu-rep --pm-sampling-interval 0 --pm-sampling-buffer-size 0 --pm-sampling-max-passes 0 python launch.py --kernel sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages --M 2048 --N 2048 --K 1024 --stages 2 --swizzle True

ncu --kernel-name sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel --launch-skip-before-match 6 --launch-count 1 --set full --force-overwrite -o sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_m2048_n2048_k1024_stages3_swizzle.ncu-rep --pm-sampling-interval 0 --pm-sampling-buffer-size 0 --pm-sampling-max-passes 0 python launch.py --kernel sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages --M 2048 --N 2048 --K 1024 --stages 3 --swizzle True

ncu --kernel-name sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel --launch-skip-before-match 6 --launch-count 1 --set full --force-overwrite -o sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_m2048_n2048_k1024_stages2.ncu-rep --pm-sampling-interval 0 --pm-sampling-buffer-size 0 --pm-sampling-max-passes 0 python launch.py --kernel sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem --M 2048 --N 2048 --K 1024 --stages 2

ncu --kernel-name sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel --launch-skip-before-match 6 --launch-count 1 --set full --force-overwrite -o sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_m2048_n2048_k1024_stages3.ncu-rep --pm-sampling-interval 0 --pm-sampling-buffer-size 0 --pm-sampling-max-passes 0 python launch.py --kernel sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem --M 2048 --N 2048 --K 1024 --stages 3

ncu --kernel-name sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel --launch-skip-before-match 6 --launch-count 1 --set full --force-overwrite -o sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_m2048_n2048_k1024_stages2_swizzle.ncu-rep --pm-sampling-interval 0 --pm-sampling-buffer-size 0 --pm-sampling-max-passes 0 python launch.py --kernel sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem --M 2048 --N 2048 --K 1024 --stages 2 --swizzle True

ncu --kernel-name sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel --launch-skip-before-match 6 --launch-count 1 --set full --force-overwrite -o sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_m2048_n2048_k1024_stages3_swizzle.ncu-rep --pm-sampling-interval 0 --pm-sampling-buffer-size 0 --pm-sampling-max-passes 0 python launch.py --kernel sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem --M 2048 --N 2048 --K 1024 --stages 3 --swizzle True

