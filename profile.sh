# TODO: modify profile.sh to support launch.py(single launch)
ncu --launch-skip 4 --launch-count 1 --set full -o sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_m2048_n2048_k1024.ncu-rep python launch.py --kernel sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf --M 2048 --N 2048 --K 1024 --iters 5

ncu --launch-skip 4 --launch-count 1 --set full -o sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_m2048_n2048_k1024.ncu-rep python launch.py --kernel sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf --M 2048 --N 2048 --K 1024 --iters 5

ncu --launch-skip 4 --launch-count 1 --set full -o sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_m2048_n2048_k1024.ncu-rep python launch.py --kernel sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf --M 2048 --N 2048 --K 1024 --iters 5

ncu --launch-skip 4 --launch-count 1 --set full -o sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async_m2048_n2048_k1024.ncu-rep python launch.py --kernel sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async --M 2048 --N 2048 --K 1024 --iters 5

ncu --launch-skip 4 --launch-count 1 --set full -o sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async_m2048_n2048_k1024.ncu-rep python launch.py --kernel sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async --M 2048 --N 2048 --K 1024 --iters 5

ncu --launch-skip 4 --launch-count 1 --set full -o sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async_m2048_n2048_k1024.ncu-rep python launch.py --kernel sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async --M 2048 --N 2048 --K 1024 --iters 5

# ncu --kernel-name sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_kernel --launch-skip-before-match 4 --launch-count 1 --set full -o sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_m2048_n2048_k1024.ncu-rep python sgemm.py