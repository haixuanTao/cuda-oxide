# gemm_fma

The same register-blocked tiled SGEMM as `gemm_migrate`, but the inner loop uses
`f32::mul_add` for a true fused multiply-add.

`mul_add` lowers to `__nv_fmaf` → a single `fma.rn.f32`, which puts the kernel on
the **libdevice path** (NVVM-IR flavor, `load_kernel_module` + `cuda_launch!`,
libNVVM/nvJitLink). It measures the true-FMA ceiling vs the plain mul+add
PTX-path variant.

```
cargo oxide run gemm_fma
```

1024³ SGEMM on an RTX 5090: **~16.3 TFLOPS** (vs ~10.5 for plain mul+add and
~7.4 for a naive shared-memory tiled GEMM), bit-exact vs CPU.

## Note for Blackwell (sm_120) / newer GPUs

Like any libdevice kernel, this needs a CUDA 12.8+ libNVVM/nvJitLink. A userspace
(no-root) fix is to point `LIBNVVM_PATH` / `LIBNVJITLINK_PATH` at the
`nvidia-cuda-nvcc-cu12` / `nvidia-nvjitlink-cu12` (12.9.x) pip wheels.
