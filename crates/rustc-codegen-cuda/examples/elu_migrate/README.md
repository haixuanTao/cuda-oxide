# elu_migrate

A small port of a real-world Rust GPU kernel onto cuda-oxide: the **ELU**
activation (`if x > 0 { x } else { x.exp() - 1 }`), one thread per element over a
`DisjointSlice<f32>`. Originally a `rust-gpu`/WGSL compute shader in the
[vortx](https://github.com/dimforge/vortx) GPU tensor library; here it is the
same arithmetic as a cuda-oxide `#[kernel]`.

It exercises the **libdevice path**: `f32::exp()` lowers to `__nv_expf`, which
puts the kernel on the NVVM-IR flavor — loaded with `load_kernel_module` + the
`cuda_launch!` macro (not `load_module_from_file`), with libNVVM/nvJitLink
linking libdevice and building the cubin at first launch.

```
cargo oxide run elu_migrate
```

Verifies the GPU output against a CPU ELU reference (max error ~6e-8, pure float
rounding).

## Note for Blackwell (sm_120) / newer GPUs

The bundled CUDA toolkit's libNVVM must be new enough for your arch. CUDA 12.0's
libNVVM cannot emit `compute_120` and cannot parse modern LLVM IR; nvJitLink
12.0 lacks `nvJitLinkCreate`. A userspace fix (no root) is to point at CUDA
12.8+ libraries from pip wheels:

```
pip download nvidia-cuda-nvcc-cu12 nvidia-nvjitlink-cu12   # 12.9.x
export LIBNVVM_PATH=.../nvidia/cuda_nvcc/nvvm/lib64/libnvvm.so
export LIBNVJITLINK_PATH=.../nvidia/nvjitlink/lib/libnvJitLink.so.12
```
