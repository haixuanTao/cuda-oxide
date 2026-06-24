#!/bin/bash
# Agnostic cubin build for the native-CUDA (cuda-oxide) zealot stack.
# All paths are $HOME-relative — no hardcoded box names. sm_120 (Blackwell).
# Builds nexus_rbd_shaders3d + vortx_shaders -> .ll -> (libdevice link) -> cubin.
set -uo pipefail
source ~/.cargo/env
ARCH=${ARCH:-sm_120}
TOOL=$HOME/llvm21/bin
LIBDEV=$HOME/nvvm-wheel/extracted/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
PTXAS=$HOME/nvvm-wheel/extracted/nvidia/cuda_nvcc/bin/ptxas
BACKEND=$HOME/cuda-oxide-src/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so
export CUDA_OXIDE_PTX_DIR=$HOME/nexus_ptx
export CUDA_OXIDE_LLC=$TOOL/llc
export LIBCLANG_PATH=$HOME/llvm21/lib
export LD_LIBRARY_PATH=$HOME/llvm21/lib:/usr/lib/x86_64-linux-gnu
export PATH=$HOME/.cargo/bin:$PATH
mkdir -p $CUDA_OXIDE_PTX_DIR
RF="-Z codegen-backend=$BACKEND -Zalways-encode-mir -Zmir-enable-passes=-JumpThreading"

ll_to_cubin() {  # $1=name (basename of .ll/.cubin)
  local LL=$CUDA_OXIDE_PTX_DIR/$1.ll
  local CUBIN=$CUDA_OXIDE_PTX_DIR/$1.cubin
  echo "  defines: $(grep -c '^define' "$LL")"
  "$TOOL/llvm-as" "$LL" -o /tmp/$1.bc
  "$TOOL/llvm-link" /tmp/$1.bc "$LIBDEV" -o /tmp/${1}_linked.bc
  "$TOOL/opt" -passes="internalize,globaldce" /tmp/${1}_linked.bc -o /tmp/${1}_pruned.bc
  "$TOOL/llc" -mcpu=$ARCH -O3 /tmp/${1}_pruned.bc -o /tmp/$1.ptx
  rm -f "$CUBIN"
  "$PTXAS" -arch=$ARCH -O3 /tmp/$1.ptx -o "$CUBIN"
  ls -la "$CUBIN"
}

echo "########## [1] NEXUS nexus_rbd_shaders3d ($ARCH) ##########"
cd ~/Documents/work/nexus-cuda
cargo clean -p nexus_rbd_shaders3d 2>/dev/null || true
CARGO_INCREMENTAL=0 RUSTFLAGS="$RF" cargo +nightly-2026-04-03 build -p nexus_rbd_shaders3d --release \
  --no-default-features --features "cuda-oxide dim3 unsafe_remove_boundchecks" \
  --target nvptx64-nvidia-cuda -Z build-std=core 2>&1 | tail -6
ll_to_cubin nexus_rbd_shaders3d

echo "########## [2] VORTX vortx_shaders ($ARCH) ##########"
cd ~/Documents/work/vortx
cargo clean -p vortx-shaders 2>/dev/null || true
CARGO_INCREMENTAL=0 RUSTFLAGS="$RF" cargo +nightly-2026-04-03 build -p vortx-shaders --release \
  --no-default-features --features "cuda-oxide unsafe_remove_boundchecks" \
  --target nvptx64-nvidia-cuda -Z build-std=core 2>&1 | tail -6
ll_to_cubin vortx_shaders

echo "########## CUBINS DONE ##########"
ls -la $CUDA_OXIDE_PTX_DIR/*.cubin
