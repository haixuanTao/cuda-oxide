#!/bin/bash
set -uo pipefail
source ~/.cargo/env
export CUDA_OXIDE_LLC=$HOME/llvm21/bin/llc
export LIBCLANG_PATH=$HOME/llvm21/lib
export LD_LIBRARY_PATH=$HOME/llvm21/lib:/usr/lib/x86_64-linux-gnu
cd ~/cuda-oxide-src/crates/rustc-codegen-cuda
echo "=== building rustc-codegen-cuda backend .so @ $(date +%H:%M:%S) ==="
cargo +nightly-2026-04-03 build 2>&1 | tail -40
echo "=== EXIT $? ==="
ls -la ~/cuda-oxide-src/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so 2>&1 || \
ls -la ~/cuda-oxide-src/target/debug/librustc_codegen_cuda.so 2>&1
echo "=== BACKEND BUILD DONE ==="
