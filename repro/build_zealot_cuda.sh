#!/bin/bash
source ~/.cargo/env
export PATH=$HOME/.cargo/bin:$PATH
unset RUSTFLAGS CARGO_INCREMENTAL CUDA_OXIDE_PTX_DIR CUDA_OXIDE_LLC
export CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=$HOME/nexus_ptx/vortx_shaders.cubin
export CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=$HOME/nexus_ptx/nexus_rbd_shaders3d.cubin
cd ~/Documents/work/zealot
echo "=== zealot CUDA build @ $(date +%H:%M:%S) ==="
cargo build --release --example biped_train_gpu --features "gpu biped_gpu cuda_backend" 2>&1
echo "=== ZEALOT BUILD EXIT $? @ $(date +%H:%M:%S) ==="
