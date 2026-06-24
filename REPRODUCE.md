# Reproduce: native-CUDA (cuda-oxide) zealot biped training on sm_120

End-to-end recipe to compile the `vortx-shaders` (41 kernels) and
`nexus_rbd_shaders3d` (266 kernels) crates to native-CUDA cubins and run
zealot's `BIPED_CUDA=1` biped training on a Blackwell (sm_120) GPU — verified
on a vast.ai **RTX 5060 (8 GB)**, and equally applicable to the RTX 5090.

The codegen fixes that make this work are in this branch
(`feat/nexus3d-vortx-native-cuda-5060-fixes`, on top of `f0f9494`). See the
commit `codegen: compile vortx + nexus3d shaders to sm_120 cubin` for the
seven backend changes. Everything else needed to reproduce is below; the
sibling-repo working-tree deltas are bundled as patches under `repro/`.

## 0. Host

- GPU: NVIDIA Blackwell, **sm_120** (RTX 5060 / 5090), recent driver (≥ 580).
- OS: Ubuntu 24.04, ~32 GB disk free (tight — LLVM21 + 6 repos + targets).
- System CUDA only needs the driver at runtime; the build uses CUDA-12.9 *wheels*.

## 1. Repo layout & branches (siblings under `~/Documents/work/`, plus `~/cuda-oxide-src`)

| repo            | branch                              | base commit | working-tree delta |
|-----------------|-------------------------------------|-------------|--------------------|
| cuda-oxide-src  | `feat/nexus3d-vortx-native-cuda-5060-fixes` | f0f9494 (this branch IS the fix) | — |
| khal            | `fix/cuda-slice-arg-element-count`  | e29c5c4     | `repro/khal.patch` |
| vortx           | `fix/reduce-generic-followup`       | d299f74     | `repro/vortx.patch` |
| nexus-cuda      | `feat/per-env-parallelism`          | ef501d3     | `repro/nexus-cuda.patch` |
| zealot          | `feat/native-cuda-e2e-bench`        | 5241a21     | `repro/zealot.patch` |

The sibling deltas enable the `cuda-oxide` feature on `khal-std` / `vortx-shaders` /
`nexus_rbd_shaders3d` (typed-param CUDA-entry generator, `khal_std` arch::cuda
builtins, the `[patch.crates-io]` redirects, glam `=0.32.1` pin). Apply with
`git apply repro/<repo>.patch` from each repo root, or rsync the working trees.
`glamx` must resolve to crates.io **0.2.0** (`nostd-libm`, `bytemuck` features
only — NOT `u32`/`i32`/`f64`, which need glamx 0.3 → glam 0.33 → breaks the pin).

## 2. Toolchain (userspace, no sudo)

- `rustup` + **nightly-2026-04-03** with `rust-src` (for `-Z build-std=core`).
- **LLVM 21.1.0** at `~/llvm21` (provides `llvm-as`, `llvm-link`, `opt`, `llc`).
- **CUDA 12.9 wheels** (pip download, userspace) for `libnvvm` + `libdevice` + `ptxas`:
  - `nvidia-cuda-nvcc-cu12==12.9.86` → `~/nvvm-wheel/extracted/nvidia/cuda_nvcc/{nvvm/libdevice/libdevice.10.bc, bin/ptxas}`
  - `nvidia-nvjitlink-cu12==12.9.86` → `~/nvjit-wheel/...` (only needed for the libNVVM/nvJitLink cubin route; the llc+ptxas route below does not use it).

## 3. Build the backend `.so`  (`repro/build_backend.sh`)

```
cd ~/cuda-oxide-src/crates/rustc-codegen-cuda
CUDA_OXIDE_LLC=~/llvm21/bin/llc LIBCLANG_PATH=~/llvm21/lib \
LD_LIBRARY_PATH=~/llvm21/lib:/usr/lib/x86_64-linux-gnu \
cargo +nightly-2026-04-03 build
# -> crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so  (~5 s incremental)
```
Rebuild this after any mir-importer/collector edit (mir-importer is a path dep).

## 4. Build the cubins  (`repro/build_cubins.sh`)

Per shader crate: rustc (cuda-oxide backend) → `<crate>.ll` → libdevice link → cubin.
Key flags: `-Z codegen-backend=<.so> -Zalways-encode-mir -Zmir-enable-passes=-JumpThreading`
(`-JumpThreading` is REQUIRED — barrier-deadlock fix), `--target nvptx64-nvidia-cuda
-Z build-std=core`, `--no-default-features --features "cuda-oxide [dim3] unsafe_remove_boundchecks"`.
**nexus additionally needs `CUDA_OXIDE_ALLOW_PANIC=1`** (lets static-string bounds
panics through; they lower to `unreachable`).

```
ARCH=sm_120
TOOL=~/llvm21/bin
LIBDEV=~/nvvm-wheel/extracted/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
PTXAS=~/nvvm-wheel/extracted/nvidia/cuda_nvcc/bin/ptxas
$TOOL/llvm-as  $C.ll              -o /tmp/$C.bc
$TOOL/llvm-link /tmp/$C.bc $LIBDEV -o /tmp/${C}_linked.bc
$TOOL/opt -passes="internalize,globaldce" /tmp/${C}_linked.bc -o /tmp/${C}_pruned.bc
$TOOL/llc -mcpu=$ARCH -O3 /tmp/${C}_pruned.bc -o /tmp/$C.ptx
$PTXAS -arch=$ARCH -O3 /tmp/$C.ptx -o ~/nexus_ptx/$C.cubin
```
Produces `~/nexus_ptx/{vortx_shaders.cubin, nexus_rbd_shaders3d.cubin}`.

## 5. Build zealot with the cubins embedded  (`repro/build_zealot_cuda.sh`)

`khal-builder` reads `CUDA_OXIDE_SHADERS_PTX_<CRATE>` (crate dir upper-cased,
`-`→`_`); when set it embeds the cubin and **skips the spirv/cargo-gpu build**.
Use a clean env (NO codegen-backend RUSTFLAGS):

```
cd ~/Documents/work/zealot
CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=~/nexus_ptx/vortx_shaders.cubin \
CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=~/nexus_ptx/nexus_rbd_shaders3d.cubin \
cargo build --release --example biped_train_gpu --features "gpu biped_gpu cuda_backend"
```

## 6. Run

Model file required at
`~/Documents/work/lerobot-humanoid-design/to_real_robot/RL_policy/robot.xml`
(the `assets/` visual meshes are optional). Then:

```
BIPED_CUDA=1 ./target/release/examples/biped_train_gpu 3 1024 /tmp/policy.safetensors
```

Expected: `[biped] backend = native CUDA (cuda-oxide)`, a per-iter table with a
stable `torso_z` (~0.66, robot stands — physics contact works natively), the
reward breakdown, and `saved → …`.

## The seven backend fixes (this branch)

1. **glamx feature set** — keep `glamx 0.2.0` with `nostd-libm`+`bytemuck` only (repo-side).
2. **ZST struct construction** — recursive `build_zst_value` (`TryFromIntError(())`).
3. **Mangled call-names** — `extract_func_info` uses `instance.mangled_name()` for
   non-generic callees (fixes `global_invocation_id` symbol).
4. **`llvm.nvvm.*` dispatch** — surface the `#[link_name]` as `pattern_name`
   (PTX sreg reads from `khal_std`).
5. **`CUDA_OXIDE_ALLOW_PANIC`** — skip the over-conservative panic-formatting
   fatal check for static-string panics.
6. **Recursive const aggregates** — `const_type_size` + `build_const_value_from_bytes`
   (glam `Mat3::IDENTITY` / `Vec3::ZERO`, ~180 nexus sites).
7. **Fat-slice const → undef** — `&str` panic messages emit `MirUndefOp`.

Fixes 2–7 are in the commit; #1 is a repo-side manifest constraint (see `repro/khal.patch`).
