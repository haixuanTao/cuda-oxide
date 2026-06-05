/*
 * Migration of the zealot vortx register-blocked tiled GEMM to cuda-oxide.
 *
 * Faithful port of vortx-shaders gemm_tiled: 64x64 output tile per block,
 * 16x16 = 256 threads, TILE_K = 16, each thread computes a 4x4 sub-tile in
 * registers, A/B staged through padded shared memory (64x17, 16x65) to dodge
 * bank conflicts. The vortx kernel used vec4 FMAs purely as a WGSL/WebGPU
 * hint; on NVIDIA SIMT cores the lanes are scalar and ptxas vectorizes the
 * shared/global loads itself, so this port uses scalar 4x4 accumulators with
 * the identical memory-access pattern -- that tiling + register blocking is
 * the real performance lever.
 */

#![allow(clippy::needless_range_loop)]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;
use std::time::Instant;

const TILE_M: usize = 64;
const TILE_N: usize = 64;
const TILE_K: usize = 16;
const WG_N: usize = 16;
const THREAD_M: usize = TILE_M / WG_N; // 4
const THREAD_N: usize = TILE_N / WG_N; // 4
const SMEM_A_STRIDE: usize = TILE_K + 1; // 17
const SMEM_B_STRIDE: usize = TILE_N + 1; // 65
const SMEM_A_SIZE: usize = TILE_M * SMEM_A_STRIDE; // 1088
const SMEM_B_SIZE: usize = TILE_K * SMEM_B_STRIDE; // 1040

#[cuda_module]
mod kernels {
    use super::*;

    /// C = A * B,  A: M x K (row-major),  B: K x N (row-major),  C: M x N.
    #[kernel]
    pub fn sgemm_reg(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32>,
    ) {
        static mut SMEM_A: SharedArray<f32, SMEM_A_SIZE> = SharedArray::UNINIT;
        static mut SMEM_B: SharedArray<f32, SMEM_B_SIZE> = SharedArray::UNINIT;

        let tid_x = thread::threadIdx_x() as usize;
        let tid_y = thread::threadIdx_y() as usize;
        let linear_tid = tid_y * WG_N + tid_x;

        let tile_row = thread::blockIdx_y() as usize * TILE_M;
        let tile_col = thread::blockIdx_x() as usize * TILE_N;

        let m = m as usize;
        let n = n as usize;
        let k = k as usize;

        // 4 rows x 4 cols register accumulator. These are 16 *scalar* locals
        // on purpose: the codegen's mem2reg promotes scalar allocas to SSA
        // registers, but does NOT SROA a `[f32; 16]` array -- an array would
        // stay in local memory and the hot loop below would `ld.local`/
        // `st.local` every accumulate, collapsing throughput.
        let mut c00 = 0.0f32; let mut c01 = 0.0f32; let mut c02 = 0.0f32; let mut c03 = 0.0f32;
        let mut c10 = 0.0f32; let mut c11 = 0.0f32; let mut c12 = 0.0f32; let mut c13 = 0.0f32;
        let mut c20 = 0.0f32; let mut c21 = 0.0f32; let mut c22 = 0.0f32; let mut c23 = 0.0f32;
        let mut c30 = 0.0f32; let mut c31 = 0.0f32; let mut c32 = 0.0f32; let mut c33 = 0.0f32;

        let mut k_tile = 0usize;
        while k_tile < k {
            // Collaboratively stage the A (64x16) and B (16x64) tiles.
            // 256 threads, 1024 elems each -> 4 loads per thread.
            unsafe {
                let mut i = linear_tid;
                while i < TILE_M * TILE_K {
                    let row = i / TILE_K;
                    let col = i % TILE_K;
                    let gr = tile_row + row;
                    let gc = k_tile + col;
                    let val = if gr < m && gc < k { a[gr * k + gc] } else { 0.0 };
                    SMEM_A[row * SMEM_A_STRIDE + col] = val;
                    i += WG_N * WG_N;
                }
                let mut j = linear_tid;
                while j < TILE_K * TILE_N {
                    let row = j / TILE_N;
                    let col = j % TILE_N;
                    let gr = k_tile + row;
                    let gc = tile_col + col;
                    let val = if gr < k && gc < n { b[gr * n + gc] } else { 0.0 };
                    SMEM_B[row * SMEM_B_STRIDE + col] = val;
                    j += WG_N * WG_N;
                }
            }

            thread::sync_threads();

            // Each thread accumulates its 4x4 block from shared memory.
            unsafe {
                let a_row_base = tid_y * THREAD_M;
                let b_col_base = tid_x * THREAD_N;
                let mut kk = 0usize;
                while kk < TILE_K {
                    let a0 = SMEM_A[a_row_base * SMEM_A_STRIDE + kk];
                    let a1 = SMEM_A[(a_row_base + 1) * SMEM_A_STRIDE + kk];
                    let a2 = SMEM_A[(a_row_base + 2) * SMEM_A_STRIDE + kk];
                    let a3 = SMEM_A[(a_row_base + 3) * SMEM_A_STRIDE + kk];
                    let b0 = SMEM_B[kk * SMEM_B_STRIDE + b_col_base];
                    let b1 = SMEM_B[kk * SMEM_B_STRIDE + b_col_base + 1];
                    let b2 = SMEM_B[kk * SMEM_B_STRIDE + b_col_base + 2];
                    let b3 = SMEM_B[kk * SMEM_B_STRIDE + b_col_base + 3];
                    // Plain mul+add keeps this on the pure-PTX path. (`mul_add`
                    // would lower to a libdevice `__nv_fmaf` call, forcing the
                    // NVVM-IR flavor + libNVVM/nvJitLink loader instead.)
                    c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                    c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                    c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                    c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
                    kk += 1;
                }
            }

            thread::sync_threads();
            k_tile += TILE_K;
        }

        // Write the 4x4 block back to global memory. Fully unrolled with
        // CONSTANT `acc` indices: dynamic indexing of `acc` would force the
        // whole accumulator into local memory and spill it out of registers
        // (which silently kills throughput in the hot FMA loop above).
        let out_row = tile_row + tid_y * THREAD_M;
        let out_col = tile_col + tid_x * THREAD_N;
        macro_rules! store {
            ($di:expr, $dj:expr, $val:expr) => {{
                let row = out_row + $di;
                let col = out_col + $dj;
                if row < m && col < n {
                    unsafe { *c.get_unchecked_mut(row * n + col) = $val; }
                }
            }};
        }
        store!(0, 0, c00); store!(0, 1, c01); store!(0, 2, c02); store!(0, 3, c03);
        store!(1, 0, c10); store!(1, 1, c11); store!(1, 2, c12); store!(1, 3, c13);
        store!(2, 0, c20); store!(2, 1, c21); store!(2, 2, c22); store!(2, 3, c23);
        store!(3, 0, c30); store!(3, 1, c31); store!(3, 2, c32); store!(3, 3, c33);
    }
}

const M: usize = 1024;
const N: usize = 1024;
const K: usize = 1024;

fn main() {
    println!("=== cuda-oxide register-blocked GEMM (zealot vortx kernel) ===");
    println!("{M}x{K} * {K}x{N} = {M}x{N}\n");

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();

    let mut a = vec![0.0f32; M * K];
    let mut b = vec![0.0f32; K * N];
    let c = vec![0.0f32; M * N];
    for i in 0..M {
        for j in 0..K {
            a[i * K + j] = ((i + j) % 10) as f32 * 0.1;
        }
    }
    for i in 0..K {
        for j in 0..N {
            b[i * N + j] = ((i * j) % 10) as f32 * 0.1;
        }
    }

    let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b).unwrap();
    let mut c_dev = DeviceBuffer::from_host(&stream, &c).unwrap();

    let module = ctx
        .load_module_from_file("gemm_migrate.ptx")
        .expect("load ptx");
    let module = kernels::from_module(module).expect("typed module");

    let cfg = LaunchConfig {
        grid_dim: ((N as u32).div_ceil(TILE_N as u32), (M as u32).div_ceil(TILE_M as u32), 1),
        block_dim: (WG_N as u32, WG_N as u32, 1),
        shared_mem_bytes: 0,
    };

    // Warmup
    module
        .sgemm_reg((stream).as_ref(), cfg, M as u32, N as u32, K as u32, &a_dev, &b_dev, &mut c_dev)
        .unwrap();
    stream.synchronize().unwrap();

    const RUNS: u32 = 20;
    let start = Instant::now();
    for _ in 0..RUNS {
        module
            .sgemm_reg((stream).as_ref(), cfg, M as u32, N as u32, K as u32, &a_dev, &b_dev, &mut c_dev)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let avg_ms = start.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;
    let gflops = 2.0 * M as f64 * N as f64 * K as f64 / (avg_ms / 1000.0) / 1e9;
    println!("avg {avg_ms:.3} ms   {gflops:.1} GFLOPS");

    let c_res = c_dev.to_host_vec(&stream).unwrap();
    let mut max_err = 0.0f32;
    for sample in 0..200 {
        let idx = sample * M * N / 200;
        let (row, col) = (idx / N, idx % N);
        let mut exp = 0.0f32;
        for kk in 0..K {
            exp += a[row * K + kk] * b[kk * N + col];
        }
        max_err = max_err.max((c_res[idx] - exp).abs());
    }
    println!("max err vs CPU: {max_err:.3e}");
    if max_err < 1e-3 {
        println!("PASS: register-blocked GEMM matches CPU reference");
    } else {
        println!("FAIL");
        std::process::exit(1);
    }
}
