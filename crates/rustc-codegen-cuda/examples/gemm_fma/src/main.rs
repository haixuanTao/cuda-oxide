/*
 * FMA variant of the zealot vortx register-blocked GEMM on cuda-oxide.
 *
 * Same kernel as gemm_migrate (64x64 tile, 4x4 scalar register block, padded
 * shared memory) but the inner loop uses `f32::mul_add`, which lowers to a
 * libdevice `__nv_fmaf` -> a single `fma.rn.f32`. That pulls the kernel onto
 * the NVVM-IR flavor + `load_kernel_module` (libNVVM/nvJitLink) loader path,
 * launched with the `cuda_launch!` macro -- the same path the ELU migration
 * used. Measures the true-FMA ceiling vs the plain mul+add pure-PTX variant.
 */

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::{cuda_launch, load_kernel_module};
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

/// C = A * B,  A: M x K (row-major),  B: K x N (row-major),  C: M x N.
#[kernel]
pub fn sgemm_reg_fma(m: u32, n: u32, k: u32, a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
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

    // 16 scalar accumulators (not an array: mem2reg won't SROA `[f32; 16]`).
    let mut c00 = 0.0f32; let mut c01 = 0.0f32; let mut c02 = 0.0f32; let mut c03 = 0.0f32;
    let mut c10 = 0.0f32; let mut c11 = 0.0f32; let mut c12 = 0.0f32; let mut c13 = 0.0f32;
    let mut c20 = 0.0f32; let mut c21 = 0.0f32; let mut c22 = 0.0f32; let mut c23 = 0.0f32;
    let mut c30 = 0.0f32; let mut c31 = 0.0f32; let mut c32 = 0.0f32; let mut c33 = 0.0f32;

    let mut k_tile = 0usize;
    while k_tile < k {
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
                // mul_add -> __nv_fmaf -> fma.rn.f32 (true fused multiply-add).
                c00 = a0.mul_add(b0, c00); c01 = a0.mul_add(b1, c01); c02 = a0.mul_add(b2, c02); c03 = a0.mul_add(b3, c03);
                c10 = a1.mul_add(b0, c10); c11 = a1.mul_add(b1, c11); c12 = a1.mul_add(b2, c12); c13 = a1.mul_add(b3, c13);
                c20 = a2.mul_add(b0, c20); c21 = a2.mul_add(b1, c21); c22 = a2.mul_add(b2, c22); c23 = a2.mul_add(b3, c23);
                c30 = a3.mul_add(b0, c30); c31 = a3.mul_add(b1, c31); c32 = a3.mul_add(b2, c32); c33 = a3.mul_add(b3, c33);
                kk += 1;
            }
        }

        thread::sync_threads();
        k_tile += TILE_K;
    }

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

const M: usize = 1024;
const N: usize = 1024;
const K: usize = 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== cuda-oxide register-blocked GEMM, FMA/libdevice path ===");
    println!("{M}x{K} * {K}x{N} = {M}x{N}\n");

    let ctx = CudaContext::new(0)?;
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

    let a_dev = DeviceBuffer::from_host(&stream, &a)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b)?;
    let mut c_dev = DeviceBuffer::from_host(&stream, &c)?;

    let module = load_kernel_module(&ctx, "gemm_fma")?;

    let cfg = LaunchConfig {
        grid_dim: ((N as u32).div_ceil(TILE_N as u32), (M as u32).div_ceil(TILE_M as u32), 1),
        block_dim: (WG_N as u32, WG_N as u32, 1),
        shared_mem_bytes: 0,
    };

    // Warmup (also triggers the libNVVM/nvJitLink cubin build).
    cuda_launch! {
        kernel: sgemm_reg_fma, stream: stream, module: module, config: cfg,
        args: [M as u32, N as u32, K as u32, slice(a_dev), slice(b_dev), slice_mut(c_dev)]
    }?;
    stream.synchronize()?;

    const RUNS: u32 = 20;
    let start = Instant::now();
    for _ in 0..RUNS {
        cuda_launch! {
            kernel: sgemm_reg_fma, stream: stream, module: module, config: cfg,
            args: [M as u32, N as u32, K as u32, slice(a_dev), slice(b_dev), slice_mut(c_dev)]
        }?;
    }
    stream.synchronize()?;
    let avg_ms = start.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;
    let gflops = 2.0 * M as f64 * N as f64 * K as f64 / (avg_ms / 1000.0) / 1e9;
    println!("avg {avg_ms:.3} ms   {gflops:.1} GFLOPS");

    let c_res = c_dev.to_host_vec(&stream)?;
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
    assert!(max_err < 1e-3, "GEMM mismatch: {max_err}");
    println!("PASS: FMA register-blocked GEMM matches CPU reference");
    Ok(())
}
