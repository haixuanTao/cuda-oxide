/*
 * Minimal migration of the zealot vortx ELU kernel to cuda-oxide.
 * Mirrors the manual_launch_libdevice template: a #[kernel] that calls
 * libdevice (v.exp() -> __nv_expf), loaded via load_kernel_module (NVVM-IR
 * + libNVVM/nvJitLink libdevice link at first launch) and launched with the
 * cuda_launch! macro.
 */

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_launch, load_kernel_module};

#[kernel]
pub fn elu(mut a: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(x) = a.get_mut(idx) {
        let v = *x;
        *x = if v > 0.0 { v } else { v.exp() - 1.0 };
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== cuda-oxide ELU migration (zealot vortx kernel) ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = load_kernel_module(&ctx, "elu_migrate")?;

    const N: usize = 4096;
    let input: Vec<f32> = (0..N).map(|i| (i as f32 - 2048.0) * 0.01).collect();
    let mut a_dev = DeviceBuffer::from_host(&stream, &input)?;

    cuda_launch! {
        kernel: elu,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(N as u32),
        args: [slice_mut(a_dev)]
    }?;

    let out = a_dev.to_host_vec(&stream)?;

    let mut maxerr = 0.0f32;
    for (i, &o) in out.iter().enumerate() {
        let v = input[i];
        let expect = if v > 0.0 { v } else { v.exp() - 1.0 };
        maxerr = maxerr.max((o - expect).abs());
    }
    println!("N = {N}, max|gpu - cpu| = {maxerr:e}");
    assert!(maxerr < 1e-4, "ELU mismatch: {maxerr}");
    println!("PASS: cuda-oxide ELU matches CPU reference");
    Ok(())
}
