# gemm_migrate

A register-blocked tiled SGEMM ported from the [vortx](https://github.com/dimforge/vortx)
tensor library to a cuda-oxide `#[kernel]`: 64×64 output tile per block, 16×16 =
256 threads, `TILE_K = 16`, each thread computing a 4×4 sub-tile in registers,
with A/B staged through padded shared memory (64×17, 16×65) to dodge bank
conflicts.

Pure arithmetic (`a*b + c`), so it stays on the **PTX path** (`load_module_from_file`)
— no libdevice, no libNVVM/nvJitLink needed.

```
cargo oxide run gemm_migrate
```

1024³ SGEMM on an RTX 5090: **~10.5 TFLOPS**, bit-exact vs CPU.

## A codegen gotcha worth knowing

The accumulator is **16 scalar locals**, not a `[f32; 16]` array, on purpose: the
codegen's mem2reg promotes scalar allocas to registers but does not SROA an
array, so an array accumulator spills to local memory (`__local_depot`,
`ld.local`/`st.local` in the hot loop) and throughput collapses (~1.2 TFLOPS, an
8.8× regression). See `gemm_fma` for the FMA variant.
