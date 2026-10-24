# M0 layer benchmark raw results

- Run (UTC): 2026-10-24T03:09:07+00:00
- Host: Windows-11-10.0.26100-SP0
- Device: Vulkan GPU
- Command: `bench\m0_layer_bench.py --layer-bench build\gpu\neuralnet_cpp\layer_bench.exe --gpu --json-out docs\benchmarks\m0-layer-bench-gpu.json --markdown-out docs\benchmarks\m0-layer-bench-gpu.md`
- Board: 19x19
- Sequence: `K=64` history moves + 1 decision query = 65
- Conv input: 17 feature planes (provisional until M3 freezes features)
- Head count: 8 (provisional until M7 freezes the model)
- Profiled layers: conv2d, transformer, linear head
- Batch: inference 64 positions; training 256 positions
- Train column = forward + backward; optimizer update is excluded

## Raw layer results

| Profile | Phase | Layer | Batch | Fwd ms | Fwd GFLOPS | Train ms | Train tok/s |
|---|---|---|---:|---:|---:|---:|---:|
| b-small | inference | conv2d | 64 | 7.260 | 62.3 |  |  |
| b-small | inference | transformer | 64 | 7.818 | 2132.7 |  |  |
| b-small | inference | linear | 64 | 0.166 | 53.6 |  |  |
| b-small | training | conv2d | 256 | 2.206 | 820.5 | 7.991 | 32034 |
| b-small | training | transformer | 256 | 13.886 | 16289.3 | 45.047 | 369394 |
| b-small | training | linear | 256 | 0.156 | 228.3 | 0.760 | 337019 |
| b-final | inference | conv2d | 64 | 0.966 | 468.2 |  |  |
| b-final | inference | transformer | 64 | 9.128 | 2600.0 |  |  |
| b-final | inference | linear | 64 | 0.154 | 77.0 |  |  |
| b-final | training | conv2d | 256 | 2.265 | 799.1 | 7.886 | 32462 |
| b-final | training | transformer | 256 | 17.123 | 17963.0 | 57.509 | 289345 |
| b-final | training | linear | 256 | 0.168 | 283.2 | 0.834 | 306881 |

## Component-sum estimate

> This is a component-sum baseline, not a measured end-to-end GoModel:
> `conv2d + N * transformer + linear_head`. Conv-stem/ResBlock count and
> feature planes remain M3/M7 decisions. Replace this estimate with the
> real model benchmark when those decisions land.

| Profile | Phase | Batch | Est. total ms | Est. positions/s |
|---|---|---:|---:|---:|
| b-small | inference | 64 | 69.970 | 914.7 |
| b-small | training | 256 | 369.127 | 693.5 |
| b-final | inference | 64 | 110.656 | 578.4 |
| b-final | training | 256 | 698.828 | 366.3 |

## Raw transcripts

```text
$ build\gpu\neuralnet_cpp\layer_bench.exe --layer conv2d,transformer,linear --batch 64 --seq 65 --dmodel 192 --heads 8 --dff 768 --in 192 --out 363 --cin 17 --cout 64 --kernel 3 --hw 19 --iter 5 --warmup 2 --gpu --forward-only
GPU 加速已启用 (Vulkan GpuEngine, GPU: NVIDIA CMP 40HX)

=== layer_bench (GPU, forward-only) ===
  conv2d         : fwd    7.260 ms       62.3 GFLOPS
  transformer    : fwd    7.818 ms     2132.7 GFLOPS
  linear         : fwd    0.166 ms       53.6 GFLOPS
$ build\gpu\neuralnet_cpp\layer_bench.exe --layer conv2d,transformer,linear --batch 256 --seq 65 --dmodel 192 --heads 8 --dff 768 --in 192 --out 363 --cin 17 --cout 64 --kernel 3 --hw 19 --iter 5 --warmup 2 --gpu
GPU 加速已启用 (Vulkan GpuEngine, GPU: NVIDIA CMP 40HX)

=== layer_bench (GPU, fwd+bwd) ===
  conv2d         : fwd    2.206 ms      820.5 GFLOPS  |  train    7.991 ms      32034 tok/s
  transformer    : fwd   13.886 ms    16289.3 GFLOPS  |  train   45.047 ms     369394 tok/s
  linear         : fwd    0.156 ms      228.3 GFLOPS  |  train    0.760 ms     337019 tok/s
$ build\gpu\neuralnet_cpp\layer_bench.exe --layer conv2d,transformer,linear --batch 64 --seq 65 --dmodel 256 --heads 8 --dff 1024 --in 256 --out 363 --cin 17 --cout 64 --kernel 3 --hw 19 --iter 5 --warmup 2 --gpu --forward-only
GPU 加速已启用 (Vulkan GpuEngine, GPU: NVIDIA CMP 40HX)

=== layer_bench (GPU, forward-only) ===
  conv2d         : fwd    0.966 ms      468.2 GFLOPS
  transformer    : fwd    9.128 ms     2600.0 GFLOPS
  linear         : fwd    0.154 ms       77.0 GFLOPS
$ build\gpu\neuralnet_cpp\layer_bench.exe --layer conv2d,transformer,linear --batch 256 --seq 65 --dmodel 256 --heads 8 --dff 1024 --in 256 --out 363 --cin 17 --cout 64 --kernel 3 --hw 19 --iter 5 --warmup 2 --gpu
GPU 加速已启用 (Vulkan GpuEngine, GPU: NVIDIA CMP 40HX)

=== layer_bench (GPU, fwd+bwd) ===
  conv2d         : fwd    2.265 ms      799.1 GFLOPS  |  train    7.886 ms      32462 tok/s
  transformer    : fwd   17.123 ms    17963.0 GFLOPS  |  train   57.509 ms     289345 tok/s
  linear         : fwd    0.168 ms      283.2 GFLOPS  |  train    0.834 ms     306881 tok/s
```
