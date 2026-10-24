# M0 layer benchmark raw results

- Run (UTC): 2026-10-24T03:01:45+00:00
- Host: Linux-7.0.0-38-generic-x86_64-with-glibc2.43
- Device: CPU
- Command: `bench/m0_layer_bench.py --layer-bench build/cpu/neuralnet_cpp/layer_bench --json-out docs/benchmarks/m0-layer-bench-cpu.json --markdown-out docs/benchmarks/m0-layer-bench-cpu.md`
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
| b-small | inference | conv2d | 64 | 23.623 | 19.2 |  |  |
| b-small | inference | transformer | 64 | 264.325 | 63.1 |  |  |
| b-small | inference | linear | 64 | 0.199 | 44.8 |  |  |
| b-small | training | conv2d | 256 | 108.470 | 16.7 | 237.261 | 1079 |
| b-small | training | transformer | 256 | 1086.765 | 208.1 | 2490.173 | 6682 |
| b-small | training | linear | 256 | 0.778 | 45.9 | 2.208 | 115964 |
| b-final | inference | conv2d | 64 | 21.008 | 21.5 |  |  |
| b-final | inference | transformer | 64 | 356.329 | 66.6 |  |  |
| b-final | inference | linear | 64 | 0.228 | 52.1 |  |  |
| b-final | training | conv2d | 256 | 130.757 | 13.8 | 271.475 | 943 |
| b-final | training | transformer | 256 | 1395.178 | 220.5 | 3144.687 | 5291 |
| b-final | training | linear | 256 | 0.736 | 64.6 | 2.162 | 118422 |

## Component-sum estimate

> This is a component-sum baseline, not a measured end-to-end GoModel:
> `conv2d + N * transformer + linear_head`. Conv-stem/ResBlock count and
> feature planes remain M3/M7 decisions. Replace this estimate with the
> real model benchmark when those decisions land.

| Profile | Phase | Batch | Est. total ms | Est. positions/s |
|---|---|---:|---:|---:|
| b-small | inference | 64 | 2138.422 | 29.9 |
| b-small | training | 256 | 20160.853 | 12.7 |
| b-final | inference | 64 | 4297.184 | 14.9 |
| b-final | training | 256 | 38009.881 | 6.7 |

## Raw transcripts

```text
$ build/cpu/neuralnet_cpp/layer_bench --layer conv2d,transformer,linear --batch 64 --seq 65 --dmodel 192 --heads 8 --dff 768 --in 192 --out 363 --cin 17 --cout 64 --kernel 3 --hw 19 --iter 5 --warmup 2 --forward-only
=== layer_bench (CPU, forward-only) ===
  conv2d         : fwd   23.623 ms       19.2 GFLOPS
  transformer    : fwd  264.325 ms       63.1 GFLOPS
  linear         : fwd    0.199 ms       44.8 GFLOPS
$ build/cpu/neuralnet_cpp/layer_bench --layer conv2d,transformer,linear --batch 256 --seq 65 --dmodel 192 --heads 8 --dff 768 --in 192 --out 363 --cin 17 --cout 64 --kernel 3 --hw 19 --iter 5 --warmup 2
=== layer_bench (CPU, fwd+bwd) ===
  conv2d         : fwd  108.470 ms       16.7 GFLOPS  |  train  237.261 ms       1079 tok/s
  transformer    : fwd 1086.765 ms      208.1 GFLOPS  |  train 2490.173 ms       6682 tok/s
  linear         : fwd    0.778 ms       45.9 GFLOPS  |  train    2.208 ms     115964 tok/s
$ build/cpu/neuralnet_cpp/layer_bench --layer conv2d,transformer,linear --batch 64 --seq 65 --dmodel 256 --heads 8 --dff 1024 --in 256 --out 363 --cin 17 --cout 64 --kernel 3 --hw 19 --iter 5 --warmup 2 --forward-only
=== layer_bench (CPU, forward-only) ===
  conv2d         : fwd   21.008 ms       21.5 GFLOPS
  transformer    : fwd  356.329 ms       66.6 GFLOPS
  linear         : fwd    0.228 ms       52.1 GFLOPS
$ build/cpu/neuralnet_cpp/layer_bench --layer conv2d,transformer,linear --batch 256 --seq 65 --dmodel 256 --heads 8 --dff 1024 --in 256 --out 363 --cin 17 --cout 64 --kernel 3 --hw 19 --iter 5 --warmup 2
=== layer_bench (CPU, fwd+bwd) ===
  conv2d         : fwd  130.757 ms       13.8 GFLOPS  |  train  271.475 ms        943 tok/s
  transformer    : fwd 1395.178 ms      220.5 GFLOPS  |  train 3144.687 ms       5291 tok/s
  linear         : fwd    0.736 ms       64.6 GFLOPS  |  train    2.162 ms     118422 tok/s
```
