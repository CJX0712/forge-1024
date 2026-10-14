# 性能基准（BENCHMARKS）

> 由 `node bench/bench.mjs` 生成 · 2026-10-14T18:19:41.761Z
> 环境：Node 22.22.2 / win32 x64 · 零依赖 · 单线程 · Float64Array 后端

| 类别 | 项目 | 耗时 (ms) | 吞吐 (ops/s) | 备注 |
|---|---|---:|---:|---|
| elementwise | add  1000000 元素 | 14.1414 | 71 |  |
| elementwise | mul  1000000 元素 | 20.4874 | 49 |  |
| elementwise | exp  1000000 元素 | 7.9961 | 125 |  |
| elementwise | sum  1000000 元素 | 10.7689 | 93 |  |
| elementwise | Kahan 求和 1000000 | 3.6608 | 273 |  |
| matmul | matmul 64×64 | 0.3276 | 3052 | 1.6 GFLOP/s |
| matmul | matmul 256×256 | 23.0055 | 43 | 1.459 GFLOP/s |
| matmul | matmul 512×512 | 152.2647 | 7 | 1.763 GFLOP/s |
| decomp | lu 64×64 | 0.3251 | 3076 |  |
| decomp | qr 64×64 | 1.0683 | 936 |  |
| decomp | chol 64×64 | 0.0637 | 15704 |  |
| decomp | eigSym 64×64 | 48.0179 | 21 |  |
| decomp | lu 200×200 | 2.9638 | 337 |  |
| decomp | qr 200×200 | 14.9323 | 67 |  |
| decomp | chol 200×200 | 1.4782 | 676 |  |
| decomp | eigSym 200×200 | 421.1291 | 2 |  |
| decomp | lu 400×400 | 22.877 | 44 |  |
| decomp | qr 400×400 | 133.452 | 7 |  |
| decomp | chol 400×400 | 11.8894 | 84 |  |
| decomp | eigSym 400×400 | 4675.6441 | 0 |  |
| autodiff | tape forward+backward (32×128) | 5.9407 | 168 |  |
| optimizer | adam.step (4096 维) | 0.3559 | 2810 |  |

## 方法学

- 每项先预热 3 次再重复计时，取平均单次耗时，避免 JIT 冷启动污染。
- `linalg.matmul` 采用分块（cache-blocking）实现，GFLOP/s 按 2·m·n·k 计算。
- 计时使用 `process.hrtime.bigint()`（纳秒分辨率），不引入任何第三方基准库。
- 遥测默认关闭计时，基准结果不被内核统计开销污染。
