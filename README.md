# forge-1024

**AI 微指令内核 · 10 域 · 1024 条可验证微指令**

[![CI](https://github.com/CJX0712/forge-1024/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/forge-1024/actions/workflows/ci.yml)
[![zero dependency](https://img.shields.io/badge/dependencies-0-brightgreen)](#零依赖)
[![node](https://img.shields.io/badge/node-%3E%3D18-blue)](#快速开始)

> 零依赖 · 零构建 · 可复现 · 每条指令自带论文出处与可验证不变量。
> 作者：**晨星**（[CJX0712](https://github.com/CJX0712)）

---

## 一句话

`forge-1024` 把 AI 的全部基础能力拆成 **1024 条微指令**（micro-instruction）：从 `linalg.cholesky` 到 `dl.attn.rotaryApply`，每条都是纯函数，携带 **签名 / 中文文档 / 论文出处 / 可验证不变量**，可被统一调度、遥测、模糊测试与文档反射。整个内核 **0 个 npm 依赖**，Node ≥ 18 直接跑。

```bash
git clone https://github.com/CJX0712/forge-1024.git
cd forge-1024
node tools/selftest.mjs     # 编目自检：1024 条硬金标准
node --test                 # 全量单元测试
node examples/04-mlp-xor.mjs
```

---

## 为什么是「微指令」

传统 AI 框架把能力封装在算子库里，使用者只看见 API，看不见**正确性契约**。`forge-1024` 反过来：

| 传统算子库 | forge-1024 微指令 |
|---|---|
| 文档由人写，易与实现漂移 | 文档由**注册表反射生成**（`tools/gen-ops-doc.mjs`），永不漂移 |
| 正确性靠框架作者保证 | 每条指令自带 `inv`（可验证不变量），由测试**独立交叉验证** |
| 黑盒算子，无法追溯来源 | 每条指令自带 `ref`（论文/标准/权威实现），可追溯到原始文献 |
| 依赖庞大（CUDA、BLAS、protobuf…） | **0 依赖**、纯 JS、Float64Array 后端 |
| 随机数不可控，结果难复现 | 全局禁用 `Math.random`，一律 PCG32 确定性 RNG |

设计基准：ONNX Operator Standard 的「命名 + 签名 + 文档 + 版本化」组织方式、PyTorch `aten` 的注册模式、JAX 的可组合变换思想。

---

## 十层域（6×102 + 4×103 = 1024）

| 域 | 代号 | 名称 | 条数 | 内容 |
|---|---|---|---:|---|
| D1 | MATH | 数学基元 | 102 | 逐元素运算、归约、范数、距离、构造器 |
| D2 | NUM | 数值方法 | 102 | 稳定 softmax/logsumexp、Kahan/Neumaier 补偿求和、求根、积分、有限差分、特殊函数 |
| D3 | LINALG | 线性代数 | 102 | LU/QR/Cholesky/Jacobi 特征/SVD、迭代求解器、矩阵函数、Toeplitz/三对角 |
| D4 | AUTODIFF | 自动微分 | 102 | 磁带式反向模式 vjp、对偶数前向模式、梯度检验 |
| D5 | OPT | 优化器 | 102 | 16 个经典优化器、12 种 LR 调度、梯度变换、线搜索 |
| D6 | PROB | 概率统计 | 102 | 17 个分布族、信息论、贝叶斯推断、采样、拟合检验 |
| D7 | ML | 机器学习 | 103 | 回归/分类/树/集成/SVM/聚类/降维/预处理全流程 |
| D8 | DL | 深度学习 | 103 | 层、激活、归一化、注意力（MHA/GQA/RoPE/KV-Cache）、RNN/LSTM/GRU、损失、初始化 |
| D9 | TRAIN | 训练与评估 | 103 | 数据管线、训练循环、交叉验证、早停、可复现性与实验指纹 |
| D10 | SYS | 系统编排 | 103 | 注册表内省、遥测、执行组合、验证器、文档生成、基准、图编排、模糊测试 |

> 编目规约：6 个基础域各承载 **102** 条；能力面最广的 4 个域（ML / DL / TRAIN / SYS）各多承载 **1** 条，合计 **6×102 + 4×103 = 1024**。

依赖分层：`D1 ← D2 ← D3 ← D4 ← {D5, D6} ← D7 ← D8 ← D9 ← D10`，D10 只依赖注册表，因此**编目能力可用于自检自身**。

完整指令清单见 **[docs/OPS.md](docs/OPS.md)**（自动生成）。

---

## 快速开始

```js
import './src/index.mjs';                    // 导入即注册全部 1024 条
import { exec } from './src/core/registry.mjs';
import { Tensor } from './src/core/tensor.mjs';

// 1) 线性代数：Cholesky 解正定系统
const A = Tensor.fromFlat([4, 1, 1, 3], [2, 2]);
const b = Tensor.fromFlat([1, 2], [2]);
exec('linalg.solveSPD', [A, b]);             // → [0.0909, 0.6364]

// 2) 自动微分：反向模式 + 梯度检验
exec('ad.tapeReset', []);
const x = Tensor.fromFlat([1, 2, 3], [3]);
const loss = exec('ad.sumAll', [exec('ad.mul', [x, x])]);
const g = exec('ad.backward', [loss]);       // → Map{ x → [2,4,6] }

// 3) 概率：多元正态密度 + 稳定 logsumexp
exec('prob.normal.cdf', [1.96]);             // → 0.975…

// 4) 深度学习：RoPE 相对位置编码 / 多头自注意力
exec('dl.attn.selfAttention', [x3, 8]);   // x3: Tensor[batch, T, C]，heads=8

// 5) 系统：把整个编目反射成 Markdown
exec('sys.doc.catalogMarkdown', [listOps()]);
```

更多可运行示例见 [`examples/`](examples/)（6 个）。

---

## 验证体系（不是「跑通就行」）

每一条指令至少有一条**可被独立实现交叉验证的硬不变量**：

| 不变量类型 | 例子 |
|---|---|
| 解析解 vs 数值解 | 线性回归：闭式解 = Cholesky 解 = SVD 解 = QR 解（示例 01） |
| 双路径一致性 | PCA：Jacobi 特征分解谱 == SVD 谱 |
| 定理级性质 | EM：对数似然单调不减（Dempster et al. 1977 定理 1） |
| 梯度检验 | 解析梯度 vs 中心差分，相对误差 < 1e-6 |
| 结构不变量 | 正交矩阵 `QᵀQ = I`、协方差对称半正定、注意力权重行随机 |
| 概率公理 | 分布质量/密度积分为 1、`KL(p‖q) ≥ 0` |
| 计算等价 | KV-Cache 增量解码 == 一次性全量因果注意力（逐位一致） |

```bash
node tools/selftest.mjs    # 编目级：1024 条 / 每域 102(ML/DL/TRAIN/SYS=103) / id 连续 / name 唯一 / 元数据齐全 / arity 推导一致
node --test                 # 域级：每域不变量 + 全算子梯度检验 + 收敛性证明
node bench/bench.mjs       # 性能基准（生成 docs/BENCHMARKS.md）
node tools/verify-bundle.mjs  # 单文件产物自洽性
```

细节见 **[docs/VERIFICATION.md](docs/VERIFICATION.md)**。

---

## 性能

零依赖纯 JS并不意味着「玩具」：矩阵乘采用**分块（cache-blocking）**，分解族采用 Householder / Jacobi / 单边 Jacobi 等经典稳定算法，求和采用 **Kahan / Neumaier / pairwise** 补偿。基准数据与方文见 **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**。

---

## 浏览器内自检（单文件）

```bash
node tools/bundle-html.mjs    # → dist/forge-1024.html（无外部依赖，双击即开）
```

产物把全部源码以 **Blob ESM** 方式内联（按依赖拓扑序重写 import 说明符，不改写任何一行源码），打开后即可在浏览器里跑编目自检、检索 1024 条指令、在线调用任意指令。

---

## 目录结构

```
forge-1024/
├── src/
│   ├── core/          内核：Tensor / RNG(PCG32) / 注册表 / 遥测 / 错误分类学
│   │   ├── tensor.mjs       广播与归约（NumPy 语义）
│   │   ├── rng.mjs          PCG32 确定性随机（禁止 Math.random）
│   │   ├── registry.mjs     defineOp / exec / tryExec / 遥测
│   │   ├── errors.mjs       ForgeError + CODES 错误分类学
│   │   └── telemetry.mjs    调用计数 / 耗时 / 追踪
│   ├── ops/           10 个域实现，每域一个文件（D1–D6 各 102 条，D7–D10 各 103 条）
│   └── index.mjs      顶层入口（导入即注册 1024 条）
├── test/              域级测试 + 编目级测试
├── bench/             微基准
├── tools/             selftest / gen-ops-doc / bundle-html / verify-bundle
├── examples/          6 个可运行示例
├── docs/              ARCHITECTURE / OPS / VERIFICATION / BENCHMARKS / REFERENCES / ROADMAP
└── dist/              单文件 HTML 产物
```

---

## 设计原则

1. **零重写（集大成）**：不重新发明轮子。每条指令对标世界级实现与原始论文，`ref` 字段强制记录出处；凡存在更强的成熟方案，以该方案为准（如 gamma 用 Stirling 级数而非 Lanczos 近似，SVD 用单边 Jacobi 而非朴素幂法）。
2. **可验证优先**：先写不变量，再写实现。没有不变量的指令不允许进编目。
3. **确定性**：全局禁用 `Math.random`，一切随机来自可播种的 PCG32，实验可逐位复现。
4. **零依赖**：只用 Node 内置能力；不引入任何 npm 包、不引入构建步骤。
5. **文档即代码**：文档由注册表反射生成，杜绝漂移。

---

## 文档

| 文档 | 内容 |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 内核架构、分层依赖、不变量体系、错误分类学 |
| [docs/OPS.md](docs/OPS.md) | 1024 条指令全集（自动生成，含签名/不变量） |
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | 验证方法学与不变量清单 |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | 性能基准与方法学 |
| [docs/REFERENCES.md](docs/REFERENCES.md) | 参考文献聚合（自动生成） |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 演进路线 |

---

## 许可

MIT © 晨星
