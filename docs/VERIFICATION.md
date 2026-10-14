# 验证体系（VERIFICATION）

> 目标：1024 条微指令**每一条都有可被独立实现交叉验证的硬不变量**。
> 原则：**先写不变量，再写实现**；没有不变量的指令不允许进编目。

---

## 1. 三层验证

| 层级 | 命令 | 断言什么 |
|---|---|---|
| **编目层** | `node tools/selftest.mjs` | 1024 条 / 每域 102(ML·DL·TRAIN·SYS=103) / id 连续唯一 / name 唯一且带域前缀 / doc·ref·inv·sig 齐全 / arity 由 impl 必需参数推导一致 / 错误码齐备 |
| **域层** | `node --test` | 每域的**数学不变量**、全算子梯度检验、优化器收敛性、分解结果代数结构 |
| **产物层** | `node tools/verify-bundle.mjs` | 单文件 HTML 自洽（无外链、1024 条指令真实注册并执行） |

CI（`.github/workflows/ci.yml`）在 **3 个操作系统 × 3 个 Node 版本**矩阵上运行上述三层 + 示例冒烟 + 文档生成 + 基准。

---

## 2. 不变量分类与实例

### 2.1 解析解 ↔ 数值解

| 指令族 | 不变量 | 交叉验证方式 |
|---|---|---|
| 最小二乘 | `w = (XᵀX)⁻¹Xᵀy` | Cholesky 解 == SVD 解 == QR 解（示例 01，差异 < 1e-9） |
| 正规方程最优性 | 残差 ⊥ 列空间 | `‖Xᵀ(y − Xŵ)‖∞ < 1e-8` |
| 矩阵指数 | `e^A` | 与 Pade scaling-squaring 对照，且 `e^{A}·e^{−A} = I` |

### 2.2 双路径一致性

| 指令族 | 路径 A | 路径 B |
|---|---|---|
| 谱分解 | 对称 Jacobi 特征分解 | SVD（Eckart–Young） |
| 求逆 | LU 分解 | QR / Cholesky（SPD 情形） |
| 梯度 | 反向模式 vjp | 中心差分（gradcheck，相对误差 < 1e-6） |
| 导数 | 对偶数前向模式 | 有限差分 |

### 2.3 定理级性质

- **EM 单调性**（Dempster, Laird & Rubin 1977 定理 1）：GMM 迭代的对数似然单调不减（示例 02）。
- **Eckart–Young–Mirsky**：截断 SVD 是最优低秩逼近，残差 = Σ_{i>k} λᵢ。
- **PCA 三条**：谱一致、投影后主成分解相关（协方差非对角元 ≈ 0）、白化后协方差 = I（示例 03）。
- **优化器收敛**：16 个优化器在凸二次 `f(x)=½Σi·xᵢ²` 上收敛且（无记忆自适应法）损失单调不增。
- **Adam 公式核对**：前 3 步与 Kingma & Ba Algorithm 1 逐位核对（含偏差修正 `1−β^t`）。

### 2.4 代数结构不变量

| 分解 | 不变量 |
|---|---|
| LU | `P·A = L·U`，L 单位下三角 |
| QR | `QᵀQ = I`，`QR = A`，R 上三角 |
| Cholesky | `LLᵀ = A`，L 下三角且对角元全正 |
| 特征分解（对称） | `Av = λv`，特征向量正交归一（误差 ~1e-16） |
| SVD | `A = UΣVᵀ`，U/V 列正交，Σ 非负降序 |
| 极分解 | `A = UP`，U 正交、P 半正定 |

### 2.5 概率公理

- 分布：PMF 求和 = 1 / PDF 积分 = 1（数值积分核对）。
- `KL(p‖q) ≥ 0` 且 `= 0 ⟺ p = q`；JS 散度有界 `≤ log 2`。
- GMM：责任度每行和 = 1、混合权重和 = 1、协方差对称正定。
- 采样：大样本矩收敛到解析矩（`sampleMoments` vs 理论值）。

### 2.6 计算等价（优化正确性）

- **KV-Cache**：逐 token 增量解码 == 一次性全量因果注意力（逐位 < 1e-10）。
- **RoPE**：`⟨RoPE(q,m), RoPE(k,n)⟩` 只依赖相对位移 `m − n`。
- **因果掩码**：下三角（含对角）= 1，严格上三角 = 0；加掩码后上三角注意力权重为 0。
- **分块 matmul**：分块实现 == 朴素三重循环（小尺寸逐位一致）。

### 2.7 数值稳定性

- Kahan / Neumaier 补偿求和 vs 朴素求和在 `[1e16, 1, −1e16]` 上的行为被**如实记录**并断言（浮点真值不可表示，测试记录实际 IEEE-754 行为而非数学真值）。
- softmax / logsumexp 平移不变性：加任意常数结果不变。
- `log(0)`、`0/0`、空数组等边界一律有定义（抛 `ForgeError` 或返回约定的单位元）。

---

## 3. 模糊测试与反射校验（D10）

D10 让「编目可以验证自己」：

| 指令 | 作用 |
|---|---|
| `sys.verify.catalog` | 编目级不变量全量检查 |
| `sys.verify.determinism` | 同参调用 n 次结果逐位相同 |
| `sys.verify.fuzz` | 用 PCG32 生成随机构造入参反复调用，断言只抛 `ForgeError` |
| `sys.verify.arityCheck` | arity 与 impl 一致性 |
| `sys.verify.docCoverage` / `refCoverage` | doc / ref 覆盖率 |
| `sys.verify.idContinuity` / `nameUnique` | id 连续与 name 唯一 |
| `sys.verify.errorTaxonomy` | 错误分类学完备性 |

---

## 4. 确定性与可复现

1. 全局禁用 `Math.random`，一律 `createRng(seed)`（PCG32）。
2. 指令 state 为纯对象 + 普通数组，可 JSON 序列化与回放。
3. D9 提供 `train.repro.fingerprint / manifest / manifestHash / assertReproducible`：同种子 → 同指纹 → 同结果。

---

## 5. 失败即失败：不放宽断言的纪律

测试若失败，**默认修实现**。只有三种情况允许调整断言，且必须在测试注释里写明理由：

1. **算法固有性质**（如 Adadelta 的 RMS(Δx) 正反馈导致最优点附近缓慢振荡；signSGD 步长 ±η 只收敛到半径 ~η 的邻域）。
2. **浮点真值不可表示**（如 `1e16 + 1` 在双精度下等于 `1e16`）。
3. **启发式初始化**（如 k-means++ 的随机性，改为固定 seed 后断言）。

每一处放宽都在测试代码中以注释形式记录了对应的**理论依据**（论文/定理），而不是「让它变绿」。
