# forge-1024 指令全集（OPS）

> 本文件由 `node tools/gen-ops-doc.mjs` 从注册表反射生成，**请勿手工编辑**。
> 内核 1.0.0 / 编目 2026.10 / 指令总数 **1024**。

## 目录

- [D1 · MATH 数学基元（102 条）](#d1--math) — 逐元素运算、归约、范数、距离、构造器
- [D2 · NUM 数值方法与稳定化（102 条）](#d2--num) — 稳定softmax、补偿求和、误差度量、求根、积分、差分、特殊函数
- [D3 · LINALG 线性代数（102 条）](#d3--linalg) — 分解族（LU/QR/Cholesky/Jacobi/SVD）、求解器、迭代法、矩阵函数
- [D4 · AUTODIFF 自动微分（102 条）](#d4--autodiff) — 反向模式 vjp、前向模式对偶数、梯度检验
- [D5 · OPT 优化器（102 条）](#d5--opt) — 16 种经典优化器、学习率调度、梯度变换、线搜索
- [D6 · PROB 概率统计（102 条）](#d6--prob) — 分布族、信息论、矩估计、贝叶斯推断、采样与拟合检验
- [D7 · ML 机器学习（102 条）](#d7--ml) — 经典模型族、评估指标、数据预处理
- [D8 · DL 深度学习（102 条）](#d8--dl) — 层、激活、注意力、循环网络、损失、初始化
- [D9 · TRAIN 训练与评估（102 条）](#d9--train) — 数据管线、训练循环、交叉验证、可复现性
- [D10 · SYS 系统编排（102 条）](#d10--sys) — 注册表内省、遥测、组合、文档生成、基准、模糊测试

---

## D1 · MATH 数学基元

逐元素运算、归约、范数、距离、构造器。共 102 条。

| id | 指令 | 签名 | 说明 | 不变量 |
|---|---|---|---|---|
| `D1-001` | `math.add` | `(Tensor a, Tensor b) -> Tensor` | 逐元素加法（NumPy 广播） | out[i] = add(a[i], b[i])；广播规则同 NumPy |
| `D1-002` | `math.addScalar` | `(Tensor a, number s) -> Tensor` | 张量与标量逐元素加法（右标量） | out[i] = add(a[i], s) |
| `D1-003` | `math.scalarAdd` | `(number s, Tensor a) -> Tensor` | 标量与张量逐元素加法（左标量） | out[i] = add(s, a[i]) |
| `D1-004` | `math.sub` | `(Tensor a, Tensor b) -> Tensor` | 逐元素减法（NumPy 广播） | out[i] = sub(a[i], b[i])；广播规则同 NumPy |
| `D1-005` | `math.subScalar` | `(Tensor a, number s) -> Tensor` | 张量与标量逐元素减法（右标量） | out[i] = sub(a[i], s) |
| `D1-006` | `math.scalarSub` | `(number s, Tensor a) -> Tensor` | 标量与张量逐元素减法（左标量） | out[i] = sub(s, a[i]) |
| `D1-007` | `math.mul` | `(Tensor a, Tensor b) -> Tensor` | 逐元素乘法（NumPy 广播） | out[i] = mul(a[i], b[i])；广播规则同 NumPy |
| `D1-008` | `math.mulScalar` | `(Tensor a, number s) -> Tensor` | 张量与标量逐元素乘法（右标量） | out[i] = mul(a[i], s) |
| `D1-009` | `math.scalarMul` | `(number s, Tensor a) -> Tensor` | 标量与张量逐元素乘法（左标量） | out[i] = mul(s, a[i]) |
| `D1-010` | `math.div` | `(Tensor a, Tensor b) -> Tensor` | 逐元素除法（NumPy 广播） | out[i] = div(a[i], b[i])；广播规则同 NumPy |
| `D1-011` | `math.divScalar` | `(Tensor a, number s) -> Tensor` | 张量与标量逐元素除法（右标量） | out[i] = div(a[i], s) |
| `D1-012` | `math.scalarDiv` | `(number s, Tensor a) -> Tensor` | 标量与张量逐元素除法（左标量） | out[i] = div(s, a[i]) |
| `D1-013` | `math.mod` | `(Tensor a, Tensor b) -> Tensor` | 逐元素取模（JS 语义，符号随被除数）（NumPy 广播） | out[i] = mod(a[i], b[i])；广播规则同 NumPy |
| `D1-014` | `math.modScalar` | `(Tensor a, number s) -> Tensor` | 张量与标量逐元素取模（JS 语义，符号随被除数）（右标量） | out[i] = mod(a[i], s) |
| `D1-015` | `math.scalarMod` | `(number s, Tensor a) -> Tensor` | 标量与张量逐元素取模（JS 语义，符号随被除数）（左标量） | out[i] = mod(s, a[i]) |
| `D1-016` | `math.pow` | `(Tensor a, Tensor b) -> Tensor` | 逐元素幂（NumPy 广播） | out[i] = pow(a[i], b[i])；广播规则同 NumPy |
| `D1-017` | `math.powScalar` | `(Tensor a, number s) -> Tensor` | 张量与标量逐元素幂（右标量） | out[i] = pow(a[i], s) |
| `D1-018` | `math.scalarPow` | `(number s, Tensor a) -> Tensor` | 标量与张量逐元素幂（左标量） | out[i] = pow(s, a[i]) |
| `D1-019` | `math.atan2` | `(Tensor a, Tensor b) -> Tensor` | 逐元素二参数反正切（NumPy 广播） | out[i] = atan2(a[i], b[i])；广播规则同 NumPy |
| `D1-020` | `math.atan2Scalar` | `(Tensor a, number s) -> Tensor` | 张量与标量逐元素二参数反正切（右标量） | out[i] = atan2(a[i], s) |
| `D1-021` | `math.scalarAtan2` | `(number s, Tensor a) -> Tensor` | 标量与张量逐元素二参数反正切（左标量） | out[i] = atan2(s, a[i]) |
| `D1-022` | `math.hypot` | `(Tensor a, Tensor b) -> Tensor` | 逐元素斜边 √(a²+b²)（NumPy 广播） | out[i] = hypot(a[i], b[i])；广播规则同 NumPy |
| `D1-023` | `math.hypotScalar` | `(Tensor a, number s) -> Tensor` | 张量与标量逐元素斜边 √(a²+b²)（右标量） | out[i] = hypot(a[i], s) |
| `D1-024` | `math.scalarHypot` | `(number s, Tensor a) -> Tensor` | 标量与张量逐元素斜边 √(a²+b²)（左标量） | out[i] = hypot(s, a[i]) |
| `D1-025` | `math.maximum` | `(Tensor a, Tensor b) -> Tensor` | 逐元素逐元素取大（NumPy ufunc 命名）（NumPy 广播） | out[i] = maximum(a[i], b[i])；广播规则同 NumPy |
| `D1-026` | `math.maximumScalar` | `(Tensor a, number s) -> Tensor` | 张量与标量逐元素逐元素取大（NumPy ufunc 命名）（右标量） | out[i] = maximum(a[i], s) |
| `D1-027` | `math.scalarMaximum` | `(number s, Tensor a) -> Tensor` | 标量与张量逐元素逐元素取大（NumPy ufunc 命名）（左标量） | out[i] = maximum(s, a[i]) |
| `D1-028` | `math.minimum` | `(Tensor a, Tensor b) -> Tensor` | 逐元素逐元素取小（NumPy ufunc 命名）（NumPy 广播） | out[i] = minimum(a[i], b[i])；广播规则同 NumPy |
| `D1-029` | `math.minimumScalar` | `(Tensor a, number s) -> Tensor` | 张量与标量逐元素逐元素取小（NumPy ufunc 命名）（右标量） | out[i] = minimum(a[i], s) |
| `D1-030` | `math.scalarMinimum` | `(number s, Tensor a) -> Tensor` | 标量与张量逐元素逐元素取小（NumPy ufunc 命名）（左标量） | out[i] = minimum(s, a[i]) |
| `D1-031` | `math.sin` | `(Tensor) -> Tensor` | 逐元素 sin | out[i] = sin(in[i]) |
| `D1-032` | `math.cos` | `(Tensor) -> Tensor` | 逐元素 cos | out[i] = cos(in[i]) |
| `D1-033` | `math.tan` | `(Tensor) -> Tensor` | 逐元素 tan | out[i] = tan(in[i]) |
| `D1-034` | `math.asin` | `(Tensor) -> Tensor` | 逐元素 asin | out[i] = asin(in[i]) |
| `D1-035` | `math.acos` | `(Tensor) -> Tensor` | 逐元素 acos | out[i] = acos(in[i]) |
| `D1-036` | `math.atan` | `(Tensor) -> Tensor` | 逐元素 atan | out[i] = atan(in[i]) |
| `D1-037` | `math.sinh` | `(Tensor) -> Tensor` | 逐元素 sinh | out[i] = sinh(in[i]) |
| `D1-038` | `math.cosh` | `(Tensor) -> Tensor` | 逐元素 cosh | out[i] = cosh(in[i]) |
| `D1-039` | `math.tanh` | `(Tensor) -> Tensor` | 逐元素 tanh | out[i] = tanh(in[i]) |
| `D1-040` | `math.asinh` | `(Tensor) -> Tensor` | 逐元素 asinh | out[i] = asinh(in[i]) |
| `D1-041` | `math.acosh` | `(Tensor) -> Tensor` | 逐元素 acosh | out[i] = acosh(in[i]) |
| `D1-042` | `math.atanh` | `(Tensor) -> Tensor` | 逐元素 atanh | out[i] = atanh(in[i]) |
| `D1-043` | `math.exp` | `(Tensor) -> Tensor` | 逐元素 exp | out[i] = exp(in[i]) |
| `D1-044` | `math.expm1` | `(Tensor) -> Tensor` | 逐元素 expm1 | out[i] = expm1(in[i]) |
| `D1-045` | `math.log` | `(Tensor) -> Tensor` | 逐元素 log | out[i] = log(in[i]) |
| `D1-046` | `math.log1p` | `(Tensor) -> Tensor` | 逐元素 log1p | out[i] = log1p(in[i]) |
| `D1-047` | `math.log2` | `(Tensor) -> Tensor` | 逐元素 log2 | out[i] = log2(in[i]) |
| `D1-048` | `math.log10` | `(Tensor) -> Tensor` | 逐元素 log10 | out[i] = log10(in[i]) |
| `D1-049` | `math.sqrt` | `(Tensor) -> Tensor` | 逐元素 sqrt | out[i] = sqrt(in[i]) |
| `D1-050` | `math.cbrt` | `(Tensor) -> Tensor` | 逐元素 cbrt | out[i] = cbrt(in[i]) |
| `D1-051` | `math.abs` | `(Tensor) -> Tensor` | 逐元素 abs | out[i] = abs(in[i]) |
| `D1-052` | `math.sign` | `(Tensor) -> Tensor` | 逐元素 sign | out[i] = sign(in[i]) |
| `D1-053` | `math.floor` | `(Tensor) -> Tensor` | 逐元素 floor | out[i] = floor(in[i]) |
| `D1-054` | `math.ceil` | `(Tensor) -> Tensor` | 逐元素 ceil | out[i] = ceil(in[i]) |
| `D1-055` | `math.round` | `(Tensor) -> Tensor` | 逐元素 round | out[i] = round(in[i]) |
| `D1-056` | `math.trunc` | `(Tensor) -> Tensor` | 逐元素 trunc | out[i] = trunc(in[i]) |
| `D1-057` | `math.square` | `(Tensor) -> Tensor` | 逐元素 square | out[i] = square(in[i]) |
| `D1-058` | `math.reciprocal` | `(Tensor) -> Tensor` | 逐元素 reciprocal | out[i] = reciprocal(in[i]) |
| `D1-059` | `math.dot` | `(Tensor x, Tensor y) -> number` | 向量点积 x·y = Σxᵢyᵢ | dot(x,y)=dot(y,x)；dot(x,x)=‖x‖² |
| `D1-060` | `math.cross` | `(Tensor x, Tensor y) -> Tensor` | 3D 向量叉积 x×y | x×y ⊥ x,y；‖x×y‖=‖x‖‖y‖sinθ；(x×y)·x=0 |
| `D1-061` | `math.outer` | `(Tensor x, Tensor y) -> Tensor` | 外积 x⊗y，shape (m,n) | outer(x,y)[i,j]=x[i]·y[j] |
| `D1-062` | `math.l1` | `(Tensor) -> number` | L1 范数 ‖x‖₁=Σ\|xᵢ\| | ‖x‖₁≥0，=0 ⟺ x=0 |
| `D1-063` | `math.l2` | `(Tensor) -> number` | L2 范数 ‖x‖₂=√(Σxᵢ²) | ‖x‖₂²=dot(x,x) |
| `D1-064` | `math.linf` | `(Tensor) -> number` | L∞ 范数 max\|xᵢ\| | ‖x‖∞≥‖x‖₂≥‖x‖₁/√n |
| `D1-065` | `math.lp` | `(Tensor, number p) -> number` | Lp 范数 (Σ\|xᵢ\|^p)^(1/p) | p=1,2,∞ 与 math.l1/l2/linf 一致 |
| `D1-066` | `math.cosine` | `(Tensor x, Tensor y) -> number` | 余弦相似度 (x·y)/(‖x‖‖y‖) | cos(x,y)∈[-1,1]；cos(x,x)=1（x≠0）；零向量返回 0 |
| `D1-067` | `math.manhattan` | `(Tensor x, Tensor y) -> number` | 向量L1 距离 | d(x,y)≥0；d(x,y)=d(y,x)；d(x,x)=0 |
| `D1-068` | `math.euclidean` | `(Tensor x, Tensor y) -> number` | 向量L2 距离（开方前） | d(x,y)≥0；d(x,y)=d(y,x)；d(x,x)=0 |
| `D1-069` | `math.sqeuclidean` | `(Tensor x, Tensor y) -> number` | 向量平方 L2 距离 | d(x,y)≥0；d(x,y)=d(y,x)；d(x,x)=0 |
| `D1-070` | `math.chebyshev` | `(Tensor x, Tensor y) -> number` | 向量L∞ 距离 | d(x,y)≥0；d(x,y)=d(y,x)；d(x,x)=0 |
| `D1-071` | `math.sum` | `(Tensor) -> number` | 全元素总和 | 空集返回 0；sum(c·x)=c·sum(x)（sum/prod） |
| `D1-072` | `math.prod` | `(Tensor) -> number` | 全元素连乘 | 空集返回 1；prod(c·x)=c·prod(x)（sum/prod） |
| `D1-073` | `math.min` | `(Tensor) -> number` | 全元素最小值 | 空集返回 Infinity；min(c·x)=c·min(x)（sum/prod） |
| `D1-074` | `math.max` | `(Tensor) -> number` | 全元素最大值 | 空集返回 -Infinity；max(c·x)=c·max(x)（sum/prod） |
| `D1-075` | `math.mean` | `(Tensor) -> number` | 全元素均值 | mean(c·x)=c·mean(x) |
| `D1-076` | `math.var` | `(Tensor) -> number` | 全元素方差（总体，两遍法数值稳定） | var≥0；var(c·x)=c²·var(x) |
| `D1-077` | `math.std` | `(Tensor) -> number` | 全元素标准差 √var | std=√var；std≥0 |
| `D1-078` | `math.argmin` | `(Tensor) -> number` | 最小元素的扁平索引（首个） | t.at(argmin(t))=min(t) |
| `D1-079` | `math.argmax` | `(Tensor) -> number` | 最大元素的扁平索引（首个） | t.at(argmax(t))=max(t) |
| `D1-080` | `math.median` | `(Tensor) -> number` | 中位数（偶数长度取中间两值均值） | 至少一半元素 ≤ median ≤ 另一半 |
| `D1-081` | `math.quantile` | `(Tensor, number q) -> number` | q 分位数（线性插值，NumPy linear 方法） | quantile(0)=min；quantile(1)=max；quantile(0.5)=median |
| `D1-082` | `math.mad` | `(Tensor) -> number` | 中位数绝对偏差 median(\|x−median(x)\|) | mad≥0；对异常值不敏感（加一个极端值不变） |
| `D1-083` | `math.cumsum` | `(Tensor) -> Tensor` | 前缀和（1D） | cumsum[n−1]=sum；cumsum[k]−cumsum[k−1]=x[k] |
| `D1-084` | `math.cumprod` | `(Tensor) -> Tensor` | 前缀积（1D） | cumprod[n−1]=prod |
| `D1-085` | `math.sumAxis` | `(Tensor, number axis) -> Tensor` | 沿轴求和 | axis 归约后该轴消失；与逐列循环一致 |
| `D1-086` | `math.meanAxis` | `(Tensor, number axis) -> Tensor` | 沿轴均值 | axis 归约后该轴消失；与逐列循环一致 |
| `D1-087` | `math.maxAxis` | `(Tensor, number axis) -> Tensor` | 沿轴最大值 | axis 归约后该轴消失；与逐列循环一致 |
| `D1-088` | `math.minAxis` | `(Tensor, number axis) -> Tensor` | 沿轴最小值 | axis 归约后该轴消失；与逐列循环一致 |
| `D1-089` | `math.argmaxAxis` | `(Tensor, number axis) -> Tensor` | 沿轴 argmax（返回扁平前该轴内索引） | 对每个输出槽位，取到的元素等于该轴最大值 |
| `D1-090` | `math.varAxis` | `(Tensor, number axis) -> Tensor` | 沿轴方差（两遍法） | varAxis(x,axis).mean(其余轴) ≈ var(x) |
| `D1-091` | `math.zeros` | `(number[] shape) -> Tensor` | 全零张量 | 所有元素为 0 |
| `D1-092` | `math.ones` | `(number[] shape) -> Tensor` | 全一张量 | 所有元素为 1 |
| `D1-093` | `math.full` | `(number[] shape, number v) -> Tensor` | 常数张量 | 所有元素为 v |
| `D1-094` | `math.eye` | `(number n, number m?) -> Tensor` | 单位矩阵 | eye·A=A；eye 对称 |
| `D1-095` | `math.arange` | `(number stop, number start?, number step?) -> Tensor` | 等差序列 | 长度=ceil((stop−start)/step) |
| `D1-096` | `math.linspace` | `(number a, number b, number n) -> Tensor` | 闭区间等分 | 首=a 尾=b；相邻差=(b−a)/(n−1) |
| `D1-097` | `math.concat` | `(Tensor[] ts, number axis?) -> Tensor` | 沿轴拼接（非轴维必须一致） | 拼接后该轴长度=各输入该轴长度之和 |
| `D1-098` | `math.stack` | `(Tensor[] ts, number axis?) -> Tensor` | 沿新轴堆叠 | 结果维数=输入维数+1；新轴长度=输入个数 |
| `D1-099` | `math.tile` | `(Tensor t, number[] reps) -> Tensor` | 整体复制拼接 | 结果形状=逐维乘 reps |
| `D1-100` | `math.pad` | `(Tensor t, number[][] pads, number v?) -> Tensor` | 常数填充（pads=[[前,后],...] 每维一对） | 输出形状=原形状+前后填充；填充区=v（默认 0） |
| `D1-101` | `math.clamp` | `(Tensor t, number lo, number hi) -> Tensor` | 截断到 [lo,hi] | clamp(clamp(x)) = clamp(x) |
| `D1-102` | `math.where` | `(Tensor mask, Tensor a, Tensor b) -> Tensor` | 按掩码选择：mask≠0 取 a 否则取 b（NumPy 广播） | where(1,a,b)=a；where(0,a,b)=b |

## D2 · NUM 数值方法与稳定化

稳定softmax、补偿求和、误差度量、求根、积分、差分、特殊函数。共 102 条。

| id | 指令 | 签名 | 说明 | 不变量 |
|---|---|---|---|---|
| `D2-001` | `num.softmax` | `(Tensor x, number axis=-1) -> Tensor` | 数值稳定 softmax（先减轴最大值） | 沿 axis 每片和=1；输入整体平移不变 |
| `D2-002` | `num.logSoftmax` | `(Tensor x, number axis=-1) -> Tensor` | 稳定 log-softmax = x - max - logsumexp | exp(logSoftmax) 与 softmax 一致；不会下溢 |
| `D2-003` | `num.logsumexp` | `(Tensor x, number axis=-1) -> Tensor` | 稳定 logΣexp（max 平移 + log） | exp(logsumexp)=Σexp（不溢出） |
| `D2-004` | `num.sigmoid` | `(Tensor) -> Tensor` | 稳定 sigmoid（分支防 exp 溢出） | sigmoid(0)=0.5；sigmoid(-x)=1-sigmoid(x)；x=±1000 不溢出 |
| `D2-005` | `num.stableExp` | `(Tensor) -> Tensor` | 饱和 exp（x>709 返回 Inf，x<-745 返回 0） | 与 Math.exp 在定义域内一致；永不返回 NaN |
| `D2-006` | `num.safeLog` | `(Tensor) -> Tensor` | 对数（x=0→-Inf，x<0→NaN，文档化契约） | safeLog(e^x)=x（x 有限）；safeLog(0)=-Inf |
| `D2-007` | `num.safeDiv` | `(Tensor a, Tensor b) -> Tensor` | IEEE 语义除法（0/0=NaN，x/0=±Inf） | 与 JS / 一致且对零不抛错 |
| `D2-008` | `num.safeSqrt` | `(Tensor) -> Tensor` | 平方根（x<0→NaN） | safeSqrt(x²)=\|x\| |
| `D2-009` | `num.safePow` | `(Tensor base, Tensor exp) -> Tensor` | 幂运算（负底非整数指数→NaN，0^0=1） | 与 Math.pow 一致且零不抛错 |
| `D2-010` | `num.softplus` | `(Tensor) -> Tensor` | 稳定 softplus log(1+e^x)（大 x 线性外推） | softplus(x)≈max(x,0)（\|x\| 大时）；softplus(0)=ln2 |
| `D2-011` | `num.kahanSum` | `(Tensor) -> number` | Kahan 补偿求和（消除累计舍入） | 对 [1e16,1,-1e16] 返回 1（朴素求和返回 0） |
| `D2-012` | `num.neumaierSum` | `(Tensor) -> number` | Neumaier 改进补偿求和 | 精度 ≥ Kahan；处理大小悬殊序列 |
| `D2-013` | `num.pairwiseSum` | `(Tensor) -> number` | 成对递归求和（误差 O(log n)） | 精度 ≥ 顺序求和；与 Kahan 同量级 |
| `D2-014` | `num.naiveSum` | `(Tensor) -> number` | 朴素顺序求和（精度基准，供对比） | 与手写 for 循环完全一致 |
| `D2-015` | `num.absErr` | `(number a, number b) -> number` | 绝对误差 \|a-b\| | absErr(a,a)=0 |
| `D2-016` | `num.relErr` | `(number a, number b) -> number` | 相对误差 \|a-b\|/max(\|a\|,\|b\|,1) | relErr(a,a)=0；对称 |
| `D2-017` | `num.ulpErr` | `(number a, number b) -> number` | ULP 差异 \|a-b\|/ulp(max(\|a\|,\|b\|)) | 同数返回 0；相邻浮点返回 1 |
| `D2-018` | `num.mae` | `(Tensor a, Tensor b) -> number` | 平均绝对误差 | mae(a,a)=0 |
| `D2-019` | `num.mse` | `(Tensor a, Tensor b) -> number` | 均方误差 | mse(a,a)=0；mse≥0 |
| `D2-020` | `num.rmse` | `(Tensor a, Tensor b) -> number` | 均方根误差 √mse | rmse=√mse |
| `D2-021` | `num.mape` | `(Tensor a, Tensor b) -> number` | 平均绝对百分比误差（b=0 项跳过） | 完美预测返回 0 |
| `D2-022` | `num.maxAbsErr` | `(Tensor a, Tensor b) -> number` | 最大绝对误差（L∞ 误差） | ≥ mae；=0 ⟺ 相等 |
| `D2-023` | `num.medAbsErr` | `(Tensor a, Tensor b) -> number` | 中位数绝对误差（稳健） | 对异常值不敏感 |
| `D2-024` | `num.isclose` | `(number a, number b, number rtol=1e-5, number atol=1e-8) -> boolean` | 逐对近似相等（NumPy isclose 语义） | isclose(a,a)=true；isclose(a,b)=isclose(b,a) |
| `D2-025` | `num.allclose` | `(Tensor a, Tensor b, number rtol=1e-5, number atol=1e-8) -> boolean` | 张量逐元素近似相等（NumPy allclose） | 形状必须一致 |
| `D2-026` | `num.arrayEqual` | `(Tensor a, Tensor b) -> boolean` | 精确逐元素相等（含 NaN 位模式） | arrayEqual(a,a)=true |
| `D2-027` | `num.isfiniteAll` | `(Tensor) -> boolean` | 全部元素有限 | 无 NaN/Inf 返回 true |
| `D2-028` | `num.countNaN` | `(Tensor) -> number` | NaN 计数 | countNaN ≤ size |
| `D2-029` | `num.countInf` | `(Tensor) -> number` | ±Inf 计数 | countInf ≤ size |
| `D2-030` | `num.nanToZero` | `(Tensor) -> Tensor` | NaN 置零（Inf 保留） | 结果无 NaN |
| `D2-031` | `num.sanitize` | `(Tensor t, number lo, number hi) -> Tensor` | 清洗：NaN→lo，+Inf→hi，-Inf→lo，其余不变 | 结果全部有限 |
| `D2-032` | `num.bisect` | `(Function f, number a, number b, object opts?) -> object` | 二分法求根（需 f(a)·f(b)<0） | 收敛速度每次减半；结果 bracket 宽度 ≤ tol |
| `D2-033` | `num.secant` | `(Function f, number x0, number x1, object opts?) -> object` | 割线法求根（超线性收敛） | 光滑函数上快于二分 |
| `D2-034` | `num.newton` | `(Function f, number x0, object opts?) -> object` | Newton 法（中心差分数值导数，二阶收敛） | 单根附近二次收敛 |
| `D2-035` | `num.brent` | `(Function f, number a, number b, object opts?) -> object` | Brent 法（逆二次插值+二分保障，鲁棒且快） | 保收敛性；通常远少于二分的迭代次数 |
| `D2-036` | `num.ridder` | `(Function f, number a, number b, object opts?) -> object` | Ridder 法（指数插值，保区间） | 收敛快于二分；区间始终含根 |
| `D2-037` | `num.rootScan` | `(Function f, number a, number b, number n) -> number[][]` | 扫描区间找全部符号变化 bracket | 每个返回 bracket 满足 f(lo)·f(hi)≤0 |
| `D2-038` | `num.trapz` | `(Tensor y, Tensor x?) -> number` | 梯形法积分（非均匀步长） | 对线性函数精确 |
| `D2-039` | `num.cumtrapz` | `(Tensor y, Tensor x?) -> Tensor` | 累积梯形积分 | 末元素 = trapz(y) |
| `D2-040` | `num.simpson` | `(Tensor y, Tensor x?) -> number` | Simpson 1/3 法（均匀步长；奇数区间末段退化梯形） | 对三次及以下多项式精确 |
| `D2-041` | `num.midpoint` | `(Function f, number a, number b, number n) -> number` | 中点法（函数形式） | n 加倍误差约减半 |
| `D2-042` | `num.romberg` | `(Function f, number a, number b, number levels=6) -> number` | Romberg 积分（ Richardson 外提加速） | 每级精度量级提升 |
| `D2-043` | `num.diff` | `(Tensor y, number n=1) -> Tensor` | n 阶前向差分 | 每阶长度减 1；常数序列一阶分为 0 |
| `D2-044` | `num.gradCentral` | `(Function f, Tensor x, number eps=1e-5) -> Tensor` | 中心差分数值梯度（O(ε²)） | 对二次函数精确到 O(ε²)；f 为 x↦标量 |
| `D2-045` | `num.gradForward` | `(Function f, Tensor x, number eps=1e-5) -> Tensor` | 前向差分数值梯度（O(ε)） | 比中心差分便宜但精度低一阶 |
| `D2-046` | `num.hessianDiag` | `(Function f, Tensor x, number eps=1e-4) -> Tensor` | Hessian 对角线（二阶中心差分） | 对 f=½xᵀAx 返回 A 的对角元 |
| `D2-047` | `num.jacobianFD` | `(Function F, Tensor x, number eps=1e-5) -> Tensor` | 数值雅可比矩阵（F: R^n→R^m） | 第 j 列为 ∂F/∂xⱼ |
| `D2-048` | `num.searchSorted` | `(Tensor xs, number xq) -> number` | 二分查找插入位置（xs 升序） | 返回 i 使 xs[i-1]≤xq<xs[i] |
| `D2-049` | `num.interpLinear` | `(Tensor xs, Tensor ys, Tensor\|number xq) -> Tensor\|number` | 分段线性插值（区间外线性外推） | 过全部数据点 |
| `D2-050` | `num.interpNearest` | `(Tensor xs, Tensor ys, number xq) -> number` | 最近邻插值 | 返回最近节点的值 |
| `D2-051` | `num.interpLagrange` | `(Tensor xs, Tensor ys, number xq) -> number` | Lagrange 全局多项式插值 | 过全部节点；≤n-1 次多项式精确 |
| `D2-052` | `num.erf` | `(number x) -> number` | 误差函数（级数+连分数，全定义域） | erf(-x)=-erf(x)；erf(∞)=1；erf(1)=0.8427007929497149 |
| `D2-053` | `num.erfc` | `(number x) -> number` | 互补误差函数 | erfc(-x)=2-erfc(x)；erfc(0)=1；erfc(∞)=0 |
| `D2-054` | `num.gamma` | `(number z) -> number` | Gamma 函数（Stirling 级数 + 递推） | gamma(n)=(n-1)!；gamma(0.5)=√π=1.7724538509055159 |
| `D2-055` | `num.lgamma` | `(number z) -> number` | log\|Γ(z)\|（Stirling + 递推，大数不溢出） | lgamma(n)=ln((n-1)!)；lgamma(0.5)=ln√π |
| `D2-056` | `num.beta` | `(number a, number b) -> number` | Beta 函数 Γ(a)Γ(b)/Γ(a+b) | beta(a,b)=beta(b,a)；beta(1,1)=1 |
| `D2-057` | `num.normalQuantile` | `(number p, boolean refine=true) -> number` | 标准正态分位数（Acklam 逆近似 + Newton 修正到机器精度） | normalQuantile(normalCdf(x))≈x（1e-12）；normalQuantile(0.5)=0 |
| `D2-058` | `num.eps` | `() -> number` | 机器 epsilon 2⁻⁵² | 1+eps>1 且 1+eps/2=1 |
| `D2-059` | `num.nextafter` | `(number x, number y) -> number` | x 之后下一个可表示双精度（朝 y 方向） | nextafter(1,2)-1=EPSILON；nextafter(0,1)=MIN_VALUE |
| `D2-060` | `num.ulp` | `(number x) -> number` | x 处的 ULP（ulp(0)=最小正规数） | ulp(1)=EPSILON；ulp(x) 随量级倍增 |
| `D2-061` | `num.decompose` | `(number x) -> object` | Dekker 分裂 x=hi+lo（\|hi\| 恰好 26 位尾数） | hi+lo=x（精确）；hi 可用 26 位表示 |
| `D2-062` | `num.twoSum` | `(number a, number b) -> object` | Knuth 两数和 s=a+b, e=误差 | s+e=a+b（精确）；e 可表示 |
| `D2-063` | `num.twoProd` | `(number a, number b) -> object` | Dekker 两乘积 p=a·b, e=误差 | p+e=a·b（精确） |
| `D2-064` | `num.quickTwoSum` | `(number a, number b) -> object` | 快速两和（要求 \|a\|≥\|b\|） | s+e=a+b（\|a\|≥\|b\| 时） |
| `D2-065` | `num.ddFromNumber` | `(number x) -> object` | 数 → double-double | ddToNumber(ddFromNumber(x))=x |
| `D2-066` | `num.ddToNumber` | `(object x) -> number` | double-double → 数（取 hi+lo） | 返回最接近的双精度 |
| `D2-067` | `num.ddAdd` | `(object x, object y) -> object` | double-double 加法 | 结果精度 ~106 位 |
| `D2-068` | `num.ddSub` | `(object x, object y) -> object` | double-double 减法 | ddSub(x,x)=0 |
| `D2-069` | `num.ddMul` | `(object x, object y) -> object` | double-double 乘法 | 结果精度 ~106 位 |
| `D2-070` | `num.welfordInit` | `() -> object` | Welford 状态初始化 | {n:0, mean:0, M2:0} |
| `D2-071` | `num.welfordUpdate` | `(object st, number x) -> object` | Welford 单点更新（方差数值稳定） | n 递增；mean/M2 与两遍法一致（1e-12） |
| `D2-072` | `num.welfordMerge` | `(object a, object b) -> object` | Welford 并行合并（Chan 公式） | merge(前半,后半)=全量 Welford（1e-12） |
| `D2-073` | `num.welfordFinalize` | `(object st) -> object` | Welford 输出 {n, mean, var, std} | var=M2/n；std=√var；n=0 时 var=0 |
| `D2-074` | `num.guard.exp` | `(number x, boolean strict=false) -> number` | 守卫 exp：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 全体实数定义域（NaN/Inf 按 IEEE 754 约定处理） |
| `D2-075` | `num.guard.log` | `(number x, boolean strict=false) -> number` | 守卫 log：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 定义域: (v) => v > 0 |
| `D2-076` | `num.guard.log1p` | `(number x, boolean strict=false) -> number` | 守卫 log1p：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 定义域: (v) => v > -1 |
| `D2-077` | `num.guard.log2` | `(number x, boolean strict=false) -> number` | 守卫 log2：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 定义域: (v) => v > 0 |
| `D2-078` | `num.guard.log10` | `(number x, boolean strict=false) -> number` | 守卫 log10：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 定义域: (v) => v > 0 |
| `D2-079` | `num.guard.sqrt` | `(number x, boolean strict=false) -> number` | 守卫 sqrt：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 定义域: (v) => v >= 0 |
| `D2-080` | `num.guard.cbrt` | `(number x, boolean strict=false) -> number` | 守卫 cbrt：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 全体实数定义域（NaN/Inf 按 IEEE 754 约定处理） |
| `D2-081` | `num.guard.sin` | `(number x, boolean strict=false) -> number` | 守卫 sin：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 全体实数定义域（NaN/Inf 按 IEEE 754 约定处理） |
| `D2-082` | `num.guard.cos` | `(number x, boolean strict=false) -> number` | 守卫 cos：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 全体实数定义域（NaN/Inf 按 IEEE 754 约定处理） |
| `D2-083` | `num.guard.tan` | `(number x, boolean strict=false) -> number` | 守卫 tan：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 全体实数定义域（NaN/Inf 按 IEEE 754 约定处理） |
| `D2-084` | `num.guard.asin` | `(number x, boolean strict=false) -> number` | 守卫 asin：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 定义域: (v) => v >= -1 && v <= 1 |
| `D2-085` | `num.guard.acos` | `(number x, boolean strict=false) -> number` | 守卫 acos：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 定义域: (v) => v >= -1 && v <= 1 |
| `D2-086` | `num.guard.atan` | `(number x, boolean strict=false) -> number` | 守卫 atan：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 全体实数定义域（NaN/Inf 按 IEEE 754 约定处理） |
| `D2-087` | `num.guard.sinh` | `(number x, boolean strict=false) -> number` | 守卫 sinh：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 全体实数定义域（NaN/Inf 按 IEEE 754 约定处理） |
| `D2-088` | `num.guard.cosh` | `(number x, boolean strict=false) -> number` | 守卫 cosh：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 全体实数定义域（NaN/Inf 按 IEEE 754 约定处理） |
| `D2-089` | `num.guard.tanh` | `(number x, boolean strict=false) -> number` | 守卫 tanh：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 全体实数定义域（NaN/Inf 按 IEEE 754 约定处理） |
| `D2-090` | `num.guard.asinh` | `(number x, boolean strict=false) -> number` | 守卫 asinh：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 全体实数定义域（NaN/Inf 按 IEEE 754 约定处理） |
| `D2-091` | `num.guard.acosh` | `(number x, boolean strict=false) -> number` | 守卫 acosh：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 定义域: (v) => v >= 1 |
| `D2-092` | `num.guard.atanh` | `(number x, boolean strict=false) -> number` | 守卫 atanh：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 定义域: (v) => v > -1 && v < 1 |
| `D2-093` | `num.guard.expm1` | `(number x, boolean strict=false) -> number` | 守卫 expm1：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 全体实数定义域（NaN/Inf 按 IEEE 754 约定处理） |
| `D2-094` | `num.guard.reciprocal` | `(number x, boolean strict=false) -> number` | 守卫 reciprocal：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 定义域: (v) => v !== 0 |
| `D2-095` | `num.guard.square` | `(number x, boolean strict=false) -> number` | 守卫 square：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 全体实数定义域（NaN/Inf 按 IEEE 754 约定处理） |
| `D2-096` | `num.guard.abs` | `(number x, boolean strict=false) -> number` | 守卫 abs：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 全体实数定义域（NaN/Inf 按 IEEE 754 约定处理） |
| `D2-097` | `num.guard.sign` | `(number x, boolean strict=false) -> number` | 守卫 sign：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值 | 全体实数定义域（NaN/Inf 按 IEEE 754 约定处理） |
| `D2-098` | `num.horner` | `(Tensor coefs, number x) -> number` | Horner 多项式求值（coefs 低次→高次） | 对 coefs=[c0] 返回 c0；horner([1,2,3],0)=1 |
| `D2-099` | `num.dotKahan` | `(Tensor a, Tensor b) -> number` | Kahan 补偿点积 | 比朴素点积更接近精确值 |
| `D2-100` | `num.richardson` | `(Tensor ests, number k=2) -> number` | Richardson 外提（步长逐倍减半的估计序列） | 外提后精度阶数提升 k 倍 |
| `D2-101` | `num.aitken` | `(Tensor seq) -> number` | Aitken Δ² 序列加速 | 线性收敛序列加速后误差 O(原始²) |
| `D2-102` | `num.welfordFromTensor` | `(Tensor t) -> object` | 对整张量一次 Welford（返回状态） | 与逐点更新一致 |

## D3 · LINALG 线性代数

分解族（LU/QR/Cholesky/Jacobi/SVD）、求解器、迭代法、矩阵函数。共 102 条。

| id | 指令 | 签名 | 说明 | 不变量 |
|---|---|---|---|---|
| `D3-001` | `linalg.matmul` | `(Tensor A[m,k], Tensor B[k,n]) -> Tensor C[m,n]` | 矩阵乘法（分块 i-k-j，缓存友好） | C[i,j]=ΣₖA[i,k]B[k,j]；matmul(A,I)=A |
| `D3-002` | `linalg.matmulVec` | `(Tensor A[m,k], Tensor x[k]) -> Tensor y[m]` | 矩阵-向量乘法 | y[i]=ΣₖA[i,k]x[k] |
| `D3-003` | `linalg.matmulAT` | `(Tensor A[k,m], Tensor B[k,n]) -> Tensor C[m,n]` | Aᵀ·B | matmulAT(A,B)=matmul(A.T,B) |
| `D3-004` | `linalg.matmulBT` | `(Tensor A[m,k], Tensor B[n,k]) -> Tensor C[m,n]` | A·Bᵀ | matmulBT(A,B)=matmul(A,B.T) |
| `D3-005` | `linalg.matmulABT` | `(Tensor A[k,m], Tensor B[n,k]) -> Tensor C[m,n]` | Aᵀ·Bᵀ | matmulABT(A,B)=matmul(A.T,B.T) |
| `D3-006` | `linalg.gemm` | `(number alpha, Tensor A, Tensor B, number beta, Tensor C) -> Tensor` | BLAS 级 GEMM：α·A·B+β·C | gemm(1,A,B,0,C)=matmul(A,B)；C 缺省形状时忽略 beta |
| `D3-007` | `linalg.lu` | `(Tensor A) -> object` | LU 分解（Doolittle + 部分选主元） | P·A=L·U；L 单位下三角；U 上三角；\|det\|=ΠUᵢᵢ |
| `D3-008` | `linalg.luNoPivot` | `(Tensor A) -> object` | LU 分解（无选主元，对角占稳矩阵用） | A=L·U（无置换） |
| `D3-009` | `linalg.forwardSubst` | `(Tensor L[n,n], Tensor b[n]) -> Tensor x[n]` | 前向替换（下三角系统 Lx=b） | 解单位下三角方程组 L·x = b（前代 forward substitution） |
| `D3-010` | `linalg.backSubst` | `(Tensor U[n,n], Tensor b[n]) -> Tensor x[n]` | 后向替换（上三角系统 Ux=b） | 解单位上三角方程组 U·x = b（回代 back substitution） |
| `D3-011` | `linalg.qr` | `(Tensor A[m,n]) -> object {Q,R}` | QR 分解（Householder 反射） | QᵀQ=I；Q·R=A；R 上三角 |
| `D3-012` | `linalg.cholesky` | `(Tensor A[n,n]) -> Tensor L[n,n]` | Cholesky 分解 A=L·Lᵀ（SPD 验证） | L·Lᵀ=A；L 下三角且 Lᵢᵢ>0；非 SPD 抛 E_NUMERICAL |
| `D3-013` | `linalg.det` | `(Tensor A[n,n]) -> number` | 行列式（LU + 选主元符号） | det(AB)=det(A)det(B)；det(Aᵀ)=det(A)；det(I)=1 |
| `D3-014` | `linalg.logdet` | `(Tensor A[n,n]) -> number` | 对数行列式（Cholesky，2ΣlnLᵢᵢ，防溢出） | exp(logdet)=det；SPD 时为实数 |
| `D3-015` | `linalg.inverse` | `(Tensor A[n,n]) -> Tensor` | 矩阵逆（Gauss-Jordan + 部分选主元） | A·A⁻¹=I；奇异矩阵抛 E_NUMERICAL |
| `D3-016` | `linalg.solve` | `(Tensor A[n,n], Tensor b[n]) -> Tensor x[n]` | 线性系统求解（LU） | A·x=b；残差 ‖Ax−b‖/‖b‖ < 1e-10 |
| `D3-017` | `linalg.solveSPD` | `(Tensor A[n,n], Tensor b[n]) -> Tensor x[n]` | SPD 系统求解（Cholesky） | 对称正定 A·x = b 的唯一解（Cholesky 分解） |
| `D3-018` | `linalg.solveT` | `(Tensor A[n,n], Tensor b[n]) -> Tensor x[n]` | 转置系统求解 Aᵀx=b | Aᵀ·x=b |
| `D3-019` | `linalg.solveMultiple` | `(Tensor A[n,n], Tensor B[n,k]) -> Tensor X[n,k]` | 多右端系统求解 A·X=B | 同时解 A·X = B 的多列右端项 |
| `D3-020` | `linalg.tridiagSolve` | `(Tensor sub[n], Tensor diag[n], Tensor sup[n], Tensor rhs[n]) -> Tensor x[n]` | Thomas 算法（次/主/超对角 + 右端，O(n)） | 重构三对角矩阵验证 A·x=rhs |
| `D3-021` | `linalg.toeplitzSolve` | `(Tensor r[n+1], Tensor b[n]) -> Tensor x[n]` | Toeplitz 系统求解（显式构造 + LU） | T[i,j]=r[\|i-j\|]；T·x=b |
| `D3-022` | `linalg.eigenSym` | `(Tensor A[n,n]) -> object {values, vectors}` | 对称矩阵特征分解（循环 Jacobi 旋转） | A·vᵢ=λᵢvᵢ（残差‖Av−λv‖<1e-10）；V 正交；values 降序 |
| `D3-023` | `linalg.eigenSymQR` | `(Tensor A[n,n]) -> object {values}` | 对称矩阵特征值（Wilkinson 位移 QR 迭代） | 与 Jacobi 结果一致（1e-8）；仅特征值 |
| `D3-024` | `linalg.eigen2x2` | `(Tensor A[2,2]) -> object {values, vectors}` | 2×2 对称矩阵解析特征分解（测试基准） | 解析解；与 Jacobi 结果一致 |
| `D3-025` | `linalg.svd` | `(Tensor A[m,n]) -> object {U,S,V}` | SVD（单边 Jacobi 正交化） | U·diag(S)·Vᵀ=A；U(m×k)/V(n×k) 列正交；S 降序非负 |
| `D3-026` | `linalg.pinv` | `(Tensor A, number rcond=1e-12) -> Tensor` | Moore-Penrose 伪逆（SVD 截断） | A·A⁺·A=A；A⁺·A·A⁺=A⁺；满秩时等于逆 |
| `D3-027` | `linalg.rank` | `(Tensor A, number tol?) -> number` | 数值秩（SVD 容差 max(m,n)·eps·σ₁） | rank ≤ min(m,n)；满秩方阵 rank=n |
| `D3-028` | `linalg.normFro` | `(Tensor A) -> number` | Frobenius 范数 √ΣAᵢⱼ² | normFro(Aᵀ)=normFro(A) |
| `D3-029` | `linalg.norm1` | `(Tensor A) -> number` | 1-范数（最大列和） | norm1(A)≥normFro(A)/√n |
| `D3-030` | `linalg.normInf` | `(Tensor A) -> number` | ∞-范数（最大行和） | normInf(A)=norm1(Aᵀ) |
| `D3-031` | `linalg.normMax` | `(Tensor A) -> number` | 最大元范数 max\|Aᵢⱼ\| | normMax ≤ normFro ≤ √(mn)·normMax |
| `D3-032` | `linalg.norm2` | `(Tensor A) -> number` | 2-范数（谱范数 σmax，SVD） | norm2=σmax；norm2(A)=norm2(Aᵀ) |
| `D3-033` | `linalg.normNuc` | `(Tensor A) -> number` | 核范数 Σσᵢ | normNuc ≥ norm2 |
| `D3-034` | `linalg.cond2` | `(Tensor A) -> number` | 2-范数条件数 σmax/σmin | cond(I)=1；cond≥1；大=病态 |
| `D3-035` | `linalg.cond1` | `(Tensor A) -> number` | 1-范数条件数 ‖A‖₁·‖A⁻¹‖₁ | cond1(A)=cond1(Aᵀ) |
| `D3-036` | `linalg.gershgorin` | `(Tensor A) -> object` | Gershgorin 圆盘（特征值包含域估计） | 全部特征值落在圆盘并集内 |
| `D3-037` | `linalg.jacobiIter` | `(Tensor A, Tensor b, number maxIter=1000, number tol=1e-10) -> object` | Jacobi 迭代（对角占优系统） | 严格对角占优必收敛 |
| `D3-038` | `linalg.gaussSeidel` | `(Tensor A, Tensor b, number maxIter=1000, number tol=1e-10) -> object` | Gauss-Seidel 迭代（用最新分量，通常快于 Jacobi） | SPD 或对角占优必收敛 |
| `D3-039` | `linalg.sor` | `(Tensor A, Tensor b, number maxIter=1000, number tol=1e-10, number omega=1.5) -> object` | SOR 超松弛迭代（ω∈(0,2)，ω=1 即 Gauss-Seidel） | 最优 ω 收敛最快 |
| `D3-040` | `linalg.conjGradient` | `(Tensor A, Tensor b, number maxIter?, number tol=1e-10) -> object` | 共轭梯度法（SPD 系统，Krylov 子空间） | 精确算术下 ≤n 步收敛；残差单调不增 |
| `D3-041` | `linalg.powerIteration` | `(Tensor A, number maxIter=1000, number tol=1e-12) -> object` | 幂迭代（按模最大特征对；负主特征时向量符号振荡，返回值取模） | \|value\|=max\|λ\|；残差 ‖Av−λv‖→0 |
| `D3-042` | `linalg.inverseIteration` | `(Tensor A, number mu=0, number maxIter=100) -> object` | 反迭代（求靠近位移 μ 的特征对） | 收敛到最接近 μ 的特征值 |
| `D3-043` | `linalg.rayleighQuotient` | `(Tensor A, Tensor v) -> number` | Rayleigh 商 vᵀAv/vᵀv | v 为特征向量时等于特征值 |
| `D3-044` | `linalg.matrixPower` | `(Tensor A, number k) -> Tensor` | 矩阵幂 Aᵏ（快速幂） | A⁰=I；A¹=A；AᵏAᵐ=Aᵏ⁺ᵐ |
| `D3-045` | `linalg.matrixExp` | `(Tensor A[n,n]) -> Tensor` | 矩阵指数 e^A（缩放平方 + Taylor 级数） | e⁰=I；e^{tA}e^{sA}=e^{(t+s)A}；d/dt e^{tA}=A·e^{tA} |
| `D3-046` | `linalg.symmetrize` | `(Tensor A) -> Tensor` | 对称化 (A+Aᵀ)/2 | symmetrize(A)ᵀ=symmetrize(A) |
| `D3-047` | `linalg.whiten` | `(Tensor X[n,d], number eps=1e-5) -> object` | ZCA 白化（协方差特征分解） | 白化后协方差 ≈ I；whiten 矩阵对称 |
| `D3-048` | `linalg.nearestSPD` | `(Tensor A) -> Tensor` | 最近对称正定矩阵（对称化 + 特征值截断修正） | 结果 SPD；对称 |
| `D3-049` | `linalg.quadForm` | `(Tensor x, Tensor A) -> number` | 二次型 xᵀAx | A 对称时 quadForm(x,A)=λ 加权和 |
| `D3-050` | `linalg.matVecT` | `(Tensor x[m], Tensor A[m,n]) -> Tensor y[n]` | xᵀ·A（行向量乘矩阵） | matVecT(x,A)=matmulVec(A.T,x) |
| `D3-051` | `linalg.vander` | `(Tensor x, number n?) -> Tensor` | Vandermonde 矩阵（列幂次 n-1..0） | vander(x,n) 列 = x 的幂 |
| `D3-052` | `linalg.polyfit` | `(Tensor x, Tensor y, number deg) -> Tensor` | 最小二乘多项式拟合（Vandermonde + lstsq） | 残差最小；deg≥n-1 时精确插值 |
| `D3-053` | `linalg.polyval` | `(Tensor coefs, Tensor x) -> Tensor` | Horner 多项式求值（coefs 高次→低次） | 对单点退化为 num.horner |
| `D3-054` | `linalg.lstsq` | `(Tensor A, Tensor b, number rcond=1e-12) -> object` | 最小二乘解（QR 高效路径；秩亏回退 SVD） | ‖Ax−b‖ 最小；满秩列时 x=A⁺b |
| `D3-055` | `linalg.nullSpace` | `(Tensor A, number tol?) -> Tensor` | 零空间基（SVD 右奇异向量） | A·N=0；N 列正交 |
| `D3-056` | `linalg.kron` | `(Tensor A[m,n], Tensor B[p,q]) -> Tensor[mp,nq]` | Kronecker 积 | kron(A,I)=块对角扩展 |
| `D3-057` | `linalg.vec` | `(Tensor A[m,n]) -> Tensor[mn]` | 列堆叠向量化 vec(A)[j·m+i]=A[i,j] | vec(Iₙ) = [1,0,..,0,1,..]；unvec(vec(A))=A |
| `D3-058` | `linalg.unvec` | `(Tensor x[mn], number m) -> Tensor[m,n]` | 列堆叠反向量化 | unvec(vec(A))=A |
| `D3-059` | `linalg.householder` | `(Tensor x) -> object {v, beta}` | Householder 向量（反射 Hx=±‖x‖e₁） | H=I−βvvᵀ；Hx 与 x 模长相同 |
| `D3-060` | `linalg.givens` | `(number a, number b) -> object {c,s,r}` | Givens 旋转（消去 b） | c²+s²=1；[c s;-s c][a;b]=[r;0]（r=√(a²+b²)） |
| `D3-061` | `linalg.convMatrix` | `(Tensor kernel[k], number n) -> Tensor[n,n]` | 1D 卷积矩阵（Toeplitz，same 语义） | convMatrix(k,n)·x = conv(x,k) |
| `D3-062` | `linalg.circulantMatvec` | `(Tensor c[n], Tensor x[n]) -> Tensor` | 循环矩阵乘法（O(n²)，无需显式构造） | circulant 矩阵作用于向量 = 循环卷积 |
| `D3-063` | `linalg.isSymmetric` | `(Tensor A) -> boolean` | 判定矩阵是否对称 | 对对称矩阵返回 true |
| `D3-064` | `linalg.isUpperTri` | `(Tensor A) -> boolean` | 判定矩阵是否上三角 | 对上三角矩阵返回 true |
| `D3-065` | `linalg.isLowerTri` | `(Tensor A) -> boolean` | 判定矩阵是否下三角 | 对下三角矩阵返回 true |
| `D3-066` | `linalg.isDiagonal` | `(Tensor A) -> boolean` | 判定矩阵是否对角 | 对对角矩阵返回 true |
| `D3-067` | `linalg.isOrthogonal` | `(Tensor A) -> boolean` | 判定矩阵是否正交 | 对正交矩阵返回 true |
| `D3-068` | `linalg.isSPD` | `(Tensor A) -> boolean` | 判定矩阵是否对称正定 | 对对称正定矩阵返回 true |
| `D3-069` | `linalg.isIdempotent` | `(Tensor A) -> boolean` | 判定矩阵是否幂等（A²=A） | 对幂等（A²=A）矩阵返回 true |
| `D3-070` | `linalg.eyeLike` | `(Tensor A) -> Tensor` | 与 A 同形状单位阵 | eyeLike(A) 与 A 同形 |
| `D3-071` | `linalg.zerosLike` | `(Tensor A) -> Tensor` | 同形状零矩阵 | 输出全零张量，形状与输入完全相同 |
| `D3-072` | `linalg.onesLike` | `(Tensor A) -> Tensor` | 同形状全一矩阵 | 输出全一张量，形状与输入完全相同 |
| `D3-073` | `linalg.fullLike` | `(Tensor A, number v) -> Tensor` | 同形状常数矩阵 | 输出全部填充标量 v，形状与输入相同 |
| `D3-074` | `linalg.concatRows` | `(Tensor A, Tensor B) -> Tensor` | 纵向拼接（列数须一致） | 沿行（轴0）拼接，输出行数等于各输入行数之和 |
| `D3-075` | `linalg.concatCols` | `(Tensor A, Tensor B) -> Tensor` | 横向拼接（行数须一致） | 沿列（轴1）拼接，输出列数等于各输入列数之和 |
| `D3-076` | `linalg.submatrixRows` | `(Tensor A, number r0, number r1) -> Tensor` | 取行区间 [r0,r1) | 行数=r1−r0 |
| `D3-077` | `linalg.submatrixCols` | `(Tensor A, number c0, number c1) -> Tensor` | 取列区间 [c0,c1) | 列数=c1−c0 |
| `D3-078` | `linalg.center` | `(Tensor X[n,d]) -> Tensor` | 按列去均值 | 每列均值=0 |
| `D3-079` | `linalg.normalizeRows` | `(Tensor X) -> Tensor` | 行 L2 归一化 | 每行 L2 范数=1 |
| `D3-080` | `linalg.rowSum` | `(Tensor A) -> Tensor` | 矩阵行和 | 与 math 轴归约一致 |
| `D3-081` | `linalg.colSum` | `(Tensor A) -> Tensor` | 矩阵列和 | 与 math 轴归约一致 |
| `D3-082` | `linalg.rowMean` | `(Tensor A) -> Tensor` | 矩阵行均值 | 与 math 轴归约一致 |
| `D3-083` | `linalg.colMean` | `(Tensor A) -> Tensor` | 矩阵列均值 | 与 math 轴归约一致 |
| `D3-084` | `linalg.rowMax` | `(Tensor A) -> Tensor` | 矩阵行最大 | 与 math 轴归约一致 |
| `D3-085` | `linalg.colMax` | `(Tensor A) -> Tensor` | 矩阵列最大 | 与 math 轴归约一致 |
| `D3-086` | `linalg.rowMin` | `(Tensor A) -> Tensor` | 矩阵行最小 | 与 math 轴归约一致 |
| `D3-087` | `linalg.colMin` | `(Tensor A) -> Tensor` | 矩阵列最小 | 与 math 轴归约一致 |
| `D3-088` | `linalg.trace` | `(Tensor A) -> number` | 矩阵迹 ΣAᵢᵢ | trace(AB)=trace(BA)；trace(A)=Σλᵢ |
| `D3-089` | `linalg.transpose` | `(Tensor A) -> Tensor` | 矩阵转置（指令化入口） | transpose(transpose(A))=A |
| `D3-090` | `linalg.diag` | `(Tensor A) -> Tensor` | 提取对角线 | diag(diagMat(v))=v |
| `D3-091` | `linalg.diagMat` | `(Tensor v, number n?) -> Tensor` | 对角线构造矩阵 | diagMat(diag(A))=A（方阵） |
| `D3-092` | `linalg.scaleMat` | `(number alpha, Tensor A) -> Tensor` | 数乘 αA | scaleMat(1,A)=A |
| `D3-093` | `linalg.addMat` | `(Tensor A, Tensor B) -> Tensor` | 矩阵加法（广播） | addMat(A,B)=addMat(B,A) |
| `D3-094` | `linalg.subMat` | `(Tensor A, Tensor B) -> Tensor` | 矩阵减法（广播） | subMat(A,A)=0 |
| `D3-095` | `linalg.hadamard` | `(Tensor A, Tensor B) -> Tensor` | Hadamard 积（逐元素） | hadamard(A,B)=hadamard(B,A) |
| `D3-096` | `linalg.triLower` | `(Tensor A, number k=0) -> Tensor` | 下三角部分（第 k 条对角以下） | triLower(triLower(A))=triLower(A) |
| `D3-097` | `linalg.triUpper` | `(Tensor A, number k=0) -> Tensor` | 上三角部分（第 k 条对角以上） | triUpper(triUpper(A))=triUpper(A) |
| `D3-098` | `linalg.polar` | `(Tensor A) -> object {U, H}` | 极分解 A=U·H（U 正交，H SPD） | U 正交；H SPD；U·H=A |
| `D3-099` | `linalg.colSpace` | `(Tensor A, number tol?) -> Tensor` | 列空间正交基（SVD 左奇异向量） | 列正交；每列在 A 的列空间中 |
| `D3-100` | `linalg.rowSpace` | `(Tensor A, number tol?) -> Tensor` | 行空间正交基（SVD 右奇异向量） | QR 正交化给出行空间的标准正交基 |
| `D3-101` | `linalg.blockDiag` | `(Tensor[] blocks) -> Tensor` | 块对角矩阵 | 对角块为输入，非块区为零 |
| `D3-102` | `linalg.flipLR` | `(Tensor A) -> Tensor` | 沿列轴翻转 | flipLR(flipLR(A))=A |

## D4 · AUTODIFF 自动微分

反向模式 vjp、前向模式对偶数、梯度检验。共 102 条。

| id | 指令 | 签名 | 说明 | 不变量 |
|---|---|---|---|---|
| `D4-001` | `ad.exp` | `(Tensor x) -> Tensor` | 指数 e^x（带梯度） | d/dx ad.exp = e^x |
| `D4-002` | `ad.log` | `(Tensor x) -> Tensor` | 对数 ln x（x>0） | d/dx ad.log = 1/x |
| `D4-003` | `ad.log1p` | `(Tensor x) -> Tensor` | ln(1+x) | d/dx ad.log1p = 1/(1+x) |
| `D4-004` | `ad.sqrt` | `(Tensor x) -> Tensor` | 平方根 √x（带梯度） | d/dx ad.sqrt = 1/(2√x) |
| `D4-005` | `ad.sin` | `(Tensor x) -> Tensor` | 正弦 sin x（带梯度） | d/dx ad.sin = cos x |
| `D4-006` | `ad.cos` | `(Tensor x) -> Tensor` | 余弦 cos x（带梯度） | d/dx ad.cos = -sin x |
| `D4-007` | `ad.tanh` | `(Tensor x) -> Tensor` | 双曲正切 | d/dx ad.tanh = 1 - tanh²x |
| `D4-008` | `ad.sigmoid` | `(Tensor x) -> Tensor` | sigmoid σ(x)=1/(1+e^{-x}) | d/dx ad.sigmoid = σ(x)(1-σ(x)) |
| `D4-009` | `ad.expm1` | `(Tensor x) -> Tensor` | e^x - 1 | d/dx ad.expm1 = e^x |
| `D4-010` | `ad.square` | `(Tensor x) -> Tensor` | 逐元素平方 x²（带梯度） | d/dx ad.square = 2x |
| `D4-011` | `ad.reciprocal` | `(Tensor x) -> Tensor` | 逐元素倒数 1/x（带梯度） | d/dx ad.reciprocal = -1/x² |
| `D4-012` | `ad.relu` | `(Tensor x) -> Tensor` | ReLU max(0,x)（0 点次梯度取 0） | x>0 时梯度 1，否则 0 |
| `D4-013` | `ad.leakyRelu` | `(Tensor x, number alpha=0.01) -> Tensor` | LeakyReLU（负区间斜率 alpha） | x>0 梯度 1，否则 alpha |
| `D4-014` | `ad.elu` | `(Tensor x, number alpha=1) -> Tensor` | ELU（x>0 取 x，否则 α(e^x−1)） | x>0 梯度 1，否则 out+α |
| `D4-015` | `ad.silu` | `(Tensor x) -> Tensor` | SiLU/Swish x·σ(x) | d/dx = σ(x)(1 + x(1-σ(x))) |
| `D4-016` | `ad.gelu` | `(Tensor x) -> Tensor` | GELU（tanh 近似） | d/dx ≈ Φ(x) + x·φ(x)（精确版） |
| `D4-017` | `ad.softplus` | `(Tensor x) -> Tensor` | softplus log(1+e^x)（稳定分支） | d/dx = σ(x) |
| `D4-018` | `ad.abs` | `(Tensor x) -> Tensor` | 绝对值（0 点次梯度 0） | d/dx = sign(x) |
| `D4-019` | `ad.clamp` | `(Tensor x, number lo, number hi) -> Tensor` | 截断（界外梯度 0） | 界内梯度 1 |
| `D4-020` | `ad.maximum` | `(Tensor x, Tensor y) -> Tensor` | 逐元素最大（平局梯度平分） | d/dx = [x≥y] |
| `D4-021` | `ad.minimum` | `(Tensor x, Tensor y) -> Tensor` | 逐元素最小（平局梯度平分） | d/dx = [x≤y] |
| `D4-022` | `ad.pow` | `(Tensor x, number p) -> Tensor` | 幂 x^p | d/dx = p·x^(p−1) |
| `D4-023` | `ad.add` | `(Tensor\|number a, Tensor\|number b) -> Tensor` | 加法（广播） | d/da = g，d/db = g |
| `D4-024` | `ad.sub` | `(Tensor\|number a, Tensor\|number b) -> Tensor` | 减法（广播） | d/da=g，d/db=−g |
| `D4-025` | `ad.mul` | `(Tensor\|number a, Tensor\|number b) -> Tensor` | 乘法（广播） | d/da = g·b，d/db = g·a |
| `D4-026` | `ad.div` | `(Tensor\|number a, Tensor\|number b) -> Tensor` | 除法（广播） | d/da = g/b，d/db = −g·a/b² |
| `D4-027` | `ad.neg` | `(Tensor x) -> Tensor` | 逐元素取负 −x（带梯度） | d/dx = −g |
| `D4-028` | `ad.matmul` | `(Tensor A[m,k], Tensor B[k,n]) -> Tensor` | 矩阵乘法（反向：dA=g·Bᵀ，dB=Aᵀ·g） | dA=g·Bᵀ；dB=Aᵀ·g |
| `D4-029` | `ad.matmulAT` | `(Tensor A[k,m], Tensor B[k,n]) -> Tensor` | Aᵀ·B | dA=g·Bᵀ 的转置形式 |
| `D4-030` | `ad.vecDot` | `(Tensor x[n], Tensor y[n]) -> Tensor` | 向量点积（标量输出张量） | dx = g·y，dy = g·x |
| `D4-031` | `ad.transpose` | `(Tensor A) -> Tensor` | 矩阵转置（反向传播仍为转置） | 转置运算的反向传播仍是转置（vjp 为自身转置） |
| `D4-032` | `ad.reshape` | `(Tensor x, number[] shape) -> Tensor` | reshape（梯度反 reshape） | 梯度回到原形状 |
| `D4-033` | `ad.broadcastTo` | `(Tensor x, number[] shape) -> Tensor` | 显式广播（梯度 sumTo 原形状） | d/dx = sumTo(g) |
| `D4-034` | `ad.sumAll` | `(Tensor x) -> Tensor` | 全元素求和（标量） | d/dx = g 广播 |
| `D4-035` | `ad.meanAll` | `(Tensor x) -> Tensor` | 全元素均值（标量） | d/dx = g/n |
| `D4-036` | `ad.sumAxis` | `(Tensor x, number axis) -> Tensor` | 沿轴求和 | 梯度沿轴广播 |
| `D4-037` | `ad.meanAxis` | `(Tensor x, number axis) -> Tensor` | 沿轴均值 | 梯度 = g广播/轴长 |
| `D4-038` | `ad.maxAll` | `(Tensor x) -> Tensor` | 全元素最大（次梯度给首个最大元） | 梯度集中在 argmax 处 |
| `D4-039` | `ad.softmax` | `(Tensor x, number axis=-1) -> Tensor` | softmax（雅可比 dx=(g−Σg⊙y)⊙y） | 沿轴和=1；Jacobian-vector product 公式 |
| `D4-040` | `ad.logSoftmax` | `(Tensor x, number axis=-1) -> Tensor` | log-softmax | dx = g − softmax(x)·Σg |
| `D4-041` | `ad.mse` | `(Tensor pred, Tensor target) -> Tensor` | 均方误差（标量） | d/dpred = 2(pred−target)/n |
| `D4-042` | `ad.mae` | `(Tensor pred, Tensor target) -> Tensor` | 平均绝对误差（标量） | d/dpred = sign(pred−target)/n |
| `D4-043` | `ad.huber` | `(Tensor pred, Tensor target, number delta=1) -> Tensor` | Huber 损失（L1/L2 混合） | \|d\|≤δ 用 L2，否则 L1 |
| `D4-044` | `ad.crossEntropy` | `(Tensor logits[n,c], Tensor labels[n,c] one-hot) -> Tensor` | 交叉熵（与 log-softmax 融合） | d/dlogits = softmax − labels |
| `D4-045` | `ad.bce` | `(Tensor logits, Tensor targets) -> Tensor` | 二元交叉熵（logits 形式，数值稳定） | d/dlogits = σ(x) − t |
| `D4-046` | `ad.nll` | `(Tensor logProbs[n,c], Tensor classIdx[n]) -> Tensor` | 负对数似然（类索引） | d/dlogProb = −1/p[cls]/n |
| `D4-047` | `ad.klDiv` | `(Tensor logP, Tensor logQ) -> Tensor` | KL 散度（log 空间输入） | KL(p‖q)≥0；p=q 时为 0 |
| `D4-048` | `ad.getItem` | `(Tensor x, number[] index) -> Tensor` | 索引取子张量（梯度 scatter 回原位） | 梯度回填到 index 处 |
| `D4-049` | `ad.concat` | `(Tensor[] ts, number axis) -> Tensor` | 拼接（梯度按段切分） | 各段梯度之和 |
| `D4-050` | `ad.gatherRows` | `(Tensor table[n,d], Tensor idx[k]) -> Tensor` | 行 gather（embedding lookup；重复索引梯度累加） | 重复行梯度累加 |
| `D4-051` | `ad.scatterAdd` | `(Tensor dst, Tensor idx, Tensor src) -> Tensor` | scatter-add（gatherRows 的逆） | 重复索引累加 |
| `D4-052` | `ad.where` | `(Tensor mask, Tensor a, Tensor b) -> Tensor` | 掩码选择（梯度走对应分支） | 梯度按 mask 分流 |
| `D4-053` | `ad.padConst` | `(Tensor x, number[][] pads, number v=0) -> Tensor` | 常数填充（梯度裁剪掉填充区） | 梯度去掉填充区 |
| `D4-054` | `ad.stack` | `(Tensor[] ts, number axis) -> Tensor` | 沿新轴堆叠 | 梯度拆回各输入 |
| `D4-055` | `ad.detach` | `(Tensor x) -> Tensor` | 切断梯度（常量视图） | 不记录到磁带 |
| `D4-056` | `ad.tapeReset` | `() -> void` | 清空磁带 | graphSize=0 |
| `D4-057` | `ad.graphSize` | `() -> number` | 当前磁带节点数 | 计算图节点数与边数均为非负整数 |
| `D4-058` | `ad.backward` | `(Tensor loss, Tensor seed?) -> Map` | 反向传播（从 loss 出发，返回 张量→梯度 Map） | 梯度累积规则：多路径自动相加 |
| `D4-059` | `ad.grad` | `(Function f, Tensor x) -> Tensor` | f(x) 对标量的梯度（自动磁带管理） | 与中心差分一致（gradcheck 1e-6） |
| `D4-060` | `ad.vjp` | `(Function f, Tensor x, Tensor u) -> Tensor` | 向量-雅可比积（f 可为向量函数） | vjp(f,x,u)·v = u·(Jv) |
| `D4-061` | `ad.noGrad` | `(Function f) -> any` | 无梯度上下文执行 | 执行期间不记录 |
| `D4-062` | `ad.gradcheck` | `(Function f, Tensor x, number eps=1e-5, number tol=1e-6) -> object` | 梯度检验（中心差分 vs 反向模式） | 最大相对误差 < tol |
| `D4-063` | `ad.gradcheckVec` | `(Function f, Tensor x, Tensor u, number eps=1e-5) -> object` | 向量函数梯度检验（随机投影法） | u·vjp 与中心差分一致 |
| `D4-064` | `ad.jacobianDiag` | `(Function f, Tensor x, number eps=1e-4) -> Tensor` | 雅可比对角线（对向量函数） | J[i,i] = ∂fᵢ/∂xᵢ |
| `D4-065` | `ad.dense` | `(Tensor x[n,in], Tensor W[in,out], Tensor b[out], Function act?) -> Tensor` | 全连接层 x·W+b + 激活 | 等价于 ad.matmul+ad.add+act 组合 |
| `D4-066` | `ad.layerNorm` | `(Tensor x[n,d], Tensor gamma[d], Tensor beta[d], number eps=1e-5) -> Tensor` | LayerNorm（按最后一维） | 输出均值≈0 方差≈1（缩放前） |
| `D4-067` | `ad.scaledDotProductAttention` | `(Tensor Q[n,d], Tensor K[n,d], Tensor V[n,d], number scale?) -> Tensor` | 缩放点积注意力 softmax(QKᵀ/√d)V | softmax 行和=1；gradcheck 通过 |
| `D4-068` | `ad.mlp` | `(Tensor x, Tensor W1, Tensor b1, Tensor W2, Tensor b2) -> Tensor` | 两层 MLP（tanh 隐层，线性输出） | 等价于 dense(dense(x,W1,b1,tanh),W2,b2) |
| `D4-069` | `ad.dropout` | `(Tensor x, number p, Tensor mask) -> Tensor` | Dropout（外置掩码，保证可复现与可微） | 训练期期望不变（inverted dropout：x·mask/(1−p)） |
| `D4-070` | `ad.dual.add` | `(Dual a, Dual b) -> Dual` | 对偶加法 | tangent = 链式法则组合 |
| `D4-071` | `ad.dual.sub` | `(Dual a, Dual b) -> Dual` | 对偶减法 | tangent = 链式法则组合 |
| `D4-072` | `ad.dual.mul` | `(Dual a, Dual b) -> Dual` | 对偶乘法 | tangent = 链式法则组合 |
| `D4-073` | `ad.dual.div` | `(Dual a, Dual b) -> Dual` | 对偶除法 | tangent = 链式法则组合 |
| `D4-074` | `ad.dual.from` | `(Tensor x, Tensor v?) -> Dual` | 由张量构造对偶数 | primal=x，tangent=v（默认 0） |
| `D4-075` | `ad.dual.primal` | `(Dual a) -> Tensor` | 取对偶数的原值部分 primal(a) | primal(dual.from(x))=x |
| `D4-076` | `ad.dual.tangent` | `(Dual a) -> Tensor` | 取切线（导数） | tangent(dual.from(x,v))=v |
| `D4-077` | `ad.dual.exp` | `(Dual a) -> Dual` | 对偶指数 | tangent = f'(primal)·tangent |
| `D4-078` | `ad.dual.log` | `(Dual a) -> Dual` | 对偶对数 | tangent = f'(primal)·tangent |
| `D4-079` | `ad.dual.sqrt` | `(Dual a) -> Dual` | 对偶平方根 | tangent = f'(primal)·tangent |
| `D4-080` | `ad.dual.sin` | `(Dual a) -> Dual` | 对偶正弦 | tangent = f'(primal)·tangent |
| `D4-081` | `ad.dual.cos` | `(Dual a) -> Dual` | 对偶余弦 | tangent = f'(primal)·tangent |
| `D4-082` | `ad.dual.tanh` | `(Dual a) -> Dual` | 对偶双曲正切 | tangent = f'(primal)·tangent |
| `D4-083` | `ad.dual.sigmoid` | `(Dual a) -> Dual` | 对偶 sigmoid | tangent = f'(primal)·tangent |
| `D4-084` | `ad.dual.matmul` | `(Dual A, Dual B) -> Dual` | 对偶矩阵乘法 | primal=A·B，tangent=Ȧ·B+A·B̄ |
| `D4-085` | `ad.dual.sum` | `(Dual a) -> Dual` | 对偶求和 | tangent 同样求和 |
| `D4-086` | `ad.addScalar` | `(Tensor x, number s) -> Tensor` | 加法（张量+标量）（记录梯度） | 等价 ${base} 的标量路径 |
| `D4-087` | `ad.subScalar` | `(Tensor x, number s) -> Tensor` | 减法（张量−标量）（记录梯度） | 等价 ${base} 的标量路径 |
| `D4-088` | `ad.mulScalar` | `(Tensor x, number s) -> Tensor` | 数乘（张量×标量）（记录梯度） | 等价 ${base} 的标量路径 |
| `D4-089` | `ad.divScalar` | `(Tensor x, number s) -> Tensor` | 除法（张量÷标量）（记录梯度） | 等价 ${base} 的标量路径 |
| `D4-090` | `ad.rsubScalar` | `(Tensor x, number s) -> Tensor` | 反减（标量−张量）（记录梯度） | 等价 ${base} 的标量路径 |
| `D4-091` | `ad.rdivScalar` | `(Tensor x, number s) -> Tensor` | 反除（标量÷张量）（记录梯度） | 等价 ${base} 的标量路径 |
| `D4-092` | `ad.softsign` | `(Tensor x) -> Tensor` | softsign x/(1+\|x\|) | d/dx ad.softsign = 1/(1+\|x\|)² |
| `D4-093` | `ad.mish` | `(Tensor x) -> Tensor` | mish x·tanh(softplus x) | d/dx ad.mish = tanh(sp)+x·σ(x)·sech²(sp) |
| `D4-094` | `ad.hardtanh` | `(Tensor x, number lo=-1, number hi=1) -> Tensor` | HardTanh（线性区截断） | 界内梯度 1 |
| `D4-095` | `ad.hardswish` | `(Tensor x) -> Tensor` | HardSwish x·ReLU6(x+3)/6 | gradcheck 通过 |
| `D4-096` | `ad.sign` | `(Tensor x) -> Tensor` | 符号函数（0 点次梯度 0） | d/dx ad.sign = 0 |
| `D4-097` | `ad.rsqrt` | `(Tensor x) -> Tensor` | 逆平方根 1/√x | d/dx ad.rsqrt = -1/(2x^{3/2}) |
| `D4-098` | `ad.scaleAdd` | `(number alpha, Tensor x, Tensor y) -> Tensor` | axpy：α·x+y | d/dx=αg，d/dy=g |
| `D4-099` | `ad.l2Reg` | `(Tensor w, number lambda) -> Tensor` | L2 正则项 λ/2·‖w‖² | d/dw = λw |
| `D4-100` | `ad.l1Reg` | `(Tensor w, number lambda) -> Tensor` | L1 正则项 λ·‖w‖₁ | d/dw = λ·sign(w) |
| `D4-101` | `ad.causalMask` | `(Tensor scores, number fill=-Infinity) -> Tensor` | 因果掩码（上三角置 fill，防未来泄漏） | 下三角保持，上三角=fill |
| `D4-102` | `ad.batchnorm1d` | `(Tensor x[n,d], Tensor gamma[d], Tensor beta[d], number eps=1e-5) -> Tensor` | BatchNorm1d（按特征维，训练期批统计） | 输出批均值≈0 方差≈1（缩放前） |

## D5 · OPT 优化器

16 种经典优化器、学习率调度、梯度变换、线搜索。共 102 条。

| id | 指令 | 签名 | 说明 | 不变量 |
|---|---|---|---|---|
| `D5-001` | `opt.sgd` | `(Tensor params, object cfg) -> object` | 创建优化器 sgd 的初始 state（t=0，动量/二阶矩缓冲区为零数组） | 更新量 = −η·g |
| `D5-002` | `opt.sgd.step` | `(Tensor params, Tensor grads, object state, object cfg) -> {params: Tensor, state: object}` | sgd 单步更新：梯度下降单步：θ ← θ − η·g | 更新量 = −η·g |
| `D5-003` | `opt.sgdMomentum` | `(Tensor params, object cfg) -> object` | 创建优化器 sgdMomentum 的初始 state（t=0，动量/二阶矩缓冲区为零数组） | v_t = β·v_{t−1} + g_t，更新量 = −η·v_t |
| `D5-004` | `opt.sgdMomentum.step` | `(Tensor params, Tensor grads, object state, object cfg) -> {params: Tensor, state: object}` | sgdMomentum 单步更新：经典动量（Heavy-ball）单步：v ← β·v + g，θ ← θ − η·v | v_t = β·v_{t−1} + g_t，更新量 = −η·v_t |
| `D5-005` | `opt.sgdNesterov` | `(Tensor params, object cfg) -> object` | 创建优化器 sgdNesterov 的初始 state（t=0，动量/二阶矩缓冲区为零数组） | 更新量 = −η·(g + β·v)，等价于在预估点处取梯度 |
| `D5-006` | `opt.sgdNesterov.step` | `(Tensor params, Tensor grads, object state, object cfg) -> {params: Tensor, state: object}` | sgdNesterov 单步更新：Nesterov 动量单步：v ← β·v + g，θ ← θ − η·(g + β·v) | 更新量 = −η·(g + β·v)，等价于在预估点处取梯度 |
| `D5-007` | `opt.adagrad` | `(Tensor params, object cfg) -> object` | 创建优化器 adagrad 的初始 state（t=0，动量/二阶矩缓冲区为零数组） | 更新量 = −η·g_i/(√G_i + ε)，G 为梯度平方累积和 |
| `D5-008` | `opt.adagrad.step` | `(Tensor params, Tensor grads, object state, object cfg) -> {params: Tensor, state: object}` | adagrad 单步更新：Adagrad 单步：G ← G + g⊙g，θ ← θ − η·g/(√G + ε) | 更新量 = −η·g_i/(√G_i + ε)，G 为梯度平方累积和 |
| `D5-009` | `opt.adadelta` | `(Tensor params, object cfg) -> object` | 创建优化器 adadelta 的初始 state（t=0，动量/二阶矩缓冲区为零数组） | Δθ_i = −η·√(EΔθ²_i+ε)/√(Eg²_i+ε)·g_i；η=1 时无需全局学习率，最优点附近存在已知 RMS(Δx) 振荡 |
| `D5-010` | `opt.adadelta.step` | `(Tensor params, Tensor grads, object state, object cfg) -> {params: Tensor, state: object}` | adadelta 单步更新：Adadelta 单步：Δθ = −η·√(EΔθ²+ε)/√(Eg²+ε)·g，二阶矩用窗口 EMA（η=1 即原文） | Δθ_i = −η·√(EΔθ²_i+ε)/√(Eg²_i+ε)·g_i；η=1 时无需全局学习率，最优点附近存在已知 RMS(Δx) 振荡 |
| `D5-011` | `opt.rmsprop` | `(Tensor params, object cfg) -> object` | 创建优化器 rmsprop 的初始 state（t=0，动量/二阶矩缓冲区为零数组） | 更新量 = −η·g/(√v+ε)，v = βv + (1−β)g² |
| `D5-012` | `opt.rmsprop.step` | `(Tensor params, Tensor grads, object state, object cfg) -> {params: Tensor, state: object}` | rmsprop 单步更新：RMSprop 单步：v ← βv + (1−β)g²，θ ← θ − η·g/(√v + ε) | 更新量 = −η·g/(√v+ε)，v = βv + (1−β)g² |
| `D5-013` | `opt.adam` | `(Tensor params, object cfg) -> object` | 创建优化器 adam 的初始 state（t=0，动量/二阶矩缓冲区为零数组） | 更新量 = −η·m̂/(√v̂+ε)，偏差修正后 m̂=m/(1−β1^t)，v̂=v/(1−β2^t) |
| `D5-014` | `opt.adam.step` | `(Tensor params, Tensor grads, object state, object cfg) -> {params: Tensor, state: object}` | adam 单步更新：Adam 单步：m,v 为 EMA，偏差修正后 θ ← θ − η·m̂/(√v̂ + ε) | 更新量 = −η·m̂/(√v̂+ε)，偏差修正后 m̂=m/(1−β1^t)，v̂=v/(1−β2^t) |
| `D5-015` | `opt.adamw` | `(Tensor params, object cfg) -> object` | 创建优化器 adamw 的初始 state（t=0，动量/二阶矩缓冲区为零数组） | 更新量 = −η·(m̂/(√v̂+ε) + λ·θ)，权重衰减不经二阶矩归一化 |
| `D5-016` | `opt.adamw.step` | `(Tensor params, Tensor grads, object state, object cfg) -> {params: Tensor, state: object}` | adamw 单步更新：AdamW 单步：Adam 更新 + 解耦权重衰减 θ ← θ − η·(m̂/(√v̂+ε) + λθ) | 更新量 = −η·(m̂/(√v̂+ε) + λ·θ)，权重衰减不经二阶矩归一化 |
| `D5-017` | `opt.nadam` | `(Tensor params, object cfg) -> object` | 创建优化器 nadam 的初始 state（t=0，动量/二阶矩缓冲区为零数组） | 更新量 = −η·ψ/(√v̂+ε)，ψ = (β1·m_t + (1−β1)·g_t)/(1−β1^t) |
| `D5-018` | `opt.nadam.step` | `(Tensor params, Tensor grads, object state, object cfg) -> {params: Tensor, state: object}` | nadam 单步更新：Nadam 单步：一阶矩 Nesterov 前瞻 ψ = (β1·m + (1−β1)·g)/(1−β1^t) | 更新量 = −η·ψ/(√v̂+ε)，ψ = (β1·m_t + (1−β1)·g_t)/(1−β1^t) |
| `D5-019` | `opt.amsgrad` | `(Tensor params, object cfg) -> object` | 创建优化器 amsgrad 的初始 state（t=0，动量/二阶矩缓冲区为零数组） | 更新量 = −η·m̂/(√v̂max+ε)，v̂max = max(v̂max, v̂) |
| `D5-020` | `opt.amsgrad.step` | `(Tensor params, Tensor grads, object state, object cfg) -> {params: Tensor, state: object}` | amsgrad 单步更新：AMSGrad 单步：用 v̂ 的历史最大值 v̂max 代替 v̂ 参与更新 | 更新量 = −η·m̂/(√v̂max+ε)，v̂max = max(v̂max, v̂) |
| `D5-021` | `opt.adamax` | `(Tensor params, object cfg) -> object` | 创建优化器 adamax 的初始 state（t=0，动量/二阶矩缓冲区为零数组） | 更新量 = −η·m̂/(u+ε)，u = max(β2·u, \|g\|) |
| `D5-022` | `opt.adamax.step` | `(Tensor params, Tensor grads, object state, object cfg) -> {params: Tensor, state: object}` | adamax 单步更新：Adamax 单步：∞-范数变体 u ← max(β2·u, \|g\|)，θ ← θ − η·m̂/(u + ε) | 更新量 = −η·m̂/(u+ε)，u = max(β2·u, \|g\|) |
| `D5-023` | `opt.lamb` | `(Tensor params, object cfg) -> object` | 创建优化器 lamb 的初始 state（t=0，动量/二阶矩缓冲区为零数组） | trust = ‖φ‖₂/‖u‖₂（u = φ + λ·θ），更新量 = −η·trust·u |
| `D5-024` | `opt.lamb.step` | `(Tensor params, Tensor grads, object state, object cfg) -> {params: Tensor, state: object}` | lamb 单步更新：LAMB 单步：Adam 步 φ 加信任比缩放，θ ← θ − η·trust·(φ + λθ) | trust = ‖φ‖₂/‖u‖₂（u = φ + λ·θ），更新量 = −η·trust·u |
| `D5-025` | `opt.rprop` | `(Tensor params, object cfg) -> object` | 创建优化器 rprop 的初始 state（t=0，动量/二阶矩缓冲区为零数组） | 符号一致 η_i ← min(η_i·η⁺, ηMax)，相反 η_i ← max(η_i·η⁻, ηMin)；更新量 = −sign(g)·η_i |
| `D5-026` | `opt.rprop.step` | `(Tensor params, Tensor grads, object state, object cfg) -> {params: Tensor, state: object}` | rprop 单步更新：Rprop 单步：仅用梯度符号，步长 η_i 按符号一致性自适应增减 | 符号一致 η_i ← min(η_i·η⁺, ηMax)，相反 η_i ← max(η_i·η⁻, ηMin)；更新量 = −sign(g)·η_i |
| `D5-027` | `opt.yogi` | `(Tensor params, object cfg) -> object` | 创建优化器 yogi 的初始 state（t=0，动量/二阶矩缓冲区为零数组） | 更新量 = −η·m̂/(√v̂+ε)，v_t = v_{t−1} − (1−β2)·sign(v_{t−1}−g²)·g² |
| `D5-028` | `opt.yogi.step` | `(Tensor params, Tensor grads, object state, object cfg) -> {params: Tensor, state: object}` | yogi 单步更新：Yogi 单步：二阶矩用符号控制增量 v ← v − (1−β2)·sign(v−g²)·g² | 更新量 = −η·m̂/(√v̂+ε)，v_t = v_{t−1} − (1−β2)·sign(v_{t−1}−g²)·g² |
| `D5-029` | `opt.lion` | `(Tensor params, object cfg) -> object` | 创建优化器 lion 的初始 state（t=0，动量/二阶矩缓冲区为零数组） | 更新量 = −η·(sign(c) + λ·θ)，c = β1·m + (1−β1)·g；m ← β2·m + (1−β2)·g |
| `D5-030` | `opt.lion.step` | `(Tensor params, Tensor grads, object state, object cfg) -> {params: Tensor, state: object}` | lion 单步更新：Lion 单步：update = −η·(sign(β1·m + (1−β1)·g) + λθ)，动量用 β2 | 更新量 = −η·(sign(c) + λ·θ)，c = β1·m + (1−β1)·g；m ← β2·m + (1−β2)·g |
| `D5-031` | `opt.signSgd` | `(Tensor params, object cfg) -> object` | 创建优化器 signSgd 的初始 state（t=0，动量/二阶矩缓冲区为零数组） | 更新量 = −η·sign(g)，逐元素取 ±η |
| `D5-032` | `opt.signSgd.step` | `(Tensor params, Tensor grads, object state, object cfg) -> {params: Tensor, state: object}` | signSgd 单步更新：SignSGD 单步：θ ← θ − η·sign(g) | 更新量 = −η·sign(g)，逐元素取 ±η |
| `D5-033` | `opt.lr.constant` | `(object cfg, number epoch) -> number` | 学习率调度 constant：常数学习率：lr(t) = η | lr(t) ≡ η |
| `D5-034` | `opt.lr.step` | `(object cfg, number epoch) -> number` | 学习率调度 step：阶梯衰减：每 stepSize 个 epoch 乘以 γ | lr(t) = η·γ^⌊t/stepSize⌋ |
| `D5-035` | `opt.lr.multistep` | `(object cfg, number epoch) -> number` | 学习率调度 multistep：多里程碑衰减：每越过一个 milestone 乘以 γ | lr(t) = η·γ^{#{m∈milestones: t≥m}} |
| `D5-036` | `opt.lr.exp` | `(object cfg, number epoch) -> number` | 学习率调度 exp：指数衰减：lr(t) = η·γ^t | lr(t) = η·γ^t |
| `D5-037` | `opt.lr.cosine` | `(object cfg, number epoch) -> number` | 学习率调度 cosine：余弦退火到 lrMin | lr(t) = lrMin + (η−lrMin)·½(1+cos(π·min(t,T)/T)) |
| `D5-038` | `opt.lr.warmup` | `(object cfg, number epoch) -> number` | 学习率调度 warmup：线性 warmup 后保持常数 | t<W 时 lr(t) = start + (η−start)·t/W，否则 η |
| `D5-039` | `opt.lr.warmupCosine` | `(object cfg, number epoch) -> number` | 学习率调度 warmupCosine：线性 warmup 后余弦退火 | warmup 段线性，之后 lr(t) = lrMin + (η−lrMin)·½(1+cos(π·(t−W)/T)) |
| `D5-040` | `opt.lr.onecycle` | `(object cfg, number epoch) -> number` | 学习率调度 onecycle：OneCycle：先升后降的单周期策略 | 峰值 η，末端 η/finalDivFactor，warmup 段占 pctStart |
| `D5-041` | `opt.lr.polynomial` | `(object cfg, number epoch) -> number` | 学习率调度 polynomial：多项式衰减到 lrMin | lr(t) = lrMin + (η−lrMin)·(1−min(t,T)/T)^power |
| `D5-042` | `opt.lr.triangular` | `(object cfg, number epoch) -> number` | 学习率调度 triangular：三角波循环学习率（CLR） | lr(t) = lrMin + (maxLr−lrMin)·max(0, 1−x)，x = \|t/stepSize−2·cycle+1\| |
| `D5-043` | `opt.lr.linearDecay` | `(object cfg, number epoch) -> number` | 学习率调度 linearDecay：线性衰减到 lrMin | lr(t) = lrMin + (η−lrMin)·(1−min(t,T)/T) |
| `D5-044` | `opt.lr.inverse` | `(object cfg, number epoch) -> number` | 学习率调度 inverse：反向时间衰减：lr(t) = η/(1+γt) | lr(t) = η/(1+γ·t) |
| `D5-045` | `opt.grad.clipNorm` | `(Tensor grads, number maxNorm) -> Tensor` | 梯度全局范数裁剪：‖g‖₂ 超过 maxNorm 时按比例缩放 | 输出范数 = min(‖g‖₂, maxNorm) |
| `D5-046` | `opt.grad.clipValue` | `(Tensor grads, number clipValue) -> Tensor` | 梯度逐元素裁剪到 [−clipValue, clipValue] | 输出逐元素满足 \|g_i\| <= clipValue |
| `D5-047` | `opt.grad.weightDecay` | `(Tensor grads, Tensor params, number wd) -> Tensor` | 耦合式 L2 权重衰减：g ← g + λ·θ | 输出 = g + λ·θ |
| `D5-048` | `opt.grad.centralization` | `(Tensor grads) -> Tensor` | 梯度中心化：减去最后一轴均值，约束更新为零均值方向 | 输出沿最后一轴求和为 0 |
| `D5-049` | `opt.lookahead` | `(Tensor fastParams, object state, object cfg) -> {params: Tensor, state: object}` | Lookahead 包装器：每 k 步执行一次慢权重同步 slow ← slow + α·(fast − slow) | 同步发生时 slow ← slow + α·(fast − slow) 且 fast ← slow；state.t 每调用 +1 |
| `D5-050` | `opt.ema` | `(Tensor params, object state, object cfg) -> {params: Tensor, state: object}` | 参数指数滑动平均：ema ← decay·ema + (1−decay)·θ | ema_t = decay·ema_{t−1} + (1−decay)·θ_t，返回 ema |
| `D5-051` | `opt.grad.scaleByBatch` | `(Tensor grads, number batchSize, number baseBatch) -> Tensor` | 梯度按 batch 大小线性缩放，用于累积/归一化损失切换 | 输出 = g·batchSize/baseBatch |
| `D5-052` | `opt.grad.accumulateReset` | `(Tensor refGrads) -> object` | 重置梯度累加状态：{sum: 零数组, count: 0} | 返回 {sum: Array(k).fill(0), count: 0}，k = refGrads.size |
| `D5-053` | `opt.grad.accumulateAdd` | `(object state, Tensor grads) -> object` | 向累加状态追加一个 batch 的梯度：sum ← sum + g，count ← count + 1 | 新 state = {sum: sum + g, count: count + 1} |
| `D5-054` | `opt.line.backtracking` | `(object fn, Tensor x, Tensor grad, object cfg) -> {alpha, xNew, fNew, iters}` | Armijo 回溯线搜索：沿 −g 方向找到满足充分下降的步长 | fNew <= f0 + c1·α·(g·d)，d = −g；α 每次失败乘以 rho |
| `D5-055` | `opt.line.wolfeLite` | `(object fn, Tensor x, Tensor grad, object cfg) -> {alpha, xNew, fNew, gNew, ok, iters}` | 轻量 Wolfe 线搜索：回溯找同时满足 Armijo 与曲率条件的步长 | fNew <= f0 + c1·α·(g·d) 且 gNew·d >= c2·(g·d)，d = −g |
| `D5-056` | `opt.line.fixedStep` | `(object fn, Tensor x, Tensor grad, object cfg) -> {alpha, xNew, fNew, gNew}` | 固定步长线搜索：沿 −g 走 cfg.alpha | xNew = x − α·g，α = cfg.alpha |
| `D5-057` | `opt.line.exactQuadratic` | `(object fn, Tensor x, Tensor grad, object cfg) -> {alpha, xNew, fNew}` | 二次函数精确线搜索：α* = (gᵀg)/(gᵀHg)，沿 −g 精确最小化 | f(x − α*g) = f(x) − (gᵀg)²/(2·gᵀHg) |
| `D5-058` | `opt.adamCosine` | `(Tensor params, Tensor grads, object\|null state, object cfg) -> {params: Tensor, state: object}` | 组合指令：adam + lr.cosine（state 传 null 初始化并执行第一步） | 每步 lr 由 lr.cosine(cfg, epoch) 给出，基础更新用 opt.adam 规则；state.epoch 每步 +1 |
| `D5-059` | `opt.adamwOnecycle` | `(Tensor params, Tensor grads, object\|null state, object cfg) -> {params: Tensor, state: object}` | 组合指令：adamw + lr.onecycle（state 传 null 初始化并执行第一步） | 每步 lr 由 lr.onecycle(cfg, epoch) 给出，基础更新用 opt.adamw 规则；state.epoch 每步 +1 |
| `D5-060` | `opt.lionWarmupCosine` | `(Tensor params, Tensor grads, object\|null state, object cfg) -> {params: Tensor, state: object}` | 组合指令：lion + lr.warmupCosine（state 传 null 初始化并执行第一步） | 每步 lr 由 lr.warmupCosine(cfg, epoch) 给出，基础更新用 opt.lion 规则；state.epoch 每步 +1 |
| `D5-061` | `opt.sgdNesterovStep` | `(Tensor params, Tensor grads, object\|null state, object cfg) -> {params: Tensor, state: object}` | 组合指令：sgdNesterov + lr.step（state 传 null 初始化并执行第一步） | 每步 lr 由 lr.step(cfg, epoch) 给出，基础更新用 opt.sgdNesterov 规则；state.epoch 每步 +1 |
| `D5-062` | `opt.rmspropMultistep` | `(Tensor params, Tensor grads, object\|null state, object cfg) -> {params: Tensor, state: object}` | 组合指令：rmsprop + lr.multistep（state 传 null 初始化并执行第一步） | 每步 lr 由 lr.multistep(cfg, epoch) 给出，基础更新用 opt.rmsprop 规则；state.epoch 每步 +1 |
| `D5-063` | `opt.adamaxWarmup` | `(Tensor params, Tensor grads, object\|null state, object cfg) -> {params: Tensor, state: object}` | 组合指令：adamax + lr.warmup（state 传 null 初始化并执行第一步） | 每步 lr 由 lr.warmup(cfg, epoch) 给出，基础更新用 opt.adamax 规则；state.epoch 每步 +1 |
| `D5-064` | `opt.lambCosine` | `(Tensor params, Tensor grads, object\|null state, object cfg) -> {params: Tensor, state: object}` | 组合指令：lamb + lr.cosine（state 传 null 初始化并执行第一步） | 每步 lr 由 lr.cosine(cfg, epoch) 给出，基础更新用 opt.lamb 规则；state.epoch 每步 +1 |
| `D5-065` | `opt.adagradExp` | `(Tensor params, Tensor grads, object\|null state, object cfg) -> {params: Tensor, state: object}` | 组合指令：adagrad + lr.exp（state 传 null 初始化并执行第一步） | 每步 lr 由 lr.exp(cfg, epoch) 给出，基础更新用 opt.adagrad 规则；state.epoch 每步 +1 |
| `D5-066` | `opt.rpropTriangular` | `(Tensor params, Tensor grads, object\|null state, object cfg) -> {params: Tensor, state: object}` | 组合指令：rprop + lr.triangular（state 传 null 初始化并执行第一步） | 每步 lr 由 lr.triangular(cfg, epoch) 给出，基础更新用 opt.rprop 规则；state.epoch 每步 +1 |
| `D5-067` | `opt.signSgdConstant` | `(Tensor params, Tensor grads, object\|null state, object cfg) -> {params: Tensor, state: object}` | 组合指令：signSgd + lr.constant（state 传 null 初始化并执行第一步） | 每步 lr 由 lr.constant(cfg, epoch) 给出，基础更新用 opt.signSgd 规则；state.epoch 每步 +1 |
| `D5-068` | `opt.momentumLinear` | `(Tensor params, Tensor grads, object\|null state, object cfg) -> {params: Tensor, state: object}` | 组合指令：sgdMomentum + lr.linearDecay（state 传 null 初始化并执行第一步） | 每步 lr 由 lr.linearDecay(cfg, epoch) 给出，基础更新用 opt.sgdMomentum 规则；state.epoch 每步 +1 |
| `D5-069` | `opt.nadamOnecycle` | `(Tensor params, Tensor grads, object\|null state, object cfg) -> {params: Tensor, state: object}` | 组合指令：nadam + lr.onecycle（state 传 null 初始化并执行第一步） | 每步 lr 由 lr.onecycle(cfg, epoch) 给出，基础更新用 opt.nadam 规则；state.epoch 每步 +1 |
| `D5-070` | `opt.util.quadraticFn` | `(object cfg) -> object` | 构造凸二次基准 f(x)=½Σ dᵢ(xᵢ−x*ᵢ)² 的 loss+grad（默认 dᵢ=i, k=10, x*=0） | g = d⊙(x−x*)，Hessian 为 diag(d) |
| `D5-071` | `opt.util.sphereFn` | `(object cfg) -> object` | 构造球函数基准 f(x)=½‖x−x*‖² 的 loss+grad | g = x − x*，Hessian 为单位矩阵 |
| `D5-072` | `opt.util.rosenbrock` | `(object cfg) -> object` | 构造 Rosenbrock 非凸基准的 loss+grad（经典 2D：f=(1−x)²+100(y−x²)²） | f(x*) = 0 且 x* = (1,1,...,1) |
| `D5-073` | `opt.util.beale` | `(object cfg) -> object` | 构造 Beale 函数基准 loss+grad（2D，x*=(3, 0.5)） | f(3, 0.5) = 0 |
| `D5-074` | `opt.util.booth` | `(object cfg) -> object` | 构造 Booth 函数基准 loss+grad（2D 凸，x*=(1,3)） | f(x,y) = (x+2y−7)² + (2x+y−5)²，f(1,3) = 0 |
| `D5-075` | `opt.util.himmelblau` | `(object cfg) -> object` | 构造 Himmelblau 函数基准 loss+grad（2D 非凸，最小点 (3,2) 等，f=0） | f(x,y) = (x²+y−11)² + (x+y²−7)²，f(3,2) = 0 |
| `D5-076` | `opt.util.mccormick` | `(object cfg) -> object` | 构造 McCormick 函数基准 loss+grad（2D 非凸，f*≈−1.9132） | f(x,y) = sin(x+y) + (x−y)² − 1.5x + 2.5y + 1 |
| `D5-077` | `opt.util.runOptimizer` | `(string name, object fn, Tensor params, number iters, object cfg) -> object` | 统一驱动：运行指定优化器（经典或组合预设），返回 loss 轨迹与最终参数 | traj[0] = f(x0)，traj[i+1] = f(x_{i+1})；最终 params 为最后一步结果 |
| `D5-078` | `opt.util.lossTrajectory` | `(string name, object fn, Tensor params, number iters, object cfg) -> number[]` | 运行优化器并仅返回 loss 轨迹数组（长度 iters+1） | 返回数组长度 = iters + 1，首元素为初始 loss |
| `D5-079` | `opt.util.converged` | `(Tensor x, Tensor xStar, number tol) -> boolean` | 收敛判定：参数与最优点的最大绝对偏差 < tol | 返回 max_i \|x_i − x*_i\| < tol |
| `D5-080` | `opt.util.gradientNorm` | `(Tensor grads) -> number` | 计算梯度 L2 范数 ‖g‖₂ | 返回 sqrt(Σ g_i²) |
| `D5-081` | `opt.util.paramDistance` | `(Tensor x, Tensor xStar) -> number` | 计算参数与最优点的 L2 距离 ‖x − x*‖₂ | 返回 sqrt(Σ (x_i − x*_i)²) |
| `D5-082` | `opt.util.compareOptimizers` | `(string[] names, object fn, Tensor params, number iters, object cfg) -> object[]` | 多个优化器在同一基准上运行，返回对比表（含最终 loss/误差/单调性） | 每行 = {name, finalLoss, finalErr, monotone, iters}，按输入顺序返回 |
| `D5-083` | `opt.util.bestOf` | `(object[] rows) -> object` | 从 compareOptimizers 结果中取最终 loss 最小的一行 | 返回 rows 中 finalLoss 最小者（并列取第一个） |
| `D5-084` | `opt.util.scheduleAt` | `(string schedName, object cfg, number epoch) -> number` | 按调度器名字（constant/step/... 之一）计算 epoch 处学习率 | 等价于调用对应的 opt.lr.<schedName> |
| `D5-085` | `opt.util.stateToJSON` | `(object state) -> string` | 把优化器 state 序列化为 JSON 字符串 | JSON.parse(stateToJSON(s)) 与 s 深度相等 |
| `D5-086` | `opt.util.stateFromJSON` | `(string json) -> object` | 从 JSON 字符串还原优化器 state | stateFromJSON(stateToJSON(s)) 与 s 深度相等 |
| `D5-087` | `opt.util.zeroState` | `(string name, number k) -> object` | 按优化器名创建长度为 k 的零初始化 state | 等价于 opt.<name>(zeros([k]), {}) |
| `D5-088` | `opt.util.assertConverged` | `(Tensor x, Tensor xStar, number tol) -> boolean` | 断言收敛：‖x − x*‖∞ >= tol 时抛 E_NUMERICAL，否则返回 true | 不抛错时必有 max_i \|x_i − x*_i\| < tol |
| `D5-089` | `opt.util.monotoneLoss` | `(number[] traj, number tol) -> boolean` | 检查 loss 轨迹单调不增（允许 tol·max(1,\|loss\|) 量级抖动） | 所有相邻项满足 traj[i] <= traj[i−1] + tol·max(1, \|traj[i−1]\|) |
| `D5-090` | `opt.util.describeOptimizer` | `(string name) -> string` | 返回优化器的注册表描述（id/name/doc/ref/inv） | 输出包含该优化器的 doc 与 inv |
| `D5-091` | `opt.util.listOptimizers` | `() -> string[]` | 列出全部 16 个经典优化器名 | 返回数组长度 = 16 且元素唯一 |
| `D5-092` | `opt.util.optimizerCount` | `() -> number` | 返回经典优化器数量（16） | 恒等于 listOptimizers().length |
| `D5-093` | `opt.util.defaultCfg` | `(string name) -> object` | 返回指定优化器的默认配置对象 | 返回 OPTSPECS[name].defaults 的浅拷贝 |
| `D5-094` | `opt.util.validateCfg` | `(string name, object cfg) -> boolean` | 校验优化器配置合法性（lr>0、beta∈[0,1)、eps>=0、全有限） | 合法返回 true，否则 false（不抛错） |
| `D5-095` | `opt.util.etaScale` | `(object cfg) -> number` | Adam 偏差修正后的等效步长尺度：lr·√(1−β2^t)/(1−β1^t) | etaScale = lr·√(1−β2^t)/(1−β1^t) |
| `D5-096` | `opt.util.biasCorrection` | `(number beta, number t) -> number` | 偏差修正因子 1 − β^t（Adam 系列 m̂/v̂ 的分母） | 返回 1 − beta^t，t >= 1 |
| `D5-097` | `opt.util.trustRatio` | `(Tensor phi, Tensor w, number wd) -> number` | LAMB 信任比：‖φ‖₂/‖φ + wd·w‖₂（任一为零范数时取 1） | 返回 ‖φ‖/‖φ+wd·w‖；‖φ+wd·w‖=0 或 ‖φ‖=0 时为 1 |
| `D5-098` | `opt.util.signUpdate` | `(Tensor grads, number lr) -> Tensor` | 符号更新量 −η·sign(g)（Lion/SignSGD 的核心更新） | 输出逐元素 = −η·sign(g_i) |
| `D5-099` | `opt.util.momentumBlend` | `(Tensor grads, Tensor v, number beta) -> Tensor` | 动量混合 β·v + (1−β)·g（EMA 形式的一阶矩更新） | 输出 = β·v + (1−β)·g |
| `D5-100` | `opt.util.warmupFactor` | `(number epoch, number warmupEpochs, number start) -> number` | warmup 线性因子：epoch < W 时 start+(1−start)·epoch/W，否则 1 | 因子 ∈ [start, 1] 且单调不减 |
| `D5-101` | `opt.util.cosineFactor` | `(number epoch, number T) -> number` | 余弦退火因子 ½(1+cos(π·min(epoch,T)/T))：从 1 降到 0 | factor(0) = 1，factor(T/2) = 0.5，factor(T) = 0 |
| `D5-102` | `opt.util.onecycleFactor` | `(number epoch, number T, number pctStart) -> number` | OneCycle 归一化位置因子：warmup 段线性 0→1，随后余弦 1→0 | factor(0)=0，峰值 1 出现在 pctStart·T 处，factor(T)=0 |

## D6 · PROB 概率统计

分布族、信息论、矩估计、贝叶斯推断、采样与拟合检验。共 102 条。

| id | 指令 | 签名 | 说明 | 不变量 |
|---|---|---|---|---|
| `D6-001` | `prob.normal.pdf` | `(x: number\|Tensor, mu: number, sigma: number) -> number\|Tensor` | 正态密度：pdf(x) = exp(−(x−μ)²/(2σ²)) / (σ√(2π)) | pdf ≥ 0，∫pdf = 1，关于 μ 对称；σ > 0 |
| `D6-002` | `prob.normal.cdf` | `(x: number\|Tensor, mu: number, sigma: number) -> number\|Tensor` | 正态 CDF：Φ((x−μ)/σ)，其中 Φ 为标准正态 CDF（经 erf 精确计算） | cdf ∈ [0,1] 单调不减；cdf(−∞)=0，cdf(+∞)=1 |
| `D6-003` | `prob.normal.sample` | `(seed: number, n: number, mu: number, sigma: number) -> Tensor[n]` | 正态抽样：μ + σ·Z，Z ~ N(0,1) 由 Box-Muller 生成（种子化可复现） | 同种子完全可复现；大样本矩 → (μ, σ²) |
| `D6-004` | `prob.normal.fit` | `(data: Tensor) -> Tensor[2]` | 正态 MLE 拟合：返回 [μ̂, σ̂]，μ̂ = x̄，σ̂ = √(Σ(x−x̄)²/n) | n ≥ 1；对正态数据 μ̂→μ、σ̂→σ |
| `D6-005` | `prob.uniform.pdf` | `(x: number\|Tensor, a: number, b: number) -> number\|Tensor` | 均匀密度：pdf(x) = 1/(b−a)（a ≤ x ≤ b），否则 0 | pdf ≥ 0，∫pdf = 1；b > a |
| `D6-006` | `prob.uniform.cdf` | `(x: number\|Tensor, a: number, b: number) -> number\|Tensor` | 均匀 CDF：clip((x−a)/(b−a), 0, 1) | cdf ∈ [0,1] 单调不减；b > a |
| `D6-007` | `prob.uniform.sample` | `(seed: number, n: number, a: number, b: number) -> Tensor[n]` | 均匀抽样：a + (b−a)·U，U ~ U(0,1) 来自 PCG 流 | 同种子可复现；大样本矩 → ((a+b)/2, (b−a)²/12) |
| `D6-008` | `prob.uniform.fit` | `(data: Tensor) -> Tensor[2]` | 均匀 MLE 拟合：返回 [min, max] | 返回 â ≤ x_i ≤ b̂ |
| `D6-009` | `prob.exponential.pdf` | `(x: number\|Tensor, rate: number) -> number\|Tensor` | 指数密度：pdf(x) = λ·e^{−λx}（x ≥ 0），否则 0 | pdf ≥ 0，∫₀^∞ pdf = 1；rate > 0 |
| `D6-010` | `prob.exponential.cdf` | `(x: number\|Tensor, rate: number) -> number\|Tensor` | 指数 CDF：1 − e^{−λx}（x ≥ 0），否则 0 | cdf ∈ [0,1] 单调不减；rate > 0 |
| `D6-011` | `prob.exponential.sample` | `(seed: number, n: number, rate: number) -> Tensor[n]` | 指数抽样（逆变换）：x = −ln(U)/λ，U ~ U(0,1) | 同种子可复现；大样本矩 → (1/λ, 1/λ²) |
| `D6-012` | `prob.exponential.fit` | `(data: Tensor) -> Tensor[1]` | 指数 MLE 拟合：返回 [λ̂]，λ̂ = 1/x̄ | 数据须 ≥ 0 且均值 > 0 |
| `D6-013` | `prob.laplace.pdf` | `(x: number\|Tensor, mu: number, b: number) -> number\|Tensor` | 拉普拉斯密度：pdf(x) = exp(−\|x−μ\|/b) / (2b) | pdf ≥ 0，∫pdf = 1，关于 μ 对称；b > 0 |
| `D6-014` | `prob.laplace.cdf` | `(x: number\|Tensor, mu: number, b: number) -> number\|Tensor` | 拉普拉斯 CDF：x<μ 时 ½·exp((x−μ)/b)，x≥μ 时 1 − ½·exp(−(x−μ)/b) | cdf ∈ [0,1] 单调不减；b > 0 |
| `D6-015` | `prob.laplace.sample` | `(seed: number, n: number, mu: number, b: number) -> Tensor[n]` | 拉普拉斯抽样（逆变换）：x = μ − b·sign(U−½)·ln(1−2\|U−½\|) | 同种子可复现；大样本矩 → (μ, 2b²) |
| `D6-016` | `prob.cauchy.pdf` | `(x: number\|Tensor, x0: number, gamma: number) -> number\|Tensor` | 柯西密度：pdf(x) = 1/(πγ(1 + ((x−x0)/γ)²)) | pdf ≥ 0，∫pdf = 1；γ > 0；均值/方差不存在 |
| `D6-017` | `prob.cauchy.cdf` | `(x: number\|Tensor, x0: number, gamma: number) -> number\|Tensor` | 柯西 CDF：½ + arctan((x−x0)/γ)/π | cdf ∈ [0,1] 单调不减；γ > 0 |
| `D6-018` | `prob.cauchy.sample` | `(seed: number, n: number, x0: number, gamma: number) -> Tensor[n]` | 柯西抽样（逆变换）：x = x0 + γ·tan(π(U−½)) | 同种子可复现；中位数 = x0 |
| `D6-019` | `prob.logistic.pdf` | `(x: number\|Tensor, mu: number, s: number) -> number\|Tensor` | 逻辑斯蒂密度：pdf(x) = e^{−z}/(s(1+e^{−z})²)，z = (x−μ)/s | pdf ≥ 0，∫pdf = 1，关于 μ 对称；s > 0 |
| `D6-020` | `prob.logistic.cdf` | `(x: number\|Tensor, mu: number, s: number) -> number\|Tensor` | 逻辑斯蒂 CDF：1/(1+e^{−z})，z = (x−μ)/s | cdf ∈ [0,1] 单调不减；s > 0 |
| `D6-021` | `prob.logistic.sample` | `(seed: number, n: number, mu: number, s: number) -> Tensor[n]` | 逻辑斯蒂抽样（逆变换）：x = μ + s·ln(U/(1−U)) | 同种子可复现；大样本矩 → (μ, π²s²/3) |
| `D6-022` | `prob.gamma.pdf` | `(x: number\|Tensor, shape: number, rate: number) -> number\|Tensor` | 伽马密度：pdf(x) = λ^k x^{k−1} e^{−λx} / Γ(k)（x ≥ 0），否则 0 | pdf ≥ 0，∫₀^∞ pdf = 1；shape > 0，rate > 0 |
| `D6-023` | `prob.gamma.cdf` | `(x: number\|Tensor, shape: number, rate: number) -> number\|Tensor` | 伽马 CDF：P(k, λx)，正则化下不完全伽马函数 | cdf ∈ [0,1] 单调不减；shape > 0，rate > 0 |
| `D6-024` | `prob.gamma.sample` | `(seed: number, n: number, shape: number, rate: number) -> Tensor[n]` | 伽马抽样：Marsaglia-Tsang 拒绝法（shape ≥ 1），shape < 1 用 boost G(α)=G(α+1)·U^(1/α) | 同种子可复现；大样本矩 → (k/λ, k/λ²) |
| `D6-025` | `prob.gamma.fit` | `(data: Tensor) -> Tensor[2]` | 伽马 MLE 拟合：解似然方程 ln k − ψ(k) = s（s = ln(x̄) − mean(ln x) ≥ 0），返回 [k̂, λ̂]，λ̂ = k̂/x̄ | 数据须严格为正；f(k)=ln k − ψ(k) − s 严格递减故解唯一 |
| `D6-026` | `prob.beta.pdf` | `(x: number\|Tensor, a: number, b: number) -> number\|Tensor` | 贝塔密度：pdf(x) = x^{a−1}(1−x)^{b−1} / B(a,b)（0 ≤ x ≤ 1），否则 0 | pdf ≥ 0，∫₀¹ pdf = 1；a > 0，b > 0 |
| `D6-027` | `prob.beta.cdf` | `(x: number\|Tensor, a: number, b: number) -> number\|Tensor` | 贝塔 CDF：I_x(a,b)，正则化不完全贝塔函数 | cdf ∈ [0,1] 单调不减；a > 0，b > 0 |
| `D6-028` | `prob.beta.sample` | `(seed: number, n: number, a: number, b: number) -> Tensor[n]` | 贝塔抽样：X = G1/(G1+G2)，G1~Gamma(a,1)、G2~Gamma(b,1) 独立 | 同种子可复现；大样本矩 → (a/(a+b), ab/((a+b)²(a+b+1))) |
| `D6-029` | `prob.chi2.pdf` | `(x: number\|Tensor, df: number) -> number\|Tensor` | 卡方密度：pdf(x) = x^{ν/2−1} e^{−x/2} / (2^{ν/2} Γ(ν/2))（x ≥ 0），否则 0 | pdf ≥ 0，∫₀^∞ pdf = 1；df > 0 |
| `D6-030` | `prob.chi2.cdf` | `(x: number\|Tensor, df: number) -> number\|Tensor` | 卡方 CDF：P(ν/2, x/2)，正则化下不完全伽马函数 | cdf ∈ [0,1] 单调不减；df > 0 |
| `D6-031` | `prob.chi2.sample` | `(seed: number, n: number, df: number) -> Tensor[n]` | 卡方抽样：Gamma(df/2, rate=1/2)（即 df 个标准正态平方和的一般化） | 同种子可复现；大样本矩 → (ν, 2ν) |
| `D6-032` | `prob.studentT.pdf` | `(x: number\|Tensor, df: number) -> number\|Tensor` | t 分布密度：pdf(x) = Γ((ν+1)/2)/(√(νπ)Γ(ν/2)) · (1 + x²/ν)^{−(ν+1)/2} | pdf ≥ 0，∫pdf = 1，关于 0 对称；df > 0；ν>1 时均值 0，ν>2 时方差 ν/(ν−2) |
| `D6-033` | `prob.studentT.cdf` | `(x: number\|Tensor, df: number) -> number\|Tensor` | t 分布 CDF：x>0 时 1 − ½·I_{ν/(ν+x²)}(ν/2, 1/2)，x≤0 时 ½·I_{ν/(ν+x²)}(ν/2, 1/2) | cdf ∈ [0,1] 单调不减，cdf(−x) = 1 − cdf(x)；df > 0 |
| `D6-034` | `prob.studentT.sample` | `(seed: number, n: number, df: number) -> Tensor[n]` | t 分布抽样：T = Z/√(V/ν)，Z~N(0,1)、V~χ²(ν) 独立 | 同种子可复现；ν>2 时大样本方差 → ν/(ν−2) |
| `D6-035` | `prob.rayleigh.pdf` | `(x: number\|Tensor, sigma: number) -> number\|Tensor` | 瑞利密度：pdf(x) = (x/σ²)·e^{−x²/(2σ²)}（x ≥ 0），否则 0 | pdf ≥ 0，∫₀^∞ pdf = 1；σ > 0 |
| `D6-036` | `prob.rayleigh.cdf` | `(x: number\|Tensor, sigma: number) -> number\|Tensor` | 瑞利 CDF：1 − e^{−x²/(2σ²)}（x ≥ 0），否则 0 | cdf ∈ [0,1] 单调不减；σ > 0 |
| `D6-037` | `prob.rayleigh.sample` | `(seed: number, n: number, sigma: number) -> Tensor[n]` | 瑞利抽样（逆变换）：x = σ·√(−2 ln U) | 同种子可复现；大样本矩 → (σ√(π/2), (4−π)σ²/2) |
| `D6-038` | `prob.weibull.pdf` | `(x: number\|Tensor, shape: number, scale: number) -> number\|Tensor` | 威布尔密度：pdf(x) = (k/λ)(x/λ)^{k−1} e^{−(x/λ)^k}（x ≥ 0），否则 0 | pdf ≥ 0，∫₀^∞ pdf = 1；k > 0，λ > 0 |
| `D6-039` | `prob.weibull.cdf` | `(x: number\|Tensor, shape: number, scale: number) -> number\|Tensor` | 威布尔 CDF：1 − e^{−(x/λ)^k}（x ≥ 0），否则 0 | cdf ∈ [0,1] 单调不减；k > 0，λ > 0 |
| `D6-040` | `prob.weibull.sample` | `(seed: number, n: number, shape: number, scale: number) -> Tensor[n]` | 威布尔抽样（逆变换）：x = λ·(−ln U)^{1/k} | 同种子可复现；大样本矩 → (λΓ(1+1/k), λ²(Γ(1+2/k)−Γ(1+1/k)²)) |
| `D6-041` | `prob.pareto.pdf` | `(x: number\|Tensor, xm: number, alpha: number) -> number\|Tensor` | 帕累托密度：pdf(x) = α·xm^α / x^{α+1}（x ≥ xm），否则 0 | pdf ≥ 0，∫_{xm}^∞ pdf = 1；α > 0；α>1 时均值 αxm/(α−1) |
| `D6-042` | `prob.pareto.cdf` | `(x: number\|Tensor, xm: number, alpha: number) -> number\|Tensor` | 帕累托 CDF：1 − (xm/x)^α（x ≥ xm），否则 0 | cdf ∈ [0,1] 单调不减；α > 0 |
| `D6-043` | `prob.pareto.sample` | `(seed: number, n: number, xm: number, alpha: number) -> Tensor[n]` | 帕累托抽样（逆变换）：x = xm·U^{−1/α} | 同种子可复现；α>1 时大样本均值 → αxm/(α−1) |
| `D6-044` | `prob.bernoulli.pmf` | `(k: number\|Tensor, p: number) -> number\|Tensor` | 伯努利 pmf：P(k) = p^k (1−p)^{1−k}（k ∈ {0,1}），非整数 k 为 0 | Σpmf = 1；0 ≤ p ≤ 1 |
| `D6-045` | `prob.bernoulli.cdf` | `(k: number\|Tensor, p: number) -> number\|Tensor` | 伯努利 CDF：k<0 为 0，0≤k<1 为 1−p，k≥1 为 1 | cdf ∈ [0,1] 单调不减；0 ≤ p ≤ 1 |
| `D6-046` | `prob.bernoulli.sample` | `(seed: number, n: number, p: number) -> Tensor[n]` | 伯努利抽样：U < p 取 1 否则 0 | 同种子可复现；大样本矩 → (p, p(1−p)) |
| `D6-047` | `prob.binomial.pmf` | `(k: number\|Tensor, nTrials: number, p: number) -> number\|Tensor` | 二项 pmf：C(N,k) p^k (1−p)^{N−k}（0 ≤ k ≤ N 整数），否则 0；组合数经 lgamma 计算 | Σₖ pmf = 1；N 为非负整数，0 ≤ p ≤ 1 |
| `D6-048` | `prob.binomial.cdf` | `(k: number\|Tensor, nTrials: number, p: number) -> number\|Tensor` | 二项 CDF：Σ_{i≤⌊k⌋} pmf(i)，用正则化不完全贝塔 I_{1−p}(N−k, k+1) 闭式计算 | cdf ∈ [0,1] 单调不减；N 为非负整数，0 ≤ p ≤ 1 |
| `D6-049` | `prob.binomial.sample` | `(seed: number, n: number, nTrials: number, p: number) -> Tensor[n]` | 二项抽样：N 次独立伯努利试验之和 | 同种子可复现；大样本矩 → (Np, Np(1−p)) |
| `D6-050` | `prob.poisson.pmf` | `(k: number\|Tensor, lambda: number) -> number\|Tensor` | 泊松 pmf：e^{−λ} λ^k / k!（k 为非负整数），否则 0 | Σₖ pmf = 1；λ > 0 |
| `D6-051` | `prob.poisson.cdf` | `(k: number\|Tensor, lambda: number) -> number\|Tensor` | 泊松 CDF：Σ_{i≤⌊k⌋} pmf(i)，用恒等式 Q(⌊k⌋+1, λ)（正则化上不完全伽马）计算 | cdf ∈ [0,1] 单调不减；λ > 0 |
| `D6-052` | `prob.poisson.sample` | `(seed: number, n: number, lambda: number) -> Tensor[n]` | 泊松抽样：λ < 30 用 Knuth 乘法法；λ ≥ 30 用正态近似 round(λ + √λ·Z) | 同种子可复现；大样本矩 → (λ, λ) |
| `D6-053` | `prob.geometric.pmf` | `(k: number\|Tensor, p: number) -> number\|Tensor` | 几何 pmf：(1−p)^{k−1} p（k 为正整数），否则 0 | Σₖ pmf = 1；0 < p ≤ 1 |
| `D6-054` | `prob.geometric.cdf` | `(k: number\|Tensor, p: number) -> number\|Tensor` | 几何 CDF：k<1 为 0，否则 1 − (1−p)^{⌊k⌋} | cdf ∈ [0,1] 单调不减；0 < p ≤ 1 |
| `D6-055` | `prob.geometric.sample` | `(seed: number, n: number, p: number) -> Tensor[n]` | 几何抽样（逆变换）：k = ⌈ln U / ln(1−p)⌉ | 同种子可复现；大样本矩 → (1/p, (1−p)/p²) |
| `D6-056` | `prob.categorical.pmf` | `(k: number\|Tensor, probs: Tensor) -> number\|Tensor` | 类别 pmf：P(k) = probs[k]/Σprobs（k ∈ [0, K−1] 整数），否则 0（自动归一化） | Σₖ pmf = 1；probs 非负且和 > 0 |
| `D6-057` | `prob.categorical.cdf` | `(k: number\|Tensor, probs: Tensor) -> number\|Tensor` | 类别 CDF：Σ_{i≤⌊k⌋} pmf(i)（自动归一化） | cdf ∈ [0,1] 单调不减；probs 非负且和 > 0 |
| `D6-058` | `prob.categorical.sample` | `(seed: number, n: number, probs: Tensor) -> Tensor[n]` | 类别抽样（逆变换）：U·Σprobs 落在第几个累积区间即返回该下标 | 同种子可复现；频率 → probs/Σprobs |
| `D6-059` | `prob.mvn.pdf` | `(x: Tensor[d]\|Tensor[n,d], mu: Tensor[d], Sigma: Tensor[d,d]) -> number\|Tensor[n]` | 多元正态密度：经 Cholesky 解 Σ⁻¹ 二次型，log\|Σ\| = 2Σln L_ii | Σ 对称正定；pdf(mu) = (2π)^{−d/2}\|Σ\|^{−1/2} |
| `D6-060` | `prob.mvn.sample` | `(seed: number, n: number, mu: Tensor[d], Sigma: Tensor[d,d]) -> Tensor[n,d]` | 多元正态抽样：x = μ + L·z，Σ = L·Lᵀ（Cholesky），z ~ N(0, I_d) | 同种子可复现；大样本协方差 → Σ |
| `D6-061` | `prob.info.entropy` | `(p: Tensor) -> number` | 香农熵（nat）：H(p) = −Σ p_i ln p_i（0·ln0 = 0；自动归一化） | H ≥ 0；H ≤ ln\|X\|，均匀分布取等 |
| `D6-062` | `prob.info.crossEntropy` | `(p: Tensor, q: Tensor) -> number` | 交叉熵（nat）：H(p,q) = −Σ p_i ln q_i（p>0 且 q_i=0 时返回 +∞） | H(p,q) ≥ H(p)，等号当且仅当 p = q |
| `D6-063` | `prob.info.klDiv` | `(p: Tensor, q: Tensor) -> number` | KL 散度（nat）：D(p‖q) = Σ p_i ln(p_i/q_i)；p>0 且 q_i=0 时返回 +∞ | D(p‖p) = 0；D ≥ 0（Gibbs 不等式）；不对称 |
| `D6-064` | `prob.info.jsDiv` | `(p: Tensor, q: Tensor) -> number` | JS 散度（nat）：½D(p‖m) + ½D(q‖m)，m = (p+q)/2 | 0 ≤ JS ≤ ln 2；JS(p,p) = 0；对称 |
| `D6-065` | `prob.info.mutualInfo` | `(joint: Tensor[m,n]) -> number` | 互信息（nat）：I(X;Y) = Σ_ij P_ij ln(P_ij / (P_i·P_j))（自动归一化） | I ≥ 0；独立时为 0；I ≤ min(H(X), H(Y)) |
| `D6-066` | `prob.info.wasserstein1` | `(x: Tensor, y: Tensor) -> number \| (mu1: number, sigma1: number, mu2: number, sigma2: number) -> number` | 一阶 Wasserstein 距离：两样本（等长）取排序后平均绝对差；四个数值参数时按正态闭式 W1 = \|μ1−μ2\| | W1 ≥ 0；W1(x,x) = 0；满足三角不等式 |
| `D6-067` | `prob.info.hellinger` | `(p: Tensor, q: Tensor) -> number` | Hellinger 距离：H(p,q) = √(½ Σ (√p_i − √q_i)²) | 0 ≤ H ≤ 1；H(p,p) = 0；H²(p,q) ≤ TVD(p,q) |
| `D6-068` | `prob.info.tvd` | `(p: Tensor, q: Tensor) -> number` | 全变差距离：TVD(p,q) = ½ Σ \|p_i − q_i\| | 0 ≤ TVD ≤ 1；TVD(p,p) = 0 |
| `D6-069` | `prob.stat.mean` | `(data: Tensor) -> number` | 样本均值：x̄ = (1/n) Σ x_i | min(data) ≤ mean ≤ max(data) |
| `D6-070` | `prob.stat.variance` | `(data: Tensor) -> number` | 样本方差（总体口径）：m2 = (1/n) Σ (x_i − x̄)² | variance ≥ 0；常数序列为 0 |
| `D6-071` | `prob.stat.std` | `(data: Tensor) -> number` | 样本标准差（总体口径）：√m2 | std ≥ 0；std² = variance |
| `D6-072` | `prob.stat.skewness` | `(data: Tensor) -> number` | 样本偏度：m3 / m2^{3/2}（总体中心矩口径） | 对称分布 ≈ 0；右偏 > 0，左偏 < 0 |
| `D6-073` | `prob.stat.kurtosis` | `(data: Tensor) -> number` | 样本超额峰度：m4 / m2² − 3（正态为 0，均匀为 −1.2） | 重尾 > 0；均匀分布 ≈ −1.2 |
| `D6-074` | `prob.stat.cov` | `(x: Tensor, y: Tensor) -> number` | 样本协方差（总体口径）：(1/n) Σ (x_i − x̄)(y_i − ȳ) | cov(x,x) = variance(x)；两序列须等长 |
| `D6-075` | `prob.stat.corr` | `(x: Tensor, y: Tensor) -> number` | Pearson 相关系数：cov(x,y) / (σ_x·σ_y) | −1 ≤ corr ≤ 1；corr(x,x) = 1；任一序列为常数时抛 E_NUMERICAL |
| `D6-076` | `prob.stat.quantile` | `(data: Tensor, q: number) -> number` | 分位数（线性插值，numpy "linear" 口径）：h = (n−1)q，x_⌊h⌋ + (h−⌊h⌋)(x_⌊h⌋+1 − x_⌊h⌋) | 0 ≤ q ≤ 1；min ≤ quantile ≤ max；关于 q 单调不减 |
| `D6-077` | `prob.stat.median` | `(data: Tensor) -> number` | 中位数：quantile(data, 0.5) | min ≤ median ≤ max；对偶数个取中间两值均值 |
| `D6-078` | `prob.stat.mad` | `(data: Tensor) -> number` | 中位数绝对偏差：median(\|x_i − median(x)\|) | mad ≥ 0；对离群点稳健 |
| `D6-079` | `prob.bayes.update` | `(prior: Tensor, likelihood: Tensor) -> Tensor` | 贝叶斯更新（离散网格）：后验 ∝ 先验 × 似然，返回归一化后验 | Σ posterior = 1；两序列等长且非负 |
| `D6-080` | `prob.bayes.posteriorPredictive` | `(posterior: Tensor[k], likelihood: Tensor[k,m]) -> Tensor[m]` | 后验预测分布：P(x̃\|data) = Σ_j posterior_j · L(j, x̃)，likelihood 行对应参数、列对应观测 | 输出非负且（按列）和为 1 当 likelihood 每行归一化时 |
| `D6-081` | `prob.bayes.evidence` | `(prior: Tensor, likelihood: Tensor) -> number` | 证据（边际似然）：Z = Σ_i prior_i · likelihood_i，即后验归一化常数 | Z ≥ 0；先验已归一化时 Z ∈ [0,1] |
| `D6-082` | `prob.bayes.mapNormal` | `(data: Tensor, priorMu: number, priorSigma: number, sigma: number) -> number` | 正态均值 MAP（方差已知 σ²，先验 N(μ0,σ0²)）：μ̂ = (μ0/σ0² + n·x̄/σ²)/(1/σ0² + n/σ²) | σ0 > 0，σ > 0；后验为_normal_时 MAP = 后验均值 |
| `D6-083` | `prob.bayes.mleNormal` | `(data: Tensor) -> Tensor[2]` | 正态 MLE：返回 [μ̂, σ̂]，σ̂ 为 ÷n 口径的总体标准差 | σ̂ ≥ 0 |
| `D6-084` | `prob.bayes.mleBernoulli` | `(data: Tensor) -> number` | 伯努利 MLE：p̂ = x̄（数据须在 [0,1]） | 0 ≤ p̂ ≤ 1 |
| `D6-085` | `prob.bayes.mlePoisson` | `(data: Tensor) -> number` | 泊松 MLE：λ̂ = x̄（数据须 ≥ 0） | λ̂ ≥ 0 |
| `D6-086` | `prob.bayes.mleExponential` | `(data: Tensor) -> number` | 指数 MLE：λ̂ = 1/x̄（数据须 ≥ 0 且均值 > 0） | λ̂ > 0 |
| `D6-087` | `prob.bayes.credibleInterval` | `(mu: number, sigma: number, level: number = 0.95) -> Tensor[2]` | 正态后验可信区间：[μ ± z_{1−(1−level)/2}·σ]，z 由逆正态分位函数计算 | σ > 0，0 < level < 1；区间关于 μ 对称 |
| `D6-088` | `prob.bayes.conjugateNormal` | `(n: number, xbar: number, priorMu: number, priorSigma: number, dataSigma: number) -> Tensor[2]` | 正态-正态共轭后验（数据方差已知 σ²）：返回 [μ_n, σ_n]，σ_n² = 1/(1/σ0² + n/σ²)，μ_n = (μ0/σ0² + n·x̄/σ²)·σ_n² | n ≥ 0，σ0 > 0，σ > 0；与闭式解逐位一致 |
| `D6-089` | `prob.samp.inverseCdf` | `(seed: number, n: number, xs: Tensor, cdf: Tensor) -> Tensor[n]` | 通用逆变换采样：对分段线性 CDF（断点 xs/cdf 值）求逆并插值，xs 须严格递增 | xs 严格递增、cdf 单调不减；同种子可复现 |
| `D6-090` | `prob.samp.rejection` | `(seed: number, n: number) -> Tensor[n]` | 拒绝采样生成 N(0,1)：建议分布 Laplace(0,1)，包络常数 M = √(2e/π) ≈ 1.3155，接受率 ≈ 0.76 | 接受概率 φ(x)/(M·g(x)) ≤ 1；同种子可复现；矩 → (0,1) |
| `D6-091` | `prob.samp.metropolis` | `(seed: number, n: number, step: number) -> Tensor[n]` | 一维随机游走 Metropolis（目标 N(0,1)）：提议 y = x + step·N(0,1)，接受率 min(1, e^{−(y²−x²)/2})；前 10% 作 burn-in | step > 0；同种子可复现；矩 → (0,1)（容差 10%） |
| `D6-092` | `prob.samp.boxMuller` | `(seed: number, n: number) -> Tensor[n]` | Box-Muller 标准正态：u1,u2 ~ U(0,1)，r = √(−2 ln u1)，z1 = r·cos(2πu2)，z2 = r·sin(2πu2) | 同种子完全可复现；大样本矩 → (0,1) |
| `D6-093` | `prob.samp.marsagliaPolar` | `(seed: number, n: number) -> Tensor[n]` | Marsaglia 极坐标法标准正态：单位圆盘内取点 (u,v)，s = u²+v²，f = √(−2 ln s/s)，输出 (u·f, v·f) | 同种子完全可复现；大样本矩 → (0,1) |
| `D6-094` | `prob.samp.stratified` | `(seed: number, n: number) -> Tensor[n]` | 分层抽样 U(0,1)：第 i 层 [(i/n), (i+1)/n) 内取一个均匀点，(i+U)/n | 每层恰一个样本；均值估计量方差 = 1/(12n³)（粗蒙特卡洛 1/(12n) 的 1/n²） |
| `D6-095` | `prob.test.ks` | `(data: Tensor, dist: 'normal'\|'uniform'\|'exponential', ...params) -> {D, pValue, n}` | 一样本 Kolmogorov-Smirnov 检验：D = sup\|F_n − F\|，p 值用 Massey 1951 渐近式 Q(λ)，λ = (√n+0.12+0.11/√n)·D | 0 ≤ D ≤ 1；同分布假设下 p 值近似均匀 |
| `D6-096` | `prob.test.ks2Sample` | `(x: Tensor, y: Tensor) -> {D, pValue, n1, n2}` | 两样本 Kolmogorov-Smirnov 检验：D = sup\|F1 − F2\|，有效样本量 ne = n1·n2/(n1+n2)，p 值同 Massey 渐近式 | 0 ≤ D ≤ 1；同分布假设下 p 值近似均匀 |
| `D6-097` | `prob.test.chi2Gof` | `(observed: Tensor, expected: Tensor) -> {statistic, df, pValue}` | 卡方拟合优度检验：χ² = Σ (O_i − n·p_i)²/(n·p_i)，df = K−1，p = Q(df/2, χ²/2)（上不完全伽马） | χ² ≥ 0；p ∈ [0,1]；观测与期望概率等长 |
| `D6-098` | `prob.test.jarqueBera` | `(data: Tensor) -> {statistic, pValue, skewness, kurtosis}` | Jarque-Bera 正态性检验：JB = (n/6)(S² + K²/4)，S 为偏度、K 为超额峰度；p = e^{−JB/2}（χ²₂ 生存函数） | JB ≥ 0；正态样本 JB → χ²₂；p ∈ [0,1] |
| `D6-099` | `prob.util.trapz` | `(y: Tensor, x?: Tensor) -> number` | 复合梯形积分：Σ ½(y_i + y_{i−1})·Δx；省略 x 时按单位间距 | x 省略或与 y 等长且严格递增 |
| `D6-100` | `prob.util.normalQuantile` | `(p: number) -> number` | 逆正态分位函数 Φ⁻¹(p)：Acklam 有理逼近 + Halley 修正，精度 ~1e-12 | 0 < p < 1；单调递增；normalQuantile(Φ(x)) ≈ x |
| `D6-101` | `prob.util.distSummary` | `(dist: string, ...params) -> {dist, params, mean, variance, entropy}` | 分布摘要（均值/方差/熵，nat）：支持 normal/uniform/exponential/laplace/logistic/gamma/beta/pareto/rayleigh/weibull/chi2 | variance ≥ 0；熵对集中分布可为负（微分熵性质） |
| `D6-102` | `prob.util.sampleMoments` | `(data: Tensor) -> {mean, variance, skewness, kurtosis}` | 样本矩摘要（总体口径）：均值、方差、偏度 m3/m2^{3/2}、超额峰度 m4/m2² − 3，供与理论矩对比 | 方差为 0 时 skewness/kurtosis 记为 0 |

## D7 · ML 机器学习

经典模型族、评估指标、数据预处理。共 103 条。

| id | 指令 | 签名 | 说明 | 不变量 |
|---|---|---|---|---|
| `D7-001` | `ml.linreg.fit` | `(Tensor X[n,d], Tensor y[n]) -> object state` | 普通最小二乘闭式解（增广截距列后走 QR 最小二乘） | 残差正交于设计矩阵：Aᵀ(y−Aŵ)≈0（A=[1,X]） |
| `D7-002` | `ml.linreg.predict` | `(object state, Tensor X[m,d]) -> Tensor y[m]` | 线性回归预测 ŷ = intercept + X·coef | predict(fit(X,y), X) 等于 fit 时的 fitted 值 |
| `D7-003` | `ml.ridge.fit` | `(Tensor X[n,d], Tensor y[n], number lambda=1) -> object state` | 岭回归：最小化 (1/(2n))‖Xw−y‖² + (λ/2)‖w‖²，即解 (XᵀX+nλI)w=Xᵀy（Cholesky） | λ→0 时退化为普通最小二乘；λ→∞ 时系数趋于 0；λ 与样本量无关（按样本平均的目标） |
| `D7-004` | `ml.ridge.predict` | `(object state, Tensor X[m,d]) -> Tensor y[m]` | 岭回归预测 | λ=0 时与 ml.linreg.predict 一致 |
| `D7-005` | `ml.lasso.fit` | `(Tensor X[n,d], Tensor y[n], number lambda=1, number iters=500) -> object state` | Lasso：最小化 (1/(2n))‖Xw−y‖² + λ‖w‖₁，ISTA 近端梯度（软阈值），特征先标准化 | λ 足够大时系数出现精确 0（稀疏）；λ=0 时等于最小二乘 |
| `D7-006` | `ml.lasso.predict` | `(object state, Tensor X[m,d]) -> Tensor y[m]` | Lasso 预测（内部使用标准化空间系数换算回原始尺度） | 与 state.coef/intercept 的线性预测一致 |
| `D7-007` | `ml.elasticNet.fit` | `(Tensor X[n,d], Tensor y[n], number lambda=1, number l1Ratio=0.5, number iters=500) -> object state` | 弹性网：最小化 (1/(2n))‖Xw−y‖² + λ[α‖w‖₁ + ((1−α)/2)‖w‖²]，近端梯度（软阈值+收缩） | l1Ratio=1 退化为 Lasso；l1Ratio=0 退化为同 λ 的 Ridge |
| `D7-008` | `ml.elasticNet.predict` | `(object state, Tensor X[m,d]) -> Tensor y[m]` | 弹性网预测 | l1Ratio=1 时与 Lasso 同解 |
| `D7-009` | `ml.poly.fit` | `(Tensor X[n,d], Tensor y[n], number degree=2) -> object state` | 多项式回归：展开总次数 ≤ degree 的全部单项式后做最小二乘 | degree=1 时等价于普通线性回归 |
| `D7-010` | `ml.poly.predict` | `(object state, Tensor X[m,d]) -> Tensor y[m]` | 多项式回归预测（按 state.exps 展开基函数） | degree=1 时与 ml.linreg.predict 一致 |
| `D7-011` | `ml.robust.fit` | `(Tensor X[n,d], Tensor y[n], number delta=1.345, number iters=25) -> object state` | Huber 稳健回归：IRLS 迭代重加权最小二乘（δ 为二次/线性损失切换点） | 无异常值时结果接近最小二乘；大残差样本权重被压低 |
| `D7-012` | `ml.robust.predict` | `(object state, Tensor X[m,d]) -> Tensor y[m]` | 稳健回归预测 | 线性预测 ŷ = intercept + X·coef |
| `D7-013` | `ml.logreg.fit` | `(Tensor X[n,d], Tensor y[n], number iters=100, number ridge=1e-8) -> object state` | 二元逻辑回归：阻尼牛顿（IRLS）+ Armijo 回溯线搜索 | 梯度 Xᵀ(p−y)≈0；线性可分数据上训练准确率 = 1 |
| `D7-014` | `ml.logreg.predict` | `(object state, Tensor X[m,d]) -> Tensor labels[m]` | 逻辑回归硬分类（返回原始标签值） | 预测标签 ∈ {negLabel, posLabel}；阈值 σ(wᵀx) ≥ 0.5 |
| `D7-015` | `ml.logreg.predictProba` | `(object state, Tensor X[m,d]) -> Tensor p[m]` | 正类后验概率 σ(wᵀ[1,x]) | 概率 ∈ (0,1) 且随决策函数单调递增 |
| `D7-016` | `ml.softmax.fit` | `(Tensor X[n,d], Tensor y[n], number lr=0.1, number iters=500) -> object state` | 多类 softmax 回归（多项逻辑回归）：全批量梯度下降 + 稳定 softmax | 预测概率每行和为 1；梯度消失时 W 收敛到交叉熵驻点 |
| `D7-017` | `ml.softmax.predict` | `(object state, Tensor X[m,d]) -> Tensor labels[m]` | softmax 回归硬分类（返回原始类别值） | 预测标签必属于 state.classes |
| `D7-018` | `ml.softmax.predictProba` | `(object state, Tensor X[m,d]) -> Tensor P[m,K]` | softmax 回归类后验概率矩阵 | 每行元素非负且和为 1 |
| `D7-019` | `ml.perceptron.fit` | `(Tensor X[n,d], Tensor y[n], number lr=0.1, number epochs=200) -> object state` | 经典感知机：错分即更新 w ← w + η·y·x | 线性可分数据上有限步内收敛（错分数 → 0） |
| `D7-020` | `ml.perceptron.predict` | `(object state, Tensor X[m,d]) -> Tensor labels[m]` | 感知机预测：sign(wᵀx + b) | 预测标签 ∈ {negLabel, posLabel} |
| `D7-021` | `ml.lda.fit` | `(Tensor X[n,d], Tensor y[n]) -> object state` | 线性判别分析：类等协方差 + 高斯似然的对数线性判别函数 | 判别得分 δ_k 线性于 x；两类等协方差时等价于 Bayes 最优分类器 |
| `D7-022` | `ml.lda.predict` | `(object state, Tensor X[m,d]) -> Tensor labels[m]` | LDA 预测：argmax_k (xᵀΣ⁻¹μ_k + const_k) | 决策边界为超平面（分段线性） |
| `D7-023` | `ml.qda.fit` | `(Tensor X[n,d], Tensor y[n]) -> object state` | 二次判别分析：每类各估计一个协方差矩阵 | 决策边界为二次曲面；类内协方差相等时退化为 LDA |
| `D7-024` | `ml.qda.predict` | `(object state, Tensor X[m,d]) -> Tensor labels[m]` | QDA 预测：−½(x−μ_k)ᵀΣ_k⁻¹(x−μ_k) − ½log\|Σ_k\| + log π_k 取最大 | 预测标签必属于 state.classes |
| `D7-025` | `ml.nb.gaussianFit` | `(Tensor X[n,d], Tensor y[n]) -> object state` | 高斯朴素贝叶斯：各类各维独立估计均值与方差 | 后验 log P(y=k\|x) ∝ log π_k + Σ_j log N(x_j; μ_kj, σ²_kj) |
| `D7-026` | `ml.nb.gaussianPredict` | `(object state, Tensor X[m,d]) -> Tensor labels[m]` | 高斯朴素贝叶斯最大后验预测 | 两类等方差高斯且均值已知时准确率趋近 Bayes 最优 |
| `D7-027` | `ml.nb.multinomialFit` | `(Tensor X[n,d], Tensor y[n], number alpha=1) -> object state` | 多项式朴素贝叶斯（文本计数模型）：Laplace/Lidstone 平滑的对数似然 | 每类 θ_k 在特征维上归一：Σ_j θ_kj = 1 |
| `D7-028` | `ml.nb.multinomialPredict` | `(object state, Tensor X[m,d]) -> Tensor labels[m]` | 多项式朴素贝叶斯预测 | 得分 = log π_k + Σ_j x_j·log θ_kj |
| `D7-029` | `ml.nb.bernoulliFit` | `(Tensor X[n,d], Tensor y[n], number alpha=1) -> object state` | 伯努利朴素贝叶斯：特征二值化后估计 P(x_j=1\|y=k) | 估计概率截断到 [ε, 1−ε]，保证 log 有界 |
| `D7-030` | `ml.nb.bernoulliPredict` | `(object state, Tensor X[m,d]) -> Tensor labels[m]` | 伯努利朴素贝叶斯预测（输入按 x>0 二值化） | 得分 = log π_k + Σ_j [x_j log p_kj + (1−x_j) log(1−p_kj)] |
| `D7-031` | `ml.tree.fit` | `(Tensor X[n,d], Tensor y[n], number maxDepth=10, number minSplit=2, string task="auto") -> object state` | CART 决策树：分类用基尼不纯度、回归用平方误差，贪心递归二分 | 每个内部节点的加权不纯度下降 ≥ 0；叶节点不纯度为 0 时训练误差为 0 |
| `D7-032` | `ml.tree.predict` | `(object state, Tensor X[m,d]) -> Tensor out[m]` | 决策树预测（分类→众数标签，回归→叶内均值） | 每个样本的预测等于其落入叶节点的代表值 |
| `D7-033` | `ml.tree.predictProba` | `(object state, Tensor X[m,d]) -> Tensor P[m,K]` | 决策树叶内类概率（回归树返回 [m,1] 的叶均值） | 分类树每行概率非负且和为 1 |
| `D7-034` | `ml.tree.importance` | `(object state) -> Tensor imp[d]` | CART 特征重要性：不纯度下降量按样本数加权后归一 | 所有分量非负且和为 1 |
| `D7-035` | `ml.tree.depth` | `(object state) -> number` | 树深度（最长根到叶路径的边数，单叶树为 0） | 深度 ≤ maxDepth；单节点树深度 = 0 |
| `D7-036` | `ml.tree.leafCount` | `(object state) -> number` | 叶节点个数 | 叶数 ≥ 1；深度 ≤ D 的二叉树叶数 ≤ 2^D |
| `D7-037` | `ml.rf.fit` | `(Tensor X[n,d], Tensor y[n], number nTrees=20, number maxDepth=10, number seed=20261014) -> object state` | 随机森林：Bagging + 随机特征子集（√d）的 CART 集成 | 每棵树只在自助样本与特征子集上训练；树数越多方差越小 |
| `D7-038` | `ml.rf.predict` | `(object state, Tensor X[m,d]) -> Tensor out[m]` | 随机森林预测（分类多数投票 / 回归取均值） | 分类输出属于 state.classes；回归输出为各树叶均值的算术平均 |
| `D7-039` | `ml.rf.importance` | `(object state) -> Tensor imp[d]` | 随机森林特征重要性：各树不纯度下降量的平均并归一 | 所有分量非负且和为 1（森林非空时） |
| `D7-040` | `ml.gb.fit` | `(Tensor X[n,d], Tensor y[n], number nEst=50, number lr=0.1, number maxDepth=3) -> object state` | 梯度提升回归树：每轮用回归树拟合负梯度（残差）并做收缩 | 训练损失随提升轮数单调不增（shrinkage 足够小时） |
| `D7-041` | `ml.gb.predict` | `(object state, Tensor X[m,d]) -> Tensor y[m]` | 梯度提升树预测：base + lr·Σ 各轮回归树输出 | nEst=0 时退化为常数 base（训练集均值） |
| `D7-042` | `ml.gb.stagedLoss` | `(object state, Tensor X[n,d], Tensor y[n]) -> Tensor loss[nEst]` | 逐轮累计的均方误差（第 m 项为前 m+1 棵树集成后的损失） | lr 足够小时序列单调不增 |
| `D7-043` | `ml.ada.fit` | `(Tensor X[n,d], Tensor y[n], number nEst=50) -> object state` | 离散 AdaBoost（SAMME）：决策树桩 + 加权多数投票，二分类退化为经典 AdaBoost.M1 | 加权训练误差随轮数下降；α_m = log((1−err)/err) + log(K−1) |
| `D7-044` | `ml.ada.predict` | `(object state, Tensor X[m,d]) -> Tensor labels[m]` | AdaBoost 预测：Σ_m α_m·[h_m(x)=k] 取最大 | 预测标签必属于 state.classes |
| `D7-045` | `ml.svm.fit` | `(Tensor X[n,d], Tensor y[n], number C=1, number epochs=300, number lr=0.01) -> object state` | 线性 SVM：hinge 损失次梯度 SGD（Pegasos 风格投影步） | 目标 (½‖w‖² + C·Σhinge) 随迭代下降；可分数据上间隔 > 0 |
| `D7-046` | `ml.svm.predict` | `(object state, Tensor X[m,d]) -> Tensor labels[m]` | SVM 硬分类：决策函数取符号（线性核与 RBF 核状态通用） | 预测标签 ∈ {negLabel, posLabel} |
| `D7-047` | `ml.svm.decision` | `(object state, Tensor X[m,d]) -> Tensor score[m]` | SVM 决策函数值：线性核为 wᵀx+b，核 SVM 为 Σα_i y_i k(x_i,x)+b | sign(decision) = predict 的标签映射 |
| `D7-048` | `ml.svm.kernelFit` | `(Tensor X[n,d], Tensor y[n], number C=1, number gamma=0.5, number tol=1e-3, number maxIter=200) -> object state` | 核 SVM 二分类：Platt 简化 SMO（RBF 核），只保留支持向量 | KKT 条件在 tol 内成立；0 ≤ α_i ≤ C |
| `D7-049` | `ml.kernel.linear` | `(Tensor A[m,d], Tensor B[n,d]) -> Tensor K[m,n]` | 线性核 K(x,z) = xᵀz | K 对称半正定；K(x,x) = ‖x‖² |
| `D7-050` | `ml.kernel.rbf` | `(Tensor A[m,d], Tensor B[n,d], number gamma=1) -> Tensor K[m,n]` | 高斯 RBF 核 K(x,z) = exp(−γ‖x−z‖²) | K 对称半正定、对角元 = 1、元素 ∈ (0,1] |
| `D7-051` | `ml.kernel.poly` | `(Tensor A[m,d], Tensor B[n,d], number degree=3, number coef0=1, number gamma=1) -> Tensor K[m,n]` | 多项式核 K(x,z) = (γ·xᵀz + coef0)^degree | degree=1 且 coef0=0 时退化为线性核的 γ 倍 |
| `D7-052` | `ml.kernel.laplacian` | `(Tensor A[m,d], Tensor B[n,d], number gamma=1) -> Tensor K[m,n]` | 拉普拉斯核 K(x,z) = exp(−γ‖x−z‖₁) | K 对称半正定、对角元 = 1 |
| `D7-053` | `ml.kernel.cosine` | `(Tensor A[m,d], Tensor B[n,d]) -> Tensor K[m,n]` | 余弦相似度核 K(x,z) = xᵀz / (‖x‖‖z‖) | 对角元 = 1；元素 ∈ [−1,1]；对向量缩放不变 |
| `D7-054` | `ml.kernel.center` | `(Tensor K[n,n]) -> Tensor Kc[n,n]` | 核矩阵中心化：Kc = (I − 11ᵀ/n) K (I − 11ᵀ/n)，即特征空间去均值 | Kc 的行和与列和均为 0 |
| `D7-055` | `ml.knn.fit` | `(Tensor X[n,d], Tensor y[n]) -> object state` | k 近邻惰性学习器：仅存储训练集（任务自动判别分类/回归） | k=1 时训练集留一误差为 0（无重复样本时训练误差为 0） |
| `D7-056` | `ml.knn.predict` | `(object state, Tensor X[m,d], number k=5) -> Tensor labels[m]` | k 近邻多数投票分类（平票取标签值较小者） | k=1 时训练集预测误差为 0 |
| `D7-057` | `ml.knn.predictProba` | `(object state, Tensor X[m,d], number k=5) -> Tensor P[m,K]` | k 近邻类概率（邻域内类别频率） | 每行非负且和为 1；列序与 state.classes 一致 |
| `D7-058` | `ml.knn.regress` | `(object state, Tensor X[m,d], number k=5) -> Tensor y[m]` | k 近邻回归（邻域标签均值） | k=1 时训练集预测值等于训练标签 |
| `D7-059` | `ml.kmeans.fit` | `(Tensor X[n,d], number k, number seed=20261014, number iters=100, string init="kmeanspp") -> object state` | k-means（Lloyd 迭代），初始化支持 k-means++ 或随机 | 每轮迭代惯量（inertia）单调不增 |
| `D7-060` | `ml.kmeans.predict` | `(object state, Tensor X[m,d]) -> Tensor labels[m]` | k-means 预测：指派到最近质心 | 标签 ∈ [0, k)；指派结果使 Σ‖x−c‖² 最小 |
| `D7-061` | `ml.kmeans.inertia` | `(object state, Tensor X[n,d]) -> number` | k-means 惯量：Σ‖x_i − c_{l_i}‖²（簇内平方和） | 惯量 ≥ 0；k=n 时为 0 |
| `D7-062` | `ml.kmeanspp.init` | `(Tensor X[n,d], number k, number seed=20261014) -> Tensor C[k,d]` | k-means++ D² 加权采样初始化质心 | 质心均取自训练样本；期望代价 ≤ O(log k)·OPT |
| `D7-063` | `ml.dbscan.fit` | `(Tensor X[n,d], number eps, number minPts=4) -> object state` | DBSCAN 基于密度的聚类，噪声点标签为 −1 | 核心点被分入某簇；稀疏区域的点标签为 −1 |
| `D7-064` | `ml.dbscan.predict` | `(object state, Tensor X[m,d]) -> Tensor labels[m]` | DBSCAN 对新样本指派：eps 内最近核心点的簇，否则 −1 | 不在任何核心点 eps 邻域内的样本标签为 −1 |
| `D7-065` | `ml.agglo.fit` | `(Tensor X[n,d], number k=2, string linkage="average") -> object state` | 凝聚层次聚类：自底向上按 linkage 合并，最后切成 k 簇 | 链接距离序列单调不减；k=1 时全部同簇，k=n 时各成一类 |
| `D7-066` | `ml.agglo.labels` | `(object state) -> Tensor labels[n]` | 取出层次聚类在指定 k 下的簇标签 | 标签取值集合的基数 ≤ k |
| `D7-067` | `ml.linkage.single` | `(Tensor X[n,d]) -> object {Z, n, method}` | 单链接层次聚类（最近邻距离），返回 scipy 风格链接矩阵 Z | Z 的合并距离单调不减；d(A∪B, C) = min(d(A,C), d(B,C)) |
| `D7-068` | `ml.linkage.complete` | `(Tensor X[n,d]) -> object {Z, n, method}` | 全链接层次聚类（最远距离） | Z 的合并距离单调不减；d(A∪B, C) = max(d(A,C), d(B,C)) |
| `D7-069` | `ml.linkage.average` | `(Tensor X[n,d]) -> object {Z, n, method}` | 均链接层次聚类（UPGMA，按簇大小加权平均） | Z 的合并距离单调不减；d(A∪B,C) = (\|A\|d(A,C)+\|B\|d(B,C))/(\|A\|+\|B\|) |
| `D7-070` | `ml.spectral.fit` | `(Tensor X[n,d], number k=2, number gamma=auto) -> object state` | 谱聚类（Ng-Jordan-Weiss）：RBF 相似图 → 归一化拉普拉斯 → k 个最小特征向量 → k-means | L_sym = I − D^{−1/2}WD^{−1/2} 半正定；特征向量行归一后行范数为 1 |
| `D7-071` | `ml.spectral.labels` | `(object state) -> Tensor labels[n]` | 取出谱聚类的簇标签 | 标签取值集合的基数 ≤ k |
| `D7-072` | `ml.pca.fit` | `(Tensor X[n,d], number nComp=d) -> object state` | 主成分分析：中心化协方差的对称特征分解，载荷矩阵按特征值降序 | 主成分得分协方差的非对角元 ≈ 0；特征值之和 = 总方差 |
| `D7-073` | `ml.pca.transform` | `(object state, Tensor X[m,d]) -> Tensor Z[m,nComp]` | PCA 投影：Z = (X − mean)·Wᵀ | 得分矩阵的列间协方差非对角元 ≈ 0 |
| `D7-074` | `ml.pca.inverse` | `(object state, Tensor Z[m,nComp]) -> Tensor X̂[m,d]` | PCA 逆变换：X̂ = Z·W + mean（主成分子空间内的重构） | nComp=d 时 inverse(transform(x)) = x 精确成立 |
| `D7-075` | `ml.pca.explainedVar` | `(object state) -> Tensor ev[nComp]` | 各主成分解释的方差（协方差特征值，降序） | nComp=d 时各分量之和 = 总方差（协方差矩阵的迹） |
| `D7-076` | `ml.kpca.fit` | `(Tensor X[n,d], number nComp=2, number gamma=auto) -> object state` | 核 PCA：RBF 核矩阵中心化后特征分解，得到特征空间主成分方向 | 中心化核的特征值非负；投影方向的核范数归一（αᵀKα = 1） |
| `D7-077` | `ml.kpca.transform` | `(object state, Tensor X[m,d]) -> Tensor Z[m,nComp]` | 核 PCA 投影：测试核矩阵按训练集统计量中心化后与 α 相乘 | 对训练集自身投影得到 √λ 缩放的主成分坐标 |
| `D7-078` | `ml.mds.classical` | `(Tensor D[n,n], number nComp=2) -> Tensor Y[n,nComp]` | 经典多维尺度分析（Torgerson）：双中心化内积矩阵的特征映射 | 嵌入的欧氏距离在 nComp 足够时逼近输入距离矩阵 |
| `D7-079` | `ml.whiten` | `(Tensor X[n,d]) -> Tensor Z[n,d]` | ZCA 白化：去相关并把各主成分缩放到单位方差（仍留在原坐标系） | 白化后各列均值 ≈ 0、协方差矩阵 ≈ 单位阵 |
| `D7-080` | `ml.scale.standard` | `(Tensor X[n,d]) -> object {X, mean, std}` | 标准化（Z-score）：每列减均值除以标准差 | 变换后每列均值 ≈ 0、标准差 ≈ 1 |
| `D7-081` | `ml.scale.minmax` | `(Tensor X[n,d], number lo=0, number hi=1) -> object {X, min, max}` | 最小-最大缩放：把每列线性映射到 [lo, hi] | 变换后每列最小值 = lo、最大值 = hi（常数列退化为 lo） |
| `D7-082` | `ml.scale.robust` | `(Tensor X[n,d]) -> object {X, center, scale}` | 稳健缩放：减中位数除以四分位距（对异常值不敏感） | 变换后每列中位数 ≈ 0、四分位距 ≈ 1 |
| `D7-083` | `ml.scale.maxabs` | `(Tensor X[n,d]) -> object {X, maxAbs}` | 最大绝对值缩放：每列除以该列绝对值的最大值（保稀疏零元） | 变换后每列元素 ∈ [−1, 1]，且至少有一个绝对值为 1 |
| `D7-084` | `ml.scale.normalize` | `(Tensor X[n,d]) -> object {X, norms}` | 样本级归一化：每行缩放到单位 L2 范数（文本/余弦场景常用） | 变换后每个行向量的 L2 范数 = 1（零行保持为 0） |
| `D7-085` | `ml.encode.onehot` | `(Tensor codes[n], number k=max+1) -> Tensor OneHot[n,k]` | 独热编码：整数类别码 → n×k 的 0/1 矩阵 | 每行恰有一个 1，其余为 0；列数 = 类别数 |
| `D7-086` | `ml.encode.label` | `(Tensor y[n]) -> object {codes, classes}` | 标签编码：把任意标量标签映射到 0..K−1 的整数码 | classes[codes[i]] = y[i]（一一对应可逆） |
| `D7-087` | `ml.impute.mean` | `(Tensor X[n,d]) -> object {X, fill}` | 均值填补缺失值（NaN 用所在列的均值替换） | 输出不含 NaN；填补值等于该列非缺失元素的均值 |
| `D7-088` | `ml.impute.median` | `(Tensor X[n,d]) -> object {X, fill}` | 中位数填补缺失值（对偏态分布比均值更稳健） | 输出不含 NaN；填补值等于该列非缺失元素的中位数 |
| `D7-089` | `ml.select.variance` | `(Tensor X[n,d], number threshold=0) -> object {scores, mask, index}` | 方差阈值筛选：剔除方差低于阈值的（近）常数列 | 保留列的方差均 >= threshold |
| `D7-090` | `ml.select.fTest` | `(Tensor X[n,d], Tensor y[n]) -> object {scores, order}` | 单变量 F 检验打分：分类用单因素 ANOVA F，回归用线性相关 F | F 统计量非负；与标签无关的特征 F ≈ 0 |
| `D7-091` | `ml.select.mutualInfo` | `(Tensor X[n,d], Tensor y[n], number bins=10) -> object {scores, order}` | 互信息特征打分：等宽分箱后的离散互信息 I(X_j; Y)（自然对数） | 互信息非负；独立特征的分箱互信息 ≈ 0 |
| `D7-092` | `ml.metric.mse` | `(Tensor yTrue[n], Tensor yPred[n]) -> number` | 均方误差 MSE = mean((y−ŷ)²) | MSE ≥ 0；完美预测时为 0 |
| `D7-093` | `ml.metric.rmse` | `(Tensor yTrue[n], Tensor yPred[n]) -> number` | 均方根误差 RMSE = √MSE（与目标同量纲） | RMSE = √MSE 且 ≥ 0 |
| `D7-094` | `ml.metric.mae` | `(Tensor yTrue[n], Tensor yPred[n]) -> number` | 平均绝对误差 MAE = mean\|y−ŷ\| | MAE ≥ 0 且 ≤ RMSE（同一样本上） |
| `D7-095` | `ml.metric.r2` | `(Tensor yTrue[n], Tensor yPred[n]) -> number` | 决定系数 R² = 1 − SS_res/SS_tot | 完美预测时 R² = 1；常数基线预测（均值）时 R² = 0 |
| `D7-096` | `ml.metric.silhouette` | `(Tensor X[n,d], Tensor labels[n]) -> object {score, samples}` | 轮廓系数：返回均值 score 与逐样本 samples | score ∈ [−1, 1]；每簇仅 1 个样本时该样本轮廓为 0 |
| `D7-097` | `ml.metric.ari` | `(Tensor a[n], Tensor b[n]) -> number` | 调整兰德指数 ARI（对随机划分做机会校正的聚类一致性） | 相同划分时 ARI = 1；期望值（随机独立划分）为 0；取值 ≤ 1 |
| `D7-098` | `ml.metric.calinski` | `(Tensor X[n,d], Tensor labels[n]) -> number` | Calinski-Harabasz 指数：簇间/簇内方差比（越大越好） | k=1 时无定义（返回 0）；簇越分离、越紧凑则指数越大 |
| `D7-099` | `ml.util.clone` | `(object state) -> object state` | 深拷贝估计器 state（要求 state 可 JSON 序列化） | 克隆与原对象值相等但不共享引用 |
| `D7-100` | `ml.util.getParams` | `(object state) -> object params` | 提取 state 中的标量超参数（number/string/boolean） | 返回值全部为标量，且键集 ⊆ state 的键集 |
| `D7-101` | `ml.util.setParams` | `(object state, object patch) -> object state` | 按补丁合并超参数，返回新的 state（不改原对象） | 返回对象的 params = 原 params 覆盖 patch；原 state 不被修改 |
| `D7-102` | `ml.util.describe` | `(object state) -> object summary` | 估计器自描述：kind、顶层键、标量参数与序列化体积 | keys 等于 state 的顶层键集合；params ⊆ keys |
| `D7-103` | `ml.metric.davies` | `(Tensor X[n,d], Tensor labels[n]) -> number` | Davies-Bouldin 指数：簇内散度与簇间距之比的最大值的均值（越小越好） | DB ≥ 0；k=1 时无定义（返回 0）；完全分离且紧凑的簇 DB → 0 |

## D8 · DL 深度学习

层、激活、注意力、循环网络、损失、初始化。共 103 条。

| id | 指令 | 签名 | 说明 | 不变量 |
|---|---|---|---|---|
| `D8-001` | `dl.act.relu` | `(Tensor x) -> Tensor` | ReLU max(0,x)，逐元素；0 点次梯度取 0 | relu(x) ≥ 0；relu(−1)=0；relu(x)+relu(−x)=\|x\| |
| `D8-002` | `dl.act.leakyRelu` | `(Tensor x, number alpha=0.01) -> Tensor` | LeakyReLU：负区间保留 alpha 倍斜率，缓解 dying ReLU | leakyRelu(x) = x (x>0) = alpha·x (x≤0)；alpha=0 退化为 relu |
| `D8-003` | `dl.act.elu` | `(Tensor x, number alpha=1) -> Tensor` | ELU：x>0 取 x，否则 alpha·(e^x−1)，负区间饱和于 −alpha | elu(0)=0；x→−∞ 时 elu(x)→−alpha；x>0 段为恒等 |
| `D8-004` | `dl.act.selu` | `(Tensor x) -> Tensor` | SELU：lambda·ELU(x, alpha)，固定 alpha=1.6732632、lambda=1.0507010，具自归一化性质 | selu(0)=0；不动点附近雅可比谱半径≈1（自归一化）；x→−∞ 时 → −lambda·alpha |
| `D8-005` | `dl.act.gelu` | `(Tensor x, string approx='exact') -> Tensor` | GELU = x·Φ(x)；approx="exact" 用 erf 精确计算，"tanh" 用 tanh 近似 | GELU(0)=0；x→∞ 时 GELU(x)→x；x→−∞ 时 →0；0<GELU(x)<x (x>0) |
| `D8-006` | `dl.act.mish` | `(Tensor x) -> Tensor` | Mish = x·tanh(softplus(x))，下界约 −0.30884，上界无界 | mish(0)=0；mish(x)≈x（x 大）；最小值 ≈ −0.30884（x≈−1.1924） |
| `D8-007` | `dl.act.swish` | `(Tensor x, number beta=1) -> Tensor` | Swish/SiLU = x·σ(beta·x)；beta=1 即 SiLU，beta→∞ 趋近 ReLU | swish(0)=0；swish(x)→x（x→∞）；swish(x)→0（x→−∞） |
| `D8-008` | `dl.act.sigmoid` | `(Tensor x) -> Tensor` | Logistic sigmoid σ(x)=1/(1+e^{−x})，输出落在 (0,1) | sigmoid(0)=0.5；sigmoid(x)+sigmoid(−x)=1；值域 ⊂ (0,1) |
| `D8-009` | `dl.act.tanh` | `(Tensor x) -> Tensor` | 双曲正切，零中心、值域 (−1,1) | tanh(0)=0；值域 ⊂ (−1,1)；tanh 为奇函数 |
| `D8-010` | `dl.act.softplus` | `(Tensor x, number beta=1, number threshold=20) -> Tensor` | Softplus = (1/beta)·log(1+e^{beta·x})；beta·x 超过 threshold 时线性化避免溢出 | softplus(0)=ln2；softplus(x)→x（x 大）；softplus(x)>0 且单调 |
| `D8-011` | `dl.act.softsign` | `(Tensor x) -> Tensor` | Softsign = x/(1+\|x\|)，tanh 的多项式型替代，尾部收敛更慢 | softsign(0)=0；值域 ⊂ (−1,1)；softsign 为奇函数 |
| `D8-012` | `dl.act.hardSigmoid` | `(Tensor x) -> Tensor` | Hard sigmoid = clamp((x+3)/6, 0, 1)，分段线性、无 exp，适合量化部署 | x≤−3 时为 0；x≥3 时为 1；在 [−3,3] 上斜率为 1/6 |
| `D8-013` | `dl.act.prelu` | `(Tensor x[...,C], Tensor alpha[C]) -> Tensor` | PReLU：负区间斜率为可学习参数 alpha（按最后一维通道广播） | alpha=0 退化为 relu；alpha=1 退化为恒等；x>0 段恒等 |
| `D8-014` | `dl.act.glu` | `(Tensor x, number axis=-1) -> Tensor` | Gated Linear Unit：沿 axis 对半切分为 (a,b)，输出 a·σ(b) | axis 维长度减半；b→−∞ 时输出→0，b→∞ 时输出→a |
| `D8-015` | `dl.norm.batchnorm` | `(Tensor x[N,C,...], Tensor gamma[C], Tensor beta[C], Tensor mean[C], Tensor var[C], number training=1, number eps=1e-5) -> Tensor` | 批归一化：training=1 用本批统计量，training=0 用传入的 running 统计量；gamma/beta 为仿射参数 | 训练态输出按通道均值≈0、方差≈1；推理态使用 running 统计；eps 防止除零 |
| `D8-016` | `dl.norm.layernorm` | `(Tensor x[...,D], Tensor gamma[D], Tensor beta[D], number eps=1e-5) -> Tensor` | 层归一化：沿最后一维去均值除标准差，再做 gamma/beta 仿射；与 batch 大小无关 | 归一化后每条样本均值≈0、方差≈1（gamma=1,beta=0 时精确成立） |
| `D8-017` | `dl.norm.rmsnorm` | `(Tensor x[...,D], Tensor gamma[D], number eps=1e-5) -> Tensor` | RMSNorm：仅按均方根缩放 x/√(mean(x²)+eps)·gamma，无减均值操作（LLaMA 采用） | 输出 RMS ≈ 1（gamma=1 时）；平移 x→x+c 不保持（无去均值） |
| `D8-018` | `dl.norm.instancenorm` | `(Tensor x[N,C,H,W], Tensor gamma[C], Tensor beta[C], number eps=1e-5) -> Tensor` | 实例归一化：对每个样本每个通道独立地在空间维 (H,W) 上归一化（风格迁移常用） | 每个 (n,c) 通道的 H·W 个元素均值≈0、方差≈1；与 batch 无关 |
| `D8-019` | `dl.norm.groupnorm` | `(Tensor x[N,C,H,W], Tensor gamma[C], Tensor beta[C], number groups, number eps=1e-5) -> Tensor` | 组归一化：把 C 个通道分为 groups 组，在每组通道×空间上归一化；groups=1 近似 LayerNorm，groups=C 即 InstanceNorm | 每组 (N/组, C/groups·H·W) 元素均值≈0、方差≈1；groups 必须整除 C |
| `D8-020` | `dl.norm.weightnorm` | `(Tensor v[out,in], Tensor g[out], number eps=1e-12) -> Tensor` | 权重归一化：把权重分解为方向 v/\|\|v\|\| 与幅度 g，输出每行范数恰为 \|g\| | 输出第 i 行的 L2 范数 = \|g_i\|；方向不变：w_i ∝ v_i |
| `D8-021` | `dl.norm.spectralnorm` | `(Tensor W[M,N], number nIter=10, number eps=1e-12) -> {W: Tensor, sigma: number}` | 谱归一化：幂迭代估计最大奇异值 sigma，返回 W/sigma（谱范数≈1），用于稳定 GAN 判别器 | 归一化后矩阵的谱范数 ≈ 1；sigma 收敛到最大奇异值；迭代不发散 |
| `D8-022` | `dl.norm.l2normalize` | `(Tensor x, number axis=-1, number eps=1e-12) -> Tensor` | L2 归一化：沿 axis 除以 L2 范数（常用于对比学习的特征球投影） | 沿 axis 的 L2 范数 = 1（范数非零时）；零向量映射为零向量 |
| `D8-023` | `dl.norm.batchStats` | `(Tensor x, number[] axes, number correction=0) -> {mean: Tensor, var: Tensor}` | 沿 axes 求均值与方差并保持维数（keepdims）；correction=1 给出无偏（贝塞尔）方差 | 输出 shape 中被归约轴置 1；correction=1 时为无偏估计；var ≥ 0 |
| `D8-024` | `dl.norm.runningUpdate` | `(Tensor running, Tensor batch, number momentum=0.9) -> Tensor` | 滑动更新 running 统计量：running ← momentum·running + (1−momentum)·batch | momentum=0 时等于 batch 值；momentum→1 时保持不变；凸组合保持有界 |
| `D8-025` | `dl.layer.dense` | `(Tensor x[N,Di], Tensor W[Di,Do], Tensor b[Do]) -> Tensor[N,Do]` | 全连接层（GEMM + 偏置）：y = x·W + b，PyTorch nn.Linear / ONNX Gemm 同源 | 输出形状 [N,Do]；b=0 且 W=I 时为恒等映射；对样本维可加（逐行独立） |
| `D8-026` | `dl.layer.conv1d` | `(Tensor x[N,C,L], Tensor w[OC,IC,K], Tensor b[OC], number stride=1, number padding=0, number dilation=1) -> Tensor[N,OC,Lout]` | 一维互相关卷积（深度学习惯例，不翻转核）；Lout = ⌊(L+2·pad−dil·(K−1)−1)/stride⌋+1 | 输出长度满足公式；核为 delta 且 b=0 时近似恒等；padding=0,stride=1,K=1 时 Lo=L |
| `D8-027` | `dl.layer.conv2d` | `(Tensor x[N,C,H,W], Tensor w[OC,IC,KH,KW], Tensor b[OC], number\|[2] stride=1, number\|[2] padding=0, number\|[2] dilation=1) -> Tensor[N,OC,Ho,Wo]` | 二维互相关卷积（NCHW）；Ho = ⌊(H+2·padH−dilH·(KH−1)−1)/strideH⌋+1 | 输出空间尺寸满足公式；与 im2col+GEMM 数值等价（误差 <1e-10）；通道数 = OC |
| `D8-028` | `dl.layer.depthwiseConv2d` | `(Tensor x[N,C,H,W], Tensor w[C,1,KH,KW], Tensor b[C], number\|[2] stride=1, number\|[2] padding=0) -> Tensor[N,C,Ho,Wo]` | 深度可分离卷积的空间部分：每个输入通道独立用一个 KH×KW 核卷积，输出通道数不变 | 输出通道数 = 输入通道数；参数量 = C·KH·KW（远小于标准卷积）；等价于 groups=C 的分组卷积 |
| `D8-029` | `dl.layer.transposeConv2d` | `(Tensor x[N,C,H,W], Tensor w[C,OC,KH,KW], Tensor b[OC], number\|[2] stride=1, number\|[2] padding=0, number\|[2] outPadding=0) -> Tensor[N,OC,Ho,Wo]` | 转置卷积（反卷积）：Ho = (H−1)·stride − 2·padding + KH + outPadding；实现为「输入插零 + 边界补 (KH−1−pad) + 翻转核的互相关」 | 输出尺寸满足公式；stride=1,pad=0 且核为 delta 时为恒等；与梯度反向传播的对偶卷积一致 |
| `D8-030` | `dl.layer.maxpool1d` | `(Tensor x[N,C,L], number kernel, number stride) -> Tensor[N,C,Lo]` | 一维最大池化：窗口内取最大值，Lo = ⌊(L−kernel)/stride⌋+1 | 输出每个元素都是对应窗口的最大值；kernel=1 时为恒等；下界不超过输入最大值 |
| `D8-031` | `dl.layer.avgpool1d` | `(Tensor x[N,C,L], number kernel, number stride) -> Tensor[N,C,Lo]` | 一维平均池化：窗口内取算术平均（计数按完整窗口，边界不补零） | 输出每个元素落在窗口最小/最大之间；常数输入保持不变 |
| `D8-032` | `dl.layer.maxpool2d` | `(Tensor x[N,C,H,W], number\|[2] kernel, number\|[2] stride) -> Tensor[N,C,Ho,Wo]` | 二维最大池化（NCHW）：Ho = ⌊(H−kH)/sH⌋+1 | 输出元素 = 对应 kH×kW 窗口的最大值；平移等变性（步长整除时） |
| `D8-033` | `dl.layer.avgpool2d` | `(Tensor x[N,C,H,W], number\|[2] kernel, number\|[2] stride) -> Tensor[N,C,Ho,Wo]` | 二维平均池化（NCHW）：窗口内算术平均，替代全连接头以降低参数量 | 输出元素 = 窗口均值，落在窗口 min/max 之间；常数输入输出不变 |
| `D8-034` | `dl.layer.globalAvgPool` | `(Tensor x[N,C,H,W]) -> Tensor[N,C]` | 全局平均池化：对每个样本每个通道求整张特征图的均值 | 输出形状 [N,C]；每个输出 = 对应通道 H·W 个元素的算术平均；与 kernel=(H,W) 的平均池化一致 |
| `D8-035` | `dl.layer.flatten` | `(Tensor x, number startDim=1) -> Tensor[lead, rest]` | 把 [startDim, ndim) 的维压平成一维，输出二维 [lead, rest]；卷积特征送入全连接前的标准操作 | 元素总数不变且行主序顺序不变；startDim=1 时 [N,C,H,W] → [N, C·H·W] |
| `D8-036` | `dl.layer.unflatten` | `(Tensor x[N,P], number[] shape) -> Tensor[N,...shape]` | flatten 的逆操作：把最后一维还原成给定形状（prod(shape) 必须等于 P） | unflatten(flatten(x)) = x（形状与元素均一致）；元素总数守恒 |
| `D8-037` | `dl.layer.embedding` | `(Tensor idx[B,T], Tensor table[V,C]) -> Tensor[B,T,C]` | 查表嵌入：把整数 id 映射为表向量；等价于以 one-hot 行向量左乘嵌入矩阵 | 输出形状 [B,T,C]；out[b,t,:] = table[idx[b,t],:]；id 越界报错 |
| `D8-038` | `dl.layer.patching` | `(Tensor img[N,C,H,W], number patch) -> Tensor[N, L, C·patch·patch]` | ViT 式分块：把图像切成不重叠 patch 并拉平为 token 序列，L = (H/patch)·(W/patch) | 输出形状 [N, L, C·p²]；元素总数与输入一致；patch=1 时 token 数 = H·W |
| `D8-039` | `dl.reg.dropout` | `(Tensor x, number p, number seed, number training=1) -> Tensor` | Inverted dropout：训练期以概率 p 置零并放大 1/(1−p) 保持期望不变；推理期恒等 | 训练期 E[out] = x；非零元素恰被放大 1/(1−p)；training=0 时 out = x |
| `D8-040` | `dl.reg.dropPath` | `(Tensor x, number p, number seed, number training=1) -> Tensor` | DropPath：整条残差分支按样本（第 0 维）同生共死地丢弃，是 Stochastic Depth 的分支版实现 | 同一样本的所有元素共享同一掩码；E[out]=x；training=0 时恒等 |
| `D8-041` | `dl.reg.stochasticDepth` | `(Tensor x, Tensor residual, number p, number seed, number training=1) -> Tensor` | 随机深度：out = x + b·residual，b ∈ {0, 1/(1−p)} 按样本随机，训练期逐层递减存活率 | E[out] = x + residual；b 只有 0 与 1/(1−p) 两种取值；training=0 时 out = x + residual |
| `D8-042` | `dl.reg.l1` | `(Tensor W, number lambda) -> number` | L1 正则项 = lambda·Σ\|w\|，诱导稀疏解（LASSO 式） | l1 ≥ 0；W=0 时为 0；对 W 缩放 c 倍则值放大 \|c\| 倍 |
| `D8-043` | `dl.reg.l2` | `(Tensor W, number lambda) -> number` | L2 正则项 = 0.5·lambda·Σw²（岭回归式；0.5 使梯度恰为 lambda·w） | l2 ≥ 0；梯度 ∂/∂w = lambda·w；W=0 时为 0 |
| `D8-044` | `dl.reg.l1l2` | `(Tensor W, number l1, number l2) -> number` | Elastic Net 式混合正则 = l1·Σ\|w\| + 0.5·l2·Σw² | l1=0 退化为 l2 项；l2=0 退化为 l1 项；非负且凸 |
| `D8-045` | `dl.reg.mixup` | `(Tensor a, Tensor b, number lam) -> Tensor` | Mixup 线性插值：out = lam·a + (1−lam)·b；lam 通常采样自 Beta(alpha, alpha) | lam=1 时 out=a，lam=0 时 out=b；lam∈[0,1] 时输出落在 a、b 的凸包内 |
| `D8-046` | `dl.attn.causalMask` | `(number T) -> Tensor[T,T]` | 因果掩码：下三角（含主对角）为 1，严格上三角为 0，用于自回归解码 | 下三角含对角为 1；严格上三角（j>i）为 0；每行元素和 = i+1 |
| `D8-047` | `dl.attn.paddingMask` | `(Tensor lengths[B], number T) -> Tensor[B,T]` | padding 掩码：长度以内的位置为 1（有效），其余为 0（填充），用于变长序列批训练 | 第 b 行前 lengths[b] 个为 1、其余为 0；每行和为 lengths[b] |
| `D8-048` | `dl.attn.scaledDot` | `(Tensor Q[B,Tq,D], Tensor K[B,Tk,D], Tensor V[B,Tk,Dv], Tensor mask?, number scale=1/√D) -> {out, attn}` | 缩放点积注意力：attn = softmax(QKᵀ·scale + 掩码)，out = attn·V；被掩位置加 −1e9 后再归一化 | attn 每行和为 1 且非负；被掩位置权重恰为 0；scale 不影响 argmax |
| `D8-049` | `dl.attn.multiHead` | `(Tensor x[B,T,C], Tensor Wq[C,C], Tensor Wk[C,C], Tensor Wv[C,C], Tensor Wo[C,C], number heads, Tensor mask?) -> Tensor[B,T,C]` | 多头自注意力（含 QKV/O 四组投影）：先投影再分头并行注意力，最后拼接并输出投影 | 输出形状 [B,T,C]；heads 整除 C；heads=1 时退化为单头注意力 |
| `D8-050` | `dl.attn.selfAttention` | `(Tensor x[B,T,C], number heads, Tensor mask?) -> {out: Tensor[B,T,C], attn: Tensor[B,H,T,T]}` | 无投影自注意力：Q=K=V=x（恒等投影），用于分析/教学与残差流上的纯注意力 | 输出形状 [B,T,C]；attn 每行和为 1；mask 位置权重为 0 |
| `D8-051` | `dl.attn.crossAttention` | `(Tensor q[B,Tq,C], Tensor kv[B,Tk,C], number heads, Tensor mask?) -> {out, attn}` | 交叉注意力：查询来自 q、键值来自 kv（恒等投影），编码器-解码器桥接结构 | 输出时间维 = Tq（查询长度）；键值时间维 = Tk；attn 每行和为 1 |
| `D8-052` | `dl.attn.groupedQuery` | `(Tensor Q[B,Tq,H·D], Tensor K[B,Tk,Hkv·D], Tensor V[B,Tk,Hkv·D], number heads, number kvHeads, Tensor mask?) -> {out, attn}` | 分组查询注意力 GQA：H 个 query 头共享 Hkv 个 KV 头（第 h 个 query 头映射到第 ⌊h·Hkv/H⌋ 个 KV 头），大幅压缩 KV 缓存 | heads 必须被 kvHeads 整除；kvHeads=heads 退化为 MHA，kvHeads=1 退化为 MQA；输出 [B,Tq,H·D] |
| `D8-053` | `dl.attn.rotaryEmbed` | `(number dim, number maxLen, number base=10000) -> {cos: Tensor[maxLen,dim/2], sin: Tensor[maxLen,dim/2]}` | RoPE 频率表：θ_m,i = m·base^{−2i/dim}，返回半长 cos/sin 表（GPT-J/LLaMA 的半分块排布） | cos²+sin²=1；m=0 时 cos=1, sin=0；频率随 i 几何衰减 base^{−2i/dim} |
| `D8-054` | `dl.attn.rotaryApply` | `(Tensor x[...,T,...,D], Tensor cos[maxLen,D/2], Tensor sin[maxLen,D/2], number start=0) -> Tensor` | 应用 RoPE（半分块形式）：x=(a,b) → (a·cos − b·sin, b·cos + a·sin)，位置取自第 1 维并偏移 start | 范数保持 \|\|R(x)\|\| = \|\|x\|\|；内积 ⟨R(q,m),R(k,n)⟩ 只依赖相对位置 m−n |
| `D8-055` | `dl.attn.alibiBias` | `(number heads, number T) -> Tensor[heads,T,T]` | ALiBi 加性位置偏置：bias[h,i,j] = −m_h·\|i−j\|，斜率 m_h = 2^{−8(h+1)/heads} 几何分布，无需位置嵌入即可外推 | 对角恒为 0；bias ≤ 0；\|i−j\| 越大越负；不同头斜率互不相同且几何衰减 |
| `D8-056` | `dl.attn.relativeBias` | `(number T, number numBuckets=32, number seed=20261014) -> Tensor[T,T]` | 相对位置偏置：把距离 \|i−j\| 分桶（小距离精确、大距离对数分桶）后查可学习表，双向对称 | B[i][j] 只依赖 \|i−j\|；B 对称；对角元素全部相同 |
| `D8-057` | `dl.attn.kvCacheInit` | `(number B, number heads, number maxLen, number headDim) -> {k: Tensor[B,H,0,D], v: Tensor[B,H,0,D], len: 0, maxLen, heads, headDim}` | 初始化 KV 缓存（长度 0 的空张量），供自回归推理逐 token 增量解码 | len=0；k/v 形状 [B,H,0,D]；append 后 len 单调增且不超过 maxLen |
| `D8-058` | `dl.attn.kvCacheAppend` | `(object cache, Tensor kNew[B,H,Tn,D], Tensor vNew[B,H,Tn,D]) -> object` | 向 KV 缓存尾部追加 Tn 个新 token 的键与值，返回新缓存（纯函数，不改入参） | 新 len = 旧 len + Tn；累积结果与一次性全序列 K/V 完全相同；超过 maxLen 报错 |
| `D8-059` | `dl.attn.flashTiling` | `(Tensor Q[B,Tq,D], Tensor K[B,Tk,D], Tensor V[B,Tk,Dv], number blockRow, number blockCol) -> {out, lse}` | FlashAttention 式分块：按 (blockRow, blockCol) 分块并用 online-softmax 增量更新，避免物化 T×T 注意力矩阵 | 输出与 dl.attn.scaledDot 数值一致（约 1e-12）；lse = logsumexp(scores)；不物化完整注意力矩阵 |
| `D8-060` | `dl.attn.entropy` | `(Tensor attn[...,T]) -> Tensor[...,1]` | 注意力分布香农熵 H = −Σ p·log(p+eps)，衡量注意力集中程度（0=one-hot，log T=均匀） | 0 ≤ H ≤ log T；one-hot 时 H=0；均匀分布时 H=log T |
| `D8-061` | `dl.attn.headSplit` | `(Tensor x[B,T,C], number heads) -> Tensor[B,H,T,D]` | 把模型维 C 拆成 heads×headDim：out[b,h,t,d] = x[b,t,h·D+d] | heads × headDim = 模型维 C；元素总数守恒；与 _headMerge 互为逆操作 |
| `D8-062` | `dl.seq.rnnCell` | `(Tensor x[N,Di], Tensor h[N,Dh], Tensor Wx[Di,Dh], Tensor Wh[Dh,Dh], Tensor b[Dh], string act='tanh') -> Tensor[N,Dh]` | Elman RNN 单元：h′ = act(x·Wx + h·Wh + b) | 输出形状 [N,Dh]；W=0,b=0 时 h′=act(0)；梯度沿时间连乘（易衰减/爆炸） |
| `D8-063` | `dl.seq.gruCell` | `(Tensor x[N,Di], Tensor h[N,H], Tensor Wx[Di,3H], Tensor Wh[H,3H], Tensor bx[3H], Tensor bh[3H]) -> Tensor[N,H]` | GRU 单元：z=σ(更新门)、r=σ(重置门)、n=tanh(候选)，h′=(1−z)·h + z·n | z=0 时 h′=h（无更新）；z=1 时 h′=n（完全替换）；h′ 落在 h 与 n 的凸组合内 |
| `D8-064` | `dl.seq.lstmCell` | `(Tensor x[N,Di], Tensor h[N,H], Tensor c[N,H], Tensor W[Di,4H], Tensor U[H,4H], Tensor b[4H]) -> {h: Tensor[N,H], c: Tensor[N,H]}` | LSTM 单元：i/f/o 为 sigmoid 门，g 为 tanh 候选；c′ = f⊙c + i⊙g，h′ = o⊙tanh(c′) | f=1,i=0 时 c 不变（恒定误差流）；c′ 与 h′ 形状 [N,H]；\|h′\| ≤ 1 |
| `D8-065` | `dl.seq.rnnForward` | `(Tensor x[B,T,Di], Tensor h0[B,H], Tensor Wx[Di,H], Tensor Wh[H,H], Tensor b[H], string act='tanh') -> {out: Tensor[B,T,H], hT: Tensor[B,H]}` | 沿时间维展开 RNN：逐时间步调用 rnnCell，返回全部隐状态与最终隐状态 | out[:,t,:] = 第 t 步隐状态；hT = out[:,T−1,:]；T=1 时退化为单元计算 |
| `D8-066` | `dl.seq.gruForward` | `(Tensor x[B,T,Di], Tensor h0[B,H], Tensor Wx[Di,3H], Tensor Wh[H,3H], Tensor bx[3H], Tensor bh[3H]) -> {out, hT}` | 沿时间维展开 GRU，返回全部隐状态序列与末状态 | out[:,t,:] = 第 t 步隐状态；hT = out[:,T−1,:]；隐状态有界（\|h\| ≤ max(\|h0\|,1)） |
| `D8-067` | `dl.seq.lstmForward` | `(Tensor x[B,T,Di], Tensor h0[B,H], Tensor c0[B,H], Tensor W[Di,4H], Tensor U[H,4H], Tensor b[4H]) -> {out, hT, cT}` | 沿时间维展开 LSTM，返回隐状态序列、末隐状态与末细胞状态 | out[:,t,:] = 第 t 步隐状态；\|h\| ≤ 1；f=1,i=0 时 c 沿时间恒定 |
| `D8-068` | `dl.seq.bidirectional` | `(Tensor xf[B,T,H], Tensor xb[B,T,H]) -> Tensor[B,T,2H]` | 双向拼接：把正向与反向（通常已各自翻转时间）的输出沿特征维拼接 | 输出最后一维 = 2H；前半段等于 xf，后半段等于 xb；元素总数守恒 |
| `D8-069` | `dl.seq.scan` | `(Tensor x[B,T,D], Function f, Tensor h0, number reverse=0) -> {out: Tensor[B,T,Dh], hT: Tensor}` | 通用 carry 扫描（JAX lax.scan 语义）：h_{t+1} = f(h_t, x_t)，reverse=1 时逆序扫描 | 输出时间步数 = T；reverse=1 时输出时间序反转；f 为恒等时 out 恒为 h0 |
| `D8-070` | `dl.loss.mse` | `(Tensor pred, Tensor target, string reduction='mean') -> number` | 均方误差 = mean((pred−target)²)（或 sum） | mse ≥ 0；pred=target 时为 0；对残差缩放 c 倍则值放大 c² 倍 |
| `D8-071` | `dl.loss.mae` | `(Tensor pred, Tensor target, string reduction='mean') -> number` | 平均绝对误差 L1 = mean(\|pred−target\|)，对离群点比 MSE 更稳健 | mae ≥ 0；pred=target 时为 0；满足三角不等式 |
| `D8-072` | `dl.loss.huber` | `(Tensor pred, Tensor target, number delta=1, string reduction='mean') -> number` | Huber 损失：\|r\|≤delta 时为 0.5r²，否则 delta(\|r\|−0.5·delta)，兼顾 L1 稳健与 L2 光滑 | huber ≥ 0 且连续可微；delta→∞ 趋于 0.5·MSE·(元素数)；\|r\|≤delta 时等于 0.5r² |
| `D8-073` | `dl.loss.bce` | `(Tensor pred∈(0,1), Tensor target∈{0,1}, string reduction='mean', number eps=1e-12) -> number` | 二元交叉熵 = −[t·log(p) + (1−t)·log(1−p)]，输入须为概率 | bce ≥ 0（p,t∈[0,1]）；p=t 且 t∈{0,1} 时为 0；关于 p=t=0.5 对称 |
| `D8-074` | `dl.loss.bceWithLogits` | `(Tensor logits, Tensor target∈{0,1}, string reduction='mean') -> number` | 数值稳定的 BCE-with-logits：直接用 logits 计算，避免先 sigmoid 造成的下溢 | 与 sigmoid(logits) 后再算 bce 一致（误差 <1e-12）；logits=0 时每元素 = ln2 |
| `D8-075` | `dl.loss.ce` | `(Tensor prob[N,C], Tensor target[N], string reduction='mean', number eps=1e-12) -> number` | 交叉熵（输入为概率分布）= −log(p_{correct}) | ce = −log(p_correct)；p_correct=1 时为 0；p_correct→0 时 →∞ |
| `D8-076` | `dl.loss.ceWithLogits` | `(Tensor logits[N,C], Tensor target[N], string reduction='mean') -> number` | logits 上的交叉熵：内部先做稳定 log-softmax 再取负对数似然 | 等于 ce(softmax(logits), target)；平移不变：logits 加常数结果不变 |
| `D8-077` | `dl.loss.nll` | `(Tensor logProb[N,C], Tensor target[N], string reduction='mean') -> number` | 负对数似然 = −logProb[n, target[n]]（输入须为对数概率） | nll = −log p_correct；与 ceWithLogits(logits)=nll(logSoftmax(logits)) 一致 |
| `D8-078` | `dl.loss.klDiv` | `(Tensor logQ[N,C]（近似分布的对数概率）, Tensor P[N,C]（目标分布概率）, string reduction='mean') -> number` | KL 散度 Σ P·(log P − log Q)，按 batch 维归约（PyTorch 的 batchmean 语义） | KL ≥ 0（Gibbs 不等式）；P=Q 时为 0；非对称：KL(P\|\|Q) ≠ KL(Q\|\|P) |
| `D8-079` | `dl.loss.focal` | `(Tensor logits[N,C], Tensor target[N], number gamma=2, number alpha=0.25, string reduction='mean') -> number` | Focal Loss = −alpha·(1−p_t)^gamma·log(p_t)，抑制易分样本梯度、聚焦难样本 | gamma=0,alpha=1 时退化为普通交叉熵；p_t=1 时为 0；gamma 越大对易样本压制越强 |
| `D8-080` | `dl.loss.dice` | `(Tensor pred∈[0,1], Tensor target∈{0,1}, number eps=1e-6) -> number` | Dice 损失 = 1 − 2\|P∩T\| / (\|P\|+\|T\|)，类别极度不均衡的分割任务常用 | dice ∈ [0,1]；完全重合时为 0；完全不重合时为 1；与样本顺序无关 |
| `D8-081` | `dl.loss.ntxent` | `(Tensor z1[N,D], Tensor z2[N,D], number temperature=0.5) -> number` | SimCLR 的 NT-Xent 对比损失：2N 个视图两两相似度，正例为同一样本的另一视图，分母排除自身 | loss ≥ 0；两视图完全一致且互为正交向量时 loss = log(2N−1)；温度越低越聚焦难负例 |
| `D8-082` | `dl.loss.triplet` | `(Tensor anchor[N,D], Tensor pos[N,D], Tensor neg[N,D], number margin=1) -> number` | 三元组损失 = mean(max(0, \|\|a−p\|\|² − \|\|a−n\|\|² + margin))，拉近正例、推远负例 | triplet ≥ 0；满足 margin 约束（d(a,n)² ≥ d(a,p)²+margin）时为 0；a=p 且 margin=0 时趋于 0 |
| `D8-083` | `dl.loss.labelSmoothCE` | `(Tensor logits[N,C], Tensor target[N], number eps=0.1, string reduction='mean') -> number` | 标签平滑交叉熵 = (1−eps)·CE(one-hot) + eps·CE(均匀分布)，抑制过度自信 | eps=0 退化为 ceWithLogits；eps=1 时 = KL(U\|\|p) + log C；损失下限为 eps·log C 量级 |
| `D8-084` | `dl.init.zeros` | `(number[] shape) -> Tensor` | 全零初始化（常用于偏置项与门控偏置的初始设置） | 所有元素恰为 0；形状与入参一致；均值=方差=0 |
| `D8-085` | `dl.init.ones` | `(number[] shape) -> Tensor` | 全一初始化（LayerNorm 的 gamma、缩放因子的常用初值） | 所有元素恰为 1；均值=1，方差=0 |
| `D8-086` | `dl.init.constant` | `(number[] shape, number value) -> Tensor` | 常数初始化（如把遗忘门偏置设为 1 以缓解早期遗忘） | 所有元素等于 value；均值=value，方差=0 |
| `D8-087` | `dl.init.uniform` | `(number[] shape, number a=-0.05, number b=0.05, number seed=20261014) -> Tensor` | 均匀分布 U(a,b) 初始化 | 元素全部落在 [a,b]；均值≈(a+b)/2，方差≈(b−a)²/12；同种子完全可复现 |
| `D8-088` | `dl.init.normal` | `(number[] shape, number mean=0, number std=0.02, number seed=20261014) -> Tensor` | 正态 N(mean, std²) 初始化（Transformer 嵌入/投影层常用 std=0.02） | 样本均值≈mean、方差≈std²（大样本下 10% 容差内）；同种子可复现 |
| `D8-089` | `dl.init.truncatedNormal` | `(number[] shape, number mean=0, number std=0.02, number seed=20261014) -> Tensor` | 截断正态（\|z−mean\| ≤ 2·std）：拒绝采样最多 8 次，超出则夹到边界，避免极端初值 | 所有元素落在 [mean−2std, mean+2std]；样本方差略小于 std²；无异常值 |
| `D8-090` | `dl.init.xavierUniform` | `(number[] shape, number gain=1, number seed=20261014) -> Tensor` | Xavier/Glorot 均匀初始化 U(−b,b)，b = gain·√(6/(fanIn+fanOut))，保持前后向信号方差一致 | 方差 = gain²·2/(fanIn+fanOut)；均值 = 0；元素落在 [−b,b] |
| `D8-091` | `dl.init.xavierNormal` | `(number[] shape, number gain=1, number seed=20261014) -> Tensor` | Xavier/Glorot 正态初始化 N(0, std²)，std = gain·√(2/(fanIn+fanOut)) | 方差 ≈ gain²·2/(fanIn+fanOut)；均值 ≈ 0；与 xavierUniform 同方差不同分布 |
| `D8-092` | `dl.init.heUniform` | `(number[] shape, number gain=√2, number seed=20261014) -> Tensor` | He 均匀初始化 U(−b,b)，b = gain·√(3/fanIn)，专为 ReLU 族设计（前向方差守恒） | 方差 = gain²/fanIn（gain=√2 时 = 2/fanIn）；均值 = 0；元素落在 [−b,b] |
| `D8-093` | `dl.init.heNormal` | `(number[] shape, number gain=√2, number seed=20261014) -> Tensor` | He 正态初始化 N(0, std²)，std = gain/√fanIn（ReLU/LeakyReLU 网络默认） | 方差 ≈ gain²/fanIn = 2/fanIn（gain=√2）；均值 ≈ 0；与 heUniform 同方差不同分布 |
| `D8-094` | `dl.model.mlpInit` | `(number[] sizes, number seed=20261014, string scheme='he') -> {sizes, W: Tensor[], b: Tensor[], count: number}` | 按 sizes 初始化全连接 MLP 参数（W_i 形状 [sizes_i, sizes_{i+1}]，偏置置零） | W 长度 = sizes 长度 −1；W_i 形状 [sizes_i, sizes_{i+1}]；偏置全零；count = Σ 参数量 |
| `D8-095` | `dl.model.mlpForward` | `(Tensor x[N,D], Tensor[] W, Tensor[] b, string hiddenAct='relu') -> Tensor[N,Dk]` | 多层感知机前向：逐层 dense 并在除最后一层外施加激活；内部复用域4 的 ad.dense，因此可被 gradcheck 求导 | 输出形状 [N, D_last]；层数 = W.length；hiddenAct="none" 时退化为纯线性映射 |
| `D8-096` | `dl.model.transformerBlock` | `(Tensor x[B,T,C], object params, number heads, Tensor mask?, number eps=1e-5) -> Tensor[B,T,C]` | Pre-LN Transformer 块：x ← x + MHA(LN?)… 采用 Post-LN 形式：a=MHA(x)；h=LN(x+a)；f=FFN(h)；out=LN(h+f) | 输出形状恒为 [B,T,C]；残差连接保证恒等通路；有限输入产生有限输出 |
| `D8-097` | `dl.model.swigluFFN` | `(Tensor x[B,T,C], Tensor Wg[C,H], Tensor bg[H], Tensor Wu[C,H], Tensor bu[H], Tensor Wd[H,C], Tensor bd[C]) -> Tensor[B,T,C]` | SwiGLU 前馈网络：out = (SiLU(x·Wg+bg) ⊙ (x·Wu+bu))·Wd + bd，LLaMA/PaLM 的默认 FFN | 输出形状与输入一致 [B,T,C]；门控值 →0 时输出趋于 0；参数量为标准 FFN 的 1.5 倍 |
| `D8-098` | `dl.model.moeRouter` | `(Tensor x[N,C], Tensor Wg[C,E], number topK, number seed=20261014) -> {probs: Tensor[N,E], topIdx: Tensor[N,topK]}` | 稀疏 MoE 路由：logits = x·Wg，取 topK 个专家并在其上做 softmax，其余置零（Shazeer 2017 的 noisy top-k 简化版） | 每行恰有 topK 个非零且和为 1；topK=1 时退化为 argmax 的 one-hot；topK=E 时为完整 softmax |
| `D8-099` | `dl.model.residual` | `(Tensor x, Tensor branch, number scale=1) -> Tensor` | 残差连接：out = x + scale·branch，为深层网络提供恒等通路（缓解梯度消失） | branch=0 时为恒等；scale=1 时 out = x + branch；形状与 x 一致 |
| `D8-100` | `dl.util.paramCount` | `(Tensor[] tensors) -> number` | 统计参数张量数组的总元素数（模型规模/显存估算） | count = Σ size(t_i)；空数组为 0；与 flattenParams 的 flat 长度一致 |
| `D8-101` | `dl.util.flattenParams` | `(Tensor[] tensors) -> {flat: Tensor[P], shapes: number[][]}` | 把参数张量数组按行主序拉平成一维向量，并记录各张量形状以便还原（供优化器/线搜索使用） | flat 长度 = Σ size；shapes 与输入一一对应；按 shapes 可无损切分还原 |
| `D8-102` | `dl.util.applyGrads` | `(Tensor[] params, Tensor[] grads, number lr) -> Tensor[]` | 最朴素的 SGD 更新：p ← p − lr·g（逐张量对应，形状必须一致） | lr=0 时参数不变；梯度为 0 时参数不变；输出形状与输入逐个一致 |
| `D8-103` | `dl.util.quantizeInt8` | `(Tensor x) -> object {q: Int8Array, scale: number, dequant: Tensor}` | INT8 对称量化与反量化（训练后静态量化，逐张量粒度） | \|dequant − x\| ≤ scale/2；q ∈ [−127,127]；scale ≥ 0（全零张量 scale=1 避免除零） |

## D9 · TRAIN 训练与评估

数据管线、训练循环、交叉验证、可复现性。共 103 条。

| id | 指令 | 签名 | 说明 | 不变量 |
|---|---|---|---|---|
| `D9-001` | `train.data.batchIterator` | `(dataset, batchSize:number, opts?:{dropLast}) -> batch[]` | 按 batchSize 顺序分批；批为 {X, y, index}，index 为原始行号；最后不满一批默认保留，opts.dropLast=true 时丢弃。 | 各批 index 并集 = [0,n)，批间不相交；默认末批大小 = n mod batchSize（非零时） |
| `D9-002` | `train.data.shuffledBatches` | `(dataset, batchSize:number, seed:number) -> batch[]` | 用 createRng(seed) 对行做 Fisher-Yates 打乱后分批；同种子产生完全相同的批序列。 | 同种子两次调用批序列逐位相同；批 index 并集 = [0,n) 且不相交 |
| `D9-003` | `train.data.stratifiedBatches` | `(dataset, batchSize:number, seed:number) -> batch[]` | 按类别分组后各自打乱，再尽可能均匀地摊到 ceil(n/batchSize) 个批中，使每批类别比例≈总体比例。 | 批 index 并集 = [0,n) 且不相交；每批类别比例与总体偏差有界（各类摊差 ≤1 个元素） |
| `D9-004` | `train.data.windowIterator` | `(series:Tensor[T]\|Tensor[T,d], window:number, stride?:number) -> {X,y,index}[]` | 把一维/二维序列切成滑动窗口样本：X 为长度 window 的片段，y 为窗口之后的 1 步；stride 控制步长。 | 窗口数 = floor((T-window)/stride)；所有窗口起点互不相同且 ∈ [0, T-window) |
| `D9-005` | `train.data.dataset` | `(X:Tensor[n,d]\|nested, y:Tensor[n]\|array) -> dataset` | 构造标准数据集 {X, y}：X 为 2D，y 为 1D 且行数一致；接受嵌套数组入参。 | ds.X.shape[0] === ds.y.shape[0]；X.ndim === 2，y.ndim === 1 |
| `D9-006` | `train.data.trainTestSplit` | `(dataset, testFrac:number, seed:number) -> {train, test}` | 种子化打乱后按比例切分；测试集大小 = round(n*testFrac)，训练集非空。 | train 与 test 的 index 不相交，并集 = [0,n)；同种子划分可复现 |
| `D9-007` | `train.data.kFoldSplit` | `(dataset, k:number, seed:number) -> fold[]` | 打乱后均分 k 折，每折 {fold, valIdx, trainIdx, train, val}；折大小差 ≤1。 | 各折 valIdx 两两不相交；∪valIdx = [0,n)；折大小 ∈ {floor(n/k), ceil(n/k)} |
| `D9-008` | `train.data.stratifiedKFoldSplit` | `(dataset, k:number, seed:number) -> fold[]` | 在每个类别内部打乱后按位置模 k 轮流分配到各折，保持各折类别比例≈总体比例。 | 各折 valIdx 不相交且并集 = [0,n)；每折中各类别计数与总体比例偏差 ≤1 个元素 |
| `D9-009` | `train.data.standardizerFit` | `(dataset) -> {kind, mean:Tensor[d], std:Tensor[d], n}` | 拟合逐列均值与总体标准差；常数列 std 置 1e-12 下限，避免 apply 除零。 | apply 后每列均值 ≈ 0、标准差 ≈ 1（常数列除外，结果为 0） |
| `D9-010` | `train.data.standardizerApply` | `(dataset, scaler) -> dataset` | 用已拟合的 {mean, std} 逐列标准化；保留数据集其它字段（如 index）。 | out.X[i,j] = (X[i,j]-mean[j])/std[j]；y 与其它字段不变 |
| `D9-011` | `train.data.minmaxFit` | `(dataset) -> {kind, min:Tensor[d], max:Tensor[d]}` | 拟合逐列最小/最大值；极差下限 1e-12（常数列缩放为 0）。 | apply 后每列值域 ∈ [0,1] |
| `D9-012` | `train.data.minmaxApply` | `(dataset, scaler) -> dataset` | 用已拟合的 {min, max} 逐列缩放到 [0,1]；极差为 0 的列输出 0。 | out.X[i,j] = (X[i,j]-min[j])/(max[j]-min[j])；常数列输出 0 |
| `D9-013` | `train.data.onehotEncode` | `(y:Tensor[n]\|array, numClasses:number) -> Tensor[n,C]` | 把整数标签 [0,C) 编成 one-hot 矩阵；numClasses 缺省由最大值推断。 | 每行恰一个 1，位置 = 标签值；行和 = 1 |
| `D9-014` | `train.data.labelEncode` | `(labels:Tensor[n]\|array) -> {classes, y:Tensor[n]}` | 把任意离散标签（字符串/数字）映射为排序后的类别表与整数索引。 | classes 为去重排序表；y[i] = classes 中 labels[i] 的位置；labelDecode(y, classes) 可还原 |
| `D9-015` | `train.data.labelDecode` | `(y:Tensor[n]\|array, classes:array) -> array` | labelEncode 的逆操作：整数索引还原为原始标签值。 | 对 labelEncode 的输出，labelDecode(y, classes) 逐项等于原始 labels |
| `D9-016` | `train.data.pipelineCompose` | `(steps:fn[]) -> (dataset) -> dataset` | 把若干数据集变换函数按顺序组合成一个函数；每步均为 ds -> ds 的纯函数。 | compose(f,g)(ds) === g(f(ds))；空 steps 返回恒等函数 |
| `D9-017` | `train.data.pipelineFitTransform` | `(dataset, fitters:(ds -> (ds -> ds))[]) -> {data, apply, fitted}` | 顺序拟合各步（每步在训练集上 fit 得到 apply 函数），返回变换后的训练数据与可复用管线。 | apply(newDs) 按相同拟合参数变换新数据；fitted.length === fitters.length |
| `D9-018` | `train.data.clipOutliers` | `(dataset, k?:number) -> dataset` | 按逐列均值 ± k 倍标准差截断离群值（默认 k=3，3σ 准则）。 | 输出每列值域 ⊆ [mean-k·std, mean+k·std]；y 不变 |
| `D9-019` | `train.data.winsorize` | `(dataset, lo?:number, hi?:number) -> dataset` | 按逐列分位数截断两端（默认 1%/99% 分位缩尾）。 | 0 ≤ lo < hi ≤ 1；输出每列值域 ⊆ [Q(lo), Q(hi)] |
| `D9-020` | `train.data.logTransform` | `(dataset, offset?:number) -> dataset` | 逐元素 log(x+offset)（默认 offset=1，即 log1p）；要求 min(x)+offset > 0。 | 输出元素 = log(X[i,j]+offset)；严格单调保持原大小序 |
| `D9-021` | `train.data.polyExpand` | `(dataset, degree?:number) -> dataset` | 展开为总次数 1..degree 的全部单项式（含交叉项），阶数升序排列；degree=1 为恒等。 | 输出列数 = Σ_{m=1..degree} C(m+d-1, d-1)；首 d 列为原始列（degree=1 时恰为原始） |
| `D9-022` | `train.data.featureScale` | `(dataset, scales:number\|number[]) -> dataset` | 逐列乘以缩放因子（标量作用于所有列，数组按列对应）；用于量纲对齐。 | out.X[i,j] = X[i,j] * (数组 ? scales[j] : scales) |
| `D9-023` | `train.loop.run` | `(model, data:{train, val?}, opts?) -> {history, params, ema, best, stopped, epochsRun}` | 通用训练循环：逐 epoch 分批（可选打乱/EMA/梯度裁剪/lr 调度/早停/回调），记录 history 与最优监控点。 opts: {epochs, lr, batchSize, update, seed, shuffle, clipNorm, ema:{decay}, metric, earlyStopping:{patience,mode,minDelta}, lrSchedule, callbacks:{onEpochStart,onEpochEnd,onBatchEnd}, timeBudgetMs} | history.length === epochsRun；同种子同配置两次运行 params 逐位相同；callback 返回 false 或 ctx.stop 即停 |
| `D9-024` | `train.loop.epoch` | `(model, params, dataset, opts?) -> {params, loss, batches}` | 在给定参数上跑一个 epoch：按 batchSize 分批（可选打乱），逐批梯度步，返回更新后参数与平均损失。 | 不修改入参 params（返回新结构）；loss 为按样本数加权的批平均 |
| `D9-025` | `train.loop.evaluate` | `(model, dataset, metric?) -> number` | 用 model.predict(model.params, X) 预测并按 metric 打分；metric 支持函数或 'accuracy'\|'mse'\|'mae'，默认整数标签→准确率、连续→MSE。 | 不修改 model.params；metric 为 null 时按标签类型选择默认指标 |
| `D9-026` | `train.stop.earlyStopping` | `(values:array\|Tensor, opts?:{patience, mode, minDelta}) -> {shouldStop, bestEpoch, bestValue, stale, trace}` | 在已产生的监控序列上评估早停：超过 patience 轮无改善（幅度 > minDelta）即应停止；返回最佳轮次与逐步 trace。 | bestEpoch/bestValue 为 mode 方向上的最优点；stale = 自最佳点以来的未改善轮数 |
| `D9-027` | `train.stop.maxEpochs` | `(epoch:number, max:number) -> boolean` | 轮数上限规则：epoch >= max 时返回 true（达到预算即停）。 | epoch < max 返回 false；epoch >= max 返回 true |
| `D9-028` | `train.stop.timeBudget` | `(startedAtMs:number, budgetMs:number, nowMs?:number) -> {elapsed, remaining, exceeded}` | 时间预算规则：返回已用/剩余毫秒与是否超预算；nowMs 缺省取当前时间（测试可显式传入以保证可复现）。 | elapsed = now - startedAt；exceeded = elapsed >= budgetMs；remaining = budgetMs - elapsed |
| `D9-029` | `train.ckpt.save` | `(params, meta?) -> {schema, params, meta}` | 把参数结构深拷贝为可序列化快照；不含时间戳，保证同参数同快照。 | 快照与 params 逐位相等且互不影响（写时拷贝） |
| `D9-030` | `train.ckpt.load` | `(ckpt) -> params` | 从快照深拷贝恢复参数；返回独立副本，不与快照共享内存。 | load(save(p)) 与 p 逐位相等；修改返回值不影响快照 |
| `D9-031` | `train.ckpt.bestTracker` | `(mode?:'min'\|'max') -> {update(value, epoch?, payload?), best, bestEpoch, updates, snapshot()}` | 有状态最优跟踪器：每次 update 返回是否刷新纪录；可按需附带 payload（如参数快照）。 | mode=min 时 best 单调不增；snapshot() 与内部状态一致 |
| `D9-032` | `train.hist.log` | `(history:array, entry:object) -> array` | 向训练历史追加一条记录（缺省 epoch 时取 length+1）；纯函数，不修改入参数组。 | 返回长度 = 入参长度 + 1；入参 history 不被修改 |
| `D9-033` | `train.hist.toMarkdown` | `(history:array, cols?:string[]) -> string` | 把历史记录渲染成 Markdown 表格；cols 缺省取所有记录键的有序并集。 | 输出首行为表头 \| ... \|，第二行为分隔行；行数 = history.length |
| `D9-034` | `train.lr.hook` | `(epoch:number, baseLr:number, schedule:fn\|{kind,...}) -> number` | 学习率调度：支持函数(epoch)->倍率，或 {kind:'step'\| 'exp'\| 'cosine'\| 'linear', ...}；epoch 从 1 计数。 | epoch=1 时倍率为 1（step/exp/cosine/linear 均如此）；返回 baseLr*倍率 |
| `D9-035` | `train.ema.update` | `(state:{decay, params}, params, decay?:number) -> state` | 指数滑动平均更新：ema ← decay·ema + (1-decay)·params；decay 缺省用 state.decay。 | 新 state 与入参互不影响；decay=0 时 state.params 等于当前 params |
| `D9-036` | `train.ema.apply` | `(state) -> params` | 返回 EMA 参数的深拷贝（用于评估/推理，不影响 state）。 | apply(state) 与 state.params 逐位相等；修改返回值不影响 state |
| `D9-037` | `train.mixup` | `(dataset, alpha:number, rng) -> {X, y, lam, perm}` | mixup：lam ~ Beta(alpha, alpha)，按随机置换线性插值样本与标签（回归值/one-hot 均适用）；alpha≤0 时 lam=1（不混合）。 | X'[i] = lam·X[i] + (1-lam)·X[perm[i]]；y' 同理；lam ∈ [0,1] |
| `D9-038` | `train.labelSmoothing` | `(targets:Tensor[n]\|Tensor[n,C], numClasses:number, eps?:number) -> Tensor[n,C]` | 标签平滑（Szegedy 2016）：硬标签 → (1-eps)·onehot + eps/C；已 soft 的 2D 目标 → 向均匀分布收缩 eps。 | 输出每行和为 1；eps=0 时对硬标签为纯 one-hot |
| `D9-039` | `train.aug.gaussianNoise` | `(X:Tensor[n,d], sigma:number, rng) -> Tensor[n,d]` | 逐元素加性高斯噪声 X + sigma·N(0,1)（等价于 Tikhonov 正则化视角的噪声训练）。 | sigma=0 时输出与输入逐位相同；同种子可复现 |
| `D9-040` | `train.aug.featureDropout` | `(X:Tensor[n,d], p:number, rng) -> Tensor[n,d]` | 逐元素输入 dropout：以概率 p 置零，保留元素乘 1/(1-p)（inverted dropout，保持期望不变）。 | p=0 时输出与输入逐位相同；零元素占比 ≈ p；同种子可复现 |
| `D9-041` | `train.aug.scaleJitter` | `(X:Tensor[n,d], sigma:number, rng) -> Tensor[n,d]` | 每行乘随机尺度因子 1 + sigma·N(0,1)（行级幅度抖动）；sigma=0 为恒等。 | sigma=0 时输出与输入逐位相同；同种子可复现 |
| `D9-042` | `train.accum.step` | `(state:{sum, count}, grad) -> state` | 把一批梯度累加到累积器（sum 与 grad 同构相加，count+1）；state 可用 accum.reset() 或 accum.reset(grad) 初始化。 | 返回新 state；sum 与 grad 同构；count 每步 +1 |
| `D9-043` | `train.accum.reset` | `(grad?) -> {sum, count}` | 重置累积器：无参 → {sum:null, count:0}；给定 grad → 预分配同构全零 sum。 | reset() 后 count=0；reset(grad) 后 sum 与 grad 同构且全零 |
| `D9-044` | `train.loop.withClipping` | `(update:fn, maxNorm:number) -> (params, grad, lr) -> params` | 包装任意 update：先把梯度按全局 L2 范数裁剪到 maxNorm，再调用原 update。 | ‖grad‖ ≤ maxNorm 时原样传递；否则缩放 maxNorm/‖grad‖ 倍 |
| `D9-045` | `train.loop.shuffleEachEpoch` | `(seed:number) -> (n:number, epoch:number) -> number[]` | 返回确定性打乱函数：同一 (seed, n, epoch) 产生同一排列，不同 epoch 产生不同排列。 | perm(n, epoch) 是 [0,n) 的排列；perm(n, e) 对同一入参可复现 |
| `D9-046` | `train.loop.batchStep` | `(model, params, batch:{X,y}, lr:number, update?:fn) -> {loss, grad, gradNorm, params}` | 对一个批次执行一次梯度步：调用 model.lossGrad，经 update(params, grad, lr) 更新；update 缺省为 SGD。 | 不修改入参 params；返回 {loss, grad, gradNorm, params(新)} |
| `D9-047` | `train.ema.init` | `(params, decay?:number) -> {decay, params}` | 用初始参数深拷贝创建 EMA 状态；decay ∈ [0,1)（默认 0.99）。 | state.params 与入参逐位相等但相互独立 |
| `D9-048` | `train.stop.criteria` | `(rules:{name, stop}[], mode?:'all'\|'any') -> {stop, triggered, mode}` | 组合多个停止规则：mode=all 时全部触发才停（默认），mode=any 时任一触发即停；返回触发的规则名。 | rules 为空数组时 stop=false；triggered ⊆ rules 中 stop=true 的项 |
| `D9-049` | `train.eval.crossValidate` | `(dataset, k:number, fitPredict:fn, opts?:{seed, metric}) -> {k, scores, mean, std, valIdx}` | k 折交叉验证：每折用 fitPredict(train, val) 产出预测并按 metric 打分；返回 k 个分数及均值±总体标准差。 | scores.length === k；mean/std 为 population 统计；valIdx 各折不相交且并集 = [0,n) |
| `D9-050` | `train.eval.repeatedCV` | `(dataset, k:number, repeats:number, fitPredict:fn, opts?:{seed, metric}) -> {repeats, meanOfMeans, stdOfMeans, allScores}` | 以不同种子重复 k 折 CV repeats 次，聚合各次 mean/std 与全部分数，降低单次划分方差。 | repeats 数组长度 = repeats；allScores.length = k*repeats；meanOfMeans = 各次 mean 的均值 |
| `D9-051` | `train.eval.learningCurve` | `(dataset, sizes:number[], fitPredict:fn, opts?:{seed, metric, valFrac}) -> {n, trainScore, valScore}[]` | 固定验证集下，对递增训练子集规模分别评估训练分与验证分，判断欠拟合/过拟合趋势。 | sizes 中每个 m ∈ [1, pool 大小]；返回长度 = sizes.length；各次子集为同一打乱序列的前缀（可比） |
| `D9-052` | `train.eval.validationCurve` | `(dataset, values:array, makeFitPredict:(v)->fn, opts?:{k, seed, metric}) -> {value, mean, std, scores}[]` | 对每个候选超参数值做 k 折 CV，返回该值的 mean±std 与各折分数，用于选模。 | 返回长度 = values.length；每项 scores.length === k；同一 (值, 种子) 可复现 |
| `D9-053` | `train.eval.bootstrapCI` | `(values:array\|Tensor, opts?:{B, alpha, seed, statistic}) -> {point, lo, hi, mean, std, B, alpha}` | 对有放回重采样 B 次计算统计量（默认均值）的分位数置信区间；固定种子完全可复现。 | 同种子两次调用结果逐位相同；point = statistic(原始样本)；默认区间为 [alpha/2, 1-alpha/2] 分位 |
| `D9-054` | `train.eval.pairedCompare` | `(scoresA:array\|Tensor, scoresB:array\|Tensor, opts?) -> {meanDiff, stdDiff, se, t, df, ciLo, ciHi, pNormal, wins, losses, ties}` | 配对差分析：meanDiff=mean(A-B)，给出配对 t 统计量、正态近似 p 值与胜/负/平计数；CI 用正态分位近似。 | meanDiff = mean(A-B)；wins+losses+ties = n；t = meanDiff/se，se = stdDiff/sqrt(n)（ddof=1） |
| `D9-055` | `train.eval.classificationReport` | `(yTrue, yPred, classes?) -> {classes, accuracy, support, perClass, macro, weighted}` | 分类报告：逐类 precision/recall/f1/support，加总 accuracy 与宏平均、按 support 加权平均。 | accuracy = Σtp/n；macro = 逐类指标等权平均；weighted = 按 support 加权平均 |
| `D9-056` | `train.eval.confusionReport` | `(yTrue, yPred, classes?) -> {labels, matrix, rowSums, colSums, total, diagonal}` | 混淆矩阵：matrix[i][j] = 真实为 labels[i] 且预测为 labels[j] 的计数；附行列和与对角线（判对数）。 | 行和 = 各类真实计数；列和 = 各类预测计数；Σmatrix = n；diagonal = 判对数 |
| `D9-057` | `train.eval.rocAnalysis` | `(yTrue:0/1, scores:number[], thresholds?) -> {thresholds, tpr, fpr, points, auc, youdenJ, optimalThreshold}` | ROC 曲线：按分数降序扫阈值得 (fpr, tpr) 点列；AUC 用 Mann-Whitney（含并列中秩）；另给 Youden J 最优阈值。 | auc ∈ [0,1]；随机排序 AUC=0.5；完美排序 AUC=1；points 按阈值降序 |
| `D9-058` | `train.eval.prAnalysis` | `(yTrue:0/1, scores:number[]) -> {thresholds, precision, recall, points, averagePrecision, bestF1}` | 精确率-召回率曲线：按分数降序扫阈值；AP 用 recall 增量插值 Σ(Rₖ-Rₖ₋₁)·Pₖ；另给最优 F1 阈值。 | recall 随阈值降低单调不减；perfect 排序 AP=1；bestF1 = maxₜ 2PR/(P+R) |
| `D9-059` | `train.eval.thresholdSweep` | `(yTrue:0/1, scores:number[], thresholds:number[]) -> {threshold, tp, fp, tn, fn, precision, recall, f1, accuracy}[]` | 对给定阈值集合逐项计算二分类指标（预测正类规则：score ≥ threshold）。 | tp+fp+tn+fn = n；precision/recall/f1 与 classificationReport 中正类口径一致 |
| `D9-060` | `train.eval.macroAvg` | `(report\|perClass:array) -> {precision, recall, f1, k}` | 宏平均：对逐类 precision/recall/f1 等权平均；接受 classificationReport 或 perClass 数组。 | 输出 ∈ [0,1]；k = 类别数；与 classificationReport.macro 一致 |
| `D9-061` | `train.eval.microAvg` | `(yTrue, yPred) -> {precision, recall, f1, accuracy}` | 微平均：单标签多分类下等价于准确率（汇总所有类的 tp/fp/fn 后计算）。 | precision = recall = f1 = accuracy = Σtp/n |
| `D9-062` | `train.eval.cohenKappa` | `(yTrue, yPred, k?:number\|array) -> number` | Cohen κ：扣除随机一致后的吻合度 (po-pe)/(1-pe)，pe 由行列边缘分布乘积给出。 | κ ∈ [-1,1]；完全一致 κ=1；pe=1 时定义 κ=0 |
| `D9-063` | `train.eval.mcc` | `(yTrue, yPred) -> number` | MCC：二分类用 (tp·tn-fp·fn)/√(...)；多类用 Gorodkin 公式 (c·s-Σpₖtₖ)/√((s²-Σpₖ²)(s²-Σtₖ²))；分母为 0 时返回 0。 | MCC ∈ [-1,1]；完美预测 1；随机 0；与准确率/ F1 平衡性互补 |
| `D9-064` | `train.eval.balancedAcc` | `(yTrue, yPred) -> number` | 平衡准确率：各类 recall 的等权平均（对类别不均衡比原始准确率更稳健）。 | 取 yTrue 中出现过的类别计算 recall 平均；二类平衡时等于 (TPR+TNR)/2 |
| `D9-065` | `train.eval.logloss` | `(yTrue, proba:Tensor[n,C]\|Tensor[n]\|array, eps?:number) -> number` | 交叉熵/logloss：2D proba 取真实类概率列，1D proba 视为正类概率；概率截断到 [eps,1-eps]。 | logloss ≥ 0；完美概率 → 0；eps 防止 log 0 |
| `D9-066` | `train.eval.brier` | `(yTrue:0/1, proba:Tensor[n]\|Tensor[n,2]\|array) -> number` | Brier 分数：正类概率预测的均方误差 mean((p-y)²)；2D 入参取第 2 列。 | brier ∈ [0,1]；完美概率 → 0；与 logloss 同为 proper scoring rule |
| `D9-067` | `train.eval.fbeta` | `(yTrue, yPred, beta?:number, opts?:{average, posLabel, classes}) -> number` | F-beta（β=1 即 F1）：默认二分类（posLabel=1）；opts.average='macro' 时对逐类 F-beta 等权平均。 | Fβ = (1+β²)tp / ((1+β²)tp + β²fn + fp)；分母 0 返回 0 |
| `D9-068` | `train.eval.topKAcc` | `(yTrue, proba:Tensor[n,C]\|nested, k:number) -> number` | Top-k 准确率：真实标签落在概率前 k 名即算对（ILSVRC 口径）。 | k=1 退化为准确率；k ≥ C 时为 1 |
| `D9-069` | `train.eval.perClass` | `(yTrue, yPred, classes?) -> {classes, perClass:[{class, tp, fp, fn, support, precision, recall, f1}]}` | 逐类统计 tp/fp/fn/support 与 precision/recall/f1；classificationReport 与 macroAvg 的基础件。 | 各类 tp+fn = support；Σtp = 判对数 |
| `D9-070` | `train.eval.majorityBaseline` | `(yTrue) -> {label, count, proportion, accuracy}` | 多数类基线：永远预测最高频类别时的准确率（零模型上界参照）。 | accuracy = max_c count(c)/n；proportion 同值 |
| `D9-071` | `train.eval.stratifiedBaseline` | `(yTrue, seed?:number) -> {accuracy, expected, labels, prior, seed}` | 分层基线：按经验类别先验随机预测的模拟准确率，并给出解析期望 Σp_c²；固定种子可复现。 | 模拟准确率随种子确定；expected = Σ_c (count(c)/n)²；prior 和为 1 |
| `D9-072` | `train.eval.resultTable` | `(rows:object[], cols?:string[]) -> string` | 把记录数组渲染为 Markdown 表；cols 缺省取键的有序并集；数字定宽格式化。 | 表头列 = cols；数据行数 = rows.length |
| `D9-073` | `train.eval.compareTable` | `(runs:object[], cols?:string[]) -> string` | 多运行对比 Markdown 表：若存在 name 列则置于首位，便于横向比较指标。 | 含 name 时首列为 name；行数 = runs.length |
| `D9-074` | `train.eval.rankRuns` | `(runs:object[], metric:string\|fn, mode?:'max'\|'min') -> array` | 按指定指标对运行排序并标注 rank（1 起）；mode='max' 降序（默认），'min' 升序；不修改入参数组。 | 返回长度 = runs.length；rank 连续从 1 起；原数组不被修改 |
| `D9-075` | `train.repro.seedAll` | `(seed:number\|string) -> {seed, rng, stream(name) -> rng}` | 统一播种入口：字符串种子先经 FNV-1a 派生为 32bit；返回主 rng 与按名字派生的确定性子流（数据/初始化/增广各用一流）。 | 同种子两次调用各流序列一致；不同流名序列不同；字符串种子走 seedFrom |
| `D9-076` | `train.repro.fingerprint` | `(env?:{seed, dataHash, config}) -> {hash, parts}` | 环境指纹：内核/目录版本 + node/平台/架构 + 可选种子/数据哈希/配置哈希的稳定摘要；不含时间。 | 同入参同指纹；任一入参变化（如 seed）指纹变；parts 可读 |
| `D9-077` | `train.repro.manifest` | `(config, opts?:{seed, dataHash, extra, timestamp}) -> manifest` | 生成实验 manifest：配置 + 种子 + 内核/目录版本 + 环境 + 数据哈希；不含时间戳（除非显式传入）→ 可稳定哈希。 | 同 (config, seed, dataHash) 两次 manifest 深度相等；manifestHash 相同 |
| `D9-078` | `train.repro.manifestHash` | `(manifest) -> hex8` | 对 manifest 做规范 JSON + FNV-1a 哈希；改种子/配置/数据哈希 → 哈希变。 | 同 manifest 同哈希；键序不影响结果（规范序列化排序键） |
| `D9-079` | `train.repro.determinismCheck` | `(fn:fn, args:array, opts?:{runs}) -> {deterministic, runs, canonical, mismatchAt}` | 以深拷贝入参重复执行 fn runs 次（默认 2），对返回做规范序列化逐位比较；用于验证训练/预处理确定性。 | deterministic=true 时所有 run 的规范序列化完全相同；mismatchAt 指出首个不一致 run |
| `D9-080` | `train.repro.snapshotParams` | `(params) -> snapshot` | 参数结构深拷贝快照（Tensor/数组/纯对象/标量），与源完全隔离。 | snapshot 与 params 逐位相等；互不影响 |
| `D9-081` | `train.repro.restoreParams` | `(snapshot) -> params` | 从快照深拷贝恢复参数；与 snapshot 隔离。 | restore(snapshot(p)) 与 p 逐位相等 |
| `D9-082` | `train.repro.envInfo` | `() -> {node, platform, arch, kernel, catalog, domains, opsInDomain9}` | 环境信息：node 版本、平台、架构、内核与目录版本、域数、域9 指令数（稳定字段，不含时间）。 | 同一进程内多次调用结果相同 |
| `D9-083` | `train.repro.logAppend` | `(log:array, entry:object) -> array` | 向实验日志追加带 seq 序号的记录；纯函数不修改入参。 | 返回长度 = 入参+1；seq 连续（缺省 length+1） |
| `D9-084` | `train.repro.experimentCompare` | `(logA:array, logB:array, key?:string) -> {onlyA, onlyB, common, diffs, identical}` | 按 key（默认 runId）对齐两份实验日志，报告独有条目与字段级差异（用于复现审计）。 | onlyA/onlyB/common 构成键的全集划分；diffs 逐字段给出两侧值；identical = 三者皆空 |
| `D9-085` | `train.repro.tableToCsv` | `(rows:object[], cols?:string[]) -> string` | 记录数组 → CSV（含表头）；含逗号/引号/换行的值按 RFC 4180 双引号转义。 | 首行为表头；行数 = rows.length+1；转义后 round-trip 不破坏列结构 |
| `D9-086` | `train.repro.tableToMarkdown` | `(rows:object[], cols?:string[]) -> string` | 记录数组 → Markdown 表（与 eval.resultTable 同渲染，供实验报告归档）。 | 表头 \| ... \| + 分隔行 + 数据行 |
| `D9-087` | `train.repro.hashData` | `(dataset) -> hex8` | 对数据集 X/y 的 Float64 字节做 FNV-1a（含 shape）；改动任一比特即变。 | 同数据同哈希；形状变化哈希变；不依赖十进制表示 |
| `D9-088` | `train.repro.hashParams` | `(params) -> hex8` | 对参数结构（Tensor/数组/对象/标量）做规范序列化 + FNV-1a 哈希。 | 同构同哈希；键序无关；Tensor 按 (shape,data) 编码 |
| `D9-089` | `train.repro.assertReproducible` | `(a, b) -> true` | 断言两个对象（manifest/运行记录）在剔除 runId/timestamp/note 后规范相等；不等则抛 E_VALUE。 | 相等返回 true；不等抛 ForgeError(E_VALUE) |
| `D9-090` | `train.repro.resetState` | `() -> true` | 清空模块级指标存储与序号计数（测试隔离用）。 | reset 后 recordMetric/metricSeries 从空开始 |
| `D9-091` | `train.repro.recordMetric` | `(runId:string, name:string, value:number, step?:number) -> record` | 向模块级存储记录一个 (runId, name, value, step) 指标点；step 缺省取全局序号。 | 返回记录含 runId/name/value/step/seq；同 runId 可多次记录 |
| `D9-092` | `train.repro.metricSeries` | `(runId:string, name:string) -> {step, value}[]` | 按 (step, seq) 排序返回某运行的某指标序列（用于画曲线/比较）。 | 长度 = 该 runId 下该 name 的记录数；按 step 升序 |
| `D9-093` | `train.repro.bestEpoch` | `(history:array, metric?:string, mode?:'min'\|'max') -> {epoch, metric, mode, value, record}` | 在训练历史中找最优 epoch：mode 缺省按指标名推断（含 loss/error/mse → min，否则 max）。 | mode=min 返回该指标最小值所在记录；缺省指标 valLoss |
| `D9-094` | `train.repro.runSummary` | `(run:{id?, seed?, config?, history?, metrics?, bestMetric?, bestMode?}) -> summary` | 运行摘要：id/种子/轮数/配置哈希/指标集合/最优 epoch；供归档、对比与 replay 使用。 | metrics 优先取 run.metrics，否则取末条历史记录（除 epoch/lr）；best 缺失指标时为 null |
| `D9-095` | `train.pipe.standardizeThenSplit` | `(dataset, testFrac:number, seed:number) -> {scaler, train, test}` | 先按种子划分，再只在训练集上拟合标准化器并应用于两侧，杜绝测试集统计量泄漏。 | scaler 仅由 train 拟合；train 标准化后每列均值≈0、std≈1；test 用同一 scaler |
| `D9-096` | `train.eval.fullReport` | `(yTrue, yPred, opts?:{proba, classes}) -> report` | 分类全量报告：accuracy、balancedAcc、宏/加权平均、逐类指标、混淆矩阵、κ、MCC、多数类基线；给 proba 时追加 logloss/brier。 | 字段与各单项指标函数口径一致；proba 缺省时不产出 logloss/brier |
| `D9-097` | `train.loop.trainValTest` | `(dataset, valFrac:number, testFrac:number, seed:number) -> {train, val, test}` | 三向划分：先切出 train，再把剩余部分按 valFrac:testFrac 比例切成 val/test；三部分 index 两两不相交。 | 0 < valFrac + testFrac < 1；三部分并集 = [0,n) 且不相交 |
| `D9-098` | `train.eval.nestedCV` | `(dataset, opts:{kOuter, kInner, candidates, makeFitPredict, seed?, metric?, mode?}) -> {outerScores, mean, std, selected, mode, inner}` | 嵌套 CV：每个外层折内用内层 CV 在候选超参数中选优（mode 默认按 metric 推断：mse/mae→min，否则 max），再评外层折。 | outerScores.length === kOuter；selected.length === kOuter；内层每候选分数长度 === kInner |
| `D9-099` | `train.repro.exportRun` | `(run:{id?, config?, seed?, dataHash?, metrics?, history?, params?, fitPredict?, data?, metric?}) -> archive` | 导出完整实验存档：manifest + 配置 + 指标 + 历史 + 参数（转纯 JSON）；附 run.fitPredict 与 run.data 时追加 replay 重放数据。 | 存档中无函数/时间戳（除非显式给）；manifestHash 可稳定重算；Tensors 转为 {shape, data} |
| `D9-100` | `train.repro.replayRun` | `(archive, fitPredict?:fn) -> {ok, manifestHash, metrics, summary, checks}` | 从 exportRun 存档重放：校验 schema、复算并比对 manifestHash、复算摘要；给 fitPredict 且存档含 replay 数据时逐位复算预测哈希。 | checks.schemaOk 恒 true（否则抛错）；manifestHashStable 为重算与存档一致；replayDeterministic 为预测哈希一致 |
| `D9-101` | `train.util.assertSplitDisjoint` | `(...splits:(number[]\|dataset)) -> true` | 断言多个划分（索引数组或带 index 的数据集）两两不相交；有重叠抛 E_VALUE，否则返回 true。 | 至少两个划分；任一重叠即抛错并指出索引 |
| `D9-102` | `train.util.coverageCheck` | `(indexArrays:number[]\|number[][], total:number) -> {ok, total, covered, missing, duplicates}` | 检查若干索引集合的并集恰好覆盖 [0,total) 一次：报告缺失与重复项。 | ok = missing 与 duplicates 皆空；covered = 出现过的索引数 |
| `D9-103` | `train.util.classWeights` | `(Tensor y[n], number k) -> Tensor[k]` | 类别均衡权重 w_c = n/(k·n_c)（scikit-learn class_weight='balanced' 口径） | Σ_c w_c·n_c = n；各类别加权总量 n/k 相等；未出现的类别权重为 0 |

## D10 · SYS 系统编排

注册表内省、遥测、组合、文档生成、基准、模糊测试。共 103 条。

| id | 指令 | 签名 | 说明 | 不变量 |
|---|---|---|---|---|
| `D10-001` | `sys.reg.count` | `(number? domain) -> number` | 指令总数；给定 domain 时返回该域条数 | count()=countOps()；count(d)=opsByDomain(d).length；∑count(d)=count() |
| `D10-002` | `sys.reg.domains` | `() -> object[]` | 列出全部域元信息（code/name/desc）及当前实际条数 | 返回 10 个域；每域 count = opsByDomain(d).length |
| `D10-003` | `sys.reg.byDomain` | `(number domain) -> op[]` | 按域取指令摘要列表（已剔除 impl，可安全序列化） | 结果全部 op.domain 等于入参；条数等于 sys.reg.count(domain) |
| `D10-004` | `sys.reg.search` | `(string q) -> op[]` | 按关键字检索指令（匹配 name / doc / id，忽略大小写） | 每条结果的 name/doc/id 之一包含小写化后的查询串 |
| `D10-005` | `sys.reg.describe` | `(string nameOrId) -> string` | 单条指令的完整一行描述（id / name / doc / sig / 参考 / 不变量） | 输出包含 op.id 与 op.name，且与 describeOp 一致 |
| `D10-006` | `sys.reg.exists` | `(string nameOrId) -> boolean` | 判断指令是否已注册（按 id 或 name） | exists(x) 为真当且仅当 getOp(x) 不抛错 |
| `D10-007` | `sys.reg.signature` | `(string nameOrId) -> string` | 取指令签名字符串 | signature(x) = getOp(x).sig |
| `D10-008` | `sys.reg.catalogHash` | `(boolean? includeDocs) -> string` | 全目录指纹：FNV-1a 64 位，按 id 排序后对 id\|domain\|name\|arity\|sig(+doc+ref+inv) 求哈希 | 相同目录两次调用结果相同；任一 op 改名 / 改签名 / 新增一条都会改变哈希 |
| `D10-009` | `sys.reg.version` | `() -> object` | 内核版本与目录版本 | version().kernel = KERNEL_VERSION；version().catalog = CATALOG_VERSION |
| `D10-010` | `sys.reg.nameIndex` | `() -> object` | name -> {id, domain, arity} 的索引表，供 O(1) 反查 | 索引键集合等于全部指令 name 集合，且每个 name 唯一 |
| `D10-011` | `sys.reg.opIds` | `(number? domain) -> string[]` | 排序后的指令 id 列表（可选按域过滤） | 列表按字典序升序；无重复；过滤后条数等于该域条数 |
| `D10-012` | `sys.reg.domainStats` | `() -> object[]` | 每个域的统计：条数、占比、首个/末个 id | 各域 count 之和 = countOps()；ratio ∈ [0,1] 且总和为 1 |
| `D10-013` | `sys.tel.summary` | `(number? k) -> row[]` | 遥测汇总：每行 {id, name, calls, errors, ms}，按调用次数降序；k 限定行数 | 按 calls 降序；k 给定时行数 ≤ k；每行 calls ≥ errors ≥ 0 |
| `D10-014` | `sys.tel.hotOps` | `(number k) -> row[]` | 调用最频繁的 k 条指令（热点分析） | 返回行数 = min(k, 总行数)，且是 summary 的前缀 |
| `D10-015` | `sys.tel.reset` | `() -> boolean` | 清空内核遥测与域10 本地观测表 | reset 后 totalCalls() 中来自内核的部分为 0 |
| `D10-016` | `sys.tel.totalCalls` | `() -> number` | 累计调用次数（内核遥测 + 本地观测） | totalCalls() = 各行 calls 之和；reset 后归零 |
| `D10-017` | `sys.tel.record` | `(string opName, boolean ok) -> object` | 手动记录一次调用观测（成功/失败各计一次调用，失败额外计一次错误） | ok=false 使该指令 errors+1；ok=true 仅使 calls+1 |
| `D10-018` | `sys.tel.errorRate` | `(string? opName) -> number` | 错误率 = 错误数 / 调用数；无调用时返回 0 | errorRate ∈ [0,1]；分母为 0 时返回 0 |
| `D10-019` | `sys.tel.latencyStats` | `() -> object` | 延迟统计（min/mean/max/p50/p95），样本来自 sys.exec.call / sys.exec.timed / sys.bench.* | 样本升序后 p50 ≤ p95 ≤ max；无样本时全部为 0 |
| `D10-020` | `sys.tel.topErrors` | `(number k) -> row[]` | 错误次数最多的 k 条指令 | 按 errors 降序；errors 为 0 的行排在末尾 |
| `D10-021` | `sys.exec.call` | `(string name, any[] args) -> any` | 统一调度执行：额外把本次耗时记入域10 延迟表 | 返回值等于 exec(name, args)；耗时被记录到 latencyStats 样本集 |
| `D10-022` | `sys.exec.try` | `(string name, any[] args) -> object` | 不抛错执行，返回 {ok, value\|error}；error 被折叠为 {code, message} | 永不抛出；ok=true 时含 value，ok=false 时含 error.code |
| `D10-023` | `sys.exec.batch` | `(call[] calls) -> object[]` | 批量执行：calls 为 [[name, args], ...] 或 [{op, args}, ...]；逐条 try，不中断 | 输出长度等于输入长度；失败项 ok=false 且不影响其它项 |
| `D10-024` | `sys.exec.pipe` | `(step[] steps, any x) -> any` | 管道串联：steps 为 [{op, args?}]，每步以「上一步输出 + args」为入参调用 | pipe([s1,s2], x) = exec(s2.op, [exec(s1.op, [x, ...s1.args]), ...s2.args]) |
| `D10-025` | `sys.exec.map` | `(string opName, any[] xs) -> any[]` | 把单参指令映射到数组每个元素 | 输出长度等于 xs 长度；第 i 项 = exec(opName, [xs[i]]) |
| `D10-026` | `sys.exec.timed` | `(string name, any[] args) -> object` | 带耗时返回：{value, ms} | ms ≥ 0；value 等于 exec(name, args) |
| `D10-027` | `sys.exec.retry` | `(string opName, any[] args, object cfg) -> object` | 失败重试：只对可重试错误码（E_NUMERICAL / E_LIMIT）重试；cfg={retries, backoffMs} | 不可重试错误码时 attempts=1；全部失败后 ok=false 且 attempts=retries+1 |
| `D10-028` | `sys.exec.memo` | `(string opName, any[] args) -> object` | 记忆化：以 name + 稳定序列化(args) 为键缓存，返回 {value, hit} | 首次 hit=false；同参再次调用 hit=true 且 value 与首次稳定序列化相同 |
| `D10-029` | `sys.exec.assertOp` | `(string name, any[] args, function\|string predicate) -> boolean` | 执行后断言：predicate 为函数（接收 value）或已注册指令名（以 [value] 调用） | 返回布尔；predicate 抛错时按 E_VALUE 归一化后向上抛出 |
| `D10-030` | `sys.verify.catalog` | `(op[] ops, object opt) -> object` | 目录级综合校验：名字唯一 / id 连续 / arity 一致 / doc 与 ref 齐备 / 域计数 | ok = (issues.length === 0)；total = ops.length；checks 覆盖 5 个子项 |
| `D10-031` | `sys.verify.invariants` | `(op[] ops) -> object` | 检查每条指令都声明了非空不变量（长度 ≥ minLen），并统计未声明项 | checked = ops.length；ok 为真当且仅当每条 op.inv 非空且长度 ≥ 5 |
| `D10-032` | `sys.verify.determinism` | `(op[] ops, string opName, any[] args, number n) -> boolean` | 确定性检验：同参重复调用 n 次，稳定序列化逐位相同则为 true | 对纯函数指令返回 true；任两次结果序列化不同即返回 false |
| `D10-033` | `sys.verify.fuzz` | `(op[] ops, string opName, object cfg={}) -> object` | 模糊测试：按签名推导类型生成随机入参反复调用，捕获但不抛出异常 | 返回 {trials, failures}；同 seed 下 trials 序列可复现；健壮指令 failures 为空 |
| `D10-034` | `sys.verify.smoke` | `(op[] ops, object argsTable) -> object` | 冒烟：按 argsTable[name] 给出的入参逐条 tryExec，统计失败项 | total = ops.length；ok 为真当且仅当 failures 为空；永不抛出 |
| `D10-035` | `sys.verify.docCoverage` | `(op[] ops, number minLen) -> object` | 文档覆盖率：doc 非空且长度 ≥ minLen 视为已覆盖 | ratio = covered/total ∈ [0,1]；全覆盖时 missing 为空且 ok 为真 |
| `D10-036` | `sys.verify.refCoverage` | `(op[] ops, number minLen) -> object` | 参考出处覆盖率：ref 非空且长度 ≥ minLen 视为已覆盖 | ratio = covered/total ∈ [0,1]；全覆盖时 missing 为空且 ok 为真 |
| `D10-037` | `sys.verify.idContinuity` | `(op[] ops) -> object` | id 连续性：每域 id 形如 D<d>-NNN，且序号 001..N 无缺号无重复 | ok 为真当且仅当每域序号集合恰为 {1..count} 且前缀与 domain 一致 |
| `D10-038` | `sys.verify.nameUnique` | `(op[] ops) -> object` | 指令名唯一性检查 | ok 为真当且仅当 name 集合无重复；dupes 列出全部重名 |
| `D10-039` | `sys.verify.arityCheck` | `(op[] ops) -> object` | arity 一致性：注册的 arity 必须等于由 impl 形参表推导出的「必需参数个数」 | ok 为真当且仅当每条指令的 arity = requiredParams(impl)，且 0 ≤ arity ≤ impl.length |
| `D10-040` | `sys.verify.domainCounts` | `(op[] ops, number\|object expected) -> object` | 域计数校验：expected 为数字则每域都须等于它；为对象则逐域比对 | ok 为真当且仅当每个出现过的域条数都符合 expected |
| `D10-041` | `sys.verify.errorTaxonomy` | `(op[] ops) -> object` | 错误分类学自洽性：每个 CODES 都有类别与建议；分类映射无未知码 | ok 为真当且仅当 codes 全覆盖且 classify 对每个码返回非 unknown 类别 |
| `D10-042` | `sys.doc.opMarkdown` | `(op[] ops, string nameOrId) -> string` | 单条指令的 Markdown 卡片 | 输出以 "### " 开头，并包含该指令的 name / sig / doc / ref / inv |
| `D10-043` | `sys.doc.domainMarkdown` | `(op[] ops, number domain) -> string` | 单个域的 Markdown 指令表 | 表格行数 = 该域条数 + 2（表头与分隔行）；标题含域编号 |
| `D10-044` | `sys.doc.catalogMarkdown` | `(op[] ops, string title) -> string` | 全目录 Markdown（按域分组 + 计数） | 包含全部传入指令的 name；域标题数量等于出现的域个数 |
| `D10-045` | `sys.doc.jsonExport` | `(op[] ops, boolean pretty) -> string` | 目录 JSON 导出（可被 JSON.parse 还原） | JSON.parse(输出).ops.length = ops.length；pretty 为真时带 2 空格缩进 |
| `D10-046` | `sys.doc.csvExport` | `(op[] ops, string delimiter) -> string` | 目录 CSV 导出（RFC 4180 引号转义） | 行数 = ops.length + 1；列数恒为 7；字段含分隔符时被双引号包裹 |
| `D10-047` | `sys.doc.htmlExport` | `(op[] ops, string title) -> string` | 目录 HTML 导出（单文件，含内联样式） | 输出以 <html 开头且以 </html> 收尾；每行指令出现在 <tr> 中 |
| `D10-048` | `sys.doc.mermaid` | `(op[] ops, string kind) -> string` | 域依赖图（Mermaid）：节点为域，边表示分层依赖 D1→D2→…→D10 | 输出以 "graph" 开头；每个出现的域各有一个节点声明 |
| `D10-049` | `sys.doc.sigTable` | `(op[] ops, number? domain) -> string` | 签名表（Markdown）：name \| 签名 \| 元数 | 行数 = 指令数 + 2；domain 给定时仅列该域 |
| `D10-050` | `sys.doc.readmeSection` | `(op[] ops, number domain) -> string` | 为 README 生成单域章节（简介 + 分组 + 签名表） | 以 "## " 开头；含域描述与全部该域指令名 |
| `D10-051` | `sys.bench.timeOp` | `(op[] ops, string name, any[] args, object cfg) -> object` | 微基准：warmup 预热后计时 n 次，返回 {ms, opsPerSec, n} | n>0 时 ms>0；opsPerSec = n / (ms/1000) 有限；失败次数记入 failures |
| `D10-052` | `sys.bench.compare` | `(op[] ops, string[] names, any[] args, object cfg) -> object[]` | 多指令横向计时对比，返回按 ms 升序的行 | 行数 = names.length；按 ms 升序；每行含 op/ms/opsPerSec |
| `D10-053` | `sys.bench.throughput` | `(op[] ops, string name, any[] args, object cfg) -> object` | 吞吐量：在 budgetMs 时间预算内尽可能多跑，返回 {opsPerSec, n, ms} | n ≥ 1；ms ≥ budgetMs（跑满预算后收尾）；opsPerSec = n/(ms/1000) |
| `D10-054` | `sys.bench.matmulFlops` | `(op[] ops, number m, number n, number k) -> object` | 矩阵乘浮点吞吐量：理论 flops = 2·m·n·k，并用朴素三重循环实测 GFLOPS | flops === 2*m*n*k；gflops = flops/(ms*1e9) 且有限非负 |
| `D10-055` | `sys.bench.memoryEstimate` | `(op[] ops, number[] shape, number dtypeBytes) -> object` | 张量内存估算：元素数 × 字节数（dtypeBytes 默认 8 = float64） | elements = ∏shape；bytes = elements·dtypeBytes；mb = bytes/2^20 |
| `D10-056` | `sys.bench.reportMarkdown` | `(op[] ops, object[] rows) -> string` | 把基准行渲染成 Markdown 报告表 | 表格行数 = rows.length + 2；rows 为空时仍输出表头 |
| `D10-057` | `sys.bench.scaling` | `(op[] ops, string name, number[] sizes, object cfg) -> object[]` | 规模扫描：对每个 size 计时一次；cfg.buildArgs(size) 可自定义入参构造（默认 [size]） | 行数 = sizes.length；每行含 size/ms/opsPerSec/ok；入参构造失败不会抛出 |
| `D10-058` | `sys.bench.suite` | `(op[] ops, object cfg) -> object` | 基准套件：对前 limit 条指令按签名生成入参并各计时一次，汇总为行 | rows.length = min(limit, ops.length)；每行含 op/ms/ok；永不抛出 |
| `D10-059` | `sys.graph.build` | `(node[] nodes) -> object` | 由图节点 {id, op?, inputs?, value?, args?} 构建图；校验 id 唯一与上游引用存在 | ids 与 nodes 一一对应；重复 id 或悬空 inputs 抛 E_VALUE / E_NOTFOUND |
| `D10-060` | `sys.graph.toposort` | `(graph) -> string[]` | Kahn 拓扑排序；存在环时抛 E_STATE | 输出中每条边的起点位置严格早于终点；无环时长度等于节点数 |
| `D10-061` | `sys.graph.execute` | `(graph, object inputs) -> object` | 按拓扑序执行图：叶子取 inputs[id] 或 node.value，其余调用 exec(node.op, 上游值 + node.args) | values 覆盖全部节点；叶子节点值 = inputs[id] ?? node.value |
| `D10-062` | `sys.graph.validate` | `(graph) -> object` | 图合法性检查：id 唯一、上游引用存在、op 已注册、无环 | ok = issues.length === 0；hasCycle 为真时必有一条 cycle 类问题 |
| `D10-063` | `sys.graph.toDot` | `(graph, string title) -> string` | 导出 Graphviz DOT 有向图 | 以 digraph 开头，以 } 收尾；每个节点与每条边各占一行 |
| `D10-064` | `sys.graph.cycleDetect` | `(graph) -> object` | DFS 三色标记检测环，返回 {hasCycle, cycle}（cycle 首尾同节点） | hasCycle 为真当且仅当 cycle 非空且 cycle[0] === cycle[cycle.length-1] |
| `D10-065` | `sys.graph.nodeCount` | `(graph) -> number` | 图节点数 | nodeCount = \|V\| ≥ 0 |
| `D10-066` | `sys.graph.edgeCount` | `(graph) -> number` | 图边数（每个 inputs 引用计一条边） | edgeCount = Σ\|node.inputs\| ≥ 0 |
| `D10-067` | `sys.cfg.merge` | `(object base, object over, boolean deep) -> object` | 配置合并：deep 为真时递归合并普通对象，数组与标量整体覆盖 | 深合并时 base 未被覆盖的键全部保留；over 的同名标量键胜出 |
| `D10-068` | `sys.cfg.validate` | `(object cfg, object schema) -> object` | 按 schema 校验：值为类型名（number/int/string/boolean/object/array/function/tensor/any）或 {type, required, min, max, enum} | ok = issues.length === 0；缺 required 键或类型不符都会产生一条 issue |
| `D10-069` | `sys.cfg.freeze` | `(object cfg) -> object` | 深度冻结配置（Object.freeze 递归）；严格模式下写入将抛 TypeError | 返回对象及其所有子对象 Object.isFrozen 为真；非严格模式下写入静默失败 |
| `D10-070` | `sys.cfg.diff` | `(object a, object b) -> object` | 配置差异：以点路径给出 added / removed / changed 三类键 | a 与 b 完全相同时三类列表均为空；diff(a,b) 与 diff(b,a) 的 added/removed 互换 |
| `D10-071` | `sys.cfg.hash` | `(object cfg) -> string` | 配置指纹：FNV-1a 64 位，基于稳定序列化（键序无关） | 键顺序不同的等价配置哈希相同；任一标量改动即改变哈希 |
| `D10-072` | `sys.cfg.pick` | `(object cfg, string[] keys) -> object` | 按键白名单抽取子配置（缺失键跳过） | 结果的键集合 ⊆ keys；每个存在的键值与原配置相同 |
| `D10-073` | `sys.cfg.defaults` | `(object cfg, object defaults) -> object` | 填充默认值：递归补全缺失键，已存在的键保持不变 | defaults 的每个键都出现在结果中；cfg 已有键的值不被覆盖 |
| `D10-074` | `sys.cfg.toEnv` | `(object cfg, string prefix) -> object` | 配置转环境变量风格键：prefix + 大写下划线路径（嵌套用双下划线） | 键数 = cfg 的叶子键数；键形如 PREFIX_KEY，全大写且仅含 [A-Z0-9_] |
| `D10-075` | `sys.err.classify` | `(string code) -> string` | 把错误码映射到类别：interface/type/shape/value/numerical/resource/lookup/state，未知为 unknown | 对每个 CODES 键返回预定义类别且不等于 unknown；未知码返回 unknown |
| `D10-076` | `sys.err.format` | `(string code, string message) -> string` | 格式化错误行：`[CODE] message`；未知码统一写作 [E_UNKNOWN] | 输出以 [ 开头且包含 message；码不在 CODES 中时前缀为 [E_UNKNOWN] |
| `D10-077` | `sys.err.codes` | `() -> object[]` | 列出全部错误码及其描述、类别、可否重试 | 条数 = \|CODES\|；每条含 code/desc/category/retryable 四个键 |
| `D10-078` | `sys.err.isRetryable` | `(string code) -> boolean` | 是否可重试：仅 E_NUMERICAL（数值失败）与 E_LIMIT（资源限额）可重试 | isRetryable("E_NUMERICAL") = true；isRetryable("E_SHAPE") = false |
| `D10-079` | `sys.err.count` | `(string? opName) -> number` | 错误计数：不给 opName 时返回全部指令的错误总数 | count(opName) ≤ count()；均为非负整数 |
| `D10-080` | `sys.err.trace` | `(string opName, any[] args) -> object` | 追踪一次调用的错误：成功返回 {ok:true, code:null}，失败返回 {ok:false, code, message} | 永不抛出；ok=false 时 code 必为字符串 |
| `D10-081` | `sys.err.suggest` | `(string code) -> string` | 给出该错误码的修复建议（未知码给通用建议） | 对每个 CODES 键返回非空建议串；未知码返回兜底建议 |
| `D10-082` | `sys.env.info` | `() -> object` | 运行环境快照：平台、架构、Node 版本、V8 版本、内核版本 | platform 与 arch 取自 process；versions.node 非空字符串 |
| `D10-083` | `sys.env.nodeVersion` | `() -> string` | Node 版本字符串（semver） | 形如 major.minor.patch；等于 sys.env.info().node |
| `D10-084` | `sys.env.memory` | `() -> object` | 进程内存用量（MB，三位小数） | heapUsed ≤ heapTotal；各项均为有限非负数 |
| `D10-085` | `sys.env.nowMs` | `() -> number` | 单调递增时钟当前毫秒（hrtime 纳秒精度） | 连续两次调用后者 ≥ 前者；单位毫秒 |
| `D10-086` | `sys.env.seed` | `(string str) -> number` | 由字符串派生 32 位种子（FNV-1a），便于用文字种子复现实验 | 同串同种子；结果 ∈ [0, 2^32) |
| `D10-087` | `sys.env.platform` | `() -> string` | 平台标识，形如 "platform-arch" | 等于 `${process.platform}-${process.arch}` |
| `D10-088` | `sys.env.uptimeMs` | `() -> number` | 进程已运行毫秒数 | 非负且单调不减；等于 process.uptime()*1000 |
| `D10-089` | `sys.self.checkAll` | `(op[] ops, object opt) -> object` | 全量自检：名字唯一 / id 连续 / arity 一致 / doc·ref 覆盖 / 域计数 / 不变量声明 | ok = issues.length === 0；byDomain 给出每个出现域的条数 |
| `D10-090` | `sys.self.summary` | `(op[] ops, object opt) -> object` | 自检摘要：版本、总数、各域条数、目录哈希、健康与否 | total = ops.length；hash 等于 sys.reg.catalogHash 对同目录的结果 |
| `D10-091` | `sys.self.exitCode` | `(op[] ops, object opt) -> number` | 自检退出码：全部通过为 0，否则为 1（供 CI 使用） | 返回值 ∈ {0,1}；checkAll().ok 为真时必为 0 |
| `D10-092` | `sys.self.domainHealth` | `(op[] ops, number domain) -> object` | 单域健康度：条数、名字唯一、id 连续、doc·ref 覆盖情况 | count = 该域条数；ok 为真当且仅当 issues 为空 |
| `D10-093` | `sys.self.doctor` | `(op[] ops) -> string` | 诊断报告（Markdown）：逐域健康度 + 问题清单 + 修复建议 | 以 "# 诊断报告" 开头；每域一行；末尾给出总体结论 |
| `D10-094` | `sys.self.report` | `(op[] ops, string title) -> string` | 自检报告（Markdown）：统计表 + 遥测热点 + 目录指纹 | 以 "# " 开头；包含条目总数与目录哈希 |
| `D10-095` | `sys.util.range` | `(number a, number b?, number step?) -> number[]` | 等差数列；单参为 [0, a)，双参为 [a, b)，step 默认 1 | 长度 = ceil((b-a)/step) 且 ≥ 0；step=1 时首项 a、末项 < b |
| `D10-096` | `sys.util.zip` | `(any[] a, any[] b) -> any[][]` | 拉链：按较短数组长度配对成 [[a0,b0], ...] | 长度 = min(a.length, b.length)；第 i 项为 [a[i], b[i]] |
| `D10-097` | `sys.util.pick` | `(object obj, string path) -> any` | 按点路径取值，如 "a.b.0"；路径不存在返回 undefined | pick(o, "k") = o.k；路径中任一段缺失返回 undefined 而不抛错 |
| `D10-098` | `sys.util.deepEqual` | `(any a, any b) -> boolean` | 深度相等：自反、对称、支持嵌套对象/数组/Tensor；-0 与 0 视为相等，NaN 视为等于自身 | deepEqual(a,a) 恒真（自反）；deepEqual(a,b) = deepEqual(b,a)（对称） |
| `D10-099` | `sys.util.typeOf` | `(any v) -> string` | 精确类型名：null/undefined/array/tensor/typedarray/number/string/boolean/function/object/bigint/symbol | typeOf(null)="null"；typeOf([])="array"；typeOf(Tensor)="tensor" |
| `D10-100` | `sys.util.formatNum` | `(number v, number digits) -> string` | 定点格式化（digits 位小数）；非有限数输出字面量 | 有限数输出不含指数记号且小数位数恰为 digits；NaN 输出 "NaN" |
| `D10-101` | `sys.util.table` | `(any[][] rows, string[] headers) -> string` | 对齐文本表（列宽取该列最长单元） | 行数 = rows.length + (headers?1:0)；同列所有行宽度一致 |
| `D10-102` | `sys.util.chunk` | `(any[] arr, number size) -> any[][]` | 按固定长度切分数组，末尾不足一组时保留为短块 | 块数 = ceil(len/size)；除末块外每块长度均为 size；拼接还原原数组 |
| `D10-103` | `sys.bench.percentile` | `(number[] xs, number p) -> number` | 分位数（Hyndman-Fan 类型 7 线性插值，NumPy 默认口径） | percentile(xs,0)=min(xs)；percentile(xs,100)=max(xs)；关于 p 单调不减 |
