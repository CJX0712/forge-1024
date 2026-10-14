// 域8 深度学习 DL（102 条微指令）
// 对标：PyTorch torch.nn（层/激活/归一化/损失/init）、TensorFlow Keras layers、
//       ONNX Operator Standard（Conv/MaxPool/Gemm/Softmax/LSTM/Attention）、
//       FlashAttention（Dao et al. 2022 分块 online-softmax）、RoFormer（Su et al. 2021）、
//       ALiBi（Press et al. ICLR 2022）、GQA（Ainslie et al. 2023）、SwiGLU（Shazeer JMLR 2020）。
// 组织结构（9 组）：
//   一、激活与门控 14 · 二、归一化 10 · 三、层 14 · 四、正则化 7 · 五、注意力 16
//   六、序列模型 8 · 七、损失 14 · 八、初始化 10 · 九、模型与工具 9
// —— 张量布局约定（全文统一，sig 中显式标注）——
//   · 卷积/池化：NCHW，即 [N, C, H, W]（一维为 [N, C, L]）
//   · 序列/注意力：[B, T, C]（多头内部展开为 [B, H, T, D]）
//   · 全连接/损失：二维 [N, D]，类别维在最后一维
//   · 卷积核：conv2d 为 [OC, IC, KH, KW]；转置卷积为 [IC, OC, KH, KW]；深度卷积为 [C, 1, KH, KW]
// —— 数值约定 ——
//   · softmax / logsumexp 一律先减最大值；log 类带 eps；除法做零保护
//   · 随机一律经 createRng(seed)，禁止 Math.random，保证可复现
import {
  Tensor, defineOp, exec, opsByDomain, ForgeError,
  expectTensor, expectNumber, createRng,
} from '../core/index.mjs';
import './d1-math.mjs';
import './d2-num.mjs';
import './d3-linalg.mjs';
import './d4-autodiff.mjs';

let _s = 0;
const nid = () => `D8-${String(++_s).padStart(3, '0')}`;

// =====================================================================
// 内部小工具（非指令）
// =====================================================================
const _prod = (sh) => { let n = 1; for (const d of sh) n *= d; return n; };

function _stridesOf(shape) {
  const st = new Array(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) { st[i] = acc; acc *= shape[i]; }
  return st;
}

// 展平下标 -> 多维下标（行主序）
function _decode(p, shape) {
  const n = shape.length;
  const idx = new Array(n);
  let r = p;
  for (let d = n - 1; d >= 0; d--) { idx[d] = r % shape[d]; r = (r - idx[d]) / shape[d]; }
  return idx;
}

// 右对齐广播到目标形状（NumPy 规则）
function _bcast(t, shape) {
  expectTensor(t, '待广播张量');
  const n = shape.length;
  if (t.ndim > n) throw new ForgeError('E_SHAPE', `广播失败: [${t.shape}] 维数多于 [${shape}]`);
  const lead = n - t.ndim;
  const st = _stridesOf(t.shape);
  for (let d = 0; d < t.ndim; d++) {
    const a = t.shape[d], b = shape[lead + d];
    if (a !== b && a !== 1) throw new ForgeError('E_SHAPE', `广播失败: [${t.shape}] -> [${shape}]`);
  }
  if (t.ndim === n && t.shape.every((v, i) => v === shape[i])) return t;
  const ost = _stridesOf(shape);
  const out = new Float64Array(_prod(shape));
  for (let p = 0; p < out.length; p++) {
    const idx = _decode(p, shape);
    let q = 0;
    for (let d = 0; d < t.ndim; d++) if (t.shape[d] !== 1) q += idx[lead + d] * st[d];
    out[p] = t.data[q];
  }
  return new Tensor(out, shape);
}

// 2D 矩阵乘 [m,k]·[k,n] -> [m,n]
function _mm(A, B) {
  expectTensor(A, 'A'); expectTensor(B, 'B');
  if (A.ndim !== 2 || B.ndim !== 2) throw new ForgeError('E_SHAPE', 'matmul 仅支持 2D');
  const [m, k] = A.shape; const [k2, n] = B.shape;
  if (k !== k2) throw new ForgeError('E_SHAPE', `matmul 内维不符: ${k} vs ${k2}`);
  const out = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let t = 0; t < k; t++) {
      const a = A.data[i * k + t];
      if (a === 0) continue;
      for (let j = 0; j < n; j++) out[i * n + j] += a * B.data[t * n + j];
    }
  }
  return new Tensor(out, [m, n]);
}

// x[m,k]·W[k,n] + b[n] -> [m,n]
function _mmBias(x, W, b) {
  const [m, k] = x.shape; const n = W.shape[1];
  const out = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) out[i * n + j] = b ? b.data[j] : 0;
    for (let t = 0; t < k; t++) {
      const a = x.data[i * k + t];
      if (a === 0) continue;
      for (let j = 0; j < n; j++) out[i * n + j] += a * W.data[t * n + j];
    }
  }
  return new Tensor(out, [m, n]);
}

// 批量矩阵乘 [B,m,k]·[B,k,n] -> [B,m,n]
function _bmm(A, B) {
  const [b0, m, k] = A.shape; const [b1, k2, n] = B.shape;
  if (b0 !== b1 || k !== k2) throw new ForgeError('E_SHAPE', `bmm 形状不符: [${A.shape}] vs [${B.shape}]`);
  const out = new Float64Array(b0 * m * n);
  for (let s = 0; s < b0; s++) {
    for (let i = 0; i < m; i++) for (let t = 0; t < k; t++) {
      const a = A.data[(s * m + i) * k + t];
      if (a === 0) continue;
      for (let j = 0; j < n; j++) out[(s * m + i) * n + j] += a * B.data[(s * k + t) * n + j];
    }
  }
  return new Tensor(out, [b0, m, n]);
}

// 沿最后一维的稳定 softmax
function _softmaxLast(x) {
  const D = x.shape[x.ndim - 1];
  const outer = x.size / D;
  const out = new Float64Array(x.size);
  for (let o = 0; o < outer; o++) {
    const base = o * D;
    let mx = -Infinity;
    for (let j = 0; j < D; j++) if (x.data[base + j] > mx) mx = x.data[base + j];
    let s = 0;
    for (let j = 0; j < D; j++) { const e = Math.exp(x.data[base + j] - mx); out[base + j] = e; s += e; }
    const inv = 1 / s;
    for (let j = 0; j < D; j++) out[base + j] *= inv;
  }
  return new Tensor(out, x.shape);
}

// 沿最后一维的稳定 log-softmax（x - logsumexp）
function _logSoftmaxLast(x) {
  const D = x.shape[x.ndim - 1];
  const outer = x.size / D;
  const out = new Float64Array(x.size);
  for (let o = 0; o < outer; o++) {
    const base = o * D;
    let mx = -Infinity;
    for (let j = 0; j < D; j++) if (x.data[base + j] > mx) mx = x.data[base + j];
    let s = 0;
    for (let j = 0; j < D; j++) s += Math.exp(x.data[base + j] - mx);
    const lse = mx + Math.log(s);
    for (let j = 0; j < D; j++) out[base + j] = x.data[base + j] - lse;
  }
  return new Tensor(out, x.shape);
}

// 沿指定轴（keepdims）求均值与方差；correction 为贝塞尔校正量（1 = 无偏）
function _statsKeep(x, axes, correction = 0) {
  const ax = (Array.isArray(axes) ? axes : [axes]).map((a) => (a < 0 ? a + x.ndim : a));
  for (const a of ax) if (a < 0 || a >= x.ndim) throw new ForgeError('E_SHAPE', `轴 ${a} 越界`);
  const red = new Set(ax);
  const outShape = x.shape.map((d, i) => (red.has(i) ? 1 : d));
  const ost = _stridesOf(outShape);
  const slots = _prod(outShape);
  const sum = new Float64Array(slots);
  const cnt = new Float64Array(slots);
  const mean = new Float64Array(slots);
  const varr = new Float64Array(slots);
  for (let p = 0; p < x.size; p++) {
    const idx = _decode(p, x.shape);
    let q = 0;
    for (let d = 0; d < x.ndim; d++) if (!red.has(d)) q += idx[d] * ost[d];
    sum[q] += x.data[p]; cnt[q] += 1;
  }
  for (let i = 0; i < slots; i++) mean[i] = sum[i] / Math.max(1, cnt[i]);
  // 两遍算法（避免 Σx²−(Σx)²/n 的抵消误差）
  for (let p = 0; p < x.size; p++) {
    const idx = _decode(p, x.shape);
    let q = 0;
    for (let d = 0; d < x.ndim; d++) if (!red.has(d)) q += idx[d] * ost[d];
    const dv = x.data[p] - mean[q];
    varr[q] += dv * dv;
  }
  const denom0 = Math.max(1, (cnt.length ? cnt[0] : 1) - correction);
  for (let i = 0; i < slots; i++) varr[i] = varr[i] / denom0;
  return { mean: new Tensor(mean, outShape), var: new Tensor(varr, outShape) };
}

// [B,T,D] 取第 t 个时间片 -> [B,D]
function _timeSlice(x, t) {
  const [B, T, D] = x.shape;
  const out = new Float64Array(B * D);
  for (let b = 0; b < B; b++) for (let d = 0; d < D; d++) out[b * D + d] = x.data[(b * T + t) * D + d];
  return new Tensor(out, [B, D]);
}

// [[B,D] × T] -> [B,T,D]
function _stackTime(list) {
  const T = list.length; const B = list[0].shape[0]; const D = list[0].shape[1];
  const out = new Float64Array(B * T * D);
  for (let t = 0; t < T; t++) for (let i = 0; i < B * D; i++) out[t * B * D + i] = list[t].data[i];
  return new Tensor(out, [B, T, D]);
}

// 归约方式：'mean' 全局平均，'sum' 求和
function _reduce(lossArr, reduction) {
  const r = reduction === undefined ? 'mean' : reduction;
  let s = 0;
  for (const v of lossArr) s += v;
  if (r === 'sum') return s;
  if (r === 'mean') return s / Math.max(1, lossArr.length);
  throw new ForgeError('E_VALUE', `未知 reduction: ${r}（支持 'mean' | 'sum'）`);
}

// 把标量或二元组规范成 [a, b]
function _pair(v, who) {
  if (typeof v === 'number') { expectNumber(v, who); return [v, v]; }
  if (Array.isArray(v) && v.length === 2) return [v[0], v[1]];
  throw new ForgeError('E_TYPE', `${who}: 期望 number 或 [number, number]`);
}

const _SQRT2 = Math.SQRT2;

// =====================================================================
// 一、激活与门控（14 条）
// =====================================================================
defineOp({
  id: nid(), domain: 8, name: 'dl.act.relu', arity: 1,
  sig: '(Tensor x) -> Tensor',
  doc: 'ReLU max(0,x)，逐元素；0 点次梯度取 0',
  ref: 'Nair & Hinton, Rectified Linear Units Improve Restricted Boltzmann Machines, ICML 2010',
  inv: 'relu(x) ≥ 0；relu(−1)=0；relu(x)+relu(−x)=|x|',
  impl: (x) => { expectTensor(x, 'x'); return x.map((v) => (v > 0 ? v : 0)); },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.act.leakyRelu', arity: 2,
  sig: '(Tensor x, number alpha=0.01) -> Tensor',
  doc: 'LeakyReLU：负区间保留 alpha 倍斜率，缓解 dying ReLU',
  ref: 'Maas et al., Rectifier Nonlinearities Improve Neural Network Acoustic Models, ICML 2013',
  inv: 'leakyRelu(x) = x (x>0) = alpha·x (x≤0)；alpha=0 退化为 relu',
  impl: (x, alpha) => {
    expectTensor(x, 'x');
    const a = alpha === undefined ? 0.01 : expectNumber(alpha, 'alpha');
    return x.map((v) => (v > 0 ? v : a * v));
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.act.elu', arity: 2,
  sig: '(Tensor x, number alpha=1) -> Tensor',
  doc: 'ELU：x>0 取 x，否则 alpha·(e^x−1)，负区间饱和于 −alpha',
  ref: 'Clevert et al., Fast and Accurate Deep Network Learning by Exponential Linear Units, ICLR 2016',
  inv: 'elu(0)=0；x→−∞ 时 elu(x)→−alpha；x>0 段为恒等',
  impl: (x, alpha) => {
    expectTensor(x, 'x');
    const a = alpha === undefined ? 1 : expectNumber(alpha, 'alpha');
    return x.map((v) => (v > 0 ? v : a * Math.expm1(Math.min(v, 0))));
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.act.selu', arity: 1,
  sig: '(Tensor x) -> Tensor',
  doc: 'SELU：lambda·ELU(x, alpha)，固定 alpha=1.6732632、lambda=1.0507010，具自归一化性质',
  ref: 'Klambauer et al., Self-Normalizing Neural Networks, NeurIPS 2017',
  inv: 'selu(0)=0；不动点附近雅可比谱半径≈1（自归一化）；x→−∞ 时 → −lambda·alpha',
  impl: (x) => {
    expectTensor(x, 'x');
    const ALPHA = 1.6732632423543772848170429916717;
    const LAMBDA = 1.0507009873554804934193349852946;
    return x.map((v) => (v > 0 ? LAMBDA * v : LAMBDA * ALPHA * Math.expm1(Math.min(v, 0))));
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.act.gelu', arity: 2,
  sig: "(Tensor x, string approx='exact') -> Tensor",
  doc: 'GELU = x·Φ(x)；approx="exact" 用 erf 精确计算，"tanh" 用 tanh 近似',
  ref: 'Hendrycks & Gimpel, Gaussian Error Linear Units (GELUs), 2016',
  inv: 'GELU(0)=0；x→∞ 时 GELU(x)→x；x→−∞ 时 →0；0<GELU(x)<x (x>0)',
  impl: (x, approx) => {
    expectTensor(x, 'x');
    const mode = approx === undefined ? 'exact' : String(approx);
    if (mode === 'tanh') {
      const c = Math.sqrt(2 / Math.PI);
      return x.map((v) => 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v * v * v))));
    }
    if (mode !== 'exact') throw new ForgeError('E_VALUE', `未知 GELU 近似: ${mode}`);
    return x.map((v) => 0.5 * v * (1 + exec('num.erf', [v / _SQRT2])));
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.act.mish', arity: 1,
  sig: '(Tensor x) -> Tensor',
  doc: 'Mish = x·tanh(softplus(x))，下界约 −0.30884，上界无界',
  ref: 'Misra, Mish: A Self Regularized Non-Monotonic Neural Activation Function, 2019',
  inv: 'mish(0)=0；mish(x)≈x（x 大）；最小值 ≈ −0.30884（x≈−1.1924）',
  impl: (x) => {
    expectTensor(x, 'x');
    return x.map((v) => {
      const sp = v > 20 ? v : Math.log1p(Math.exp(v));
      return v * Math.tanh(sp);
    });
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.act.swish', arity: 2,
  sig: '(Tensor x, number beta=1) -> Tensor',
  doc: 'Swish/SiLU = x·σ(beta·x)；beta=1 即 SiLU，beta→∞ 趋近 ReLU',
  ref: 'Ramachandran et al., Searching for Activation Functions, 2017',
  inv: 'swish(0)=0；swish(x)→x（x→∞）；swish(x)→0（x→−∞）',
  impl: (x, beta) => {
    expectTensor(x, 'x');
    const b = beta === undefined ? 1 : expectNumber(beta, 'beta');
    return x.map((v) => v / (1 + Math.exp(-b * v)));
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.act.sigmoid', arity: 1,
  sig: '(Tensor x) -> Tensor',
  doc: 'Logistic sigmoid σ(x)=1/(1+e^{−x})，输出落在 (0,1)',
  ref: '标准 logistic 函数（Berkson 1944；深度学习通用门控）',
  inv: 'sigmoid(0)=0.5；sigmoid(x)+sigmoid(−x)=1；值域 ⊂ (0,1)',
  impl: (x) => { expectTensor(x, 'x'); return x.map((v) => 1 / (1 + Math.exp(-v))); },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.act.tanh', arity: 1,
  sig: '(Tensor x) -> Tensor',
  doc: '双曲正切，零中心、值域 (−1,1)',
  ref: '标准激活（LeCun et al. 1998 高效反向传播推崇 tanh）',
  inv: 'tanh(0)=0；值域 ⊂ (−1,1)；tanh 为奇函数',
  impl: (x) => { expectTensor(x, 'x'); return x.map(Math.tanh); },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.act.softplus', arity: 3,
  sig: '(Tensor x, number beta=1, number threshold=20) -> Tensor',
  doc: 'Softplus = (1/beta)·log(1+e^{beta·x})；beta·x 超过 threshold 时线性化避免溢出',
  ref: 'Dugas et al., Incorporating Second-Order Functional Knowledge for Better Option Pricing, NIPS 2000',
  inv: 'softplus(0)=ln2；softplus(x)→x（x 大）；softplus(x)>0 且单调',
  impl: (x, beta, threshold) => {
    expectTensor(x, 'x');
    const b = beta === undefined ? 1 : expectNumber(beta, 'beta');
    const th = threshold === undefined ? 20 : expectNumber(threshold, 'threshold');
    if (!(b > 0)) throw new ForgeError('E_VALUE', 'beta 必须 > 0');
    return x.map((v) => {
      const t = b * v;
      return t > th ? v : Math.log1p(Math.exp(t)) / b;
    });
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.act.softsign', arity: 1,
  sig: '(Tensor x) -> Tensor',
  doc: 'Softsign = x/(1+|x|)，tanh 的多项式型替代，尾部收敛更慢',
  ref: 'Bergstra et al., Theano 激活函数库, 2010；Glorot & Bengio 2010 提及',
  inv: 'softsign(0)=0；值域 ⊂ (−1,1)；softsign 为奇函数',
  impl: (x) => { expectTensor(x, 'x'); return x.map((v) => v / (1 + Math.abs(v))); },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.act.hardSigmoid', arity: 1,
  sig: '(Tensor x) -> Tensor',
  doc: 'Hard sigmoid = clamp((x+3)/6, 0, 1)，分段线性、无 exp，适合量化部署',
  ref: 'Courbariaux et al., BinaryNet, 2016；Howard et al., MobileNets V3（h-swish 同源）, 2019',
  inv: 'x≤−3 时为 0；x≥3 时为 1；在 [−3,3] 上斜率为 1/6',
  impl: (x) => {
    expectTensor(x, 'x');
    return x.map((v) => Math.min(1, Math.max(0, (v + 3) / 6)));
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.act.prelu', arity: 2,
  sig: '(Tensor x[...,C], Tensor alpha[C]) -> Tensor',
  doc: 'PReLU：负区间斜率为可学习参数 alpha（按最后一维通道广播）',
  ref: 'He et al., Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet, ICCV 2015',
  inv: 'alpha=0 退化为 relu；alpha=1 退化为恒等；x>0 段恒等',
  impl: (x, alpha) => {
    expectTensor(x, 'x'); expectTensor(alpha, 'alpha');
    const C = x.shape[x.ndim - 1];
    if (alpha.size !== C && alpha.size !== 1) {
      throw new ForgeError('E_SHAPE', `alpha 元素数 ${alpha.size} 与通道数 ${C} 不符`);
    }
    return x.map((v, i) => {
      if (v > 0) return v;
      const c = i % C;
      return v * alpha.data[alpha.size === 1 ? 0 : c];
    });
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.act.glu', arity: 2,
  sig: '(Tensor x, number axis=-1) -> Tensor',
  doc: 'Gated Linear Unit：沿 axis 对半切分为 (a,b)，输出 a·σ(b)',
  ref: 'Dauphin et al., Language Modeling with Gated Convolutional Networks, ICML 2017',
  inv: 'axis 维长度减半；b→−∞ 时输出→0，b→∞ 时输出→a',
  impl: (x, axis) => {
    expectTensor(x, 'x');
    const a0 = axis === undefined ? -1 : axis;
    const ax = a0 < 0 ? x.ndim + a0 : a0;
    if (ax < 0 || ax >= x.ndim) throw new ForgeError('E_SHAPE', `轴 ${ax} 越界`);
    const D = x.shape[ax];
    if (D % 2 !== 0) throw new ForgeError('E_SHAPE', `GLU 要求 axis 维为偶数，当前 ${D}`);
    const half = D / 2;
    const outShape = x.shape.slice(); outShape[ax] = half;
    const st = _stridesOf(x.shape);
    const out = new Float64Array(_prod(outShape));
    for (let p = 0; p < out.length; p++) {
      const idx = _decode(p, outShape);
      let base = 0;
      for (let d = 0; d < x.ndim; d++) if (d !== ax) base += idx[d] * st[d];
      const off = idx[ax];
      const a = x.data[base + off * st[ax]];
      const b = x.data[base + (off + half) * st[ax]];
      out[p] = a / (1 + Math.exp(-b));
    }
    return new Tensor(out, outShape);
  },
});

// =====================================================================
// 二、归一化（10 条）
// =====================================================================
// NCHW 下 BatchNorm 的统计轴：除通道维(1)以外的全部轴
function _bnAxes(ndim) {
  const ax = [0];
  for (let d = 2; d < ndim; d++) ax.push(d);
  return ax;
}

defineOp({
  id: nid(), domain: 8, name: 'dl.norm.batchnorm', arity: 7,
  sig: '(Tensor x[N,C,...], Tensor gamma[C], Tensor beta[C], Tensor mean[C], Tensor var[C], number training=1, number eps=1e-5) -> Tensor',
  doc: '批归一化：training=1 用本批统计量，training=0 用传入的 running 统计量；gamma/beta 为仿射参数',
  ref: 'Ioffe & Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, ICML 2015',
  inv: '训练态输出按通道均值≈0、方差≈1；推理态使用 running 统计；eps 防止除零',
  impl: (x, gamma, beta, mean, varT, training, eps) => {
    expectTensor(x, 'x');
    const e = eps === undefined ? 1e-5 : expectNumber(eps, 'eps');
    const train = training === undefined ? 1 : training;
    if (x.ndim < 2) throw new ForgeError('E_SHAPE', 'batchnorm 需要 ndim>=2');
    const C = x.shape[1];
    const spatial = _prod(x.shape.slice(2)); // 各通道的空间元素数（2D 输入时为 1）
    const CS = C * spatial;
    // 每通道一份统计量（训练态在本批上按 N 与空间维归约）
    const mu = new Float64Array(C);
    const va = new Float64Array(C);
    if (train) {
      const st = _statsKeep(x, _bnAxes(x.ndim), 0);
      const stm = _stridesOf(st.mean.shape);
      for (let c = 0; c < C; c++) {
        let q = 0;
        for (let d = 0; d < x.ndim; d++) if (st.mean.shape[d] !== 1) q += (d === 1 ? c : 0) * stm[d];
        mu[c] = st.mean.data[q]; va[c] = st.var.data[q];
      }
    } else {
      expectTensor(mean, 'mean'); expectTensor(varT, 'var');
      if (mean.size !== C || varT.size !== C) {
        throw new ForgeError('E_SHAPE', `running 统计量元素数应为 ${C}`);
      }
      for (let c = 0; c < C; c++) { mu[c] = mean.data[c]; va[c] = varT.data[c]; }
    }
    if (gamma != null && gamma.size !== C) throw new ForgeError('E_SHAPE', `gamma 元素数应为 ${C}`);
    if (beta != null && beta.size !== C) throw new ForgeError('E_SHAPE', `beta 元素数应为 ${C}`);
    const out = new Float64Array(x.size);
    for (let p = 0; p < x.size; p++) {
      const n = Math.floor(p / CS);
      const c = Math.floor((p - n * CS) / spatial);
      const inv = 1 / Math.sqrt(va[c] + e);
      const z = (x.data[p] - mu[c]) * inv;
      out[p] = (gamma ? gamma.data[c] * z : z) + (beta ? beta.data[c] : 0);
    }
    return new Tensor(out, x.shape);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.norm.layernorm', arity: 4,
  sig: '(Tensor x[...,D], Tensor gamma[D], Tensor beta[D], number eps=1e-5) -> Tensor',
  doc: '层归一化：沿最后一维去均值除标准差，再做 gamma/beta 仿射；与 batch 大小无关',
  ref: 'Ba et al., Layer Normalization, 2016',
  inv: '归一化后每条样本均值≈0、方差≈1（gamma=1,beta=0 时精确成立）',
  impl: (x, gamma, beta, eps) => {
    expectTensor(x, 'x');
    const e = eps === undefined ? 1e-5 : expectNumber(eps, 'eps');
    const D = x.shape[x.ndim - 1];
    const outer = x.size / D;
    const out = new Float64Array(x.size);
    for (let o = 0; o < outer; o++) {
      const base = o * D;
      let mu = 0;
      for (let j = 0; j < D; j++) mu += x.data[base + j];
      mu /= D;
      let v = 0;
      for (let j = 0; j < D; j++) { const dv = x.data[base + j] - mu; v += dv * dv; }
      v /= D;
      const inv = 1 / Math.sqrt(v + e);
      for (let j = 0; j < D; j++) {
        const z = (x.data[base + j] - mu) * inv;
        out[base + j] = z * (gamma ? gamma.data[j % gamma.size] : 1) + (beta ? beta.data[j % beta.size] : 0);
      }
    }
    return new Tensor(out, x.shape);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.norm.rmsnorm', arity: 3,
  sig: '(Tensor x[...,D], Tensor gamma[D], number eps=1e-5) -> Tensor',
  doc: 'RMSNorm：仅按均方根缩放 x/√(mean(x²)+eps)·gamma，无减均值操作（LLaMA 采用）',
  ref: 'Zhang & Sennrich, Root Mean Square Layer Normalization, NeurIPS 2019',
  inv: '输出 RMS ≈ 1（gamma=1 时）；平移 x→x+c 不保持（无去均值）',
  impl: (x, gamma, eps) => {
    expectTensor(x, 'x');
    const e = eps === undefined ? 1e-5 : expectNumber(eps, 'eps');
    const D = x.shape[x.ndim - 1];
    const outer = x.size / D;
    const out = new Float64Array(x.size);
    for (let o = 0; o < outer; o++) {
      const base = o * D;
      let ms = 0;
      for (let j = 0; j < D; j++) ms += x.data[base + j] * x.data[base + j];
      const inv = 1 / Math.sqrt(ms / D + e);
      for (let j = 0; j < D; j++) {
        out[base + j] = x.data[base + j] * inv * (gamma ? gamma.data[j % gamma.size] : 1);
      }
    }
    return new Tensor(out, x.shape);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.norm.instancenorm', arity: 4,
  sig: '(Tensor x[N,C,H,W], Tensor gamma[C], Tensor beta[C], number eps=1e-5) -> Tensor',
  doc: '实例归一化：对每个样本每个通道独立地在空间维 (H,W) 上归一化（风格迁移常用）',
  ref: 'Ulyanov et al., Instance Normalization: The Missing Ingredient for Fast Stylization, 2016',
  inv: '每个 (n,c) 通道的 H·W 个元素均值≈0、方差≈1；与 batch 无关',
  impl: (x, gamma, beta, eps) => {
    expectTensor(x, 'x');
    if (x.ndim < 3) throw new ForgeError('E_SHAPE', 'instancenorm 需要 ndim>=3');
    const e = eps === undefined ? 1e-5 : expectNumber(eps, 'eps');
    const st = _statsKeep(x, x.shape.map((_, i) => i).slice(2), 0);
    const spatial = _prod(x.shape.slice(2));
    const C = x.shape[1];
    if (gamma != null && gamma.size !== C) throw new ForgeError('E_SHAPE', `gamma 元素数应为 ${C}`);
    if (beta != null && beta.size !== C) throw new ForgeError('E_SHAPE', `beta 元素数应为 ${C}`);
    const stm = _stridesOf(st.mean.shape);
    const out = new Float64Array(x.size);
    for (let p = 0; p < x.size; p++) {
      const idx = _decode(p, x.shape);
      let q = 0;
      for (let d = 0; d < x.ndim; d++) if (st.mean.shape[d] !== 1) q += idx[d] * stm[d];
      const c = idx[1];
      const inv = 1 / Math.sqrt(st.var.data[q] + e);
      const z = (x.data[p] - st.mean.data[q]) * inv;
      out[p] = z * (gamma ? gamma.data[c] : 1) + (beta ? beta.data[c] : 0);
    }
    void spatial;
    return new Tensor(out, x.shape);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.norm.groupnorm', arity: 5,
  sig: '(Tensor x[N,C,H,W], Tensor gamma[C], Tensor beta[C], number groups, number eps=1e-5) -> Tensor',
  doc: '组归一化：把 C 个通道分为 groups 组，在每组通道×空间上归一化；groups=1 近似 LayerNorm，groups=C 即 InstanceNorm',
  ref: 'Wu & He, Group Normalization, ECCV 2018',
  inv: '每组 (N/组, C/groups·H·W) 元素均值≈0、方差≈1；groups 必须整除 C',
  impl: (x, gamma, beta, groups, eps) => {
    expectTensor(x, 'x');
    if (x.ndim < 3) throw new ForgeError('E_SHAPE', 'groupnorm 需要 ndim>=3');
    const G = expectNumber(groups, 'groups');
    const [N, C] = x.shape;
    if (!Number.isInteger(G) || G < 1 || C % G !== 0) {
      throw new ForgeError('E_VALUE', `groups=${G} 必须为正整数且整除通道数 ${C}`);
    }
    const e = eps === undefined ? 1e-5 : expectNumber(eps, 'eps');
    const per = C / G;
    const spatial = _prod(x.shape.slice(2));
    if (gamma != null && gamma.size !== C) throw new ForgeError('E_SHAPE', `gamma 元素数应为 ${C}`);
    if (beta != null && beta.size !== C) throw new ForgeError('E_SHAPE', `beta 元素数应为 ${C}`);
    const out = new Float64Array(x.size);
    // 以 (n, group) 为槽位
    const slots = N * G;
    const mean = new Float64Array(slots);
    const varr = new Float64Array(slots);
    const cnt = N * G * per * spatial;
    for (let n = 0; n < N; n++) for (let gg = 0; gg < G; gg++) {
      let s = 0;
      for (let c = gg * per; c < (gg + 1) * per; c++) {
        const base = ((n * C) + c) * spatial;
        for (let i = 0; i < spatial; i++) s += x.data[base + i];
      }
      mean[n * G + gg] = s / (per * spatial);
    }
    for (let n = 0; n < N; n++) for (let gg = 0; gg < G; gg++) {
      let s = 0;
      for (let c = gg * per; c < (gg + 1) * per; c++) {
        const base = ((n * C) + c) * spatial;
        for (let i = 0; i < spatial; i++) { const dv = x.data[base + i] - mean[n * G + gg]; s += dv * dv; }
      }
      varr[n * G + gg] = s / (per * spatial);
    }
    void cnt;
    for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) {
      const gg = Math.floor(c / per);
      const inv = 1 / Math.sqrt(varr[n * G + gg] + e);
      const mu = mean[n * G + gg];
      const base = (n * C + c) * spatial;
      for (let i = 0; i < spatial; i++) {
        const z = (x.data[base + i] - mu) * inv;
        out[base + i] = z * (gamma ? gamma.data[c] : 1) + (beta ? beta.data[c] : 0);
      }
    }
    return new Tensor(out, x.shape);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.norm.weightnorm', arity: 3,
  sig: '(Tensor v[out,in], Tensor g[out], number eps=1e-12) -> Tensor',
  doc: '权重归一化：把权重分解为方向 v/||v|| 与幅度 g，输出每行范数恰为 |g|',
  ref: 'Salimans & Kingma, Weight Normalization: A Simple Reparameterization to Accelerate Training, NIPS 2016',
  inv: '输出第 i 行的 L2 范数 = |g_i|；方向不变：w_i ∝ v_i',
  impl: (v, g, eps) => {
    expectTensor(v, 'v'); expectTensor(g, 'g');
    if (v.ndim !== 2) throw new ForgeError('E_SHAPE', 'weightnorm 的 v 需为 2D [out,in]');
    const e = eps === undefined ? 1e-12 : expectNumber(eps, 'eps');
    const [O, I] = v.shape;
    if (g.size !== O && g.size !== 1) throw new ForgeError('E_SHAPE', `g 元素数 ${g.size} != out ${O}`);
    const out = new Float64Array(O * I);
    for (let i = 0; i < O; i++) {
      let s = 0;
      for (let j = 0; j < I; j++) s += v.data[i * I + j] ** 2;
      const inv = 1 / Math.max(e, Math.sqrt(s));
      const gi = g.data[g.size === 1 ? 0 : i];
      for (let j = 0; j < I; j++) out[i * I + j] = v.data[i * I + j] * inv * gi;
    }
    return new Tensor(out, [O, I]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.norm.spectralnorm', arity: 3,
  sig: '(Tensor W[M,N], number nIter=10, number eps=1e-12) -> {W: Tensor, sigma: number}',
  doc: '谱归一化：幂迭代估计最大奇异值 sigma，返回 W/sigma（谱范数≈1），用于稳定 GAN 判别器',
  ref: 'Miyato et al., Spectral Normalization for Generative Adversarial Networks, ICLR 2018',
  inv: '归一化后矩阵的谱范数 ≈ 1；sigma 收敛到最大奇异值；迭代不发散',
  impl: (W, nIter, eps) => {
    expectTensor(W, 'W');
    if (W.ndim !== 2) throw new ForgeError('E_SHAPE', 'spectralnorm 仅支持 2D');
    const [M, N] = W.shape;
    const it = nIter === undefined ? 10 : expectNumber(nIter, 'nIter');
    const e = eps === undefined ? 1e-12 : expectNumber(eps, 'eps');
    // 确定性初始化（禁止 Math.random）：交替符号向量，几乎不与主奇异向量正交
    let v = new Float64Array(N);
    for (let i = 0; i < N; i++) v[i] = (i % 2 === 0 ? 1 : -1) / Math.sqrt(N);
    let u = new Float64Array(M);
    let sigma = 0;
    for (let k = 0; k < Math.max(1, Math.round(it)); k++) {
      // u ← W v / ||W v||
      for (let i = 0; i < M; i++) {
        let s = 0;
        for (let j = 0; j < N; j++) s += W.data[i * N + j] * v[j];
        u[i] = s;
      }
      let nu = 0;
      for (let i = 0; i < M; i++) nu += u[i] * u[i];
      nu = Math.sqrt(nu);
      if (nu < e) { for (let i = 0; i < M; i++) u[i] = 1 / Math.sqrt(M); nu = 1; }
      for (let i = 0; i < M; i++) u[i] /= nu;
      // v ← Wᵀ u / ||Wᵀ u||
      const nv0 = new Float64Array(N);
      for (let j = 0; j < N; j++) {
        let s = 0;
        for (let i = 0; i < M; i++) s += W.data[i * N + j] * u[i];
        nv0[j] = s;
      }
      let nn = 0;
      for (let j = 0; j < N; j++) nn += nv0[j] * nv0[j];
      nn = Math.sqrt(nn);
      if (nn < e) break;
      for (let j = 0; j < N; j++) v[j] = nv0[j] / nn;
      sigma = nn;
    }
    if (!(sigma > e)) sigma = e;
    const out = new Float64Array(M * N);
    for (let i = 0; i < M * N; i++) out[i] = W.data[i] / sigma;
    return { W: new Tensor(out, [M, N]), sigma };
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.norm.l2normalize', arity: 3,
  sig: '(Tensor x, number axis=-1, number eps=1e-12) -> Tensor',
  doc: 'L2 归一化：沿 axis 除以 L2 范数（常用于对比学习的特征球投影）',
  ref: '对比学习惯例（Chen et al. SimCLR 2020 的投影头归一化）',
  inv: '沿 axis 的 L2 范数 = 1（范数非零时）；零向量映射为零向量',
  impl: (x, axis, eps) => {
    expectTensor(x, 'x');
    const ax0 = axis === undefined ? -1 : axis;
    const ax = ax0 < 0 ? x.ndim + ax0 : ax0;
    if (ax < 0 || ax >= x.ndim) throw new ForgeError('E_SHAPE', `轴 ${ax} 越界`);
    const e = eps === undefined ? 1e-12 : expectNumber(eps, 'eps');
    const D = x.shape[ax];
    const outer = x.size / D;
    const st = _stridesOf(x.shape);
    const out = new Float64Array(x.size);
    for (let o = 0; o < outer; o++) {
      // 槽位展平下标：把 ax 维去掉
      const remain = x.shape.filter((_, i) => i !== ax);
      const idxr = _decode(o, remain.length ? remain : [1]);
      let base = 0;
      let ri = 0;
      for (let d = 0; d < x.ndim; d++) if (d !== ax) base += (idxr[ri++] ?? 0) * st[d];
      let s = 0;
      for (let k = 0; k < D; k++) { const v = x.data[base + k * st[ax]]; s += v * v; }
      const inv = 1 / Math.max(e, Math.sqrt(s));
      for (let k = 0; k < D; k++) out[base + k * st[ax]] = x.data[base + k * st[ax]] * inv;
    }
    return new Tensor(out, x.shape);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.norm.batchStats', arity: 3,
  sig: '(Tensor x, number[] axes, number correction=0) -> {mean: Tensor, var: Tensor}',
  doc: '沿 axes 求均值与方差并保持维数（keepdims）；correction=1 给出无偏（贝塞尔）方差',
  ref: 'Ioffe & Szegedy 2015 的批统计；Welford 1962 数值稳定方差',
  inv: '输出 shape 中被归约轴置 1；correction=1 时为无偏估计；var ≥ 0',
  impl: (x, axes, correction) => {
    expectTensor(x, 'x');
    const ax = Array.isArray(axes) ? axes : [axes === undefined ? 0 : axes];
    const c = correction === undefined ? 0 : expectNumber(correction, 'correction');
    return _statsKeep(x, ax, c);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.norm.runningUpdate', arity: 3,
  sig: '(Tensor running, Tensor batch, number momentum=0.9) -> Tensor',
  doc: '滑动更新 running 统计量：running ← momentum·running + (1−momentum)·batch',
  ref: 'Ioffe & Szegedy 2015 的 running_mean / running_var 更新式',
  inv: 'momentum=0 时等于 batch 值；momentum→1 时保持不变；凸组合保持有界',
  impl: (running, batch, momentum) => {
    expectTensor(running, 'running'); expectTensor(batch, 'batch');
    const m = momentum === undefined ? 0.9 : expectNumber(momentum, 'momentum');
    if (running.size !== batch.size) throw new ForgeError('E_SHAPE', 'running/batch 元素数不符');
    const out = new Float64Array(running.size);
    for (let i = 0; i < running.size; i++) out[i] = m * running.data[i] + (1 - m) * batch.data[i];
    return new Tensor(out, running.shape);
  },
});

// =====================================================================
// 三、层（14 条）
// =====================================================================
defineOp({
  id: nid(), domain: 8, name: 'dl.layer.dense', arity: 3,
  sig: '(Tensor x[N,Di], Tensor W[Di,Do], Tensor b[Do]) -> Tensor[N,Do]',
  doc: '全连接层（GEMM + 偏置）：y = x·W + b，PyTorch nn.Linear / ONNX Gemm 同源',
  ref: 'Rosenblatt 1958 感知机；ONNX Operator Standard: Gemm',
  inv: '输出形状 [N,Do]；b=0 且 W=I 时为恒等映射；对样本维可加（逐行独立）',
  impl: (x, W, b) => {
    expectTensor(x, 'x'); expectTensor(W, 'W');
    if (x.ndim !== 2 || W.ndim !== 2) throw new ForgeError('E_SHAPE', 'dense 需要 2D 输入');
    if (x.shape[1] !== W.shape[0]) throw new ForgeError('E_SHAPE', `dense 内维不符: ${x.shape[1]} vs ${W.shape[0]}`);
    const bt = b == null ? null : expectTensor(b, 'b');
    return _mmBias(x, W, bt);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.layer.conv1d', arity: 6,
  sig: '(Tensor x[N,C,L], Tensor w[OC,IC,K], Tensor b[OC], number stride=1, number padding=0, number dilation=1) -> Tensor[N,OC,Lout]',
  doc: '一维互相关卷积（深度学习惯例，不翻转核）；Lout = ⌊(L+2·pad−dil·(K−1)−1)/stride⌋+1',
  ref: 'LeCun et al. 1998；ONNX Operator Standard: Conv',
  inv: '输出长度满足公式；核为 delta 且 b=0 时近似恒等；padding=0,stride=1,K=1 时 Lo=L',
  impl: (x, w, b, stride, padding, dilation) => {
    expectTensor(x, 'x'); expectTensor(w, 'w');
    if (x.ndim !== 3 || w.ndim !== 3) throw new ForgeError('E_SHAPE', 'conv1d 需要 x[N,C,L] 与 w[OC,IC,K]');
    const [N, C, L] = x.shape; const [OC, IC, K] = w.shape;
    if (C !== IC) throw new ForgeError('E_SHAPE', `conv1d 通道不符: ${C} vs ${IC}`);
    const s = stride === undefined ? 1 : expectNumber(stride, 'stride');
    const p = padding === undefined ? 0 : expectNumber(padding, 'padding');
    const d = dilation === undefined ? 1 : expectNumber(dilation, 'dilation');
    const span = d * (K - 1) + 1;
    const Lo = Math.floor((L + 2 * p - span) / s) + 1;
    if (Lo <= 0) throw new ForgeError('E_SHAPE', 'conv1d 输出长度非正，请检查 stride/padding/kernel');
    const out = new Float64Array(N * OC * Lo);
    const bt = b == null ? null : expectTensor(b, 'b');
    for (let n = 0; n < N; n++) for (let oc = 0; oc < OC; oc++) {
      for (let o = 0; o < Lo; o++) {
        let acc = bt ? bt.data[oc] : 0;
        for (let ic = 0; ic < IC; ic++) {
          const xb = ((n * C) + ic) * L;
          const wb = ((oc * IC) + ic) * K;
          for (let k = 0; k < K; k++) {
            const li = o * s - p + k * d;
            if (li < 0 || li >= L) continue;
            acc += x.data[xb + li] * w.data[wb + k];
          }
        }
        out[((n * OC) + oc) * Lo + o] = acc;
      }
    }
    return new Tensor(out, [N, OC, Lo]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.layer.conv2d', arity: 6,
  sig: '(Tensor x[N,C,H,W], Tensor w[OC,IC,KH,KW], Tensor b[OC], number|[2] stride=1, number|[2] padding=0, number|[2] dilation=1) -> Tensor[N,OC,Ho,Wo]',
  doc: '二维互相关卷积（NCHW）；Ho = ⌊(H+2·padH−dilH·(KH−1)−1)/strideH⌋+1',
  ref: 'LeCun et al., Gradient-Based Learning Applied to Document Recognition, 1998；ONNX Conv',
  inv: '输出空间尺寸满足公式；与 im2col+GEMM 数值等价（误差 <1e-10）；通道数 = OC',
  impl: (x, w, b, stride, padding, dilation) => {
    expectTensor(x, 'x'); expectTensor(w, 'w');
    if (x.ndim !== 4 || w.ndim !== 4) throw new ForgeError('E_SHAPE', 'conv2d 需要 x[N,C,H,W] 与 w[OC,IC,KH,KW]');
    const [N, C, H, Wd] = x.shape; const [OC, IC, KH, KW] = w.shape;
    if (C !== IC) throw new ForgeError('E_SHAPE', `conv2d 通道不符: ${C} vs ${IC}`);
    const [sh, sw] = _pair(stride === undefined ? 1 : stride, 'stride');
    const [ph, pw] = _pair(padding === undefined ? 0 : padding, 'padding');
    const [dh, dw] = _pair(dilation === undefined ? 1 : dilation, 'dilation');
    const spanH = dh * (KH - 1) + 1, spanW = dw * (KW - 1) + 1;
    const Ho = Math.floor((H + 2 * ph - spanH) / sh) + 1;
    const Wo = Math.floor((Wd + 2 * pw - spanW) / sw) + 1;
    if (Ho <= 0 || Wo <= 0) throw new ForgeError('E_SHAPE', 'conv2d 输出尺寸非正');
    const out = new Float64Array(N * OC * Ho * Wo);
    const bt = b == null ? null : expectTensor(b, 'b');
    for (let n = 0; n < N; n++) for (let oc = 0; oc < OC; oc++) {
      for (let oh = 0; oh < Ho; oh++) for (let ow = 0; ow < Wo; ow++) {
        let acc = bt ? bt.data[oc] : 0;
        for (let ic = 0; ic < IC; ic++) {
          const xb = (((n * C) + ic) * H) * Wd;
          const wb = (((oc * IC) + ic) * KH) * KW;
          for (let kh = 0; kh < KH; kh++) {
            const hi = oh * sh - ph + kh * dh;
            if (hi < 0 || hi >= H) continue;
            for (let kw = 0; kw < KW; kw++) {
              const wi = ow * sw - pw + kw * dw;
              if (wi < 0 || wi >= Wd) continue;
              acc += x.data[xb + hi * Wd + wi] * w.data[wb + kh * KW + kw];
            }
          }
        }
        out[(((n * OC) + oc) * Ho + oh) * Wo + ow] = acc;
      }
    }
    return new Tensor(out, [N, OC, Ho, Wo]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.layer.depthwiseConv2d', arity: 5,
  sig: '(Tensor x[N,C,H,W], Tensor w[C,1,KH,KW], Tensor b[C], number|[2] stride=1, number|[2] padding=0) -> Tensor[N,C,Ho,Wo]',
  doc: '深度可分离卷积的空间部分：每个输入通道独立用一个 KH×KW 核卷积，输出通道数不变',
  ref: 'Howard et al., MobileNets: Efficient Convolutional Neural Networks for Mobile Vision, 2017',
  inv: '输出通道数 = 输入通道数；参数量 = C·KH·KW（远小于标准卷积）；等价于 groups=C 的分组卷积',
  impl: (x, w, b, stride, padding) => {
    expectTensor(x, 'x'); expectTensor(w, 'w');
    if (x.ndim !== 4) throw new ForgeError('E_SHAPE', 'depthwiseConv2d 需要 x[N,C,H,W]');
    let wt = w;
    if (w.ndim === 3) wt = w.reshape([w.shape[0], 1, w.shape[1], w.shape[2]]); // 允许 [C,KH,KW]
    if (wt.ndim !== 4 || wt.shape[1] !== 1) throw new ForgeError('E_SHAPE', '核需为 [C,1,KH,KW] 或 [C,KH,KW]');
    const [N, C, H, Wd] = x.shape; const KH = wt.shape[2], KW = wt.shape[3];
    if (wt.shape[0] !== C) throw new ForgeError('E_SHAPE', `核通道 ${wt.shape[0]} != 输入通道 ${C}`);
    const [sh, sw] = _pair(stride === undefined ? 1 : stride, 'stride');
    const [ph, pw] = _pair(padding === undefined ? 0 : padding, 'padding');
    const Ho = Math.floor((H + 2 * ph - KH) / sh) + 1;
    const Wo = Math.floor((Wd + 2 * pw - KW) / sw) + 1;
    if (Ho <= 0 || Wo <= 0) throw new ForgeError('E_SHAPE', 'depthwiseConv2d 输出尺寸非正');
    const out = new Float64Array(N * C * Ho * Wo);
    const bt = b == null ? null : expectTensor(b, 'b');
    for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) {
      const xb = (((n * C) + c) * H) * Wd;
      const wb = c * KH * KW;
      for (let oh = 0; oh < Ho; oh++) for (let ow = 0; ow < Wo; ow++) {
        let acc = bt ? bt.data[c] : 0;
        for (let kh = 0; kh < KH; kh++) {
          const hi = oh * sh - ph + kh;
          if (hi < 0 || hi >= H) continue;
          for (let kw = 0; kw < KW; kw++) {
            const wi = ow * sw - pw + kw;
            if (wi < 0 || wi >= Wd) continue;
            acc += x.data[xb + hi * Wd + wi] * wt.data[wb + kh * KW + kw];
          }
        }
        out[(((n * C) + c) * Ho + oh) * Wo + ow] = acc;
      }
    }
    return new Tensor(out, [N, C, Ho, Wo]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.layer.transposeConv2d', arity: 6,
  sig: '(Tensor x[N,C,H,W], Tensor w[C,OC,KH,KW], Tensor b[OC], number|[2] stride=1, number|[2] padding=0, number|[2] outPadding=0) -> Tensor[N,OC,Ho,Wo]',
  doc: '转置卷积（反卷积）：Ho = (H−1)·stride − 2·padding + KH + outPadding；实现为「输入插零 + 边界补 (KH−1−pad) + 翻转核的互相关」',
  ref: 'Dumoulin & Visin, A Guide to Convolution Arithmetic for Deep Learning, 2016；ONNX ConvTranspose',
  inv: '输出尺寸满足公式；stride=1,pad=0 且核为 delta 时为恒等；与梯度反向传播的对偶卷积一致',
  impl: (x, w, b, stride, padding, outPadding) => {
    expectTensor(x, 'x'); expectTensor(w, 'w');
    if (x.ndim !== 4 || w.ndim !== 4) throw new ForgeError('E_SHAPE', 'transposeConv2d 需要 x[N,C,H,W] 与 w[C,OC,KH,KW]');
    const [N, C, H, Wd] = x.shape; const [IC, OC, KH, KW] = w.shape;
    if (IC !== C) throw new ForgeError('E_SHAPE', `核输入通道 ${IC} != ${C}`);
    const [sh, sw] = _pair(stride === undefined ? 1 : stride, 'stride');
    const [ph, pw] = _pair(padding === undefined ? 0 : padding, 'padding');
    const [oph, opw] = _pair(outPadding === undefined ? 0 : outPadding, 'outPadding');
    const Ho = (H - 1) * sh - 2 * ph + KH + oph;
    const Wo = (Wd - 1) * sw - 2 * pw + KW + opw;
    if (Ho <= 0 || Wo <= 0) throw new ForgeError('E_SHAPE', 'transposeConv2d 输出尺寸非正');
    // 输入按 stride 插零
    const H1 = (H - 1) * sh + 1, W1 = (Wd - 1) * sw + 1;
    const dil = new Float64Array(N * C * H1 * W1);
    for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) for (let i = 0; i < H; i++) for (let j = 0; j < Wd; j++) {
      dil[(((n * C) + c) * H1 + i * sh) * W1 + j * sw] = x.data[(((n * C) + c) * H + i) * Wd + j];
    }
    // 边界补 (KH-1-ph)：等价于卷积的转置
    const padH = KH - 1 - ph, padW = KW - 1 - pw;
    const H2 = H1 + 2 * padH + oph, W2 = W1 + 2 * padW + opw;
    const pad = new Float64Array(N * C * H2 * W2);
    for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) for (let i = 0; i < H1; i++) for (let j = 0; j < W1; j++) {
      pad[(((n * C) + c) * H2 + i + padH) * W2 + j + padW] = dil[(((n * C) + c) * H1 + i) * W1 + j];
    }
    const out = new Float64Array(N * OC * Ho * Wo);
    const bt = b == null ? null : expectTensor(b, 'b');
    for (let n = 0; n < N; n++) for (let oc = 0; oc < OC; oc++) {
      for (let oh = 0; oh < Ho; oh++) for (let ow = 0; ow < Wo; ow++) {
        let acc = bt ? bt.data[oc] : 0;
        for (let ic = 0; ic < C; ic++) {
          const xb = (((n * C) + ic) * H2) * W2;
          const wb = (((ic * OC) + oc) * KH) * KW;
          for (let kh = 0; kh < KH; kh++) {
            const hi = oh + kh;
            if (hi >= H2) continue;
            for (let kw = 0; kw < KW; kw++) {
              const wi = ow + kw;
              if (wi >= W2) continue;
              // 翻转核 → 真正的卷积（而非互相关）
              acc += pad[xb + hi * W2 + wi] * w.data[wb + (KH - 1 - kh) * KW + (KW - 1 - kw)];
            }
          }
        }
        out[(((n * OC) + oc) * Ho + oh) * Wo + ow] = acc;
      }
    }
    return new Tensor(out, [N, OC, Ho, Wo]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.layer.maxpool1d', arity: 3,
  sig: '(Tensor x[N,C,L], number kernel, number stride) -> Tensor[N,C,Lo]',
  doc: '一维最大池化：窗口内取最大值，Lo = ⌊(L−kernel)/stride⌋+1',
  ref: 'Ranzato et al. 2007；LeCun et al. 1998 的子采样层',
  inv: '输出每个元素都是对应窗口的最大值；kernel=1 时为恒等；下界不超过输入最大值',
  impl: (x, kernel, stride) => {
    expectTensor(x, 'x');
    if (x.ndim !== 3) throw new ForgeError('E_SHAPE', 'maxpool1d 需要 x[N,C,L]');
    const [N, C, L] = x.shape;
    const k = expectNumber(kernel, 'kernel');
    const s = stride === undefined ? k : expectNumber(stride, 'stride');
    const Lo = Math.floor((L - k) / s) + 1;
    if (Lo <= 0) throw new ForgeError('E_SHAPE', 'maxpool1d 输出长度非正');
    const out = new Float64Array(N * C * Lo);
    for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) for (let o = 0; o < Lo; o++) {
      let m = -Infinity;
      for (let t = 0; t < k; t++) {
        const v = x.data[((n * C) + c) * L + o * s + t];
        if (v > m) m = v;
      }
      out[((n * C) + c) * Lo + o] = m;
    }
    return new Tensor(out, [N, C, Lo]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.layer.avgpool1d', arity: 3,
  sig: '(Tensor x[N,C,L], number kernel, number stride) -> Tensor[N,C,Lo]',
  doc: '一维平均池化：窗口内取算术平均（计数按完整窗口，边界不补零）',
  ref: 'LeCun et al. 1998；Lin et al., Network In Network, 2014（全局平均池化思想）',
  inv: '输出每个元素落在窗口最小/最大之间；常数输入保持不变',
  impl: (x, kernel, stride) => {
    expectTensor(x, 'x');
    if (x.ndim !== 3) throw new ForgeError('E_SHAPE', 'avgpool1d 需要 x[N,C,L]');
    const [N, C, L] = x.shape;
    const k = expectNumber(kernel, 'kernel');
    const s = stride === undefined ? k : expectNumber(stride, 'stride');
    const Lo = Math.floor((L - k) / s) + 1;
    if (Lo <= 0) throw new ForgeError('E_SHAPE', 'avgpool1d 输出长度非正');
    const out = new Float64Array(N * C * Lo);
    for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) for (let o = 0; o < Lo; o++) {
      let s2 = 0;
      for (let t = 0; t < k; t++) s2 += x.data[((n * C) + c) * L + o * s + t];
      out[((n * C) + c) * Lo + o] = s2 / k;
    }
    return new Tensor(out, [N, C, Lo]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.layer.maxpool2d', arity: 3,
  sig: '(Tensor x[N,C,H,W], number|[2] kernel, number|[2] stride) -> Tensor[N,C,Ho,Wo]',
  doc: '二维最大池化（NCHW）：Ho = ⌊(H−kH)/sH⌋+1',
  ref: 'Ranzato et al., Unsupervised Learning of Invariant Feature Hierarchies, 2007；ONNX MaxPool',
  inv: '输出元素 = 对应 kH×kW 窗口的最大值；平移等变性（步长整除时）',
  impl: (x, kernel, stride) => {
    expectTensor(x, 'x');
    if (x.ndim !== 4) throw new ForgeError('E_SHAPE', 'maxpool2d 需要 x[N,C,H,W]');
    const [N, C, H, Wd] = x.shape;
    const kk = _pair(kernel, 'kernel');
    const ss = _pair(stride === undefined ? kernel : stride, 'stride');
    const Ho = Math.floor((H - kk[0]) / ss[0]) + 1;
    const Wo = Math.floor((Wd - kk[1]) / ss[1]) + 1;
    if (Ho <= 0 || Wo <= 0) throw new ForgeError('E_SHAPE', 'maxpool2d 输出尺寸非正');
    const out = new Float64Array(N * C * Ho * Wo);
    for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) {
      const xb = ((n * C) + c) * H * Wd;
      for (let oh = 0; oh < Ho; oh++) for (let ow = 0; ow < Wo; ow++) {
        let m = -Infinity;
        for (let kh = 0; kh < kk[0]; kh++) for (let kw = 0; kw < kk[1]; kw++) {
          const v = x.data[xb + (oh * ss[0] + kh) * Wd + ow * ss[1] + kw];
          if (v > m) m = v;
        }
        out[(((n * C) + c) * Ho + oh) * Wo + ow] = m;
      }
    }
    return new Tensor(out, [N, C, Ho, Wo]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.layer.avgpool2d', arity: 3,
  sig: '(Tensor x[N,C,H,W], number|[2] kernel, number|[2] stride) -> Tensor[N,C,Ho,Wo]',
  doc: '二维平均池化（NCHW）：窗口内算术平均，替代全连接头以降低参数量',
  ref: 'Lin et al., Network In Network, ICLR 2014；He et al. ResNet 2015',
  inv: '输出元素 = 窗口均值，落在窗口 min/max 之间；常数输入输出不变',
  impl: (x, kernel, stride) => {
    expectTensor(x, 'x');
    if (x.ndim !== 4) throw new ForgeError('E_SHAPE', 'avgpool2d 需要 x[N,C,H,W]');
    const [N, C, H, Wd] = x.shape;
    const kk = _pair(kernel, 'kernel');
    const ss = _pair(stride === undefined ? kernel : stride, 'stride');
    const Ho = Math.floor((H - kk[0]) / ss[0]) + 1;
    const Wo = Math.floor((Wd - kk[1]) / ss[1]) + 1;
    if (Ho <= 0 || Wo <= 0) throw new ForgeError('E_SHAPE', 'avgpool2d 输出尺寸非正');
    const out = new Float64Array(N * C * Ho * Wo);
    const area = kk[0] * kk[1];
    for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) {
      const xb = ((n * C) + c) * H * Wd;
      for (let oh = 0; oh < Ho; oh++) for (let ow = 0; ow < Wo; ow++) {
        let s = 0;
        for (let kh = 0; kh < kk[0]; kh++) for (let kw = 0; kw < kk[1]; kw++) {
          s += x.data[xb + (oh * ss[0] + kh) * Wd + ow * ss[1] + kw];
        }
        out[(((n * C) + c) * Ho + oh) * Wo + ow] = s / area;
      }
    }
    return new Tensor(out, [N, C, Ho, Wo]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.layer.globalAvgPool', arity: 1,
  sig: '(Tensor x[N,C,H,W]) -> Tensor[N,C]',
  doc: '全局平均池化：对每个样本每个通道求整张特征图的均值',
  ref: 'Lin et al., Network In Network, ICLR 2014',
  inv: '输出形状 [N,C]；每个输出 = 对应通道 H·W 个元素的算术平均；与 kernel=(H,W) 的平均池化一致',
  impl: (x) => {
    expectTensor(x, 'x');
    if (x.ndim < 3) throw new ForgeError('E_SHAPE', 'globalAvgPool 需要 ndim>=3');
    const [N, C] = x.shape;
    const spatial = _prod(x.shape.slice(2));
    const out = new Float64Array(N * C);
    for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) {
      let s = 0;
      const base = (n * C + c) * spatial;
      for (let i = 0; i < spatial; i++) s += x.data[base + i];
      out[n * C + c] = s / spatial;
    }
    return new Tensor(out, [N, C]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.layer.flatten', arity: 2,
  sig: '(Tensor x, number startDim=1) -> Tensor[lead, rest]',
  doc: '把 [startDim, ndim) 的维压平成一维，输出二维 [lead, rest]；卷积特征送入全连接前的标准操作',
  ref: 'PyTorch nn.Flatten / Keras Flatten 层',
  inv: '元素总数不变且行主序顺序不变；startDim=1 时 [N,C,H,W] → [N, C·H·W]',
  impl: (x, startDim) => {
    expectTensor(x, 'x');
    const sd = startDim === undefined ? 1 : expectNumber(startDim, 'startDim');
    if (sd < 0 || sd > x.ndim) throw new ForgeError('E_SHAPE', `startDim ${sd} 越界`);
    const lead = Math.max(1, _prod(x.shape.slice(0, sd)));
    const rest = Math.max(1, _prod(x.shape.slice(sd)));
    return new Tensor(x.data.slice(), [lead, rest]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.layer.unflatten', arity: 2,
  sig: '(Tensor x[N,P], number[] shape) -> Tensor[N,...shape]',
  doc: 'flatten 的逆操作：把最后一维还原成给定形状（prod(shape) 必须等于 P）',
  ref: 'PyTorch nn.Unflatten / TensorFlow reshape 语义',
  inv: 'unflatten(flatten(x)) = x（形状与元素均一致）；元素总数守恒',
  impl: (x, shape) => {
    expectTensor(x, 'x');
    if (!Array.isArray(shape) || shape.length === 0) throw new ForgeError('E_TYPE', 'shape 需为非空数组');
    const P = x.shape[x.ndim - 1];
    if (_prod(shape) !== P) throw new ForgeError('E_SHAPE', `unflatten 元素数不符: prod(shape)=${_prod(shape)} vs ${P}`);
    const lead = x.size / P;
    const leadShape = x.shape.slice(0, x.ndim - 1);
    return new Tensor(x.data.slice(), [...leadShape, ...shape]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.layer.embedding', arity: 2,
  sig: '(Tensor idx[B,T], Tensor table[V,C]) -> Tensor[B,T,C]',
  doc: '查表嵌入：把整数 id 映射为表向量；等价于以 one-hot 行向量左乘嵌入矩阵',
  ref: 'Bengio et al., A Neural Probabilistic Language Model, JMLR 2003；Mikolov et al. word2vec 2013',
  inv: '输出形状 [B,T,C]；out[b,t,:] = table[idx[b,t],:]；id 越界报错',
  impl: (idx, table) => {
    expectTensor(idx, 'idx'); expectTensor(table, 'table');
    if (table.ndim !== 2) throw new ForgeError('E_SHAPE', 'table 需为 2D [V,C]');
    const [V, C] = table.shape;
    const outShape = [...idx.shape, C];
    const out = new Float64Array(_prod(outShape));
    for (let i = 0; i < idx.size; i++) {
      const id = idx.data[i];
      if (!Number.isInteger(id) || id < 0 || id >= V) {
        throw new ForgeError('E_VALUE', `embedding id 越界或非整数: ${id}（词表大小 ${V}）`);
      }
      for (let c = 0; c < C; c++) out[i * C + c] = table.data[id * C + c];
    }
    return new Tensor(out, outShape);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.layer.patching', arity: 2,
  sig: '(Tensor img[N,C,H,W], number patch) -> Tensor[N, L, C·patch·patch]',
  doc: 'ViT 式分块：把图像切成不重叠 patch 并拉平为 token 序列，L = (H/patch)·(W/patch)',
  ref: 'Dosovitskiy et al., An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, ICLR 2021',
  inv: '输出形状 [N, L, C·p²]；元素总数与输入一致；patch=1 时 token 数 = H·W',
  impl: (img, patch) => {
    expectTensor(img, 'img');
    if (img.ndim !== 4) throw new ForgeError('E_SHAPE', 'patching 需要 [N,C,H,W]');
    const [N, C, H, Wd] = img.shape;
    const p = expectNumber(patch, 'patch');
    if (!Number.isInteger(p) || p < 1 || H % p !== 0 || Wd % p !== 0) {
      throw new ForgeError('E_VALUE', `patch=${p} 必须整除 H=${H} 与 W=${Wd}`);
    }
    const gh = H / p, gw = Wd / p, L = gh * gw, D = C * p * p;
    const out = new Float64Array(N * L * D);
    for (let n = 0; n < N; n++) for (let gi = 0; gi < gh; gi++) for (let gj = 0; gj < gw; gj++) {
      const tok = (gi * gw + gj) * D;
      let o = 0;
      for (let c = 0; c < C; c++) for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) {
        out[(n * L) * D + tok + o++] = img.data[(((n * C) + c) * H + gi * p + i) * Wd + gj * p + j];
      }
    }
    return new Tensor(out, [N, L, D]);
  },
});

// =====================================================================
// 四、正则化（7 条）
// =====================================================================
defineOp({
  id: nid(), domain: 8, name: 'dl.reg.dropout', arity: 4,
  sig: '(Tensor x, number p, number seed, number training=1) -> Tensor',
  doc: 'Inverted dropout：训练期以概率 p 置零并放大 1/(1−p) 保持期望不变；推理期恒等',
  ref: 'Srivastava et al., Dropout: A Simple Way to Prevent Neural Networks from Overfitting, JMLR 2014',
  inv: '训练期 E[out] = x；非零元素恰被放大 1/(1−p)；training=0 时 out = x',
  impl: (x, p, seed, training) => {
    expectTensor(x, 'x');
    const pr = expectNumber(p, 'p');
    if (!(pr >= 0 && pr < 1)) throw new ForgeError('E_VALUE', `dropout 概率 p 必须 ∈ [0,1)，当前 ${pr}`);
    const train = training === undefined ? 1 : training;
    if (!train) return x.clone();
    const rng = createRng(seed === undefined ? 20261014 : (seed >>> 0));
    const scale = 1 / (1 - pr);
    return x.map((v) => (rng.next() >= pr ? v * scale : 0));
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.reg.dropPath', arity: 4,
  sig: '(Tensor x, number p, number seed, number training=1) -> Tensor',
  doc: 'DropPath：整条残差分支按样本（第 0 维）同生共死地丢弃，是 Stochastic Depth 的分支版实现',
  ref: 'Larsson et al., FractalNet: Ultra-Deep Neural Networks without Residuals, ICLR 2017；Huang et al., Deep Networks with Stochastic Depth, ECCV 2016',
  inv: '同一样本的所有元素共享同一掩码；E[out]=x；training=0 时恒等',
  impl: (x, p, seed, training) => {
    expectTensor(x, 'x');
    const pr = expectNumber(p, 'p');
    if (!(pr >= 0 && pr < 1)) throw new ForgeError('E_VALUE', `dropPath 概率 p 必须 ∈ [0,1)，当前 ${pr}`);
    const train = training === undefined ? 1 : training;
    if (!train) return x.clone();
    const rng = createRng(seed === undefined ? 20261014 : (seed >>> 0));
    const N = x.shape[0];
    const keep = new Float64Array(N);
    for (let i = 0; i < N; i++) keep[i] = rng.next() >= pr ? 1 / (1 - pr) : 0;
    const inner = x.size / N;
    return x.map((v, i) => v * keep[Math.floor(i / inner)]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.reg.stochasticDepth', arity: 5,
  sig: '(Tensor x, Tensor residual, number p, number seed, number training=1) -> Tensor',
  doc: '随机深度：out = x + b·residual，b ∈ {0, 1/(1−p)} 按样本随机，训练期逐层递减存活率',
  ref: 'Huang et al., Deep Networks with Stochastic Depth, ECCV 2016',
  inv: 'E[out] = x + residual；b 只有 0 与 1/(1−p) 两种取值；training=0 时 out = x + residual',
  impl: (x, residual, p, seed, training) => {
    expectTensor(x, 'x'); expectTensor(residual, 'residual');
    if (x.size !== residual.size) throw new ForgeError('E_SHAPE', 'x 与 residual 元素数不符');
    const pr = expectNumber(p, 'p');
    if (!(pr >= 0 && pr < 1)) throw new ForgeError('E_VALUE', `p 必须 ∈ [0,1)，当前 ${pr}`);
    const train = training === undefined ? 1 : training;
    const N = x.shape[0];
    const inner = x.size / N;
    const keep = new Float64Array(N);
    if (train) {
      const rng = createRng(seed === undefined ? 20261014 : (seed >>> 0));
      for (let i = 0; i < N; i++) keep[i] = rng.next() >= pr ? 1 / (1 - pr) : 0;
    } else {
      keep.fill(1);
    }
    const out = new Float64Array(x.size);
    for (let i = 0; i < x.size; i++) {
      const k = keep[Math.floor(i / inner)];
      out[i] = x.data[i] + k * residual.data[i % residual.size];
    }
    return new Tensor(out, x.shape);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.reg.l1', arity: 2,
  sig: '(Tensor W, number lambda) -> number',
  doc: 'L1 正则项 = lambda·Σ|w|，诱导稀疏解（LASSO 式）',
  ref: 'Tibshirani, Regression Shrinkage and Selection via the Lasso, JRSS-B 1996；深度学习权重衰减实践',
  inv: 'l1 ≥ 0；W=0 时为 0；对 W 缩放 c 倍则值放大 |c| 倍',
  impl: (W, lambda) => {
    expectTensor(W, 'W');
    const l = expectNumber(lambda, 'lambda');
    let s = 0;
    for (let i = 0; i < W.size; i++) s += Math.abs(W.data[i]);
    return l * s;
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.reg.l2', arity: 2,
  sig: '(Tensor W, number lambda) -> number',
  doc: 'L2 正则项 = 0.5·lambda·Σw²（岭回归式；0.5 使梯度恰为 lambda·w）',
  ref: 'Hoerl & Kennard, Ridge Regression, 1970；Krogh & Hertz 1992 权重衰减',
  inv: 'l2 ≥ 0；梯度 ∂/∂w = lambda·w；W=0 时为 0',
  impl: (W, lambda) => {
    expectTensor(W, 'W');
    const l = expectNumber(lambda, 'lambda');
    let s = 0;
    for (let i = 0; i < W.size; i++) s += W.data[i] * W.data[i];
    return 0.5 * l * s;
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.reg.l1l2', arity: 3,
  sig: '(Tensor W, number l1, number l2) -> number',
  doc: 'Elastic Net 式混合正则 = l1·Σ|w| + 0.5·l2·Σw²',
  ref: 'Zou & Hastie, Regularization and Variable Selection via the Elastic Net, JRSS-B 2005',
  inv: 'l1=0 退化为 l2 项；l2=0 退化为 l1 项；非负且凸',
  impl: (W, l1, l2) => {
    expectTensor(W, 'W');
    const a = expectNumber(l1, 'l1'); const b = expectNumber(l2, 'l2');
    let s1 = 0, s2 = 0;
    for (let i = 0; i < W.size; i++) { s1 += Math.abs(W.data[i]); s2 += W.data[i] * W.data[i]; }
    return a * s1 + 0.5 * b * s2;
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.reg.mixup', arity: 3,
  sig: '(Tensor a, Tensor b, number lam) -> Tensor',
  doc: 'Mixup 线性插值：out = lam·a + (1−lam)·b；lam 通常采样自 Beta(alpha, alpha)',
  ref: 'Zhang et al., mixup: Beyond Empirical Risk Minimization, ICLR 2018',
  inv: 'lam=1 时 out=a，lam=0 时 out=b；lam∈[0,1] 时输出落在 a、b 的凸包内',
  impl: (a, b, lam) => {
    expectTensor(a, 'a'); expectTensor(b, 'b');
    if (a.size !== b.size) throw new ForgeError('E_SHAPE', 'mixup 两输入元素数不符');
    const l = expectNumber(lam, 'lam');
    const out = new Float64Array(a.size);
    for (let i = 0; i < a.size; i++) out[i] = l * a.data[i] + (1 - l) * b.data[i];
    return new Tensor(out, a.shape);
  },
});

// =====================================================================
// 五、注意力（16 条）
// =====================================================================
// 通用缩放点积注意力（内核）：Q[K] 形状 [B,Tq,D]，V [B,Tk,Dv]
function _sdpaCore(Q, K, V, mask, scale) {
  const [B, Tq, D] = Q.shape;
  const Tk = K.shape[1];
  const Dv = V.shape[2];
  const s = scale === undefined || scale === null ? 1 / Math.sqrt(D) : scale;
  const scores = new Float64Array(B * Tq * Tk);
  for (let b = 0; b < B; b++) for (let i = 0; i < Tq; i++) for (let j = 0; j < Tk; j++) {
    let acc = 0;
    const qb = (b * Tq + i) * D, kb = (b * Tk + j) * D;
    for (let d = 0; d < D; d++) acc += Q.data[qb + d] * K.data[kb + d];
    scores[(b * Tq + i) * Tk + j] = acc * s;
  }
  // 加性/乘性掩码统一处理：mask 为 0/1 乘性遮罩，被遮处加 −1e9
  let msk = null;
  if (mask != null) {
    expectTensor(mask, 'mask');
    msk = _bcast(mask, [B, Tq, Tk]);
  }
  if (msk) for (let p = 0; p < scores.length; p++) {
    scores[p] = msk.data[p] !== 0 ? scores[p] : scores[p] - 1e9;
  }
  const attn = _softmaxLast(Tensor.fromFlat(scores, [B, Tq, Tk]));
  const out = new Float64Array(B * Tq * Dv);
  for (let b = 0; b < B; b++) for (let i = 0; i < Tq; i++) for (let d = 0; d < Dv; d++) {
    let acc = 0;
    for (let j = 0; j < Tk; j++) acc += attn.data[(b * Tq + i) * Tk + j] * V.data[(b * Tk + j) * Dv + d];
    out[(b * Tq + i) * Dv + d] = acc;
  }
  return { out: new Tensor(out, [B, Tq, Dv]), attn };
}
// 批量 head 展开：[B,T,C] -> [B,H,T,D]
function _headSplit(x, heads) {
  const [B, T, C] = x.shape;
  if (!Number.isInteger(heads) || heads < 1 || C % heads !== 0) {
    throw new ForgeError('E_VALUE', `heads=${heads} 必须整除模型维 ${C}`);
  }
  const D = C / heads;
  const out = new Float64Array(B * heads * T * D);
  for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) for (let h = 0; h < heads; h++) for (let d = 0; d < D; d++) {
    out[(((b * heads) + h) * T + t) * D + d] = x.data[(b * T + t) * C + h * D + d];
  }
  return new Tensor(out, [B, heads, T, D]);
}
// 批量 head 合并：[B,H,T,D] -> [B,T,H·D]
function _headMerge(y) {
  const [B, H, T, D] = y.shape;
  const C = H * D;
  const out = new Float64Array(B * T * C);
  for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) for (let h = 0; h < H; h++) for (let d = 0; d < D; d++) {
    out[(b * T + t) * C + h * D + d] = y.data[(((b * H) + h) * T + t) * D + d];
  }
  return new Tensor(out, [B, T, C]);
}
// 多头注意力内核：输入已投影的 QKV 张量（[B,T,C]）与 heads
function _mhaCore(Q, K, V, heads, mask) {
  const [B, Tq, C] = Q.shape;
  const Tk = K.shape[1];
  const D = C / heads;
  const qh = _headSplit(Q, heads), kh = _headSplit(K, heads), vh = _headSplit(V, heads);
  const outH = new Float64Array(B * heads * Tq * D);
  const attnH = new Float64Array(B * heads * Tq * Tk);
  for (let h = 0; h < heads; h++) {
    const q = new Float64Array(B * Tq * D), k = new Float64Array(B * Tk * D), v = new Float64Array(B * Tk * D);
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < Tq; t++) for (let d = 0; d < D; d++) q[(b * Tq + t) * D + d] = qh.data[(((b * heads) + h) * Tq + t) * D + d];
      for (let t = 0; t < Tk; t++) for (let d = 0; d < D; d++) { k[(b * Tk + t) * D + d] = kh.data[(((b * heads) + h) * Tk + t) * D + d]; v[(b * Tk + t) * D + d] = vh.data[(((b * heads) + h) * Tk + t) * D + d]; }
    }
    const r = _sdpaCore(Tensor.fromFlat(q, [B, Tq, D]), Tensor.fromFlat(k, [B, Tk, D]), Tensor.fromFlat(v, [B, Tk, D]), mask, 1 / Math.sqrt(D));
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < Tq; t++) for (let d = 0; d < D; d++) outH[(((b * heads) + h) * Tq + t) * D + d] = r.out.data[(b * Tq + t) * D + d];
      for (let t = 0; t < Tq; t++) for (let t2 = 0; t2 < Tk; t2++) attnH[(((b * heads) + h) * Tq + t) * Tk + t2] = r.attn.data[(b * Tq + t) * Tk + t2];
    }
  }
  return { out: _headMerge(Tensor.fromFlat(outH, [B, heads, Tq, D])), attn: Tensor.fromFlat(attnH, [B, heads, Tq, Tk]), D };
}

defineOp({
  id: nid(), domain: 8, name: 'dl.attn.causalMask', arity: 1,
  sig: '(number T) -> Tensor[T,T]',
  doc: '因果掩码：下三角（含主对角）为 1，严格上三角为 0，用于自回归解码',
  ref: 'Vaswani et al., Attention Is All You Need, NeurIPS 2017（decoder 的 subsequent mask）',
  inv: '下三角含对角为 1；严格上三角（j>i）为 0；每行元素和 = i+1',
  impl: (T) => {
    const t = expectNumber(T, 'T');
    if (!Number.isInteger(t) || t < 1) throw new ForgeError('E_VALUE', 'T 必须为正整数');
    const out = new Float64Array(t * t);
    for (let i = 0; i < t; i++) for (let j = 0; j <= i; j++) out[i * t + j] = 1;
    return new Tensor(out, [t, t]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.attn.paddingMask', arity: 2,
  sig: '(Tensor lengths[B], number T) -> Tensor[B,T]',
  doc: 'padding 掩码：长度以内的位置为 1（有效），其余为 0（填充），用于变长序列批训练',
  ref: 'Transformer 变长批处理惯例（Vaswani et al. 2017 的工程实现）',
  inv: '第 b 行前 lengths[b] 个为 1、其余为 0；每行和为 lengths[b]',
  impl: (lengths, T) => {
    expectTensor(lengths, 'lengths');
    const t = expectNumber(T, 'T');
    if (!Number.isInteger(t) || t < 1) throw new ForgeError('E_VALUE', 'T 必须为正整数');
    const B = lengths.size;
    const out = new Float64Array(B * t);
    for (let b = 0; b < B; b++) {
      const len = Math.max(0, Math.min(t, Math.floor(lengths.data[b])));
      for (let j = 0; j < len; j++) out[b * t + j] = 1;
    }
    return new Tensor(out, [B, t]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.attn.scaledDot', arity: 3,
  sig: '(Tensor Q[B,Tq,D], Tensor K[B,Tk,D], Tensor V[B,Tk,Dv], Tensor mask?, number scale=1/√D) -> {out, attn}',
  doc: '缩放点积注意力：attn = softmax(QKᵀ·scale + 掩码)，out = attn·V；被掩位置加 −1e9 后再归一化',
  ref: 'Vaswani et al., Attention Is All You Need, NeurIPS 2017',
  inv: 'attn 每行和为 1 且非负；被掩位置权重恰为 0；scale 不影响 argmax',
  impl: (Q, K, V, mask = null, scale = undefined) => {
    expectTensor(Q, 'Q'); expectTensor(K, 'K'); expectTensor(V, 'V');
    if (Q.ndim !== 3 || K.ndim !== 3 || V.ndim !== 3) throw new ForgeError('E_SHAPE', '注意力输入需为 3D [B,T,D]');
    if (Q.shape[0] !== K.shape[0] || K.shape[0] !== V.shape[0]) throw new ForgeError('E_SHAPE', 'batch 维不一致');
    if (Q.shape[2] !== K.shape[2]) throw new ForgeError('E_SHAPE', 'Q/K 头维不一致');
    if (K.shape[1] !== V.shape[1]) throw new ForgeError('E_SHAPE', 'K/V 时间维不一致');
    if (scale === undefined) scale = 1 / Math.sqrt(Q.shape[2]);
    return _sdpaCore(Q, K, V, mask, scale);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.attn.multiHead', arity: 7,
  sig: '(Tensor x[B,T,C], Tensor Wq[C,C], Tensor Wk[C,C], Tensor Wv[C,C], Tensor Wo[C,C], number heads, Tensor mask?) -> Tensor[B,T,C]',
  doc: '多头自注意力（含 QKV/O 四组投影）：先投影再分头并行注意力，最后拼接并输出投影',
  ref: 'Vaswani et al., Attention Is All You Need, NeurIPS 2017',
  inv: '输出形状 [B,T,C]；heads 整除 C；heads=1 时退化为单头注意力',
  impl: (x, Wq, Wk, Wv, Wo, heads, mask) => {
    expectTensor(x, 'x');
    if (x.ndim !== 3) throw new ForgeError('E_SHAPE', 'multiHead 需要 x[B,T,C]');
    const [B, T, C] = x.shape;
    const xf = x.reshape([B * T, C]);
    const Q = _mm(xf, expectTensor(Wq, 'Wq')).reshape([B, T, Wq.shape[1]]);
    const K = _mm(xf, expectTensor(Wk, 'Wk')).reshape([B, T, Wk.shape[1]]);
    const V = _mm(xf, expectTensor(Wv, 'Wv')).reshape([B, T, Wv.shape[1]]);
    const r = _mhaCore(Q, K, V, heads, mask);
    const merged = r.out.reshape([B * T, r.out.shape[2]]);
    return _mm(merged, expectTensor(Wo, 'Wo')).reshape([B, T, Wo.shape[1]]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.attn.selfAttention', arity: 3,
  sig: '(Tensor x[B,T,C], number heads, Tensor mask?) -> {out: Tensor[B,T,C], attn: Tensor[B,H,T,T]}',
  doc: '无投影自注意力：Q=K=V=x（恒等投影），用于分析/教学与残差流上的纯注意力',
  ref: 'Vaswani et al. 2017 的注意力核心（去掉线性投影的裸形式）',
  inv: '输出形状 [B,T,C]；attn 每行和为 1；mask 位置权重为 0',
  impl: (x, heads, mask) => {
    expectTensor(x, 'x');
    if (x.ndim !== 3) throw new ForgeError('E_SHAPE', 'selfAttention 需要 x[B,T,C]');
    const r = _mhaCore(x, x, x, heads, mask);
    return { out: r.out, attn: r.attn };
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.attn.crossAttention', arity: 4,
  sig: '(Tensor q[B,Tq,C], Tensor kv[B,Tk,C], number heads, Tensor mask?) -> {out, attn}',
  doc: '交叉注意力：查询来自 q、键值来自 kv（恒等投影），编码器-解码器桥接结构',
  ref: 'Vaswani et al. 2017 的 encoder-decoder attention',
  inv: '输出时间维 = Tq（查询长度）；键值时间维 = Tk；attn 每行和为 1',
  impl: (q, kv, heads, mask) => {
    expectTensor(q, 'q'); expectTensor(kv, 'kv');
    if (q.ndim !== 3 || kv.ndim !== 3) throw new ForgeError('E_SHAPE', 'crossAttention 输入需为 3D');
    if (q.shape[2] !== kv.shape[2]) throw new ForgeError('E_SHAPE', 'q/kv 模型维不一致');
    const r = _mhaCore(q, kv, kv, heads, mask);
    return { out: r.out, attn: r.attn };
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.attn.groupedQuery', arity: 6,
  sig: '(Tensor Q[B,Tq,H·D], Tensor K[B,Tk,Hkv·D], Tensor V[B,Tk,Hkv·D], number heads, number kvHeads, Tensor mask?) -> {out, attn}',
  doc: '分组查询注意力 GQA：H 个 query 头共享 Hkv 个 KV 头（第 h 个 query 头映射到第 ⌊h·Hkv/H⌋ 个 KV 头），大幅压缩 KV 缓存',
  ref: 'Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, EMNLP 2023',
  inv: 'heads 必须被 kvHeads 整除；kvHeads=heads 退化为 MHA，kvHeads=1 退化为 MQA；输出 [B,Tq,H·D]',
  impl: (Q, K, V, heads, kvHeads, mask) => {
    expectTensor(Q, 'Q'); expectTensor(K, 'K'); expectTensor(V, 'V');
    const H = expectNumber(heads, 'heads'); const Hkv = expectNumber(kvHeads, 'kvHeads');
    if (H % Hkv !== 0) throw new ForgeError('E_VALUE', `heads=${H} 必须被 kvHeads=${Hkv} 整除`);
    const [B, Tq, C] = Q.shape;
    const Tk = K.shape[1];
    const D = C / H;
    // K/V 按 kvHeads 拆头（注意 [B,T,H·D] → [B,H,T,D] 是带转置的重排，不能简单 reinterpret）
    const qh = _headSplit(Q, H);
    const kh = _headSplit(K, Hkv);
    const vh = _headSplit(V, Hkv);
    const outH = new Float64Array(B * H * Tq * D);
    const attnH = new Float64Array(B * H * Tq * Tk);
    for (let h = 0; h < H; h++) {
      const g = Math.floor(h * Hkv / H);
      const q = new Float64Array(B * Tq * D), k = new Float64Array(B * Tk * D), v = new Float64Array(B * Tk * D);
      for (let b = 0; b < B; b++) {
        for (let t = 0; t < Tq; t++) for (let d = 0; d < D; d++) q[(b * Tq + t) * D + d] = qh.data[(((b * H) + h) * Tq + t) * D + d];
        for (let t = 0; t < Tk; t++) for (let d = 0; d < D; d++) { k[(b * Tk + t) * D + d] = kh.data[(((b * Hkv) + g) * Tk + t) * D + d]; v[(b * Tk + t) * D + d] = vh.data[(((b * Hkv) + g) * Tk + t) * D + d]; }
      }
      const r = _sdpaCore(Tensor.fromFlat(q, [B, Tq, D]), Tensor.fromFlat(k, [B, Tk, D]), Tensor.fromFlat(v, [B, Tk, D]), mask, 1 / Math.sqrt(D));
      for (let b = 0; b < B; b++) {
        for (let t = 0; t < Tq; t++) for (let d = 0; d < D; d++) outH[(((b * H) + h) * Tq + t) * D + d] = r.out.data[(b * Tq + t) * D + d];
        for (let t = 0; t < Tq; t++) for (let t2 = 0; t2 < Tk; t2++) attnH[(((b * H) + h) * Tq + t) * Tk + t2] = r.attn.data[(b * Tq + t) * Tk + t2];
      }
    }
    return { out: _headMerge(Tensor.fromFlat(outH, [B, H, Tq, D])), attn: Tensor.fromFlat(attnH, [B, H, Tq, Tk]) };
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.attn.rotaryEmbed', arity: 3,
  sig: '(number dim, number maxLen, number base=10000) -> {cos: Tensor[maxLen,dim/2], sin: Tensor[maxLen,dim/2]}',
  doc: 'RoPE 频率表：θ_m,i = m·base^{−2i/dim}，返回半长 cos/sin 表（GPT-J/LLaMA 的半分块排布）',
  ref: 'Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding, 2021',
  inv: 'cos²+sin²=1；m=0 时 cos=1, sin=0；频率随 i 几何衰减 base^{−2i/dim}',
  impl: (dim, maxLen, base) => {
    const D = expectNumber(dim, 'dim');
    const L = expectNumber(maxLen, 'maxLen');
    if (!Number.isInteger(D) || D < 2 || D % 2 !== 0) throw new ForgeError('E_VALUE', 'dim 必须为 >=2 的偶数');
    if (!Number.isInteger(L) || L < 1) throw new ForgeError('E_VALUE', 'maxLen 必须为正整数');
    const b0 = base === undefined ? 10000 : expectNumber(base, 'base');
    const half = D / 2;
    const cos = new Float64Array(L * half), sin = new Float64Array(L * half);
    for (let m = 0; m < L; m++) for (let i = 0; i < half; i++) {
      const theta = m * Math.pow(b0, -2 * i / D);
      cos[m * half + i] = Math.cos(theta);
      sin[m * half + i] = Math.sin(theta);
    }
    return { cos: new Tensor(cos, [L, half]), sin: new Tensor(sin, [L, half]) };
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.attn.rotaryApply', arity: 4,
  sig: '(Tensor x[...,T,...,D], Tensor cos[maxLen,D/2], Tensor sin[maxLen,D/2], number start=0) -> Tensor',
  doc: '应用 RoPE（半分块形式）：x=(a,b) → (a·cos − b·sin, b·cos + a·sin)，位置取自第 1 维并偏移 start',
  ref: 'Su et al., RoFormer, 2021；GPT-J/LLaMA 的 half-split 实现惯例',
  inv: '范数保持 ||R(x)|| = ||x||；内积 ⟨R(q,m),R(k,n)⟩ 只依赖相对位置 m−n',
  impl: (x, cos, sin, start) => {
    expectTensor(x, 'x'); expectTensor(cos, 'cos'); expectTensor(sin, 'sin');
    if (x.ndim < 2) throw new ForgeError('E_SHAPE', 'rotaryApply 需要 ndim>=2（时间维在轴 1）');
    const D = x.shape[x.ndim - 1];
    const half = D / 2;
    if (!Number.isInteger(half) || D % 2 !== 0) throw new ForgeError('E_VALUE', '最后一维必须为偶数');
    if (cos.shape[1] !== half || sin.shape[1] !== half) throw new ForgeError('E_SHAPE', 'cos/sin 的半长与 D/2 不符');
    const st2 = start === undefined ? 0 : expectNumber(start, 'start');
    const T = x.shape[1];
    if (T + st2 > cos.shape[0]) throw new ForgeError('E_VALUE', `位置 ${T + st2} 超出 maxLen ${cos.shape[0]}`);
    const st = _stridesOf(x.shape);
    const out = new Float64Array(x.size);
    for (let p = 0; p < x.size; p++) {
      const idx = _decode(p, x.shape);
      const t = idx[1] + st2;
      const d = idx[x.ndim - 1];
      let base = 0;
      for (let k = 0; k < x.ndim - 1; k++) base += idx[k] * st[k];
      const cb = t * half, sb = t * half;
      if (d < half) {
        out[p] = x.data[base + d] * cos.data[cb + d] - x.data[base + d + half] * sin.data[sb + d];
      } else {
        const dd = d - half;
        out[p] = x.data[base + d] * cos.data[cb + dd] + x.data[base + dd] * sin.data[sb + dd];
      }
    }
    return new Tensor(out, x.shape);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.attn.alibiBias', arity: 2,
  sig: '(number heads, number T) -> Tensor[heads,T,T]',
  doc: 'ALiBi 加性位置偏置：bias[h,i,j] = −m_h·|i−j|，斜率 m_h = 2^{−8(h+1)/heads} 几何分布，无需位置嵌入即可外推',
  ref: 'Press et al., Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation, ICLR 2022',
  inv: '对角恒为 0；bias ≤ 0；|i−j| 越大越负；不同头斜率互不相同且几何衰减',
  impl: (heads, T) => {
    const H = expectNumber(heads, 'heads'); const t = expectNumber(T, 'T');
    if (!Number.isInteger(H) || H < 1) throw new ForgeError('E_VALUE', 'heads 必须为正整数');
    if (!Number.isInteger(t) || t < 1) throw new ForgeError('E_VALUE', 'T 必须为正整数');
    const out = new Float64Array(H * t * t);
    for (let h = 0; h < H; h++) {
      const slope = Math.pow(2, -8 * (h + 1) / H);
      for (let i = 0; i < t; i++) for (let j = 0; j < t; j++) out[(h * t + i) * t + j] = -slope * Math.abs(i - j);
    }
    return new Tensor(out, [H, t, t]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.attn.relativeBias', arity: 3,
  sig: '(number T, number numBuckets=32, number seed=20261014) -> Tensor[T,T]',
  doc: '相对位置偏置：把距离 |i−j| 分桶（小距离精确、大距离对数分桶）后查可学习表，双向对称',
  ref: 'Raffel et al., Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, JMLR 2020',
  inv: 'B[i][j] 只依赖 |i−j|；B 对称；对角元素全部相同',
  impl: (T, numBuckets, seed) => {
    const t = expectNumber(T, 'T');
    if (!Number.isInteger(t) || t < 1) throw new ForgeError('E_VALUE', 'T 必须为正整数');
    const nb = numBuckets === undefined ? 32 : expectNumber(numBuckets, 'numBuckets');
    if (!Number.isInteger(nb) || nb < 2 || nb % 2 !== 0) throw new ForgeError('E_VALUE', 'numBuckets 必须为 >=2 的偶数');
    const rng = createRng((seed === undefined ? 20261014 : seed) >>> 0);
    const table = new Float64Array(nb);
    for (let i = 0; i < nb; i++) table[i] = 0.02 * rng.normal();
    const maxExact = Math.max(1, Math.floor(nb / 2));
    const maxDist = 128;
    // 对称分桶：小距离逐个精确分桶，大距离按对数压缩到剩余桶
    const bucketOf = (dist) => {
      if (dist < maxExact) return dist;
      const ratio = Math.log(dist / maxExact) / Math.log(maxDist / maxExact);
      const span = nb - maxExact;
      return maxExact + Math.min(span - 1, Math.floor(ratio * span));
    };
    const out = new Float64Array(t * t);
    for (let i = 0; i < t; i++) for (let j = 0; j < t; j++) out[i * t + j] = table[bucketOf(Math.abs(j - i))];
    return new Tensor(out, [t, t]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.attn.kvCacheInit', arity: 4,
  sig: '(number B, number heads, number maxLen, number headDim) -> {k: Tensor[B,H,0,D], v: Tensor[B,H,0,D], len: 0, maxLen, heads, headDim}',
  doc: '初始化 KV 缓存（长度 0 的空张量），供自回归推理逐 token 增量解码',
  ref: '自回归推理 KV 缓存惯例（Pope et al., Efficiently Scaling Transformer Inference, 2022）',
  inv: 'len=0；k/v 形状 [B,H,0,D]；append 后 len 单调增且不超过 maxLen',
  impl: (B, heads, maxLen, headDim) => {
    const b = expectNumber(B, 'B'); const h = expectNumber(heads, 'heads');
    const m = expectNumber(maxLen, 'maxLen'); const d = expectNumber(headDim, 'headDim');
    for (const [n, v] of [['B', b], ['heads', h], ['maxLen', m], ['headDim', d]]) {
      if (!Number.isInteger(v) || v < (n === 'maxLen' ? 0 : 1)) throw new ForgeError('E_VALUE', `${n} 取值非法: ${v}`);
    }
    return {
      k: Tensor.zeros([b, h, 0, d]), v: Tensor.zeros([b, h, 0, d]),
      len: 0, maxLen: m, heads: h, headDim: d,
    };
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.attn.kvCacheAppend', arity: 3,
  sig: '(object cache, Tensor kNew[B,H,Tn,D], Tensor vNew[B,H,Tn,D]) -> object',
  doc: '向 KV 缓存尾部追加 Tn 个新 token 的键与值，返回新缓存（纯函数，不改入参）',
  ref: '增量解码 KV 缓存（Pope et al. 2022；HuggingFace generate 实现）',
  inv: '新 len = 旧 len + Tn；累积结果与一次性全序列 K/V 完全相同；超过 maxLen 报错',
  impl: (cache, kNew, vNew) => {
    if (cache == null || typeof cache !== 'object') throw new ForgeError('E_TYPE', 'cache 必须为对象');
    expectTensor(kNew, 'kNew'); expectTensor(vNew, 'vNew');
    if (kNew.shape[0] !== cache.k.shape[0] || kNew.shape[1] !== cache.heads) {
      throw new ForgeError('E_SHAPE', 'kNew 的 [B,H] 与缓存不符');
    }
    if (kNew.shape[2] !== vNew.shape[2]) throw new ForgeError('E_SHAPE', 'kNew/vNew 时间维不一致');
    const Tn = kNew.shape[2];
    if (cache.len + Tn > cache.maxLen) throw new ForgeError('E_LIMIT', `KV 缓存溢出: ${cache.len}+${Tn} > maxLen ${cache.maxLen}`);
    const [B, H, oldT, D] = cache.k.shape;
    const newT = oldT + Tn;
    const k = new Float64Array(B * H * newT * D), v = new Float64Array(B * H * newT * D);
    // 旧段需按「时间维变长」重新定位：每个 (b,h) 块在块内连续，整体做定长搬移
    const blockOld = oldT * D, blockNew = newT * D;
    for (let i = 0; i < B * H; i++) {
      k.set(cache.k.data.subarray(i * blockOld, (i + 1) * blockOld), i * blockNew);
      v.set(cache.v.data.subarray(i * blockOld, (i + 1) * blockOld), i * blockNew);
    }
    for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) for (let t = 0; t < Tn; t++) for (let d = 0; d < D; d++) {
      k[(((b * H) + h) * newT + oldT + t) * D + d] = kNew.data[(((b * H) + h) * Tn + t) * D + d];
      v[(((b * H) + h) * newT + oldT + t) * D + d] = vNew.data[(((b * H) + h) * Tn + t) * D + d];
    }
    return {
      k: new Tensor(k, [B, H, newT, D]), v: new Tensor(v, [B, H, newT, D]),
      len: cache.len + Tn, maxLen: cache.maxLen, heads: H, headDim: D,
    };
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.attn.flashTiling', arity: 5,
  sig: '(Tensor Q[B,Tq,D], Tensor K[B,Tk,D], Tensor V[B,Tk,Dv], number blockRow, number blockCol) -> {out, lse}',
  doc: 'FlashAttention 式分块：按 (blockRow, blockCol) 分块并用 online-softmax 增量更新，避免物化 T×T 注意力矩阵',
  ref: 'Dao et al., FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, NeurIPS 2022',
  inv: '输出与 dl.attn.scaledDot 数值一致（约 1e-12）；lse = logsumexp(scores)；不物化完整注意力矩阵',
  impl: (Q, K, V, blockRow, blockCol) => {
    expectTensor(Q, 'Q'); expectTensor(K, 'K'); expectTensor(V, 'V');
    if (Q.ndim !== 3 || K.ndim !== 3 || V.ndim !== 3) throw new ForgeError('E_SHAPE', 'flashTiling 输入需为 3D');
    const [B, Tq, D] = Q.shape; const Tk = K.shape[1]; const Dv = V.shape[2];
    const br = Math.max(1, Math.min(Tq, Math.floor(blockRow === undefined ? 16 : expectNumber(blockRow, 'blockRow'))));
    const bc = Math.max(1, Math.min(Tk, Math.floor(blockCol === undefined ? 16 : expectNumber(blockCol, 'blockCol'))));
    const scale = 1 / Math.sqrt(D);
    const out = new Float64Array(B * Tq * Dv);
    const lse = new Float64Array(B * Tq);
    for (let b = 0; b < B; b++) {
      for (let r0 = 0; r0 < Tq; r0 += br) {
        const r1 = Math.min(Tq, r0 + br);
        const m = new Float64Array(r1 - r0).fill(-Infinity);
        const l = new Float64Array(r1 - r0);
        const acc = new Float64Array((r1 - r0) * Dv);
        const blk = new Float64Array(bc); // 当前列块的分数缓存（避免重复计算 QKᵀ）
        for (let c0 = 0; c0 < Tk; c0 += bc) {
          const c1 = Math.min(Tk, c0 + bc);
          for (let i = r0; i < r1; i++) {
            const ii = i - r0;
            // 1) 本列块的分数与本块最大值
            let mx = m[ii];
            for (let j = c0; j < c1; j++) {
              let s = 0;
              for (let d = 0; d < D; d++) s += Q.data[(b * Tq + i) * D + d] * K.data[(b * Tk + j) * D + d];
              const sv = s * scale;
              blk[j - c0] = sv;
              if (sv > mx) mx = sv;
            }
            const mNew = mx;
            // 2) 先把历史累积量按新基准缩放（exp(m_old − m_new)）
            const corr = m[ii] === -Infinity ? 0 : Math.exp(m[ii] - mNew);
            for (let d = 0; d < Dv; d++) acc[ii * Dv + d] *= corr;
            // 3) 再把本块贡献按新基准累加（不再乘 corr）
            let ls = 0;
            for (let j = c0; j < c1; j++) {
              const p = Math.exp(blk[j - c0] - mNew);
              ls += p;
              for (let d = 0; d < Dv; d++) acc[ii * Dv + d] += p * V.data[(b * Tk + j) * Dv + d];
            }
            l[ii] = l[ii] * corr + ls;
            m[ii] = mNew;
          }
        }
        for (let i = r0; i < r1; i++) {
          const ii = i - r0;
          const inv = 1 / Math.max(1e-300, l[ii]);
          for (let d = 0; d < Dv; d++) out[(b * Tq + i) * Dv + d] = acc[ii * Dv + d] * inv;
          lse[b * Tq + i] = m[ii] + Math.log(Math.max(1e-300, l[ii]));
        }
      }
    }
    return { out: new Tensor(out, [B, Tq, Dv]), lse: new Tensor(lse, [B, Tq, 1]) };
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.attn.entropy', arity: 1,
  sig: '(Tensor attn[...,T]) -> Tensor[...,1]',
  doc: '注意力分布香农熵 H = −Σ p·log(p+eps)，衡量注意力集中程度（0=one-hot，log T=均匀）',
  ref: 'Shannon 1948；注意力可解释性分析（ entropy 作为注意力塌缩指标）',
  inv: '0 ≤ H ≤ log T；one-hot 时 H=0；均匀分布时 H=log T',
  impl: (attn) => {
    expectTensor(attn, 'attn');
    const T = attn.shape[attn.ndim - 1];
    const outer = attn.size / T;
    const outShape = attn.shape.slice(0, -1).concat([1]);
    const out = new Float64Array(outer);
    for (let o = 0; o < outer; o++) {
      let h = 0;
      for (let j = 0; j < T; j++) {
        const p = attn.data[o * T + j];
        if (p > 0) h -= p * Math.log(p); // p=0 项贡献 0（0·log0 取极限 0）
      }
      out[o] = h;
    }
    return new Tensor(out, outShape.length ? outShape : [1]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.attn.headSplit', arity: 2,
  sig: '(Tensor x[B,T,C], number heads) -> Tensor[B,H,T,D]',
  doc: '把模型维 C 拆成 heads×headDim：out[b,h,t,d] = x[b,t,h·D+d]',
  ref: 'Vaswani et al. 2017 的多头 reshape 惯例',
  inv: 'heads × headDim = 模型维 C；元素总数守恒；与 _headMerge 互为逆操作',
  impl: (x, heads) => {
    expectTensor(x, 'x');
    if (x.ndim !== 3) throw new ForgeError('E_SHAPE', 'headSplit 需要 x[B,T,C]');
    const h = expectNumber(heads, 'heads');
    return _headSplit(x, h);
  },
});

// =====================================================================
// 六、序列模型（8 条）
// =====================================================================
// 通用：z[m, kH] 的第 g 个门 -> [m, H]（PyTorch 的 i,f,g,o 门序）
function _pickGate(z, H, g) {
  const m = z.shape[0];
  const out = new Float64Array(m * H);
  for (let i = 0; i < m; i++) for (let j = 0; j < H; j++) out[i * H + j] = z.data[i * (z.shape[1]) + g * H + j];
  return new Tensor(out, [m, H]);
}
const _sigmoidT = (t) => t.map((v) => 1 / (1 + Math.exp(-v)));
const _tanhT = (t) => t.map(Math.tanh);
const _actByName = (t, name) => {
  if (name === null || name === undefined || name === 'none' || name === 'linear') return t;
  if (name === 'tanh') return _tanhT(t);
  if (name === 'relu') return t.map((v) => (v > 0 ? v : 0));
  if (name === 'sigmoid') return _sigmoidT(t);
  if (name === 'gelu') return t.map((v) => 0.5 * v * (1 + exec('num.erf', [v / _SQRT2])));
  throw new ForgeError('E_VALUE', `未知激活: ${name}`);
};

defineOp({
  id: nid(), domain: 8, name: 'dl.seq.rnnCell', arity: 6,
  sig: "(Tensor x[N,Di], Tensor h[N,Dh], Tensor Wx[Di,Dh], Tensor Wh[Dh,Dh], Tensor b[Dh], string act='tanh') -> Tensor[N,Dh]",
  doc: 'Elman RNN 单元：h′ = act(x·Wx + h·Wh + b)',
  ref: 'Elman, Finding Structure in Time, Cognitive Science 1990；Rumelhart et al. 1986',
  inv: '输出形状 [N,Dh]；W=0,b=0 时 h′=act(0)；梯度沿时间连乘（易衰减/爆炸）',
  impl: (x, h, Wx, Wh, b, act) => {
    expectTensor(x, 'x'); expectTensor(h, 'h');
    const a = act === undefined ? 'tanh' : act;
    const z = _mmBias(x, expectTensor(Wx, 'Wx'), expectTensor(b, 'b'));
    const zh = _mm(h, expectTensor(Wh, 'Wh'));
    return _actByName(z.map((v, i) => v + zh.data[i]), a);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.seq.gruCell', arity: 6,
  sig: '(Tensor x[N,Di], Tensor h[N,H], Tensor Wx[Di,3H], Tensor Wh[H,3H], Tensor bx[3H], Tensor bh[3H]) -> Tensor[N,H]',
  doc: 'GRU 单元：z=σ(更新门)、r=σ(重置门)、n=tanh(候选)，h′=(1−z)·h + z·n',
  ref: 'Cho et al., Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation, EMNLP 2014',
  inv: 'z=0 时 h′=h（无更新）；z=1 时 h′=n（完全替换）；h′ 落在 h 与 n 的凸组合内',
  impl: (x, h, Wx, Wh, bx, bh) => {
    expectTensor(x, 'x'); expectTensor(h, 'h');
    const zx = _mmBias(x, expectTensor(Wx, 'Wx'), expectTensor(bx, 'bx'));
    const zh = _mmBias(h, expectTensor(Wh, 'Wh'), expectTensor(bh, 'bh'));
    const H = h.shape[1];
    const zi = _sigmoidT(zx.map((v, i) => v + zh.data[i]));
    const z = _pickGate(zi, H, 0), r = _pickGate(zi, H, 1);
    // 候选：n = tanh(xWn + bxn + r ⊙ (hUn + bhn))
    const nx = _pickGate(zx, H, 2);
    const nhp = _pickGate(zh, H, 2);
    const n = _tanhT(nx.map((v, i) => v + r.data[i] * nhp.data[i]));
    return h.map((v, i) => (1 - z.data[i]) * v + z.data[i] * n.data[i]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.seq.lstmCell', arity: 6,
  sig: '(Tensor x[N,Di], Tensor h[N,H], Tensor c[N,H], Tensor W[Di,4H], Tensor U[H,4H], Tensor b[4H]) -> {h: Tensor[N,H], c: Tensor[N,H]}',
  doc: 'LSTM 单元：i/f/o 为 sigmoid 门，g 为 tanh 候选；c′ = f⊙c + i⊙g，h′ = o⊙tanh(c′)',
  ref: 'Hochreiter & Schmidhuber, Long Short-Term Memory, Neural Computation 1997',
  inv: 'f=1,i=0 时 c 不变（恒定误差流）；c′ 与 h′ 形状 [N,H]；|h′| ≤ 1',
  impl: (x, h, c, W, U, b) => {
    expectTensor(x, 'x'); expectTensor(h, 'h'); expectTensor(c, 'c');
    const H = h.shape[1];
    const z = _mmBias(x, expectTensor(W, 'W'), expectTensor(b, 'b'));
    const zh = _mm(h, expectTensor(U, 'U'));
    const gz = z.map((v, i) => v + zh.data[i]);
    const i2 = _sigmoidT(_pickGate(gz, H, 0));
    const f2 = _sigmoidT(_pickGate(gz, H, 1));
    const g2 = _tanhT(_pickGate(gz, H, 2));
    const o2 = _sigmoidT(_pickGate(gz, H, 3));
    const nc = c.map((v, k) => f2.data[k] * v + i2.data[k] * g2.data[k]);
    const nh = o2.map((v, k) => v * Math.tanh(nc.data[k]));
    return { h: nh, c: nc };
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.seq.rnnForward', arity: 6,
  sig: "(Tensor x[B,T,Di], Tensor h0[B,H], Tensor Wx[Di,H], Tensor Wh[H,H], Tensor b[H], string act='tanh') -> {out: Tensor[B,T,H], hT: Tensor[B,H]}",
  doc: '沿时间维展开 RNN：逐时间步调用 rnnCell，返回全部隐状态与最终隐状态',
  ref: 'Elman 1990；PyTorch nn.RNN 的前向语义',
  inv: 'out[:,t,:] = 第 t 步隐状态；hT = out[:,T−1,:]；T=1 时退化为单元计算',
  impl: (x, h0, Wx, Wh, b, act) => {
    expectTensor(x, 'x'); expectTensor(h0, 'h0');
    if (x.ndim !== 3) throw new ForgeError('E_SHAPE', 'rnnForward 需要 x[B,T,Di]');
    const T = x.shape[1];
    let h = h0;
    const list = [];
    for (let t = 0; t < T; t++) {
      h = exec('dl.seq.rnnCell', [_timeSlice(x, t), h, Wx, Wh, b, act === undefined ? 'tanh' : act]);
      list.push(h);
    }
    return { out: _stackTime(list), hT: h };
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.seq.gruForward', arity: 6,
  sig: '(Tensor x[B,T,Di], Tensor h0[B,H], Tensor Wx[Di,3H], Tensor Wh[H,3H], Tensor bx[3H], Tensor bh[3H]) -> {out, hT}',
  doc: '沿时间维展开 GRU，返回全部隐状态序列与末状态',
  ref: 'Cho et al. 2014；PyTorch nn.GRU 前向语义',
  inv: 'out[:,t,:] = 第 t 步隐状态；hT = out[:,T−1,:]；隐状态有界（|h| ≤ max(|h0|,1)）',
  impl: (x, h0, Wx, Wh, bx, bh) => {
    expectTensor(x, 'x'); expectTensor(h0, 'h0');
    if (x.ndim !== 3) throw new ForgeError('E_SHAPE', 'gruForward 需要 x[B,T,Di]');
    const T = x.shape[1];
    let h = h0;
    const list = [];
    for (let t = 0; t < T; t++) {
      h = exec('dl.seq.gruCell', [_timeSlice(x, t), h, Wx, Wh, bx, bh]);
      list.push(h);
    }
    return { out: _stackTime(list), hT: h };
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.seq.lstmForward', arity: 6,
  sig: '(Tensor x[B,T,Di], Tensor h0[B,H], Tensor c0[B,H], Tensor W[Di,4H], Tensor U[H,4H], Tensor b[4H]) -> {out, hT, cT}',
  doc: '沿时间维展开 LSTM，返回隐状态序列、末隐状态与末细胞状态',
  ref: 'Hochreiter & Schmidhuber 1997；PyTorch nn.LSTM 前向语义',
  inv: 'out[:,t,:] = 第 t 步隐状态；|h| ≤ 1；f=1,i=0 时 c 沿时间恒定',
  impl: (x, h0, c0, W, U, b) => {
    expectTensor(x, 'x'); expectTensor(h0, 'h0'); expectTensor(c0, 'c0');
    if (x.ndim !== 3) throw new ForgeError('E_SHAPE', 'lstmForward 需要 x[B,T,Di]');
    const T = x.shape[1];
    let h = h0, c = c0;
    const list = [];
    for (let t = 0; t < T; t++) {
      const r = exec('dl.seq.lstmCell', [_timeSlice(x, t), h, c, W, U, b]);
      h = r.h; c = r.c;
      list.push(h);
    }
    return { out: _stackTime(list), hT: h, cT: c };
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.seq.bidirectional', arity: 2,
  sig: '(Tensor xf[B,T,H], Tensor xb[B,T,H]) -> Tensor[B,T,2H]',
  doc: '双向拼接：把正向与反向（通常已各自翻转时间）的输出沿特征维拼接',
  ref: 'Schuster & Paliwal, Bidirectional Recurrent Neural Networks, IEEE TSP 1997',
  inv: '输出最后一维 = 2H；前半段等于 xf，后半段等于 xb；元素总数守恒',
  impl: (xf, xb) => {
    expectTensor(xf, 'xf'); expectTensor(xb, 'xb');
    if (xf.ndim !== 3 || xb.ndim !== 3) throw new ForgeError('E_SHAPE', 'bidirectional 需要 3D [B,T,H]');
    const [B, T, H] = xf.shape; const Hb = xb.shape[2];
    if (xf.shape[0] !== xb.shape[0] || xf.shape[1] !== xb.shape[1]) {
      throw new ForgeError('E_SHAPE', '两路输出的 [B,T] 不一致');
    }
    const out = new Float64Array(B * T * (H + Hb));
    for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) {
      const o = (b * T + t) * (H + Hb);
      for (let d = 0; d < H; d++) out[o + d] = xf.data[(b * T + t) * H + d];
      for (let d = 0; d < Hb; d++) out[o + H + d] = xb.data[(b * T + t) * Hb + d];
    }
    return new Tensor(out, [B, T, H + Hb]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.seq.scan', arity: 4,
  sig: '(Tensor x[B,T,D], Function f, Tensor h0, number reverse=0) -> {out: Tensor[B,T,Dh], hT: Tensor}',
  doc: '通用 carry 扫描（JAX lax.scan 语义）：h_{t+1} = f(h_t, x_t)，reverse=1 时逆序扫描',
  ref: 'JAX lax.scan 语义；Hochreiter & Schmidhuber 1997 的时间展开抽象',
  inv: '输出时间步数 = T；reverse=1 时输出时间序反转；f 为恒等时 out 恒为 h0',
  impl: (x, f, h0, reverse) => {
    expectTensor(x, 'x'); expectTensor(h0, 'h0');
    if (typeof f !== 'function') throw new ForgeError('E_TYPE', 'f 必须为函数 (h, xt) -> h');
    if (x.ndim !== 3) throw new ForgeError('E_SHAPE', 'scan 需要 x[B,T,D]');
    const T = x.shape[1];
    const rev = reverse === undefined ? 0 : reverse;
    let h = h0;
    const list = new Array(T);
    for (let k = 0; k < T; k++) {
      const t = rev ? T - 1 - k : k;
      h = f(h, _timeSlice(x, t));
      if (!(h instanceof Tensor)) throw new ForgeError('E_TYPE', 'f 必须返回 Tensor');
      list[k] = h;
    }
    return { out: _stackTime(list), hT: h };
  },
});

// =====================================================================
// 七、损失函数（14 条）
// =====================================================================
// 读取类别标签张量并做越界检查
function _labels(target, C) {
  expectTensor(target, 'target');
  const out = new Array(target.size);
  for (let i = 0; i < target.size; i++) {
    const v = target.data[i];
    if (!Number.isInteger(v) || v < 0 || v >= C) {
      throw new ForgeError('E_VALUE', `标签越界或非整数: ${v}（类别数 ${C}）`);
    }
    out[i] = v;
  }
  return out;
}
// 同形状逐元素损失 + 归约
function _elemLoss(pred, target, fn) {
  expectTensor(pred, 'pred'); expectTensor(target, 'target');
  if (pred.size !== target.size) throw new ForgeError('E_SHAPE', 'pred/target 元素数不符');
  const out = new Float64Array(pred.size);
  for (let i = 0; i < pred.size; i++) out[i] = fn(pred.data[i], target.data[i]);
  return out;
}

defineOp({
  id: nid(), domain: 8, name: 'dl.loss.mse', arity: 3,
  sig: "(Tensor pred, Tensor target, string reduction='mean') -> number",
  doc: '均方误差 = mean((pred−target)²)（或 sum）',
  ref: 'Gauss 1809 最小二乘；PyTorch nn.MSELoss',
  inv: 'mse ≥ 0；pred=target 时为 0；对残差缩放 c 倍则值放大 c² 倍',
  impl: (pred, target, reduction) => {
    const a = _elemLoss(pred, target, (p, t) => (p - t) * (p - t));
    return _reduce(a, reduction);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.loss.mae', arity: 3,
  sig: "(Tensor pred, Tensor target, string reduction='mean') -> number",
  doc: '平均绝对误差 L1 = mean(|pred−target|)，对离群点比 MSE 更稳健',
  ref: 'Laplace 1774 最小绝对偏差；PyTorch nn.L1Loss',
  inv: 'mae ≥ 0；pred=target 时为 0；满足三角不等式',
  impl: (pred, target, reduction) => {
    const a = _elemLoss(pred, target, (p, t) => Math.abs(p - t));
    return _reduce(a, reduction);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.loss.huber', arity: 4,
  sig: "(Tensor pred, Tensor target, number delta=1, string reduction='mean') -> number",
  doc: 'Huber 损失：|r|≤delta 时为 0.5r²，否则 delta(|r|−0.5·delta)，兼顾 L1 稳健与 L2 光滑',
  ref: 'Huber, Robust Estimation of a Location Parameter, Ann. Math. Statist. 1964',
  inv: 'huber ≥ 0 且连续可微；delta→∞ 趋于 0.5·MSE·(元素数)；|r|≤delta 时等于 0.5r²',
  impl: (pred, target, delta, reduction) => {
    const d = delta === undefined ? 1 : expectNumber(delta, 'delta');
    if (!(d > 0)) throw new ForgeError('E_VALUE', 'delta 必须 > 0');
    const a = _elemLoss(pred, target, (p, t) => {
      const r = Math.abs(p - t);
      return r <= d ? 0.5 * r * r : d * (r - 0.5 * d);
    });
    return _reduce(a, reduction);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.loss.bce', arity: 4,
  sig: "(Tensor pred∈(0,1), Tensor target∈{0,1}, string reduction='mean', number eps=1e-12) -> number",
  doc: '二元交叉熵 = −[t·log(p) + (1−t)·log(1−p)]，输入须为概率',
  ref: '标准对数似然；PyTorch nn.BCELoss',
  inv: 'bce ≥ 0（p,t∈[0,1]）；p=t 且 t∈{0,1} 时为 0；关于 p=t=0.5 对称',
  impl: (pred, target, reduction, eps) => {
    const e = eps === undefined ? 1e-12 : expectNumber(eps, 'eps');
    const a = _elemLoss(pred, target, (p, t) => {
      const pc = Math.min(1 - e, Math.max(e, p));
      return -(t * Math.log(pc) + (1 - t) * Math.log(1 - pc));
    });
    return _reduce(a, reduction);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.loss.bceWithLogits', arity: 3,
  sig: "(Tensor logits, Tensor target∈{0,1}, string reduction='mean') -> number",
  doc: '数值稳定的 BCE-with-logits：直接用 logits 计算，避免先 sigmoid 造成的下溢',
  ref: 'PyTorch nn.BCEWithLogitsLoss（log-sum-exp 稳定化）',
  inv: '与 sigmoid(logits) 后再算 bce 一致（误差 <1e-12）；logits=0 时每元素 = ln2',
  impl: (logits, target, reduction) => {
    const a = _elemLoss(logits, target, (z, t) => {
      // max(z,0) − z·t + log(1+e^{−|z|})
      return Math.max(z, 0) - z * t + Math.log1p(Math.exp(-Math.abs(z)));
    });
    return _reduce(a, reduction);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.loss.ce', arity: 4,
  sig: "(Tensor prob[N,C], Tensor target[N], string reduction='mean', number eps=1e-12) -> number",
  doc: '交叉熵（输入为概率分布）= −log(p_{correct})',
  ref: 'Shannon 1948；PyTorch 的 NLLLoss 与概率输入的交叉熵',
  inv: 'ce = −log(p_correct)；p_correct=1 时为 0；p_correct→0 时 →∞',
  impl: (prob, target, reduction, eps) => {
    expectTensor(prob, 'prob');
    if (prob.ndim !== 2) throw new ForgeError('E_SHAPE', 'ce 需要 prob[N,C]');
    const [N, C] = prob.shape;
    const lab = _labels(target, C);
    if (lab.length !== N) throw new ForgeError('E_SHAPE', `标签数 ${lab.length} != 样本数 ${N}`);
    const e = eps === undefined ? 1e-12 : expectNumber(eps, 'eps');
    const a = new Float64Array(N);
    for (let n = 0; n < N; n++) a[n] = -Math.log(Math.max(e, prob.data[n * C + lab[n]]));
    return _reduce(a, reduction);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.loss.ceWithLogits', arity: 3,
  sig: "(Tensor logits[N,C], Tensor target[N], string reduction='mean') -> number",
  doc: 'logits 上的交叉熵：内部先做稳定 log-softmax 再取负对数似然',
  ref: 'PyTorch nn.CrossEntropyLoss（LogSoftmax + NLLLoss 融合）',
  inv: '等于 ce(softmax(logits), target)；平移不变：logits 加常数结果不变',
  impl: (logits, target, reduction) => {
    expectTensor(logits, 'logits');
    if (logits.ndim !== 2) throw new ForgeError('E_SHAPE', 'ceWithLogits 需要 logits[N,C]');
    const [N, C] = logits.shape;
    const lab = _labels(target, C);
    if (lab.length !== N) throw new ForgeError('E_SHAPE', `标签数 ${lab.length} != 样本数 ${N}`);
    const lp = _logSoftmaxLast(logits);
    const a = new Float64Array(N);
    for (let n = 0; n < N; n++) a[n] = -lp.data[n * C + lab[n]];
    return _reduce(a, reduction);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.loss.nll', arity: 3,
  sig: "(Tensor logProb[N,C], Tensor target[N], string reduction='mean') -> number",
  doc: '负对数似然 = −logProb[n, target[n]]（输入须为对数概率）',
  ref: 'PyTorch nn.NLLLoss；Fisher 1912 极大似然',
  inv: 'nll = −log p_correct；与 ceWithLogits(logits)=nll(logSoftmax(logits)) 一致',
  impl: (logProb, target, reduction) => {
    expectTensor(logProb, 'logProb');
    if (logProb.ndim !== 2) throw new ForgeError('E_SHAPE', 'nll 需要 logProb[N,C]');
    const [N, C] = logProb.shape;
    const lab = _labels(target, C);
    if (lab.length !== N) throw new ForgeError('E_SHAPE', `标签数 ${lab.length} != 样本数 ${N}`);
    const a = new Float64Array(N);
    for (let n = 0; n < N; n++) a[n] = -logProb.data[n * C + lab[n]];
    return _reduce(a, reduction);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.loss.klDiv', arity: 3,
  sig: "(Tensor logQ[N,C]（近似分布的对数概率）, Tensor P[N,C]（目标分布概率）, string reduction='mean') -> number",
  doc: 'KL 散度 Σ P·(log P − log Q)，按 batch 维归约（PyTorch 的 batchmean 语义）',
  ref: 'Kullback & Leibler, On Information and Sufficiency, Ann. Math. Statist. 1951',
  inv: 'KL ≥ 0（Gibbs 不等式）；P=Q 时为 0；非对称：KL(P||Q) ≠ KL(Q||P)',
  impl: (logQ, P, reduction) => {
    expectTensor(logQ, 'logQ'); expectTensor(P, 'P');
    if (logQ.ndim !== 2 || P.ndim !== 2) throw new ForgeError('E_SHAPE', 'klDiv 需要 2D 输入');
    if (logQ.size !== P.size) throw new ForgeError('E_SHAPE', 'logQ/P 元素数不符');
    const N = logQ.shape[0], C = logQ.shape[1];
    const a = new Float64Array(N);
    for (let n = 0; n < N; n++) {
      let s = 0;
      for (let c = 0; c < C; c++) {
        const p = P.data[n * C + c];
        if (p > 0) s += p * (Math.log(p) - logQ.data[n * C + c]);
      }
      a[n] = s;
    }
    return _reduce(a, reduction);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.loss.focal', arity: 5,
  sig: "(Tensor logits[N,C], Tensor target[N], number gamma=2, number alpha=0.25, string reduction='mean') -> number",
  doc: 'Focal Loss = −alpha·(1−p_t)^gamma·log(p_t)，抑制易分样本梯度、聚焦难样本',
  ref: 'Lin et al., Focal Loss for Dense Object Detection, ICCV 2017',
  inv: 'gamma=0,alpha=1 时退化为普通交叉熵；p_t=1 时为 0；gamma 越大对易样本压制越强',
  impl: (logits, target, gamma, alpha, reduction) => {
    expectTensor(logits, 'logits');
    if (logits.ndim !== 2) throw new ForgeError('E_SHAPE', 'focal 需要 logits[N,C]');
    const [N, C] = logits.shape;
    const lab = _labels(target, C);
    if (lab.length !== N) throw new ForgeError('E_SHAPE', `标签数 ${lab.length} != 样本数 ${N}`);
    const g = gamma === undefined ? 2 : expectNumber(gamma, 'gamma');
    const al = alpha === undefined ? 0.25 : expectNumber(alpha, 'alpha');
    const p = _softmaxLast(logits);
    const a = new Float64Array(N);
    for (let n = 0; n < N; n++) {
      const pt = Math.max(1e-12, p.data[n * C + lab[n]]);
      a[n] = -al * Math.pow(1 - pt, g) * Math.log(pt);
    }
    return _reduce(a, reduction);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.loss.dice', arity: 3,
  sig: '(Tensor pred∈[0,1], Tensor target∈{0,1}, number eps=1e-6) -> number',
  doc: 'Dice 损失 = 1 − 2|P∩T| / (|P|+|T|)，类别极度不均衡的分割任务常用',
  ref: 'Milletari et al., V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation, 3DV 2016',
  inv: 'dice ∈ [0,1]；完全重合时为 0；完全不重合时为 1；与样本顺序无关',
  impl: (pred, target, eps) => {
    const e = eps === undefined ? 1e-6 : expectNumber(eps, 'eps');
    expectTensor(pred, 'pred'); expectTensor(target, 'target');
    if (pred.size !== target.size) throw new ForgeError('E_SHAPE', 'pred/target 元素数不符');
    let inter = 0, sp = 0, st = 0;
    for (let i = 0; i < pred.size; i++) {
      inter += pred.data[i] * target.data[i];
      sp += pred.data[i];
      st += target.data[i];
    }
    return 1 - (2 * inter + e) / (sp + st + e);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.loss.ntxent', arity: 3,
  sig: '(Tensor z1[N,D], Tensor z2[N,D], number temperature=0.5) -> number',
  doc: 'SimCLR 的 NT-Xent 对比损失：2N 个视图两两相似度，正例为同一样本的另一视图，分母排除自身',
  ref: 'Chen et al., A Simple Framework for Contrastive Learning of Visual Representations (SimCLR), ICML 2020',
  inv: 'loss ≥ 0；两视图完全一致且互为正交向量时 loss = log(2N−1)；温度越低越聚焦难负例',
  impl: (z1, z2, temperature) => {
    expectTensor(z1, 'z1'); expectTensor(z2, 'z2');
    if (z1.ndim !== 2 || z2.ndim !== 2) throw new ForgeError('E_SHAPE', 'ntxent 需要 2D 输入');
    if (z1.shape[0] !== z2.shape[0] || z1.shape[1] !== z2.shape[1]) {
      throw new ForgeError('E_SHAPE', 'z1/z2 形状必须一致');
    }
    const N = z1.shape[0], D = z1.shape[1];
    const T = temperature === undefined ? 0.5 : expectNumber(temperature, 'temperature');
    if (!(T > 0)) throw new ForgeError('E_VALUE', 'temperature 必须 > 0');
    // L2 归一化后拼接 [2N, D]
    const z = new Float64Array(2 * N * D);
    for (let i = 0; i < 2 * N; i++) {
      const src = i < N ? z1.data : z2.data;
      const off = (i < N ? i : i - N) * D;
      let s = 0;
      for (let d = 0; d < D; d++) s += src[off + d] ** 2;
      const inv = 1 / Math.max(1e-12, Math.sqrt(s));
      for (let d = 0; d < D; d++) z[i * D + d] = src[off + d] * inv;
    }
    const M = 2 * N;
    let total = 0;
    for (let i = 0; i < M; i++) {
      const pos = i < N ? i + N : i - N;
      // 稳定化：减去行最大值
      const sim = new Float64Array(M);
      let mx = -Infinity;
      for (let j = 0; j < M; j++) {
        let s = 0;
        for (let d = 0; d < D; d++) s += z[i * D + d] * z[j * D + d];
        sim[j] = s / T;
        if (j !== i && sim[j] > mx) mx = sim[j];
      }
      let den = 0, posExp = 0;
      for (let j = 0; j < M; j++) {
        if (j === i) continue;
        const e = Math.exp(sim[j] - mx);
        den += e;
        if (j === pos) posExp = e;
      }
      total += -(sim[pos] - mx - Math.log(den));
      void posExp;
    }
    return total / M;
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.loss.triplet', arity: 4,
  sig: '(Tensor anchor[N,D], Tensor pos[N,D], Tensor neg[N,D], number margin=1) -> number',
  doc: '三元组损失 = mean(max(0, ||a−p||² − ||a−n||² + margin))，拉近正例、推远负例',
  ref: 'Schroff et al., FaceNet: A Unified Embedding for Face Recognition and Clustering, CVPR 2015',
  inv: 'triplet ≥ 0；满足 margin 约束（d(a,n)² ≥ d(a,p)²+margin）时为 0；a=p 且 margin=0 时趋于 0',
  impl: (anchor, pos, neg, margin) => {
    expectTensor(anchor, 'anchor'); expectTensor(pos, 'pos'); expectTensor(neg, 'neg');
    if (anchor.size !== pos.size || anchor.size !== neg.size) {
      throw new ForgeError('E_SHAPE', 'anchor/pos/neg 元素数不符');
    }
    const m = margin === undefined ? 1 : expectNumber(margin, 'margin');
    const D = anchor.shape[anchor.ndim - 1];
    const N = anchor.size / D;
    const a = new Float64Array(N);
    for (let n = 0; n < N; n++) {
      let dp = 0, dn = 0;
      for (let d = 0; d < D; d++) {
        const av = anchor.data[n * D + d];
        dp += (av - pos.data[n * D + d]) ** 2;
        dn += (av - neg.data[n * D + d]) ** 2;
      }
      a[n] = Math.max(0, dp - dn + m);
    }
    return _reduce(a, 'mean');
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.loss.labelSmoothCE', arity: 4,
  sig: "(Tensor logits[N,C], Tensor target[N], number eps=0.1, string reduction='mean') -> number",
  doc: '标签平滑交叉熵 = (1−eps)·CE(one-hot) + eps·CE(均匀分布)，抑制过度自信',
  ref: 'Szegedy et al., Rethinking the Inception Architecture for Computer Vision, CVPR 2016',
  inv: 'eps=0 退化为 ceWithLogits；eps=1 时 = KL(U||p) + log C；损失下限为 eps·log C 量级',
  impl: (logits, target, eps, reduction) => {
    expectTensor(logits, 'logits');
    if (logits.ndim !== 2) throw new ForgeError('E_SHAPE', 'labelSmoothCE 需要 logits[N,C]');
    const [N, C] = logits.shape;
    const lab = _labels(target, C);
    if (lab.length !== N) throw new ForgeError('E_SHAPE', `标签数 ${lab.length} != 样本数 ${N}`);
    const e = eps === undefined ? 0.1 : expectNumber(eps, 'eps');
    if (!(e >= 0 && e <= 1)) throw new ForgeError('E_VALUE', `eps 必须 ∈ [0,1]，当前 ${e}`);
    const lp = _logSoftmaxLast(logits);
    const a = new Float64Array(N);
    for (let n = 0; n < N; n++) {
      const nll = -lp.data[n * C + lab[n]];
      let sm = 0;
      for (let c = 0; c < C; c++) sm += -lp.data[n * C + c];
      a[n] = (1 - e) * nll + e * (sm / C);
    }
    return _reduce(a, reduction);
  },
});

// =====================================================================
// 八、参数初始化（10 条）
// =====================================================================
// fan-in / fan-out：全连接 [fanOut, fanIn]；卷积 [OC, IC, *kernel]
function _fanInOut(shape) {
  if (!Array.isArray(shape) || shape.length === 0) throw new ForgeError('E_VALUE', 'shape 非法');
  if (shape.length === 1) return { fanIn: shape[0], fanOut: shape[0] };
  if (shape.length === 2) return { fanIn: shape[1], fanOut: shape[0] };
  const rec = _prod(shape.slice(2));
  return { fanIn: shape[1] * rec, fanOut: shape[0] * rec };
}
function _initRandom(shape, seed, fill) {
  if (!Array.isArray(shape) || shape.length === 0) throw new ForgeError('E_VALUE', 'shape 需为非空数组');
  for (const d of shape) if (!Number.isInteger(d) || d < 0) throw new ForgeError('E_VALUE', `非法维度 ${d}`);
  const n = _prod(shape);
  const rng = createRng((seed === undefined ? 20261014 : seed) >>> 0);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = fill(rng);
  return new Tensor(out, shape.slice());
}
function _shapeArr(shape) {
  if (!Array.isArray(shape) || shape.length === 0) throw new ForgeError('E_VALUE', 'shape 需为非空数组');
  for (const d of shape) if (!Number.isInteger(d) || d < 0) throw new ForgeError('E_VALUE', `非法维度 ${d}`);
  return shape.slice();
}

defineOp({
  id: nid(), domain: 8, name: 'dl.init.zeros', arity: 1,
  sig: '(number[] shape) -> Tensor',
  doc: '全零初始化（常用于偏置项与门控偏置的初始设置）',
  ref: 'PyTorch nn.init.zeros_ / TensorFlow zeros_initializer',
  inv: '所有元素恰为 0；形状与入参一致；均值=方差=0',
  impl: (shape) => Tensor.zeros(_shapeArr(shape)),
});
defineOp({
  id: nid(), domain: 8, name: 'dl.init.ones', arity: 1,
  sig: '(number[] shape) -> Tensor',
  doc: '全一初始化（LayerNorm 的 gamma、缩放因子的常用初值）',
  ref: 'PyTorch nn.init.ones_ / TensorFlow ones_initializer',
  inv: '所有元素恰为 1；均值=1，方差=0',
  impl: (shape) => Tensor.ones(_shapeArr(shape)),
});
defineOp({
  id: nid(), domain: 8, name: 'dl.init.constant', arity: 2,
  sig: '(number[] shape, number value) -> Tensor',
  doc: '常数初始化（如把遗忘门偏置设为 1 以缓解早期遗忘）',
  ref: 'PyTorch nn.init.constant_；Gers et al. 2000 遗忘门偏置技巧',
  inv: '所有元素等于 value；均值=value，方差=0',
  impl: (shape, value) => {
    const sh = _shapeArr(shape);
    return Tensor.full(sh, expectNumber(value, 'value'));
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.init.uniform', arity: 4,
  sig: '(number[] shape, number a=-0.05, number b=0.05, number seed=20261014) -> Tensor',
  doc: '均匀分布 U(a,b) 初始化',
  ref: 'PyTorch nn.init.uniform_；早期神经网络的对称破缺做法',
  inv: '元素全部落在 [a,b]；均值≈(a+b)/2，方差≈(b−a)²/12；同种子完全可复现',
  impl: (shape, a, b, seed) => {
    const lo = a === undefined ? -0.05 : expectNumber(a, 'a');
    const hi = b === undefined ? 0.05 : expectNumber(b, 'b');
    if (!(hi > lo)) throw new ForgeError('E_VALUE', `需要 b > a，当前 a=${lo}, b=${hi}`);
    return _initRandom(shape, seed, (rng) => lo + (hi - lo) * rng.next());
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.init.normal', arity: 4,
  sig: '(number[] shape, number mean=0, number std=0.02, number seed=20261014) -> Tensor',
  doc: '正态 N(mean, std²) 初始化（Transformer 嵌入/投影层常用 std=0.02）',
  ref: 'PyTorch nn.init.normal_；Radford et al. GPT-2 的 0.02 缩放惯例',
  inv: '样本均值≈mean、方差≈std²（大样本下 10% 容差内）；同种子可复现',
  impl: (shape, mean, std, seed) => {
    const mu = mean === undefined ? 0 : expectNumber(mean, 'mean');
    const sd = std === undefined ? 0.02 : expectNumber(std, 'std');
    if (!(sd >= 0)) throw new ForgeError('E_VALUE', `std 必须 >= 0，当前 ${sd}`);
    return _initRandom(shape, seed, (rng) => mu + sd * rng.normal());
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.init.truncatedNormal', arity: 4,
  sig: '(number[] shape, number mean=0, number std=0.02, number seed=20261014) -> Tensor',
  doc: '截断正态（|z−mean| ≤ 2·std）：拒绝采样最多 8 次，超出则夹到边界，避免极端初值',
  ref: 'TensorFlow tf.truncated_normal_initializer；Salimans & Kingma 2016 提及',
  inv: '所有元素落在 [mean−2std, mean+2std]；样本方差略小于 std²；无异常值',
  impl: (shape, mean, std, seed) => {
    const mu = mean === undefined ? 0 : expectNumber(mean, 'mean');
    const sd = std === undefined ? 0.02 : expectNumber(std, 'std');
    if (!(sd >= 0)) throw new ForgeError('E_VALUE', `std 必须 >= 0，当前 ${sd}`);
    return _initRandom(shape, seed, (rng) => {
      for (let k = 0; k < 8; k++) {
        const z = rng.normal();
        if (Math.abs(z) <= 2) return mu + sd * z;
      }
      const z = rng.normal();
      return mu + sd * Math.max(-2, Math.min(2, z));
    });
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.init.xavierUniform', arity: 3,
  sig: '(number[] shape, number gain=1, number seed=20261014) -> Tensor',
  doc: 'Xavier/Glorot 均匀初始化 U(−b,b)，b = gain·√(6/(fanIn+fanOut))，保持前后向信号方差一致',
  ref: 'Glorot & Bengio, Understanding the Difficulty of Training Deep Feedforward Neural Networks, AISTATS 2010',
  inv: '方差 = gain²·2/(fanIn+fanOut)；均值 = 0；元素落在 [−b,b]',
  impl: (shape, gain, seed) => {
    const sh = _shapeArr(shape);
    const g = gain === undefined ? 1 : expectNumber(gain, 'gain');
    const { fanIn, fanOut } = _fanInOut(sh);
    const b = g * Math.sqrt(6 / Math.max(1, fanIn + fanOut));
    return _initRandom(sh, seed, (rng) => (2 * rng.next() - 1) * b);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.init.xavierNormal', arity: 3,
  sig: '(number[] shape, number gain=1, number seed=20261014) -> Tensor',
  doc: 'Xavier/Glorot 正态初始化 N(0, std²)，std = gain·√(2/(fanIn+fanOut))',
  ref: 'Glorot & Bengio, AISTATS 2010',
  inv: '方差 ≈ gain²·2/(fanIn+fanOut)；均值 ≈ 0；与 xavierUniform 同方差不同分布',
  impl: (shape, gain, seed) => {
    const sh = _shapeArr(shape);
    const g = gain === undefined ? 1 : expectNumber(gain, 'gain');
    const { fanIn, fanOut } = _fanInOut(sh);
    const sd = g * Math.sqrt(2 / Math.max(1, fanIn + fanOut));
    return _initRandom(sh, seed, (rng) => sd * rng.normal());
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.init.heUniform', arity: 3,
  sig: '(number[] shape, number gain=√2, number seed=20261014) -> Tensor',
  doc: 'He 均匀初始化 U(−b,b)，b = gain·√(3/fanIn)，专为 ReLU 族设计（前向方差守恒）',
  ref: 'He et al., Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet, ICCV 2015',
  inv: '方差 = gain²/fanIn（gain=√2 时 = 2/fanIn）；均值 = 0；元素落在 [−b,b]',
  impl: (shape, gain, seed) => {
    const sh = _shapeArr(shape);
    const g = gain === undefined ? Math.SQRT2 : expectNumber(gain, 'gain');
    const { fanIn } = _fanInOut(sh);
    const b = g * Math.sqrt(3 / Math.max(1, fanIn));
    return _initRandom(sh, seed, (rng) => (2 * rng.next() - 1) * b);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.init.heNormal', arity: 3,
  sig: '(number[] shape, number gain=√2, number seed=20261014) -> Tensor',
  doc: 'He 正态初始化 N(0, std²)，std = gain/√fanIn（ReLU/LeakyReLU 网络默认）',
  ref: 'He et al., ICCV 2015；PyTorch kaiming_normal_',
  inv: '方差 ≈ gain²/fanIn = 2/fanIn（gain=√2）；均值 ≈ 0；与 heUniform 同方差不同分布',
  impl: (shape, gain, seed) => {
    const sh = _shapeArr(shape);
    const g = gain === undefined ? Math.SQRT2 : expectNumber(gain, 'gain');
    const { fanIn } = _fanInOut(sh);
    const sd = g / Math.sqrt(Math.max(1, fanIn));
    return _initRandom(sh, seed, (rng) => sd * rng.normal());
  },
});

// =====================================================================
// 九、模型与工具（9 条）
// =====================================================================
// 把激活名映射到 ad 的带梯度算子（供 mlpForward 复用域4 的反向模式）
function _adActFn(name) {
  if (name === null || name === undefined || name === 'none' || name === 'linear') return null;
  if (name === 'tanh') return (t) => exec('ad.tanh', [t]);
  if (name === 'relu') return (t) => exec('ad.relu', [t]);
  if (name === 'sigmoid') return (t) => exec('ad.sigmoid', [t]);
  if (name === 'gelu') return (t) => exec('ad.gelu', [t]);
  if (typeof name === 'function') return name;
  throw new ForgeError('E_VALUE', `未知激活: ${name}`);
}
// 通用的「投影 + 偏置」：绕过 ad（用于不需要梯度的模块）
function _ffn(x, W, b) {
  const xf = x.reshape([x.size / x.shape[x.ndim - 1], x.shape[x.ndim - 1]]);
  const y = _mm(xf, W);
  const out = new Float64Array(y.size);
  const n = y.shape[1];
  for (let i = 0; i < y.size; i++) out[i] = y.data[i] + (b ? b.data[i % n] : 0);
  return new Tensor(out, [...x.shape.slice(0, -1), n]);
}

defineOp({
  id: nid(), domain: 8, name: 'dl.model.mlpInit', arity: 3,
  sig: "(number[] sizes, number seed=20261014, string scheme='he') -> {sizes, W: Tensor[], b: Tensor[], count: number}",
  doc: '按 sizes 初始化全连接 MLP 参数（W_i 形状 [sizes_i, sizes_{i+1}]，偏置置零）',
  ref: 'He et al. 2015 / Glorot & Bengio 2010；PyTorch nn.Sequential 的参数构造',
  inv: 'W 长度 = sizes 长度 −1；W_i 形状 [sizes_i, sizes_{i+1}]；偏置全零；count = Σ 参数量',
  impl: (sizes, seed, scheme) => {
    if (!Array.isArray(sizes) || sizes.length < 2) throw new ForgeError('E_VALUE', 'sizes 至少含 2 个元素');
    for (const d of sizes) if (!Number.isInteger(d) || d < 1) throw new ForgeError('E_VALUE', `非法层宽 ${d}`);
    const sch = scheme === undefined ? 'he' : String(scheme);
    const sd = (seed === undefined ? 20261014 : seed) >>> 0;
    const W = [], b = [];
    let count = 0;
    for (let i = 0; i < sizes.length - 1; i++) {
      const sh = [sizes[i], sizes[i + 1]];
      let w;
      if (sch === 'xavier') w = exec('dl.init.xavierUniform', [sh, 1, sd + i]);
      else if (sch === 'normal') w = exec('dl.init.normal', [sh, 0, 0.02, sd + i]);
      else if (sch === 'he') w = exec('dl.init.heNormal', [sh, Math.SQRT2, sd + i]);
      else throw new ForgeError('E_VALUE', `未知初始化方案: ${sch}`);
      W.push(w);
      b.push(Tensor.zeros([sizes[i + 1]]));
      count += w.size + sizes[i + 1];
    }
    return { sizes: sizes.slice(), W, b, count };
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.model.mlpForward', arity: 4,
  sig: "(Tensor x[N,D], Tensor[] W, Tensor[] b, string hiddenAct='relu') -> Tensor[N,Dk]",
  doc: '多层感知机前向：逐层 dense 并在除最后一层外施加激活；内部复用域4 的 ad.dense，因此可被 gradcheck 求导',
  ref: 'Hornik et al. 1989 通用逼近定理；PyTorch nn.Sequential 前向语义',
  inv: '输出形状 [N, D_last]；层数 = W.length；hiddenAct="none" 时退化为纯线性映射',
  impl: (x, W, b, hiddenAct) => {
    expectTensor(x, 'x');
    if (!Array.isArray(W) || W.length === 0) throw new ForgeError('E_VALUE', 'W 需为非空 Tensor 数组');
    if (!Array.isArray(b) || b.length !== W.length) throw new ForgeError('E_VALUE', 'b 需与 W 等长');
    const actName = hiddenAct === undefined ? 'relu' : hiddenAct;
    let h = x;
    for (let i = 0; i < W.length; i++) {
      const last = i === W.length - 1;
      const fn = last ? null : _adActFn(actName);
      h = exec('ad.dense', [h, W[i], b[i], fn]);
    }
    return h;
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.model.transformerBlock', arity: 5,
  sig: '(Tensor x[B,T,C], object params, number heads, Tensor mask?, number eps=1e-5) -> Tensor[B,T,C]',
  doc: 'Pre-LN Transformer 块：x ← x + MHA(LN?)… 采用 Post-LN 形式：a=MHA(x)；h=LN(x+a)；f=FFN(h)；out=LN(h+f)',
  ref: 'Vaswani et al., Attention Is All You Need, NeurIPS 2017',
  inv: '输出形状恒为 [B,T,C]；残差连接保证恒等通路；有限输入产生有限输出',
  impl: (x, params, heads, mask, eps) => {
    expectTensor(x, 'x');
    if (x.ndim !== 3) throw new ForgeError('E_SHAPE', 'transformerBlock 需要 x[B,T,C]');
    if (params == null || typeof params !== 'object') throw new ForgeError('E_TYPE', 'params 必须为对象');
    const need = ['Wq', 'Wk', 'Wv', 'Wo', 'ln1g', 'ln1b', 'ffW1', 'ffb1', 'ffW2', 'ffb2', 'ln2g', 'ln2b'];
    for (const k of need) if (params[k] == null) throw new ForgeError('E_VALUE', `params 缺少字段 ${k}`);
    const e = eps === undefined ? 1e-5 : expectNumber(eps, 'eps');
    // 1) 多头自注意力 + 残差
    const attn = exec('dl.attn.multiHead', [x, params.Wq, params.Wk, params.Wv, params.Wo, heads, mask]);
    const h1 = exec('dl.model.residual', [x, attn, 1]);
    const h1n = exec('dl.norm.layernorm', [h1, params.ln1g, params.ln1b, e]);
    // 2) 前馈网络（GELU 隐层）+ 残差
    const ff = _ffn(h1n, params.ffW1, params.ffb1).map((v) => 0.5 * v * (1 + exec('num.erf', [v / _SQRT2])));
    const ff2 = _ffn(ff, params.ffW2, params.ffb2);
    const h2 = exec('dl.model.residual', [h1n, ff2, 1]);
    return exec('dl.norm.layernorm', [h2, params.ln2g, params.ln2b, e]);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.model.swigluFFN', arity: 7,
  sig: '(Tensor x[B,T,C], Tensor Wg[C,H], Tensor bg[H], Tensor Wu[C,H], Tensor bu[H], Tensor Wd[H,C], Tensor bd[C]) -> Tensor[B,T,C]',
  doc: 'SwiGLU 前馈网络：out = (SiLU(x·Wg+bg) ⊙ (x·Wu+bu))·Wd + bd，LLaMA/PaLM 的默认 FFN',
  ref: 'Shazeer, GLU Variants Improve Transformer, 2020；Touvron et al., LLaMA, 2023',
  inv: '输出形状与输入一致 [B,T,C]；门控值 →0 时输出趋于 0；参数量为标准 FFN 的 1.5 倍',
  impl: (x, Wg, bg, Wu, bu, Wd, bd) => {
    expectTensor(x, 'x');
    for (const [n2, t] of [['Wg', Wg], ['Wu', Wu], ['Wd', Wd]]) expectTensor(t, n2);
    const g = _ffn(x, Wg, bg);
    const u = _ffn(x, Wu, bu);
    const h = g.map((v, i) => (v / (1 + Math.exp(-v))) * u.data[i]);
    return _ffn(h, Wd, bd);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.model.moeRouter', arity: 4,
  sig: '(Tensor x[N,C], Tensor Wg[C,E], number topK, number seed=20261014) -> {probs: Tensor[N,E], topIdx: Tensor[N,topK]}',
  doc: '稀疏 MoE 路由：logits = x·Wg，取 topK 个专家并在其上做 softmax，其余置零（Shazeer 2017 的 noisy top-k 简化版）',
  ref: 'Shazeer et al., Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer, ICLR 2017',
  inv: '每行恰有 topK 个非零且和为 1；topK=1 时退化为 argmax 的 one-hot；topK=E 时为完整 softmax',
  impl: (x, Wg, topK, seed) => {
    expectTensor(x, 'x'); expectTensor(Wg, 'Wg');
    if (x.ndim !== 2 || Wg.ndim !== 2) throw new ForgeError('E_SHAPE', 'moeRouter 需要 2D 输入');
    if (x.shape[1] !== Wg.shape[0]) throw new ForgeError('E_SHAPE', 'x 与 Wg 内维不符');
    const [N, E] = [x.shape[0], Wg.shape[1]];
    const K = expectNumber(topK, 'topK');
    if (!Number.isInteger(K) || K < 1 || K > E) throw new ForgeError('E_VALUE', `topK 必须 ∈ [1,${E}]`);
    const rng = createRng((seed === undefined ? 20261014 : seed) >>> 0);
    const logits = _mmBias(x, Wg, null);
    const probs = new Float64Array(N * E);
    const topIdx = new Float64Array(N * K);
    for (let n = 0; n < N; n++) {
      // 1e-6 量级的确定性抖动：仅在并列时打破平局，不影响常规 top-k 选择
      const z = new Float64Array(E);
      for (let e = 0; e < E; e++) z[e] = logits.data[n * E + e] + 1e-6 * rng.normal();
      const order = Array.from({ length: E }, (_, i) => i).sort((a, b2) => z[b2] - z[a]);
      let mx = -Infinity;
      for (let k = 0; k < K; k++) if (z[order[k]] > mx) mx = z[order[k]];
      let s = 0;
      const exps = new Float64Array(K);
      for (let k = 0; k < K; k++) { exps[k] = Math.exp(z[order[k]] - mx); s += exps[k]; }
      for (let k = 0; k < K; k++) {
        probs[n * E + order[k]] = exps[k] / s;
        topIdx[n * K + k] = order[k];
      }
    }
    return { probs: new Tensor(probs, [N, E]), topIdx: new Tensor(topIdx, [N, K]) };
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.model.residual', arity: 3,
  sig: '(Tensor x, Tensor branch, number scale=1) -> Tensor',
  doc: '残差连接：out = x + scale·branch，为深层网络提供恒等通路（缓解梯度消失）',
  ref: 'He et al., Deep Residual Learning for Image Recognition, CVPR 2016',
  inv: 'branch=0 时为恒等；scale=1 时 out = x + branch；形状与 x 一致',
  impl: (x, branch, scale) => {
    expectTensor(x, 'x'); expectTensor(branch, 'branch');
    const s = scale === undefined ? 1 : expectNumber(scale, 'scale');
    if (x.size !== branch.size) throw new ForgeError('E_SHAPE', 'x 与 branch 元素数不符');
    const out = new Float64Array(x.size);
    for (let i = 0; i < x.size; i++) out[i] = x.data[i] + s * branch.data[i];
    return new Tensor(out, x.shape);
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.util.paramCount', arity: 1,
  sig: '(Tensor[] tensors) -> number',
  doc: '统计参数张量数组的总元素数（模型规模/显存估算）',
  ref: '模型参数量统计惯例（PyTorch numel 求和）',
  inv: 'count = Σ size(t_i)；空数组为 0；与 flattenParams 的 flat 长度一致',
  impl: (tensors) => {
    if (!Array.isArray(tensors)) throw new ForgeError('E_TYPE', 'tensors 需为 Tensor 数组');
    let n = 0;
    for (const t of tensors) n += expectTensor(t, 'tensor').size;
    return n;
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.util.flattenParams', arity: 1,
  sig: '(Tensor[] tensors) -> {flat: Tensor[P], shapes: number[][]}',
  doc: '把参数张量数组按行主序拉平成一维向量，并记录各张量形状以便还原（供优化器/线搜索使用）',
  ref: 'PyTorch parameters_to_vector（torch.nn.utils）',
  inv: 'flat 长度 = Σ size；shapes 与输入一一对应；按 shapes 可无损切分还原',
  impl: (tensors) => {
    if (!Array.isArray(tensors)) throw new ForgeError('E_TYPE', 'tensors 需为 Tensor 数组');
    const shapes = [];
    let n = 0;
    for (const t of tensors) { expectTensor(t, 'tensor'); shapes.push(t.shape.slice()); n += t.size; }
    const flat = new Float64Array(n);
    let o = 0;
    for (const t of tensors) { flat.set(t.data, o); o += t.size; }
    return { flat: new Tensor(flat, [n]), shapes };
  },
});
defineOp({
  id: nid(), domain: 8, name: 'dl.util.applyGrads', arity: 3,
  sig: '(Tensor[] params, Tensor[] grads, number lr) -> Tensor[]',
  doc: '最朴素的 SGD 更新：p ← p − lr·g（逐张量对应，形状必须一致）',
  ref: 'Robbins & Monro, A Stochastic Approximation Method, Ann. Math. Statist. 1951',
  inv: 'lr=0 时参数不变；梯度为 0 时参数不变；输出形状与输入逐个一致',
  impl: (params, grads, lr) => {
    if (!Array.isArray(params) || !Array.isArray(grads)) throw new ForgeError('E_TYPE', 'params/grads 需为数组');
    if (params.length !== grads.length) throw new ForgeError('E_VALUE', 'params/grads 长度不一致');
    const eta = expectNumber(lr, 'lr');
    const out = [];
    for (let i = 0; i < params.length; i++) {
      const p = expectTensor(params[i], `params[${i}]`);
      const g = expectTensor(grads[i], `grads[${i}]`);
      if (p.size !== g.size) throw new ForgeError('E_SHAPE', `第 ${i} 组参数与梯度元素数不符`);
      const d = new Float64Array(p.size);
      for (let k = 0; k < p.size; k++) d[k] = p.data[k] - eta * g.data[k];
      out.push(new Tensor(d, p.shape.slice()));
    }
    return out;
  },
});


// INT8 对称量化：scale = max|x| / 127，q = round(x/scale)，反量化 x̂ = q·scale。
// 与 dl.util.* 族同层，是推理端落地的最后一条链路（训练后静态量化 PTQ）。
defineOp({
  id: nid(), domain: 8, name: 'dl.util.quantizeInt8', arity: 1,
  sig: '(Tensor x) -> object {q: Int8Array, scale: number, dequant: Tensor}',
  doc: 'INT8 对称量化与反量化（训练后静态量化，逐张量粒度）',
  ref: 'Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference, CVPR 2018',
  inv: '|dequant − x| ≤ scale/2；q ∈ [−127,127]；scale ≥ 0（全零张量 scale=1 避免除零）',
  impl: (x) => {
    expectTensor(x, 'x');
    let amax = 0;
    for (let i = 0; i < x.size; i++) {
      const v = Math.abs(x.data[i]);
      if (!Number.isFinite(v)) throw new ForgeError('E_VALUE', `x 第 ${i} 项非有限`);
      if (v > amax) amax = v;
    }
    const scale = amax > 0 ? amax / 127 : 1;      // 全零张量：scale 取 1，量化结果全 0
    const q = new Int8Array(x.size);
    const back = new Float64Array(x.size);
    for (let i = 0; i < x.size; i++) {
      const r = Math.round(x.data[i] / scale);
      q[i] = Math.max(-127, Math.min(127, r));    // 对称量化钳位到 int8 有效区间
      back[i] = q[i] * scale;
    }
    return { q, scale, dequant: new Tensor(back, x.shape) };
  },
});

// —— 文件底部自检：域8 必须恰好 103 条，否则 import 即失败 ——
if (opsByDomain(8).length !== 103) {
  throw new Error(`域8 指令数错误: 期望 103，实际 ${opsByDomain(8).length}`);
}
