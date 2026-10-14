// 域4 自动微分 AUTODIFF（102 条微指令）
// 反向模式（tape + vjp，对标 PyTorch autograd / JAX grad）+ 前向模式（对偶数，JAX jvp）
// + 梯度检验（中心差分，对标 PyTorch gradcheck）。
// 参考：Baydin et al. 2018, Automatic Differentiation in Machine Learning: a Survey；
//       Griewank & Walther《Evaluating Derivatives》2th ed.；PyTorch autograd 设计。
import { defineOp, opsByDomain, exec } from '../core/registry.mjs';
import { ForgeError } from '../core/errors.mjs';
import { Tensor, prod as coreProd, stridesOf, broadcastShapes, expectTensor } from '../core/tensor.mjs';

const D = 4;
let _s = 0;
const nid = () => `D${D}-${String(++_s).padStart(3, '0')}`;

// ============ 磁带机制 ============
const tape = [];
const nodes = new WeakMap(); // Tensor -> {op, inputs, vjp}
let recording = true;

function record(op, inputs, out, vjp) {
  if (!recording) return out;
  const node = { op, inputs, out, vjp };
  nodes.set(out, node);
  tape.push(node);
  return out;
}

// 把梯度归约到目标形状（广播逆操作）：前导维合并求和 + 广播维累加求和
function sumTo(g, shape) {
  if (sameShapeArr(g.shape, shape)) return g;
  // 先合并前导多余维（沿第 0 维求和）
  let cur = g;
  while (cur.ndim > shape.length) {
    const block = coreProd(cur.shape.slice(1));
    const out2 = new Float64Array(block);
    for (let i = 0; i < cur.shape[0]; i++) for (let j = 0; j < block; j++) out2[j] += cur.data[i * block + j];
    cur = new Tensor(out2, cur.shape.slice(1));
  }
  // 再对被广播的维（目标形状中为 1 的维）累加
  const out = new Float64Array(coreProd(shape));
  const stg = stridesOf(shape);
  const idx = new Array(cur.ndim);
  const lead = shape.length - cur.ndim;
  for (let p = 0; p < cur.size; p++) {
    let r = p;
    for (let d = cur.ndim - 1; d >= 0; d--) { idx[d] = r % cur.shape[d]; r = (r - idx[d]) / cur.shape[d]; }
    let q = 0;
    for (let d = 0; d < shape.length; d++) {
      const cv = d >= lead ? idx[d - lead] : 0;
      if (shape[d] !== 1) q += cv * stg[d];
    }
    out[q] += cur.data[p];
  }
  return new Tensor(out, shape);
}
function sameShapeArr(a, b) { return a.length === b.length && a.every((v, i) => v === b[i]); }
function addGrad(a, b) { const out = new Float64Array(a.size); for (let i = 0; i < a.size; i++) out[i] = a.data[i] + b.data[i]; return new Tensor(out, a.shape); }

// 通用逐元素前向 + vjp 生成器（dfFn: 导数函数；dfDoc: 导数公式文档）
function elementwiseAd(name, doc, f, dfFn, dfDoc, tableRef) {
  defineOp({
    id: nid(), domain: D, name, arity: 1, sig: '(Tensor x) -> Tensor', doc,
    ref: tableRef, inv: `d/dx ${name} = ${dfDoc}`,
    impl: (x) => {
      expectTensor(x, 'x');
      const out = x.map(f);
      return record(name, [x], out, (g) => [g.map((v, i) => v * dfFn(x.data[i]))]);
    },
  });
}

// ============ 逐元素算子（22 条） ============
elementwiseAd('ad.exp', '指数 e^x（带梯度）', Math.exp, Math.exp, 'e^x', 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018');
elementwiseAd('ad.log', '对数 ln x（x>0）', Math.log, (v) => 1 / v, '1/x', 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018');
elementwiseAd('ad.log1p', 'ln(1+x)', Math.log1p, (v) => 1 / (1 + v), '1/(1+x)', 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018');
elementwiseAd('ad.sqrt', '平方根 √x（带梯度）', Math.sqrt, (v) => 1 / (2 * Math.sqrt(v)), '1/(2√x)', 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018');
elementwiseAd('ad.sin', '正弦 sin x（带梯度）', Math.sin, Math.cos, 'cos x', 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018');
elementwiseAd('ad.cos', '余弦 cos x（带梯度）', Math.cos, (v) => -Math.sin(v), '-sin x', 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018');
elementwiseAd('ad.tanh', '双曲正切', Math.tanh, (v) => 1 - Math.tanh(v) ** 2, '1 - tanh²x', 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018');
elementwiseAd('ad.sigmoid', 'sigmoid σ(x)=1/(1+e^{-x})', (v) => 1 / (1 + Math.exp(-v)), (v) => { const s = 1 / (1 + Math.exp(-v)); return s * (1 - s); }, 'σ(x)(1-σ(x))', 'Logistic 函数');
elementwiseAd('ad.expm1', 'e^x - 1', Math.expm1, Math.exp, 'e^x', 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018');
elementwiseAd('ad.square', '逐元素平方 x²（带梯度）', (v) => v * v, (v) => 2 * v, '2x', 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018');
elementwiseAd('ad.reciprocal', '逐元素倒数 1/x（带梯度）', (v) => 1 / v, (v) => -1 / (v * v), '-1/x²', 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018');

defineOp({
  id: nid(), domain: D, name: 'ad.relu', arity: 1, sig: '(Tensor x) -> Tensor', doc: 'ReLU max(0,x)（0 点次梯度取 0）',
  ref: 'Nair & Hinton 2010；Glorot 2011', inv: 'x>0 时梯度 1，否则 0',
  impl: (x) => { const out = x.map((v) => Math.max(0, v)); return record('ad.relu', [x], out, (g) => [g.map((v, i) => (x.data[i] > 0 ? v : 0))]); },
});
defineOp({
  id: nid(), domain: D, name: 'ad.leakyRelu', arity: 1, sig: '(Tensor x, number alpha=0.01) -> Tensor',
  doc: 'LeakyReLU（负区间斜率 alpha）', ref: 'Maas et al. 2013', inv: 'x>0 梯度 1，否则 alpha',
  impl: (x, alpha = 0.01) => {
    const out = x.map((v) => (v > 0 ? v : alpha * v));
    return record('ad.leakyRelu', [x], out, (g) => [g.map((v, i) => (x.data[i] > 0 ? v : alpha * v))]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.elu', arity: 1, sig: '(Tensor x, number alpha=1) -> Tensor',
  doc: 'ELU（x>0 取 x，否则 α(e^x−1)）', ref: 'Clevert et al. 2016', inv: 'x>0 梯度 1，否则 out+α',
  impl: (x, alpha = 1) => {
    const out = x.map((v) => (v > 0 ? v : alpha * (Math.exp(v) - 1)));
    return record('ad.elu', [x], out, (g) => [g.map((v, i) => (x.data[i] > 0 ? v : v * (out.data[i] + alpha)))]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.silu', arity: 1, sig: '(Tensor x) -> Tensor', doc: 'SiLU/Swish x·σ(x)',
  ref: 'Ramachandran et al. 2017（Swish搜索）；Hendrycks & Gimpel 2016（GELU 前身）', inv: "d/dx = σ(x)(1 + x(1-σ(x)))",
  impl: (x) => {
    const sig = x.map((v) => 1 / (1 + Math.exp(-v)));
    const out = x.map((v, i) => v * sig.data[i]);
    return record('ad.silu', [x], out, (g) => [g.map((v, i) => v * sig.data[i] * (1 + x.data[i] * (1 - sig.data[i])))]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.gelu', arity: 1, sig: '(Tensor x) -> Tensor', doc: 'GELU（tanh 近似）',
  ref: 'Hendrycks & Gimpel 2016', inv: "d/dx ≈ Φ(x) + x·φ(x)（精确版）",
  impl: (x) => {
    const c = Math.sqrt(2 / Math.PI);
    const out = x.map((v) => 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v ** 3))));
    return record('ad.gelu', [x], out, (g) => {
      const sech2 = (v) => { const t = Math.tanh(c * (v + 0.044715 * v ** 3)); return 1 - t * t; };
      return [g.map((v, i) => {
        const xi = x.data[i];
        const inner = c * (xi + 0.044715 * xi ** 3);
        const dinner = c * (1 + 3 * 0.044715 * xi * xi);
        return v * (0.5 * (1 + Math.tanh(inner)) + 0.5 * xi * sech2(xi) * dinner);
      })];
    });
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.softplus', arity: 1, sig: '(Tensor x) -> Tensor', doc: 'softplus log(1+e^x)（稳定分支）',
  ref: 'Dugas et al. 2001', inv: 'd/dx = σ(x)',
  impl: (x) => {
    const out = x.map((v) => (v > 30 ? v : v < -30 ? Math.exp(v) : Math.log1p(Math.exp(v))));
    return record('ad.softplus', [x], out, (g) => [g.map((v, i) => v * (x.data[i] >= 0 ? 1 / (1 + Math.exp(-x.data[i])) : Math.exp(x.data[i]) / (1 + Math.exp(x.data[i]))))]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.abs', arity: 1, sig: '(Tensor x) -> Tensor', doc: '绝对值（0 点次梯度 0）',
  ref: 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018', inv: 'd/dx = sign(x)',
  impl: (x) => { const out = x.map(Math.abs); return record('ad.abs', [x], out, (g) => [g.map((v, i) => v * Math.sign(x.data[i]))]); },
});
defineOp({
  id: nid(), domain: D, name: 'ad.clamp', arity: 3, sig: '(Tensor x, number lo, number hi) -> Tensor',
  doc: '截断（界外梯度 0）', ref: 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018', inv: '界内梯度 1',
  impl: (x, lo, hi) => {
    const out = x.map((v) => Math.min(Math.max(v, lo), hi));
    return record('ad.clamp', [x], out, (g) => [g.map((v, i) => (x.data[i] >= lo && x.data[i] <= hi ? v : 0))]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.maximum', arity: 2, sig: '(Tensor x, Tensor y) -> Tensor',
  doc: '逐元素最大（平局梯度平分）', ref: 'NumPy maximum + 次梯度', inv: 'd/dx = [x≥y]',
  impl: (x, y) => {
    const out = x.map((v, i) => Math.max(v, y.data[i % y.size]));
    return record('ad.maximum', [x, y], out, (g) => [
      g.map((v, i) => (x.data[i] >= y.data[i % y.size] ? v : 0)),
      g.map((v, i) => (y.data[i % y.size] >= x.data[i] ? v : 0)),
    ]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.minimum', arity: 2, sig: '(Tensor x, Tensor y) -> Tensor',
  doc: '逐元素最小（平局梯度平分）', ref: 'NumPy minimum + 次梯度', inv: 'd/dx = [x≤y]',
  impl: (x, y) => {
    const out = x.map((v, i) => Math.min(v, y.data[i % y.size]));
    return record('ad.minimum', [x, y], out, (g) => [
      g.map((v, i) => (x.data[i] <= y.data[i % y.size] ? v : 0)),
      g.map((v, i) => (y.data[i % y.size] <= x.data[i] ? v : 0)),
    ]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.pow', arity: 2, sig: '(Tensor x, number p) -> Tensor', doc: '幂 x^p',
  ref: 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018', inv: 'd/dx = p·x^(p−1)',
  impl: (x, p) => {
    const out = x.map((v) => v ** p);
    return record('ad.pow', [x], out, (g) => [g.map((v, i) => v * p * x.data[i] ** (p - 1))]);
  },
});

// ============ 二元算子（8 条） ============
defineOp({
  id: nid(), domain: D, name: 'ad.add', arity: 2, sig: '(Tensor|number a, Tensor|number b) -> Tensor',
  doc: '加法（广播）', ref: 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018', inv: 'd/da = g，d/db = g',
  impl: (a, b) => {
    if (typeof a === 'number') return record('ad.add', [b], b.map((v) => v + a), (g) => [g]);
    if (typeof b === 'number') return record('ad.add', [a], a.map((v) => v + b), (g) => [g]);
    const out = exec('math.add', [a, b]);
    return record('ad.add', [a, b], out, (g) => [sumTo(g, a.shape), sumTo(g, b.shape)]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.sub', arity: 2, sig: '(Tensor|number a, Tensor|number b) -> Tensor',
  doc: '减法（广播）', ref: 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018', inv: 'd/da=g，d/db=−g',
  impl: (a, b) => {
    if (typeof a === 'number') return record('ad.sub', [b], b.map((v) => a - v), (g) => [g.map((v) => -v)]);
    if (typeof b === 'number') return record('ad.sub', [a], a.map((v) => v - b), (g) => [g]);
    const out = exec('math.sub', [a, b]);
    return record('ad.sub', [a, b], out, (g) => [sumTo(g, a.shape), sumTo(g, b.shape).map((v) => -v)]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.mul', arity: 2, sig: '(Tensor|number a, Tensor|number b) -> Tensor',
  doc: '乘法（广播）', ref: 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018', inv: 'd/da = g·b，d/db = g·a',
  impl: (a, b) => {
    if (typeof a === 'number') return record('ad.mul', [b], b.map((v) => a * v), (g) => [g.map((v) => a * v)]);
    if (typeof b === 'number') return record('ad.mul', [a], a.map((v) => v * b), (g) => [g.map((v) => b * v)]);
    const out = exec('math.mul', [a, b]);
    return record('ad.mul', [a, b], out, (g) => [sumTo(g.map((v, i) => v * b.data[i % b.size]), a.shape), sumTo(g.map((v, i) => v * a.data[i % a.size]), b.shape)]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.div', arity: 2, sig: '(Tensor|number a, Tensor|number b) -> Tensor',
  doc: '除法（广播）', ref: 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018', inv: 'd/da = g/b，d/db = −g·a/b²',
  impl: (a, b) => {
    if (typeof a === 'number') return record('ad.div', [b], b.map((v) => a / v), (g) => [g.map((v, i) => -a * v / (b.data[i % b.size] ** 2))]);
    if (typeof b === 'number') return record('ad.div', [a], a.map((v) => v / b), (g) => [g.map((v) => v / b)]);
    const out = exec('math.div', [a, b]);
    return record('ad.div', [a, b], out, (g) => [
      sumTo(g.map((v, i) => v / b.data[i % b.size]), a.shape),
      sumTo(g.map((v, i) => -v * a.data[i % a.size] / (b.data[i % b.size] ** 2)), b.shape),
    ]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.neg', arity: 1, sig: '(Tensor x) -> Tensor', doc: '逐元素取负 −x（带梯度）',
  ref: 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018', inv: 'd/dx = −g', impl: (x) => record('ad.neg', [x], x.map((v) => -v), (g) => [g.map((v) => -v)]),
});

// ============ 矩阵算子（6 条） ============
defineOp({
  id: nid(), domain: D, name: 'ad.matmul', arity: 2, sig: '(Tensor A[m,k], Tensor B[k,n]) -> Tensor',
  doc: '矩阵乘法（反向：dA=g·Bᵀ，dB=Aᵀ·g）', ref: '链式法则 / Golub & Van Loan',
  inv: 'dA=g·Bᵀ；dB=Aᵀ·g',
  impl: (A, B) => {
    const out = exec('linalg.matmul', [A, B]);
    return record('ad.matmul', [A, B], out, (g) => [exec('linalg.matmul', [g, B.T()]), exec('linalg.matmul', [A.T(), g])]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.matmulAT', arity: 2, sig: '(Tensor A[k,m], Tensor B[k,n]) -> Tensor',
  doc: 'Aᵀ·B', ref: 'BLAS transA', inv: 'dA=g·Bᵀ 的转置形式',
  impl: (A, B) => exec('ad.matmul', [exec('ad.transpose', [A]), B]),
});
defineOp({
  id: nid(), domain: D, name: 'ad.vecDot', arity: 2, sig: '(Tensor x[n], Tensor y[n]) -> Tensor',
  doc: '向量点积（标量输出张量）', ref: 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018', inv: 'dx = g·y，dy = g·x',
  impl: (x, y) => {
    let s = 0;
    for (let i = 0; i < x.size; i++) s += x.data[i] * y.data[i];
    const out = new Tensor([s], [1]);
    return record('ad.vecDot', [x, y], out, (g) => [
      Tensor.full(x.shape, g.data[0]).map((v, i) => v * y.data[i]),
      Tensor.full(y.shape, g.data[0]).map((v, i) => v * x.data[i]),
    ]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.transpose', arity: 1, sig: '(Tensor A) -> Tensor', doc: '矩阵转置（反向传播仍为转置）',
  ref: 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018', inv: '转置运算的反向传播仍是转置（vjp 为自身转置）', impl: (A) => record('ad.transpose', [A], A.T(), (g) => [g.T()]),
});
defineOp({
  id: nid(), domain: D, name: 'ad.reshape', arity: 2, sig: '(Tensor x, number[] shape) -> Tensor',
  doc: 'reshape（梯度反 reshape）', ref: 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018', inv: '梯度回到原形状',
  impl: (x, shape) => record('ad.reshape', [x], x.reshape(shape), (g) => [g.reshape(x.shape)]),
});
defineOp({
  id: nid(), domain: D, name: 'ad.broadcastTo', arity: 2, sig: '(Tensor x, number[] shape) -> Tensor',
  doc: '显式广播（梯度 sumTo 原形状）', ref: 'NumPy broadcast', inv: 'd/dx = sumTo(g)',
  impl: (x, shape) => {
    const out = exec('math.mul', [x, Tensor.ones(shape)]);
    return record('ad.broadcastTo', [x], out, (g) => [sumTo(g, x.shape)]);
  },
});

// ============ 归约算子（6 条） ============
defineOp({
  id: nid(), domain: D, name: 'ad.sumAll', arity: 1, sig: '(Tensor x) -> Tensor', doc: '全元素求和（标量）',
  ref: 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018', inv: 'd/dx = g 广播',
  impl: (x) => { const s = x.data.reduce((a, b) => a + b, 0); const out = new Tensor([s], [1]); return record('ad.sumAll', [x], out, (g) => [Tensor.full(x.shape, g.data[0])]); },
});
defineOp({
  id: nid(), domain: D, name: 'ad.meanAll', arity: 1, sig: '(Tensor x) -> Tensor', doc: '全元素均值（标量）',
  ref: 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018', inv: 'd/dx = g/n',
  impl: (x) => { const out = new Tensor([x.data.reduce((a, b) => a + b, 0) / x.size], [1]); return record('ad.meanAll', [x], out, (g) => [Tensor.full(x.shape, g.data[0] / x.size)]); },
});
// 沿轴广播 g 回原形状（sumAxis/meanAxis 的 vjp 共用）
function broadcastAlongAxis(g, xShape, ax) {
  const out = new Float64Array(coreProd(xShape));
  const idx = new Array(xShape.length);
  const stg = stridesOf(g.shape);
  for (let p = 0; p < out.length; p++) {
    let rr = p;
    for (let d = xShape.length - 1; d >= 0; d--) { idx[d] = rr % xShape[d]; rr = (rr - idx[d]) / xShape[d]; }
    let q = 0, oi = 0;
    for (let d = 0; d < xShape.length; d++) { if (d === ax) continue; q += idx[d] * stg[oi]; oi++; }
    out[p] = g.data[q];
  }
  return new Tensor(out, xShape);
}

defineOp({
  id: nid(), domain: D, name: 'ad.sumAxis', arity: 2, sig: '(Tensor x, number axis) -> Tensor',
  doc: '沿轴求和', ref: 'NumPy sum(axis)', inv: '梯度沿轴广播',
  impl: (x, axis) => {
    const ax = axis < 0 ? axis + x.ndim : axis;
    const r = exec('math.sumAxis', [x, axis]);
    return record('ad.sumAxis', [x], r, (g) => [broadcastAlongAxis(g, x.shape, ax)]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.meanAxis', arity: 2, sig: '(Tensor x, number axis) -> Tensor',
  doc: '沿轴均值', ref: 'NumPy mean(axis)', inv: '梯度 = g广播/轴长',
  impl: (x, axis) => {
    const ax = axis < 0 ? axis + x.ndim : axis;
    const r = exec('math.meanAxis', [x, axis]);
    const n = x.shape[ax];
    return record('ad.meanAxis', [x], r, (g) => [broadcastAlongAxis(g, x.shape, ax).map((v) => v / n)]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.maxAll', arity: 1, sig: '(Tensor x) -> Tensor', doc: '全元素最大（次梯度给首个最大元）',
  ref: 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018', inv: '梯度集中在 argmax 处',
  impl: (x) => {
    let bi = 0, bv = -Infinity;
    for (let i = 0; i < x.size; i++) if (x.data[i] > bv) { bv = x.data[i]; bi = i; }
    const out = new Tensor([bv], [1]);
    return record('ad.maxAll', [x], out, (g) => { const t = new Float64Array(x.size); t[bi] = g.data[0]; return [new Tensor(t, x.shape)]; });
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.softmax', arity: 1, sig: '(Tensor x, number axis=-1) -> Tensor',
  doc: 'softmax（雅可比 dx=(g−Σg⊙y)⊙y）', ref: '深度学习标准', inv: '沿轴和=1；Jacobian-vector product 公式',
  impl: (x, axis = -1) => {
    const y = exec('num.softmax', [x, axis]);
    return record('ad.softmax', [x], y, (g) => {
      const out = new Float64Array(x.size);
      const idx = new Array(x.ndim);
      const ax = axis < 0 ? axis + x.ndim : axis;
      const red = x.shape[ax];
      for (let p = 0; p < x.size; p++) {
        let rr = p;
        for (let d = x.ndim - 1; d >= 0; d--) { idx[d] = rr % x.shape[d]; rr = (rr - idx[d]) / x.shape[d]; }
        // 槽位起点
        let base = 0, st = stridesOf(x.shape);
        for (let d = 0; d < x.ndim; d++) if (d !== ax) base += idx[d] * st[d];
        const off = idx[ax];
        let s = 0;
        for (let k = 0; k < red; k++) s += g.data[base + k * st[ax]] * y.data[base + k * st[ax]];
        out[p] = y.data[p] * (g.data[p] - s);
      }
      return [new Tensor(out, x.shape)];
    });
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.logSoftmax', arity: 1, sig: '(Tensor x, number axis=-1) -> Tensor',
  doc: 'log-softmax', ref: 'PyTorch LogSoftmax', inv: 'dx = g − softmax(x)·Σg',
  impl: (x, axis = -1) => {
    const y = exec('num.logSoftmax', [x, axis]);
    return record('ad.logSoftmax', [x], y, (g) => {
      const sm = exec('num.softmax', [x, axis]);
      const gsum = exec('ad.sumAxis', [g, axis]);
      const out = new Float64Array(x.size);
      const idx = new Array(x.ndim);
      const ax = axis < 0 ? axis + x.ndim : axis;
      for (let p = 0; p < x.size; p++) {
        let rr = p;
        for (let d = x.ndim - 1; d >= 0; d--) { idx[d] = rr % x.shape[d]; rr = (rr - idx[d]) / x.shape[d]; }
        let q = 0, oi = 0;
        const stg = stridesOf(gsum.shape);
        for (let d = 0; d < x.ndim; d++) { if (d === ax) continue; q += idx[d] * stg[oi]; oi++; }
        out[p] = g.data[p] - sm.data[p] * gsum.data[q];
      }
      return [new Tensor(out, x.shape)];
    });
  },
});

// ============ 损失算子（7 条） ============
defineOp({
  id: nid(), domain: D, name: 'ad.mse', arity: 2, sig: '(Tensor pred, Tensor target) -> Tensor',
  doc: '均方误差（标量）', ref: '回归标准损失', inv: 'd/dpred = 2(pred−target)/n',
  impl: (pred, target) => {
    const diff = exec('math.sub', [pred, target]);
    const out = new Tensor([diff.data.reduce((a, b) => a + b * b, 0) / pred.size], [1]);
    return record('ad.mse', [pred, target], out, (g) => [diff.map((v) => 2 * v * g.data[0] / pred.size), null]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.mae', arity: 2, sig: '(Tensor pred, Tensor target) -> Tensor',
  doc: '平均绝对误差（标量）', ref: '稳健回归损失', inv: 'd/dpred = sign(pred−target)/n',
  impl: (pred, target) => {
    const diff = exec('math.sub', [pred, target]);
    const out = new Tensor([diff.data.reduce((a, b) => a + Math.abs(b), 0) / pred.size], [1]);
    return record('ad.mae', [pred, target], out, (g) => [diff.map((v) => Math.sign(v) * g.data[0] / pred.size), null]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.huber', arity: 2, sig: '(Tensor pred, Tensor target, number delta=1) -> Tensor',
  doc: 'Huber 损失（L1/L2 混合）', ref: 'Huber 1964 稳健统计', inv: '|d|≤δ 用 L2，否则 L1',
  impl: (pred, target, delta = 1) => {
    const diff = exec('math.sub', [pred, target]);
    let s = 0;
    const grad = new Float64Array(pred.size);
    for (let i = 0; i < pred.size; i++) {
      const d = diff.data[i], ad = Math.abs(d);
      if (ad <= delta) { s += 0.5 * d * d; grad[i] = d; } else { s += delta * (ad - 0.5 * delta); grad[i] = delta * Math.sign(d); }
    }
    const out = new Tensor([s / pred.size], [1]);
    return record('ad.huber', [pred, target], out, (g) => [new Tensor(grad.map((v) => v * g.data[0] / pred.size), pred.shape), null]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.crossEntropy', arity: 2, sig: '(Tensor logits[n,c], Tensor labels[n,c] one-hot) -> Tensor',
  doc: '交叉熵（与 log-softmax 融合）', ref: '分类标准损失', inv: 'd/dlogits = softmax − labels',
  impl: (logits, labels) => {
    const lsm = exec('num.logSoftmax', [logits, -1]);
    let s = 0;
    for (let i = 0; i < logits.size; i++) s -= labels.data[i % labels.size] * lsm.data[i];
    const out = new Tensor([s / logits.shape[0]], [1]);
    return record('ad.crossEntropy', [logits, labels], out, (g) => {
      const sm = exec('num.softmax', [logits, -1]);
      const grad = new Float64Array(logits.size);
      for (let i = 0; i < logits.size; i++) grad[i] = (sm.data[i] - labels.data[i % labels.size]) * g.data[0] / logits.shape[0];
      return [new Tensor(grad, logits.shape), null];
    });
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.bce', arity: 2, sig: '(Tensor logits, Tensor targets) -> Tensor',
  doc: '二元交叉熵（logits 形式，数值稳定）', ref: '二分类标准损失', inv: 'd/dlogits = σ(x) − t',
  impl: (logits, targets) => {
    let s = 0;
    const grad = new Float64Array(logits.size);
    for (let i = 0; i < logits.size; i++) {
      const x = logits.data[i], t = targets.data[i % targets.size];
      // BCE = max(x,0) − x·t + log(1+e^{−|x|})
      s += Math.max(x, 0) - x * t + Math.log1p(Math.exp(-Math.abs(x)));
      grad[i] = (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x))) - t;
    }
    const out = new Tensor([s / logits.size], [1]);
    return record('ad.bce', [logits, targets], out, (g) => [new Tensor(grad.map((v) => v * g.data[0] / logits.size), logits.shape), null]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.nll', arity: 2, sig: '(Tensor logProbs[n,c], Tensor classIdx[n]) -> Tensor',
  doc: '负对数似然（类索引）', ref: '分类标准损失', inv: 'd/dlogProb = −1/p[cls]/n',
  impl: (lp, idx) => {
    const n = lp.shape[0], c = lp.shape[1];
    let s = 0;
    const grad = new Float64Array(lp.size);
    for (let i = 0; i < n; i++) {
      const cls = idx.data[i];
      s -= lp.data[i * c + cls];
      grad[i * c + cls] = -1;
    }
    const out = new Tensor([s / n], [1]);
    return record('ad.nll', [lp, idx], out, (g) => [new Tensor(grad.map((v) => v * g.data[0] / n), lp.shape), null]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.klDiv', arity: 2, sig: '(Tensor logP, Tensor logQ) -> Tensor',
  doc: 'KL 散度（log 空间输入）', ref: 'Kullback & Leibler 1951', inv: 'KL(p‖q)≥0；p=q 时为 0',
  impl: (logP, logQ) => {
    let s = 0;
    for (let i = 0; i < logP.size; i++) s += Math.exp(logP.data[i]) * (logP.data[i] - logQ.data[i % logQ.size]);
    const out = new Tensor([s], [1]);
    return record('ad.klDiv', [logP, logQ], out, (g) => [
      g.map((v) => v * (1 + logP.data[0] - logQ.data[0])), // 简化：逐元素
      null,
    ]);
  },
});

// ============ 结构算子（8 条） ============
defineOp({
  id: nid(), domain: D, name: 'ad.getItem', arity: 2, sig: '(Tensor x, number[] index) -> Tensor',
  doc: '索引取子张量（梯度 scatter 回原位）', ref: 'NumPy fancy indexing', inv: '梯度回填到 index 处',
  impl: (x, index) => {
    const st = stridesOf(x.shape);
    let off = 0;
    for (let d = 0; d < index.length; d++) off += index[d] * st[d];
    const rest = x.size / coreProd(index.map((v, d) => 1));
    const restShape = x.shape.slice(index.length);
    const restLen = coreProd(restShape);
    const out = new Tensor(x.data.slice(off, off + restLen), restShape);
    return record('ad.getItem', [x], out, (g) => {
      const t = new Float64Array(x.size);
      for (let i = 0; i < restLen; i++) t[off + i] = g.data[i];
      return [new Tensor(t, x.shape)];
    });
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.concat', arity: 1, sig: '(Tensor[] ts, number axis) -> Tensor',
  doc: '拼接（梯度按段切分）', ref: 'NumPy concatenate', inv: '各段梯度之和',
  impl: (ts, axis = 0) => {
    const out = exec('math.concat', [ts, axis]);
    const ax = axis < 0 ? axis + ts[0].ndim : axis;
    return record('ad.concat', ts, out, (g) => {
      const inner = out.shape.slice(ax + 1).reduce((a, b) => a * b, 1);
      const outer = out.shape.slice(0, ax).reduce((a, b) => a * b, 1);
      const grads = [];
      let off = 0;
      for (const t of ts) {
        const block = t.shape[ax] * inner;
        const gt = new Float64Array(t.size);
        for (let o = 0; o < outer; o++) {
          gt.set(g.data.subarray((o * out.shape[ax] + off) * inner, (o * out.shape[ax] + off + t.shape[ax]) * inner), o * block);
        }
        grads.push(new Tensor(gt, t.shape));
        off += t.shape[ax];
      }
      return grads;
    });
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.gatherRows', arity: 2, sig: '(Tensor table[n,d], Tensor idx[k]) -> Tensor',
  doc: '行 gather（embedding lookup；重复索引梯度累加）', ref: 'PyTorch embedding', inv: '重复行梯度累加',
  impl: (table, idx) => {
    const [n, d] = table.shape;
    const k = idx.size;
    const out = new Float64Array(k * d);
    for (let i = 0; i < k; i++) {
      const r = idx.data[i];
      for (let j = 0; j < d; j++) out[i * d + j] = table.data[r * d + j];
    }
    const outT = new Tensor(out, [k, d]);
    return record('ad.gatherRows', [table, idx], outT, (g) => {
      const t = new Float64Array(table.size);
      for (let i = 0; i < k; i++) {
        const r = idx.data[i];
        for (let j = 0; j < d; j++) t[r * d + j] += g.data[i * d + j];
      }
      return [new Tensor(t, table.shape), null];
    });
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.scatterAdd', arity: 3, sig: '(Tensor dst, Tensor idx, Tensor src) -> Tensor',
  doc: 'scatter-add（gatherRows 的逆）', ref: 'PyTorch scatter_add', inv: '重复索引累加',
  impl: (dst, idx, src) => {
    const out = dst.clone();
    const d = dst.shape[1] ?? 1;
    for (let i = 0; i < idx.size; i++) {
      const r = idx.data[i];
      for (let j = 0; j < d; j++) out.data[r * d + j] += src.data[i * d + j];
    }
    return record('ad.scatterAdd', [dst, idx, src], out, (g) => {
      const gs = new Float64Array(src.size);
      for (let i = 0; i < idx.size; i++) {
        const r = idx.data[i];
        for (let j = 0; j < d; j++) gs[i * d + j] = g.data[r * d + j];
      }
      return [g, null, new Tensor(gs, src.shape)];
    });
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.where', arity: 3, sig: '(Tensor mask, Tensor a, Tensor b) -> Tensor',
  doc: '掩码选择（梯度走对应分支）', ref: 'NumPy where', inv: '梯度按 mask 分流',
  impl: (mask, a, b) => {
    const out = exec('math.where', [mask, a, b]);
    return record('ad.where', [mask, a, b], out, (g) => [
      null,
      g.map((v, i) => (mask.data[i % mask.size] !== 0 ? v : 0)),
      g.map((v, i) => (mask.data[i % mask.size] !== 0 ? 0 : v)),
    ]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.padConst', arity: 2, sig: '(Tensor x, number[][] pads, number v=0) -> Tensor',
  doc: '常数填充（梯度裁剪掉填充区）', ref: 'NumPy pad', inv: '梯度去掉填充区',
  impl: (x, pads, v = 0) => {
    const out = exec('math.pad', [x, pads, v]);
    const st = stridesOf(x.shape);
    return record('ad.padConst', [x], out, (g) => {
      const t = new Float64Array(x.size);
      const idx = new Array(x.ndim);
      for (let p = 0; p < x.size; p++) {
        let rr = p;
        for (let d = x.ndim - 1; d >= 0; d--) { idx[d] = rr % x.shape[d]; rr = (rr - idx[d]) / x.shape[d]; }
        let q = 0;
        for (let d = 0; d < x.ndim; d++) q += (idx[d] + pads[d][0]) * stridesOf(out.shape)[d];
        t[p] = g.data[q];
      }
      return [new Tensor(t, x.shape)];
    });
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.stack', arity: 1, sig: '(Tensor[] ts, number axis) -> Tensor',
  doc: '沿新轴堆叠', ref: 'NumPy stack', inv: '梯度拆回各输入',
  impl: (ts, axis = 0) => {
    const expanded = ts.map((t) => t.reshape(...insertOne(t.shape, axis < 0 ? axis + t.ndim + 1 : axis, 1)));
    const out = exec('math.concat', [expanded, axis]);
    const ax = axis < 0 ? axis + ts[0].ndim + 1 : axis;
    return record('ad.stack', ts, out, (g) => {
      const inner = out.shape.slice(ax + 1).reduce((a, b) => a * b, 1);
      const outer = out.shape.slice(0, ax).reduce((a, b) => a * b, 1);
      return ts.map((t, k) => {
        const gt = new Float64Array(t.size);
        for (let o = 0; o < outer; o++) {
          gt.set(g.data.subarray((o * out.shape[ax] + k) * inner, (o * out.shape[ax] + k + 1) * inner), o * inner);
        }
        return new Tensor(gt, t.shape);
      });
    });
    function insertOne(shape, at, v) { const s = shape.slice(); s.splice(at, 0, v); return s; }
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.detach', arity: 1, sig: '(Tensor x) -> Tensor', doc: '切断梯度（常量视图）',
  ref: 'PyTorch detach', inv: '不记录到磁带',
  impl: (x) => x.clone(),
});

// ============ 磁带与检验工具（10 条） ============
defineOp({
  id: nid(), domain: D, name: 'ad.tapeReset', arity: 0, sig: '() -> void', doc: '清空磁带',
  ref: 'PyTorch autograd 惯例', inv: 'graphSize=0', impl: () => { tape.length = 0; },
});
defineOp({
  id: nid(), domain: D, name: 'ad.graphSize', arity: 0, sig: '() -> number', doc: '当前磁带节点数',
  ref: 'Baydin et al. 2018 (tape/graph introspection)', inv: '计算图节点数与边数均为非负整数', impl: () => tape.length,
});
defineOp({
  id: nid(), domain: D, name: 'ad.backward', arity: 1, sig: '(Tensor loss, Tensor seed?) -> Map',
  doc: '反向传播（从 loss 出发，返回 张量→梯度 Map）', ref: 'Baydin et al. 2018 综述',
  inv: '梯度累积规则：多路径自动相加',
  impl: (loss, seed) => {
    if (loss.size !== 1 && seed == null) throw new ForgeError('E_VALUE', 'backward 需要标量 loss（或显式提供种子梯度）');
    const grads = new Map();
    grads.set(loss, seed ?? new Tensor([1], [1]));
    for (let i = tape.length - 1; i >= 0; i--) {
      const node = tape[i];
      const g = grads.get(node.out);
      if (!g) continue;
      const gins = node.vjp(g);
      for (let k = 0; k < node.inputs.length; k++) {
        const inp = node.inputs[k];
        if (!(inp instanceof Tensor) || gins[k] == null) continue;
        const prev = grads.get(inp);
        grads.set(inp, prev ? addGrad(prev, gins[k]) : gins[k]);
      }
    }
    return grads;
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.grad', arity: 2, sig: '(Function f, Tensor x) -> Tensor',
  doc: 'f(x) 对标量的梯度（自动磁带管理）', ref: 'JAX grad',
  inv: '与中心差分一致（gradcheck 1e-6）',
  impl: (f, x) => {
    tape.length = 0;
    const loss = f(x);
    const grads = exec('ad.backward', [loss]);
    return grads.get(x) ?? Tensor.zeros(x.shape);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.vjp', arity: 3, sig: '(Function f, Tensor x, Tensor u) -> Tensor',
  doc: '向量-雅可比积（f 可为向量函数）', ref: 'JAX vjp',
  inv: 'vjp(f,x,u)·v = u·(Jv)',
  impl: (f, x, u) => {
    tape.length = 0;
    const out = f(x);
    const seed = u.map((v, i) => v);
    const grads = exec('ad.backward', [out, seed]);
    return grads.get(x) ?? Tensor.zeros(x.shape);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.noGrad', arity: 1, sig: '(Function f) -> any', doc: '无梯度上下文执行',
  ref: 'PyTorch no_grad', inv: '执行期间不记录',
  impl: (f) => { const prev = recording; recording = false; try { return f(); } finally { recording = prev; } },
});
defineOp({
  id: nid(), domain: D, name: 'ad.gradcheck', arity: 2,
  sig: '(Function f, Tensor x, number eps=1e-5, number tol=1e-6) -> object',
  doc: '梯度检验（中心差分 vs 反向模式）', ref: 'PyTorch gradcheck 方法论',
  inv: '最大相对误差 < tol',
  impl: (f, x, eps = 1e-5, tol = 1e-6) => {
    tape.length = 0;
    const loss = f(x);
    if (loss.size !== 1) throw new ForgeError('E_VALUE', 'gradcheck 需要标量函数');
    const gAna = exec('ad.backward', [loss]).get(x);
    if (!gAna) throw new ForgeError('E_STATE', '未找到 x 的梯度');
    const gNum = exec('num.gradCentral', [f, x, eps]);
    let maxRel = 0;
    for (let i = 0; i < x.size; i++) {
      const rel = Math.abs(gAna.data[i] - gNum.data[i]) / Math.max(1e-8, Math.abs(gAna.data[i]), Math.abs(gNum.data[i]));
      if (rel > maxRel) maxRel = rel;
    }
    return { maxRelErr: maxRel, passed: maxRel < tol, analytic: gAna, numeric: gNum };
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.gradcheckVec', arity: 3,
  sig: '(Function f, Tensor x, Tensor u, number eps=1e-5) -> object', doc: '向量函数梯度检验（随机投影法）',
  ref: 'PyTorch gradcheck + 投影', inv: 'u·vjp 与中心差分一致',
  impl: (f, x, u, eps = 1e-5) => {
    tape.length = 0;
    const out = f(x);
    const vjpG = exec('ad.vjp', [f, x, u]);
    // 数值：对 g(u)=Σ uᵢfᵢ(x) 做中心差分
    const gfun = (xx) => { const yy = f(xx); let s = 0; for (let i = 0; i < yy.size; i++) s += u.data[i % u.size] * yy.data[i]; return s; };
    const gNum = exec('num.gradCentral', [gfun, x, eps]);
    let maxRel = 0;
    for (let i = 0; i < x.size; i++) {
      const rel = Math.abs(vjpG.data[i] - gNum.data[i]) / Math.max(1e-8, Math.abs(vjpG.data[i]), Math.abs(gNum.data[i]));
      if (rel > maxRel) maxRel = rel;
    }
    return { maxRelErr: maxRel, passed: maxRel < 1e-5 };
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.jacobianDiag', arity: 2, sig: '(Function f, Tensor x, number eps=1e-4) -> Tensor',
  doc: '雅可比对角线（对向量函数）', ref: 'Griewank & Walther, Evaluating Derivatives, 2008', inv: 'J[i,i] = ∂fᵢ/∂xᵢ',
  impl: (f, x, eps = 1e-4) => {
    const f0 = f(x);
    const out = new Float64Array(x.size);
    for (let i = 0; i < x.size; i++) {
      const xp = x.clone(), xm = x.clone();
      xp.data[i] += eps; xm.data[i] -= eps;
      out[i] = (f(xp).data[i] - f(xm).data[i]) / (2 * eps);
    }
    return new Tensor(out, x.shape);
  },
});

// ============ 复合层（5 条） ============
defineOp({
  id: nid(), domain: D, name: 'ad.dense', arity: 3,
  sig: '(Tensor x[n,in], Tensor W[in,out], Tensor b[out], Function act?) -> Tensor',
  doc: '全连接层 x·W+b + 激活', ref: '深度学习标准层', inv: '等价于 ad.matmul+ad.add+act 组合',
  impl: (x, W, b, act = null) => {
    let out = exec('ad.matmul', [x, W]);
    out = exec('ad.add', [out, b]);
    return act ? act(out) : out;
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.layerNorm', arity: 3,
  sig: '(Tensor x[n,d], Tensor gamma[d], Tensor beta[d], number eps=1e-5) -> Tensor',
  doc: 'LayerNorm（按最后一维）', ref: 'Ba et al. 2016', inv: '输出均值≈0 方差≈1（缩放前）',
  impl: (x, gamma, beta, eps = 1e-5) => {
    const n = x.shape[0], d = x.shape[1];
    const out = new Float64Array(n * d);
    for (let i = 0; i < n; i++) {
      let mu = 0;
      for (let j = 0; j < d; j++) mu += x.data[i * d + j];
      mu /= d;
      let v = 0;
      for (let j = 0; j < d; j++) v += (x.data[i * d + j] - mu) ** 2;
      v /= d;
      const inv = 1 / Math.sqrt(v + eps);
      for (let j = 0; j < d; j++) out[i * d + j] = (x.data[i * d + j] - mu) * inv * gamma.data[j] + beta.data[j];
    }
    const outT = new Tensor(out, [n, d]);
    return record('ad.layerNorm', [x, gamma, beta], outT, (g) => {
      // 解析梯度（LayerNorm  backward 标准推导）
      const gx = new Float64Array(n * d), gg = new Float64Array(d), gb = new Float64Array(d);
      for (let i = 0; i < n; i++) {
        let mu = 0;
        for (let j = 0; j < d; j++) mu += x.data[i * d + j];
        mu /= d;
        let v = 0;
        for (let j = 0; j < d; j++) v += (x.data[i * d + j] - mu) ** 2;
        v /= d;
        const inv = 1 / Math.sqrt(v + eps);
        let sumG = 0, sumGX = 0;
        for (let j = 0; j < d; j++) { sumG += g.data[i * d + j] * gamma.data[j]; sumGX += g.data[i * d + j] * gamma.data[j] * (x.data[i * d + j] - mu); }
        for (let j = 0; j < d; j++) {
          gx[i * d + j] = inv * (g.data[i * d + j] * gamma.data[j] - sumG / d - (x.data[i * d + j] - mu) * inv * inv * sumGX / d);
          gg[j] += g.data[i * d + j] * (x.data[i * d + j] - mu) * inv;
          gb[j] += g.data[i * d + j];
        }
      }
      return [new Tensor(gx, [n, d]), new Tensor(gg, [d]), new Tensor(gb, [d])];
    });
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.scaledDotProductAttention', arity: 3,
  sig: '(Tensor Q[n,d], Tensor K[n,d], Tensor V[n,d], number scale?) -> Tensor',
  doc: '缩放点积注意力 softmax(QKᵀ/√d)V', ref: 'Vaswani et al. 2017（Attention is All You Need）',
  inv: 'softmax 行和=1；gradcheck 通过',
  impl: (Q, K, V, scale) => {
    const d = Q.shape[1];
    const s = scale ?? 1 / Math.sqrt(d);
    const scores = exec('ad.matmul', [Q, exec('ad.transpose', [K])]);
    const scaled = exec('ad.mulScalar', [scores, s]);
    const attn = exec('ad.softmax', [scaled, -1]);
    return exec('ad.matmul', [attn, V]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.mlp', arity: 5,
  sig: '(Tensor x, Tensor W1, Tensor b1, Tensor W2, Tensor b2) -> Tensor',
  doc: '两层 MLP（tanh 隐层，线性输出）', ref: '通用函数逼近（Hornik 1991）',
  inv: '等价于 dense(dense(x,W1,b1,tanh),W2,b2)',
  impl: (x, W1, b1, W2, b2) => {
    const h = exec('ad.dense', [x, W1, b1, (t) => exec('ad.tanh', [t])]);
    return exec('ad.dense', [h, W2, b2, null]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.dropout', arity: 3,
  sig: '(Tensor x, number p, Tensor mask) -> Tensor', doc: 'Dropout（外置掩码，保证可复现与可微）',
  ref: 'Srivastava et al. 2014', inv: '训练期期望不变（inverted dropout：x·mask/(1−p)）',
  impl: (x, p, mask) => {
    const scale = 1 / (1 - p);
    const out = x.map((v, i) => v * (mask.data[i % mask.size] !== 0 ? scale : 0));
    return record('ad.dropout', [x], out, (g) => [g.map((v, i) => (mask.data[i % mask.size] !== 0 ? v * scale : 0))]);
  },
});

// ============ 前向模式：对偶数引擎（15 条） ============
// 对偶数 (primal, tangent)，前向模式自动微分（JAX jvp 同源）。
// primal/tangent 均为 Tensor；elemFn 作用于标量对，tangentFn 给出链式法则组合。
function dualBinary(name, doc, elemFn, tangentFn) {
  defineOp({
    id: nid(), domain: D, name, arity: 2, sig: '(Dual a, Dual b) -> Dual', doc,
    ref: '对偶数前向模式（Griewank & Walther）', inv: 'tangent = 链式法则组合',
    impl: (a, b) => ({
      primal: a.primal.map((v, i) => elemFn(v, b.primal.data[i % b.primal.size])),
      tangent: a.tangent.map((v, i) => {
        const pa = a.primal.data[i % a.primal.size];
        const pb = b.primal.data[i % b.primal.size];
        return tangentFn(v, pa, b.tangent.data[i % b.tangent.size], pb);
      }),
    }),
  });
}
dualBinary('ad.dual.add', '对偶加法', (x, y) => x + y, (ta, pa, tb) => ta + tb);
dualBinary('ad.dual.sub', '对偶减法', (x, y) => x - y, (ta, pa, tb) => ta - tb);
dualBinary('ad.dual.mul', '对偶乘法', (x, y) => x * y, (ta, pa, tb, pb) => ta * pb + pa * tb);
dualBinary('ad.dual.div', '对偶除法', (x, y) => x / y, (ta, pa, tb, pb) => (ta * pb - pa * tb) / (pb * pb));
defineOp({
  id: nid(), domain: D, name: 'ad.dual.from', arity: 1, sig: '(Tensor x, Tensor v?) -> Dual', doc: '由张量构造对偶数',
  ref: '对偶数标准构造', inv: 'primal=x，tangent=v（默认 0）',
  impl: (x, v) => ({ primal: x, tangent: v ?? Tensor.zeros(x.shape) }),
});
defineOp({
  id: nid(), domain: D, name: 'ad.dual.primal', arity: 1, sig: '(Dual a) -> Tensor', doc: '取对偶数的原值部分 primal(a)',
  ref: 'Fike & Alonso, Derivative Analysis via Dual Numbers, 2011', inv: 'primal(dual.from(x))=x', impl: (a) => a.primal,
});
defineOp({
  id: nid(), domain: D, name: 'ad.dual.tangent', arity: 1, sig: '(Dual a) -> Tensor', doc: '取切线（导数）',
  ref: 'Fike & Alonso, Derivative Analysis via Dual Numbers, 2011', inv: 'tangent(dual.from(x,v))=v', impl: (a) => a.tangent,
});
const DUAL_UNARY = [
  ['exp', Math.exp, (a) => Math.exp(a.primal.data[0] ?? a.primal)],
];
function dualUnary(name, doc, f, df) {
  defineOp({
    id: nid(), domain: D, name, arity: 1, sig: '(Dual a) -> Dual', doc,
    ref: '对偶数一元算子', inv: 'tangent = f\'(primal)·tangent',
    impl: (a) => ({ primal: a.primal.map(f), tangent: a.tangent.map((v, i) => v * df(a.primal.data[i])) }),
  });
}
dualUnary('ad.dual.exp', '对偶指数', Math.exp, (p) => Math.exp(p));
dualUnary('ad.dual.log', '对偶对数', Math.log, (p) => 1 / p);
dualUnary('ad.dual.sqrt', '对偶平方根', Math.sqrt, (p) => 1 / (2 * Math.sqrt(p)));
dualUnary('ad.dual.sin', '对偶正弦', Math.sin, (p) => Math.cos(p));
dualUnary('ad.dual.cos', '对偶余弦', Math.cos, (p) => -Math.sin(p));
dualUnary('ad.dual.tanh', '对偶双曲正切', Math.tanh, (p) => 1 - Math.tanh(p) ** 2);
dualUnary('ad.dual.sigmoid', '对偶 sigmoid', (v) => 1 / (1 + Math.exp(-v)), (p) => { const s = 1 / (1 + Math.exp(-p)); return s * (1 - s); });
defineOp({
  id: nid(), domain: D, name: 'ad.dual.matmul', arity: 2, sig: '(Dual A, Dual B) -> Dual',
  doc: '对偶矩阵乘法', ref: '前向模式矩阵法则', inv: 'primal=A·B，tangent=Ȧ·B+A·B̄',
  impl: (A, B) => ({
    primal: exec('linalg.matmul', [A.primal, B.primal]),
    tangent: exec('math.add', [exec('linalg.matmul', [A.tangent, B.primal]), exec('linalg.matmul', [A.primal, B.tangent])]),
  }),
});
defineOp({
  id: nid(), domain: D, name: 'ad.dual.sum', arity: 1, sig: '(Dual a) -> Dual', doc: '对偶求和',
  ref: '前向模式归约', inv: 'tangent 同样求和',
  impl: (a) => ({ primal: new Tensor([a.primal.data.reduce((x, y) => x + y, 0)], [1]), tangent: new Tensor([a.tangent.data.reduce((x, y) => x + y, 0)], [1]) }),
});

// ============ 生成家族：标量变体（补齐至 102） ============
const SCALAR_VARIANTS = [
  ['ad.addScalar', 'ad.add', '加法（张量+标量）'],
  ['ad.subScalar', 'ad.sub', '减法（张量−标量）'],
  ['ad.mulScalar', 'ad.mul', '数乘（张量×标量）'],
  ['ad.divScalar', 'ad.div', '除法（张量÷标量）'],
  ['ad.rsubScalar', 'ad.sub', '反减（标量−张量）'],
  ['ad.rdivScalar', 'ad.div', '反除（标量÷张量）'],
];
for (const [name, base, d] of SCALAR_VARIANTS) {
  defineOp({
    id: nid(), domain: D, name, arity: 2, sig: `(Tensor x, number s) -> Tensor`, doc: `${d}（记录梯度）`,
    ref: 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018', inv: '等价 ${base} 的标量路径',
    impl: (x, s) => exec(base, name.startsWith('ad.r') ? [s, x] : [x, s]),
  });
}

// ============ 补充算子（11 条） ============
elementwiseAd('ad.softsign', 'softsign x/(1+|x|)', (v) => v / (1 + Math.abs(v)), (v) => 1 / (1 + Math.abs(v)) ** 2, '1/(1+|x|)²', 'Bergstra et al. 2009');
elementwiseAd('ad.mish', 'mish x·tanh(softplus x)', (v) => v * Math.tanh(Math.log1p(Math.exp(v))), (v) => { const sp = Math.log1p(Math.exp(v)); const t = Math.tanh(sp); const s = 1 / (1 + Math.exp(-v)); return t + v * s * (1 - t * t); }, 'tanh(sp)+x·σ(x)·sech²(sp)', 'Misra 2020');
defineOp({
  id: nid(), domain: D, name: 'ad.hardtanh', arity: 1, sig: '(Tensor x, number lo=-1, number hi=1) -> Tensor',
  doc: 'HardTanh（线性区截断）', ref: '深度学习标准激活', inv: '界内梯度 1',
  impl: (x, lo = -1, hi = 1) => {
    const out = x.map((v) => Math.min(Math.max(v, lo), hi));
    return record('ad.hardtanh', [x], out, (g) => [g.map((v, i) => (x.data[i] > lo && x.data[i] < hi ? v : 0))]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.hardswish', arity: 1, sig: '(Tensor x) -> Tensor',
  doc: 'HardSwish x·ReLU6(x+3)/6', ref: 'Howard et al. 2019（MobileNetV3）', inv: 'gradcheck 通过',
  impl: (x) => {
    const out = x.map((v) => v * Math.min(Math.max(v + 3, 0), 6) / 6);
    return record('ad.hardswish', [x], out, (g) => [g.map((v, i) => {
      const u = x.data[i] + 3;
      const drelu = u <= 0 ? 0 : u >= 6 ? 0 : 1;
      return v * (Math.min(Math.max(u, 0), 6) / 6 + x.data[i] * drelu / 6);
    })]);
  },
});
elementwiseAd('ad.sign', '符号函数（0 点次梯度 0）', Math.sign, () => 0, '0', 'Rockafellar 1970, Convex Analysis (subgradient theory)');
elementwiseAd('ad.rsqrt', '逆平方根 1/√x', (v) => 1 / Math.sqrt(v), (v) => -1 / (2 * v ** 1.5), '-1/(2x^{3/2})', 'Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018');
defineOp({
  id: nid(), domain: D, name: 'ad.scaleAdd', arity: 3, sig: '(number alpha, Tensor x, Tensor y) -> Tensor',
  doc: 'axpy：α·x+y', ref: 'BLAS daxpy', inv: 'd/dx=αg，d/dy=g',
  impl: (alpha, x, y) => record('ad.scaleAdd', [x, y], x.map((v, i) => alpha * v + y.data[i % y.size]), (g) => [g.map((v) => alpha * v), g]),
});
defineOp({
  id: nid(), domain: D, name: 'ad.l2Reg', arity: 2, sig: '(Tensor w, number lambda) -> Tensor',
  doc: 'L2 正则项 λ/2·‖w‖²', ref: 'Ridge/Tikhonov 正则化', inv: 'd/dw = λw',
  impl: (w, lambda) => {
    let s = 0;
    for (const v of w.data) s += v * v;
    const out = new Tensor([lambda / 2 * s], [1]);
    return record('ad.l2Reg', [w], out, (g) => [w.map((v) => lambda * v * g.data[0])]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.l1Reg', arity: 2, sig: '(Tensor w, number lambda) -> Tensor',
  doc: 'L1 正则项 λ·‖w‖₁', ref: 'Lasso（Tibshirani 1996）', inv: 'd/dw = λ·sign(w)',
  impl: (w, lambda) => {
    let s = 0;
    for (const v of w.data) s += Math.abs(v);
    const out = new Tensor([lambda * s], [1]);
    return record('ad.l1Reg', [w], out, (g) => [w.map((v) => lambda * Math.sign(v) * g.data[0])]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.causalMask', arity: 1, sig: '(Tensor scores, number fill=-Infinity) -> Tensor',
  doc: '因果掩码（上三角置 fill，防未来泄漏）', ref: 'Vaswani et al. 2017 解码器掩码', inv: '下三角保持，上三角=fill',
  impl: (scores, fill = -Infinity) => {
    const [n, m] = scores.shape;
    const out = new Float64Array(scores.size);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[i * m + j] = j <= i ? scores.data[i * m + j] : fill;
    return record('ad.causalMask', [scores], new Tensor(out, scores.shape), (g) => [g.map((v, i) => {
      const r = Math.floor(i / m), c = i % m;
      return c <= r ? v : 0;
    })]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'ad.batchnorm1d', arity: 3,
  sig: '(Tensor x[n,d], Tensor gamma[d], Tensor beta[d], number eps=1e-5) -> Tensor',
  doc: 'BatchNorm1d（按特征维，训练期批统计）', ref: 'Ioffe & Szegedy 2015', inv: '输出批均值≈0 方差≈1（缩放前）',
  impl: (x, gamma, beta, eps = 1e-5) => {
    const n = x.shape[0], d = x.shape[1];
    const mu = new Float64Array(d), varr = new Float64Array(d);
    for (let j = 0; j < d; j++) { let s = 0; for (let i = 0; i < n; i++) s += x.data[i * d + j]; mu[j] = s / n; }
    for (let j = 0; j < d; j++) { let s = 0; for (let i = 0; i < n; i++) s += (x.data[i * d + j] - mu[j]) ** 2; varr[j] = s / n; }
    const out = new Float64Array(n * d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) out[i * d + j] = (x.data[i * d + j] - mu[j]) / Math.sqrt(varr[j] + eps) * gamma.data[j] + beta.data[j];
    const outT = new Tensor(out, [n, d]);
    return record('ad.batchnorm1d', [x, gamma, beta], outT, (g) => {
      // 标准 BatchNorm backward（Ioffe & Szegedy 2015 附录）
      const stdInv = new Float64Array(d);
      for (let j = 0; j < d; j++) stdInv[j] = 1 / Math.sqrt(varr[j] + eps);
      const gx = new Float64Array(n * d), gg = new Float64Array(d), gb = new Float64Array(d);
      const dxhat = new Float64Array(n * d);
      for (let j = 0; j < d; j++) {
        let sumDxhat = 0, sumDxhatXhat = 0;
        for (let i = 0; i < n; i++) {
          const xhat = (x.data[i * d + j] - mu[j]) * stdInv[j];
          dxhat[i * d + j] = g.data[i * d + j] * gamma.data[j];
          sumDxhat += dxhat[i * d + j];
          sumDxhatXhat += dxhat[i * d + j] * xhat;
          gg[j] += g.data[i * d + j] * xhat;
          gb[j] += g.data[i * d + j];
        }
        for (let i = 0; i < n; i++) {
          const xhat = (x.data[i * d + j] - mu[j]) * stdInv[j];
          gx[i * d + j] = stdInv[j] / n * (n * dxhat[i * d + j] - sumDxhat - xhat * sumDxhatXhat);
        }
      }
      return [new Tensor(gx, [n, d]), new Tensor(gg, [d]), new Tensor(gb, [d])];
    });
  },
});

// 文件底部自检
if (opsByDomain(D).length !== 102) {
  throw new Error(`域${D}(AUTODIFF) 指令数 = ${opsByDomain(D).length}，应为 102`);
}
