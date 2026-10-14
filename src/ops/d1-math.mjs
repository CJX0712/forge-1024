// 域1 数学基元 MATH（102 条微指令）
// 内容：逐元素运算族（广播/标量变体）、归约族、范数与距离、构造器、形状操作。
// 参考：NumPy ufunc 语义与广播规则；Golub & Van Loan《Matrix Computations》。
import { defineOp, opsByDomain, exec } from '../core/registry.mjs';
import { ForgeError } from '../core/errors.mjs';
import { Tensor, prod as coreProd, stridesOf, broadcastShapes, elementwise, reduceTensor, expectTensor } from '../core/tensor.mjs';

const D = 1;
let _s = 0;
const nid = () => `D${D}-${String(++_s).padStart(3, '0')}`;

// ============ 生成式家族 A：二元逐元素（10 基础 × {TT, TS, ST} = 30 条） ============
// TT: Tensor⊗Tensor（广播）；TS: Tensor⊕标量；ST: 标量⊕Tensor。
export const BINARY_TABLE = [
  ['add', (a, b) => a + b, '加法'],
  ['sub', (a, b) => a - b, '减法'],
  ['mul', (a, b) => a * b, '乘法'],
  ['div', (a, b) => a / b, '除法'],
  ['mod', (a, b) => a % b, '取模（JS 语义，符号随被除数）'],
  ['pow', (a, b) => a ** b, '幂'],
  ['atan2', Math.atan2, '二参数反正切'],
  ['hypot', Math.hypot, '斜边 √(a²+b²)'],
  ['maximum', Math.max, '逐元素取大（NumPy ufunc 命名）'],
  ['minimum', Math.min, '逐元素取小（NumPy ufunc 命名）'],
];
for (const [n, f, d] of BINARY_TABLE) {
  defineOp({
    id: nid(), domain: D, name: `math.${n}`,
    sig: '(Tensor a, Tensor b) -> Tensor', doc: `逐元素${d}（NumPy 广播）`,
    ref: 'NumPy ufunc 语义 / IEEE 754', inv: `out[i] = ${n}(a[i], b[i])；广播规则同 NumPy`,
    impl: (a, b) => { expectTensor(a, 'a'); expectTensor(b, 'b'); return elementwise(a, b, f); },
  });
  defineOp({
    id: nid(), domain: D, name: `math.${n}Scalar`,
    sig: '(Tensor a, number s) -> Tensor', doc: `张量与标量逐元素${d}（右标量）`,
    ref: 'NumPy ufunc 语义', inv: `out[i] = ${n}(a[i], s)`,
    impl: (a, s) => { expectTensor(a, 'a'); return a.map(v => f(v, s)); },
  });
  defineOp({
    id: nid(), domain: D, name: `math.scalar${n[0].toUpperCase()}${n.slice(1)}`,
    sig: '(number s, Tensor a) -> Tensor', doc: `标量与张量逐元素${d}（左标量）`,
    ref: 'NumPy ufunc 语义', inv: `out[i] = ${n}(s, a[i])`,
    impl: (s, a) => { expectTensor(a, 'a'); return a.map(v => f(s, v)); },
  });
}

// ============ 生成式家族 B：一元逐元素（28 条） ============
export const UNARY_TABLE = [
  ['sin', Math.sin], ['cos', Math.cos], ['tan', Math.tan],
  ['asin', Math.asin], ['acos', Math.acos], ['atan', Math.atan],
  ['sinh', Math.sinh], ['cosh', Math.cosh], ['tanh', Math.tanh],
  ['asinh', Math.asinh], ['acosh', Math.acosh], ['atanh', Math.atanh],
  ['exp', Math.exp], ['expm1', Math.expm1],
  ['log', Math.log], ['log1p', Math.log1p], ['log2', Math.log2], ['log10', Math.log10],
  ['sqrt', Math.sqrt], ['cbrt', Math.cbrt],
  ['abs', Math.abs], ['sign', Math.sign],
  ['floor', Math.floor], ['ceil', Math.ceil], ['round', Math.round], ['trunc', Math.trunc],
  ['square', (x) => x * x], ['reciprocal', (x) => 1 / x],
];
for (const [n, f] of UNARY_TABLE) {
  defineOp({
    id: nid(), domain: D, name: `math.${n}`,
    sig: '(Tensor) -> Tensor', doc: `逐元素 ${n}`,
    ref: 'IEEE 754 / ECMAScript Math', inv: `out[i] = ${n}(in[i])`,
    impl: (t) => { expectTensor(t, 't'); return t.map(f); },
  });
}

// ============ 手写核心：内积与范数 ============
function vec1d(t, who) {
  expectTensor(t, who);
  if (t.ndim !== 1) throw new ForgeError('E_SHAPE', `${who}: 期望 1D 向量，得到 shape [${t.shape}]`);
  return t.data;
}

defineOp({
  id: nid(), domain: D, name: 'math.dot', sig: '(Tensor x, Tensor y) -> number',
  doc: '向量点积 x·y = Σxᵢyᵢ', ref: 'Golub & Van Loan §1.3 / BLAS ddot',
  inv: 'dot(x,y)=dot(y,x)；dot(x,x)=‖x‖²',
  impl: (x, y) => {
    const a = vec1d(x, 'x'), b = vec1d(y, 'y');
    if (a.length !== b.length) throw new ForgeError('E_SHAPE', `点积长度不符: ${a.length} vs ${b.length}`);
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  },
});

defineOp({
  id: nid(), domain: D, name: 'math.cross', sig: '(Tensor x, Tensor y) -> Tensor',
  doc: '3D 向量叉积 x×y', ref: '线性代数标准定义',
  inv: 'x×y ⊥ x,y；‖x×y‖=‖x‖‖y‖sinθ；(x×y)·x=0',
  impl: (x, y) => {
    const a = vec1d(x, 'x'), b = vec1d(y, 'y');
    if (a.length !== 3 || b.length !== 3) throw new ForgeError('E_VALUE', '叉积仅支持 3D 向量');
    return Tensor.fromFlat([
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ], [3]);
  },
});

defineOp({
  id: nid(), domain: D, name: 'math.outer', sig: '(Tensor x, Tensor y) -> Tensor',
  doc: '外积 x⊗y，shape (m,n)', ref: 'Golub & Van Loan §1.3',
  inv: 'outer(x,y)[i,j]=x[i]·y[j]',
  impl: (x, y) => {
    const a = vec1d(x, 'x'), b = vec1d(y, 'y');
    const out = new Float64Array(a.length * b.length);
    for (let i = 0; i < a.length; i++) for (let j = 0; j < b.length; j++) out[i * b.length + j] = a[i] * b[j];
    return new Tensor(out, [a.length, b.length]);
  },
});

function flatData(t) { expectTensor(t, 't'); return t.data; }

defineOp({
  id: nid(), domain: D, name: 'math.l1', sig: '(Tensor) -> number', doc: 'L1 范数 ‖x‖₁=Σ|xᵢ|',
  ref: 'NumPy norm / 泛函分析', inv: '‖x‖₁≥0，=0 ⟺ x=0',
  impl: (t) => { let s = 0; for (const v of flatData(t)) s += Math.abs(v); return s; },
});
defineOp({
  id: nid(), domain: D, name: 'math.l2', sig: '(Tensor) -> number', doc: 'L2 范数 ‖x‖₂=√(Σxᵢ²)',
  ref: 'NumPy norm / BLAS dnrm2', inv: '‖x‖₂²=dot(x,x)',
  impl: (t) => { let s = 0; for (const v of flatData(t)) s += v * v; return Math.sqrt(s); },
});
defineOp({
  id: nid(), domain: D, name: 'math.linf', sig: '(Tensor) -> number', doc: 'L∞ 范数 max|xᵢ|',
  ref: 'NumPy norm', inv: '‖x‖∞≥‖x‖₂≥‖x‖₁/√n',
  impl: (t) => { let m = 0; for (const v of flatData(t)) m = Math.max(m, Math.abs(v)); return m; },
});
defineOp({
  id: nid(), domain: D, name: 'math.lp', sig: '(Tensor, number p) -> number', doc: 'Lp 范数 (Σ|xᵢ|^p)^(1/p)',
  ref: '泛函分析 / NumPy norm', inv: 'p=1,2,∞ 与 math.l1/l2/linf 一致',
  impl: (t, p) => {
    if (!(p >= 1)) throw new ForgeError('E_VALUE', 'p 必须 ≥1');
    let s = 0;
    for (const v of flatData(t)) s += Math.abs(v) ** p;
    return s ** (1 / p);
  },
});
defineOp({
  id: nid(), domain: D, name: 'math.cosine', sig: '(Tensor x, Tensor y) -> number',
  doc: '余弦相似度 (x·y)/(‖x‖‖y‖)', ref: '信息检索标准度量',
  inv: 'cos(x,y)∈[-1,1]；cos(x,x)=1（x≠0）；零向量返回 0',
  impl: (x, y) => {
    const a = vec1d(x, 'x'), b = vec1d(y, 'y');
    if (a.length !== b.length) throw new ForgeError('E_SHAPE', '余弦相似度长度不符');
    let d = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    if (na === 0 || nb === 0) return 0;
    return d / (Math.sqrt(na) * Math.sqrt(nb));
  },
});

const DISTS = [
  ['manhattan', (d) => Math.abs(d), 'L1 距离'],
  ['euclidean', (d) => d * d, 'L2 距离（开方前）'],
  ['sqeuclidean', (d) => d * d, '平方 L2 距离'],
  ['chebyshev', null, 'L∞ 距离'],
];
for (const [n, f, d] of DISTS) {
  defineOp({
    id: nid(), domain: D, name: `math.${n}`, sig: '(Tensor x, Tensor y) -> number',
    doc: `向量${d}`, ref: 'Minkowski 距离族', inv: `d(x,y)≥0；d(x,y)=d(y,x)；d(x,x)=0`,
    impl: (x, y) => {
      const a = vec1d(x, 'x'), b = vec1d(y, 'y');
      if (a.length !== b.length) throw new ForgeError('E_SHAPE', `${n} 距离长度不符`);
      if (n === 'chebyshev') { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; }
      let s = 0;
      for (let i = 0; i < a.length; i++) s += f(a[i] - b[i]);
      return n === 'euclidean' ? Math.sqrt(s) : s;
    },
  });
}

// ============ 归约族（全元素 -> 标量） ============
const SCALAR_REDUCTIONS = [
  ['sum', 0, (a, v) => a + v, '总和'],
  ['prod', 1, (a, v) => a * v, '连乘'],
  ['min', Infinity, (a, v) => Math.min(a, v), '最小值'],
  ['max', -Infinity, (a, v) => Math.max(a, v), '最大值'],
];
for (const [n, init, f, d] of SCALAR_REDUCTIONS) {
  defineOp({
    id: nid(), domain: D, name: `math.${n}`, sig: '(Tensor) -> number', doc: `全元素${d}`,
    ref: 'NumPy reduction', inv: `空集返回 ${init}；${n}(c·x)=c·${n}(x)（sum/prod）`,
    impl: (t) => { const r = reduceTensor(t, null, init, f); return r.tensor.scalar(); },
  });
}

defineOp({
  id: nid(), domain: D, name: 'math.mean', sig: '(Tensor) -> number', doc: '全元素均值',
  ref: 'NumPy mean', inv: 'mean(c·x)=c·mean(x)',
  impl: (t) => { const r = reduceTensor(t, null, 0, (a, v) => a + v); return r.tensor.scalar() / r.count; },
});
defineOp({
  id: nid(), domain: D, name: 'math.var', sig: '(Tensor) -> number', doc: '全元素方差（总体，两遍法数值稳定）',
  ref: 'NumPy var / 统计学定义', inv: 'var≥0；var(c·x)=c²·var(x)',
  impl: (t) => {
    const r = reduceTensor(t, null, 0, (a, v) => a + v);
    const mu = r.tensor.scalar() / r.count;
    const r2 = reduceTensor(t, null, 0, (a, v) => a + (v - mu) * (v - mu));
    return r2.tensor.scalar() / r.count;
  },
});
defineOp({
  id: nid(), domain: D, name: 'math.std', sig: '(Tensor) -> number', doc: '全元素标准差 √var',
  ref: 'NumPy std', inv: 'std=√var；std≥0',
  impl: (t) => Math.sqrt(exec('math.var', [t])),
});
defineOp({
  id: nid(), domain: D, name: 'math.argmin', sig: '(Tensor) -> number', doc: '最小元素的扁平索引（首个）',
  ref: 'NumPy argmin', inv: 't.at(argmin(t))=min(t)',
  impl: (t) => { let bi = -1, bv = Infinity; for (let i = 0; i < t.size; i++) if (t.data[i] < bv) { bv = t.data[i]; bi = i; } return bi; },
});
defineOp({
  id: nid(), domain: D, name: 'math.argmax', sig: '(Tensor) -> number', doc: '最大元素的扁平索引（首个）',
  ref: 'NumPy argmax', inv: 't.at(argmax(t))=max(t)',
  impl: (t) => { let bi = -1, bv = -Infinity; for (let i = 0; i < t.size; i++) if (t.data[i] > bv) { bv = t.data[i]; bi = i; } return bi; },
});
defineOp({
  id: nid(), domain: D, name: 'math.median', sig: '(Tensor) -> number', doc: '中位数（偶数长度取中间两值均值）',
  ref: 'Hampel, Ronchetti, Rousseeuw & Stahel, Robust Statistics: The Approach Based on Influence Functions, 1986',
  inv: '至少一半元素 ≤ median ≤ 另一半',
  impl: (t) => quantileSorted(Array.from(t.data).sort((a, b) => a - b), 0.5),
});
defineOp({
  id: nid(), domain: D, name: 'math.quantile', sig: '(Tensor, number q) -> number',
  doc: 'q 分位数（线性插值，NumPy linear 方法）', ref: 'NumPy quantile',
  inv: 'quantile(0)=min；quantile(1)=max；quantile(0.5)=median',
  impl: (t, q) => {
    if (q < 0 || q > 1) throw new ForgeError('E_VALUE', 'q∈[0,1]');
    return quantileSorted(Array.from(t.data).sort((a, b) => a - b), q);
  },
});
defineOp({
  id: nid(), domain: D, name: 'math.mad', sig: '(Tensor) -> number',
  doc: '中位数绝对偏差 median(|x−median(x)|)', ref: 'Hampel 1974, JASA 69:383-393; Leys et al. 2013 JESP',
  inv: 'mad≥0；对异常值不敏感（加一个极端值不变）',
  impl: (t) => {
    const sorted = Array.from(t.data).sort((a, b) => a - b);
    const m = quantileSorted(sorted, 0.5);
    const dev = sorted.map(v => Math.abs(v - m)).sort((a, b) => a - b);
    return quantileSorted(dev, 0.5);
  },
});

function quantileSorted(sorted, q) {
  const n = sorted.length;
  if (n === 0) throw new ForgeError('E_VALUE', '空张量无分位数');
  if (n === 1) return sorted[0];
  const pos = q * (n - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

// ============ 前缀和族 ============
defineOp({
  id: nid(), domain: D, name: 'math.cumsum', sig: '(Tensor) -> Tensor',
  doc: '前缀和（1D）', ref: 'NumPy cumsum', inv: 'cumsum[n−1]=sum；cumsum[k]−cumsum[k−1]=x[k]',
  impl: (t) => {
    const a = vec1d(t, 't');
    const out = new Float64Array(a.length);
    let s = 0;
    for (let i = 0; i < a.length; i++) { s += a[i]; out[i] = s; }
    return new Tensor(out, [a.length]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'math.cumprod', sig: '(Tensor) -> Tensor',
  doc: '前缀积（1D）', ref: 'NumPy cumprod', inv: 'cumprod[n−1]=prod',
  impl: (t) => {
    const a = vec1d(t, 't');
    const out = new Float64Array(a.length);
    let s = 1;
    for (let i = 0; i < a.length; i++) { s *= a[i]; out[i] = s; }
    return new Tensor(out, [a.length]);
  },
});

// ============ 轴归约族（去掉该轴，NumPy keepdims=false 语义） ============
const AXIS_REDUCTIONS = [
  ['sumAxis', 0, (a, v) => a + v, '求和'],
  ['meanAxis', 0, (a, v) => a + v, '均值'],
  ['maxAxis', -Infinity, (a, v) => Math.max(a, v), '最大值'],
  ['minAxis', Infinity, (a, v) => Math.min(a, v), '最小值'],
];
for (const [n, init, f, d] of AXIS_REDUCTIONS) {
  defineOp({
    id: nid(), domain: D, name: `math.${n}`, sig: '(Tensor, number axis) -> Tensor', doc: `沿轴${d}`,
    ref: 'NumPy axis reduction', inv: `axis 归约后该轴消失；与逐列循环一致`,
    impl: (t, axis) => {
      const r = reduceTensor(t, [axis], init, f);
      // 去掉长度为 1 的被归约轴（NumPy keepdims=false）
      const outShape = r.tensor.shape.filter((_, i) => !(i === (axis < 0 ? axis + t.ndim : axis)));
      const dropped = r.tensor.reshape(outShape);
      if (n === 'meanAxis') return dropped.map(v => v / t.shape[axis < 0 ? axis + t.ndim : axis]);
      return dropped;
    },
  });
}
defineOp({
  id: nid(), domain: D, name: 'math.argmaxAxis', sig: '(Tensor, number axis) -> Tensor',
  doc: '沿轴 argmax（返回扁平前该轴内索引）', ref: 'NumPy argmax(axis)',
  inv: '对每个输出槽位，取到的元素等于该轴最大值',
  impl: (t, axis) => {
    const ax = axis < 0 ? axis + t.ndim : axis;
    const outShape = t.shape.filter((_, i) => i !== ax);
    const out = new Float64Array(coreProd(outShape));
    const inner = t.shape.slice(ax + 1).reduce((a, b) => a * b, 1);
    const outer = t.shape.slice(0, ax).reduce((a, b) => a * b, 1);
    const red = t.shape[ax];
    for (let o = 0; o < outer; o++) for (let i = 0; i < inner; i++) {
      let best = -Infinity, bi = 0;
      for (let r = 0; r < red; r++) {
        const v = t.data[(o * red + r) * inner + i];
        if (v > best) { best = v; bi = r; }
      }
      out[o * inner + i] = bi;
    }
    return new Tensor(out, outShape);
  },
});
defineOp({
  id: nid(), domain: D, name: 'math.varAxis', sig: '(Tensor, number axis) -> Tensor',
  doc: '沿轴方差（两遍法）', ref: 'NumPy var(axis)', inv: 'varAxis(x,axis).mean(其余轴) ≈ var(x)',
  impl: (t, axis) => {
    const ax = axis < 0 ? axis + t.ndim : axis;
    const m = exec('math.meanAxis', [t, ax]);
    // 逐槽两遍法：先展开每槽均值，再累加平方差
    const outShape = t.shape.filter((_, i) => i !== ax);
    const red = t.shape[ax];
    const out = new Float64Array(coreProd(outShape));
    const inner = t.shape.slice(ax + 1).reduce((a, b) => a * b, 1);
    const outerN = t.shape.slice(0, ax).reduce((a, b) => a * b, 1);
    for (let o = 0; o < outerN; o++) for (let i = 0; i < inner; i++) {
      let s = 0;
      const mu = m.data[o * inner + i];
      for (let r2 = 0; r2 < red; r2++) { const v = t.data[(o * red + r2) * inner + i]; s += (v - mu) * (v - mu); }
      out[o * inner + i] = s / red;
    }
    return new Tensor(out, outShape);
  },
});

// ============ 构造器 ============
defineOp({
  id: nid(), domain: D, name: 'math.zeros', sig: '(number[] shape) -> Tensor', doc: '全零张量',
  ref: 'NumPy zeros', inv: '所有元素为 0', impl: (shape) => Tensor.zeros(shape),
});
defineOp({
  id: nid(), domain: D, name: 'math.ones', sig: '(number[] shape) -> Tensor', doc: '全一张量',
  ref: 'NumPy ones', inv: '所有元素为 1', impl: (shape) => Tensor.ones(shape),
});
defineOp({
  id: nid(), domain: D, name: 'math.full', sig: '(number[] shape, number v) -> Tensor', doc: '常数张量',
  ref: 'NumPy full', inv: '所有元素为 v', impl: (shape, v) => Tensor.full(shape, v),
});
defineOp({
  id: nid(), domain: D, name: 'math.eye', sig: '(number n, number m?) -> Tensor', doc: '单位矩阵',
  ref: 'NumPy eye', inv: 'eye·A=A；eye 对称', impl: (n, m) => Tensor.eye(n, m ?? n),
});
defineOp({
  id: nid(), domain: D, name: 'math.arange', sig: '(number stop, number start?, number step?) -> Tensor',
  doc: '等差序列', ref: 'NumPy arange', inv: '长度=ceil((stop−start)/step)',
  impl: (stop, start, step) => Tensor.arange(stop, start ?? 0, step ?? 1),
});
defineOp({
  id: nid(), domain: D, name: 'math.linspace', sig: '(number a, number b, number n) -> Tensor',
  doc: '闭区间等分', ref: 'NumPy linspace', inv: '首=a 尾=b；相邻差=(b−a)/(n−1)',
  impl: (a, b, n) => Tensor.linspace(a, b, n),
});

// ============ 形状操作 ============
defineOp({
  id: nid(), domain: D, name: 'math.concat', sig: '(Tensor[] ts, number axis?) -> Tensor',
  doc: '沿轴拼接（非轴维必须一致）', ref: 'NumPy concatenate',
  inv: '拼接后该轴长度=各输入该轴长度之和',
  impl: (ts, axis = 0) => {
    if (!Array.isArray(ts) || ts.length === 0) throw new ForgeError('E_VALUE', 'concat 需要非空张量数组');
    const ax = axis < 0 ? axis + ts[0].ndim : axis;
    for (const t of ts) {
      expectTensor(t, 't');
      if (t.ndim !== ts[0].ndim) throw new ForgeError('E_SHAPE', 'concat 维数不一致');
      for (let i = 0; i < t.ndim; i++) if (i !== ax && t.shape[i] !== ts[0].shape[i]) throw new ForgeError('E_SHAPE', `concat 非轴维不一致: 轴${i}`);
    }
    const outShape = ts[0].shape.slice();
    outShape[ax] = ts.reduce((s, t) => s + t.shape[ax], 0);
    const out = new Float64Array(coreProd(outShape));
    const inner = outShape.slice(ax + 1).reduce((a, b) => a * b, 1);
    const outer = outShape.slice(0, ax).reduce((a, b) => a * b, 1);
    let off = 0;
    for (const t of ts) {
      const block = t.shape[ax] * inner;
      for (let o = 0; o < outer; o++) {
        out.set(t.data.subarray(o * block, (o + 1) * block), (o * outShape[ax] + off) * inner);
      }
      off += t.shape[ax];
    }
    return new Tensor(out, outShape);
  },
});
defineOp({
  id: nid(), domain: D, name: 'math.stack', sig: '(Tensor[] ts, number axis?) -> Tensor',
  doc: '沿新轴堆叠', ref: 'NumPy stack', inv: '结果维数=输入维数+1；新轴长度=输入个数',
  impl: (ts, axis = 0) => {
    if (!Array.isArray(ts) || ts.length === 0) throw new ForgeError('E_VALUE', 'stack 需要非空张量数组');
    const n = ts.length;
    const expanded = ts.map(t => t.reshape(...insertOne(t.shape, axis < 0 ? axis + t.ndim + 1 : axis, 1)));
    return exec('math.concat', [expanded, axis < 0 ? axis + t0ndim(expanded) : axis]);
  },
});
function insertOne(shape, at, v) { const s = shape.slice(); s.splice(at, 0, v); return s; }
function t0ndim(ts) { return ts[0].ndim; }

defineOp({
  id: nid(), domain: D, name: 'math.tile', sig: '(Tensor t, number[] reps) -> Tensor',
  doc: '整体复制拼接', ref: 'NumPy tile', inv: '结果形状=逐维乘 reps',
  impl: (t, reps) => {
    expectTensor(t, 't');
    const outShape = t.shape.map((d, i) => d * (reps[i] ?? 1));
    const out = new Float64Array(coreProd(outShape));
    const idx = new Array(t.ndim);
    for (let p = 0; p < out.length; p++) {
      let r = p;
      for (let d = t.ndim - 1; d >= 0; d--) { idx[d] = r % outShape[d]; r = (r - idx[d]) / outShape[d]; }
      const st = stridesOf(t.shape);
      let src = 0;
      for (let d = 0; d < t.ndim; d++) src += (idx[d] % t.shape[d]) * st[d];
      out[p] = t.data[src];
    }
    return new Tensor(out, outShape);
  },
});
defineOp({
  id: nid(), domain: D, name: 'math.pad', sig: '(Tensor t, number[][] pads, number v?) -> Tensor',
  doc: '常数填充（pads=[[前,后],...] 每维一对）', ref: 'NumPy pad',
  inv: '输出形状=原形状+前后填充；填充区=v（默认 0）',
  impl: (t, pads, v = 0) => {
    expectTensor(t, 't');
    if (pads.length !== t.ndim) throw new ForgeError('E_SHAPE', 'pads 维数不符');
    const outShape = t.shape.map((d, i) => d + pads[i][0] + pads[i][1]);
    const out = new Float64Array(coreProd(outShape)).fill(v);
    const inner = t.shape.slice(1).reduce((a, b) => a * b, 1);
    for (let i = 0; i < t.shape[0]; i++) {
      const srcOff = i * inner;
      const dstOff = (i + pads[0][0]) * (outShape.slice(1).reduce((a, b) => a * b, 1));
      out.set(t.data.subarray(srcOff, srcOff + inner), dstOff);
    }
    return new Tensor(out, outShape);
  },
});
defineOp({
  id: nid(), domain: D, name: 'math.clamp', sig: '(Tensor t, number lo, number hi) -> Tensor',
  doc: '截断到 [lo,hi]', ref: 'NumPy clip', inv: 'clamp(clamp(x)) = clamp(x)',
  impl: (t, lo, hi) => { expectTensor(t, 't'); if (lo > hi) throw new ForgeError('E_VALUE', 'lo>hi'); return t.map(v => Math.min(Math.max(v, lo), hi)); },
});
defineOp({
  id: nid(), domain: D, name: 'math.where', sig: '(Tensor mask, Tensor a, Tensor b) -> Tensor',
  doc: '按掩码选择：mask≠0 取 a 否则取 b（NumPy 广播）', ref: 'NumPy where',
  inv: 'where(1,a,b)=a；where(0,a,b)=b',
  impl: (mask, a, b) => {
    expectTensor(mask, 'mask'); expectTensor(a, 'a'); expectTensor(b, 'b');
    const outShape = broadcastShapes(broadcastShapes(mask.shape, a.shape), b.shape);
    const zeros = Tensor.zeros(outShape);
    // 借广播把三方对齐到同一形状后逐元素选择
    const M = elementwise(mask, zeros, (x) => x);
    const A = elementwise(a, zeros, (x) => x);
    const B = elementwise(b, zeros, (x) => x);
    const out = new Float64Array(outShape.reduce((x, y) => x * y, 1));
    for (let i = 0; i < out.length; i++) out[i] = M.data[i] !== 0 ? A.data[i] : B.data[i];
    return new Tensor(out, outShape);
  },
});

// 文件底部自检：本域必须恰好 102 条
if (opsByDomain(D).length !== 102) {
  throw new Error(`域${D}(MATH) 指令数 = ${opsByDomain(D).length}，应为 102`);
}
