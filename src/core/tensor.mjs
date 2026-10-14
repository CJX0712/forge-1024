// 稠密张量 Tensor：Float64Array + shape，行主序（C order）。
// 语义子集对标 NumPy ndarray：广播规则、reshape 视图、permute 拷贝、轴归约。
// 参考：NumPy broadcasting 文档；Golub & Van Loan《Matrix Computations》§1.3（存储序）。
import { ForgeError } from './errors.mjs';

export function prod(shape) {
  let n = 1;
  for (const d of shape) {
    if (!Number.isInteger(d) || d < 0) throw new ForgeError('E_SHAPE', `非法维度 ${d}`);
    n *= d;
  }
  return n;
}

export function stridesOf(shape) {
  const st = new Array(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) { st[i] = acc; acc *= shape[i]; }
  return st;
}

export function sameShape(a, b) {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

// NumPy 广播规则：从尾部对齐；1 可扩展，缺失补 1；其余必须相等。
export function broadcastShapes(s1, s2) {
  const n = Math.max(s1.length, s2.length);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const d1 = s1[s1.length - 1 - i] ?? 1;
    const d2 = s2[s2.length - 1 - i] ?? 1;
    if (d1 !== d2 && d1 !== 1 && d2 !== 1) {
      throw new ForgeError('E_SHAPE', `形状不可广播: [${s1}] vs [${s2}]`);
    }
    out[n - 1 - i] = Math.max(d1, d2);
  }
  return out;
}

export class Tensor {
  constructor(data, shape) {
    this.data = data instanceof Float64Array ? data : Float64Array.from(data);
    this.shape = shape.slice();
    if (prod(this.shape) !== this.data.length) {
      throw new ForgeError('E_SHAPE', `shape [${this.shape}] 与数据长度 ${this.data.length} 不符`);
    }
  }

  // —— 静态构造器 ——
  static fromNested(arr) {
    const shape = [];
    let node = arr;
    while (Array.isArray(node)) {
      shape.push(node.length);
      if (node.length === 0) break;
      node = node[0];
    }
    const flat = [];
    const fill = (n, depth) => {
      if (depth === shape.length) { flat.push(Number(n)); return; }
      if (!Array.isArray(n) || n.length !== shape[depth]) {
        throw new ForgeError('E_SHAPE', '嵌套数组非矩形，无法推断 shape');
      }
      for (const c of n) fill(c, depth + 1);
    };
    fill(arr, 0);
    return new Tensor(flat, shape);
  }

  static zeros(shape) { return new Tensor(new Float64Array(prod(shape)), shape); }
  static ones(shape) { const t = Tensor.zeros(shape); t.data.fill(1); return t; }
  static full(shape, v) { const t = Tensor.zeros(shape); t.data.fill(v); return t; }
  static eye(n, m = n) {
    const t = Tensor.zeros([n, m]);
    for (let i = 0; i < Math.min(n, m); i++) t.data[i * m + i] = 1;
    return t;
  }
  static arange(stop, start = 0, step = 1) {
    const n = Math.max(0, Math.ceil((stop - start) / step));
    const d = new Float64Array(n);
    for (let i = 0; i < n; i++) d[i] = start + i * step;
    return new Tensor(d, [n]);
  }
  static linspace(a, b, n) {
    if (n < 1) throw new ForgeError('E_VALUE', 'linspace 需要 n>=1');
    const d = new Float64Array(n);
    if (n === 1) { d[0] = a; return new Tensor(d, [1]); }
    for (let i = 0; i < n; i++) d[i] = a + (b - a) * (i / (n - 1));
    return new Tensor(d, [n]);
  }
  static fromFlat(data, shape) { return new Tensor(Float64Array.from(data), shape); }

  get size() { return this.data.length; }
  get ndim() { return this.shape.length; }
  scalar() {
    if (this.size !== 1) throw new ForgeError('E_SHAPE', `scalar() 仅接受 size=1，当前 ${this.size}`);
    return this.data[0];
  }

  at(i) { return this.data[i]; }
  get2(i, j) {
    if (this.ndim !== 2) throw new ForgeError('E_SHAPE', 'get2 仅限 2D');
    return this.data[i * this.shape[1] + j];
  }
  set2(i, j, v) {
    if (this.ndim !== 2) throw new ForgeError('E_SHAPE', 'set2 仅限 2D');
    this.data[i * this.shape[1] + j] = v;
    return this;
  }
  row(i) {
    if (this.ndim !== 2) throw new ForgeError('E_SHAPE', 'row 仅限 2D');
    const n = this.shape[1];
    return this.data.subarray(i * n, (i + 1) * n);
  }
  col(j) {
    if (this.ndim !== 2) throw new ForgeError('E_SHAPE', 'col 仅限 2D');
    const [m, n] = this.shape;
    const out = new Float64Array(m);
    for (let i = 0; i < m; i++) out[i] = this.data[i * n + j];
    return out;
  }
  getAt(idx) {
    const st = stridesOf(this.shape);
    let p = 0;
    for (let d = 0; d < idx.length; d++) p += idx[d] * st[d];
    return this.data[p];
  }
  setAt(idx, v) {
    const st = stridesOf(this.shape);
    let p = 0;
    for (let d = 0; d < idx.length; d++) p += idx[d] * st[d];
    this.data[p] = v;
    return this;
  }

  // reshape 返回视图（共享底层数据，与 NumPy 一致）；-1 自动推断。
  reshape(...dims) {
    const d = (dims.length === 1 && Array.isArray(dims[0])) ? dims[0] : dims;
    const out = d.slice();
    const neg = out.indexOf(-1);
    if (neg >= 0) {
      let rest = 1;
      out.forEach((v, i) => { if (i !== neg) rest *= v; });
      out[neg] = this.size / rest;
    }
    if (prod(out) !== this.size) throw new ForgeError('E_SHAPE', `reshape [${this.shape}] -> [${out}] 元素数不符`);
    return new Tensor(this.data, out);
  }
  flatten() { return new Tensor(this.data, [this.size]); }

  T() {
    if (this.ndim !== 2) throw new ForgeError('E_SHAPE', 'T 仅限 2D');
    const [m, n] = this.shape;
    const out = new Float64Array(m * n);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[j * m + i] = this.data[i * n + j];
    return new Tensor(out, [n, m]);
  }
  permute(...axes) {
    if (axes.length !== this.ndim) throw new ForgeError('E_SHAPE', 'permute 轴数不符');
    const outShape = axes.map(a => this.shape[a]);
    const out = new Float64Array(this.size);
    const st = stridesOf(this.shape), so = stridesOf(outShape);
    const idx = new Array(this.ndim);
    for (let p = 0; p < this.size; p++) {
      let r = p;
      for (let d = this.ndim - 1; d >= 0; d--) { idx[d] = r % outShape[d]; r = (r - idx[d]) / outShape[d]; }
      let q = 0;
      for (let d = 0; d < this.ndim; d++) q += idx[d] * st[axes[d]];
      out[p] = this.data[q];
    }
    return new Tensor(out, outShape);
  }
  clone() { return new Tensor(this.data.slice(), this.shape); }
  map(fn) {
    const out = new Float64Array(this.size);
    const d = this.data;
    for (let i = 0; i < d.length; i++) out[i] = fn(d[i], i);
    return new Tensor(out, this.shape);
  }
  fill(v) { this.data.fill(v); return this; }

  toArray() {
    const build = (offset, dims) => {
      if (dims.length === 1) return Array.from(this.data.subarray(offset, offset + dims[0]));
      const [d, ...rest] = dims;
      const block = prod(rest);
      const out = [];
      for (let i = 0; i < d; i++) out.push(build(offset + i * block, rest));
      return out;
    };
    return build(0, this.shape);
  }
  toString() { return `Tensor(shape=[${this.shape}], data=[${Array.from(this.data.slice(0, 8)).map(v => v.toFixed(4))}${this.size > 8 ? ', ...' : ''}])`; }
}

export function isTensor(v) { return v instanceof Tensor; }

export function expectTensor(v, who = '参数') {
  if (!(v instanceof Tensor)) throw new ForgeError('E_TYPE', `${who}: 期望 Tensor，得到 ${v === null ? 'null' : typeof v}`);
  return v;
}

export function expectNumber(v, who = '参数') {
  if (typeof v !== 'number' || !Number.isFinite(v)) throw new ForgeError('E_TYPE', `${who}: 期望有限 number，得到 ${v}`);
  return v;
}

// 逐元素运算：支持 NumPy 广播；number 自动提升为标量。
export function elementwise(a, b, fn) {
  const ta = typeof a === 'number', tb = typeof b === 'number';
  if (ta && tb) return fn(a, b);
  if (ta) return b.map(v => fn(a, v));
  if (tb) return a.map(v => fn(v, b));
  const outShape = broadcastShapes(a.shape, b.shape);
  const n = outShape.length;
  const pa = new Array(n).fill(1), pb = new Array(n).fill(1);
  for (let i = 0; i < a.shape.length; i++) pa[n - a.shape.length + i] = a.shape[i];
  for (let i = 0; i < b.shape.length; i++) pb[n - b.shape.length + i] = b.shape[i];
  // stride 补齐到 n 维（前置补 0：补齐维尺寸为 1，索引恒 0）
  const padStrides = (shape) => {
    const st = stridesOf(shape);
    const out = new Array(n).fill(0);
    for (let i = 0; i < shape.length; i++) out[n - shape.length + i] = st[i];
    return out;
  };
  const sa = padStrides(a.shape), sb = padStrides(b.shape);
  const out = new Float64Array(prod(outShape));
  const idx = new Array(n);
  for (let p = 0; p < out.length; p++) {
    let r = p;
    for (let d = n - 1; d >= 0; d--) { idx[d] = r % outShape[d]; r = (r - idx[d]) / outShape[d]; }
    let ia = 0, ib = 0;
    for (let d = 0; d < n; d++) {
      if (pa[d] !== 1) ia += idx[d] * sa[d];
      if (pb[d] !== 1) ib += idx[d] * sb[d];
    }
    out[p] = fn(a.data[ia], b.data[ib]);
  }
  return new Tensor(out, outShape);
}

// 沿 axes 归约（axes=null 表示全部；支持负轴）。输出保持维数（keepDims 语义：被归约轴置 1）。
// 每个输出槽位独立累积：combine(slotAcc, val, flatIdx) -> newSlotAcc。
// 返回 {tensor, count}；count = 每个输出槽位合并的元素数（用于 mean 等）。
export function reduceTensor(t, axes, init, combine) {
  const all = axes == null;
  let ax;
  if (all) ax = t.shape.map((_, i) => i);
  else ax = (Array.isArray(axes) ? axes : [axes]).map(a => (a < 0 ? a + t.shape.length : a));
  for (const a of ax) if (a < 0 || a >= t.ndim) throw new ForgeError('E_SHAPE', `归约轴 ${a} 越界`);
  const reduced = new Set(ax);
  const outShape = t.shape.map((d, i) => (reduced.has(i) ? 1 : d));
  const so = stridesOf(outShape);
  const accs = new Array(prod(outShape)).fill(init);
  const idx = new Array(t.ndim);
  for (let p = 0; p < t.size; p++) {
    let r = p, q = 0;
    for (let d = t.ndim - 1; d >= 0; d--) { idx[d] = r % t.shape[d]; r = (r - idx[d]) / t.shape[d]; if (!reduced.has(d)) q += idx[d] * so[d]; }
    accs[q] = combine(accs[q], t.data[p], p);
  }
  let count = 1;
  for (let i = 0; i < t.ndim; i++) if (reduced.has(i)) count *= t.shape[i];
  return { tensor: new Tensor(Float64Array.from(accs), outShape), count };
}
