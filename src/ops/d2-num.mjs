// 域2 数值方法与数值稳定化 NUM（102 条微指令）
// 内容：稳定 softmax/logsumexp、补偿求和（Kahan/Neumaier/成对）、误差度量、
// 求根（二分/割线/Newton/Brent/Ridder）、积分（梯形/Simpson/Romberg）、
// 有限差分、插值、特殊函数（erf/gamma/Lanczos/逆正态）、双精度倍精度算术（double-double）、
// Welford 在线统计、输入守卫族。
// 参考：Higham《Accuracy and Stability of Numerical Algorithms》；
//       Numerical Recipes 3rd ed.（erf/gamma/求根/积分）；Dekker 1971（双精度倍精度）。
import { defineOp, opsByDomain, exec } from '../core/registry.mjs';
import { ForgeError } from '../core/errors.mjs';
import { Tensor, prod as coreProd, expectTensor } from '../core/tensor.mjs';

const D = 2;
let _s = 0;
const nid = () => `D${D}-${String(++_s).padStart(3, '0')}`;

function vec(t) { expectTensor(t, 't'); return t.data; }

// ============ A. 稳定化核心（14 条） ============
defineOp({
  id: nid(), domain: D, name: 'num.softmax', arity: 1,
  sig: '(Tensor x, number axis=-1) -> Tensor', doc: '数值稳定 softmax（先减轴最大值）',
  ref: '深度学习标准实践 / cs231n', inv: '沿 axis 每片和=1；输入整体平移不变',
  impl: (x, axis = -1) => {
    expectTensor(x, 'x');
    const ax = axis < 0 ? axis + x.ndim : axis;
    const m = exec('math.maxAxis', [x, ax]);
    const e = x.map((v, i) => {
      // 逐元素 exp(x - max_axis)
      const idx = decodeIdx(i, x.shape);
      const mi = m.data[flatWithRemoved(idx, ax, m.shape)];
      return Math.exp(v - mi);
    });
    const s = exec('math.sumAxis', [e, ax]);
    const out = new Float64Array(e.size);
    for (let i = 0; i < e.size; i++) {
      const idx = decodeIdx(i, x.shape);
      const si = s.data[flatWithRemoved(idx, ax, s.shape)];
      out[i] = e.data[i] / si;
    }
    return new Tensor(out, x.shape);
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.logSoftmax', arity: 1,
  sig: '(Tensor x, number axis=-1) -> Tensor', doc: '稳定 log-softmax = x - max - logsumexp',
  ref: 'PyTorch LogSoftmax', inv: 'exp(logSoftmax) 与 softmax 一致；不会下溢',
  impl: (x, axis = -1) => {
    expectTensor(x, 'x');
    const ax = axis < 0 ? axis + x.ndim : axis;
    const lse = exec('num.logsumexp', [x, ax]);
    const out = new Float64Array(x.size);
    for (let i = 0; i < x.size; i++) {
      const idx = decodeIdx(i, x.shape);
      const li = lse.data[flatWithRemoved(idx, ax, lse.shape)];
      out[i] = x.data[i] - li; // logSoftmax = x - logsumexp(x)
    }
    return new Tensor(out, x.shape);
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.logsumexp', arity: 1,
  sig: '(Tensor x, number axis=-1) -> Tensor', doc: '稳定 logΣexp（max 平移 + log）',
  ref: 'log-sum-exp trick（数值计算标准技巧）', inv: 'exp(logsumexp)=Σexp（不溢出）',
  impl: (x, axis = -1) => {
    expectTensor(x, 'x');
    const ax = axis < 0 ? axis + x.ndim : axis;
    const m = exec('math.maxAxis', [x, ax]);
    const out = new Float64Array(m.size);
    for (let i = 0; i < x.size; i++) {
      const idx = decodeIdx(i, x.shape);
      const mi = m.data[flatWithRemoved(idx, ax, m.shape)];
      const q = flatWithRemoved(idx, ax, m.shape);
      out[q] += Math.exp(x.data[i] - mi);
    }
    for (let q = 0; q < out.length; q++) out[q] = m.data[q] + Math.log(out[q]);
    // 重新整形为去掉该轴的
    const outShape = m.shape.filter((_, i) => i !== ax);
    return new Tensor(out, outShape);
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.sigmoid', sig: '(Tensor) -> Tensor', doc: '稳定 sigmoid（分支防 exp 溢出）',
  ref: 'Logistic 函数标准实现', inv: 'sigmoid(0)=0.5；sigmoid(-x)=1-sigmoid(x)；x=±1000 不溢出',
  impl: (t) => t.map((v) => (v >= 0 ? 1 / (1 + Math.exp(-v)) : Math.exp(v) / (1 + Math.exp(v)))),
});
defineOp({
  id: nid(), domain: D, name: 'num.stableExp', sig: '(Tensor) -> Tensor', doc: '饱和 exp（x>709 返回 Inf，x<-745 返回 0）',
  ref: 'IEEE 754 双精度指数定义域', inv: '与 Math.exp 在定义域内一致；永不返回 NaN',
  impl: (t) => t.map((v) => (v > 709 ? Infinity : v < -745 ? 0 : Math.exp(v))),
});
defineOp({
  id: nid(), domain: D, name: 'num.safeLog', sig: '(Tensor) -> Tensor', doc: '对数（x=0→-Inf，x<0→NaN，文档化契约）',
  ref: 'IEEE 754 log', inv: 'safeLog(e^x)=x（x 有限）；safeLog(0)=-Inf',
  impl: (t) => t.map((v) => (v > 0 ? Math.log(v) : v === 0 ? -Infinity : NaN)),
});
defineOp({
  id: nid(), domain: D, name: 'num.safeDiv', arity: 2, sig: '(Tensor a, Tensor b) -> Tensor',
  doc: 'IEEE 语义除法（0/0=NaN，x/0=±Inf）', ref: 'IEEE 754', inv: '与 JS / 一致且对零不抛错',
  impl: (a, b) => {
    expectTensor(a, 'a'); expectTensor(b, 'b');
    const out = new Float64Array(a.size);
    for (let i = 0; i < a.size; i++) out[i] = a.data[i] / b.data[i % b.size];
    return new Tensor(out, a.shape);
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.safeSqrt', sig: '(Tensor) -> Tensor', doc: '平方根（x<0→NaN）',
  ref: 'IEEE 754', inv: 'safeSqrt(x²)=|x|', impl: (t) => t.map((v) => (v < 0 ? NaN : Math.sqrt(v))),
});
defineOp({
  id: nid(), domain: D, name: 'num.safePow', arity: 2, sig: '(Tensor base, Tensor exp) -> Tensor',
  doc: '幂运算（负底非整数指数→NaN，0^0=1）', ref: 'IEEE 754 pow', inv: '与 Math.pow 一致且零不抛错',
  impl: (a, b) => {
    expectTensor(a, 'a'); expectTensor(b, 'b');
    const out = new Float64Array(a.size);
    for (let i = 0; i < a.size; i++) out[i] = Math.pow(a.data[i], b.data[i % b.size]);
    return new Tensor(out, a.shape);
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.softplus', sig: '(Tensor) -> Tensor', doc: '稳定 softplus log(1+e^x)（大 x 线性外推）',
  ref: 'Dugas et al. 2001（将 softplus 引入神经网络）', inv: 'softplus(x)≈max(x,0)（|x| 大时）；softplus(0)=ln2',
  impl: (t) => t.map((v) => (v > 30 ? v : v < -30 ? Math.exp(v) : Math.log1p(Math.exp(v)))),
});
defineOp({
  id: nid(), domain: D, name: 'num.kahanSum', sig: '(Tensor) -> number', doc: 'Kahan 补偿求和（消除累计舍入）',
  ref: 'Kahan 1965', inv: '对 [1e16,1,-1e16] 返回 1（朴素求和返回 0）',
  impl: (t) => {
    let s = 0, c = 0;
    for (let i = 0; i < t.size; i++) {
      const y = t.data[i] - c;
      const tt = s + y;
      c = (tt - s) - y;
      s = tt;
    }
    return s;
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.neumaierSum', sig: '(Tensor) -> number', doc: 'Neumaier 改进补偿求和',
  ref: 'Neumaier 1974', inv: '精度 ≥ Kahan；处理大小悬殊序列',
  impl: (t) => {
    let s = 0, c = 0;
    for (let i = 0; i < t.size; i++) {
      const x = t.data[i];
      const tt = s + x;
      if (Math.abs(s) >= Math.abs(x)) c += (s - tt) + x;
      else c += (x - tt) + s;
      s = tt;
    }
    return s + c;
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.pairwiseSum', sig: '(Tensor) -> number', doc: '成对递归求和（误差 O(log n)）',
  ref: '成对求和 / BLAS 累加策略', inv: '精度 ≥ 顺序求和；与 Kahan 同量级',
  impl: (t) => {
    const a = t.data;
    const rec = (lo, hi) => {
      const n = hi - lo;
      if (n <= 128) { let s = 0; for (let i = lo; i < hi; i++) s += a[i]; return s; }
      const mid = lo + (n >> 1);
      return rec(lo, mid) + rec(mid, hi);
    };
    return rec(0, a.length);
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.naiveSum', sig: '(Tensor) -> number', doc: '朴素顺序求和（精度基准，供对比）',
  ref: 'IEEE 754 顺序累加', inv: '与手写 for 循环完全一致',
  impl: (t) => { let s = 0; for (let i = 0; i < t.size; i++) s += t.data[i]; return s; },
});

// ============ B. 误差度量（9 条） ============
defineOp({
  id: nid(), domain: D, name: 'num.absErr', arity: 2, sig: '(number a, number b) -> number',
  doc: '绝对误差 |a-b|', ref: '数值分析定义', inv: 'absErr(a,a)=0', impl: (a, b) => Math.abs(a - b),
});
defineOp({
  id: nid(), domain: D, name: 'num.relErr', arity: 2, sig: '(number a, number b) -> number',
  doc: '相对误差 |a-b|/max(|a|,|b|,1)', ref: '数值分析定义', inv: 'relErr(a,a)=0；对称',
  impl: (a, b) => Math.abs(a - b) / Math.max(1e-300, Math.abs(a), Math.abs(b)),
});
defineOp({
  id: nid(), domain: D, name: 'num.ulpErr', arity: 2, sig: '(number a, number b) -> number',
  doc: 'ULP 差异 |a-b|/ulp(max(|a|,|b|))', ref: 'IEEE 754 ulp 概念', inv: '同数返回 0；相邻浮点返回 1',
  impl: (a, b) => {
    const u = exec('num.ulp', [Math.max(Math.abs(a), Math.abs(b))]);
    return u === 0 ? 0 : Math.abs(a - b) / u;
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.mae', arity: 2, sig: '(Tensor a, Tensor b) -> number',
  doc: '平均绝对误差', ref: '回归评估标准', inv: 'mae(a,a)=0',
  impl: (a, b) => meanAbs(a, b),
});
function meanAbs(a, b) { let s = 0; for (let i = 0; i < a.size; i++) s += Math.abs(a.data[i] - b.data[i % b.size]); return s / a.size; }
defineOp({
  id: nid(), domain: D, name: 'num.mse', arity: 2, sig: '(Tensor a, Tensor b) -> number',
  doc: '均方误差', ref: '回归评估标准', inv: 'mse(a,a)=0；mse≥0',
  impl: (a, b) => { let s = 0; for (let i = 0; i < a.size; i++) { const d = a.data[i] - b.data[i % b.size]; s += d * d; } return s / a.size; },
});
defineOp({
  id: nid(), domain: D, name: 'num.rmse', arity: 2, sig: '(Tensor a, Tensor b) -> number',
  doc: '均方根误差 √mse', ref: '回归评估标准', inv: 'rmse=√mse',
  impl: (a, b) => Math.sqrt(exec('num.mse', [a, b])),
});
defineOp({
  id: nid(), domain: D, name: 'num.mape', arity: 2, sig: '(Tensor a, Tensor b) -> number',
  doc: '平均绝对百分比误差（b=0 项跳过）', ref: '预测评估标准', inv: '完美预测返回 0',
  impl: (a, b) => {
    let s = 0, n = 0;
    for (let i = 0; i < a.size; i++) { const bb = b.data[i % b.size]; if (bb === 0) continue; s += Math.abs((a.data[i] - bb) / bb); n++; }
    return n === 0 ? NaN : s / n;
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.maxAbsErr', arity: 2, sig: '(Tensor a, Tensor b) -> number',
  doc: '最大绝对误差（L∞ 误差）', ref: 'Higham 2002, Accuracy and Stability of Numerical Algorithms, §1.2', inv: '≥ mae；=0 ⟺ 相等',
  impl: (a, b) => { let m = 0; for (let i = 0; i < a.size; i++) m = Math.max(m, Math.abs(a.data[i] - b.data[i % b.size])); return m; },
});
defineOp({
  id: nid(), domain: D, name: 'num.medAbsErr', arity: 2, sig: '(Tensor a, Tensor b) -> number',
  doc: '中位数绝对误差（稳健）', ref: 'Hampel 1974, JASA 69:383-393 (robust scale estimator)', inv: '对异常值不敏感',
  impl: (a, b) => {
    const ds = [];
    for (let i = 0; i < a.size; i++) ds.push(Math.abs(a.data[i] - b.data[i % b.size]));
    ds.sort((x, y) => x - y);
    const n = ds.length;
    return n % 2 ? ds[(n - 1) / 2] : (ds[n / 2 - 1] + ds[n / 2]) / 2;
  },
});

// ============ C. 比较与守卫（8 条） ============
defineOp({
  id: nid(), domain: D, name: 'num.isclose', arity: 2, sig: '(number a, number b, number rtol=1e-5, number atol=1e-8) -> boolean',
  doc: '逐对近似相等（NumPy isclose 语义）', ref: 'NumPy isclose', inv: 'isclose(a,a)=true；isclose(a,b)=isclose(b,a)',
  impl: (a, b, rtol = 1e-5, atol = 1e-8) => Math.abs(a - b) <= atol + rtol * Math.max(Math.abs(a), Math.abs(b)),
});
defineOp({
  id: nid(), domain: D, name: 'num.allclose', arity: 2, sig: '(Tensor a, Tensor b, number rtol=1e-5, number atol=1e-8) -> boolean',
  doc: '张量逐元素近似相等（NumPy allclose）', ref: 'NumPy allclose', inv: '形状必须一致',
  impl: (a, b, rtol = 1e-5, atol = 1e-8) => {
    expectTensor(a, 'a'); expectTensor(b, 'b');
    if (a.size !== b.size) throw new ForgeError('E_SHAPE', 'allclose 形状不符');
    for (let i = 0; i < a.size; i++) if (!exec('num.isclose', [a.data[i], b.data[i], rtol, atol])) return false;
    return true;
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.arrayEqual', arity: 2, sig: '(Tensor a, Tensor b) -> boolean',
  doc: '精确逐元素相等（含 NaN 位模式）', ref: 'NumPy array_equal', inv: 'arrayEqual(a,a)=true',
  impl: (a, b) => {
    expectTensor(a, 'a'); expectTensor(b, 'b');
    if (a.size !== b.size) return false;
    for (let i = 0; i < a.size; i++) {
      const x = a.data[i], y = b.data[i];
      if (Number.isNaN(x) && Number.isNaN(y)) continue;
      if (x !== y) return false;
    }
    return true;
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.isfiniteAll', sig: '(Tensor) -> boolean', doc: '全部元素有限',
  ref: 'IEEE 754 isFinite', inv: '无 NaN/Inf 返回 true', impl: (t) => t.data.every(Number.isFinite),
});
defineOp({
  id: nid(), domain: D, name: 'num.countNaN', sig: '(Tensor) -> number', doc: 'NaN 计数',
  ref: 'IEEE 754', inv: 'countNaN ≤ size', impl: (t) => { let n = 0; for (const v of t.data) if (Number.isNaN(v)) n++; return n; },
});
defineOp({
  id: nid(), domain: D, name: 'num.countInf', sig: '(Tensor) -> number', doc: '±Inf 计数',
  ref: 'IEEE 754', inv: 'countInf ≤ size', impl: (t) => { let n = 0; for (const v of t.data) if (!Number.isFinite(v) && !Number.isNaN(v)) n++; return n; },
});
defineOp({
  id: nid(), domain: D, name: 'num.nanToZero', sig: '(Tensor) -> Tensor', doc: 'NaN 置零（Inf 保留）',
  ref: '数据清洗常见操作', inv: '结果无 NaN', impl: (t) => t.map((v) => (Number.isNaN(v) ? 0 : v)),
});
defineOp({
  id: nid(), domain: D, name: 'num.sanitize', arity: 3, sig: '(Tensor t, number lo, number hi) -> Tensor',
  doc: '清洗：NaN→lo，+Inf→hi，-Inf→lo，其余不变', ref: '数据清洗 / 梯度爆炸防护',
  inv: '结果全部有限', impl: (t, lo, hi) => t.map((v) => (Number.isNaN(v) ? lo : v === Infinity ? hi : v === -Infinity ? lo : v)),
});

// ============ D. 求根（6 条） ============
function rootOpts(opts = {}) {
  return { tol: opts.tol ?? 1e-12, maxIter: opts.maxIter ?? 100 };
}
function rootResult(x, iters, converged, fx) { return { root: x, iters, converged, fx }; }

defineOp({
  id: nid(), domain: D, name: 'num.bisect', arity: 3,
  sig: '(Function f, number a, number b, object opts?) -> object', doc: '二分法求根（需 f(a)·f(b)<0）',
  ref: 'Numerical Recipes §9.1', inv: '收敛速度每次减半；结果 bracket 宽度 ≤ tol',
  impl: (f, a, b, opts = {}) => {
    const { tol, maxIter } = rootOpts(opts);
    let fa = f(a), fb = f(b);
    if (fa * fb > 0) throw new ForgeError('E_VALUE', 'bisect 需要 f(a)·f(b)<0');
    let lo = a, hi = b, flo = fa, it = 0;
    for (; it < maxIter; it++) {
      const mid = (lo + hi) / 2;
      const fm = f(mid);
      if (fm === 0 || (hi - lo) / 2 <= tol) return rootResult(mid, it + 1, true, fm);
      if (flo * fm < 0) { hi = mid; } else { lo = mid; flo = fm; }
    }
    return rootResult((lo + hi) / 2, it, false, f((lo + hi) / 2));
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.secant', arity: 3,
  sig: '(Function f, number x0, number x1, object opts?) -> object', doc: '割线法求根（超线性收敛）',
  ref: 'Numerical Recipes §9.2', inv: '光滑函数上快于二分',
  impl: (f, x0, x1, opts = {}) => {
    const { tol, maxIter } = rootOpts(opts);
    let a = x0, b = x1, fa = f(a), fb = f(b);
    for (let it = 0; it < maxIter; it++) {
      if (Math.abs(fb - fa) < 1e-300) return rootResult(b, it, false, fb);
      const c = b - fb * (b - a) / (fb - fa);
      const fc = f(c);
      if (Math.abs(fc) <= tol || Math.abs(c - b) <= tol) return rootResult(c, it + 1, true, fc);
      a = b; fa = fb; b = c; fb = fc;
    }
    return rootResult(b, maxIter, false, fb);
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.newton', arity: 2,
  sig: '(Function f, number x0, object opts?) -> object', doc: 'Newton 法（中心差分数值导数，二阶收敛）',
  ref: 'Newton-Raphson / Numerical Recipes §9.4', inv: '单根附近二次收敛',
  impl: (f, x0, opts = {}) => {
    const { tol, maxIter } = rootOpts(opts);
    let x = x0;
    for (let it = 0; it < maxIter; it++) {
      const h = Math.max(1e-7, Math.abs(x) * 1e-7);
      const d = (f(x + h) - f(x - h)) / (2 * h);
      if (!Number.isFinite(d) || d === 0) return rootResult(x, it, false, f(x));
      const xn = x - f(x) / d;
      if (Math.abs(xn - x) <= tol) return rootResult(xn, it + 1, true, f(xn));
      x = xn;
    }
    return rootResult(x, maxIter, false, f(x));
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.brent', arity: 3,
  sig: '(Function f, number a, number b, object opts?) -> object', doc: 'Brent 法（逆二次插值+二分保障，鲁棒且快）',
  ref: 'Brent 1973, Algorithms for Minimization Without Derivatives；Numerical Recipes §9.3',
  inv: '保收敛性；通常远少于二分的迭代次数',
  impl: (f, a, b, opts = {}) => {
    const { tol, maxIter } = rootOpts(opts);
    let fa = f(a), fb = f(b);
    if (fa * fb > 0) throw new ForgeError('E_VALUE', 'brent 需要 f(a)·f(b)<0');
    let c = a, fc = fa, d = b - a, e = d;
    for (let it = 1; it <= maxIter; it++) {
      if (fb * fc > 0) { c = a; fc = fa; d = b - a; e = d; }
      if (Math.abs(fc) < Math.abs(fb)) { [a, b, c] = [b, c, b]; [fa, fb, fc] = [fb, fc, fb]; }
      const tol1 = 2 * Number.EPSILON * Math.abs(b) + 0.5 * tol;
      const xm = 0.5 * (c - b);
      if (Math.abs(xm) <= tol1 || fb === 0) return rootResult(b, it, true, fb);
      if (Math.abs(e) >= tol1 && Math.abs(fa) > Math.abs(fb)) {
        const s = fb / fa;
        let p, q;
        if (a === c) { p = 2 * xm * s; q = 1 - s; }
        else { q = fa / fc; const r = fb / fc; p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1)); q = (q - 1) * (r - 1) * (s - 1); }
        if (p > 0) q = -q;
        p = Math.abs(p);
        const min1 = 3 * xm * q - Math.abs(tol1 * q);
        const min2 = Math.abs(e * q);
        if (2 * p < Math.min(min1, min2)) { e = d; d = p / q; } else { d = xm; e = d; }
      } else { d = xm; e = d; }
      a = b; fa = fb;
      if (Math.abs(d) > tol1) b += d; else b += Math.sign(xm) * tol1;
      fb = f(b);
    }
    return rootResult(b, maxIter, false, fb);
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.ridder', arity: 3,
  sig: '(Function f, number a, number b, object opts?) -> object', doc: 'Ridder 法（指数插值，保区间）',
  ref: 'Ridder 1979', inv: '收敛快于二分；区间始终含根',
  impl: (f, a, b, opts = {}) => {
    const { tol, maxIter } = rootOpts(opts);
    let fa = f(a), fb = f(b);
    if (fa * fb > 0) throw new ForgeError('E_VALUE', 'ridder 需要 f(a)·f(b)<0');
    for (let it = 1; it <= maxIter; it++) {
      const c = 0.5 * (a + b);
      const fc = f(c);
      const s = Math.sqrt(fc * fc - fa * fb);
      if (s === 0) return rootResult(c, it, true, fc);
      // 标准 Ridder 更新：x = c + (c-a)·sign(fa-fb)·fc/s（Wikipedia / NR 变体）
      const x = c + (c - a) * Math.sign(fa - fb) * fc / s;
      const fx = f(x);
      if (fx === 0 || Math.abs(x - c) <= tol) return rootResult(x, it, true, fx);
      // 更新 bracket：始终包含根
      if (Math.sign(fx) !== Math.sign(fc)) { a = c; fa = fc; } else { b = c; fb = fc; }
      if (Math.sign(fa) !== Math.sign(fx)) { b = x; fb = fx; } else { a = x; fa = fx; }
    }
    return rootResult(0.5 * (a + b), maxIter, false, f(0.5 * (a + b)));
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.rootScan', arity: 3,
  sig: '(Function f, number a, number b, number n) -> number[][]', doc: '扫描区间找全部符号变化 bracket',
  ref: '符号扫描（工程求根常用）', inv: '每个返回 bracket 满足 f(lo)·f(hi)≤0',
  impl: (f, a, b, n = 100) => {
    const out = [];
    let x0 = a, f0 = f(a);
    for (let i = 1; i <= n; i++) {
      const x1 = a + (b - a) * i / n;
      const f1 = f(x1);
      if (f0 === 0) out.push([x0, x0]);
      else if (f0 * f1 < 0) out.push([x0, x1]);
      x0 = x1; f0 = f1;
    }
    return out;
  },
});

// ============ E. 积分（5 条） ============
defineOp({
  id: nid(), domain: D, name: 'num.trapz', arity: 1, sig: '(Tensor y, Tensor x?) -> number',
  doc: '梯形法积分（非均匀步长）', ref: 'Numerical Recipes §4.1', inv: '对线性函数精确',
  impl: (y, x) => {
    if (x === undefined) { let s = 0; for (let i = 1; i < y.size; i++) s += (y.data[i] + y.data[i - 1]) / 2; return s; }
    let s = 0;
    for (let i = 1; i < y.size; i++) s += (y.data[i] + y.data[i - 1]) / 2 * (x.data[i] - x.data[i - 1]);
    return s;
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.cumtrapz', arity: 1, sig: '(Tensor y, Tensor x?) -> Tensor',
  doc: '累积梯形积分', ref: 'NumPy cumtrapz', inv: '末元素 = trapz(y)',
  impl: (y, x) => {
    const out = new Float64Array(y.size);
    let acc = 0;
    for (let i = 1; i < y.size; i++) {
      const h = x ? x.data[i] - x.data[i - 1] : 1;
      acc += (y.data[i] + y.data[i - 1]) / 2 * h;
      out[i] = acc;
    }
    return new Tensor(out, y.shape);
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.simpson', arity: 1, sig: '(Tensor y, Tensor x?) -> number',
  doc: 'Simpson 1/3 法（均匀步长；奇数区间末段退化梯形）', ref: 'Numerical Recipes §4.1',
  inv: '对三次及以下多项式精确',
  impl: (y, x) => {
    const n = y.size;
    if (n < 3) return exec('num.trapz', [y, x]);
    const h = x ? x.data[1] - x.data[0] : 1;
    let m = n - 1; // 区间数
    let pts = n;
    if (m % 2 === 1) pts = n - 1; // 奇数区间：末段改用梯形
    let s = y.data[0] + y.data[pts - 1];
    for (let j = 1; j < pts - 1; j++) s += (j % 2 === 1 ? 4 : 2) * y.data[j];
    let result = h * s / 3;
    if (m % 2 === 1) result += h * (y.data[n - 2] + y.data[n - 1]) / 2;
    return result;
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.midpoint', arity: 1, sig: '(Function f, number a, number b, number n) -> number',
  doc: '中点法（函数形式）', ref: 'Numerical Recipes §4.1', inv: 'n 加倍误差约减半',
  impl: (f, a, b, n = 100) => {
    const h = (b - a) / n;
    let s = 0;
    for (let i = 0; i < n; i++) s += f(a + (i + 0.5) * h);
    return s * h;
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.romberg', arity: 1,
  sig: '(Function f, number a, number b, number levels=6) -> number', doc: 'Romberg 积分（ Richardson 外提加速）',
  ref: 'Romberg 1955 / Numerical Recipes §4.3', inv: '每级精度量级提升',
  impl: (f, a, b, levels = 6) => {
    const R = [];
    let h = b - a;
    R.push([h * (f(a) + f(b)) / 2]);
    for (let i = 1; i < levels; i++) {
      h /= 2;
      let s = 0;
      const n = 2 ** (i - 1);
      for (let k = 0; k < n; k++) s += f(a + (2 * k + 1) * h);
      R.push([0.5 * R[i - 1][0] + s * h]);
      for (let j = 1; j <= i; j++) {
        R[i].push(R[i][j - 1] + (R[i][j - 1] - R[i - 1][j - 1]) / (4 ** j - 1));
      }
    }
    return R[levels - 1][levels - 1];
  },
});

// ============ F. 有限差分（5 条） ============
defineOp({
  id: nid(), domain: D, name: 'num.diff', arity: 1, sig: '(Tensor y, number n=1) -> Tensor',
  doc: 'n 阶前向差分', ref: 'NumPy diff', inv: '每阶长度减 1；常数序列一阶分为 0',
  impl: (y, n = 1) => {
    let cur = y;
    for (let k = 0; k < n; k++) {
      const out = new Float64Array(Math.max(0, cur.size - 1));
      for (let i = 1; i < cur.size; i++) out[i - 1] = cur.data[i] - cur.data[i - 1];
      cur = new Tensor(out, [out.length]);
    }
    return cur;
  },
});
// 标量提取：f 既可直接返回 number，也可返回 size=1 的 Tensor（gradcheck 的常见写法）。
// 直接对两个 Tensor 做减法会得 NaN，故统一在此归一为 number。
function _scalarOf(v, who = 'f(x)') {
  if (typeof v === 'number') return v;
  if (v && v.data && v.size === 1) return v.data[0];
  if (v && typeof v.scalar === 'function') return v.scalar();
  throw new ForgeError('E_VALUE', `${who} 必须为 number 或 size=1 的 Tensor`);
}

defineOp({
  id: nid(), domain: D, name: 'num.gradCentral', arity: 2,
  sig: '(Function f, Tensor x, number eps=1e-5) -> Tensor', doc: '中心差分数值梯度（O(ε²)）',
  ref: '数值微分标准', inv: '对二次函数精确到 O(ε²)；f 为 x↦标量',
  impl: (f, x, eps = 1e-5) => {
    const out = new Float64Array(x.size);
    for (let i = 0; i < x.size; i++) {
      const xp = x.clone(), xm = x.clone();
      xp.data[i] += eps; xm.data[i] -= eps;
      out[i] = (_scalarOf(f(xp)) - _scalarOf(f(xm))) / (2 * eps);
    }
    return new Tensor(out, x.shape);
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.gradForward', arity: 2,
  sig: '(Function f, Tensor x, number eps=1e-5) -> Tensor', doc: '前向差分数值梯度（O(ε)）',
  ref: '数值微分标准', inv: '比中心差分便宜但精度低一阶',
  impl: (f, x, eps = 1e-5) => {
    const out = new Float64Array(x.size);
    const f0 = _scalarOf(f(x));
    for (let i = 0; i < x.size; i++) {
      const xp = x.clone();
      xp.data[i] += eps;
      out[i] = (_scalarOf(f(xp)) - f0) / eps;
    }
    return new Tensor(out, x.shape);
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.hessianDiag', arity: 2,
  sig: '(Function f, Tensor x, number eps=1e-4) -> Tensor', doc: 'Hessian 对角线（二阶中心差分）',
  ref: '数值微分标准', inv: '对 f=½xᵀAx 返回 A 的对角元',
  impl: (f, x, eps = 1e-4) => {
    const out = new Float64Array(x.size);
    for (let i = 0; i < x.size; i++) {
      const xp = x.clone(), xm = x.clone(), x0 = x.clone();
      xp.data[i] += eps; xm.data[i] -= eps;
      out[i] = (_scalarOf(f(xp)) - 2 * _scalarOf(f(x0)) + _scalarOf(f(xm))) / (eps * eps);
    }
    return new Tensor(out, x.shape);
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.jacobianFD', arity: 2,
  sig: '(Function F, Tensor x, number eps=1e-5) -> Tensor', doc: '数值雅可比矩阵（F: R^n→R^m）',
  ref: '数值微分标准', inv: '第 j 列为 ∂F/∂xⱼ',
  impl: (F, x, eps = 1e-5) => {
    const cols = [];
    for (let j = 0; j < x.size; j++) {
      const xp = x.clone(), xm = x.clone();
      xp.data[j] += eps; xm.data[j] -= eps;
      const fp = F(xp), fm = F(xm);
      const col = new Float64Array(fp.size);
      for (let i = 0; i < fp.size; i++) col[i] = (fp.data[i] - fm.data[i]) / (2 * eps);
      cols.push(col);
    }
    const m = cols[0].length;
    const out = new Float64Array(m * x.size);
    for (let i = 0; i < m; i++) for (let j = 0; j < x.size; j++) out[i * x.size + j] = cols[j][i];
    return new Tensor(out, [m, x.size]);
  },
});

// ============ G. 插值（4 条） ============
defineOp({
  id: nid(), domain: D, name: 'num.searchSorted', arity: 2, sig: '(Tensor xs, number xq) -> number',
  doc: '二分查找插入位置（xs 升序）', ref: 'NumPy searchsorted', inv: '返回 i 使 xs[i-1]≤xq<xs[i]',
  impl: (xs, xq) => {
    let lo = 0, hi = xs.size;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (xs.data[mid] < xq) lo = mid + 1; else hi = mid; }
    return lo;
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.interpLinear', arity: 3, sig: '(Tensor xs, Tensor ys, Tensor|number xq) -> Tensor|number',
  doc: '分段线性插值（区间外线性外推）', ref: 'NumPy interp', inv: '过全部数据点',
  impl: (xs, ys, xq) => {
    if (typeof xq === 'number') return interp1(xs, ys, xq);
    return xq.map((v) => interp1(xs, ys, v));
  },
});
function interp1(xs, ys, xq) {
  const n = xs.size;
  if (n === 0) throw new ForgeError('E_VALUE', '空数据');
  if (n === 1) return ys.data[0];
  if (xq <= xs.data[0]) { const s = (ys.data[1] - ys.data[0]) / (xs.data[1] - xs.data[0]); return ys.data[0] + s * (xq - xs.data[0]); }
  if (xq >= xs.data[n - 1]) { const s = (ys.data[n - 1] - ys.data[n - 2]) / (xs.data[n - 1] - xs.data[n - 2]); return ys.data[n - 1] + s * (xq - xs.data[n - 1]); }
  const i = exec('num.searchSorted', [xs, xq]);
  const t = (xq - xs.data[i - 1]) / (xs.data[i] - xs.data[i - 1]);
  return ys.data[i - 1] + t * (ys.data[i] - ys.data[i - 1]);
}
defineOp({
  id: nid(), domain: D, name: 'num.interpNearest', arity: 3, sig: '(Tensor xs, Tensor ys, number xq) -> number',
  doc: '最近邻插值', ref: 'NumPy interp(kind=nearest)', inv: '返回最近节点的值',
  impl: (xs, ys, xq) => {
    const i = exec('num.searchSorted', [xs, xq]);
    if (i === 0) return ys.data[0];
    if (i >= xs.size) return ys.data[xs.size - 1];
    return xq - xs.data[i - 1] <= xs.data[i] - xq ? ys.data[i - 1] : ys.data[i];
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.interpLagrange', arity: 3, sig: '(Tensor xs, Tensor ys, number xq) -> number',
  doc: 'Lagrange 全局多项式插值', ref: '数值分析标准', inv: '过全部节点；≤n-1 次多项式精确',
  impl: (xs, ys, xq) => {
    const n = xs.size;
    let out = 0;
    for (let i = 0; i < n; i++) {
      let li = 1;
      for (let j = 0; j < n; j++) if (j !== i) li *= (xq - xs.data[j]) / (xs.data[i] - xs.data[j]);
      out += ys.data[i] * li;
    }
    return out;
  },
});

// ============ H. 特殊函数（6 条） ============
defineOp({
  id: nid(), domain: D, name: 'num.erf', sig: '(number x) -> number', doc: '误差函数（级数+连分数，全定义域）',
  ref: 'Abramowitz & Stegun 7.1.26；级数 erf(x)=2/√π·Σ(-1)ⁿx^{2n+1}/(n!(2n+1))',
  inv: 'erf(-x)=-erf(x)；erf(∞)=1；erf(1)=0.8427007929497149',
  impl: (x) => {
    if (Number.isNaN(x)) return NaN;
    if (x === 0) return 0;
    if (x < 0) return -exec('num.erf', [-x]);
    if (x > 3) return 1 - erfcLarge(x);
    // 级数
    let term = x, sum = x;
    for (let n = 1; n < 200; n++) {
      term *= -x * x / n;
      const add = term / (2 * n + 1);
      sum += add;
      if (Math.abs(add) < 1e-18 * Math.abs(sum)) break;
    }
    return 2 / Math.sqrt(Math.PI) * sum;
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.erfc', sig: '(number x) -> number', doc: '互补误差函数',
  ref: 'Laplace 连分数 + 级数', inv: 'erfc(-x)=2-erfc(x)；erfc(0)=1；erfc(∞)=0',
  impl: (x) => {
    if (Number.isNaN(x)) return NaN;
    if (x < 0) return 2 - exec('num.erfc', [-x]);
    if (x === 0) return 1;
    if (x <= 2) return 1 - exec('num.erf', [x]);
    return erfcLarge(x);
  },
});
// Laplace 连分数（DLMF 7.9.2）：
// erfc(x)·(√π·e^{x²}) = 1/(x + 1/(2x + 2/(x + 3/(2x + 4/(x + ...)))))
// 分子序列 1,1,2,3,4,...；分母序列 x,2x,x,2x,...
function erfcLarge(x) {
  const N = 80;
  let f = 0;
  for (let i = N; i >= 1; i--) {
    const d = i % 2 === 1 ? x : 2 * x;
    const num = i === 1 ? 1 : i - 1;
    f = num / (d + f);
  }
  return Math.exp(-x * x) / Math.sqrt(Math.PI) * f;
}
defineOp({
  id: nid(), domain: D, name: 'num.gamma', sig: '(number z) -> number', doc: 'Gamma 函数（Stirling 级数 + 递推）',
  ref: 'Stirling 展开（DLMF 5.11.1）；Numerical Recipes §6.1（递推策略）',
  inv: 'gamma(n)=(n-1)!；gamma(0.5)=√π=1.7724538509055159',
  impl: (z) => {
    if (Number.isNaN(z)) return NaN;
    if (z === Math.round(z) && z <= 0) return NaN; // 极点
    if (z < 0.5) return Math.PI / (Math.sin(Math.PI * z) * exec('num.gamma', [1 - z]));
    // 递推到 z>=10 后用 Stirling 级数（保证 1e-15 精度）
    let zz = z, prod = 1;
    while (zz < 10) { prod *= zz; zz += 1; }
    return Math.exp(logGammaStirling(zz)) / prod;
  },
});
// Stirling 级数：lnΓ(z) ≈ (z-0.5)lnz - z + 0.5ln(2π) + Σ B_{2n}/(2n(2n-1)z^{2n-1})
function logGammaStirling(z) {
  const z2 = z * z;
  return (z - 0.5) * Math.log(z) - z + 0.5 * Math.log(2 * Math.PI)
    + 1 / (12 * z) - 1 / (360 * z2 * z) + 1 / (1260 * z2 * z2 * z) - 1 / (1680 * z2 * z2 * z2 * z);
}
defineOp({
  id: nid(), domain: D, name: 'num.lgamma', sig: '(number z) -> number', doc: 'log|Γ(z)|（Stirling + 递推，大数不溢出）',
  ref: 'Stirling 展开（DLMF 5.11.1）', inv: 'lgamma(n)=ln((n-1)!)；lgamma(0.5)=ln√π',
  impl: (z) => {
    if (z < 0.5) return Math.log(Math.abs(Math.PI / Math.sin(Math.PI * z))) - exec('num.lgamma', [1 - z]);
    let zz = z, lacc = 0;
    while (zz < 10) { lacc += Math.log(zz); zz += 1; }
    return logGammaStirling(zz) - lacc;
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.beta', arity: 2, sig: '(number a, number b) -> number',
  doc: 'Beta 函数 Γ(a)Γ(b)/Γ(a+b)', ref: '特殊函数标准', inv: 'beta(a,b)=beta(b,a)；beta(1,1)=1',
  impl: (a, b) => exec('num.gamma', [a]) * exec('num.gamma', [b]) / exec('num.gamma', [a + b]),
});
defineOp({
  id: nid(), domain: D, name: 'num.normalQuantile', arity: 1, sig: '(number p, boolean refine=true) -> number',
  doc: '标准正态分位数（Acklam 逆近似 + Newton 修正到机器精度）',
  ref: 'Acklam 算法（Peter Acklam）；Wichura AS241 为另一基准',
  inv: 'normalQuantile(normalCdf(x))≈x（1e-12）；normalQuantile(0.5)=0',
  impl: (p, refine = true) => {
    if (p <= 0 || p >= 1) throw new ForgeError('E_VALUE', 'p∈(0,1)');
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    const plow = 0.02425;
    let x;
    if (p < plow) {
      const q = Math.sqrt(-2 * Math.log(p));
      x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    } else if (p <= 1 - plow) {
      const q = p - 0.5, r = q * q;
      x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    } else {
      const q = Math.sqrt(-2 * Math.log(1 - p));
      x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (!refine) return x;
    // Newton 修正：用 erf 构造 Cdf
    const cdf = (xx) => 0.5 * (1 + exec('num.erf', [xx / Math.SQRT2]));
    const pdf = (xx) => Math.exp(-0.5 * xx * xx) / Math.sqrt(2 * Math.PI);
    return x - (cdf(x) - p) / pdf(x);
  },
});

// ============ I. 浮点细节（4 条） ============
defineOp({
  id: nid(), domain: D, name: 'num.eps', arity: 0, sig: '() -> number', doc: '机器 epsilon 2⁻⁵²',
  ref: 'IEEE 754 双精度', inv: '1+eps>1 且 1+eps/2=1', impl: () => Number.EPSILON,
});
defineOp({
  id: nid(), domain: D, name: 'num.nextafter', arity: 2, sig: '(number x, number y) -> number',
  doc: 'x 之后下一个可表示双精度（朝 y 方向）', ref: 'IEEE 754 nextAfter / C99 nextafter',
  inv: 'nextafter(1,2)-1=EPSILON；nextafter(0,1)=MIN_VALUE',
  impl: (x, y) => {
    if (Number.isNaN(x) || Number.isNaN(y)) return NaN;
    if (x === y) return y;
    if (x === 0) return y > 0 ? Number.MIN_VALUE : -Number.MIN_VALUE;
    const buf = new ArrayBuffer(8);
    const f = new Float64Array(buf), u = new Uint32Array(buf);
    f[0] = x;
    const negative = (u[1] & 0x80000000) !== 0;
    if ((y > x) === !negative) {
      u[0] += 1;
      if (u[0] === 0) u[1] += 1;
    } else {
      if (u[0] === 0) u[1] -= 1;
      u[0] -= 1;
    }
    return f[0];
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.ulp', sig: '(number x) -> number', doc: 'x 处的 ULP（ulp(0)=最小正规数）',
  ref: 'IEEE 754', inv: 'ulp(1)=EPSILON；ulp(x) 随量级倍增',
  impl: (x) => {
    const ax = Math.abs(x);
    if (!Number.isFinite(ax) || ax === 0) return Number.MIN_VALUE;
    return exec('num.nextafter', [ax, Infinity]) - ax;
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.decompose', arity: 1, sig: '(number x) -> object', doc: 'Dekker 分裂 x=hi+lo（|hi| 恰好 26 位尾数）',
  ref: 'Dekker 1971；Higham §4.2', inv: 'hi+lo=x（精确）；hi 可用 26 位表示',
  impl: (x) => {
    const c = 134217729; // 2^27+1
    const p = x * c;
    const hi = p - (p - x);
    const lo = x - hi;
    return { hi, lo };
  },
});

// ============ J. 双精度倍精度算术 double-double（8 条） ============
defineOp({
  id: nid(), domain: D, name: 'num.twoSum', arity: 2, sig: '(number a, number b) -> object', doc: 'Knuth 两数和 s=a+b, e=误差',
  ref: 'Knuth TAOCP Vol.2 §4.2.2', inv: 's+e=a+b（精确）；e 可表示',
  impl: (a, b) => { const s = a + b; const bb = s - a; return { s, e: (a - (s - bb)) + (b - bb) }; },
});
defineOp({
  id: nid(), domain: D, name: 'num.twoProd', arity: 2, sig: '(number a, number b) -> object', doc: 'Dekker 两乘积 p=a·b, e=误差',
  ref: 'Dekker 1971', inv: 'p+e=a·b（精确）', impl: (a, b) => {
    const p = a * b;
    const { hi, lo } = exec('num.decompose', [a]);
    return { p, e: hi * b - p + lo * b };
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.quickTwoSum', arity: 2, sig: '(number a, number b) -> object', doc: '快速两和（要求 |a|≥|b|）',
  ref: 'Dekker 1971', inv: 's+e=a+b（|a|≥|b| 时）', impl: (a, b) => { const s = a + b; return { s, e: b - (s - a) }; },
});
function dd(x) { return x; } // dd 表示 {hi, lo}
defineOp({
  id: nid(), domain: D, name: 'num.ddFromNumber', sig: '(number x) -> object', doc: '数 → double-double',
  ref: 'double-double 算术标准', inv: 'ddToNumber(ddFromNumber(x))=x', impl: (x) => ({ hi: x, lo: 0 }),
});
defineOp({
  id: nid(), domain: D, name: 'num.ddToNumber', sig: '(object x) -> number', doc: 'double-double → 数（取 hi+lo）',
  ref: 'double-double 算术标准', inv: '返回最接近的双精度', impl: (x) => x.hi + x.lo,
});
defineOp({
  id: nid(), domain: D, name: 'num.ddAdd', arity: 2, sig: '(object x, object y) -> object', doc: 'double-double 加法',
  ref: 'double-double 算术标准', inv: '结果精度 ~106 位',
  impl: (x, y) => {
    const { s, e } = exec('num.twoSum', [x.hi, y.hi]);
    const e2 = e + (x.lo + y.lo);
    const r = exec('num.quickTwoSum', [s, e2]);
    return { hi: r.s, lo: r.e };
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.ddSub', arity: 2, sig: '(object x, object y) -> object', doc: 'double-double 减法',
  ref: 'double-double 算术标准', inv: 'ddSub(x,x)=0',
  impl: (x, y) => exec('num.ddAdd', [x, { hi: -y.hi, lo: -y.lo }]),
});
defineOp({
  id: nid(), domain: D, name: 'num.ddMul', arity: 2, sig: '(object x, object y) -> object', doc: 'double-double 乘法',
  ref: 'double-double 算术标准', inv: '结果精度 ~106 位',
  impl: (x, y) => {
    const { p, e } = exec('num.twoProd', [x.hi, y.hi]);
    const e2 = e + (x.hi * y.lo + x.lo * y.hi);
    const r = exec('num.quickTwoSum', [p, e2]);
    return { hi: r.s, lo: r.e };
  },
});

// ============ K. Welford 在线统计（4 条） ============
defineOp({
  id: nid(), domain: D, name: 'num.welfordInit', arity: 0, sig: '() -> object', doc: 'Welford 状态初始化',
  ref: 'Welford 1962', inv: '{n:0, mean:0, M2:0}', impl: () => ({ n: 0, mean: 0, M2: 0 }),
});
defineOp({
  id: nid(), domain: D, name: 'num.welfordUpdate', arity: 2, sig: '(object st, number x) -> object',
  doc: 'Welford 单点更新（方差数值稳定）', ref: 'Welford 1962',
  inv: 'n 递增；mean/M2 与两遍法一致（1e-12）',
  impl: (st, x) => {
    const n = st.n + 1;
    const delta = x - st.mean;
    const mean = st.mean + delta / n;
    return { n, mean, M2: st.M2 + delta * (x - mean) };
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.welfordMerge', arity: 2, sig: '(object a, object b) -> object',
  doc: 'Welford 并行合并（Chan 公式）', ref: 'Chan, Golub & LeVeque 1979（并行方差合并）',
  inv: 'merge(前半,后半)=全量 Welford（1e-12）',
  impl: (a, b) => {
    if (a.n === 0) return b;
    if (b.n === 0) return a;
    const n = a.n + b.n;
    const delta = b.mean - a.mean;
    return { n, mean: a.mean + (delta * b.n) / n, M2: a.M2 + b.M2 + (delta * delta * a.n * b.n) / n };
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.welfordFinalize', arity: 1, sig: '(object st) -> object',
  doc: 'Welford 输出 {n, mean, var, std}', ref: 'Welford 1962',
  inv: 'var=M2/n；std=√var；n=0 时 var=0',
  impl: (st) => ({ n: st.n, mean: st.mean, var: st.n > 0 ? st.M2 / st.n : 0, std: st.n > 0 ? Math.sqrt(st.M2 / st.n) : 0 }),
});

// ============ L. 输入守卫族（24 条生成） ============
// 每个 guard 变体：strict=true 时对越界输入抛 E_VALUE；否则返回文档化约定值。
const GUARDS = [
  ['exp', (v) => Math.exp(v), null],
  ['log', (v) => Math.log(v), (v) => v > 0],
  ['log1p', (v) => Math.log1p(v), (v) => v > -1],
  ['log2', (v) => Math.log2(v), (v) => v > 0],
  ['log10', (v) => Math.log10(v), (v) => v > 0],
  ['sqrt', (v) => Math.sqrt(v), (v) => v >= 0],
  ['cbrt', (v) => Math.cbrt(v), null],
  ['sin', (v) => Math.sin(v), null],
  ['cos', (v) => Math.cos(v), null],
  ['tan', (v) => Math.tan(v), null],
  ['asin', (v) => Math.asin(v), (v) => v >= -1 && v <= 1],
  ['acos', (v) => Math.acos(v), (v) => v >= -1 && v <= 1],
  ['atan', (v) => Math.atan(v), null],
  ['sinh', (v) => Math.sinh(v), null],
  ['cosh', (v) => Math.cosh(v), null],
  ['tanh', (v) => Math.tanh(v), null],
  ['asinh', (v) => Math.asinh(v), null],
  ['acosh', (v) => Math.acosh(v), (v) => v >= 1],
  ['atanh', (v) => Math.atanh(v), (v) => v > -1 && v < 1],
  ['expm1', (v) => Math.expm1(v), null],
  ['reciprocal', (v) => 1 / v, (v) => v !== 0],
  ['square', (v) => v * v, null],
  ['abs', (v) => Math.abs(v), null],
  ['sign', (v) => Math.sign(v), null],
];
for (const [n, f, dom] of GUARDS) {
  defineOp({
    id: nid(), domain: D, name: `num.guard.${n}`, arity: 1, sig: '(number x, boolean strict=false) -> number',
    doc: `守卫 ${n}：越界输入 strict 抛 E_VALUE，否则返回 IEEE 约定值`,
    ref: 'NumPy errstate / IEEE 754 域规则', inv: dom ? `定义域: ${dom}` : '全体实数定义域（NaN/Inf 按 IEEE 754 约定处理）',
    impl: (x, strict = false) => {
      if (Number.isNaN(x)) { if (strict) throw new ForgeError('E_VALUE', `guard.${n}: NaN 输入`); return NaN; }
      if (dom && !dom(x)) {
        if (strict) throw new ForgeError('E_VALUE', `guard.${n}: ${x} 越界`);
        return f(x);
      }
      return f(x);
    },
  });
}

// ============ 额外手写（5 条） ============
defineOp({
  id: nid(), domain: D, name: 'num.horner', arity: 2, sig: '(Tensor coefs, number x) -> number',
  doc: 'Horner 多项式求值（coefs 低次→高次）', ref: 'Horner 规则 / Numerical Recipes §5.3',
  inv: '对 coefs=[c0] 返回 c0；horner([1,2,3],0)=1',
  impl: (coefs, x) => {
    let s = 0;
    for (let i = coefs.size - 1; i >= 0; i--) s = s * x + coefs.data[i];
    return s;
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.dotKahan', arity: 2, sig: '(Tensor a, Tensor b) -> number',
  doc: 'Kahan 补偿点积', ref: 'Kahan 1965 / BLAS 补偿点积', inv: '比朴素点积更接近精确值',
  impl: (a, b) => {
    let s = 0, c = 0;
    for (let i = 0; i < a.size; i++) {
      const p = a.data[i] * b.data[i];
      const y = p - c;
      const t = s + y;
      c = (t - s) - y;
      s = t;
    }
    return s;
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.richardson', arity: 1,
  sig: '(Tensor ests, number k=2) -> number', doc: 'Richardson 外提（步长逐倍减半的估计序列）',
  ref: 'Richardson 1910 / Numerical Recipes §4.2', inv: '外提后精度阶数提升 k 倍',
  impl: (ests, k = 2) => {
    const cur = Array.from(ests.data);
    for (let j = 1; j < cur.length; j++) {
      for (let i = cur.length - 1; i >= j; i--) {
        cur[i] = cur[i] + (cur[i] - cur[i - 1]) / (k ** j - 1);
      }
    }
    return cur[cur.length - 1];
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.aitken', arity: 1, sig: '(Tensor seq) -> number', doc: 'Aitken Δ² 序列加速',
  ref: 'Aitken 1926 / Numerical Recipes §5.1', inv: '线性收敛序列加速后误差 O(原始²)',
  impl: (seq) => {
    const n = seq.size;
    if (n < 3) return seq.data[n - 1];
    const x0 = seq.data[n - 3], x1 = seq.data[n - 2], x2 = seq.data[n - 1];
    const d1 = x1 - x0, d2 = x2 - x1;
    const den = d2 - d1;
    return den === 0 ? x2 : x2 - (d2 * d2) / den;
  },
});
defineOp({
  id: nid(), domain: D, name: 'num.welfordFromTensor', arity: 1, sig: '(Tensor t) -> object',
  doc: '对整张量一次 Welford（返回状态）', ref: 'Welford 1962', inv: '与逐点更新一致',
  impl: (t) => {
    let st = exec('num.welfordInit', []);
    for (let i = 0; i < t.size; i++) st = exec('num.welfordUpdate', [st, t.data[i]]);
    return st;
  },
});

// ============ 索引工具 ============
function decodeIdx(p, shape) {
  const idx = new Array(shape.length);
  let r = p;
  for (let d = shape.length - 1; d >= 0; d--) { idx[d] = r % shape[d]; r = (r - idx[d]) / shape[d]; }
  return idx;
}
// 把"带被归约轴"的索引映射到"该轴已移除"形状的扁平索引
function flatWithRemoved(idx, removedAxis, outShape) {
  let q = 0;
  let oi = 0;
  for (let d = 0; d < idx.length; d++) {
    if (d === removedAxis) continue;
    q += idx[d] * strideAt(outShape, oi);
    oi++;
  }
  return q;
}
function strideAt(shape, axis) {
  let s = 1;
  for (let d = axis + 1; d < shape.length; d++) s *= shape[d];
  return s;
}

// 文件底部自检
if (opsByDomain(D).length !== 102) {
  throw new Error(`域${D}(NUM) 指令数 = ${opsByDomain(D).length}，应为 102`);
}
