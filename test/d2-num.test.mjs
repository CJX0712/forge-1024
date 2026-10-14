// 域2 数值方法与稳定化测试
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { exec, opsByDomain } from '../src/core/registry.mjs';
import { Tensor } from '../src/core/index.mjs';
import '../src/ops/d1-math.mjs'; // 域2 依赖域1（math.maxAxis 等）
import '../src/ops/d2-num.mjs'; // 导入即注册本域指令
import { approx, tensorClose } from './helpers.mjs';

const call = (name, args) => exec(name, args);
const N = (name, ...args) => call(name, args);

test('域2 恰好 102 条指令且 id 连续', () => {
  const ops = opsByDomain(2);
  assert.equal(ops.length, 102);
  const ids = ops.map(o => o.id).sort();
  assert.equal(ids[0], 'D2-001');
  assert.equal(ids[101], 'D2-102');
  assert.equal(new Set(ids).size, 102);
});

test('稳定 softmax / logSoftmax / logsumexp', () => {
  const x = Tensor.fromFlat([1000, 1000], [2]);
  const sm = N('num.softmax', x);
  assert.deepEqual(sm.toArray(), [0.5, 0.5]); // 不减 max 会溢出
  const x2 = Tensor.fromNested([[1, 2, 3], [1000, 1000, 1000]]);
  const sm2 = N('num.softmax', x2, -1);
  assert.deepEqual(Array.from(sm2.row(0)).map((v) => +v.toFixed(6)), [0.090031, 0.244728, 0.665241]);
  assert.deepEqual(Array.from(sm2.row(1)), [1 / 3, 1 / 3, 1 / 3]);
  // 每片和 = 1
  for (let i = 0; i < 2; i++) {
    let s = 0;
    for (let j = 0; j < 3; j++) s += sm2.get2(i, j);
    assert.ok(approx(s, 1, 1e-12), `第${i}片和=${s}`);
  }
  // 平移不变
  const sm3 = N('num.softmax', x2.map((v) => v + 777), -1);
  tensorClose(sm3, sm2, 1e-12);
  // logsumexp
  const lse = N('num.logsumexp', Tensor.fromFlat([1, 2, 3], [3]));
  assert.ok(approx(lse.scalar(), 3 + Math.log(1 + Math.exp(-1) + Math.exp(-2)), 1e-12));
  const lse2 = N('num.logsumexp', Tensor.fromFlat([1000, 1000], [2]));
  assert.ok(approx(lse2.scalar(), 1000 + Math.log(2), 1e-12));
  // logSoftmax = log(softmax)
  const lsm = N('num.logSoftmax', x2, -1);
  for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) assert.ok(approx(lsm.get2(i, j), Math.log(sm2.get2(i, j)), 1e-10));
});

test('sigmoid / softplus / stableExp 稳定性', () => {
  assert.equal(N('num.sigmoid', Tensor.fromFlat([0], [1])).scalar(), 0.5);
  const big = Tensor.fromFlat([1000, -1000], [2]);
  const s = N('num.sigmoid', big);
  assert.equal(s.data[0], 1);
  assert.equal(s.data[1], 0);
  assert.ok(approx(N('num.softplus', Tensor.fromFlat([0], [1])).scalar(), Math.log(2), 1e-12));
  assert.ok(approx(N('num.softplus', Tensor.fromFlat([50], [1])).scalar(), 50, 1e-12));
  assert.equal(N('num.stableExp', Tensor.fromFlat([1000], [1])).scalar(), Infinity);
  assert.equal(N('num.stableExp', Tensor.fromFlat([-1000], [1])).scalar(), 0);
});

test('补偿求和：Kahan/Neumaier/成对 vs 朴素', () => {
  // [1e16,1,-1e16]：数学真值 1，但 (1e16±1) 不可表示 —— 记录各算法实际行为
  const catastrophic = Tensor.fromFlat([1e16, 1, -1e16], [3]);
  assert.equal(N('num.naiveSum', catastrophic), 0);
  assert.equal(N('num.kahanSum', catastrophic), 0);
  assert.equal(N('num.neumaierSum', catastrophic), 1); // Neumaier 误差项恰好补偿出真值
  assert.equal(N('num.pairwiseSum', catastrophic), 0);
  // 长序列：1e6 个 1 加到 1e16 上，naive 全丢，Kahan 大部分保留
  const many = new Float64Array(1000000);
  many.fill(1);
  const t = new Tensor(many, [many.length]);
  const base = 1e16;
  const t2 = new Tensor(Float64Array.from([base, ...many]), [many.length + 1]);
  const naive = N('num.naiveSum', t2);
  const kahan = N('num.kahanSum', t2);
  assert.equal(naive, 1e16);
  assert.ok(kahan > 1e16 + 9e5, `kahan=${kahan} 应保留大部分增量`);
  // 交替序列：补偿不差于朴素
  const rngVals = [];
  let truth = 0;
  for (let i = 0; i < 10000; i++) { const v = (i % 2 ? 1 : -1) * (1 + i * 1e-3); rngVals.push(v); truth += v; }
  const t3 = Tensor.fromFlat(rngVals, [rngVals.length]);
  const naive3 = N('num.naiveSum', t3);
  const kahan3 = N('num.kahanSum', t3);
  assert.ok(Math.abs(kahan3 - truth) <= Math.abs(naive3 - truth), `kahan(${kahan3}) 应不差于 naive(${naive3})`);
  // 补偿点积：对 [1,1,1]·[1e16,1,1]
  const dk = N('num.dotKahan', Tensor.fromFlat([1, 1, 1], [3]), Tensor.fromFlat([1e16, 1, 1], [3]));
  assert.ok(dk > 1e16 + 1, `dotKahan=${dk}`);
});

test('误差度量', () => {
  const a = Tensor.fromFlat([1, 2, 3], [3]);
  const b = Tensor.fromFlat([1.1, 1.8, 3.5], [3]);
  assert.ok(approx(N('num.mae', a, b), (0.1 + 0.2 + 0.5) / 3, 1e-12));
  assert.ok(approx(N('num.mse', a, b), (0.01 + 0.04 + 0.25) / 3, 1e-12));
  assert.ok(approx(N('num.rmse', a, b), Math.sqrt((0.01 + 0.04 + 0.25) / 3), 1e-12));
  assert.ok(approx(N('num.maxAbsErr', a, b), 0.5, 1e-12));
  assert.ok(approx(N('num.medAbsErr', a, b), 0.2, 1e-12));
  assert.equal(N('num.mae', a, a), 0);
  assert.equal(N('num.absErr', 3, 5), 2);
  assert.ok(approx(N('num.relErr', 100, 101), 1 / 101, 1e-12));
});

test('isclose/allclose/arrayEqual（NumPy 语义）', () => {
  assert.equal(N('num.isclose', 1.0, 1.0 + 1e-10, 1e-5, 1e-8), true);
  assert.equal(N('num.isclose', 1.0, 1.1), false);
  assert.equal(N('num.isclose', 1e-9, 0, 0, 1e-8), true); // atol 生效
  const a = Tensor.fromFlat([1, 2], [2]), b = Tensor.fromFlat([1 + 1e-12, 2], [2]);
  assert.equal(N('num.allclose', a, b), true);
  assert.equal(N('num.allclose', a, Tensor.fromFlat([1, 3], [2])), false);
  assert.equal(N('num.arrayEqual', a, a), true);
  assert.equal(N('num.arrayEqual', a, b), false);
  assert.equal(N('num.isfiniteAll', a), true);
  assert.equal(N('num.isfiniteAll', Tensor.fromFlat([1, NaN], [2])), false);
  assert.equal(N('num.countNaN', Tensor.fromFlat([1, NaN, NaN], [3])), 2);
  assert.equal(N('num.countInf', Tensor.fromFlat([1, Infinity, -Infinity], [3])), 2);
  assert.deepEqual(N('num.nanToZero', Tensor.fromFlat([1, NaN], [2])).toArray(), [1, 0]);
});

test('求根：bisect/secant/newton/brent/ridder', () => {
  const f = (x) => x * x - 2;
  const sqrt2 = Math.SQRT2;
  const r1 = N('num.bisect', f, 1, 2, { tol: 1e-12 });
  assert.ok(approx(r1.root, sqrt2, 1e-12), `bisect=${r1.root}`);
  assert.ok(r1.converged);
  assert.ok(r1.iters >= 35 && r1.iters <= 100, `bisect iters=${r1.iters}`);
  const r2 = N('num.secant', f, 1, 2, { tol: 1e-12 });
  assert.ok(approx(r2.root, sqrt2, 1e-12));
  const r3 = N('num.newton', f, 1, { tol: 1e-12 });
  assert.ok(approx(r3.root, sqrt2, 1e-12));
  assert.ok(r3.iters < 10); // 二次收敛
  const r4 = N('num.brent', f, 1, 2, { tol: 1e-12 });
  assert.ok(approx(r4.root, sqrt2, 1e-12));
  assert.ok(r4.iters < r1.iters, `brent(${r4.iters}) 应快于 bisect(${r1.iters})`);
  const r5 = N('num.ridder', f, 1, 2, { tol: 1e-12 });
  assert.ok(approx(r5.root, sqrt2, 1e-12));
  // 非 bracket 抛错
  assert.throws(() => N('num.bisect', f, 2, 3), (e) => e.code === 'E_VALUE');
  // rootScan 找 sin 在 [0, 10] 的根
  const roots = N('num.rootScan', Math.sin, 0.5, 9.5, 90);
  assert.ok(roots.length >= 3);
  for (const [lo, hi] of roots) assert.ok(Math.sin(lo) * Math.sin(hi) <= 0);
});

test('积分：trapz/simpson/midpoint/romberg', () => {
  // ∫₀¹ x² dx = 1/3
  const xs = Tensor.linspace(0, 1, 101);
  const ys = xs.map((v) => v * v);
  const t = N('num.trapz', ys, xs);
  const s = N('num.simpson', ys, xs);
  assert.ok(approx(t, 1 / 3, 1e-4), `trapz=${t}`);
  assert.ok(approx(s, 1 / 3, 1e-9), `simpson=${s}`); // Simpson 对二次精确
  const m = N('num.midpoint', (x) => x * x, 0, 1, 1000);
  assert.ok(approx(m, 1 / 3, 1e-5));
  const r = N('num.romberg', (x) => x * x, 0, 1, 5);
  assert.ok(approx(r, 1 / 3, 1e-12), `romberg=${r}`);
  // ∫₀^π sin = 2（Simpson 对 sin 高精度）
  const xs2 = Tensor.linspace(0, Math.PI, 101);
  const ys2 = xs2.map(Math.sin);
  assert.ok(approx(N('num.simpson', ys2, xs2), 2, 1e-8));
  // cumtrapz 末元素 = trapz
  const ct = N('num.cumtrapz', ys, xs);
  assert.ok(approx(ct.data[ct.size - 1], t, 1e-12));
});

test('有限差分', () => {
  const f = (x) => { let s = 0; for (let i = 0; i < x.size; i++) s += x.data[i] ** 3; return s; };
  const x = Tensor.fromFlat([1, 2, -1], [3]);
  const g = N('num.gradCentral', f, x, 1e-5);
  assert.deepEqual(g.toArray().map((v) => +v.toFixed(6)), [3, 12, 3]);
  const gf = N('num.gradForward', f, x, 1e-5);
  assert.ok(approx(gf.data[0], 3, 1e-4));
  const hd = N('num.hessianDiag', (x2) => 0.5 * (x2.data[0] ** 2 + 3 * x2.data[1] ** 2), Tensor.fromFlat([1, 1], [2]), 1e-3);
  assert.deepEqual(hd.toArray().map((v) => +v.toFixed(3)), [1, 3]);
  const F = (x2) => Tensor.fromFlat([x2.data[0] ** 2, x2.data[0] * x2.data[1]], [2]);
  const J = N('num.jacobianFD', F, Tensor.fromFlat([2, 3], [2]), 1e-5);
  assert.deepEqual(J.toArray().map((r) => r.map((v) => +v.toFixed(4))), [[4, 0], [3, 2]]);
  assert.deepEqual(N('num.diff', Tensor.fromFlat([1, 4, 9, 16], [4])).toArray(), [3, 5, 7]);
  assert.deepEqual(N('num.diff', Tensor.fromFlat([1, 4, 9, 16], [4]), 2).toArray(), [2, 2]);
});

test('插值', () => {
  const xs = Tensor.fromFlat([0, 1, 2], [3]);
  const ys = Tensor.fromFlat([0, 10, 20], [3]);
  assert.equal(N('num.interpLinear', xs, ys, 0.5), 5);
  assert.equal(N('num.interpLinear', xs, ys, 1.5), 15);
  assert.equal(N('num.interpLinear', xs, ys, -1), -10); // 线性外推
  assert.equal(N('num.interpLinear', xs, ys, 3), 30);
  assert.equal(N('num.interpNearest', xs, ys, 0.4), 0);
  assert.equal(N('num.interpNearest', xs, ys, 0.6), 10);
  // Lagrange 对二次多项式精确
  const xs2 = Tensor.fromFlat([0, 1, 2, 3], [4]);
  const ys2 = xs2.map((v) => v * v + 1);
  assert.ok(approx(N('num.interpLagrange', xs2, ys2, 1.5), 1.5 * 1.5 + 1, 1e-10));
  assert.equal(N('num.searchSorted', xs, 1.5), 2);
  assert.equal(N('num.searchSorted', xs, 0), 0);
  // 批量插值
  const q = Tensor.fromFlat([0.25, 0.75, 1.25], [3]);
  assert.deepEqual(N('num.interpLinear', xs, ys, q).toArray(), [2.5, 7.5, 12.5]);
});

test('特殊函数（已知高精度值）', () => {
  assert.ok(approx(N('num.erf', 0), 0, 1e-15));
  assert.ok(approx(N('num.erf', 1), 0.8427007929497149, 1e-12));
  assert.ok(approx(N('num.erf', 0.5), 0.5204998778130465, 1e-12));
  assert.ok(approx(N('num.erf', 3), 0.9999779095030014, 1e-12));
  assert.ok(approx(N('num.erf', -1), -0.8427007929497149, 1e-12));
  assert.ok(approx(N('num.erfc', 0), 1, 1e-15));
  assert.ok(approx(N('num.erfc', 1), 0.15729920705028513, 1e-13));
  assert.ok(approx(N('num.erfc', 3), 2.209049699858544e-5, 1e-13));
  assert.ok(approx(N('num.erfc', -1), 1.8427007929497149, 1e-13));
  assert.ok(approx(N('num.gamma', 5), 24, 1e-10));
  assert.ok(approx(N('num.gamma', 0.5), 1.7724538509055159, 1e-12));
  assert.ok(approx(N('num.gamma', 1.5), 0.8862269254527580, 1e-12));
  assert.ok(approx(N('num.lgamma', 5), Math.log(24), 1e-12));
  assert.ok(approx(N('num.lgamma', 100), 359.1342053695754, 1e-8));
  assert.ok(approx(N('num.beta', 2, 3), 1 / 12, 1e-12));
  assert.ok(approx(N('num.beta', 0.5, 0.5), Math.PI, 1e-12));
});

test('逆正态分位数', () => {
  assert.ok(approx(N('num.normalQuantile', 0.5), 0, 1e-12));
  assert.ok(approx(N('num.normalQuantile', 0.975), 1.959963984540054, 1e-10));
  assert.ok(approx(N('num.normalQuantile', 0.001), -3.090232306167813, 1e-8));
  assert.ok(approx(N('num.normalQuantile', 0.99), 2.3263478740408408, 1e-10));
  // 往返一致性
  for (const p of [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]) {
    const x = N('num.normalQuantile', p);
    const cdf = 0.5 * (1 + N('num.erf', x / Math.SQRT2));
    assert.ok(approx(cdf, p, 1e-10), `p=${p} x=${x} cdf=${cdf}`);
  }
  assert.throws(() => N('num.normalQuantile', 0), (e) => e.code === 'E_VALUE');
  assert.throws(() => N('num.normalQuantile', 1), (e) => e.code === 'E_VALUE');
});

test('浮点细节：eps/nextafter/ulp/decompose', () => {
  assert.equal(N('num.eps'), Number.EPSILON);
  assert.ok(approx(N('num.nextafter', 1, 2) - 1, Number.EPSILON, 1e-30));
  assert.equal(N('num.nextafter', 0, 1), Number.MIN_VALUE);
  assert.equal(N('num.nextafter', 1, 0) - 1, -Number.EPSILON / 2);
  assert.ok(approx(N('num.ulp', 1), Number.EPSILON, 1e-30));
  assert.ok(approx(N('num.ulp', 0), Number.MIN_VALUE, 1e-30));
  const d = N('num.decompose', 1 + 2 ** -27);
  assert.ok(approx(d.hi + d.lo, 1 + 2 ** -27, 1e-30));
  assert.ok(approx(N('num.ulpErr', 1, N('num.nextafter', 1, 2)), 1, 1e-6));
  assert.equal(N('num.ulpErr', 1, 1), 0);
});

test('double-double 算术', () => {
  const ts = N('num.twoSum', 1, 1e-17);
  assert.ok(approx(ts.s + ts.e, 1 + 1e-17, 1e-30));
  const tp = N('num.twoProd', 1 + 2 ** -27, 1 + 2 ** -27);
  assert.ok(approx(tp.p + tp.e, (1 + 2 ** -27) ** 2, 1e-30));
  const dd = N('num.ddAdd', { hi: 1, lo: 2 ** -53 }, { hi: 2 ** -60, lo: 0 });
  assert.ok(approx(dd.hi + dd.lo, 1 + 2 ** -53 + 2 ** -60, 1e-15));
  assert.equal(N('num.ddSub', { hi: 1, lo: 0 }, { hi: 1, lo: 0 }).hi + N('num.ddSub', { hi: 1, lo: 0 }, { hi: 1, lo: 0 }).lo, 0);
  assert.equal(N('num.ddToNumber', N('num.ddFromNumber', 3.5)), 3.5);
  const m = N('num.ddMul', { hi: 1 + 2 ** -52, lo: 0 }, { hi: 1 + 2 ** -52, lo: 0 });
  assert.ok(approx(m.hi + m.lo, (1 + 2 ** -52) ** 2, 1e-15));
});

test('Welford 在线统计与并行合并', () => {
  const rng = { next: () => 0 };
  const data = Array.from({ length: 500 }, (_, i) => Math.sin(i) * 3 + i * 0.01);
  const t = Tensor.fromFlat(data, [data.length]);
  const st = N('num.welfordFromTensor', t);
  const fin = N('num.welfordFinalize', st);
  const mean2 = data.reduce((a, b) => a + b, 0) / data.length;
  const var2 = data.reduce((a, b) => a + (b - mean2) ** 2, 0) / data.length;
  assert.ok(approx(fin.mean, mean2, 1e-10), `mean ${fin.mean} vs ${mean2}`);
  assert.ok(approx(fin.var, var2, 1e-8), `var ${fin.var} vs ${var2}`);
  // 并行合并 = 全量
  const half = 250;
  const s1 = N('num.welfordFromTensor', new Tensor(data.slice(0, half), [half]));
  const s2 = N('num.welfordFromTensor', new Tensor(data.slice(half), [data.length - half]));
  const merged = N('num.welfordFinalize', N('num.welfordMerge', s1, s2));
  assert.ok(approx(merged.mean, fin.mean, 1e-10));
  assert.ok(approx(merged.var, fin.var, 1e-8));
  const init = N('num.welfordInit');
  assert.deepEqual(init, { n: 0, mean: 0, M2: 0 });
});

test('输入守卫族（strict 与宽松模式）', () => {
  assert.equal(N('num.guard.log', Math.E), 1);
  assert.equal(N('num.guard.log', 0, false), -Infinity);
  assert.equal(Number.isNaN(N('num.guard.log', -1, false)), true);
  assert.throws(() => N('num.guard.log', -1, true), (e) => e.code === 'E_VALUE');
  assert.throws(() => N('num.guard.log', NaN, true), (e) => e.code === 'E_VALUE');
  assert.equal(N('num.guard.sqrt', 4), 2);
  assert.equal(Number.isNaN(N('num.guard.sqrt', -4, false)), true);
  assert.throws(() => N('num.guard.sqrt', -4, true), (e) => e.code === 'E_VALUE');
  assert.equal(N('num.guard.asin', 1), Math.PI / 2);
  assert.equal(Number.isNaN(N('num.guard.asin', 2, false)), true);
  assert.equal(N('num.guard.exp', 1), Math.E);
  assert.equal(N('num.guard.acosh', 1), 0);
  assert.equal(N('num.guard.reciprocal', 4), 0.25);
  assert.equal(N('num.guard.atanh', 1, false), Infinity); // JS Math.atanh(1)=Infinity
  assert.equal(N('num.guard.atanh', 0), 0);
});

test('Horner / 补偿点积 / Richardson / Aitken', () => {
  // 2x²+3x+1 at x=2 → 15（coefs 低次→高次）
  assert.equal(N('num.horner', Tensor.fromFlat([1, 3, 2], [3]), 2), 15);
  assert.equal(N('num.horner', Tensor.fromFlat([5], [1]), 100), 5);
  // Richardson：对 ∫x² 用步长减半的 Simpson 序列外提应接近精确
  const f = (x) => x * x;
  const trap = (h) => { let s = 0; for (let x = h / 2; x < 1; x += h) s += f(x); return s * h; };
  const ests = Tensor.fromFlat([trap(0.1), trap(0.05), trap(0.025), trap(0.0125)], [4]);
  const rich = N('num.richardson', ests, 2);
  assert.ok(approx(rich, 1 / 3, 1e-6), `richardson=${rich}`);
  // Aitken 加速 x_{n+1}=cos(x_n) 的收敛序列
  let x = 1;
  const seq = [];
  for (let i = 0; i < 30; i++) { x = Math.cos(x); seq.push(x); }
  const acc = N('num.aitken', Tensor.fromFlat(seq.slice(-3), [3]));
  assert.ok(approx(acc, 0.7390851332151607, 1e-9), `aitken=${acc}`);
  const dk = N('num.dotKahan', Tensor.fromFlat([1e16, 1, 1], [3]), Tensor.fromFlat([1, 1, 1], [3]));
  assert.ok(approx(dk, 1e16 + 2, 1e-4)); // 补偿后保留 +2 中的一部分
});
