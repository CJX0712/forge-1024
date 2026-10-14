// 域6 概率统计 PROB 测试：102 条微指令的数值性质验证。
// 运行：node --test test/d6-prob.test.mjs
// 所有随机数据均经 createRng(seed)（PCG 流）生成，固定种子完全可复现。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import '../src/ops/d6-prob.mjs';
import { exec, tryExec, getOp, opsByDomain, describeOp } from '../src/core/index.mjs';
import { Tensor, createRng } from '../src/core/index.mjs';
import { approx, relErr } from './helpers.mjs';

const call = (name, args) => exec(name, args);
const T1 = (arr) => Tensor.fromFlat(arr, [arr.length]);

// —— 采样矩核对助手：|mean−μ| ≤ tol·max(1,|μ|,σ)，|var−σ²| ≤ tol·max(σ²,1e-3) ——
function checkMoments(vals, mu, v, tol = 0.05, label = '') {
  const n = vals.length;
  let m = 0;
  for (const x of vals) m += x;
  m /= n;
  let s2 = 0;
  for (const x of vals) s2 += (x - m) ** 2;
  s2 /= n;
  assert.ok(Math.abs(m - mu) <= tol * Math.max(1, Math.abs(mu), Math.sqrt(v)),
    `${label} 均值 ${m} vs 理论 ${mu}`);
  assert.ok(Math.abs(s2 - v) <= tol * Math.max(v, 1e-3),
    `${label} 方差 ${s2} vs 理论 ${v}`);
  return { mean: m, var: s2 };
}

// —— 连续分布规格表：pdf/cdf 取值 + 积分网格 + 理论矩 ——
// 网格按各分布尾部衰减速度选取，保证「截断尾质量 + 梯形误差」< 1e-6。
const CONT = [
  {
    name: 'normal', pdf: (x) => call('prob.normal.pdf', [x, 0, 1]),
    cdf: (x) => call('prob.normal.cdf', [x, 0, 1]),
    grid: [-10, 10, 20001], lo: -10, hi: 10,
    mean: 0, var: 1,
    sample: (seed, n) => call('prob.normal.sample', [seed, n, 0, 1]),
  },
  {
    name: 'uniform', pdf: (x) => call('prob.uniform.pdf', [x, 2, 5]),
    cdf: (x) => call('prob.uniform.cdf', [x, 2, 5]),
    grid: [2, 5, 30001], lo: 2, hi: 5,
    mean: 3.5, var: 9 / 12,
    sample: (seed, n) => call('prob.uniform.sample', [seed, n, 2, 5]),
  },
  {
    name: 'exponential', pdf: (x) => call('prob.exponential.pdf', [x, 1.5]),
    cdf: (x) => call('prob.exponential.cdf', [x, 1.5]),
    grid: [0, 30, 300001], lo: 0, hi: 30,
    mean: 1 / 1.5, var: 1 / 1.5 ** 2,
    sample: (seed, n) => call('prob.exponential.sample', [seed, n, 1.5]),
  },
  {
    name: 'laplace', pdf: (x) => call('prob.laplace.pdf', [x, 0, 1]),
    cdf: (x) => call('prob.laplace.cdf', [x, 0, 1]),
    grid: [-40, 40, 80001], lo: -40, hi: 40,
    mean: 0, var: 2,
    sample: (seed, n) => call('prob.laplace.sample', [seed, n, 0, 1]),
  },
  {
    name: 'logistic', pdf: (x) => call('prob.logistic.pdf', [x, 0, 1]),
    cdf: (x) => call('prob.logistic.cdf', [x, 0, 1]),
    grid: [-40, 40, 80001], lo: -40, hi: 40,
    mean: 0, var: Math.PI ** 2 / 3,
    sample: (seed, n) => call('prob.logistic.sample', [seed, n, 0, 1]),
  },
  {
    name: 'gamma', pdf: (x) => call('prob.gamma.pdf', [x, 2.5, 1.5]),
    cdf: (x) => call('prob.gamma.cdf', [x, 2.5, 1.5]),
    grid: [0, 40, 400001], lo: 0, hi: 40,
    mean: 2.5 / 1.5, var: 2.5 / 1.5 ** 2,
    sample: (seed, n) => call('prob.gamma.sample', [seed, n, 2.5, 1.5]),
  },
  {
    name: 'beta', pdf: (x) => call('prob.beta.pdf', [x, 2, 3]),
    cdf: (x) => call('prob.beta.cdf', [x, 2, 3]),
    grid: [0, 1, 100001], lo: 0, hi: 1,
    mean: 0.4, var: 0.04,
    sample: (seed, n) => call('prob.beta.sample', [seed, n, 2, 3]),
  },
  {
    name: 'chi2', pdf: (x) => call('prob.chi2.pdf', [x, 4]),
    cdf: (x) => call('prob.chi2.cdf', [x, 4]),
    grid: [0, 60, 600001], lo: 0, hi: 60,
    mean: 4, var: 8,
    sample: (seed, n) => call('prob.chi2.sample', [seed, n, 4]),
  },
  {
    name: 'studentT', pdf: (x) => call('prob.studentT.pdf', [x, 5]),
    cdf: (x) => call('prob.studentT.cdf', [x, 5]),
    grid: [-500, 500, 500001], lo: -500, hi: 500,
    mean: 0, var: NaN, // t5 方差无穷，矩核对单独用 ν=10
    sample: (seed, n) => call('prob.studentT.sample', [seed, n, 10]),
    theoryForSample: { mean: 0, var: 10 / 8 },
  },
  {
    name: 'rayleigh', pdf: (x) => call('prob.rayleigh.pdf', [x, 1.5]),
    cdf: (x) => call('prob.rayleigh.cdf', [x, 1.5]),
    grid: [0, 40, 400001], lo: 0, hi: 40,
    mean: 1.5 * Math.sqrt(Math.PI / 2), var: (4 - Math.PI) / 2 * 1.5 ** 2,
    // rayleigh 4 阶矩大（κ≈40），方差收敛慢，矩核对用 20 万样本
    sample: (seed, n) => call('prob.rayleigh.sample', [seed, n, 1.5]), bigN: true,
  },
  {
    name: 'weibull', pdf: (x) => call('prob.weibull.pdf', [x, 1.5, 2]),
    cdf: (x) => call('prob.weibull.cdf', [x, 1.5, 2]),
    grid: [0, 30, 300001], lo: 0, hi: 30,
    mean: 2 * Math.exp(lgammaTest(1 + 1 / 1.5)),
    var: 4 * (Math.exp(lgammaTest(1 + 2 / 1.5)) - Math.exp(lgammaTest(1 + 1 / 1.5)) ** 2),
    sample: (seed, n) => call('prob.weibull.sample', [seed, n, 1.5, 2]), bigN: true,
  },
];

// 测试内独立的 lgamma（Lanczos），用于构造 weibull 理论矩期望值
function lgammaTest(x) {
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgammaTest(1 - x);
  const z = x - 1;
  let a = c[0];
  const t = z + 7.5;
  for (let i = 1; i < 9; i++) a += c[i] / (z + i);
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

// ============================================================================
test('D6 注册表：恰好 102 条、id 连续 D6-001..D6-102、元数据完整', () => {
  const ops = opsByDomain(6);
  assert.equal(ops.length, 102);
  const ids = ops.map((o) => o.id).sort();
  assert.deepEqual(ids, Array.from({ length: 102 }, (_, i) => `D6-${String(i + 1).padStart(3, '0')}`));
  const names = ops.map((o) => o.name);
  assert.equal(new Set(names).size, 102, 'name 必须全局唯一');
  for (const o of ops) {
    assert.equal(o.domain, 6);
    assert.ok(o.name.startsWith('prob.'), `${o.id} name 缺 prob. 前缀`);
    assert.equal(typeof o.impl, 'function');
    for (const k of ['sig', 'doc', 'ref', 'inv']) {
      assert.ok(typeof o[k] === 'string' && o[k].length >= 2, `${o.id} 字段 ${k} 缺失`);
    }
  }
  // describeOp 可读
  assert.ok(describeOp('prob.normal.pdf').includes('D6-001'));
  assert.ok(getOp('D6-102').name === 'prob.util.sampleMoments');
});

// ============================================================================
test('连续分布 pdf 积分 ≈ 1（trapz，容差 1e-6）', () => {
  for (const d of CONT) {
    const [lo, hi, n] = d.grid;
    const xs = Tensor.linspace(lo, hi, n);
    const ys = d.pdf(xs);
    const I = call('prob.util.trapz', [ys, xs]);
    assert.ok(Math.abs(I - 1) <= 1e-6, `${d.name} pdf 积分 = ${I}（误差 ${Math.abs(I - 1)}）`);
  }
});

test('柯西 pdf：切线变换 x = tan θ 后 trapz ≈ 1（尾 1/x² 专用）', () => {
  const N = 20001;
  const th = Tensor.linspace(-Math.PI / 2, Math.PI / 2, N);
  // dx/dθ = sec²θ = 1 + tan²θ；pdf(tanθ)·sec²θ = 1/π（常数）
  const ys = th.map((t) => {
    const x = Math.tan(t);
    return call('prob.cauchy.pdf', [x, 0, 1]) * (1 + x * x);
  });
  const I = call('prob.util.trapz', [ys, th]);
  assert.ok(Math.abs(I - 1) <= 1e-6, `cauchy(切线变换) 积分 = ${I}`);
  // 直接宽网格粗校验（尾质量 ~6.4e-4）
  const xs = Tensor.linspace(-1000, 1000, 400001);
  const ys2 = call('prob.cauchy.pdf', [xs, 0, 1]);
  const I2 = call('prob.util.trapz', [ys2, xs]);
  assert.ok(Math.abs(I2 - 1) <= 1e-3, `cauchy 直接积分 = ${I2}`);
});

test('帕累托 pdf：对数尾变换 x = xm·e^t 后 trapz ≈ 1', () => {
  // x = 2·e^t，dx = 2e^t dt ⇒ pdf(x)·dx/dt = α·e^{−αt}，t ∈ [0, 9] 尾质量 e^{−22.5} ≈ 1.7e-10
  const N = 90001;
  const ts = Tensor.linspace(0, 9, N);
  const ys = ts.map((t) => {
    const x = 2 * Math.exp(t);
    return call('prob.pareto.pdf', [x, 2, 2.5]) * 2 * Math.exp(t);
  });
  const I = call('prob.util.trapz', [ys, ts]);
  assert.ok(Math.abs(I - 1) <= 1e-6, `pareto(对数变换) 积分 = ${I}`);
});

// ============================================================================
test('连续分布 cdf：单调不减、端点正确、与 pdf 积分一致', () => {
  for (const d of CONT) {
    const [lo, hi] = [d.lo, d.hi];
    const n = 2001;
    const xs = Tensor.linspace(lo, hi, n);
    const cdf = d.cdf(xs).data;
    for (let i = 1; i < n; i++) {
      assert.ok(cdf[i] >= cdf[i - 1] - 1e-12, `${d.name} cdf 在 ${xs.at(i)} 处递减`);
      assert.ok(cdf[i] >= -1e-15 && cdf[i] <= 1 + 1e-15, `${d.name} cdf 越界 [0,1]`);
    }
    // 端点（按支撑调整到网格内）
    assert.ok(d.cdf(lo) <= 1e-6, `${d.name} cdf(lo) = ${d.cdf(lo)}`);
    assert.ok(d.cdf(hi) >= 1 - 1e-6, `${d.name} cdf(hi) = ${d.cdf(hi)}`);
  }
  // 支撑外取值：pdf=0 / cdf 饱和
  assert.equal(call('prob.normal.pdf', [-1e9, 0, 1]), 0);
  assert.equal(call('prob.normal.cdf', [-1e9, 0, 1]), 0);
  assert.equal(call('prob.normal.cdf', [1e9, 0, 1]), 1);
  assert.equal(call('prob.exponential.pdf', [-1, 2]), 0);
  assert.equal(call('prob.beta.pdf', [1.5, 2, 3]), 0);
  assert.equal(call('prob.pareto.pdf', [1, 2, 2.5]), 0);
});

test('连续分布 cdf 差 = pdf 的 trapz 积分（容差 1e-6）', () => {
  const spans = [
    ['normal', -3, 3], ['uniform', 2.5, 4.5], ['exponential', 0, 5], ['laplace', -5, 5],
    ['logistic', -5, 5], ['gamma', 0.1, 8], ['beta', 0.05, 0.95], ['chi2', 0.1, 15],
    ['studentT', -6, 6], ['rayleigh', 0.1, 8], ['weibull', 0.1, 8], ['pareto', 2.1, 20],
    ['cauchy', -5, 5],
  ];
  for (const [name, a, b] of spans) {
    const d = CONT.find((x) => x.name === name) || { pdf: (x) => call('prob.cauchy.pdf', [x, 0, 1]), cdf: (x) => call('prob.cauchy.cdf', [x, 0, 1]) };
    const n = 200001;
    const xs = Tensor.linspace(a, b, n);
    const I = call('prob.util.trapz', [d.pdf(xs), xs]);
    const D = d.cdf(b) - d.cdf(a);
    assert.ok(Math.abs(I - D) <= 1e-6, `${name}: trapz=${I} vs cdf差=${D}`);
  }
});

// ============================================================================
test('连续分布采样矩 ≈ 理论矩（固定种子，容差 5%）', () => {
  for (const d of CONT) {
    const n = d.bigN ? 200000 : 20000;
    const s = d.sample(1234, n);
    assert.equal(s.shape[0], n);
    const theory = d.theoryForSample || { mean: d.mean, var: d.var };
    checkMoments(Array.from(s.data), theory.mean, theory.var, 0.05, d.name);
    // 同种子完全可复现
    const s2 = d.sample(1234, 1000);
    const s3 = d.sample(1234, 1000);
    assert.deepEqual(Array.from(s2.data), Array.from(s3.data), `${d.name} 采样不可复现`);
  }
});

test('柯西采样：无矩，样本中位数 ≈ x0', () => {
  const s = call('prob.cauchy.sample', [77, 20000, 0, 1]);
  const med = call('prob.stat.median', [s]);
  assert.ok(Math.abs(med) <= 0.05, `cauchy 样本中位数 ${med}（理论 0）`);
  const s2 = call('prob.cauchy.sample', [78, 20000, 2.5, 0.5]);
  const med2 = call('prob.stat.median', [s2]);
  assert.ok(Math.abs(med2 - 2.5) <= 0.05, `cauchy 样本中位数 ${med2}（理论 2.5）`);
});

// ============================================================================
// 离散分布规格表
const DISC = [
  {
    name: 'bernoulli', pmf: (k) => call('prob.bernoulli.pmf', [k, 0.3]),
    cdf: (k) => call('prob.bernoulli.cdf', [k, 0.3]),
    ks: [0, 1], mean: 0.3, var: 0.21,
    sample: (seed, n) => call('prob.bernoulli.sample', [seed, n, 0.3]),
  },
  {
    name: 'binomial', pmf: (k) => call('prob.binomial.pmf', [k, 10, 0.4]),
    cdf: (k) => call('prob.binomial.cdf', [k, 10, 0.4]),
    ks: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], mean: 4, var: 2.4,
    sample: (seed, n) => call('prob.binomial.sample', [seed, n, 10, 0.4]),
  },
  {
    name: 'poisson', pmf: (k) => call('prob.poisson.pmf', [k, 4]),
    cdf: (k) => call('prob.poisson.cdf', [k, 4]),
    ks: Array.from({ length: 101 }, (_, i) => i), mean: 4, var: 4,
    sample: (seed, n) => call('prob.poisson.sample', [seed, n, 4]),
  },
  {
    name: 'geometric', pmf: (k) => call('prob.geometric.pmf', [k, 0.2]),
    cdf: (k) => call('prob.geometric.cdf', [k, 0.2]),
    ks: Array.from({ length: 500 }, (_, i) => i + 1), mean: 5, var: 20,
    sample: (seed, n) => call('prob.geometric.sample', [seed, n, 0.2]),
  },
  {
    name: 'categorical', pmf: (k) => call('prob.categorical.pmf', [k, T1([0.1, 0.2, 0.3, 0.4])]),
    cdf: (k) => call('prob.categorical.cdf', [k, T1([0.1, 0.2, 0.3, 0.4])]),
    ks: [0, 1, 2, 3], mean: 2, var: 1,
    sample: (seed, n) => call('prob.categorical.sample', [seed, n, T1([0.1, 0.2, 0.3, 0.4])]),
  },
];

test('离散分布 pmf 求和 = 1 且 cdf 端点/单调正确（容差 1e-9）', () => {
  for (const d of DISC) {
    let s = 0;
    for (const k of d.ks) s += d.pmf(k);
    assert.ok(Math.abs(s - 1) <= 1e-9, `${d.name} pmf 求和 = ${s}`);
    // cdf 单调 + 端点
    let prev = -1;
    for (const k of d.ks) {
      const c = d.cdf(k);
      assert.ok(c >= prev - 1e-12, `${d.name} cdf 在 k=${k} 递减`);
      assert.ok(c >= -1e-12 && c <= 1 + 1e-12, `${d.name} cdf 越界`);
      prev = c;
    }
    assert.ok(d.cdf(d.ks[0] - 1) === 0 || d.cdf(d.ks[0] - 1) < 1e-12, `${d.name} cdf 左端非 0`);
    assert.ok(d.cdf(d.ks[d.ks.length - 1]) > 1 - 1e-9, `${d.name} cdf 右端非 1：${d.cdf(d.ks[d.ks.length - 1])}`);
    // cdf(k) = Σ_{i≤k} pmf
    let acc = 0;
    for (const k of d.ks) {
      acc += d.pmf(k);
      assert.ok(approx(d.cdf(k), acc, 1e-9), `${d.name} cdf(${k})=${d.cdf(k)} vs Σpmf=${acc}`);
    }
  }
});

test('离散分布 pmf/cdf 已知值核对', () => {
  assert.equal(call('prob.bernoulli.pmf', [1, 0.3]), 0.3);
  assert.equal(call('prob.bernoulli.pmf', [0, 0.3]), 0.7);
  assert.equal(call('prob.bernoulli.pmf', [2, 0.3]), 0); // 非 {0,1}
  assert.equal(call('prob.bernoulli.cdf', [0, 0.3]), 0.7);
  assert.equal(call('prob.bernoulli.cdf', [1, 0.3]), 1);
  assert.ok(approx(call('prob.binomial.pmf', [5, 10, 0.5]), 0.24609375, 1e-12));
  assert.ok(approx(call('prob.binomial.pmf', [3, 10, 0.4]), 0.214990848, 1e-9));
  assert.ok(approx(call('prob.binomial.pmf', [0, 10, 0.5]), 1 / 1024, 1e-12));
  assert.ok(approx(call('prob.binomial.cdf', [5, 10, 0.5]), 0.623046875, 1e-12));
  assert.ok(approx(call('prob.poisson.pmf', [2, 4]), 0.1465251111, 1e-9));
  assert.ok(approx(call('prob.poisson.pmf', [7, 4]), 0.0595403, 1e-6));
  assert.ok(approx(call('prob.poisson.cdf', [4, 4]), 0.628837, 1e-6));
  assert.ok(approx(call('prob.geometric.pmf', [1, 0.25]), 0.25, 1e-12));
  assert.ok(approx(call('prob.geometric.pmf', [3, 0.25]), 0.140625, 1e-12));
  assert.ok(approx(call('prob.geometric.cdf', [2, 0.25]), 0.4375, 1e-12));
  assert.ok(approx(call('prob.categorical.pmf', [2, T1([0.1, 0.2, 0.3, 0.4])]), 0.3, 1e-12));
  assert.ok(approx(call('prob.categorical.cdf', [1, T1([0.1, 0.2, 0.3, 0.4])]), 0.3, 1e-12));
  // 非整数 k → pmf 0
  assert.equal(call('prob.poisson.pmf', [2.5, 4]), 0);
  assert.equal(call('prob.binomial.pmf', [-1, 10, 0.4]), 0);
  assert.equal(call('prob.geometric.pmf', [0, 0.2]), 0);
});

test('离散分布采样矩 ≈ 理论矩（固定种子，容差 5%）', () => {
  for (const d of DISC) {
    const s = d.sample(4321, 20000);
    checkMoments(Array.from(s.data), d.mean, d.var, 0.05, d.name);
    const s2 = d.sample(4321, 1000);
    const s3 = d.sample(4321, 1000);
    assert.deepEqual(Array.from(s2.data), Array.from(s3.data), `${d.name} 采样不可复现`);
  }
});

// ============================================================================
test('多元正态：pdf 闭式值、批量求值、采样协方差', () => {
  const mu = T1([1, -2]);
  const Sigma = Tensor.fromNested([[4, 1.2], [1.2, 1]]);
  // |Σ| = 4·1 − 1.2² = 2.56，pdf(μ) = 1/(2π√2.56)
  const det = 2.56;
  assert.ok(approx(call('prob.mvn.pdf', [mu, mu, Sigma]), 1 / (2 * Math.PI * Math.sqrt(det)), 1e-12));
  // x = [2,−1]：(x−μ)ᵀΣ⁻¹(x−μ) = (1+4−2.4)/2.56 = 1.015625
  const q = (1 + 4 - 2.4) / det;
  assert.ok(approx(call('prob.mvn.pdf', [T1([2, -1]), mu, Sigma]),
    Math.exp(-0.5 * q) / (2 * Math.PI * Math.sqrt(det)), 1e-12));
  // 批量
  const X = Tensor.fromNested([[1, -2], [2, -1]]);
  const out = call('prob.mvn.pdf', [X, mu, Sigma]);
  assert.deepEqual(out.shape, [2]);
  assert.ok(approx(out.at(0), 1 / (2 * Math.PI * Math.sqrt(det)), 1e-12));
  // 采样矩
  const S = call('prob.mvn.sample', [9, 20000, mu, Sigma]);
  assert.deepEqual(S.shape, [20000, 2]);
  const d = S.data;
  const n = S.shape[0];
  let m1 = 0, m2 = 0;
  for (let i = 0; i < n; i++) { m1 += d[2 * i]; m2 += d[2 * i + 1]; }
  m1 /= n; m2 /= n;
  let c = 0, v1 = 0, v2 = 0;
  for (let i = 0; i < n; i++) {
    c += (d[2 * i] - m1) * (d[2 * i + 1] - m2);
    v1 += (d[2 * i] - m1) ** 2;
    v2 += (d[2 * i + 1] - m2) ** 2;
  }
  c /= n; v1 /= n; v2 /= n;
  assert.ok(Math.abs(m1 - 1) <= 0.05 && Math.abs(m2 + 2) <= 0.05, `mvn 样本均值 [${m1},${m2}]`);
  assert.ok(Math.abs(v1 - 4) <= 0.2 && Math.abs(v2 - 1) <= 0.05, `mvn 样本方差 [${v1},${v2}]`);
  assert.ok(Math.abs(c - 1.2) <= 0.1, `mvn 样本协方差 ${c}`);
  // 1 维退化情形的 pdf 与正态一致
  const mu1 = T1([0.5]), S1 = Tensor.fromNested([[2.25]]);
  assert.ok(approx(call('prob.mvn.pdf', [T1([1]), mu1, S1]),
    call('prob.normal.pdf', [1, 0.5, 1.5]), 1e-12));
});

// ============================================================================
test('信息论：entropy / crossEntropy / klDiv / jsDiv', () => {
  const P = T1([0.5, 0.5]);
  assert.ok(approx(call('prob.info.entropy', [P]), Math.log(2), 1e-12));
  assert.ok(approx(call('prob.info.entropy', [T1([1])]), 0, 1e-12));
  assert.ok(approx(call('prob.info.entropy', [T1([0.25, 0.25, 0.25, 0.25])]), Math.log(4), 1e-12));
  // 均匀分布最大化熵
  const rng = createRng(5);
  for (let t = 0; t < 20; t++) {
    const raw = Array.from({ length: 5 }, () => rng.next() + 0.01);
    const s = raw.reduce((a, b) => a + b, 0);
    const p = T1(raw.map((v) => v / s));
    assert.ok(call('prob.info.entropy', [p]) <= Math.log(5) + 1e-12, '熵不应超过 ln|X|');
  }
  // 交叉熵 ≥ 熵，等号当 p = q；H([.5,.5],[.25,.75]) = ½ln4 + ½ln(4/3)
  const Q = T1([0.25, 0.75]);
  assert.ok(approx(call('prob.info.crossEntropy', [P, Q]), 0.5 * Math.log(4) + 0.5 * Math.log(4 / 3), 1e-12));
  assert.ok(call('prob.info.crossEntropy', [P, Q]) >= call('prob.info.entropy', [P]) - 1e-12);
  assert.ok(approx(call('prob.info.crossEntropy', [P, P]), Math.log(2), 1e-12));
  // KL：D(p‖p)=0、D≥0、已知值 D([.5,.5]‖[.25,.75]) = ½ln2 + ½ln(2/3) = ½ln(4/3)
  assert.ok(approx(call('prob.info.klDiv', [P, P]), 0, 1e-12));
  assert.ok(approx(call('prob.info.klDiv', [P, Q]), 0.5 * Math.log(4 / 3), 1e-12));
  const A = T1([0.1, 0.2, 0.3, 0.4]);
  const B = T1([0.4, 0.3, 0.2, 0.1]);
  assert.ok(call('prob.info.klDiv', [A, B]) >= -1e-12);
  assert.ok(call('prob.info.klDiv', [A, B]) > 0.01);
  // JS：∈[0, ln2]，对称，JS(p,p)=0，不相交时为 ln2
  assert.ok(approx(call('prob.info.jsDiv', [P, P]), 0, 1e-12));
  assert.ok(approx(call('prob.info.jsDiv', [T1([1, 0]), T1([0, 1])]), Math.log(2), 1e-12));
  const js = call('prob.info.jsDiv', [A, B]);
  assert.ok(js >= 0 && js <= Math.log(2) + 1e-12, `JS=${js}`);
  assert.ok(approx(call('prob.info.jsDiv', [A, B]), call('prob.info.jsDiv', [B, A]), 1e-12));
  // 交叉验证：JS = ½KL(p‖m) + ½KL(q‖m)
  const M = T1([0.25, 0.25, 0.25, 0.25]);
  assert.ok(approx(js, 0.5 * call('prob.info.klDiv', [A, M]) + 0.5 * call('prob.info.klDiv', [B, M]), 1e-12));
});

test('信息论：mutualInfo / wasserstein1 / hellinger / tvd', () => {
  // 互信息：独立联合 → 0；完全相关 → ln2
  const indep = Tensor.fromNested([[0.25, 0.25], [0.25, 0.25]]);
  assert.ok(approx(call('prob.info.mutualInfo', [indep]), 0, 1e-12));
  const perfect = Tensor.fromNested([[0.5, 0], [0, 0.5]]);
  assert.ok(approx(call('prob.info.mutualInfo', [perfect]), Math.log(2), 1e-12));
  const mixed = Tensor.fromNested([[0.3, 0.1], [0.2, 0.4]]);
  const mi = call('prob.info.mutualInfo', [mixed]);
  assert.ok(mi > 0 && mi <= Math.log(2) + 1e-12, `MI=${mi}`);
  // 手工核对：I = Σ P ln(P/(Px·Py))
  {
    const rows = [[0.3, 0.1], [0.2, 0.4]];
    const px = [0.4, 0.6], py = [0.5, 0.5];
    let exp = 0;
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) exp += rows[i][j] * Math.log(rows[i][j] / (px[i] * py[j]));
    assert.ok(approx(mi, exp, 1e-12));
  }
  // Wasserstein：正态闭式 |μ1−μ2|（与尺度无关）+ 经验式
  assert.ok(approx(call('prob.info.wasserstein1', [0, 1, 3, 1]), 3, 1e-12));
  assert.ok(approx(call('prob.info.wasserstein1', [0, 1, 1, 2]), 1, 1e-12));
  assert.ok(approx(call('prob.info.wasserstein1', [5, 3, 5, 0.1]), 0, 1e-12));
  assert.ok(approx(call('prob.info.wasserstein1', [T1([1, 2, 3]), T1([4, 5, 6])]), 3, 1e-12));
  assert.ok(approx(call('prob.info.wasserstein1', [T1([1, 2, 3]), T1([1, 2, 3])]), 0, 1e-12));
  // 经验 W1 与正态样本矩一致性（同分布 → 小）
  const rng = createRng(66);
  const xs = Array.from({ length: 5000 }, () => rng.normal());
  const ys = Array.from({ length: 5000 }, () => rng.normal());
  assert.ok(call('prob.info.wasserstein1', [T1(xs), T1(ys)]) < 0.05, '同分布经验 W1 应接近 0');
  // Hellinger 与 TVD
  const U = T1([0.5, 0.5]);
  assert.ok(approx(call('prob.info.hellinger', [U, U]), 0, 1e-12));
  assert.ok(approx(call('prob.info.hellinger', [T1([1, 0]), T1([0, 1])]), 1, 1e-12));
  // H²([1,0],[.5,.5]) = ½[(1−√.5)² + .5] ≈ 0.2929
  assert.ok(approx(call('prob.info.hellinger', [T1([1, 0]), U]) ** 2, 0.5 * ((1 - Math.sqrt(0.5)) ** 2 + 0.5), 1e-12));
  assert.ok(call('prob.info.hellinger', [T1([1, 0]), U]) ** 2 <= call('prob.info.tvd', [T1([1, 0]), U]) + 1e-12, 'H² ≤ TVD');
  assert.ok(approx(call('prob.info.tvd', [U, U]), 0, 1e-12));
  assert.ok(approx(call('prob.info.tvd', [T1([1, 0]), T1([0, 1])]), 1, 1e-12));
  const A = T1([0.1, 0.2, 0.3, 0.4]), B = T1([0.4, 0.3, 0.2, 0.1]);
  const tvd = call('prob.info.tvd', [A, B]);
  assert.ok(tvd >= 0 && tvd <= 1);
  assert.ok(approx(tvd, 0.5 * (0.3 + 0.1 + 0.1 + 0.3), 1e-12));
});

// ============================================================================
test('矩与统计量：已知值核对（总体矩口径）', () => {
  assert.ok(approx(call('prob.stat.mean', [T1([1, 2, 3, 4])]), 2.5, 1e-12));
  assert.ok(approx(call('prob.stat.variance', [T1([1, 2, 3, 4])]), 1.25, 1e-12));
  assert.ok(approx(call('prob.stat.std', [T1([1, 2, 3, 4])]), Math.sqrt(1.25), 1e-12));
  assert.ok(approx(call('prob.stat.skewness', [T1([1, 2, 3, 4, 5])]), 0, 1e-12));
  assert.ok(approx(call('prob.stat.kurtosis', [T1([1, 2, 3, 4, 5])]), -1.3, 1e-12));
  // cov/corr 手工值：x=[1..5], y=[2,4,5,4,5] ⇒ Σdev·dev=6，cov=1.2，σx²=2，σy²=1.2
  assert.ok(approx(call('prob.stat.cov', [T1([1, 2, 3, 4, 5]), T1([2, 4, 5, 4, 5])]), 1.2, 1e-12));
  assert.ok(approx(call('prob.stat.corr', [T1([1, 2, 3, 4, 5]), T1([2, 4, 5, 4, 5])]), 1.2 / Math.sqrt(2 * 1.2), 1e-12));
  assert.ok(approx(call('prob.stat.cov', [T1([1, 2, 3]), T1([1, 2, 3])]), call('prob.stat.variance', [T1([1, 2, 3])]), 1e-12));
  assert.ok(approx(call('prob.stat.corr', [T1([1, 2, 3]), T1([1, 2, 3])]), 1, 1e-12));
  assert.ok(approx(call('prob.stat.corr', [T1([1, 2, 3]), T1([-1, -2, -3])]), -1, 1e-12));
  // 分位数（线性插值口径）
  assert.ok(approx(call('prob.stat.quantile', [T1([1, 2, 3, 4]), 0.5]), 2.5, 1e-12));
  assert.ok(approx(call('prob.stat.quantile', [T1([1, 2, 3, 4, 5]), 0.25]), 2, 1e-12));
  assert.ok(approx(call('prob.stat.quantile', [T1([1, 2, 3, 4, 5]), 0.9]), 4.6, 1e-12));
  assert.ok(approx(call('prob.stat.median', [T1([3, 1, 2])]), 2, 1e-12));
  assert.ok(approx(call('prob.stat.median', [T1([4, 1, 3, 2])]), 2.5, 1e-12));
  // MAD：median(|x − median|)
  assert.ok(approx(call('prob.stat.mad', [T1([1, 2, 3, 4, 100])]), 1, 1e-12));
  assert.ok(approx(call('prob.stat.mad', [T1([1, 2, 3, 4, 5])]), 1, 1e-12));
  // sampleMoments
  const m = call('prob.util.sampleMoments', [T1([1, 2, 3, 4, 5])]);
  assert.ok(approx(m.mean, 3, 1e-12) && approx(m.variance, 2, 1e-12));
  assert.ok(approx(m.skewness, 0, 1e-12) && approx(m.kurtosis, -1.3, 1e-12));
  const m2 = call('prob.util.sampleMoments', [T1([1, 1, 1, 1, 2])]);
  assert.ok(approx(m2.skewness, 1.5, 1e-12) && approx(m2.kurtosis, 0.25, 1e-12));
});

test('矩与统计量：采样矩 vs 理论矩（sampleMoments/distSummary 闭环）', () => {
  // 均匀样本：偏度 0、超额峰度 −1.2
  const u = call('prob.uniform.sample', [21, 20000, 0, 1]);
  const mu = call('prob.util.sampleMoments', [u]);
  assert.ok(Math.abs(mu.skewness) <= 0.05, `uniform 偏度 ${mu.skewness}`);
  assert.ok(Math.abs(mu.kurtosis + 1.2) <= 0.06, `uniform 峰度 ${mu.kurtosis}`);
  // 指数样本：偏度 ≈ 2、峰度 ≈ 6（峰度收敛慢，用 20 万样本）
  const e = call('prob.exponential.sample', [22, 200000, 1]);
  const me = call('prob.util.sampleMoments', [e]);
  assert.ok(Math.abs(me.skewness - 2) <= 0.05 * 2, `exponential 偏度 ${me.skewness}`);
  assert.ok(Math.abs(me.kurtosis - 6) <= 0.05 * 6, `exponential 峰度 ${me.kurtosis}`);
  // distSummary 与采样矩闭环（normal/gamma）
  const ns = call('prob.normal.sample', [23, 20000, 2, 3]);
  const sm = call('prob.util.distSummary', ['normal', 2, 3]);
  const nm = call('prob.util.sampleMoments', [ns]);
  assert.ok(Math.abs(nm.mean - sm.mean) <= 0.05 * 3 && Math.abs(nm.variance - sm.variance) <= 0.05 * 9,
    `normal 采样矩 [${nm.mean},${nm.variance}] vs 摘要 [${sm.mean},${sm.variance}]`);
  assert.ok(approx(sm.entropy, 0.5 * Math.log(2 * Math.PI * Math.E * 9), 1e-12));
});

test('distSummary：各分布均值/方差/熵与闭式一致', () => {
  const G = (a) => Math.exp(lgammaTest(a));
  const cases = [
    ['normal', [2, 3], 2, 9, 0.5 * Math.log(2 * Math.PI * Math.E * 9)],
    ['uniform', [1, 4], 2.5, 0.75, Math.log(3)],
    ['exponential', [2], 0.5, 0.25, 1 - Math.log(2)],
    ['laplace', [0, 2], 0, 8, 1 + Math.log(4)],
    ['logistic', [0, 3], 0, Math.PI ** 2 * 9 / 3, 2 + Math.log(3)],
    ['gamma', [2, 3], 2 / 3, 2 / 9, 2 - Math.log(3) - (1 - 0.5772156649015329)],
    ['beta', [0.5, 0.5], 0.5, 0.125, Math.log(Math.PI / 4)],
    ['pareto', [1, 5], 1.25, 5 / 48, Math.log(0.2) + 1.2],
    ['rayleigh', [2], 2 * Math.sqrt(Math.PI / 2), (4 - Math.PI) / 2 * 4, 1 + Math.log(2 / Math.SQRT2) + 0.5772156649015329 / 2],
    ['weibull', [2, 3], 3 * G(1.5), 9 * (1 - G(1.5) ** 2), 0.5772156649015329 * 0.5 + Math.log(1.5) + 1],
    ['chi2', [6], 6, 12, Math.log(2 * G(3)) - 2 * (1.5 - 0.5772156649015329) + 3],
  ];
  for (const [dist, params, mean, variance, entropy] of cases) {
    const r = call('prob.util.distSummary', [dist, ...params]);
    assert.ok(approx(r.mean, mean, 1e-10), `${dist} 均值 ${r.mean} vs ${mean}`);
    assert.ok(approx(r.variance, variance, 1e-10), `${dist} 方差 ${r.variance} vs ${variance}`);
    assert.ok(approx(r.entropy, entropy, 1e-10), `${dist} 熵 ${r.entropy} vs ${entropy}`);
  }
});

// ============================================================================
test('贝叶斯：update / evidence / posteriorPredictive', () => {
  const prior = T1([1, 1]); // 未归一化先验
  const like = T1([1, 3]);
  const post = call('prob.bayes.update', [prior, like]);
  assert.deepEqual(Array.from(post.data), [0.25, 0.75]);
  assert.ok(approx(call('prob.bayes.evidence', [prior, like]), 4, 1e-12));
  // 后验预测：posterior [0.5,0.5]，L = [[0.9,0.1],[0.2,0.8]] ⇒ [0.55,0.45]
  const pp = call('prob.bayes.posteriorPredictive', [T1([0.5, 0.5]), Tensor.fromNested([[0.9, 0.1], [0.2, 0.8]])]);
  assert.ok(approx(pp.at(0), 0.55, 1e-12) && approx(pp.at(1), 0.45, 1e-12));
  // update 后再预测 = evidence 加权
  const post2 = call('prob.bayes.update', [prior, like]);
  const pp2 = call('prob.bayes.posteriorPredictive', [post2, Tensor.fromNested([[0.9, 0.1], [0.2, 0.8]])]);
  assert.ok(approx(pp2.at(0), 0.9 * 0.25 + 0.2 * 0.75, 1e-12));
});

test('贝叶斯：conjugateNormal / mapNormal / credibleInterval 与闭式解一致（1e-9）', () => {
  const n = 10, xbar = 1.5, mu0 = 0, s0 = 2, s = 1;
  const prec = 1 / (s0 * s0) + n / (s * s);
  const muN = (mu0 / (s0 * s0) + n * xbar / (s * s)) / prec;
  const sN = 1 / Math.sqrt(prec);
  const r = call('prob.bayes.conjugateNormal', [n, xbar, mu0, s0, s]);
  assert.ok(approx(r.at(0), muN, 1e-12) && approx(r.at(1), sN, 1e-12), `conjugateNormal = ${r.toArray()}`);
  // mapNormal 走数据：10 个均值恰为 1.5 的样本
  const data = T1(new Array(10).fill(1.5));
  assert.ok(approx(call('prob.bayes.mapNormal', [data, mu0, s0, s]), muN, 1e-12));
  // 无信息先验极限：σ0→∞ 时 μ̂ → x̄
  assert.ok(approx(call('prob.bayes.mapNormal', [data, 100, 1e8, 1]), 1.5, 1e-6));
  // 可信区间
  const ci = call('prob.bayes.credibleInterval', [0, 1, 0.95]);
  const z95 = 1.959963984540054;
  assert.ok(approx(ci.at(0), -z95, 1e-9) && approx(ci.at(1), z95, 1e-9));
  const ci90 = call('prob.bayes.credibleInterval', [10, 2, 0.9]);
  assert.ok(approx(ci90.at(0), 10 - 1.6448536269514722 * 2, 1e-9));
  assert.ok(approx(ci90.at(1), 10 + 1.6448536269514722 * 2, 1e-9));
  assert.ok(approx(call('prob.bayes.credibleInterval', [0, 1]).at(1), z95, 1e-9)); // 默认 0.95
});

test('贝叶斯：mle 四兄弟', () => {
  const nrm = call('prob.bayes.mleNormal', [T1([1, 2, 3])]);
  assert.ok(approx(nrm.at(0), 2, 1e-12) && approx(nrm.at(1), Math.sqrt(2 / 3), 1e-12));
  assert.ok(approx(call('prob.bayes.mleBernoulli', [T1([1, 0, 1, 1])]), 0.75, 1e-12));
  assert.ok(approx(call('prob.bayes.mlePoisson', [T1([2, 4, 6])]), 4, 1e-12));
  assert.ok(approx(call('prob.bayes.mleExponential', [T1([1, 2, 3])]), 0.5, 1e-12));
  // fit 与 mle 一致
  const data = call('prob.normal.sample', [31, 5000, 1.2, 2.5]);
  const fit = call('prob.normal.fit', [data]);
  const mle = call('prob.bayes.mleNormal', [data]);
  assert.deepEqual(Array.from(fit.data), Array.from(mle.data));
});

test('分布 fit：normal / uniform / exponential / gamma（容差 5%）', () => {
  const n = call('prob.normal.fit', [call('prob.normal.sample', [41, 20000, 1.2, 2.5])]);
  assert.ok(Math.abs(n.at(0) - 1.2) <= 0.05 * 2.5 && Math.abs(n.at(1) - 2.5) <= 0.05 * 2.5, `normal.fit = ${n.toArray()}`);
  const u = call('prob.uniform.fit', [call('prob.uniform.sample', [42, 20000, -1, 3])]);
  assert.ok(Math.abs(u.at(0) + 1) <= 0.05 * 4 && Math.abs(u.at(1) - 3) <= 0.05 * 4, `uniform.fit = ${u.toArray()}`);
  const ex = call('prob.exponential.fit', [call('prob.exponential.sample', [43, 20000, 2])]);
  assert.ok(Math.abs(ex.at(0) - 2) <= 0.05 * 2, `exponential.fit = ${ex.toArray()}`);
  const gm = call('prob.gamma.fit', [call('prob.gamma.sample', [44, 20000, 2.5, 1.5])]);
  assert.ok(Math.abs(gm.at(0) - 2.5) <= 0.05 * 2.5 && Math.abs(gm.at(1) - 1.5) <= 0.05 * 1.5, `gamma.fit = ${gm.toArray()}`);
  const gm2 = call('prob.gamma.fit', [call('prob.gamma.sample', [45, 20000, 0.7, 2])]);
  assert.ok(Math.abs(gm2.at(0) - 0.7) <= 0.05 * 0.7 && Math.abs(gm2.at(1) - 2) <= 0.05 * 2, `gamma.fit(0.7,2) = ${gm2.toArray()}`);
});

// ============================================================================
test('采样：boxMuller / marsagliaPolar 可复现且矩正确', () => {
  for (const name of ['prob.samp.boxMuller', 'prob.samp.marsagliaPolar']) {
    const s = call(name, [5, 20000]);
    checkMoments(Array.from(s.data), 0, 1, 0.05, name);
    const a = call(name, [5, 500]);
    const b = call(name, [5, 500]);
    assert.deepEqual(Array.from(a.data), Array.from(b.data), `${name} 不可复现`);
    const c = call(name, [6, 500]);
    assert.notDeepEqual(Array.from(a.data), Array.from(c.data), `${name} 不同种子应不同`);
  }
});

test('采样：rejection（Laplace 包络，M = √(2e/π)）矩正确', () => {
  const s = call('prob.samp.rejection', [11, 20000]);
  checkMoments(Array.from(s.data), 0, 1, 0.05, 'rejection');
  const a = call('prob.samp.rejection', [11, 500]);
  const b = call('prob.samp.rejection', [11, 500]);
  assert.deepEqual(Array.from(a.data), Array.from(b.data));
});

test('采样：metropolis 一维随机游走（容差 10%）', () => {
  const s = call('prob.samp.metropolis', [7, 20000, 2.4]);
  checkMoments(Array.from(s.data), 0, 1, 0.10, 'metropolis');
  // 步长鲁棒性：0.5 与 4.0 也能收敛
  checkMoments(Array.from(call('prob.samp.metropolis', [8, 20000, 0.5]).data), 0, 1, 0.10, 'metropolis(0.5)');
  checkMoments(Array.from(call('prob.samp.metropolis', [9, 20000, 4.0]).data), 0, 1, 0.10, 'metropolis(4.0)');
  const a = call('prob.samp.metropolis', [7, 500, 2.4]);
  const b = call('prob.samp.metropolis', [7, 500, 2.4]);
  assert.deepEqual(Array.from(a.data), Array.from(b.data));
});

test('采样：stratified 分层抽样（每层一个、均值方差 1/(12n²)）', () => {
  const n = 100;
  const s = call('prob.samp.stratified', [5, n]);
  const xs = Array.from(s.data).sort((a, b) => a - b);
  for (let i = 0; i < n; i++) {
    assert.ok(xs[i] >= i / n - 1e-12 && xs[i] < (i + 1) / n + 1e-12, `第 ${i} 层样本越界: ${xs[i]}`);
  }
  // 200 次独立运行的均值估计量方差 ≈ 1/(12n³) = 8.333e-8（粗蒙特卡洛 1/(12n) 的 1/n²）
  const means = [];
  for (let r = 0; r < 200; r++) {
    const t = call('prob.samp.stratified', [1000 + r, n]);
    let m = 0;
    for (const v of t.data) m += v;
    means.push(m / n);
  }
  const mm = means.reduce((a, b) => a + b, 0) / means.length;
  const mv = means.reduce((a, b) => a + (b - mm) ** 2, 0) / (means.length - 1);
  assert.ok(Math.abs(mm - 0.5) <= 0.01, `分层均值 ${mm}`);
  const expectVar = 1 / (12 * n ** 3);
  assert.ok(Math.abs(mv - expectVar) <= 0.25 * expectVar, `估计量方差 ${mv} vs ${expectVar}`);
  // 与粗蒙特卡洛对比：方差比 ≈ n² = 10⁴
  const crude = [];
  const rng = createRng(999);
  for (let r = 0; r < 200; r++) {
    let m = 0;
    for (let i = 0; i < n; i++) m += rng.next();
    crude.push(m / n);
  }
  const cm = crude.reduce((a, b) => a + b, 0) / crude.length;
  const cv = crude.reduce((a, b) => a + (b - cm) ** 2, 0) / (crude.length - 1);
  const ratio = cv / mv;
  assert.ok(ratio > (n * n) / 4 && ratio < (n * n) * 4, `方差比 ${ratio} vs n²=${n * n}`);
});

test('采样：inverseCdf 通用逆变换（正态断点）', () => {
  const xs = Tensor.linspace(-8, 8, 1601);
  const ys = call('prob.normal.cdf', [xs, 0, 1]);
  const s = call('prob.samp.inverseCdf', [42, 20000, xs, ys]);
  checkMoments(Array.from(s.data), 0, 1, 0.05, 'inverseCdf');
  // 分位数点核对：样本 0.8413 分位 ≈ 1
  const q = call('prob.stat.quantile', [s, 0.8413447461]);
  assert.ok(Math.abs(q - 1) <= 0.03, `inverseCdf 84.13% 分位 ${q} vs 1`);
  const a = call('prob.samp.inverseCdf', [42, 500, xs, ys]);
  const b = call('prob.samp.inverseCdf', [42, 500, xs, ys]);
  assert.deepEqual(Array.from(a.data), Array.from(b.data));
});

// ============================================================================
test('拟合检验：KS 一样本（同分布 p>0.01，平移 3σ 的 D≈0.866 且 p<1e-6）', () => {
  const s1 = call('prob.normal.sample', [101, 20000, 0, 1]);
  const r1 = call('prob.test.ks', [s1, 'normal', 0, 1]);
  assert.ok(r1.D >= 0 && r1.D <= 1);
  assert.ok(r1.pValue > 0.01, `同分布 KS p = ${r1.pValue}（D = ${r1.D}）`);
  // 均值平移 3σ：理论上 sup|Φ(x−3)−Φ(x)| = Φ(1.5)−Φ(−1.5) ≈ 0.8664
  const s2 = call('prob.normal.sample', [202, 20000, 3, 1]);
  const r2 = call('prob.test.ks', [s2, 'normal', 0, 1]);
  assert.ok(r2.D > 0.8 && r2.D < 0.93, `平移样本 D = ${r2.D}（理论 0.8664）`);
  assert.ok(r2.pValue < 1e-6, `平移样本 p = ${r2.pValue}`);
  // 均匀与指数参考
  const ru = call('prob.test.ks', [call('prob.uniform.sample', [303, 20000, 2, 5]), 'uniform', 2, 5]);
  assert.ok(ru.pValue > 0.01, `uniform KS p = ${ru.pValue}`);
  const re = call('prob.test.ks', [call('prob.exponential.sample', [404, 20000, 1.5]), 'exponential', 1.5]);
  assert.ok(re.pValue > 0.01, `exponential KS p = ${re.pValue}`);
  // 错配：指数样本按正态检验
  const rm = call('prob.test.ks', [call('prob.exponential.sample', [505, 20000, 1.5]), 'normal', 1, 1]);
  assert.ok(rm.pValue < 0.01, `错配 KS p = ${rm.pValue}`);
});

test('拟合检验：KS 两样本', () => {
  const x = call('prob.normal.sample', [11, 20000, 0, 1]);
  const y = call('prob.normal.sample', [22, 20000, 0, 1]);
  const r1 = call('prob.test.ks2Sample', [x, y]);
  assert.ok(r1.pValue > 0.01, `同分布两样本 p = ${r1.pValue}（D = ${r1.D}）`);
  const z = call('prob.normal.sample', [33, 20000, 3, 1]);
  const r2 = call('prob.test.ks2Sample', [x, z]);
  assert.ok(r2.D > 0.8, `平移两样本 D = ${r2.D}`);
  assert.ok(r2.pValue < 1e-6, `平移两样本 p = ${r2.pValue}`);
  // 手工：完全分离的两个样本 D = 1（渐近 p 值在小样本下偏保守，取 p<0.01）
  const r3 = call('prob.test.ks2Sample', [T1(Array.from({ length: 10 }, (_, i) => i + 1)), T1(Array.from({ length: 10 }, (_, i) => i + 21))]);
  assert.ok(approx(r3.D, 1, 1e-12));
  assert.ok(r3.pValue < 0.01, `分离样本 p = ${r3.pValue}`);
});

test('拟合检验：chi2Gof（统计量与 p 值）', () => {
  // observed [40,30,30] vs 均匀 1/3：E=33.33，χ² = (6.67²+3.33²+3.33²)/33.33 = 2，p = e^{−1}
  const r = call('prob.test.chi2Gof', [T1([40, 30, 30]), T1([1 / 3, 1 / 3, 1 / 3])]);
  assert.ok(approx(r.statistic, 2, 1e-9), `χ² = ${r.statistic}`);
  assert.equal(r.df, 2);
  assert.ok(approx(r.pValue, Math.exp(-1), 1e-9), `p = ${r.pValue}`);
  // 均匀骰子 600 次投掷（固定种子）→ p > 0.01
  const rng = createRng(606);
  const counts = new Array(6).fill(0);
  for (let i = 0; i < 600; i++) counts[Math.floor(rng.next() * 6)]++;
  const rd = call('prob.test.chi2Gof', [T1(counts), T1(new Array(6).fill(1 / 6))]);
  assert.ok(rd.pValue > 0.01, `公平骰子 p = ${rd.pValue}（χ² = ${rd.statistic}）`);
  // 偏载骰子 → p 极小
  const rl = call('prob.test.chi2Gof', [T1([200, 10, 10, 180, 100, 100]), T1(new Array(6).fill(1 / 6))]);
  assert.ok(rl.statistic > 300 && rl.pValue < 1e-60, `偏载 χ² = ${rl.statistic}, p = ${rl.pValue}`);
});

test('拟合检验：jarqueBera（正态样本 p>0.01，偏态样本 p<0.01）', () => {
  // 手工：x = [−2,−1,0,1,2] ⇒ S=0, K=−1.3, JB = (5/6)(1.69/4) ≈ 0.3521, p = e^{−JB/2}
  const h = call('prob.test.jarqueBera', [T1([-2, -1, 0, 1, 2])]);
  assert.ok(approx(h.statistic, (5 / 6) * (1.69 / 4), 1e-12), `JB = ${h.statistic}`);
  assert.ok(approx(h.pValue, Math.exp(-h.statistic / 2), 1e-12));
  const n1 = call('prob.test.jarqueBera', [call('prob.normal.sample', [71, 20000, 0, 1])]);
  assert.ok(n1.pValue > 0.01, `正态样本 JB p = ${n1.pValue}（JB = ${n1.statistic}）`);
  const e1 = call('prob.test.jarqueBera', [call('prob.exponential.sample', [72, 20000, 1])]);
  assert.ok(e1.pValue < 0.01 && e1.statistic > 100, `指数样本 JB = ${e1.statistic}, p = ${e1.pValue}`);
});

// ============================================================================
test('工具：trapz 梯形积分', () => {
  // 常数函数精确积分
  const y1 = T1(new Array(1001).fill(1));
  assert.ok(approx(call('prob.util.trapz', [y1, Tensor.linspace(0, 1, 1001)]), 1, 1e-12));
  // 单位间距
  assert.ok(approx(call('prob.util.trapz', [y1]), 1000, 1e-12));
  // x² over [0,1] = 1/3
  const xs = Tensor.linspace(0, 1, 100001);
  const ys = xs.map((v) => v * v);
  assert.ok(approx(call('prob.util.trapz', [ys, xs]), 1 / 3, 1e-9));
  // 非均匀网格：y = x，xs = [0,1,3] ⇒ ∫₀³ x dx = 4.5
  assert.ok(approx(call('prob.util.trapz', [T1([0, 1, 3]), T1([0, 1, 3])]), 4.5, 1e-12));
});

test('工具：normalQuantile 逆正态（Φ⁻¹(Φ(x)) ≈ x）', () => {
  assert.ok(approx(call('prob.util.normalQuantile', [0.5]), 0, 1e-12));
  const pts = [0.001, 0.01, 0.1, 0.25, 1 / 3, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999];
  for (const p of pts) {
    const x = call('prob.util.normalQuantile', [p]);
    const back = call('prob.normal.cdf', [x, 0, 1]);
    assert.ok(Math.abs(back - p) <= 1e-12, `normalQuantile(${p}) = ${x}，Φ 回去 = ${back}`);
  }
  // 与已知分位点核对
  assert.ok(approx(call('prob.util.normalQuantile', [0.975]), 1.959963984540054, 1e-10));
  assert.ok(approx(call('prob.util.normalQuantile', [0.995]), 2.5758293035489004, 1e-10));
  assert.ok(approx(call('prob.util.normalQuantile', [0.9]), 1.2815515655446004, 1e-10));
  // 单调性
  let prev = -Infinity;
  for (let p = 0.001; p < 1; p += 0.01) {
    const x = call('prob.util.normalQuantile', [p]);
    assert.ok(x > prev, 'normalQuantile 应单调递增');
    prev = x;
  }
  // Φ 自身精度：与已知值核对
  assert.ok(approx(call('prob.normal.cdf', [1.96, 0, 1]), 0.9750021048517795, 1e-12));
  assert.ok(approx(call('prob.normal.cdf', [-1, 0, 1]), 0.15865525393145707, 1e-12));
});

// ============================================================================
test('错误处理：参数校验与错误码', () => {
  const bad = (name, args, code) => {
    const r = tryExec(name, args);
    assert.equal(r.ok, false, `${name}(${args}) 应抛错`);
    assert.equal(r.error.code, code, `${name}(${args}) 错误码 ${r.error.code} != ${code}`);
  };
  bad('prob.normal.pdf', [0, 0, -1], 'E_VALUE'); // σ ≤ 0
  bad('prob.normal.pdf', [0, 0, 0], 'E_VALUE');
  bad('prob.normal.pdf', ['x', 0, 1], 'E_TYPE'); // x 类型
  bad('prob.normal.pdf', [0, 1], 'E_ARITY'); // 参数个数
  bad('prob.exponential.pdf', [-1, -2], 'E_VALUE');
  bad('prob.beta.pdf', [0.5, -1, 2], 'E_VALUE');
  bad('prob.beta.pdf', [0.5, 1, 0], 'E_VALUE');
  bad('prob.bernoulli.pmf', [1, 1.5], 'E_VALUE'); // p 越界
  bad('prob.poisson.pmf', [1, 0], 'E_VALUE'); // λ ≤ 0
  bad('prob.categorical.sample', [1, 5, T1([-0.1, 1.1])], 'E_VALUE'); // 负概率
  bad('prob.stat.quantile', [T1([1, 2, 3]), 1.5], 'E_VALUE'); // q 越界
  bad('prob.stat.corr', [T1([1, 1, 1]), T1([1, 2, 3])], 'E_NUMERICAL'); // 常量序列
  bad('prob.stat.cov', [T1([1, 2]), T1([1, 2, 3])], 'E_VALUE'); // 长度不符
  bad('prob.util.normalQuantile', [0], 'E_VALUE');
  bad('prob.util.normalQuantile', [1.5], 'E_VALUE');
  bad('prob.test.ks', [T1([1, 2, 3]), 'cauchy', 0, 1], 'E_VALUE'); // 不支持的参考分布
  bad('prob.util.distSummary', ['cauchy', 0, 1], 'E_VALUE');
  bad('prob.util.distSummary', ['nope'], 'E_VALUE');
  bad('prob.gamma.fit', [T1([1, 0, 2])], 'E_VALUE'); // 非正数据
  bad('prob.gamma.fit', [T1([1, 1, 1])], 'E_VALUE'); // 常数数据
  bad('prob.mvn.pdf', [T1([1, 2]), T1([0, 0]), Tensor.fromNested([[1, 2], [2, 1]])], 'E_NUMERICAL'); // 非正定
  bad('prob.mvn.pdf', [T1([1, 2]), T1([0]), Tensor.fromNested([[1]])], 'E_SHAPE'); // 维度不符
  bad('prob.bayes.mleBernoulli', [T1([0.5, 2])], 'E_VALUE');
  bad('prob.bayes.mleExponential', [T1([0, 0])], 'E_VALUE');
  bad('prob.info.entropy', [T1([-0.1, 1.1])], 'E_VALUE');
  bad('prob.util.trapz', [T1([1, 2]), T1([1])], 'E_SHAPE');
  bad('prob.samp.inverseCdf', [1, 10, T1([3, 2, 1]), T1([0.1, 0.5, 1])], 'E_VALUE'); // xs 非递增
  bad('prob.samp.metropolis', [1, 10, -1], 'E_VALUE');
  // 未知指令
  const r = tryExec('prob.nope', []);
  assert.equal(r.ok, false);
  assert.equal(r.error.code, 'E_NOTFOUND');
  // 合法边界不抛错
  assert.equal(call('prob.exponential.pdf', [-1, 2]), 0);
  assert.equal(call('prob.poisson.pmf', [2.5, 4]), 0);
  assert.equal(call('prob.beta.pdf', [0, 2, 3]), 0); // a>1 时 x=0 处密度为 0
  assert.ok(approx(call('prob.beta.pdf', [0, 1, 2]), 2, 1e-12)); // a=1 时 pdf(0) = 1/B(1,2) = 2
});

// __TEST_APPEND__
