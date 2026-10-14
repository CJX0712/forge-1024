// ============================================================================
// 域6 · 概率统计 PROB —— 102 条微指令（D6-001 … D6-102）
//
// 覆盖：
//   连续分布 43：normal / uniform / exponential / laplace / cauchy / logistic /
//                gamma(shape,rate) / beta / chi2 / studentT / rayleigh /
//                weibull / pareto 各含 pdf+cdf+sample（39），外加 normal /
//                uniform / exponential / gamma 的 fit（4）；
//   离散分布 15：bernoulli / binomial / poisson / geometric / categorical 各 pmf+cdf+sample；
//   多元正态  2：mvn.pdf / mvn.sample；
//   信息论    8：entropy / crossEntropy / klDiv / jsDiv / mutualInfo /
//                wasserstein1 / hellinger / tvd；
//   矩与统计量 10：mean / variance / std / skewness / kurtosis / cov / corr /
//                quantile / median / mad；
//   贝叶斯与估计 10：update / posteriorPredictive / evidence / mapNormal /
//                mleNormal / mleBernoulli / mlePoisson / mleExponential /
//                credibleInterval / conjugateNormal；
//   采样方法  6：inverseCdf / rejection / metropolis / boxMuller /
//                marsagliaPolar / stratified；
//   拟合检验  4：ks / ks2Sample / chi2Gof / jarqueBera；
//   工具      4：trapz / normalQuantile / distSummary / sampleMoments。
//
// 约定：
//   * 随机数一律来自 core/rng.mjs 的 createRng（PCG-XSL-RR 32bit，同种子完全可复现），
//     禁止 Math.random；
//   * 参数校验抛 ForgeError（E_TYPE / E_VALUE / E_SHAPE / E_NUMERICAL）；
//   * 分布参数命名沿用 Johnson, Kotz & Balakrishnan《Continuous Univariate
//     Distributions》惯例：gamma(shape, rate)、weibull(shape, scale) 等；
//   * 统计量（variance/skewness/kurtosis/cov）采用总体矩口径（÷n，即 MLE 口径）。
// ============================================================================
import { defineOp, opsByDomain } from '../core/registry.mjs';
import { Tensor } from '../core/tensor.mjs';
import { createRng } from '../core/rng.mjs';
import { ForgeError } from '../core/errors.mjs';

let _s = 0;
const nid = () => `D6-${String(++_s).padStart(3, '0')}`;

// ============================================================================
// 一、内部特殊函数（非微指令，仅供本域实现使用；关键公式见注释）
// ============================================================================

const SQRT_2PI = Math.sqrt(2 * Math.PI);
const EULER_GAMMA = 0.5772156649015328606; // 欧拉-马歇罗尼常数 γ

// Lanczos 逼近（g=7, n=9），Γ(x) 相对误差 ~1e-15。
// 参考：Lanczos, "A Precision Approximation of the Gamma Function", 1964。
const LANCZOS_G = 7.5;
const LANCZOS_C = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];
function lgamma(x) {
  if (x <= 0) {
    // 反射公式 Γ(x)Γ(1−x) = π / sin(πx)（x 非整数）
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  }
  const z = x - 1;
  let a = LANCZOS_C[0];
  const t = z + LANCZOS_G;
  for (let i = 1; i < LANCZOS_C.length; i++) a += LANCZOS_C[i] / (z + i);
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

// Digamma ψ(x) = d/dx ln Γ(x)：递推 ψ(x) = ψ(x+1) − 1/x 提升到 x≥10 后用渐近展开
// ψ(x) ≈ ln x − 1/(2x) − 1/(12x²) + 1/(120x⁴) − 1/(252x⁶) + 1/(240x⁸)。
function digamma(x) {
  let r = 0;
  while (x < 10) { r -= 1 / x; x += 1; }
  const f = 1 / (x * x);
  return r + Math.log(x) - 0.5 / x + f * (-1 / 12 + f * (1 / 120 + f * (-1 / 252 + f / 240)));
}

// Trigamma ψ'(x)：递推 ψ'(x) = ψ'(x+1) + 1/x² 提升到 x≥8 后用渐近展开
// ψ'(x) ≈ 1/x + 1/(2x²) + 1/(6x³) − 1/(30x⁵) + 1/(42x⁷)。
function trigamma(x) {
  let r = 0;
  while (x < 8) { r += 1 / (x * x); x += 1; }
  const f = 1 / (x * x);
  return r + 1 / x + 0.5 * f + f * f * (1 / 6 + f * (-1 / 30 + f / 42));
}

// 正则化不完全伽马 P(a,x)（级数，x < a+1）与 Q(a,x)（连分式，x ≥ a+1）。
// 参考：Press et al., Numerical Recipes in C, §6.2（gser / gcf）。
function gammaSeriesP(a, x) {
  if (x <= 0) return 0;
  let ap = a;
  let sum = 1 / a;
  let del = sum;
  for (let n = 1; n <= 10000; n++) {
    ap += 1;
    del *= x / ap;
    sum += del;
    if (Math.abs(del) < Math.abs(sum) * 1e-16) break;
  }
  return sum * Math.exp(-x + a * Math.log(x) - lgamma(a));
}
function gammaCFQ(a, x) {
  const FPMIN = 1e-300;
  const b0 = x + 1 - a;
  let c = 1 / FPMIN;
  let d = 1 / b0;
  let h = d;
  for (let i = 1; i <= 10000; i++) {
    const an = -i * (i - a);
    let b = b0 + 2 * i;
    d = an * d + b;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = b + an / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-16) break;
  }
  return Math.exp(-x + a * Math.log(x) - lgamma(a)) * h;
}
// 正则化下不完全伽马 P(a,x) = γ(a,x)/Γ(a)
function gammap(a, x) {
  if (x < 0) throw new ForgeError('E_VALUE', `gammap: x 必须 ≥ 0，得到 ${x}`);
  if (a <= 0) throw new ForgeError('E_VALUE', `gammap: a 必须 > 0，得到 ${a}`);
  return x < a + 1 ? gammaSeriesP(a, x) : 1 - gammaCFQ(a, x);
}
// 正则化上不完全伽马 Q(a,x) = Γ(a,x)/Γ(a)
function gammaq(a, x) {
  if (x < 0) throw new ForgeError('E_VALUE', `gammaq: x 必须 ≥ 0，得到 ${x}`);
  if (a <= 0) throw new ForgeError('E_VALUE', `gammaq: a 必须 > 0，得到 ${a}`);
  return x < a + 1 ? 1 - gammaSeriesP(a, x) : gammaCFQ(a, x);
}

// 误差函数：erf(x) = P(1/2, x²)（x≥0）；erfc(x) = Q(1/2, x²)。
function erf(x) { return x >= 0 ? gammap(0.5, x * x) : -gammap(0.5, x * x); }
function erfc(x) { return x >= 0 ? gammaq(0.5, x * x) : 2 - gammaq(0.5, x * x); }
// 标准正态 CDF：Φ(x) = ½ erfc(−x/√2)
function normCdf(x) { return 0.5 * erfc(-x / Math.SQRT2); }

// 正则化不完全 beta I_x(a,b)（连分式 + 对称反射）。
// 参考：Press et al., Numerical Recipes in C, §6.4（betacf）。
function betaCF(a, b, x) {
  const FPMIN = 1e-300;
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1;
  let d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 10000; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-16) break;
  }
  return h;
}
function betai(a, b, x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b); // ln B(a,b)
  // bt = x^a (1−x)^b / B(a,b)
  const bt = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lbeta);
  // 对称反射保证连分式收敛速度
  if (x < (a + 1) / (a + b + 2)) return bt * betaCF(a, b, x) / a;
  return 1 - bt * betaCF(b, a, 1 - x) / b;
}

// 逆正态分位函数：Acklam 有理逼近 + 一步 Halley 修正（用精确 Φ 反代）。
// 参考：P.J. Acklam（Beasley-Springer-Moro 族算法）；相对误差 < 1.15e-9，Halley 后 ~1e-15。
function normalQuantileRaw(p) {
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
    1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
    6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
    -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const pLow = 0.02425;
  let x;
  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= 1 - pLow) {
    const q = p - 0.5, r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  // Halley 修正：e = Φ(x) − p，u = e·√(2π)·exp(x²/2)，x ← x − u/(1 + x·u/2)
  const e = normCdf(x) - p;
  const u = e * SQRT_2PI * Math.exp(0.5 * x * x);
  return x - u / (1 + 0.5 * x * u);
}

// 下三角 Cholesky 分解 Σ = L·Lᵀ（要求正定）。
function choleskyLower(S, d) {
  const L = new Float64Array(d * d);
  for (let i = 0; i < d; i++) {
    for (let j = 0; j <= i; j++) {
      let s = S[i * d + j];
      for (let k = 0; k < j; k++) s -= L[i * d + k] * L[j * d + k];
      if (i === j) {
        if (!(s > 0)) throw new ForgeError('E_NUMERICAL', '协方差矩阵非正定，Cholesky 失败');
        L[i * d + j] = Math.sqrt(s);
      } else {
        L[i * d + j] = s / L[j * d + j];
      }
    }
  }
  return L;
}

// ============================================================================
// 二、输入校验助手
// ============================================================================

function isNum(v) { return typeof v === 'number' && Number.isFinite(v); }

function num(v, who) {
  if (!isNum(v)) {
    throw new ForgeError('E_TYPE', `${who}: 期望有限 number，得到 ${v === null ? 'null' : typeof v === 'number' ? String(v) : typeof v}`);
  }
  return v;
}
function positive(v, who) {
  num(v, who);
  if (v <= 0) throw new ForgeError('E_VALUE', `${who}: 必须 > 0，得到 ${v}`);
  return v;
}
function nonNeg(v, who) {
  num(v, who);
  if (v < 0) throw new ForgeError('E_VALUE', `${who}: 必须 ≥ 0，得到 ${v}`);
  return v;
}
function unitInterval(v, who) {
  num(v, who);
  if (v < 0 || v > 1) throw new ForgeError('E_VALUE', `${who}: 必须在 [0,1]，得到 ${v}`);
  return v;
}

// 取一维数据张量的 Float64Array 视图（拒绝空张量/非有限值）。
function dataOf(t, who = 'data') {
  if (!(t instanceof Tensor)) {
    throw new ForgeError('E_TYPE', `${who}: 期望 Tensor，得到 ${t === null ? 'null' : typeof t}`);
  }
  if (t.size === 0) throw new ForgeError('E_VALUE', `${who}: 不能为空`);
  const d = t.data;
  for (let i = 0; i < d.length; i++) {
    if (!Number.isFinite(d[i])) throw new ForgeError('E_VALUE', `${who}: 第 ${i} 项非有限（${d[i]}）`);
  }
  return d;
}

// 概率向量：非负、和 > 0；返回 {data, sum}（内部按需归一化）。
function probVecOf(t, who = 'p') {
  const d = dataOf(t, who);
  let s = 0;
  for (const v of d) {
    if (v < 0) throw new ForgeError('E_VALUE', `${who}: 概率不能为负（${v}）`);
    s += v;
  }
  if (!(s > 0)) throw new ForgeError('E_VALUE', `${who}: 概率和必须 > 0`);
  return { d, s };
}

// pdf/cdf 的取值点：number 或 Tensor。
function point(x, who = 'x') {
  if (x instanceof Tensor) return x;
  if (isNum(x)) return x;
  throw new ForgeError('E_TYPE', `${who}: 期望 number 或 Tensor，得到 ${x === null ? 'null' : typeof x}`);
}
// 逐点求值提升：number -> number，Tensor -> 同形状 Tensor。
function liftPoint(x, f) { return x instanceof Tensor ? x.map(f) : f(x); }

// ============================================================================
// 三、采样核心（全部基于 createRng，禁止 Math.random）
// ============================================================================

// Box-Muller 正态生成器（成对生成 + 本地缓存）。
// 参考：Box & Muller 1958；u1,u2 ~ U(0,1) ⇒ r = √(−2 ln u1)，z1 = r·cos(2πu2)，z2 = r·sin(2πu2)。
function normalGen(rng) {
  let cache = null;
  return () => {
    if (cache !== null) { const v = cache; cache = null; return v; }
    let u1 = rng.next();
    if (u1 < 1e-300) u1 = 1e-300;
    const u2 = rng.next();
    const r = Math.sqrt(-2 * Math.log(u1));
    cache = r * Math.sin(2 * Math.PI * u2);
    return r * Math.cos(2 * Math.PI * u2);
  };
}

// Gamma(shape, rate) 采样：Marsaglia-Tsang 拒绝法（shape ≥ 1），shape < 1 用 boost 技巧
// G(α) = G(α+1)·U^(1/α)。参考：Marsaglia & Tsang, ACM TOMS 26(2), 2000。
function gammaSample(rng, shape, rate) {
  if (!(shape > 0) || !(rate > 0)) throw new ForgeError('E_VALUE', 'gammaSample: shape 与 rate 必须 > 0');
  if (shape < 1) {
    let u = rng.next();
    if (u < 1e-300) u = 1e-300;
    return gammaSample(rng, shape + 1, rate) * Math.pow(u, 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  const g = normalGen(rng);
  for (let iter = 0; iter < 1000000; iter++) {
    let x = 0, v = 0;
    do { x = g(); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    let u = rng.next();
    if (u < 1e-300) u = 1e-300;
    // 接受条件：ln u < x²/2 + d − d·v + d·ln v
    if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v / rate;
  }
  throw new ForgeError('E_NUMERICAL', 'gammaSample: 拒绝采样未收敛');
}

// KS 检验 p 值：Kolmogorov 分布渐近式 Q(λ) = 2·Σ_{k≥1} (−1)^{k−1} e^{−2k²λ²}，
// λ = (√n + 0.12 + 0.11/√n)·D（Stephens 修正）。参考：Massey 1951；Stephens 1974。
function ksPValue(d, n) {
  const en = Math.sqrt(n);
  const lambda = (en + 0.12 + 0.11 / en) * d;
  if (lambda <= 0) return 1;
  let s = 0;
  for (let k = 1; k <= 200; k++) {
    const t = 2 * Math.exp(-2 * k * k * lambda * lambda);
    s += (k % 2 === 1 ? t : -t);
    if (t < 1e-17) break;
  }
  return Math.min(1, Math.max(0, s));
}

// 总体矩助手：返回 [m1, m2, m3, m4]（÷n 口径）。
function centralMoments(d) {
  const n = d.length;
  let m1 = 0;
  for (const v of d) m1 += v;
  m1 /= n;
  let m2 = 0, m3 = 0, m4 = 0;
  for (const v of d) {
    const z = v - m1;
    m2 += z * z;
    m3 += z * z * z;
    m4 += z * z * z * z;
  }
  return [m1, m2 / n, m3 / n, m4 / n];
}

// ============================================================================
// 四、连续分布（43 条：D6-001 … D6-043）
// ============================================================================

// —— 1. 正态分布 Normal(μ, σ²) ——
defineOp({
  id: nid(), domain: 6, name: 'prob.normal.pdf', arity: 3,
  sig: '(x: number|Tensor, mu: number, sigma: number) -> number|Tensor',
  doc: '正态密度：pdf(x) = exp(−(x−μ)²/(2σ²)) / (σ√(2π))',
  ref: 'Evans & Rosenthal, Probability and Statistics（正态分布）',
  inv: 'pdf ≥ 0，∫pdf = 1，关于 μ 对称；σ > 0',
  impl: (x, mu, sigma) => {
    point(x); num(mu, 'mu'); positive(sigma, 'sigma');
    return liftPoint(x, v => Math.exp(-0.5 * ((v - mu) / sigma) ** 2) / (sigma * SQRT_2PI));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.normal.cdf', arity: 3,
  sig: '(x: number|Tensor, mu: number, sigma: number) -> number|Tensor',
  doc: '正态 CDF：Φ((x−μ)/σ)，其中 Φ 为标准正态 CDF（经 erf 精确计算）',
  ref: 'Evans & Rosenthal, Probability and Statistics（正态分布）',
  inv: 'cdf ∈ [0,1] 单调不减；cdf(−∞)=0，cdf(+∞)=1',
  impl: (x, mu, sigma) => {
    point(x); num(mu, 'mu'); positive(sigma, 'sigma');
    return liftPoint(x, v => normCdf((v - mu) / sigma));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.normal.sample', arity: 4,
  sig: '(seed: number, n: number, mu: number, sigma: number) -> Tensor[n]',
  doc: '正态抽样：μ + σ·Z，Z ~ N(0,1) 由 Box-Muller 生成（种子化可复现）',
  ref: 'Box & Muller 1958（正态变量生成）',
  inv: '同种子完全可复现；大样本矩 → (μ, σ²)',
  impl: (seed, n, mu, sigma) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n'); num(mu, 'mu'); positive(sigma, 'sigma');
    const rng = createRng(seed);
    const g = normalGen(rng);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = mu + sigma * g();
    return new Tensor(out, [n]);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.normal.fit', arity: 1,
  sig: '(data: Tensor) -> Tensor[2]',
  doc: '正态 MLE 拟合：返回 [μ̂, σ̂]，μ̂ = x̄，σ̂ = √(Σ(x−x̄)²/n)',
  ref: 'Evans & Rosenthal, Probability and Statistics（极大似然估计）',
  inv: 'n ≥ 1；对正态数据 μ̂→μ、σ̂→σ',
  impl: (data) => {
    const d = dataOf(data);
    const [m1, m2] = centralMoments(d);
    return Tensor.fromFlat([m1, Math.sqrt(m2)], [2]);
  },
});

// —— 2. 连续均匀分布 Uniform(a, b) ——
defineOp({
  id: nid(), domain: 6, name: 'prob.uniform.pdf', arity: 3,
  sig: '(x: number|Tensor, a: number, b: number) -> number|Tensor',
  doc: '均匀密度：pdf(x) = 1/(b−a)（a ≤ x ≤ b），否则 0',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 2（uniform）',
  inv: 'pdf ≥ 0，∫pdf = 1；b > a',
  impl: (x, a, b) => {
    point(x); num(a, 'a'); num(b, 'b');
    if (!(b > a)) throw new ForgeError('E_VALUE', `uniform.pdf: 需要 b > a，得到 a=${a}, b=${b}`);
    return liftPoint(x, v => (v >= a && v <= b ? 1 / (b - a) : 0));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.uniform.cdf', arity: 3,
  sig: '(x: number|Tensor, a: number, b: number) -> number|Tensor',
  doc: '均匀 CDF：clip((x−a)/(b−a), 0, 1)',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 2（uniform）',
  inv: 'cdf ∈ [0,1] 单调不减；b > a',
  impl: (x, a, b) => {
    point(x); num(a, 'a'); num(b, 'b');
    if (!(b > a)) throw new ForgeError('E_VALUE', `uniform.cdf: 需要 b > a，得到 a=${a}, b=${b}`);
    return liftPoint(x, v => Math.min(1, Math.max(0, (v - a) / (b - a))));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.uniform.sample', arity: 4,
  sig: '(seed: number, n: number, a: number, b: number) -> Tensor[n]',
  doc: '均匀抽样：a + (b−a)·U，U ~ U(0,1) 来自 PCG 流',
  ref: 'Devroye 1986, Non-Uniform Random Variate Generation（逆变换法）',
  inv: '同种子可复现；大样本矩 → ((a+b)/2, (b−a)²/12)',
  impl: (seed, n, a, b) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n'); num(a, 'a'); num(b, 'b');
    if (!(b > a)) throw new ForgeError('E_VALUE', `uniform.sample: 需要 b > a`);
    const rng = createRng(seed);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = a + (b - a) * rng.next();
    return new Tensor(out, [n]);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.uniform.fit', arity: 1,
  sig: '(data: Tensor) -> Tensor[2]',
  doc: '均匀 MLE 拟合：返回 [min, max]',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 2',
  inv: '返回 â ≤ x_i ≤ b̂',
  impl: (data) => {
    const d = dataOf(data);
    let lo = Infinity, hi = -Infinity;
    for (const v of d) { if (v < lo) lo = v; if (v > hi) hi = v; }
    return Tensor.fromFlat([lo, hi], [2]);
  },
});

// —— 3. 指数分布 Exponential(rate λ) ——
defineOp({
  id: nid(), domain: 6, name: 'prob.exponential.pdf', arity: 2,
  sig: '(x: number|Tensor, rate: number) -> number|Tensor',
  doc: '指数密度：pdf(x) = λ·e^{−λx}（x ≥ 0），否则 0',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1（exponential）',
  inv: 'pdf ≥ 0，∫₀^∞ pdf = 1；rate > 0',
  impl: (x, rate) => {
    point(x); positive(rate, 'rate');
    return liftPoint(x, v => (v < 0 ? 0 : rate * Math.exp(-rate * v)));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.exponential.cdf', arity: 2,
  sig: '(x: number|Tensor, rate: number) -> number|Tensor',
  doc: '指数 CDF：1 − e^{−λx}（x ≥ 0），否则 0',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1',
  inv: 'cdf ∈ [0,1] 单调不减；rate > 0',
  impl: (x, rate) => {
    point(x); positive(rate, 'rate');
    return liftPoint(x, v => (v < 0 ? 0 : -Math.expm1(-rate * v)));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.exponential.sample', arity: 3,
  sig: '(seed: number, n: number, rate: number) -> Tensor[n]',
  doc: '指数抽样（逆变换）：x = −ln(U)/λ，U ~ U(0,1)',
  ref: 'Devroye 1986, Non-Uniform Random Variate Generation（逆变换法）',
  inv: '同种子可复现；大样本矩 → (1/λ, 1/λ²)',
  impl: (seed, n, rate) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n'); positive(rate, 'rate');
    const rng = createRng(seed);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let u = rng.next();
      if (u < 1e-300) u = 1e-300;
      out[i] = -Math.log(u) / rate;
    }
    return new Tensor(out, [n]);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.exponential.fit', arity: 1,
  sig: '(data: Tensor) -> Tensor[1]',
  doc: '指数 MLE 拟合：返回 [λ̂]，λ̂ = 1/x̄',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1',
  inv: '数据须 ≥ 0 且均值 > 0',
  impl: (data) => {
    const d = dataOf(data);
    let s = 0;
    for (const v of d) { nonNeg(v, 'data'); s += v; }
    const mean = s / d.length;
    if (!(mean > 0)) throw new ForgeError('E_VALUE', 'exponential.fit: 数据均值为 0，λ̂ 无定义');
    return Tensor.fromFlat([1 / mean], [1]);
  },
});

// —— 4. 拉普拉斯分布 Laplace(μ, b) ——
defineOp({
  id: nid(), domain: 6, name: 'prob.laplace.pdf', arity: 3,
  sig: '(x: number|Tensor, mu: number, b: number) -> number|Tensor',
  doc: '拉普拉斯密度：pdf(x) = exp(−|x−μ|/b) / (2b)',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1（Laplace）',
  inv: 'pdf ≥ 0，∫pdf = 1，关于 μ 对称；b > 0',
  impl: (x, mu, b) => {
    point(x); num(mu, 'mu'); positive(b, 'b');
    return liftPoint(x, v => Math.exp(-Math.abs(v - mu) / b) / (2 * b));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.laplace.cdf', arity: 3,
  sig: '(x: number|Tensor, mu: number, b: number) -> number|Tensor',
  doc: '拉普拉斯 CDF：x<μ 时 ½·exp((x−μ)/b)，x≥μ 时 1 − ½·exp(−(x−μ)/b)',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1',
  inv: 'cdf ∈ [0,1] 单调不减；b > 0',
  impl: (x, mu, b) => {
    point(x); num(mu, 'mu'); positive(b, 'b');
    return liftPoint(x, v => (v < mu ? 0.5 * Math.exp((v - mu) / b) : 1 - 0.5 * Math.exp(-(v - mu) / b)));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.laplace.sample', arity: 4,
  sig: '(seed: number, n: number, mu: number, b: number) -> Tensor[n]',
  doc: '拉普拉斯抽样（逆变换）：x = μ − b·sign(U−½)·ln(1−2|U−½|)',
  ref: 'Devroye 1986, Non-Uniform Random Variate Generation（逆变换法）',
  inv: '同种子可复现；大样本矩 → (μ, 2b²)',
  impl: (seed, n, mu, b) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n'); num(mu, 'mu'); positive(b, 'b');
    const rng = createRng(seed);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const u = rng.next() - 0.5;
      out[i] = mu - b * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
    }
    return new Tensor(out, [n]);
  },
});

// —— 5. 柯西分布 Cauchy(x0, γ) ——
defineOp({
  id: nid(), domain: 6, name: 'prob.cauchy.pdf', arity: 3,
  sig: '(x: number|Tensor, x0: number, gamma: number) -> number|Tensor',
  doc: '柯西密度：pdf(x) = 1/(πγ(1 + ((x−x0)/γ)²))',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1（Cauchy）',
  inv: 'pdf ≥ 0，∫pdf = 1；γ > 0；均值/方差不存在',
  impl: (x, x0, gamma) => {
    point(x); num(x0, 'x0'); positive(gamma, 'gamma');
    return liftPoint(x, v => 1 / (Math.PI * gamma * (1 + ((v - x0) / gamma) ** 2)));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.cauchy.cdf', arity: 3,
  sig: '(x: number|Tensor, x0: number, gamma: number) -> number|Tensor',
  doc: '柯西 CDF：½ + arctan((x−x0)/γ)/π',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1',
  inv: 'cdf ∈ [0,1] 单调不减；γ > 0',
  impl: (x, x0, gamma) => {
    point(x); num(x0, 'x0'); positive(gamma, 'gamma');
    return liftPoint(x, v => 0.5 + Math.atan((v - x0) / gamma) / Math.PI);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.cauchy.sample', arity: 4,
  sig: '(seed: number, n: number, x0: number, gamma: number) -> Tensor[n]',
  doc: '柯西抽样（逆变换）：x = x0 + γ·tan(π(U−½))',
  ref: 'Devroye 1986, Non-Uniform Random Variate Generation（逆变换法）',
  inv: '同种子可复现；中位数 = x0',
  impl: (seed, n, x0, gamma) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n'); num(x0, 'x0'); positive(gamma, 'gamma');
    const rng = createRng(seed);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let u = rng.next();
      if (u < 1e-300) u = 1e-300;
      out[i] = x0 + gamma * Math.tan(Math.PI * (u - 0.5));
    }
    return new Tensor(out, [n]);
  },
});

// —— 6. 逻辑斯蒂分布 Logistic(μ, s) ——
defineOp({
  id: nid(), domain: 6, name: 'prob.logistic.pdf', arity: 3,
  sig: '(x: number|Tensor, mu: number, s: number) -> number|Tensor',
  doc: '逻辑斯蒂密度：pdf(x) = e^{−z}/(s(1+e^{−z})²)，z = (x−μ)/s',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 2（logistic）',
  inv: 'pdf ≥ 0，∫pdf = 1，关于 μ 对称；s > 0',
  impl: (x, mu, s) => {
    point(x); num(mu, 'mu'); positive(s, 's');
    return liftPoint(x, v => {
      const z = (v - mu) / s;
      // 数值稳定：|z| 大时用 e^{−|z|} 形式
      const e = Math.exp(-Math.abs(z));
      return e / (s * (1 + e) ** 2);
    });
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.logistic.cdf', arity: 3,
  sig: '(x: number|Tensor, mu: number, s: number) -> number|Tensor',
  doc: '逻辑斯蒂 CDF：1/(1+e^{−z})，z = (x−μ)/s',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 2',
  inv: 'cdf ∈ [0,1] 单调不减；s > 0',
  impl: (x, mu, s) => {
    point(x); num(mu, 'mu'); positive(s, 's');
    return liftPoint(x, v => {
      const z = (v - mu) / s;
      return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
    });
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.logistic.sample', arity: 4,
  sig: '(seed: number, n: number, mu: number, s: number) -> Tensor[n]',
  doc: '逻辑斯蒂抽样（逆变换）：x = μ + s·ln(U/(1−U))',
  ref: 'Devroye 1986, Non-Uniform Random Variate Generation（逆变换法）',
  inv: '同种子可复现；大样本矩 → (μ, π²s²/3)',
  impl: (seed, n, mu, s) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n'); num(mu, 'mu'); positive(s, 's');
    const rng = createRng(seed);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let u = rng.next();
      if (u < 1e-300) u = 1e-300;
      if (u >= 1) u = 1 - 1e-16;
      out[i] = mu + s * Math.log(u / (1 - u));
    }
    return new Tensor(out, [n]);
  },
});

// —— 7. 伽马分布 Gamma(shape k, rate λ) ——
defineOp({
  id: nid(), domain: 6, name: 'prob.gamma.pdf', arity: 3,
  sig: '(x: number|Tensor, shape: number, rate: number) -> number|Tensor',
  doc: '伽马密度：pdf(x) = λ^k x^{k−1} e^{−λx} / Γ(k)（x ≥ 0），否则 0',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1（gamma）',
  inv: 'pdf ≥ 0，∫₀^∞ pdf = 1；shape > 0，rate > 0',
  impl: (x, shape, rate) => {
    point(x); positive(shape, 'shape'); positive(rate, 'rate');
    const lg = lgamma(shape);
    return liftPoint(x, v => {
      if (v < 0) return 0;
      if (v === 0) {
        if (shape > 1) return 0;
        if (shape === 1) return rate; // 退化为指数分布
        return Infinity;
      }
      return Math.exp((shape - 1) * Math.log(v) - rate * v + shape * Math.log(rate) - lg);
    });
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.gamma.cdf', arity: 3,
  sig: '(x: number|Tensor, shape: number, rate: number) -> number|Tensor',
  doc: '伽马 CDF：P(k, λx)，正则化下不完全伽马函数',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1',
  inv: 'cdf ∈ [0,1] 单调不减；shape > 0，rate > 0',
  impl: (x, shape, rate) => {
    point(x); positive(shape, 'shape'); positive(rate, 'rate');
    return liftPoint(x, v => (v <= 0 ? 0 : gammap(shape, rate * v)));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.gamma.sample', arity: 4,
  sig: '(seed: number, n: number, shape: number, rate: number) -> Tensor[n]',
  doc: '伽马抽样：Marsaglia-Tsang 拒绝法（shape ≥ 1），shape < 1 用 boost G(α)=G(α+1)·U^(1/α)',
  ref: 'Marsaglia & Tsang, ACM Trans. Math. Software 26(2), 2000',
  inv: '同种子可复现；大样本矩 → (k/λ, k/λ²)',
  impl: (seed, n, shape, rate) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n'); positive(shape, 'shape'); positive(rate, 'rate');
    const rng = createRng(seed);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = gammaSample(rng, shape, rate);
    return new Tensor(out, [n]);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.gamma.fit', arity: 1,
  sig: '(data: Tensor) -> Tensor[2]',
  doc: '伽马 MLE 拟合：解似然方程 ln k − ψ(k) = s（s = ln(x̄) − mean(ln x) ≥ 0），返回 [k̂, λ̂]，λ̂ = k̂/x̄',
  ref: 'Minka, "Estimating a Gamma Distribution", 2002；Thom 1968（初值近似）',
  inv: '数据须严格为正；f(k)=ln k − ψ(k) − s 严格递减故解唯一',
  impl: (data) => {
    const d = dataOf(data);
    let sx = 0, sl = 0;
    for (const v of d) {
      if (!(v > 0)) throw new ForgeError('E_VALUE', `gamma.fit: 数据须严格为正（${v}）`);
      sx += v;
      sl += Math.log(v);
    }
    const n = d.length;
    const mean = sx / n;
    const s = Math.log(mean) - sl / n; // Jensen：s ≥ 0；s→0⁺ 时 k̂→∞，s→∞ 时 k̂→0⁺
    if (!(s > 0)) throw new ForgeError('E_VALUE', 'gamma.fit: 数据为常数，shape MLE 无有限解');
    //  safeguarded Newton：f(k) = ln k − ψ(k) − s 严格递减，括号 [lo, hi] 保证收敛
    let lo = 1e-8, hi = 1e8;
    let k = Math.min(hi, Math.max(lo, (3 - s + Math.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)));
    for (let it = 0; it < 300; it++) {
      const f = Math.log(k) - digamma(k) - s;
      if (f > 0) lo = k; else hi = k;
      const fp = 1 / k - trigamma(k); // f'(k) < 0
      let kn = k - f / fp;
      if (!(kn > lo && kn < hi)) kn = 0.5 * (lo + hi); // 越界则二分回退
      if (Math.abs(kn - k) <= 1e-14 * Math.max(1, kn)) { k = kn; break; }
      k = kn;
    }
    return Tensor.fromFlat([k, k / mean], [2]);
  },
});

// —— 8. 贝塔分布 Beta(α, β) ——
defineOp({
  id: nid(), domain: 6, name: 'prob.beta.pdf', arity: 3,
  sig: '(x: number|Tensor, a: number, b: number) -> number|Tensor',
  doc: '贝塔密度：pdf(x) = x^{a−1}(1−x)^{b−1} / B(a,b)（0 ≤ x ≤ 1），否则 0',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 2（beta）',
  inv: 'pdf ≥ 0，∫₀¹ pdf = 1；a > 0，b > 0',
  impl: (x, a, b) => {
    point(x); positive(a, 'a'); positive(b, 'b');
    const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
    return liftPoint(x, v => {
      if (v < 0 || v > 1) return 0;
      if (v === 0) return a > 1 ? 0 : a === 1 ? Math.exp(-lbeta) : Infinity;
      if (v === 1) return b > 1 ? 0 : b === 1 ? Math.exp(-lbeta) : Infinity;
      return Math.exp((a - 1) * Math.log(v) + (b - 1) * Math.log(1 - v) - lbeta);
    });
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.beta.cdf', arity: 3,
  sig: '(x: number|Tensor, a: number, b: number) -> number|Tensor',
  doc: '贝塔 CDF：I_x(a,b)，正则化不完全贝塔函数',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 2',
  inv: 'cdf ∈ [0,1] 单调不减；a > 0，b > 0',
  impl: (x, a, b) => {
    point(x); positive(a, 'a'); positive(b, 'b');
    return liftPoint(x, v => (v <= 0 ? 0 : v >= 1 ? 1 : betai(a, b, v)));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.beta.sample', arity: 4,
  sig: '(seed: number, n: number, a: number, b: number) -> Tensor[n]',
  doc: '贝塔抽样：X = G1/(G1+G2)，G1~Gamma(a,1)、G2~Gamma(b,1) 独立',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 2（beta 抽样）',
  inv: '同种子可复现；大样本矩 → (a/(a+b), ab/((a+b)²(a+b+1)))',
  impl: (seed, n, a, b) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n'); positive(a, 'a'); positive(b, 'b');
    const rng = createRng(seed);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const g1 = gammaSample(rng, a, 1);
      const g2 = gammaSample(rng, b, 1);
      out[i] = g1 / (g1 + g2);
    }
    return new Tensor(out, [n]);
  },
});

// —— 9. 卡方分布 Chi2(ν) ——
defineOp({
  id: nid(), domain: 6, name: 'prob.chi2.pdf', arity: 2,
  sig: '(x: number|Tensor, df: number) -> number|Tensor',
  doc: '卡方密度：pdf(x) = x^{ν/2−1} e^{−x/2} / (2^{ν/2} Γ(ν/2))（x ≥ 0），否则 0',
  ref: 'Evans & Rosenthal, Probability and Statistics（chi-square 分布）',
  inv: 'pdf ≥ 0，∫₀^∞ pdf = 1；df > 0',
  impl: (x, df) => {
    point(x); positive(df, 'df');
    const h = df / 2;
    const c = Math.exp(h * Math.log(0.5) - lgamma(h));
    return liftPoint(x, v => {
      if (v < 0) return 0;
      if (v === 0) return h > 1 ? 0 : h === 1 ? c : Infinity;
      return c * Math.exp((h - 1) * Math.log(v) - 0.5 * v);
    });
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.chi2.cdf', arity: 2,
  sig: '(x: number|Tensor, df: number) -> number|Tensor',
  doc: '卡方 CDF：P(ν/2, x/2)，正则化下不完全伽马函数',
  ref: 'Evans & Rosenthal, Probability and Statistics（chi-square 分布）',
  inv: 'cdf ∈ [0,1] 单调不减；df > 0',
  impl: (x, df) => {
    point(x); positive(df, 'df');
    return liftPoint(x, v => (v <= 0 ? 0 : gammap(df / 2, v / 2)));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.chi2.sample', arity: 3,
  sig: '(seed: number, n: number, df: number) -> Tensor[n]',
  doc: '卡方抽样：Gamma(df/2, rate=1/2)（即 df 个标准正态平方和的一般化）',
  ref: 'Marsaglia & Tsang 2000（伽马抽样）；Evans & Rosenthal',
  inv: '同种子可复现；大样本矩 → (ν, 2ν)',
  impl: (seed, n, df) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n'); positive(df, 'df');
    const rng = createRng(seed);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = gammaSample(rng, df / 2, 0.5);
    return new Tensor(out, [n]);
  },
});

// —— 10. 学生 t 分布 StudentT(ν) ——
defineOp({
  id: nid(), domain: 6, name: 'prob.studentT.pdf', arity: 2,
  sig: '(x: number|Tensor, df: number) -> number|Tensor',
  doc: 't 分布密度：pdf(x) = Γ((ν+1)/2)/(√(νπ)Γ(ν/2)) · (1 + x²/ν)^{−(ν+1)/2}',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 2（Student t）',
  inv: 'pdf ≥ 0，∫pdf = 1，关于 0 对称；df > 0；ν>1 时均值 0，ν>2 时方差 ν/(ν−2)',
  impl: (x, df) => {
    point(x); positive(df, 'df');
    const c = Math.exp(lgamma((df + 1) / 2) - lgamma(df / 2) - 0.5 * Math.log(df * Math.PI));
    return liftPoint(x, v => c * Math.pow(1 + v * v / df, -(df + 1) / 2));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.studentT.cdf', arity: 2,
  sig: '(x: number|Tensor, df: number) -> number|Tensor',
  doc: 't 分布 CDF：x>0 时 1 − ½·I_{ν/(ν+x²)}(ν/2, 1/2)，x≤0 时 ½·I_{ν/(ν+x²)}(ν/2, 1/2)',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 2',
  inv: 'cdf ∈ [0,1] 单调不减，cdf(−x) = 1 − cdf(x)；df > 0',
  impl: (x, df) => {
    point(x); positive(df, 'df');
    const h = df / 2;
    return liftPoint(x, v => {
      const p = betai(h, 0.5, df / (df + v * v));
      return v > 0 ? 1 - 0.5 * p : 0.5 * p;
    });
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.studentT.sample', arity: 3,
  sig: '(seed: number, n: number, df: number) -> Tensor[n]',
  doc: 't 分布抽样：T = Z/√(V/ν)，Z~N(0,1)、V~χ²(ν) 独立',
  ref: 'Evans & Rosenthal, Probability and Statistics（t 分布表示）',
  inv: '同种子可复现；ν>2 时大样本方差 → ν/(ν−2)',
  impl: (seed, n, df) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n'); positive(df, 'df');
    const rng = createRng(seed);
    const g = normalGen(rng);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const z = g();
      const v = gammaSample(rng, df / 2, 0.5);
      out[i] = z / Math.sqrt(v / df);
    }
    return new Tensor(out, [n]);
  },
});

// —— 11. 瑞利分布 Rayleigh(σ) ——
defineOp({
  id: nid(), domain: 6, name: 'prob.rayleigh.pdf', arity: 2,
  sig: '(x: number|Tensor, sigma: number) -> number|Tensor',
  doc: '瑞利密度：pdf(x) = (x/σ²)·e^{−x²/(2σ²)}（x ≥ 0），否则 0',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1（Rayleigh）',
  inv: 'pdf ≥ 0，∫₀^∞ pdf = 1；σ > 0',
  impl: (x, sigma) => {
    point(x); positive(sigma, 'sigma');
    return liftPoint(x, v => (v < 0 ? 0 : (v / (sigma * sigma)) * Math.exp(-v * v / (2 * sigma * sigma))));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.rayleigh.cdf', arity: 2,
  sig: '(x: number|Tensor, sigma: number) -> number|Tensor',
  doc: '瑞利 CDF：1 − e^{−x²/(2σ²)}（x ≥ 0），否则 0',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1',
  inv: 'cdf ∈ [0,1] 单调不减；σ > 0',
  impl: (x, sigma) => {
    point(x); positive(sigma, 'sigma');
    return liftPoint(x, v => (v <= 0 ? 0 : -Math.expm1(-v * v / (2 * sigma * sigma))));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.rayleigh.sample', arity: 3,
  sig: '(seed: number, n: number, sigma: number) -> Tensor[n]',
  doc: '瑞利抽样（逆变换）：x = σ·√(−2 ln U)',
  ref: 'Devroye 1986, Non-Uniform Random Variate Generation（逆变换法）',
  inv: '同种子可复现；大样本矩 → (σ√(π/2), (4−π)σ²/2)',
  impl: (seed, n, sigma) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n'); positive(sigma, 'sigma');
    const rng = createRng(seed);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let u = rng.next();
      if (u < 1e-300) u = 1e-300;
      out[i] = sigma * Math.sqrt(-2 * Math.log(u));
    }
    return new Tensor(out, [n]);
  },
});

// —— 12. 威布尔分布 Weibull(shape k, scale λ) ——
defineOp({
  id: nid(), domain: 6, name: 'prob.weibull.pdf', arity: 3,
  sig: '(x: number|Tensor, shape: number, scale: number) -> number|Tensor',
  doc: '威布尔密度：pdf(x) = (k/λ)(x/λ)^{k−1} e^{−(x/λ)^k}（x ≥ 0），否则 0',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1（Weibull）',
  inv: 'pdf ≥ 0，∫₀^∞ pdf = 1；k > 0，λ > 0',
  impl: (x, shape, scale) => {
    point(x); positive(shape, 'shape'); positive(scale, 'scale');
    return liftPoint(x, v => {
      if (v < 0) return 0;
      if (v === 0) return shape > 1 ? 0 : shape === 1 ? 1 / scale : Infinity;
      return (shape / scale) * Math.pow(v / scale, shape - 1) * Math.exp(-Math.pow(v / scale, shape));
    });
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.weibull.cdf', arity: 3,
  sig: '(x: number|Tensor, shape: number, scale: number) -> number|Tensor',
  doc: '威布尔 CDF：1 − e^{−(x/λ)^k}（x ≥ 0），否则 0',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1',
  inv: 'cdf ∈ [0,1] 单调不减；k > 0，λ > 0',
  impl: (x, shape, scale) => {
    point(x); positive(shape, 'shape'); positive(scale, 'scale');
    return liftPoint(x, v => (v <= 0 ? 0 : -Math.expm1(-Math.pow(v / scale, shape))));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.weibull.sample', arity: 4,
  sig: '(seed: number, n: number, shape: number, scale: number) -> Tensor[n]',
  doc: '威布尔抽样（逆变换）：x = λ·(−ln U)^{1/k}',
  ref: 'Devroye 1986, Non-Uniform Random Variate Generation（逆变换法）',
  inv: '同种子可复现；大样本矩 → (λΓ(1+1/k), λ²(Γ(1+2/k)−Γ(1+1/k)²))',
  impl: (seed, n, shape, scale) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n'); positive(shape, 'shape'); positive(scale, 'scale');
    const rng = createRng(seed);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let u = rng.next();
      if (u < 1e-300) u = 1e-300;
      out[i] = scale * Math.pow(-Math.log(u), 1 / shape);
    }
    return new Tensor(out, [n]);
  },
});

// —— 13. 帕累托分布 Pareto(xm, α) ——
defineOp({
  id: nid(), domain: 6, name: 'prob.pareto.pdf', arity: 3,
  sig: '(x: number|Tensor, xm: number, alpha: number) -> number|Tensor',
  doc: '帕累托密度：pdf(x) = α·xm^α / x^{α+1}（x ≥ xm），否则 0',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1（Pareto）',
  inv: 'pdf ≥ 0，∫_{xm}^∞ pdf = 1；α > 0；α>1 时均值 αxm/(α−1)',
  impl: (x, xm, alpha) => {
    point(x); positive(xm, 'xm'); positive(alpha, 'alpha');
    return liftPoint(x, v => (v < xm ? 0 : alpha * Math.pow(xm, alpha) / Math.pow(v, alpha + 1)));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.pareto.cdf', arity: 3,
  sig: '(x: number|Tensor, xm: number, alpha: number) -> number|Tensor',
  doc: '帕累托 CDF：1 − (xm/x)^α（x ≥ xm），否则 0',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1',
  inv: 'cdf ∈ [0,1] 单调不减；α > 0',
  impl: (x, xm, alpha) => {
    point(x); positive(xm, 'xm'); positive(alpha, 'alpha');
    return liftPoint(x, v => (v < xm ? 0 : -Math.expm1(-alpha * Math.log(v / xm))));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.pareto.sample', arity: 4,
  sig: '(seed: number, n: number, xm: number, alpha: number) -> Tensor[n]',
  doc: '帕累托抽样（逆变换）：x = xm·U^{−1/α}',
  ref: 'Devroye 1986, Non-Uniform Random Variate Generation（逆变换法）',
  inv: '同种子可复现；α>1 时大样本均值 → αxm/(α−1)',
  impl: (seed, n, xm, alpha) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n'); positive(xm, 'xm'); positive(alpha, 'alpha');
    const rng = createRng(seed);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let u = rng.next();
      if (u < 1e-300) u = 1e-300;
      out[i] = xm * Math.pow(u, -1 / alpha);
    }
    return new Tensor(out, [n]);
  },
});

// ============================================================================
// 五、离散分布（15 条：D6-044 … D6-058）
// ============================================================================

// —— 1. 伯努利分布 Bernoulli(p) ——
defineOp({
  id: nid(), domain: 6, name: 'prob.bernoulli.pmf', arity: 2,
  sig: '(k: number|Tensor, p: number) -> number|Tensor',
  doc: '伯努利 pmf：P(k) = p^k (1−p)^{1−k}（k ∈ {0,1}），非整数 k 为 0',
  ref: 'Johnson, Kotz & Kemp, Discrete Univariate Distributions（Bernoulli）',
  inv: 'Σpmf = 1；0 ≤ p ≤ 1',
  impl: (k, p) => {
    point(k); unitInterval(p, 'p');
    return liftPoint(k, v => (v === 0 ? 1 - p : v === 1 ? p : 0));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.bernoulli.cdf', arity: 2,
  sig: '(k: number|Tensor, p: number) -> number|Tensor',
  doc: '伯努利 CDF：k<0 为 0，0≤k<1 为 1−p，k≥1 为 1',
  ref: 'Johnson, Kotz & Kemp, Discrete Univariate Distributions',
  inv: 'cdf ∈ [0,1] 单调不减；0 ≤ p ≤ 1',
  impl: (k, p) => {
    point(k); unitInterval(p, 'p');
    return liftPoint(k, v => (v < 0 ? 0 : v < 1 ? 1 - p : 1));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.bernoulli.sample', arity: 3,
  sig: '(seed: number, n: number, p: number) -> Tensor[n]',
  doc: '伯努利抽样：U < p 取 1 否则 0',
  ref: 'Devroye 1986, Non-Uniform Random Variate Generation',
  inv: '同种子可复现；大样本矩 → (p, p(1−p))',
  impl: (seed, n, p) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n'); unitInterval(p, 'p');
    const rng = createRng(seed);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = rng.next() < p ? 1 : 0;
    return new Tensor(out, [n]);
  },
});

// —— 2. 二项分布 Binomial(nTrials, p) ——
defineOp({
  id: nid(), domain: 6, name: 'prob.binomial.pmf', arity: 3,
  sig: '(k: number|Tensor, nTrials: number, p: number) -> number|Tensor',
  doc: '二项 pmf：C(N,k) p^k (1−p)^{N−k}（0 ≤ k ≤ N 整数），否则 0；组合数经 lgamma 计算',
  ref: 'Johnson, Kotz & Kemp, Discrete Univariate Distributions（binomial）',
  inv: 'Σₖ pmf = 1；N 为非负整数，0 ≤ p ≤ 1',
  impl: (k, nTrials, p) => {
    point(k); nonNeg(nTrials | 0, 'nTrials'); unitInterval(p, 'p');
    const N = nTrials | 0;
    return liftPoint(k, v => {
      const ki = Math.floor(v);
      if (ki !== v || ki < 0 || ki > N) return 0;
      if (p === 0) return ki === 0 ? 1 : 0;
      if (p === 1) return ki === N ? 1 : 0;
      const lc = lgamma(N + 1) - lgamma(ki + 1) - lgamma(N - ki + 1);
      return Math.exp(lc + ki * Math.log(p) + (N - ki) * Math.log(1 - p));
    });
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.binomial.cdf', arity: 3,
  sig: '(k: number|Tensor, nTrials: number, p: number) -> number|Tensor',
  doc: '二项 CDF：Σ_{i≤⌊k⌋} pmf(i)，用正则化不完全贝塔 I_{1−p}(N−k, k+1) 闭式计算',
  ref: 'Johnson, Kotz & Kemp, Discrete Univariate Distributions',
  inv: 'cdf ∈ [0,1] 单调不减；N 为非负整数，0 ≤ p ≤ 1',
  impl: (k, nTrials, p) => {
    point(k); nonNeg(nTrials | 0, 'nTrials'); unitInterval(p, 'p');
    const N = nTrials | 0;
    return liftPoint(k, v => {
      const ki = Math.floor(v);
      if (ki < 0) return 0;
      if (ki >= N) return 1;
      if (p === 0) return 1;
      if (p === 1) return 0;
      return betai(N - ki, ki + 1, 1 - p);
    });
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.binomial.sample', arity: 4,
  sig: '(seed: number, n: number, nTrials: number, p: number) -> Tensor[n]',
  doc: '二项抽样：N 次独立伯努利试验之和',
  ref: 'Devroye 1986, Non-Uniform Random Variate Generation',
  inv: '同种子可复现；大样本矩 → (Np, Np(1−p))',
  impl: (seed, n, nTrials, p) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n'); nonNeg(nTrials | 0, 'nTrials'); unitInterval(p, 'p');
    const N = nTrials | 0;
    const rng = createRng(seed);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < N; j++) if (rng.next() < p) s++;
      out[i] = s;
    }
    return new Tensor(out, [n]);
  },
});

// —— 3. 泊松分布 Poisson(λ) ——
defineOp({
  id: nid(), domain: 6, name: 'prob.poisson.pmf', arity: 2,
  sig: '(k: number|Tensor, lambda: number) -> number|Tensor',
  doc: '泊松 pmf：e^{−λ} λ^k / k!（k 为非负整数），否则 0',
  ref: 'Johnson, Kotz & Kemp, Discrete Univariate Distributions（Poisson）',
  inv: 'Σₖ pmf = 1；λ > 0',
  impl: (k, lambda) => {
    point(k); positive(lambda, 'lambda');
    return liftPoint(k, v => {
      const ki = Math.floor(v);
      if (ki !== v || ki < 0) return 0;
      return Math.exp(-lambda + ki * Math.log(lambda) - lgamma(ki + 1));
    });
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.poisson.cdf', arity: 2,
  sig: '(k: number|Tensor, lambda: number) -> number|Tensor',
  doc: '泊松 CDF：Σ_{i≤⌊k⌋} pmf(i)，用恒等式 Q(⌊k⌋+1, λ)（正则化上不完全伽马）计算',
  ref: 'Johnson, Kotz & Kemp, Discrete Univariate Distributions',
  inv: 'cdf ∈ [0,1] 单调不减；λ > 0',
  impl: (k, lambda) => {
    point(k); positive(lambda, 'lambda');
    return liftPoint(k, v => (v < 0 ? 0 : gammaq(Math.floor(v) + 1, lambda)));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.poisson.sample', arity: 3,
  sig: '(seed: number, n: number, lambda: number) -> Tensor[n]',
  doc: '泊松抽样：λ < 30 用 Knuth 乘法法；λ ≥ 30 用正态近似 round(λ + √λ·Z)',
  ref: 'Knuth, TAOCP Vol.2 §3.4.1（Knuth 算法）；Devroye 1986',
  inv: '同种子可复现；大样本矩 → (λ, λ)',
  impl: (seed, n, lambda) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n'); positive(lambda, 'lambda');
    const rng = createRng(seed);
    const out = new Float64Array(n);
    if (lambda < 30) {
      const L = Math.exp(-lambda);
      for (let i = 0; i < n; i++) {
        let k = 0, p = 1;
        do { k++; p *= rng.next(); } while (p > L);
        out[i] = k - 1;
      }
    } else {
      const g = normalGen(rng);
      const sd = Math.sqrt(lambda);
      for (let i = 0; i < n; i++) out[i] = Math.max(0, Math.round(lambda + sd * g()));
    }
    return new Tensor(out, [n]);
  },
});

// —— 4. 几何分布 Geometric(p)（支撑 {1,2,…}，首次成功所需试验次数） ——
defineOp({
  id: nid(), domain: 6, name: 'prob.geometric.pmf', arity: 2,
  sig: '(k: number|Tensor, p: number) -> number|Tensor',
  doc: '几何 pmf：(1−p)^{k−1} p（k 为正整数），否则 0',
  ref: 'Johnson, Kotz & Kemp, Discrete Univariate Distributions（geometric）',
  inv: 'Σₖ pmf = 1；0 < p ≤ 1',
  impl: (k, p) => {
    point(k); unitInterval(p, 'p');
    if (p === 0) throw new ForgeError('E_VALUE', 'geometric.pmf: p 必须 > 0');
    return liftPoint(k, v => {
      const ki = Math.floor(v);
      if (ki !== v || ki < 1) return 0;
      return Math.exp((ki - 1) * Math.log(1 - p) + Math.log(p));
    });
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.geometric.cdf', arity: 2,
  sig: '(k: number|Tensor, p: number) -> number|Tensor',
  doc: '几何 CDF：k<1 为 0，否则 1 − (1−p)^{⌊k⌋}',
  ref: 'Johnson, Kotz & Kemp, Discrete Univariate Distributions',
  inv: 'cdf ∈ [0,1] 单调不减；0 < p ≤ 1',
  impl: (k, p) => {
    point(k); unitInterval(p, 'p');
    if (p === 0) throw new ForgeError('E_VALUE', 'geometric.cdf: p 必须 > 0');
    return liftPoint(k, v => (v < 1 ? 0 : -Math.expm1(Math.floor(v) * Math.log(1 - p))));
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.geometric.sample', arity: 3,
  sig: '(seed: number, n: number, p: number) -> Tensor[n]',
  doc: '几何抽样（逆变换）：k = ⌈ln U / ln(1−p)⌉',
  ref: 'Devroye 1986, Non-Uniform Random Variate Generation（逆变换法）',
  inv: '同种子可复现；大样本矩 → (1/p, (1−p)/p²)',
  impl: (seed, n, p) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n'); unitInterval(p, 'p');
    if (p === 0) throw new ForgeError('E_VALUE', 'geometric.sample: p 必须 > 0');
    const rng = createRng(seed);
    const lq = Math.log(1 - p);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let u = rng.next();
      if (u < 1e-300) u = 1e-300;
      out[i] = Math.ceil(Math.log(u) / lq);
    }
    return new Tensor(out, [n]);
  },
});

// —— 5. 类别分布 Categorical(probs) ——
defineOp({
  id: nid(), domain: 6, name: 'prob.categorical.pmf', arity: 2,
  sig: '(k: number|Tensor, probs: Tensor) -> number|Tensor',
  doc: '类别 pmf：P(k) = probs[k]/Σprobs（k ∈ [0, K−1] 整数），否则 0（自动归一化）',
  ref: 'Johnson, Kotz & Kemp, Discrete Univariate Distributions（categorical）',
  inv: 'Σₖ pmf = 1；probs 非负且和 > 0',
  impl: (k, probs) => {
    point(k);
    const { d, s } = probVecOf(probs, 'probs');
    return liftPoint(k, v => {
      const ki = Math.floor(v);
      if (ki !== v || ki < 0 || ki >= d.length) return 0;
      return d[ki] / s;
    });
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.categorical.cdf', arity: 2,
  sig: '(k: number|Tensor, probs: Tensor) -> number|Tensor',
  doc: '类别 CDF：Σ_{i≤⌊k⌋} pmf(i)（自动归一化）',
  ref: 'Johnson, Kotz & Kemp, Discrete Univariate Distributions',
  inv: 'cdf ∈ [0,1] 单调不减；probs 非负且和 > 0',
  impl: (k, probs) => {
    point(k);
    const { d, s } = probVecOf(probs, 'probs');
    return liftPoint(k, v => {
      const ki = Math.floor(v);
      if (ki < 0) return 0;
      let acc = 0;
      for (let i = 0; i <= Math.min(ki, d.length - 1); i++) acc += d[i];
      return acc / s;
    });
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.categorical.sample', arity: 3,
  sig: '(seed: number, n: number, probs: Tensor) -> Tensor[n]',
  doc: '类别抽样（逆变换）：U·Σprobs 落在第几个累积区间即返回该下标',
  ref: 'Devroye 1986, Non-Uniform Random Variate Generation（逆变换法）',
  inv: '同种子可复现；频率 → probs/Σprobs',
  impl: (seed, n, probs) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n');
    const { d, s } = probVecOf(probs, 'probs');
    const rng = createRng(seed);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const u = rng.next() * s;
      let acc = 0, j = 0;
      for (; j < d.length - 1; j++) { acc += d[j]; if (u < acc) break; }
      out[i] = j;
    }
    return new Tensor(out, [n]);
  },
});

// ============================================================================
// 六、多元正态分布（2 条：D6-059 … D6-060）
// ============================================================================

// 多元正态 pdf：N(x; μ, Σ) = (2π)^{−d/2} |Σ|^{−1/2} exp(−½ (x−μ)ᵀ Σ⁻¹ (x−μ))
defineOp({
  id: nid(), domain: 6, name: 'prob.mvn.pdf', arity: 3,
  sig: '(x: Tensor[d]|Tensor[n,d], mu: Tensor[d], Sigma: Tensor[d,d]) -> number|Tensor[n]',
  doc: '多元正态密度：经 Cholesky 解 Σ⁻¹ 二次型，log|Σ| = 2Σln L_ii',
  ref: 'Tong, The Multivariate Normal Distribution；Evans & Rosenthal',
  inv: 'Σ 对称正定；pdf(mu) = (2π)^{−d/2}|Σ|^{−1/2}',
  impl: (x, mu, Sigma) => {
    if (!(x instanceof Tensor)) throw new ForgeError('E_TYPE', `x: 期望 Tensor，得到 ${typeof x}`);
    const m = dataOf(mu, 'mu');
    const d = m.length;
    if (!(Sigma instanceof Tensor) || Sigma.shape.length !== 2 || Sigma.shape[0] !== d || Sigma.shape[1] !== d) {
      throw new ForgeError('E_SHAPE', `Sigma: 期望 [${d},${d}] 方阵`);
    }
    const S = dataOf(Sigma, 'Sigma');
    // 对称性校验
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < i; j++) {
        if (S[i * d + j] !== S[j * d + i]) throw new ForgeError('E_VALUE', 'Sigma: 协方差矩阵必须对称');
      }
    }
    const L = choleskyLower(S, d);
    let logDet = 0;
    for (let i = 0; i < d; i++) logDet += Math.log(L[i * d + i]);
    logDet *= 2; // |Σ| = |L·Lᵀ| = |L|² ⇒ ln|Σ| = 2·Σ ln L_ii
    const logNorm = -0.5 * (d * Math.log(2 * Math.PI) + logDet);
    const quad = (row, off) => {
      const y = new Float64Array(d);
      for (let i = 0; i < d; i++) {
        let s = row[off + i] - m[i];
        for (let j = 0; j < i; j++) s -= L[i * d + j] * y[j];
        y[i] = s / L[i * d + i];
      }
      let q = 0;
      for (let i = 0; i < d; i++) q += y[i] * y[i];
      return q;
    };
    if (x.ndim === 1) {
      const xd = dataOf(x, 'x');
      if (xd.length !== d) throw new ForgeError('E_SHAPE', `x: 期望长度 ${d}，得到 ${xd.length}`);
      return Math.exp(logNorm - 0.5 * quad(xd, 0));
    }
    if (x.ndim === 2) {
      const [n, cols] = x.shape;
      if (cols !== d) throw new ForgeError('E_SHAPE', `x: 期望每行长度 ${d}，得到 ${cols}`);
      const out = new Float64Array(n);
      for (let r = 0; r < n; r++) out[r] = Math.exp(logNorm - 0.5 * quad(x.data, r * d));
      return new Tensor(out, [n]);
    }
    throw new ForgeError('E_SHAPE', `x: 期望 1D [d] 或 2D [n,d]，得到 ndim=${x.ndim}`);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.mvn.sample', arity: 4,
  sig: '(seed: number, n: number, mu: Tensor[d], Sigma: Tensor[d,d]) -> Tensor[n,d]',
  doc: '多元正态抽样：x = μ + L·z，Σ = L·Lᵀ（Cholesky），z ~ N(0, I_d)',
  ref: 'Tong, The Multivariate Normal Distribution（Cholesky 抽样法）',
  inv: '同种子可复现；大样本协方差 → Σ',
  impl: (seed, n, mu, Sigma) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n');
    const m = dataOf(mu, 'mu');
    const d = m.length;
    if (!(Sigma instanceof Tensor) || Sigma.shape.length !== 2 || Sigma.shape[0] !== d || Sigma.shape[1] !== d) {
      throw new ForgeError('E_SHAPE', `Sigma: 期望 [${d},${d}] 方阵`);
    }
    const L = choleskyLower(dataOf(Sigma, 'Sigma'), d);
    const rng = createRng(seed);
    const g = normalGen(rng);
    const out = new Float64Array(n * d);
    for (let r = 0; r < n; r++) {
      const z = new Float64Array(d);
      for (let i = 0; i < d; i++) z[i] = g();
      for (let i = 0; i < d; i++) {
        let s = m[i];
        for (let j = 0; j <= i; j++) s += L[i * d + j] * z[j];
        out[r * d + i] = s;
      }
    }
    return new Tensor(out, [n, d]);
  },
});

// ============================================================================
// 七、信息论（8 条：D6-061 … D6-068）
// ============================================================================

defineOp({
  id: nid(), domain: 6, name: 'prob.info.entropy', arity: 1,
  sig: '(p: Tensor) -> number',
  doc: '香农熵（nat）：H(p) = −Σ p_i ln p_i（0·ln0 = 0；自动归一化）',
  ref: 'Shannon 1948；Cover & Thomas, Elements of Information Theory §2.1',
  inv: 'H ≥ 0；H ≤ ln|X|，均匀分布取等',
  impl: (p) => {
    const { d, s } = probVecOf(p, 'p');
    let h = 0;
    for (const v of d) if (v > 0) h -= (v / s) * Math.log(v / s);
    return h;
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.info.crossEntropy', arity: 2,
  sig: '(p: Tensor, q: Tensor) -> number',
  doc: '交叉熵（nat）：H(p,q) = −Σ p_i ln q_i（p>0 且 q_i=0 时返回 +∞）',
  ref: 'Cover & Thomas, Elements of Information Theory §2.2',
  inv: 'H(p,q) ≥ H(p)，等号当且仅当 p = q',
  impl: (p, q) => {
    const P = probVecOf(p, 'p');
    const Q = probVecOf(q, 'q');
    if (P.d.length !== Q.d.length) throw new ForgeError('E_VALUE', 'crossEntropy: p 与 q 长度须一致');
    let h = 0;
    for (let i = 0; i < P.d.length; i++) {
      const pi = P.d[i] / P.s;
      if (pi <= 0) continue;
      if (Q.d[i] <= 0) return Infinity;
      h -= pi * Math.log(Q.d[i] / Q.s);
    }
    return h;
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.info.klDiv', arity: 2,
  sig: '(p: Tensor, q: Tensor) -> number',
  doc: 'KL 散度（nat）：D(p‖q) = Σ p_i ln(p_i/q_i)；p>0 且 q_i=0 时返回 +∞',
  ref: 'Kullback & Leibler 1951（KL 散度原始文献）',
  inv: 'D(p‖p) = 0；D ≥ 0（Gibbs 不等式）；不对称',
  impl: (p, q) => {
    const P = probVecOf(p, 'p');
    const Q = probVecOf(q, 'q');
    if (P.d.length !== Q.d.length) throw new ForgeError('E_VALUE', 'klDiv: p 与 q 长度须一致');
    let d = 0;
    for (let i = 0; i < P.d.length; i++) {
      const pi = P.d[i] / P.s;
      if (pi <= 0) continue;
      if (Q.d[i] <= 0) return Infinity;
      d += pi * Math.log(pi / (Q.d[i] / Q.s));
    }
    return d;
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.info.jsDiv', arity: 2,
  sig: '(p: Tensor, q: Tensor) -> number',
  doc: 'JS 散度（nat）：½D(p‖m) + ½D(q‖m)，m = (p+q)/2',
  ref: 'Lin 1991（JS divergence）；Endres & Schindelin 2003',
  inv: '0 ≤ JS ≤ ln 2；JS(p,p) = 0；对称',
  impl: (p, q) => {
    const P = probVecOf(p, 'p');
    const Q = probVecOf(q, 'q');
    if (P.d.length !== Q.d.length) throw new ForgeError('E_VALUE', 'jsDiv: p 与 q 长度须一致');
    const n = P.d.length;
    const m = new Float64Array(n);
    for (let i = 0; i < n; i++) m[i] = 0.5 * (P.d[i] / P.s + Q.d[i] / Q.s);
    let js = 0;
    for (let i = 0; i < n; i++) {
      const mi = m[i];
      if (mi <= 0) continue;
      const pi = P.d[i] / P.s, qi = Q.d[i] / Q.s;
      if (pi > 0) js += 0.5 * pi * Math.log(pi / mi);
      if (qi > 0) js += 0.5 * qi * Math.log(qi / mi);
    }
    return js;
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.info.mutualInfo', arity: 1,
  sig: '(joint: Tensor[m,n]) -> number',
  doc: '互信息（nat）：I(X;Y) = Σ_ij P_ij ln(P_ij / (P_i·P_j))（自动归一化）',
  ref: 'Cover & Thomas, Elements of Information Theory §2.3；Shannon 1948',
  inv: 'I ≥ 0；独立时为 0；I ≤ min(H(X), H(Y))',
  impl: (joint) => {
    if (!(joint instanceof Tensor) || joint.ndim !== 2) {
      throw new ForgeError('E_SHAPE', 'mutualInfo: joint 须为 2D Tensor [m,n]');
    }
    const [m, n] = joint.shape;
    const J = dataOf(joint, 'joint');
    let s = 0;
    for (const v of J) { if (v < 0) throw new ForgeError('E_VALUE', 'mutualInfo: 联合概率不能为负'); s += v; }
    if (!(s > 0)) throw new ForgeError('E_VALUE', 'mutualInfo: 联合概率和须 > 0');
    const px = new Float64Array(m), py = new Float64Array(n);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) { px[i] += J[i * n + j]; py[j] += J[i * n + j]; }
    let mi = 0;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        const pij = J[i * n + j] / s;
        if (pij <= 0) continue;
        mi += pij * Math.log(pij / ((px[i] / s) * (py[j] / s)));
      }
    }
    return mi;
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.info.wasserstein1', arity: null,
  sig: '(x: Tensor, y: Tensor) -> number | (mu1: number, sigma1: number, mu2: number, sigma2: number) -> number',
  doc: '一阶 Wasserstein 距离：两样本（等长）取排序后平均绝对差；四个数值参数时按正态闭式 W1 = |μ1−μ2|',
  ref: 'Vallender 1974；Olkin & Pukelsheim 1982（正态间 W1 = |μ1−μ2|）',
  inv: 'W1 ≥ 0；W1(x,x) = 0；满足三角不等式',
  impl: (a, b, c = undefined, d = undefined) => {
    if (a instanceof Tensor && b instanceof Tensor && c === undefined) {
      const x = Array.from(dataOf(a, 'x')).sort((u, v) => u - v);
      const y = Array.from(dataOf(b, 'y')).sort((u, v) => u - v);
      if (x.length !== y.length) throw new ForgeError('E_VALUE', 'wasserstein1: 两样本须等长');
      let s = 0;
      for (let i = 0; i < x.length; i++) s += Math.abs(x[i] - y[i]);
      return s / x.length;
    }
    if (isNum(a) && isNum(b) && isNum(c) && isNum(d)) {
      positive(b, 'sigma1'); positive(d, 'sigma2');
      return Math.abs(a - c); // 正态间 1-Wasserstein 与尺度无关：W1 = |μ1−μ2|
    }
    throw new ForgeError('E_TYPE', 'wasserstein1: 需要 (Tensor,Tensor) 或 (mu1,sigma1,mu2,sigma2)');
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.info.hellinger', arity: 2,
  sig: '(p: Tensor, q: Tensor) -> number',
  doc: 'Hellinger 距离：H(p,q) = √(½ Σ (√p_i − √q_i)²)',
  ref: 'Jeffreys 1946；Le Cam 1964（Hellinger 距离）',
  inv: '0 ≤ H ≤ 1；H(p,p) = 0；H²(p,q) ≤ TVD(p,q)',
  impl: (p, q) => {
    const P = probVecOf(p, 'p');
    const Q = probVecOf(q, 'q');
    if (P.d.length !== Q.d.length) throw new ForgeError('E_VALUE', 'hellinger: p 与 q 长度须一致');
    let s = 0;
    for (let i = 0; i < P.d.length; i++) {
      const diff = Math.sqrt(P.d[i] / P.s) - Math.sqrt(Q.d[i] / Q.s);
      s += diff * diff;
    }
    return Math.sqrt(0.5 * s);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.info.tvd', arity: 2,
  sig: '(p: Tensor, q: Tensor) -> number',
  doc: '全变差距离：TVD(p,q) = ½ Σ |p_i − q_i|',
  ref: 'Cover & Thomas, Elements of Information Theory（total variation）',
  inv: '0 ≤ TVD ≤ 1；TVD(p,p) = 0',
  impl: (p, q) => {
    const P = probVecOf(p, 'p');
    const Q = probVecOf(q, 'q');
    if (P.d.length !== Q.d.length) throw new ForgeError('E_VALUE', 'tvd: p 与 q 长度须一致');
    let s = 0;
    for (let i = 0; i < P.d.length; i++) s += Math.abs(P.d[i] / P.s - Q.d[i] / Q.s);
    return 0.5 * s;
  },
});

// ============================================================================
// 八、矩与统计量（10 条：D6-069 … D6-078）
// 口径说明：variance/std/skewness/kurtosis/cov 均为总体矩（÷n，MLE 口径）；
// kurtosis 为超额峰度（m4/m2² − 3，正态为 0）；quantile 用线性插值（numpy 'linear'）。
// ============================================================================

defineOp({
  id: nid(), domain: 6, name: 'prob.stat.mean', arity: 1,
  sig: '(data: Tensor) -> number',
  doc: '样本均值：x̄ = (1/n) Σ x_i',
  ref: 'Kendall & Stuart, The Advanced Theory of Statistics, Vol.1',
  inv: 'min(data) ≤ mean ≤ max(data)',
  impl: (data) => {
    const d = dataOf(data);
    let s = 0;
    for (const v of d) s += v;
    return s / d.length;
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.stat.variance', arity: 1,
  sig: '(data: Tensor) -> number',
  doc: '样本方差（总体口径）：m2 = (1/n) Σ (x_i − x̄)²',
  ref: 'Kendall & Stuart, The Advanced Theory of Statistics, Vol.1',
  inv: 'variance ≥ 0；常数序列为 0',
  impl: (data) => {
    const d = dataOf(data);
    return centralMoments(d)[1];
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.stat.std', arity: 1,
  sig: '(data: Tensor) -> number',
  doc: '样本标准差（总体口径）：√m2',
  ref: 'Kendall & Stuart, The Advanced Theory of Statistics, Vol.1',
  inv: 'std ≥ 0；std² = variance',
  impl: (data) => {
    const d = dataOf(data);
    return Math.sqrt(centralMoments(d)[1]);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.stat.skewness', arity: 1,
  sig: '(data: Tensor) -> number',
  doc: '样本偏度：m3 / m2^{3/2}（总体中心矩口径）',
  ref: 'Kendall & Stuart, The Advanced Theory of Statistics, Vol.1',
  inv: '对称分布 ≈ 0；右偏 > 0，左偏 < 0',
  impl: (data) => {
    const d = dataOf(data);
    const [, m2, m3] = centralMoments(d);
    if (m2 === 0) return 0;
    return m3 / Math.pow(m2, 1.5);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.stat.kurtosis', arity: 1,
  sig: '(data: Tensor) -> number',
  doc: '样本超额峰度：m4 / m2² − 3（正态为 0，均匀为 −1.2）',
  ref: 'Kendall & Stuart, The Advanced Theory of Statistics, Vol.1',
  inv: '重尾 > 0；均匀分布 ≈ −1.2',
  impl: (data) => {
    const d = dataOf(data);
    const [, m2, , m4] = centralMoments(d);
    if (m2 === 0) return 0;
    return m4 / (m2 * m2) - 3;
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.stat.cov', arity: 2,
  sig: '(x: Tensor, y: Tensor) -> number',
  doc: '样本协方差（总体口径）：(1/n) Σ (x_i − x̄)(y_i − ȳ)',
  ref: 'Kendall & Stuart, The Advanced Theory of Statistics, Vol.1',
  inv: 'cov(x,x) = variance(x)；两序列须等长',
  impl: (x, y) => {
    const dx = dataOf(x, 'x');
    const dy = dataOf(y, 'y');
    if (dx.length !== dy.length) throw new ForgeError('E_VALUE', `cov: 两序列须等长（${dx.length} vs ${dy.length}）`);
    const n = dx.length;
    let mx = 0, my = 0;
    for (let i = 0; i < n; i++) { mx += dx[i]; my += dy[i]; }
    mx /= n; my /= n;
    let s = 0;
    for (let i = 0; i < n; i++) s += (dx[i] - mx) * (dy[i] - my);
    return s / n;
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.stat.corr', arity: 2,
  sig: '(x: Tensor, y: Tensor) -> number',
  doc: 'Pearson 相关系数：cov(x,y) / (σ_x·σ_y)',
  ref: 'Kendall & Stuart, The Advanced Theory of Statistics, Vol.1',
  inv: '−1 ≤ corr ≤ 1；corr(x,x) = 1；任一序列为常数时抛 E_NUMERICAL',
  impl: (x, y) => {
    const dx = dataOf(x, 'x');
    const dy = dataOf(y, 'y');
    if (dx.length !== dy.length) throw new ForgeError('E_VALUE', `corr: 两序列须等长（${dx.length} vs ${dy.length}）`);
    const n = dx.length;
    let mx = 0, my = 0;
    for (let i = 0; i < n; i++) { mx += dx[i]; my += dy[i]; }
    mx /= n; my /= n;
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < n; i++) {
      const ax = dx[i] - mx, ay = dy[i] - my;
      sxy += ax * ay; sxx += ax * ax; syy += ay * ay;
    }
    if (sxx === 0 || syy === 0) throw new ForgeError('E_NUMERICAL', 'corr: 序列为常数，相关系数无定义');
    return sxy / Math.sqrt(sxx * syy);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.stat.quantile', arity: 2,
  sig: '(data: Tensor, q: number) -> number',
  doc: '分位数（线性插值，numpy "linear" 口径）：h = (n−1)q，x_⌊h⌋ + (h−⌊h⌋)(x_⌊h⌋+1 − x_⌊h⌋)',
  ref: 'Kendall & Stuart, The Advanced Theory of Statistics, Vol.1（分位数定义）',
  inv: '0 ≤ q ≤ 1；min ≤ quantile ≤ max；关于 q 单调不减',
  impl: (data, q) => {
    const d = dataOf(data);
    unitInterval(q, 'q');
    const xs = Array.from(d).sort((a, b) => a - b);
    const n = xs.length;
    const h = (n - 1) * q;
    const lo = Math.floor(h);
    if (lo >= n - 1) return xs[n - 1];
    return xs[lo] + (h - lo) * (xs[lo + 1] - xs[lo]);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.stat.median', arity: 1,
  sig: '(data: Tensor) -> number',
  doc: '中位数：quantile(data, 0.5)',
  ref: 'Kendall & Stuart, The Advanced Theory of Statistics, Vol.1',
  inv: 'min ≤ median ≤ max；对偶数个取中间两值均值',
  impl: (data) => {
    const d = dataOf(data);
    const xs = Array.from(d).sort((a, b) => a - b);
    const n = xs.length;
    const mid = n >> 1;
    return n % 2 === 1 ? xs[mid] : 0.5 * (xs[mid - 1] + xs[mid]);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.stat.mad', arity: 1,
  sig: '(data: Tensor) -> number',
  doc: '中位数绝对偏差：median(|x_i − median(x)|)',
  ref: 'Kendall & Stuart, The Advanced Theory of Statistics, Vol.1（稳健统计量）',
  inv: 'mad ≥ 0；对离群点稳健',
  impl: (data) => {
    const d = dataOf(data);
    const xs = Array.from(d).sort((a, b) => a - b);
    const n = xs.length;
    const mid = n >> 1;
    const med = n % 2 === 1 ? xs[mid] : 0.5 * (xs[mid - 1] + xs[mid]);
    const ad = Array.from(d, v => Math.abs(v - med)).sort((a, b) => a - b);
    const m2i = ad.length >> 1;
    return ad.length % 2 === 1 ? ad[m2i] : 0.5 * (ad[m2i - 1] + ad[m2i]);
  },
});

// ============================================================================
// 九、贝叶斯与估计（10 条：D6-079 … D6-088）
// ============================================================================

defineOp({
  id: nid(), domain: 6, name: 'prob.bayes.update', arity: 2,
  sig: '(prior: Tensor, likelihood: Tensor) -> Tensor',
  doc: '贝叶斯更新（离散网格）：后验 ∝ 先验 × 似然，返回归一化后验',
  ref: 'DeGroot, Optimal Statistical Decisions；Gelman et al., Bayesian Data Analysis ch.2',
  inv: 'Σ posterior = 1；两序列等长且非负',
  impl: (prior, likelihood) => {
    const P = dataOf(prior, 'prior');
    const L = dataOf(likelihood, 'likelihood');
    if (P.length !== L.length) throw new ForgeError('E_VALUE', `bayes.update: 先验与似然须等长（${P.length} vs ${L.length}）`);
    const out = new Float64Array(P.length);
    let s = 0;
    for (let i = 0; i < P.length; i++) {
      if (P[i] < 0 || L[i] < 0) throw new ForgeError('E_VALUE', 'bayes.update: 先验与似然不能为负');
      out[i] = P[i] * L[i];
      s += out[i];
    }
    if (!(s > 0)) throw new ForgeError('E_VALUE', 'bayes.update: 先验×似然全为 0，后验无定义');
    for (let i = 0; i < out.length; i++) out[i] /= s;
    return new Tensor(out, P.length === 1 ? [1] : [P.length]);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.bayes.posteriorPredictive', arity: 2,
  sig: '(posterior: Tensor[k], likelihood: Tensor[k,m]) -> Tensor[m]',
  doc: '后验预测分布：P(x̃|data) = Σ_j posterior_j · L(j, x̃)，likelihood 行对应参数、列对应观测',
  ref: 'DeGroot, Optimal Statistical Decisions；Gelman et al., BDA ch.1',
  inv: '输出非负且（按列）和为 1 当 likelihood 每行归一化时',
  impl: (posterior, likelihood) => {
    const P = dataOf(posterior, 'posterior');
    if (!(likelihood instanceof Tensor) || likelihood.ndim !== 2 || likelihood.shape[0] !== P.length) {
      throw new ForgeError('E_SHAPE', `posteriorPredictive: likelihood 须为 [${P.length}, m] 的 2D Tensor`);
    }
    const L = dataOf(likelihood, 'likelihood');
    const m = likelihood.shape[1];
    let ps = 0;
    for (const v of P) { if (v < 0) throw new ForgeError('E_VALUE', 'posteriorPredictive: 后验不能为负'); ps += v; }
    const out = new Float64Array(m);
    for (let j = 0; j < P.length; j++) {
      const w = P[j] / ps;
      for (let i = 0; i < m; i++) out[i] += w * L[j * m + i];
    }
    return new Tensor(out, [m]);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.bayes.evidence', arity: 2,
  sig: '(prior: Tensor, likelihood: Tensor) -> number',
  doc: '证据（边际似然）：Z = Σ_i prior_i · likelihood_i，即后验归一化常数',
  ref: 'DeGroot, Optimal Statistical Decisions；Gelman et al., BDA ch.1',
  inv: 'Z ≥ 0；先验已归一化时 Z ∈ [0,1]',
  impl: (prior, likelihood) => {
    const P = dataOf(prior, 'prior');
    const L = dataOf(likelihood, 'likelihood');
    if (P.length !== L.length) throw new ForgeError('E_VALUE', 'evidence: 先验与似然须等长');
    let s = 0;
    for (let i = 0; i < P.length; i++) s += P[i] * L[i];
    return s;
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.bayes.mapNormal', arity: 4,
  sig: '(data: Tensor, priorMu: number, priorSigma: number, sigma: number) -> number',
  doc: '正态均值 MAP（方差已知 σ²，先验 N(μ0,σ0²)）：μ̂ = (μ0/σ0² + n·x̄/σ²)/(1/σ0² + n/σ²)',
  ref: 'Gelman et al., Bayesian Data Analysis ch.2（正态-正态共轭）',
  inv: 'σ0 > 0，σ > 0；后验为_normal_时 MAP = 后验均值',
  impl: (data, priorMu, priorSigma, sigma) => {
    const d = dataOf(data);
    num(priorMu, 'priorMu'); positive(priorSigma, 'priorSigma'); positive(sigma, 'sigma');
    const n = d.length;
    let xbar = 0;
    for (const v of d) xbar += v;
    xbar /= n;
    const prec = 1 / (priorSigma * priorSigma) + n / (sigma * sigma);
    return (priorMu / (priorSigma * priorSigma) + n * xbar / (sigma * sigma)) / prec;
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.bayes.mleNormal', arity: 1,
  sig: '(data: Tensor) -> Tensor[2]',
  doc: '正态 MLE：返回 [μ̂, σ̂]，σ̂ 为 ÷n 口径的总体标准差',
  ref: 'Evans & Rosenthal, Probability and Statistics（MLE）',
  inv: 'σ̂ ≥ 0',
  impl: (data) => {
    const d = dataOf(data);
    const [m1, m2] = centralMoments(d);
    return Tensor.fromFlat([m1, Math.sqrt(m2)], [2]);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.bayes.mleBernoulli', arity: 1,
  sig: '(data: Tensor) -> number',
  doc: '伯努利 MLE：p̂ = x̄（数据须在 [0,1]）',
  ref: 'Johnson, Kotz & Kemp, Discrete Univariate Distributions',
  inv: '0 ≤ p̂ ≤ 1',
  impl: (data) => {
    const d = dataOf(data);
    let s = 0;
    for (const v of d) { unitInterval(v, 'data'); s += v; }
    return s / d.length;
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.bayes.mlePoisson', arity: 1,
  sig: '(data: Tensor) -> number',
  doc: '泊松 MLE：λ̂ = x̄（数据须 ≥ 0）',
  ref: 'Johnson, Kotz & Kemp, Discrete Univariate Distributions',
  inv: 'λ̂ ≥ 0',
  impl: (data) => {
    const d = dataOf(data);
    let s = 0;
    for (const v of d) { nonNeg(v, 'data'); s += v; }
    return s / d.length;
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.bayes.mleExponential', arity: 1,
  sig: '(data: Tensor) -> number',
  doc: '指数 MLE：λ̂ = 1/x̄（数据须 ≥ 0 且均值 > 0）',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1',
  inv: 'λ̂ > 0',
  impl: (data) => {
    const d = dataOf(data);
    let s = 0;
    for (const v of d) { nonNeg(v, 'data'); s += v; }
    const mean = s / d.length;
    if (!(mean > 0)) throw new ForgeError('E_VALUE', 'mleExponential: 数据均值为 0，λ̂ 无定义');
    return 1 / mean;
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.bayes.credibleInterval', arity: null,
  sig: '(mu: number, sigma: number, level: number = 0.95) -> Tensor[2]',
  doc: '正态后验可信区间：[μ ± z_{1−(1−level)/2}·σ]，z 由逆正态分位函数计算',
  ref: 'Gelman et al., Bayesian Data Analysis ch.2（可信区间）',
  inv: 'σ > 0，0 < level < 1；区间关于 μ 对称',
  impl: (mu, sigma, level = 0.95) => {
    num(mu, 'mu'); positive(sigma, 'sigma'); num(level, 'level');
    if (!(level > 0 && level < 1)) throw new ForgeError('E_VALUE', `credibleInterval: level 须在 (0,1)，得到 ${level}`);
    const z = normalQuantileRaw(1 - (1 - level) / 2);
    return Tensor.fromFlat([mu - z * sigma, mu + z * sigma], [2]);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.bayes.conjugateNormal', arity: 5,
  sig: '(n: number, xbar: number, priorMu: number, priorSigma: number, dataSigma: number) -> Tensor[2]',
  doc: '正态-正态共轭后验（数据方差已知 σ²）：返回 [μ_n, σ_n]，σ_n² = 1/(1/σ0² + n/σ²)，μ_n = (μ0/σ0² + n·x̄/σ²)·σ_n²',
  ref: 'Gelman et al., Bayesian Data Analysis ch.2；DeGroot, Optimal Statistical Decisions',
  inv: 'n ≥ 0，σ0 > 0，σ > 0；与闭式解逐位一致',
  impl: (n, xbar, priorMu, priorSigma, dataSigma) => {
    nonNeg(n | 0, 'n'); num(xbar, 'xbar'); num(priorMu, 'priorMu'); positive(priorSigma, 'priorSigma'); positive(dataSigma, 'dataSigma');
    const N = n | 0;
    const prec = 1 / (priorSigma * priorSigma) + N / (dataSigma * dataSigma);
    const muN = (priorMu / (priorSigma * priorSigma) + N * xbar / (dataSigma * dataSigma)) / prec;
    return Tensor.fromFlat([muN, 1 / Math.sqrt(prec)], [2]);
  },
});

// ============================================================================
// 十、采样方法（6 条：D6-089 … D6-094）
// ============================================================================

defineOp({
  id: nid(), domain: 6, name: 'prob.samp.inverseCdf', arity: 4,
  sig: '(seed: number, n: number, xs: Tensor, cdf: Tensor) -> Tensor[n]',
  doc: '通用逆变换采样：对分段线性 CDF（断点 xs/cdf 值）求逆并插值，xs 须严格递增',
  ref: 'Devroye 1986, Non-Uniform Random Variate Generation（inverse CDF method）',
  inv: 'xs 严格递增、cdf 单调不减；同种子可复现',
  impl: (seed, n, xs, cdf) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n');
    const x = dataOf(xs, 'xs');
    const y = dataOf(cdf, 'cdf');
    if (x.length !== y.length) throw new ForgeError('E_SHAPE', `inverseCdf: xs 与 cdf 须等长（${x.length} vs ${y.length}）`);
    if (x.length < 2) throw new ForgeError('E_VALUE', 'inverseCdf: 至少需要 2 个断点');
    for (let i = 1; i < x.length; i++) if (!(x[i] > x[i - 1])) throw new ForgeError('E_VALUE', 'inverseCdf: xs 必须严格递增');
    for (let i = 1; i < y.length; i++) if (y[i] < y[i - 1]) throw new ForgeError('E_VALUE', 'inverseCdf: cdf 值必须单调不减');
    const rng = createRng(seed);
    const out = new Float64Array(n);
    const last = y.length - 1;
    for (let s = 0; s < n; s++) {
      const u = Math.min(Math.max(rng.next(), y[0]), y[last]);
      let lo = 0, hi = last;
      while (lo < hi - 1) { const mid = (lo + hi) >> 1; if (y[mid] <= u) lo = mid; else hi = mid; }
      const dy = y[lo + 1] - y[lo];
      const t = dy > 0 ? (u - y[lo]) / dy : 0;
      out[s] = x[lo] + t * (x[lo + 1] - x[lo]);
    }
    return new Tensor(out, [n]);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.samp.rejection', arity: 2,
  sig: '(seed: number, n: number) -> Tensor[n]',
  doc: '拒绝采样生成 N(0,1)：建议分布 Laplace(0,1)，包络常数 M = √(2e/π) ≈ 1.3155，接受率 ≈ 0.76',
  ref: 'von Neumann 1951（拒绝采样）；Marsaglia & Bray 1964（正态的 Laplace 包络）',
  inv: '接受概率 φ(x)/(M·g(x)) ≤ 1；同种子可复现；矩 → (0,1)',
  impl: (seed, n) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n');
    const rng = createRng(seed);
    const M = Math.sqrt(2 * Math.E / Math.PI);
    const out = new Float64Array(n);
    let i = 0;
    while (i < n) {
      const u = rng.next();
      // Laplace(0,1) 逆变换
      const x = u < 0.5 ? Math.log(2 * u) : -Math.log(2 * (1 - u));
      const v = rng.next();
      // φ(x) / g(x) = √(2/π)·e^{−x²/2+|x|} ≤ M（|x|=1 处取等）
      const ratio = Math.sqrt(2 / Math.PI) * Math.exp(-0.5 * x * x + Math.abs(x));
      if (v * M < ratio) out[i++] = x;
    }
    return new Tensor(out, [n]);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.samp.metropolis', arity: 3,
  sig: '(seed: number, n: number, step: number) -> Tensor[n]',
  doc: '一维随机游走 Metropolis（目标 N(0,1)）：提议 y = x + step·N(0,1)，接受率 min(1, e^{−(y²−x²)/2})；前 10% 作 burn-in',
  ref: 'Metropolis, Rosenbluth, Rosenbluth, Teller & Teller 1953',
  inv: 'step > 0；同种子可复现；矩 → (0,1)（容差 10%）',
  impl: (seed, n, step) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n'); positive(step, 'step');
    const rng = createRng(seed);
    const g = normalGen(rng);
    const burn = Math.max(1, Math.floor(n / 10));
    let x = 0;
    const out = new Float64Array(n);
    for (let i = 0; i < n + burn; i++) {
      const y = x + step * g();
      // 目标对数密度之差（正态化常数抵消）
      const logRatio = -0.5 * (y * y - x * x);
      let u = rng.next();
      if (u < 1e-300) u = 1e-300;
      if (Math.log(u) < logRatio) x = y;
      if (i >= burn) out[i - burn] = x;
    }
    return new Tensor(out, [n]);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.samp.boxMuller', arity: 2,
  sig: '(seed: number, n: number) -> Tensor[n]',
  doc: 'Box-Muller 标准正态：u1,u2 ~ U(0,1)，r = √(−2 ln u1)，z1 = r·cos(2πu2)，z2 = r·sin(2πu2)',
  ref: 'Box & Muller, Annals of Mathematical Statistics 29(2), 1958',
  inv: '同种子完全可复现；大样本矩 → (0,1)',
  impl: (seed, n) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n');
    const rng = createRng(seed);
    const out = new Float64Array(n);
    let i = 0;
    while (i < n) {
      let u1 = rng.next();
      if (u1 < 1e-300) u1 = 1e-300;
      const u2 = rng.next();
      const r = Math.sqrt(-2 * Math.log(u1));
      out[i++] = r * Math.cos(2 * Math.PI * u2);
      if (i < n) out[i++] = r * Math.sin(2 * Math.PI * u2);
    }
    return new Tensor(out, [n]);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.samp.marsagliaPolar', arity: 2,
  sig: '(seed: number, n: number) -> Tensor[n]',
  doc: 'Marsaglia 极坐标法标准正态：单位圆盘内取点 (u,v)，s = u²+v²，f = √(−2 ln s/s)，输出 (u·f, v·f)',
  ref: 'Marsaglia & Bray, CACM 7(1), 1964',
  inv: '同种子完全可复现；大样本矩 → (0,1)',
  impl: (seed, n) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n');
    const rng = createRng(seed);
    const out = new Float64Array(n);
    let i = 0;
    while (i < n) {
      const u = rng.range(-1, 1), v = rng.range(-1, 1);
      const s = u * u + v * v;
      if (s >= 1 || s === 0) continue;
      const f = Math.sqrt(-2 * Math.log(s) / s);
      out[i++] = u * f;
      if (i < n) out[i++] = v * f;
    }
    return new Tensor(out, [n]);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.samp.stratified', arity: 2,
  sig: '(seed: number, n: number) -> Tensor[n]',
  doc: '分层抽样 U(0,1)：第 i 层 [(i/n), (i+1)/n) 内取一个均匀点，(i+U)/n',
  ref: 'Cochran, Sampling Techniques §5.2（分层抽样/JIPS）',
  inv: '每层恰一个样本；均值估计量方差 = 1/(12n³)（粗蒙特卡洛 1/(12n) 的 1/n²）',
  impl: (seed, n) => {
    num(seed, 'seed'); nonNeg(n | 0, 'n');
    if (n < 1) throw new ForgeError('E_VALUE', 'stratified: n 须 ≥ 1');
    const rng = createRng(seed);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = (i + rng.next()) / n;
    return new Tensor(out, [n]);
  },
});

// ============================================================================
// 十一、拟合检验（4 条：D6-095 … D6-098）
// ============================================================================

defineOp({
  id: nid(), domain: 6, name: 'prob.test.ks', arity: null,
  sig: "(data: Tensor, dist: 'normal'|'uniform'|'exponential', ...params) -> {D, pValue, n}",
  doc: '一样本 Kolmogorov-Smirnov 检验：D = sup|F_n − F|，p 值用 Massey 1951 渐近式 Q(λ)，λ = (√n+0.12+0.11/√n)·D',
  ref: 'Massey 1951（Kolmogorov 分布临界值）；Stephens 1974（EDF 统计量修正）',
  inv: '0 ≤ D ≤ 1；同分布假设下 p 值近似均匀',
  impl: (data, dist, ...ps) => {
    const d = dataOf(data);
    const n = d.length;
    const sorted = Array.from(d).sort((a, b) => a - b);
    let F;
    if (dist === 'normal') {
      const mu = num(ps[0], 'mu'), sigma = positive(ps[1], 'sigma');
      F = (x) => normCdf((x - mu) / sigma);
    } else if (dist === 'uniform') {
      const a = num(ps[0], 'a'), b = num(ps[1], 'b');
      if (!(b > a)) throw new ForgeError('E_VALUE', 'ks: uniform 需要 b > a');
      F = (x) => (x < a ? 0 : x > b ? 1 : (x - a) / (b - a));
    } else if (dist === 'exponential') {
      const rate = positive(ps[0], 'rate');
      F = (x) => (x < 0 ? 0 : -Math.expm1(-rate * x));
    } else {
      throw new ForgeError('E_VALUE', `ks: 不支持的参考分布 "${dist}"（支持 normal / uniform / exponential）`);
    }
    let D = 0;
    for (let i = 0; i < n; i++) {
      const f = F(sorted[i]);
      // 经验 CDF 在 x_(i) 处为 (i+1)/n，左极限为 i/n
      D = Math.max(D, (i + 1) / n - f, f - i / n);
    }
    return { D, pValue: ksPValue(D, n), n };
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.test.ks2Sample', arity: 2,
  sig: '(x: Tensor, y: Tensor) -> {D, pValue, n1, n2}',
  doc: '两样本 Kolmogorov-Smirnov 检验：D = sup|F1 − F2|，有效样本量 ne = n1·n2/(n1+n2)，p 值同 Massey 渐近式',
  ref: 'Smirnov 1939；Massey 1951（两样本 KS 分布）',
  inv: '0 ≤ D ≤ 1；同分布假设下 p 值近似均匀',
  impl: (x, y) => {
    const dx = Array.from(dataOf(x, 'x')).sort((a, b) => a - b);
    const dy = Array.from(dataOf(y, 'y')).sort((a, b) => a - b);
    const n1 = dx.length, n2 = dy.length;
    const all = [...new Set([...dx, ...dy])].sort((a, b) => a - b);
    let D = 0;
    let i = 0, j = 0;
    for (const v of all) {
      while (i < n1 && dx[i] <= v) i++;
      while (j < n2 && dy[j] <= v) j++;
      D = Math.max(D, Math.abs(i / n1 - j / n2));
    }
    const ne = (n1 * n2) / (n1 + n2);
    return { D, pValue: ksPValue(D, ne), n1, n2 };
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.test.chi2Gof', arity: 2,
  sig: '(observed: Tensor, expected: Tensor) -> {statistic, df, pValue}',
  doc: '卡方拟合优度检验：χ² = Σ (O_i − n·p_i)²/(n·p_i)，df = K−1，p = Q(df/2, χ²/2)（上不完全伽马）',
  ref: 'Pearson 1900（卡方准则）；Cochran 1952（GOF 适用条件）',
  inv: 'χ² ≥ 0；p ∈ [0,1]；观测与期望概率等长',
  impl: (observed, expected) => {
    const O = dataOf(observed, 'observed');
    const { d: E, s } = probVecOf(expected, 'expected');
    if (O.length !== E.length) throw new ForgeError('E_VALUE', `chi2Gof: 观测与期望须等长（${O.length} vs ${E.length}）`);
    let n = 0;
    for (const v of O) { nonNeg(v, 'observed'); n += v; }
    if (!(n > 0)) throw new ForgeError('E_VALUE', 'chi2Gof: 观测总数须 > 0');
    let stat = 0;
    for (let i = 0; i < O.length; i++) {
      const ei = n * (E[i] / s);
      if (ei <= 0) throw new ForgeError('E_VALUE', 'chi2Gof: 期望频数须 > 0');
      stat += (O[i] - ei) ** 2 / ei;
    }
    const df = O.length - 1;
    const pValue = df >= 1 ? gammaq(df / 2, stat / 2) : NaN;
    return { statistic: stat, df, pValue };
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.test.jarqueBera', arity: 1,
  sig: '(data: Tensor) -> {statistic, pValue, skewness, kurtosis}',
  doc: 'Jarque-Bera 正态性检验：JB = (n/6)(S² + K²/4)，S 为偏度、K 为超额峰度；p = e^{−JB/2}（χ²₂ 生存函数）',
  ref: 'Jarque & Bera, International Statistical Review 55(2), 1987',
  inv: 'JB ≥ 0；正态样本 JB → χ²₂；p ∈ [0,1]',
  impl: (data) => {
    const d = dataOf(data);
    const n = d.length;
    if (n < 4) throw new ForgeError('E_VALUE', 'jarqueBera: 样本量须 ≥ 4');
    const [, m2, m3, m4] = centralMoments(d);
    if (m2 === 0) return { statistic: 0, pValue: 1, skewness: 0, kurtosis: 0 };
    const S = m3 / Math.pow(m2, 1.5);
    const K = m4 / (m2 * m2) - 3;
    const jb = (n / 6) * (S * S + K * K / 4);
    return { statistic: jb, pValue: Math.exp(-jb / 2), skewness: S, kurtosis: K };
  },
});

// ============================================================================
// 十二、数值工具（4 条：D6-099 … D6-102）
// ============================================================================

defineOp({
  id: nid(), domain: 6, name: 'prob.util.trapz', arity: null,
  sig: '(y: Tensor, x?: Tensor) -> number',
  doc: '复合梯形积分：Σ ½(y_i + y_{i−1})·Δx；省略 x 时按单位间距',
  ref: '数值分析复合梯形公式（误差 O(h²)）',
  inv: 'x 省略或与 y 等长且严格递增',
  impl: (y, x = undefined) => {
    const ys = dataOf(y, 'y');
    let xs = null;
    if (x !== undefined) {
      if (!(x instanceof Tensor)) throw new ForgeError('E_TYPE', `x: 期望 Tensor，得到 ${typeof x}`);
      xs = dataOf(x, 'x');
      if (xs.length !== ys.length) throw new ForgeError('E_SHAPE', `trapz: x 与 y 须等长（${xs.length} vs ${ys.length}）`);
      for (let i = 1; i < xs.length; i++) if (!(xs[i] > xs[i - 1])) throw new ForgeError('E_VALUE', 'trapz: x 必须严格递增');
    }
    let s = 0;
    for (let i = 1; i < ys.length; i++) s += 0.5 * (ys[i] + ys[i - 1]) * (xs ? xs[i] - xs[i - 1] : 1);
    return s;
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.util.normalQuantile', arity: 1,
  sig: '(p: number) -> number',
  doc: '逆正态分位函数 Φ⁻¹(p)：Acklam 有理逼近 + Halley 修正，精度 ~1e-12',
  ref: 'P.J. Acklam（Beasley-Springer-Moro 族逆正态算法）；Halley 迭代',
  inv: '0 < p < 1；单调递增；normalQuantile(Φ(x)) ≈ x',
  impl: (p) => {
    num(p, 'p');
    if (!(p > 0 && p < 1)) throw new ForgeError('E_VALUE', `normalQuantile: p 须在 (0,1)，得到 ${p}`);
    return normalQuantileRaw(p);
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.util.distSummary', arity: null,
  sig: "(dist: string, ...params) -> {dist, params, mean, variance, entropy}",
  doc: '分布摘要（均值/方差/熵，nat）：支持 normal/uniform/exponential/laplace/logistic/gamma/beta/pareto/rayleigh/weibull/chi2',
  ref: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions（各分布矩与熵闭式）',
  inv: 'variance ≥ 0；熵对集中分布可为负（微分熵性质）',
  impl: (dist, ...ps) => {
    let mean = 0, variance = 0, entropy = 0;
    if (dist === 'normal') {
      const mu = num(ps[0], 'mu'), sigma = positive(ps[1], 'sigma');
      mean = mu; variance = sigma * sigma; entropy = 0.5 * Math.log(2 * Math.PI * Math.E * variance);
    } else if (dist === 'uniform') {
      const a = num(ps[0], 'a'), b = num(ps[1], 'b');
      if (!(b > a)) throw new ForgeError('E_VALUE', 'distSummary: uniform 需要 b > a');
      mean = 0.5 * (a + b); variance = (b - a) ** 2 / 12; entropy = Math.log(b - a);
    } else if (dist === 'exponential') {
      const rate = positive(ps[0], 'rate');
      mean = 1 / rate; variance = 1 / (rate * rate); entropy = 1 - Math.log(rate);
    } else if (dist === 'laplace') {
      const mu = num(ps[0], 'mu'), b = positive(ps[1], 'b');
      mean = mu; variance = 2 * b * b; entropy = 1 + Math.log(2 * b);
    } else if (dist === 'logistic') {
      const mu = num(ps[0], 'mu'), s = positive(ps[1], 's');
      mean = mu; variance = Math.PI * Math.PI * s * s / 3; entropy = 2 + Math.log(s);
    } else if (dist === 'gamma') {
      const k = positive(ps[0], 'shape'), rate = positive(ps[1], 'rate');
      mean = k / rate; variance = k / (rate * rate);
      entropy = k - Math.log(rate) + lgamma(k) + (1 - k) * digamma(k);
    } else if (dist === 'beta') {
      const a = positive(ps[0], 'a'), b = positive(ps[1], 'b');
      mean = a / (a + b); variance = a * b / ((a + b) ** 2 * (a + b + 1));
      entropy = lgamma(a) + lgamma(b) - lgamma(a + b)
        - (a - 1) * digamma(a) - (b - 1) * digamma(b) + (a + b - 2) * digamma(a + b);
    } else if (dist === 'pareto') {
      const xm = positive(ps[0], 'xm'), alpha = num(ps[1], 'alpha');
      if (!(alpha > 2)) throw new ForgeError('E_VALUE', `distSummary: pareto 方差要求 alpha > 2，得到 ${alpha}`);
      mean = alpha * xm / (alpha - 1);
      variance = alpha * xm * xm / ((alpha - 1) ** 2 * (alpha - 2));
      entropy = Math.log(xm / alpha) + 1 + 1 / alpha;
    } else if (dist === 'rayleigh') {
      const sigma = positive(ps[0], 'sigma');
      mean = sigma * Math.sqrt(Math.PI / 2);
      variance = (4 - Math.PI) / 2 * sigma * sigma;
      entropy = 1 + Math.log(sigma / Math.SQRT2) + EULER_GAMMA / 2;
    } else if (dist === 'weibull') {
      const k = positive(ps[0], 'shape'), scale = positive(ps[1], 'scale');
      const g1 = Math.exp(lgamma(1 + 1 / k));
      mean = scale * g1;
      variance = scale * scale * (Math.exp(lgamma(1 + 2 / k)) - g1 * g1);
      entropy = EULER_GAMMA * (1 - 1 / k) + Math.log(scale / k) + 1;
    } else if (dist === 'chi2') {
      const df = positive(ps[0], 'df');
      mean = df; variance = 2 * df;
      entropy = Math.log(2 * Math.exp(lgamma(df / 2))) + (1 - df / 2) * digamma(df / 2) + df / 2;
    } else {
      throw new ForgeError('E_VALUE', `distSummary: 不支持的分布 "${dist}"`);
    }
    return { dist, params: ps.slice(), mean, variance, entropy };
  },
});
defineOp({
  id: nid(), domain: 6, name: 'prob.util.sampleMoments', arity: 1,
  sig: '(data: Tensor) -> {mean, variance, skewness, kurtosis}',
  doc: '样本矩摘要（总体口径）：均值、方差、偏度 m3/m2^{3/2}、超额峰度 m4/m2² − 3，供与理论矩对比',
  ref: 'Kendall & Stuart, The Advanced Theory of Statistics, Vol.1',
  inv: '方差为 0 时 skewness/kurtosis 记为 0',
  impl: (data) => {
    const d = dataOf(data);
    const [m1, m2, m3, m4] = centralMoments(d);
    return {
      mean: m1,
      variance: m2,
      skewness: m2 === 0 ? 0 : m3 / Math.pow(m2, 1.5),
      kurtosis: m2 === 0 ? 0 : m4 / (m2 * m2) - 3,
    };
  },
});

// ============================================================================
// 底部自检：本域必须恰好注册 102 条指令，否则导入即抛错。
// ============================================================================
if (opsByDomain(6).length !== 102) {
  throw new ForgeError('E_VALUE', `D6 域应注册 102 条微指令，实际 ${opsByDomain(6).length}`);
}
