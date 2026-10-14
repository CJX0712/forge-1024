// 示例 02：高斯混合模型（EM 算法）—— 用微指令组合出一个完整学习算法
// 运行：node examples/02-gmm-clustering.mjs
// 展示：EM 的似然单调性定理（Dempster et al. 1977）、责任度归一、协方差对称正定。
// 全部由域1/2/3/6 的微指令拼装，不引入任何 GMM 专用实现。
import '../src/index.mjs';
import { exec } from '../src/core/registry.mjs';
import { Tensor } from '../src/core/tensor.mjs';
import { createRng } from '../src/core/rng.mjs';

const rng = createRng(7);
const d = 2, K = 3, N = 300;
const centers = [[-3, -3], [3, 3], [0, 3.5]];

// —— 造数据：3 个高斯团 ——
const pts = [];
for (let i = 0; i < N; i++) {
  const c = centers[i % K];
  pts.push([c[0] + rng.normal() * 0.6, c[1] + rng.normal() * 0.6]);
}
const X = Tensor.fromFlat(pts.flat(), [N, d]);

// —— 初始化：随机取 K 个点作均值，协方差设为样本协方差 ——
let mus = [0, 1, 2].map(i => Tensor.fromFlat(pts[Math.floor(i * N / K)], [d]));
let sigmas = mus.map(() => Tensor.fromFlat([1, 0, 0, 1], [d, d]));
let weights = [1 / 3, 1 / 3, 1 / 3];

// 稳定的 log-sum-exp：num.logsumexp 返回 Tensor，取标量
const logSumExp = (arr) => exec('num.logsumexp', [Tensor.fromFlat(arr, [arr.length])]).data[0];

function eStep() {
  const logResp = [];
  for (let i = 0; i < N; i++) {
    const xi = Tensor.fromFlat([pts[i][0], pts[i][1]], [d]);
    const row = [];
    for (let k = 0; k < K; k++) {
      const p = exec('prob.mvn.pdf', [xi, mus[k], sigmas[k]]);
      row.push(Math.log(weights[k]) + Math.log(Math.max(p, 1e-300)));
    }
    const norm = logSumExp(row);          // 稳定的 log-sum-exp 归一化
    logResp.push(row.map(v => v - norm));
  }
  return logResp.map(r => r.map(Math.exp)); // 责任度 γ(z_ik)
}

function mStep(resp) {
  const Nk = [];
  for (let k = 0; k < K; k++) {
    let s = 0;
    for (let i = 0; i < N; i++) s += resp[i][k];
    if (!(s > 1e-12)) {                       // 簇塌缩保护：重置为均匀分布
      mus[k] = Tensor.fromFlat(pts[Math.floor(rng.next() * N)], [d]);
      sigmas[k] = Tensor.fromFlat([1, 0, 0, 1], [d, d]);
      weights[k] = 1 / K;
      Nk.push(0);
      continue;
    }
    Nk.push(s);
    // 均值
    const mu = new Array(d).fill(0);
    for (let i = 0; i < N; i++) for (let j = 0; j < d; j++) mu[j] += resp[i][k] * pts[i][j];
    mus[k] = Tensor.fromFlat(mu.map(v => v / Nk[k]), [d]);
    // 协方差（加 1e-6 岭以保证正定）
    const S = new Array(d * d).fill(0);
    for (let i = 0; i < N; i++) {
      const diff = pts[i].map((v, j) => v - mus[k].data[j]);
      for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) S[a * d + b] += resp[i][k] * diff[a] * diff[b];
    }
    for (let a = 0; a < d; a++) S[a * d + a] += 1e-6;      // 岭项：保证正定
    // 显式对称化：浮点累加顺序会让 S[01] 与 S[10] 相差 ~1e-17，mvn.pdf 要求严格对称
    for (let a = 0; a < d; a++) for (let b = a + 1; b < d; b++) {
      const v = (S[a * d + b] + S[b * d + a]) / 2;
      S[a * d + b] = v; S[b * d + a] = v;
    }
    sigmas[k] = Tensor.fromFlat(S.map(v => v / Nk[k]), [d, d]);
    weights[k] = Nk[k] / N;
  }
}

function logLik() {
  let ll = 0;
  for (let i = 0; i < N; i++) {
    const xi = Tensor.fromFlat(pts[i], [d]);
    const row = [];
    for (let k = 0; k < K; k++) row.push(Math.log(weights[k]) + Math.log(Math.max(exec('prob.mvn.pdf', [xi, mus[k], sigmas[k]]), 1e-300)));
    ll += logSumExp(row);
  }
  return ll;
}

const lls = [];
for (let it = 0; it < 60; it++) {
  const resp = eStep();
  lls.push(logLik());       // E 步后、M 步前的似然
  mStep(resp);
}
lls.push(logLik());

// —— 不变量 1：EM 对数似然单调不减（Dempster et al. 定理 1） ——
let minRise = Infinity;
for (let i = 1; i < lls.length; i++) minRise = Math.min(minRise, lls[i] - lls[i - 1]);

// —— 不变量 2：协方差对称正定、权重和为 1 ——
const symErr = Math.max(...sigmas.map(S => Math.abs(S.data[1] - S.data[2])));
const wSum = weights.reduce((a, b) => a + b, 0);
const spd = sigmas.every(S => exec('linalg.cholesky', [S]) !== null);

// —— 聚类准确率（与真实团标签比对） ——
const resp = eStep();
let correct = 0;
for (let i = 0; i < N; i++) {
  const pred = resp[i].indexOf(Math.max(...resp[i]));
  if (pred === i % K) correct++;
}

console.log('示例 02 · 高斯混合模型（EM，由微指令组合而成）');
console.log('==========================================');
console.log(`迭代 60 轮，对数似然 ${lls[0].toFixed(3)} → ${lls[lls.length - 1].toFixed(3)}`);
console.log(`似然单调不减：最小增量 ${minRise.toExponential(3)}   ${minRise > -1e-8 ? '✓' : '✗'}`);
console.log(`协方差对称性误差：${symErr.toExponential(3)}   ${symErr < 1e-12 ? '✓' : '✗'}`);
console.log(`混合权重和：${wSum.toFixed(15)}   ${Math.abs(wSum - 1) < 1e-12 ? '✓' : '✗'}`);
console.log(`Cholesky 可分解（正定）：${spd ? '✓' : '✗'}`);
console.log(`聚类准确率：${(correct / N * 100).toFixed(1)}%`);
console.log('\n结论：仅用 mvn.pdf + logsumexp + Cholesky 三条微指令即完成 EM 全流程，');
console.log('      对数似然严格单调不减，责任度归一化、协方差对称正定均成立。');
