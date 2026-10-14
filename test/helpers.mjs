// 测试公共助手：断言、数据生成、数值比较。
// 所有随机数据必须经 createRng(seed) 生成，保证测试可复现。
import assert from 'node:assert/strict';
import { Tensor, createRng } from '../src/core/index.mjs';

export { assert };

export function approx(a, b, eps = 1e-9) {
  if (Number.isNaN(a) && Number.isNaN(b)) return true; // NaN 安全（域外输入双方均为 NaN 视为一致）
  return Math.abs(a - b) <= eps * Math.max(1, Math.abs(a), Math.abs(b));
}

export function relErr(a, b) {
  const d = Math.max(1e-300, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) / d;
}

// 张量 vs 期望（嵌套数组/Tensor/标量）逐元素近似比较
export function tensorClose(actual, expected, eps = 1e-9, msg = '') {
  const exp = expected instanceof Tensor ? expected.toArray() : expected;
  const act = actual instanceof Tensor ? actual.toArray() : actual;
  assert.deepEqual(act.length, Array.isArray(exp) ? exp.length : 1, `长度不符 ${msg}`);
  const flatA = flatten(act), flatE = flatten(Array.isArray(exp) ? exp : [exp]);
  assert.equal(flatA.length, flatE.length, `元素数不符: ${flatA.length} vs ${flatE.length} ${msg}`);
  let maxErr = 0;
  for (let i = 0; i < flatA.length; i++) {
    const e = Math.abs(flatA[i] - flatE[i]);
    if (e > maxErr) maxErr = e;
    assert.ok(approx(flatA[i], flatE[i], eps), `第${i}项: ${flatA[i]} != ${flatE[i]} (err=${e}) ${msg}`);
  }
  return maxErr;
}

export function flatten(arr) {
  const out = [];
  const walk = (a) => { for (const v of a) Array.isArray(v) ? walk(v) : out.push(v); };
  walk(arr);
  return out;
}

export function maxAbsDiff(a, b) {
  const ta = a instanceof Tensor ? a.data : Float64Array.from(a);
  const tb = b instanceof Tensor ? b.data : Float64Array.from(b);
  assert.equal(ta.length, tb.length, 'maxAbsDiff 长度不符');
  let m = 0;
  for (let i = 0; i < ta.length; i++) m = Math.max(m, Math.abs(ta[i] - tb[i]));
  return m;
}

// —— 标准测试数据（全部种子化，可复现） ——

// 高斯团 blobs：返回 {X: Tensor[n,d], y: Tensor[n]}
export function blobs({ n = 60, centers = [[0, 0], [5, 5]], std = 0.6, seed = 42, dim = 2 }) {
  const rng = createRng(seed);
  const k = centers.length;
  const per = Math.floor(n / k);
  const rows = [], ys = [];
  for (let c = 0; c < k; c++) {
    const count = c === k - 1 ? n - per * (k - 1) : per;
    for (let i = 0; i < count; i++) {
      const row = [];
      for (let d = 0; d < dim; d++) row.push(centers[c][d] + std * rng.normal());
      rows.push(row); ys.push(c);
    }
  }
  return { X: Tensor.fromNested(rows), y: Tensor.fromFlat(ys, [ys.length]) };
}

// 一维回归数据：y = 2x + 1 + 噪声
export function linData(n = 50, seed = 7, noise = 0.1) {
  const rng = createRng(seed);
  const rows = [], ys = [];
  for (let i = 0; i < n; i++) {
    const x = rng.range(-3, 3);
    rows.push([x]); ys.push(2 * x + 1 + noise * rng.normal());
  }
  return { X: Tensor.fromNested(rows), y: Tensor.fromFlat(ys, [n]) };
}

// XOR 数据（4 个高斯团）
export function xorData(n = 200, seed = 3) {
  const rng = createRng(seed);
  const centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]];
  const labels = [0, 0, 1, 1];
  const rows = [], ys = [];
  for (let i = 0; i < n; i++) {
    const c = i % 4;
    rows.push([centers[c][0] + 0.35 * rng.normal(), centers[c][1] + 0.35 * rng.normal()]);
    ys.push(labels[c]);
  }
  return { X: Tensor.fromNested(rows), y: Tensor.fromFlat(ys, [n]) };
}

// 线性可分二分类
export function linSep(n = 100, seed = 11) {
  const rng = createRng(seed);
  const rows = [], ys = [];
  for (let i = 0; i < n; i++) {
    const y = i % 2 === 0 ? 1 : -1;
    rows.push([rng.range(-2, 2), y * 1.5 + 0.3 * rng.normal()]);
    ys.push(y);
  }
  return { X: Tensor.fromNested(rows), y: Tensor.fromFlat(ys, [n]) };
}

// 随机稠密矩阵（元素 ~ N(0,1)）
export function randMat(m, n, seed = 5) {
  const rng = createRng(seed);
  return Tensor.fromFlat(rng.shuffle(Array.from({ length: m * n }, () => rng.normal())), [m, n]);
}

// 随机对称正定矩阵 AᵀA + n·I
export function randSPD(n, seed = 5, ridge = 1) {
  const A = randMat(n, n, seed);
  const At = A.T();
  const out = Tensor.zeros([n, n]);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let s = 0;
    for (let k = 0; k < n; k++) s += At.get2(i, k) * A.get2(k, j);
    out.set2(i, j, s + (i === j ? ridge : 0));
  }
  return out;
}
