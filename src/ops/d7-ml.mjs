// 域7 机器学习 ML（102 条微指令）
// 对标：scikit-learn 经典估计器族（linear_model / naive_bayes / tree / ensemble /
//       svm / neighbors / cluster / decomposition / preprocessing / feature_selection / metrics），
//       统计侧参考 Hastie, Tibshirani & Friedman《Elements of Statistical Learning》2nd ed. (2009)。
// 组织：A 回归 12 / B 线性分类 10 / C 判别与朴素贝叶斯 8 / D 树与集成 14 / E SVM 与核 10 /
//       F 近邻 4 / G 聚类 13 / H 降维 8 / I 预处理 9 / J 特征选择 3 / K 指标 7 / L 工具 4 = 102。
// 约定：
//   1) fit 返回可 JSON 序列化的纯对象 state（内部数组一律普通 Array，不用 Float64Array）；
//   2) predict 统一签名 (state, X)；
//   3) 预处理类指令返回 {X: Tensor, ...参数}，便于逆变换与管线复用；
//   4) 随机数一律来自 createRng(seed)（PCG32），禁止 Math.random；
//   5) 数值稳健：log/除零 eps 保护、softmax 减最大值、协方差加 ridge 再 Cholesky。
// 依赖：仅 src/core 与 src/ops/d3-linalg.mjs（D3 <- D7 分层）。
import { defineOp, opsByDomain, exec } from '../core/registry.mjs';
import { ForgeError } from '../core/errors.mjs';
import { Tensor, expectTensor, expectNumber, createRng } from '../core/index.mjs';
import '../ops/d3-linalg.mjs';

const D = 7;
let _s = 0;
const nid = () => `D${D}-${String(++_s).padStart(3, '0')}`;

// =====================================================================
// 零、内部小工具（非指令）
// =====================================================================
const _EPS = 1e-12;

function _asMat(X, who = 'X') {
  expectTensor(X, who);
  if (X.ndim === 1) return X.reshape(X.shape[0], 1);
  if (X.ndim !== 2) throw new ForgeError('E_SHAPE', `${who} 需为 1D/2D Tensor，当前 [${X.shape}]`);
  return X;
}

function _asVec(v, who = 'y') {
  expectTensor(v, who);
  if (v.ndim === 2 && (v.shape[0] === 1 || v.shape[1] === 1)) return v.flatten();
  if (v.ndim !== 1) throw new ForgeError('E_SHAPE', `${who} 需为向量 Tensor，当前 [${v.shape}]`);
  return v;
}

// Tensor[n,d] -> number[][]
function _rows(X) {
  const [n, d] = X.shape;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const r = new Array(d);
    for (let j = 0; j < d; j++) r[j] = X.data[i * d + j];
    out[i] = r;
  }
  return out;
}

// number[][] -> Tensor[n,d]
function _T2(rows) {
  const n = rows.length;
  if (n === 0) throw new ForgeError('E_SHAPE', '空矩阵不可构造 Tensor');
  const d = rows[0].length;
  const flat = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    if (rows[i].length !== d) throw new ForgeError('E_SHAPE', '矩阵行长度不一致');
    for (let j = 0; j < d; j++) flat[i * d + j] = rows[i][j];
  }
  return new Tensor(flat, [n, d]);
}

const _T1 = (a) => new Tensor(Float64Array.from(a), [a.length]);
const _flat = (t) => Array.from(t.data);
const _dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const _sum = (a) => { let s = 0; for (const v of a) s += v; return s; };
const _mean = (a) => (a.length ? _sum(a) / a.length : 0);
const _clone = (a) => a.slice();

function _dist2(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const t = a[i] - b[i]; s += t * t; }
  return s;
}
function _dist1(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s;
}

// 列均值 / 列样本标准差
function _colMean(R, d) {
  const m = new Array(d).fill(0);
  for (const r of R) for (let j = 0; j < d; j++) m[j] += r[j];
  for (let j = 0; j < d; j++) m[j] /= R.length;
  return m;
}
function _center(R, m) { return R.map((r) => r.map((v, j) => v - m[j])); }

// 线性求解：优先 Cholesky（SPD），失败回退 LU；ridge 加到对角
function _solve(M, b, ridge = 0) {
  const p = M.length;
  const A = M.map((row, i) => row.map((v, j) => v + (i === j ? ridge : 0)));
  try {
    return _flat(exec('linalg.solveSPD', [_T2(A), _T1(b)]));
  } catch {
    return _flat(exec('linalg.solve', [_T2(A), _T1(b)]));
  }
}

// 最小二乘（QR 路径，秩亏回退 SVD 伪逆）
function _lstsq(A, b) {
  const r = exec('linalg.lstsq', [_T2(A), _T1(b)]);
  return _flat(r.x);
}

// 对称特征分解：返回 {values: number[], vectors: number[][]}（vectors[k] = 第 k 个特征向量，按列取）
function _eigSym(M) {
  const r = exec('linalg.eigenSym', [_T2(M)]);
  const values = Array.from(r.values.data);
  const V = _rows(r.vectors);
  const p = values.length;
  const vectors = new Array(p);
  for (let k = 0; k < p; k++) {
    const v = new Array(p);
    for (let i = 0; i < p; i++) v[i] = V[i][k];
    vectors[k] = v;
  }
  return { values, vectors };
}

// 矩阵逆（回退：岭正则后再求逆）
function _inv(M) {
  try { return _rows(exec('linalg.inverse', [_T2(M)])); } catch { return null; }
}

function _choleskyLogDet(M) {
  try {
    const L = exec('linalg.cholesky', [_T2(M)]);
    let ld = 0;
    for (let i = 0; i < M.length; i++) {
      const v = L.data[i * M.length + i];
      ld += 2 * Math.log(Math.max(v, 1e-300));
    }
    return ld;
  } catch { return 0; }
}

// 稳定 sigmoid（分象限避免 exp 溢出）
function _sigmoid(z) {
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

// 稳定 softmax（减最大值），in-place 写入 out，返回行和
function _softmaxRow(z, out) {
  let mx = -Infinity;
  for (const v of z) if (v > mx) mx = v;
  let s = 0;
  for (let k = 0; k < z.length; k++) { const e = Math.exp(z[k] - mx); out[k] = e; s += e; }
  if (!(s > 0)) { for (let k = 0; k < z.length; k++) out[k] = 1 / z.length; return 1; }
  for (let k = 0; k < z.length; k++) out[k] /= s;
  return s;
}

// 二分类标签规范化：返回 {neg, pos, t: 0/1 数组}
function _binLabels(ys) {
  const uniq = Array.from(new Set(ys)).sort((a, b) => a - b);
  if (uniq.length !== 2) {
    throw new ForgeError('E_VALUE', `二分类需要恰好 2 个不同标签，实际 ${uniq.length} 个`);
  }
  const neg = uniq[0], pos = uniq[1];
  return { neg, pos, t: ys.map((v) => (v === pos ? 1 : 0)) };
}

// 类别清单（升序去重）
const _classesOf = (ys) => Array.from(new Set(ys)).sort((a, b) => a - b);

// 任务自动判别：整数标签且类别数不多 → 分类
function _isClassify(ys) {
  const uniq = new Set(ys);
  if (uniq.size < 2) return true;
  if (uniq.size > Math.max(20, Math.floor(ys.length * 0.5))) return false;
  for (const v of uniq) if (!Number.isInteger(v)) return false;
  return true;
}

// 候选切分阈值（相邻唯一值的中点，最多 maxC 个等距抽样）
function _candThr(R, idx, j, maxC = 32) {
  const vals = Array.from(new Set(idx.map((i) => R[i][j])));
  if (vals.length <= 1) return [];
  vals.sort((a, b) => a - b);
  const mids = new Array(vals.length - 1);
  for (let i = 0; i < vals.length - 1; i++) mids[i] = (vals[i] + vals[i + 1]) / 2;
  if (mids.length <= maxC) return mids;
  const step = mids.length / maxC;
  const out = [];
  for (let i = 0; i < maxC; i++) out.push(mids[Math.min(mids.length - 1, Math.floor(i * step))]);
  return Array.from(new Set(out));
}

// 节点统计：分类→基尼 + 众数 + 类概率；回归→均值 + 方差
function _nodeStat(idx, ys, task, classes) {
  const n = idx.length;
  if (task === 'regress') {
    let s = 0;
    for (const i of idx) s += ys[i];
    const m = s / n;
    let v = 0;
    for (const i of idx) v += (ys[i] - m) ** 2;
    return { value: m, impurity: v / n, probs: null, n };
  }
  const cnt = new Map();
  for (const i of idx) cnt.set(ys[i], (cnt.get(ys[i]) || 0) + 1);
  let gini = 1;
  for (const v of cnt.values()) { const p = v / n; gini -= p * p; }
  let bestLab = classes[0], bestC = -1;
  for (const [k, v] of cnt) if (v > bestC) { bestC = v; bestLab = k; }
  const probs = classes.map((c) => (cnt.get(c) || 0) / n);
  return { value: bestLab, impurity: gini, probs, n };
}

const _giniFrom = (cnt, n) => {
  let g = 1;
  for (const v of cnt) { if (v > 0) { const p = v / n; g -= p * p; } }
  return g;
};

// CART 递归建树：cfg = {maxDepth, minSplit, task, classes, feats}
function _buildTree(R, ys, idx, depth, cfg) {
  const stat = _nodeStat(idx, ys, cfg.task, cfg.classes);
  const leaf = { leaf: true, value: stat.value, probs: stat.probs, n: stat.n, impurity: stat.impurity };
  if (depth >= cfg.maxDepth || idx.length < cfg.minSplit || stat.impurity <= 1e-14) return leaf;
  let best = null;
  for (const j of cfg.feats) {
    for (const thr of _candThr(R, idx, j)) {
      const L = [], Rt = [];
      for (const i of idx) (R[i][j] <= thr ? L : Rt).push(i);
      if (L.length === 0 || Rt.length === 0) continue;
      let wImp;
      if (cfg.task === 'regress') {
        const mL = _mean(L.map((i) => ys[i])), mR = _mean(Rt.map((i) => ys[i]));
        let sL = 0, sR = 0;
        for (const i of L) sL += (ys[i] - mL) ** 2;
        for (const i of Rt) sR += (ys[i] - mR) ** 2;
        wImp = (sL + sR) / idx.length;
      } else {
        const K = cfg.classes.length;
        const cL = new Array(K).fill(0), cR = new Array(K).fill(0);
        for (const i of L) cL[cfg.classes.indexOf(ys[i])]++;
        for (const i of Rt) cR[cfg.classes.indexOf(ys[i])]++;
        wImp = (L.length * _giniFrom(cL, L.length) + Rt.length * _giniFrom(cR, Rt.length)) / idx.length;
      }
      const gain = stat.impurity - wImp;
      if (gain > 1e-12 && (best === null || gain > best.gain)) best = { gain, j, thr, L, R: Rt };
    }
  }
  if (!best) return leaf;
  return {
    leaf: false,
    feat: best.j,
    thr: best.thr,
    n: idx.length,
    impurity: stat.impurity,
    gain: best.gain,
    left: _buildTree(R, ys, best.L, depth + 1, cfg),
    right: _buildTree(R, ys, best.R, depth + 1, cfg),
  };
}

function _treeDown(node, row) {
  let cur = node;
  while (!cur.leaf) cur = row[cur.feat] <= cur.thr ? cur.left : cur.right;
  return cur;
}
const _treeValue = (node, row) => _treeDown(node, row).value;
const _treeProbs = (node, row) => _treeDown(node, row).probs;

function _treeDepth(node) {
  if (node.leaf) return 0;
  return 1 + Math.max(_treeDepth(node.left), _treeDepth(node.right));
}
function _treeLeaves(node) {
  if (node.leaf) return 1;
  return _treeLeaves(node.left) + _treeLeaves(node.right);
}
function _treeImportance(node, acc) {
  if (node.leaf) return;
  acc[node.feat] += node.gain * node.n;
  _treeImportance(node.left, acc);
  _treeImportance(node.right, acc);
}

// 决策树桩（AdaBoost 弱学习器）：阈值 + 左右众数类别
function _bestStump(R, tIdx, w, classes, feats) {
  const n = R.length;
  const all = Array.from({ length: n }, (_, i) => i);
  let best = null;
  for (const j of feats) {
    for (const thr of _candThr(R, all, j)) {
      const cL = new Array(classes.length).fill(0), cR = new Array(classes.length).fill(0);
      for (let i = 0; i < n; i++) (R[i][j] <= thr ? cL : cR)[tIdx[i]] += w[i];
      let a = 0; for (let k = 1; k < cL.length; k++) if (cL[k] > cL[a]) a = k;
      let b = 0; for (let k = 1; k < cR.length; k++) if (cR[k] > cR[b]) b = k;
      let err = 0;
      for (let i = 0; i < n; i++) if ((R[i][j] <= thr ? a : b) !== tIdx[i]) err += w[i];
      if (best === null || err < best.err) best = { err, feat: j, thr, cL: a, cR: b };
    }
  }
  return best;
}

// Lloyd k-means（k-means++ 或随机初始化；空簇重投到最远点）
function _kmeansppPick(R, k, rng) {
  const n = R.length;
  const centers = [_clone(R[rng.int(n)])];
  const d2 = new Array(n).fill(Infinity);
  for (let c = 1; c < k; c++) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const dd = _dist2(R[i], centers[centers.length - 1]);
      if (dd < d2[i]) d2[i] = dd;
      sum += d2[i];
    }
    if (!(sum > 0)) { centers.push(_clone(R[rng.int(n)])); continue; }
    let target = rng.next() * sum, acc = 0, chosen = n - 1;
    for (let i = 0; i < n; i++) { acc += d2[i]; if (acc >= target) { chosen = i; break; } }
    centers.push(_clone(R[chosen]));
  }
  return centers;
}

function _kmeans(R, k, seed, iters, init) {
  const rng = createRng(seed === undefined ? 20261014 : seed);
  const n = R.length, d = R[0].length;
  if (!(k >= 1) || k > n) throw new ForgeError('E_VALUE', `k 必须 ∈ [1, ${n}]，当前 ${k}`);
  let centers = init === 'random' ? rng.sample(R, k).map(_clone) : _kmeansppPick(R, k, rng);
  const labels = new Array(n).fill(0);
  const maxIt = iters === undefined ? 100 : Math.max(1, Math.floor(iters));
  for (let it = 0; it < maxIt; it++) {
    let moved = false;
    for (let i = 0; i < n; i++) {
      let bi = 0, bd = Infinity;
      for (let c = 0; c < k; c++) {
        const dd = _dist2(R[i], centers[c]);
        if (dd < bd) { bd = dd; bi = c; }
      }
      if (labels[i] !== bi) { labels[i] = bi; moved = true; }
    }
    const sums = Array.from({ length: k }, () => new Array(d).fill(0));
    const cnts = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      const c = labels[i];
      cnts[c]++;
      for (let j = 0; j < d; j++) sums[c][j] += R[i][j];
    }
    for (let c = 0; c < k; c++) {
      if (cnts[c] > 0) centers[c] = sums[c].map((v) => v / cnts[c]);
      else {
        // 空簇：取距自身簇心最远的样本作为新中心
        let fi = 0, fd = -1;
        for (let i = 0; i < n; i++) {
          const dd = _dist2(R[i], centers[labels[i]]);
          if (dd > fd) { fd = dd; fi = i; }
        }
        centers[c] = _clone(R[fi]);
        moved = true;
      }
    }
    if (!moved && it > 0) break;
  }
  let inertia = 0;
  for (let i = 0; i < n; i++) inertia += _dist2(R[i], centers[labels[i]]);
  return { centers, labels, inertia, k, d, n };
}

// 层次聚类（Lance-Williams 增量），返回 scipy 风格 Z：[[a, b, dist, size], ...]
function _linkage(R, method) {
  const n = R.length;
  const D0 = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const v = Math.sqrt(_dist2(R[i], R[j]));
    D0[i][j] = v; D0[j][i] = v;
  }
  let M = D0.map((r) => r.slice());
  const sizes = new Array(n).fill(1);
  const ids = Array.from({ length: n }, (_, i) => i);
  const Z = [];
  let nextId = n;
  while (ids.length > 1) {
    let bi = 0, bj = 1, bd = Infinity;
    for (let a = 0; a < ids.length; a++) {
      for (let b = a + 1; b < ids.length; b++) {
        if (M[a][b] < bd) { bd = M[a][b]; bi = a; bj = b; }
      }
    }
    const ia = ids[bi], ib = ids[bj], sa = sizes[bi], sb = sizes[bj];
    Z.push([Math.min(ia, ib), Math.max(ia, ib), bd, sa + sb]);
    const row = [];
    for (let x = 0; x < ids.length; x++) {
      if (x === bi || x === bj) continue;
      const da = M[bi][x], db = M[bj][x];
      let d;
      if (method === 'single') d = Math.min(da, db);
      else if (method === 'complete') d = Math.max(da, db);
      else d = (sa * da + sb * db) / (sa + sb); // average（UPGMA）
      row.push(d);
    }
    const keep = [];
    for (let x = 0; x < ids.length; x++) if (x !== bi && x !== bj) keep.push(x);
    const newIds = keep.map((x) => ids[x]);
    const newM = newIds.map(() => new Array(newIds.length).fill(0));
    for (let a = 0; a < newIds.length; a++) {
      for (let b = 0; b < newIds.length; b++) {
        if (a === b) continue;
        if (keep[a] === bi || keep[a] === bj || keep[b] === bi || keep[b] === bj) continue;
        const oa = M[keep[a]][keep[b]];
        newM[a][b] = oa;
      }
    }
    const mi = newIds.length;
    newM.push(new Array(mi + 1).fill(0));
    for (let a = 0; a < mi + 1; a++) newM[a].length = mi + 1;
    for (let a = 0; a < mi; a++) { newM[a][mi] = row[a]; newM[mi][a] = row[a]; }
    newM[mi][mi] = 0;
    const newSizes = keep.map((x) => sizes[x]);
    newSizes.push(sa + sb);
    newIds.push(nextId++);
    M = newM; sizes.length = 0; sizes.push(...newSizes);
    ids.length = 0; ids.push(...newIds);
  }
  return { Z, n, method };
}

// 由链接矩阵切出 k 簇标签
function _cutLabels(Z, n, k) {
  const members = new Map();
  for (let i = 0; i < n; i++) members.set(i, [i]);
  for (let m = 0; m < Z.length; m++) {
    const id = n + m;
    const a = members.get(Z[m][0]) || [], b = members.get(Z[m][1]) || [];
    members.set(id, a.concat(b));
  }
  // 自顶向下分裂直到 k 个簇
  let clusters = [members.get(Z.length ? n + Z.length - 1 : 0) || []];
  if (Z.length === 0) clusters = [Array.from({ length: n }, (_, i) => i)];
  for (let m = Z.length - 1; m >= 0 && clusters.length < k; m--) {
    for (let c = 0; c < clusters.length && clusters.length < k; c++) {
      const id = n + m;
      const a = members.get(Z[m][0]) || [], b = members.get(Z[m][1]) || [];
      if (clusters[c].length !== a.length + b.length) continue;
      const setA = new Set(a);
      if (!clusters[c].every((v) => setA.has(v) || b.includes(v))) continue;
      clusters.splice(c, 1, a, b);
      break;
    }
  }
  const labels = new Array(n).fill(0);
  clusters.forEach((cl, ci) => { for (const i of cl) labels[i] = ci; });
  return labels;
}

// 高斯核
function _rbf(a, b, gamma) { return Math.exp(-gamma * _dist2(a, b)); }

// 核矩阵中心化（特征空间去均值）：Kc = K − 1ₙK − K1ₙ + 1ₙK1ₙ
function _centerKernel(K) {
  const n = K.length;
  const rowMean = new Array(n).fill(0), colMean = new Array(n).fill(0);
  let all = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { rowMean[i] += K[i][j]; colMean[j] += K[i][j]; all += K[i][j]; }
  for (let i = 0; i < n; i++) { rowMean[i] /= n; colMean[i] /= n; }
  all /= n * n;
  return K.map((r, i) => r.map((v, j) => v - rowMean[i] - colMean[j] + all));
}

// RBF 带宽启发式：1 / (2·中位数距离²)
function _autoGamma(R) {
  const ds = [];
  for (let i = 0; i < R.length; i++) for (let j = i + 1; j < R.length; j++) ds.push(_dist2(R[i], R[j]));
  ds.sort((a, b) => a - b);
  const med = ds.length ? ds[Math.floor(ds.length / 2)] : 1;
  return 1 / (2 * Math.max(med, 1e-12));
}

// 调整兰德指数（ARI）
function _ari(a, b) {
  const n = a.length;
  if (n !== b.length) throw new ForgeError('E_SHAPE', `ARI 标签长度不符: ${n} vs ${b.length}`);
  const cc = new Map(), ca = new Map(), cb = new Map();
  for (let i = 0; i < n; i++) {
    const key = `${a[i]}|${b[i]}`;
    cc.set(key, (cc.get(key) || 0) + 1);
    ca.set(a[i], (ca.get(a[i]) || 0) + 1);
    cb.set(b[i], (cb.get(b[i]) || 0) + 1);
  }
  const C2 = (v) => v * (v - 1) / 2;
  let sIJ = 0;
  for (const v of cc.values()) sIJ += C2(v);
  let sA = 0;
  for (const v of ca.values()) sA += C2(v);
  let sB = 0;
  for (const v of cb.values()) sB += C2(v);
  const comb = C2(n);
  if (comb === 0) return 0;
  const exp = (sA * sB) / comb;
  const max = (sA + sB) / 2;
  if (Math.abs(max - exp) < 1e-15) return 0;
  return (sIJ - exp) / (max - exp);
}

// 轮廓系数（逐样本）
function _silhouetteSamples(R, labels) {
  const n = R.length;
  const D = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const v = Math.sqrt(_dist2(R[i], R[j]));
    D[i][j] = v; D[j][i] = v;
  }
  const groups = new Map();
  for (let i = 0; i < n; i++) {
    if (!groups.has(labels[i])) groups.set(labels[i], []);
    groups.get(labels[i]).push(i);
  }
  const out = new Array(n).fill(0);
  if (groups.size < 2) return out;
  for (let i = 0; i < n; i++) {
    const own = groups.get(labels[i]);
    let a = 0;
    for (const j of own) if (j !== i) a += D[i][j];
    a = own.length > 1 ? a / (own.length - 1) : 0;
    let b = Infinity;
    for (const [lab, mem] of groups) {
      if (lab === labels[i]) continue;
      let s = 0;
      for (const j of mem) s += D[i][j];
      b = Math.min(b, s / mem.length);
    }
    const den = Math.max(a, b);
    out[i] = den > 0 ? (b - a) / den : 0;
  }
  return out;
}

// 等宽离散化（用于互信息估计）
function _discretize(a, bins) {
  let mn = Infinity, mx = -Infinity;
  for (const v of a) { if (v < mn) mn = v; if (v > mx) mx = v; }
  if (!(mx > mn)) return a.map(() => 0);
  const w = (mx - mn) / bins;
  return a.map((v) => Math.min(bins - 1, Math.max(0, Math.floor((v - mn) / w - 1e-12))));
}

// 熵（自然对数）
function _entropy(counts, total) {
  let h = 0;
  for (const v of counts) if (v > 0) { const p = v / total; h -= p * Math.log(p); }
  return h;
}

// —— A 组公用：把 X 增广为 [1, x] 并做最小二乘 ——
function _linregCore(X, y) {
  const R = _rows(X), ys = _flat(y);
  const A = R.map((r) => [1, ...r]);
  const w = _lstsq(A, ys);
  return { R, ys, intercept: w[0], coef: w.slice(1), w };
}

// —— 逻辑回归：阻尼牛顿（IRLS）+ Armijo 回溯 ——
function _logLoss(A, t, w) {
  let L = 0;
  for (let i = 0; i < A.length; i++) {
    const p = Math.min(1 - 1e-15, Math.max(1e-15, _sigmoid(_dot(w, A[i]))));
    L -= t[i] * Math.log(p) + (1 - t[i]) * Math.log(1 - p);
  }
  return L;
}
function _fitLogReg(A, t, iters, ridge) {
  const n = A.length, p = A[0].length;
  let w = new Array(p).fill(0);
  let L = _logLoss(A, t, w);
  const maxIt = iters === undefined ? 100 : Math.max(1, Math.floor(iters));
  for (let it = 0; it < maxIt; it++) {
    const pr = new Array(n);
    for (let i = 0; i < n; i++) pr[i] = _sigmoid(_dot(w, A[i]));
    const g = new Array(p).fill(0);
    for (let i = 0; i < n; i++) {
      const e = pr[i] - t[i], r = A[i];
      for (let j = 0; j < p; j++) g[j] += e * r[j];
    }
    let gn = 0;
    for (const v of g) gn += v * v;
    gn = Math.sqrt(gn);
    if (gn < 1e-12) break;
    const H = Array.from({ length: p }, () => new Array(p).fill(0));
    for (let i = 0; i < n; i++) {
      const ww = Math.max(pr[i] * (1 - pr[i]), 1e-10), r = A[i];
      for (let a = 0; a < p; a++) {
        const ra = r[a];
        if (ra === 0) continue;
        for (let b = a; b < p; b++) H[a][b] += ww * ra * r[b];
      }
    }
    for (let a = 0; a < p; a++) {
      for (let b = a + 1; b < p; b++) H[b][a] = H[a][b];
      H[a][a] += ridge;
    }
    let dir;
    try {
      const s = _solve(H, g, 0);
      dir = s.map((v) => -v);
    } catch {
      dir = g.map((v) => -v / Math.max(1, gn));
    }
    if (!dir.every(Number.isFinite)) dir = g.map((v) => -v / Math.max(1, gn));
    let nl = _logLoss(A, t, w.map((v, i) => v + dir[i]));
    let tries = 0;
    while (!(nl < L) && tries < 40) {
      dir = dir.map((v) => v * 0.5);
      nl = _logLoss(A, t, w.map((v, i) => v + dir[i]));
      tries++;
    }
    if (!(nl < L)) break;
    const nw = w.map((v, i) => v + dir[i]);
    if (!nw.every(Number.isFinite)) break;
    const drop = L - nl;
    w = nw; L = nl;
    if (L < 1e-14 || drop < 1e-16) break;
  }
  return { w, loss: L };
}

// =====================================================================
// 一、A 回归（12 条）：linreg / ridge / lasso / elasticNet / poly / robust
// =====================================================================
defineOp({
  id: nid(), domain: D, name: 'ml.linreg.fit', arity: 2,
  sig: '(Tensor X[n,d], Tensor y[n]) -> object state',
  doc: '普通最小二乘闭式解（增广截距列后走 QR 最小二乘）',
  ref: 'Gauss 1809 / Legendre 1805；Golub & Van Loan《Matrix Computations》§5.3',
  inv: '残差正交于设计矩阵：Aᵀ(y−Aŵ)≈0（A=[1,X]）',
  impl: (X, y) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    if (n < 1) throw new ForgeError('E_VALUE', '至少需要 1 个样本');
    const { R, ys, intercept, coef } = _linregCore(X, y);
    const fitted = R.map((r) => intercept + _dot(coef, r));
    return { kind: 'ml.linreg', n, d, intercept, coef, fitted, rank: Math.min(n, d + 1) };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.linreg.predict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor y[m]',
  doc: '线性回归预测 ŷ = intercept + X·coef',
  ref: 'Hastie, Tibshirani & Friedman, ESL 2nd ed., 2009, §3.2',
  inv: 'predict(fit(X,y), X) 等于 fit 时的 fitted 值',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    return _T1(R.map((r) => state.intercept + _dot(state.coef, r)));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.ridge.fit', arity: 3,
  sig: '(Tensor X[n,d], Tensor y[n], number lambda=1) -> object state',
  doc: '岭回归：最小化 (1/(2n))‖Xw−y‖² + (λ/2)‖w‖²，即解 (XᵀX+nλI)w=Xᵀy（Cholesky）',
  ref: 'Hoerl & Kennard, Ridge Regression: Biased Estimation for Nonorthogonal Problems, 1970',
  inv: 'λ→0 时退化为普通最小二乘；λ→∞ 时系数趋于 0；λ 与样本量无关（按样本平均的目标）',
  impl: (X, y, lambda) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const lam = lambda === undefined ? 1 : expectNumber(lambda, 'lambda');
    if (lam < 0) throw new ForgeError('E_VALUE', `lambda 必须 >= 0，当前 ${lam}`);
    const R = _rows(X), ys = _flat(y);
    const mx = _colMean(R, d), my = _mean(ys);
    const Rc = _center(R, mx), yc = ys.map((v) => v - my);
    const G = Array.from({ length: d }, () => new Array(d).fill(0));
    for (let a = 0; a < d; a++) for (let b = a; b < d; b++) {
      let s = 0;
      for (const r of Rc) s += r[a] * r[b];
      G[a][b] = s; G[b][a] = s;
    }
    const b = new Array(d).fill(0);
    for (let a = 0; a < d; a++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += Rc[i][a] * yc[i];
      b[a] = s;
    }
    // 目标按样本平均（1/(2n)‖r‖² + (λ/2)‖w‖²），故正规方程的对角增量为 n·λ
    const reg = lam > 0 ? lam * n : 1e-12 * (1 + _sum(G.map((r, i) => r[i])) / Math.max(1, d));
    const w = _solve(G, b, reg);
    const intercept = my - _dot(w, mx);
    return { kind: 'ml.ridge', n, d, lambda: lam, coef: w, intercept, meanX: mx, meanY: my };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.ridge.predict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor y[m]',
  doc: '岭回归预测',
  ref: 'Hoerl & Kennard 1970；Hastie et al., ESL 2009 §3.4.1',
  inv: 'λ=0 时与 ml.linreg.predict 一致',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    return _T1(R.map((r) => state.intercept + _dot(state.coef, r)));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.lasso.fit', arity: 4,
  sig: '(Tensor X[n,d], Tensor y[n], number lambda=1, number iters=500) -> object state',
  doc: 'Lasso：最小化 (1/(2n))‖Xw−y‖² + λ‖w‖₁，ISTA 近端梯度（软阈值），特征先标准化',
  ref: 'Tibshirani, Regression Shrinkage and Selection via the Lasso, JRSS-B, 1996；Beck & Teboulle, FISTA, 2009',
  inv: 'λ 足够大时系数出现精确 0（稀疏）；λ=0 时等于最小二乘',
  impl: (X, y, lambda, iters) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const lam = lambda === undefined ? 1 : expectNumber(lambda, 'lambda');
    if (lam < 0) throw new ForgeError('E_VALUE', `lambda 必须 >= 0，当前 ${lam}`);
    const maxIt = iters === undefined ? 500 : Math.max(1, Math.floor(iters));
    const R = _rows(X), ys = _flat(y);
    const mx = _colMean(R, d), my = _mean(ys);
    const sd = new Array(d).fill(0);
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += (R[i][j] - mx[j]) ** 2;
      sd[j] = Math.sqrt(s / n) || 1;
    }
    const Xs = R.map((r) => r.map((v, j) => (v - mx[j]) / sd[j]));
    const yc = ys.map((v) => v - my);
    const G = Array.from({ length: d }, () => new Array(d).fill(0));
    for (let a = 0; a < d; a++) for (let b = a; b < d; b++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += Xs[i][a] * Xs[i][b];
      G[a][b] = s; G[b][a] = s;
    }
    const Xty = new Array(d).fill(0);
    for (let a = 0; a < d; a++) { let s = 0; for (let i = 0; i < n; i++) s += Xs[i][a] * yc[i]; Xty[a] = s; }
    // Lipschitz 常数 L = λmax(XᵀX)/n
    let L = 0;
    try { L = _eigSym(G).values[0] / n; } catch { for (const r of G) L = Math.max(L, _sum(r.map(Math.abs)) / n); }
    if (!(L > 0)) L = 1;
    const w = new Array(d).fill(0);
    const thr = lam / L;
    const step = 1 / L;
    for (let it = 0; it < maxIt; it++) {
      const grad = new Array(d).fill(0);
      for (let a = 0; a < d; a++) {
        let s = 0;
        for (let b = 0; b < d; b++) s += G[a][b] * w[b];
        grad[a] = (s - Xty[a]) / n;
      }
      let moved = 0;
      for (let a = 0; a < d; a++) {
        const z = w[a] - step * grad[a];
        const nv = Math.sign(z) * Math.max(0, Math.abs(z) - thr);
        moved = Math.max(moved, Math.abs(nv - w[a]));
        w[a] = nv;
      }
      if (moved < 1e-14) break;
    }
    const coef = w.map((v, j) => v / sd[j]);
    const intercept = my - _dot(coef, mx);
    return {
      kind: 'ml.lasso', n, d, lambda: lam, iters: maxIt,
      coef, intercept, meanX: mx, meanY: my, stdX: sd,
      nonzero: coef.filter((v) => v !== 0).length,
    };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.lasso.predict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor y[m]',
  doc: 'Lasso 预测（内部使用标准化空间系数换算回原始尺度）',
  ref: 'Tibshirani 1996；Hastie et al., ESL 2009 §3.4.2',
  inv: '与 state.coef/intercept 的线性预测一致',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    return _T1(R.map((r) => state.intercept + _dot(state.coef, r)));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.elasticNet.fit', arity: 5,
  sig: '(Tensor X[n,d], Tensor y[n], number lambda=1, number l1Ratio=0.5, number iters=500) -> object state',
  doc: '弹性网：最小化 (1/(2n))‖Xw−y‖² + λ[α‖w‖₁ + ((1−α)/2)‖w‖²]，近端梯度（软阈值+收缩）',
  ref: 'Zou & Hastie, Regularization and Variable Selection via the Elastic Net, JRSS-B, 2005',
  inv: 'l1Ratio=1 退化为 Lasso；l1Ratio=0 退化为同 λ 的 Ridge',
  impl: (X, y, lambda, l1Ratio, iters) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const lam = lambda === undefined ? 1 : expectNumber(lambda, 'lambda');
    const a1 = l1Ratio === undefined ? 0.5 : expectNumber(l1Ratio, 'l1Ratio');
    if (lam < 0) throw new ForgeError('E_VALUE', `lambda 必须 >= 0，当前 ${lam}`);
    if (!(a1 >= 0 && a1 <= 1)) throw new ForgeError('E_VALUE', `l1Ratio 必须 ∈ [0,1]，当前 ${a1}`);
    const maxIt = iters === undefined ? 500 : Math.max(1, Math.floor(iters));
    const R = _rows(X), ys = _flat(y);
    const mx = _colMean(R, d), my = _mean(ys);
    const sd = new Array(d).fill(0);
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += (R[i][j] - mx[j]) ** 2;
      sd[j] = Math.sqrt(s / n) || 1;
    }
    const Xs = R.map((r) => r.map((v, j) => (v - mx[j]) / sd[j]));
    const yc = ys.map((v) => v - my);
    const G = Array.from({ length: d }, () => new Array(d).fill(0));
    for (let p = 0; p < d; p++) for (let q = p; q < d; q++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += Xs[i][p] * Xs[i][q];
      G[p][q] = s; G[q][p] = s;
    }
    const Xty = new Array(d).fill(0);
    for (let p = 0; p < d; p++) { let s = 0; for (let i = 0; i < n; i++) s += Xs[i][p] * yc[i]; Xty[p] = s; }
    let L = 0;
    try { L = _eigSym(G).values[0] / n; } catch { for (const r of G) L = Math.max(L, _sum(r.map(Math.abs)) / n); }
    if (!(L > 0)) L = 1;
    const w = new Array(d).fill(0);
    const thr = (lam * a1) / L;
    const shrink = 1 / (1 + (lam * (1 - a1)) / L);
    for (let it = 0; it < maxIt; it++) {
      const grad = new Array(d).fill(0);
      for (let p = 0; p < d; p++) {
        let s = 0;
        for (let q = 0; q < d; q++) s += G[p][q] * w[q];
        grad[p] = (s - Xty[p]) / n;
      }
      let moved = 0;
      for (let p = 0; p < d; p++) {
        const z = (w[p] - grad[p] / L) * shrink;
        const nv = Math.sign(z) * Math.max(0, Math.abs(z) - thr * shrink);
        moved = Math.max(moved, Math.abs(nv - w[p]));
        w[p] = nv;
      }
      if (moved < 1e-14) break;
    }
    const coef = w.map((v, j) => v / sd[j]);
    const intercept = my - _dot(coef, mx);
    return {
      kind: 'ml.elasticNet', n, d, lambda: lam, l1Ratio: a1, iters: maxIt,
      coef, intercept, meanX: mx, meanY: my, stdX: sd,
      nonzero: coef.filter((v) => v !== 0).length,
    };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.elasticNet.predict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor y[m]',
  doc: '弹性网预测',
  ref: 'Zou & Hastie 2005',
  inv: 'l1Ratio=1 时与 Lasso 同解',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    return _T1(R.map((r) => state.intercept + _dot(state.coef, r)));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.poly.fit', arity: 3,
  sig: '(Tensor X[n,d], Tensor y[n], number degree=2) -> object state',
  doc: '多项式回归：展开总次数 ≤ degree 的全部单项式后做最小二乘',
  ref: 'Hastie et al., ESL 2009 §7.2（基函数展开）；Forsythe 1957（多项式拟合）',
  inv: 'degree=1 时等价于普通线性回归',
  impl: (X, y, degree) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const deg = degree === undefined ? 2 : Math.floor(expectNumber(degree, 'degree'));
    if (!(deg >= 1)) throw new ForgeError('E_VALUE', `degree 必须 >= 1，当前 ${deg}`);
    const R = _rows(X), ys = _flat(y);
    // 生成总次数 1..deg 的指数多重指标
    const exps = [];
    const cur = new Array(d).fill(0);
    const rec = (pos) => {
      if (pos === d) {
        const s = _sum(cur);
        if (s >= 1 && s <= deg) exps.push(_clone(cur));
        return;
      }
      for (let e = 0; e <= deg; e++) { cur[pos] = e; rec(pos + 1); }
      cur[pos] = 0;
    };
    rec(0);
    const A = R.map((r) => [1, ...exps.map((e) => {
      let v = 1;
      for (let j = 0; j < d; j++) v *= r[j] ** e[j];
      return v;
    })]);
    const w = _lstsq(A, ys);
    return {
      kind: 'ml.poly', n, d, degree: deg, exps,
      intercept: w[0], coef: w.slice(1),
    };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.poly.predict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor y[m]',
  doc: '多项式回归预测（按 state.exps 展开基函数）',
  ref: 'Hastie et al., ESL 2009 §7.2',
  inv: 'degree=1 时与 ml.linreg.predict 一致',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    const exps = state.exps, coef = state.coef, d = state.d;
    return _T1(R.map((r) => {
      let v = state.intercept;
      for (let k = 0; k < exps.length; k++) {
        let mono = 1;
        for (let j = 0; j < d; j++) mono *= r[j] ** exps[k][j];
        v += coef[k] * mono;
      }
      return v;
    }));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.robust.fit', arity: 4,
  sig: '(Tensor X[n,d], Tensor y[n], number delta=1.345, number iters=25) -> object state',
  doc: 'Huber 稳健回归：IRLS 迭代重加权最小二乘（δ 为二次/线性损失切换点）',
  ref: 'Huber, Robust Estimation of a Location Parameter, Ann. Math. Statist., 1964',
  inv: '无异常值时结果接近最小二乘；大残差样本权重被压低',
  impl: (X, y, delta, iters) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const maxIt = iters === undefined ? 25 : Math.max(1, Math.floor(iters));
    const R = _rows(X), ys = _flat(y);
    const A = R.map((r) => [1, ...r]);
    let w = _lstsq(A, ys);
    let deltaEff = delta === undefined ? 0 : expectNumber(delta, 'delta');
    for (let it = 0; it < maxIt; it++) {
      const res = A.map((r, i) => ys[i] - _dot(w, r));
      const abs = res.map(Math.abs).sort((a, b) => a - b);
      const mad = abs[Math.floor(n / 2)] || 0;
      const sigma = 1.4826 * mad;
      const dl = deltaEff > 0 ? deltaEff : Math.max(1e-8, 1.345 * sigma);
      const wt = res.map((r) => {
        const a = Math.abs(r);
        return a <= dl ? 1 : dl / Math.max(a, _EPS);
      });
      const G = Array.from({ length: d + 1 }, () => new Array(d + 1).fill(0));
      for (let a = 0; a <= d; a++) for (let b = a; b <= d; b++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += wt[i] * A[i][a] * A[i][b];
        G[a][b] = s; G[b][a] = s;
      }
      const rhs = new Array(d + 1).fill(0);
      for (let a = 0; a <= d; a++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += wt[i] * A[i][a] * ys[i];
        rhs[a] = s;
      }
      const nw = _solve(G, rhs, 1e-10);
      if (!nw.every(Number.isFinite)) break;
      let diff = 0;
      for (let a = 0; a <= d; a++) diff = Math.max(diff, Math.abs(nw[a] - w[a]));
      w = nw;
      if (diff < 1e-12) break;
    }
    return { kind: 'ml.robust', n, d, delta: deltaEff, intercept: w[0], coef: w.slice(1) };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.robust.predict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor y[m]',
  doc: '稳健回归预测',
  ref: 'Huber 1964；Hampel et al.《Robust Statistics》1986',
  inv: '线性预测 ŷ = intercept + X·coef',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    return _T1(R.map((r) => state.intercept + _dot(state.coef, r)));
  },
});

// =====================================================================
// 二、B 线性分类（10 条）：logreg / softmax / perceptron / lda
// =====================================================================
defineOp({
  id: nid(), domain: D, name: 'ml.logreg.fit', arity: 4,
  sig: '(Tensor X[n,d], Tensor y[n], number iters=100, number ridge=1e-8) -> object state',
  doc: '二元逻辑回归：阻尼牛顿（IRLS）+ Armijo 回溯线搜索',
  ref: 'Cox, The Regression Analysis of Binary Sequences, JRSS-B, 1958；Nelder & Wedderburn, GLM, 1972',
  inv: '梯度 Xᵀ(p−y)≈0；线性可分数据上训练准确率 = 1',
  impl: (X, y, iters, ridge) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const ys = _flat(y);
    const { neg, pos, t } = _binLabels(ys);
    const A = _rows(X).map((r) => [1, ...r]);
    const rg = ridge === undefined ? 1e-8 : expectNumber(ridge, 'ridge');
    const { w, loss } = _fitLogReg(A, t, iters, Math.max(0, rg));
    return { kind: 'ml.logreg', n, d, negLabel: neg, posLabel: pos, w, loss };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.logreg.predict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor labels[m]',
  doc: '逻辑回归硬分类（返回原始标签值）',
  ref: 'Cox 1958；Bishop, PRML, 2006 §4.3',
  inv: '预测标签 ∈ {negLabel, posLabel}；阈值 σ(wᵀx) ≥ 0.5',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    return _T1(R.map((r) => (_sigmoid(_dot(state.w, [1, ...r])) >= 0.5 ? state.posLabel : state.negLabel)));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.logreg.predictProba', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor p[m]',
  doc: '正类后验概率 σ(wᵀ[1,x])',
  ref: 'Cox 1958；Platt, Probabilistic Outputs for SVMs, 1999',
  inv: '概率 ∈ (0,1) 且随决策函数单调递增',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    return _T1(R.map((r) => _sigmoid(_dot(state.w, [1, ...r]))));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.softmax.fit', arity: 4,
  sig: '(Tensor X[n,d], Tensor y[n], number lr=0.1, number iters=500) -> object state',
  doc: '多类 softmax 回归（多项逻辑回归）：全批量梯度下降 + 稳定 softmax',
  ref: 'Bridle, Probabilistic Interpretation of Feedforward Network Outputs, 1990；Bishop PRML §4.3.4',
  inv: '预测概率每行和为 1；梯度消失时 W 收敛到交叉熵驻点',
  impl: (X, y, lr, iters) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const ys = _flat(y);
    const classes = _classesOf(ys);
    const K = classes.length;
    if (K < 2) throw new ForgeError('E_VALUE', 'softmax 需要至少 2 个类别');
    const eta = lr === undefined ? 0.1 : expectNumber(lr, 'lr');
    if (!(eta > 0)) throw new ForgeError('E_VALUE', `lr 必须 > 0，当前 ${eta}`);
    const maxIt = iters === undefined ? 500 : Math.max(1, Math.floor(iters));
    const A = _rows(X).map((r) => [1, ...r]);
    const p = d + 1;
    const W = Array.from({ length: p }, () => new Array(K).fill(0));
    const probs = new Array(K);
    let loss = 0;
    for (let it = 0; it < maxIt; it++) {
      const G = Array.from({ length: p }, () => new Array(K).fill(0));
      loss = 0;
      for (let i = 0; i < n; i++) {
        const z = new Array(K);
        for (let k = 0; k < K; k++) z[k] = _dot(W.map((row) => row[k]), A[i]);
        _softmaxRow(z, probs);
        const ti = classes.indexOf(ys[i]);
        loss -= Math.log(Math.max(probs[ti], 1e-300));
        for (let k = 0; k < K; k++) {
          const e = probs[k] - (k === ti ? 1 : 0);
          for (let j = 0; j < p; j++) G[j][k] += e * A[i][j];
        }
      }
      for (let j = 0; j < p; j++) for (let k = 0; k < K; k++) W[j][k] -= (eta / n) * G[j][k];
      loss /= n;
    }
    return { kind: 'ml.softmax', n, d, classes, W, loss, lr: eta, iters: maxIt };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.softmax.predict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor labels[m]',
  doc: 'softmax 回归硬分类（返回原始类别值）',
  ref: 'Bridle 1990',
  inv: '预测标签必属于 state.classes',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    const classes = state.classes, K = classes.length, W = state.W;
    const probs = new Array(K);
    const z = new Array(K);
    return _T1(R.map((r) => {
      const a = [1, ...r];
      for (let k = 0; k < K; k++) z[k] = _dot(W.map((row) => row[k]), a);
      _softmaxRow(z, probs);
      let bi = 0;
      for (let k = 1; k < K; k++) if (probs[k] > probs[bi]) bi = k;
      return classes[bi];
    }));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.softmax.predictProba', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor P[m,K]',
  doc: 'softmax 回归类后验概率矩阵',
  ref: 'Bridle 1990；Bishop PRML §4.3.4',
  inv: '每行元素非负且和为 1',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    const classes = state.classes, K = classes.length, W = state.W;
    const out = new Float64Array(R.length * K);
    const probs = new Array(K), z = new Array(K);
    R.forEach((r, i) => {
      const a = [1, ...r];
      for (let k = 0; k < K; k++) z[k] = _dot(W.map((row) => row[k]), a);
      _softmaxRow(z, probs);
      for (let k = 0; k < K; k++) out[i * K + k] = probs[k];
    });
    return new Tensor(out, [R.length, K]);
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.perceptron.fit', arity: 4,
  sig: '(Tensor X[n,d], Tensor y[n], number lr=0.1, number epochs=200) -> object state',
  doc: '经典感知机：错分即更新 w ← w + η·y·x',
  ref: 'Rosenblatt, The Perceptron: A Probabilistic Model for Information Storage, 1958',
  inv: '线性可分数据上有限步内收敛（错分数 → 0）',
  impl: (X, y, lr, epochs) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const ys = _flat(y);
    const { neg, pos } = _binLabels(ys);
    const eta = lr === undefined ? 0.1 : expectNumber(lr, 'lr');
    if (!(eta > 0)) throw new ForgeError('E_VALUE', `lr 必须 > 0，当前 ${eta}`);
    const ep = epochs === undefined ? 200 : Math.max(1, Math.floor(epochs));
    const R = _rows(X);
    const t = ys.map((v) => (v === pos ? 1 : -1));
    const w = new Array(d).fill(0);
    let b = 0, updates = 0;
    for (let e = 0; e < ep; e++) {
      let errs = 0;
      for (let i = 0; i < n; i++) {
        const z = t[i] * (_dot(w, R[i]) + b);
        if (z <= 0) {
          for (let j = 0; j < d; j++) w[j] += eta * t[i] * R[i][j];
          b += eta * t[i];
          updates++; errs++;
        }
      }
      if (errs === 0) break;
    }
    return { kind: 'ml.perceptron', n, d, negLabel: neg, posLabel: pos, w, b, lr: eta, epochs: ep, updates };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.perceptron.predict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor labels[m]',
  doc: '感知机预测：sign(wᵀx + b)',
  ref: 'Rosenblatt 1958',
  inv: '预测标签 ∈ {negLabel, posLabel}',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    return _T1(R.map((r) => (_dot(state.w, r) + state.b >= 0 ? state.posLabel : state.negLabel)));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.lda.fit', arity: 2,
  sig: '(Tensor X[n,d], Tensor y[n]) -> object state',
  doc: '线性判别分析：类等协方差 + 高斯似然的对数线性判别函数',
  ref: 'Fisher, The Use of Multiple Measurements in Taxonomic Problems, 1936；Hastie et al., ESL 2009 §4.3',
  inv: '判别得分 δ_k 线性于 x；两类等协方差时等价于 Bayes 最优分类器',
  impl: (X, y) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const R = _rows(X), ys = _flat(y);
    const classes = _classesOf(ys);
    const K = classes.length;
    if (K < 2) throw new ForgeError('E_VALUE', 'LDA 需要至少 2 个类别');
    const groups = classes.map((c) => R.filter((_, i) => ys[i] === c));
    const means = groups.map((g) => _colMean(g, d));
    const prior = groups.map((g) => g.length / n);
    // 合并协方差（无偏，分母 n−K）
    const S = Array.from({ length: d }, () => new Array(d).fill(0));
    for (let k = 0; k < K; k++) {
      for (const r of groups[k]) {
        const c = r.map((v, j) => v - means[k][j]);
        for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) S[a][b] += c[a] * c[b];
      }
    }
    const den = Math.max(1, n - K);
    for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) S[a][b] /= den;
    let tr = 0;
    for (let a = 0; a < d; a++) tr += S[a][a];
    const ridge = 1e-8 * (1 + tr / d);
    const SinvMu = means.map((m) => _solve(S, m, ridge));
    const consts = means.map((m, k) => -0.5 * _dot(m, SinvMu[k]) + Math.log(Math.max(prior[k], _EPS)));
    return { kind: 'ml.lda', n, d, classes, means, prior, S, SinvMu, consts, ridge };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.lda.predict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor labels[m]',
  doc: 'LDA 预测：argmax_k (xᵀΣ⁻¹μ_k + const_k)',
  ref: 'Fisher 1936；Hastie et al., ESL 2009 §4.3',
  inv: '决策边界为超平面（分段线性）',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    const K = state.classes.length;
    return _T1(R.map((r) => {
      let bi = 0, bv = -Infinity;
      for (let k = 0; k < K; k++) {
        const s = _dot(r, state.SinvMu[k]) + state.consts[k];
        if (s > bv) { bv = s; bi = k; }
      }
      return state.classes[bi];
    }));
  },
});

// =====================================================================
// 三、C 判别与朴素贝叶斯（8 条）：qda / nb.gaussian / nb.multinomial / nb.bernoulli
// =====================================================================
defineOp({
  id: nid(), domain: D, name: 'ml.qda.fit', arity: 2,
  sig: '(Tensor X[n,d], Tensor y[n]) -> object state',
  doc: '二次判别分析：每类各估计一个协方差矩阵',
  ref: 'Hastie et al., ESL 2009 §4.3；McLachlan《Discriminant Analysis and Statistical Pattern Recognition》1992',
  inv: '决策边界为二次曲面；类内协方差相等时退化为 LDA',
  impl: (X, y) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const R = _rows(X), ys = _flat(y);
    const classes = _classesOf(ys);
    const K = classes.length;
    if (K < 2) throw new ForgeError('E_VALUE', 'QDA 需要至少 2 个类别');
    const groups = classes.map((c) => R.filter((_, i) => ys[i] === c));
    const means = groups.map((g) => _colMean(g, d));
    const prior = groups.map((g) => g.length / n);
    const covs = [], invs = [], logdets = [];
    for (let k = 0; k < K; k++) {
      const S = Array.from({ length: d }, () => new Array(d).fill(0));
      const nk = Math.max(1, groups[k].length - 1);
      for (const r of groups[k]) {
        const c = r.map((v, j) => v - means[k][j]);
        for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) S[a][b] += c[a] * c[b];
      }
      for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) S[a][b] /= nk;
      let tr = 0;
      for (let a = 0; a < d; a++) tr += S[a][a];
      const ridge = 1e-6 * (1 + tr / d);
      const Sr = S.map((row, i) => row.map((v, j) => v + (i === j ? ridge : 0)));
      covs.push(Sr);
      invs.push(_inv(Sr));
      logdets.push(_choleskyLogDet(Sr));
    }
    return { kind: 'ml.qda', n, d, classes, means, prior, covs, invs, logdets };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.qda.predict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor labels[m]',
  doc: 'QDA 预测：−½(x−μ_k)ᵀΣ_k⁻¹(x−μ_k) − ½log|Σ_k| + log π_k 取最大',
  ref: 'Hastie et al., ESL 2009 §4.3',
  inv: '预测标签必属于 state.classes',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    const K = state.classes.length;
    return _T1(R.map((r) => {
      let bi = 0, bv = -Infinity;
      for (let k = 0; k < K; k++) {
        const c = r.map((v, j) => v - state.means[k][j]);
        const quad = _dot(c, state.invs[k].map((row) => _dot(row, c)));
        const s = -0.5 * quad - 0.5 * state.logdets[k] + Math.log(Math.max(state.prior[k], _EPS));
        if (s > bv) { bv = s; bi = k; }
      }
      return state.classes[bi];
    }));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.nb.gaussianFit', arity: 2,
  sig: '(Tensor X[n,d], Tensor y[n]) -> object state',
  doc: '高斯朴素贝叶斯：各类各维独立估计均值与方差',
  ref: 'Duda & Hart, Pattern Classification and Scene Analysis, 1973；Hastie et al., ESL 2009 §6.6.3',
  inv: '后验 log P(y=k|x) ∝ log π_k + Σ_j log N(x_j; μ_kj, σ²_kj)',
  impl: (X, y) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const R = _rows(X), ys = _flat(y);
    const classes = _classesOf(ys);
    const groups = classes.map((c) => R.filter((_, i) => ys[i] === c));
    const means = groups.map((g) => _colMean(g, d));
    const vars = means.map((m, k) => {
      const g = groups[k], nk = Math.max(1, g.length - 1);
      return m.map((_, j) => {
        let s = 0;
        for (const r of g) s += (r[j] - m[j]) ** 2;
        return Math.max(s / nk, 1e-9);
      });
    });
    return {
      kind: 'ml.nb.gaussian', n, d, classes,
      prior: groups.map((g) => g.length / n),
      means, vars,
    };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.nb.gaussianPredict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor labels[m]',
  doc: '高斯朴素贝叶斯最大后验预测',
  ref: 'Duda & Hart 1973；Hastie et al., ESL 2009 §6.6.3',
  inv: '两类等方差高斯且均值已知时准确率趋近 Bayes 最优',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    const K = state.classes.length, d = state.d;
    return _T1(R.map((r) => {
      let bi = 0, bv = -Infinity;
      for (let k = 0; k < K; k++) {
        let s = Math.log(Math.max(state.prior[k], _EPS));
        for (let j = 0; j < d; j++) {
          const v = state.vars[k][j];
          s += -0.5 * Math.log(2 * Math.PI * v) - ((r[j] - state.means[k][j]) ** 2) / (2 * v);
        }
        if (s > bv) { bv = s; bi = k; }
      }
      return state.classes[bi];
    }));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.nb.multinomialFit', arity: 3,
  sig: '(Tensor X[n,d], Tensor y[n], number alpha=1) -> object state',
  doc: '多项式朴素贝叶斯（文本计数模型）：Laplace/Lidstone 平滑的对数似然',
  ref: 'McCallum & Nigam, A Comparison of Event Models for Naive Bayes Text Classification, AAAI-98',
  inv: '每类 θ_k 在特征维上归一：Σ_j θ_kj = 1',
  impl: (X, y, alpha) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const a = alpha === undefined ? 1 : expectNumber(alpha, 'alpha');
    if (a < 0) throw new ForgeError('E_VALUE', `alpha 必须 >= 0，当前 ${a}`);
    const R = _rows(X), ys = _flat(y);
    const classes = _classesOf(ys);
    const groups = classes.map((c) => R.filter((_, i) => ys[i] === c));
    const logTheta = groups.map((g) => {
      const cnt = new Array(d).fill(0);
      let tot = 0;
      for (const r of g) for (let j = 0; j < d; j++) { cnt[j] += Math.max(0, r[j]); tot += Math.max(0, r[j]); }
      return cnt.map((c) => Math.log((c + a) / (tot + a * d)));
    });
    return {
      kind: 'ml.nb.multinomial', n, d, classes, alpha: a,
      prior: groups.map((g) => g.length / n), logTheta,
    };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.nb.multinomialPredict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor labels[m]',
  doc: '多项式朴素贝叶斯预测',
  ref: 'McCallum & Nigam, AAAI-98',
  inv: '得分 = log π_k + Σ_j x_j·log θ_kj',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    const K = state.classes.length, d = state.d;
    return _T1(R.map((r) => {
      let bi = 0, bv = -Infinity;
      for (let k = 0; k < K; k++) {
        let s = Math.log(Math.max(state.prior[k], _EPS));
        for (let j = 0; j < d; j++) s += Math.max(0, r[j]) * state.logTheta[k][j];
        if (s > bv) { bv = s; bi = k; }
      }
      return state.classes[bi];
    }));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.nb.bernoulliFit', arity: 3,
  sig: '(Tensor X[n,d], Tensor y[n], number alpha=1) -> object state',
  doc: '伯努利朴素贝叶斯：特征二值化后估计 P(x_j=1|y=k)',
  ref: 'McCallum & Nigam, AAAI-98；Hastie et al., ESL 2009 §6.6.3',
  inv: '估计概率截断到 [ε, 1−ε]，保证 log 有界',
  impl: (X, y, alpha) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const a = alpha === undefined ? 1 : expectNumber(alpha, 'alpha');
    if (a < 0) throw new ForgeError('E_VALUE', `alpha 必须 >= 0，当前 ${a}`);
    const R = _rows(X), ys = _flat(y);
    const classes = _classesOf(ys);
    const groups = classes.map((c) => R.filter((_, i) => ys[i] === c));
    const p = groups.map((g) => {
      const nk = Math.max(1, g.length);
      const cnt = new Array(d).fill(0);
      for (const r of g) for (let j = 0; j < d; j++) if (r[j] > 0) cnt[j]++;
      return cnt.map((c) => Math.min(1 - 1e-12, Math.max(1e-12, (c + a) / (nk + 2 * a))));
    });
    return {
      kind: 'ml.nb.bernoulli', n, d, classes, alpha: a,
      prior: groups.map((g) => g.length / n), p,
    };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.nb.bernoulliPredict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor labels[m]',
  doc: '伯努利朴素贝叶斯预测（输入按 x>0 二值化）',
  ref: 'McCallum & Nigam, AAAI-98',
  inv: '得分 = log π_k + Σ_j [x_j log p_kj + (1−x_j) log(1−p_kj)]',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    const K = state.classes.length, d = state.d;
    return _T1(R.map((r) => {
      let bi = 0, bv = -Infinity;
      for (let k = 0; k < K; k++) {
        let s = Math.log(Math.max(state.prior[k], _EPS));
        for (let j = 0; j < d; j++) {
          const x = r[j] > 0 ? 1 : 0, pk = state.p[k][j];
          s += x * Math.log(pk) + (1 - x) * Math.log(1 - pk);
        }
        if (s > bv) { bv = s; bi = k; }
      }
      return state.classes[bi];
    }));
  },
});

// =====================================================================
// 四、D 树与集成（14 条）：CART / 随机森林 / 梯度提升 / AdaBoost
// =====================================================================
defineOp({
  id: nid(), domain: D, name: 'ml.tree.fit', arity: 5,
  sig: '(Tensor X[n,d], Tensor y[n], number maxDepth=10, number minSplit=2, string task="auto") -> object state',
  doc: 'CART 决策树：分类用基尼不纯度、回归用平方误差，贪心递归二分',
  ref: 'Breiman, Friedman, Olshen & Stone, Classification and Regression Trees, 1984',
  inv: '每个内部节点的加权不纯度下降 ≥ 0；叶节点不纯度为 0 时训练误差为 0',
  impl: (X, y, maxDepth, minSplit, task) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const ys = _flat(y);
    const tk = task === undefined || task === 'auto' ? (_isClassify(ys) ? 'classify' : 'regress') : String(task);
    if (tk !== 'classify' && tk !== 'regress') {
      throw new ForgeError('E_VALUE', `task 必须为 auto|classify|regress，当前 ${task}`);
    }
    const md = maxDepth === undefined ? 10 : Math.max(0, Math.floor(expectNumber(maxDepth, 'maxDepth')));
    const ms = minSplit === undefined ? 2 : Math.max(2, Math.floor(expectNumber(minSplit, 'minSplit')));
    const classes = tk === 'classify' ? _classesOf(ys) : [];
    const R = _rows(X);
    const root = _buildTree(R, ys, Array.from({ length: n }, (_, i) => i), 0, {
      maxDepth: md, minSplit: ms, task: tk, classes, feats: Array.from({ length: d }, (_, j) => j),
    });
    return { kind: 'ml.tree', n, d, task: tk, classes, maxDepth: md, minSplit: ms, root };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.tree.predict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor out[m]',
  doc: '决策树预测（分类→众数标签，回归→叶内均值）',
  ref: 'Breiman et al., CART, 1984',
  inv: '每个样本的预测等于其落入叶节点的代表值',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    return _T1(R.map((r) => _treeValue(state.root, r)));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.tree.predictProba', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor P[m,K]',
  doc: '决策树叶内类概率（回归树返回 [m,1] 的叶均值）',
  ref: 'Breiman et al., CART, 1984',
  inv: '分类树每行概率非负且和为 1',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    if (state.task === 'regress') {
      return new Tensor(Float64Array.from(R.map((r) => _treeValue(state.root, r))), [R.length, 1]);
    }
    const K = state.classes.length;
    const out = new Float64Array(R.length * K);
    R.forEach((r, i) => {
      const p = _treeProbs(state.root, r) || state.classes.map(() => 1 / K);
      for (let k = 0; k < K; k++) out[i * K + k] = p[k];
    });
    return new Tensor(out, [R.length, K]);
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.tree.importance', arity: 1,
  sig: '(object state) -> Tensor imp[d]',
  doc: 'CART 特征重要性：不纯度下降量按样本数加权后归一',
  ref: 'Breiman et al., CART, 1984；Breiman, Random Forests, Machine Learning, 2001',
  inv: '所有分量非负且和为 1',
  impl: (state) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    const acc = new Array(state.d).fill(0);
    _treeImportance(state.root, acc);
    const tot = _sum(acc);
    const out = tot > 0 ? acc.map((v) => v / tot) : acc.map(() => 0);
    return _T1(out);
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.tree.depth', arity: 1,
  sig: '(object state) -> number',
  doc: '树深度（最长根到叶路径的边数，单叶树为 0）',
  ref: 'Breiman et al., CART, 1984',
  inv: '深度 ≤ maxDepth；单节点树深度 = 0',
  impl: (state) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    return _treeDepth(state.root);
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.tree.leafCount', arity: 1,
  sig: '(object state) -> number',
  doc: '叶节点个数',
  ref: 'Breiman et al., CART, 1984',
  inv: '叶数 ≥ 1；深度 ≤ D 的二叉树叶数 ≤ 2^D',
  impl: (state) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    return _treeLeaves(state.root);
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.rf.fit', arity: 5,
  sig: '(Tensor X[n,d], Tensor y[n], number nTrees=20, number maxDepth=10, number seed=20261014) -> object state',
  doc: '随机森林：Bagging + 随机特征子集（√d）的 CART 集成',
  ref: 'Breiman, Random Forests, Machine Learning, 2001；Ho, Random Decision Forests, 1995',
  inv: '每棵树只在自助样本与特征子集上训练；树数越多方差越小',
  impl: (X, y, nTrees, maxDepth, seed) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const T = nTrees === undefined ? 20 : Math.max(1, Math.floor(expectNumber(nTrees, 'nTrees')));
    const md = maxDepth === undefined ? 10 : Math.max(0, Math.floor(expectNumber(maxDepth, 'maxDepth')));
    const sd = seed === undefined ? 20261014 : Math.floor(expectNumber(seed, 'seed'));
    const rng = createRng(sd);
    const ys = _flat(y);
    const task = _isClassify(ys) ? 'classify' : 'regress';
    const classes = task === 'classify' ? _classesOf(ys) : [];
    const R = _rows(X);
    const mtry = Math.max(1, Math.floor(Math.sqrt(d)));
    const trees = [];
    for (let t = 0; t < T; t++) {
      const bag = Array.from({ length: n }, () => rng.int(n));
      const feats = rng.sample(Array.from({ length: d }, (_, j) => j), Math.min(mtry, d));
      const root = _buildTree(R, ys, bag, 0, {
        maxDepth: md, minSplit: 2, task, classes, feats,
      });
      trees.push({ root, feats });
    }
    return { kind: 'ml.rf', n, d, task, classes, nTrees: T, maxDepth: md, mtry, seed: sd, trees };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.rf.predict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor out[m]',
  doc: '随机森林预测（分类多数投票 / 回归取均值）',
  ref: 'Breiman, Random Forests, 2001',
  inv: '分类输出属于 state.classes；回归输出为各树叶均值的算术平均',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    const T = state.trees.length;
    if (state.task === 'regress') {
      return _T1(R.map((r) => {
        let s = 0;
        for (let t = 0; t < T; t++) s += _treeValue(state.trees[t].root, r);
        return s / T;
      }));
    }
    return _T1(R.map((r) => {
      const cnt = new Map();
      for (let t = 0; t < T; t++) {
        const v = _treeValue(state.trees[t].root, r);
        cnt.set(v, (cnt.get(v) || 0) + 1);
      }
      let bl = state.classes[0], bc = -1;
      for (const [k, v] of cnt) if (v > bc) { bc = v; bl = k; }
      return bl;
    }));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.rf.importance', arity: 1,
  sig: '(object state) -> Tensor imp[d]',
  doc: '随机森林特征重要性：各树不纯度下降量的平均并归一',
  ref: 'Breiman, Random Forests, 2001；Louppe et al., Understanding Variable Importances in Forests, 2013',
  inv: '所有分量非负且和为 1（森林非空时）',
  impl: (state) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    const d = state.d;
    const acc = new Array(d).fill(0);
    for (const t of state.trees) _treeImportance(t.root, acc);
    const tot = _sum(acc);
    return _T1(tot > 0 ? acc.map((v) => v / tot) : acc.map(() => 0));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.gb.fit', arity: 5,
  sig: '(Tensor X[n,d], Tensor y[n], number nEst=50, number lr=0.1, number maxDepth=3) -> object state',
  doc: '梯度提升回归树：每轮用回归树拟合负梯度（残差）并做收缩',
  ref: 'Friedman, Greedy Function Approximation: A Gradient Boosting Machine, Annals of Statistics, 2001',
  inv: '训练损失随提升轮数单调不增（shrinkage 足够小时）',
  impl: (X, y, nEst, lr, maxDepth) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const M = nEst === undefined ? 50 : Math.max(1, Math.floor(expectNumber(nEst, 'nEst')));
    const eta = lr === undefined ? 0.1 : expectNumber(lr, 'lr');
    if (!(eta > 0)) throw new ForgeError('E_VALUE', `lr 必须 > 0，当前 ${eta}`);
    const md = maxDepth === undefined ? 3 : Math.max(0, Math.floor(expectNumber(maxDepth, 'maxDepth')));
    const R = _rows(X), ys = _flat(y);
    const base = _mean(ys);
    const all = Array.from({ length: n }, (_, i) => i);
    const pred = new Array(n).fill(base);
    const trees = [];
    for (let m = 0; m < M; m++) {
      const res = ys.map((v, i) => v - pred[i]);
      const root = _buildTree(R, res, all, 0, {
        maxDepth: md, minSplit: 2, task: 'regress', classes: [],
        feats: Array.from({ length: d }, (_, j) => j),
      });
      for (let i = 0; i < n; i++) pred[i] += eta * _treeValue(root, R[i]);
      trees.push(root);
    }
    return { kind: 'ml.gb', n, d, nEst: M, lr: eta, maxDepth: md, base, trees };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.gb.predict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor y[m]',
  doc: '梯度提升树预测：base + lr·Σ 各轮回归树输出',
  ref: 'Friedman, Greedy Function Approximation: A Gradient Boosting Machine, 2001',
  inv: 'nEst=0 时退化为常数 base（训练集均值）',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    return _T1(R.map((r) => {
      let v = state.base;
      for (const t of state.trees) v += state.lr * _treeValue(t, r);
      return v;
    }));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.gb.stagedLoss', arity: 3,
  sig: '(object state, Tensor X[n,d], Tensor y[n]) -> Tensor loss[nEst]',
  doc: '逐轮累计的均方误差（第 m 项为前 m+1 棵树集成后的损失）',
  ref: 'Friedman 2001；Hastie et al., ESL 2009 §10.9-10.10',
  inv: 'lr 足够小时序列单调不增',
  impl: (state, X, y) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const R = _rows(X), ys = _flat(y);
    if (ys.length !== R.length) throw new ForgeError('E_SHAPE', 'X 行数与 y 长度不符');
    const pred = new Array(R.length).fill(state.base);
    const out = new Float64Array(state.trees.length);
    for (let m = 0; m < state.trees.length; m++) {
      for (let i = 0; i < R.length; i++) pred[i] += state.lr * _treeValue(state.trees[m], R[i]);
      let s = 0;
      for (let i = 0; i < R.length; i++) s += (ys[i] - pred[i]) ** 2;
      out[m] = s / R.length;
    }
    return new Tensor(out, [state.trees.length]);
  },
});

defineOp({
  id: nid(), domain: D,   name: 'ml.ada.fit', arity: 3,
  sig: '(Tensor X[n,d], Tensor y[n], number nEst=50) -> object state',
  doc: '离散 AdaBoost（SAMME）：决策树桩 + 加权多数投票，二分类退化为经典 AdaBoost.M1',
  ref: 'Freund & Schapire, A Decision-Theoretic Generalization of On-Line Learning, JCSS, 1997；Zhu et al., SAMME, 2009',
  inv: '加权训练误差随轮数下降；α_m = log((1−err)/err) + log(K−1)',
  impl: (X, y, nEst) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const M = nEst === undefined ? 50 : Math.max(1, Math.floor(expectNumber(nEst, 'nEst')));
    const R = _rows(X), ys = _flat(y);
    const classes = _classesOf(ys);
    const K = classes.length;
    if (K < 2) throw new ForgeError('E_VALUE', 'AdaBoost 需要至少 2 个类别');
    const tIdx = ys.map((v) => classes.indexOf(v));
    const w = new Array(n).fill(1 / n);
    const feats = Array.from({ length: d }, (_, j) => j);
    const learners = [];
    let errHist = [];
    for (let m = 0; m < M; m++) {
      const st = _bestStump(R, tIdx, w, classes, feats);
      if (!st) break;
      const err = Math.min(1 - 1e-12, Math.max(1e-12, st.err));
      if (err >= 1 - 1 / K - 1e-12) break;
      const alpha = Math.log((1 - err) / err) + Math.log(K - 1);
      for (let i = 0; i < n; i++) {
        const pred = R[i][st.feat] <= st.thr ? st.cL : st.cR;
        if (pred !== tIdx[i]) w[i] *= Math.exp(alpha);
      }
      const z = _sum(w);
      if (!(z > 0) || !Number.isFinite(z)) break;
      for (let i = 0; i < n; i++) w[i] /= z;
      learners.push({ feat: st.feat, thr: st.thr, cL: st.cL, cR: st.cR, alpha, err });
      errHist.push(st.err);
    }
    return { kind: 'ml.ada', n, d, classes, nEst: M, learners, errHist, fitted: learners.length };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.ada.predict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor labels[m]',
  doc: 'AdaBoost 预测：Σ_m α_m·[h_m(x)=k] 取最大',
  ref: 'Freund & Schapire 1997；Zhu et al., SAMME, 2009',
  inv: '预测标签必属于 state.classes',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    const classes = state.classes, K = classes.length;
    return _T1(R.map((r) => {
      const score = new Array(K).fill(0);
      for (const l of state.learners) {
        const c = r[l.feat] <= l.thr ? l.cL : l.cR;
        score[c] += l.alpha;
      }
      let bi = 0;
      for (let k = 1; k < K; k++) if (score[k] > score[bi]) bi = k;
      return classes[bi];
    }));
  },
});

// =====================================================================
// 五、E SVM 与核（10 条）：线性 SVM（hinge SGD）/ 简化 SMO / 核函数族
// =====================================================================
defineOp({
  id: nid(), domain: D, name: 'ml.svm.fit', arity: 5,
  sig: '(Tensor X[n,d], Tensor y[n], number C=1, number epochs=300, number lr=0.01) -> object state',
  doc: '线性 SVM：hinge 损失次梯度 SGD（Pegasos 风格投影步）',
  ref: 'Cortes & Vapnik, Support-Vector Networks, Machine Learning, 1995；Shalev-Shwartz et al., Pegasos, 2011',
  inv: '目标 (½‖w‖² + C·Σhinge) 随迭代下降；可分数据上间隔 > 0',
  impl: (X, y, C, epochs, lr) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const ys = _flat(y);
    const { neg, pos } = _binLabels(ys);
    const c = C === undefined ? 1 : expectNumber(C, 'C');
    if (!(c > 0)) throw new ForgeError('E_VALUE', `C 必须 > 0，当前 ${c}`);
    const eta = lr === undefined ? 0.01 : expectNumber(lr, 'lr');
    if (!(eta > 0)) throw new ForgeError('E_VALUE', `lr 必须 > 0，当前 ${eta}`);
    const ep = epochs === undefined ? 300 : Math.max(1, Math.floor(expectNumber(epochs, 'epochs')));
    const R = _rows(X);
    const t = ys.map((v) => (v === pos ? 1 : -1));
    const w = new Array(d).fill(0);
    let b = 0, obj = 0;
    for (let e = 0; e < ep; e++) {
      for (let i = 0; i < n; i++) {
        const margin = t[i] * (_dot(w, R[i]) + b);
        if (margin < 1) {
          for (let j = 0; j < d; j++) w[j] = w[j] * (1 - eta) + eta * c * t[i] * R[i][j];
          b += eta * c * t[i];
        } else {
          for (let j = 0; j < d; j++) w[j] *= (1 - eta);
        }
      }
    }
    let hinge = 0;
    for (let i = 0; i < n; i++) hinge += Math.max(0, 1 - t[i] * (_dot(w, R[i]) + b));
    obj = 0.5 * _dot(w, w) + c * hinge / Math.max(1, n);
    return { kind: 'ml.svm', n, d, negLabel: neg, posLabel: pos, w, b, C: c, lr: eta, epochs: ep, obj };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.svm.predict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor labels[m]',
  doc: 'SVM 硬分类：决策函数取符号（线性核与 RBF 核状态通用）',
  ref: 'Cortes & Vapnik 1995；Boser, Guyon & Vapnik 1992',
  inv: '预测标签 ∈ {negLabel, posLabel}',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    const score = (r) => {
      if (state.kind === 'ml.svm.kernel') {
        let s = state.b;
        for (let i = 0; i < state.sv.length; i++) s += state.svAlpha[i] * state.svY[i] * _rbf(state.sv[i], r, state.gamma);
        return s;
      }
      return _dot(state.w, r) + state.b;
    };
    return _T1(R.map((r) => (score(r) >= 0 ? state.posLabel : state.negLabel)));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.svm.decision', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor score[m]',
  doc: 'SVM 决策函数值：线性核为 wᵀx+b，核 SVM 为 Σα_i y_i k(x_i,x)+b',
  ref: 'Cortes & Vapnik 1995；Boser, Guyon & Vapnik 1992',
  inv: 'sign(decision) = predict 的标签映射',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X);
    if (state.kind === 'ml.svm.kernel') {
      return _T1(R.map((r) => {
        let s = state.b;
        for (let i = 0; i < state.sv.length; i++) s += state.svAlpha[i] * state.svY[i] * _rbf(state.sv[i], r, state.gamma);
        return s;
      }));
    }
    return _T1(R.map((r) => _dot(state.w, r) + state.b));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.svm.kernelFit', arity: 6,
  sig: '(Tensor X[n,d], Tensor y[n], number C=1, number gamma=0.5, number tol=1e-3, number maxIter=200) -> object state',
  doc: '核 SVM 二分类：Platt 简化 SMO（RBF 核），只保留支持向量',
  ref: 'Platt, Fast Training of Support Vector Machines using SMO, 1998；Boser, Guyon & Vapnik, 1992',
  inv: 'KKT 条件在 tol 内成立；0 ≤ α_i ≤ C',
  impl: (X, y, C, gamma, tol, maxIter) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const ys = _flat(y);
    const { neg, pos } = _binLabels(ys);
    const c = C === undefined ? 1 : expectNumber(C, 'C');
    const g = gamma === undefined ? 0.5 : expectNumber(gamma, 'gamma');
    if (!(c > 0)) throw new ForgeError('E_VALUE', `C 必须 > 0，当前 ${c}`);
    if (!(g > 0)) throw new ForgeError('E_VALUE', `gamma 必须 > 0，当前 ${g}`);
    const tv = tol === undefined ? 1e-3 : Math.max(0, expectNumber(tol, 'tol'));
    const mi = maxIter === undefined ? 200 : Math.max(1, Math.floor(expectNumber(maxIter, 'maxIter')));
    const R = _rows(X);
    const t = ys.map((v) => (v === pos ? 1 : -1));
    const K = Array.from({ length: n }, (_, i) => R.map((r) => _rbf(R[i], r, g)));
    const alphas = new Array(n).fill(0);
    let b = 0;
    const E = new Array(n).fill(0);
    const evalE = (i) => {
      let s = -t[i];
      for (let j = 0; j < n; j++) s += alphas[j] * t[j] * K[j][i];
      return s + b;
    };
    for (let it = 0; it < mi; it++) {
      let changed = 0;
      for (let i = 0; i < n; i++) {
        const Ei = evalE(i);
        E[i] = Ei;
        if (!((t[i] * Ei < -tv && alphas[i] < c) || (t[i] * Ei > tv && alphas[i] > 0))) continue;
        // 选使 |Ei − Ej| 最大的 j
        let j = -1, best = -1;
        for (let q = 0; q < n; q++) {
          if (q === i) continue;
          const eq = E[q] !== undefined && alphas[q] > 0 && alphas[q] < c ? E[q] : evalE(q);
          E[q] = eq;
          const diff = Math.abs(Ei - eq);
          if (diff > best) { best = diff; j = q; }
        }
        if (j < 0) continue;
        const Ej = E[j];
        const ai = alphas[i], aj = alphas[j];
        let Lp, Hp;
        if (t[i] === t[j]) { const gg = ai + aj; Lp = Math.max(0, gg - c); Hp = Math.min(c, gg); }
        else { const gg = aj - ai; Lp = Math.max(0, gg); Hp = Math.min(c, c + gg); }
        if (Math.abs(Hp - Lp) < 1e-15) continue;
        const eta = K[i][i] + K[j][j] - 2 * K[i][j];
        if (!(eta > 1e-12)) continue;
        let ajNew = aj + (t[j] * (Ei - Ej)) / eta;
        if (ajNew < Lp) ajNew = Lp;
        if (ajNew > Hp) ajNew = Hp;
        if (Math.abs(ajNew - aj) < 1e-12) continue;
        const aiNew = ai + t[i] * t[j] * (aj - ajNew);
        const b1 = b - Ei - t[i] * (aiNew - ai) * K[i][i] - t[j] * (ajNew - aj) * K[i][j];
        const b2 = b - Ej - t[i] * (aiNew - ai) * K[i][j] - t[j] * (ajNew - aj) * K[j][j];
        if (aiNew > 0 && aiNew < c) b = b1;
        else if (ajNew > 0 && ajNew < c) b = b2;
        else b = (b1 + b2) / 2;
        alphas[i] = aiNew; alphas[j] = ajNew;
        changed++;
      }
      if (changed === 0) break;
    }
    const svIdx = [];
    for (let i = 0; i < n; i++) if (alphas[i] > 1e-8) svIdx.push(i);
    return {
      kind: 'ml.svm.kernel', n, d, negLabel: neg, posLabel: pos,
      gamma: g, C: c, b,
      sv: svIdx.map((i) => R[i]),
      svY: svIdx.map((i) => t[i]),
      svAlpha: svIdx.map((i) => alphas[i]),
      nSV: svIdx.length,
    };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.kernel.linear', arity: 2,
  sig: '(Tensor A[m,d], Tensor B[n,d]) -> Tensor K[m,n]',
  doc: '线性核 K(x,z) = xᵀz',
  ref: 'Schölkopf & Smola, Learning with Kernels, 2002 §2.1',
  inv: 'K 对称半正定；K(x,x) = ‖x‖²',
  impl: (A, B) => {
    A = _asMat(A, 'A'); B = _asMat(B, 'B');
    if (A.shape[1] !== B.shape[1]) throw new ForgeError('E_SHAPE', 'A/B 特征维不一致');
    const ra = _rows(A), rb = _rows(B);
    const out = new Float64Array(ra.length * rb.length);
    ra.forEach((a, i) => rb.forEach((b, j) => { out[i * rb.length + j] = _dot(a, b); }));
    return new Tensor(out, [ra.length, rb.length]);
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.kernel.rbf', arity: 3,
  sig: '(Tensor A[m,d], Tensor B[n,d], number gamma=1) -> Tensor K[m,n]',
  doc: '高斯 RBF 核 K(x,z) = exp(−γ‖x−z‖²)',
  ref: 'Schölkopf & Smola, Learning with Kernels, 2002；Mercer 1909（正定性）',
  inv: 'K 对称半正定、对角元 = 1、元素 ∈ (0,1]',
  impl: (A, B, gamma) => {
    A = _asMat(A, 'A'); B = _asMat(B, 'B');
    if (A.shape[1] !== B.shape[1]) throw new ForgeError('E_SHAPE', 'A/B 特征维不一致');
    const g = gamma === undefined ? 1 : expectNumber(gamma, 'gamma');
    if (!(g > 0)) throw new ForgeError('E_VALUE', `gamma 必须 > 0，当前 ${g}`);
    const ra = _rows(A), rb = _rows(B);
    const out = new Float64Array(ra.length * rb.length);
    ra.forEach((a, i) => rb.forEach((b, j) => { out[i * rb.length + j] = _rbf(a, b, g); }));
    return new Tensor(out, [ra.length, rb.length]);
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.kernel.poly', arity: 5,
  sig: '(Tensor A[m,d], Tensor B[n,d], number degree=3, number coef0=1, number gamma=1) -> Tensor K[m,n]',
  doc: '多项式核 K(x,z) = (γ·xᵀz + coef0)^degree',
  ref: 'Schölkopf & Smola, Learning with Kernels, 2002；Vapnik 1995',
  inv: 'degree=1 且 coef0=0 时退化为线性核的 γ 倍',
  impl: (A, B, degree, coef0, gamma) => {
    A = _asMat(A, 'A'); B = _asMat(B, 'B');
    if (A.shape[1] !== B.shape[1]) throw new ForgeError('E_SHAPE', 'A/B 特征维不一致');
    const deg = degree === undefined ? 3 : Math.max(1, Math.floor(expectNumber(degree, 'degree')));
    const c0 = coef0 === undefined ? 1 : expectNumber(coef0, 'coef0');
    const g = gamma === undefined ? 1 : expectNumber(gamma, 'gamma');
    const ra = _rows(A), rb = _rows(B);
    const out = new Float64Array(ra.length * rb.length);
    ra.forEach((a, i) => rb.forEach((b, j) => { out[i * rb.length + j] = (g * _dot(a, b) + c0) ** deg; }));
    return new Tensor(out, [ra.length, rb.length]);
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.kernel.laplacian', arity: 3,
  sig: '(Tensor A[m,d], Tensor B[n,d], number gamma=1) -> Tensor K[m,n]',
  doc: '拉普拉斯核 K(x,z) = exp(−γ‖x−z‖₁)',
  ref: 'Schölkopf & Smola, Learning with Kernels, 2002 §4.4（L1 距离对应的平稳核）',
  inv: 'K 对称半正定、对角元 = 1',
  impl: (A, B, gamma) => {
    A = _asMat(A, 'A'); B = _asMat(B, 'B');
    if (A.shape[1] !== B.shape[1]) throw new ForgeError('E_SHAPE', 'A/B 特征维不一致');
    const g = gamma === undefined ? 1 : expectNumber(gamma, 'gamma');
    if (!(g > 0)) throw new ForgeError('E_VALUE', `gamma 必须 > 0，当前 ${g}`);
    const ra = _rows(A), rb = _rows(B);
    const out = new Float64Array(ra.length * rb.length);
    ra.forEach((a, i) => rb.forEach((b, j) => { out[i * rb.length + j] = Math.exp(-g * _dist1(a, b)); }));
    return new Tensor(out, [ra.length, rb.length]);
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.kernel.cosine', arity: 2,
  sig: '(Tensor A[m,d], Tensor B[n,d]) -> Tensor K[m,n]',
  doc: '余弦相似度核 K(x,z) = xᵀz / (‖x‖‖z‖)',
  ref: 'Schölkopf & Smola 2002；Salton & McGill《Introduction to Modern IR》1983（余弦测度）',
  inv: '对角元 = 1；元素 ∈ [−1,1]；对向量缩放不变',
  impl: (A, B) => {
    A = _asMat(A, 'A'); B = _asMat(B, 'B');
    if (A.shape[1] !== B.shape[1]) throw new ForgeError('E_SHAPE', 'A/B 特征维不一致');
    const ra = _rows(A), rb = _rows(B);
    const na = ra.map((a) => Math.sqrt(_dot(a, a)) || 1);
    const nb = rb.map((b) => Math.sqrt(_dot(b, b)) || 1);
    const out = new Float64Array(ra.length * rb.length);
    ra.forEach((a, i) => rb.forEach((b, j) => { out[i * rb.length + j] = _dot(a, b) / (na[i] * nb[j]); }));
    return new Tensor(out, [ra.length, rb.length]);
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.kernel.center', arity: 1,
  sig: '(Tensor K[n,n]) -> Tensor Kc[n,n]',
  doc: '核矩阵中心化：Kc = (I − 11ᵀ/n) K (I − 11ᵀ/n)，即特征空间去均值',
  ref: 'Schölkopf, Smola & Müller, Nonlinear Component Analysis as a Kernel Eigenvalue Problem, 1998',
  inv: 'Kc 的行和与列和均为 0',
  impl: (K) => {
    K = _asMat(K, 'K');
    if (K.shape[0] !== K.shape[1]) throw new ForgeError('E_SHAPE', 'K 必须是方阵');
    const M = _rows(K);
    return _T2(_centerKernel(M));
  },
});

// =====================================================================
// 六、F 近邻（4 条）：k-NN 分类 / 概率 / 回归
// =====================================================================
defineOp({
  id: nid(), domain: D, name: 'ml.knn.fit', arity: 2,
  sig: '(Tensor X[n,d], Tensor y[n]) -> object state',
  doc: 'k 近邻惰性学习器：仅存储训练集（任务自动判别分类/回归）',
  ref: 'Cover & Hart, Nearest Neighbor Pattern Classification, IEEE Trans. IT, 1967',
  inv: 'k=1 时训练集留一误差为 0（无重复样本时训练误差为 0）',
  impl: (X, y) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const ys = _flat(y);
    const task = _isClassify(ys) ? 'classify' : 'regress';
    return {
      kind: 'ml.knn', n, d, task,
      rows: _rows(X), labels: ys, classes: task === 'classify' ? _classesOf(ys) : [],
    };
  },
});

function _knnNeighbors(state, X, k) {
  X = _asMat(X, 'X');
  const kk = Math.max(1, Math.floor(expectNumber(k, 'k')));
  const R = _rows(X), train = state.rows;
  const out = [];
  for (const r of R) {
    const ds = train.map((t, i) => [_dist2(r, t), i]);
    ds.sort((a, b) => a[0] - b[0]);
    out.push(ds.slice(0, Math.min(kk, ds.length)).map((v) => v[1]));
  }
  return { R, out, kk };
}

defineOp({
  id: nid(), domain: D, name: 'ml.knn.predict', arity: 3,
  sig: '(object state, Tensor X[m,d], number k=5) -> Tensor labels[m]',
  doc: 'k 近邻多数投票分类（平票取标签值较小者）',
  ref: 'Cover & Hart 1967',
  inv: 'k=1 时训练集预测误差为 0',
  impl: (state, X, k) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    const { out, kk } = _knnNeighbors(state, X, k);
    return _T1(out.map((nb) => {
      const cnt = new Map();
      for (const i of nb) {
        const v = state.labels[i];
        cnt.set(v, (cnt.get(v) || 0) + 1);
      }
      let bl = null, bc = -1;
      for (const [lab, c] of cnt) if (c > bc || (c === bc && lab < bl)) { bc = c; bl = lab; }
      return bl;
    }));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.knn.predictProba', arity: 3,
  sig: '(object state, Tensor X[m,d], number k=5) -> Tensor P[m,K]',
  doc: 'k 近邻类概率（邻域内类别频率）',
  ref: 'Cover & Hart 1967',
  inv: '每行非负且和为 1；列序与 state.classes 一致',
  impl: (state, X, k) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    if (state.task !== 'classify') throw new ForgeError('E_VALUE', 'predictProba 仅适用于分类任务');
    const { R, out } = _knnNeighbors(state, X, k);
    const K = state.classes.length;
    const res = new Float64Array(R.length * K);
    out.forEach((nb, i) => {
      for (const j of nb) {
        const ci = state.classes.indexOf(state.labels[j]);
        if (ci >= 0) res[i * K + ci] += 1 / nb.length;
      }
    });
    return new Tensor(res, [R.length, K]);
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.knn.regress', arity: 3,
  sig: '(object state, Tensor X[m,d], number k=5) -> Tensor y[m]',
  doc: 'k 近邻回归（邻域标签均值）',
  ref: 'Cover & Hart 1967；Altman, An Introduction to Kernel and Nearest-Neighbor Nonparametric Regression, 1992',
  inv: 'k=1 时训练集预测值等于训练标签',
  impl: (state, X, k) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    const { R, out } = _knnNeighbors(state, X, k);
    return _T1(out.map((nb) => {
      let s = 0;
      for (const j of nb) s += state.labels[j];
      return s / nb.length;
    }));
  },
});

// =====================================================================
// 七、G 聚类（13 条）：k-means / DBSCAN / 层次 / 谱聚类 / 轮廓系数
// =====================================================================
defineOp({
  id: nid(), domain: D, name: 'ml.kmeans.fit', arity: 5,
  sig: '(Tensor X[n,d], number k, number seed=20261014, number iters=100, string init="kmeanspp") -> object state',
  doc: 'k-means（Lloyd 迭代），初始化支持 k-means++ 或随机',
  ref: 'Lloyd, Least Squares Quantization in PCM, 1982；Arthur & Vassilvitskii, k-means++, SODA 2007',
  inv: '每轮迭代惯量（inertia）单调不增',
  impl: (X, k, seed, iters, init) => {
    X = _asMat(X, 'X');
    expectNumber(k, 'k');
    const R = _rows(X);
    const kk = Math.floor(k);
    if (!(kk >= 1) || kk > R.length) throw new ForgeError('E_VALUE', `k 必须 ∈ [1, ${R.length}]，当前 ${kk}`);
    const ini = init === undefined ? 'kmeanspp' : String(init);
    if (ini !== 'kmeanspp' && ini !== 'random') throw new ForgeError('E_VALUE', `init 必须为 kmeanspp|random，当前 ${ini}`);
    const r = _kmeans(R, kk, seed, iters, ini);
    return { kind: 'ml.kmeans', centers: r.centers, labels: r.labels, inertia: r.inertia, k: r.k, d: r.d, n: r.n, init: ini };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.kmeans.predict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor labels[m]',
  doc: 'k-means 预测：指派到最近质心',
  ref: 'Lloyd 1982',
  inv: '标签 ∈ [0, k)；指派结果使 Σ‖x−c‖² 最小',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X), C = state.centers;
    return _T1(R.map((r) => {
      let bi = 0, bd = Infinity;
      for (let c = 0; c < C.length; c++) {
        const dd = _dist2(r, C[c]);
        if (dd < bd) { bd = dd; bi = c; }
      }
      return bi;
    }));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.kmeans.inertia', arity: 2,
  sig: '(object state, Tensor X[n,d]) -> number',
  doc: 'k-means 惯量：Σ‖x_i − c_{l_i}‖²（簇内平方和）',
  ref: 'Lloyd 1982；Hastie et al., ESL 2009 §14.3',
  inv: '惯量 ≥ 0；k=n 时为 0',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X), C = state.centers;
    let s = 0;
    for (const r of R) {
      let bd = Infinity, bi = 0;
      for (let c = 0; c < C.length; c++) {
        const dd = _dist2(r, C[c]);
        if (dd < bd) { bd = dd; bi = c; }
      }
      s += _dist2(r, C[bi]);
    }
    return s;
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.kmeanspp.init', arity: 3,
  sig: '(Tensor X[n,d], number k, number seed=20261014) -> Tensor C[k,d]',
  doc: 'k-means++ D² 加权采样初始化质心',
  ref: 'Arthur & Vassilvitskii, k-means++: The Advantages of Careful Seeding, SODA 2007',
  inv: '质心均取自训练样本；期望代价 ≤ O(log k)·OPT',
  impl: (X, k, seed) => {
    X = _asMat(X, 'X');
    expectNumber(k, 'k');
    const R = _rows(X);
    const kk = Math.floor(k);
    if (!(kk >= 1) || kk > R.length) throw new ForgeError('E_VALUE', `k 必须 ∈ [1, ${R.length}]，当前 ${kk}`);
    const sd = seed === undefined ? 20261014 : Math.floor(expectNumber(seed, 'seed'));
    return _T2(_kmeansppPick(R, kk, createRng(sd)));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.dbscan.fit', arity: 3,
  sig: '(Tensor X[n,d], number eps, number minPts=4) -> object state',
  doc: 'DBSCAN 基于密度的聚类，噪声点标签为 −1',
  ref: 'Ester, Kriegel, Sander & Xu, A Density-Based Algorithm for Discovering Clusters, KDD 1996',
  inv: '核心点被分入某簇；稀疏区域的点标签为 −1',
  impl: (X, eps, minPts) => {
    X = _asMat(X, 'X');
    const e = expectNumber(eps, 'eps');
    if (!(e > 0)) throw new ForgeError('E_VALUE', `eps 必须 > 0，当前 ${e}`);
    const mp = minPts === undefined ? 4 : Math.max(1, Math.floor(expectNumber(minPts, 'minPts')));
    const R = _rows(X), n = R.length;
    const eps2 = e * e;
    const nbrs = R.map((r) => {
      const out = [];
      for (let j = 0; j < n; j++) if (_dist2(r, R[j]) <= eps2) out.push(j);
      return out;
    });
    const core = nbrs.map((v) => v.length >= mp);
    const labels = new Array(n).fill(-1);
    let cid = 0;
    for (let i = 0; i < n; i++) {
      if (labels[i] !== -1 || !core[i]) continue;
      const c = cid++;
      labels[i] = c;
      const queue = nbrs[i].filter((v) => v !== i);
      while (queue.length) {
        const q = queue.pop();
        if (labels[q] === -1) {
          labels[q] = c;
          if (core[q]) for (const v of nbrs[q]) if (labels[v] === -1) queue.push(v);
        }
      }
    }
    return { kind: 'ml.dbscan', n, d: X.shape[1], eps: e, minPts: mp, labels, core, rows: R, nClusters: cid };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.dbscan.predict', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor labels[m]',
  doc: 'DBSCAN 对新样本指派：eps 内最近核心点的簇，否则 −1',
  ref: 'Ester et al., KDD 1996',
  inv: '不在任何核心点 eps 邻域内的样本标签为 −1',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X), train = state.rows, labs = state.labels, core = state.core;
    const eps2 = state.eps * state.eps;
    return _T1(R.map((r) => {
      let bi = -1, bd = Infinity;
      for (let i = 0; i < train.length; i++) {
        if (!core[i]) continue;
        const dd = _dist2(r, train[i]);
        if (dd < bd) { bd = dd; bi = i; }
      }
      return bd <= eps2 ? labs[bi] : -1;
    }));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.agglo.fit', arity: 3,
  sig: '(Tensor X[n,d], number k=2, string linkage="average") -> object state',
  doc: '凝聚层次聚类：自底向上按 linkage 合并，最后切成 k 簇',
  ref: 'Sneath & Sokal《Numerical Taxonomy》1973；Sokal & Michener, A Statistical Method for Evaluating Systematic Relationships, 1958',
  inv: '链接距离序列单调不减；k=1 时全部同簇，k=n 时各成一类',
  impl: (X, k, linkage) => {
    X = _asMat(X, 'X');
    const R = _rows(X), n = R.length;
    const kk = k === undefined ? 2 : Math.max(1, Math.floor(expectNumber(k, 'k')));
    if (kk > n) throw new ForgeError('E_VALUE', `k 不能大于样本数 ${n}，当前 ${kk}`);
    const lk = linkage === undefined ? 'average' : String(linkage);
    if (lk !== 'single' && lk !== 'complete' && lk !== 'average') {
      throw new ForgeError('E_VALUE', `linkage 必须为 single|complete|average，当前 ${lk}`);
    }
    const { Z, method } = _linkage(R, lk);
    const labels = kk <= 1 ? new Array(n).fill(0) : (kk >= n ? Array.from({ length: n }, (_, i) => i) : _cutLabels(Z, n, kk));
    return { kind: 'ml.agglo', n, d: X.shape[1], k: kk, linkage: method, Z, labels };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.agglo.labels', arity: 1,
  sig: '(object state) -> Tensor labels[n]',
  doc: '取出层次聚类在指定 k 下的簇标签',
  ref: 'Sneath & Sokal 1973',
  inv: '标签取值集合的基数 ≤ k',
  impl: (state) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    return _T1(state.labels);
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.linkage.single', arity: 1,
  sig: '(Tensor X[n,d]) -> object {Z, n, method}',
  doc: '单链接层次聚类（最近邻距离），返回 scipy 风格链接矩阵 Z',
  ref: 'Sneath 1957；Lance & Williams, A General Theory of Classificatory Sorting Strategies, 1967',
  inv: 'Z 的合并距离单调不减；d(A∪B, C) = min(d(A,C), d(B,C))',
  impl: (X) => {
    X = _asMat(X, 'X');
    return _linkage(_rows(X), 'single');
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.linkage.complete', arity: 1,
  sig: '(Tensor X[n,d]) -> object {Z, n, method}',
  doc: '全链接层次聚类（最远距离）',
  ref: 'Sokal & Michener 1958；Lance & Williams 1967',
  inv: 'Z 的合并距离单调不减；d(A∪B, C) = max(d(A,C), d(B,C))',
  impl: (X) => {
    X = _asMat(X, 'X');
    return _linkage(_rows(X), 'complete');
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.linkage.average', arity: 1,
  sig: '(Tensor X[n,d]) -> object {Z, n, method}',
  doc: '均链接层次聚类（UPGMA，按簇大小加权平均）',
  ref: 'Sokal & Michener 1958；Lance & Williams 1967',
  inv: 'Z 的合并距离单调不减；d(A∪B,C) = (|A|d(A,C)+|B|d(B,C))/(|A|+|B|)',
  impl: (X) => {
    X = _asMat(X, 'X');
    return _linkage(_rows(X), 'average');
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.spectral.fit', arity: 3,
  sig: '(Tensor X[n,d], number k=2, number gamma=auto) -> object state',
  doc: '谱聚类（Ng-Jordan-Weiss）：RBF 相似图 → 归一化拉普拉斯 → k 个最小特征向量 → k-means',
  ref: 'Ng, Jordan & Weiss, On Spectral Clustering: Analysis and an Algorithm, NeurIPS 2001；Shi & Malik, PAMI 2000',
  inv: 'L_sym = I − D^{−1/2}WD^{−1/2} 半正定；特征向量行归一后行范数为 1',
  impl: (X, k, gamma) => {
    X = _asMat(X, 'X');
    const R = _rows(X), n = R.length, d = X.shape[1];
    const kk = k === undefined ? 2 : Math.max(1, Math.floor(expectNumber(k, 'k')));
    if (kk > n) throw new ForgeError('E_VALUE', `k 不能大于样本数 ${n}，当前 ${kk}`);
    const g = gamma === undefined ? _autoGamma(R) : expectNumber(gamma, 'gamma');
    if (!(g > 0)) throw new ForgeError('E_VALUE', `gamma 必须 > 0，当前 ${g}`);
    const W = R.map((a) => R.map((b) => _rbf(a, b, g)));
    for (let i = 0; i < n; i++) W[i][i] = 0;
    const deg = W.map((r) => _sum(r));
    const L = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = (i === j ? 1 : 0) - W[i][j] / Math.sqrt(Math.max(deg[i] * deg[j], _EPS));
        L[i][j] = v;
      }
    }
    const { values, vectors } = _eigSym(L);
    // 取最小的 k 个特征值（eigenSym 降序，故取末尾）
    const sel = [];
    for (let m = 0; m < kk; m++) sel.push(vectors[n - 1 - m]);
    const V = R.map((_, i) => sel.map((v) => v[i]));
    const Vn = V.map((r) => {
      const nn = Math.sqrt(_dot(r, r));
      return nn > 0 ? r.map((v) => v / nn) : r.map(() => 0);
    });
    const km = _kmeans(Vn, kk, 20261014, 50, 'kmeanspp');
    return {
      kind: 'ml.spectral', n, d, k: kk, gamma: g, labels: km.labels,
      eigen: sel.map((_, m) => values[n - 1 - m]),
      embedding: Vn,
    };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.spectral.labels', arity: 1,
  sig: '(object state) -> Tensor labels[n]',
  doc: '取出谱聚类的簇标签',
  ref: 'Ng, Jordan & Weiss, NeurIPS 2001',
  inv: '标签取值集合的基数 ≤ k',
  impl: (state) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    return _T1(state.labels);
  },
});

// =====================================================================
// 八、H 降维（8 条）：PCA / 核 PCA / 经典 MDS / 白化
// =====================================================================
defineOp({
  id: nid(), domain: D, name: 'ml.pca.fit', arity: 2,
  sig: '(Tensor X[n,d], number nComp=d) -> object state',
  doc: '主成分分析：中心化协方差的对称特征分解，载荷矩阵按特征值降序',
  ref: 'Pearson, On Lines and Planes of Closest Fit, 1901；Hotelling, Analysis of a Complex of Statistical Variables, 1933',
  inv: '主成分得分协方差的非对角元 ≈ 0；特征值之和 = 总方差',
  impl: (X, nComp) => {
    X = _asMat(X, 'X');
    const n = X.shape[0], d = X.shape[1];
    if (n < 1) throw new ForgeError('E_VALUE', '至少需要 1 个样本');
    const nc = nComp === undefined ? d : Math.max(1, Math.min(d, Math.floor(expectNumber(nComp, 'nComp'))));
    const R = _rows(X);
    const mean = _colMean(R, d);
    const Rc = _center(R, mean);
    const den = Math.max(1, n - 1);
    const S = Array.from({ length: d }, () => new Array(d).fill(0));
    for (let a = 0; a < d; a++) for (let b = a; b < d; b++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += Rc[i][a] * Rc[i][b];
      S[a][b] = s / den; S[b][a] = S[a][b];
    }
    const { values, vectors } = _eigSym(S);
    const components = [];
    for (let k = 0; k < nc; k++) components.push(vectors[k]);
    const total = _sum(values);
    return {
      kind: 'ml.pca', n, d, nComp: nc, mean, components,
      eigenvalues: values, totalVariance: total,
      explainedVarRatio: values.map((v) => v / Math.max(total, _EPS)),
    };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.pca.transform', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor Z[m,nComp]',
  doc: 'PCA 投影：Z = (X − mean)·Wᵀ',
  ref: 'Pearson 1901；Hotelling 1933；Jolliffe《Principal Component Analysis》2002',
  inv: '得分矩阵的列间协方差非对角元 ≈ 0',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X), mean = state.mean, W = state.components, nc = W.length;
    const out = new Float64Array(R.length * nc);
    R.forEach((r, i) => {
      for (let k = 0; k < nc; k++) {
        let s = 0;
        for (let j = 0; j < mean.length; j++) s += (r[j] - mean[j]) * W[k][j];
        out[i * nc + k] = s;
      }
    });
    return new Tensor(out, [R.length, nc]);
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.pca.inverse', arity: 2,
  sig: '(object state, Tensor Z[m,nComp]) -> Tensor X̂[m,d]',
  doc: 'PCA 逆变换：X̂ = Z·W + mean（主成分子空间内的重构）',
  ref: 'Pearson 1901；Jolliffe 2002',
  inv: 'nComp=d 时 inverse(transform(x)) = x 精确成立',
  impl: (state, Z) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    Z = _asMat(Z, 'Z');
    if (Z.shape[1] !== state.components.length) {
      throw new ForgeError('E_SHAPE', `Z 列数 ${Z.shape[1]} 与 nComp ${state.components.length} 不符`);
    }
    const rows = _rows(Z), W = state.components, mean = state.mean, d = state.d;
    const out = new Float64Array(rows.length * d);
    rows.forEach((z, i) => {
      for (let j = 0; j < d; j++) {
        let s = mean[j];
        for (let k = 0; k < W.length; k++) s += z[k] * W[k][j];
        out[i * d + j] = s;
      }
    });
    return new Tensor(out, [rows.length, d]);
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.pca.explainedVar', arity: 1,
  sig: '(object state) -> Tensor ev[nComp]',
  doc: '各主成分解释的方差（协方差特征值，降序）',
  ref: 'Jolliffe《Principal Component Analysis》2002 §2.6',
  inv: 'nComp=d 时各分量之和 = 总方差（协方差矩阵的迹）',
  impl: (state) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    return _T1(state.eigenvalues.slice(0, state.nComp));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.kpca.fit', arity: 3,
  sig: '(Tensor X[n,d], number nComp=2, number gamma=auto) -> object state',
  doc: '核 PCA：RBF 核矩阵中心化后特征分解，得到特征空间主成分方向',
  ref: 'Schölkopf, Smola & Müller, Nonlinear Component Analysis as a Kernel Eigenvalue Problem, Neural Computation, 1998',
  inv: '中心化核的特征值非负；投影方向的核范数归一（αᵀKα = 1）',
  impl: (X, nComp, gamma) => {
    X = _asMat(X, 'X');
    const R = _rows(X), n = R.length, d = X.shape[1];
    if (n < 2) throw new ForgeError('E_VALUE', '核 PCA 至少需要 2 个样本');
    const nc = nComp === undefined ? 2 : Math.max(1, Math.min(n, Math.floor(expectNumber(nComp, 'nComp'))));
    const g = gamma === undefined ? _autoGamma(R) : expectNumber(gamma, 'gamma');
    if (!(g > 0)) throw new ForgeError('E_VALUE', `gamma 必须 > 0，当前 ${g}`);
    const K = R.map((a) => R.map((b) => _rbf(a, b, g)));
    const colMean = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) colMean[j] += K[i][j];
    for (let j = 0; j < n; j++) colMean[j] /= n;
    let all = 0;
    for (const v of colMean) all += v;
    all /= n;
    const Kc = _centerKernel(K);
    const { values, vectors } = _eigSym(Kc);
    const alphas = [];
    const eig = [];
    for (let k = 0; k < nc; k++) {
      const lam = Math.max(values[k], 0);
      // α = v/√λ 使特征空间主成分方向单位范数（αᵀKα = 1）；
      // 训练集投影坐标 = √λ·v，其无偏方差 = λ/(n−1)，与 PCA 的解释方差口径一致
      const scale = Math.sqrt(lam);
      alphas.push(vectors[k].map((v) => (scale > _EPS ? v / scale : 0)));
      eig.push(lam / Math.max(1, n - 1));
    }
    return {
      kind: 'ml.kpca', n, d, nComp: nc, gamma: g,
      xs: R, alphas, eigenvalues: eig, kernelColMean: colMean, kernelMean: all,
    };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.kpca.transform', arity: 2,
  sig: '(object state, Tensor X[m,d]) -> Tensor Z[m,nComp]',
  doc: '核 PCA 投影：测试核矩阵按训练集统计量中心化后与 α 相乘',
  ref: 'Schölkopf, Smola & Müller 1998',
  inv: '对训练集自身投影得到 √λ 缩放的主成分坐标',
  impl: (state, X) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    X = _asMat(X, 'X');
    const R = _rows(X), xs = state.xs, n = xs.length, alphas = state.alphas, nc = alphas.length;
    const out = new Float64Array(R.length * nc);
    R.forEach((r, i) => {
      const row = xs.map((s) => _rbf(r, s, state.gamma));
      let rowMean = 0;
      for (const v of row) rowMean += v;
      rowMean /= n;
      for (let k = 0; k < nc; k++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += (row[j] - state.kernelColMean[j] - rowMean + state.kernelMean) * alphas[k][j];
        out[i * nc + k] = s;
      }
    });
    return new Tensor(out, [R.length, nc]);
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.mds.classical', arity: 2,
  sig: '(Tensor D[n,n], number nComp=2) -> Tensor Y[n,nComp]',
  doc: '经典多维尺度分析（Torgerson）：双中心化内积矩阵的特征映射',
  ref: 'Torgerson, Multidimensional Scaling: I. Theory and Method, Psychometrika, 1952；Mardia et al. 1979',
  inv: '嵌入的欧氏距离在 nComp 足够时逼近输入距离矩阵',
  impl: (D, nComp) => {
    D = _asMat(D, 'D');
    const n = D.shape[0];
    if (D.shape[1] !== n) throw new ForgeError('E_SHAPE', 'D 必须是方阵距离矩阵');
    const nc = nComp === undefined ? 2 : Math.max(1, Math.min(n, Math.floor(expectNumber(nComp, 'nComp'))));
    const M = _rows(D);
    // B = −½·J·D²·J（J = I − 11ᵀ/n）
    const rowMean = new Array(n).fill(0), colMean = new Array(n).fill(0);
    let all = 0;
    const D2 = M.map((r) => r.map((v) => v * v));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { rowMean[i] += D2[i][j]; colMean[j] += D2[i][j]; all += D2[i][j]; }
    for (let i = 0; i < n; i++) { rowMean[i] /= n; colMean[i] /= n; }
    all /= n * n;
    const B = D2.map((r, i) => r.map((v, j) => -0.5 * (v - rowMean[i] - colMean[j] + all)));
    const { values, vectors } = _eigSym(B);
    const out = new Float64Array(n * nc);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < nc; k++) out[i * nc + k] = vectors[k][i] * Math.sqrt(Math.max(values[k], 0));
    }
    return new Tensor(out, [n, nc]);
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.whiten', arity: 1,
  sig: '(Tensor X[n,d]) -> Tensor Z[n,d]',
  doc: 'ZCA 白化：去相关并把各主成分缩放到单位方差（仍留在原坐标系）',
  ref: 'Bell & Sejnowski, The "Independent Components" of Natural Scenes, Vision Research, 1997；Hyvärinen et al., ICA, 2001',
  inv: '白化后各列均值 ≈ 0、协方差矩阵 ≈ 单位阵',
  impl: (X) => {
    X = _asMat(X, 'X');
    const n = X.shape[0], d = X.shape[1];
    if (n < 2) throw new ForgeError('E_VALUE', '白化至少需要 2 个样本');
    const R = _rows(X);
    const mean = _colMean(R, d);
    const Rc = _center(R, mean);
    const den = Math.max(1, n - 1);
    const S = Array.from({ length: d }, () => new Array(d).fill(0));
    for (let a = 0; a < d; a++) for (let b = a; b < d; b++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += Rc[i][a] * Rc[i][b];
      S[a][b] = s / den; S[b][a] = S[a][b];
    }
    const { values, vectors } = _eigSym(S);
    const out = new Float64Array(n * d);
    for (let i = 0; i < n; i++) {
      const z = new Array(d).fill(0);
      for (let k = 0; k < d; k++) {
        const lam = Math.max(values[k], 1e-12);
        let s = 0;
        for (let j = 0; j < d; j++) s += Rc[i][j] * vectors[k][j];
        const c = s / Math.sqrt(lam);
        for (let j = 0; j < d; j++) z[j] += c * vectors[k][j];
      }
      for (let j = 0; j < d; j++) out[i * d + j] = z[j];
    }
    return new Tensor(out, [n, d]);
  },
});

// =====================================================================
// 九、I 预处理（9 条）：标准化 / 缩放 / 编码 / 缺失值填补
// =====================================================================
defineOp({
  id: nid(), domain: D, name: 'ml.scale.standard', arity: 1,
  sig: '(Tensor X[n,d]) -> object {X, mean, std}',
  doc: '标准化（Z-score）：每列减均值除以标准差',
  ref: 'Hastie et al., ESL 2009 §3.4；scikit-learn StandardScaler',
  inv: '变换后每列均值 ≈ 0、标准差 ≈ 1',
  impl: (X) => {
    X = _asMat(X, 'X');
    const R = _rows(X), d = X.shape[1], n = R.length;
    const mean = _colMean(R, d);
    const std = new Array(d).fill(0);
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += (R[i][j] - mean[j]) ** 2;
      std[j] = Math.sqrt(s / Math.max(1, n)) || 1;
    }
    const out = new Float64Array(n * d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) out[i * d + j] = (R[i][j] - mean[j]) / std[j];
    return { X: new Tensor(out, [n, d]), mean, std };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.scale.minmax', arity: 3,
  sig: '(Tensor X[n,d], number lo=0, number hi=1) -> object {X, min, max}',
  doc: '最小-最大缩放：把每列线性映射到 [lo, hi]',
  ref: 'scikit-learn MinMaxScaler；Han & Kamber《Data Mining》2006 §3.5',
  inv: '变换后每列最小值 = lo、最大值 = hi（常数列退化为 lo）',
  impl: (X, lo, hi) => {
    X = _asMat(X, 'X');
    const a = lo === undefined ? 0 : expectNumber(lo, 'lo');
    const b = hi === undefined ? 1 : expectNumber(hi, 'hi');
    if (b <= a) throw new ForgeError('E_VALUE', `hi 必须 > lo，当前 [${a}, ${b}]`);
    const R = _rows(X), d = X.shape[1], n = R.length;
    const mn = new Array(d).fill(Infinity), mx = new Array(d).fill(-Infinity);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) {
      if (R[i][j] < mn[j]) mn[j] = R[i][j];
      if (R[i][j] > mx[j]) mx[j] = R[i][j];
    }
    const out = new Float64Array(n * d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) {
      const span = mx[j] - mn[j];
      out[i * d + j] = span > 0 ? a + ((R[i][j] - mn[j]) / span) * (b - a) : a;
    }
    return { X: new Tensor(out, [n, d]), min: mn, max: mx };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.scale.robust', arity: 1,
  sig: '(Tensor X[n,d]) -> object {X, center, scale}',
  doc: '稳健缩放：减中位数除以四分位距（对异常值不敏感）',
  ref: 'Huber 1964；scikit-learn RobustScaler（IQR 定义同 Tukey 1977）',
  inv: '变换后每列中位数 ≈ 0、四分位距 ≈ 1',
  impl: (X) => {
    X = _asMat(X, 'X');
    const R = _rows(X), d = X.shape[1], n = R.length;
    const center = new Array(d).fill(0), scale = new Array(d).fill(1);
    for (let j = 0; j < d; j++) {
      const col = R.map((r) => r[j]).sort((p, q) => p - q);
      const q = (p) => {
        const idx = p * (col.length - 1);
        const lo = Math.floor(idx), hi = Math.ceil(idx);
        return lo === hi ? col[lo] : col[lo] + (col[hi] - col[lo]) * (idx - lo);
      };
      const q1 = q(0.25), q3 = q(0.75);
      center[j] = q(0.5);
      scale[j] = q3 - q1 > 0 ? q3 - q1 : 1;
    }
    const out = new Float64Array(n * d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) out[i * d + j] = (R[i][j] - center[j]) / scale[j];
    return { X: new Tensor(out, [n, d]), center, scale };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.scale.maxabs', arity: 1,
  sig: '(Tensor X[n,d]) -> object {X, maxAbs}',
  doc: '最大绝对值缩放：每列除以该列绝对值的最大值（保稀疏零元）',
  ref: 'scikit-learn MaxAbsScaler；Hastie et al., ESL 2009 §3.4',
  inv: '变换后每列元素 ∈ [−1, 1]，且至少有一个绝对值为 1',
  impl: (X) => {
    X = _asMat(X, 'X');
    const R = _rows(X), d = X.shape[1], n = R.length;
    const ma = new Array(d).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) ma[j] = Math.max(ma[j], Math.abs(R[i][j]));
    const out = new Float64Array(n * d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) out[i * d + j] = R[i][j] / (ma[j] || 1);
    return { X: new Tensor(out, [n, d]), maxAbs: ma };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.scale.normalize', arity: 1,
  sig: '(Tensor X[n,d]) -> object {X, norms}',
  doc: '样本级归一化：每行缩放到单位 L2 范数（文本/余弦场景常用）',
  ref: 'Salton & McGill 1983；scikit-learn normalize',
  inv: '变换后每个行向量的 L2 范数 = 1（零行保持为 0）',
  impl: (X) => {
    X = _asMat(X, 'X');
    const R = _rows(X), d = X.shape[1], n = R.length;
    const norms = R.map((r) => Math.sqrt(_dot(r, r)));
    const out = new Float64Array(n * d);
    for (let i = 0; i < n; i++) {
      const s = norms[i] > 0 ? norms[i] : 1;
      for (let j = 0; j < d; j++) out[i * d + j] = R[i][j] / s;
    }
    return { X: new Tensor(out, [n, d]), norms };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.encode.onehot', arity: 2,
  sig: '(Tensor codes[n], number k=max+1) -> Tensor OneHot[n,k]',
  doc: '独热编码：整数类别码 → n×k 的 0/1 矩阵',
  ref: 'Hastie et al., ESL 2009 §2.4（定性变量的哑变量编码）',
  inv: '每行恰有一个 1，其余为 0；列数 = 类别数',
  impl: (codes, k) => {
    codes = _asVec(codes, 'codes');
    const n = codes.shape[0];
    const c = _flat(codes);
    let K = k === undefined ? 0 : Math.floor(expectNumber(k, 'k'));
    if (k === undefined) {
      for (const v of c) K = Math.max(K, Math.floor(v) + 1);
    }
    if (!(K >= 1)) throw new ForgeError('E_VALUE', `k 必须 >= 1，当前 ${K}`);
    const out = new Float64Array(n * K);
    c.forEach((v, i) => {
      const idx = Math.floor(v);
      if (idx < 0 || idx >= K) throw new ForgeError('E_VALUE', `类别码 ${idx} 越界（k=${K}）`);
      out[i * K + idx] = 1;
    });
    return new Tensor(out, [n, K]);
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.encode.label', arity: 1,
  sig: '(Tensor y[n]) -> object {codes, classes}',
  doc: '标签编码：把任意标量标签映射到 0..K−1 的整数码',
  ref: 'Hastie et al., ESL 2009 §2.4；scikit-learn LabelEncoder',
  inv: 'classes[codes[i]] = y[i]（一一对应可逆）',
  impl: (y) => {
    y = _asVec(y, 'y');
    const ys = _flat(y);
    const classes = _classesOf(ys);
    const codes = ys.map((v) => classes.indexOf(v));
    return { codes: _T1(codes), classes };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.impute.mean', arity: 1,
  sig: '(Tensor X[n,d]) -> object {X, fill}',
  doc: '均值填补缺失值（NaN 用所在列的均值替换）',
  ref: 'Little & Rubin《Statistical Analysis with Missing Data》2nd ed., 2002 §4.2',
  inv: '输出不含 NaN；填补值等于该列非缺失元素的均值',
  impl: (X) => {
    X = _asMat(X, 'X');
    const R = _rows(X), d = X.shape[1], n = R.length;
    const fill = new Array(d).fill(0);
    for (let j = 0; j < d; j++) {
      let s = 0, c = 0;
      for (let i = 0; i < n; i++) if (Number.isFinite(R[i][j])) { s += R[i][j]; c++; }
      fill[j] = c > 0 ? s / c : 0;
    }
    const out = new Float64Array(n * d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) out[i * d + j] = Number.isFinite(R[i][j]) ? R[i][j] : fill[j];
    return { X: new Tensor(out, [n, d]), fill };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.impute.median', arity: 1,
  sig: '(Tensor X[n,d]) -> object {X, fill}',
  doc: '中位数填补缺失值（对偏态分布比均值更稳健）',
  ref: 'Little & Rubin 2002；Tukey《Exploratory Data Analysis》1977',
  inv: '输出不含 NaN；填补值等于该列非缺失元素的中位数',
  impl: (X) => {
    X = _asMat(X, 'X');
    const R = _rows(X), d = X.shape[1], n = R.length;
    const fill = new Array(d).fill(0);
    for (let j = 0; j < d; j++) {
      const col = [];
      for (let i = 0; i < n; i++) if (Number.isFinite(R[i][j])) col.push(R[i][j]);
      col.sort((p, q) => p - q);
      fill[j] = col.length ? col[Math.floor(col.length / 2)] : 0;
    }
    const out = new Float64Array(n * d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) out[i * d + j] = Number.isFinite(R[i][j]) ? R[i][j] : fill[j];
    return { X: new Tensor(out, [n, d]), fill };
  },
});

// =====================================================================
// 十、J 特征选择（3 条）：方差阈值 / F 检验 / 互信息
// =====================================================================
defineOp({
  id: nid(), domain: D, name: 'ml.select.variance', arity: 2,
  sig: '(Tensor X[n,d], number threshold=0) -> object {scores, mask, index}',
  doc: '方差阈值筛选：剔除方差低于阈值的（近）常数列',
  ref: 'Guyon & Elisseeff, An Introduction to Variable and Feature Selection, JMLR, 2003；scikit-learn VarianceThreshold',
  inv: '保留列的方差均 >= threshold',
  impl: (X, threshold) => {
    X = _asMat(X, 'X');
    const th = threshold === undefined ? 0 : expectNumber(threshold, 'threshold');
    const R = _rows(X), d = X.shape[1], n = R.length;
    const mean = _colMean(R, d);
    const scores = new Array(d).fill(0);
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += (R[i][j] - mean[j]) ** 2;
      scores[j] = s / Math.max(1, n);
    }
    const mask = scores.map((v) => v >= th);
    const index = [];
    mask.forEach((m, j) => { if (m) index.push(j); });
    return { scores: _T1(scores), mask, index };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.select.fTest', arity: 2,
  sig: '(Tensor X[n,d], Tensor y[n]) -> object {scores, order}',
  doc: '单变量 F 检验打分：分类用单因素 ANOVA F，回归用线性相关 F',
  ref: 'Fisher 1925《Statistical Methods for Research Workers》；Guyon & Elisseeff, JMLR 2003',
  inv: 'F 统计量非负；与标签无关的特征 F ≈ 0',
  impl: (X, y) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const R = _rows(X), ys = _flat(y);
    const classify = _isClassify(ys);
    const scores = new Array(d).fill(0);
    for (let j = 0; j < d; j++) {
      const col = R.map((r) => r[j]);
      if (classify) {
        const classes = _classesOf(ys);
        const K = classes.length;
        const gm = _mean(col);
        let ssb = 0, ssw = 0;
        for (const c of classes) {
          const g = col.filter((_, i) => ys[i] === c);
          const m = _mean(g);
          ssb += g.length * (m - gm) ** 2;
          for (const v of g) ssw += (v - m) ** 2;
        }
        const dfw = Math.max(1, n - K);
        scores[j] = (ssb / Math.max(1, K - 1)) / Math.max(ssw / dfw, _EPS);
      } else {
        const mx = _mean(col), my = _mean(ys);
        let sxy = 0, sxx = 0, syy = 0;
        for (let i = 0; i < n; i++) {
          const a = col[i] - mx, b = ys[i] - my;
          sxy += a * b; sxx += a * a; syy += b * b;
        }
        const r2 = sxx > 0 && syy > 0 ? (sxy * sxy) / (sxx * syy) : 0;
        scores[j] = r2 < 1 ? (r2 / (1 - r2)) * Math.max(1, n - 2) : 0;
      }
    }
    const order = scores.map((v, j) => [v, j]).sort((p, q) => q[0] - p[0]).map((v) => v[1]);
    return { scores: _T1(scores), order };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.select.mutualInfo', arity: 3,
  sig: '(Tensor X[n,d], Tensor y[n], number bins=10) -> object {scores, order}',
  doc: '互信息特征打分：等宽分箱后的离散互信息 I(X_j; Y)（自然对数）',
  ref: 'Cover & Thomas《Elements of Information Theory》2nd ed., 2006；Guyon & Elisseeff, JMLR 2003',
  inv: '互信息非负；独立特征的分箱互信息 ≈ 0',
  impl: (X, y, bins) => {
    X = _asMat(X, 'X'); y = _asVec(y, 'y');
    const n = X.shape[0], d = X.shape[1];
    if (y.shape[0] !== n) throw new ForgeError('E_SHAPE', `X 行数 ${n} 与 y 长度 ${y.shape[0]} 不符`);
    const B = bins === undefined ? 10 : Math.max(2, Math.floor(expectNumber(bins, 'bins')));
    const R = _rows(X), ys = _flat(y);
    const yd = _isClassify(ys) && _classesOf(ys).length <= B
      ? ys.map((v) => _classesOf(ys).indexOf(v))
      : _discretize(ys, B);
    const scores = new Array(d).fill(0);
    for (let j = 0; j < d; j++) {
      const xd = _discretize(R.map((r) => r[j]), B);
      const nx = Math.max(1, Math.max(...xd) + 1);
      const ny = Math.max(1, Math.max(...yd) + 1);
      const joint = new Array(nx * ny).fill(0);
      const px = new Array(nx).fill(0), py = new Array(ny).fill(0);
      for (let i = 0; i < n; i++) {
        joint[xd[i] * ny + yd[i]]++;
        px[xd[i]]++; py[yd[i]]++;
      }
      const hx = _entropy(px, n), hy = _entropy(py, n), hxy = _entropy(joint, n);
      scores[j] = Math.max(0, hx + hy - hxy);
    }
    const order = scores.map((v, j) => [v, j]).sort((p, q) => q[0] - p[0]).map((v) => v[1]);
    return { scores: _T1(scores), order };
  },
});

// =====================================================================
// 十一、K 指标（7 条）：回归误差 / 决定系数 / 轮廓 / ARI / CH
// =====================================================================
defineOp({
  id: nid(), domain: D, name: 'ml.metric.mse', arity: 2,
  sig: '(Tensor yTrue[n], Tensor yPred[n]) -> number',
  doc: '均方误差 MSE = mean((y−ŷ)²)',
  ref: 'Hastie et al., ESL 2009 §2.4；Gauss 最小二乘传统',
  inv: 'MSE ≥ 0；完美预测时为 0',
  impl: (yTrue, yPred) => {
    yTrue = _asVec(yTrue, 'yTrue'); yPred = _asVec(yPred, 'yPred');
    const a = _flat(yTrue), b = _flat(yPred);
    if (a.length !== b.length) throw new ForgeError('E_SHAPE', `长度不符: ${a.length} vs ${b.length}`);
    if (a.length === 0) throw new ForgeError('E_VALUE', '至少需要 1 个样本');
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
    return s / a.length;
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.metric.rmse', arity: 2,
  sig: '(Tensor yTrue[n], Tensor yPred[n]) -> number',
  doc: '均方根误差 RMSE = √MSE（与目标同量纲）',
  ref: 'Hastie et al., ESL 2009 §2.4',
  inv: 'RMSE = √MSE 且 ≥ 0',
  impl: (yTrue, yPred) => {
    yTrue = _asVec(yTrue, 'yTrue'); yPred = _asVec(yPred, 'yPred');
    const a = _flat(yTrue), b = _flat(yPred);
    if (a.length !== b.length) throw new ForgeError('E_SHAPE', `长度不符: ${a.length} vs ${b.length}`);
    if (a.length === 0) throw new ForgeError('E_VALUE', '至少需要 1 个样本');
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
    return Math.sqrt(s / a.length);
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.metric.mae', arity: 2,
  sig: '(Tensor yTrue[n], Tensor yPred[n]) -> number',
  doc: '平均绝对误差 MAE = mean|y−ŷ|',
  ref: 'Hastie et al., ESL 2009 §2.4；Huber 1964（L1 损失）',
  inv: 'MAE ≥ 0 且 ≤ RMSE（同一样本上）',
  impl: (yTrue, yPred) => {
    yTrue = _asVec(yTrue, 'yTrue'); yPred = _asVec(yPred, 'yPred');
    const a = _flat(yTrue), b = _flat(yPred);
    if (a.length !== b.length) throw new ForgeError('E_SHAPE', `长度不符: ${a.length} vs ${b.length}`);
    if (a.length === 0) throw new ForgeError('E_VALUE', '至少需要 1 个样本');
    let s = 0;
    for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
    return s / a.length;
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.metric.r2', arity: 2,
  sig: '(Tensor yTrue[n], Tensor yPred[n]) -> number',
  doc: '决定系数 R² = 1 − SS_res/SS_tot',
  ref: 'Wright, Correlation and Causation, J. Agricultural Research, 1921；Hastie et al., ESL 2009 §3.2',
  inv: '完美预测时 R² = 1；常数基线预测（均值）时 R² = 0',
  impl: (yTrue, yPred) => {
    yTrue = _asVec(yTrue, 'yTrue'); yPred = _asVec(yPred, 'yPred');
    const a = _flat(yTrue), b = _flat(yPred);
    if (a.length !== b.length) throw new ForgeError('E_SHAPE', `长度不符: ${a.length} vs ${b.length}`);
    if (a.length === 0) throw new ForgeError('E_VALUE', '至少需要 1 个样本');
    const m = _mean(a);
    let ss = 0, tt = 0;
    for (let i = 0; i < a.length; i++) { ss += (a[i] - b[i]) ** 2; tt += (a[i] - m) ** 2; }
    if (tt <= 0) return ss <= 0 ? 1 : 0;
    return 1 - ss / tt;
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.metric.silhouette', arity: 2,
  sig: '(Tensor X[n,d], Tensor labels[n]) -> object {score, samples}',
  doc: '轮廓系数：返回均值 score 与逐样本 samples',
  ref: 'Rousseeuw, Silhouettes: A Graphical Aid to the Interpretation of Cluster Analysis, 1987',
  inv: 'score ∈ [−1, 1]；每簇仅 1 个样本时该样本轮廓为 0',
  impl: (X, labels) => {
    X = _asMat(X, 'X'); labels = _asVec(labels, 'labels');
    const R = _rows(X), lab = _flat(labels);
    if (lab.length !== R.length) throw new ForgeError('E_SHAPE', `labels 长度 ${lab.length} 与样本数 ${R.length} 不符`);
    const s = _silhouetteSamples(R, lab);
    return { score: _mean(s), samples: _T1(s) };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.metric.ari', arity: 2,
  sig: '(Tensor a[n], Tensor b[n]) -> number',
  doc: '调整兰德指数 ARI（对随机划分做机会校正的聚类一致性）',
  ref: 'Rand, Objective Criteria for the Evaluation of Clustering Methods, 1971；Hubert & Arabie, Comparing Partitions, 1985',
  inv: '相同划分时 ARI = 1；期望值（随机独立划分）为 0；取值 ≤ 1',
  impl: (a, b) => {
    a = _asVec(a, 'a'); b = _asVec(b, 'b');
    return _ari(_flat(a), _flat(b));
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.metric.calinski', arity: 2,
  sig: '(Tensor X[n,d], Tensor labels[n]) -> number',
  doc: 'Calinski-Harabasz 指数：簇间/簇内方差比（越大越好）',
  ref: 'Caliński & Harabasz, A Dendrite Method for Cluster Analysis, Communications in Statistics, 1974',
  inv: 'k=1 时无定义（返回 0）；簇越分离、越紧凑则指数越大',
  impl: (X, labels) => {
    X = _asMat(X, 'X'); labels = _asVec(labels, 'labels');
    const R = _rows(X), lab = _flat(labels);
    if (lab.length !== R.length) throw new ForgeError('E_SHAPE', `labels 长度 ${lab.length} 与样本数 ${R.length} 不符`);
    const groups = new Map();
    for (let i = 0; i < R.length; i++) {
      if (!groups.has(lab[i])) groups.set(lab[i], []);
      groups.get(lab[i]).push(i);
    }
    const k = groups.size;
    if (k < 2 || R.length <= k) return 0;
    const d = X.shape[1];
    const gm = new Array(d).fill(0);
    for (const r of R) for (let j = 0; j < d; j++) gm[j] += r[j];
    for (let j = 0; j < d; j++) gm[j] /= R.length;
    let between = 0, within = 0;
    for (const mem of groups.values()) {
      const c = new Array(d).fill(0);
      for (const i of mem) for (let j = 0; j < d; j++) c[j] += R[i][j];
      for (let j = 0; j < d; j++) c[j] /= mem.length;
      for (let j = 0; j < d; j++) between += mem.length * (c[j] - gm[j]) ** 2;
      for (const i of mem) within += _dist2(R[i], c);
    }
    const denom = within / (R.length - k);
    if (!(denom > 0)) return 0;
    return (between / (k - 1)) / denom;
  },
});

// =====================================================================
// 十二、L 工具（4 条）：克隆 / 参数读写 / 自描述
// =====================================================================
defineOp({
  id: nid(), domain: D, name: 'ml.util.clone', arity: 1,
  sig: '(object state) -> object state',
  doc: '深拷贝估计器 state（要求 state 可 JSON 序列化）',
  ref: 'scikit-learn BaseEstimator.clone / get_params 设计；Gamma et al.《Design Patterns》原型模式 1994',
  inv: '克隆与原对象值相等但不共享引用',
  impl: (state) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    try {
      return JSON.parse(JSON.stringify(state));
    } catch {
      throw new ForgeError('E_VALUE', 'state 不可 JSON 序列化，无法克隆');
    }
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.util.getParams', arity: 1,
  sig: '(object state) -> object params',
  doc: '提取 state 中的标量超参数（number/string/boolean）',
  ref: 'scikit-learn BaseEstimator.get_params 约定（Buitinck et al., API design for ML software, 2013）',
  inv: '返回值全部为标量，且键集 ⊆ state 的键集',
  impl: (state) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    const out = {};
    for (const k of Object.keys(state)) {
      const v = state[k];
      if (typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean') out[k] = v;
    }
    return out;
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.util.setParams', arity: 2,
  sig: '(object state, object patch) -> object state',
  doc: '按补丁合并超参数，返回新的 state（不改原对象）',
  ref: 'scikit-learn BaseEstimator.set_params 约定；Buitinck et al. 2013',
  inv: '返回对象的 params = 原 params 覆盖 patch；原 state 不被修改',
  impl: (state, patch) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    if (!patch || typeof patch !== 'object') throw new ForgeError('E_TYPE', 'patch 必须为对象');
    return { ...state, ...patch };
  },
});

defineOp({
  id: nid(), domain: D, name: 'ml.util.describe', arity: 1,
  sig: '(object state) -> object summary',
  doc: '估计器自描述：kind、顶层键、标量参数与序列化体积',
  ref: 'scikit-learn 估计器内省 API；Buitinck et al. 2013',
  inv: 'keys 等于 state 的顶层键集合；params ⊆ keys',
  impl: (state) => {
    if (!state || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 必须为对象');
    const keys = Object.keys(state);
    const params = {};
    for (const k of keys) {
      const v = state[k];
      if (typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean') params[k] = v;
    }
    let bytes = 0;
    try { bytes = JSON.stringify(state).length; } catch { bytes = NaN; }
    return { kind: state.kind ?? 'unknown', keys, params, bytes };
  },
});


// Davies–Bouldin 指数：DB = (1/k)·Σᵢ max_{j≠i} (sᵢ+sⱼ)/dᵢⱼ，s 为簇内平均到质心距离。
// 与 silhouette / Calinski-Harabasz 互补，三者共同构成聚类内部评估的三条独立标尺。
defineOp({
  id: nid(), domain: D, name: 'ml.metric.davies', arity: 2,
  sig: '(Tensor X[n,d], Tensor labels[n]) -> number',
  doc: 'Davies-Bouldin 指数：簇内散度与簇间距之比的最大值的均值（越小越好）',
  ref: 'Davies & Bouldin, A Cluster Separation Measure, IEEE TPAMI 1979',
  inv: 'DB ≥ 0；k=1 时无定义（返回 0）；完全分离且紧凑的簇 DB → 0',
  impl: (X, labels) => {
    X = _asMat(X, 'X'); labels = _asVec(labels, 'labels');
    const R = _rows(X), lab = _flat(labels);
    if (lab.length !== R.length) throw new ForgeError('E_SHAPE', `labels 长度 ${lab.length} 与样本数 ${R.length} 不符`);
    const groups = new Map();
    for (let i = 0; i < R.length; i++) {
      if (!groups.has(lab[i])) groups.set(lab[i], []);
      groups.get(lab[i]).push(i);
    }
    const k = groups.size;
    if (k < 2) return 0;
    const d = X.shape[1];
    // 质心与簇内平均散度 sᵢ
    const cent = [], scatter = [];
    for (const mem of groups.values()) {
      const c = new Array(d).fill(0);
      for (const i of mem) for (let j = 0; j < d; j++) c[j] += R[i][j];
      for (let j = 0; j < d; j++) c[j] /= mem.length;
      cent.push(c);
      let sSum = 0;
      for (const i of mem) {
        let t = 0;
        for (let j = 0; j < d; j++) t += (R[i][j] - c[j]) ** 2;
        sSum += Math.sqrt(t);
      }
      scatter.push(sSum / mem.length);
    }
    // DB = (1/k)·Σᵢ max_{j≠i} (sᵢ+sⱼ)/‖cᵢ−cⱼ‖
    let total = 0;
    for (let i = 0; i < k; i++) {
      let worst = -Infinity;
      for (let j = 0; j < k; j++) {
        if (i === j) continue;
        let dist = 0;
        for (let t = 0; t < d; t++) dist += (cent[i][t] - cent[j][t]) ** 2;
        dist = Math.sqrt(dist);
        worst = Math.max(worst, dist > 0 ? (scatter[i] + scatter[j]) / dist : Infinity);
      }
      if (!Number.isFinite(worst)) return Infinity;   // 两簇质心重合：分离度无界
      total += worst;
    }
    return total / k;
  },
});

// —— 文件底部自检：域7 必须恰好 103 条，否则 import 即失败 ——
if (opsByDomain(D).length !== 103) {
  throw new Error(`域7 指令数错误: 期望 103，实际 ${opsByDomain(D).length}`);
}

