// ============================================================================
// 域9 训练与评估（TRAIN）：数据管线 / 训练循环 / 评估分析 / 可复现性 / 组合工具
// ----------------------------------------------------------------------------
// 设计约束：
//   - 零依赖，仅依赖 core（registry / errors / tensor / rng）；
//   - 随机数一律走 createRng（PCG-XSL-RR），禁止 Math.random；
//   - 训练循环面向通用模型接口：{ params, predict(params, X), lossGrad(params, X, y) }
//     （predict 返回预测 Tensor，lossGrad 返回 { loss: number, grad: 与 params 同构 }），
//     不硬编码任何具体模型，可被域7（ML）直接复用；
//   - 数据集统一为 { X: Tensor[n,d], y: Tensor[n] }（可附带 index 等字段，变换时保留）。
// ============================================================================
import { defineOp, opsByDomain, DOMAINS, KERNEL_VERSION, CATALOG_VERSION } from '../core/registry.mjs';
import { ForgeError } from '../core/errors.mjs';
import { Tensor } from '../core/tensor.mjs';
import { createRng, seedFrom } from '../core/rng.mjs';

let _s = 0;
const nid = () => `D9-${String(++_s).padStart(3, '0')}`;

// —— 内部错误快捷方式 ——
const err = (code, msg) => new ForgeError(code, msg);

// —— 内部纯函数工具 ——
const range = (n) => Array.from({ length: n }, (_, i) => i);
const isPlainObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v) && !(v instanceof Tensor);
const arr = (v) => (v instanceof Tensor ? v.toArray() : Array.from(v));

// 数据集校验：返回归一化视图 {X, y, n, d}
function dsOf(ds, who = 'dataset') {
  if (!ds || !(ds.X instanceof Tensor) || !(ds.y instanceof Tensor)) {
    throw err('E_TYPE', `${who}: 期望 {X: Tensor, y: Tensor}`);
  }
  if (ds.X.ndim !== 2) throw err('E_SHAPE', `${who}: X 必须为 2D [n,d]`);
  if (ds.y.ndim !== 1) throw err('E_SHAPE', `${who}: y 必须为 1D [n]`);
  if (ds.X.shape[0] !== ds.y.shape[0]) {
    throw err('E_SHAPE', `${who}: X 行数 ${ds.X.shape[0]} 与 y 长度 ${ds.y.shape[0]} 不符`);
  }
  return { X: ds.X, y: ds.y, n: ds.X.shape[0], d: ds.X.shape[1] };
}

// 按索引数组取行（拷贝，不共享内存）
function rowsOf(X, idx) {
  const d = X.shape[1];
  const out = new Float64Array(idx.length * d);
  for (let i = 0; i < idx.length; i++) {
    if (idx[i] < 0 || idx[i] >= X.shape[0]) throw err('E_VALUE', `行索引越界: ${idx[i]}`);
    out.set(X.data.subarray(idx[i] * d, idx[i] * d + d), i * d);
  }
  return new Tensor(out, [idx.length, d]);
}

function pickY(y, idx) {
  const out = new Float64Array(idx.length);
  for (let i = 0; i < idx.length; i++) out[i] = y.data[idx[i]];
  return new Tensor(out, [idx.length]);
}

// 取子集并保留原数据集的其它字段（如 index）；剔除 dsOf 派生的 n/d，避免过期值
function takeRows(ds, idx) {
  const out = { ...ds };
  delete out.n;
  delete out.d;
  out.X = rowsOf(ds.X, idx);
  out.y = pickY(ds.y, idx);
  out.index = idx.slice();
  return out;
}

// 沿轴 0 连续切片（支持 1D/2D）
function headRows(t, i, count) {
  if (t.ndim === 1) return new Tensor(t.data.slice(i, i + count), [count]);
  const d = t.shape[1];
  const out = new Float64Array(count * d);
  out.set(t.data.subarray(i * d, (i + count) * d), 0);
  return new Tensor(out, [count, d]);
}

// 按顺序数组切批（批内保持给定顺序，附原始行号 index）
function batchesFromIndex(ds, order, bs, dropLast = false) {
  const out = [];
  for (let i = 0; i < order.length; i += bs) {
    const end = Math.min(i + bs, order.length);
    if (dropLast && end - i < bs) break;
    const idx = order.slice(i, end);
    out.push({ X: rowsOf(ds.X, idx), y: pickY(ds.y, idx), index: idx });
  }
  return out;
}

// 均匀分配 n 个元素到 k 份（前 n%k 份多 1 个）
function distribute(n, k) {
  const base = Math.floor(n / k), rem = n % k;
  return range(k).map(i => base + (i < rem ? 1 : 0));
}

function sortedUnique(a) {
  const allNum = a.every(v => typeof v === 'number');
  return [...new Set(a)].sort(allNum ? (x, y) => x - y : (x, y) => String(x).localeCompare(String(y)));
}

function meanOf(a) { let s = 0; for (const v of a) s += v; return a.length ? s / a.length : NaN; }
// 总体标准差（ddof=0）
function stdOf(a) { const m = meanOf(a); let s = 0; for (const v of a) s += (v - m) ** 2; return a.length ? Math.sqrt(s / a.length) : NaN; }
// 样本标准差（ddof=1）
function sampleStd(a) {
  if (a.length < 2) return 0;
  const m = meanOf(a); let s = 0; for (const v of a) s += (v - m) ** 2;
  return Math.sqrt(s / (a.length - 1));
}

// 线性插值分位数（Hyndman & Fan 1996, type 7）
function quantileSorted(sorted, q) {
  if (sorted.length === 0) return NaN;
  const pos = (sorted.length - 1) * Math.min(Math.max(q, 0), 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

// 深度拷贝参数结构（Tensor / 数组 / 纯对象 / 标量）
function deepClone(v) {
  if (v instanceof Tensor) return v.clone();
  if (Array.isArray(v)) return v.map(deepClone);
  if (isPlainObj(v)) { const o = {}; for (const k of Object.keys(v)) o[k] = deepClone(v[k]); return o; }
  return v;
}

// 同构参数逐元素映射：fn(aLeaf, bLeaf) -> newLeaf
function mapParams(a, b, fn) {
  if (a instanceof Tensor && b instanceof Tensor) {
    if (a.size !== b.size) throw err('E_SHAPE', `参数结构大小不符: ${a.size} vs ${b.size}`);
    const out = new Float64Array(a.size);
    for (let i = 0; i < a.size; i++) out[i] = fn(a.data[i], b.data[i]);
    return new Tensor(out, a.shape.slice());
  }
  if (Array.isArray(a) && Array.isArray(b)) return a.map((v, i) => mapParams(v, b[i], fn));
  if (typeof a === 'number' && typeof b === 'number') return fn(a, b);
  if (isPlainObj(a) && isPlainObj(b)) {
    const o = {};
    for (const k of Object.keys(a)) o[k] = mapParams(a[k], b[k], fn);
    return o;
  }
  throw err('E_TYPE', '参数结构不可映射（类型/形状不匹配）');
}

const sgdUpdate = (params, grad, lr) => mapParams(params, grad, (p, g) => p - lr * g);

// 参数全局 L2 范数
function paramsNorm(grad) {
  let s = 0;
  const walk = (v) => {
    if (v instanceof Tensor) { for (let i = 0; i < v.size; i++) s += v.data[i] * v.data[i]; }
    else if (Array.isArray(v)) v.forEach(walk);
    else if (isPlainObj(v)) Object.values(v).forEach(walk);
    else if (typeof v === 'number') s += v * v;
  };
  walk(grad);
  return Math.sqrt(s);
}

function scaleParams(grad, f) {
  return mapParams(grad, grad, v => v * f);
}

// 梯度全局范数裁剪（Pascanu et al. 2013）
function clipGradImpl(grad, maxNorm) {
  const nrm = paramsNorm(grad);
  if (!(nrm > maxNorm)) return grad;
  return scaleParams(grad, maxNorm / nrm);
}

function zeroLike(grad) { return mapParams(grad, grad, () => 0); }

// —— 稳定序列化与哈希（可复现性基础设施） ——
// 规范 JSON：对象键排序、Tensor 按 (shape,data) 展开，保证同构同串。
function canonicalJson(v) {
  if (v instanceof Tensor) {
    return `T[${v.shape.join(',')}](${Array.from(v.data).map(x => (Number.isInteger(x) ? String(x) : x.toString())).join(',')})`;
  }
  if (Array.isArray(v)) return `[${v.map(canonicalJson).join(',')}]`;
  if (isPlainObj(v)) {
    return `{${Object.keys(v).sort().map(k => `${JSON.stringify(k)}:${canonicalJson(v[k])}`).join(',')}}`;
  }
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toString();
  if (v === undefined || typeof v === 'function' || typeof v === 'bigint') return 'null';
  return JSON.stringify(v);
}

// FNV-1a 32bit（Fowler/Noll/Vo 1991），输出 8 位十六进制
function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return (h >>> 0).toString(16).padStart(8, '0');
}

// 逐位哈希 Float64Array（按字节，不受十进制表示影响）
function hashF64(f64) {
  const bytes = new Uint8Array(f64.buffer, f64.byteOffset, f64.byteLength);
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) { h ^= bytes[i]; h = Math.imul(h, 0x01000193); }
  return (h >>> 0).toString(16).padStart(8, '0');
}

// 正态 CDF（Abramowitz & Stegun 7.1.26 近似，|误差| < 1.5e-7）
function erfApprox(x) {
  const s = x < 0 ? -1 : 1;
  const z = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * z);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
  return s * y;
}
const normalCdf = (z) => 0.5 * (1 + erfApprox(z / Math.SQRT2));

// Gamma 采样（Marsaglia & Tsang 2000），用于 Beta(α,α)（mixup 的 λ）
function gammaSample(rng, a) {
  if (!(a > 0)) throw err('E_VALUE', 'Gamma 形状参数必须为正');
  if (a < 1) {
    const u = Math.max(rng.next(), 1e-300);
    return gammaSample(rng, a + 1) * Math.pow(u, 1 / a);
  }
  const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do { x = rng.normal(); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = rng.next();
    const x2 = x * x;
    if (u < 1 - 0.0331 * x2 * x2) return d * v;
    if (Math.log(u) < 0.5 * x2 + d * (1 - v + Math.log(v))) return d * v;
  }
}

// —— 内置指标（供 opts.metric 传字符串使用） ——
function argmaxRows(t) {
  const [n, c] = t.shape;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    let best = 0;
    for (let j = 1; j < c; j++) if (t.get2(i, j) > t.get2(i, best)) best = j;
    out[i] = best;
  }
  return out;
}
const mAccuracy = (pred, y) => {
  const P = pred instanceof Tensor && pred.ndim === 2 ? argmaxRows(pred) : arr(pred);
  const Y = arr(y);
  let c = 0;
  for (let i = 0; i < Y.length; i++) if (Math.round(P[i]) === Y[i]) c++;
  return Y.length ? c / Y.length : NaN;
};
const mMse = (pred, y) => { const P = arr(pred), Y = arr(y); let s = 0; for (let i = 0; i < Y.length; i++) s += (P[i] - Y[i]) ** 2; return Y.length ? s / Y.length : NaN; };
const mMae = (pred, y) => { const P = arr(pred), Y = arr(y); let s = 0; for (let i = 0; i < Y.length; i++) s += Math.abs(P[i] - Y[i]); return Y.length ? s / Y.length : NaN; };

// 指标解析：函数 | 'accuracy' | 'mse' | 'mae' | null
function resolveMetric(m) {
  if (m == null) return null;
  if (typeof m === 'function') return m;
  if (m === 'accuracy' || m === 'acc') return mAccuracy;
  if (m === 'mse') return mMse;
  if (m === 'mae') return mMae;
  throw err('E_VALUE', `未知指标: ${m}（支持 'accuracy' | 'mse' | 'mae' 或函数）`);
}

// 默认指标：整数标签 → 准确率；连续标签 → MSE
function defaultMetric(pred, y) {
  const Y = arr(y);
  if (pred instanceof Tensor && pred.ndim === 2) return mAccuracy(pred, y);
  const P = arr(pred);
  if (Y.every(Number.isInteger)) {
    let c = 0;
    for (let i = 0; i < Y.length; i++) if (Math.round(P[i]) === Y[i]) c++;
    return Y.length ? c / Y.length : NaN;
  }
  return mMse(P, Y);
}

// fitPredict 返回值 → 预测（Tensor | 数组 | {pred}）
function predOf(res) {
  if (res instanceof Tensor) return res;
  if (Array.isArray(res)) return Tensor.fromFlat(res, [res.length]);
  if (res && res.pred !== undefined) return predOf(res.pred);
  throw err('E_TYPE', 'fitPredict 返回值无法识别为预测（应为 Tensor / 数组 / {pred}）');
}

// 折分数：允许 fitPredict 直接返回数值分数
function scoreOf(res, y, metric) {
  if (typeof res === 'number') return res;
  const pred = predOf(res);
  return metric ? metric(pred, y) : defaultMetric(pred, y);
}

// —— 表格渲染（markdown / csv 共用） ——
function orderedKeys(rows) {
  const ks = [], seen = new Set();
  for (const r of rows) {
    if (r && typeof r === 'object' && !(r instanceof Tensor)) {
      for (const k of Object.keys(r)) if (!seen.has(k)) { seen.add(k); ks.push(k); }
    }
  }
  return ks;
}
function fmtCell(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'number') {
    if (Number.isInteger(v)) return String(v);
    if (!Number.isFinite(v)) return String(v);
    const a = Math.abs(v);
    return a < 1e-4 || a >= 1e7 ? v.toExponential(4) : v.toFixed(6);
  }
  if (v instanceof Tensor) return `[${Array.from(v.data).map(fmtCell).join(',')}]`;
  if (Array.isArray(v)) return `[${v.map(fmtCell).join(',')}]`;
  return String(v);
}
function tableMarkdown(rows, cols) {
  const list = Array.isArray(rows) ? rows : [rows];
  const cs = cols && cols.length ? cols : orderedKeys(list);
  const head = `| ${cs.join(' | ')} |`;
  const sep = `| ${cs.map(() => '---').join(' | ')} |`;
  const body = list.map(r => `| ${cs.map(c => fmtCell(r?.[c])).join(' | ')} |`);
  return [head, sep, ...body].join('\n');
}

// ============================================================================
// 一、数据管线（D9-001 ~ D9-022）
// ============================================================================

// D9-001 顺序分批：最后不满一批默认保留，dropLast 可丢弃
defineOp({
  id: nid(), domain: 9, name: 'train.data.batchIterator', arity: 3,
  sig: '(dataset, batchSize:number, opts?:{dropLast}) -> batch[]',
  doc: '按 batchSize 顺序分批；批为 {X, y, index}，index 为原始行号；最后不满一批默认保留，opts.dropLast=true 时丢弃。',
  ref: '小批随机梯度下降的批处理惯例：Bottou (2010) Large-Scale Machine Learning with Stochastic Gradient Descent, COMPSTAT',
  inv: '各批 index 并集 = [0,n)，批间不相交；默认末批大小 = n mod batchSize（非零时）',
  impl: (ds, batchSize, opts = {}) => {
    const d = dsOf(ds);
    if (!Number.isInteger(batchSize) || batchSize < 1) throw err('E_VALUE', `batchSize 必须为正整数，收到 ${batchSize}`);
    return batchesFromIndex(d, range(d.n), batchSize, !!(opts && opts.dropLast));
  },
});

// D9-002 种子化打乱后分批：同种子同顺序（可复现）
defineOp({
  id: nid(), domain: 9, name: 'train.data.shuffledBatches', arity: 3,
  sig: '(dataset, batchSize:number, seed:number) -> batch[]',
  doc: '用 createRng(seed) 对行做 Fisher-Yates 打乱后分批；同种子产生完全相同的批序列。',
  ref: 'Fisher-Yates 打乱 + PCG 流：O\'Neill (2014) PCG: A Family of Simple Fast Space-Efficient Statistically Good RNGs',
  inv: '同种子两次调用批序列逐位相同；批 index 并集 = [0,n) 且不相交',
  impl: (ds, batchSize, seed) => {
    const d = dsOf(ds);
    if (!Number.isInteger(batchSize) || batchSize < 1) throw err('E_VALUE', `batchSize 必须为正整数，收到 ${batchSize}`);
    const perm = createRng(seed).shuffle(range(d.n));
    return batchesFromIndex(d, perm, batchSize, false);
  },
});

// D9-003 分层分批：各类别按比例进入各批（近似保持总体类别比例）
defineOp({
  id: nid(), domain: 9, name: 'train.data.stratifiedBatches', arity: 3,
  sig: '(dataset, batchSize:number, seed:number) -> batch[]',
  doc: '按类别分组后各自打乱，再尽可能均匀地摊到 ceil(n/batchSize) 个批中，使每批类别比例≈总体比例。',
  ref: '分层抽样：Kohavi (1995) A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection, IJCAI',
  inv: '批 index 并集 = [0,n) 且不相交；每批类别比例与总体偏差有界（各类摊差 ≤1 个元素）',
  impl: (ds, batchSize, seed) => {
    const d = dsOf(ds);
    if (!Number.isInteger(batchSize) || batchSize < 1) throw err('E_VALUE', `batchSize 必须为正整数，收到 ${batchSize}`);
    const classes = sortedUnique(arr(d.y));
    const rng = createRng(seed);
    const k = Math.max(1, Math.ceil(d.n / batchSize));
    const buckets = Array.from({ length: k }, () => []);
    for (const c of classes) {
      const idxs = rng.shuffle(range(d.n).filter(i => d.y.data[i] === c));
      const sizes = distribute(idxs.length, k);
      let p = 0;
      for (let b = 0; b < k; b++) { buckets[b].push(...idxs.slice(p, p + sizes[b])); p += sizes[b]; }
    }
    return buckets
      .filter(b => b.length > 0)
      .map(b => { const s = b.slice().sort((x, y2) => x - y2); return { X: rowsOf(d.X, s), y: pickY(d.y, s), index: s }; });
  },
});

// D9-004 序列窗口迭代：X=series[i:i+w]，y=series[i+w]（下一步预测）
defineOp({
  id: nid(), domain: 9, name: 'train.data.windowIterator', arity: 3,
  sig: '(series:Tensor[T]|Tensor[T,d], window:number, stride?:number) -> {X,y,index}[]',
  doc: '把一维/二维序列切成滑动窗口样本：X 为长度 window 的片段，y 为窗口之后的 1 步；stride 控制步长。',
  ref: '时间序列滞后特征与窗口：Box & Jenkins (1970) Time Series Analysis: Forecasting and Control',
  inv: '窗口数 = floor((T-window)/stride)；所有窗口起点互不相同且 ∈ [0, T-window)',
  impl: (series, window, stride = 1) => {
    if (!(series instanceof Tensor)) throw err('E_TYPE', 'series 必须为 Tensor');
    const w = window, st = stride;
    if (!Number.isInteger(w) || w < 1) throw err('E_VALUE', 'window 必须为正整数');
    if (!Number.isInteger(st) || st < 1) throw err('E_VALUE', 'stride 必须为正整数');
    const T = series.shape[0];
    const out = [];
    for (let i = 0; i + w < T; i += st) {
      out.push({ X: headRows(series, i, w), y: headRows(series, i + w, 1), index: i });
    }
    return out;
  },
});

// D9-005 数据集构造：校验并规范化 {X, y}
defineOp({
  id: nid(), domain: 9, name: 'train.data.dataset', arity: 2,
  sig: '(X:Tensor[n,d]|nested, y:Tensor[n]|array) -> dataset',
  doc: '构造标准数据集 {X, y}：X 为 2D，y 为 1D 且行数一致；接受嵌套数组入参。',
  ref: '数据矩阵约定：Pedregosa et al. (2011) Scikit-learn: Machine Learning in Python, JMLR 12',
  inv: 'ds.X.shape[0] === ds.y.shape[0]；X.ndim === 2，y.ndim === 1',
  impl: (X, y) => {
    const Xt = X instanceof Tensor ? X : Tensor.fromNested(X);
    const yt = y instanceof Tensor ? y : Tensor.fromFlat(y, [y.length]);
    dsOf({ X: Xt, y: yt }, 'train.data.dataset');
    return { X: Xt, y: yt };
  },
});

// D9-006 训练/测试划分（保持可复现）
defineOp({
  id: nid(), domain: 9, name: 'train.data.trainTestSplit', arity: 3,
  sig: '(dataset, testFrac:number, seed:number) -> {train, test}',
  doc: '种子化打乱后按比例切分；测试集大小 = round(n*testFrac)，训练集非空。',
  ref: 'Holdout 验证：Devroye, Györfi & Lugosi (1996) A Probabilistic Theory of Pattern Recognition',
  inv: 'train 与 test 的 index 不相交，并集 = [0,n)；同种子划分可复现',
  impl: (ds, testFrac, seed) => {
    const d = dsOf(ds);
    if (!(typeof testFrac === 'number' && testFrac >= 0 && testFrac < 1)) throw err('E_VALUE', `testFrac 必须 ∈ [0,1)，收到 ${testFrac}`);
    const testN = Math.min(Math.round(d.n * testFrac), d.n - 1);
    const perm = createRng(seed).shuffle(range(d.n));
    const testIdx = perm.slice(0, testN).sort((a, b) => a - b);
    const trainIdx = perm.slice(testN).sort((a, b) => a - b);
    return { train: takeRows(d, trainIdx), test: takeRows(d, testIdx) };
  },
});

// —— k 折划分内核（供 op 与交叉验证复用） ——
function kFoldSplitImpl(ds, k, seed) {
  const d = dsOf(ds, 'kFoldSplit');
  if (!Number.isInteger(k) || k < 2 || k > d.n) throw err('E_VALUE', `k 必须为 [2,n] 内的整数，收到 k=${k}, n=${d.n}`);
  const perm = createRng(seed).shuffle(range(d.n));
  const sizes = distribute(d.n, k);
  const folds = [];
  let p = 0;
  for (let f = 0; f < k; f++) {
    const valIdx = perm.slice(p, p + sizes[f]).sort((a, b) => a - b);
    p += sizes[f];
    const valSet = new Set(valIdx);
    const trainIdx = range(d.n).filter(i => !valSet.has(i));
    folds.push({
      fold: f, valIdx, trainIdx,
      val: { X: rowsOf(d.X, valIdx), y: pickY(d.y, valIdx), index: valIdx },
      train: { X: rowsOf(d.X, trainIdx), y: pickY(d.y, trainIdx), index: trainIdx },
    });
  }
  return folds;
}

// D9-007 k 折划分：各折验证索引不相交、并集为全集
defineOp({
  id: nid(), domain: 9, name: 'train.data.kFoldSplit', arity: 3,
  sig: '(dataset, k:number, seed:number) -> fold[]',
  doc: '打乱后均分 k 折，每折 {fold, valIdx, trainIdx, train, val}；折大小差 ≤1。',
  ref: 'K 折交叉验证：Stone (1974) Cross-Validatory Choice and Assessment of Statistical Predictions, JRSS-B',
  inv: '各折 valIdx 两两不相交；∪valIdx = [0,n)；折大小 ∈ {floor(n/k), ceil(n/k)}',
  impl: (ds, k, seed) => kFoldSplitImpl(ds, k, seed),
});

// D9-008 分层 k 折：各折类别比例≈总体
defineOp({
  id: nid(), domain: 9, name: 'train.data.stratifiedKFoldSplit', arity: 3,
  sig: '(dataset, k:number, seed:number) -> fold[]',
  doc: '在每个类别内部打乱后按位置模 k 轮流分配到各折，保持各折类别比例≈总体比例。',
  ref: '分层交叉验证：Kohavi (1995) IJCAI；Sechidis, Tsoumakas & Vlahavas (2012) On the Stratification of Multi-Label Data, ECML',
  inv: '各折 valIdx 不相交且并集 = [0,n)；每折中各类别计数与总体比例偏差 ≤1 个元素',
  impl: (ds, k, seed) => {
    const d = dsOf(ds);
    if (!Number.isInteger(k) || k < 2 || k > d.n) throw err('E_VALUE', `k 必须为 [2,n] 内的整数，收到 k=${k}, n=${d.n}`);
    const classes = sortedUnique(arr(d.y));
    const rng = createRng(seed);
    const foldIdx = Array.from({ length: k }, () => []);
    for (const c of classes) {
      const idxs = rng.shuffle(range(d.n).filter(i => d.y.data[i] === c));
      idxs.forEach((v, pos) => foldIdx[pos % k].push(v));
    }
    return foldIdx.map((raw, f) => {
      const valIdx = raw.slice().sort((a, b) => a - b);
      const valSet = new Set(valIdx);
      const trainIdx = range(d.n).filter(i => !valSet.has(i));
      return {
        fold: f, valIdx, trainIdx,
        val: { X: rowsOf(d.X, valIdx), y: pickY(d.y, valIdx), index: valIdx },
        train: { X: rowsOf(d.X, trainIdx), y: pickY(d.y, trainIdx), index: trainIdx },
      };
    });
  },
});

// D9-009 标准化器拟合：逐列均值/标准差（std 下限 1e-12 防除零）
defineOp({
  id: nid(), domain: 9, name: 'train.data.standardizerFit', arity: 1,
  sig: '(dataset) -> {kind, mean:Tensor[d], std:Tensor[d], n}',
  doc: '拟合逐列均值与总体标准差；常数列 std 置 1e-12 下限，避免 apply 除零。',
  ref: 'Z-score 标准化与 3σ 离群准则：Iglewicz & Hoaglin (1993) How to Detect and Handle Outliers, ASQ Press',
  inv: 'apply 后每列均值 ≈ 0、标准差 ≈ 1（常数列除外，结果为 0）',
  impl: (ds) => {
    const { X, n, d } = dsOf(ds);
    if (n === 0) throw err('E_VALUE', '空数据集无法拟合标准化器');
    const mean = new Float64Array(d), std = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      let m = 0;
      for (let i = 0; i < n; i++) m += X.get2(i, j);
      m /= n;
      let s = 0;
      for (let i = 0; i < n; i++) s += (X.get2(i, j) - m) ** 2;
      mean[j] = m;
      std[j] = Math.max(Math.sqrt(s / n), 1e-12);
    }
    return { kind: 'standardizer', mean: new Tensor(mean, [d]), std: new Tensor(std, [d]), n };
  },
});

// D9-010 标准化应用：(X-mean)/std
defineOp({
  id: nid(), domain: 9, name: 'train.data.standardizerApply', arity: 2,
  sig: '(dataset, scaler) -> dataset',
  doc: '用已拟合的 {mean, std} 逐列标准化；保留数据集其它字段（如 index）。',
  ref: '训练集统计量应用于测试集（避免数据泄漏）：Kohavi (1995) IJCAI',
  inv: 'out.X[i,j] = (X[i,j]-mean[j])/std[j]；y 与其它字段不变',
  impl: (ds, scaler) => {
    const { X, d } = dsOf(ds);
    if (!scaler || !(scaler.mean instanceof Tensor) || !(scaler.std instanceof Tensor)) throw err('E_TYPE', 'scaler 需要 {mean: Tensor, std: Tensor}');
    if (scaler.mean.shape[0] !== d || scaler.std.shape[0] !== d) throw err('E_SHAPE', `scaler 维度 ${scaler.mean.shape[0]} 与数据列数 ${d} 不符`);
    const out = new Float64Array(X.size);
    for (let i = 0; i < X.shape[0]; i++) {
      for (let j = 0; j < d; j++) out[i * d + j] = (X.get2(i, j) - scaler.mean.data[j]) / scaler.std.data[j];
    }
    return { ...ds, X: new Tensor(out, X.shape.slice()) };
  },
});

// D9-011 最小-最大缩放器拟合
defineOp({
  id: nid(), domain: 9, name: 'train.data.minmaxFit', arity: 1,
  sig: '(dataset) -> {kind, min:Tensor[d], max:Tensor[d]}',
  doc: '拟合逐列最小/最大值；极差下限 1e-12（常数列缩放为 0）。',
  ref: 'Min-Max 归一化：Pedregosa et al. (2011) Scikit-learn, JMLR 12',
  inv: 'apply 后每列值域 ∈ [0,1]',
  impl: (ds) => {
    const { X, n, d } = dsOf(ds);
    if (n === 0) throw err('E_VALUE', '空数据集无法拟合 minmax');
    const mn = new Float64Array(d), mx = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < n; i++) { const v = X.get2(i, j); if (v < lo) lo = v; if (v > hi) hi = v; }
      mn[j] = lo; mx[j] = hi;
    }
    return { kind: 'minmax', min: new Tensor(mn, [d]), max: new Tensor(mx, [d]) };
  },
});

// D9-012 最小-最大缩放应用
defineOp({
  id: nid(), domain: 9, name: 'train.data.minmaxApply', arity: 2,
  sig: '(dataset, scaler) -> dataset',
  doc: '用已拟合的 {min, max} 逐列缩放到 [0,1]；极差为 0 的列输出 0。',
  ref: 'Pedregosa et al. (2011) Scikit-learn: Machine Learning in Python, JMLR 12',
  inv: 'out.X[i,j] = (X[i,j]-min[j])/(max[j]-min[j])；常数列输出 0',
  impl: (ds, scaler) => {
    const { X, d } = dsOf(ds);
    if (!scaler || !(scaler.min instanceof Tensor) || !(scaler.max instanceof Tensor)) throw err('E_TYPE', 'scaler 需要 {min: Tensor, max: Tensor}');
    if (scaler.min.shape[0] !== d || scaler.max.shape[0] !== d) throw err('E_SHAPE', `scaler 维度与数据列数 ${d} 不符`);
    const out = new Float64Array(X.size);
    for (let i = 0; i < X.shape[0]; i++) {
      for (let j = 0; j < d; j++) {
        const rg = scaler.max.data[j] - scaler.min.data[j];
        out[i * d + j] = rg > 1e-12 ? (X.get2(i, j) - scaler.min.data[j]) / rg : 0;
      }
    }
    return { ...ds, X: new Tensor(out, X.shape.slice()) };
  },
});

// D9-013 独热编码
defineOp({
  id: nid(), domain: 9, name: 'train.data.onehotEncode', arity: 2,
  sig: '(y:Tensor[n]|array, numClasses:number) -> Tensor[n,C]',
  doc: '把整数标签 [0,C) 编成 one-hot 矩阵；numClasses 缺省由最大值推断。',
  ref: '哑变量/独热编码约定：Pedregosa et al. (2011) Scikit-learn, JMLR 12',
  inv: '每行恰一个 1，位置 = 标签值；行和 = 1',
  impl: (y, numClasses) => {
    const Y = arr(y);
    const C = numClasses == null ? Math.max(...Y) + 1 : numClasses;
    if (!Number.isInteger(C) || C < 1) throw err('E_VALUE', `numClasses 必须为正整数，收到 ${C}`);
    const out = new Float64Array(Y.length * C);
    for (let i = 0; i < Y.length; i++) {
      const k = Y[i];
      if (!Number.isInteger(k) || k < 0 || k >= C) throw err('E_VALUE', `标签 ${k} 越界（numClasses=${C}）`);
      out[i * C + k] = 1;
    }
    return new Tensor(out, [Y.length, C]);
  },
});

// D9-014 标签编码：任意离散值 → 0..C-1 索引
defineOp({
  id: nid(), domain: 9, name: 'train.data.labelEncode', arity: 1,
  sig: '(labels:Tensor[n]|array) -> {classes, y:Tensor[n]}',
  doc: '把任意离散标签（字符串/数字）映射为排序后的类别表与整数索引。',
  ref: '标签编码约定：Pedregosa et al. (2011) Scikit-learn, JMLR 12',
  inv: 'classes 为去重排序表；y[i] = classes 中 labels[i] 的位置；labelDecode(y, classes) 可还原',
  impl: (labels) => {
    const L = arr(labels);
    const classes = sortedUnique(L);
    const index = new Map(classes.map((c, i) => [c, i]));
    const out = new Float64Array(L.length);
    for (let i = 0; i < L.length; i++) out[i] = index.get(L[i]);
    return { classes, y: new Tensor(out, [L.length]) };
  },
});

// D9-015 标签解码：索引 → 原始标签
defineOp({
  id: nid(), domain: 9, name: 'train.data.labelDecode', arity: 2,
  sig: '(y:Tensor[n]|array, classes:array) -> array',
  doc: 'labelEncode 的逆操作：整数索引还原为原始标签值。',
  ref: 'Pedregosa et al. (2011) Scikit-learn: Machine Learning in Python, JMLR 12',
  inv: '对 labelEncode 的输出，labelDecode(y, classes) 逐项等于原始 labels',
  impl: (y, classes) => {
    const Y = arr(y);
    if (!Array.isArray(classes) || classes.length === 0) throw err('E_TYPE', 'classes 必须为非空数组');
    const out = [];
    for (const v of Y) {
      if (!Number.isInteger(v) || v < 0 || v >= classes.length) throw err('E_VALUE', `索引 ${v} 越界（|classes|=${classes.length}）`);
      out.push(classes[v]);
    }
    return out;
  },
});

// D9-016 管线组合：顺序应用若干 ds -> ds 变换
defineOp({
  id: nid(), domain: 9, name: 'train.data.pipelineCompose', arity: 1,
  sig: '(steps:fn[]) -> (dataset) -> dataset',
  doc: '把若干数据集变换函数按顺序组合成一个函数；每步均为 ds -> ds 的纯函数。',
  ref: '函数式数据流管线：Hughes (1984) Why Functional Programming Matters；scikit-learn Pipeline（Pedregosa et al. 2011）',
  inv: 'compose(f,g)(ds) === g(f(ds))；空 steps 返回恒等函数',
  impl: (steps) => {
    if (!Array.isArray(steps) || steps.some(f => typeof f !== 'function')) throw err('E_TYPE', 'steps 必须为函数数组');
    return (ds) => steps.reduce((acc, f) => f(acc), ds);
  },
});

// D9-017 管线拟合-变换：返回变换后数据与可复用于新数据的 apply
defineOp({
  id: nid(), domain: 9, name: 'train.data.pipelineFitTransform', arity: 2,
  sig: '(dataset, fitters:(ds -> (ds -> ds))[]) -> {data, apply, fitted}',
  doc: '顺序拟合各步（每步在训练集上 fit 得到 apply 函数），返回变换后的训练数据与可复用管线。',
  ref: 'fit/transform 分离与数据泄漏防范：Kohavi (1995) IJCAI；Pedregosa et al. (2011) JMLR',
  inv: 'apply(newDs) 按相同拟合参数变换新数据；fitted.length === fitters.length',
  impl: (ds, fitters) => {
    if (!Array.isArray(fitters) || fitters.some(f => typeof f !== 'function')) throw err('E_TYPE', 'fitters 必须为函数数组');
    let cur = ds;
    const fitted = [];
    for (const f of fitters) {
      const t = f(cur);
      if (typeof t !== 'function') throw err('E_TYPE', 'fitter 必须返回 apply 函数 (ds -> ds)');
      fitted.push(t);
      cur = t(cur);
    }
    return {
      data: cur,
      fitted,
      apply: (ds2) => fitted.reduce((acc, t) => t(acc), ds2),
    };
  },
});

// D9-018 离群值截断：逐列裁剪到 mean ± k·std
defineOp({
  id: nid(), domain: 9, name: 'train.data.clipOutliers', arity: 2,
  sig: '(dataset, k?:number) -> dataset',
  doc: '按逐列均值 ± k 倍标准差截断离群值（默认 k=3，3σ 准则）。',
  ref: '3σ / z-score 离群处理：Iglewicz & Hoaglin (1993) How to Detect and Handle Outliers, ASQ Press',
  inv: '输出每列值域 ⊆ [mean-k·std, mean+k·std]；y 不变',
  impl: (ds, k = 3) => {
    const { X, n, d } = dsOf(ds);
    if (!(typeof k === 'number' && k > 0)) throw err('E_VALUE', `k 必须为正数，收到 ${k}`);
    const out = new Float64Array(X.size);
    for (let j = 0; j < d; j++) {
      let m = 0;
      for (let i = 0; i < n; i++) m += X.get2(i, j);
      m /= n;
      let s = 0;
      for (let i = 0; i < n; i++) s += (X.get2(i, j) - m) ** 2;
      const sd = Math.sqrt(s / n), lo = m - k * sd, hi = m + k * sd;
      for (let i = 0; i < n; i++) out[i * d + j] = Math.min(Math.max(X.get2(i, j), lo), hi);
    }
    return { ...ds, X: new Tensor(out, X.shape.slice()) };
  },
});

// D9-019 缩尾处理：逐列裁剪到分位数
defineOp({
  id: nid(), domain: 9, name: 'train.data.winsorize', arity: 3,
  sig: '(dataset, lo?:number, hi?:number) -> dataset',
  doc: '按逐列分位数截断两端（默认 1%/99% 分位缩尾）。',
  ref: '缩尾（Winsorizing）：Tukey (1962) The Future of Data Analysis, Ann. Math. Statist.；Dixon (1960) Ann. Math. Statist.',
  inv: '0 ≤ lo < hi ≤ 1；输出每列值域 ⊆ [Q(lo), Q(hi)]',
  impl: (ds, lo = 0.01, hi = 0.99) => {
    const { X, n, d } = dsOf(ds);
    if (!(lo >= 0 && hi <= 1 && lo < hi)) throw err('E_VALUE', `需要 0 ≤ lo < hi ≤ 1，收到 (${lo}, ${hi})`);
    const out = new Float64Array(X.size);
    for (let j = 0; j < d; j++) {
      const col = Array.from(X.col(j)).sort((a, b) => a - b);
      const qlo = quantileSorted(col, lo), qhi = quantileSorted(col, hi);
      for (let i = 0; i < n; i++) out[i * d + j] = Math.min(Math.max(X.get2(i, j), qlo), qhi);
    }
    return { ...ds, X: new Tensor(out, X.shape.slice()) };
  },
});

// D9-020 对数变换：log(x + offset)
defineOp({
  id: nid(), domain: 9, name: 'train.data.logTransform', arity: 2,
  sig: '(dataset, offset?:number) -> dataset',
  doc: '逐元素 log(x+offset)（默认 offset=1，即 log1p）；要求 min(x)+offset > 0。',
  ref: '方差稳定化变换：Box & Cox (1964) An Analysis of Transformations, JRSS-B；Tukey (1957) Ann. Math. Statist.',
  inv: '输出元素 = log(X[i,j]+offset)；严格单调保持原大小序',
  impl: (ds, offset = 1) => {
    const { X } = dsOf(ds);
    if (!(typeof offset === 'number' && offset > 0)) throw err('E_VALUE', `offset 必须为正数，收到 ${offset}`);
    let mn = Infinity;
    for (let i = 0; i < X.size; i++) if (X.data[i] < mn) mn = X.data[i];
    if (!(mn + offset > 0)) throw err('E_VALUE', `log 参数非正：min=${mn}, offset=${offset}`);
    return { ...ds, X: X.map(v => Math.log(v + offset)) };
  },
});

// D9-021 多项式特征展开：1..degree 阶全部单项式（含交叉项）
defineOp({
  id: nid(), domain: 9, name: 'train.data.polyExpand', arity: 2,
  sig: '(dataset, degree?:number) -> dataset',
  doc: '展开为总次数 1..degree 的全部单项式（含交叉项），阶数升序排列；degree=1 为恒等。',
  ref: '多项式/MARS 基展开：Friedman (1991) Multivariate Adaptive Regression Splines, Ann. Statist.',
  inv: '输出列数 = Σ_{m=1..degree} C(m+d-1, d-1)；首 d 列为原始列（degree=1 时恰为原始）',
  impl: (ds, degree = 2) => {
    const { X, d } = dsOf(ds);
    if (!Number.isInteger(degree) || degree < 1) throw err('E_VALUE', `degree 必须为正整数，收到 ${degree}`);
    // 枚举指数向量：按总次数从 1 到 degree；高维优先分配幂次（sklearn PolynomialFeatures 的首列优先序）
    const exps = [];
    const cur = new Array(d).fill(0);
    const rec = (pos, remaining) => {
      if (pos === d - 1) { cur[pos] = remaining; exps.push(cur.slice()); cur[pos] = 0; return; }
      for (let e = remaining; e >= 0; e--) { cur[pos] = e; rec(pos + 1, remaining - e); }
      cur[pos] = 0;
    };
    for (let deg = 1; deg <= degree; deg++) rec(0, deg);
    const n = X.shape[0];
    const out = new Float64Array(n * exps.length);
    for (let i = 0; i < n; i++) {
      for (let e = 0; e < exps.length; e++) {
        let v = 1;
        for (let j = 0; j < d; j++) { for (let t = 0; t < exps[e][j]; t++) v *= X.get2(i, j); }
        out[i * exps.length + e] = v;
      }
    }
    return { ...ds, X: new Tensor(out, [n, exps.length]) };
  },
});

// D9-022 特征列缩放：逐列乘以缩放因子
defineOp({
  id: nid(), domain: 9, name: 'train.data.featureScale', arity: 2,
  sig: '(dataset, scales:number|number[]) -> dataset',
  doc: '逐列乘以缩放因子（标量作用于所有列，数组按列对应）；用于量纲对齐。',
  ref: '特征缩放与量纲一致性：Pedregosa et al. (2011) Scikit-learn, JMLR 12',
  inv: 'out.X[i,j] = X[i,j] * (数组 ? scales[j] : scales)',
  impl: (ds, scales) => {
    const { X, d } = dsOf(ds);
    const sc = Array.isArray(scales) ? scales : range(d).map(() => scales);
    if (sc.length !== d || sc.some(v => typeof v !== 'number' || !Number.isFinite(v))) throw err('E_VALUE', `scales 需要 ${d} 个有限数`);
    const out = new Float64Array(X.size);
    for (let i = 0; i < X.shape[0]; i++) for (let j = 0; j < d; j++) out[i * d + j] = X.get2(i, j) * sc[j];
    return { ...ds, X: new Tensor(out, X.shape.slice()) };
  },
});

// ============================================================================
// 二、训练循环（D9-023 ~ D9-048）
// 通用模型接口：{ params, predict(params, X) -> Tensor, lossGrad(params, X, y) -> {loss, grad} }
// ============================================================================

// 单批次梯度步（供 loop.epoch / loop.run 复用）
function batchStepImpl(model, params, batch, lr, update, clipNorm) {
  const { loss, grad } = model.lossGrad(params, batch.X, batch.y);
  if (typeof loss !== 'number' || !Number.isFinite(loss)) throw err('E_NUMERICAL', `lossGrad 返回的 loss 非有限数: ${loss}`);
  const g = clipNorm != null ? clipGradImpl(grad, clipNorm) : grad;
  const newParams = update(params, g, lr);
  return { loss, grad: g, gradNorm: paramsNorm(g), params: newParams };
}

// 学习率调度内核：函数 | {kind:'step'|'exp'|'cosine'|'linear', ...}
function lrHookImpl(epoch, baseLr, schedule) {
  if (typeof schedule === 'function') return baseLr * schedule(epoch);
  if (schedule && typeof schedule === 'object') {
    const e = Math.max(0, epoch - 1); // epoch 从 1 计数
    switch (schedule.kind) {
      case 'step': return baseLr * Math.pow(schedule.drop ?? 0.5, Math.floor(e / (schedule.every ?? 10)));
      case 'exp': return baseLr * Math.pow(schedule.gamma ?? 0.95, e);
      case 'cosine': return baseLr * 0.5 * (1 + Math.cos(Math.PI * Math.min(1, e / (schedule.total ?? 100))));
      case 'linear': return baseLr * Math.max(0, 1 - e / (schedule.total ?? 100));
      default: throw err('E_VALUE', `未知 lrSchedule.kind: ${schedule.kind}`);
    }
  }
  throw err('E_TYPE', 'lrSchedule 必须为函数或 {kind,...} 对象');
}

// 早停内核（在已产生的监控序列上评估）
function earlyStoppingImpl(values, opts) {
  const vals = arr(values);
  if (vals.length === 0) throw err('E_VALUE', 'monitor 序列不能为空');
  const patience = opts?.patience ?? 1;
  const mode = opts?.mode ?? 'min';
  const minDelta = opts?.minDelta ?? 0;
  if (!Number.isInteger(patience) || patience < 1) throw err('E_VALUE', `patience 必须为正整数，收到 ${patience}`);
  let best = vals[0], bestEpoch = 1, stale = 0;
  const trace = [{ epoch: 1, value: vals[0], improved: true, stale: 0 }];
  for (let i = 1; i < vals.length; i++) {
    const improved = mode === 'max' ? vals[i] > best + minDelta : vals[i] < best - minDelta;
    if (improved) { best = vals[i]; bestEpoch = i + 1; stale = 0; } else stale++;
    trace.push({ epoch: i + 1, value: vals[i], improved, stale });
  }
  return { shouldStop: stale >= patience, bestEpoch, bestValue: best, stale, trace, patience, mode };
}

// EMA 状态初始化 / 更新内核
function emaInitImpl(params, decay = 0.99) {
  if (!(decay >= 0 && decay < 1)) throw err('E_VALUE', `decay 必须 ∈ [0,1)，收到 ${decay}`);
  return { decay, params: deepClone(params) };
}
function emaUpdateImpl(state, params, decay) {
  if (!state || state.params === undefined) throw err('E_STATE', 'EMA 状态未初始化');
  const d = decay ?? state.decay;
  if (!(d >= 0 && d < 1)) throw err('E_VALUE', `decay 必须 ∈ [0,1)，收到 ${d}`);
  return { decay: d, params: mapParams(state.params, params, (e, p) => d * e + (1 - d) * p) };
}

// D9-023 完整训练循环
defineOp({
  id: nid(), domain: 9, name: 'train.loop.run', arity: 3,
  sig: '(model, data:{train, val?}, opts?) -> {history, params, ema, best, stopped, epochsRun}',
  doc: '通用训练循环：逐 epoch 分批（可选打乱/EMA/梯度裁剪/lr 调度/早停/回调），记录 history 与最优监控点。'
    + ' opts: {epochs, lr, batchSize, update, seed, shuffle, clipNorm, ema:{decay}, metric, earlyStopping:{patience,mode,minDelta}, lrSchedule, callbacks:{onEpochStart,onEpochEnd,onBatchEnd}, timeBudgetMs}',
  ref: '训练循环与早停：Prechelt (1998) Early Stopping — But When?; Goodfellow et al. (2016) Deep Learning, ch.8',
  inv: 'history.length === epochsRun；同种子同配置两次运行 params 逐位相同；callback 返回 false 或 ctx.stop 即停',
  impl: (model, data, opts = {}) => {
    if (!model || typeof model.lossGrad !== 'function' || typeof model.predict !== 'function' || model.params === undefined) {
      throw err('E_TYPE', 'model 需要 {params, predict(params,X), lossGrad(params,X,y)}');
    }
    const train = dsOf(data?.train, 'data.train');
    const val = data?.val ? dsOf(data.val, 'data.val') : null;
    const epochs = opts.epochs ?? 100;
    if (!Number.isInteger(epochs) || epochs < 1) throw err('E_VALUE', `epochs 必须为正整数，收到 ${epochs}`);
    const baseLr = opts.lr ?? 0.01;
    const bs = opts.batchSize && opts.batchSize > 0 ? opts.batchSize : train.n;
    const update = opts.update ?? sgdUpdate;
    const metric = resolveMetric(opts.metric);
    const es = opts.earlyStopping ?? null;
    const cbs = opts.callbacks ?? {};
    let params = deepClone(model.params);
    let emaState = opts.ema ? emaInitImpl(params, opts.ema.decay ?? 0.99) : null;
    const rng = createRng(opts.seed ?? 0);
    const history = [];
    let best = null, stale = 0, stopReason = 'epochs';
    const t0 = opts.timeBudgetMs != null ? Date.now() : null;
    for (let epoch = 1; epoch <= epochs; epoch++) {
      if (typeof cbs.onEpochStart === 'function') cbs.onEpochStart({ epoch, params, history });
      const lr = opts.lrSchedule ? lrHookImpl(epoch, baseLr, opts.lrSchedule) : baseLr;
      const order = opts.shuffle ? rng.shuffle(range(train.n)) : range(train.n);
      const batches = batchesFromIndex(train, order, bs, false);
      let lossSum = 0, seen = 0;
      for (const b of batches) {
        const step = batchStepImpl(model, params, b, lr, update, opts.clipNorm ?? null);
        params = step.params;
        lossSum += step.loss * b.index.length;
        seen += b.index.length;
        if (typeof cbs.onBatchEnd === 'function') cbs.onBatchEnd({ epoch, batch: b, loss: step.loss, gradNorm: step.gradNorm, params });
      }
      const trainLoss = seen > 0 ? lossSum / seen : NaN;
      let valLoss = null, valMetric = null;
      if (val) {
        valLoss = model.lossGrad(params, val.X, val.y).loss;
        if (metric) valMetric = metric(model.predict(params, val.X), val.y);
      }
      history.push({ epoch, lr, trainLoss, valLoss, valMetric });
      if (emaState) emaState = emaUpdateImpl(emaState, params);
      // 监控值优先取 valMetric，其次 valLoss，最后 trainLoss
      const monitor = valMetric != null ? valMetric : (valLoss != null ? valLoss : trainLoss);
      if (es) {
        const mode = es.mode ?? 'min', md = es.minDelta ?? 0;
        if (best === null || (mode === 'max' ? monitor > best.value + md : monitor < best.value - md)) {
          best = { epoch, value: monitor };
          stale = 0;
        } else stale++;
      } else if (best === null) {
        best = { epoch, value: monitor };
      }
      let brk = null;
      if (es && stale >= (es.patience ?? 1)) brk = 'earlyStopping';
      if (opts.timeBudgetMs != null && t0 != null && Date.now() - t0 >= opts.timeBudgetMs) brk = 'timeBudget';
      if (typeof cbs.onEpochEnd === 'function') {
        const ctx = { epoch, params, trainLoss, valLoss, valMetric, history, lr, stop: false };
        const r = cbs.onEpochEnd(ctx);
        if (r === false || ctx.stop) brk = 'callback';
      }
      if (brk) { stopReason = brk; break; }
    }
    return { history, params, ema: emaState, best, stopped: stopReason, epochsRun: history.length };
  },
});

// D9-024 单轮训练
defineOp({
  id: nid(), domain: 9, name: 'train.loop.epoch', arity: 4,
  sig: '(model, params, dataset, opts?) -> {params, loss, batches}',
  doc: '在给定参数上跑一个 epoch：按 batchSize 分批（可选打乱），逐批梯度步，返回更新后参数与平均损失。',
  ref: 'Epoch/小批训练惯例：Bottou (2010) COMPSTAT；Goodfellow et al. (2016) Deep Learning, ch.8',
  inv: '不修改入参 params（返回新结构）；loss 为按样本数加权的批平均',
  impl: (model, params, data, opts = {}) => {
    const train = dsOf(data, 'train.loop.epoch');
    if (!model || typeof model.lossGrad !== 'function') throw err('E_TYPE', 'model 需要 lossGrad');
    const bs = opts.batchSize && opts.batchSize > 0 ? opts.batchSize : train.n;
    const update = opts.update ?? sgdUpdate;
    const lr = opts.lr ?? 0.01;
    const order = opts.shuffle ? createRng(opts.seed ?? 0).shuffle(range(train.n)) : range(train.n);
    const batches = batchesFromIndex(train, order, bs, false);
    let p = deepClone(params), lossSum = 0, seen = 0;
    for (const b of batches) {
      const step = batchStepImpl(model, p, b, lr, update, opts.clipNorm ?? null);
      p = step.params;
      lossSum += step.loss * b.index.length;
      seen += b.index.length;
    }
    return { params: p, loss: seen > 0 ? lossSum / seen : NaN, batches: batches.length };
  },
});

// D9-025 在数据集上评估模型
defineOp({
  id: nid(), domain: 9, name: 'train.loop.evaluate', arity: 3,
  sig: '(model, dataset, metric?) -> number',
  doc: '用 model.predict(model.params, X) 预测并按 metric 打分；metric 支持函数或 \'accuracy\'|\'mse\'|\'mae\'，默认整数标签→准确率、连续→MSE。',
  ref: '泛化误差评估：Hastie, Tibshirani & Friedman (2009) The Elements of Statistical Learning, §7',
  inv: '不修改 model.params；metric 为 null 时按标签类型选择默认指标',
  impl: (model, data, metric = null) => {
    const ds = dsOf(data, 'train.loop.evaluate');
    if (!model || typeof model.predict !== 'function' || model.params === undefined) throw err('E_TYPE', 'model 需要 {params, predict}');
    const pred = model.predict(model.params, ds.X);
    const m = resolveMetric(metric);
    return m ? m(pred, ds.y) : defaultMetric(pred, ds.y);
  },
});

// D9-026 早停判定（纯函数，作用于监控序列）
defineOp({
  id: nid(), domain: 9, name: 'train.stop.earlyStopping', arity: 2,
  sig: '(values:array|Tensor, opts?:{patience, mode, minDelta}) -> {shouldStop, bestEpoch, bestValue, stale, trace}',
  doc: '在已产生的监控序列上评估早停：超过 patience 轮无改善（幅度 > minDelta）即应停止；返回最佳轮次与逐步 trace。',
  ref: 'Prechelt (1998) Early Stopping — But When? (Neural Networks: Tricks of the Trade); Prechelt & Probst (1996) 收敛监控',
  inv: 'bestEpoch/bestValue 为 mode 方向上的最优点；stale = 自最佳点以来的未改善轮数',
  impl: (values, opts = {}) => earlyStoppingImpl(values, opts),
});

// D9-027 最大轮数停止
defineOp({
  id: nid(), domain: 9, name: 'train.stop.maxEpochs', arity: 2,
  sig: '(epoch:number, max:number) -> boolean',
  doc: '轮数上限规则：epoch >= max 时返回 true（达到预算即停）。',
  ref: '训练预算控制：Domhan et al. (2015) Speeding Up Automatic Hyperparameter Optimization of DNNs, ICML',
  inv: 'epoch < max 返回 false；epoch >= max 返回 true',
  impl: (epoch, max) => {
    if (!Number.isInteger(epoch) || epoch < 1) throw err('E_VALUE', `epoch 必须为正整数，收到 ${epoch}`);
    if (!Number.isInteger(max) || max < 1) throw err('E_VALUE', `max 必须为正整数，收到 ${max}`);
    return epoch >= max;
  },
});

// D9-028 时间预算停止
defineOp({
  id: nid(), domain: 9, name: 'train.stop.timeBudget', arity: 3,
  sig: '(startedAtMs:number, budgetMs:number, nowMs?:number) -> {elapsed, remaining, exceeded}',
  doc: '时间预算规则：返回已用/剩余毫秒与是否超预算；nowMs 缺省取当前时间（测试可显式传入以保证可复现）。',
  ref: '预算受限训练与提前终止：Domhan et al. (2015) ICML；Real et al. (2017) Large-Scale Evolution of Image Classifiers',
  inv: 'elapsed = now - startedAt；exceeded = elapsed >= budgetMs；remaining = budgetMs - elapsed',
  impl: (startedAtMs, budgetMs, nowMs) => {
    if (!(typeof startedAtMs === 'number' && Number.isFinite(startedAtMs))) throw err('E_TYPE', 'startedAtMs 必须为有限数');
    if (!(typeof budgetMs === 'number' && budgetMs > 0)) throw err('E_VALUE', `budgetMs 必须为正数，收到 ${budgetMs}`);
    const now = nowMs ?? Date.now();
    const elapsed = now - startedAtMs;
    return { startedAtMs, budgetMs, elapsed, remaining: budgetMs - elapsed, exceeded: elapsed >= budgetMs };
  },
});

// D9-029 检查点保存（纯数据快照，无时间戳，可复现）
defineOp({
  id: nid(), domain: 9, name: 'train.ckpt.save', arity: 2,
  sig: '(params, meta?) -> {schema, params, meta}',
  doc: '把参数结构深拷贝为可序列化快照；不含时间戳，保证同参数同快照。',
  ref: '检查点/快照语义：Pedregosa et al. (2011) Scikit-learn, JMLR 12（模型持久化约定）',
  inv: '快照与 params 逐位相等且互不影响（写时拷贝）',
  impl: (params, meta = null) => ({
    schema: 'd9.ckpt/1',
    params: deepClone(params),
    meta: meta === null || meta === undefined ? null : deepClone(meta),
  }),
});

// D9-030 检查点加载
defineOp({
  id: nid(), domain: 9, name: 'train.ckpt.load', arity: 1,
  sig: '(ckpt) -> params',
  doc: '从快照深拷贝恢复参数；返回独立副本，不与快照共享内存。',
  ref: 'Pedregosa et al. (2011) Scikit-learn: Machine Learning in Python, JMLR 12',
  inv: 'load(save(p)) 与 p 逐位相等；修改返回值不影响快照',
  impl: (ckpt) => {
    if (!ckpt || ckpt.params === undefined) throw err('E_VALUE', 'ckpt 缺少 params 字段');
    return deepClone(ckpt.params);
  },
});

// D9-031 最优值跟踪器（有状态对象）
defineOp({
  id: nid(), domain: 9, name: 'train.ckpt.bestTracker', arity: 1,
  sig: "(mode?:'min'|'max') -> {update(value, epoch?, payload?), best, bestEpoch, updates, snapshot()}",
  doc: '有状态最优跟踪器：每次 update 返回是否刷新纪录；可按需附带 payload（如参数快照）。',
  ref: 'Prechelt (1998) Early Stopping — But When?；验证集选模惯例：Hastie et al. (2009) ESL §7',
  inv: 'mode=min 时 best 单调不增；snapshot() 与内部状态一致',
  impl: (mode = 'min') => {
    if (mode !== 'min' && mode !== 'max') throw err('E_VALUE', `mode 必须为 'min'|'max'，收到 ${mode}`);
    const st = { mode, best: null, bestEpoch: null, bestPayload: null, updates: 0 };
    st.update = (value, epoch, payload) => {
      if (typeof value !== 'number' || !Number.isFinite(value)) throw err('E_TYPE', `value 必须为有限数，收到 ${value}`);
      st.updates++;
      const improved = st.best === null || (mode === 'max' ? value > st.best : value < st.best);
      if (improved) {
        st.best = value;
        st.bestEpoch = epoch ?? st.updates;
        st.bestPayload = payload === undefined || payload === null ? null : deepClone(payload);
      }
      return { improved, best: st.best, bestEpoch: st.bestEpoch, updates: st.updates };
    };
    st.snapshot = () => ({ mode: st.mode, best: st.best, bestEpoch: st.bestEpoch, updates: st.updates });
    return st;
  },
});

// D9-032 训练历史追加记录
defineOp({
  id: nid(), domain: 9, name: 'train.hist.log', arity: 2,
  sig: '(history:array, entry:object) -> array',
  doc: '向训练历史追加一条记录（缺省 epoch 时取 length+1）；纯函数，不修改入参数组。',
  ref: '训练日志惯例：Goodfellow et al. (2016) Deep Learning, ch.8',
  inv: '返回长度 = 入参长度 + 1；入参 history 不被修改',
  impl: (history, entry) => {
    const h = Array.isArray(history) ? history.slice() : [];
    if (!entry || typeof entry !== 'object') throw err('E_TYPE', 'entry 必须为对象');
    const rec = { epoch: h.length + 1, ...entry }; // epoch 置首位，保证列序稳定
    h.push(rec);
    return h;
  },
});

// D9-033 训练历史渲染为 Markdown 表
defineOp({
  id: nid(), domain: 9, name: 'train.hist.toMarkdown', arity: 2,
  sig: '(history:array, cols?:string[]) -> string',
  doc: '把历史记录渲染成 Markdown 表格；cols 缺省取所有记录键的有序并集。',
  ref: '表格化报告惯例：Demšar (2006) Statistical Comparisons of Classifiers over Multiple Data Sets, JMLR',
  inv: '输出首行为表头 | ... |，第二行为分隔行；行数 = history.length',
  impl: (history, cols = null) => {
    if (!Array.isArray(history)) throw err('E_TYPE', 'history 必须为数组');
    return tableMarkdown(history, cols);
  },
});

// D9-034 学习率钩子：按 epoch 计算当前 lr
defineOp({
  id: nid(), domain: 9, name: 'train.lr.hook', arity: 3,
  sig: '(epoch:number, baseLr:number, schedule:fn|{kind,...}) -> number',
  doc: '学习率调度：支持函数(epoch)->倍率，或 {kind:\'step\'| \'exp\'| \'cosine\'| \'linear\', ...}；epoch 从 1 计数。',
  ref: 'Smith (2017) Cyclical Learning Rates for Training Neural Networks (SGDR); Loshchilov & Hutter (2017) SGDR, ICLR',
  inv: 'epoch=1 时倍率为 1（step/exp/cosine/linear 均如此）；返回 baseLr*倍率',
  impl: (epoch, baseLr, schedule) => {
    if (!Number.isInteger(epoch) || epoch < 1) throw err('E_VALUE', `epoch 必须为正整数，收到 ${epoch}`);
    if (!(typeof baseLr === 'number' && baseLr > 0)) throw err('E_VALUE', `baseLr 必须为正数，收到 ${baseLr}`);
    return lrHookImpl(epoch, baseLr, schedule);
  },
});

// D9-035 EMA 更新（Polyak 平均）
defineOp({
  id: nid(), domain: 9, name: 'train.ema.update', arity: 3,
  sig: '(state:{decay, params}, params, decay?:number) -> state',
  doc: '指数滑动平均更新：ema ← decay·ema + (1-decay)·params；decay 缺省用 state.decay。',
  ref: 'Polyak & Juditsky (1992) Acceleration of Stochastic Approximation by Averaging, SIAM J. Control Optim.',
  inv: '新 state 与入参互不影响；decay=0 时 state.params 等于当前 params',
  impl: (state, params, decay = null) => emaUpdateImpl(state, params, decay ?? undefined),
});

// D9-036 取出 EMA 参数
defineOp({
  id: nid(), domain: 9, name: 'train.ema.apply', arity: 1,
  sig: '(state) -> params',
  doc: '返回 EMA 参数的深拷贝（用于评估/推理，不影响 state）。',
  ref: 'Polyak & Juditsky (1992) SIAM J. Control Optim.',
  inv: 'apply(state) 与 state.params 逐位相等；修改返回值不影响 state',
  impl: (state) => {
    if (!state || state.params === undefined) throw err('E_STATE', 'EMA 状态未初始化');
    return deepClone(state.params);
  },
});

// D9-037 mixup 数据增强（Zhang 2018）
defineOp({
  id: nid(), domain: 9, name: 'train.mixup', arity: 3,
  sig: '(dataset, alpha:number, rng) -> {X, y, lam, perm}',
  doc: 'mixup：lam ~ Beta(alpha, alpha)，按随机置换线性插值样本与标签（回归值/one-hot 均适用）；alpha≤0 时 lam=1（不混合）。',
  ref: 'Zhang et al. (2018) mixup: Beyond Empirical Risk Minimization, ICLR',
  inv: 'X\'[i] = lam·X[i] + (1-lam)·X[perm[i]]；y\' 同理；lam ∈ [0,1]',
  impl: (ds, alpha, rng) => {
    const { X, y, n } = dsOf(ds);
    if (!rng || typeof rng.next !== 'function') throw err('E_TYPE', 'rng 必须为 createRng 实例');
    if (!(typeof alpha === 'number' && alpha >= 0)) throw err('E_VALUE', `alpha 必须为非负数，收到 ${alpha}`);
    const perm = rng.shuffle(range(n));
    let lam = 1;
    if (alpha > 0) {
      const g1 = gammaSample(rng, alpha), g2 = gammaSample(rng, alpha);
      lam = g1 / (g1 + g2);
    }
    const Xout = new Float64Array(X.size);
    const d = X.shape[1];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) Xout[i * d + j] = lam * X.get2(i, j) + (1 - lam) * X.get2(perm[i], j);
    }
    const yout = new Float64Array(n);
    for (let i = 0; i < n; i++) yout[i] = lam * y.data[i] + (1 - lam) * y.data[perm[i]];
    return {
      X: new Tensor(Xout, X.shape.slice()),
      y: new Tensor(yout, [n]),
      lam, perm,
    };
  },
});

// D9-038 标签平滑
defineOp({
  id: nid(), domain: 9, name: 'train.labelSmoothing', arity: 3,
  sig: '(targets:Tensor[n]|Tensor[n,C], numClasses:number, eps?:number) -> Tensor[n,C]',
  doc: '标签平滑（Szegedy 2016）：硬标签 → (1-eps)·onehot + eps/C；已 soft 的 2D 目标 → 向均匀分布收缩 eps。',
  ref: 'Szegedy et al. (2016) Rethinking the Inception Architecture for Computer Vision, CVPR; Müller et al. (2019) When Does Label Smoothing Help?, NeurIPS',
  inv: '输出每行和为 1；eps=0 时对硬标签为纯 one-hot',
  impl: (targets, numClasses, eps = 0.1) => {
    const C = numClasses;
    if (!Number.isInteger(C) || C < 2) throw err('E_VALUE', `numClasses 必须 ≥2 的整数，收到 ${C}`);
    if (!(eps >= 0 && eps < 1)) throw err('E_VALUE', `eps 必须 ∈ [0,1)，收到 ${eps}`);
    if (!(targets instanceof Tensor)) throw err('E_TYPE', 'targets 必须为 Tensor');
    if (targets.ndim === 1) {
      const Y = targets.toArray();
      const out = new Float64Array(Y.length * C);
      for (let i = 0; i < Y.length; i++) {
        const k = Y[i];
        if (!Number.isInteger(k) || k < 0 || k >= C) throw err('E_VALUE', `标签 ${k} 越界（C=${C}）`);
        for (let c = 0; c < C; c++) out[i * C + c] = eps / C;
        out[i * C + k] += 1 - eps;
      }
      return new Tensor(out, [Y.length, C]);
    }
    if (targets.ndim === 2) {
      const [n, c2] = targets.shape;
      if (c2 !== C) throw err('E_SHAPE', `targets 列数 ${c2} 与 numClasses ${C} 不符`);
      const out = new Float64Array(targets.size);
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < C; c++) out[i * C + c] = (1 - eps) * targets.get2(i, c) + eps / C;
      }
      return new Tensor(out, [n, C]);
    }
    throw err('E_SHAPE', 'targets 必须为 1D（硬标签）或 2D（soft 标签）');
  },
});

// D9-039 高斯噪声增强
defineOp({
  id: nid(), domain: 9, name: 'train.aug.gaussianNoise', arity: 3,
  sig: '(X:Tensor[n,d], sigma:number, rng) -> Tensor[n,d]',
  doc: '逐元素加性高斯噪声 X + sigma·N(0,1)（等价于 Tikhonov 正则化视角的噪声训练）。',
  ref: 'Bishop (1995) Training with Noise is Equivalent to Tikhonov Regularization, Neural Computation 6(1)',
  inv: 'sigma=0 时输出与输入逐位相同；同种子可复现',
  impl: (X, sigma, rng) => {
    if (!(X instanceof Tensor)) throw err('E_TYPE', 'X 必须为 Tensor');
    if (!(typeof sigma === 'number' && sigma >= 0)) throw err('E_VALUE', `sigma 必须为非负数，收到 ${sigma}`);
    if (!rng || typeof rng.normal !== 'function') throw err('E_TYPE', 'rng 必须为 createRng 实例');
    const out = new Float64Array(X.size);
    for (let i = 0; i < X.size; i++) out[i] = X.data[i] + sigma * rng.normal();
    return new Tensor(out, X.shape.slice());
  },
});

// D9-040 输入特征 dropout 增强
defineOp({
  id: nid(), domain: 9, name: 'train.aug.featureDropout', arity: 3,
  sig: '(X:Tensor[n,d], p:number, rng) -> Tensor[n,d]',
  doc: '逐元素输入 dropout：以概率 p 置零，保留元素乘 1/(1-p)（inverted dropout，保持期望不变）。',
  ref: 'Srivastava et al. (2014) Dropout: A Simple Way to Prevent Neural Networks from Overfitting, JMLR 15',
  inv: 'p=0 时输出与输入逐位相同；零元素占比 ≈ p；同种子可复现',
  impl: (X, p, rng) => {
    if (!(X instanceof Tensor)) throw err('E_TYPE', 'X 必须为 Tensor');
    if (!(p >= 0 && p < 1)) throw err('E_VALUE', `p 必须 ∈ [0,1)，收到 ${p}`);
    if (!rng || typeof rng.next !== 'function') throw err('E_TYPE', 'rng 必须为 createRng 实例');
    const keep = 1 / (1 - p);
    const out = new Float64Array(X.size);
    for (let i = 0; i < X.size; i++) out[i] = rng.next() < p ? 0 : X.data[i] * keep;
    return new Tensor(out, X.shape.slice());
  },
});

// D9-041 随机尺度抖动增强
defineOp({
  id: nid(), domain: 9, name: 'train.aug.scaleJitter', arity: 3,
  sig: '(X:Tensor[n,d], sigma:number, rng) -> Tensor[n,d]',
  doc: '每行乘随机尺度因子 1 + sigma·N(0,1)（行级幅度抖动）；sigma=0 为恒等。',
  ref: '数据增强综述：Shorten & Khoshgoftaar (2019) A Survey on Image Data Augmentation for Deep Learning, Applied Intelligence',
  inv: 'sigma=0 时输出与输入逐位相同；同种子可复现',
  impl: (X, sigma, rng) => {
    if (!(X instanceof Tensor)) throw err('E_TYPE', 'X 必须为 Tensor');
    if (!(typeof sigma === 'number' && sigma >= 0)) throw err('E_VALUE', `sigma 必须为非负数，收到 ${sigma}`);
    if (!rng || typeof rng.normal !== 'function') throw err('E_TYPE', 'rng 必须为 createRng 实例');
    const out = new Float64Array(X.size);
    const d = X.shape[1];
    for (let i = 0; i < X.shape[0]; i++) {
      const s = 1 + sigma * rng.normal();
      for (let j = 0; j < d; j++) out[i * d + j] = X.get2(i, j) * s;
    }
    return new Tensor(out, X.shape.slice());
  },
});

// D9-042 梯度累积：累加一步
defineOp({
  id: nid(), domain: 9, name: 'train.accum.step', arity: 2,
  sig: '(state:{sum, count}, grad) -> state',
  doc: '把一批梯度累加到累积器（sum 与 grad 同构相加，count+1）；state 可用 accum.reset() 或 accum.reset(grad) 初始化。',
  ref: '梯度累积（大有效批）：Huang et al. (2019) GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism, NeurIPS',
  inv: '返回新 state；sum 与 grad 同构；count 每步 +1',
  impl: (state, grad) => {
    if (!state || typeof state !== 'object') throw err('E_TYPE', 'state 必须为 {sum, count}');
    if (state.sum === null || state.sum === undefined) return { sum: deepClone(grad), count: 1 };
    return { sum: mapParams(state.sum, grad, (a, b) => a + b), count: (state.count ?? 0) + 1 };
  },
});

// D9-043 梯度累积器重置
defineOp({
  id: nid(), domain: 9, name: 'train.accum.reset', arity: 1,
  sig: '(grad?) -> {sum, count}',
  doc: '重置累积器：无参 → {sum:null, count:0}；给定 grad → 预分配同构全零 sum。',
  ref: 'Huang et al. (2019) GPipe, NeurIPS',
  inv: 'reset() 后 count=0；reset(grad) 后 sum 与 grad 同构且全零',
  impl: (grad = null) => (grad === null || grad === undefined ? { sum: null, count: 0 } : { sum: zeroLike(grad), count: 0 }),
});

// D9-044 带梯度裁剪的更新函数包装器
defineOp({
  id: nid(), domain: 9, name: 'train.loop.withClipping', arity: 2,
  sig: '(update:fn, maxNorm:number) -> (params, grad, lr) -> params',
  doc: '包装任意 update：先把梯度按全局 L2 范数裁剪到 maxNorm，再调用原 update。',
  ref: 'Pascanu, Mikolov & Bengio (2013) On the Difficulty of Training Recurrent Neural Models, ICML',
  inv: '‖grad‖ ≤ maxNorm 时原样传递；否则缩放 maxNorm/‖grad‖ 倍',
  impl: (update, maxNorm) => {
    if (typeof update !== 'function') throw err('E_TYPE', 'update 必须为函数 (params, grad, lr) -> params');
    if (!(typeof maxNorm === 'number' && maxNorm > 0)) throw err('E_VALUE', `maxNorm 必须为正数，收到 ${maxNorm}`);
    return (params, grad, lr) => update(params, clipGradImpl(grad, maxNorm), lr);
  },
});

// D9-045 每 epoch 打乱器：确定性生成第 epoch 轮的行顺序
defineOp({
  id: nid(), domain: 9, name: 'train.loop.shuffleEachEpoch', arity: 1,
  sig: '(seed:number) -> (n:number, epoch:number) -> number[]',
  doc: '返回确定性打乱函数：同一 (seed, n, epoch) 产生同一排列，不同 epoch 产生不同排列。',
  ref: 'Bottou (2010) Large-Scale Machine Learning with SGD, COMPSTAT（每轮重排）',
  inv: 'perm(n, epoch) 是 [0,n) 的排列；perm(n, e) 对同一入参可复现',
  impl: (seed) => {
    const s = seed >>> 0;
    return (n, epoch) => {
      if (!Number.isInteger(n) || n < 1) throw err('E_VALUE', `n 必须为正整数，收到 ${n}`);
      if (!Number.isInteger(epoch) || epoch < 1) throw err('E_VALUE', `epoch 必须为正整数，收到 ${epoch}`);
      return createRng(seedFrom(`${s}|epoch${epoch}`)).shuffle(range(n));
    };
  },
});

// D9-046 单批次梯度步（对外暴露）
defineOp({
  id: nid(), domain: 9, name: 'train.loop.batchStep', arity: 5,
  sig: '(model, params, batch:{X,y}, lr:number, update?:fn) -> {loss, grad, gradNorm, params}',
  doc: '对一个批次执行一次梯度步：调用 model.lossGrad，经 update(params, grad, lr) 更新；update 缺省为 SGD。',
  ref: 'Goodfellow et al. (2016) Deep Learning, §8.3（小批梯度下降）',
  inv: '不修改入参 params；返回 {loss, grad, gradNorm, params(新)}',
  impl: (model, params, batch, lr, update = null) => {
    if (!model || typeof model.lossGrad !== 'function') throw err('E_TYPE', 'model 需要 lossGrad');
    if (!batch || !(batch.X instanceof Tensor) || !(batch.y instanceof Tensor)) throw err('E_TYPE', 'batch 需要 {X, y}');
    if (!(typeof lr === 'number' && lr > 0)) throw err('E_VALUE', `lr 必须为正数，收到 ${lr}`);
    return batchStepImpl(model, params, batch, lr, update ?? sgdUpdate, null);
  },
});

// D9-047 EMA 状态初始化
defineOp({
  id: nid(), domain: 9, name: 'train.ema.init', arity: 2,
  sig: '(params, decay?:number) -> {decay, params}',
  doc: '用初始参数深拷贝创建 EMA 状态；decay ∈ [0,1)（默认 0.99）。',
  ref: 'Polyak & Juditsky (1992) Acceleration of Stochastic Approximation by Averaging, SIAM J. Control Optim.',
  inv: 'state.params 与入参逐位相等但相互独立',
  impl: (params, decay = 0.99) => emaInitImpl(params, decay),
});

// D9-048 组合停止准则
defineOp({
  id: nid(), domain: 9, name: 'train.stop.criteria', arity: 2,
  sig: "(rules:{name, stop}[], mode?:'all'|'any') -> {stop, triggered, mode}",
  doc: '组合多个停止规则：mode=all 时全部触发才停（默认），mode=any 时任一触发即停；返回触发的规则名。',
  ref: '早停与预算控制的组合策略：Domhan et al. (2015) ICML；Prechelt (1998)',
  inv: 'rules 为空数组时 stop=false；triggered ⊆ rules 中 stop=true 的项',
  impl: (rules, mode = 'all') => {
    if (!Array.isArray(rules) || rules.some(r => !r || typeof r.stop !== 'boolean')) throw err('E_TYPE', 'rules 必须为 {name, stop}[]');
    if (mode !== 'all' && mode !== 'any') throw err('E_VALUE', `mode 必须为 'all'|'any'，收到 ${mode}`);
    const triggered = rules.filter(r => r.stop).map(r => r.name);
    // 空规则集恒不停止（vacuous truth 不作为停止依据）
    const stop = rules.length === 0 ? false : mode === 'all' ? triggered.length === rules.length : triggered.length > 0;
    return { stop, triggered, mode };
  },
});

// ============================================================================
// 三、评估分析（D9-049 ~ D9-074）
// fitPredict 约定：(train:dataset, val:dataset) -> 预测(Tensor|数组|{pred}) 或数值分数
// ============================================================================

function trainTestSplitImpl(ds, testFrac, seed) {
  const d = dsOf(ds, 'trainTestSplit');
  if (!(typeof testFrac === 'number' && testFrac >= 0 && testFrac < 1)) throw err('E_VALUE', `testFrac 必须 ∈ [0,1)，收到 ${testFrac}`);
  const testN = Math.min(Math.round(d.n * testFrac), d.n - 1);
  const perm = createRng(seed).shuffle(range(d.n));
  const testIdx = perm.slice(0, testN).sort((a, b) => a - b);
  const trainIdx = perm.slice(testN).sort((a, b) => a - b);
  return { train: takeRows(d, trainIdx), test: takeRows(d, testIdx) };
}

// k 折交叉验证内核
function crossValidateImpl(ds, k, fitPredict, opts = {}) {
  const d = dsOf(ds, 'crossValidate');
  if (typeof fitPredict !== 'function') throw err('E_TYPE', 'fitPredict 必须为函数');
  const folds = kFoldSplitImpl(d, k, opts.seed ?? 20261014);
  const metric = resolveMetric(opts.metric);
  const scores = folds.map(f => scoreOf(fitPredict(f.train, f.val), f.val.y, metric));
  return { k, scores, mean: meanOf(scores), std: stdOf(scores), valIdx: folds.map(f => f.valIdx) };
}

// D9-049 k 折交叉验证
defineOp({
  id: nid(), domain: 9, name: 'train.eval.crossValidate', arity: 4,
  sig: '(dataset, k:number, fitPredict:fn, opts?:{seed, metric}) -> {k, scores, mean, std, valIdx}',
  doc: 'k 折交叉验证：每折用 fitPredict(train, val) 产出预测并按 metric 打分；返回 k 个分数及均值±总体标准差。',
  ref: 'Stone (1974) JRSS-B; Kohavi (1995) A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection, IJCAI',
  inv: 'scores.length === k；mean/std 为 population 统计；valIdx 各折不相交且并集 = [0,n)',
  impl: (ds, k, fitPredict, opts = {}) => crossValidateImpl(ds, k, fitPredict, opts),
});

// D9-050 重复 k 折交叉验证
defineOp({
  id: nid(), domain: 9, name: 'train.eval.repeatedCV', arity: 5,
  sig: '(dataset, k:number, repeats:number, fitPredict:fn, opts?:{seed, metric}) -> {repeats, meanOfMeans, stdOfMeans, allScores}',
  doc: '以不同种子重复 k 折 CV repeats 次，聚合各次 mean/std 与全部分数，降低单次划分方差。',
  ref: 'Dietterich (1998) Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms, Neural Computation',
  inv: 'repeats 数组长度 = repeats；allScores.length = k*repeats；meanOfMeans = 各次 mean 的均值',
  impl: (ds, k, repeats, fitPredict, opts = {}) => {
    if (!Number.isInteger(repeats) || repeats < 1) throw err('E_VALUE', `repeats 必须为正整数，收到 ${repeats}`);
    const seed0 = opts?.seed ?? 20261014;
    const reps = [];
    const allScores = [];
    for (let r = 0; r < repeats; r++) {
      const cv = crossValidateImpl(ds, k, fitPredict, { ...opts, seed: seed0 + r * 7919 });
      reps.push({ repeat: r, seed: seed0 + r * 7919, scores: cv.scores, mean: cv.mean, std: cv.std });
      allScores.push(...cv.scores);
    }
    return {
      repeats: reps,
      k,
      meanOfMeans: meanOf(reps.map(r => r.mean)),
      stdOfMeans: stdOf(reps.map(r => r.mean)),
      allScores,
      grandMean: meanOf(allScores),
    };
  },
});

// D9-051 学习曲线：随训练规模变化的训练/验证分数
defineOp({
  id: nid(), domain: 9, name: 'train.eval.learningCurve', arity: 4,
  sig: '(dataset, sizes:number[], fitPredict:fn, opts?:{seed, metric, valFrac}) -> {n, trainScore, valScore}[]',
  doc: '固定验证集下，对递增训练子集规模分别评估训练分与验证分，判断欠拟合/过拟合趋势。',
  ref: '学习曲线分析：Perlich, Provost & Simonoff (2003) Tree Induction vs. Logistic Regression: A Learning-Curve Analysis, Machine Learning',
  inv: 'sizes 中每个 m ∈ [1, pool 大小]；返回长度 = sizes.length；各次子集为同一打乱序列的前缀（可比）',
  impl: (ds, sizes, fitPredict, opts = {}) => {
    const d = dsOf(ds, 'learningCurve');
    const seed = opts?.seed ?? 20261014;
    const metric = resolveMetric(opts?.metric) ?? defaultMetric;
    const { train: pool, test: val } = trainTestSplitImpl(d, opts?.valFrac ?? 0.25, seed);
    const poolN = pool.X.shape[0];
    const perm = createRng(seed + 1).shuffle(range(poolN));
    const out = [];
    for (const m of arr(sizes)) {
      if (!Number.isInteger(m) || m < 1 || m > poolN) throw err('E_VALUE', `规模 ${m} 越界（pool=${poolN}）`);
      const sub = takeRows(pool, perm.slice(0, m));
      const trainScore = scoreOf(fitPredict(sub, sub), sub.y, metric);
      const valScore = scoreOf(fitPredict(sub, val), val.y, metric);
      out.push({ n: m, trainScore, valScore });
    }
    return out;
  },
});

// D9-052 验证曲线：随超参数变化的 CV 分数
defineOp({
  id: nid(), domain: 9, name: 'train.eval.validationCurve', arity: 4,
  sig: '(dataset, values:array, makeFitPredict:(v)->fn, opts?:{k, seed, metric}) -> {value, mean, std, scores}[]',
  doc: '对每个候选超参数值做 k 折 CV，返回该值的 mean±std 与各折分数，用于选模。',
  ref: '验证曲线/模型选择：Hastie, Tibshirani & Friedman (2009) The Elements of Statistical Learning, §7.10',
  inv: '返回长度 = values.length；每项 scores.length === k；同一 (值, 种子) 可复现',
  impl: (ds, values, makeFitPredict, opts = {}) => {
    const d = dsOf(ds, 'validationCurve');
    if (typeof makeFitPredict !== 'function') throw err('E_TYPE', 'makeFitPredict 必须为函数');
    const k = opts?.k ?? 5;
    const seed = opts?.seed ?? 20261014;
    const metric = resolveMetric(opts?.metric);
    return arr(values).map(v => {
      const cv = crossValidateImpl(d, k, makeFitPredict(v), { seed, metric });
      return { value: v, mean: cv.mean, std: cv.std, scores: cv.scores };
    });
  },
});

// D9-053 Bootstrap 置信区间
defineOp({
  id: nid(), domain: 9, name: 'train.eval.bootstrapCI', arity: 2,
  sig: '(values:array|Tensor, opts?:{B, alpha, seed, statistic}) -> {point, lo, hi, mean, std, B, alpha}',
  doc: '对有放回重采样 B 次计算统计量（默认均值）的分位数置信区间；固定种子完全可复现。',
  ref: 'Efron (1979) Bootstrap Methods: Another Look at the Jackknife, Ann. Statist. 7(1); Efron & Tibshirani (1993) An Introduction to the Bootstrap',
  inv: '同种子两次调用结果逐位相同；point = statistic(原始样本)；默认区间为 [alpha/2, 1-alpha/2] 分位',
  impl: (values, opts = {}) => {
    const v = arr(values);
    if (v.length === 0) throw err('E_VALUE', 'values 不能为空');
    const B = opts?.B ?? 1000;
    const alpha = opts?.alpha ?? 0.05;
    const seed = opts?.seed ?? 20261014;
    if (!Number.isInteger(B) || B < 1) throw err('E_VALUE', `B 必须为正整数，收到 ${B}`);
    if (!(alpha > 0 && alpha < 1)) throw err('E_VALUE', `alpha 必须 ∈ (0,1)，收到 ${alpha}`);
    let stat = opts?.statistic;
    if (stat == null) stat = meanOf;
    else if (typeof stat === 'string') {
      if (stat === 'mean') stat = meanOf;
      else if (stat === 'median') stat = (a) => quantileSorted(a.slice().sort((x, y) => x - y), 0.5);
      else throw err('E_VALUE', `未知 statistic: ${stat}`);
    } else if (typeof stat !== 'function') throw err('E_TYPE', 'statistic 必须为函数或 \'mean\'|\'median\'');
    const point = stat(v);
    const rng = createRng(seed);
    const samples = new Array(B);
    for (let b = 0; b < B; b++) {
      const rs = new Array(v.length);
      for (let i = 0; i < v.length; i++) rs[i] = v[rng.int(v.length)];
      samples[b] = stat(rs);
    }
    samples.sort((x, y) => x - y);
    return {
      point,
      lo: quantileSorted(samples, alpha / 2),
      hi: quantileSorted(samples, 1 - alpha / 2),
      mean: meanOf(samples),
      std: stdOf(samples),
      B, alpha,
      sampleSize: v.length,
    };
  },
});

// D9-054 配对比较（两模型同折分数）
defineOp({
  id: nid(), domain: 9, name: 'train.eval.pairedCompare', arity: 3,
  sig: '(scoresA:array|Tensor, scoresB:array|Tensor, opts?) -> {meanDiff, stdDiff, se, t, df, ciLo, ciHi, pNormal, wins, losses, ties}',
  doc: '配对差分析：meanDiff=mean(A-B)，给出配对 t 统计量、正态近似 p 值与胜/负/平计数；CI 用正态分位近似。',
  ref: 'Dietterich (1998) Neural Computation; Nadeau & Bengio (2003) Inference for the Generalization Error, NIPS',
  inv: 'meanDiff = mean(A-B)；wins+losses+ties = n；t = meanDiff/se，se = stdDiff/sqrt(n)（ddof=1）',
  impl: (a, b, opts = {}) => {
    const A = arr(a), B = arr(b);
    if (A.length === 0 || A.length !== B.length) throw err('E_VALUE', `scoresA/B 需等长非空，收到 ${A.length} / ${B.length}`);
    const diffs = A.map((v, i) => v - B[i]);
    const md = meanOf(diffs);
    const sd = sampleStd(diffs);
    const n = A.length;
    const se = n > 1 ? sd / Math.sqrt(n) : 0;
    const t = se > 0 ? md / se : 0;
    const z = 1.959963984540054; // Φ⁻¹(0.975)
    let wins = 0, losses = 0, ties = 0;
    for (const d of diffs) { if (d > 0) wins++; else if (d < 0) losses++; else ties++; }
    return {
      meanDiff: md, stdDiff: sd, se, t, df: n - 1,
      ciLo: md - z * se, ciHi: md + z * se,
      pNormal: 2 * (1 - normalCdf(Math.abs(t))),
      wins, losses, ties, n,
    };
  },
});

// —— 混淆矩阵内核：行=真实，列=预测 ——
function evalConfusion(yTrue, yPred, classes) {
  const Y = arr(yTrue), P = arr(yPred);
  if (Y.length !== P.length) throw err('E_SHAPE', `yTrue 长度 ${Y.length} 与 yPred 长度 ${P.length} 不符`);
  const labels = classes ? arr(classes) : sortedUnique([...Y, ...P]);
  const index = new Map(labels.map((l, i) => [l, i]));
  const C = labels.length;
  const matrix = Array.from({ length: C }, () => new Array(C).fill(0));
  for (let i = 0; i < Y.length; i++) {
    const r = index.get(Y[i]), c = index.get(P[i]);
    if (r === undefined) throw err('E_VALUE', `真实标签 ${Y[i]} 不在 classes 中`);
    if (c === undefined) throw err('E_VALUE', `预测标签 ${P[i]} 不在 classes 中`);
    matrix[r][c]++;
  }
  const rowSums = matrix.map(r => r.reduce((a, b) => a + b, 0));
  const colSums = labels.map((_, j) => matrix.reduce((a, r) => a + r[j], 0));
  return {
    labels, matrix, rowSums, colSums,
    total: rowSums.reduce((a, b) => a + b, 0),
    diagonal: labels.reduce((a, _, i) => a + matrix[i][i], 0),
  };
}

// 每类指标内核
function evalPerClass(yTrue, yPred, classes) {
  const conf = evalConfusion(yTrue, yPred, classes);
  const perClass = conf.labels.map((c, i) => {
    const tp = conf.matrix[i][i];
    const fp = conf.colSums[i] - tp;
    const fn = conf.rowSums[i] - tp;
    const support = conf.rowSums[i];
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    return { class: c, tp, fp, fn, support, precision, recall, f1 };
  });
  return { classes: conf.labels, perClass };
}

// D9-055 分类报告（逐类 + 宏/加权平均）
defineOp({
  id: nid(), domain: 9, name: 'train.eval.classificationReport', arity: 3,
  sig: '(yTrue, yPred, classes?) -> {classes, accuracy, support, perClass, macro, weighted}',
  doc: '分类报告：逐类 precision/recall/f1/support，加总 accuracy 与宏平均、按 support 加权平均。',
  ref: '分类报告惯例：Sokolova & Lapalme (2009) A Systematic Analysis of Performance Measures for Classification Tasks, IP&M 45(4)',
  inv: 'accuracy = Σtp/n；macro = 逐类指标等权平均；weighted = 按 support 加权平均',
  impl: (yTrue, yPred, classes = null) => {
    const { classes: cls, perClass } = evalPerClass(yTrue, yPred, classes);
    const total = perClass.reduce((a, r) => a + r.support, 0);
    if (total === 0) throw err('E_VALUE', '空样本无法生成报告');
    const acc = perClass.reduce((a, r) => a + r.tp, 0) / total;
    const macro = {
      precision: meanOf(perClass.map(r => r.precision)),
      recall: meanOf(perClass.map(r => r.recall)),
      f1: meanOf(perClass.map(r => r.f1)),
    };
    const weighted = {
      precision: perClass.reduce((a, r) => a + r.support * r.precision, 0) / total,
      recall: perClass.reduce((a, r) => a + r.support * r.recall, 0) / total,
      f1: perClass.reduce((a, r) => a + r.support * r.f1, 0) / total,
    };
    return { classes: cls, accuracy: acc, support: total, perClass, macro, weighted };
  },
});

// D9-056 混淆矩阵报告
defineOp({
  id: nid(), domain: 9, name: 'train.eval.confusionReport', arity: 3,
  sig: '(yTrue, yPred, classes?) -> {labels, matrix, rowSums, colSums, total, diagonal}',
  doc: '混淆矩阵：matrix[i][j] = 真实为 labels[i] 且预测为 labels[j] 的计数；附行列和与对角线（判对数）。',
  ref: '混淆矩阵：Stehman (1997) Selecting and Interpreting Measures of Thematic Classification Accuracy, Remote Sens. Environ.',
  inv: '行和 = 各类真实计数；列和 = 各类预测计数；Σmatrix = n；diagonal = 判对数',
  impl: (yTrue, yPred, classes = null) => evalConfusion(yTrue, yPred, classes),
});

// ROC 曲线与 AUC 内核
function evalRoc(yTrue, scores, thresholds) {
  const y = arr(yTrue).map(v => (v === 1 || v === true ? 1 : 0));
  const s = arr(scores);
  if (y.length !== s.length) throw err('E_SHAPE', `yTrue 长度 ${y.length} 与 scores 长度 ${s.length} 不符`);
  const P = y.reduce((a, v) => a + v, 0);
  const N = y.length - P;
  if (P === 0 || N === 0) throw err('E_VALUE', 'ROC 需要同时包含正负样本');
  // AUC：Mann-Whitney U（并列取中秩）
  const order = range(s.length).sort((a, b) => s[a] - s[b]);
  const ranks = new Array(s.length);
  let i = 0;
  while (i < order.length) {
    let j = i;
    while (j + 1 < order.length && s[order[j + 1]] === s[order[i]]) j++;
    const avg = (i + j) / 2 + 1; // 1 起始秩
    for (let k = i; k <= j; k++) ranks[order[k]] = avg;
    i = j + 1;
  }
  let sumPos = 0;
  for (let k = 0; k < y.length; k++) if (y[k] === 1) sumPos += ranks[k];
  const auc = (sumPos - (P * (P + 1)) / 2) / (P * N);
  const ths = thresholds ? arr(thresholds).slice().sort((a, b) => b - a) : [...new Set(s)].sort((a, b) => b - a);
  const tpr = [], fpr = [], points = [];
  let bestJ = -Infinity, bestTh = null;
  for (const t of ths) {
    let tp = 0, fp = 0;
    for (let k = 0; k < s.length; k++) if (s[k] >= t) { if (y[k] === 1) tp++; else fp++; }
    const tprV = tp / P, fprV = fp / N;
    tpr.push(tprV); fpr.push(fprV);
    points.push({ threshold: t, tp, fp, tpr: tprV, fpr: fprV });
    if (tprV - fprV > bestJ) { bestJ = tprV - fprV; bestTh = t; }
  }
  return { thresholds: ths, tpr, fpr, points, auc, youdenJ: bestJ, optimalThreshold: bestTh, positives: P, negatives: N };
}

// D9-057 ROC 分析
defineOp({
  id: nid(), domain: 9, name: 'train.eval.rocAnalysis', arity: 3,
  sig: '(yTrue:0/1, scores:number[], thresholds?) -> {thresholds, tpr, fpr, points, auc, youdenJ, optimalThreshold}',
  doc: 'ROC 曲线：按分数降序扫阈值得 (fpr, tpr) 点列；AUC 用 Mann-Whitney（含并列中秩）；另给 Youden J 最优阈值。',
  ref: 'Fawcett (2006) An Introduction to ROC Analysis, Pattern Recognition Letters 27(8); Hanley & McNeil (1982) Radiology 143(1)',
  inv: 'auc ∈ [0,1]；随机排序 AUC=0.5；完美排序 AUC=1；points 按阈值降序',
  impl: (yTrue, scores, thresholds = null) => evalRoc(yTrue, scores, thresholds),
});

// PR 曲线与 AP 内核
function evalPr(yTrue, scores) {
  const y = arr(yTrue).map(v => (v === 1 || v === true ? 1 : 0));
  const s = arr(scores);
  if (y.length !== s.length) throw err('E_SHAPE', `yTrue 长度 ${y.length} 与 scores 长度 ${s.length} 不符`);
  const P = y.reduce((a, v) => a + v, 0);
  if (P === 0) throw err('E_VALUE', 'PR 曲线需要至少一个正样本');
  const ths = [...new Set(s)].sort((a, b) => b - a);
  const precision = [], recall = [], points = [];
  let ap = 0, prevR = 0, bestF1 = 0, bestTh = null;
  for (const t of ths) {
    let tp = 0, fp = 0;
    for (let k = 0; k < s.length; k++) if (s[k] >= t) { if (y[k] === 1) tp++; else fp++; }
    const pr = tp + fp > 0 ? tp / (tp + fp) : 0;
    const rc = tp / P;
    precision.push(pr); recall.push(rc);
    points.push({ threshold: t, tp, fp, precision: pr, recall: rc });
    if (rc > prevR) { ap += (rc - prevR) * pr; prevR = rc; }
    const f1 = pr + rc > 0 ? (2 * pr * rc) / (pr + rc) : 0;
    if (f1 > bestF1) { bestF1 = f1; bestTh = t; }
  }
  return { thresholds: ths, precision, recall, points, averagePrecision: ap, bestF1, bestF1Threshold: bestTh };
}

// D9-058 PR 分析
defineOp({
  id: nid(), domain: 9, name: 'train.eval.prAnalysis', arity: 2,
  sig: '(yTrue:0/1, scores:number[]) -> {thresholds, precision, recall, points, averagePrecision, bestF1}',
  doc: '精确率-召回率曲线：按分数降序扫阈值；AP 用 recall 增量插值 Σ(Rₖ-Rₖ₋₁)·Pₖ；另给最优 F1 阈值。',
  ref: 'Davis & Goadrich (2006) The Relationship Between Precision-Recall and ROC Curves, ICML',
  inv: 'recall 随阈值降低单调不减；perfect 排序 AP=1；bestF1 = maxₜ 2PR/(P+R)',
  impl: (yTrue, scores) => evalPr(yTrue, scores),
});

// D9-059 阈值扫描
defineOp({
  id: nid(), domain: 9, name: 'train.eval.thresholdSweep', arity: 3,
  sig: '(yTrue:0/1, scores:number[], thresholds:number[]) -> {threshold, tp, fp, tn, fn, precision, recall, f1, accuracy}[]',
  doc: '对给定阈值集合逐项计算二分类指标（预测正类规则：score ≥ threshold）。',
  ref: '阈值扫描与工作点选择：Fawcett (2006) Pattern Recognition Letters 27(8)',
  inv: 'tp+fp+tn+fn = n；precision/recall/f1 与 classificationReport 中正类口径一致',
  impl: (yTrue, scores, thresholds) => {
    const y = arr(yTrue).map(v => (v === 1 || v === true ? 1 : 0));
    const s = arr(scores);
    if (y.length !== s.length) throw err('E_SHAPE', 'yTrue 与 scores 长度不符');
    const n = y.length;
    return arr(thresholds).map(t => {
      let tp = 0, fp = 0, tn = 0, fn = 0;
      for (let i = 0; i < n; i++) {
        const pred = s[i] >= t ? 1 : 0;
        if (pred === 1 && y[i] === 1) tp++;
        else if (pred === 1 && y[i] === 0) fp++;
        else if (pred === 0 && y[i] === 0) tn++;
        else fn++;
      }
      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
      return { threshold: t, tp, fp, tn, fn, precision, recall, f1, accuracy: (tp + tn) / n };
    });
  },
});

// D9-060 宏平均
defineOp({
  id: nid(), domain: 9, name: 'train.eval.macroAvg', arity: 1,
  sig: '(report|perClass:array) -> {precision, recall, f1, k}',
  doc: '宏平均：对逐类 precision/recall/f1 等权平均；接受 classificationReport 或 perClass 数组。',
  ref: 'Opitz & Burst (2019) Macro F1 and Macro F1+, arXiv:1911.03347; Sokolova & Lapalme (2009) IP&M 45(4)',
  inv: '输出 ∈ [0,1]；k = 类别数；与 classificationReport.macro 一致',
  impl: (input) => {
    const list = Array.isArray(input) ? input : input?.perClass;
    if (!Array.isArray(list) || list.length === 0) throw err('E_TYPE', '需要 perClass 数组或含 perClass 的报告');
    return {
      precision: meanOf(list.map(r => r.precision)),
      recall: meanOf(list.map(r => r.recall)),
      f1: meanOf(list.map(r => r.f1)),
      k: list.length,
    };
  },
});

// D9-061 微平均
defineOp({
  id: nid(), domain: 9, name: 'train.eval.microAvg', arity: 2,
  sig: '(yTrue, yPred) -> {precision, recall, f1, accuracy}',
  doc: '微平均：单标签多分类下等价于准确率（汇总所有类的 tp/fp/fn 后计算）。',
  ref: 'Sokolova & Lapalme (2009) A Systematic Analysis of Performance Measures for Classification Tasks, IP&M 45(4)',
  inv: 'precision = recall = f1 = accuracy = Σtp/n',
  impl: (yTrue, yPred) => {
    const conf = evalConfusion(yTrue, yPred, null);
    if (conf.total === 0) throw err('E_VALUE', '空样本');
    const acc = conf.diagonal / conf.total;
    return { precision: acc, recall: acc, f1: acc, accuracy: acc };
  },
});

// D9-062 Cohen's kappa
defineOp({
  id: nid(), domain: 9, name: 'train.eval.cohenKappa', arity: 3,
  sig: '(yTrue, yPred, k?:number|array) -> number',
  doc: 'Cohen κ：扣除随机一致后的吻合度 (po-pe)/(1-pe)，pe 由行列边缘分布乘积给出。',
  ref: 'Cohen (1960) A Coefficient of Agreement for Nominal Scales, Educational and Psychological Measurement 20(1)',
  inv: 'κ ∈ [-1,1]；完全一致 κ=1；pe=1 时定义 κ=0',
  impl: (yTrue, yPred, k = null) => {
    const conf = evalConfusion(yTrue, yPred, k);
    const n = conf.total;
    if (n === 0) throw err('E_VALUE', '空样本');
    const po = conf.diagonal / n;
    let pe = 0;
    for (let i = 0; i < conf.labels.length; i++) pe += (conf.rowSums[i] / n) * (conf.colSums[i] / n);
    return pe >= 1 ? 0 : (po - pe) / (1 - pe);
  },
});

// D9-063 Matthews 相关系数（二分类/多类统一）
defineOp({
  id: nid(), domain: 9, name: 'train.eval.mcc', arity: 2,
  sig: '(yTrue, yPred) -> number',
  doc: 'MCC：二分类用 (tp·tn-fp·fn)/√(...)；多类用 Gorodkin 公式 (c·s-Σpₖtₖ)/√((s²-Σpₖ²)(s²-Σtₖ²))；分母为 0 时返回 0。',
  ref: 'Matthews (1975) Biochimie 62(6); Gorodkin (2004) Correntropy…/K-correlation for multi-class, Pattern Recognition Letters',
  inv: 'MCC ∈ [-1,1]；完美预测 1；随机 0；与准确率/ F1 平衡性互补',
  impl: (yTrue, yPred) => {
    const Y = arr(yTrue).map(v => (v === true ? 1 : v === false ? 0 : v));
    const P = arr(yPred).map(v => (v === true ? 1 : v === false ? 0 : v));
    const labels = sortedUnique([...Y, ...P]);
    const conf = evalConfusion(Y, P, labels);
    const isBinary = labels.length === 2 && labels.every(l => l === 0 || l === 1 || l === -1);
    if (isBinary) {
      const pi = labels.indexOf(1);
      const ni = pi === 0 ? 1 : 0;
      const tp = conf.matrix[pi][pi], tn = conf.matrix[ni][ni];
      const fp = conf.matrix[ni][pi], fn = conf.matrix[pi][ni];
      const den = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
      return den === 0 ? 0 : (tp * tn - fp * fn) / den;
    }
    const n = conf.total;
    let c = 0, sumPt = 0, sumP2 = 0, sumT2 = 0;
    for (let i = 0; i < labels.length; i++) {
      c += conf.matrix[i][i];
      sumPt += conf.colSums[i] * conf.rowSums[i];
      sumP2 += conf.colSums[i] ** 2;
      sumT2 += conf.rowSums[i] ** 2;
    }
    const den = Math.sqrt((n * n - sumP2) * (n * n - sumT2));
    return den === 0 ? 0 : (c * n - sumPt) / den;
  },
});

// D9-064 平衡准确率
defineOp({
  id: nid(), domain: 9, name: 'train.eval.balancedAcc', arity: 2,
  sig: '(yTrue, yPred) -> number',
  doc: '平衡准确率：各类 recall 的等权平均（对类别不均衡比原始准确率更稳健）。',
  ref: 'Brodersen et al. (2010) The Balanced Accuracy and Its Posterior Distribution, ICPR',
  inv: '取 yTrue 中出现过的类别计算 recall 平均；二类平衡时等于 (TPR+TNR)/2',
  impl: (yTrue, yPred) => {
    const { perClass } = evalPerClass(yTrue, yPred, sortedUnique(arr(yTrue)));
    return meanOf(perClass.map(r => r.recall));
  },
});

// D9-065 对数损失
defineOp({
  id: nid(), domain: 9, name: 'train.eval.logloss', arity: 3,
  sig: '(yTrue, proba:Tensor[n,C]|Tensor[n]|array, eps?:number) -> number',
  doc: '交叉熵/logloss：2D proba 取真实类概率列，1D proba 视为正类概率；概率截断到 [eps,1-eps]。',
  ref: 'Brier (1950) Monthly Weather Review 78(1); Murphy (1972) / proper scoring rules 综述',
  inv: 'logloss ≥ 0；完美概率 → 0；eps 防止 log 0',
  impl: (yTrue, proba, eps = 1e-15) => {
    const Y = arr(yTrue);
    if (!(eps > 0 && eps < 0.5)) throw err('E_VALUE', `eps 必须 ∈ (0,0.5)，收到 ${eps}`);
    if (proba instanceof Tensor && proba.ndim === 2) {
      const [n, C] = proba.shape;
      if (n !== Y.length) throw err('E_SHAPE', `proba 行数 ${n} 与 yTrue 长度 ${Y.length} 不符`);
      let s = 0;
      for (let i = 0; i < n; i++) {
        const k = Y[i];
        if (!Number.isInteger(k) || k < 0 || k >= C) throw err('E_VALUE', `标签 ${k} 越界（C=${C}）`);
        const p = Math.min(Math.max(proba.get2(i, k), eps), 1 - eps);
        s -= Math.log(p);
      }
      return s / n;
    }
    const P = arr(proba);
    if (P.length !== Y.length) throw err('E_SHAPE', 'proba 与 yTrue 长度不符');
    let s = 0;
    for (let i = 0; i < Y.length; i++) {
      const p = Math.min(Math.max(P[i], eps), 1 - eps);
      s -= Y[i] === 1 ? Math.log(p) : Math.log(1 - p);
    }
    return s / Y.length;
  },
});

// D9-066 Brier 分数
defineOp({
  id: nid(), domain: 9, name: 'train.eval.brier', arity: 2,
  sig: '(yTrue:0/1, proba:Tensor[n]|Tensor[n,2]|array) -> number',
  doc: 'Brier 分数：正类概率预测的均方误差 mean((p-y)²)；2D 入参取第 2 列。',
  ref: 'Brier (1950) Verification of Forecasts Expressed in Terms of Probability, Monthly Weather Review 78(1)',
  inv: 'brier ∈ [0,1]；完美概率 → 0；与 logloss 同为 proper scoring rule',
  impl: (yTrue, proba) => {
    const Y = arr(yTrue).map(v => (v === 1 || v === true ? 1 : 0));
    const P = proba instanceof Tensor && proba.ndim === 2 ? Array.from(proba.col(1)) : arr(proba);
    if (P.length !== Y.length) throw err('E_SHAPE', 'proba 与 yTrue 长度不符');
    let s = 0;
    for (let i = 0; i < Y.length; i++) s += (P[i] - Y[i]) ** 2;
    return s / Y.length;
  },
});

// D9-067 F-beta
defineOp({
  id: nid(), domain: 9, name: 'train.eval.fbeta', arity: 4,
  sig: '(yTrue, yPred, beta?:number, opts?:{average, posLabel, classes}) -> number',
  doc: 'F-beta（β=1 即 F1）：默认二分类（posLabel=1）；opts.average=\'macro\' 时对逐类 F-beta 等权平均。',
  ref: 'Van Rijsbergen (1979) Information Retrieval, 2nd ed.（F 度量）; Sokolova & Lapalme (2009) IP&M',
  inv: 'Fβ = (1+β²)tp / ((1+β²)tp + β²fn + fp)；分母 0 返回 0',
  impl: (yTrue, yPred, beta = 1, opts = {}) => {
    if (!(typeof beta === 'number' && beta > 0)) throw err('E_VALUE', `beta 必须为正数，收到 ${beta}`);
    const b2 = beta * beta;
    if (opts?.average === 'macro') {
      const { perClass } = evalPerClass(yTrue, yPred, opts?.classes ?? null);
      return meanOf(perClass.map(r => ((1 + b2) * r.tp) / ((1 + b2) * r.tp + b2 * r.fn + r.fp) || 0));
    }
    const pos = opts?.posLabel ?? 1;
    const Y = arr(yTrue), P = arr(yPred);
    if (Y.length !== P.length) throw err('E_SHAPE', 'yTrue 与 yPred 长度不符');
    let tp = 0, fp = 0, fn = 0;
    for (let i = 0; i < Y.length; i++) {
      if (Y[i] === pos && P[i] === pos) tp++;
      else if (P[i] === pos) fp++;
      else if (Y[i] === pos) fn++;
    }
    const den = (1 + b2) * tp + b2 * fn + fp;
    return den === 0 ? 0 : ((1 + b2) * tp) / den;
  },
});

// D9-068 Top-k 准确率
defineOp({
  id: nid(), domain: 9, name: 'train.eval.topKAcc', arity: 3,
  sig: '(yTrue, proba:Tensor[n,C]|nested, k:number) -> number',
  doc: 'Top-k 准确率：真实标签落在概率前 k 名即算对（ILSVRC 口径）。',
  ref: 'Russakovsky et al. (2015) ImageNet Large Scale Visual Recognition Challenge, IJCV 115(3)',
  inv: 'k=1 退化为准确率；k ≥ C 时为 1',
  impl: (yTrue, proba, k) => {
    const Y = arr(yTrue);
    const Pt = proba instanceof Tensor ? proba : Tensor.fromNested(proba);
    if (Pt.ndim !== 2) throw err('E_SHAPE', 'proba 必须为 2D [n,C]');
    const [n, C] = Pt.shape;
    if (n !== Y.length) throw err('E_SHAPE', 'proba 行数与 yTrue 长度不符');
    if (!Number.isInteger(k) || k < 1 || k > C) throw err('E_VALUE', `k 必须 ∈ [1,C]，收到 k=${k}, C=${C}`);
    let correct = 0;
    for (let i = 0; i < n; i++) {
      const row = Pt.row(i);
      const top = range(C).sort((a, b) => row[b] - row[a]).slice(0, k);
      if (top.includes(Y[i])) correct++;
    }
    return correct / n;
  },
});

// D9-069 逐类指标
defineOp({
  id: nid(), domain: 9, name: 'train.eval.perClass', arity: 3,
  sig: '(yTrue, yPred, classes?) -> {classes, perClass:[{class, tp, fp, fn, support, precision, recall, f1}]}',
  doc: '逐类统计 tp/fp/fn/support 与 precision/recall/f1；classificationReport 与 macroAvg 的基础件。',
  ref: 'Sokolova & Lapalme (2009) A Systematic Analysis of Performance Measures for Classification Tasks, IP&M 45(4)',
  inv: '各类 tp+fn = support；Σtp = 判对数',
  impl: (yTrue, yPred, classes = null) => evalPerClass(yTrue, yPred, classes),
});

// D9-070 多数类基线
defineOp({
  id: nid(), domain: 9, name: 'train.eval.majorityBaseline', arity: 1,
  sig: '(yTrue) -> {label, count, proportion, accuracy}',
  doc: '多数类基线：永远预测最高频类别时的准确率（零模型上界参照）。',
  ref: 'Devroye, Györfi & Lugosi (1996) A Probabilistic Theory of Pattern Recognition（零规则基线）',
  inv: 'accuracy = max_c count(c)/n；proportion 同值',
  impl: (yTrue) => {
    const Y = arr(yTrue);
    if (Y.length === 0) throw err('E_VALUE', '空样本');
    const labels = sortedUnique(Y);
    let best = labels[0], bestN = 0;
    for (const c of labels) {
      let n = 0;
      for (const v of Y) if (v === c) n++;
      if (n > bestN) { bestN = n; best = c; }
    }
    return { label: best, count: bestN, proportion: bestN / Y.length, accuracy: bestN / Y.length };
  },
});

// D9-071 分层随机基线
defineOp({
  id: nid(), domain: 9, name: 'train.eval.stratifiedBaseline', arity: 2,
  sig: '(yTrue, seed?:number) -> {accuracy, expected, labels, prior, seed}',
  doc: '分层基线：按经验类别先验随机预测的模拟准确率，并给出解析期望 Σp_c²；固定种子可复现。',
  ref: 'Devroye, Györfi & Lugosi (1996) A Probabilistic Theory of Pattern Recognition',
  inv: '模拟准确率随种子确定；expected = Σ_c (count(c)/n)²；prior 和为 1',
  impl: (yTrue, seed = 20261014) => {
    const Y = arr(yTrue);
    if (Y.length === 0) throw err('E_VALUE', '空样本');
    const labels = sortedUnique(Y);
    const counts = labels.map(c => Y.reduce((a, v) => a + (v === c ? 1 : 0), 0));
    const prior = counts.map(c => c / Y.length);
    const rng = createRng(seed);
    const cum = [];
    let acc = 0;
    for (const p of prior) { acc += p; cum.push(acc); }
    let correct = 0;
    for (const v of Y) {
      const u = rng.next();
      let pred = labels[labels.length - 1];
      for (let i = 0; i < labels.length; i++) { if (u < cum[i]) { pred = labels[i]; break; } }
      if (pred === v) correct++;
    }
    return { accuracy: correct / Y.length, expected: prior.reduce((a, p) => a + p * p, 0), labels, prior, seed };
  },
});

// D9-072 结果表（Markdown）
defineOp({
  id: nid(), domain: 9, name: 'train.eval.resultTable', arity: 2,
  sig: '(rows:object[], cols?:string[]) -> string',
  doc: '把记录数组渲染为 Markdown 表；cols 缺省取键的有序并集；数字定宽格式化。',
  ref: 'Demšar (2006) Statistical Comparisons of Classifiers over Multiple Data Sets, JMLR',
  inv: '表头列 = cols；数据行数 = rows.length',
  impl: (rows, cols = null) => {
    if (!Array.isArray(rows)) throw err('E_TYPE', 'rows 必须为对象数组');
    return tableMarkdown(rows, cols);
  },
});

// D9-073 运行对比表（name 列优先）
defineOp({
  id: nid(), domain: 9, name: 'train.eval.compareTable', arity: 2,
  sig: '(runs:object[], cols?:string[]) -> string',
  doc: '多运行对比 Markdown 表：若存在 name 列则置于首位，便于横向比较指标。',
  ref: 'Demšar (2006) Statistical Comparisons of Classifiers over Multiple Data Sets, JMLR',
  inv: '含 name 时首列为 name；行数 = runs.length',
  impl: (runs, cols = null) => {
    const list = Array.isArray(runs) ? runs : [runs];
    let cs = cols && cols.length ? cols : orderedKeys(list);
    if (cs.includes('name')) cs = ['name', ...cs.filter(c => c !== 'name')];
    return tableMarkdown(list, cs);
  },
});

// D9-074 运行排名
defineOp({
  id: nid(), domain: 9, name: 'train.eval.rankRuns', arity: 3,
  sig: "(runs:object[], metric:string|fn, mode?:'max'|'min') -> array",
  doc: '按指定指标对运行排序并标注 rank（1 起）；mode=\'max\' 降序（默认），\'min\' 升序；不修改入参数组。',
  ref: 'Demšar (2006) Statistical Comparisons of Classifiers over Multiple Data Sets, JMLR',
  inv: '返回长度 = runs.length；rank 连续从 1 起；原数组不被修改',
  impl: (runs, metric, mode = 'max') => {
    const list = Array.isArray(runs) ? runs.slice() : [runs];
    if (mode !== 'max' && mode !== 'min') throw err('E_VALUE', `mode 必须为 'max'|'min'，收到 ${mode}`);
    const get = typeof metric === 'function' ? metric : (r) => r?.[metric];
    if (list.some(r => typeof get(r) !== 'number' || !Number.isFinite(get(r)))) throw err('E_TYPE', 'metric 取值必须为有限数');
    list.sort((a, b) => (mode === 'max' ? get(b) - get(a) : get(a) - get(b)));
    return list.map((r, i) => ({ ...r, rank: i + 1 }));
  },
});

// ============================================================================
// 四、可复现性（D9-075 ~ D9-094）
// ============================================================================

// 环境信息（稳定字段，不含时间）
function envInfoImpl() {
  const node = typeof process !== 'undefined' && process.versions ? process.versions.node : 'unknown';
  const platform = typeof process !== 'undefined' && process.platform ? process.platform : 'unknown';
  const arch = typeof process !== 'undefined' && process.arch ? process.arch : 'unknown';
  return { node, platform, arch, kernel: KERNEL_VERSION, catalog: CATALOG_VERSION, domains: Object.keys(DOMAINS).length, opsInDomain9: opsByDomain(9).length };
}

// manifest 内核（不含时间戳 → 同配置同种子同哈希）
function manifestImpl(config, opts = {}) {
  const env = envInfoImpl();
  return {
    schema: 'd9.manifest/1',
    config: deepClone(config ?? {}),
    seed: opts?.seed ?? null,
    versions: { kernel: env.kernel, catalog: env.catalog },
    env: { node: env.node, platform: env.platform, arch: env.arch },
    dataHash: opts?.dataHash ?? null,
    extra: opts?.extra == null ? null : deepClone(opts.extra),
    timestamp: opts?.timestamp ?? null,
  };
}
const manifestHashImpl = (m) => fnv1a(`m|${canonicalJson(m)}`);

// 最优 epoch 内核
function bestEpochImpl(history, metric, mode) {
  const h = Array.isArray(history) ? history : [];
  if (h.length === 0) throw err('E_VALUE', 'history 为空');
  const m = mode ?? (/loss|error|err|mse|mae/i.test(String(metric)) ? 'min' : 'max');
  let best = null;
  for (const r of h) {
    const v = r?.[metric];
    if (typeof v !== 'number' || !Number.isFinite(v)) continue;
    if (best === null) { best = r; continue; }
    const bv = best[metric];
    if (m === 'max' ? v > bv : v < bv) best = r;
  }
  if (best === null) throw err('E_VALUE', `history 中无有效指标 ${metric}`);
  return { epoch: best.epoch, metric, mode: m, value: best[metric], record: best };
}

// run 摘要内核
function runSummaryImpl(run) {
  const history = Array.isArray(run?.history) ? run.history : [];
  let metrics = {};
  if (run?.metrics && Object.keys(run.metrics).length > 0) metrics = { ...run.metrics };
  else if (history.length > 0) {
    const last = history[history.length - 1];
    for (const k of Object.keys(last)) if (k !== 'epoch' && k !== 'lr') metrics[k] = last[k];
  }
  let best = null;
  if (history.length > 0) {
    try { best = bestEpochImpl(history, run?.bestMetric ?? 'valLoss', run?.bestMode); } catch { best = null; }
  }
  return {
    id: run?.id ?? null,
    seed: run?.seed ?? null,
    epochs: history.length > 0 ? history.length : null,
    configHash: run?.config ? hashParamsImpl(run.config) : null,
    metrics,
    best,
  };
}
const hashParamsImpl = (p) => fnv1a(`p|${canonicalJson(p)}`);

// 参数结构 → 纯 JSON 结构（Tensor → {shape, data}）
function toPlain(v) {
  if (v instanceof Tensor) return { shape: v.shape.slice(), data: Array.from(v.data) };
  if (Array.isArray(v)) return v.map(toPlain);
  if (isPlainObj(v)) { const o = {}; for (const k of Object.keys(v)) o[k] = toPlain(v[k]); return o; }
  return v;
}

// 模块级指标存储（recordMetric / metricSeries / resetState）
const metricStore = new Map();
let metricSeq = 0;

// D9-075 统一播种：主 RNG + 命名子流
defineOp({
  id: nid(), domain: 9, name: 'train.repro.seedAll', arity: 1,
  sig: '(seed:number|string) -> {seed, rng, stream(name) -> rng}',
  doc: '统一播种入口：字符串种子先经 FNV-1a 派生为 32bit；返回主 rng 与按名字派生的确定性子流（数据/初始化/增广各用一流）。',
  ref: 'PCG 随机流与命名派生：O\'Neill (2014) PCG RNG; FNV-1a: Fowler/Noll/Vo (1991)',
  inv: '同种子两次调用各流序列一致；不同流名序列不同；字符串种子走 seedFrom',
  impl: (seed) => {
    const s = typeof seed === 'string' ? seedFrom(seed) : seed;
    if (!Number.isInteger(s)) throw err('E_TYPE', `seed 必须为整数或字符串，收到 ${typeof seed}`);
    return {
      seed: s >>> 0,
      rng: createRng(s),
      stream(name) { return createRng(seedFrom(`${s >>> 0}|${name}`)); },
    };
  },
});

// D9-076 环境指纹（稳定摘要）
defineOp({
  id: nid(), domain: 9, name: 'train.repro.fingerprint', arity: 1,
  sig: '(env?:{seed, dataHash, config}) -> {hash, parts}',
  doc: '环境指纹：内核/目录版本 + node/平台/架构 + 可选种子/数据哈希/配置哈希的稳定摘要；不含时间。',
  ref: '可复现构建与确定性摘要：Debian Reproducible Builds 项目（2013–）；Perez-Riverol et al. (2016) Ten Simple Rules for Taking Advantage of Git and GitHub, PLOS CB',
  inv: '同入参同指纹；任一入参变化（如 seed）指纹变；parts 可读',
  impl: (env = {}) => {
    const e = env ?? {};
    const base = envInfoImpl();
    const parts = {
      kernel: base.kernel,
      catalog: base.catalog,
      node: base.node,
      platform: base.platform,
      arch: base.arch,
      domains: base.domains,
      seed: e.seed ?? null,
      dataHash: e.dataHash ?? null,
      configHash: e.config ? hashParamsImpl(e.config) : null,
    };
    return { hash: fnv1a(`f|${canonicalJson(parts)}`), parts };
  },
});

// D9-077 实验清单 manifest
defineOp({
  id: nid(), domain: 9, name: 'train.repro.manifest', arity: 2,
  sig: '(config, opts?:{seed, dataHash, extra, timestamp}) -> manifest',
  doc: '生成实验 manifest：配置 + 种子 + 内核/目录版本 + 环境 + 数据哈希；不含时间戳（除非显式传入）→ 可稳定哈希。',
  ref: 'Perez-Riverol et al. (2016) Ten Simple Rules for Taking Advantage of Git and GitHub, PLOS Comput. Biol. 12(7)',
  inv: '同 (config, seed, dataHash) 两次 manifest 深度相等；manifestHash 相同',
  impl: (config, opts = {}) => manifestImpl(config, opts),
});

// D9-078 manifest 哈希
defineOp({
  id: nid(), domain: 9, name: 'train.repro.manifestHash', arity: 1,
  sig: '(manifest) -> hex8',
  doc: '对 manifest 做规范 JSON + FNV-1a 哈希；改种子/配置/数据哈希 → 哈希变。',
  ref: 'FNV-1a：Fowler/Noll/Vo (1991)；Perez-Riverol et al. (2016) PLOS CB',
  inv: '同 manifest 同哈希；键序不影响结果（规范序列化排序键）',
  impl: (manifest) => {
    if (!manifest || typeof manifest !== 'object') throw err('E_TYPE', 'manifest 必须为对象');
    return manifestHashImpl(manifest);
  },
});

// D9-079 逐位确定性检查
defineOp({
  id: nid(), domain: 9, name: 'train.repro.determinismCheck', arity: 3,
  sig: '(fn:fn, args:array, opts?:{runs}) -> {deterministic, runs, canonical, mismatchAt}',
  doc: '以深拷贝入参重复执行 fn runs 次（默认 2），对返回做规范序列化逐位比较；用于验证训练/预处理确定性。',
  ref: 'Hoare (1969) An Axiomatic Basis for Computer Programming, CACM 12(10)（断言式不变量）',
  inv: 'deterministic=true 时所有 run 的规范序列化完全相同；mismatchAt 指出首个不一致 run',
  impl: (fn, args, opts = {}) => {
    if (typeof fn !== 'function') throw err('E_TYPE', 'fn 必须为函数');
    const runs = opts?.runs ?? 2;
    if (!Number.isInteger(runs) || runs < 2) throw err('E_VALUE', `runs 必须 ≥2 的整数，收到 ${runs}`);
    const a = Array.isArray(args) ? args : [args];
    let first = null;
    for (let i = 0; i < runs; i++) {
      const c = canonicalJson(fn(...deepClone(a)));
      if (first === null) first = c;
      else if (c !== first) return { deterministic: false, runs, canonical: first, mismatchAt: i + 1 };
    }
    return { deterministic: true, runs, canonical: first, mismatchAt: null };
  },
});

// D9-080 参数快照
defineOp({
  id: nid(), domain: 9, name: 'train.repro.snapshotParams', arity: 1,
  sig: '(params) -> snapshot',
  doc: '参数结构深拷贝快照（Tensor/数组/纯对象/标量），与源完全隔离。',
  ref: 'Pedregosa et al. (2011) Scikit-learn, JMLR 12（模型持久化）',
  inv: 'snapshot 与 params 逐位相等；互不影响',
  impl: (params) => deepClone(params),
});

// D9-081 参数恢复
defineOp({
  id: nid(), domain: 9, name: 'train.repro.restoreParams', arity: 1,
  sig: '(snapshot) -> params',
  doc: '从快照深拷贝恢复参数；与 snapshot 隔离。',
  ref: 'Pedregosa et al. (2011) Scikit-learn, JMLR 12',
  inv: 'restore(snapshot(p)) 与 p 逐位相等',
  impl: (snapshot) => deepClone(snapshot),
});

// D9-082 环境信息
defineOp({
  id: nid(), domain: 9, name: 'train.repro.envInfo', arity: 0,
  sig: '() -> {node, platform, arch, kernel, catalog, domains, opsInDomain9}',
  doc: '环境信息：node 版本、平台、架构、内核与目录版本、域数、域9 指令数（稳定字段，不含时间）。',
  ref: 'Perez-Riverol et al. (2016) Ten Simple Rules for Taking Advantage of Git and GitHub, PLOS CB',
  inv: '同一进程内多次调用结果相同',
  impl: () => envInfoImpl(),
});

// D9-083 实验日志追加
defineOp({
  id: nid(), domain: 9, name: 'train.repro.logAppend', arity: 2,
  sig: '(log:array, entry:object) -> array',
  doc: '向实验日志追加带 seq 序号的记录；纯函数不修改入参。',
  ref: 'Kitzes et al. (2018) The Practice of Reproducible Research, UC Press',
  inv: '返回长度 = 入参+1；seq 连续（缺省 length+1）',
  impl: (log, entry) => {
    const l = Array.isArray(log) ? log.slice() : [];
    if (!entry || typeof entry !== 'object') throw err('E_TYPE', 'entry 必须为对象');
    const rec = { seq: l.length + 1, ...entry };
    if (entry.seq !== undefined) rec.seq = entry.seq;
    l.push(rec);
    return l;
  },
});

// D9-084 实验日志对比
defineOp({
  id: nid(), domain: 9, name: 'train.repro.experimentCompare', arity: 3,
  sig: '(logA:array, logB:array, key?:string) -> {onlyA, onlyB, common, diffs, identical}',
  doc: '按 key（默认 runId）对齐两份实验日志，报告独有条目与字段级差异（用于复现审计）。',
  ref: 'Kitzes et al. (2018) The Practice of Reproducible Research, UC Press',
  inv: 'onlyA/onlyB/common 构成键的全集划分；diffs 逐字段给出两侧值；identical = 三者皆空',
  impl: (logA, logB, key = 'runId') => {
    const toMap = (log) => {
      const m = new Map();
      for (const e of Array.isArray(log) ? log : []) {
        if (e && typeof e === 'object') m.set(String(e[key]), e);
      }
      return m;
    };
    const A = toMap(logA), B = toMap(logB);
    const onlyA = [...A.keys()].filter(k => !B.has(k));
    const onlyB = [...B.keys()].filter(k => !A.has(k));
    const common = [...A.keys()].filter(k => B.has(k));
    const diffs = [];
    for (const k of common) {
      const a = A.get(k), b = B.get(k);
      const fields = orderedKeys([a, b]).filter(f => f !== key);
      for (const f of fields) {
        if (canonicalJson(a[f]) !== canonicalJson(b[f])) diffs.push({ [key]: k, field: f, a: a[f], b: b[f] });
      }
    }
    return { onlyA, onlyB, common, diffs, identical: onlyA.length === 0 && onlyB.length === 0 && diffs.length === 0 };
  },
});

// D9-085 表格转 CSV
defineOp({
  id: nid(), domain: 9, name: 'train.repro.tableToCsv', arity: 2,
  sig: '(rows:object[], cols?:string[]) -> string',
  doc: '记录数组 → CSV（含表头）；含逗号/引号/换行的值按 RFC 4180 双引号转义。',
  ref: 'CSV 规范：Shafranovich (2005) RFC 4180, IETF',
  inv: '首行为表头；行数 = rows.length+1；转义后 round-trip 不破坏列结构',
  impl: (rows, cols = null) => {
    const list = Array.isArray(rows) ? rows : [rows];
    const cs = cols && cols.length ? cols : orderedKeys(list);
    const esc = (v) => {
      const s = fmtCell(v);
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    return [cs.map(esc).join(','), ...list.map(r => cs.map(c => esc(r?.[c])).join(','))].join('\n');
  },
});

// D9-086 表格转 Markdown
defineOp({
  id: nid(), domain: 9, name: 'train.repro.tableToMarkdown', arity: 2,
  sig: '(rows:object[], cols?:string[]) -> string',
  doc: '记录数组 → Markdown 表（与 eval.resultTable 同渲染，供实验报告归档）。',
  ref: 'Demšar (2006) Statistical Comparisons of Classifiers over Multiple Data Sets, JMLR',
  inv: '表头 | ... | + 分隔行 + 数据行',
  impl: (rows, cols = null) => {
    if (!Array.isArray(rows)) throw err('E_TYPE', 'rows 必须为对象数组');
    return tableMarkdown(rows, cols);
  },
});

// D9-087 数据哈希（逐位）
defineOp({
  id: nid(), domain: 9, name: 'train.repro.hashData', arity: 1,
  sig: '(dataset) -> hex8',
  doc: '对数据集 X/y 的 Float64 字节做 FNV-1a（含 shape）；改动任一比特即变。',
  ref: 'FNV-1a：Fowler/Noll/Vo (1991)；数据版本控制：Perez-Riverol et al. (2016) PLOS CB',
  inv: '同数据同哈希；形状变化哈希变；不依赖十进制表示',
  impl: (ds) => {
    const { X, y } = dsOf(ds, 'hashData');
    return fnv1a(`d9d|${X.shape.join('x')}|${hashF64(X.data)}|${y.shape.join('x')}|${hashF64(y.data)}`);
  },
});

// D9-088 参数哈希
defineOp({
  id: nid(), domain: 9, name: 'train.repro.hashParams', arity: 1,
  sig: '(params) -> hex8',
  doc: '对参数结构（Tensor/数组/对象/标量）做规范序列化 + FNV-1a 哈希。',
  ref: 'FNV-1a：Fowler/Noll/Vo (1991)',
  inv: '同构同哈希；键序无关；Tensor 按 (shape,data) 编码',
  impl: (params) => hashParamsImpl(params),
});

// D9-089 复现性断言
defineOp({
  id: nid(), domain: 9, name: 'train.repro.assertReproducible', arity: 2,
  sig: '(a, b) -> true',
  doc: '断言两个对象（manifest/运行记录）在剔除 runId/timestamp/note 后规范相等；不等则抛 E_VALUE。',
  ref: 'Hoare (1969) An Axiomatic Basis for Computer Programming, CACM 12(10)',
  inv: '相等返回 true；不等抛 ForgeError(E_VALUE)',
  impl: (a, b) => {
    if (!a || !b || typeof a !== 'object' || typeof b !== 'object') throw err('E_TYPE', '需要两个对象');
    const strip = (m) => {
      const c = deepClone(m);
      delete c.runId; delete c.timestamp; delete c.note;
      return c;
    };
    const ha = fnv1a(`a|${canonicalJson(strip(a))}`);
    const hb = fnv1a(`a|${canonicalJson(strip(b))}`);
    if (ha !== hb) throw err('E_VALUE', `复现性断言失败: ${ha} != ${hb}`);
    return true;
  },
});

// D9-090 重置可复现性模块状态
defineOp({
  id: nid(), domain: 9, name: 'train.repro.resetState', arity: 0,
  sig: '() -> true',
  doc: '清空模块级指标存储与序号计数（测试隔离用）。',
  ref: '测试隔离与状态复位：Hoare (1969) CACM 12(10)',
  inv: 'reset 后 recordMetric/metricSeries 从空开始',
  impl: () => { metricStore.clear(); metricSeq = 0; return true; },
});

// D9-091 记录指标点
defineOp({
  id: nid(), domain: 9, name: 'train.repro.recordMetric', arity: 4,
  sig: '(runId:string, name:string, value:number, step?:number) -> record',
  doc: '向模块级存储记录一个 (runId, name, value, step) 指标点；step 缺省取全局序号。',
  ref: 'Kitzes et al. (2018) The Practice of Reproducible Research, UC Press',
  inv: '返回记录含 runId/name/value/step/seq；同 runId 可多次记录',
  impl: (runId, name, value, step = null) => {
    if (typeof value !== 'number' || !Number.isFinite(value)) throw err('E_TYPE', `value 必须为有限数，收到 ${value}`);
    const rid = String(runId), nm = String(name);
    const seq = ++metricSeq;
    const rec = { runId: rid, name: nm, value, step: step ?? seq, seq };
    if (!metricStore.has(rid)) metricStore.set(rid, []);
    metricStore.get(rid).push(rec);
    return { ...rec };
  },
});

// D9-092 读取指标序列
defineOp({
  id: nid(), domain: 9, name: 'train.repro.metricSeries', arity: 2,
  sig: '(runId:string, name:string) -> {step, value}[]',
  doc: '按 (step, seq) 排序返回某运行的某指标序列（用于画曲线/比较）。',
  ref: 'Kitzes et al. (2018) The Practice of Reproducible Research, UC Press',
  inv: '长度 = 该 runId 下该 name 的记录数；按 step 升序',
  impl: (runId, name) => {
    const rid = String(runId), nm = String(name);
    const list = (metricStore.get(rid) ?? []).filter(r => r.name === nm);
    list.sort((a, b) => (a.step - b.step) || (a.seq - b.seq));
    return list.map(r => ({ step: r.step, value: r.value }));
  },
});

// D9-093 最优 epoch
defineOp({
  id: nid(), domain: 9, name: 'train.repro.bestEpoch', arity: 3,
  sig: "(history:array, metric?:string, mode?:'min'|'max') -> {epoch, metric, mode, value, record}",
  doc: '在训练历史中找最优 epoch：mode 缺省按指标名推断（含 loss/error/mse → min，否则 max）。',
  ref: 'Prechelt (1998) Early Stopping — But When?（最佳点选取）',
  inv: 'mode=min 返回该指标最小值所在记录；缺省指标 valLoss',
  impl: (history, metric = 'valLoss', mode = null) => bestEpochImpl(history, metric, mode ?? undefined),
});

// D9-094 运行摘要
defineOp({
  id: nid(), domain: 9, name: 'train.repro.runSummary', arity: 1,
  sig: '(run:{id?, seed?, config?, history?, metrics?, bestMetric?, bestMode?}) -> summary',
  doc: '运行摘要：id/种子/轮数/配置哈希/指标集合/最优 epoch；供归档、对比与 replay 使用。',
  ref: 'Perez-Riverol et al. (2016) Ten Simple Rules for Taking Advantage of Git and GitHub, PLOS CB',
  inv: 'metrics 优先取 run.metrics，否则取末条历史记录（除 epoch/lr）；best 缺失指标时为 null',
  impl: (run) => {
    if (!run || typeof run !== 'object') throw err('E_TYPE', 'run 必须为对象');
    return runSummaryImpl(run);
  },
});

// ============================================================================
// 五、组合与工具（D9-095 ~ D9-102）
// ============================================================================

// D9-095 先划分后标准化（防数据泄漏）
defineOp({
  id: nid(), domain: 9, name: 'train.pipe.standardizeThenSplit', arity: 3,
  sig: '(dataset, testFrac:number, seed:number) -> {scaler, train, test}',
  doc: '先按种子划分，再只在训练集上拟合标准化器并应用于两侧，杜绝测试集统计量泄漏。',
  ref: '数据泄漏防范：Kohavi (1995) A Study of Cross-Validation and Bootstrap, IJCAI',
  inv: 'scaler 仅由 train 拟合；train 标准化后每列均值≈0、std≈1；test 用同一 scaler',
  impl: (ds, testFrac, seed) => {
    const { train, test } = trainTestSplitImpl(ds, testFrac, seed);
    const f = standardizerFitImpl(train);
    const scaler = { kind: 'standardizer', mean: f.mean, std: f.std };
    return {
      scaler,
      train: { ...train, X: standardizeWith(train.X, scaler) },
      test: { ...test, X: standardizeWith(test.X, scaler) },
    };
  },
});

// 内部：标准izerFit 的复用实现（避免经 exec 间接调用）
function standardizerFitImpl(ds) {
  const { X, n, d } = dsOf(ds);
  const mean = new Float64Array(d), std = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    let m = 0;
    for (let i = 0; i < n; i++) m += X.get2(i, j);
    m /= n;
    let s = 0;
    for (let i = 0; i < n; i++) s += (X.get2(i, j) - m) ** 2;
    mean[j] = m;
    std[j] = Math.max(Math.sqrt(s / n), 1e-12);
  }
  return { mean: new Tensor(mean, [d]), std: new Tensor(std, [d]) };
}
function standardizeWith(X, scaler) {
  const d = X.shape[1];
  const out = new Float64Array(X.size);
  for (let i = 0; i < X.shape[0]; i++) {
    for (let j = 0; j < d; j++) out[i * d + j] = (X.get2(i, j) - scaler.mean.data[j]) / scaler.std.data[j];
  }
  return new Tensor(out, X.shape.slice());
}

// D9-096 分类全指标报告
defineOp({
  id: nid(), domain: 9, name: 'train.eval.fullReport', arity: 3,
  sig: '(yTrue, yPred, opts?:{proba, classes}) -> report',
  doc: '分类全量报告：accuracy、balancedAcc、宏/加权平均、逐类指标、混淆矩阵、κ、MCC、多数类基线；给 proba 时追加 logloss/brier。',
  ref: 'Sokolova & Lapalme (2009) IP&M 45(4); Cohen (1960); Matthews (1975); Brier (1950)',
  inv: '字段与各单项指标函数口径一致；proba 缺省时不产出 logloss/brier',
  impl: (yTrue, yPred, opts = {}) => {
    const proba = opts?.proba ?? null;
    const classes = opts?.classes ?? null;
    const report = execClassificationReport(yTrue, yPred, classes);
    const out = {
      accuracy: report.accuracy,
      balancedAcc: execBalancedAcc(yTrue, yPred),
      macro: report.macro,
      weighted: report.weighted,
      perClass: report.perClass,
      confusion: evalConfusion(yTrue, yPred, classes),
      cohenKappa: execCohenKappa(yTrue, yPred, classes),
      mcc: execMcc(yTrue, yPred),
      majorityBaseline: execMajorityBaseline(yTrue),
    };
    if (proba) {
      out.logloss = execLogloss(yTrue, proba);
      const labels = sortedUnique(arr(yTrue));
      if (labels.length === 2 && labels.every(l => l === 0 || l === 1)) out.brier = execBrier(yTrue, proba);
    }
    return out;
  },
});

// —— 组合件用的小型执行包装（避免与已注册 op 的实现重复） ——
function execClassificationReport(yTrue, yPred, classes) {
  const { classes: cls, perClass } = evalPerClass(yTrue, yPred, classes);
  const total = perClass.reduce((a, r) => a + r.support, 0);
  const acc = perClass.reduce((a, r) => a + r.tp, 0) / total;
  return {
    classes: cls, accuracy: acc, support: total, perClass,
    macro: { precision: meanOf(perClass.map(r => r.precision)), recall: meanOf(perClass.map(r => r.recall)), f1: meanOf(perClass.map(r => r.f1)) },
    weighted: {
      precision: perClass.reduce((a, r) => a + r.support * r.precision, 0) / total,
      recall: perClass.reduce((a, r) => a + r.support * r.recall, 0) / total,
      f1: perClass.reduce((a, r) => a + r.support * r.f1, 0) / total,
    },
  };
}
function execBalancedAcc(yTrue, yPred) {
  const { perClass } = evalPerClass(yTrue, yPred, sortedUnique(arr(yTrue)));
  return meanOf(perClass.map(r => r.recall));
}
function execCohenKappa(yTrue, yPred, classes) {
  const conf = evalConfusion(yTrue, yPred, classes);
  const n = conf.total;
  const po = conf.diagonal / n;
  let pe = 0;
  for (let i = 0; i < conf.labels.length; i++) pe += (conf.rowSums[i] / n) * (conf.colSums[i] / n);
  return pe >= 1 ? 0 : (po - pe) / (1 - pe);
}
function execMcc(yTrue, yPred) {
  const Y = arr(yTrue).map(v => (v === true ? 1 : v === false ? 0 : v));
  const P = arr(yPred).map(v => (v === true ? 1 : v === false ? 0 : v));
  const labels = sortedUnique([...Y, ...P]);
  const conf = evalConfusion(Y, P, labels);
  const isBinary = labels.length === 2 && labels.every(l => l === 0 || l === 1 || l === -1);
  if (isBinary) {
    const pi = labels.indexOf(1);
    const ni = pi === 0 ? 1 : 0;
    const tp = conf.matrix[pi][pi], tn = conf.matrix[ni][ni];
    const fp = conf.matrix[ni][pi], fn = conf.matrix[pi][ni];
    const den = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
    return den === 0 ? 0 : (tp * tn - fp * fn) / den;
  }
  const n = conf.total;
  let c = 0, sumPt = 0, sumP2 = 0, sumT2 = 0;
  for (let i = 0; i < labels.length; i++) {
    c += conf.matrix[i][i];
    sumPt += conf.colSums[i] * conf.rowSums[i];
    sumP2 += conf.colSums[i] ** 2;
    sumT2 += conf.rowSums[i] ** 2;
  }
  const den = Math.sqrt((n * n - sumP2) * (n * n - sumT2));
  return den === 0 ? 0 : (c * n - sumPt) / den;
}
function execMajorityBaseline(yTrue) {
  const Y = arr(yTrue);
  const labels = sortedUnique(Y);
  let best = labels[0], bestN = 0;
  for (const c of labels) {
    const n = Y.reduce((a, v) => a + (v === c ? 1 : 0), 0);
    if (n > bestN) { bestN = n; best = c; }
  }
  return { label: best, count: bestN, proportion: bestN / Y.length, accuracy: bestN / Y.length };
}
function execLogloss(yTrue, proba, eps = 1e-15) {
  const Y = arr(yTrue);
  if (proba instanceof Tensor && proba.ndim === 2) {
    const [n, C] = proba.shape;
    let s = 0;
    for (let i = 0; i < n; i++) {
      const p = Math.min(Math.max(proba.get2(i, Y[i]), eps), 1 - eps);
      s -= Math.log(p);
    }
    return s / n;
  }
  const P = arr(proba);
  let s = 0;
  for (let i = 0; i < Y.length; i++) {
    const p = Math.min(Math.max(P[i], eps), 1 - eps);
    s -= Y[i] === 1 ? Math.log(p) : Math.log(1 - p);
  }
  return s / Y.length;
}
function execBrier(yTrue, proba) {
  const Y = arr(yTrue).map(v => (v === 1 || v === true ? 1 : 0));
  const P = proba instanceof Tensor && proba.ndim === 2 ? Array.from(proba.col(1)) : arr(proba);
  let s = 0;
  for (let i = 0; i < Y.length; i++) s += (P[i] - Y[i]) ** 2;
  return s / Y.length;
}

// D9-097 训练/验证/测试三向划分
defineOp({
  id: nid(), domain: 9, name: 'train.loop.trainValTest', arity: 4,
  sig: '(dataset, valFrac:number, testFrac:number, seed:number) -> {train, val, test}',
  doc: '三向划分：先切出 train，再把剩余部分按 valFrac:testFrac 比例切成 val/test；三部分 index 两两不相交。',
  ref: '三向 holdout 惯例：Hastie, Tibshirani & Friedman (2009) The Elements of Statistical Learning, §7',
  inv: '0 < valFrac + testFrac < 1；三部分并集 = [0,n) 且不相交',
  impl: (ds, valFrac, testFrac, seed) => {
    const d = dsOf(ds, 'trainValTest');
    const rest = valFrac + testFrac;
    if (!(rest > 0 && rest < 1)) throw err('E_VALUE', `valFrac+testFrac 必须 ∈ (0,1)，收到 ${rest}`);
    const first = trainTestSplitImpl(d, rest, seed);
    const valInRest = valFrac / rest;
    // second 的 testFrac 切出的是 val（占 rest 的 valFrac/rest），其余为 test
    const second = trainTestSplitImpl(first.test, valInRest, seed + 1);
    // 二次切分的 index 是相对于 first.test 的局部索引，映射回全局
    const remap = (s) => ({ ...s, index: s.index.map((li) => first.test.index[li]) });
    return { train: first.train, val: remap(second.test), test: remap(second.train) };
  },
});

// D9-098 嵌套交叉验证（外层评估 + 内层选模）
defineOp({
  id: nid(), domain: 9, name: 'train.eval.nestedCV', arity: 2,
  sig: '(dataset, opts:{kOuter, kInner, candidates, makeFitPredict, seed?, metric?, mode?}) -> {outerScores, mean, std, selected, mode, inner}',
  doc: '嵌套 CV：每个外层折内用内层 CV 在候选超参数中选优（mode 默认按 metric 推断：mse/mae→min，否则 max），再评外层折。',
  ref: 'Varma & Simon (2006) Bias in Error Estimation When Using Cross-Validation for Model Selection, BMC Bioinformatics 7:91; Stone (1974) JRSS-B',
  inv: 'outerScores.length === kOuter；selected.length === kOuter；内层每候选分数长度 === kInner',
  impl: (ds, opts = {}) => {
    const d = dsOf(ds, 'nestedCV');
    const kOuter = opts?.kOuter ?? 5, kInner = opts?.kInner ?? 3;
    const candidates = opts?.candidates;
    const make = opts?.makeFitPredict;
    if (!Array.isArray(candidates) || candidates.length === 0) throw err('E_VALUE', 'candidates 必须为非空数组');
    if (typeof make !== 'function') throw err('E_TYPE', 'makeFitPredict 必须为函数');
    const metric = resolveMetric(opts?.metric);
    const seed = opts?.seed ?? 20261014;
    const mode = opts?.mode ?? (opts?.metric === 'mse' || opts?.metric === 'mae' ? 'min' : 'max');
    const outerFolds = kFoldSplitImpl(d, kOuter, seed);
    const outerScores = [], selected = [], inner = [];
    for (const f of outerFolds) {
      const innerFolds = kFoldSplitImpl(f.train, kInner, seed + 7919);
      const candScores = candidates.map(cv => {
        const fp = make(cv);
        const ss = innerFolds.map(g => scoreOf(fp(g.train, g.val), g.val.y, metric));
        return { value: cv, scores: ss, mean: meanOf(ss) };
      });
      const best = mode === 'max'
        ? candScores.reduce((a, b) => (b.mean > a.mean ? b : a))
        : candScores.reduce((a, b) => (b.mean < a.mean ? b : a));
      selected.push(best.value);
      inner.push({ fold: f.fold, candidates: candScores, best: best.value });
      outerScores.push(scoreOf(make(best.value)(f.train, f.val), f.val.y, metric));
    }
    return { outerScores, mean: meanOf(outerScores), std: stdOf(outerScores), selected, mode, kOuter, kInner, inner };
  },
});

// D9-099 导出运行存档（JSON 可序列化）
defineOp({
  id: nid(), domain: 9, name: 'train.repro.exportRun', arity: 1,
  sig: '(run:{id?, config?, seed?, dataHash?, metrics?, history?, params?, fitPredict?, data?, metric?}) -> archive',
  doc: '导出完整实验存档：manifest + 配置 + 指标 + 历史 + 参数（转纯 JSON）；附 run.fitPredict 与 run.data 时追加 replay 重放数据。',
  ref: 'Perez-Riverol et al. (2016) Ten Simple Rules for Taking Advantage of Git and GitHub, PLOS CB',
  inv: '存档中无函数/时间戳（除非显式给）；manifestHash 可稳定重算；Tensors 转为 {shape, data}',
  impl: (run) => {
    if (!run || typeof run !== 'object') throw err('E_TYPE', 'run 必须为对象');
    const manifest = run.manifest ?? manifestImpl(run.config ?? {}, { seed: run.seed ?? null, dataHash: run.dataHash ?? null });
    const archive = {
      schema: 'd9.run/1',
      id: run.id ?? null,
      manifest,
      manifestHash: manifestHashImpl(manifest),
      config: toPlain(run.config ?? {}),
      seed: run.seed ?? null,
      history: toPlain(run.history ?? []),
      metrics: toPlain(run.metrics ?? {}),
      params: run.params ? toPlain(run.params) : null,
      summary: runSummaryImpl(run),
    };
    if (typeof run.fitPredict === 'function' && run.data) {
      const d = dsOf(run.data, 'exportRun.data');
      const pred = predOf(run.fitPredict(d, d));
      const m = resolveMetric(run.metric);
      archive.replay = {
        X: d.X.toArray(),
        y: d.y.toArray(),
        predHash: hashF64(pred.data),
        metric: m ? m(pred, d.y) : null,
      };
    }
    return archive;
  },
});

// D9-100 从存档重放摘要
defineOp({
  id: nid(), domain: 9, name: 'train.repro.replayRun', arity: 2,
  sig: '(archive, fitPredict?:fn) -> {ok, manifestHash, metrics, summary, checks}',
  doc: '从 exportRun 存档重放：校验 schema、复算并比对 manifestHash、复算摘要；给 fitPredict 且存档含 replay 数据时逐位复算预测哈希。',
  ref: 'Kitzes et al. (2018) The Practice of Reproducible Research, UC Press',
  inv: 'checks.schemaOk 恒 true（否则抛错）；manifestHashStable 为重算与存档一致；replayDeterministic 为预测哈希一致',
  impl: (archive, fitPredict = null) => {
    if (!archive || archive.schema !== 'd9.run/1') throw err('E_VALUE', 'archive 需为 d9.run/1 存档');
    const checks = { schemaOk: true };
    const h = manifestHashImpl(archive.manifest);
    checks.manifestHashStable = archive.manifestHash ? h === archive.manifestHash : null;
    const reSummary = runSummaryImpl({
      id: archive.id, seed: archive.seed, config: archive.config,
      metrics: archive.metrics, history: archive.history,
    });
    checks.summaryMatches = canonicalJson(reSummary) === canonicalJson(archive.summary);
    if (typeof fitPredict === 'function' && archive.replay) {
      const X = Tensor.fromNested(archive.replay.X);
      const y = Tensor.fromFlat(archive.replay.y, [archive.replay.y.length]);
      const pred = predOf(fitPredict({ X, y }, { X, y }));
      checks.replayDeterministic = hashF64(pred.data) === archive.replay.predHash;
    }
    return { ok: true, manifestHash: h, metrics: archive.metrics ?? {}, summary: archive.summary ?? null, checks };
  },
});

// D9-101 划分不相交断言
defineOp({
  id: nid(), domain: 9, name: 'train.util.assertSplitDisjoint', arity: null,
  sig: '(...splits:(number[]|dataset)) -> true',
  doc: '断言多个划分（索引数组或带 index 的数据集）两两不相交；有重叠抛 E_VALUE，否则返回 true。',
  ref: 'Hoare (1969) An Axiomatic Basis for Computer Programming, CACM 12(10)（不变量断言）',
  inv: '至少两个划分；任一重叠即抛错并指出索引',
  impl: (...splits) => {
    if (splits.length < 2) throw err('E_VALUE', '至少需要两个划分');
    const sets = splits.map((s, i) => {
      const idx = Array.isArray(s) ? s : (s && Array.isArray(s.index) ? s.index : null);
      if (!idx) throw err('E_TYPE', `划分 ${i} 需要为索引数组或带 index 的数据集`);
      return new Set(idx.map(Number));
    });
    for (let i = 0; i < sets.length; i++) {
      for (let j = i + 1; j < sets.length; j++) {
        for (const v of sets[i]) {
          if (sets[j].has(v)) throw err('E_VALUE', `划分 ${i} 与 ${j} 存在重叠索引 ${v}`);
        }
      }
    }
    return true;
  },
});

// D9-102 覆盖检查
defineOp({
  id: nid(), domain: 9, name: 'train.util.coverageCheck', arity: 2,
  sig: '(indexArrays:number[]|number[][], total:number) -> {ok, total, covered, missing, duplicates}',
  doc: '检查若干索引集合的并集恰好覆盖 [0,total) 一次：报告缺失与重复项。',
  ref: 'Hoare (1969) An Axiomatic Basis for Computer Programming, CACM 12(10)',
  inv: 'ok = missing 与 duplicates 皆空；covered = 出现过的索引数',
  impl: (indexArrays, total) => {
    if (!Number.isInteger(total) || total < 0) throw err('E_VALUE', `total 必须为非负整数，收到 ${total}`);
    let lists;
    if (Array.isArray(indexArrays) && indexArrays.length > 0 && indexArrays.every(Array.isArray)) lists = indexArrays;
    else if (Array.isArray(indexArrays)) lists = [indexArrays];
    else throw err('E_TYPE', 'indexArrays 需要为索引数组或索引数组的数组');
    const seen = new Map();
    for (const l of lists) {
      for (const i of l) {
        if (!Number.isInteger(i) || i < 0 || i >= total) throw err('E_VALUE', `索引 ${i} 越界（total=${total}）`);
        seen.set(i, (seen.get(i) || 0) + 1);
      }
    }
    const missing = [], duplicates = [];
    for (let i = 0; i < total; i++) {
      const c = seen.get(i) ?? 0;
      if (c === 0) missing.push(i);
      else if (c > 1) duplicates.push(i);
    }
    return { ok: missing.length === 0 && duplicates.length === 0, total, covered: seen.size, missing, duplicates };
  },
});


// 类别均衡权重（'balanced'）：w_c = n / (k · n_c)，使加权后每个类别的总权重相等。
// 类别不平衡时配合带权损失使用；与 train.eval.balancedAcc 构成「度量—加权」闭环。
defineOp({
  id: nid(), domain: 9, name: 'train.util.classWeights', arity: 2,
  sig: '(Tensor y[n], number k) -> Tensor[k]',
  doc: "类别均衡权重 w_c = n/(k·n_c)（scikit-learn class_weight='balanced' 口径）",
  ref: "scikit-learn class_weight='balanced'；King & Zeng, Logistic Regression in Rare Events Data, 2001",
  inv: 'Σ_c w_c·n_c = n；各类别加权总量 n/k 相等；未出现的类别权重为 0',
  impl: (y, k) => {
    if (!(y instanceof Tensor)) throw new ForgeError('E_TYPE', 'y: 期望 Tensor');
    expectNumber(k, 'k');
    if (!Number.isInteger(k) || k < 1) throw new ForgeError('E_VALUE', 'k 必须为正整数');
    const n = y.size;
    if (n === 0) throw new ForgeError('E_VALUE', 'y 不能为空');
    const cnt = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      const c = Math.round(y.data[i]);
      if (!Number.isFinite(c) || c < 0 || c >= k) throw new ForgeError('E_VALUE', `y 第 ${i} 项 ${y.data[i]} 越界（k=${k}）`);
      cnt[c]++;
    }
    const out = new Float64Array(k);
    for (let c = 0; c < k; c++) out[c] = cnt[c] > 0 ? n / (k * cnt[c]) : 0;
    return new Tensor(out, [k]);
  },
});

// ============================================================================
// 自检：域9 必须恰好 103 条，且 id 连续为 D9-001..D9-103（否则导入即抛错）
// ============================================================================
{
  const ops = opsByDomain(9);
  if (ops.length !== 103) {
    throw new ForgeError('E_VALUE', `域9 指令数必须为 103，当前 ${ops.length}`);
  }
  ops.forEach((op, i) => {
    const expect = `D9-${String(i + 1).padStart(3, '0')}`;
    if (op.id !== expect) {
      throw new ForgeError('E_VALUE', `域9 指令 id 不连续: 第 ${i + 1} 条为 ${op.id}（期望 ${expect}）`);
    }
  });
}
