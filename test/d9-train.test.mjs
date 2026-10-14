// 域9 训练与评估（TRAIN）测试：覆盖 103 条微指令的行为契约。
// 所有随机数据经 createRng(seed) 生成；可复现性断言以逐位相等为准。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { exec, getOp, hasOp, opsByDomain } from '../src/core/index.mjs';
import { Tensor, createRng } from '../src/core/index.mjs';
import '../src/ops/d9-train.mjs'; // 导入即注册，并触发 103 条自检
import { approx, maxAbsDiff, tensorClose, blobs, linData, randMat } from './helpers.mjs';

const run = (name, args) => exec(name, args);
const range = (n) => Array.from({ length: n }, (_, i) => i);

// —— 测试本地模型与闭式解（不依赖其他域文件） ——

// 线性回归模型（通用接口 {params, predict, lossGrad}）
function linearModel(d, seed = 1) {
  const rng = createRng(seed);
  const w0 = Tensor.fromFlat(Array.from({ length: d }, () => rng.normal() * 0.1), [d]);
  const predict = (params, X) => {
    const n = X.shape[0];
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = params.b;
      for (let j = 0; j < X.shape[1]; j++) s += X.get2(i, j) * params.w.at(j);
      out[i] = s;
    }
    return new Tensor(out, [n]);
  };
  return {
    params: { w: w0, b: 0 },
    predict,
    lossGrad(params, X, y) {
      const n = X.shape[0], d2 = X.shape[1];
      const pred = predict(params, X);
      let loss = 0;
      const gw = new Float64Array(d2);
      let gb = 0;
      for (let i = 0; i < n; i++) {
        const e = pred.at(i) - y.at(i);
        loss += e * e;
        for (let j = 0; j < d2; j++) gw[j] += 2 * e * X.get2(i, j);
        gb += 2 * e;
      }
      return { loss: loss / n, grad: { w: new Tensor(gw, [d2]).map(v => v / n), b: gb / n } };
    },
  };
}

const sgdLocal = (p, g, lr) => ({
  w: new Tensor(Float64Array.from([p.w.data[0] - lr * g.w.data[0]]), [1]),
  b: p.b - lr * g.b,
});

// 高斯消元（部分选主元）
function gaussSolve(A, b) {
  const n = b.length;
  const M = A.map(r => Float64Array.from(r));
  const x = Float64Array.from(b);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-14) throw new Error('奇异矩阵');
    [M[c], M[piv]] = [M[piv], M[c]];
    [x[c], x[piv]] = [x[piv], x[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k < n; k++) M[r][k] -= f * M[c][k];
      x[r] -= f * x[c];
    }
  }
  const out = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = x[r];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * out[k];
    out[r] = s / M[r][r];
  }
  return out;
}

// 岭回归闭式解 w = (XᵀX + λI)⁻¹ Xᵀy
function ridgeSolve(X, y, lam) {
  const [n, d] = X.shape;
  const A = Array.from({ length: d }, () => new Float64Array(d));
  const b = new Float64Array(d);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < d; a++) {
      b[a] += X.get2(i, a) * y.at(i);
      for (let c = 0; c < d; c++) A[a][c] += X.get2(i, a) * X.get2(i, c);
    }
  }
  for (let a = 0; a < d; a++) A[a][a] += lam;
  return gaussSolve(A, b);
}
function ridgePredict(X, w) {
  const n = X.shape[0], d = X.shape[1];
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < d; j++) s += X.get2(i, j) * w[j];
    out[i] = s;
  }
  return out;
}
function ridgeFitPredict(lam) {
  return (train, val) => Tensor.fromFlat(ridgePredict(val.X, ridgeSolve(train.X, train.y, lam)), [val.X.shape[0]]);
}

// 近质心分类器（分类口径的 fitPredict）
function centroidFitPredict(train, val) {
  const classes = [...new Set(train.y.toArray())];
  const cents = classes.map(c => {
    const idx = train.y.toArray().map((v, i) => (v === c ? i : -1)).filter(i => i >= 0);
    const d = train.X.shape[1];
    const mean = new Float64Array(d);
    for (const i of idx) for (let j = 0; j < d; j++) mean[j] += train.X.get2(i, j) / idx.length;
    return { c, mean };
  });
  const n = val.X.shape[0];
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let best = null, bd = Infinity;
    for (const { c, mean } of cents) {
      let dist = 0;
      for (let j = 0; j < mean.length; j++) { const dv = val.X.get2(i, j) - mean[j]; dist += dv * dv; }
      if (dist < bd) { bd = dist; best = c; }
    }
    out[i] = best;
  }
  return new Tensor(out, [n]);
}

// ============================================================================
test('域9 自检：恰好 103 条，id 连续、name 唯一且带 train. 前缀', () => {
  const ops = opsByDomain(9);
  assert.equal(ops.length, 103);
  assert.deepEqual(ops.map(o => o.id), range(103).map(i => `D9-${String(i + 1).padStart(3, '0')}`));
  const names = ops.map(o => o.name);
  assert.equal(new Set(names).size, 103);
  for (const o of ops) {
    assert.equal(o.domain, 9);
    assert.ok(o.name.startsWith('train.'), o.name);
    assert.ok(o.sig.length > 0 && o.doc.length > 0 && o.ref.length > 0 && o.inv.length > 0, `${o.id} 元数据不全`);
    assert.ok(hasOp(o.name) && hasOp(o.id), `${o.id} 无法按 name/id 索引`);
  }
});

test('分批：覆盖、不相交、末批与 dropLast；shuffle 可复现；stratified 保持比例', () => {
  const n = 23;
  const X = randMat(n, 3, 5);
  const y = Tensor.fromFlat(range(n).map(i => i % 3), [n]);
  const ds = { X, y };
  const batches = run('train.data.batchIterator', [ds, 5, {}]);
  assert.equal(batches.length, 5);
  assert.deepEqual(batches.map(b => b.index.length), [5, 5, 5, 5, 3]);
  assert.deepEqual(batches[0].index, [0, 1, 2, 3, 4]);
  assert.deepEqual(batches[4].index, [20, 21, 22]);
  // 内容抽查：批 X/y 与原始行一致
  assert.equal(batches[1].X.get2(0, 0), X.get2(5, 0));
  assert.equal(batches[1].y.at(0), y.at(5));
  // 覆盖 + 不相交
  const all = batches.flatMap(b => b.index);
  assert.equal(new Set(all).size, n);
  assert.deepEqual([...all].sort((a, b) => a - b), range(n));
  // dropLast：末批 3 < 5 被丢弃
  const dropped = run('train.data.batchIterator', [ds, 5, { dropLast: true }]);
  assert.equal(dropped.length, 4);
  // shuffle：同种子同顺序、不同种子不同顺序
  const s1 = run('train.data.shuffledBatches', [ds, 5, 42]);
  const s2 = run('train.data.shuffledBatches', [ds, 5, 42]);
  const s3 = run('train.data.shuffledBatches', [ds, 5, 43]);
  assert.deepEqual(s1.map(b => b.index), s2.map(b => b.index));
  assert.notDeepEqual(s1.map(b => b.index), s3.map(b => b.index));
  assert.equal(new Set(s1.flatMap(b => b.index)).size, n);
  // stratified：各批类别比例 ≈ 总体（每类摊差 ≤1 个元素）
  const st = run('train.data.stratifiedBatches', [ds, 5, 7]);
  assert.equal(new Set(st.flatMap(b => b.index)).size, n);
  const overall = [0, 1, 2].map(c => y.toArray().filter(v => v === c).length / n);
  for (const b of st) {
    for (const c of [0, 1, 2]) {
      const p = b.index.filter(i => y.data[i] === c).length / b.index.length;
      assert.ok(Math.abs(p - overall[c]) <= 0.15, `类别 ${c} 比例 ${p} vs 总体 ${overall[c]}`);
    }
  }
});

test('序列窗口 / 数据集构造 / 离群与列变换', () => {
  const series = Tensor.fromFlat([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10]);
  const wins = run('train.data.windowIterator', [series, 3, 1]);
  assert.equal(wins.length, 7);
  assert.deepEqual(wins[0].X.toArray(), [1, 2, 3]);
  assert.deepEqual(wins[0].y.toArray(), [4]);
  assert.deepEqual(wins[6].X.toArray(), [7, 8, 9]);
  assert.deepEqual(wins[6].y.toArray(), [10]);
  const wins2 = run('train.data.windowIterator', [series, 3, 2]);
  assert.equal(wins2.length, 4); // 起点 0,2,4,6
  assert.deepEqual(wins2[2].X.toArray(), [5, 6, 7]);
  assert.deepEqual(wins2[3].X.toArray(), [7, 8, 9]);
  assert.deepEqual(wins2[3].y.toArray(), [10]);
  // 二维序列窗口
  const s2d = Tensor.fromNested([[1, 10], [2, 20], [3, 30], [4, 40]]);
  const w2 = run('train.data.windowIterator', [s2d, 2, 1]);
  assert.equal(w2.length, 2);
  assert.deepEqual(w2[0].X.toArray(), [[1, 10], [2, 20]]);
  assert.deepEqual(w2[0].y.toArray(), [[3, 30]]);
  // dataset 构造与校验
  const ds = run('train.data.dataset', [[[1, 2], [3, 4]], [0, 1]]);
  assert.equal(ds.X.shape[0], 2);
  assert.equal(ds.y.at(1), 1);
  assert.throws(() => run('train.data.dataset', [[[1, 2], [3, 4]], [0]]), (e) => e.code === 'E_SHAPE');
  // clipOutliers
  const dsc = { X: Tensor.fromNested([[1], [2], [3], [4], [100]]), y: Tensor.fromFlat([0, 0, 0, 0, 0], [5]) };
  const clipped = run('train.data.clipOutliers', [dsc, 1.5]);
  const col = clipped.X.toArray().map(r => r[0]);
  assert.ok(Math.max(...col) < 100);
  assert.equal(minOf(col), 1); // 下界 mean-1.5σ ≤ 1，而原最小值即 1
  // winsorize：1%/99% → 20%/80% 分位（type7 插值）
  const win = run('train.data.winsorize', [dsc, 0.2, 0.8]);
  const colw = win.X.toArray().map(r => r[0]);
  assert.ok(approx(colw[4], 23.2, 1e-9), `上尾应压到 Q(0.8)=23.2，实际 ${colw[4]}`);
  assert.ok(approx(colw[0], 1.8, 1e-9), `下尾应抬到 Q(0.2)=1.8，实际 ${colw[0]}`);
  // logTransform
  const lg = run('train.data.logTransform', [dsc, 1]);
  assert.ok(approx(lg.X.at(0), Math.log(2), 1e-12));
  // polyExpand：degree=2, d=2 → [x1, x2, x1², x1x2, x2²]
  const poly = run('train.data.polyExpand', [{ X: Tensor.fromNested([[2, 3]]), y: Tensor.fromFlat([0], [1]) }, 2]);
  assert.deepEqual(poly.X.toArray(), [[2, 3, 4, 6, 9]]);
  // featureScale
  const fs = run('train.data.featureScale', [{ X: Tensor.fromNested([[2, 4]]), y: Tensor.fromFlat([0], [1]) }, [0.5, 2]]);
  assert.deepEqual(fs.X.toArray(), [[1, 8]]);
});

function minOf(a) { return a.reduce((x, y) => Math.min(x, y), Infinity); }

test('trainTestSplit / kFold / stratifiedKFold', () => {
  const n = 60;
  const X = randMat(n, 2, 11);
  const y = Tensor.fromFlat(range(n).map(i => i % 3), [n]);
  const ds = { X, y };
  const { train, test } = run('train.data.trainTestSplit', [ds, 0.25, 3]);
  assert.equal(train.X.shape[0] + test.X.shape[0], n);
  assert.equal(test.X.shape[0], 15);
  assert.equal(train.X.shape[0], 45);
  run('train.util.assertSplitDisjoint', [train.index, test.index]);
  assert.equal(run('train.util.coverageCheck', [[train.index, test.index], n]).ok, true);
  // kFold：不相交、并集=全集、折大小均衡
  const folds = run('train.data.kFoldSplit', [ds, 5, 9]);
  assert.equal(folds.length, 5);
  assert.deepEqual(folds.map(f => f.fold), [0, 1, 2, 3, 4]);
  assert.deepEqual(folds.map(f => f.valIdx.length), [12, 12, 12, 12, 12]);
  assert.deepEqual(folds.map(f => f.trainIdx.length), [48, 48, 48, 48, 48]);
  run('train.util.assertSplitDisjoint', folds.map(f => f.valIdx));
  assert.equal(run('train.util.coverageCheck', [folds.map(f => f.valIdx), n]).ok, true);
  assert.equal(run('train.util.coverageCheck', [folds.map(f => f.trainIdx), n]).ok, false); // 训练集本就重叠
  // stratifiedKFold：类别 [10,20,30] → 各折 [2,4,6]，比例≈总体
  const yStrat = Tensor.fromFlat([...Array(10).fill(0), ...Array(20).fill(1), ...Array(30).fill(2)], [60]);
  const ds2 = { X: randMat(60, 2, 13), y: yStrat };
  const sf = run('train.data.stratifiedKFoldSplit', [ds2, 5, 4]);
  assert.equal(sf.length, 5);
  run('train.util.assertSplitDisjoint', sf.map(f => f.valIdx));
  assert.equal(run('train.util.coverageCheck', [sf.map(f => f.valIdx), 60]).ok, true);
  const overall = [10 / 60, 20 / 60, 30 / 60];
  for (const f of sf) {
    assert.equal(f.valIdx.length, 12);
    for (const c of [0, 1, 2]) {
      const p = f.valIdx.filter(i => yStrat.data[i] === c).length / 12;
      assert.ok(Math.abs(p - overall[c]) <= 0.1, `fold ${f.fold} 类 ${c}: ${p} vs ${overall[c]}`);
    }
  }
});

test('标准化器与 minmax：均值≈0、std≈1、值域∈[0,1]', () => {
  const { X, y } = blobs({ n: 60, seed: 5 });
  const ds = { X, y };
  const sc = run('train.data.standardizerFit', [ds]);
  const out = run('train.data.standardizerApply', [ds, sc]);
  const Xt = out.X;
  assert.equal(out.y, y); // y 保留
  for (let j = 0; j < 2; j++) {
    let m = 0;
    for (let i = 0; i < 60; i++) m += Xt.get2(i, j);
    m /= 60;
    let s = 0;
    for (let i = 0; i < 60; i++) s += (Xt.get2(i, j) - m) ** 2;
    s = Math.sqrt(s / 60);
    assert.ok(approx(m, 0, 1e-9), `mean[${j}]=${m}`);
    assert.ok(approx(s, 1, 1e-9), `std[${j}]=${s}`);
  }
  const mm = run('train.data.minmaxFit', [ds]);
  const outm = run('train.data.minmaxApply', [ds, mm]);
  for (let i = 0; i < 60; i++) {
    for (let j = 0; j < 2; j++) {
      const v = outm.X.get2(i, j);
      assert.ok(v >= -1e-12 && v <= 1 + 1e-12, `值越界 ${v}`);
    }
  }
  // 每列恰有 0 与 1（原 min/max 行）
  for (let j = 0; j < 2; j++) {
    const col = Array.from(outm.X.col(j));
    assert.ok(approx(Math.min(...col), 0, 1e-12));
    assert.ok(approx(Math.max(...col), 1, 1e-12));
  }
});

test('onehot/label 编解码互逆', () => {
  const labels = ['cat', 'dog', 'cat', 'bird', 'dog'];
  const enc = run('train.data.labelEncode', [labels]);
  assert.deepEqual(enc.classes, ['bird', 'cat', 'dog']);
  assert.deepEqual(enc.y.toArray(), [1, 2, 1, 0, 2]);
  const dec = run('train.data.labelDecode', [enc.y, enc.classes]);
  assert.deepEqual(dec, labels);
  const oh = run('train.data.onehotEncode', [enc.y, 3]);
  assert.deepEqual(oh.toArray(), [[0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]]);
  // 每行和为 1
  for (let i = 0; i < 5; i++) assert.equal(Array.from(oh.row(i)).reduce((a, b) => a + b, 0), 1);
  // onehot → argmax → 标签（互逆的另一半）
  const backIdx = oh.toArray().map(r => r.indexOf(1));
  assert.deepEqual(backIdx, enc.y.toArray());
  // numClasses 越界应抛错
  assert.throws(() => run('train.data.onehotEncode', [[2], 2]), (e) => e.code === 'E_VALUE');
});

test('管线组合与拟合-变换（防泄漏 apply）', () => {
  const { X, y } = blobs({ n: 40, seed: 8 });
  const ds = { X, y };
  const step1 = (d) => run('train.data.featureScale', [d, 2]);
  const step2 = (d) => run('train.data.featureScale', [d, 3]);
  const pipe = run('train.data.pipelineCompose', [[step1, step2]]);
  const out = pipe(ds);
  assert.deepEqual(out.X.toArray(), X.map(v => v * 6).toArray());
  const id = run('train.data.pipelineCompose', [[]]);
  assert.deepEqual(id(ds).X.toArray(), X.toArray());
  // fit/transform 分离
  const fitters = [(d) => {
    const sc = run('train.data.standardizerFit', [d]);
    return (d2) => run('train.data.standardizerApply', [d2, sc]);
  }];
  // fit/transform 分离：在训练子集上拟合，再应用于测试子集
  const { train, test } = run('train.data.trainTestSplit', [ds, 0.3, 5]);
  const { data, apply } = run('train.data.pipelineFitTransform', [train, fitters]);
  for (let j = 0; j < 2; j++) {
    let m = 0;
    for (let i = 0; i < train.X.shape[0]; i++) m += data.X.get2(i, j);
    m /= train.X.shape[0];
    assert.ok(approx(m, 0, 1e-9));
  }
  // 新数据用同一拟合参数变换（与手动一致）
  const t2 = apply(test);
  const t2manual = run('train.data.standardizerApply', [test, run('train.data.standardizerFit', [train])]);
  assert.equal(maxAbsDiff(t2.X, t2manual.X), 0);
});

test('batchStep / loop.epoch / loop.evaluate（通用模型接口）', () => {
  const { X, y } = linData(30, 7);
  const model = linearModel(1, 1);
  const Xb = new Tensor(X.data.slice(0, 5), [5, 1]);
  const yb = new Tensor(y.data.slice(0, 5), [5]);
  const step = run('train.loop.batchStep', [model, model.params, { X: Xb, y: yb }, 0.1, null]);
  assert.ok(step.loss > 0);
  assert.ok(step.gradNorm > 0);
  // 手算 SGD 一步核对
  const lg = model.lossGrad(model.params, Xb, yb);
  assert.ok(approx(step.params.w.data[0], model.params.w.data[0] - 0.1 * lg.grad.w.data[0], 1e-12));
  assert.ok(approx(step.params.b, model.params.b - 0.1 * lg.grad.b, 1e-12));
  assert.ok(approx(model.params.w.data[0], step.params.w.data[0] + 0.1 * lg.grad.w.data[0], 1e-12)); // 入参未被修改
  // epoch：3 批、loss 有限
  const ep = run('train.loop.epoch', [model, model.params, { X, y }, { batchSize: 10, lr: 0.1 }]);
  assert.equal(ep.batches, 3);
  assert.ok(ep.loss > 0 && Number.isFinite(ep.loss));
  assert.notEqual(ep.params.w.data[0], model.params.w.data[0]);
  // evaluate
  const mse = run('train.loop.evaluate', [model, { X, y }, 'mse']);
  assert.ok(mse > 0);
  const mse2 = run('train.loop.evaluate', [model, { X, y }, null]);
  assert.ok(approx(mse2, mse, 1e-12));
});

test('训练循环：收敛、history、回调、早停集成、EMA、lr 调度、shuffle 可复现', () => {
  const { X, y } = linData(60, 7);
  const ds = { X, y };
  const r = run('train.loop.run', [linearModel(1, 3), { train: ds }, { epochs: 300, lr: 0.08, batchSize: 0 }]);
  assert.equal(r.epochsRun, 300);
  assert.equal(r.history.length, 300);
  assert.equal(r.stopped, 'epochs');
  assert.ok(r.history[299].trainLoss < 0.2 * r.history[0].trainLoss, `loss ${r.history[0].trainLoss} -> ${r.history[299].trainLoss}`);
  assert.ok(approx(r.params.w.data[0], 2, 0.1), `w=${r.params.w.data[0]}`);
  assert.ok(approx(r.params.b, 1, 0.1), `b=${r.params.b}`);
  assert.deepEqual(r.history.map(h => h.epoch), range(300).map(i => i + 1));
  // 回调停止（onEpochEnd 返回 false）
  let seen = 0;
  const r2 = run('train.loop.run', [linearModel(1, 3), { train: ds }, {
    epochs: 100, lr: 0.08,
    callbacks: { onEpochEnd: (ctx) => { seen++; if (ctx.epoch >= 10) return false; } },
  }]);
  assert.equal(r2.stopped, 'callback');
  assert.equal(r2.epochsRun, 10);
  assert.equal(seen, 10);
  // 早停集成：构造 loss 单调上升的假模型（valLoss 监控）
  let calls = 0;
  const fakeModel = {
    params: { w: Tensor.fromFlat([0], [1]) },
    predict: (p) => p.w,
    lossGrad: () => { calls++; return { loss: 1 + 0.1 * calls, grad: { w: Tensor.fromFlat([0], [1]) } }; },
  };
  const r3 = run('train.loop.run', [fakeModel, { train: ds, val: ds }, {
    epochs: 20, lr: 0.01, earlyStopping: { patience: 3, mode: 'min' },
  }]);
  assert.equal(r3.stopped, 'earlyStopping');
  assert.equal(r3.epochsRun, 4);
  assert.equal(r3.best.epoch, 1);
  assert.ok(approx(r3.best.value, 1.2, 1e-9));
  // EMA：apply 与内部一致且隔离
  const r4 = run('train.loop.run', [linearModel(1, 3), { train: ds }, { epochs: 50, lr: 0.08, ema: { decay: 0.9 } }]);
  assert.ok(r4.ema);
  const emaP = run('train.ema.apply', [r4.ema]);
  assert.ok(approx(emaP.w.data[0], r4.ema.params.w.data[0], 0));
  emaP.w.data[0] = -99;
  assert.notEqual(r4.ema.params.w.data[0], -99);
  // lr 调度（step 每 10 轮折半）
  const seenLr = [];
  run('train.loop.run', [linearModel(1, 3), { train: ds }, {
    epochs: 25, lr: 1.0, lrSchedule: { kind: 'step', drop: 0.5, every: 10 },
    callbacks: { onEpochEnd: (ctx) => { seenLr.push(ctx.lr); } },
  }]);
  assert.ok(approx(seenLr[0], 1.0, 1e-12));
  assert.ok(approx(seenLr[9], 1.0, 1e-12));
  assert.ok(approx(seenLr[10], 0.5, 1e-12));
  assert.ok(approx(seenLr[20], 0.25, 1e-12));
  // shuffle + 同种子逐位可复现；不同种子不同
  // shuffle + 同种子逐位可复现；不同种子不同（小批使批组成随排列变化）
  const so = { epochs: 5, lr: 0.1, shuffle: true, batchSize: 12 };
  const a1 = run('train.loop.run', [linearModel(1, 3), { train: ds }, { ...so, seed: 42 }]);
  const a2 = run('train.loop.run', [linearModel(1, 3), { train: ds }, { ...so, seed: 42 }]);
  const a3 = run('train.loop.run', [linearModel(1, 3), { train: ds }, { ...so, seed: 43 }]);
  assert.equal(maxAbsDiff(a1.params.w, a2.params.w), 0);
  assert.equal(a1.params.b, a2.params.b);
  assert.ok(maxAbsDiff(a1.params.w, a3.params.w) > 0 || a1.params.b !== a3.params.b, '不同种子应产生不同参数');
  // withClipping：梯度 norm=5 裁到 0.5 → 缩放 0.1
  const wrapped = run('train.loop.withClipping', [sgdLocal, 0.5]);
  const p0 = { w: Tensor.fromFlat([1], [1]), b: 0 };
  const p1 = wrapped(p0, { w: Tensor.fromFlat([3], [1]), b: 4 }, 1.0);
  assert.ok(approx(p1.w.data[0], 0.7, 1e-12));
  assert.ok(approx(p1.b, -0.4, 1e-12));
});

test('早停 / 最大轮数 / 时间预算 / 组合准则', () => {
  const es = run('train.stop.earlyStopping', [[1.0, 0.8, 0.7, 0.75, 0.82, 0.85], { patience: 2, mode: 'min' }]);
  assert.equal(es.shouldStop, true);
  assert.equal(es.bestEpoch, 3);
  assert.ok(approx(es.bestValue, 0.7, 1e-12));
  assert.equal(es.stale, 3);
  assert.equal(es.trace.length, 6);
  assert.equal(es.trace[2].improved, true);
  assert.equal(es.trace[3].improved, false);
  const es2 = run('train.stop.earlyStopping', [[1.0, 0.8, 0.7, 0.75, 0.82, 0.85], { patience: 4, mode: 'min' }]);
  assert.equal(es2.shouldStop, false);
  assert.equal(es2.stale, 3);
  // max 模式
  const es3 = run('train.stop.earlyStopping', [[0.5, 0.6, 0.55, 0.52], { patience: 1, mode: 'max' }]);
  assert.equal(es3.bestEpoch, 2);
  assert.equal(es3.shouldStop, true);
  // minDelta：0.05 以内的改善不算改善
  const es4 = run('train.stop.earlyStopping', [[1.0, 0.99, 0.98], { patience: 1, mode: 'min', minDelta: 0.05 }]);
  assert.equal(es4.bestEpoch, 1);
  assert.equal(es4.shouldStop, true);
  // maxEpochs / timeBudget
  assert.equal(run('train.stop.maxEpochs', [5, 5]), true);
  assert.equal(run('train.stop.maxEpochs', [4, 5]), false);
  // timeBudget：startedAt=1000, budget=500 → now=1600 超预算（elapsed=600, remaining=-100）
  const tb = run('train.stop.timeBudget', [1000, 500, 1600]);
  assert.equal(tb.exceeded, true);
  assert.equal(tb.elapsed, 600);
  assert.equal(tb.remaining, -100);
  const tb2 = run('train.stop.timeBudget', [1000, 500, 1200]);
  assert.equal(tb2.exceeded, false);
  assert.equal(tb2.elapsed, 200);
  assert.equal(tb2.remaining, 300);
  // criteria
  const rules = [{ name: 'a', stop: true }, { name: 'b', stop: false }];
  const cr = run('train.stop.criteria', [rules, 'any']);
  assert.deepEqual(cr.triggered, ['a']);
  assert.equal(cr.stop, true);
  const cr2 = run('train.stop.criteria', [rules, 'all']);
  assert.equal(cr2.stop, false);
  const cr3 = run('train.stop.criteria', [[{ name: 'a', stop: true }, { name: 'b', stop: true }], 'all']);
  assert.equal(cr3.stop, true);
  assert.equal(run('train.stop.criteria', [[], 'all']).stop, false);
});

test('检查点保存/加载/最优跟踪', () => {
  const params = { w: Tensor.fromFlat([1, 2], [2]), b: 0.5 };
  const ck = run('train.ckpt.save', [params, { epoch: 3 }]);
  const back = run('train.ckpt.load', [ck]);
  assert.equal(maxAbsDiff(back.w, params.w), 0);
  assert.equal(back.b, params.b);
  back.w.data[0] = 99;
  assert.equal(ck.params.w.data[0], 1); // 快照隔离
  assert.equal(params.w.data[0], 1);
  const tr = run('train.ckpt.bestTracker', ['min']);
  const r1 = tr.update(0.5, 1);
  const r2 = tr.update(0.3, 2);
  const r3 = tr.update(0.4, 3);
  assert.equal(r1.improved, true);
  assert.equal(r2.improved, true);
  assert.equal(r3.improved, false);
  assert.equal(tr.bestEpoch, 2);
  assert.equal(tr.snapshot().best, 0.3);
  assert.equal(tr.snapshot().updates, 3);
});

test('历史记录 / Markdown / lr 钩子', () => {
  let h = [];
  h = run('train.hist.log', [h, { trainLoss: 0.5 }]);
  h = run('train.hist.log', [h, { trainLoss: 0.3, valLoss: 0.4 }]);
  assert.equal(h.length, 2);
  assert.equal(h[0].epoch, 1);
  assert.equal(h[1].epoch, 2);
  const md = run('train.hist.toMarkdown', [h, null]);
  assert.ok(md.startsWith('| epoch | trainLoss | valLoss |'));
  assert.equal(md.split('\n').length, 4); // 表头 + 分隔 + 2 行
  assert.ok(md.includes('0.500000'));
  // lr.hook：epoch 从 1 计数
  assert.ok(approx(run('train.lr.hook', [1, 1.0, { kind: 'step', drop: 0.5, every: 10 }]), 1.0, 1e-12));
  assert.ok(approx(run('train.lr.hook', [11, 1.0, { kind: 'step', drop: 0.5, every: 10 }]), 0.5, 1e-12));
  assert.ok(approx(run('train.lr.hook', [21, 1.0, { kind: 'step', drop: 0.5, every: 10 }]), 0.25, 1e-12));
  assert.ok(approx(run('train.lr.hook', [2, 1.0, { kind: 'exp', gamma: 0.1 }]), 0.1, 1e-12));
  assert.ok(approx(run('train.lr.hook', [51, 1.0, { kind: 'linear', total: 100 }]), 0.5, 1e-12));
  assert.ok(approx(run('train.lr.hook', [51, 1.0, { kind: 'cosine', total: 100 }]), 0.5, 1e-9));
  assert.ok(approx(run('train.lr.hook', [3, 2.0, (e) => 0.1 * e]), 0.6, 1e-12));
});

test('EMA / 梯度累积 / 每轮打乱', () => {
  const params = { w: Tensor.fromFlat([1, 2], [2]) };
  const st = run('train.ema.init', [params, 0.9]);
  const st2 = run('train.ema.update', [st, { w: Tensor.fromFlat([3, 4], [2]) }, null]);
  assert.deepEqual(st2.params.w.toArray(), [1.2, 2.2]); // 0.9·1 + 0.1·3
  const applied = run('train.ema.apply', [st2]);
  applied.w.data[0] = -1;
  assert.equal(st2.params.w.data[0], 1.2); // apply 隔离
  const st3 = run('train.ema.update', [st, { w: Tensor.fromFlat([7, 8], [2]) }, 0]);
  assert.deepEqual(st3.params.w.toArray(), [7, 8]); // decay=0 → 取当前
  // accum
  let acc = run('train.accum.reset', [null]);
  assert.equal(acc.count, 0);
  assert.equal(acc.sum, null);
  acc = run('train.accum.step', [acc, { w: Tensor.fromFlat([1, 2], [2]) }]);
  acc = run('train.accum.step', [acc, { w: Tensor.fromFlat([3, 4], [2]) }]);
  assert.deepEqual(acc.sum.w.toArray(), [4, 6]);
  assert.equal(acc.count, 2);
  const acc2 = run('train.accum.reset', [{ w: Tensor.fromFlat([9, 9], [2]) }]);
  assert.deepEqual(acc2.sum.w.toArray(), [0, 0]);
  assert.equal(acc2.count, 0);
  // shuffleEachEpoch：确定性 + 是排列 + 不同 epoch 不同
  const sh = run('train.loop.shuffleEachEpoch', [7]);
  const p1 = sh(10, 1);
  assert.deepEqual(p1, sh(10, 1));
  assert.deepEqual([...p1].sort((a, b) => a - b), range(10));
  assert.notDeepEqual(p1, sh(10, 2));
});

test('mixup / 标签平滑 / 数据增强', () => {
  const X = Tensor.fromNested([[1, 2], [3, 4], [5, 6]]);
  const y = Tensor.fromFlat([0, 1, 0], [3]);
  const mx = run('train.mixup', [{ X, y }, 1, createRng(42)]);
  assert.ok(mx.lam >= 0 && mx.lam <= 1);
  assert.equal(mx.perm.length, 3);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 2; j++) {
      assert.ok(approx(mx.X.get2(i, j), mx.lam * X.get2(i, j) + (1 - mx.lam) * X.get2(mx.perm[i], j), 1e-12));
    }
    assert.ok(approx(mx.y.at(i), mx.lam * y.at(i) + (1 - mx.lam) * y.at(mx.perm[i]), 1e-12));
  }
  // Beta(1,1)=Uniform(0,1) → lam 均值≈0.5
  const rng2 = createRng(1);
  const lams = [];
  for (let t = 0; t < 400; t++) lams.push(run('train.mixup', [{ X, y }, 1, rng2]).lam);
  assert.ok(approx(lams.reduce((a, b) => a + b, 0) / lams.length, 0.5, 0.06), `lam 均值 ${lams.reduce((a, b) => a + b, 0) / lams.length}`);
  // alpha=0 → lam=1（不混合）
  const mx0 = run('train.mixup', [{ X, y }, 0, createRng(2)]);
  assert.equal(mx0.lam, 1);
  assert.equal(maxAbsDiff(mx0.X, X), 0);
  // labelSmoothing
  const sm = run('train.labelSmoothing', [Tensor.fromFlat([0, 2], [2]), 3, 0.1]);
  const e = 0.1 / 3;
  tensorClose(sm, [[0.9 + e, e, e], [e, e, 0.9 + e]], 1e-12);
  for (let i = 0; i < 2; i++) assert.equal(Array.from(sm.row(i)).reduce((a, b) => a + b, 0), 1);
  const sm0 = run('train.labelSmoothing', [Tensor.fromFlat([1], [1]), 2, 0]);
  assert.deepEqual(sm0.toArray(), [[0, 1]]);
  // gaussianNoise
  assert.equal(maxAbsDiff(run('train.aug.gaussianNoise', [X, 0, createRng(3)]), X), 0);
  const gn = run('train.aug.gaussianNoise', [X, 0.5, createRng(3)]);
  assert.equal(maxAbsDiff(gn, run('train.aug.gaussianNoise', [X, 0.5, createRng(3)])), 0); // 同种子可复现
  assert.notEqual(maxAbsDiff(gn, X), 0);
  // featureDropout：p=0 恒等；p=0.5 有零值；同种子可复现
  assert.equal(maxAbsDiff(run('train.aug.featureDropout', [X, 0, createRng(4)]), X), 0);
  const fd = run('train.aug.featureDropout', [X, 0.5, createRng(4)]);
  assert.equal(maxAbsDiff(fd, run('train.aug.featureDropout', [X, 0.5, createRng(4)])), 0);
  const zeros = Array.from(fd.data).filter(v => v === 0).length;
  assert.ok(zeros >= 1 && zeros <= 5, `零元素数 ${zeros}`);
  // scaleJitter
  assert.equal(maxAbsDiff(run('train.aug.scaleJitter', [X, 0, createRng(5)]), X), 0);
  const sj = run('train.aug.scaleJitter', [X, 0.1, createRng(5)]);
  assert.equal(maxAbsDiff(sj, run('train.aug.scaleJitter', [X, 0.1, createRng(5)])), 0);
  assert.notEqual(maxAbsDiff(sj, X), 0);
});

// ============================================================================
test('crossValidate / repeatedCV（岭回归闭式解，fold 0 手算核对）', () => {
  const { X, y } = linData(50, 7);
  const ds = { X, y };
  const k = 5;
  const r = run('train.eval.crossValidate', [ds, k, ridgeFitPredict(0.5), { seed: 11, metric: 'mse' }]);
  assert.equal(r.scores.length, k);
  assert.ok(r.scores.every(s => s > 0 && Number.isFinite(s)));
  // mean/std 自洽（population）
  const m = r.scores.reduce((a, b) => a + b, 0) / k;
  const sd = Math.sqrt(r.scores.reduce((a, b) => a + (b - m) ** 2, 0) / k);
  assert.ok(approx(r.mean, m, 1e-12));
  assert.ok(approx(r.std, sd, 1e-12));
  // fold 0 手算：同种子同划分 → 同折 → 同分数
  const folds = run('train.data.kFoldSplit', [ds, k, 11]);
  const f0 = folds[0];
  const w = ridgeSolve(f0.train.X, f0.train.y, 0.5);
  const pred = ridgePredict(f0.val.X, w);
  let s = 0;
  for (let i = 0; i < pred.length; i++) s += (pred[i] - f0.val.y.at(i)) ** 2;
  assert.ok(approx(r.scores[0], s / pred.length, 1e-9), `fold0: ${r.scores[0]} vs ${s / pred.length}`);
  // 折索引不相交且覆盖全集
  run('train.util.assertSplitDisjoint', folds.map(f => f.valIdx));
  assert.equal(run('train.util.coverageCheck', [folds.map(f => f.valIdx), 50]).ok, true);
  // 分类口径：近质心分类器
  const bl = blobs({ n: 60, seed: 5 });
  const rc = run('train.eval.crossValidate', [{ X: bl.X, y: bl.y }, 4, centroidFitPredict, { seed: 3, metric: 'accuracy' }]);
  assert.ok(rc.mean > 0.8, `centroid CV acc=${rc.mean}`);
  // repeatedCV
  const rr = run('train.eval.repeatedCV', [ds, 4, 3, ridgeFitPredict(0.5), { seed: 21, metric: 'mse' }]);
  assert.equal(rr.repeats.length, 3);
  assert.equal(rr.allScores.length, 12);
  assert.ok(approx(rr.meanOfMeans, rr.repeats.reduce((a, x) => a + x.mean, 0) / 3, 1e-12));
  assert.ok(approx(rr.grandMean, rr.allScores.reduce((a, b) => a + b, 0) / 12, 1e-12));
  assert.ok(rr.repeats.every(x => x.scores.length === 4));
});

test('学习曲线 / 验证曲线', () => {
  const { X, y } = linData(80, 7);
  const ds = { X, y };
  const lc = run('train.eval.learningCurve', [ds, [10, 20, 40, 60], ridgeFitPredict(0.5), { seed: 5, metric: 'mse', valFrac: 0.25 }]);
  assert.equal(lc.length, 4);
  assert.deepEqual(lc.map(r => r.n), [10, 20, 40, 60]);
  assert.ok(lc.every(r => r.trainScore > 0 && r.valScore > 0));
  // 线性关系下数据量增大验证 MSE 不升
  assert.ok(lc[3].valScore <= lc[0].valScore + 0.05, `valScore ${lc[0].valScore} -> ${lc[3].valScore}`);
  // 验证曲线
  const vc = run('train.eval.validationCurve', [ds, [0.001, 0.01, 0.1, 1, 10], (lam) => ridgeFitPredict(lam), { k: 4, seed: 6, metric: 'mse' }]);
  assert.equal(vc.length, 5);
  assert.ok(vc.every(r => r.scores.length === 4));
  assert.ok(vc.every(r => approx(r.mean, r.scores.reduce((a, b) => a + b, 0) / 4, 1e-12)));
  // 过度正则（λ=10）不应是最优
  const means = vc.map(r => r.mean);
  assert.ok(means[4] > Math.min(...means) + 1e-12, `means=${means}`);
});

test('bootstrapCI / pairedCompare', () => {
  const values = [0.82, 0.85, 0.79, 0.88, 0.75, 0.9, 0.83, 0.86];
  const b1 = run('train.eval.bootstrapCI', [values, { B: 800, alpha: 0.05, seed: 11 }]);
  const b2 = run('train.eval.bootstrapCI', [values, { B: 800, alpha: 0.05, seed: 11 }]);
  assert.deepEqual(b1, b2); // 固定种子可复现
  const point = values.reduce((a, b) => a + b, 0) / values.length;
  assert.ok(approx(b1.point, point, 1e-12));
  assert.ok(b1.lo <= point && point <= b1.hi, `CI [${b1.lo}, ${b1.hi}] 应包含点估计 ${point}`);
  assert.ok(b1.hi - b1.lo > 0);
  assert.ok(b1.lo >= Math.min(...values) && b1.hi <= Math.max(...values));
  const b3 = run('train.eval.bootstrapCI', [values, { B: 800, alpha: 0.05, seed: 12 }]);
  assert.notDeepEqual(b1, b3); // 不同种子区间不同
  // median 统计量：点估计 = 0.84
  const bm = run('train.eval.bootstrapCI', [values, { B: 400, seed: 3, statistic: 'median' }]);
  assert.ok(approx(bm.point, 0.84, 1e-12));
  // pairedCompare
  const a = [0.8, 0.85, 0.9, 0.82];
  const bb = [0.75, 0.8, 0.82, 0.81];
  const pc = run('train.eval.pairedCompare', [a, bb, {}]);
  const diffs = a.map((v, i) => v - bb[i]);
  assert.ok(approx(pc.meanDiff, diffs.reduce((x, y2) => x + y2, 0) / 4, 1e-12));
  assert.equal(pc.wins, 4);
  assert.equal(pc.losses, 0);
  assert.equal(pc.ties, 0);
  assert.ok(pc.t > 0);
  assert.ok(pc.pNormal < 0.05);
  assert.ok(pc.ciLo <= pc.meanDiff && pc.meanDiff <= pc.ciHi);
  assert.equal(pc.df, 3);
});

test('classificationReport / confusionReport / perClass / macroAvg / microAvg（手算核对）', () => {
  const yTrue = [0, 0, 0, 1, 1, 1];
  const yPred = [0, 0, 1, 1, 1, 0];
  const rep = run('train.eval.classificationReport', [yTrue, yPred, null]);
  assert.ok(approx(rep.accuracy, 4 / 6, 1e-12));
  assert.equal(rep.support, 6);
  const c0 = rep.perClass.find(r => r.class === 0);
  const c1 = rep.perClass.find(r => r.class === 1);
  // 类0：tp=2, fp=1, fn=1
  assert.deepEqual([c0.tp, c0.fp, c0.fn, c0.support], [2, 1, 1, 3]);
  assert.ok(approx(c0.precision, 2 / 3, 1e-12));
  assert.ok(approx(c0.recall, 2 / 3, 1e-12));
  assert.ok(approx(c0.f1, 2 / 3, 1e-12));
  // 类1：tp=2, fp=1, fn=1（对称）
  assert.ok(approx(c1.precision, 2 / 3, 1e-12));
  assert.ok(approx(rep.macro.precision, 2 / 3, 1e-12));
  assert.ok(approx(rep.macro.f1, 2 / 3, 1e-12));
  assert.ok(approx(rep.weighted.f1, 2 / 3, 1e-12));
  // confusionReport：行列和
  const conf = run('train.eval.confusionReport', [yTrue, yPred, null]);
  assert.deepEqual(conf.labels, [0, 1]);
  assert.deepEqual(conf.matrix, [[2, 1], [1, 2]]);
  assert.deepEqual(conf.rowSums, [3, 3]);
  assert.deepEqual(conf.colSums, [3, 3]);
  assert.equal(conf.total, 6);
  assert.equal(conf.diagonal, 4);
  // perClass / macroAvg / microAvg
  const pc = run('train.eval.perClass', [yTrue, yPred, null]);
  assert.equal(pc.perClass.length, 2);
  const ma = run('train.eval.macroAvg', [pc]);
  assert.ok(approx(ma.f1, 2 / 3, 1e-12));
  assert.equal(ma.k, 2);
  const ma2 = run('train.eval.macroAvg', [rep]);
  assert.ok(approx(ma2.f1, 2 / 3, 1e-12));
  const mi = run('train.eval.microAvg', [yTrue, yPred]);
  assert.ok(approx(mi.f1, 4 / 6, 1e-12));
  assert.ok(approx(mi.accuracy, 4 / 6, 1e-12));
});

test('ROC / PR / 阈值扫描', () => {
  const y = [1, 1, 1, 1, 0, 0, 0, 0];
  const s = [0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.52];
  const roc = run('train.eval.rocAnalysis', [y, s, null]);
  assert.ok(approx(roc.auc, 1, 1e-12));
  assert.equal(roc.thresholds.length, 8);
  assert.deepEqual(roc.thresholds, [0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.52]);
  assert.equal(roc.tpr[roc.tpr.length - 1], 1);
  assert.equal(roc.fpr[roc.fpr.length - 1], 1);
  assert.ok(approx(roc.optimalThreshold, 0.6, 1e-12)); // Youden J 首次达 1
  // 已知 AUC=0.75（2 对组合中赢 3 对）
  const roc2 = run('train.eval.rocAnalysis', [[1, 1, 0, 0], [0.9, 0.6, 0.8, 0.4], null]);
  assert.ok(approx(roc2.auc, 0.75, 1e-12));
  // 完全并列 → 0.5
  const roc3 = run('train.eval.rocAnalysis', [[1, 0], [0.5, 0.5], null]);
  assert.ok(approx(roc3.auc, 0.5, 1e-12));
  // 缺一类应抛错
  assert.throws(() => run('train.eval.rocAnalysis', [[1, 1], [0.9, 0.8], null]), (e) => e.code === 'E_VALUE');
  // PR
  const pr = run('train.eval.prAnalysis', [y, s]);
  assert.ok(approx(pr.averagePrecision, 1, 1e-12));
  assert.ok(approx(pr.bestF1, 1, 1e-12));
  // 阈值扫描
  const sw = run('train.eval.thresholdSweep', [[1, 1, 0, 0], [0.9, 0.6, 0.8, 0.4], [0.5, 0.7]]);
  assert.equal(sw.length, 2);
  const at05 = sw[0];
  assert.equal(at05.threshold, 0.5);
  // s>=0.5 → pred=[1,1,1,0]：tp=2, fp=1(0.8), tn=1, fn=0
  assert.deepEqual([at05.tp, at05.fp, at05.tn, at05.fn], [2, 1, 1, 0]);
  assert.ok(approx(at05.precision, 2 / 3, 1e-12));
  assert.ok(approx(at05.recall, 1, 1e-12));
  const at07 = sw[1];
  // s>=0.7 → pred=[1,0,1,0]：tp=1, fp=1, tn=1, fn=1
  assert.deepEqual([at07.tp, at07.fp, at07.tn, at07.fn], [1, 1, 1, 1]);
  assert.ok(approx(at07.precision, 1 / 2, 1e-12));
  assert.ok(approx(at07.recall, 1 / 2, 1e-12));
});

test('cohenKappa / mcc / balancedAcc / logloss / brier / fbeta / topKAcc', () => {
  // yTrue=[0,0,1,1], yPred=[0,1,1,1]：po=3/4, pe=1/2 → κ=0.5；tp=2,tn=1,fp=1,fn=0 → MCC=2/√12
  const k = run('train.eval.cohenKappa', [[0, 0, 1, 1], [0, 1, 1, 1], null]);
  assert.ok(approx(k, 0.5, 1e-12));
  const m = run('train.eval.mcc', [[0, 0, 1, 1], [0, 1, 1, 1]]);
  assert.ok(approx(m, 2 / Math.sqrt(12), 1e-12), `mcc=${m}`);
  // 多类 MCC（Gorodkin）：c=4, s=6, Σp·t=12, Σp²=14, Σt²=12
  const yT = [0, 0, 1, 1, 2, 2];
  const yP = [0, 1, 1, 1, 2, 0];
  const m2 = run('train.eval.mcc', [yT, yP]);
  const expect2 = (4 * 6 - 12) / Math.sqrt((36 - 14) * (36 - 12));
  assert.ok(approx(m2, expect2, 1e-12), `mcc=${m2} vs ${expect2}`);
  // balancedAcc：recalls = [0.5, 1, 0.5] → 2/3
  const ba = run('train.eval.balancedAcc', [yT, yP]);
  assert.ok(approx(ba, 2 / 3, 1e-12));
  // logloss / brier
  const proba = Tensor.fromNested([[0.9, 0.1], [0.2, 0.8]]);
  const ll = run('train.eval.logloss', [[0, 1], proba, 1e-15]);
  assert.ok(approx(ll, (-Math.log(0.9) - Math.log(0.8)) / 2, 1e-12));
  const br = run('train.eval.brier', [[1, 0], [0.9, 0.2]]);
  assert.ok(approx(br, (0.01 + 0.04) / 2, 1e-12));
  // fbeta：tp=2, fp=0, fn=1 → F1=4/5, F2=10/14
  const fb = run('train.eval.fbeta', [[1, 1, 0, 1], [1, 0, 0, 1], 1, {}]);
  assert.ok(approx(fb, 0.8, 1e-12));
  const fb2 = run('train.eval.fbeta', [[1, 1, 0, 1], [1, 0, 0, 1], 2, {}]);
  assert.ok(approx(fb2, 10 / 14, 1e-12));
  // topKAcc
  const pk = Tensor.fromNested([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3]]);
  assert.ok(approx(run('train.eval.topKAcc', [[0, 2], pk, 1]), 0, 1e-12));
  // top2：样本0 的类0(0.1) 不在 top2(0.6,0.3) 中 → 0.5
  assert.ok(approx(run('train.eval.topKAcc', [[0, 2], pk, 2]), 0.5, 1e-12));
  assert.ok(approx(run('train.eval.topKAcc', [[0, 2], pk, 3]), 1, 1e-12));
});

test('多数类 / 分层随机基线', () => {
  const y = [0, 0, 0, 0, 1, 1, 2];
  const mb = run('train.eval.majorityBaseline', [y]);
  assert.equal(mb.label, 0);
  assert.equal(mb.count, 4);
  assert.ok(approx(mb.accuracy, 4 / 7, 1e-12));
  assert.ok(approx(mb.proportion, 4 / 7, 1e-12));
  const sb = run('train.eval.stratifiedBaseline', [y, 9]);
  assert.ok(approx(sb.expected, (4 / 7) ** 2 + (2 / 7) ** 2 + (1 / 7) ** 2, 1e-12));
  assert.ok(sb.accuracy >= 0 && sb.accuracy <= 1);
  assert.equal(sb.accuracy, run('train.eval.stratifiedBaseline', [y, 9]).accuracy); // 可复现
  assert.deepEqual(sb.prior.map(p => +(p * 7).toFixed(9)), [4, 2, 1]);
});

test('结果表 / 对比表 / 排名', () => {
  const rows = [
    { name: 'ridge', mse: 0.05, acc: 0.9 },
    { name: 'lasso', mse: 0.07, acc: 0.88 },
  ];
  const md = run('train.eval.resultTable', [rows, null]);
  assert.ok(md.startsWith('| name | mse | acc |'));
  assert.equal(md.split('\n').length, 4);
  assert.ok(md.includes('0.050000'));
  const cmp = run('train.eval.compareTable', [rows, null]);
  assert.ok(cmp.startsWith('| name |'));
  const ranked = run('train.eval.rankRuns', [rows, 'mse', 'min']);
  assert.deepEqual(ranked.map(r => r.name), ['ridge', 'lasso']);
  assert.deepEqual(ranked.map(r => r.rank), [1, 2]);
  assert.deepEqual(rows.map(r => r.name), ['ridge', 'lasso']); // 原数组未被修改
  const ranked2 = run('train.eval.rankRuns', [rows, (r) => r.acc, 'max']);
  assert.equal(ranked2[0].name, 'ridge');
  assert.equal(ranked2[1].name, 'lasso');
});

test('seedAll / fingerprint / manifest / manifestHash / envInfo / logAppend / experimentCompare', () => {
  const s1 = run('train.repro.seedAll', [42]);
  const s2 = run('train.repro.seedAll', [42]);
  const take = (rng) => Array.from({ length: 5 }, () => rng.next());
  assert.deepEqual(take(s1.rng), take(s2.rng));
  assert.deepEqual(take(s1.stream('data')), take(s2.stream('data')));
  assert.notDeepEqual(take(s1.stream('data')), take(s1.stream('init')));
  assert.equal(run('train.repro.seedAll', ['forge']).seed, run('train.repro.seedAll', ['forge']).seed);
  assert.notEqual(run('train.repro.seedAll', ['forge']).seed, run('train.repro.seedAll', ['forge1']).seed);
  // fingerprint
  const f1 = run('train.repro.fingerprint', [{ seed: 42 }]);
  const f2 = run('train.repro.fingerprint', [{ seed: 42 }]);
  const f3 = run('train.repro.fingerprint', [{ seed: 43 }]);
  assert.equal(f1.hash, f2.hash);
  assert.notEqual(f1.hash, f3.hash);
  assert.equal(f1.parts.seed, 42);
  // manifest：同配置同种子 → 同哈希；改种子 → 变
  const cfg = { model: 'ridge', layers: [2, 1] };
  const m1 = run('train.repro.manifest', [cfg, { seed: 42, dataHash: 'abc' }]);
  const m2 = run('train.repro.manifest', [cfg, { seed: 42, dataHash: 'abc' }]);
  const m3 = run('train.repro.manifest', [cfg, { seed: 43, dataHash: 'abc' }]);
  assert.equal(run('train.repro.manifestHash', [m1]), run('train.repro.manifestHash', [m2]));
  assert.notEqual(run('train.repro.manifestHash', [m1]), run('train.repro.manifestHash', [m3]));
  // 键序无关（规范序列化）
  const m4 = run('train.repro.manifest', [{ layers: [2, 1], model: 'ridge' }, { seed: 42, dataHash: 'abc' }]);
  assert.equal(run('train.repro.manifestHash', [m4]), run('train.repro.manifestHash', [m1]));
  assert.equal(m1.timestamp, null); // 默认不含时间戳
  // envInfo
  const env = run('train.repro.envInfo', []);
  assert.ok(env.node.length > 0);
  assert.equal(env.opsInDomain9, 103);
  assert.equal(env.domains, 10);
  // logAppend / experimentCompare
  let logA = [];
  logA = run('train.repro.logAppend', [logA, { runId: 'a', mse: 0.1 }]);
  logA = run('train.repro.logAppend', [logA, { runId: 'b', mse: 0.2 }]);
  assert.deepEqual(logA.map(e => e.seq), [1, 2]);
  let logB = [];
  logB = run('train.repro.logAppend', [logB, { runId: 'a', mse: 0.1 }]);
  logB = run('train.repro.logAppend', [logB, { runId: 'c', mse: 0.3 }]);
  const cmp = run('train.repro.experimentCompare', [logA, logB, undefined]);
  assert.deepEqual(cmp.common, ['a']);
  assert.deepEqual(cmp.onlyA, ['b']);
  assert.deepEqual(cmp.onlyB, ['c']);
  assert.equal(cmp.identical, false);
  assert.equal(run('train.repro.experimentCompare', [logA, logA, undefined]).identical, true);
  const logC = run('train.repro.logAppend', [[], { runId: 'a', mse: 0.15 }]);
  const cmpDiff = run('train.repro.experimentCompare', [logA, logC, undefined]);
  assert.equal(cmpDiff.diffs.length, 1);
  assert.equal(cmpDiff.diffs[0].field, 'mse');
  assert.ok(approx(cmpDiff.diffs[0].a, 0.1, 1e-12));
  assert.ok(approx(cmpDiff.diffs[0].b, 0.15, 1e-12));
});

test('determinismCheck：同种子逐位相同，不同种子不同', () => {
  const trainOnce = (seed) => {
    const { X, y } = linData(40, 7);
    return run('train.loop.run', [linearModel(1, seed), { train: { X, y } }, { epochs: 25, lr: 0.1, shuffle: true, seed }]).params;
  };
  const chk = run('train.repro.determinismCheck', [trainOnce, [42], {}]);
  assert.equal(chk.deterministic, true);
  assert.equal(chk.runs, 2);
  assert.equal(chk.mismatchAt, null);
  const chk2 = run('train.repro.determinismCheck', [trainOnce, [42], { runs: 3 }]);
  assert.equal(chk2.deterministic, true);
  // 不同种子应不同
  const p1 = trainOnce(42);
  const p2 = trainOnce(43);
  assert.notEqual(maxAbsDiff(p1.w, p2.w), 0);
  // snapshot / restore 隔离
  const snap = run('train.repro.snapshotParams', [p1]);
  const back = run('train.repro.restoreParams', [snap]);
  assert.equal(maxAbsDiff(back.w, p1.w), 0);
  back.w.data[0] = 123;
  assert.notEqual(p1.w.data[0], 123);
  assert.notEqual(snap.w.data[0], 123);
});

test('表格导出 / 数据与参数哈希 / 复现断言 / 指标存储 / bestEpoch / runSummary', () => {
  const rows = [{ a: 1, b: 2 }, { a: 'x,y', b: 'q"q' }];
  const csv = run('train.repro.tableToCsv', [rows, null]);
  assert.equal(csv.split('\n')[0], 'a,b');
  assert.ok(csv.includes('"x,y"'));
  assert.ok(csv.includes('"q""q"'));
  const md = run('train.repro.tableToMarkdown', [rows, null]);
  assert.ok(md.includes('| a | b |'));
  assert.equal(md.split('\n').length, 4);
  // hashData：逐位敏感
  const { X, y } = linData(20, 3);
  const ds = { X, y };
  const h1 = run('train.repro.hashData', [ds]);
  assert.equal(h1, run('train.repro.hashData', [ds]));
  const X2 = X.clone();
  X2.data[0] += 1e-12;
  assert.notEqual(run('train.repro.hashData', [{ X: X2, y }]), h1);
  // hashParams
  const params = { w: Tensor.fromFlat([1, 2], [2]), b: 0.5 };
  assert.equal(run('train.repro.hashParams', [params]), run('train.repro.hashParams', [run('train.repro.snapshotParams', [params])]));
  assert.notEqual(run('train.repro.hashParams', [{ w: Tensor.fromFlat([1, 2.0000001], [2]), b: 0.5 }]), run('train.repro.hashParams', [params]));
  // assertReproducible
  const m1 = run('train.repro.manifest', [{ a: 1 }, { seed: 1 }]);
  const m2 = run('train.repro.manifest', [{ a: 1 }, { seed: 1 }]);
  assert.equal(run('train.repro.assertReproducible', [m1, m2]), true);
  const m3 = run('train.repro.manifest', [{ a: 1 }, { seed: 2 }]);
  assert.throws(() => run('train.repro.assertReproducible', [m1, m3]), (e) => e.code === 'E_VALUE');
  // 指标存储（先复位）
  run('train.repro.resetState', []);
  run('train.repro.recordMetric', ['r1', 'loss', 0.5, null]);
  run('train.repro.recordMetric', ['r1', 'loss', 0.3, 2]);
  run('train.repro.recordMetric', ['r1', 'acc', 0.9, 1]);
  const series = run('train.repro.metricSeries', ['r1', 'loss']);
  assert.equal(series.length, 2);
  assert.deepEqual(series.map(s => s.value), [0.5, 0.3]);
  assert.deepEqual(series.map(s => s.step), [1, 2]);
  assert.equal(run('train.repro.metricSeries', ['r1', 'nope']).length, 0);
  run('train.repro.resetState', []);
  assert.equal(run('train.repro.metricSeries', ['r1', 'loss']).length, 0);
  // bestEpoch
  const hist = [
    { epoch: 1, valLoss: 0.5, acc: 0.8 },
    { epoch: 2, valLoss: 0.3, acc: 0.9 },
    { epoch: 3, valLoss: 0.4, acc: 0.85 },
  ];
  const be = run('train.repro.bestEpoch', [hist, 'valLoss', null]);
  assert.equal(be.epoch, 2);
  assert.ok(approx(be.value, 0.3, 1e-12));
  assert.equal(be.mode, 'min');
  const be2 = run('train.repro.bestEpoch', [hist, 'acc', 'max']);
  assert.equal(be2.epoch, 2);
  assert.ok(approx(be2.value, 0.9, 1e-12));
  // runSummary
  const sum = run('train.repro.runSummary', [{ id: 'r1', seed: 42, config: { model: 'ridge' }, metrics: { mse: 0.05 }, history: hist }]);
  assert.equal(sum.id, 'r1');
  assert.equal(sum.seed, 42);
  assert.equal(sum.epochs, 3);
  assert.ok(approx(sum.metrics.mse, 0.05, 1e-12));
  assert.equal(sum.best.epoch, 2);
  assert.ok(typeof sum.configHash === 'string' && sum.configHash.length === 8);
});

test('standardizeThenSplit / trainValTest / assertSplitDisjoint / coverageCheck', () => {
  const { X, y } = blobs({ n: 100, seed: 21 });
  const ds = { X, y };
  const r = run('train.pipe.standardizeThenSplit', [ds, 0.3, 5]);
  assert.equal(r.train.X.shape[0], 70);
  assert.equal(r.test.X.shape[0], 30);
  // scaler 仅来自 train（train 每列均值≈0）
  for (let j = 0; j < 2; j++) {
    let m = 0;
    for (let i = 0; i < 70; i++) m += r.train.X.get2(i, j);
    m /= 70;
    assert.ok(approx(m, 0, 1e-9), `train mean[${j}]=${m}`);
  }
  // r.train 已标准化：对其再 fit 应得均值≈0；scaler.mean 为原始 train 均值（非零）
  assert.ok(Array.from(run('train.data.standardizerFit', [r.train]).mean.data).every(v => Math.abs(v) < 1e-9));
  assert.ok(Array.from(r.scaler.mean.data).some(v => Math.abs(v) > 1e-6));
  assert.ok(Array.from(r.test.X.data).every(Number.isFinite));
  run('train.util.assertSplitDisjoint', [r.train.index, r.test.index]);
  // trainValTest：60/20/20
  const tv = run('train.loop.trainValTest', [ds, 0.2, 0.2, 9]);
  assert.equal(tv.train.X.shape[0], 60);
  assert.equal(tv.val.X.shape[0], 20);
  assert.equal(tv.test.X.shape[0], 20);
  run('train.util.assertSplitDisjoint', [tv.train.index, tv.val.index, tv.test.index]);
  assert.equal(run('train.util.coverageCheck', [[tv.train.index, tv.val.index, tv.test.index], 100]).ok, true);
  // 比例不等：val 0.1 / test 0.3
  const tv2 = run('train.loop.trainValTest', [ds, 0.1, 0.3, 9]);
  assert.equal(tv2.train.X.shape[0] + tv2.val.X.shape[0] + tv2.test.X.shape[0], 100);
  assert.equal(tv2.val.X.shape[0], 10);
  assert.equal(tv2.test.X.shape[0], 30);
  // assertSplitDisjoint 抛错 / 通过
  assert.throws(() => run('train.util.assertSplitDisjoint', [[0, 1, 2], [2, 3]]), (e) => e.code === 'E_VALUE');
  assert.equal(run('train.util.assertSplitDisjoint', [[0, 1], [2, 3]]), true);
  assert.equal(run('train.util.assertSplitDisjoint', [tv.train, tv.test]), true); // 带 index 的数据集
  // coverageCheck：缺失与重复
  const cov2 = run('train.util.coverageCheck', [[[0, 1], [1, 2]], 4]);
  assert.equal(cov2.ok, false);
  assert.deepEqual(cov2.missing, [3]);
  assert.deepEqual(cov2.duplicates, [1]);
  assert.equal(cov2.covered, 3);
  assert.equal(run('train.util.coverageCheck', [[0, 1, 2, 3], 4]).ok, true);
  assert.equal(run('train.util.coverageCheck', [[[], []], 0]).ok, true);
});

test('eval.fullReport / eval.nestedCV', () => {
  const yTrue = [0, 0, 0, 1, 1, 1];
  const yPred = [0, 0, 1, 1, 1, 0];
  const fr = run('train.eval.fullReport', [yTrue, yPred, {}]);
  assert.ok(approx(fr.accuracy, 4 / 6, 1e-12));
  assert.deepEqual(fr.confusion.matrix, [[2, 1], [1, 2]]);
  assert.equal(fr.perClass.length, 2);
  assert.ok(approx(fr.balancedAcc, 2 / 3, 1e-12));
  assert.ok(approx(fr.cohenKappa, 1 / 3, 1e-12));
  assert.ok(approx(fr.mcc, 1 / 3, 1e-12));
  assert.equal(fr.majorityBaseline.label, 0);
  assert.ok(fr.logloss === undefined); // 无 proba 不产出
  // 带 proba：追加 logloss / brier
  const proba = Tensor.fromNested([[0.8, 0.2], [0.7, 0.3], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.6, 0.4]]);
  const fr2 = run('train.eval.fullReport', [yTrue, yPred, { proba }]);
  assert.ok(fr2.logloss > 0 && fr2.brier > 0);
  // nestedCV：外层 4 折、内层 3 折、4 个候选
  const { X, y } = linData(60, 7);
  const nc = run('train.eval.nestedCV', [{ X, y }, {
    kOuter: 4, kInner: 3,
    candidates: [0.01, 0.1, 1, 10],
    makeFitPredict: (lam) => ridgeFitPredict(lam),
    seed: 13, metric: 'mse',
  }]);
  assert.equal(nc.outerScores.length, 4);
  assert.equal(nc.selected.length, 4);
  assert.equal(nc.mode, 'min');
  assert.equal(nc.inner.length, 4);
  assert.ok(nc.inner.every(x => x.candidates.length === 4 && x.candidates.every(c => c.scores.length === 3)));
  assert.ok(nc.outerScores.every(s => s > 0));
  assert.ok(approx(nc.mean, nc.outerScores.reduce((a, b) => a + b, 0) / 4, 1e-12));
  // 选中的正则都不应是最过度正则的 10（过拟偏差）
  assert.ok(nc.selected.every(v => v !== 10), `selected=${nc.selected}`);
});

test('exportRun / replayRun（完整实验存档与重放）', () => {
  const { X, y } = linData(40, 7);
  const ds = { X, y };
  const run1 = {
    id: 'run-1',
    config: { model: 'ridge', lambda: 0.5 },
    seed: 42,
    dataHash: run('train.repro.hashData', [ds]),
    metrics: { mse: 0.05 },
    history: [
      { epoch: 1, trainLoss: 0.2, valLoss: 0.1 },
      { epoch: 2, trainLoss: 0.1, valLoss: 0.08 },
    ],
    params: { w: Tensor.fromFlat([1.9, 1.1], [2]), b: 0.5 },
    fitPredict: ridgeFitPredict(0.5),
    data: ds,
    metric: 'mse',
  };
  const arc = run('train.repro.exportRun', [run1]);
  assert.equal(arc.schema, 'd9.run/1');
  assert.equal(arc.id, 'run-1');
  assert.equal(arc.seed, 42);
  // 纯 JSON：Tensors 已转 {shape, data}，无函数
  const json = JSON.stringify(arc);
  assert.ok(json.length > 0);
  assert.deepEqual(Object.keys(arc.params).sort(), ['b', 'w']);
  assert.deepEqual(arc.params.w.shape, [2]);
  assert.equal(arc.replay.metric > 0, true);
  assert.equal(arc.summary.epochs, 2);
  assert.equal(arc.summary.best.epoch, 2);
  // replay：哈希稳定、摘要一致、预测逐位可复现
  const rp = run('train.repro.replayRun', [arc, ridgeFitPredict(0.5)]);
  assert.equal(rp.ok, true);
  assert.equal(rp.checks.schemaOk, true);
  assert.equal(rp.checks.manifestHashStable, true);
  assert.equal(rp.checks.summaryMatches, true);
  assert.equal(rp.checks.replayDeterministic, true);
  assert.equal(rp.manifestHash, arc.manifestHash);
  // 换一个不同的 fitPredict → 预测哈希不符
  const rp2 = run('train.repro.replayRun', [arc, ridgeFitPredict(0.01)]);
  assert.equal(rp2.checks.replayDeterministic, false);
  // 无 fitPredict → 不做 replay 检查
  const rp3 = run('train.repro.replayRun', [arc, null]);
  assert.equal(rp3.checks.replayDeterministic, undefined);
  assert.equal(rp3.checks.summaryMatches, true);
  // 坏存档抛错
  assert.throws(() => run('train.repro.replayRun', [{ schema: 'nope' }, null]), (e) => e.code === 'E_VALUE');
});
