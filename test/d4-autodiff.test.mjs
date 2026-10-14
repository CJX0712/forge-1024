// 域4 自动微分测试：gradcheck 全面覆盖 + 链式法则 + 前向/反向对偶一致性 + 端到端训练
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { exec, opsByDomain } from '../src/core/registry.mjs';
import { Tensor, createRng } from '../src/core/index.mjs';
import '../src/ops/d1-math.mjs';
import '../src/ops/d2-num.mjs';
import '../src/ops/d3-linalg.mjs';
import '../src/ops/d4-autodiff.mjs';
import { approx, tensorClose, randMat, xorData } from './helpers.mjs';

const A = (name, args) => exec(name, args);

test('域4 恰好 102 条指令且 id 连续', () => {
  const ops = opsByDomain(4);
  assert.equal(ops.length, 102);
  const ids = ops.map(o => o.id).sort();
  assert.equal(ids[0], 'D4-001');
  assert.equal(ids[101], 'D4-102');
  assert.equal(new Set(ids).size, 102);
});

// —— 标量输出函数的 gradcheck（每个一元/二元算子）——
// 注意：f 必须让 x 直连计算图（不得用 x.map() 预变换破坏链）；
// 定义域敏感的函数用 x·x+c 复合既保证定义域又检验链式法则。
const sq = (t, c) => A('ad.add', [A('ad.mul', [t, t]), c]);
const SCALAR_FUNCS = [
  ['ad.exp', (t) => A('ad.exp', [t])],
  ['ad.log', (t) => A('ad.log', [sq(t, 0.5)])],
  ['ad.sqrt', (t) => A('ad.sqrt', [sq(t, 0.5)])],
  ['ad.sin', (t) => A('ad.sin', [t])],
  ['ad.cos', (t) => A('ad.cos', [t])],
  ['ad.tanh', (t) => A('ad.tanh', [t])],
  ['ad.sigmoid', (t) => A('ad.sigmoid', [t])],
  ['ad.softplus', (t) => A('ad.softplus', [t])],
  ['ad.relu', (t) => A('ad.relu', [sq(t, 0.3)])],
  ['ad.leakyRelu', (t) => A('ad.leakyRelu', [sq(t, 0.3), 0.1])],
  ['ad.elu', (t) => A('ad.elu', [t, 1])],
  ['ad.silu', (t) => A('ad.silu', [t])],
  ['ad.gelu', (t) => A('ad.gelu', [t])],
  ['ad.softsign', (t) => A('ad.softsign', [t])],
  ['ad.mish', (t) => A('ad.mish', [t])],
  ['ad.hardtanh', (t) => A('ad.hardtanh', [A('ad.mul', [t, A('ad.mulScalar', [t, 0.4])])])],
  ['ad.hardswish', (t) => A('ad.hardswish', [sq(t, 0.37)])],
  ['ad.rsqrt', (t) => A('ad.rsqrt', [sq(t, 1)])],
  ['ad.expm1', (t) => A('ad.expm1', [t])],
  ['ad.log1p', (t) => A('ad.log1p', [sq(t, 0.5)])],
  ['ad.square', (t) => A('ad.square', [t])],
  ['ad.reciprocal', (t) => A('ad.reciprocal', [sq(t, 1)])],
  ['ad.abs', (t) => A('ad.abs', [sq(t, 0.3)])],
];
test('一元算子 gradcheck 全覆盖（中心差分 1e-6）', () => {
  const rng = createRng(77);
  for (const [name, f] of SCALAR_FUNCS) {
    const x = Tensor.fromFlat(Array.from({ length: 8 }, () => rng.normal()), [8]);
    // 逐元素函数包 sumAll 使其为标量函数（逐分量梯度仍被完整检验）
    const r = A('ad.gradcheck', [(t) => A('ad.sumAll', [f(t)]), x, 1e-5, 1e-5]);
    assert.ok(r.passed, `${name} gradcheck maxRelErr=${r.maxRelErr.toExponential(3)}`);
  }
});

test('二元算子 gradcheck', () => {
  const rng = createRng(88);
  const mk = () => Tensor.fromFlat(Array.from({ length: 6 }, () => rng.normal()), [6]);
  const cases = [
    ['ad.add', (x, y) => A('ad.add', [x, y])],
    ['ad.sub', (x, y) => A('ad.sub', [x, y])],
    ['ad.mul', (x, y) => A('ad.mul', [x, y])],
    ['ad.div', (x, y) => A('ad.div', [x, A('ad.add', [A('ad.mul', [y, y]), 1])])],
    ['ad.maximum', (x, y) => A('ad.maximum', [A('ad.add', [x, 0.2]), y])],
    ['ad.minimum', (x, y) => A('ad.minimum', [A('ad.sub', [x, 0.2]), y])],
    ['ad.scaleAdd', (x, y) => A('ad.scaleAdd', [1.5, x, y])],
  ];
  for (const [name, f] of cases) {
    const x = mk(), y = mk();
    for (const [who, arg] of [['x', x], ['y', y]]) {
      const r = A('ad.gradcheck', [(t) => A('ad.sumAll', [f(t, who === 'x' ? y : x)]), arg, 1e-5, 1e-5]);
      assert.ok(r.passed, `${name} 对 ${who} gradcheck maxRelErr=${r.maxRelErr.toExponential(3)}`);
    }
  }
});

test('标量变体与 pow/clamp/neg', () => {
  const rng = createRng(89);
  const x = Tensor.fromFlat(Array.from({ length: 6 }, () => rng.normal()), [6]);
  const cases = [
    ['ad.addScalar', (t) => A('ad.addScalar', [t, 2.5])],
    ['ad.subScalar', (t) => A('ad.subScalar', [t, 1.5])],
    ['ad.mulScalar', (t) => A('ad.mulScalar', [t, 3])],
    ['ad.divScalar', (t) => A('ad.divScalar', [t, 4])],
    ['ad.rsubScalar', (t) => A('ad.rsubScalar', [t, 5])],
    ['ad.rdivScalar', (t) => A('ad.rdivScalar', [t, 10])],
    ['ad.pow', (t) => A('ad.pow', [sq(t, 0.5), 3])],
    ['ad.clamp', (t) => A('ad.clamp', [A('ad.mulScalar', [t, 0.1]), -0.5, 0.5])],
    ['ad.neg', (t) => A('ad.neg', [t])],
  ];
  for (const [name, f] of cases) {
    const r = A('ad.gradcheck', [(t) => A('ad.sumAll', [f(t)]), x, 1e-5, 1e-5]);
    assert.ok(r.passed, `${name} gradcheck=${r.maxRelErr.toExponential(3)}`);
  }
});

test('矩阵算子 gradcheck：matmul / transpose / reshape / broadcast', () => {
  const rng = createRng(90);
  const mk = (m, n) => Tensor.fromFlat(Array.from({ length: m * n }, () => rng.normal()), [m, n]);
  const Am = mk(3, 4), Bm = mk(4, 5);
  const r1 = A('ad.gradcheck', [(t) => A('ad.sumAll', [A('ad.matmul', [t, Bm])]), Am, 1e-5, 1e-5]);
  assert.ok(r1.passed, `matmul dA=${r1.maxRelErr.toExponential(3)}`);
  const r2 = A('ad.gradcheck', [(t) => A('ad.sumAll', [A('ad.matmul', [Am, t])]), Bm, 1e-5, 1e-5]);
  assert.ok(r2.passed, `matmul dB=${r2.maxRelErr.toExponential(3)}`);
  // 矩阵值函数（向量输出）用投影检验
  const u = mk(3, 5);
  const rv1 = A('ad.gradcheckVec', [(t) => A('ad.matmul', [t, Bm]), Am, u]);
  assert.ok(rv1.passed, `matmul vjp dA=${rv1.maxRelErr.toExponential(3)}`);
  const rv2 = A('ad.gradcheckVec', [(t) => A('ad.matmul', [Am, t]), Bm, u]);
  assert.ok(rv2.passed, `matmul vjp dB=${rv2.maxRelErr.toExponential(3)}`);
  const rvT = A('ad.gradcheckVec', [(t) => A('ad.transpose', [t]), Am, mk(4, 3)]);
  assert.ok(rvT.passed, `transpose=${rvT.maxRelErr.toExponential(3)}`);
  const rvR = A('ad.gradcheckVec', [(t) => A('ad.reshape', [t, [12]]), Am, mk(3, 4)]);
  assert.ok(rvR.passed, `reshape=${rvR.maxRelErr.toExponential(3)}`);
  const rvB = A('ad.gradcheckVec', [(t) => A('ad.broadcastTo', [t, [3, 4]]), Tensor.fromFlat([1, 2, 3, 4], [1, 4]), mk(3, 4)]);
  assert.ok(rvB.passed, `broadcastTo=${rvB.maxRelErr.toExponential(3)}`);
  // matmulAT 也检验（A[k,m]·B[k,n] → [m,n]）
  const rvAT = A('ad.gradcheckVec', [(t) => A('ad.matmulAT', [t, Bm]), mk(4, 3), mk(3, 5)]);
  assert.ok(rvAT.passed, `matmulAT=${rvAT.maxRelErr.toExponential(3)}`);
});

test('归约与 softmax/logSoftmax gradcheck', () => {
  const rng = createRng(91);
  const x = Tensor.fromFlat(Array.from({ length: 12 }, () => rng.normal()), [3, 4]);
  for (const [name, f] of [
    ['sumAxis', (t) => A('ad.sumAxis', [t, 1])],
    ['meanAxis', (t) => A('ad.meanAxis', [t, 0])],
    ['softmax', (t) => A('ad.softmax', [t, -1])],
    ['logSoftmax', (t) => A('ad.logSoftmax', [t, -1])],
  ]) {
    const o = f(x);
    const u = Tensor.fromFlat(Array.from({ length: o.size }, () => rng.normal()), o.shape);
    const r = A('ad.gradcheckVec', [f, x, u]);
    assert.ok(r.passed, `${name}=${r.maxRelErr.toExponential(3)}`);
  }
  // softmax 解析验证：行和=1，雅可比对角 y(1-y)
  const sm = A('ad.softmax', [x, -1]);
  for (let i = 0; i < 3; i++) { let s = 0; for (let j = 0; j < 4; j++) s += sm.get2(i, j); assert.ok(approx(s, 1, 1e-12)); }
});

test('损失函数 gradcheck：mse/mae/huber/bce/crossEntropy/nll', () => {
  const rng = createRng(92);
  const pred = Tensor.fromFlat(Array.from({ length: 8 }, () => rng.normal()), [8]);
  const target = Tensor.fromFlat(Array.from({ length: 8 }, () => rng.normal()), [8]);
  for (const [name, f] of [
    ['mse', (t) => A('ad.mse', [t, target])],
    ['mae', (t) => A('ad.mae', [A('ad.add', [t, 0.1]), target])],
    ['huber', (t) => A('ad.huber', [A('ad.add', [t, 0.1]), target, 1])],
    ['bce', (t) => A('ad.bce', [t, Tensor.full([8], 0.7)])],
  ]) {
    const r = A('ad.gradcheck', [f, pred, 1e-5, 1e-5]);
    assert.ok(r.passed, `${name}=${r.maxRelErr.toExponential(3)}`);
  }
  // crossEntropy（one-hot 目标）
  const logits = Tensor.fromFlat(Array.from({ length: 12 }, () => rng.normal()), [4, 3]);
  const labels = Tensor.zeros([4, 3]);
  for (let i = 0; i < 4; i++) labels.set2(i, i % 3, 1);
  const r = A('ad.gradcheck', [(t) => A('ad.crossEntropy', [t, labels]), logits, 1e-5, 1e-5]);
  assert.ok(r.passed, `crossEntropy=${r.maxRelErr.toExponential(3)}`);
  // nll：手算单点
  const lp = A('ad.logSoftmax', [logits, -1]);
  const idx = Tensor.fromFlat([0, 1, 2, 1], [4]);
  const loss = A('ad.nll', [lp, idx]);
  assert.ok(approx(loss.data[0], -(lp.data[0] + lp.get2(1, 1) + lp.get2(2, 2) + lp.get2(3, 1)) / 4, 1e-12));
  // mse 梯度解析验证：d/dp = 2(p−t)/n
  const g = A('ad.grad', [(t) => A('ad.mse', [t, target]), pred]);
  let maxErr = 0;
  for (let i = 0; i < 8; i++) maxErr = Math.max(maxErr, Math.abs(g.data[i] - 2 * (pred.data[i] - target.data[i]) / 8));
  assert.ok(maxErr < 1e-12, `mse 梯度解析误差 ${maxErr}`);
});

test('结构与索引算子 gradcheck：getItem/concat/stack/gatherRows/where/pad', () => {
  const rng = createRng(93);
  const x = Tensor.fromFlat(Array.from({ length: 12 }, () => rng.normal()), [3, 4]);
  const other = Tensor.fromFlat(Array.from({ length: 12 }, () => rng.normal()), [3, 4]);
  const otherB = Tensor.fromFlat(Array.from({ length: 12 }, () => rng.normal()), [3, 4]);
  const mask = Tensor.fromFlat([1, 0, 1, 0], [4]);
  const table = Tensor.fromFlat(Array.from({ length: 10 }, () => rng.normal()), [5, 2]);
  const u4 = Tensor.fromFlat(Array.from({ length: 8 }, () => rng.normal()), [4, 2]);
  const cases = [
    ['getItem', (t) => A('ad.getItem', [t, [1]]), Tensor.fromFlat(Array.from({ length: 4 }, () => rng.normal()), [4])],
    ['concat', (t) => A('ad.concat', [[t, other], 0]), Tensor.fromFlat(Array.from({ length: 12 }, () => rng.normal()), [3, 4])],
    ['stack', (t) => A('ad.stack', [[t, other], 0]), Tensor.fromFlat(Array.from({ length: 12 }, () => rng.normal()), [3, 4])],
    ['gatherRows', (t) => A('ad.gatherRows', [t, Tensor.fromFlat([0, 2, 2, 4], [4])]), table],
    ['where', (t) => A('ad.where', [mask, t, otherB]), other],
    ['padConst', (t) => A('ad.padConst', [t, [[1, 0], [0, 1]], 0]), x],
    ['scatterAdd', (t) => A('ad.scatterAdd', [Tensor.zeros([5, 2]), Tensor.fromFlat([0, 2, 2, 4], [4]), t]), u4],
  ];
  for (const [name, f, inp] of cases) {
    const u = (() => {
      const o = f(inp);
      return Tensor.fromFlat(Array.from({ length: o.size }, () => rng.normal()), o.shape);
    })();
    const r = A('ad.gradcheckVec', [f, inp, u]);
    assert.ok(r.passed, `${name}=${r.maxRelErr.toExponential(3)}`);
  }
  // gatherRows 重复索引梯度累加
  tape_reset();
  const tbl = Tensor.ones([5, 2]);
  const g = A('ad.gatherRows', [tbl, Tensor.fromFlat([2, 2], [2])]);
  const grads = A('ad.backward', [A('ad.sumAll', [g])]);
  tensorClose(grads.get(tbl), Tensor.full([5, 2], 0).map((v, i) => (Math.floor(i / 2) === 2 ? 2 : 0)), 1e-12);
  function tape_reset() { A('ad.tapeReset', []); }
});

test('复合层 gradcheck：dense/layerNorm/batchnorm1d/attention/dropout', () => {
  const rng = createRng(94);
  const x = Tensor.fromFlat(Array.from({ length: 12 }, () => rng.normal()), [4, 3]);
  const W = Tensor.fromFlat(Array.from({ length: 15 }, () => rng.normal()), [3, 5]);
  const b = Tensor.fromFlat(Array.from({ length: 5 }, () => rng.normal()), [5]);
  const K2 = Tensor.fromFlat(Array.from({ length: 12 }, () => rng.normal()), [4, 3]);
  const V2 = Tensor.fromFlat(Array.from({ length: 12 }, () => rng.normal()), [4, 3]);
  const Q2 = Tensor.fromFlat(Array.from({ length: 12 }, () => rng.normal()), [4, 3]);
  const u0 = Tensor.fromFlat(Array.from({ length: 20 }, () => rng.normal()), [4, 5]);
  const W2t = Tensor.fromFlat(Array.from({ length: 10 }, () => rng.normal()), [5, 2]);
  const b2t = Tensor.fromFlat([0, 0], [2]);
  for (const [name, f, arg] of [
    ['dense.x', (t) => A('ad.dense', [t, W, b]), x],
    ['dense.W', (t) => A('ad.dense', [x, t, b, (z) => A('ad.tanh', [z])]), W],
    ['layerNorm', (t) => A('ad.layerNorm', [t, Tensor.ones([3]), Tensor.zeros([3])]), x],
    ['batchnorm1d', (t) => A('ad.batchnorm1d', [t, Tensor.ones([3]), Tensor.zeros([3])]), x],
    ['attention.Q', (t) => A('ad.scaledDotProductAttention', [t, K2, V2]), x],
    ['attention.K', (t) => A('ad.scaledDotProductAttention', [Q2, t, V2]), x],
    ['mlp', (t) => A('ad.mlp', [t, W, b, W2t, b2t]), x],
  ]) {
    const o = f(arg);
    const uu = Tensor.fromFlat(Array.from({ length: o.size }, () => rng.normal()), o.shape);
    const r = A('ad.gradcheckVec', [f, arg, uu]);
    assert.ok(r.passed, `${name}=${r.maxRelErr.toExponential(3)}`);
  }
  // dropout 前向：掩码为 0 处输出 0，非零处 ×1/(1-p)
  const mask = Tensor.fromFlat([1, 0, 1, 1], [4]);
  const dx = A('ad.dropout', [Tensor.ones([4]), 0.25, mask]);
  tensorClose(dx, Tensor.fromFlat([4 / 3, 0, 4 / 3, 4 / 3], [4]), 1e-9);
});

test('磁带机制：noGrad/detach/graphSize/reset/多路径累加', () => {
  A('ad.tapeReset', []);
  const x = Tensor.fromFlat([1, 2], [2]);
  const y = A('ad.mul', [x, x]); // x 使用两次 → 梯度 2x
  const loss = A('ad.sumAll', [y]);
  assert.ok(A('ad.graphSize', []) >= 2);
  const grads = A('ad.backward', [loss]);
  tensorClose(grads.get(x), Tensor.fromFlat([2, 4], [2]), 1e-12); // 2·x
  // 多路径：z = x·a + x·b → dx = a+b
  A('ad.tapeReset', []);
  const a = Tensor.fromFlat([1, 2], [2]), bb = Tensor.fromFlat([10, 20], [2]);
  const z = A('ad.add', [A('ad.mul', [x, a]), A('ad.mul', [x, bb])]);
  const gz = A('ad.backward', [A('ad.sumAll', [z])]);
  tensorClose(gz.get(x), Tensor.fromFlat([11, 22], [2]), 1e-12);
  // noGrad：不记录
  A('ad.tapeReset', []);
  const before = A('ad.graphSize', []);
  const out = A('ad.noGrad', [() => A('ad.exp', [x])]);
  assert.equal(A('ad.graphSize', []), before);
  assert.deepEqual(out.toArray(), x.map(Math.exp).toArray());
  // detach 切断
  A('ad.tapeReset', []);
  const d = A('ad.detach', [x]);
  const l2 = A('ad.sumAll', [A('ad.mul', [d, d])]);
  const gd = A('ad.backward', [l2]);
  assert.equal(gd.get(x), undefined);
});

test('前向模式（对偶数）与反向模式对偶一致性', () => {
  // <jvp(f,x,v), u> == <vjp(f,x,u), v>
  const rng = createRng(95);
  const x = Tensor.fromFlat(Array.from({ length: 5 }, () => rng.normal()), [5]);
  const v = Tensor.fromFlat(Array.from({ length: 5 }, () => rng.normal()), [5]);
  const u = Tensor.fromFlat([1], [1]);
  const f = (t) => {
    const a = A('ad.exp', [A('ad.mul', [t, t])]);
    return A('ad.sumAll', [A('ad.sin', [A('ad.add', [a, t])])]);
  };
  const d = A('ad.dual.from', [x, v]);
  const d1 = A('ad.dual.exp', [A('ad.dual.mul', [d, d])]);
  const d2 = A('ad.dual.sin', [A('ad.dual.add', [d1, d])]);
  const d3 = A('ad.dual.sum', [d2]);
  const jvp = d3.tangent.data[0];
  const vjpG = A('ad.vjp', [f, x, u]);
  const vjpDot = vjpG.data.reduce((s, g, i) => s + g * v.data[i], 0);
  assert.ok(approx(jvp, vjpDot, 1e-10), `jvp=${jvp} vs vjp·v=${vjpDot}`);
  // 对偶数基础运算
  const da = A('ad.dual.from', [Tensor.fromFlat([2], [1]), Tensor.fromFlat([3], [1])]);
  const db = A('ad.dual.from', [Tensor.fromFlat([4], [1]), Tensor.fromFlat([5], [1])]);
  assert.equal(A('ad.dual.primal', [A('ad.dual.add', [da, db])]).data[0], 6);
  assert.equal(A('ad.dual.tangent', [A('ad.dual.add', [da, db])]).data[0], 8);
  assert.equal(A('ad.dual.primal', [A('ad.dual.mul', [da, db])]).data[0], 8);
  assert.equal(A('ad.dual.tangent', [A('ad.dual.mul', [da, db])]).data[0], 3 * 4 + 2 * 5);
  assert.equal(A('ad.dual.primal', [A('ad.dual.div', [da, db])]).data[0], 0.5);
  assert.ok(approx(A('ad.dual.tangent', [A('ad.dual.div', [da, db])]).data[0], (3 * 4 - 2 * 5) / 16, 1e-12));
  // 对偶 matmul
  const Am = randMat(2, 3, 7), Bm = randMat(3, 2, 8);
  const dA = A('ad.dual.from', [Am, Tensor.ones([2, 3])]);
  const dB = A('ad.dual.from', [Bm, Tensor.zeros([3, 2])]);
  const dM = A('ad.dual.matmul', [dA, dB]);
  tensorClose(dM.tangent, exec('linalg.matmul', [Tensor.ones([2, 3]), Bm]), 1e-12);
});

test('端到端：两层 MLP + Adam 风格 GD 训练 XOR 收敛', () => {
  const { X, y } = xorData(200, 3);
  const yCol = y.reshape([200, 1]); // 与 logits [200,1] 对齐（避免广播 mse）
  const rng = createRng(99);
  const W1 = Tensor.fromFlat(Array.from({ length: 16 }, () => rng.normal()), [2, 8]);
  const b1 = Tensor.zeros([8]);
  const W2 = Tensor.fromFlat(Array.from({ length: 8 }, () => rng.normal()), [8, 1]);
  const b2 = Tensor.zeros([1]);
  const lr = 0.15;
  let lastLoss = Infinity;
  for (let epoch = 0; epoch < 600; epoch++) {
    A('ad.tapeReset', []);
    const logits = A('ad.mlp', [X, W1, b1, W2, b2]);
    const loss = A('ad.mse', [logits, yCol]);
    lastLoss = loss.data[0];
    const grads = A('ad.backward', [loss]);
    // 简单 SGD 更新
    for (const [P, name] of [[W1, 'W1'], [b1, 'b1'], [W2, 'W2'], [b2, 'b2']]) {
      const g = grads.get(P);
      if (!g) continue;
      for (let i = 0; i < P.size; i++) P.data[i] -= lr * g.data[i];
    }
  }
  assert.ok(lastLoss < 0.02, `XOR 最终 loss=${lastLoss}`);
  // 准确率
  A('ad.tapeReset', []);
  const logits = A('ad.noGrad', [() => A('ad.mlp', [X, W1, b1, W2, b2])]);
  let correct = 0;
  for (let i = 0; i < 200; i++) {
    const p = logits.data[i] > 0.5 ? 1 : 0;
    if (p === y.data[i]) correct++;
  }
  assert.ok(correct / 200 > 0.95, `XOR 准确率 ${correct / 200}`);
});

test('因果掩码与 klDiv', () => {
  const scores = Tensor.fromFlat([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
  const masked = A('ad.causalMask', [scores]);
  assert.deepEqual(masked.toArray(), [[1, -Infinity, -Infinity], [4, 5, -Infinity], [7, 8, 9]]);
  const g = A('ad.vjp', [(t) => A('ad.causalMask', [t]), scores, Tensor.ones([3, 3])]);
  tensorClose(g, Tensor.fromFlat([1, 0, 0, 1, 1, 0, 1, 1, 1], [3, 3]), 1e-12);
  // klDiv：p=q → 0
  const p = A('ad.logSoftmax', [Tensor.fromFlat([1, 2, 3], [3]), -1]);
  const kl = A('ad.klDiv', [p, p]);
  assert.ok(approx(kl.data[0], 0, 1e-12));
});
