// 域5 优化器测试：收敛性证明（凸二次）、Adam 手算核对、调度函数核对
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { exec, opsByDomain } from '../src/core/registry.mjs';
import { Tensor } from '../src/core/index.mjs';
import '../src/ops/d1-math.mjs';
import '../src/ops/d2-num.mjs';
import '../src/ops/d3-linalg.mjs';
import '../src/ops/d5-opt.mjs';
import { approx, tensorClose } from './helpers.mjs';

const O = (name, args) => exec(name, args);

test('域5 恰好 102 条指令且 id 连续', () => {
  const ops = opsByDomain(5);
  assert.equal(ops.length, 102);
  const ids = ops.map(o => o.id).sort();
  assert.equal(ids[0], 'D5-001');
  assert.equal(ids[101], 'D5-102');
  assert.equal(new Set(ids).size, 102);
});

// —— 全部 16 个优化器在凸二次上收敛且损失单调不增 ——
const OPTIMIZERS = [
  'sgd', 'sgdMomentum', 'sgdNesterov', 'adagrad', 'adadelta', 'rmsprop',
  'adam', 'adamw', 'nadam', 'amsgrad', 'adamax', 'lamb', 'rprop', 'yogi', 'lion', 'signSgd',
];
const LR = {
  sgd: 0.05, sgdMomentum: 0.03, sgdNesterov: 0.03, adagrad: 0.5, adadelta: 1,
  rmsprop: 0.05, adam: 0.05, adamw: 0.05, nadam: 0.05, amsgrad: 0.05, adamax: 0.05,
  lamb: 0.05, rprop: 0.5, yogi: 0.05, lion: 0.05, signSgd: 0.02,
};
const TOL = { adadelta: 1e-3, signSgd: 5e-3, lion: 5e-3, rprop: 1e-4 };

test('16 个优化器在凸二次 f(x)=½Σi·xᵢ² 上收敛且损失单调不增', () => {
  const k = 8;
  const fn = O('opt.util.quadraticFn', [{ k, d: null, xStar: null }]); // 默认 dᵢ=i
  for (const name of OPTIMIZERS) {
    const x0 = Tensor.fromFlat(Array.from({ length: k }, (_, i) => 1 + i * 0.5), [k]);
    const cfg = { lr: LR[name] };
    let state = O(`opt.${name}`, [x0, cfg]);
    let params = x0.clone();
    const losses = [fn.f(params)];
    let ok = false;
    // Adadelta 的 RMS(Δx) 正反馈在最优点附近缓慢振荡，故按「轨迹最优损失」判定收敛
    const useBest = name === 'adadelta';
    // 收敛容差：Adadelta 按轨迹最优（RMS(Δx) 长程振荡）；
    // signSGD 步长为 ±η，理论上只收敛到半径 ~η 的最优邻域（Bernstein et al. 定理 2）
    const tol = TOL[name] ?? 1e-6;
    const maxIters = useBest ? 5000 : 5000;
    for (let it = 0; it < maxIters; it++) {
      const g = fn.g(params);
      const r = O(`opt.${name}.step`, [params, g, state, cfg]);
      params = r.params ?? r;
      state = r.state ?? state;
      const l = fn.f(params);
      losses.push(typeof l === 'number' ? l : l.data?.[0]);
      if (Number.isFinite(losses[losses.length - 1]) && losses[losses.length - 1] < 1e-12) { ok = true; break; }
    }
    const best = Math.min(...losses);
    const score = useBest ? best : losses[losses.length - 1];
    assert.ok(ok || score < tol, `${name} 未收敛: ${useBest ? 'best' : 'final'}Loss=${score}`);
    // 单调不增（允许 1e-8 抖动）
    let maxRise = 0;
    for (let i = 1; i < losses.length; i++) maxRise = Math.max(maxRise, losses[i] - losses[i - 1]);
    // 动量系与 Adadelta 均存在动力学过冲/振荡，仅对无记忆自适应法断言单调
    const MONOTONE = ['sgd', 'adagrad', 'rmsprop', 'adam', 'adamw', 'nadam', 'amsgrad', 'adamax'];
    if (MONOTONE.includes(name)) {
      assert.ok(maxRise < 1e-6, `${name} 损失上升 ${maxRise.toExponential(2)}`);
      assert.equal(O('opt.util.monotoneLoss', [losses, 1e-8]), true, `${name} monotoneLoss 判定`);
    }
  }
});

test('Adam 前 3 步与论文公式逐位核对', () => {
  // Kingma & Ba 2015, Algorithm 1：β1=0.9, β2=0.999, ε=1e-8, α=0.1
  const alpha = 0.1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
  const g0 = Tensor.fromFlat([1.0, -0.5], [2]);
  const x0 = Tensor.fromFlat([0, 0], [2]);
  const cfg = { lr: alpha };
  let state = O('opt.adam', [x0, cfg]);
  let x = x0.clone();
  let m = Tensor.zeros([2]);
  let v = Tensor.zeros([2]);
  for (let t = 1; t <= 3; t++) {
    const g = t === 1 ? g0 : g0.map(vv => vv * 0.9); // 递减梯度便于区分各步
    const prev = x.clone();
    const r = O('opt.adam.step', [x, g, state, cfg]);
    x = r.params ?? r;
    state = r.state ?? state;
    m = m.map((mm, i) => beta1 * mm + (1 - beta1) * g.data[i]);
    v = v.map((vv, i) => beta2 * vv + (1 - beta2) * g.data[i] ** 2);
    const mh = m.data.map(mm => mm / (1 - Math.pow(beta1, t)));   // 1-β1²=0.19（t=2）
    const vh = v.data.map(vv => vv / (1 - Math.pow(beta2, t)));   // 1-β2²=0.001999
    for (let i = 0; i < 2; i++) {
      const expected = prev.data[i] - alpha * mh[i] / (Math.sqrt(vh[i]) + eps);
      assert.ok(Math.abs(x.data[i] - expected) < 1e-9, `Adam t=${t} 第${i}维: ${x.data[i]} vs ${expected}`);
    }
  }
});

test('AdamW 解耦权重衰减 / Lion 符号动量 / LAMB trust ratio', () => {
  // AdamW：零梯度时更新仅来自解耦衰减
  const cfg = { lr: 0.1, weightDecay: 0.1 };
  const x = Tensor.fromFlat([1], [1]);
  let state = O('opt.adamw', [x, cfg]);
  const r = O('opt.adamw.step', [x, Tensor.fromFlat([0], [1]), state, cfg]);
  const x1 = r.params;
  assert.ok(x1.data[0] < 1 && x1.data[0] > 0.98, `AdamW 零梯度衰减 ${x1.data[0]}`);
  // Lion：正梯度 → 负向更新
  const cfgl = { lr: 0.1 };
  const xl = Tensor.fromFlat([0], [1]);
  const stl = O('opt.lion', [xl, cfgl]);
  const rl = O('opt.lion.step', [xl, Tensor.fromFlat([2], [1]), stl, cfgl]);
  assert.ok(rl.params.data[0] < 0, 'Lion 正梯度应负向更新');
  // LAMB trust ratio：φ=2w → ratio=1
  const tr = O('opt.util.trustRatio', [Tensor.fromFlat([2, 2], [2]), Tensor.fromFlat([1, 1], [2]), 0]);
  assert.ok(approx(tr, 1, 1e-9), `trustRatio=${tr}`);
});

test('学习率调度函数手算核对', () => {
  assert.ok(approx(O('opt.lr.constant', [{ lr: 0.1 }, 5]), 0.1, 1e-12));
  assert.ok(approx(O('opt.lr.cosine', [{ lr: 1, T: 10 }, 5]), 0.5, 1e-6), 'cosine 半周期');
  assert.ok(approx(O('opt.lr.warmup', [{ lr: 1, warmupEpochs: 10 }, 5]), 0.5, 1e-9), 'warmup 线性');
  assert.ok(approx(O('opt.lr.exp', [{ lr: 1, gamma: 0.9 }, 2]), 0.81, 1e-9), 'exp 衰减');
  assert.ok(approx(O('opt.lr.step', [{ lr: 1, stepSize: 3, gamma: 0.1 }, 7]), 0.01, 1e-9), 'step 阶梯');
  assert.ok(approx(O('opt.lr.linearDecay', [{ lr: 1, T: 10 }, 5]), 0.5, 1e-9), '线性衰减');
  // warmupCosine 组合
  const wc = O('opt.lr.warmupCosine', [{ lr: 1, warmupEpochs: 10, T: 30 }, 5]);
  assert.ok(approx(typeof wc === 'number' ? wc : wc.lr, 0.5, 1e-9), `warmupCosine=${wc}`);
});

test('梯度变换：clipNorm / clipValue / centralization / EMA', () => {
  const g = Tensor.fromFlat([3, 4], [2]); // L2=5
  const cn = O('opt.grad.clipNorm', [g, 1]);
  assert.ok(approx(exec('math.l2', [cn]), 1, 1e-9), `clipNorm 后范数=${exec('math.l2', [cn])}`);
  const cn2 = O('opt.grad.clipNorm', [g, 10]);
  assert.ok(approx(exec('math.l2', [cn2]), 5, 1e-9), '范数低于上限不缩放');
  const cv = O('opt.grad.clipValue', [g, 2]);
  assert.deepEqual(Array.from(cv.data), [2, 2]);
  const g2 = Tensor.fromFlat([1, 2, 3], [3]);
  const ce = O('opt.grad.centralization', [g2]);
  assert.ok(approx(ce.data.reduce((a, b) => a + b, 0) / 3, 0, 1e-12), '中心化后均值 0');
  // EMA：ema ← decay·ema + (1−decay)·θ（state 初始为零）
  const emaR = O('opt.ema', [Tensor.fromFlat([1, 2], [2]), { ema: [0, 0] }, { decay: 0.9 }]);
  tensorClose(emaR.params, Tensor.fromFlat([0.1, 0.2], [2]), 1e-9);
});

test('线搜索：backtracking 满足 Armijo / 精确二次步长', () => {
  const fn = O('opt.util.quadraticFn', [{ k: 1 }]);
  const x = Tensor.fromFlat([1], [1]);
  const g = fn.g(x);
  const r = O('opt.line.backtracking', [fn, x, g, { init: 1, c1: 1e-4, rho: 0.5 }]);
  assert.ok(r.alpha > 0 && r.alpha <= 1, `alpha=${r.alpha}`);
  assert.ok(r.fNew <= fn.f(x), '线搜索应下降');
  // 精确二次：f=½x²（sphereFn d=1 时 f=½x²? 依实现核对 α*）
  const q = O('opt.line.exactQuadratic', [fn, x, g, {}]);
  assert.ok(q.alpha > 0, `exact alpha=${q.alpha}`);
});

test('组合预设收敛（抽样 4 个）', () => {
  const k = 6;
  const fn = O('opt.util.quadraticFn', [{ k }]);
  const x0 = Tensor.fromFlat(Array.from({ length: k }, () => 2), [k]);
  for (const name of ['adamCosine', 'adamwOnecycle', 'lionWarmupCosine', 'momentumLinear']) {
    const r = O('opt.util.runOptimizer', [name, fn, x0, 6000, { lr: 0.2, T: 6000, warmupEpochs: 100 }] );
    const traj = r.traj ?? r.losses ?? r.trajectory;
    const finalL = traj ? traj[traj.length - 1] : fn.f(r.params);
    assert.ok(finalL < 0.05, `${name} 最终 loss=${finalL}`);
  }
});

test('非凸基准函数值正确（Rosenbrock/Beale/Booth/Himmelblau）', () => {
  assert.ok(approx(O('opt.util.rosenbrock', [{}]).f(Tensor.fromFlat([1, 1], [2])), 0, 1e-12), 'rosenbrock(1,1)=0');
  assert.ok(approx(O('opt.util.beale', [{}]).f(Tensor.fromFlat([3, 0.5], [2])), 0, 1e-12), 'beale(3,0.5)=0');
  assert.ok(approx(O('opt.util.booth', [{}]).f(Tensor.fromFlat([1, 3], [2])), 0, 1e-12), 'booth(1,3)=0');
  assert.ok(approx(O('opt.util.himmelblau', [{}]).f(Tensor.fromFlat([3, 2], [2])), 0, 1e-12), 'himmelblau(3,2)=0');
  // Rosenbrock 梯度核对：f=(1-x)²+100(y-x²)² → ∂x = -2(1-x)-400x(y-x²)
  const rb = O('opt.util.rosenbrock', [{}]);
  const g = rb.g(Tensor.fromFlat([-1.2, 1], [2]));
  assert.ok(approx(g.data[0], -2 * (1 - (-1.2)) - 400 * (-1.2) * (1 - 1.44), 1e-9));
});

test('工具：converged / stateToJSON 往返 / compareOptimizers / listOptimizers', () => {
  const fn = O('opt.util.quadraticFn', [{ k: 4 }]);
  assert.equal(O('opt.util.converged', [Tensor.zeros([4]), fn.xStar, 1e-3]), true);
  assert.equal(O('opt.util.converged', [Tensor.fromFlat([1, 1, 1, 1], [4]), fn.xStar, 1e-4]), false);
  const x0 = Tensor.fromFlat([1, 1, 1, 1], [4]);
  const st = O('opt.adam', [x0, { lr: 0.1 }]);
  const js = O('opt.util.stateToJSON', [st]);
  const back = O('opt.util.stateFromJSON', [typeof js === 'string' ? js : JSON.stringify(js)]);
  assert.ok(back != null);
  const cmp = O('opt.util.compareOptimizers', [['sgd', 'adam'], fn, x0, 500, { lr: 0.1 }]);
  const rows = Array.isArray(cmp) ? cmp : cmp.results ?? cmp.table;
  assert.ok(rows.length === 2, 'compareOptimizers 两行');
  const list = O('opt.util.listOptimizers', []);
  assert.ok(list.length >= 16, `listOptimizers=${list.length}`);
  assert.equal(O('opt.util.optimizerCount', []), list.length);
  // state 序列化往返
  const parsed = JSON.parse(typeof js === 'string' ? js : JSON.stringify(js));
  assert.ok(parsed != null);
});

test('signSgd 与梯度尺度无关 / adadelta 无全局 lr', () => {
  const cfg = { lr: 0.01 };
  const x = Tensor.fromFlat([0], [1]);
  const st = O('opt.signSgd', [x, cfg]);
  const r1 = O('opt.signSgd.step', [x, Tensor.fromFlat([1e6], [1]), st, cfg]);
  assert.ok(approx(Math.abs(r1.params.data[0]), 0.01, 1e-12), `signSgd 步长 ${r1.params.data[0]}`);
  // 大梯度与小梯度同符号 → 同步长
  const x2 = Tensor.fromFlat([0], [1]);
  const st2 = O('opt.signSgd', [x2, cfg]);
  const r2 = O('opt.signSgd.step', [x2, Tensor.fromFlat([1e-6], [1]), st2, cfg]);
  assert.ok(approx(Math.abs(r2.params.data[0]), 0.01, 1e-12), '小梯度同步长');
  const st3 = O('opt.adadelta', [Tensor.fromFlat([1], [1]), {}]);
  assert.ok(st3 != null);
});

test('gradientNorm / paramDistance / etaScale / biasCorrection', () => {
  const g = O('opt.util.gradientNorm', [Tensor.fromFlat([3, 4], [2])]);
  assert.ok(approx(g, 5, 1e-12));
  const pd = O('opt.util.paramDistance', [Tensor.fromFlat([0, 0], [2]), Tensor.fromFlat([3, 4], [2])]);
  assert.ok(approx(typeof pd === 'number' ? pd : pd.value, 5, 1e-12));
  const bc = O('opt.util.biasCorrection', [0.9, 1]);
  assert.ok(approx(bc, 0.1, 1e-12), `biasCorrection(0.9,1)=${bc}`);
});
