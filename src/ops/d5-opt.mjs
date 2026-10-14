// 域5 优化器 OPT（102 条微指令）
// 对标：PyTorch torch.optim 优化器族与 LRScheduler、TensorFlow optimizers。
// 组织：16 经典优化器（create+step 各一条，共 32 条）、12 学习率调度、9 梯度变换、
//       4 线搜索、12 组合预设、33 内省与工具。零依赖、纯函数、state 可 JSON 序列化。
// 状态约定：所有 state 为纯对象+数组（无 Float64Array），t 为已执行步数。
import {
  Tensor, defineOp, getOp, exec, opsByDomain, ForgeError, expectTensor, expectNumber,
} from '../core/index.mjs';

let _s = 0;
const nid = () => `D5-${String(++_s).padStart(3, '0')}`;

// —— 内部小工具（非指令）——
const _vec = (t) => Array.from(t.data);               // Tensor -> 纯数组
const _mk = (a, shape) => Tensor.fromFlat(a, shape);  // 纯数组 -> Tensor
const _norm2 = (a) => { let s = 0; for (const v of a) s += v * v; return Math.sqrt(s); };
const _fill = (k, v = 0) => new Array(k).fill(v);
function _checkSameShape(p, g) {
  if (p.shape.length !== g.shape.length || p.shape.some((d, i) => d !== g.shape[i])) {
    throw new ForgeError('E_SHAPE', `params/grads 形状不符: [${p.shape}] vs [${g.shape}]`);
  }
}
// 数值校验：cfg 合并默认值后，所有字段必须有限；lr>0；beta∈[0,1)；eps>=0
function _checkCfg(name, c) {
  for (const k of Object.keys(c)) {
    const v = c[k];
    if (typeof v === 'number' && !Number.isFinite(v)) {
      throw new ForgeError('E_VALUE', `opt.${name} 配置项 ${k} 非有限: ${v}`);
    }
  }
  if ('lr' in c) { if (!(c.lr > 0)) throw new ForgeError('E_VALUE', `opt.${name}: lr 必须 > 0`); }
  for (const b of ['beta1', 'beta2', 'momentum', 'alpha', 'rho']) {
    if (b in c && !(c[b] >= 0 && c[b] < 1)) throw new ForgeError('E_VALUE', `opt.${name}: ${b} 必须 ∈ [0,1)`);
  }
  if ('eps' in c && c.eps < 0) throw new ForgeError('E_VALUE', `opt.${name}: eps 必须 >= 0`);
}

// =====================================================================
// 一、16 个经典优化器（每条：opt.<name> 创建 state + opt.<name>.step 单步更新）
//     step 返回 {params: Tensor, state: object}；state 为纯对象+数组。
// =====================================================================
const OPTSPECS = {
  // SGD：θ ← θ − η·g（Robbins & Monro 1951；Polyak & Juditsky 1992 平均化）
  sgd: {
    defaults: { lr: 0.1 },
    init: () => ({ t: 0 }),
    step(p, g, st, c) {
      const out = p.map((v, i) => v - c.lr * g[i]);
      return { p: out, st: { t: st.t + 1 } };
    },
  },
  // 经典动量（Heavy-ball）：v ← β·v + g；θ ← θ − η·v
  sgdMomentum: {
    defaults: { lr: 0.01, momentum: 0.9 },
    init: (k) => ({ t: 0, v: _fill(k) }),
    step(p, g, st, c) {
      const nv = st.v.map((v, i) => c.momentum * v + g[i]);
      const out = p.map((v, i) => v - c.lr * nv[i]);
      return { p: out, st: { t: st.t + 1, v: nv } };
    },
  },
  // Nesterov 动量（Sutskever 2013 形式）：v ← β·v + g；θ ← θ − η·(g + β·v)
  sgdNesterov: {
    defaults: { lr: 0.01, momentum: 0.9 },
    init: (k) => ({ t: 0, v: _fill(k) }),
    step(p, g, st, c) {
      const nv = st.v.map((v, i) => c.momentum * v + g[i]);
      const out = p.map((v, i) => v - c.lr * (g[i] + c.momentum * nv[i]));
      return { p: out, st: { t: st.t + 1, v: nv } };
    },
  },
  // Adagrad：G ← G + g⊙g；θ ← θ − η·g/(√G + ε)（Duchi, Hazan & Singer, JMLR 2011）
  adagrad: {
    defaults: { lr: 0.5, eps: 1e-8 },
    init: (k) => ({ t: 0, G: _fill(k) }),
    step(p, g, st, c) {
      const nG = st.G.map((v, i) => v + g[i] * g[i]);
      const out = p.map((v, i) => v - c.lr * g[i] / (Math.sqrt(nG[i]) + c.eps));
      return { p: out, st: { t: st.t + 1, G: nG } };
    },
  },
  // Adadelta：Eg² ← ρEg² + (1−ρ)g²；Δθ = −η·√(EΔθ²+ε)/√(Eg²+ε)·g（Zeiler, 2012）
  //   η 默认 1 ⇒ 与原文 Algorithm 1 严格一致；PyTorch/TF 额外暴露 η 以便工程调优。
  //   已知局限：RMS(Δx) 正反馈使比值 √(EΔθ²)/√(Eg²) 在最优解附近失稳，
  //   长程会在最优点附近缓慢振荡（本文件 inv 与测试均显式记录该行为）。
  adadelta: {
    defaults: { lr: 1, rho: 0.9, eps: 1e-6 },
    init: (k) => ({ t: 0, eg2: _fill(k), edx2: _fill(k) }),
    step(p, g, st, c) {
      const eg2 = st.eg2.map((v, i) => c.rho * v + (1 - c.rho) * g[i] * g[i]);
      const dx = g.map((gi, i) => -c.lr * Math.sqrt(st.edx2[i] + c.eps) / Math.sqrt(eg2[i] + c.eps) * gi);
      const edx2 = dx.map((d, i) => c.rho * st.edx2[i] + (1 - c.rho) * d * d);
      const out = p.map((v, i) => v + dx[i]);
      return { p: out, st: { t: st.t + 1, eg2, edx2 } };
    },
  },
  // RMSprop：v ← βv + (1−β)g²；θ ← θ − η·g/(√v + ε)（Hinton 课程；Graves 2013）
  rmsprop: {
    defaults: { lr: 0.01, beta2: 0.99, eps: 1e-8 },
    init: (k) => ({ t: 0, v: _fill(k) }),
    step(p, g, st, c) {
      const nv = st.v.map((v, i) => c.beta2 * v + (1 - c.beta2) * g[i] * g[i]);
      const out = p.map((v, i) => v - c.lr * g[i] / (Math.sqrt(nv[i]) + c.eps));
      return { p: out, st: { t: st.t + 1, v: nv } };
    },
  },
  // Adam：m,v 为 EMA，偏差修正 m̂=m/(1−β1^t), v̂=v/(1−β2^t)；
  //       θ ← θ − η·m̂/(√v̂ + ε)（Kingma & Ba, ICLR 2015）
  adam: {
    defaults: { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
    init: (k) => ({ t: 0, m: _fill(k), v: _fill(k) }),
    step(p, g, st, c) {
      const t = st.t + 1, bc1 = 1 - Math.pow(c.beta1, t), bc2 = 1 - Math.pow(c.beta2, t);
      const m = st.m.map((v, i) => c.beta1 * v + (1 - c.beta1) * g[i]);
      const v = st.v.map((x, i) => c.beta2 * x + (1 - c.beta2) * g[i] * g[i]);
      const out = p.map((x, i) => x - c.lr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + c.eps));
      return { p: out, st: { t, m, v } };
    },
  },
  // AdamW：Adam 更新 + 解耦权重衰减 θ ← θ − η·(m̂/(√v̂+ε) + λθ)
  //        （Loshchilov & Hutter, Decoupled Weight Decay Regularization, ICLR 2019）
  adamw: {
    defaults: { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8, wd: 0.01 },
    init: (k) => ({ t: 0, m: _fill(k), v: _fill(k) }),
    step(p, g, st, c) {
      const t = st.t + 1, bc1 = 1 - Math.pow(c.beta1, t), bc2 = 1 - Math.pow(c.beta2, t);
      const m = st.m.map((v, i) => c.beta1 * v + (1 - c.beta1) * g[i]);
      const v = st.v.map((x, i) => c.beta2 * x + (1 - c.beta2) * g[i] * g[i]);
      const out = p.map((x, i) => x - c.lr * (m[i] / bc1 / (Math.sqrt(v[i] / bc2) + c.eps) + c.wd * x));
      return { p: out, st: { t, m, v } };
    },
  },
  // Nadam：一阶矩 Nesterov 前瞻 ψ = (β1·m_t + (1−β1)·g_t)/(1−β1^t)；
  //        θ ← θ − η·ψ/(√v̂ + ε)（Dozat, ICLR 2016）
  nadam: {
    defaults: { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
    init: (k) => ({ t: 0, m: _fill(k), v: _fill(k) }),
    step(p, g, st, c) {
      const t = st.t + 1, bc1 = 1 - Math.pow(c.beta1, t), bc2 = 1 - Math.pow(c.beta2, t);
      const m = st.m.map((v, i) => c.beta1 * v + (1 - c.beta1) * g[i]);
      const v = st.v.map((x, i) => c.beta2 * x + (1 - c.beta2) * g[i] * g[i]);
      const out = p.map((x, i) => x - c.lr * ((c.beta1 * m[i] + (1 - c.beta1) * g[i]) / bc1) / (Math.sqrt(v[i] / bc2) + c.eps));
      return { p: out, st: { t, m, v } };
    },
  },
  // AMSGrad：用历史最大的 v̂ 代替当前 v̂（Reddi, Kale, Kumar, ICLR 2018）
  amsgrad: {
    defaults: { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
    init: (k) => ({ t: 0, m: _fill(k), v: _fill(k), vmax: _fill(k) }),
    step(p, g, st, c) {
      const t = st.t + 1, bc1 = 1 - Math.pow(c.beta1, t), bc2 = 1 - Math.pow(c.beta2, t);
      const m = st.m.map((v, i) => c.beta1 * v + (1 - c.beta1) * g[i]);
      const v = st.v.map((x, i) => c.beta2 * x + (1 - c.beta2) * g[i] * g[i]);
      const vmax = st.vmax.map((x, i) => Math.max(x, v[i] / bc2));
      const out = p.map((x, i) => x - c.lr * (m[i] / bc1) / (Math.sqrt(vmax[i]) + c.eps));
      return { p: out, st: { t, m, v, vmax } };
    },
  },
  // Adamax：∞-范数变体 u ← max(β2·u, |g|)；θ ← θ − η·m̂/(u + ε)
  //          （Kingma & Ba, Adam 论文 §7.2）
  adamax: {
    defaults: { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
    init: (k) => ({ t: 0, m: _fill(k), u: _fill(k) }),
    step(p, g, st, c) {
      const t = st.t + 1, bc1 = 1 - Math.pow(c.beta1, t);
      const m = st.m.map((v, i) => c.beta1 * v + (1 - c.beta1) * g[i]);
      const u = st.u.map((x, i) => Math.max(c.beta2 * x, Math.abs(g[i])));
      const out = p.map((x, i) => x - c.lr * (m[i] / bc1) / (u[i] + c.eps));
      return { p: out, st: { t, m, u } };
    },
  },
  // LAMB：Adam 步 φ = m̂/(√v̂+ε)，u = φ + λθ，信任比 trust = ‖φ‖/‖u‖，
  //       θ ← θ − η·trust·u（You et al., Large Batch Training of Deep Networks, 2017；
  //       信任比按本域规范取 ‖φ‖/‖φ+λθ‖，论文原式为 ‖θ‖/‖u‖）
  lamb: {
    defaults: { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8, wd: 0.01 },
    init: (k) => ({ t: 0, m: _fill(k), v: _fill(k) }),
    step(p, g, st, c) {
      const t = st.t + 1, bc1 = 1 - Math.pow(c.beta1, t), bc2 = 1 - Math.pow(c.beta2, t);
      const m = st.m.map((v, i) => c.beta1 * v + (1 - c.beta1) * g[i]);
      const v = st.v.map((x, i) => c.beta2 * x + (1 - c.beta2) * g[i] * g[i]);
      const phi = p.map((x, i) => m[i] / bc1 / (Math.sqrt(v[i] / bc2) + c.eps));
      const u = p.map((x, i) => phi[i] + c.wd * x);
      const nu = _norm2(u), np = _norm2(phi);
      const trust = nu > 0 && np > 0 ? np / nu : 1;
      const out = p.map((x, i) => x - c.lr * trust * u[i]);
      return { p: out, st: { t, m, v } };
    },
  },
  // Rprop：仅用梯度符号；符号一致 η_i·=η⁺，相反 η_i·=η⁻，更新 θ ← θ − sign(g)·η_i
  //        （Riedmiller & Braun, A Direct Adaptive Method for Faster BP, 1993）
  //        cfg.lr 为全局步长倍率（默认 1，即标准 Rprop）
  rprop: {
    defaults: { lr: 1, eta0: 0.01, etaPlus: 1.2, etaMinus: 0.5, etaMin: 1e-6, etaMax: 50 },
    init: (k) => ({ t: 0, gPrev: _fill(k), eta: _fill(k) }),
    step(p, g, st, c) {
      const eta = st.eta.map((e, i) => {
        const base = st.t === 0 ? c.eta0 : e;
        const s = g[i] * st.gPrev[i];
        let v = base;
        if (s > 0) v = Math.min(base * c.etaPlus, c.etaMax);
        else if (s < 0) v = Math.max(base * c.etaMinus, c.etaMin);
        return v;
      });
      const out = p.map((x, i) => x - c.lr * Math.sign(g[i]) * eta[i]);
      return { p: out, st: { t: st.t + 1, gPrev: g.slice(), eta } };
    },
  },
  // Yogi：v ← v − (1−β2)·sign(v − g²)·g²，控制二阶矩增速
  //        （Zaheer et al., Adaptive Methods for Nonconvex Optimization, 2018）
  yogi: {
    defaults: { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
    init: (k) => ({ t: 0, m: _fill(k), v: _fill(k) }),
    step(p, g, st, c) {
      const t = st.t + 1, bc1 = 1 - Math.pow(c.beta1, t), bc2 = 1 - Math.pow(c.beta2, t);
      const m = st.m.map((v, i) => c.beta1 * v + (1 - c.beta1) * g[i]);
      const v = st.v.map((x, i) => x - (1 - c.beta2) * Math.sign(x - g[i] * g[i]) * g[i] * g[i]);
      const out = p.map((x, i) => x - c.lr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + c.eps));
      return { p: out, st: { t, m, v } };
    },
  },
  // Lion：update = −η·(sign(β1·m + (1−β1)·g) + λθ)；m ← β2·m + (1−β2)·g
  //        （Chen et al., Symbolic Discovery of Optimization Algorithms, 2023）
  lion: {
    defaults: { lr: 0.05, beta1: 0.9, beta2: 0.99, wd: 0.0 },
    init: (k) => ({ t: 0, m: _fill(k) }),
    step(p, g, st, c) {
      const c1 = p.map((x, i) => c.beta1 * st.m[i] + (1 - c.beta1) * g[i]);
      const out = p.map((x, i) => x - c.lr * (Math.sign(c1[i]) + c.wd * x));
      const m = p.map((x, i) => c.beta2 * st.m[i] + (1 - c.beta2) * g[i]);
      return { p: out, st: { t: st.t + 1, m } };
    },
  },
  // SignSGD：θ ← θ − η·sign(g)（Bernstein et al., signSGD, ICML 2018）
  signSgd: {
    defaults: { lr: 0.05 },
    init: () => ({ t: 0 }),
    step(p, g, st, c) {
      const out = p.map((x, i) => x - c.lr * Math.sign(g[i]));
      return { p: out, st: { t: st.t + 1 } };
    },
  },
};

const OPT_NAMES = Object.keys(OPTSPECS);

// 统一 step 入口：参数校验 -> 委托 spec.step -> 返回 {params, state}
function _optStep(name, params, grads, state, cfg) {
  expectTensor(params, `opt.${name}.step params`);
  expectTensor(grads, `opt.${name}.step grads`);
  _checkSameShape(params, grads);
  if (state === null || typeof state !== 'object') {
    throw new ForgeError('E_STATE', `opt.${name}.step: state 未初始化（先用 opt.${name} 创建）`);
  }
  const c = { ...OPTSPECS[name].defaults, ...(cfg || {}) };
  _checkCfg(name, c);
  const p = _vec(params), g = _vec(grads);
  const r = OPTSPECS[name].step(p, g, state, c);
  return { params: _mk(r.p, params.shape), state: r.st };
}

// 每条优化器的文档/引用/不变量（供注册使用）
const OPT_DOCS = {
  sgd: {
    doc: '梯度下降单步：θ ← θ − η·g',
    ref: 'Robbins & Monro, A Stochastic Approximation Method, Ann. Math. Statist. 1951',
    inv: '更新量 = −η·g',
  },
  sgdMomentum: {
    doc: '经典动量（Heavy-ball）单步：v ← β·v + g，θ ← θ − η·v',
    ref: 'Polyak, Some Methods of Speeding Up the Convergence of Iteration Methods, 1964',
    inv: 'v_t = β·v_{t−1} + g_t，更新量 = −η·v_t',
  },
  sgdNesterov: {
    doc: 'Nesterov 动量单步：v ← β·v + g，θ ← θ − η·(g + β·v)',
    ref: 'Sutskever et al., On the Importance of Initialization and Momentum in Deep Learning, ICML 2013',
    inv: '更新量 = −η·(g + β·v)，等价于在预估点处取梯度',
  },
  adagrad: {
    doc: 'Adagrad 单步：G ← G + g⊙g，θ ← θ − η·g/(√G + ε)',
    ref: 'Duchi, Hazan & Singer, Adaptive Subgradient Methods for Online Learning, JMLR 2011',
    inv: '更新量 = −η·g_i/(√G_i + ε)，G 为梯度平方累积和',
  },
  adadelta: {
    doc: 'Adadelta 单步：Δθ = −η·√(EΔθ²+ε)/√(Eg²+ε)·g，二阶矩用窗口 EMA（η=1 即原文）',
    ref: 'Zeiler, ADADELTA: An Adaptive Learning Rate Method, 2012',
    inv: 'Δθ_i = −η·√(EΔθ²_i+ε)/√(Eg²_i+ε)·g_i；η=1 时无需全局学习率，最优点附近存在已知 RMS(Δx) 振荡',
  },
  rmsprop: {
    doc: 'RMSprop 单步：v ← βv + (1−β)g²，θ ← θ − η·g/(√v + ε)',
    ref: 'Graves, Generating Sequences With Recurrent Neural Networks, 2013（RMSprop 形式）',
    inv: '更新量 = −η·g/(√v+ε)，v = βv + (1−β)g²',
  },
  adam: {
    doc: 'Adam 单步：m,v 为 EMA，偏差修正后 θ ← θ − η·m̂/(√v̂ + ε)',
    ref: 'Kingma & Ba, Adam: A Method for Stochastic Optimization, ICLR 2015',
    inv: '更新量 = −η·m̂/(√v̂+ε)，偏差修正后 m̂=m/(1−β1^t)，v̂=v/(1−β2^t)',
  },
  adamw: {
    doc: 'AdamW 单步：Adam 更新 + 解耦权重衰减 θ ← θ − η·(m̂/(√v̂+ε) + λθ)',
    ref: 'Loshchilov & Hutter, Decoupled Weight Decay Regularization (AdamW), ICLR 2019',
    inv: '更新量 = −η·(m̂/(√v̂+ε) + λ·θ)，权重衰减不经二阶矩归一化',
  },
  nadam: {
    doc: 'Nadam 单步：一阶矩 Nesterov 前瞻 ψ = (β1·m + (1−β1)·g)/(1−β1^t)',
    ref: 'Dozat, Incorporating Nesterov Momentum into Adam, ICLR 2016',
    inv: '更新量 = −η·ψ/(√v̂+ε)，ψ = (β1·m_t + (1−β1)·g_t)/(1−β1^t)',
  },
  amsgrad: {
    doc: 'AMSGrad 单步：用 v̂ 的历史最大值 v̂max 代替 v̂ 参与更新',
    ref: 'Reddi, Kale & Kumar, On the Convergence of Adam and Beyond, ICLR 2018',
    inv: '更新量 = −η·m̂/(√v̂max+ε)，v̂max = max(v̂max, v̂)',
  },
  adamax: {
    doc: 'Adamax 单步：∞-范数变体 u ← max(β2·u, |g|)，θ ← θ − η·m̂/(u + ε)',
    ref: 'Kingma & Ba, Adam 论文 §7.2（Adamax 算法）',
    inv: '更新量 = −η·m̂/(u+ε)，u = max(β2·u, |g|)',
  },
  lamb: {
    doc: 'LAMB 单步：Adam 步 φ 加信任比缩放，θ ← θ − η·trust·(φ + λθ)',
    ref: 'You et al., Large Batch Training of Deep Networks (LAMB), 2017',
    inv: 'trust = ‖φ‖₂/‖u‖₂（u = φ + λ·θ），更新量 = −η·trust·u',
  },
  rprop: {
    doc: 'Rprop 单步：仅用梯度符号，步长 η_i 按符号一致性自适应增减',
    ref: 'Riedmiller & Braun, A Direct Adaptive Method for Faster Backpropagation Learning, 1993',
    inv: '符号一致 η_i ← min(η_i·η⁺, ηMax)，相反 η_i ← max(η_i·η⁻, ηMin)；更新量 = −sign(g)·η_i',
  },
  yogi: {
    doc: 'Yogi 单步：二阶矩用符号控制增量 v ← v − (1−β2)·sign(v−g²)·g²',
    ref: 'Zaheer et al., Adaptive Methods for Nonconvex Optimization, NeurIPS 2018',
    inv: '更新量 = −η·m̂/(√v̂+ε)，v_t = v_{t−1} − (1−β2)·sign(v_{t−1}−g²)·g²',
  },
  lion: {
    doc: 'Lion 单步：update = −η·(sign(β1·m + (1−β1)·g) + λθ)，动量用 β2',
    ref: 'Chen et al., Symbolic Discovery of Optimization Algorithms (Lion), 2023',
    inv: '更新量 = −η·(sign(c) + λ·θ)，c = β1·m + (1−β1)·g；m ← β2·m + (1−β2)·g',
  },
  signSgd: {
    doc: 'SignSGD 单步：θ ← θ − η·sign(g)',
    ref: 'Bernstein et al., signSGD: Compressed Optimisation for Non-Convex Problems, ICML 2018',
    inv: '更新量 = −η·sign(g)，逐元素取 ±η',
  },
};

// 注册 32 条：每条优化器 create + step
for (const name of OPT_NAMES) {
  const spec = OPTSPECS[name];
  const doc = OPT_DOCS[name];
  defineOp({
    id: nid(), domain: 5, name: `opt.${name}`, arity: 2,
    sig: '(Tensor params, object cfg) -> object',
    doc: `创建优化器 ${name} 的初始 state（t=0，动量/二阶矩缓冲区为零数组）`,
    ref: doc.ref || '', inv: doc.inv || 'state.t = 0 且各缓冲区为长度 k 的零数组',
    impl: (params, cfg) => {
      expectTensor(params, `opt.${name} params`);
      const c = { ...spec.defaults, ...(cfg || {}) };
      _checkCfg(name, c);
      return spec.init(params.size);
    },
  });
  defineOp({
    id: nid(), domain: 5, name: `opt.${name}.step`, arity: 4,
    sig: '(Tensor params, Tensor grads, object state, object cfg) -> {params: Tensor, state: object}',
    doc: `${name} 单步更新：${doc.doc || ''}`,
    ref: doc.ref || '', inv: doc.inv || '',
    impl: (params, grads, state, cfg) => _optStep(name, params, grads, state, cfg),
  });
}

// =====================================================================
// 二、学习率调度 12 条：统一签名 (cfg, epoch) -> lr，epoch 从 0 起算
// =====================================================================
const SCHED_DEFAULTS = {
  constant: { lr: 0.1 },
  step: { lr: 0.1, stepSize: 10, gamma: 0.1 },
  multistep: { lr: 0.1, milestones: [30, 60, 90], gamma: 0.1 },
  exp: { lr: 0.1, gamma: 0.95 },
  cosine: { lr: 0.1, T: 50, lrMin: 0 },
  warmup: { lr: 0.1, warmupEpochs: 5, warmupStartLr: 0 },
  warmupCosine: { lr: 0.1, T: 50, lrMin: 0, warmupEpochs: 5, warmupStartLr: 0 },
  onecycle: { lr: 0.1, T: 50, pctStart: 0.3, divFactor: 25, finalDivFactor: 1e4 },
  polynomial: { lr: 0.1, T: 50, lrMin: 0, power: 1 },
  triangular: { lrMin: 0.01, maxLr: 0.1, stepSize: 10 },
  linearDecay: { lr: 0.1, T: 50, lrMin: 0 },
  inverse: { lr: 0.1, gamma: 0.01 },
};

const SCHED = {
  // 常数：lr(t) = η
  constant: (c, e) => c.lr,
  // 阶梯：lr(t) = η·γ^{⌊t/stepSize⌋}
  step: (c, e) => c.lr * Math.pow(c.gamma, Math.floor(e / c.stepSize)),
  // 多里程碑：lr(t) = η·γ^{|{m ∈ milestones : t ≥ m}|}
  multistep: (c, e) => c.lr * Math.pow(c.gamma, c.milestones.filter((m) => e >= m).length),
  // 指数衰减：lr(t) = η·γ^t
  exp: (c, e) => c.lr * Math.pow(c.gamma, e),
  // 余弦退火：lr(t) = lrMin + (η−lrMin)·½(1+cos(π·min(t,T)/T))（SGDR, Loshchilov & Hutter 2016）
  cosine: (c, e) => {
    const t = Math.min(e, c.T), lo = c.lrMin ?? 0;
    return lo + (c.lr - lo) * 0.5 * (1 + Math.cos(Math.PI * t / c.T));
  },
  // 线性 warmup 后保持常数：t < W 时 η(t) = start + (η−start)·t/W
  warmup: (c, e) => {
    const w = c.warmupEpochs, s = c.warmupStartLr ?? 0;
    return e < w ? s + (c.lr - s) * (e / w) : c.lr;
  },
  // warmup + 余弦退火到 lrMin
  warmupCosine: (c, e) => {
    const w = c.warmupEpochs, s = c.warmupStartLr ?? 0;
    if (e < w) return s + (c.lr - s) * (e / w);
    const t = Math.min(e - w, c.T), lo = c.lrMin ?? 0;
    return lo + (c.lr - lo) * 0.5 * (1 + Math.cos(Math.PI * t / c.T));
  },
  // OneCycle：先线性升到 η，再余弦降到 η/finalDivFactor（Smith, 2018）
  onecycle: (c, e) => {
    const T = c.T, pct = c.pctStart ?? 0.3;
    const lr0 = c.lr / (c.divFactor ?? 25), lrEnd = c.lr / (c.finalDivFactor ?? 1e4);
    const warm = Math.max(1, Math.round(pct * T));
    if (e < warm) return lr0 + (c.lr - lr0) * (e / warm);
    if (e < T) {
      const t = (e - warm) / Math.max(1, T - warm);
      return lrEnd + (c.lr - lrEnd) * 0.5 * (1 + Math.cos(Math.PI * t));
    }
    return lrEnd;
  },
  // 多项式衰减：lr(t) = lrMin + (η−lrMin)·(1 − min(t,T)/T)^power
  polynomial: (c, e) => {
    const lo = c.lrMin ?? 0, t = Math.min(e, c.T) / c.T;
    return lo + (c.lr - lo) * Math.pow(1 - t, c.power ?? 1);
  },
  // 三角波循环：lr(t) = lrMin + (maxLr−lrMin)·max(0, 1−x)，
  // x = |t/stepSize − 2·cycle + 1|，cycle = ⌊1 + t/(2·stepSize)⌋（Smith, CLR 2017）
  triangular: (c, e) => {
    const s = c.stepSize, cyc = Math.floor(1 + e / (2 * s));
    const x = Math.abs(e / s - 2 * cyc + 1);
    const peak = c.maxLr ?? c.lr, lo = c.lrMin ?? 0;
    return lo + (peak - lo) * Math.max(0, 1 - x);
  },
  // 线性衰减到 lrMin：lr(t) = lrMin + (η−lrMin)·(1 − min(t,T)/T)
  linearDecay: (c, e) => {
    const lo = c.lrMin ?? 0, t = Math.min(e, c.T) / c.T;
    return lo + (c.lr - lo) * (1 - t);
  },
  // 反向衰减：lr(t) = η/(1 + γ·t)
  inverse: (c, e) => c.lr / (1 + c.gamma * e),
};

const SCHED_DOCS = {
  constant: ['常数学习率：lr(t) = η', 'PyTorch 文档：lr_scheduler.ConstantLR', 'lr(t) ≡ η'],
  step: ['阶梯衰减：每 stepSize 个 epoch 乘以 γ', 'PyTorch 文档：lr_scheduler.StepLR', 'lr(t) = η·γ^⌊t/stepSize⌋'],
  multistep: ['多里程碑衰减：每越过一个 milestone 乘以 γ', 'PyTorch 文档：lr_scheduler.MultiStepLR', 'lr(t) = η·γ^{#{m∈milestones: t≥m}}'],
  exp: ['指数衰减：lr(t) = η·γ^t', 'PyTorch 文档：lr_scheduler.ExponentialLR', 'lr(t) = η·γ^t'],
  cosine: ['余弦退火到 lrMin', 'Loshchilov & Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts, ICLR 2017', 'lr(t) = lrMin + (η−lrMin)·½(1+cos(π·min(t,T)/T))'],
  warmup: ['线性 warmup 后保持常数', 'Goyal et al., Accurate Large Minibatch SGD, 2017（warmup 策略）', 't<W 时 lr(t) = start + (η−start)·t/W，否则 η'],
  warmupCosine: ['线性 warmup 后余弦退火', 'Goyal et al., Accurate Large Minibatch SGD, 2017 + Loshchilov & Hutter, SGDR, ICLR 2017', 'warmup 段线性，之后 lr(t) = lrMin + (η−lrMin)·½(1+cos(π·(t−W)/T))'],
  onecycle: ['OneCycle：先升后降的单周期策略', 'Smith, A Disciplined Approach to Neural Network Hyper-Parameters, 2018', '峰值 η，末端 η/finalDivFactor，warmup 段占 pctStart'],
  polynomial: ['多项式衰减到 lrMin', 'PyTorch 文档：lr_scheduler.PolynomialLR', 'lr(t) = lrMin + (η−lrMin)·(1−min(t,T)/T)^power'],
  triangular: ['三角波循环学习率（CLR）', 'Smith, Cyclical Learning Rates for Training Neural Networks, 2017', 'lr(t) = lrMin + (maxLr−lrMin)·max(0, 1−x)，x = |t/stepSize−2·cycle+1|'],
  linearDecay: ['线性衰减到 lrMin', '常见训练脚本实践（如 Transformer 的 Noam 逆平方根的线性变体）', 'lr(t) = lrMin + (η−lrMin)·(1−min(t,T)/T)'],
  inverse: ['反向时间衰减：lr(t) = η/(1+γt)', 'Dark Knowledge / 常见 SGD 变体实践', 'lr(t) = η/(1+γ·t)'],
};

for (const name of Object.keys(SCHED)) {
  const [doc, ref, inv] = SCHED_DOCS[name];
  defineOp({
    id: nid(), domain: 5, name: `opt.lr.${name}`, arity: 2,
    sig: '(object cfg, number epoch) -> number',
    doc: `学习率调度 ${name}：${doc}`,
    ref, inv,
    impl: (cfg, epoch) => {
      expectNumber(epoch, 'epoch');
      if (epoch < 0) throw new ForgeError('E_VALUE', 'epoch 必须 >= 0');
      const c = { ...SCHED_DEFAULTS[name], ...(cfg || {}) };
      if ('lr' in c && !(c.lr > 0)) throw new ForgeError('E_VALUE', `opt.lr.${name}: lr 必须 > 0`);
      return SCHED[name](c, epoch);
    },
  });
}

// =====================================================================
// 三、梯度变换 9 条
// =====================================================================
// 按 L2 范数裁剪：若 ‖g‖₂ > maxNorm 则 g ← g·maxNorm/‖g‖₂
defineOp({
  id: nid(), domain: 5, name: 'opt.grad.clipNorm', arity: 2,
  sig: '(Tensor grads, number maxNorm) -> Tensor',
  doc: '梯度全局范数裁剪：‖g‖₂ 超过 maxNorm 时按比例缩放',
  ref: 'Pascanu, Mikolov & Bengio, On the Difficulty of Training RNNs, ICML 2013',
  inv: '输出范数 = min(‖g‖₂, maxNorm)',
  impl: (grads, maxNorm) => {
    expectTensor(grads, 'grads'); expectNumber(maxNorm, 'maxNorm');
    if (maxNorm <= 0) throw new ForgeError('E_VALUE', 'maxNorm 必须 > 0');
    const a = _vec(grads), n = _norm2(a);
    const s = n > maxNorm ? maxNorm / n : 1;
    return _mk(a.map((v) => v * s), grads.shape);
  },
});
// 逐元素值域裁剪：g_i ← clamp(g_i, −clip, clip)
defineOp({
  id: nid(), domain: 5, name: 'opt.grad.clipValue', arity: 2,
  sig: '(Tensor grads, number clipValue) -> Tensor',
  doc: '梯度逐元素裁剪到 [−clipValue, clipValue]',
  ref: '常见训练实践（TensorFlow tf.clip_by_value）',
  inv: '输出逐元素满足 |g_i| <= clipValue',
  impl: (grads, clipValue) => {
    expectTensor(grads, 'grads'); expectNumber(clipValue, 'clipValue');
    if (clipValue <= 0) throw new ForgeError('E_VALUE', 'clipValue 必须 > 0');
    return _mk(_vec(grads).map((v) => Math.max(-clipValue, Math.min(clipValue, v))), grads.shape);
  },
});
// L2 权重衰减（耦合式）：g ← g + wd·θ
defineOp({
  id: nid(), domain: 5, name: 'opt.grad.weightDecay', arity: 3,
  sig: '(Tensor grads, Tensor params, number wd) -> Tensor',
  doc: '耦合式 L2 权重衰减：g ← g + λ·θ',
  ref: 'Krogh & Hertz, A Simple Weight Decay Can Improve Generalization, 1992',
  inv: '输出 = g + λ·θ',
  impl: (grads, params, wd) => {
    expectTensor(grads, 'grads'); expectTensor(params, 'params'); expectNumber(wd, 'wd');
    _checkSameShape(grads, params);
    const g = _vec(grads), p = _vec(params);
    return _mk(g.map((v, i) => v + wd * p[i]), grads.shape);
  },
});
// 梯度中心化：减去最后一轴的均值（1D 减全局均值）
defineOp({
  id: nid(), domain: 5, name: 'opt.grad.centralization', arity: 1,
  sig: '(Tensor grads) -> Tensor',
  doc: '梯度中心化：减去最后一轴均值，约束更新为零均值方向',
  ref: 'Yong et al., Gradient Centralization for Better Training Performance, CVPR 2020',
  inv: '输出沿最后一轴求和为 0',
  impl: (grads) => {
    expectTensor(grads, 'grads');
    const a = _vec(grads), shape = grads.shape;
    if (shape.length === 1) {
      const m = a.reduce((s, v) => s + v, 0) / a.length;
      return _mk(a.map((v) => v - m), shape);
    }
    // 2D：逐行减均值（行 = shape[0]，每行长度 shape[1]）
    const rows = shape[0], cols = shape[1], out = new Array(a.length);
    for (let r = 0; r < rows; r++) {
      let s = 0;
      for (let j = 0; j < cols; j++) s += a[r * cols + j];
      const m = s / cols;
      for (let j = 0; j < cols; j++) out[r * cols + j] = a[r * cols + j] - m;
    }
    return _mk(out, shape);
  },
});
// Lookahead：每 k 步把慢权重向快权重同步 slow ← slow + α·(fast − slow)，fast ← slow
defineOp({
  id: nid(), domain: 5, name: 'opt.lookahead', arity: 3,
  sig: '(Tensor fastParams, object state, object cfg) -> {params: Tensor, state: object}',
  doc: 'Lookahead 包装器：每 k 步执行一次慢权重同步 slow ← slow + α·(fast − slow)',
  ref: 'Zhang, Hinton et al., Lookahead Optimizer: k Steps Forward, 1 Step Back, NeurIPS 2019',
  inv: '同步发生时 slow ← slow + α·(fast − slow) 且 fast ← slow；state.t 每调用 +1',
  impl: (fast, state, cfg) => {
    expectTensor(fast, 'fastParams');
    if (state === null || typeof state !== 'object') throw new ForgeError('E_STATE', 'opt.lookahead: state 未初始化');
    const c = { alpha: 0.5, k: 5, ...(cfg || {}) };
    const t = (state.t ?? 0) + 1;
    const slow = state.slow ?? _vec(fast);
    const f = _vec(fast);
    if (t % c.k === 0) {
      const ns = slow.map((v, i) => v + c.alpha * (f[i] - v));
      return { params: _mk(ns, fast.shape), state: { t, slow: ns } };
    }
    return { params: _mk(f, fast.shape), state: { t, slow } };
  },
});
// 参数 EMA：ema ← decay·ema + (1−decay)·θ，返回 ema 作为新参数
defineOp({
  id: nid(), domain: 5, name: 'opt.ema', arity: 3,
  sig: '(Tensor params, object state, object cfg) -> {params: Tensor, state: object}',
  doc: '参数指数滑动平均：ema ← decay·ema + (1−decay)·θ',
  ref: 'Polyak & Juditsky, Averaging of Iterative Methods, 1992；深度学习常见 EMA 实践（如 Mean Teacher）',
  inv: 'ema_t = decay·ema_{t−1} + (1−decay)·θ_t，返回 ema',
  impl: (params, state, cfg) => {
    expectTensor(params, 'params');
    if (state === null || typeof state !== 'object') throw new ForgeError('E_STATE', 'opt.ema: state 未初始化');
    const c = { decay: 0.999, ...(cfg || {}) };
    const p = _vec(params);
    const ema = (state.ema ?? p).map((v, i) => c.decay * v + (1 - c.decay) * p[i]);
    return { params: _mk(ema, params.shape), state: { t: (state.t ?? 0) + 1, ema } };
  },
});
// 梯度按 batch 大小缩放：g ← g·batchSize/baseBatch（等效损失归约切换）
defineOp({
  id: nid(), domain: 5, name: 'opt.grad.scaleByBatch', arity: 3,
  sig: '(Tensor grads, number batchSize, number baseBatch) -> Tensor',
  doc: '梯度按 batch 大小线性缩放，用于累积/归一化损失切换',
  ref: '常见混合精度与梯度累积实践（如 PyTorch GradScaler 逆缩放思想）',
  inv: '输出 = g·batchSize/baseBatch',
  impl: (grads, batchSize, baseBatch) => {
    expectTensor(grads, 'grads'); expectNumber(batchSize, 'batchSize'); expectNumber(baseBatch, 'baseBatch');
    if (batchSize <= 0 || baseBatch <= 0) throw new ForgeError('E_VALUE', 'batchSize/baseBatch 必须 > 0');
    const s = batchSize / baseBatch;
    return _mk(_vec(grads).map((v) => v * s), grads.shape);
  },
});
// 累加状态重置：按参考梯度形状返回全零累加状态
defineOp({
  id: nid(), domain: 5, name: 'opt.grad.accumulateReset', arity: 1,
  sig: '(Tensor refGrads) -> object',
  doc: '重置梯度累加状态：{sum: 零数组, count: 0}',
  ref: '梯度累积训练实践（TensorFlow/PyTorch accumulation）',
  inv: '返回 {sum: Array(k).fill(0), count: 0}，k = refGrads.size',
  impl: (refGrads) => {
    expectTensor(refGrads, 'refGrads');
    return { sum: _fill(refGrads.size), count: 0 };
  },
});
// 梯度累加：sum ← sum + g，count ← count + 1（纯函数，返回新 state）
defineOp({
  id: nid(), domain: 5, name: 'opt.grad.accumulateAdd', arity: 2,
  sig: '(object state, Tensor grads) -> object',
  doc: '向累加状态追加一个 batch 的梯度：sum ← sum + g，count ← count + 1',
  ref: '梯度累积训练实践（等效放大 batch）',
  inv: '新 state = {sum: sum + g, count: count + 1}',
  impl: (state, grads) => {
    if (state === null || typeof state !== 'object' || !Array.isArray(state.sum)) {
      throw new ForgeError('E_STATE', 'opt.grad.accumulateAdd: state 应为 {sum, count}');
    }
    expectTensor(grads, 'grads');
    const g = _vec(grads);
    if (g.length !== state.sum.length) {
      throw new ForgeError('E_SHAPE', `累加长度不符: ${state.sum.length} vs ${g.length}`);
    }
    return { sum: state.sum.map((v, i) => v + g[i]), count: state.count + 1 };
  },
});

// =====================================================================
// 四、线搜索 4 条。fn 约定：{f: Tensor->number, g: Tensor->Tensor, hv?: Tensor->Tensor}
// =====================================================================
const _dirNorm2 = (g) => { let s = 0; for (const v of _vec(g)) s += v * v; return s; };
// Armijo 回溯线搜索：f(x − αd) ≤ f(x) + c1·α·∇f·d，d = −g
defineOp({
  id: nid(), domain: 5, name: 'opt.line.backtracking', arity: 4,
  sig: '(object fn, Tensor x, Tensor grad, object cfg) -> {alpha, xNew, fNew, iters}',
  doc: 'Armijo 回溯线搜索：沿 −g 方向找到满足充分下降的步长',
  ref: 'Armijo, Minimization of Functions Having Lipschitz Continuous First Partial Derivatives, 1966',
  inv: 'fNew <= f0 + c1·α·(g·d)，d = −g；α 每次失败乘以 rho',
  impl: (fn, x, grad, cfg) => {
    expectTensor(x, 'x'); expectTensor(grad, 'grad');
    const c = { alpha0: 1, rho: 0.5, c1: 1e-4, maxIter: 60, ...(cfg || {}) };
    const f0 = fn.f(x), gd = -_dirNorm2(grad);
    let a = c.alpha0, iters = 0, xNew = x, fNew = f0;
    for (; iters < c.maxIter; iters++) {
      const xn = _mk(_vec(x).map((v, i) => v - a * _vec(grad)[i]), x.shape);
      const fnv = fn.f(xn);
      if (fnv <= f0 + c.c1 * a * gd) { xNew = xn; fNew = fnv; break; }
      a *= c.rho;
    }
    if (iters === c.maxIter) throw new ForgeError('E_NUMERICAL', 'backtracking: 超过 maxIter 未满足 Armijo');
    return { alpha: a, xNew, fNew, iters: iters + 1 };
  },
});
// 轻量 Wolfe 线搜索：Armijo 充分下降 + 曲率条件 g_new·d ≥ c2·g·d
defineOp({
  id: nid(), domain: 5, name: 'opt.line.wolfeLite', arity: 4,
  sig: '(object fn, Tensor x, Tensor grad, object cfg) -> {alpha, xNew, fNew, gNew, ok, iters}',
  doc: '轻量 Wolfe 线搜索：回溯找同时满足 Armijo 与曲率条件的步长',
  ref: 'Wolfe, Convergence Conditions for Ascent Methods, SIAM Review 1971；Nocedal & Wright《Numerical Optimization》§3.1',
  inv: 'fNew <= f0 + c1·α·(g·d) 且 gNew·d >= c2·(g·d)，d = −g',
  impl: (fn, x, grad, cfg) => {
    expectTensor(x, 'x'); expectTensor(grad, 'grad');
    const c = { alpha0: 1, rho: 0.5, c1: 1e-4, c2: 0.9, maxIter: 60, ...(cfg || {}) };
    const g0 = _vec(grad), d = g0.map((v) => -v);
    const gd0 = g0.reduce((s, v, i) => s + v * d[i], 0);
    const f0 = fn.f(x);
    let a = c.alpha0;
    for (let it = 1; it <= c.maxIter; it++) {
      const xn = _mk(_vec(x).map((v, i) => v + a * d[i]), x.shape);
      const fNew = fn.f(xn), gNew = fn.g(xn);
      const gd = _vec(gNew).reduce((s, v, i) => s + v * d[i], 0);
      if (fNew <= f0 + c.c1 * a * gd0 && gd >= c.c2 * gd0) {
        return { alpha: a, xNew: xn, fNew, gNew, ok: true, iters: it };
      }
      a *= c.rho;
    }
    throw new ForgeError('E_NUMERICAL', 'wolfeLite: 超过 maxIter 未满足 Wolfe 条件');
  },
});
// 固定步长：直接返回 α = cfg.alpha 的试验点
defineOp({
  id: nid(), domain: 5, name: 'opt.line.fixedStep', arity: 4,
  sig: '(object fn, Tensor x, Tensor grad, object cfg) -> {alpha, xNew, fNew, gNew}',
  doc: '固定步长线搜索：沿 −g 走 cfg.alpha',
  ref: '最速下降法固定步长形式（Nocedal & Wright《Numerical Optimization》§3.1）',
  inv: 'xNew = x − α·g，α = cfg.alpha',
  impl: (fn, x, grad, cfg) => {
    expectTensor(x, 'x'); expectTensor(grad, 'grad');
    const c = { alpha: 0.1, ...(cfg || {}) };
    const g = _vec(grad);
    const xNew = _mk(_vec(x).map((v, i) => v - c.alpha * g[i]), x.shape);
    return { alpha: c.alpha, xNew, fNew: fn.f(xNew), gNew: fn.g(xNew) };
  },
});
// 二次函数精确线搜索：α* = (gᵀg)/(gᵀHg)，需 fn.hv(v) = H·v
defineOp({
  id: nid(), domain: 5, name: 'opt.line.exactQuadratic', arity: 4,
  sig: '(object fn, Tensor x, Tensor grad, object cfg) -> {alpha, xNew, fNew}',
  doc: '二次函数精确线搜索：α* = (gᵀg)/(gᵀHg)，沿 −g 精确最小化',
  ref: 'Nocedal & Wright《Numerical Optimization》§3.1（精确线搜索）',
  inv: 'f(x − α*g) = f(x) − (gᵀg)²/(2·gᵀHg)',
  impl: (fn, x, grad, cfg) => {
    expectTensor(x, 'x'); expectTensor(grad, 'grad');
    const g = _vec(grad);
    const hv = _vec(fn.hv(grad));
    const gHg = g.reduce((s, v, i) => s + v * hv[i], 0);
    const gg = g.reduce((s, v) => s + v * v, 0);
    if (!(gHg > 0)) throw new ForgeError('E_NUMERICAL', 'exactQuadratic: gᵀHg 非正，非凸方向');
    const a = gg / gHg;
    const xNew = _mk(_vec(x).map((v, i) => v - a * g[i]), x.shape);
    return { alpha: a, xNew, fNew: fn.f(xNew) };
  },
});

// =====================================================================
// 五、组合预设 12 条：基础优化器 + 学习率调度，state 传 null 表示初始化
// =====================================================================
const PRESETS = {
  adamCosine: { base: 'adam', sched: 'cosine' },
  adamwOnecycle: { base: 'adamw', sched: 'onecycle' },
  lionWarmupCosine: { base: 'lion', sched: 'warmupCosine' },
  sgdNesterovStep: { base: 'sgdNesterov', sched: 'step' },
  rmspropMultistep: { base: 'rmsprop', sched: 'multistep' },
  adamaxWarmup: { base: 'adamax', sched: 'warmup' },
  lambCosine: { base: 'lamb', sched: 'cosine' },
  adagradExp: { base: 'adagrad', sched: 'exp' },
  rpropTriangular: { base: 'rprop', sched: 'triangular' },
  signSgdConstant: { base: 'signSgd', sched: 'constant' },
  momentumLinear: { base: 'sgdMomentum', sched: 'linearDecay' },
  nadamOnecycle: { base: 'nadam', sched: 'onecycle' },
};

function _presetStep(pname, params, grads, state, cfg) {
  const { base, sched } = PRESETS[pname];
  let st = state;
  if (st === null || st === undefined) st = { epoch: 0, base: OPTSPECS[base].init(params.size) };
  const c0 = { ...SCHED_DEFAULTS[sched], ...(cfg || {}) };
  const lr = SCHED[sched](c0, st.epoch);
  const c = { ...OPTSPECS[base].defaults, ...(cfg || {}), lr };
  const r = OPTSPECS[base].step(_vec(params), _vec(grads), st.base, c);
  return {
    params: _mk(r.p, params.shape),
    state: { epoch: st.epoch + 1, base: r.st },
  };
}

for (const pname of Object.keys(PRESETS)) {
  const { base, sched } = PRESETS[pname];
  defineOp({
    id: nid(), domain: 5, name: `opt.${pname}`, arity: 4,
    sig: '(Tensor params, Tensor grads, object|null state, object cfg) -> {params: Tensor, state: object}',
    doc: `组合指令：${base} + lr.${sched}（state 传 null 初始化并执行第一步）`,
    ref: `组合预设：${OPT_DOCS[base].ref} + ${SCHED_DOCS[sched][1]}`,
    inv: `每步 lr 由 lr.${sched}(cfg, epoch) 给出，基础更新用 opt.${base} 规则；state.epoch 每步 +1`,
    impl: (params, grads, state, cfg) => {
      expectTensor(params, `${pname} params`);
      expectTensor(grads, `${pname} grads`);
      _checkSameShape(params, grads);
      return _presetStep(pname, params, grads, state, cfg);
    },
  });
}

// =====================================================================
// 六、内省与工具 33 条
// =====================================================================
// —— 基准函数（均返回 {name, dim, xStar, f, g}，f/g 接收 Tensor）——
// 凸二次：f(x) = ½Σᵢ dᵢ(xᵢ−x*ᵢ)²，默认 dᵢ = i（1..k），k = 10，x* = 0
defineOp({
  id: nid(), domain: 5, name: 'opt.util.quadraticFn', arity: 1,
  sig: '(object cfg) -> object',
  doc: '构造凸二次基准 f(x)=½Σ dᵢ(xᵢ−x*ᵢ)² 的 loss+grad（默认 dᵢ=i, k=10, x*=0）',
  ref: '优化基准标准形式（Nocedal & Wright《Numerical Optimization》第 3 章二次模型）',
  inv: 'g = d⊙(x−x*)，Hessian 为 diag(d)',
  impl: (cfg) => {
    const c = { k: 10, xStar: null, diag: null, ...(cfg || {}) };
    const k = c.k;
    if (!Number.isInteger(k) || k < 1) throw new ForgeError('E_VALUE', 'k 必须为正整数');
    const xs = c.xStar ? c.xStar.slice(0, k) : _fill(k);
    const d = c.diag ? c.diag.slice(0, k) : Array.from({ length: k }, (_, i) => i + 1);
    const xsT = _mk(xs, [k]);
    return {
      name: 'quadratic', dim: k, xStar: xsT,
      f: (x) => { const a = _vec(x); let s = 0; for (let i = 0; i < k; i++) { const e = a[i] - xs[i]; s += d[i] * e * e; } return 0.5 * s; },
      g: (x) => _mk(_vec(x).map((v, i) => d[i] * (v - xs[i])), [k]),
      hv: (v) => _mk(_vec(v).map((x, i) => d[i] * x), [k]),
    };
  },
});
// 球函数：f(x) = ½Σ(xᵢ−x*ᵢ)²
defineOp({
  id: nid(), domain: 5, name: 'opt.util.sphereFn', arity: 1,
  sig: '(object cfg) -> object',
  doc: '构造球函数基准 f(x)=½‖x−x*‖² 的 loss+grad',
  ref: '优化测试函数标准形式（Jamil & Yang, A Literature Survey of Benchmark Functions, 2013）',
  inv: 'g = x − x*，Hessian 为单位矩阵',
  impl: (cfg) => {
    const c = { k: 10, xStar: null, ...(cfg || {}) };
    const k = c.k;
    if (!Number.isInteger(k) || k < 1) throw new ForgeError('E_VALUE', 'k 必须为正整数');
    const xs = c.xStar ? c.xStar.slice(0, k) : _fill(k);
    const xsT = _mk(xs, [k]);
    return {
      name: 'sphere', dim: k, xStar: xsT,
      f: (x) => { const a = _vec(x); let s = 0; for (let i = 0; i < k; i++) { const e = a[i] - xs[i]; s += e * e; } return 0.5 * s; },
      g: (x) => _mk(_vec(x).map((v, i) => v - xs[i]), [k]),
      hv: (v) => _mk(_vec(v), [k]),
    };
  },
});
// Rosenbrock（非凸）：f = Σ [100(x_{2i+1}−x_{2i}²)² + (1−x_{2i})²]，x* = (1,..,1)
defineOp({
  id: nid(), domain: 5, name: 'opt.util.rosenbrock', arity: 1,
  sig: '(object cfg) -> object',
  doc: '构造 Rosenbrock 非凸基准的 loss+grad（经典 2D：f=(1−x)²+100(y−x²)²）',
  ref: 'Rosenbrock, An Automatic Method for Finding the Greatest or Least Value of a Function, 1960',
  inv: 'f(x*) = 0 且 x* = (1,1,...,1)',
  impl: (cfg) => {
    const c = { n: 2, ...(cfg || {}) };
    const n = c.n;
    if (n < 2 || n % 2 !== 0) throw new ForgeError('E_VALUE', 'rosenbrock 需要偶数 n >= 2');
    const xsT = _mk(_fill(n, 1), [n]);
    return {
      name: 'rosenbrock', dim: n, xStar: xsT,
      f: (x) => {
        const a = _vec(x); let s = 0;
        for (let i = 0; i < n; i += 2) { const u = a[i + 1] - a[i] * a[i], v = 1 - a[i]; s += 100 * u * u + v * v; }
        return s;
      },
      g: (x) => {
        const a = _vec(x), out = _fill(n);
        for (let i = 0; i < n; i += 2) {
          const u = a[i + 1] - a[i] * a[i], v = 1 - a[i];
          out[i] += -400 * a[i] * u - 2 * v;
          out[i + 1] += 200 * u;
        }
        return _mk(out, [n]);
      },
    };
  },
});
// Beale（2D 非凸，经典最优点 (3, 0.5)）
defineOp({
  id: nid(), domain: 5, name: 'opt.util.beale', arity: 1,
  sig: '(object cfg) -> object',
  doc: '构造 Beale 函数基准 loss+grad（2D，x*=(3, 0.5)）',
  ref: 'Beale 函数，标准优化测试函数（Jamil & Yang, 2013）',
  inv: 'f(3, 0.5) = 0',
  impl: () => {
    const xsT = _mk([3, 0.5], [2]);
    const f = (x, y) => (1.5 - x + x * y) ** 2 + (2.25 - x + x * y * y) ** 2 + (2.625 - x + x * y * y * y) ** 2;
    return {
      name: 'beale', dim: 2, xStar: xsT,
      f: (t) => f(_vec(t)[0], _vec(t)[1]),
      g: (t) => {
        const [x, y] = _vec(t);
        const t1 = 1.5 - x + x * y, t2 = 2.25 - x + x * y * y, t3 = 2.625 - x + x * y * y * y;
        const gx = 2 * t1 * (y - 1) + 2 * t2 * (y * y - 1) + 2 * t3 * (y * y * y - 1);
        const gy = 2 * t1 * x + 2 * t2 * 2 * x * y + 2 * t3 * 3 * x * y * y;
        return _mk([gx, gy], [2]);
      },
    };
  },
});
// Booth（2D 凸二次，x*=(1,3)）
defineOp({
  id: nid(), domain: 5, name: 'opt.util.booth', arity: 1,
  sig: '(object cfg) -> object',
  doc: '构造 Booth 函数基准 loss+grad（2D 凸，x*=(1,3)）',
  ref: 'Booth 函数，标准优化测试函数（Jamil & Yang, 2013）',
  inv: 'f(x,y) = (x+2y−7)² + (2x+y−5)²，f(1,3) = 0',
  impl: () => {
    const xsT = _mk([1, 3], [2]);
    return {
      name: 'booth', dim: 2, xStar: xsT,
      f: (t) => { const [x, y] = _vec(t); return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2; },
      g: (t) => {
        const [x, y] = _vec(t);
        return _mk([2 * (x + 2 * y - 7) + 4 * (2 * x + y - 5), 4 * (x + 2 * y - 7) + 2 * (2 * x + y - 5)], [2]);
      },
      hv: (v) => _mk([10 * _vec(v)[0] + 8 * _vec(v)[1], 8 * _vec(v)[0] + 10 * _vec(v)[1]], [2]),
    };
  },
});
// Himmelblau（2D 非凸，四个全局最小点，f=0）
defineOp({
  id: nid(), domain: 5, name: 'opt.util.himmelblau', arity: 1,
  sig: '(object cfg) -> object',
  doc: '构造 Himmelblau 函数基准 loss+grad（2D 非凸，最小点 (3,2) 等，f=0）',
  ref: 'Himmelblau 函数，标准优化测试函数（Jamil & Yang, 2013）',
  inv: 'f(x,y) = (x²+y−11)² + (x+y²−7)²，f(3,2) = 0',
  impl: () => {
    const xsT = _mk([3, 2], [2]);
    return {
      name: 'himmelblau', dim: 2, xStar: xsT,
      f: (t) => { const [x, y] = _vec(t); return (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2; },
      g: (t) => {
        const [x, y] = _vec(t);
        return _mk([4 * x * (x * x + y - 11) + 2 * (x + y * y - 7), 2 * (x * x + y - 11) + 4 * y * (x + y * y - 7)], [2]);
      },
    };
  },
});
// McCormick（2D 非凸，x*≈(−0.54719, −1.54719)，f*≈−1.91322）
defineOp({
  id: nid(), domain: 5, name: 'opt.util.mccormick', arity: 1,
  sig: '(object cfg) -> object',
  doc: '构造 McCormick 函数基准 loss+grad（2D 非凸，f*≈−1.9132）',
  ref: 'McCormick, A Mathematical Programming Approach to the Global Optimization of Chemical Processes, 1972',
  inv: 'f(x,y) = sin(x+y) + (x−y)² − 1.5x + 2.5y + 1',
  impl: () => {
    const xsT = _mk([-0.5471975602064441, -1.5471975602064441], [2]);
    return {
      name: 'mccormick', dim: 2, xStar: xsT,
      f: (t) => { const [x, y] = _vec(t); return Math.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1; },
      g: (t) => {
        const [x, y] = _vec(t), cc = Math.cos(x + y);
        return _mk([cc + 2 * (x - y) - 1.5, cc - 2 * (x - y) + 2.5], [2]);
      },
    };
  },
});

// —— 优化器驱动 ——
function _runOpt(name, fn, params, iters, cfg) {
  const nm = name.startsWith('opt.') ? name.slice(4) : name;
  const isClassic = Object.prototype.hasOwnProperty.call(OPTSPECS, nm);
  const isPreset = Object.prototype.hasOwnProperty.call(PRESETS, nm);
  if (!isClassic && !isPreset) throw new ForgeError('E_NOTFOUND', `未知优化器: ${name}`);
  const p0 = params instanceof Tensor ? params.clone() : _mk(params.slice(), [params.length]);
  const c = cfg || {};
  let p = p0, st = null;
  const traj = [fn.f(p)];
  for (let i = 0; i < iters; i++) {
    const g = fn.g(p);
    if (isClassic) {
      if (st === null) st = OPTSPECS[nm].init(p.size);
      const r = _optStep(nm, p, g, st, c);
      p = r.params; st = r.state;
    } else {
      const r = _presetStep(nm, p, g, st, c);
      p = r.params; st = r.state;
    }
    traj.push(fn.f(p));
  }
  return { name: nm, traj, params: p, finalLoss: traj[traj.length - 1], iters };
}

// 统一驱动：给定优化器名 + 基准 + 迭代数，返回 loss 轨迹与最终参数
defineOp({
  id: nid(), domain: 5, name: 'opt.util.runOptimizer', arity: 5,
  sig: '(string name, object fn, Tensor params, number iters, object cfg) -> object',
  doc: '统一驱动：运行指定优化器（经典或组合预设），返回 loss 轨迹与最终参数',
  ref: '对标 PyTorch 训练循环的最小化抽象',
  inv: 'traj[0] = f(x0)，traj[i+1] = f(x_{i+1})；最终 params 为最后一步结果',
  impl: (name, fn, params, iters, cfg) => {
    if (typeof name !== 'string') throw new ForgeError('E_TYPE', 'name 应为 string');
    expectNumber(iters, 'iters');
    if (iters < 0) throw new ForgeError('E_VALUE', 'iters 必须 >= 0');
    expectTensor(params, 'params');
    return _runOpt(name, fn, params, iters, cfg);
  },
});
// 仅取 loss 轨迹数组
defineOp({
  id: nid(), domain: 5, name: 'opt.util.lossTrajectory', arity: 5,
  sig: '(string name, object fn, Tensor params, number iters, object cfg) -> number[]',
  doc: '运行优化器并仅返回 loss 轨迹数组（长度 iters+1）',
  ref: 'runOptimizer 的轨迹切片视图',
  inv: '返回数组长度 = iters + 1，首元素为初始 loss',
  impl: (name, fn, params, iters, cfg) => _runOpt(name, fn, params, iters, cfg).traj,
});
// 收敛判定：‖x − x*‖∞ < tol
defineOp({
  id: nid(), domain: 5, name: 'opt.util.converged', arity: 3,
  sig: '(Tensor x, Tensor xStar, number tol) -> boolean',
  doc: '收敛判定：参数与最优点的最大绝对偏差 < tol',
  ref: '优化终止准则（Nocedal & Wright《Numerical Optimization》§2.2）',
  inv: '返回 max_i |x_i − x*_i| < tol',
  impl: (x, xStar, tol) => {
    expectTensor(x, 'x'); expectTensor(xStar, 'xStar');
    const t = tol === undefined ? 1e-6 : tol;
    if (!(t > 0)) throw new ForgeError('E_VALUE', 'tol 必须 > 0');
    const a = _vec(x), b = _vec(xStar);
    if (a.length !== b.length) throw new ForgeError('E_SHAPE', 'x 与 xStar 长度不符');
    return a.every((v, i) => Math.abs(v - b[i]) < t);
  },
});
// 梯度 L2 范数
defineOp({
  id: nid(), domain: 5, name: 'opt.util.gradientNorm', arity: 1,
  sig: '(Tensor grads) -> number',
  doc: '计算梯度 L2 范数 ‖g‖₂',
  ref: 'Nocedal & Wright《Numerical Optimization》§2.2（梯度范数终止准则）',
  inv: '返回 sqrt(Σ g_i²)',
  impl: (grads) => { expectTensor(grads, 'grads'); return _norm2(_vec(grads)); },
});
// 参数与最优点的 L2 距离
defineOp({
  id: nid(), domain: 5, name: 'opt.util.paramDistance', arity: 2,
  sig: '(Tensor x, Tensor xStar) -> number',
  doc: '计算参数与最优点的 L2 距离 ‖x − x*‖₂',
  ref: '收敛度量标准定义',
  inv: '返回 sqrt(Σ (x_i − x*_i)²)',
  impl: (x, xStar) => {
    expectTensor(x, 'x'); expectTensor(xStar, 'xStar');
    const a = _vec(x), b = _vec(xStar);
    if (a.length !== b.length) throw new ForgeError('E_SHAPE', '长度不符');
    return _norm2(a.map((v, i) => v - b[i]));
  },
});
// 多优化器同基准对比表
defineOp({
  id: nid(), domain: 5, name: 'opt.util.compareOptimizers', arity: 5,
  sig: '(string[] names, object fn, Tensor params, number iters, object cfg) -> object[]',
  doc: '多个优化器在同一基准上运行，返回对比表（含最终 loss/误差/单调性）',
  ref: '基准测试方法论（如 MLPerf 训练基准的最小化对照）',
  inv: '每行 = {name, finalLoss, finalErr, monotone, iters}，按输入顺序返回',
  impl: (names, fn, params, iters, cfg) => {
    if (!Array.isArray(names) || names.length === 0) throw new ForgeError('E_VALUE', 'names 应为非空数组');
    expectTensor(params, 'params');
    expectNumber(iters, 'iters');
    const xs = fn.xStar ? _vec(fn.xStar) : null;
    return names.map((nm) => {
      const r = _runOpt(nm, fn, params, iters, cfg);
      const xf = _vec(r.params);
      const finalErr = xs ? Math.max(...xf.map((v, i) => Math.abs(v - xs[i]))) : NaN;
      return { name: r.name, finalLoss: r.finalLoss, finalErr, monotone: _monotone(r.traj), iters: r.iters };
    });
  },
});
// 取对比表中最终 loss 最小的一行
defineOp({
  id: nid(), domain: 5, name: 'opt.util.bestOf', arity: 1,
  sig: '(object[] rows) -> object',
  doc: '从 compareOptimizers 结果中取最终 loss 最小的一行',
  ref: '基准对比后处理',
  inv: '返回 rows 中 finalLoss 最小者（并列取第一个）',
  impl: (rows) => {
    if (!Array.isArray(rows) || rows.length === 0) throw new ForgeError('E_VALUE', 'rows 应为非空数组');
    let best = rows[0];
    for (const r of rows) if (r.finalLoss < best.finalLoss) best = r;
    return best;
  },
});
// 按调度名取某 epoch 的学习率
defineOp({
  id: nid(), domain: 5, name: 'opt.util.scheduleAt', arity: 3,
  sig: '(string schedName, object cfg, number epoch) -> number',
  doc: '按调度器名字（constant/step/... 之一）计算 epoch 处学习率',
  ref: '调度器分派入口',
  inv: '等价于调用对应的 opt.lr.<schedName>',
  impl: (schedName, cfg, epoch) => {
    if (typeof schedName !== 'string') throw new ForgeError('E_TYPE', 'schedName 应为 string');
    const nm = schedName.replace(/^opt\.lr\./, '');
    if (!SCHED[nm]) throw new ForgeError('E_NOTFOUND', `未知调度器: ${schedName}`);
    expectNumber(epoch, 'epoch');
    const c = { ...SCHED_DEFAULTS[nm], ...(cfg || {}) };
    return SCHED[nm](c, epoch);
  },
});
// state -> JSON 字符串
defineOp({
  id: nid(), domain: 5, name: 'opt.util.stateToJSON', arity: 1,
  sig: '(object state) -> string',
  doc: '把优化器 state 序列化为 JSON 字符串',
  ref: '序列化约定：state 必须为纯对象+数组',
  inv: 'JSON.parse(stateToJSON(s)) 与 s 深度相等',
  impl: (state) => {
    if (state === null || typeof state !== 'object') throw new ForgeError('E_TYPE', 'state 应为 object');
    return JSON.stringify(state);
  },
});
// JSON 字符串 -> state
defineOp({
  id: nid(), domain: 5, name: 'opt.util.stateFromJSON', arity: 1,
  sig: '(string json) -> object',
  doc: '从 JSON 字符串还原优化器 state',
  ref: '序列化约定：state 必须为纯对象+数组',
  inv: 'stateFromJSON(stateToJSON(s)) 与 s 深度相等',
  impl: (json) => {
    if (typeof json !== 'string') throw new ForgeError('E_TYPE', 'json 应为 string');
    const s = JSON.parse(json);
    if (s === null || typeof s !== 'object') throw new ForgeError('E_VALUE', 'JSON 必须表示对象');
    return s;
  },
});
// 按优化器名创建零状态
defineOp({
  id: nid(), domain: 5, name: 'opt.util.zeroState', arity: 2,
  sig: '(string name, number k) -> object',
  doc: '按优化器名创建长度为 k 的零初始化 state',
  ref: '与 opt.<name> 的 init 一致',
  inv: '等价于 opt.<name>(zeros([k]), {})',
  impl: (name, k) => {
    if (typeof name !== 'string') throw new ForgeError('E_TYPE', 'name 应为 string');
    const nm = name.startsWith('opt.') ? name.slice(4) : name;
    if (!OPTSPECS[nm]) throw new ForgeError('E_NOTFOUND', `未知优化器: ${name}`);
    if (!Number.isInteger(k) || k < 1) throw new ForgeError('E_VALUE', 'k 必须为正整数');
    return OPTSPECS[nm].init(k);
  },
});
// 未收敛则抛错（供测试/流水线断言）
defineOp({
  id: nid(), domain: 5, name: 'opt.util.assertConverged', arity: 3,
  sig: '(Tensor x, Tensor xStar, number tol) -> boolean',
  doc: '断言收敛：‖x − x*‖∞ >= tol 时抛 E_NUMERICAL，否则返回 true',
  ref: '测试断言与训练早停',
  inv: '不抛错时必有 max_i |x_i − x*_i| < tol',
  impl: (x, xStar, tol) => {
    const t = tol === undefined ? 1e-6 : tol;
    if (!(t > 0)) throw new ForgeError('E_VALUE', 'tol 必须 > 0');
    const ok = exec('opt.util.converged', [x, xStar, t]);
    if (!ok) throw new ForgeError('E_NUMERICAL', `未收敛: ‖x−x*‖∞ >= ${t}`);
    return true;
  },
});
// 轨迹单调性检查（允许 tol 量级抖动）
function _monotone(traj, tol = 1e-12) {
  for (let i = 1; i < traj.length; i++) {
    if (traj[i] > traj[i - 1] + tol * Math.max(1, Math.abs(traj[i - 1]))) return false;
  }
  return true;
}
defineOp({
  id: nid(), domain: 5, name: 'opt.util.monotoneLoss', arity: 2,
  sig: '(number[] traj, number tol) -> boolean',
  doc: '检查 loss 轨迹单调不增（允许 tol·max(1,|loss|) 量级抖动）',
  ref: '凸优化下降方法理论（Nocedal & Wright 第 3 章）',
  inv: '所有相邻项满足 traj[i] <= traj[i−1] + tol·max(1, |traj[i−1]|)',
  impl: (traj, tol) => {
    if (!Array.isArray(traj)) throw new ForgeError('E_TYPE', 'traj 应为数组');
    const t = tol === undefined ? 1e-12 : tol;
    if (!(t >= 0)) throw new ForgeError('E_VALUE', 'tol 必须 >= 0');
    return _monotone(traj, t);
  },
});
// 优化器描述（走注册表元数据）
defineOp({
  id: nid(), domain: 5, name: 'opt.util.describeOptimizer', arity: 1,
  sig: '(string name) -> string',
  doc: '返回优化器的注册表描述（id/name/doc/ref/inv）',
  ref: '注册表内省（describeOp）',
  inv: '输出包含该优化器的 doc 与 inv',
  impl: (name) => {
    const nm = typeof name === 'string' && name.startsWith('opt.') ? name : `opt.${name}`;
    const op = getOp(nm);
    return `${op.id} ${op.name} — ${op.doc} | ${op.sig} | 参考: ${op.ref} | 不变量: ${op.inv}`;
  },
});
// 优化器名列表（16 个）
defineOp({
  id: nid(), domain: 5, name: 'opt.util.listOptimizers', arity: 0,
  sig: '() -> string[]',
  doc: '列出全部 16 个经典优化器名',
  ref: 'forge registry introspection (self-describing op catalog)',
  inv: '返回数组长度 = 16 且元素唯一',
  impl: () => OPT_NAMES.slice(),
});
// 优化器数量
defineOp({
  id: nid(), domain: 5, name: 'opt.util.optimizerCount', arity: 0,
  sig: '() -> number',
  doc: '返回经典优化器数量（16）',
  ref: 'forge registry introspection (self-describing op catalog)',
  inv: '恒等于 listOptimizers().length',
  impl: () => OPT_NAMES.length,
});
// 默认配置
defineOp({
  id: nid(), domain: 5, name: 'opt.util.defaultCfg', arity: 1,
  sig: '(string name) -> object',
  doc: '返回指定优化器的默认配置对象',
  ref: '对标 PyTorch optim 的 defaults 字典',
  inv: '返回 OPTSPECS[name].defaults 的浅拷贝',
  impl: (name) => {
    if (typeof name !== 'string') throw new ForgeError('E_TYPE', 'name 应为 string');
    const nm = name.startsWith('opt.') ? name.slice(4) : name;
    if (!OPTSPECS[nm]) throw new ForgeError('E_NOTFOUND', `未知优化器: ${name}`);
    return { ...OPTSPECS[nm].defaults };
  },
});
// 配置合法性校验（返回布尔，不抛错）
defineOp({
  id: nid(), domain: 5, name: 'opt.util.validateCfg', arity: 2,
  sig: '(string name, object cfg) -> boolean',
  doc: '校验优化器配置合法性（lr>0、beta∈[0,1)、eps>=0、全有限）',
  ref: '对标 PyTorch 的参数校验',
  inv: '合法返回 true，否则 false（不抛错）',
  impl: (name, cfg) => {
    if (typeof name !== 'string' || cfg === null || typeof cfg !== 'object') return false;
    const nm = name.startsWith('opt.') ? name.slice(4) : name;
    if (!OPTSPECS[nm]) return false;
    try { _checkCfg(nm, { ...OPTSPECS[nm].defaults, ...cfg }); return true; } catch { return false; }
  },
});
// Adam 偏差修正后的等效步长尺度 η·√(1−β2^t)/(1−β1^t)
defineOp({
  id: nid(), domain: 5, name: 'opt.util.etaScale', arity: 1,
  sig: '(object cfg) -> number',
  doc: 'Adam 偏差修正后的等效步长尺度：lr·√(1−β2^t)/(1−β1^t)',
  ref: 'Kingma & Ba, ICLR 2015（偏差修正分析）',
  inv: 'etaScale = lr·√(1−β2^t)/(1−β1^t)',
  impl: (cfg) => {
    const c = { lr: 0.001, beta1: 0.9, beta2: 0.999, t: 1, ...(cfg || {}) };
    expectNumber(c.t, 't');
    if (!Number.isInteger(c.t) || c.t < 1) throw new ForgeError('E_VALUE', 't 必须为正整数');
    return c.lr * Math.sqrt(1 - Math.pow(c.beta2, c.t)) / (1 - Math.pow(c.beta1, c.t));
  },
});
// 偏差修正因子 1 − β^t
defineOp({
  id: nid(), domain: 5, name: 'opt.util.biasCorrection', arity: 2,
  sig: '(number beta, number t) -> number',
  doc: '偏差修正因子 1 − β^t（Adam 系列 m̂/v̂ 的分母）',
  ref: 'Kingma & Ba, ICLR 2015',
  inv: '返回 1 − beta^t，t >= 1',
  impl: (beta, t) => {
    expectNumber(beta, 'beta'); expectNumber(t, 't');
    if (!(beta >= 0 && beta < 1)) throw new ForgeError('E_VALUE', 'beta 必须 ∈ [0,1)');
    if (!Number.isInteger(t) || t < 1) throw new ForgeError('E_VALUE', 't 必须为正整数');
    return 1 - Math.pow(beta, t);
  },
});
// LAMB 信任比 ‖φ‖/‖φ+wd·w‖
defineOp({
  id: nid(), domain: 5, name: 'opt.util.trustRatio', arity: 3,
  sig: '(Tensor phi, Tensor w, number wd) -> number',
  doc: 'LAMB 信任比：‖φ‖₂/‖φ + wd·w‖₂（任一为零范数时取 1）',
  ref: 'You et al., Large Batch Training of Deep Networks (LAMB), 2017',
  inv: '返回 ‖φ‖/‖φ+wd·w‖；‖φ+wd·w‖=0 或 ‖φ‖=0 时为 1',
  impl: (phi, w, wd) => {
    expectTensor(phi, 'phi'); expectTensor(w, 'w');
    const d = wd === undefined ? 0 : wd;
    expectNumber(d, 'wd');
    _checkSameShape(phi, w);
    const a = _vec(phi), b = _vec(w);
    const nu = _norm2(a.map((v, i) => v + d * b[i])), np = _norm2(a);
    return nu > 0 && np > 0 ? np / nu : 1;
  },
});
// Lion/SignSGD 符号更新 −η·sign(g)
defineOp({
  id: nid(), domain: 5, name: 'opt.util.signUpdate', arity: 2,
  sig: '(Tensor grads, number lr) -> Tensor',
  doc: '符号更新量 −η·sign(g)（Lion/SignSGD 的核心更新）',
  ref: 'Bernstein et al., signSGD, ICML 2018；Chen et al., Lion, 2023',
  inv: '输出逐元素 = −η·sign(g_i)',
  impl: (grads, lr) => {
    expectTensor(grads, 'grads'); expectNumber(lr, 'lr');
    if (!(lr > 0)) throw new ForgeError('E_VALUE', 'lr 必须 > 0');
    return _mk(_vec(grads).map((v) => -lr * Math.sign(v)), grads.shape);
  },
});
// 动量混合 β·v + (1−β)·g
defineOp({
  id: nid(), domain: 5, name: 'opt.util.momentumBlend', arity: 3,
  sig: '(Tensor grads, Tensor v, number beta) -> Tensor',
  doc: '动量混合 β·v + (1−β)·g（EMA 形式的一阶矩更新）',
  ref: 'Polyak 动量 / EMA 标准形式',
  inv: '输出 = β·v + (1−β)·g',
  impl: (grads, v, beta) => {
    expectTensor(grads, 'grads'); expectTensor(v, 'v');
    expectNumber(beta, 'beta');
    _checkSameShape(grads, v);
    if (!(beta >= 0 && beta < 1)) throw new ForgeError('E_VALUE', 'beta 必须 ∈ [0,1)');
    const g = _vec(grads), a = _vec(v);
    return _mk(g.map((x, i) => beta * a[i] + (1 - beta) * x), grads.shape);
  },
});
// warmup 线性因子
defineOp({
  id: nid(), domain: 5, name: 'opt.util.warmupFactor', arity: 3,
  sig: '(number epoch, number warmupEpochs, number start) -> number',
  doc: 'warmup 线性因子：epoch < W 时 start+(1−start)·epoch/W，否则 1',
  ref: 'Goyal et al., Accurate Large Minibatch SGD, 2017',
  inv: '因子 ∈ [start, 1] 且单调不减',
  impl: (epoch, warmupEpochs, start) => {
    expectNumber(epoch, 'epoch'); expectNumber(warmupEpochs, 'warmupEpochs');
    const s = start === undefined ? 0 : start;
    if (epoch < 0) throw new ForgeError('E_VALUE', 'epoch 必须 >= 0');
    if (!(warmupEpochs > 0)) throw new ForgeError('E_VALUE', 'warmupEpochs 必须 > 0');
    if (!(s >= 0 && s <= 1)) throw new ForgeError('E_VALUE', 'start 必须 ∈ [0,1]');
    return epoch < warmupEpochs ? s + (1 - s) * (epoch / warmupEpochs) : 1;
  },
});
// 余弦因子 ½(1+cos(π·min(t,T)/T))
defineOp({
  id: nid(), domain: 5, name: 'opt.util.cosineFactor', arity: 2,
  sig: '(number epoch, number T) -> number',
  doc: '余弦退火因子 ½(1+cos(π·min(epoch,T)/T))：从 1 降到 0',
  ref: 'Loshchilov & Hutter, SGDR, ICLR 2017',
  inv: 'factor(0) = 1，factor(T/2) = 0.5，factor(T) = 0',
  impl: (epoch, T) => {
    expectNumber(epoch, 'epoch'); expectNumber(T, 'T');
    if (epoch < 0) throw new ForgeError('E_VALUE', 'epoch 必须 >= 0');
    if (!(T > 0)) throw new ForgeError('E_VALUE', 'T 必须 > 0');
    return 0.5 * (1 + Math.cos(Math.PI * Math.min(epoch, T) / T));
  },
});
// OneCycle 归一化位置因子：warmup 段 0→1，退火段 1→0
defineOp({
  id: nid(), domain: 5, name: 'opt.util.onecycleFactor', arity: 3,
  sig: '(number epoch, number T, number pctStart) -> number',
  doc: 'OneCycle 归一化位置因子：warmup 段线性 0→1，随后余弦 1→0',
  ref: 'Smith, A Disciplined Approach to Neural Network Hyper-Parameters, 2018',
  inv: 'factor(0)=0，峰值 1 出现在 pctStart·T 处，factor(T)=0',
  impl: (epoch, T, pctStart) => {
    expectNumber(epoch, 'epoch'); expectNumber(T, 'T');
    const pct = pctStart === undefined ? 0.3 : pctStart;
    if (epoch < 0) throw new ForgeError('E_VALUE', 'epoch 必须 >= 0');
    if (!(T > 0)) throw new ForgeError('E_VALUE', 'T 必须 > 0');
    if (!(pct > 0 && pct < 1)) throw new ForgeError('E_VALUE', 'pctStart 必须 ∈ (0,1)');
    const warm = Math.max(1, Math.round(pct * T));
    if (epoch < warm) return Math.min(1, epoch / warm);
    if (epoch < T) {
      const t = (epoch - warm) / Math.max(1, T - warm);
      return 0.5 * (1 + Math.cos(Math.PI * t));
    }
    return 0;
  },
});

// —— 文件底部自检：域5 必须恰好 102 条，否则 import 即失败 ——
if (opsByDomain(5).length !== 102) {
  throw new Error(`域5 指令数错误: 期望 102，实际 ${opsByDomain(5).length}`);
}


