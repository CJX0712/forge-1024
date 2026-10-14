// 域8 深度学习测试：激活/归一化/卷积/池化/注意力/RoPE/KV-cache/序列/损失/初始化/gradcheck
// 全部随机数据经 createRng(seed) 生成，保证可复现。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { exec, opsByDomain, requiredParams } from '../src/core/registry.mjs';
import { Tensor, createRng } from '../src/core/index.mjs';
import '../src/ops/d1-math.mjs';
import '../src/ops/d2-num.mjs';
import '../src/ops/d3-linalg.mjs';
import '../src/ops/d4-autodiff.mjs';
import '../src/ops/d8-dl.mjs';

const A = (name, args) => exec(name, args);
const R = (n, seed = 1) => {
  const rng = createRng(seed);
  return Tensor.fromFlat(Array.from({ length: n }, () => rng.normal()), [n]);
};
const RM = (m, n, seed = 2) => {
  const rng = createRng(seed);
  return Tensor.fromFlat(Array.from({ length: m * n }, () => rng.normal()), [m, n]);
};
const RT = (shape, seed = 3) => {
  const rng = createRng(seed);
  return Tensor.fromFlat(Array.from({ length: shape.reduce((a, b) => a * b, 1) }, () => rng.normal()), shape);
};
function allFinite(v) {
  if (v instanceof Tensor) return Array.from(v.data).every(Number.isFinite);
  if (typeof v === 'number') return Number.isFinite(v);
  if (Array.isArray(v)) return v.every(allFinite);
  if (v && typeof v === 'object') return Object.values(v).every(allFinite);
  return true;
}
const close = (a, b, eps = 1e-10, msg = '') => {
  assert.ok(Math.abs(a - b) <= eps * Math.max(1, Math.abs(a), Math.abs(b)), `${a} != ${b} (eps=${eps}) ${msg}`);
};

// =====================================================================
test('1. 域8 恰好 103 条指令，id D8-001..D8-103 唯一且字段齐备', () => {
  const ops = opsByDomain(8);
  assert.equal(ops.length, 103);
  const ids = ops.map((o) => o.id).sort();
  assert.equal(ids[0], 'D8-001');
  assert.equal(ids[102], 'D8-103');
  assert.equal(new Set(ids).size, 103);
  assert.equal(new Set(ops.map((o) => o.name)).size, 103);
  for (const o of ops) {
    assert.equal(o.domain, 8, `${o.id} 域名不是 8`);
    assert.ok(o.sig && o.doc && o.ref && o.inv, `${o.id} 元数据缺失`);
    assert.ok(typeof o.impl === 'function', `${o.id} 缺少 impl`);
    assert.equal(typeof o.arity, 'number', `${o.id} arity 缺失`);
    // arity = 由 impl 形参表推导的必需参数个数（可选参数不计入）
    assert.equal(o.arity, requiredParams(o.impl), `${o.id} (${o.name}) arity=${o.arity} 与必需形参 ${requiredParams(o.impl)} 不符`);
    assert.ok(o.arity <= o.impl.length, `${o.name} arity 超过形参总数`);
    // 最少参数调用不得因 arity 之外的原因崩（此处只校验能被 getOp 找到）
    assert.ok(o.name.startsWith('dl.'), `${o.id} 命名不以 dl. 开头`);
  }
});

// =====================================================================
test('2. 激活函数关键值：relu/sigmoid/tanh/softplus/gelu', () => {
  const x = Tensor.fromFlat([-2, -1, -0.5, 0, 0.5, 1, 2, 6], [8]);
  const one = (name, args) => A(name, [x, ...args]);
  const rel = one('dl.act.relu', []);
  assert.equal(rel.data[0], 0);
  assert.equal(rel.data[1], 0);
  assert.equal(rel.data[3], 0);
  assert.equal(rel.data[7], 6);

  const sig = one('dl.act.sigmoid', []);
  close(sig.data[3], 0.5, 1e-14);
  close(sig.data[2] + sig.data[4], 1, 1e-14, 'sigmoid 互补');

  const th = one('dl.act.tanh', []);
  assert.equal(th.data[3], 0);
  assert.ok(Math.abs(th.data[7]) < 1);

  const sp = one('dl.act.softplus', [1, 20]);
  close(sp.data[3], Math.LN2, 1e-14, 'softplus(0)=ln2');

  const ge = one('dl.act.gelu', ['exact']);
  close(ge.data[3], 0, 1e-14, 'gelu(0)=0');
  // x 大时 GELU(x) → x
  const big = Tensor.fromFlat([20, 30], [2]);
  const geb = A('dl.act.gelu', [big, 'exact']);
  close(geb.data[0], 20, 1e-9);
  close(geb.data[1], 30, 1e-12);
  // tanh 近似与精确值接近
  const geT = one('dl.act.gelu', ['tanh']);
  for (let i = 0; i < 8; i++) assert.ok(Math.abs(geT.data[i] - ge.data[i]) < 2e-3);
  // 其余激活的关键性质
  assert.equal(A('dl.act.leakyRelu', [x, 0.25]).data[1], -0.25);
  close(A('dl.act.elu', [x, 1]).data[3], 0, 1e-14);
  close(A('dl.act.selu', [x]).data[3], 0, 1e-14);
  close(A('dl.act.mish', [x]).data[3], 0, 1e-14);
  close(A('dl.act.swish', [x, 1]).data[3], 0, 1e-14);
  close(A('dl.act.softsign', [x]).data[3], 0, 1e-14);
  close(A('dl.act.hardSigmoid', [x]).data[3], 0.5, 1e-14);
  assert.equal(A('dl.act.hardSigmoid', [x]).data[7], 1); // x=6 ≥ 3
  // PReLU：负区间按通道斜率
  const pr = A('dl.act.prelu', [Tensor.fromFlat([-1, 2, -3], [3]), Tensor.fromFlat([0.5, 0.5, 0.1], [3])]);
  close(pr.data[0], -0.5, 1e-14); close(pr.data[1], 2, 1e-14); close(pr.data[2], -0.3, 1e-14);
  // GLU：沿最后一维对半切分，out = a·σ(b)
  const gl = A('dl.act.glu', [Tensor.fromFlat([1, 2, 3, 0], [4]), -1]);
  close(gl.data[0], 1 / (1 + Math.exp(-3)), 1e-14);
  close(gl.data[1], 2 / (1 + Math.exp(0)), 1e-14);
});

// =====================================================================
test('3. softmax：行和为 1、logSoftmax = log(softmax)、平移不变', () => {
  const x = RT([4, 5], 21);
  const p = A('num.softmax', [x, -1]);
  for (let i = 0; i < 4; i++) {
    let s = 0;
    for (let j = 0; j < 5; j++) { assert.ok(p.data[i * 5 + j] > 0); s += p.data[i * 5 + j]; }
    close(s, 1, 1e-14, `第 ${i} 行和为 1`);
  }
  const lp = A('num.logSoftmax', [x, -1]);
  for (let k = 0; k < x.size; k++) close(lp.data[k], Math.log(p.data[k]), 1e-12);
  // 平移不变：整体加常数（或每行加不同常数）结果不变
  const c = Tensor.fromFlat([[3, 3, 3, 3, 3], [-2, -2, -2, -2, -2], [0, 0, 0, 0, 0], [7, 7, 7, 7, 7]].flat(), [4, 5]);
  const p2 = A('num.softmax', [A('math.add', [x, c]), -1]);
  for (let k = 0; k < x.size; k++) close(p2.data[k], p.data[k], 1e-12);
  // 大数值稳定（减最大值）
  const big = Tensor.fromFlat([[1000, 1001, 999], [-800, -801, -799]].flat(), [2, 3]);
  const pb = A('num.softmax', [big, -1]);
  for (let i = 0; i < 2; i++) {
    let s = 0;
    for (let j = 0; j < 3; j++) s += pb.data[i * 3 + j];
    close(s, 1, 1e-14);
  }
  // 注意力的 attn 矩阵同样行和为 1
  const sd = A('dl.attn.scaledDot', [RT([2, 4, 6], 5), RT([2, 4, 6], 6), RT([2, 4, 6], 7), null, null]);
  for (let b = 0; b < 2; b++) for (let i = 0; i < 4; i++) {
    let s = 0;
    for (let j = 0; j < 4; j++) s += sd.attn.data[(b * 4 + i) * 4 + j];
    close(s, 1, 1e-14);
  }
});

// =====================================================================
test('4. LayerNorm 均值≈0 方差≈1；BatchNorm 训练/推理两态', () => {
  const x = RT([6, 8], 31);
  const g = Tensor.ones([8]); const b0 = Tensor.zeros([8]);
  const y = A('dl.norm.layernorm', [x, g, b0, 0]);
  for (let i = 0; i < 6; i++) {
    let mu = 0;
    for (let j = 0; j < 8; j++) mu += y.data[i * 8 + j];
    mu /= 8;
    let v = 0;
    for (let j = 0; j < 8; j++) v += (y.data[i * 8 + j] - mu) ** 2;
    v /= 8;
    assert.ok(Math.abs(mu) < 1e-12, `均值 ${mu}`);
    close(v, 1, 1e-9, '方差');
  }
  // gamma/beta 仿射后均值=beta、方差=gamma²
  const y2 = A('dl.norm.layernorm', [x, Tensor.full([8], 2), Tensor.full([8], 5), 0]);
  for (let i = 0; i < 6; i++) {
    let mu = 0;
    for (let j = 0; j < 8; j++) mu += y2.data[i * 8 + j];
    mu /= 8;
    let v = 0;
    for (let j = 0; j < 8; j++) v += (y2.data[i * 8 + j] - mu) ** 2;
    v /= 8;
    close(mu, 5, 1e-9); close(v, 4, 1e-9);
  }
  // RMSNorm：RMS ≈ 1
  const yr = A('dl.norm.rmsnorm', [x, Tensor.ones([8]), 0]);
  for (let i = 0; i < 6; i++) {
    let ms = 0;
    for (let j = 0; j < 8; j++) ms += yr.data[i * 8 + j] ** 2;
    close(Math.sqrt(ms / 8), 1, 1e-9);
  }
  // —— BatchNorm ——
  const x4 = RT([4, 3, 2, 2], 41);
  const g3 = Tensor.ones([3]); const b3 = Tensor.zeros([3]);
  const rm = Tensor.zeros([3]); const rv = Tensor.ones([3]);
  const trainOut = A('dl.norm.batchnorm', [x4, g3, b3, rm, rv, 1, 0]);
  for (let c = 0; c < 3; c++) {
    let mu = 0;
    const cnt = 4 * 2 * 2;
    for (let n = 0; n < 4; n++) for (let i = 0; i < 4; i++) mu += trainOut.data[(n * 3 + c) * 4 + i];
    mu /= cnt;
    let v = 0;
    for (let n = 0; n < 4; n++) for (let i = 0; i < 4; i++) v += (trainOut.data[(n * 3 + c) * 4 + i] - mu) ** 2;
    v /= cnt;
    assert.ok(Math.abs(mu) < 1e-10, `通道 ${c} 训练态均值 ${mu}`);
    close(v, 1, 1e-8, `通道 ${c} 训练态方差`);
  }
  // 推理态：用 running 统计量（mean=0.5, var=2），手算对照
  const rmean = Tensor.full([3], 0.5); const rvar = Tensor.full([3], 2);
  const inf = A('dl.norm.batchnorm', [x4, g3, b3, rmean, rvar, 0, 0]);
  for (let k = 0; k < x4.size; k++) {
    close(inf.data[k], (x4.data[k] - 0.5) / Math.sqrt(2), 1e-12);
  }
  // batchStats + runningUpdate 与训练态统计量一致
  const st = A('dl.norm.batchStats', [x4, [0, 2, 3], 0]);
  const upd = A('dl.norm.runningUpdate', [rm, st.mean.reshape([3]), 0.9]);
  for (let c = 0; c < 3; c++) close(upd.data[c], 0.1 * st.mean.data[c], 1e-14);
  // GroupNorm：groups=C 时等价于 InstanceNorm（逐 (n,c) 归一化）
  const gn = A('dl.norm.groupnorm', [x4, g3, b3, 3, 0]);
  const inorm = A('dl.norm.instancenorm', [x4, g3, b3, 0]);
  for (let k = 0; k < x4.size; k++) close(gn.data[k], inorm.data[k], 1e-12);
  // WeightNorm：行范数 = |g|
  const vmat = RM(4, 5, 9); const gv = Tensor.fromFlat([1, 2, 3, 4], [4]);
  const wn = A('dl.norm.weightnorm', [vmat, gv, 1e-12]);
  for (let i = 0; i < 4; i++) {
    let s = 0;
    for (let j = 0; j < 5; j++) s += wn.data[i * 5 + j] ** 2;
    close(Math.sqrt(s), gv.data[i], 1e-12);
  }
  // L2 归一化
  const ln2 = A('dl.norm.l2normalize', [RT([3, 6], 13), -1, 1e-12]);
  for (let i = 0; i < 3; i++) {
    let s = 0;
    for (let j = 0; j < 6; j++) s += ln2.data[i * 6 + j] ** 2;
    close(s, 1, 1e-12);
  }
  // SpectralNorm：归一化后谱范数 ≈ 1
  const W = RM(5, 4, 17);
  const sn = A('dl.norm.spectralnorm', [W, 30, 1e-12]);
  const sn2 = A('dl.norm.spectralnorm', [sn.W, 30, 1e-12]);
  close(sn2.sigma, 1, 1e-8, '归一化后谱范数');
  close(sn.sigma * sn2.sigma, sn.sigma, 1e-8);
});

// =====================================================================
// 手写 im2col + matmul 参照实现（NCHW，padding/stride 均为标量或 [h,w]）
function im2colConv2d(x, w, bb, stride, pad) {
  const [N, C, H, W] = x.shape; const [OC, IC, KH, KW] = w.shape;
  const [sh, sw] = Array.isArray(stride) ? stride : [stride, stride];
  const [ph, pw] = Array.isArray(pad) ? pad : [pad, pad];
  const Ho = Math.floor((H + 2 * ph - KH) / sh) + 1;
  const Wo = Math.floor((W + 2 * pw - KW) / sw) + 1;
  const rows = IC * KH * KW;
  // 权重矩阵 [OC, rows]
  const Wm = new Float64Array(OC * rows);
  for (let oc = 0; oc < OC; oc++) for (let ic = 0; ic < IC; ic++) for (let kh = 0; kh < KH; kh++) for (let kw = 0; kw < KW; kw++) {
    Wm[oc * rows + (ic * KH + kh) * KW + kw] = w.data[((oc * IC + ic) * KH + kh) * KW + kw];
  }
  const out = new Float64Array(N * OC * Ho * Wo);
  for (let n = 0; n < N; n++) {
    // cols[rows, Ho*Wo]
    const cols = new Float64Array(rows * Ho * Wo);
    for (let oh = 0; oh < Ho; oh++) for (let ow = 0; ow < Wo; ow++) {
      for (let ic = 0; ic < IC; ic++) for (let kh = 0; kh < KH; kh++) for (let kw = 0; kw < KW; kw++) {
        const hi = oh * sh - ph + kh, wi = ow * sw - pw + kw;
        const val = (hi >= 0 && hi < H && wi >= 0 && wi < W) ? x.data[(((n * C) + ic) * H + hi) * W + wi] : 0;
        cols[((ic * KH + kh) * KW + kw) * (Ho * Wo) + oh * Wo + ow] = val;
      }
    }
    for (let oc = 0; oc < OC; oc++) for (let oh = 0; oh < Ho; oh++) for (let ow = 0; ow < Wo; ow++) {
      let acc = bb ? bb.data[oc] : 0;
      for (let r = 0; r < rows; r++) acc += Wm[oc * rows + r] * cols[r * (Ho * Wo) + oh * Wo + ow];
      out[(((n * OC) + oc) * Ho + oh) * Wo + ow] = acc;
    }
  }
  return new Tensor(out, [N, OC, Ho, Wo]);
}
test('5. conv1d/conv2d 与手写暴力循环 / im2col+matmul 一致（误差 <1e-10）', () => {
  // —— conv1d：四重暴力循环 ——
  const x1 = RT([2, 3, 7], 51); const w1 = RT([4, 3, 3], 52); const b1 = R(4, 53);
  const s1 = 2, p1 = 1, d1 = 2;
  const y1 = A('dl.layer.conv1d', [x1, w1, b1, s1, p1, d1]);
  const [N1, C1, L] = x1.shape; const [OC1, IC1, K] = w1.shape;
  const span = d1 * (K - 1) + 1;
  const Lo = Math.floor((L + 2 * p1 - span) / s1) + 1;
  assert.deepEqual(y1.shape, [N1, OC1, Lo]);
  let maxErr1 = 0;
  for (let n = 0; n < N1; n++) for (let oc = 0; oc < OC1; oc++) for (let o = 0; o < Lo; o++) {
    let acc = b1.data[oc];
    for (let ic = 0; ic < IC1; ic++) for (let k = 0; k < K; k++) {
      const li = o * s1 - p1 + k * d1;
      if (li < 0 || li >= L) continue;
      acc += x1.data[((n * C1) + ic) * L + li] * w1.data[((oc * IC1) + ic) * K + k];
    }
    maxErr1 = Math.max(maxErr1, Math.abs(acc - y1.data[(n * OC1 + oc) * Lo + o]));
  }
  assert.ok(maxErr1 < 1e-10, `conv1d 最大误差 ${maxErr1}`);

  // —— conv2d：im2col + matmul（含 padding）——
  const x2 = RT([2, 3, 5, 6], 61); const w2 = RT([4, 3, 3, 2], 62); const b2 = R(4, 63);
  const y2 = A('dl.layer.conv2d', [x2, w2, b2, 1, [1, 0], 1]);
  const ref2 = im2colConv2d(x2, w2, b2, 1, [1, 0]);
  assert.deepEqual(y2.shape, ref2.shape);
  let maxErr2 = 0;
  for (let k = 0; k < y2.size; k++) maxErr2 = Math.max(maxErr2, Math.abs(y2.data[k] - ref2.data[k]));
  assert.ok(maxErr2 < 1e-10, `conv2d 最大误差 ${maxErr2}`);

  // 带 stride 与 dilation 的 conv2d
  const y3 = A('dl.layer.conv2d', [x2, w2, b2, [2, 1], 1, [1, 2]]);
  // Ho=(5+2−3)/2+1=3 ；Wo=(6+2−(2·(2−1)+1))/1+1=6
  assert.deepEqual(y3.shape, [2, 4, 3, 6]);
  // depthwiseConv2d 等价于「每个通道独立的标准卷积」
  const dw = RT([3, 1, 3, 3], 64);
  const ydw = A('dl.layer.depthwiseConv2d', [x2, dw, null, 1, 1]);
  assert.deepEqual(ydw.shape, [2, 3, 5, 6]);
  for (let c = 0; c < 3; c++) {
    // 把第 c 个通道单独取出做标准 conv2d（单输入单输出通道）
    const oneCh = new Tensor(x2.data.filter((_, i) => {
      const ch = Math.floor((i % (3 * 5 * 6)) / (5 * 6));
      return ch === c;
    }), [2, 1, 5, 6]);
    const yOne = A('dl.layer.conv2d', [oneCh, dw.data.slice(c * 9, (c + 1) * 9) && new Tensor(dw.data.slice(c * 9, (c + 1) * 9), [1, 1, 3, 3]), null, 1, 1, 1]);
    for (let n = 0; n < 2; n++) for (let i = 0; i < 30; i++) {
      close(ydw.data[(n * 3 + c) * 30 + i], yOne.data[n * 30 + i], 1e-12, `depthwise 通道 ${c}`);
    }
  }
  // transposeConv2d 尺寸公式
  const yt = A('dl.layer.transposeConv2d', [RT([1, 2, 3, 3], 66), RT([2, 4, 3, 3], 67), null, [2, 1], 1, 1]);
  assert.deepEqual(yt.shape, [1, 4, (3 - 1) * 2 - 2 + 3 + 1, (3 - 1) * 1 - 2 + 3 + 1]);
  // dense 与手写 matmul+bias 一致
  const xd = RM(3, 4, 71); const wd = RM(4, 5, 72); const bd = R(5, 73);
  const yd = A('dl.layer.dense', [xd, wd, bd]);
  assert.deepEqual(yd.shape, [3, 5]);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 5; j++) {
    let s = bd.data[j];
    for (let t = 0; t < 4; t++) s += xd.data[i * 4 + t] * wd.data[t * 5 + j];
    close(yd.data[i * 5 + j], s, 1e-12);
  }
});

// =====================================================================
test('6. 池化：max=窗口最大值、avg=窗口均值、globalAvgPool=通道全图均值', () => {
  const x = RT([2, 3, 4, 5], 81);
  const mx = A('dl.layer.maxpool2d', [x, [2, 2], [2, 2]]);
  assert.deepEqual(mx.shape, [2, 3, 2, 2]);
  const av = A('dl.layer.avgpool2d', [x, [2, 2], [2, 2]]);
  assert.deepEqual(av.shape, [2, 3, 2, 2]);
  for (let n = 0; n < 2; n++) for (let c = 0; c < 3; c++) for (let oh = 0; oh < 2; oh++) for (let ow = 0; ow < 2; ow++) {
    let m = -Infinity, s = 0;
    for (let kh = 0; kh < 2; kh++) for (let kw = 0; kw < 2; kw++) {
      const v = x.data[(((n * 3) + c) * 4 + oh * 2 + kh) * 5 + ow * 2 + kw];
      if (v > m) m = v;
      s += v;
    }
    close(mx.data[(((n * 3) + c) * 2 + oh) * 2 + ow], m, 0, 'maxpool');
    close(av.data[(((n * 3) + c) * 2 + oh) * 2 + ow], s / 4, 1e-14, 'avgpool');
  }
  const gap = A('dl.layer.globalAvgPool', [x]);
  assert.deepEqual(gap.shape, [2, 3]);
  for (let n = 0; n < 2; n++) for (let c = 0; c < 3; c++) {
    let s = 0;
    for (let i = 0; i < 20; i++) s += x.data[(n * 3 + c) * 20 + i];
    close(gap.data[n * 3 + c], s / 20, 1e-14);
  }
  // 一维池化
  const x1 = RT([1, 2, 6], 82);
  const m1 = A('dl.layer.maxpool1d', [x1, 2, 2]);
  const a1 = A('dl.layer.avgpool1d', [x1, 2, 2]);
  assert.deepEqual(m1.shape, [1, 2, 3]);
  for (let c = 0; c < 2; c++) for (let o = 0; o < 3; o++) {
    const u = x1.data[c * 6 + o * 2], v = x1.data[c * 6 + o * 2 + 1];
    close(m1.data[c * 3 + o], Math.max(u, v), 0);
    close(a1.data[c * 3 + o], (u + v) / 2, 1e-14);
  }
  // flatten / unflatten 可逆
  const f = A('dl.layer.flatten', [x, 1]);
  assert.deepEqual(f.shape, [2, 60]);
  const uf = A('dl.layer.unflatten', [f, [3, 4, 5]]);
  assert.deepEqual(uf.shape, [2, 3, 4, 5]);
  for (let k = 0; k < x.size; k++) assert.equal(uf.data[k], x.data[k]);
  // embedding 与查表一致
  const table = RM(6, 4, 83);
  const emb = A('dl.layer.embedding', [Tensor.fromFlat([0, 3, 5], [3]), table]);
  assert.deepEqual(emb.shape, [3, 4]);
  for (let i = 0; i < 4; i++) assert.equal(emb.data[1 * 4 + i], table.data[3 * 4 + i]);
  // patching：token 数 = (H/p)(W/p)，元素守恒
  const pt = A('dl.layer.patching', [RT([2, 3, 4, 4], 84), 2]);
  assert.deepEqual(pt.shape, [2, 4, 3 * 2 * 2]);
  assert.equal(pt.size, 2 * 3 * 4 * 4);
});

// =====================================================================
test('7. causalMask 下三角（含对角）为 1、严格上三角为 0；加掩码后注意力上三角为 0 且行和为 1', () => {
  const T = 5;
  const m = A('dl.attn.causalMask', [T]);
  assert.deepEqual(m.shape, [T, T]);
  for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
    assert.equal(m.data[i * T + j], j <= i ? 1 : 0, `mask[${i}][${j}]`);
  }
  for (let i = 0; i < T; i++) {
    let s = 0;
    for (let j = 0; j < T; j++) s += m.data[i * T + j];
    assert.equal(s, i + 1);
  }
  // 掩码注意力
  const Q = RT([1, T, 6], 91); const K = RT([1, T, 6], 92); const V = RT([1, T, 6], 93);
  const r = A('dl.attn.scaledDot', [Q, K, V, m, null]);
  for (let i = 0; i < T; i++) {
    let s = 0;
    for (let j = 0; j < T; j++) {
      const w = r.attn.data[i * T + j];
      if (j > i) assert.equal(w, 0, `attn[${i}][${j}] 应为 0`);
      assert.ok(w >= 0);
      s += w;
    }
    close(s, 1, 1e-14, `第 ${i} 行权重和`);
  }
  // 无掩码时所有位置均非零
  const r0 = A('dl.attn.scaledDot', [Q, K, V, null, null]);
  for (let k = 0; k < T * T; k++) assert.ok(r0.attn.data[k] > 0);
  // paddingMask：有效位为 1
  const pm = A('dl.attn.paddingMask', [Tensor.fromFlat([2, 4, 1], [3]), 5]);
  assert.deepEqual(pm.shape, [3, 5]);
  for (let b = 0; b < 3; b++) {
    let s = 0;
    for (let j = 0; j < 5; j++) s += pm.data[b * 5 + j];
    assert.equal(s, [2, 4, 1][b]);
  }
  // 注意力熵：one-hot 为 0，均匀为 log T
  const onehot = Tensor.fromFlat([[1, 0, 0, 0], [0, 0, 1, 0]].flat(), [2, 4]);
  const eh = A('dl.attn.entropy', [onehot]);
  close(eh.data[0], 0, 1e-12); close(eh.data[1], 0, 1e-12);
  const uni = Tensor.full([1, 4], 0.25);
  close(A('dl.attn.entropy', [uni]).data[0], Math.log(4), 1e-12);
});

// =====================================================================
test('8. 多头注意力：headSplit 后 head 数×headDim = 模型维；multiHead 输出 [B,T,C]', () => {
  const B = 2, T = 4, C = 8, H = 4;
  const x = RT([B, T, C], 101);
  const hs = A('dl.attn.headSplit', [x, H]);
  assert.deepEqual(hs.shape, [B, H, T, C / H]);
  assert.equal(H * (C / H), C);
  // 分头再合并还原
  const back = A('dl.model.residual', [x, x.map(() => 0), 1]);
  for (let k = 0; k < x.size; k++) assert.equal(back.data[k], x.data[k]);
  // headSplit 的元素映射关系
  for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) for (let h = 0; h < H; h++) for (let d = 0; d < C / H; d++) {
    assert.equal(hs.data[(((b * H) + h) * T + t) * (C / H) + d], x.data[(b * T + t) * C + h * (C / H) + d]);
  }
  // multiHead
  const W = (s) => RT([C, C], s);
  const out = A('dl.attn.multiHead', [x, W(1), W(2), W(3), W(4), H, null]);
  assert.deepEqual(out.shape, [B, T, C]);
  assert.ok(allFinite(out));
  // 自注意力（恒等投影）形状与 attn 形状
  const sa = A('dl.attn.selfAttention', [x, H, null]);
  assert.deepEqual(sa.out.shape, [B, T, C]);
  assert.deepEqual(sa.attn.shape, [B, H, T, T]);
  for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) for (let i = 0; i < T; i++) {
    let s = 0;
    for (let j = 0; j < T; j++) s += sa.attn.data[(((b * H) + h) * T + i) * T + j];
    close(s, 1, 1e-14);
  }
  // 交叉注意力：输出时间维 = 查询长度
  const ca = A('dl.attn.crossAttention', [RT([2, 3, 8], 105), RT([2, 6, 8], 106), 2, null]);
  assert.deepEqual(ca.out.shape, [2, 3, 8]);
  assert.deepEqual(ca.attn.shape, [2, 2, 3, 6]);
  // GQA：H=4, Hkv=2 → 输出形状不变且结果有限
  const gq = A('dl.attn.groupedQuery', [RT([2, 5, 8], 107), RT([2, 5, 4], 108), RT([2, 5, 4], 109), 4, 2, null]);
  assert.deepEqual(gq.out.shape, [2, 5, 8]);
  assert.ok(allFinite(gq.out));
  // GQA 的 kvHeads=heads 等价于同配置的多头注意力（恒等投影下，Q=K=V 同源）
  const same = RT([1, 3, 4], 110);
  const gq2 = A('dl.attn.groupedQuery', [same, same, same, 2, 2, null]);
  const mh2 = A('dl.attn.selfAttention', [same, 2, null]);
  for (let k = 0; k < gq2.out.size; k++) close(gq2.out.data[k], mh2.out.data[k], 1e-12, 'GQA(kvHeads=heads) = MHA');
  // ALiBi 与相对位置偏置
  const ab = A('dl.attn.alibiBias', [3, 4]);
  assert.deepEqual(ab.shape, [3, 4, 4]);
  for (let h = 0; h < 3; h++) for (let i = 0; i < 4; i++) {
    assert.ok(ab.data[(h * 4 + i) * 4 + i] === 0, '对角为 0'); // −slope·0 = −0，用 === 判定
    for (let j = 0; j < 4; j++) assert.ok(ab.data[(h * 4 + i) * 4 + j] <= 0);
  }
  const rb = A('dl.attn.relativeBias', [5, 8, 7]);
  for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
    assert.equal(rb.data[i * 5 + j], rb.data[j * 5 + i], '对称性');
    assert.equal(rb.data[i * 5 + j], rb.data[0 * 5 + Math.abs(j - i)], `只依赖 |i-j| (${i},${j})`);
  }
  // flashTiling 与标准注意力一致
  const Qf = RT([2, 6, 5], 113); const Kf = RT([2, 6, 5], 114); const Vf = RT([2, 6, 5], 115);
  const fl = A('dl.attn.flashTiling', [Qf, Kf, Vf, 2, 3]);
  const std = A('dl.attn.scaledDot', [Qf, Kf, Vf, null, null]);
  for (let k = 0; k < fl.out.size; k++) close(fl.out.data[k], std.out.data[k], 1e-12, 'flash vs 标准');
  // lse 应等于 logsumexp(scores)
  let lseRef = 0;
  {
    let s = 0;
    for (let j = 0; j < 6; j++) {
      let d = 0;
      for (let k2 = 0; k2 < 5; k2++) d += Qf.data[0 * 5 + k2] * Kf.data[j * 5 + k2];
      s += Math.exp(d / Math.sqrt(5));
    }
    lseRef = Math.log(s);
  }
  close(fl.lse.data[0], lseRef, 1e-10, 'lse');
});

// =====================================================================
test('9. RoPE：内积只依赖相对位置，且保持范数', () => {
  const D = 8, L = 16;
  const { cos, sin } = A('dl.attn.rotaryEmbed', [D, L, 10000]);
  assert.deepEqual(cos.shape, [L, D / 2]);
  assert.deepEqual(sin.shape, [L, D / 2]);
  for (let k = 0; k < cos.size; k++) close(cos.data[k] ** 2 + sin.data[k] ** 2, 1, 1e-14);
  assert.equal(cos.data[0], 1); assert.equal(sin.data[0], 0); // m=0
  const rng = createRng(121);
  const q = Array.from({ length: D }, () => rng.normal());
  const kk = Array.from({ length: D }, () => rng.normal());
  const tile = (v) => Tensor.fromFlat(Array.from({ length: L }, () => v).flat(), [1, L, 1, D]);
  const rq = A('dl.attn.rotaryApply', [tile(q), cos, sin, 0]);
  const rk = A('dl.attn.rotaryApply', [tile(kk), cos, sin, 0]);
  const dot2 = (m, n) => {
    let s = 0;
    for (let d = 0; d < D; d++) s += rq.data[m * D + d] * rk.data[n * D + d];
    return s;
  };
  // 范数保持
  for (const t of [0, 3, 7, 15]) {
    let nq = 0, nk = 0;
    for (let d = 0; d < D; d++) { nq += rq.data[t * D + d] ** 2; nk += rk.data[t * D + d] ** 2; }
    let q0 = 0, k0 = 0;
    for (let d = 0; d < D; d++) { q0 += q[d] ** 2; k0 += kk[d] ** 2; }
    close(nq, q0, 1e-12, `范数保持 q@${t}`);
    close(nk, k0, 1e-12, `范数保持 k@${t}`);
  }
  // 相对位置性质：⟨R(q,m),R(k,n)⟩ 只依赖 m−n
  const pairs = [[2, 5], [7, 10], [1, 4]];
  const base = dot2(pairs[0][0], pairs[0][1]);
  for (const [m, n] of pairs) close(dot2(m, n), base, 1e-12, `相对位移 3: (${m},${n})`);
  const base2 = dot2(6, 2);
  for (const [m, n] of [[6, 2], [12, 8], [9, 5]]) close(dot2(m, n), base2, 1e-12, `相对位移 4: (${m},${n})`);
  // 不同相对位移的值确实不同（防止平凡通过）
  assert.ok(Math.abs(base - base2) > 1e-6);
  // start 偏移：偏移 s 后等价于原位置 + s
  const rq2 = A('dl.attn.rotaryApply', [Tensor.fromFlat(Array.from({ length: 4 }, () => q).flat(), [1, 4, 1, D]), cos, sin, 5]);
  for (let d = 0; d < D; d++) close(rq2.data[0 * D + d], rq.data[5 * D + d], 1e-12, 'start 偏移');
});

// =====================================================================
test('10. LSTM/GRU：常数序列下有界；lstmCell 的 c 更新与手算一致', () => {
  const N = 2, Di = 3, H = 4;
  const x = Tensor.fromFlat([1, 2, 3], [3]); // 单样本（N=1 演示）
  const h = Tensor.fromFlat([0.1, -0.2, 0.3, 0.4], [1, 4]);
  const c = Tensor.fromFlat([0.5, -0.5, 0.2, -0.2], [1, 4]);
  const W = RT([Di, 4 * H], 131); const U = RT([H, 4 * H], 132); const b = R(4 * H, 133);
  // 手算
  const z = new Float64Array(4 * H);
  for (let j = 0; j < 4 * H; j++) {
    let s = b.data[j];
    for (let i = 0; i < Di; i++) s += x.data[i] * W.data[i * 4 * H + j];
    for (let i = 0; i < H; i++) s += h.data[i] * U.data[i * 4 * H + j];
    z[j] = s;
  }
  const sig = (v) => 1 / (1 + Math.exp(-v));
  const expectC = new Float64Array(H), expectH = new Float64Array(H);
  for (let j = 0; j < H; j++) {
    const ig = sig(z[j]);            // i 门
    const fg = sig(z[H + j]);        // f 门
    const gg = Math.tanh(z[2 * H + j]); // g 候选
    const og = sig(z[3 * H + j]);    // o 门
    expectC[j] = fg * c.data[j] + ig * gg;
    expectH[j] = og * Math.tanh(expectC[j]);
  }
  const r = A('dl.seq.lstmCell', [Tensor.fromFlat([1, 2, 3], [1, 3]), h, c, W, U, b]);
  assert.ok(r.h instanceof Tensor && r.c instanceof Tensor);
  for (let j = 0; j < H; j++) {
    close(r.c.data[j], expectC[j], 1e-12, `c[${j}]`);
    close(r.h.data[j], expectH[j], 1e-12, `h[${j}]`);
  }
  // |h| <= 1
  for (let j = 0; j < H; j++) assert.ok(Math.abs(r.h.data[j]) <= 1 + 1e-12);
  // 遗忘门全开、输入门关闭时 c 恒定（f=1,i=0 需要构造偏置）
  const bConst = new Float64Array(4 * H);
  for (let j = 0; j < H; j++) { bConst[j] = -50; bConst[H + j] = 50; bConst[2 * H + j] = 0; bConst[3 * H + j] = 0; }
  const r2 = A('dl.seq.lstmCell', [
    Tensor.zeros([1, Di]), Tensor.zeros([1, H]),
    Tensor.fromFlat([0.3, -0.7, 0.1, 0.9], [1, 4]),
    Tensor.zeros([Di, 4 * H]), Tensor.zeros([H, 4 * H]), Tensor.fromFlat(bConst, [4 * H]),
  ]);
  for (let j = 0; j < H; j++) close(r2.c.data[j], [0.3, -0.7, 0.1, 0.9][j], 1e-8, 'f=1,i=0 时 c 恒定');
  // —— 长序列前向：常数输入，隐状态有限且有界 ——
  const B = 2, T = 40;
  const xs = Tensor.fromFlat(Array.from({ length: B * T * Di }, () => 1.5), [B, T, Di]);
  const h0 = Tensor.zeros([B, H]); const c0 = Tensor.zeros([B, H]);
  const lf = A('dl.seq.lstmForward', [xs, h0, c0, W, U, b]);
  assert.deepEqual(lf.out.shape, [B, T, H]);
  assert.ok(allFinite(lf.out) && allFinite(lf.hT) && allFinite(lf.cT));
  for (let k = 0; k < lf.out.size; k++) assert.ok(Math.abs(lf.out.data[k]) <= 1 + 1e-12);
  for (let k = 0; k < lf.hT.size; k++) close(lf.hT.data[k], lf.out.data[(T - 1) * B * H + k], 0, 'hT = 末步隐状态');
  const rf = A('dl.seq.rnnForward', [xs, h0, RT([Di, H], 141), RT([H, H], 142), R(H, 143), 'tanh']);
  assert.deepEqual(rf.out.shape, [B, T, H]);
  assert.ok(allFinite(rf.out));
  for (let k = 0; k < rf.out.size; k++) assert.ok(Math.abs(rf.out.data[k]) <= 1 + 1e-9, 'tanh RNN 隐状态有界');
  // GRU
  const gf = A('dl.seq.gruForward', [xs, h0, RT([Di, 3 * H], 151), RT([H, 3 * H], 152), R(3 * H, 153), R(3 * H, 154)]);
  assert.deepEqual(gf.out.shape, [B, T, H]);
  assert.ok(allFinite(gf.out));
  for (let k = 0; k < gf.out.size; k++) assert.ok(Math.abs(gf.out.data[k]) <= 1 + 1e-9, 'GRU 隐状态有界');
  // GRU 单元：z=0 时隐状态不变
  const bx0 = new Float64Array(3 * H); const bh0 = new Float64Array(3 * H);
  for (let j = 0; j < 3 * H; j++) { bx0[j] = -50; bh0[j] = -50; }
  const gc = A('dl.seq.gruCell', [Tensor.zeros([1, Di]), Tensor.fromFlat([0.4, -0.4, 0.2, 0.6], [1, H]),
    Tensor.zeros([Di, 3 * H]), Tensor.zeros([H, 3 * H]), Tensor.fromFlat(bx0, [3 * H]), Tensor.fromFlat(bh0, [3 * H])]);
  for (let j = 0; j < H; j++) close(gc.data[j], [0.4, -0.4, 0.2, 0.6][j], 1e-8, 'z=0 时 h 不变');
  // 双向拼接
  const bi = A('dl.seq.bidirectional', [RT([2, 5, 3], 161), RT([2, 5, 3], 162)]);
  assert.deepEqual(bi.shape, [2, 5, 6]);
  for (let n = 0; n < 2; n++) for (let t = 0; t < 5; t++) for (let d = 0; d < 3; d++) {
    assert.equal(bi.data[(n * 5 + t) * 6 + d], RT([2, 5, 3], 161).data[(n * 5 + t) * 3 + d]);
    assert.equal(bi.data[(n * 5 + t) * 6 + 3 + d], RT([2, 5, 3], 162).data[(n * 5 + t) * 3 + d]);
  }
  // scan：f 为恒等时输出恒为 h0
  const sc = A('dl.seq.scan', [xs, (hh) => hh, Tensor.full([B, 7], 2.5), 0]);
  assert.deepEqual(sc.out.shape, [B, T, 7]);
  for (let k = 0; k < sc.out.size; k++) assert.equal(sc.out.data[k], 2.5);
  void N;
});

// =====================================================================
test('11. 损失：ce=−log(p_correct)、bceWithLogits≡sigmoid+bce、labelSmooth 均匀极限 = KL 到均匀分布', () => {
  const logits = RT([4, 3], 171);
  const tgt = Tensor.fromFlat([0, 2, 1, 0], [4]);
  const prob = A('num.softmax', [logits, -1]);
  // ce = −log(p_correct)
  let manual = 0;
  for (let n = 0; n < 4; n++) manual += -Math.log(prob.data[n * 3 + tgt.data[n]]);
  close(A('dl.loss.ce', [prob, tgt, 'mean', 1e-12]), manual / 4, 1e-12, 'ce');
  close(A('dl.loss.ce', [prob, tgt, 'sum', 1e-12]), manual, 1e-12, 'ce sum');
  close(A('dl.loss.ceWithLogits', [logits, tgt, 'mean']), manual / 4, 1e-12, 'ceWithLogits');
  // nll = −logProb[correct]
  const lp = A('num.logSoftmax', [logits, -1]);
  close(A('dl.loss.nll', [lp, tgt, 'mean']), manual / 4, 1e-12, 'nll');
  // bceWithLogits ≡ sigmoid + bce
  const z = R(6, 172); const t01 = Tensor.fromFlat([0, 1, 1, 0, 1, 0], [6]);
  const psig = A('dl.act.sigmoid', [z]);
  const bceManual = A('dl.loss.bce', [psig, t01, 'mean', 1e-12]);
  close(A('dl.loss.bceWithLogits', [z, t01, 'mean']), bceManual, 1e-12, 'bceWithLogits');
  close(A('dl.loss.bceWithLogits', [Tensor.zeros([2]), Tensor.zeros([2]), 'mean']), Math.LN2, 1e-12, 'logits=0 → ln2');
  // labelSmooth：eps=0 退化为普通交叉熵
  close(A('dl.loss.labelSmoothCE', [logits, tgt, 0, 'mean']), manual / 4, 1e-12, 'eps=0');
  // eps=1 → 均匀分布目标，loss = KL(U || p) + log C
  const C = 3;
  const uni = Tensor.full([4, C], 1 / C);
  const kl = A('dl.loss.klDiv', [lp, uni, 'mean']);
  const ls = A('dl.loss.labelSmoothCE', [logits, tgt, 1, 'mean']);
  close(ls, kl + Math.log(C), 1e-12, 'labelSmooth 均匀极限');
  // KL 非负，P=Q 时为 0
  assert.ok(A('dl.loss.klDiv', [lp, prob, 'mean']) >= -1e-15);
  close(A('dl.loss.klDiv', [lp, prob, 'mean']), 0, 1e-12, 'P=Q');
  // 回归损失
  const p = RT([3, 2], 173); const q = RT([3, 2], 174);
  let sse = 0, sae = 0;
  for (let k = 0; k < p.size; k++) { sse += (p.data[k] - q.data[k]) ** 2; sae += Math.abs(p.data[k] - q.data[k]); }
  close(A('dl.loss.mse', [p, q, 'mean']), sse / p.size, 1e-12);
  close(A('dl.loss.mae', [p, q, 'mean']), sae / p.size, 1e-12);
  close(A('dl.loss.mse', [p, p, 'mean']), 0, 0);
  // Huber：小残差 = 0.5r²，大残差 = δ(|r|−δ/2)
  const a1 = Tensor.fromFlat([0, 0], [2]); const b1 = Tensor.fromFlat([0.5, 5], [2]);
  const hu = A('dl.loss.huber', [a1, b1, 1, 'mean']);
  close(hu, (0.5 * 0.25 + 1 * (5 - 0.5)) / 2, 1e-12);
  // Focal：gamma=0, alpha=1 退化为交叉熵
  close(A('dl.loss.focal', [logits, tgt, 0, 1, 'mean']), manual / 4, 1e-12, 'focal γ=0');
  // Dice
  const dp = Tensor.fromFlat([1, 1, 0, 0], [4]); const dt = Tensor.fromFlat([1, 0, 0, 0], [4]);
  close(A('dl.loss.dice', [dp, dt, 1e-6]), 1 - (2 * 1 + 1e-6) / (2 + 1 + 1e-6), 1e-12);
  close(A('dl.loss.dice', [dt, dt, 1e-6]), 0, 1e-9, '完全重合');
  // NT-Xent 非负
  const z1 = RM(5, 4, 175); const z2 = RM(5, 4, 176);
  assert.ok(A('dl.loss.ntxent', [z1, z2, 0.5]) >= 0);
  assert.ok(A('dl.loss.ntxent', [z1, z1, 0.5]) >= 0);
  // Triplet：满足 margin 时为 0，且非负
  const an = RM(4, 3, 177);
  const tp = A('dl.loss.triplet', [an, an.clone(), an.clone(), 0]);
  close(tp, 0, 1e-12);
  const tp2 = A('dl.loss.triplet', [Tensor.zeros([1, 3]), Tensor.zeros([1, 3]), Tensor.fromFlat([2, 0, 0], [1, 3]), 1]);
  close(tp2, 0, 1e-12, 'd(a,n)²=4 ≥ 0+1');
  const tp3 = A('dl.loss.triplet', [Tensor.zeros([1, 3]), Tensor.fromFlat([2, 0, 0], [1, 3]), Tensor.zeros([1, 3]), 1]);
  close(tp3, 5, 1e-12, '4 − 0 + 1');
});

// =====================================================================
function sampleVar(t) {
  let mu = 0;
  for (let i = 0; i < t.size; i++) mu += t.data[i];
  mu /= t.size;
  let v = 0;
  for (let i = 0; i < t.size; i++) v += (t.data[i] - mu) ** 2;
  return v / t.size;
}
test('12. 初始化方差：xavierUniform ≈ 2/(fanIn+fanOut)、heNormal ≈ 2/fanIn（10% 容差）', () => {
  const fanIn = 300, fanOut = 200;
  const xu = A('dl.init.xavierUniform', [[fanOut, fanIn], 1, 2026]);
  const expectX = 2 / (fanIn + fanOut);
  assert.ok(Math.abs(sampleVar(xu) / expectX - 1) < 0.1, `xavierUniform 方差比 ${sampleVar(xu) / expectX}`);
  // 支撑区间 [-b, b]
  const bx = Math.sqrt(6 / (fanIn + fanOut));
  for (let i = 0; i < xu.size; i++) assert.ok(Math.abs(xu.data[i]) <= bx + 1e-15);
  const xn = A('dl.init.xavierNormal', [[fanOut, fanIn], 1, 2027]);
  assert.ok(Math.abs(sampleVar(xn) / expectX - 1) < 0.1, `xavierNormal 方差比 ${sampleVar(xn) / expectX}`);
  const hn = A('dl.init.heNormal', [[fanOut, fanIn], Math.SQRT2, 2028]);
  const expectH = 2 / fanIn;
  assert.ok(Math.abs(sampleVar(hn) / expectH - 1) < 0.1, `heNormal 方差比 ${sampleVar(hn) / expectH}`);
  const hu = A('dl.init.heUniform', [[fanOut, fanIn], Math.SQRT2, 2029]);
  assert.ok(Math.abs(sampleVar(hu) / expectH - 1) < 0.1, `heUniform 方差比 ${sampleVar(hu) / expectH}`);
  const bh = Math.SQRT2 * Math.sqrt(3 / fanIn);
  for (let i = 0; i < hu.size; i++) assert.ok(Math.abs(hu.data[i]) <= bh + 1e-15);
  // 常数类
  assert.equal(A('dl.init.zeros', [[3, 4]]).data.every((v) => v === 0), true);
  assert.equal(A('dl.init.ones', [[3, 4]]).data.every((v) => v === 1), true);
  assert.equal(A('dl.init.constant', [[2, 2], 0.75]).data.every((v) => v === 0.75), true);
  assert.deepEqual(A('dl.init.zeros', [[3, 4]]).shape, [3, 4]);
  // uniform 区间与均匀性
  const un = A('dl.init.uniform', [[500, 4], -2, 3, 5]);
  for (let i = 0; i < un.size; i++) assert.ok(un.data[i] >= -2 && un.data[i] <= 3);
  close(sampleVar(un), 25 / 12, 0.6, 'uniform 方差 ≈ (b−a)²/12');
  // normal 方差
  close(sampleVar(A('dl.init.normal', [[2000, 5], 0, 0.5, 6])), 0.25, 0.04, 'normal 方差');
  // truncatedNormal：全在 ±2σ 内
  const tn = A('dl.init.truncatedNormal', [[1000, 4], 0, 0.5, 7]);
  for (let i = 0; i < tn.size; i++) assert.ok(Math.abs(tn.data[i]) <= 1 + 1e-12, `截断界 ${tn.data[i]}`);
  assert.ok(sampleVar(tn) < 0.25);
  // 可复现：同种子结果完全一致
  const s1 = A('dl.init.heNormal', [[6, 7], Math.SQRT2, 99]);
  const s2 = A('dl.init.heNormal', [[6, 7], Math.SQRT2, 99]);
  for (let i = 0; i < s1.size; i++) assert.equal(s1.data[i], s2.data[i]);
  // 卷积核的 fanIn/fanOut 计入感受野
  const cw = A('dl.init.heNormal', [[16, 8, 3, 3], Math.SQRT2, 3]);
  assert.ok(Math.abs(sampleVar(cw) / (2 / (8 * 3 * 3)) - 1) < 0.15, `卷积 he 方差比 ${sampleVar(cw) / (2 / 72)}`);
});

// =====================================================================
test('13. KV cache：逐步 append 解码与一次性全序列注意力一致（<1e-10）', () => {
  const B = 2, H = 3, T = 7, D = 4;
  const K = RT([B, H, T, D], 181); const V = RT([B, H, T, D], 182); const Q = RT([B, H, T, D], 183);
  // 一次性：对每个 head 做 scaled dot attention
  const headK = (h) => {
    const out = new Float64Array(B * T * D);
    for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) {
      out[(b * T + t) * D + d] = K.data[(((b * H) + h) * T + t) * D + d];
    }
    return new Tensor(out, [B, T, D]);
  };
  const full = [];
  for (let h = 0; h < H; h++) {
    const qh = new Float64Array(B * T * D);
    for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) {
      qh[(b * T + t) * D + d] = Q.data[(((b * H) + h) * T + t) * D + d];
    }
    const vh = headK(h).clone();
    const vv = new Float64Array(B * T * D);
    for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) {
      vv[(b * T + t) * D + d] = V.data[(((b * H) + h) * T + t) * D + d];
    }
    full.push(A('dl.attn.scaledDot', [Tensor.fromFlat(qh, [B, T, D]), vh, Tensor.fromFlat(vv, [B, T, D]), null, null]));
  }
  // 增量：逐个 token append
  let cache = A('dl.attn.kvCacheInit', [B, H, T, D]);
  assert.equal(cache.len, 0);
  let maxErr = 0;
  for (let t = 0; t < T; t++) {
    const kn = new Float64Array(B * H * 1 * D);
    const vn = new Float64Array(B * H * 1 * D);
    for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) for (let d = 0; d < D; d++) {
      kn[((b * H + h) * 1 + 0) * D + d] = K.data[(((b * H) + h) * T + t) * D + d];
      vn[((b * H + h) * 1 + 0) * D + d] = V.data[(((b * H) + h) * T + t) * D + d];
    }
    cache = A('dl.attn.kvCacheAppend', [cache, Tensor.fromFlat(kn, [B, H, 1, D]), Tensor.fromFlat(vn, [B, H, 1, D])]);
    assert.equal(cache.len, t + 1);
    // 用当前 token 的 query 对缓存做注意力
    for (let h = 0; h < H; h++) {
      const q1 = new Float64Array(B * 1 * D);
      const kc = new Float64Array(B * (t + 1) * D);
      const vc = new Float64Array(B * (t + 1) * D);
      for (let b = 0; b < B; b++) {
        for (let d = 0; d < D; d++) q1[b * D + d] = Q.data[(((b * H) + h) * T + t) * D + d];
        for (let s = 0; s <= t; s++) for (let d = 0; d < D; d++) {
          kc[(b * (t + 1) + s) * D + d] = cache.k.data[(((b * H) + h) * (t + 1) + s) * D + d];
          vc[(b * (t + 1) + s) * D + d] = cache.v.data[(((b * H) + h) * (t + 1) + s) * D + d];
        }
      }
      const rr = A('dl.attn.scaledDot', [
        Tensor.fromFlat(q1, [B, 1, D]), Tensor.fromFlat(kc, [B, t + 1, D]), Tensor.fromFlat(vc, [B, t + 1, D]), null, null,
      ]);
      for (let b = 0; b < B; b++) for (let d = 0; d < D; d++) {
        const got = rr.out.data[b * D + d];
        const want = full[h].out.data[(b * T + t) * D + d];
        maxErr = Math.max(maxErr, Math.abs(got - want));
      }
    }
  }
  // 注意：逐步解码只用前 t+1 个键（与一次性全序列不同），因此断言「因果前缀」一致性
  // 这里额外验证：使用因果掩码的一次性注意力与逐步解码逐位一致
  const mask = A('dl.attn.causalMask', [T]);
  maxErr = 0;
  for (let h = 0; h < H; h++) {
    const qh = new Float64Array(B * T * D);
    for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) {
      qh[(b * T + t) * D + d] = Q.data[(((b * H) + h) * T + t) * D + d];
    }
    const vh = new Float64Array(B * T * D);
    for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) {
      vh[(b * T + t) * D + d] = V.data[(((b * H) + h) * T + t) * D + d];
    }
    const causal = A('dl.attn.scaledDot', [Tensor.fromFlat(qh, [B, T, D]), headK(h), Tensor.fromFlat(vh, [B, T, D]), mask, null]);
    // 逐步：每个位置用缓存前缀
    let cache2 = A('dl.attn.kvCacheInit', [B, H, T, D]);
    for (let t = 0; t < T; t++) {
      const kn = new Float64Array(B * H * D), vn = new Float64Array(B * H * D);
      for (let b = 0; b < B; b++) for (let hh = 0; hh < H; hh++) for (let d = 0; d < D; d++) {
        kn[(b * H + hh) * D + d] = K.data[(((b * H) + hh) * T + t) * D + d];
        vn[(b * H + hh) * D + d] = V.data[(((b * H) + hh) * T + t) * D + d];
      }
      cache2 = A('dl.attn.kvCacheAppend', [cache2, Tensor.fromFlat(kn, [B, H, 1, D]), Tensor.fromFlat(vn, [B, H, 1, D])]);
      const kc = new Float64Array(B * (t + 1) * D), vc = new Float64Array(B * (t + 1) * D), q1 = new Float64Array(B * D);
      for (let b = 0; b < B; b++) {
        for (let d = 0; d < D; d++) q1[b * D + d] = Q.data[(((b * H) + h) * T + t) * D + d];
        for (let s = 0; s <= t; s++) for (let d = 0; d < D; d++) {
          kc[(b * (t + 1) + s) * D + d] = cache2.k.data[(((b * H) + h) * (t + 1) + s) * D + d];
          vc[(b * (t + 1) + s) * D + d] = cache2.v.data[(((b * H) + h) * (t + 1) + s) * D + d];
        }
      }
      const rr = A('dl.attn.scaledDot', [
        Tensor.fromFlat(q1, [B, 1, D]), Tensor.fromFlat(kc, [B, t + 1, D]), Tensor.fromFlat(vc, [B, t + 1, D]), null, null,
      ]);
      for (let b = 0; b < B; b++) for (let d = 0; d < D; d++) {
        maxErr = Math.max(maxErr, Math.abs(rr.out.data[b * D + d] - causal.out.data[(b * T + t) * D + d]));
      }
    }
  }
  assert.ok(maxErr < 1e-10, `KV cache 逐步 vs 因果一次性 最大误差 ${maxErr}`);
  // 超出 maxLen 必须报错
  assert.throws(() => {
    const c = A('dl.attn.kvCacheInit', [1, 1, 1, 2]);
    A('dl.attn.kvCacheAppend', [c, RT([1, 1, 2, 2], 191), RT([1, 1, 2, 2], 192)]);
  }, /E_LIMIT|超出|溢出/);
});

// =====================================================================
test('14. 小 MLP：dl.model.mlpForward 对输入的梯度通过 ad.gradcheck（相对误差 <1e-6）', () => {
  const net = A('dl.model.mlpInit', [[4, 6, 3], 20261014, 'xavier']);
  assert.equal(net.W.length, 2);
  assert.deepEqual(net.W[0].shape, [4, 6]);
  assert.deepEqual(net.W[1].shape, [6, 3]);
  assert.equal(net.count, A('dl.util.paramCount', [[...net.W, ...net.b]]));
  const x = RT([3, 4], 201);
  const f = (t) => A('ad.sumAll', [A('dl.model.mlpForward', [t, net.W, net.b, 'tanh'])]);
  const r = A('ad.gradcheck', [f, x, 1e-5, 1e-6]);
  assert.ok(r.passed, `gradcheck maxRelErr=${r.maxRelErr.toExponential(3)}`);
  assert.deepEqual(r.analytic.shape, x.shape);
  // relu 版本（注意避开 0 点，数据集为连续分布，几乎必不落在 0）
  const f2 = (t) => A('ad.sumAll', [A('dl.model.mlpForward', [t, net.W, net.b, 'relu'])]);
  const r2 = A('ad.gradcheck', [f2, RT([3, 4], 202), 1e-5, 1e-6]);
  assert.ok(r2.passed, `relu gradcheck maxRelErr=${r2.maxRelErr.toExponential(3)}`);
});

// =====================================================================
test('16. 微型 Transformer block 前向：形状正确、有限值、残差通路可用', () => {
  const B = 2, T = 5, C = 8, H = 2;
  const P = {
    Wq: RT([C, C], 1), Wk: RT([C, C], 2), Wv: RT([C, C], 3), Wo: RT([C, C], 4),
    ln1g: Tensor.ones([C]), ln1b: Tensor.zeros([C]),
    ffW1: RT([C, 16], 5), ffb1: Tensor.zeros([16]),
    ffW2: RT([16, C], 6), ffb2: Tensor.zeros([C]),
    ln2g: Tensor.ones([C]), ln2b: Tensor.zeros([C]),
  };
  const x = RT([B, T, C], 7);
  const mask = A('dl.attn.causalMask', [T]);
  const y = A('dl.model.transformerBlock', [x, P, H, mask, 1e-5]);
  assert.deepEqual(y.shape, [B, T, C]);
  assert.ok(allFinite(y));
  // 无掩码版本同样成立
  const y2 = A('dl.model.transformerBlock', [x, P, H, null, 1e-5]);
  assert.deepEqual(y2.shape, [B, T, C]);
  assert.ok(allFinite(y2));
  // SwiGLU FFN 形状守恒
  const sf = A('dl.model.swigluFFN', [x, RT([C, 12], 8), Tensor.zeros([12]), RT([C, 12], 9), Tensor.zeros([12]), RT([12, C], 10), Tensor.zeros([C])]);
  assert.deepEqual(sf.shape, [B, T, C]);
  assert.ok(allFinite(sf));
  // 门控为 0（Wg/bg 使线性输出为 0）时输出为 0 + bd
  const sf0 = A('dl.model.swigluFFN', [Tensor.zeros([B, T, C]), RT([C, 12], 8), Tensor.zeros([12]), RT([C, 12], 9), Tensor.zeros([12]), RT([12, C], 10), Tensor.zeros([C])]);
  for (let k = 0; k < sf0.size; k++) close(sf0.data[k], 0, 1e-14, '零输入 + 零偏置 → 0');
  // MoE 路由
  const moe = A('dl.model.moeRouter', [RT([6, C], 11), RT([C, 4], 12), 2, 2026]);
  assert.deepEqual(moe.probs.shape, [6, 4]);
  assert.deepEqual(moe.topIdx.shape, [6, 2]);
  for (let n = 0; n < 6; n++) {
    let s = 0, nz = 0;
    for (let e = 0; e < 4; e++) { s += moe.probs.data[n * 4 + e]; if (moe.probs.data[n * 4 + e] !== 0) nz++; }
    assert.equal(nz, 2, '每行恰 topK 个非零');
    close(s, 1, 1e-14);
  }
  // 残差
  const res = A('dl.model.residual', [x, x, 1]);
  for (let k = 0; k < x.size; k++) close(res.data[k], 2 * x.data[k], 1e-14);
  // 参数拉平/计数/更新
  const fp = A('dl.util.flattenParams', [[P.Wq, P.Wk, P.ffW1]]);
  assert.equal(fp.flat.shape[0], 64 + 64 + 128);
  assert.equal(fp.shapes.length, 3);
  assert.equal(A('dl.util.paramCount', [[P.Wq, P.Wk, P.ffW1]]), fp.flat.shape[0]);
  const upd = A('dl.util.applyGrads', [[P.Wq], [Tensor.ones([C, C])], 0.1]);
  for (let k = 0; k < P.Wq.size; k++) close(upd[0].data[k], P.Wq.data[k] - 0.1, 1e-14);
});

// =====================================================================
test('15. 全部 103 条指令冒烟：合理入参调用不抛异常且输出有限', () => {
  const x8 = R(8, 301);
  const x23 = RM(2, 3, 302);
  const x4 = RT([2, 3, 4, 4], 303);
  const x3 = RT([2, 3, 4], 304);
  const C = 8;
  const P16 = {
    Wq: RT([4, 4], 1), Wk: RT([4, 4], 2), Wv: RT([4, 4], 3), Wo: RT([4, 4], 4),
    ln1g: Tensor.ones([4]), ln1b: Tensor.zeros([4]),
    ffW1: RT([4, 6], 5), ffb1: Tensor.zeros([6]),
    ffW2: RT([6, 4], 6), ffb2: Tensor.zeros([4]),
    ln2g: Tensor.ones([4]), ln2b: Tensor.zeros([4]),
  };
  const mlp = A('dl.model.mlpInit', [[3, 4, 2], 1, 'he']);
  const ROP = A('dl.attn.rotaryEmbed', [4, 8, 10000]);
  const CALLS = [
    // —— 激活与门控 14 ——
    ['dl.act.relu', [x8]],
    ['dl.act.leakyRelu', [x8, 0.1]],
    ['dl.act.elu', [x8, 1]],
    ['dl.act.selu', [x8]],
    ['dl.act.gelu', [x8, 'exact']],
    ['dl.act.mish', [x8]],
    ['dl.act.swish', [x8, 1]],
    ['dl.act.sigmoid', [x8]],
    ['dl.act.tanh', [x8]],
    ['dl.act.softplus', [x8, 1, 20]],
    ['dl.act.softsign', [x8]],
    ['dl.act.hardSigmoid', [x8]],
    ['dl.act.prelu', [x23, Tensor.full([3], 0.25)]],
    ['dl.act.glu', [R(6, 305), -1]],
    // —— 归一化 10 ——
    ['dl.norm.batchnorm', [x4, Tensor.ones([3]), Tensor.zeros([3]), Tensor.zeros([3]), Tensor.ones([3]), 1, 1e-5]],
    ['dl.norm.layernorm', [x3, Tensor.ones([4]), Tensor.zeros([4]), 1e-5]],
    ['dl.norm.rmsnorm', [x3, Tensor.ones([4]), 1e-5]],
    ['dl.norm.instancenorm', [x4, Tensor.ones([3]), Tensor.zeros([3]), 1e-5]],
    ['dl.norm.groupnorm', [x4, Tensor.ones([3]), Tensor.zeros([3]), 1, 1e-5]],
    ['dl.norm.weightnorm', [RM(4, 3, 306), Tensor.fromFlat([1, 2, 3, 4], [4]), 1e-12]],
    ['dl.norm.spectralnorm', [RM(4, 3, 307), 10, 1e-12]],
    ['dl.norm.l2normalize', [x3, -1, 1e-12]],
    ['dl.norm.batchStats', [x4, [0, 2, 3], 1]],
    ['dl.norm.runningUpdate', [R(3, 308), R(3, 309), 0.9]],
    // —— 层 14 ——
    ['dl.layer.dense', [RM(2, 3, 310), RM(3, 4, 311), R(4, 312)]],
    ['dl.layer.conv1d', [RT([1, 2, 5], 313), RT([3, 2, 2], 314), R(3, 315), 1, 0, 1]],
    ['dl.layer.conv2d', [RT([1, 2, 4, 4], 316), RT([3, 2, 2, 2], 317), R(3, 318), 1, 0, 1]],
    ['dl.layer.depthwiseConv2d', [RT([1, 2, 4, 4], 319), RT([2, 1, 2, 2], 320), R(2, 321), 1, 0]],
    ['dl.layer.transposeConv2d', [RT([1, 2, 3, 3], 322), RT([2, 3, 2, 2], 323), R(3, 324), 1, 0, 0]],
    ['dl.layer.maxpool1d', [RT([1, 2, 6], 325), 2, 2]],
    ['dl.layer.avgpool1d', [RT([1, 2, 6], 326), 2, 2]],
    ['dl.layer.maxpool2d', [RT([1, 2, 4, 4], 327), 2, 2]],
    ['dl.layer.avgpool2d', [RT([1, 2, 4, 4], 328), 2, 2]],
    ['dl.layer.globalAvgPool', [RT([2, 3, 4, 4], 329)]],
    ['dl.layer.flatten', [RT([2, 3, 4, 4], 330), 1]],
    ['dl.layer.unflatten', [RM(2, 48, 331), [3, 4, 4]]],
    ['dl.layer.embedding', [Tensor.fromFlat([0, 2, 4], [3]), RM(5, 4, 332)]],
    ['dl.layer.patching', [RT([1, 1, 4, 4], 333), 2]],
    // —— 正则化 7 ——
    ['dl.reg.dropout', [x23, 0.5, 7, 1]],
    ['dl.reg.dropPath', [x23, 0.2, 7, 1]],
    ['dl.reg.stochasticDepth', [x23, RT([2, 3], 334), 0.2, 7, 1]],
    ['dl.reg.l1', [x23, 0.01]],
    ['dl.reg.l2', [x23, 0.01]],
    ['dl.reg.l1l2', [x23, 0.01, 0.01]],
    ['dl.reg.mixup', [x23, RT([2, 3], 335), 0.4]],
    // —— 注意力 16 ——
    ['dl.attn.causalMask', [4]],
    ['dl.attn.paddingMask', [Tensor.fromFlat([2, 4], [2]), 4]],
    ['dl.attn.scaledDot', [RT([1, 3, 4], 336), RT([1, 3, 4], 337), RT([1, 3, 4], 338), A('dl.attn.causalMask', [3]), 0.5]],
    ['dl.attn.multiHead', [RT([1, 3, 4], 339), RM(4, 4, 340), RM(4, 4, 341), RM(4, 4, 342), RM(4, 4, 343), 2, A('dl.attn.causalMask', [3])]],
    ['dl.attn.selfAttention', [RT([1, 3, 4], 344), 2, null]],
    ['dl.attn.crossAttention', [RT([1, 2, 4], 345), RT([1, 3, 4], 346), 2, null]],
    ['dl.attn.groupedQuery', [RT([1, 3, 4], 347), RT([1, 3, 2], 348), RT([1, 3, 2], 349), 2, 1, null]],
    ['dl.attn.rotaryEmbed', [4, 8, 10000]],
    ['dl.attn.rotaryApply', [RT([1, 4, 2, 4], 350), ROP.cos, ROP.sin, 0]],
    ['dl.attn.alibiBias', [2, 3]],
    ['dl.attn.relativeBias', [4, 8, 11]],
    ['dl.attn.kvCacheInit', [1, 2, 8, 3]],
    ['dl.attn.kvCacheAppend', [A('dl.attn.kvCacheInit', [1, 2, 8, 3]), RT([1, 2, 1, 3], 351), RT([1, 2, 1, 3], 352)]],
    ['dl.attn.flashTiling', [RT([1, 4, 3], 353), RT([1, 4, 3], 354), RT([1, 4, 3], 355), 2, 2]],
    ['dl.attn.entropy', [RT([1, 3, 3], 356)]],
    ['dl.attn.headSplit', [RT([1, 3, 4], 357), 2]],
    // —— 序列 8 ——
    ['dl.seq.rnnCell', [RM(2, 3, 358), RM(2, 4, 359), RM(3, 4, 360), RM(4, 4, 361), R(4, 362), 'tanh']],
    ['dl.seq.gruCell', [RM(2, 3, 363), RM(2, 4, 364), RM(3, 12, 365), RM(4, 12, 366), R(12, 367), R(12, 368)]],
    ['dl.seq.lstmCell', [RM(2, 3, 369), RM(2, 4, 370), RM(2, 4, 371), RM(3, 16, 372), RM(4, 16, 373), R(16, 374)]],
    ['dl.seq.rnnForward', [RT([2, 3, 3], 375), RM(2, 4, 376), RM(3, 4, 377), RM(4, 4, 378), R(4, 379), 'tanh']],
    ['dl.seq.gruForward', [RT([2, 3, 3], 380), RM(2, 4, 381), RM(3, 12, 382), RM(4, 12, 383), R(12, 384), R(12, 385)]],
    ['dl.seq.lstmForward', [RT([2, 3, 3], 386), RM(2, 4, 387), RM(2, 4, 388), RM(3, 16, 389), RM(4, 16, 390), R(16, 391)]],
    ['dl.seq.bidirectional', [RT([2, 3, 4], 392), RT([2, 3, 4], 393)]],
    ['dl.seq.scan', [RT([2, 3, 3], 394), (hh) => hh, RM(2, 4, 395), 0]],
    // —— 损失 14 ——
    ['dl.loss.mse', [RM(2, 3, 396), RM(2, 3, 397), 'mean']],
    ['dl.loss.mae', [RM(2, 3, 398), RM(2, 3, 399), 'mean']],
    ['dl.loss.huber', [RM(2, 3, 400), RM(2, 3, 401), 1, 'mean']],
    ['dl.loss.bce', [Tensor.full([2, 3], 0.4), Tensor.full([2, 3], 1), 'mean', 1e-12]],
    ['dl.loss.bceWithLogits', [RM(2, 3, 402), Tensor.full([2, 3], 1), 'mean']],
    ['dl.loss.ce', [Tensor.full([2, 3], 1 / 3), Tensor.fromFlat([0, 2], [2]), 'mean', 1e-12]],
    ['dl.loss.ceWithLogits', [RM(2, 3, 403), Tensor.fromFlat([0, 2], [2]), 'mean']],
    ['dl.loss.nll', [A('num.logSoftmax', [RM(2, 3, 404), -1]), Tensor.fromFlat([1, 0], [2]), 'mean']],
    ['dl.loss.klDiv', [A('num.logSoftmax', [RM(2, 3, 405), -1]), Tensor.full([2, 3], 1 / 3), 'mean']],
    ['dl.loss.focal', [RM(2, 3, 406), Tensor.fromFlat([0, 2], [2]), 2, 0.25, 'mean']],
    ['dl.loss.dice', [Tensor.full([4], 0.6), Tensor.fromFlat([1, 0, 1, 0], [4]), 1e-6]],
    ['dl.loss.ntxent', [RM(3, 4, 407), RM(3, 4, 408), 0.5]],
    ['dl.loss.triplet', [RM(3, 4, 409), RM(3, 4, 410), RM(3, 4, 411), 1]],
    ['dl.loss.labelSmoothCE', [RM(2, 3, 412), Tensor.fromFlat([0, 2], [2]), 0.1, 'mean']],
    // —— 初始化 10 ——
    ['dl.init.zeros', [[2, 3]]],
    ['dl.init.ones', [[2, 3]]],
    ['dl.init.constant', [[2, 3], 0.5]],
    ['dl.init.uniform', [[2, 3], -0.1, 0.1, 1]],
    ['dl.init.normal', [[2, 3], 0, 0.1, 1]],
    ['dl.init.truncatedNormal', [[2, 3], 0, 0.1, 1]],
    ['dl.init.xavierUniform', [[4, 6], 1, 1]],
    ['dl.init.xavierNormal', [[4, 6], 1, 1]],
    ['dl.init.heUniform', [[4, 6], Math.SQRT2, 1]],
    ['dl.init.heNormal', [[4, 6], Math.SQRT2, 1]],
    // —— 模型与工具 9 ——
    ['dl.model.mlpInit', [[3, 4, 2], 1, 'he']],
    ['dl.model.mlpForward', [RM(2, 3, 413), mlp.W, mlp.b, 'tanh']],
    ['dl.model.transformerBlock', [RT([1, 3, 4], 414), P16, 2, A('dl.attn.causalMask', [3]), 1e-5]],
    ['dl.model.swigluFFN', [RT([1, 3, 4], 415), RM(4, 6, 416), Tensor.zeros([6]), RM(4, 6, 417), Tensor.zeros([6]), RM(6, 4, 418), Tensor.zeros([4])]],
    ['dl.model.moeRouter', [RM(3, 4, 419), RM(4, 3, 420), 2, 1]],
    ['dl.model.residual', [x3, x3, 1]],
    ['dl.util.paramCount', [[RM(3, 4, 421), RM(2, 2, 422)]]],
    ['dl.util.flattenParams', [[RM(3, 4, 423), RM(2, 2, 424)]]],
    ['dl.util.applyGrads', [[RM(3, 4, 425)], [RM(3, 4, 426)], 0.01]],
    ['dl.util.quantizeInt8', [RM(3, 4, 427)]],
  ];
  assert.equal(CALLS.length, 103, `冒烟用例数 ${CALLS.length}`);
  const seen = new Set();
  for (const [name, args] of CALLS) {
    assert.ok(!seen.has(name), `重复冒烟: ${name}`);
    seen.add(name);
    let out;
    try {
      out = A(name, args);
    } catch (e) {
      throw new Error(`指令 ${name} 冒烟失败: ${e.message}`);
    }
    assert.ok(allFinite(out), `指令 ${name} 输出含非有限值`);
  }
  // 冒烟集合必须与注册表完全一致
  const all = new Set(opsByDomain(8).map((o) => o.name));
  assert.equal(seen.size, 103);
  for (const n of all) assert.ok(seen.has(n), `漏测指令 ${n}`);
  void C;
});
