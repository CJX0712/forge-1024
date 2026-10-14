// 域1 数学基元测试：生成家族属性测试 + 手写核心不变量测试
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { exec, opsByDomain } from '../src/core/registry.mjs';
import { Tensor, createRng } from '../src/core/index.mjs';
import { BINARY_TABLE, UNARY_TABLE } from '../src/ops/d1-math.mjs';
import { approx, tensorClose, randMat } from './helpers.mjs';

const call = (name, args) => exec(name, args);

test('域1 恰好 102 条指令且 id 连续', () => {
  const ops = opsByDomain(1);
  assert.equal(ops.length, 102);
  const ids = ops.map(o => o.id).sort();
  assert.equal(ids[0], 'D1-001');
  assert.equal(ids[101], 'D1-102');
  assert.equal(new Set(ids).size, 102);
});

// —— 生成家族：与朴素参考逐点核对（含广播） ——
test('二元逐元素 TT：张量⊗张量（含广播）', () => {
  const rng = createRng(101);
  const mk = (shape) => Tensor.fromFlat(Array.from({ length: shape.reduce((a, b) => a * b, 1) }, () => rng.range(-2, 2)), shape);
  const cases = [
    [[3, 4], [3, 4]], [[3, 1], [1, 4]], [[2, 3], [3]], [[5], [3, 1]], [[2, 3, 4], [4]],
  ];
  for (const [n, f] of BINARY_TABLE) {
    for (const [sa, sb] of cases) {
      const a = mk(sa), b = mk(sb);
      const out = call(`math.${n}`, [a, b]);
      const outShape = [];
      const nn = Math.max(sa.length, sb.length);
      for (let i = 0; i < nn; i++) {
        const d1 = sa[sa.length - 1 - i] ?? 1, d2 = sb[sb.length - 1 - i] ?? 1;
        outShape[nn - 1 - i] = Math.max(d1, d2);
      }
      assert.deepEqual(out.shape, outShape, `${n} [${sa}]x[${sb}] 形状`);
      // 朴素参考（padded 形状 + 行主序 stride）
      const pa = sa.slice(), pb = sb.slice();
      while (pa.length < nn) pa.unshift(1);
      while (pb.length < nn) pb.unshift(1);
      const stra = new Array(nn).fill(1), strb = new Array(nn).fill(1);
      for (let d = nn - 2; d >= 0; d--) { stra[d] = stra[d + 1] * pa[d + 1]; strb[d] = strb[d + 1] * pb[d + 1]; }
      const idx = new Array(nn);
      for (let p = 0; p < out.size; p++) {
        let r = p;
        for (let d = nn - 1; d >= 0; d--) { idx[d] = r % outShape[d]; r = (r - idx[d]) / outShape[d]; }
        let ia = 0, ib = 0;
        for (let d = 0; d < nn; d++) { ia += (pa[d] === 1 ? 0 : idx[d]) * stra[d]; ib += (pb[d] === 1 ? 0 : idx[d]) * strb[d]; }
        assert.ok(approx(out.data[p], f(a.data[ia], b.data[ib]), 1e-12), `${n} [${sa}]x[${sb}] 第${p}项`);
      }
    }
  }
});

test('二元逐元素 TS/ST：标量变体', () => {
  const rng = createRng(202);
  const a = Tensor.fromFlat(Array.from({ length: 12 }, () => rng.range(-3, 3)), [3, 4]);
  const s = 2.5;
  for (const [n, f] of BINARY_TABLE) {
    const ts = call(`math.${n}Scalar`, [a, s]);
    const st = call(`math.scalar${n[0].toUpperCase()}${n.slice(1)}`, [s, a]);
    for (let i = 0; i < a.size; i++) {
      assert.ok(approx(ts.data[i], f(a.data[i], s), 1e-12), `${n}Scalar 第${i}项`);
      assert.ok(approx(st.data[i], f(s, a.data[i]), 1e-12), `scalar${n} 第${i}项`);
    }
  }
});

test('一元逐元素：与 Math 参考逐点核对', () => {
  const rng = createRng(303);
  const xs = Array.from({ length: 50 }, () => rng.range(-0.9, 0.9));
  const t = Tensor.fromFlat(xs, [50]);
  for (const [n, f] of UNARY_TABLE) {
    const out = call(`math.${n}`, [t]);
    for (let i = 0; i < xs.length; i++) {
      assert.ok(approx(out.data[i], f(xs[i]), 1e-12), `${n} 第${i}项: ${out.data[i]} != ${f(xs[i])}`);
    }
  }
});

test('点积/叉积/外积', () => {
  const x = Tensor.fromFlat([1, 2, 3], [3]);
  const y = Tensor.fromFlat([4, 5, 6], [3]);
  assert.equal(call('math.dot', [x, y]), 32);
  assert.ok(approx(call('math.l2', [x]), Math.sqrt(14), 1e-12));
  const c = call('math.cross', [x, y]);
  assert.deepEqual(c.toArray(), [-3, 6, -3]);
  assert.ok(approx(call('math.dot', [c, x]), 0, 1e-12));
  assert.ok(approx(call('math.dot', [c, y]), 0, 1e-12));
  const o = call('math.outer', [x, y]);
  assert.deepEqual(o.shape, [3, 3]);
  assert.deepEqual(o.toArray(), [[4, 5, 6], [8, 10, 12], [12, 15, 18]]);
});

test('范数与距离', () => {
  const x = Tensor.fromFlat([3, -4], [2]);
  assert.equal(call('math.l1', [x]), 7);
  assert.equal(call('math.l2', [x]), 5);
  assert.equal(call('math.linf', [x]), 4);
  assert.ok(approx(call('math.lp', [x, 2]), 5, 1e-12));
  assert.ok(approx(call('math.lp', [x, 1]), 7, 1e-12));
  const y = Tensor.fromFlat([0, 0], [2]);
  assert.equal(call('math.manhattan', [x, y]), 7);
  assert.equal(call('math.euclidean', [x, y]), 5);
  assert.equal(call('math.sqeuclidean', [x, y]), 25);
  assert.equal(call('math.chebyshev', [x, y]), 4);
  assert.equal(call('math.cosine', [x, y]), 0); // 零向量 -> 0
  const a = Tensor.fromFlat([1, 0], [2]), b = Tensor.fromFlat([2, 0], [2]);
  assert.ok(approx(call('math.cosine', [a, b]), 1, 1e-12));
});

test('全元素归约 vs 朴素循环', () => {
  const rng = createRng(404);
  const vals = Array.from({ length: 37 }, () => rng.normal());
  const t = Tensor.fromFlat(vals, [37]);
  const sum = vals.reduce((a, b) => a + b, 0);
  assert.ok(approx(call('math.sum', [t]), sum, 1e-10));
  assert.ok(approx(call('math.mean', [t]), sum / 37, 1e-10));
  const mu = sum / 37;
  const varr = vals.reduce((a, b) => a + (b - mu) ** 2, 0) / 37;
  assert.ok(approx(call('math.var', [t]), varr, 1e-10));
  assert.ok(approx(call('math.std', [t]), Math.sqrt(varr), 1e-10));
  assert.equal(call('math.min', [t]), Math.min(...vals));
  assert.equal(call('math.max', [t]), Math.max(...vals));
  assert.equal(call('math.argmax', [t]), vals.indexOf(Math.max(...vals)));
  assert.equal(call('math.argmin', [t]), vals.indexOf(Math.min(...vals)));
  assert.ok(approx(call('math.prod', [t]), vals.reduce((a, b) => a * b, 1), 1e-6));
  // 空张量约定
  const empty = Tensor.zeros([0]);
  assert.equal(call('math.sum', [empty]), 0);
  assert.equal(call('math.prod', [empty]), 1);
});

test('分位数与中位数', () => {
  const t = Tensor.fromFlat([1, 2, 3, 4, 5], [5]);
  assert.equal(call('math.median', [t]), 3);
  assert.equal(call('math.quantile', [t, 0]), 1);
  assert.equal(call('math.quantile', [t, 1]), 5);
  assert.equal(call('math.quantile', [t, 0.5]), 3);
  const t2 = Tensor.fromFlat([1, 2, 3, 4], [4]);
  assert.equal(call('math.median', [t2]), 2.5);
  assert.ok(approx(call('math.quantile', [t2, 0.25]), 1.75, 1e-12));
  // mad：稳健性
  const t3 = Tensor.fromFlat([1, 1, 1, 1, 100], [5]);
  assert.equal(call('math.mad', [t3]), 0);
});

test('前缀和/积', () => {
  const t = Tensor.fromFlat([1, 2, 3, 4], [4]);
  assert.deepEqual(call('math.cumsum', [t]).toArray(), [1, 3, 6, 10]);
  assert.deepEqual(call('math.cumprod', [t]).toArray(), [1, 2, 6, 24]);
  const cs = call('math.cumsum', [t]);
  assert.ok(approx(cs.data[3], call('math.sum', [t]), 1e-12));
});

test('轴归约', () => {
  const t = Tensor.fromNested([[1, 2, 3], [4, 5, 6]]);
  assert.deepEqual(call('math.sumAxis', [t, 0]).toArray(), [5, 7, 9]);
  assert.deepEqual(call('math.sumAxis', [t, 1]).toArray(), [6, 15]);
  assert.deepEqual(call('math.meanAxis', [t, 1]).toArray(), [2, 5]);
  assert.deepEqual(call('math.maxAxis', [t, 0]).toArray(), [4, 5, 6]);
  assert.deepEqual(call('math.minAxis', [t, 1]).toArray(), [1, 4]);
  assert.deepEqual(call('math.argmaxAxis', [t, 1]).toArray(), [2, 2]);
  assert.deepEqual(call('math.argmaxAxis', [t, 0]).toArray(), [1, 1, 1]);
  const v = call('math.varAxis', [t, 0]);
  assert.deepEqual(v.toArray(), [2.25, 2.25, 2.25]);
  // 负轴
  assert.deepEqual(call('math.sumAxis', [t, -1]).toArray(), [6, 15]);
  // 3D
  const c = Tensor.arange(24).reshape(2, 3, 4);
  const s = call('math.sumAxis', [c, 1]);
  assert.deepEqual(s.shape, [2, 4]);
  assert.equal(s.data[0], 0 + 4 + 8);
});

test('构造器', () => {
  assert.deepEqual(call('math.zeros', [[2, 2]]).toArray(), [[0, 0], [0, 0]]);
  assert.deepEqual(call('math.ones', [[2]]).toArray(), [1, 1]);
  assert.deepEqual(call('math.full', [[2], 7]).toArray(), [7, 7]);
  assert.deepEqual(call('math.eye', [2]).toArray(), [[1, 0], [0, 1]]);
  assert.deepEqual(call('math.arange', [5]).toArray(), [0, 1, 2, 3, 4]);
  assert.deepEqual(call('math.linspace', [0, 1, 3]).toArray(), [0, 0.5, 1]);
});

test('concat/stack/tile/pad/clamp/where', () => {
  const a = Tensor.fromNested([[1, 2], [3, 4]]);
  const b = Tensor.fromNested([[5, 6]]);
  assert.deepEqual(call('math.concat', [[a, b], 0]).toArray(), [[1, 2], [3, 4], [5, 6]]);
  assert.deepEqual(call('math.concat', [[a, b.T()], 1]).toArray(), [[1, 2, 5], [3, 4, 6]]);
  const c = Tensor.fromNested([[5, 6], [7, 8]]);
  const s = call('math.stack', [[a, c], 0]);
  assert.deepEqual(s.shape, [2, 2, 2]);
  assert.deepEqual(s.toArray(), [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
  const s1 = call('math.stack', [[a, c], 1]);
  assert.deepEqual(s1.toArray(), [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]);
  assert.throws(() => call('math.stack', [[a, b], 0]), (e) => e.code === 'E_SHAPE');
  const t = Tensor.fromFlat([1, 2], [2]);
  assert.deepEqual(call('math.tile', [t, [2]]).toArray(), [1, 2, 1, 2]);
  const t2 = Tensor.fromNested([[1, 2]]);
  assert.deepEqual(call('math.tile', [t2, [2, 3]]).toArray(), [[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]]);
  const p = call('math.pad', [t2, [[1, 0], [0, 1]], 9]);
  assert.deepEqual(p.toArray(), [[9, 9, 9], [1, 2, 9]]);
  assert.deepEqual(call('math.clamp', [Tensor.fromFlat([-5, 0, 5], [3]), -1, 1]).toArray(), [-1, 0, 1]);
  const mask = Tensor.fromFlat([1, 0, 1], [3]);
  const x = Tensor.fromFlat([10, 20, 30], [3]);
  const y = Tensor.fromFlat([1, 2, 3], [3]);
  assert.deepEqual(call('math.where', [mask, x, y]).toArray(), [10, 2, 30]);
  // where 广播
  const w2 = call('math.where', [Tensor.fromFlat([1, 0], [2, 1]), Tensor.ones([2, 2]), Tensor.zeros([2, 2])]);
  assert.deepEqual(w2.toArray(), [[1, 1], [0, 0]]);
});

test('数学恒等式：l2²=dot(x,x)、范数单调', () => {
  const rng = createRng(505);
  const x = Tensor.fromFlat(Array.from({ length: 20 }, () => rng.normal()), [20]);
  assert.ok(approx(call('math.l2', [x]) ** 2, call('math.dot', [x, x]), 1e-10));
  // ‖x‖∞ ≤ ‖x‖₂ ≤ ‖x‖₁（Cauchy-Schwarz）
  assert.ok(call('math.linf', [x]) <= call('math.l2', [x]) + 1e-12, 'linf<=l2');
  assert.ok(call('math.l2', [x]) <= call('math.l1', [x]) + 1e-12, 'l2<=l1');
  assert.ok(call('math.l2', [x]) >= call('math.l1', [x]) / Math.sqrt(20) - 1e-12);
});
