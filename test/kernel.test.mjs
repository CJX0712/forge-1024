// 内核核心测试：Tensor / RNG / 注册表 / 执行器 / 遥测 / 错误分类学
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  Tensor, prod, stridesOf, sameShape, broadcastShapes, elementwise, reduceTensor,
  createRng, seedFrom, ForgeError, CODES,
  defineOp, exec, tryExec, getOp, hasOp, countOps, listOps, opsByDomain, searchOps, describeOp,
  DOMAINS, KERNEL_VERSION, telemetry, kernel,
} from '../src/core/index.mjs';
import { approx, tensorClose, maxAbsDiff, randMat, randSPD } from './helpers.mjs';

test('Tensor 构造与静态工厂', () => {
  const t = Tensor.fromNested([[1, 2], [3, 4]]);
  assert.equal(t.ndim, 2);
  assert.deepEqual(t.shape, [2, 2]);
  assert.equal(t.size, 4);
  assert.equal(t.get2(1, 0), 3);
  assert.deepEqual(Tensor.zeros([2, 3]).shape, [2, 3]);
  assert.deepEqual(Tensor.eye(3).toArray(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  assert.deepEqual(Array.from(Tensor.arange(5).data), [0, 1, 2, 3, 4]);
  assert.deepEqual(Array.from(Tensor.linspace(0, 1, 5).data), [0, 0.25, 0.5, 0.75, 1]);
  assert.equal(Tensor.full([2], 3.5).at(1), 3.5);
});

test('reshape 视图与 -1 推断', () => {
  const t = Tensor.arange(6).reshape(2, 3);
  const v = t.reshape(3, -1);
  assert.deepEqual(v.shape, [3, 2]);
  assert.equal(v.data, t.data); // 视图共享
  assert.deepEqual(t.flatten().toArray(), [0, 1, 2, 3, 4, 5]);
});

test('转置与置换', () => {
  const t = Tensor.fromNested([[1, 2, 3], [4, 5, 6]]);
  assert.deepEqual(t.T().toArray(), [[1, 4], [2, 5], [3, 6]]);
  const c = Tensor.fromNested([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
  assert.deepEqual(c.permute(1, 0, 2).toArray(), [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]);
});

test('广播规则', () => {
  assert.deepEqual(broadcastShapes([3, 1], [1, 4]), [3, 4]);
  assert.deepEqual(broadcastShapes([2, 3, 5], [5]), [2, 3, 5]);
  assert.throws(() => broadcastShapes([2, 3], [4]), (e) => e instanceof ForgeError && e.code === 'E_SHAPE');
});

test('elementwise 广播逐元素', () => {
  const a = Tensor.fromNested([[1, 2], [3, 4]]);
  const b = Tensor.fromNested([[10], [20]]);
  assert.deepEqual(elementwise(a, b, (x, y) => x + y).toArray(), [[11, 12], [23, 24]]);
  assert.deepEqual(elementwise(a, 2, (x, y) => x * y).toArray(), [[2, 4], [6, 8]]);
  assert.deepEqual(elementwise(2, a, (x, y) => y - x).toArray(), [[-1, 0], [1, 2]]);
});

test('reduceTensor 轴归约（每槽独立累积）', () => {
  const t = Tensor.fromNested([[1, 2], [3, 4], [5, 6]]); // 3x2
  const sum0 = reduceTensor(t, [0], 0, (a, v) => a + v);
  assert.deepEqual(sum0.tensor.shape, [1, 2]);
  assert.deepEqual(sum0.tensor.toArray(), [[9, 12]]);
  assert.equal(sum0.count, 3);
  const sum1 = reduceTensor(t, 1, 0, (a, v) => a + v);
  assert.deepEqual(sum1.tensor.toArray(), [[3], [7], [11]]);
  const all = reduceTensor(t, null, 0, (a, v) => a + v);
  assert.equal(all.tensor.scalar(), 21);
  // max / min / prod（数值累积）
  const mx = reduceTensor(t, [1], -Infinity, (a, v) => Math.max(a, v));
  assert.deepEqual(mx.tensor.toArray(), [[2], [4], [6]]);
  const mn = reduceTensor(t, [0], Infinity, (a, v) => Math.min(a, v));
  assert.deepEqual(mn.tensor.toArray(), [[1, 2]]);
  const pd = reduceTensor(t, null, 1, (a, v) => a * v);
  assert.equal(pd.tensor.scalar(), 720);
  // 负轴
  const neg = reduceTensor(t, [-1], 0, (a, v) => a + v);
  assert.deepEqual(neg.tensor.toArray(), [[3], [7], [11]]);
});

test('PCG32 RNG 可复现且分布合理', () => {
  const a = createRng(123), b = createRng(123), c = createRng(124);
  const seqA = Array.from({ length: 100 }, () => a.next());
  const seqB = Array.from({ length: 100 }, () => b.next());
  const seqC = Array.from({ length: 100 }, () => c.next());
  assert.deepEqual(seqA, seqB, '同种子必须同序列');
  assert.notDeepEqual(seqA, seqC, '不同种子应不同序列');
  const mean = seqA.reduce((s, v) => s + v, 0) / seqA.length;
  assert.ok(approx(mean, 0.5, 0.05), `均匀均值偏离: ${mean}`);
  // 正态矩
  const r = createRng(7);
  const ns = Array.from({ length: 20000 }, () => r.normal());
  const m = ns.reduce((s, v) => s + v, 0) / ns.length;
  const v2 = ns.reduce((s, v) => s + (v - m) ** 2, 0) / ns.length;
  assert.ok(approx(m, 0, 0.02), `正态均值 ${m}`);
  assert.ok(approx(v2, 1, 0.02), `正态方差 ${v2}`);
  // int/shuffle
  const r2 = createRng(9);
  const idx = Array.from({ length: 1000 }, () => r2.int(10));
  assert.ok(idx.every(v => v >= 0 && v < 10));
  const arr = [1, 2, 3, 4, 5];
  const sh = createRng(3).shuffle(arr);
  assert.deepEqual([...sh].sort(), [1, 2, 3, 4, 5]);
  assert.deepEqual(arr, [1, 2, 3, 4, 5], 'shuffle 不得改原数组');
  assert.equal(seedFrom('forge'), seedFrom('forge'));
  assert.notEqual(seedFrom('forge'), seedFrom('forge1'));
});

test('注册表：defineOp/exec/错误分类学/遥测', () => {
  defineOp({
    id: 'T-001', domain: 10, name: 'test.addConst', arity: 2,
    sig: '(Tensor, number) -> Tensor', doc: '测试指令：加常数',
    ref: '测试', inv: 'out[i]=in[i]+c', impl: (t, c) => t.map(v => v + c),
  });
  assert.ok(hasOp('test.addConst'));
  assert.ok(hasOp('T-001'));
  const out = exec('test.addConst', [Tensor.fromFlat([1, 2], [2]), 10]);
  assert.deepEqual(out.toArray(), [11, 12]);
  // name 与 id 均可索引
  assert.deepEqual(exec('T-001', [Tensor.fromFlat([1], [1]), 1]).toArray(), [2]);
  // 未知指令
  const r1 = tryExec('nope.nope', []);
  assert.equal(r1.ok, false);
  assert.equal(r1.error.code, 'E_NOTFOUND');
  // 参数个数
  const r2 = tryExec('test.addConst', [Tensor.zeros([1])]);
  assert.equal(r2.ok, false);
  assert.equal(r2.error.code, 'E_ARITY');
  // 运行期错误归一化为 E_NUMERICAL
  defineOp({
    id: 'T-002', domain: 10, name: 'test.boom', arity: 0,
    sig: '() -> void', doc: '测试指令：抛错', ref: '测试', inv: '抛错',
    impl: () => { throw new Error('boom'); },
  });
  const r3 = tryExec('test.boom', []);
  assert.equal(r3.ok, false);
  assert.equal(r3.error.code, 'E_NUMERICAL');
  // 重复注册
  assert.throws(() => defineOp({
    id: 'T-001', domain: 10, name: 'test.dup', sig: '', doc: '', ref: '', inv: '', impl: () => 0,
  }), (e) => e.code === 'E_VALUE');
  // 遥测
  telemetry.reset();
  telemetry.setTiming(true);
  exec('test.addConst', [Tensor.zeros([2]), 1]);
  exec('test.addConst', [Tensor.zeros([2]), 1]);
  const stats = telemetry.summary().filter(s => s.id === 'T-001');
  assert.equal(stats.length, 1);
  assert.equal(stats[0].calls, 2);
  assert.ok(stats[0].ms >= 0);
  telemetry.setTiming(false);
  assert.equal(telemetry.totalCalls(), 2);
  // 搜索与描述
  assert.ok(searchOps('addConst').length >= 1);
  assert.ok(describeOp('T-001').includes('测试指令'));
  // 查询接口
  assert.ok(listOps().length >= 2);
  assert.ok(opsByDomain(10).length >= 2);
  assert.equal(kernel.version, KERNEL_VERSION);
  assert.ok(Object.keys(DOMAINS).length === 10);
});

test('矩阵随机工厂可复现', () => {
  const A = randMat(4, 4, 1), B = randMat(4, 4, 1);
  assert.equal(maxAbsDiff(A, B), 0);
  const S = randSPD(3, 2);
  // 对称
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) assert.ok(approx(S.get2(i, j), S.get2(j, i), 1e-12));
  tensorClose(S, [[5.5, 0, 0], [0, 5.5, 0], [0, 0, 5.5]], 100); // 仅验证有限
  assert.ok(S.data.every(Number.isFinite));
});

test('prod / stridesOf / sameShape', () => {
  assert.equal(prod([2, 3, 4]), 24);
  assert.equal(prod([]), 1);
  assert.deepEqual(stridesOf([2, 3, 4]), [12, 4, 1]);
  assert.equal(sameShape([2, 2], [2, 2]), true);
  assert.equal(sameShape([2, 2], [2, 3]), false);
});
