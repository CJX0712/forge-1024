// 域10 系统编排测试：注册表内省、遥测、执行组合、验证、文档、基准、图、配置、错误、环境、自检
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { exec, opsByDomain, defineOp, requiredParams } from '../src/core/registry.mjs';
import { ForgeError, CODES, Tensor } from '../src/core/index.mjs';
import '../src/ops/d1-math.mjs';
import '../src/ops/d10-sys.mjs';
import { approx } from './helpers.mjs';

const O = (name, args = []) => exec(name, args);

// 快照：在任何临时 op 注册之前取好，保证后续断言不受夹具影响。
// 域10 的 verify/doc/bench/self 全部以入参接收 op 列表，故只注册部分域也能验证。
const ALL = [...opsByDomain(1), ...opsByDomain(10)];
// 编目规约：1024 = 6×102 + 4×103（D7–D10 四个能力面最广的域各多承载 1 条）
const EXPECT_PER_DOMAIN = { 1: 102, 2: 102, 3: 102, 4: 102, 5: 102, 6: 102, 7: 103, 8: 103, 9: 103, 10: 103 };
const D10 = opsByDomain(10);

// —— 冒烟入参表：为每条域10 指令给出一组可执行的入参 ——
const T = (a, s = [a.length]) => Tensor.fromFlat(a, s);
const G1 = [
  { id: 'a', op: null, inputs: [], value: T([1, 2]) },
  { id: 'b', op: 'math.add', inputs: ['a', 'a'] },
];
const SMOKE = {
  'sys.reg.count': [undefined],
  'sys.reg.domains': [],
  'sys.reg.byDomain': [10],
  'sys.reg.search': ['sys'],
  'sys.reg.describe': ['sys.reg.count'],
  'sys.reg.exists': ['sys.reg.count'],
  'sys.reg.signature': ['sys.reg.count'],
  'sys.reg.catalogHash': [true],
  'sys.reg.version': [],
  'sys.reg.nameIndex': [],
  'sys.reg.opIds': [undefined],
  'sys.reg.domainStats': [],

  'sys.tel.summary': [undefined],
  'sys.tel.hotOps': [5],
  'sys.tel.reset': [],
  'sys.tel.totalCalls': [],
  'sys.tel.record': ['sys.reg.count', true],
  'sys.tel.errorRate': [undefined],
  'sys.tel.latencyStats': [],
  'sys.tel.topErrors': [5],

  'sys.exec.call': ['sys.reg.version', []],
  'sys.exec.try': ['sys.reg.version', []],
  'sys.exec.batch': [[['sys.reg.version', []]]],
  'sys.exec.pipe': [[{ op: 'math.addScalar', args: [1] }], T([1, 2])],
  'sys.exec.map': ['sys.util.typeOf', [1, -2, 3]],
  'sys.exec.timed': ['sys.reg.version', []],
  'sys.exec.retry': ['sys.reg.version', [], { retries: 1 }],
  'sys.exec.memo': ['sys.reg.version', []],
  'sys.exec.assertOp': ['sys.reg.version', [], () => true],

  'sys.verify.catalog': [ALL, undefined],
  'sys.verify.invariants': [ALL],
  'sys.verify.determinism': [ALL, 'sys.reg.version', [], 3],
  'sys.verify.fuzz': [ALL, 'sys.util.typeOf', { trials: 3, seed: 7 }],
  'sys.verify.smoke': [ALL.slice(0, 2), {}],
  'sys.verify.docCoverage': [ALL, 4],
  'sys.verify.refCoverage': [ALL, 4],
  'sys.verify.idContinuity': [ALL],
  'sys.verify.nameUnique': [ALL],
  'sys.verify.arityCheck': [ALL],
  'sys.verify.domainCounts': [ALL, 102],
  'sys.verify.errorTaxonomy': [ALL],

  'sys.doc.opMarkdown': [ALL, 'sys.reg.count'],
  'sys.doc.domainMarkdown': [ALL, 10],
  'sys.doc.catalogMarkdown': [ALL, '目录'],
  'sys.doc.jsonExport': [ALL, false],
  'sys.doc.csvExport': [ALL, ','],
  'sys.doc.htmlExport': [ALL, '标题'],
  'sys.doc.mermaid': [ALL, 'graph'],
  'sys.doc.sigTable': [ALL, 10],
  'sys.doc.readmeSection': [ALL, 10],

  'sys.bench.timeOp': [ALL, 'sys.reg.version', [], { n: 3, warmup: 0 }],
  'sys.bench.compare': [ALL, ['sys.reg.version'], [], { n: 2, warmup: 0 }],
  'sys.bench.throughput': [ALL, 'sys.reg.version', [], { budgetMs: 2 }],
  'sys.bench.matmulFlops': [ALL, 8, 8, 8],
  'sys.bench.memoryEstimate': [ALL, [10, 10], 8],
  'sys.bench.reportMarkdown': [ALL, []],
  'sys.bench.scaling': [ALL, 'sys.util.range', [1, 2, 4], { n: 1 }],
  'sys.bench.suite': [ALL.slice(0, 3), { limit: 3, trials: 1, warmup: 0 }],

  'sys.graph.build': [G1],
  'sys.graph.toposort': [G1],
  'sys.graph.execute': [G1, {}],
  'sys.graph.validate': [G1],
  'sys.graph.toDot': [G1, 'g'],
  'sys.graph.cycleDetect': [G1],
  'sys.graph.nodeCount': [G1],
  'sys.graph.edgeCount': [G1],

  'sys.cfg.merge': [{ a: 1 }, { b: 2 }, true],
  'sys.cfg.validate': [{ a: 1 }, { a: 'number' }],
  'sys.cfg.freeze': [{ a: 1 }],
  'sys.cfg.diff': [{ a: 1 }, { a: 2 }],
  'sys.cfg.hash': [{ a: 1 }],
  'sys.cfg.pick': [{ a: 1, b: 2 }, ['a']],
  'sys.cfg.defaults': [{ a: 1 }, { b: 2 }],
  'sys.cfg.toEnv': [{ a: 1 }, 'APP'],

  'sys.err.classify': ['E_SHAPE'],
  'sys.err.format': ['E_SHAPE', 'msg'],
  'sys.err.codes': [],
  'sys.err.isRetryable': ['E_NUMERICAL'],
  'sys.err.count': [undefined],
  'sys.err.trace': ['sys.reg.version', []],
  'sys.err.suggest': ['E_SHAPE'],

  'sys.env.info': [],
  'sys.env.nodeVersion': [],
  'sys.env.memory': [],
  'sys.env.nowMs': [],
  'sys.env.seed': ['abc'],
  'sys.env.platform': [],
  'sys.env.uptimeMs': [],

  'sys.self.checkAll': [ALL, undefined],
  'sys.self.summary': [ALL, undefined],
  'sys.self.exitCode': [ALL, undefined],
  'sys.self.domainHealth': [ALL, 10],
  'sys.self.doctor': [ALL],
  'sys.self.report': [ALL, '报告'],

  'sys.util.range': [0, 5, 1],
  'sys.util.zip': [[1, 2], [3, 4]],
  'sys.util.pick': [{ a: { b: 1 } }, 'a.b'],
  'sys.util.deepEqual': [{ a: 1 }, { a: 1 }],
  'sys.util.typeOf': [1],
  'sys.util.formatNum': [3.14159, 2],
  'sys.util.table': [[['a', 'b']], ['A', 'B']],
  'sys.util.chunk': [[1, 2, 3, 4, 5], 2],

  'sys.bench.percentile': [[1, 2, 3, 4, 5], 50],
};

test('1. 域10 恰好 103 条，id D10-001..D10-103 且唯一', () => {
  assert.equal(D10.length, 103);
  const ids = D10.map(o => o.id).sort();
  assert.equal(ids[0], 'D10-001');
  assert.equal(ids[102], 'D10-103');
  assert.equal(new Set(ids).size, 103);
  // 每条都带齐元数据
  for (const o of D10) {
    assert.ok(o.name.startsWith('sys.'), `${o.id} 名字前缀异常: ${o.name}`);
    assert.equal(o.domain, 10);
    assert.ok(o.sig && o.doc && o.ref && o.inv, `${o.name} 元数据缺失`);
    assert.equal(o.arity, requiredParams(o.impl), `${o.name} arity(${o.arity}) != 必需形参数(${requiredParams(o.impl)})`);
    assert.ok(o.arity <= o.impl.length, `${o.name} arity 超过形参总数`);
  }
});

test('2. sys.reg.catalogHash 稳定；新增一条 op 后改变', () => {
  const h1 = O('sys.reg.catalogHash', [true]);
  const h2 = O('sys.reg.catalogHash', [true]);
  assert.equal(h1, h2);
  assert.match(h1, /^0x[0-9a-f]{16}$/);
  defineOp({
    id: 'D1-903', domain: 1, name: '_tmp.hashprobe', arity: 0,
    sig: '() -> number', doc: '临时探针指令，用于验证目录哈希敏感性',
    ref: '单元测试夹具', inv: '恒返回 1',
    impl: () => 1,
  });
  const h3 = O('sys.reg.catalogHash', [true]);
  assert.notEqual(h1, h3);
});

test('3. verify.nameUnique / idContinuity / domainCounts / docCoverage / refCoverage 全部通过', () => {
  const nu = O('sys.verify.nameUnique', [ALL]);
  assert.equal(nu.ok, true, `重名: ${nu.dupes}`);
  assert.equal(nu.total, ALL.length);

  const ic = O('sys.verify.idContinuity', [ALL]);
  assert.equal(ic.ok, true, JSON.stringify(ic.issues));
  assert.equal(ic.byDomain['1'], 102);
  assert.equal(ic.byDomain['10'], 103);

  const dc = O('sys.verify.domainCounts', [ALL, EXPECT_PER_DOMAIN]);
  assert.equal(dc.ok, true, JSON.stringify(dc.issues));
  assert.equal(dc.counts['1'], 102);
  assert.equal(dc.counts['10'], 103);

  const doc = O('sys.verify.docCoverage', [ALL, 4]);
  assert.equal(doc.ok, true, `缺文档: ${doc.missing}`);
  assert.equal(doc.ratio, 1);

  const ref = O('sys.verify.refCoverage', [ALL, 4]);
  assert.equal(ref.ok, true, `缺参考: ${ref.missing}`);
  assert.equal(ref.ratio, 1);

  const ar = O('sys.verify.arityCheck', [ALL]);
  assert.equal(ar.ok, true, JSON.stringify(ar.issues));

  const inv = O('sys.verify.invariants', [ALL]);
  assert.equal(inv.ok, true, JSON.stringify(inv.failures));

  const tx = O('sys.verify.errorTaxonomy', [ALL]);
  assert.equal(tx.ok, true);
});

test('4. sys.exec.pipe 串联 3 条 math op = 手工逐步计算', () => {
  const x = T([1, 2]);
  const steps = [
    { op: 'math.addScalar', args: [3] },
    { op: 'math.mulScalar', args: [2] },
    { op: 'math.subScalar', args: [1] },
  ];
  const got = O('sys.exec.pipe', [steps, x]);
  const manual = O('math.subScalar', [O('math.mulScalar', [O('math.addScalar', [x, 3]), 2]), 1]);
  assert.deepEqual(Array.from(got.data), [7, 9]);
  assert.deepEqual(Array.from(got.data), Array.from(manual.data));
  // 空管道恒等
  assert.equal(O('sys.exec.pipe', [[], x]), x);
});

test('5. sys.exec.memo 第二次调用命中', () => {
  O('sys.tel.reset', []);
  const r1 = O('sys.exec.memo', ['math.l2', [T([3, 4])]]);
  const r2 = O('sys.exec.memo', ['math.l2', [T([3, 4])]]);
  assert.equal(r1.hit, false);
  assert.equal(r2.hit, true);
  assert.ok(approx(r2.value, 5, 1e-12));
  // 不同入参不命中
  const r3 = O('sys.exec.memo', ['math.l2', [T([6, 8])]]);
  assert.equal(r3.hit, false);
  assert.ok(approx(r3.value, 10, 1e-12));
});

test('6. sys.exec.retry：可重试错误重试成功，不可重试错误立即失败', () => {
  let flaky = 0;
  defineOp({
    id: 'D1-901', domain: 1, name: '_tmp.flaky', arity: 0,
    sig: '() -> number', doc: '夹具：前两次抛 E_LIMIT，之后返回 42',
    ref: '单元测试夹具', inv: '第 3 次起返回 42',
    impl: () => { flaky++; if (flaky <= 2) throw new ForgeError('E_LIMIT', '临时失败'); return 42; },
  });
  defineOp({
    id: 'D1-902', domain: 1, name: '_tmp.hard', arity: 0,
    sig: '() -> number', doc: '夹具：恒抛 E_SHAPE',
    ref: '单元测试夹具', inv: '恒抛 E_SHAPE',
    impl: () => { throw new ForgeError('E_SHAPE', '形状错误'); },
  });

  const r1 = O('sys.exec.retry', ['_tmp.flaky', [], { retries: 3 }]);
  assert.equal(r1.ok, true);
  assert.equal(r1.value, 42);
  assert.equal(r1.attempts, 3);

  const r2 = O('sys.exec.retry', ['_tmp.hard', [], { retries: 5 }]);
  assert.equal(r2.ok, false);
  assert.equal(r2.attempts, 1, '不可重试错误必须立即失败');
  assert.equal(r2.error.code, 'E_SHAPE');

  assert.equal(O('sys.err.isRetryable', ['E_LIMIT']), true);
  assert.equal(O('sys.err.isRetryable', ['E_SHAPE']), false);
});

test('7. sys.graph.toposort 满足全部边；cycleDetect 能发现环', () => {
  const g = O('sys.graph.build', [[
    { id: 'a', op: null, inputs: [], value: 1 },
    { id: 'b', op: null, inputs: [], value: 2 },
    { id: 'c', op: 'sys.util.typeOf', inputs: ['a', 'b'] },
    { id: 'd', op: 'sys.util.typeOf', inputs: ['c'] },
    { id: 'e', op: 'sys.util.typeOf', inputs: ['d', 'a'] },
  ]]);
  const order = O('sys.graph.toposort', [g]);
  assert.equal(order.length, 5);
  assert.equal(O('sys.graph.nodeCount', [g]), 5);
  assert.equal(O('sys.graph.edgeCount', [g]), 5);
  const pos = new Map(order.map((id, i) => [id, i]));
  for (const n of g.nodes) {
    for (const s of n.inputs) assert.ok(pos.get(s) < pos.get(n.id), `边 ${s}->${n.id} 顺序错误`);
  }
  assert.equal(O('sys.graph.cycleDetect', [g]).hasCycle, false);
  assert.equal(O('sys.graph.validate', [g]).ok, true);

  const cyc = O('sys.graph.build', [[
    { id: 'a', op: null, inputs: ['c'], value: 1 },
    { id: 'b', op: 'sys.util.typeOf', inputs: ['a'] },
    { id: 'c', op: 'sys.util.typeOf', inputs: ['b'] },
  ]]);
  const det = O('sys.graph.cycleDetect', [cyc]);
  assert.equal(det.hasCycle, true);
  assert.equal(det.cycle[0], det.cycle[det.cycle.length - 1]);
  assert.throws(() => O('sys.graph.toposort', [cyc]), /E_STATE/);
  assert.equal(O('sys.graph.validate', [cyc]).hasCycle, true);

  const dot = O('sys.graph.toDot', [g, 'demo']);
  assert.ok(dot.startsWith('digraph'));
  assert.ok(dot.includes('"a" -> "c"'));
});

test('8. sys.graph.execute 计算 (a+b)*c 与直算一致', () => {
  const a = T([2]), b = T([3]), c = T([4]);
  const g = O('sys.graph.build', [[
    { id: 'a', op: null, inputs: [], value: a },
    { id: 'b', op: null, inputs: [], value: b },
    { id: 'c', op: null, inputs: [], value: c },
    { id: 's', op: 'math.add', inputs: ['a', 'b'] },
    { id: 'p', op: 'math.mul', inputs: ['s', 'c'] },
  ]]);
  const r = O('sys.graph.execute', [g, {}]);
  const direct = O('math.mul', [O('math.add', [a, b]), c]);
  assert.deepEqual(Array.from(r.values.p.data), Array.from(direct.data));
  assert.deepEqual(Array.from(r.values.p.data), [20]);
  assert.equal(r.order.length, 5);
  // 外部输入覆盖叶子
  const r2 = O('sys.graph.execute', [g, { a: T([10]) }]);
  assert.deepEqual(Array.from(r2.values.p.data), [52]);
});

test('9. sys.cfg.merge 深合并 / freeze 只读 / diff 给出变更键', () => {
  const base = { a: 1, nest: { x: 1, y: 2 }, arr: [1, 2] };
  const over = { b: 2, nest: { y: 9 } };
  const m = O('sys.cfg.merge', [base, over, true]);
  assert.deepEqual(m, { a: 1, b: 2, nest: { x: 1, y: 9 }, arr: [1, 2] });
  const shallow = O('sys.cfg.merge', [base, { nest: { y: 9 } }, false]);
  assert.deepEqual(shallow.nest, { y: 9 });

  const f = O('sys.cfg.freeze', [{ a: 1, nest: { b: 2 } }]);
  assert.equal(Object.isFrozen(f), true);
  assert.equal(Object.isFrozen(f.nest), true);
  assert.throws(() => { f.a = 5; }, TypeError);
  assert.equal(f.a, 1);

  const d = O('sys.cfg.diff', [{ a: 1, b: 2, c: 3 }, { a: 1, b: 20, d: 4 }]);
  assert.deepEqual(d.changed, ['b']);
  assert.deepEqual(d.added, ['d']);
  assert.deepEqual(d.removed, ['c']);
  assert.deepEqual(O('sys.cfg.diff', [{ a: 1 }, { a: 1 }]), { added: [], removed: [], changed: [] });

  const h1 = O('sys.cfg.hash', [{ a: 1, b: 2 }]);
  const h2 = O('sys.cfg.hash', [{ b: 2, a: 1 }]);
  assert.equal(h1, h2, '键序无关');
  assert.notEqual(h1, O('sys.cfg.hash', [{ a: 1, b: 3 }]));

  assert.deepEqual(O('sys.cfg.pick', [{ a: 1, b: 2 }, ['a', 'z']]), { a: 1 });
  assert.deepEqual(O('sys.cfg.defaults', [{ a: 1 }, { a: 9, b: 2 }]), { a: 1, b: 2 });
  assert.deepEqual(O('sys.cfg.toEnv', [{ maxIter: 5 }, 'app']), { APP_MAX_ITER: '5' });
  assert.equal(O('sys.cfg.validate', [{ a: 1 }, { a: 'number' }]).ok, true);
  assert.equal(O('sys.cfg.validate', [{ a: 'x' }, { a: { type: 'number', required: true } }]).ok, false);
});

test('10. sys.util.deepEqual 自反/对称/嵌套/数组；chunk 切分正确', () => {
  const eq = (a, b) => O('sys.util.deepEqual', [a, b]);
  const nested = { a: [1, { b: 2 }], c: 'x' };
  const nested2 = { a: [1, { b: 2 }], c: 'x' };
  assert.equal(eq(nested, nested), true, '自反');
  assert.equal(eq(nested, nested2), true, '结构相等');
  assert.equal(eq(nested2, nested), true, '对称');
  assert.equal(eq(nested, { a: [1, { b: 3 }], c: 'x' }), false);
  assert.equal(eq([1, 2, 3], [1, 2, 3]), true);
  assert.equal(eq([1, 2, 3], [1, 2]), false);
  assert.equal(eq(T([1, 2], [2]), T([1, 2], [2])), true);
  assert.equal(eq(T([1, 2], [2]), T([1, 2], [1, 2])), false, 'shape 不同');
  assert.equal(eq(null, {}), false);
  assert.equal(eq(1, '1'), false);

  assert.deepEqual(O('sys.util.chunk', [[1, 2, 3, 4, 5], 2]), [[1, 2], [3, 4], [5]]);
  assert.deepEqual(O('sys.util.chunk', [[1, 2, 3], 3]), [[1, 2, 3]]);
  assert.throws(() => O('sys.util.chunk', [[1], 0]), /E_VALUE/);

  assert.deepEqual(O('sys.util.range', [0, 5, 1]), [0, 1, 2, 3, 4]);
  assert.deepEqual(O('sys.util.range', [3, undefined, undefined]), [0, 1, 2]);
  assert.deepEqual(O('sys.util.zip', [[1, 2, 3], ['a', 'b']]), [[1, 'a'], [2, 'b']]);
  assert.equal(O('sys.util.pick', [{ a: { b: [7, 8] } }, 'a.b.1']), 8);
  assert.equal(O('sys.util.typeOf', [null]), 'null');
  assert.equal(O('sys.util.typeOf', [[]]), 'array');
  assert.equal(O('sys.util.typeOf', [T([1])]), 'tensor');
  assert.equal(O('sys.util.formatNum', [3.14159, 2]), '3.14');
  const tbl = O('sys.util.table', [[['a', 'bb'], ['ccc', 'd']], ['A', 'B']]);
  assert.ok(tbl.split('\n')[1].includes('---'), '含分隔行');
});

test('11. sys.err.classify 覆盖全部 CODES；isRetryable 区分 E_NUMERICAL / E_SHAPE', () => {
  const CAT = {
    E_ARITY: 'interface', E_TYPE: 'type', E_SHAPE: 'shape', E_VALUE: 'value',
    E_NUMERICAL: 'numerical', E_LIMIT: 'resource', E_NOTFOUND: 'lookup', E_STATE: 'state',
  };
  for (const code of Object.keys(CODES)) {
    assert.equal(O('sys.err.classify', [code]), CAT[code], `classify(${code})`);
    assert.notEqual(O('sys.err.suggest', [code]).length, 0);
  }
  assert.equal(O('sys.err.classify', ['E_NOPE']), 'unknown');
  assert.equal(O('sys.err.isRetryable', ['E_NUMERICAL']), true);
  assert.equal(O('sys.err.isRetryable', ['E_SHAPE']), false);
  assert.equal(O('sys.err.isRetryable', ['E_ARITY']), false);

  assert.equal(O('sys.err.format', ['E_SHAPE', 'x']), '[E_SHAPE] x');
  assert.equal(O('sys.err.format', ['E_ZZZ', 'x']), '[E_UNKNOWN] x');

  const codes = O('sys.err.codes', []);
  assert.equal(codes.length, Object.keys(CODES).length);
  for (const c of codes) {
    assert.ok(c.code && c.desc && c.category && typeof c.retryable === 'boolean');
  }

  const tr = O('sys.err.trace', ['_tmp.hard', []]);
  assert.equal(tr.ok, false);
  assert.equal(tr.code, 'E_SHAPE');
  const tr2 = O('sys.err.trace', ['sys.reg.version', []]);
  assert.equal(tr2.ok, true);
  assert.equal(tr2.code, null);
  assert.ok(Number.isInteger(O('sys.err.count', [undefined])));
});

test('12. sys.doc.* 导出可被解析：JSON / CSV / HTML / Mermaid', () => {
  const json = O('sys.doc.jsonExport', [ALL, false]);
  const parsed = JSON.parse(json);
  assert.equal(parsed.ops.length, ALL.length);
  assert.equal(parsed.count, ALL.length);
  assert.equal(parsed.kernel, '1.0.0');
  const item = parsed.ops.find(o => o.name === 'sys.reg.count');
  assert.equal(item.id, 'D10-001');

  const csv = O('sys.doc.csvExport', [ALL, ',']);
  const lines = csv.split('\n');
  assert.equal(lines.length, ALL.length + 1);
  assert.equal(lines[0].split(',').length, 7);

  const html = O('sys.doc.htmlExport', [ALL, '目录']);
  assert.ok(html.includes('<html'));
  assert.ok(html.includes('</html>'));
  assert.ok(html.includes('sys.reg.count'));

  const mm = O('sys.doc.mermaid', [ALL, 'graph']);
  assert.ok(mm.startsWith('graph'), `mermaid 应以 graph 开头: ${mm.slice(0, 20)}`);
  assert.ok(mm.includes('D10["D10 SYS'));

  const md = O('sys.doc.opMarkdown', [ALL, 'sys.reg.count']);
  assert.ok(md.startsWith('### '));
  assert.ok(md.includes('sys.reg.count') && md.includes('D10-001'));
  const dm = O('sys.doc.domainMarkdown', [ALL, 10]);
  assert.ok(dm.includes('## 域10 SYS 系统编排'));
  assert.ok(O('sys.doc.catalogMarkdown', [ALL, '目录']).includes('# 目录'));
  assert.ok(O('sys.doc.sigTable', [ALL, 10]).includes('| `sys.reg.count` |'));
  assert.ok(O('sys.doc.readmeSection', [ALL, 10]).startsWith('## 域10'));
});

test('13. sys.bench.timeOp ms>0 且 opsPerSec 有限；matmulFlops = 2·m·n·k', () => {
  const r = O('sys.bench.timeOp', [ALL, 'sys.reg.version', [], { n: 50, warmup: 2 }]);
  assert.ok(r.ms > 0, `ms 应 > 0，实际 ${r.ms}`);
  assert.ok(Number.isFinite(r.opsPerSec), `opsPerSec 应有限: ${r.opsPerSec}`);
  assert.equal(r.n, 50);
  assert.equal(r.failures, 0);

  const cmp = O('sys.bench.compare', [ALL, ['sys.reg.version', 'sys.reg.domains'], [], { n: 20, warmup: 1 }]);
  assert.equal(cmp.length, 2);
  assert.ok(cmp[0].ms <= cmp[1].ms, '按 ms 升序');

  const tp = O('sys.bench.throughput', [ALL, 'sys.reg.version', [], { budgetMs: 5, maxN: 5000 }]);
  assert.ok(tp.n >= 1 && Number.isFinite(tp.opsPerSec));

  const f = O('sys.bench.matmulFlops', [ALL, 16, 32, 64]);
  assert.equal(f.flops, 2 * 16 * 32 * 64);
  assert.ok(f.gflops > 0 && Number.isFinite(f.gflops));

  const mem = O('sys.bench.memoryEstimate', [ALL, [10, 10], 8]);
  assert.equal(mem.elements, 100);
  assert.equal(mem.bytes, 800);
  assert.ok(approx(mem.mb, 800 / 1048576, 1e-12));

  const sc = O('sys.bench.scaling', [ALL, 'sys.util.range', [1, 2, 4], {}]);
  assert.equal(sc.length, 3);
  assert.ok(sc.every(x => Number.isFinite(x.ms)));

  const suite = O('sys.bench.suite', [ALL.slice(0, 5), { limit: 5, trials: 1, warmup: 0 }]);
  assert.equal(suite.rows.length, 5);
  assert.ok(suite.rows.every(x => Number.isFinite(x.ms)));
  assert.ok(O('sys.bench.reportMarkdown', [ALL, [r]]).includes('| 基准报告') === false);
  assert.ok(O('sys.bench.reportMarkdown', [ALL, [r]]).includes('`sys.reg.version`'));
});

test('14. sys.verify.determinism 对纯函数为真；sys.verify.fuzz 健壮指令 0 失败', () => {
  const a = T([1, 2, 3]), b = T([4, 5, 6]);
  assert.equal(O('sys.verify.determinism', [ALL, 'math.add', [a, b], 5]), true);
  assert.equal(O('sys.verify.determinism', [ALL, 'math.l2', [a], 4]), true);
  assert.equal(O('sys.verify.determinism', [ALL, 'sys.reg.version', [], 3]), true);

  for (const nm of ['math.add', 'math.sub', 'math.mul', 'math.dot', 'math.l2', 'math.sum', 'math.mean']) {
    const fz = O('sys.verify.fuzz', [ALL, nm, { trials: 12, seed: 20261014 }]);
    assert.equal(fz.failures.length, 0, `${nm} 模糊测试失败: ${JSON.stringify(fz.failures.slice(0, 2))}`);
    assert.equal(fz.trials, 12);
  }
  // 恒抛错的指令应被记录为失败但不抛出
  const fz2 = O('sys.verify.fuzz', [ALL, '_tmp.hard', { trials: 3, seed: 1 }]);
  assert.equal(fz2.failures.length, 3);
  assert.equal(fz2.failures[0].code, 'E_SHAPE');
});

test('15. sys.self.checkAll 返回结构化结果（含每域计数）', () => {
  const chk = O('sys.self.checkAll', [ALL, { expected: EXPECT_PER_DOMAIN }]);
  assert.equal(chk.ok, true, JSON.stringify(chk.issues));
  assert.equal(chk.total, ALL.length);
  assert.equal(chk.byDomain['1'], 102);
  assert.equal(chk.byDomain['10'], 103);
  for (const key of ['nameUnique', 'idContinuity', 'arityCheck', 'docCoverage', 'refCoverage', 'invariants', 'domainCounts']) {
    assert.ok(chk.checks[key], `缺少子检查 ${key}`);
  }
  assert.equal(O('sys.self.exitCode', [ALL, { expected: EXPECT_PER_DOMAIN }]), 0);

  const sm = O('sys.self.summary', [ALL, { expected: EXPECT_PER_DOMAIN }]);
  assert.equal(sm.total, ALL.length);
  assert.equal(sm.ok, true);
  assert.match(sm.hash, /^0x[0-9a-f]{16}$/);
  assert.equal(sm.byDomain['10'], 103);

  const dh = O('sys.self.domainHealth', [ALL, 10]);
  assert.equal(dh.domain, 10);
  assert.equal(dh.code, 'SYS');
  assert.equal(dh.count, 103);
  assert.equal(dh.ok, true, JSON.stringify(dh.issues));

  const doc = O('sys.self.doctor', [ALL]);
  assert.ok(doc.startsWith('# 诊断报告'));
  assert.ok(doc.includes('| D10 | SYS'));
  const rep = O('sys.self.report', [ALL, '自检']);
  assert.ok(rep.startsWith('# 自检'));
  assert.ok(rep.includes(sm.hash));

  // 人为损坏一条：应被检出且退出码为 1
  const broken = ALL.map((o, i) => (i === 0 ? { ...o, name: ALL[1].name } : o));
  assert.equal(O('sys.self.checkAll', [broken, undefined]).ok, false);
  assert.equal(O('sys.self.exitCode', [broken, undefined]), 1);
});

test('16. 全部 103 条域10 指令冒烟调用不抛异常', () => {
  assert.equal(Object.keys(SMOKE).length, 103, '冒烟入参表须覆盖全部 103 条');
  const r = O('sys.verify.smoke', [D10, SMOKE]);
  assert.equal(r.total, 103);
  assert.equal(r.failures.length, 0, JSON.stringify(r.failures));
  assert.equal(r.ok, 103);
});

test('17. sys.reg.* 内省与 sys.tel.* 遥测基本行为', () => {
  assert.equal(O('sys.reg.count', [10]), 103);
  assert.ok(O('sys.reg.count', [undefined]) >= 103);
  const doms = O('sys.reg.domains', []);
  assert.equal(doms.length, 10);
  assert.equal(doms[9].code, 'SYS');
  const by = O('sys.reg.byDomain', [10]);
  assert.equal(by.length, 103);
  assert.ok(by.every(o => o.domain === 10 && !('impl' in o)));
  assert.ok(O('sys.reg.search', ['toposort']).some(o => o.name === 'sys.graph.toposort'));
  assert.ok(O('sys.reg.describe', ['sys.reg.count']).includes('D10-001'));
  assert.equal(O('sys.reg.exists', ['sys.reg.count']), true);
  assert.equal(O('sys.reg.exists', ['sys.nope']), false);
  assert.equal(O('sys.reg.signature', ['sys.util.chunk']), '(any[] arr, number size) -> any[][]');
  assert.equal(O('sys.reg.version', []).kernel, '1.0.0');
  assert.equal(O('sys.reg.nameIndex', [])['sys.util.chunk'].id, 'D10-102');
  const ids = O('sys.reg.opIds', [10]);
  assert.equal(ids.length, 103);
  assert.equal(ids[0], 'D10-001');
  const st = O('sys.reg.domainStats', []);
  assert.equal(st.length, 10);
  assert.equal(st[9].count, 103);
  assert.ok(approx(st.reduce((a, x) => a + x.ratio, 0), 1, 1e-9));

  O('sys.tel.reset', []);
  const before = O('sys.tel.totalCalls', []);
  O('sys.exec.call', ['sys.reg.version', []]);
  O('sys.exec.timed', ['sys.reg.version', []]);
  assert.ok(O('sys.tel.totalCalls', []) > before);
  const rec = O('sys.tel.record', ['sys.reg.version', false]);
  assert.ok(rec.errors >= 1 && rec.calls >= 1);
  const sm = O('sys.tel.summary', [3]);
  assert.ok(sm.length <= 3);
  assert.ok(O('sys.tel.hotOps', [2]).length <= 2);
  const er = O('sys.tel.errorRate', ['sys.reg.version']);
  assert.ok(er >= 0 && er <= 1);
  const lat = O('sys.tel.latencyStats', []);
  assert.ok(lat.n >= 2 && lat.min <= lat.p50 && lat.p50 <= lat.p95 && lat.p95 <= lat.max);
  assert.ok(O('sys.tel.topErrors', [3]).length <= 3);
  assert.equal(O('sys.tel.reset', []), true);
});

test('18. sys.env.* 环境内省与 sys.exec.* 其余组合', () => {
  const info = O('sys.env.info', []);
  assert.equal(info.platform, process.platform);
  assert.equal(info.arch, process.arch);
  assert.equal(info.node, process.versions.node);
  assert.equal(O('sys.env.nodeVersion', []), process.versions.node);
  assert.equal(O('sys.env.platform', []), `${process.platform}-${process.arch}`);
  const mem = O('sys.env.memory', []);
  assert.ok(mem.heapUsed >= 0 && mem.heapUsed <= mem.heapTotal);
  const t1 = O('sys.env.nowMs', []);
  const t2 = O('sys.env.nowMs', []);
  assert.ok(t2 >= t1);
  assert.ok(O('sys.env.uptimeMs', []) > 0);
  assert.equal(O('sys.env.seed', ['abc']), O('sys.env.seed', ['abc']));
  assert.notEqual(O('sys.env.seed', ['abc']), O('sys.env.seed', ['abd']));

  const bt = O('sys.exec.batch', [[['sys.reg.version', []], ['sys.nope', []], ['math.l2', [T([3, 4])]]]]);
  assert.equal(bt.length, 3);
  assert.equal(bt[0].ok, true);
  assert.equal(bt[1].ok, false);
  assert.equal(bt[2].ok, true);
  assert.ok(approx(bt[2].value, 5, 1e-12));

  const tk = O('sys.exec.try', ['sys.nope', []]);
  assert.equal(tk.ok, false);
  assert.equal(tk.error.code, 'E_NOTFOUND');

  const tm = O('sys.exec.timed', ['math.dot', [T([1, 2]), T([3, 4])]]);
  assert.ok(approx(tm.value, 11, 1e-12));
  assert.ok(tm.ms >= 0);

  assert.deepEqual(O('sys.exec.map', ['sys.util.typeOf', [1, 'a', null]]), ['number', 'string', 'null']);
  assert.equal(O('sys.exec.assertOp', ['math.l2', [T([3, 4])], (v) => Math.abs(v - 5) < 1e-9]), true);
  assert.equal(O('sys.exec.assertOp', ['math.l2', [T([3, 4])], (v) => v > 100]), false);
});
