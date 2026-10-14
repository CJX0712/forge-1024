// 编目级测试：1024 条微指令的硬金标准（不变量层）
import { test } from 'node:test';
import assert from 'node:assert/strict';
import '../src/index.mjs';
import {
  listOps, countOps, opsByDomain, DOMAINS, KERNEL_VERSION, CATALOG_VERSION, getOp, exec, requiredParams,
} from '../src/core/registry.mjs';
import { ForgeError, CODES } from '../src/core/errors.mjs';

// 编目规约：1024 = 6×102 + 4×103（D7–D10 四个能力面最广的域各多承载 1 条）
const EXPECT = { 1: 102, 2: 102, 3: 102, 4: 102, 5: 102, 6: 102, 7: 103, 8: 103, 9: 103, 10: 103 };
const PREFIX = { 1: 'math', 2: 'num', 3: 'linalg', 4: 'ad', 5: 'opt', 6: 'prob', 7: 'ml', 8: 'dl', 9: 'train', 10: 'sys' };

test('指令总数恰好 1024（6×102 + 4×103）', () => {
  assert.equal(countOps(), 1024);
  assert.equal(Object.keys(DOMAINS).length, 10);
  const sum = Object.values(EXPECT).reduce((a, b) => a + b, 0);
  assert.equal(sum, 1024, '各域期望条数之和必须为 1024');
  for (const d of Object.keys(DOMAINS).map(Number)) {
    assert.equal(opsByDomain(d).length, EXPECT[d], `域${d} 指令数应为 ${EXPECT[d]}`);
  }
});

test('id 形如 D<n>-001..N，全局唯一且每域连续', () => {
  const seen = new Set();
  for (const d of Object.keys(DOMAINS).map(Number)) {
    const ids = opsByDomain(d).map(o => o.id).sort();
    assert.equal(new Set(ids).size, EXPECT[d], `域${d} id 重复`);
    ids.forEach((id, i) => {
      assert.equal(id, `D${d}-${String(i + 1).padStart(3, '0')}`, `域${d} id 不连续于 ${id}`);
      assert.ok(!seen.has(id), `id 全局重复: ${id}`);
      seen.add(id);
    });
  }
  assert.equal(seen.size, 1024);
});

test('name 全局唯一且带域前缀', () => {
  const names = listOps().map(o => o.name);
  assert.equal(new Set(names).size, 1024);
  for (const o of listOps()) {
    assert.ok(o.name.startsWith(PREFIX[o.domain] + '.'), `${o.name} 缺少域前缀 ${PREFIX[o.domain]}.`);
  }
});

test('每条指令元数据齐全：doc / ref / inv / sig', () => {
  for (const o of listOps()) {
    assert.ok(o.doc && o.doc.length >= 4, `${o.name} doc 缺失`);
    assert.ok(o.ref && o.ref.length >= 6, `${o.name} ref 缺失或过短: ${o.ref}`);
    assert.ok(o.inv && o.inv.length >= 6, `${o.name} inv 缺失或过短: ${o.inv}`);
    assert.ok(o.sig && o.sig.includes('->'), `${o.name} sig 缺失`);
  }
});

test('arity = 由 impl 形参表推导的必需参数个数（1024 条全部成立）', () => {
  // arity 由注册表从实现反推，杜绝手写元数据漂移；同时必须满足 arity ≤ impl 形参总数
  const bad = listOps().filter(o => typeof o.impl !== 'function'
    || o.arity !== requiredParams(o.impl) || o.arity > o.impl.length);
  assert.equal(bad.length, 0,
    `arity 不符: ${bad.slice(0, 8).map(o => `${o.name}(${o.arity} vs ${requiredParams(o.impl)}/${o.impl.length})`).join(', ')}`);
  // 抽查：至少存在带可选参数的指令（证明推导器能识别可选性，而非一律取形参数）
  const withOptional = listOps().filter(o => o.impl.length > o.arity);
  assert.ok(withOptional.length > 0, '应存在带可选参数的指令');
});

test('getOp 支持 id 与 name 双索引；未知指令抛 E_NOTFOUND', () => {
  const o = listOps()[0];
  assert.equal(getOp(o.id).name, o.name);
  assert.equal(getOp(o.name).id, o.id);
  assert.throws(() => getOp('definitely.not.an.op'),
    (e) => e instanceof ForgeError && e.code === 'E_NOTFOUND');
});

test('错误分类学：E_SHAPE/E_VALUE/E_NUMERICAL/E_ARITY/E_NOTFOUND 齐备', () => {
  for (const c of ['E_SHAPE', 'E_VALUE', 'E_NUMERICAL', 'E_ARITY', 'E_NOTFOUND']) {
    assert.ok(c in CODES, `缺少错误码 ${c}`);
  }
  assert.ok(Object.keys(CODES).length >= 6);
});

test('版本常量非空', () => {
  assert.ok(KERNEL_VERSION && typeof KERNEL_VERSION === 'string');
  assert.ok(CATALOG_VERSION && typeof CATALOG_VERSION === 'string');
});

test('全 1024 条：参数不足必抛 E_ARITY，且不会抛出非 ForgeError 的编程错误', () => {
  const bad = [];
  for (const o of listOps()) {
    // 1) 少传一个参数（arity>0 时传 arity−1 个）必须报 E_ARITY
    if (o.arity > 0) {
      try {
        exec(o.name, new Array(o.arity - 1).fill(null));
        bad.push(`${o.name}: 参数不足未报 E_ARITY`);
      } catch (e) {
        if (!(e instanceof ForgeError) || e.code !== 'E_ARITY') {
          bad.push(`${o.name}: 参数不足抛出 ${e.code || e.constructor.name}`);
        }
      }
    }
  }
  assert.equal(bad.length, 0, bad.slice(0, 8).join(' | '));
});
