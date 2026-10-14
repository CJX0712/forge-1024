// 示例 06：指令编目自检与文档生成（域10 系统编排能力演示）
// 运行：node examples/06-op-catalog.mjs
// 展示：注册表内省、编目哈希、不变量检查、文档反射、遥测统计。
import '../src/index.mjs';
import { exec, listOps } from '../src/core/registry.mjs';
import { countOps, opsByDomain, DOMAINS, KERNEL_VERSION, CATALOG_VERSION } from '../src/core/registry.mjs';
import { Tensor } from '../src/core/tensor.mjs';

console.log('示例 06 · 指令编目与系统编排');
console.log('============================');
console.log(`内核 ${KERNEL_VERSION} / 编目 ${CATALOG_VERSION} / 指令总数 ${countOps()}\n`);

console.log('域分布：');
for (const d of Object.keys(DOMAINS).map(Number).sort((a, b) => a - b)) {
  const info = DOMAINS[d];
  const n = opsByDomain(d).length;
  const bar = '█'.repeat(Math.round(n / 102 * 20));
  console.log(`  D${String(d).padStart(2)} ${info.code.padEnd(9)} ${info.name.padEnd(12)} ${String(n).padStart(4)}  ${bar}`);
}

// —— 编目哈希（对内容敏感的稳定摘要） ——
const h1 = exec('sys.reg.catalogHash', []);
const h2 = exec('sys.reg.catalogHash', []);
console.log(`\n编目哈希：${h1}`);
console.log(`二次调用一致：${h1 === h2 ? '✓' : '✗'}`);

// —— 元数据完整度 ——
const cov = exec('sys.verify.docCoverage', [opsByDomain(1)]);
const refc = exec('sys.verify.refCoverage', [opsByDomain(1)]);
console.log(`域1 文档覆盖率：${JSON.stringify(cov)}`);
console.log(`域1 参考覆盖率：${JSON.stringify(refc)}`);

// —— 唯一性与 id 连续性 ——
console.log(`name 唯一：${JSON.stringify(exec('sys.verify.nameUnique', [opsByDomain(1).concat(opsByDomain(2))]))}`);
console.log(`id 连续  ：${JSON.stringify(exec('sys.verify.idContinuity', [opsByDomain(1)]))}`);

// —— 文档反射：把域1 渲染成 Markdown（前 6 行预览） ——
const md = exec('sys.doc.domainMarkdown', [opsByDomain(1), 1]);
console.log('\n域1 Markdown 反射（前 6 行）：');
console.log(String(md).split('\n').slice(0, 6).map(l => '  ' + l).join('\n'));

// —— 模糊测试：对 math.add 做随机入参健壮性测试 ——
const fuzz = exec('sys.verify.fuzz', [listOps(), 'math.add', { trials: 200, seed: 1 }]);
console.log(`\n模糊测试 math.add：${JSON.stringify(fuzz)}`);

// —— 遥测 ——
exec('sys.tel.reset', []);
for (let i = 0; i < 50; i++) exec('math.add', [Tensor.fromFlat([1], [1]), Tensor.fromFlat([2], [1])]);
console.log(`遥测总调用数：${exec('sys.tel.totalCalls', [])}`);
console.log(`热点指令 top3：${JSON.stringify((exec('sys.tel.hotOps', [3]) || []).map(r => `${r.id}×${r.calls}`))}`);

console.log('\n结论：编目可由代码完全反射——哈希稳定、元数据齐备、文档自动生成、');
console.log('      健壮性可模糊测试、调用可遥测，1024 条指令构成一个自描述、自验证的内核。');
