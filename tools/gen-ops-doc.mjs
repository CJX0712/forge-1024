// 文档生成：从注册表反射出 docs/OPS.md（指令全集）与 docs/REFERENCES.md（参考文献）。
// 运行：node tools/gen-ops-doc.mjs
// 原则：文档由代码生成，永不手写，杜绝「文档与实现漂移」。
import { writeFileSync, mkdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import '../src/index.mjs';
import { listOps, opsByDomain, DOMAINS, KERNEL_VERSION, CATALOG_VERSION, countOps } from '../src/core/registry.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
mkdirSync(resolve(root, 'docs'), { recursive: true });

const esc = (s) => String(s).replace(/\|/g, '\\|').replace(/\n/g, ' ');
const all = listOps();

// —— OPS.md ——
const L = [];
L.push('# forge-1024 指令全集（OPS）');
L.push('');
L.push(`> 本文件由 \`node tools/gen-ops-doc.mjs\` 从注册表反射生成，**请勿手工编辑**。`);
L.push(`> 内核 ${KERNEL_VERSION} / 编目 ${CATALOG_VERSION} / 指令总数 **${countOps()}**。`);
L.push('');
L.push('## 目录');
L.push('');
for (const d of Object.keys(DOMAINS).map(Number).sort((a, b) => a - b)) {
  const info = DOMAINS[d];
  L.push(`- [D${d} · ${info.code} ${info.name}（102 条）](#d${d}--${info.code.toLowerCase()}) — ${info.desc}`);
}
L.push('');
L.push('---');
L.push('');
for (const d of Object.keys(DOMAINS).map(Number).sort((a, b) => a - b)) {
  const info = DOMAINS[d];
  const ops = opsByDomain(d).sort((a, b) => a.id.localeCompare(b.id));
  L.push(`## D${d} · ${info.code} ${info.name}`);
  L.push('');
  L.push(`${info.desc}。共 ${ops.length} 条。`);
  L.push('');
  L.push('| id | 指令 | 签名 | 说明 | 不变量 |');
  L.push('|---|---|---|---|---|');
  for (const o of ops) {
    L.push(`| \`${o.id}\` | \`${o.name}\` | \`${esc(o.sig)}\` | ${esc(o.doc)} | ${esc(o.inv)} |`);
  }
  L.push('');
}
writeFileSync(resolve(root, 'docs/OPS.md'), L.join('\n'), 'utf8');

// —— REFERENCES.md ——
const refs = new Map(); // ref -> [name]
for (const o of all) {
  if (!refs.has(o.ref)) refs.set(o.ref, []);
  refs.get(o.ref).push(o.name);
}
const R = [];
R.push('# 参考文献（REFERENCES）');
R.push('');
R.push('> 本文件由 `node tools/gen-ops-doc.mjs` 从指令元数据 `ref` 字段自动聚合生成。');
R.push(`> 共 ${refs.size} 条独立出处，覆盖 ${all.length} 条微指令。`);
R.push('');
R.push('每条微指令都携带 `ref`（论文 / 标准 / 权威实现）与 `inv`（可验证不变量），');
R.push('使得任意一条实现都能被追溯到原始文献，并被独立的数学性质交叉验证。');
R.push('');
const sorted = [...refs.entries()].sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]));
R.push('| 出处 | 引用指令数 | 指令示例 |');
R.push('|---|---|---|');
for (const [ref, names] of sorted) {
  R.push(`| ${esc(ref)} | ${names.length} | ${names.slice(0, 3).map(n => `\`${n}\``).join('、')}${names.length > 3 ? ' 等' : ''} |`);
}
R.push('');
writeFileSync(resolve(root, 'docs/REFERENCES.md'), R.join('\n'), 'utf8');

console.log(`已生成 docs/OPS.md（${all.length} 条指令）与 docs/REFERENCES.md（${refs.size} 条出处）`);
