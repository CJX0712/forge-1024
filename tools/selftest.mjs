// 编目自检：1024 条微指令的硬金标准。
// 运行：node tools/selftest.mjs
// 检查项：总数 / 每域 102(ML/DL/TRAIN/SYS 为 103) / id 连续唯一 / name 唯一 / 元数据齐全 / arity 推导一致 / 错误分类学。
import '../src/index.mjs';
import {
  listOps, countOps, opsByDomain, DOMAINS, KERNEL_VERSION, CATALOG_VERSION, requiredParams,
} from '../src/core/registry.mjs';
import { CODES } from '../src/core/errors.mjs';

// 编目规约：1024 = 6×102 + 4×103（D7–D10 四个能力面最广的域各多承载 1 条）
const PER_DOMAIN = { 1: 102, 2: 102, 3: 102, 4: 102, 5: 102, 6: 102, 7: 103, 8: 103, 9: 103, 10: 103 };

const GREEN = '\x1b[32m', RED = '\x1b[31m', DIM = '\x1b[2m', BOLD = '\x1b[1m', OFF = '\x1b[0m';

const results = [];
const check = (name, ok, detail = '') => results.push({ name, ok: !!ok, detail });

// —— 1. 总数 ——
const total = countOps();
check('指令总数 = 1024', total === 1024, `实际 ${total}`);

// —— 2. 每域恰好 102 且 id 连续 ——
const expectDomainCounts = Object.keys(DOMAINS).length;
check('域数量 = 10', expectDomainCounts === 10, `实际 ${expectDomainCounts}`);

for (const d of Object.keys(DOMAINS).map(Number).sort((a, b) => a - b)) {
  const ops = opsByDomain(d);
  const ids = ops.map(o => o.id).sort();
  const prefix = `D${d}-`;
  const okCount = ops.length === PER_DOMAIN[d];
  const okPrefix = ids.every(i => i.startsWith(prefix));
  const okUnique = new Set(ids).size === ids.length;
  const okContinuous = ids.every((id, i) => id === `${prefix}${String(i + 1).padStart(3, '0')}`);
  const code = DOMAINS[d].code;
  check(`域${d} (${code}) 指令数 = ${PER_DOMAIN[d]}`, okCount, `实际 ${ops.length}`);
  check(`域${d} (${code}) id 前缀/唯一/连续`, okPrefix && okUnique && okContinuous,
    `prefix=${okPrefix} unique=${okUnique} continuous=${okContinuous}`);
}

// —— 3. name 全局唯一 + 命名规范（域前缀） ——
const all = listOps();
const names = all.map(o => o.name);
const dupNames = names.filter((n, i) => names.indexOf(n) !== i);
check('name 全局唯一', dupNames.length === 0, dupNames.slice(0, 5).join(', '));
check('name 全局 1024 个不重复', new Set(names).size === 1024, `唯一 ${new Set(names).size}`);

const PREFIX = { 1: 'math', 2: 'num', 3: 'linalg', 4: 'ad', 5: 'opt', 6: 'prob', 7: 'ml', 8: 'dl', 9: 'train', 10: 'sys' };
const badPrefix = all.filter(o => !o.name.startsWith(PREFIX[o.domain] + '.') && o.name !== PREFIX[o.domain]);
check('name 使用域前缀', badPrefix.length === 0, badPrefix.slice(0, 5).map(o => o.name).join(', '));

// —— 4. 元数据齐全（doc/ref/inv/sig 非空且有意义长度） ——
const noDoc = all.filter(o => !o.doc || o.doc.length < 4);
const noRef = all.filter(o => !o.ref || o.ref.length < 6);
const noInv = all.filter(o => !o.inv || o.inv.length < 6);
const noSig = all.filter(o => !o.sig || !o.sig.includes('->'));
check('每条指令都有 doc', noDoc.length === 0, noDoc.slice(0, 3).map(o => o.name).join(', '));
check('每条指令都有 ref（论文/标准/出处）', noRef.length === 0, noRef.slice(0, 3).map(o => o.name).join(', '));
check('每条指令都有 inv（可验证不变量）', noInv.length === 0, noInv.slice(0, 3).map(o => o.name).join(', '));
check('每条指令都有 sig（含 ->）', noSig.length === 0, noSig.slice(0, 3).map(o => o.name).join(', '));

// —— 5. arity 由 impl 形参表推导的必需参数个数（机器反推，杜绝元数据漂移） ——
const badArity = all.filter(o => typeof o.impl !== 'function'
  || o.arity !== requiredParams(o.impl) || o.arity > o.impl.length);
check('arity = 由 impl 必需的形参个数（全部 1024 条）', badArity.length === 0,
  badArity.slice(0, 6).map(o => `${o.name}(decl=${o.arity},req=${requiredParams(o.impl)},impl=${o.impl.length})`).join(', '));

// —— 6. 错误分类学 ——
const REQUIRED_CODES = ['E_SHAPE', 'E_VALUE', 'E_NUMERICAL', 'E_ARITY', 'E_NOTFOUND'];
const missing = REQUIRED_CODES.filter(c => !(c in CODES));
check('错误分类学含 5 个必需码', missing.length === 0, missing.join(', '));
check('CODES 数量 >= 5', Object.keys(CODES).length >= 5, `实际 ${Object.keys(CODES).length}`);

// —— 7. 版本 ——
check('KERNEL_VERSION 非空', !!KERNEL_VERSION, KERNEL_VERSION);
check('CATALOG_VERSION 非空', !!CATALOG_VERSION, CATALOG_VERSION);

// —— 汇总 ——
const failed = results.filter(r => !r.ok);
console.log(`\n${BOLD}forge-1024 编目自检${OFF}  ${DIM}kernel ${KERNEL_VERSION} / catalog ${CATALOG_VERSION}${OFF}\n`);
for (const r of results) {
  const tag = r.ok ? `${GREEN}✓${OFF}` : `${RED}✗${OFF}`;
  console.log(`  ${tag} ${r.name}${r.detail && !r.ok ? `  ${RED}${r.detail}${OFF}` : ''}`);
}
const perDomain = Object.keys(DOMAINS).map(Number).sort((a, b) => a - b)
  .map(d => `${DOMAINS[d].code}:${opsByDomain(d).length}`).join('  ');
console.log(`\n  ${DIM}${perDomain}${OFF}`);
console.log(`\n  ${BOLD}总计 ${results.length} 项检查：${GREEN}${results.length - failed.length} 通过${OFF}` +
  (failed.length ? `，${RED}${failed.length} 失败${OFF}` : '') + '\n');
process.exit(failed.length ? 1 : 0);
