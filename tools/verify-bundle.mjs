// 校验 dist/forge-1024.html：单文件自洽 + 内核可真正执行（1024 条注册）。
// 运行：node tools/verify-bundle.mjs
// 方法：从 HTML 抽取内联的 JSON 模块载荷，落盘到临时目录（保持相对结构），
//       用 Node 真正 import 入口，统计注册指令数。这是执行级校验，不依赖字面量文本。
import { readFileSync, existsSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { join, resolve, dirname, posix } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { tmpdir } from 'node:os';
import { execSync } from 'node:child_process';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const target = join(root, 'dist', 'forge-1024.html');

const fail = [];
const ok = (cond, msg) => { console.log(`${cond ? '\x1b[32m✓\x1b[0m' : '\x1b[31m✗\x1b[0m'} ${msg}`); if (!cond) fail.push(msg); };

ok(existsSync(target), `产物存在: dist/forge-1024.html`);
if (!existsSync(target)) process.exit(1);

const html = readFileSync(target, 'utf8');

ok(html.includes('<!DOCTYPE html>') && html.trimEnd().endsWith('</html>'), 'HTML 结构完整');
ok(!/<script[^>]+src=/i.test(html), '无外部脚本引用（完全自包含）');
ok(!/<link[^>]+href=/i.test(html), '无外部样式引用（完全自包含）');
ok(!/https?:\/\/(?!www\.w3\.org)/i.test(html.replace(/https?:\/\/[^\s"']*github[^\s"']*/gi, '')),
  '无外部网络依赖');

// —— 抽取 JSON 模块载荷（src/<path> -> source）——
const m = html.match(/<script id="src" type="application\/json">([\s\S]*?)<\/script>/);
ok(!!m, '内联 JSON 模块载荷存在');
if (!m) process.exit(1);
let SOURCES;
try { SOURCES = JSON.parse(m[1]); }
catch (e) { ok(false, 'JSON 载荷可解析: ' + e.message); process.exit(1); }

const fileCount = Object.keys(SOURCES).length;
ok(fileCount >= 18, `内联模块数 ${fileCount} ≥ 18`);

// —— 落盘到临时目录并保持相对结构，真正 import 入口 ——
const sandbox = join(tmpdir(), 'forge-bundle-check-' + Date.now());
mkdirSync(sandbox, { recursive: true });
try {
  for (const [rel, src] of Object.entries(SOURCES)) {
    const p = join(sandbox, rel);
    mkdirSync(dirname(p), { recursive: true });
    writeFileSync(p, src, 'utf8');
  }
  const entry = pathToFileURL(join(sandbox, 'src', 'index.mjs')).href;
  const mod = await import(entry);
  const { listOps, countOps, opsByDomain, DOMAINS } = mod;

  const ALL = countOps();
  ok(ALL === 1024, `内核真正注册 ${ALL} 条指令（=1024）`);

  // 编目规约：6×102 + 4×103
  const PER_DOMAIN = { 1: 102, 2: 102, 3: 102, 4: 102, 5: 102, 6: 102, 7: 103, 8: 103, 9: 103, 10: 103 };
  for (const d of Object.keys(DOMAINS).map(Number).sort((a, b) => a - b)) {
    ok(opsByDomain(d).length === PER_DOMAIN[d], `域${d} ${DOMAINS[d].code} = ${PER_DOMAIN[d]} 条（实际 ${opsByDomain(d).length}）`);
  }

  // 元数据齐全
  const all = listOps();
  const badMeta = all.filter(o => !o.doc || !o.ref || !o.inv || !o.sig || !o.sig.includes('->'));
  ok(badMeta.length === 0, `每条指令 doc/ref/inv/sig 齐全（缺 ${badMeta.length}）`);

  // name 全局唯一
  const names = all.map(o => o.name);
  ok(new Set(names).size === names.length, 'name 全局唯一');
} finally {
  rmSync(sandbox, { recursive: true, force: true });
}

// 产物体积
const kb = html.length / 1024;
console.log(`\n  单文件体积 ${kb.toFixed(1)} KB`);
console.log(`  ${fail.length ? `\x1b[31m${fail.length} 项未通过\x1b[0m` : '\x1b[32m全部通过\x1b[0m'}\n`);
process.exit(fail.length ? 1 : 0);
