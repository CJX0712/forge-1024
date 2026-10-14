// 单文件 HTML 打包：把整个内核（10 域 1024 条）内联进一个 .html，浏览器内可自检。
// 运行：node tools/bundle-html.mjs  → dist/forge-1024.html
// 方案：保留真正的 ES module 语义——按依赖拓扑序为每个源文件创建 Blob URL，
//       并把相对 import 说明符重写为对应 Blob URL，最后动态 import 入口。
//       这样不改写任何一行源码，也不引入打包器（零依赖原则）。
import { readFileSync, writeFileSync, mkdirSync, readdirSync, statSync } from 'node:fs';
import { join, resolve, dirname, posix } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');

// —— 依赖拓扑序（core 先于 ops；域间按 D1→D10）——
const ORDER = [
  'src/core/errors.mjs',
  'src/core/tensor.mjs',
  'src/core/rng.mjs',
  'src/core/telemetry.mjs',
  'src/core/registry.mjs',
  'src/core/index.mjs',
  'src/ops/d1-math.mjs',
  'src/ops/d2-num.mjs',
  'src/ops/d3-linalg.mjs',
  'src/ops/d4-autodiff.mjs',
  'src/ops/d5-opt.mjs',
  'src/ops/d6-prob.mjs',
  'src/ops/d7-ml.mjs',
  'src/ops/d8-dl.mjs',
  'src/ops/d9-train.mjs',
  'src/ops/d10-sys.mjs',
  'src/ops/index.mjs',
  'src/index.mjs',
];

// —— 校验：src 下不得有遗漏文件 ——
function walk(dir, out = []) {
  for (const e of readdirSync(dir)) {
    const p = join(dir, e);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (p.endsWith('.mjs')) out.push(p.replace(/\\/g, '/').replace(resolve(root).replace(/\\/g, '/') + '/', ''));
  }
  return out;
}
const allFiles = walk(join(root, 'src')).sort();
const missing = allFiles.filter(f => !ORDER.includes(f));
if (missing.length) {
  console.error('打包顺序遗漏文件（请补进 ORDER）:', missing);
  process.exit(1);
}

const sources = {};
for (const f of ORDER) {
  sources[f] = readFileSync(join(root, f), 'utf8');
}

const esc = (s) => JSON.stringify(s).replace(/<\/script/gi, '<\\/script');
const payload = `{${ORDER.map(f => `${JSON.stringify(f)}:${esc(sources[f])}`).join(',\n')}}`;

const html = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>forge-1024 · AI 微指令内核（浏览器内自检）</title>
<style>
  :root { --bg:#0f1115; --fg:#e6e8ee; --dim:#8b93a7; --acc:#5cc8ff; --ok:#3ddc84; --bad:#ff6b6b; --card:#171a21; --line:#262b36; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  .wrap { max-width: 980px; margin: 0 auto; padding: 28px 20px 80px; }
  h1 { font-size: 20px; margin: 0 0 4px; letter-spacing:.5px; }
  h2 { font-size: 15px; margin: 26px 0 10px; color: var(--acc); border-bottom:1px solid var(--line); padding-bottom:6px; }
  .sub { color: var(--dim); font-size: 12px; margin-bottom: 18px; }
  .card { background: var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:14px; }
  .row { display:flex; gap:10px; flex-wrap:wrap; }
  .stat { flex:1 1 160px; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
  .stat b { display:block; font-size:22px; color:var(--acc); }
  .stat span { color:var(--dim); font-size:11px; }
  .ok { color: var(--ok); } .bad { color: var(--bad); }
  input, select { background:#0c0e13; color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:7px 9px; font:inherit; }
  input { flex:1 1 240px; }
  table { width:100%; border-collapse: collapse; font-size:12px; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align: top; }
  th { color: var(--dim); font-weight:600; position: sticky; top:0; background:var(--bg); }
  td.id { color: var(--dim); white-space: nowrap; }
  .scroll { max-height: 380px; overflow:auto; border:1px solid var(--line); border-radius:8px; }
  pre { margin:0; white-space: pre-wrap; word-break: break-all; font-size:12px; color:var(--dim); }
  button { background:#1b2230; color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:7px 12px; cursor:pointer; font:inherit; }
  button:hover { border-color: var(--acc); }
  code { color: var(--acc); }
</style>
</head>
<body>
<div class="wrap">
  <h1>forge-1024 · AI 微指令内核</h1>
  <div class="sub">零依赖 · 零构建 · 浏览器内自检 —— 10 域 1024 条可验证微指令（6×102 + 4×103）。作者：晨星。</div>

  <div id="boot" class="card">正在装载内核…</div>

  <div class="row" id="stats" style="display:none"></div>

  <h2>编目自检</h2>
  <div class="card"><pre id="selftest">—</pre></div>

  <h2>指令检索（1024 条全量索引）</h2>
  <div class="card">
    <div class="row" style="margin-bottom:10px">
      <input id="q" placeholder="按名称 / 说明 / id 搜索，例如 softmax、svd、klDiv">
      <select id="dom">
        <option value="">全部域</option>
      </select>
    </div>
    <div class="scroll">
      <table>
        <thead><tr><th>id</th><th>指令</th><th>签名</th><th>说明</th><th>不变量</th><th>参考</th></tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
    <div class="sub" id="count" style="margin:10px 0 0"></div>
  </div>

  <h2>在线执行</h2>
  <div class="card">
    <div class="row" style="margin-bottom:10px">
      <input id="callName" style="flex:0 1 260px" placeholder="指令名，例如 math.add">
      <input id="callArgs" placeholder="JSON 参数数组，例如 [1,2]">
      <button id="run">执行</button>
    </div>
    <pre id="out">—</pre>
  </div>
</div>

<script id="src" type="application/json">${payload}</script>
<script type="module">
const SOURCES = JSON.parse(document.getElementById('src').textContent);
const ORDER = ${JSON.stringify(ORDER)};
const urls = new Map();

// 相对说明符 -> 规范化项目内路径
function resolveSpec(fromFile, spec) {
  const dir = fromFile.split('/').slice(0, -1).join('/');
  const parts = (dir + '/' + spec).split('/');
  const out = [];
  for (const p of parts) {
    if (p === '.' || p === '') continue;
    if (p === '..') { out.pop(); continue; }
    out.push(p);
  }
  return out.join('/');
}

// 把源码中的 import ... from './x.mjs' 重写为已生成的 Blob URL
function rewrite(file, src) {
  return src.replace(/(from\\s*|import\\s*\\(\\s*)(['"])(\\.{1,2}\\/[^'"]+)\\2/g, (m, kw, q, spec) => {
    const key = resolveSpec(file, spec);
    if (!urls.has(key)) throw new Error('依赖未按拓扑序装载: ' + key + ' (被 ' + file + ' 引用)');
    return kw + q + urls.get(key) + q;
  });
}

const boot = document.getElementById('boot');
try {
  for (const f of ORDER) {
    const code = rewrite(f, SOURCES[f]);
    urls.set(f, URL.createObjectURL(new Blob([code], { type: 'text/javascript' })));
  }
  const mod = await import(urls.get('src/index.mjs'));
  window.FORGE = mod;
  boot.className = 'card';
  boot.innerHTML = '<span class="ok">✓ 内核装载成功</span> —— ' + ORDER.length + ' 个模块以 Blob ESM 方式内联，未改写任何源码。';
  start(mod);
} catch (e) {
  boot.innerHTML = '<span class="bad">✗ 装载失败: ' + String(e && e.message || e) + '</span>';
  throw e;
}

function start(mod) {
  const { listOps, countOps, opsByDomain, DOMAINS, exec, tryExec, KERNEL_VERSION, CATALOG_VERSION, describeOp } = mod;
  const all = listOps();

  // —— 统计卡 ——
  const stats = [
    ['指令总数', countOps()],
    ['域', Object.keys(DOMAINS).length],
    ['每域', '102 / 103'],
    ['内核版本', KERNEL_VERSION],
    ['编目版本', CATALOG_VERSION],
  ];
  const box = document.getElementById('stats');
  box.style.display = 'flex';
  box.innerHTML = stats.map(([k, v]) => '<div class="stat"><b>' + v + '</b><span>' + k + '</span></div>').join('');

  // —— 编目自检 ——
  const lines = [];
  const chk = (n, ok, d) => lines.push((ok ? '\\u2713 ' : '\\u2717 ') + n + (d ? '  — ' + d : ''));
  chk('指令总数 = 1024', countOps() === 1024, String(countOps()));
  const PER_DOMAIN = { 1: 102, 2: 102, 3: 102, 4: 102, 5: 102, 6: 102, 7: 103, 8: 103, 9: 103, 10: 103 };
  for (const d of Object.keys(DOMAINS).map(Number).sort((a, b) => a - b)) {
    const ops = opsByDomain(d);
    const ids = ops.map(o => o.id).sort();
    const cont = ids.every((id, i) => id === 'D' + d + '-' + String(i + 1).padStart(3, '0'));
    chk('D' + d + ' ' + DOMAINS[d].code + ' ' + PER_DOMAIN[d] + ' 条 / id 连续', ops.length === PER_DOMAIN[d] && cont, ops.length + ' 条');
  }
  const names = all.map(o => o.name);
  chk('name 全局唯一', new Set(names).size === names.length);
  chk('每条含 doc / ref / inv / sig', all.every(o => o.doc && o.ref && o.inv && o.sig && o.sig.includes('->')));
  chk('arity = 由 impl 必需参数推导', all.every(o => o.arity <= o.impl.length),
      all.filter(o => o.arity > o.impl.length).length + ' 条不符');
  document.getElementById('selftest').textContent = lines.join('\\n');

  // —— 检索 ——
  const domSel = document.getElementById('dom');
  for (const d of Object.keys(DOMAINS).map(Number).sort((a, b) => a - b)) {
    const o = document.createElement('option');
    o.value = String(d); o.textContent = 'D' + d + ' ' + DOMAINS[d].code + ' ' + DOMAINS[d].name;
    domSel.appendChild(o);
  }
  const rows = document.getElementById('rows');
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  function render() {
    const q = document.getElementById('q').value.trim().toLowerCase();
    const d = domSel.value;
    const hit = all.filter(o => (!d || String(o.domain) === d) &&
      (!q || o.name.toLowerCase().includes(q) || o.doc.toLowerCase().includes(q) || o.id.toLowerCase().includes(q)));
    document.getElementById('count').textContent = '命中 ' + hit.length + ' / ' + all.length + ' 条';
    rows.innerHTML = hit.slice(0, 300).map(o => '<tr><td class="id">' + o.id + '</td><td><code>' + esc(o.name) +
      '</code></td><td>' + esc(o.sig) + '</td><td>' + esc(o.doc) + '</td><td>' + esc(o.inv) +
      '</td><td>' + esc(o.ref) + '</td></tr>').join('');
  }
  document.getElementById('q').addEventListener('input', render);
  domSel.addEventListener('change', render);
  render();

  // —— 在线执行 ——
  const out = document.getElementById('out');
  document.getElementById('run').addEventListener('click', () => {
    const name = document.getElementById('callName').value.trim();
    let args = [];
    try { args = JSON.parse(document.getElementById('callArgs').value || '[]'); } catch (e) { out.textContent = '参数 JSON 解析失败: ' + e.message; return; }
    if (!Array.isArray(args)) args = [args];
    const r = tryExec(name, args);
    out.textContent = r.ok ? ('\\u2713 ' + JSON.stringify(r.value, (k, v) => v instanceof Float64Array ? Array.from(v) : v, 1))
                           : ('\\u2717 ' + (r.error && r.error.message || String(r.error)));
  });
}
</script>
</body>
</html>
`;

mkdirSync(join(root, 'dist'), { recursive: true });
const target = join(root, 'dist', 'forge-1024.html');
writeFileSync(target, html, 'utf8');
console.log(`已生成 ${target}（${(html.length / 1024).toFixed(1)} KB，内联 ${ORDER.length} 个模块）`);
