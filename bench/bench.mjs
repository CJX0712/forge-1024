// 性能基准：无依赖微基准，输出 Markdown 表格到 stdout 与 docs/BENCHMARKS.md。
// 运行：node bench/bench.mjs
// 方法学：每项目先预热，再重复计时取总耗时；报告 ops/s（越大越好）与 ms。
import '../src/index.mjs';
import { exec } from '../src/core/registry.mjs';
import { Tensor } from '../src/core/tensor.mjs';

const SIZES = { small: 64, mid: 256, big: 512 };

function timeIt(label, fn, { warmup = 3, reps = 10 } = {}) {
  for (let i = 0; i < warmup; i++) fn();
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < reps; i++) fn();
  const t1 = process.hrtime.bigint();
  const ms = Number(t1 - t0) / 1e6 / reps;
  return { label, ms: +ms.toFixed(4), opsPerSec: Math.round(1000 / ms) };
}

const rows = [];
const push = (group, r) => rows.push({ group, ...r });

// —— 1. 逐元素 / 归约（memory-bound） ——
{
  const n = 1e6;
  const a = Tensor.fromFlat(Array.from({ length: n }, (_, i) => Math.sin(i)), [n]);
  const b = Tensor.fromFlat(Array.from({ length: n }, (_, i) => Math.cos(i)), [n]);
  push('elementwise', timeIt(`add  ${n} 元素`, () => exec('math.add', [a, b])));
  push('elementwise', timeIt(`mul  ${n} 元素`, () => exec('math.mul', [a, b])));
  push('elementwise', timeIt(`exp  ${n} 元素`, () => exec('math.exp', [a])));
  push('elementwise', timeIt(`sum  ${n} 元素`, () => exec('math.sum', [a])));
  push('elementwise', timeIt(`Kahan 求和 ${n}`, () => exec('num.kahanSum', [a])));
}

// —— 2. 矩阵乘（compute-bound，分块） ——
for (const [name, m] of Object.entries(SIZES)) {
  const A = Tensor.fromFlat(Array.from({ length: m * m }, (_, i) => (i % 97) / 97), [m, m]);
  const B = Tensor.fromFlat(Array.from({ length: m * m }, (_, i) => (i % 89) / 89), [m, m]);
  const r = timeIt(`matmul ${m}×${m}`, () => exec('linalg.matmul', [A, B]), { reps: m >= 512 ? 3 : 10 });
  const gflops = (2 * m * m * m) / (r.ms / 1000) / 1e9;
  push('matmul', { ...r, gflops: +gflops.toFixed(3) });
}

// —— 3. 分解 ——
for (const [name, m] of Object.entries({ small: 64, mid: 200, big: 400 })) {
  const A = Tensor.fromFlat(Array.from({ length: m * m }, (_, i) => {
    const r0 = Math.floor(i / m), c0 = i % m;
    return r0 === c0 ? m + 1 : 1 / (1 + Math.abs(r0 - c0));
  }), [m, m]);
  push('decomp', timeIt(`lu ${m}×${m}`, () => exec('linalg.lu', [A]), { reps: 5 }));
  push('decomp', timeIt(`qr ${m}×${m}`, () => exec('linalg.qr', [A]), { reps: 5 }));
  push('decomp', timeIt(`chol ${m}×${m}`, () => exec('linalg.cholesky', [A]), { reps: 5 }));
  push('decomp', timeIt(`eigSym ${m}×${m}`, () => exec('linalg.eigenSym', [A]), { reps: 3 }));
}

// —— 4. 自动微分 ——
{
  const d = 128, h = 256;
  const X = Tensor.fromFlat(Array.from({ length: 32 * d }, (_, i) => Math.sin(i) * 0.5), [32, d]);
  push('autodiff', timeIt(`tape forward+backward (32×${d})`, () => {
    const t = exec('ad.tapeReset', []);
    const W = Tensor.fromFlat(Array.from({ length: d * h }, (_, i) => Math.sin(i) * 0.01), [d, h]);
    const y = exec('ad.matmul', [X, W, t]);
    const s = exec('ad.sumAll', [y]);
    exec('ad.backward', [s, t]);
  }, { reps: 20 }));
}

// —— 5. 优化器步长 ——
{
  const k = 4096;
  const p = Tensor.fromFlat(Array.from({ length: k }, () => 0.1), [k]);
  const g = Tensor.fromFlat(Array.from({ length: k }, () => 0.01), [k]);
  let st = exec('opt.adam', [p, { lr: 0.01 }]);
  push('optimizer', timeIt('adam.step (4096 维)', () => { st = exec('opt.adam.step', [p, g, st, { lr: 0.01 }]).state; }, { reps: 200 }));
}

// —— 输出 ——
const now = new Date();
const env = `Node ${process.versions.node} / ${process.platform} ${process.arch}`;
const L = [];
L.push('# 性能基准（BENCHMARKS）');
L.push('');
L.push(`> 由 \`node bench/bench.mjs\` 生成 · ${now.toISOString()}`);
L.push(`> 环境：${env} · 零依赖 · 单线程 · Float64Array 后端`);
L.push('');
L.push('| 类别 | 项目 | 耗时 (ms) | 吞吐 (ops/s) | 备注 |');
L.push('|---|---|---:|---:|---|');
for (const r of rows) {
  L.push(`| ${r.group} | ${r.label} | ${r.ms} | ${r.opsPerSec} | ${r.gflops ? `${r.gflops} GFLOP/s` : ''} |`);
}
L.push('');
L.push('## 方法学');
L.push('');
L.push('- 每项先预热 3 次再重复计时，取平均单次耗时，避免 JIT 冷启动污染。');
L.push('- `linalg.matmul` 采用分块（cache-blocking）实现，GFLOP/s 按 2·m·n·k 计算。');
L.push('- 计时使用 `process.hrtime.bigint()`（纳秒分辨率），不引入任何第三方基准库。');
L.push('- 遥测默认关闭计时，基准结果不被内核统计开销污染。');
L.push('');

const { writeFileSync, mkdirSync } = await import('node:fs');
mkdirSync(new URL('../docs/', import.meta.url), { recursive: true });
writeFileSync(new URL('../docs/BENCHMARKS.md', import.meta.url), L.join('\n'), 'utf8');

console.log(L.join('\n'));
console.log('已写入 docs/BENCHMARKS.md');
