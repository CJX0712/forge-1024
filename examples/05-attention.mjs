// 示例 05：注意力机制 —— 缩放点积 / 因果掩码 / 多头 / 旋转位置编码 / KV 缓存
// 运行：node examples/05-attention.mjs
// 展示：注意力权重行随机、因果掩码上三角为 0、RoPE 相对位置不变性、KV 缓存与全量计算等价。
// 注：dl.attn.* 系列采用严格的 3D 约定 [B,T,D]；ad.* 教学版接受 2D [n,d]。
import '../src/index.mjs';
import { exec } from '../src/core/registry.mjs';
import { Tensor } from '../src/core/tensor.mjs';
import { createRng } from '../src/core/rng.mjs';

const rng = createRng(11);
const T = 6, C = 8, HEADS = 2;

const rand = (rows, cols) => Tensor.fromFlat(
  Array.from({ length: rows * cols }, () => rng.normal() * 0.5), [rows, cols]);
const Q = rand(T, C), K = rand(T, C), V = rand(T, C);
// dl.attn.* 严格 3D：[B,T,D]
const Q3 = Q.reshape([1, T, C]), K3 = K.reshape([1, T, C]), V3 = V.reshape([1, T, C]);

// —— 1. 缩放点积注意力（域4 可微教学版，2D [n,d]） ——
const out = exec('ad.scaledDotProductAttention', [Q, K, V, null]);
console.log('示例 05 · 注意力机制');
console.log('====================');
console.log(`1) 缩放点积注意力(教学版)：Q/K/V ∈ R^{${T}×${C}} → 输出 R^{${T}×${C}}，有限值 ${Array.from(out.data).every(Number.isFinite) ? '✓' : '✗'}`);

// —— 2. 因果掩码 + 严格 3D 缩放点积 ——
const mask = exec('dl.attn.causalMask', [T]);
let causalOk = true;
for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
  const want = j <= i ? 1 : 0;
  if (mask.data[i * T + j] !== want) causalOk = false;
}
const masked = exec('dl.attn.scaledDot', [Q3, K3, V3, mask]);
// 行随机性：每行和为 1
let rowSumOk = true;
for (let i = 0; i < T; i++) {
  let s = 0;
  for (let j = 0; j < T; j++) s += masked.attn.data[i * T + j];
  if (Math.abs(s - 1) > 1e-9) rowSumOk = false;
}
console.log(`2) 因果掩码：下三角(含对角)=1、严格上三角=0  ${causalOk ? '✓' : '✗'}；注意力行和=1  ${rowSumOk ? '✓' : '✗'}；输出有限 ${Array.from(masked.out.data).every(Number.isFinite) ? '✓' : '✗'}`);

// —— 3. 多头自注意力（域8，3D 严格约定） ——
const mh = exec('dl.attn.selfAttention', [Q3, HEADS, mask]);
const mhShapeOk = mh.out.shape[0] === 1 && mh.out.shape[1] === T && mh.out.shape[2] === C;
console.log(`3) 多头自注意力(heads=${HEADS})：输出形状 [${mh.out.shape.join('×')}]  ${mhShapeOk ? '✓' : '✗'}`);

// —— 4. RoPE 相对位置不变性：⟨RoPE(x,m),RoPE(x,n)⟩ 只依赖 m−n ——
const D = 4;
const { cos, sin } = exec('dl.attn.rotaryEmbed', [D, 32, 10000]);
const xh = rand(1, D).reshape([1, 1, D]); // [B=1,T=1,D]
const R = (pos) => exec('dl.attn.rotaryApply', [xh, cos, sin, pos]); // 返回 Tensor [1,1,D]
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.data.length; i++) s += a.data[i] * b.data[i]; return s; };
const s1 = dot(R(3), R(5)), s2 = dot(R(7), R(9)); // 相对位移同为 −2
const ropeOk = Math.abs(s1 - s2) < 1e-9;
console.log(`4) RoPE 相对位置不变性：⟨x₃,x₅⟩=${s1.toFixed(9)}  ⟨x₇,x₉⟩=${s2.toFixed(9)}  差 ${Math.abs(s1 - s2).toExponential(3)}  ${ropeOk ? '✓' : '✗'}`);

// —— 5. KV 缓存：逐 token 解码 == 一次性全量因果注意力（单头 H=1 简化） ——
let cache = exec('dl.attn.kvCacheInit', [1, 1, T, C]); // k/v ∈ [1,1,0,C]
const stepwise = [];
for (let t = 0; t < T; t++) {
  const qt = Tensor.fromFlat(Array.from({ length: C }, (_, j) => Q.data[t * C + j]), [1, 1, C]);
  const kNew = Tensor.fromFlat(Array.from({ length: C }, (_, j) => K.data[t * C + j]), [1, 1, 1, C]);
  const vNew = Tensor.fromFlat(Array.from({ length: C }, (_, j) => V.data[t * C + j]), [1, 1, 1, C]);
  const r = exec('dl.attn.kvCacheAppend', [cache, kNew, vNew]);
  cache = r; // 返回新缓存（纯函数）
  const Kmat = cache.k.reshape([1, cache.k.shape[2], C]);
  const Vmat = cache.v.reshape([1, cache.v.shape[2], C]);
  const o = exec('dl.attn.scaledDot', [qt, Kmat, Vmat]); // 无掩码：缓存按构造已因果
  stepwise.push(Array.from(o.out.data));
}
const full = exec('dl.attn.scaledDot', [Q3, K3, V3, mask]);
let cacheEq = true;
for (let t = 0; t < T; t++) {
  for (let j = 0; j < C; j++) {
    if (Math.abs(stepwise[t][j] - full.out.data[t * C + j]) > 1e-9) { cacheEq = false; break; }
  }
  if (!cacheEq) break;
}
console.log(`5) KV 缓存：逐 token 解码与全量因果注意力逐行一致  ${cacheEq ? '✓' : '✗'}`);

console.log('\n结论：缩放点积注意力在严格 3D 约定下正确实现因果掩码、多头拆分与');
console.log('      RoPE 相对位置编码；KV 缓存的增量解码与全序列计算数值等价。');
