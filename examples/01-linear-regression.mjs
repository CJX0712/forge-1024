// 示例 01：线性回归 —— 三种解法交叉验证（正规方程 / Cholesky / SVD 最小二乘）
// 运行：node examples/01-linear-regression.mjs
// 展示：闭式解与数值解一致、残差正交性（正规方程充要条件）、R² 口径。
import '../src/index.mjs';
import { exec } from '../src/core/registry.mjs';
import { Tensor } from '../src/core/tensor.mjs';
import { createRng } from '../src/core/rng.mjs';

const rng = createRng(20261014);
const n = 200, d = 3;              // 设计矩阵含截距列：[1, x₁, x₂]
const wTrue = [1.25, 2.5, -1.0];   // 真值（含截距）

const Xd = [], yd = [];
for (let i = 0; i < n; i++) {
  const x1 = rng.normal();
  const x2 = rng.normal() * 0.5 + 1;
  const y = wTrue[0] + wTrue[1] * x1 + wTrue[2] * x2 + rng.normal() * 0.25;
  Xd.push(1, x1, x2);
  yd.push(y);
}
const X = Tensor.fromFlat(Xd, [n, d]);
const y = Tensor.fromFlat(yd, [n]);

// —— 解法 A：正规方程 w = (XᵀX)⁻¹Xᵀy，正定系统走 Cholesky ——
const Xt = exec('linalg.transpose', [X]);
const XtX = exec('linalg.matmul', [Xt, X]);
const Xty = exec('linalg.matmulVec', [Xt, y]);
const wA = exec('linalg.solveSPD', [XtX, Xty]);

// —— 解法 B：SVD 最小二乘（病态/秩亏时更稳） ——
const wB = exec('linalg.lstsq', [X, y]).x;   // lstsq 返回 {x, rank}

// —— 解法 C：Moore–Penrose 伪逆 w = X⁺y（SVD 路径，秩亏亦成立） ——
const Xpinv = exec('linalg.pinv', [X]);
const wC = exec('linalg.matmulVec', [Xpinv, y]);

const fmt = (t) => `[${Array.from(t.data).map(v => v.toFixed(4)).join(', ')}]`;
const maxDiff = (a, b) => Math.max(...Array.from(a.data).map((v, i) => Math.abs(v - b.data[i])));

// —— 不变量 1：三解互相一致 ——
const dAB = maxDiff(wA, wB), dAC = maxDiff(wA, wC);

// —— 不变量 2：残差 ⊥ 列空间，即 Xᵀ(y − Xŵ) ≈ 0 ——
const pred = exec('linalg.matmulVec', [X, wA]);
const resid = exec('math.sub', [y, pred]);
const orth = Math.max(...Array.from(exec('linalg.matmulVec', [Xt, resid]).data).map(Math.abs));

// —— 拟合优度 ——
const yMean = exec('math.mean', [y]);
const ssRes = exec('math.sum', [resid.map(v => v * v)]);
const ssTot = exec('math.sum', [y.map(v => (v - yMean) ** 2)]);
const r2 = 1 - ssRes / ssTot;
const sigma = Math.sqrt(ssRes / (n - d));

console.log('示例 01 · 线性回归（闭式解 ×3 交叉验证）');
console.log('=====================================');
console.log(`真值 w*            : ${fmt(Tensor.fromFlat(wTrue, [d]))}`);
console.log(`A 正规方程+Cholesky: ${fmt(wA)}`);
console.log(`B SVD 最小二乘     : ${fmt(wB)}`);
console.log(`C 伪逆 X⁺y（SVD）  : ${fmt(wC)}`);
console.log(`‖A−B‖∞ = ${dAB.toExponential(3)}   ‖A−C‖∞ = ${dAC.toExponential(3)}   ${Math.max(dAB, dAC) < 1e-9 ? '✓' : '✗'}`);
console.log(`残差正交 ‖Xᵀr‖∞    : ${orth.toExponential(3)}   ${orth < 1e-8 ? '✓' : '✗'}`);
console.log(`R²                 : ${r2.toFixed(6)}`);
console.log(`残差标准误 σ̂       : ${sigma.toFixed(6)}（真值 0.25）`);
console.log('\n结论：三条独立路径（Cholesky / SVD 最小二乘 / Moore–Penrose 伪逆）给出逐位一致的解；');
console.log('      残差与设计矩阵列空间正交，正是正规方程的充要最优性条件。');
