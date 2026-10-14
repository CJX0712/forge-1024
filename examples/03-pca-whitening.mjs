// 示例 03：PCA 与白化 —— 特征分解路径 vs SVD 路径交叉验证
// 运行：node examples/03-pca-whitening.mjs
// 展示：Eckart–Young 最优低秩逼近、主成分互不相关、白化后协方差 = I。
import '../src/index.mjs';
import { exec } from '../src/core/registry.mjs';
import { Tensor } from '../src/core/tensor.mjs';
import { createRng } from '../src/core/rng.mjs';

const rng = createRng(42);
const n = 400, d = 5;

// —— 造相关数据：x = A·z，z 独立标准正态 ——
const A = [[2, 0, 0, 0, 0], [1.5, 1.2, 0, 0, 0], [0.8, 0.9, 0.6, 0, 0], [0.3, 0.4, 0.5, 0.35, 0], [0.2, 0.2, 0.2, 0.2, 0.15]];
const rows = [];
for (let i = 0; i < n; i++) {
  const z = [0, 1, 2, 3, 4].map(() => rng.normal());
  const x = A.map(row => row.reduce((s, a, j) => s + a * z[j], 0));
  rows.push(...x);
}
const X = Tensor.fromFlat(rows, [n, d]);

// —— 中心化 ——
const means = [];
for (let j = 0; j < d; j++) {
  let s = 0;
  for (let i = 0; i < n; i++) s += X.data[i * d + j];
  means.push(s / n);
}
const Xc = X.map((v, idx) => v - means[idx % d]);

// —— 协方差矩阵（对称半正定） ——
const XcT = exec('linalg.transpose', [Xc]);
const C = exec('linalg.matmul', [XcT, Xc]).map(v => v / (n - 1));
const symErr = exec('math.max', [exec('math.sub', [C, exec('linalg.transpose', [C])]).map(Math.abs)]);

// —— 路径 A：对称 Jacobi 特征分解 ——
const { values: evA, vectors: VA } = exec('linalg.eigenSym', [C]);

// —— 路径 B：SVD（Eckart–Young） ——
const { U, S, V } = exec('linalg.svd', [Xc.map(v => v / Math.sqrt(n - 1))]);
const evB = Array.from(S.data).map(s => s * s);

// —— 不变量 1：两条路径的特征值一致（降序） ——
const evAd = Array.from(evA.data).sort((a, b) => b - a);
const evBd = evB.sort((a, b) => b - a);
const evDiff = Math.max(...evAd.map((v, i) => Math.abs(v - evBd[i])));

// —— 不变量 2：投影后各主成分互不相关（协方差非对角元 ≈ 0） ——
const P = exec('linalg.matmul', [Xc, VA]);
const Pt = exec('linalg.transpose', [P]);
const CP = exec('linalg.matmul', [Pt, P]).map(v => v / (n - 1));
let offDiag = 0;
for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) if (i !== j) offDiag = Math.max(offDiag, Math.abs(CP.data[i * d + j]));

// —— 不变量 3：特征值之和 = 总方差 ——
const totalVar = evAd.reduce((a, b) => a + b, 0);
const traceC = exec('linalg.trace', [C]);

// —— 不变量 4：白化后协方差 = I ——
// whiten 返回 {W, mean, Xw}；eps=0 表示不加岭项（本例 λ_min≈0.017，无需正则）
const { Xw: Z } = exec('linalg.whiten', [Xc, 0]);
const CZ = exec('linalg.matmul', [exec('linalg.transpose', [Z]), Z]).map(v => v / (n - 1));
let identErr = 0;
for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) {
  identErr = Math.max(identErr, Math.abs(CZ.data[i * d + j] - (i === j ? 1 : 0)));
}

// —— 低秩逼近误差（Eckart–Young：截断 SVD 是最优低秩逼近） ——
const k = 2;
const approxErr = evAd.slice(k).reduce((a, b) => a + b, 0); // 理论残差平方和

console.log('示例 03 · PCA 与白化');
console.log('===================');
console.log(`协方差对称性误差 ‖C−Cᵀ‖∞ : ${symErr.toExponential(3)}   ${symErr < 1e-14 ? '✓' : '✗'}`);
console.log(`特征值（Jacobi） : [${evAd.map(v => v.toFixed(4)).join(', ')}]`);
console.log(`特征值（SVD）    : [${evBd.map(v => v.toFixed(4)).join(', ')}]`);
console.log(`两路径最大差异   : ${evDiff.toExponential(3)}   ${evDiff < 1e-9 ? '✓' : '✗'}`);
console.log(`主成分协方差非对角元最大绝对值 : ${offDiag.toExponential(3)}   ${offDiag < 1e-10 ? '✓' : '✗'}`);
console.log(`Σλᵢ = ${totalVar.toFixed(6)}  tr(C) = ${traceC.toFixed(6)}   差 ${Math.abs(totalVar - traceC).toExponential(3)}   ${Math.abs(totalVar - traceC) < 1e-9 ? '✓' : '✗'}`);
console.log(`白化后 ‖C_Z − I‖∞ : ${identErr.toExponential(3)}   ${identErr < 1e-8 ? '✓' : '✗'}`);
console.log(`秩−${k} 逼近理论残差（Σ_{i>${k}} λᵢ）: ${approxErr.toFixed(6)}`);
console.log(`前 ${k} 主成分解释方差比 : ${(evAd.slice(0, k).reduce((a, b) => a + b, 0) / totalVar * 100).toFixed(2)}%`);
console.log('\n结论：Jacobi 特征分解与 SVD 两条独立路径给出一致谱；投影后主成分解相关；');
console.log('      白化使协方差精确变为单位阵——PCA 的三条核心不变量全部成立。');
