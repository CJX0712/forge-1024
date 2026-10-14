// 域3 线性代数 LINALG（102 条微指令）
// 分解族：LU（Doolittle+部分选主元）/ QR（Householder）/ Cholesky /
//          特征分解（循环 Jacobi + Wilkinson 位移 QR 迭代）/ SVD（单边 Jacobi）
// 求解器：LU/Cholesky/三角/Toeplitz(Levinson)/迭代（Jacobi,GS,SOR,CG）
// 参考：Golub & Van Loan《Matrix Computations》4th ed.；
//       LAPACK（dgetrf/dgeqrf/dpotrf/dgesvd 算法同源）；Demmel《Applied Numerical Linear Algebra》。
import { defineOp, opsByDomain, exec } from '../core/registry.mjs';
import { ForgeError } from '../core/errors.mjs';
import { Tensor, prod as coreProd, stridesOf, expectTensor } from '../core/tensor.mjs';
import { createRng } from '../core/rng.mjs';

const D = 3;
let _s = 0;
const nid = () => `D${D}-${String(++_s).padStart(3, '0')}`;

function mat2d(t, who) {
  expectTensor(t, who);
  if (t.ndim !== 2) throw new ForgeError('E_SHAPE', `${who}: 期望 2D 矩阵，得到 shape [${t.shape}]`);
  return t;
}

// ============ 矩阵乘法族（6 条） ============
// 分块 i-k-j 循环（缓存友好，零元素跳过）
function matmulRaw(A, B, m, k, n) {
  const out = new Float64Array(m * n);
  const BM = 48;
  for (let i0 = 0; i0 < m; i0 += BM) {
    const iMax = Math.min(i0 + BM, m);
    for (let k0 = 0; k0 < k; k0 += BM) {
      const kMax = Math.min(k0 + BM, k);
      for (let j0 = 0; j0 < n; j0 += BM) {
        const jMax = Math.min(j0 + BM, n);
        for (let i = i0; i < iMax; i++) {
          for (let kk = k0; kk < kMax; kk++) {
            const a = A[i * k + kk];
            if (a === 0) continue;
            const rowOff = i * n;
            const bOff = kk * n;
            for (let j = j0; j < jMax; j++) out[rowOff + j] += a * B[bOff + j];
          }
        }
      }
    }
  }
  return out;
}

defineOp({
  id: nid(), domain: D, name: 'linalg.matmul', arity: 2,
  sig: '(Tensor A[m,k], Tensor B[k,n]) -> Tensor C[m,n]', doc: '矩阵乘法（分块 i-k-j，缓存友好）',
  ref: 'Golub & Van Loan §1.1；BLAS dgemm；Goto & van de Geijn 分块思想',
  inv: 'C[i,j]=ΣₖA[i,k]B[k,j]；matmul(A,I)=A',
  impl: (A, B) => {
    A = mat2d(A, 'A'); B = mat2d(B, 'B');
    const [m, k] = A.shape, [k2, n] = B.shape;
    if (k !== k2) throw new ForgeError('E_SHAPE', `matmul 内维不符: [${A.shape}]×[${B.shape}]`);
    return new Tensor(matmulRaw(A.data, B.data, m, k, n), [m, n]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.matmulVec', arity: 2,
  sig: '(Tensor A[m,k], Tensor x[k]) -> Tensor y[m]', doc: '矩阵-向量乘法',
  ref: 'BLAS dgemv', inv: 'y[i]=ΣₖA[i,k]x[k]',
  impl: (A, x) => {
    A = mat2d(A, 'A');
    const [m, k] = A.shape;
    if (x.ndim !== 1 || x.shape[0] !== k) throw new ForgeError('E_SHAPE', 'matmulVec 向量长度不符');
    const out = new Float64Array(m);
    for (let i = 0; i < m; i++) { let s = 0; for (let j = 0; j < k; j++) s += A.data[i * k + j] * x.data[j]; out[i] = s; }
    return new Tensor(out, [m]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.matmulAT', arity: 2,
  sig: '(Tensor A[k,m], Tensor B[k,n]) -> Tensor C[m,n]', doc: 'Aᵀ·B',
  ref: 'BLAS dgemm(transA)', inv: 'matmulAT(A,B)=matmul(A.T,B)',
  impl: (A, B) => exec('linalg.matmul', [A.T(), B]),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.matmulBT', arity: 2,
  sig: '(Tensor A[m,k], Tensor B[n,k]) -> Tensor C[m,n]', doc: 'A·Bᵀ',
  ref: 'BLAS dgemm(transB)', inv: 'matmulBT(A,B)=matmul(A,B.T)',
  impl: (A, B) => exec('linalg.matmul', [A, B.T()]),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.matmulABT', arity: 2,
  sig: '(Tensor A[k,m], Tensor B[n,k]) -> Tensor C[m,n]', doc: 'Aᵀ·Bᵀ',
  ref: 'BLAS dgemm(transA,transB)', inv: 'matmulABT(A,B)=matmul(A.T,B.T)',
  impl: (A, B) => exec('linalg.matmul', [A.T(), B.T()]),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.gemm', arity: 5,
  sig: '(number alpha, Tensor A, Tensor B, number beta, Tensor C) -> Tensor', doc: 'BLAS 级 GEMM：α·A·B+β·C',
  ref: 'BLAS dgemm', inv: 'gemm(1,A,B,0,C)=matmul(A,B)；C 缺省形状时忽略 beta',
  impl: (alpha, A, B, beta, C) => {
    A = mat2d(A, 'A'); B = mat2d(B, 'B');
    const [m, k] = A.shape, [k2, n] = B.shape;
    if (k !== k2) throw new ForgeError('E_SHAPE', 'gemm 内维不符');
    const out = matmulRaw(A.data, B.data, m, k, n);
    if (C !== undefined && C !== null) {
      C = mat2d(C, 'C');
      for (let i = 0; i < out.length; i++) out[i] = alpha * out[i] + beta * C.data[i];
    } else {
      for (let i = 0; i < out.length; i++) out[i] *= alpha;
    }
    return new Tensor(out, [m, n]);
  },
});

// ============ 分解族（11 条） ============
// LU：Doolittle + 部分选主元。返回 {L,U,P,pivots,sign}
function luFactor(A) {
  A = mat2d(A, 'A');
  const n = A.shape[0];
  if (A.shape[1] !== n) throw new ForgeError('E_SHAPE', 'LU 需要方阵');
  const a = A.data.slice();
  const pivots = new Array(n);
  let sign = 1;
  for (let k = 0; k < n; k++) {
    let p = k, max = Math.abs(a[k * n + k]);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(a[i * n + k]);
      if (v > max) { max = v; p = i; }
    }
    pivots[k] = p;
    if (max === 0) continue; // 奇异：该列全零
    if (p !== k) {
      for (let j = 0; j < n; j++) { const t = a[k * n + j]; a[k * n + j] = a[p * n + j]; a[p * n + j] = t; }
      sign = -sign;
    }
    const akk = a[k * n + k];
    for (let i = k + 1; i < n; i++) {
      const f = a[i * n + k] / akk;
      a[i * n + k] = f;
      if (f !== 0) for (let j = k + 1; j < n; j++) a[i * n + j] -= f * a[k * n + j];
    }
  }
  const L = Tensor.zeros([n, n]), U = Tensor.zeros([n, n]);
  for (let i = 0; i < n; i++) {
    L.set2(i, i, 1);
    for (let j = 0; j < i; j++) L.set2(i, j, a[i * n + j]);
    for (let j = i; j < n; j++) U.set2(i, j, a[i * n + j]);
  }
  return { L, U, pivots, sign, n };
}

defineOp({
  id: nid(), domain: D, name: 'linalg.lu', arity: 1, sig: '(Tensor A) -> object',
  doc: 'LU 分解（Doolittle + 部分选主元）', ref: 'LAPACK dgetrf；Golub & Van Loan §3.2',
  inv: 'P·A=L·U；L 单位下三角；U 上三角；|det|=ΠUᵢᵢ',
  impl: (A) => luFactor(A),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.luNoPivot', arity: 1, sig: '(Tensor A) -> object',
  doc: 'LU 分解（无选主元，对角占稳矩阵用）', ref: 'Golub & Van Loan §3.2',
  inv: 'A=L·U（无置换）',
  impl: (A) => {
    A = mat2d(A, 'A');
    const n = A.shape[0];
    const a = A.data.slice();
    for (let k = 0; k < n; k++) {
      const akk = a[k * n + k];
      if (akk === 0) throw new ForgeError('E_NUMERICAL', 'luNoPivot 遇零主元');
      for (let i = k + 1; i < n; i++) {
        const f = a[i * n + k] / akk;
        a[i * n + k] = f;
        if (f !== 0) for (let j = k + 1; j < n; j++) a[i * n + j] -= f * a[k * n + j];
      }
    }
    const L = Tensor.zeros([n, n]), U = Tensor.zeros([n, n]);
    for (let i = 0; i < n; i++) {
      L.set2(i, i, 1);
      for (let j = 0; j < i; j++) L.set2(i, j, a[i * n + j]);
      for (let j = i; j < n; j++) U.set2(i, j, a[i * n + j]);
    }
    return { L, U, pivots: Array.from({ length: n }, (_, i) => i), sign: 1, n };
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.forwardSubst', arity: 2,
  sig: '(Tensor L[n,n], Tensor b[n]) -> Tensor x[n]', doc: '前向替换（下三角系统 Lx=b）',
  ref: 'Golub & Van Loan §3.1', inv: '解单位下三角方程组 L·x = b（前代 forward substitution）',
  impl: (L, b) => {
    L = mat2d(L, 'L');
    const n = L.shape[0];
    if (b.ndim !== 1 || b.shape[0] !== n) throw new ForgeError('E_SHAPE', 'forwardSubst b 长度不符');
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = b.data[i];
      for (let j = 0; j < i; j++) s -= L.data[i * n + j] * x[j];
      x[i] = s / L.data[i * n + i];
    }
    return new Tensor(x, [n]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.backSubst', arity: 2,
  sig: '(Tensor U[n,n], Tensor b[n]) -> Tensor x[n]', doc: '后向替换（上三角系统 Ux=b）',
  ref: 'Golub & Van Loan §3.1', inv: '解单位上三角方程组 U·x = b（回代 back substitution）',
  impl: (U, b) => {
    U = mat2d(U, 'U');
    const n = U.shape[0];
    if (b.ndim !== 1 || b.shape[0] !== n) throw new ForgeError('E_SHAPE', 'backSubst b 长度不符');
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = b.data[i];
      for (let j = i + 1; j < n; j++) s -= U.data[i * n + j] * x[j];
      x[i] = s / U.data[i * n + i];
    }
    return new Tensor(x, [n]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.qr', arity: 1, sig: '(Tensor A[m,n]) -> object {Q,R}',
  doc: 'QR 分解（Householder 反射）', ref: 'LAPACK dgeqrf；Golub & Van Loan §5.2',
  inv: 'QᵀQ=I；Q·R=A；R 上三角',
  impl: (A) => {
    A = mat2d(A, 'A');
    const [m, n] = A.shape;
    const R = A.clone();
    const Q = Tensor.eye(m);
    const vs = [];
    for (let k = 0; k < Math.min(m - 1, n); k++) {
      // x = R[k:m, k]
      let norm = 0;
      for (let i = k; i < m; i++) norm += R.data[i * n + k] ** 2;
      norm = Math.sqrt(norm);
      if (norm === 0) { vs.push(null); continue; }
      const alpha = R.data[k * n + k] > 0 ? -norm : norm;
      const v = new Float64Array(m - k);
      v[0] = R.data[k * n + k] - alpha;
      for (let i = k + 1; i < m; i++) v[i - k] = R.data[i * n + k];
      let vnorm2 = 0;
      for (const vv of v) vnorm2 += vv * vv;
      if (vnorm2 === 0) { vs.push(null); continue; }
      vs.push({ v, vnorm2 });
      // R ← (I - 2vvᵀ/vᵀv)R（作用于列 k:n）
      for (let j = k; j < n; j++) {
        let dot = 0;
        for (let i = k; i < m; i++) dot += v[i - k] * R.data[i * n + j];
        const f = 2 * dot / vnorm2;
        for (let i = k; i < m; i++) R.data[i * n + j] -= f * v[i - k];
      }
      // Q ← Q·(I - 2vvᵀ/vᵀv)（作用于行 k:m）
      for (let i = 0; i < m; i++) {
        let dot = 0;
        for (let j = k; j < m; j++) dot += Q.data[i * m + j] * v[j - k];
        const f = 2 * dot / vnorm2;
        for (let j = k; j < m; j++) Q.data[i * m + j] -= f * v[j - k];
      }
    }
    // 清理 R 下三角数值噪声
    for (let i = 0; i < m; i++) for (let j = 0; j < Math.min(i, n); j++) R.data[i * n + j] = 0;
    return { Q, R };
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.cholesky', arity: 1, sig: '(Tensor A[n,n]) -> Tensor L[n,n]',
  doc: 'Cholesky 分解 A=L·Lᵀ（SPD 验证）', ref: 'LAPACK dpotrf；Golub & Van Loan §4.2',
  inv: 'L·Lᵀ=A；L 下三角且 Lᵢᵢ>0；非 SPD 抛 E_NUMERICAL',
  impl: (A) => {
    A = mat2d(A, 'A');
    const n = A.shape[0];
    const L = Tensor.zeros([n, n]);
    for (let j = 0; j < n; j++) {
      let d = A.get2(j, j);
      for (let k = 0; k < j; k++) d -= L.get2(j, k) ** 2;
      if (d <= 0) throw new ForgeError('E_NUMERICAL', `cholesky: 主元 ${d} ≤ 0，矩阵非 SPD`);
      const ljj = Math.sqrt(d);
      L.set2(j, j, ljj);
      for (let i = j + 1; i < n; i++) {
        let s = A.get2(i, j);
        for (let k = 0; k < j; k++) s -= L.get2(i, k) * L.get2(j, k);
        L.set2(i, j, s / ljj);
      }
    }
    return L;
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.det', arity: 1, sig: '(Tensor A[n,n]) -> number',
  doc: '行列式（LU + 选主元符号）', ref: 'LAPACK dgetrf 后取 ΠUᵢᵢ',
  inv: 'det(AB)=det(A)det(B)；det(Aᵀ)=det(A)；det(I)=1',
  impl: (A) => {
    const { U, sign } = luFactor(A);
    let d = sign;
    for (let i = 0; i < U.shape[0]; i++) d *= U.get2(i, i);
    return d;
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.logdet', arity: 1, sig: '(Tensor A[n,n]) -> number',
  doc: '对数行列式（Cholesky，2ΣlnLᵢᵢ，防溢出）', ref: '统计学标准（多元正态归一化）',
  inv: 'exp(logdet)=det；SPD 时为实数',
  impl: (A) => {
    const L = exec('linalg.cholesky', [A]);
    let s = 0;
    for (let i = 0; i < L.shape[0]; i++) s += Math.log(L.get2(i, i));
    return 2 * s;
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.inverse', arity: 1, sig: '(Tensor A[n,n]) -> Tensor',
  doc: '矩阵逆（Gauss-Jordan + 部分选主元）', ref: 'Golub & Van Loan §2.3；LAPACK dgetri',
  inv: 'A·A⁻¹=I；奇异矩阵抛 E_NUMERICAL',
  impl: (A) => {
    A = mat2d(A, 'A');
    const n = A.shape[0];
    const a = A.data.slice();
    const inv = Tensor.eye(n).data;
    for (let k = 0; k < n; k++) {
      let p = k, max = Math.abs(a[k * n + k]);
      for (let i = k + 1; i < n; i++) { const v = Math.abs(a[i * n + k]); if (v > max) { max = v; p = i; } }
      if (max < 1e-300) throw new ForgeError('E_NUMERICAL', 'inverse: 矩阵奇异');
      if (p !== k) {
        for (let j = 0; j < n; j++) { const t = a[k * n + j]; a[k * n + j] = a[p * n + j]; a[p * n + j] = t; }
        for (let j = 0; j < n; j++) { const t = inv[k * n + j]; inv[k * n + j] = inv[p * n + j]; inv[p * n + j] = t; }
      }
      const akk = a[k * n + k];
      for (let j = 0; j < n; j++) { a[k * n + j] /= akk; inv[k * n + j] /= akk; }
      for (let i = 0; i < n; i++) {
        if (i === k) continue;
        const f = a[i * n + k];
        if (f === 0) continue;
        for (let j = 0; j < n; j++) { a[i * n + j] -= f * a[k * n + j]; inv[i * n + j] -= f * inv[k * n + j]; }
      }
    }
    return new Tensor(inv, [n, n]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.solve', arity: 2, sig: '(Tensor A[n,n], Tensor b[n]) -> Tensor x[n]',
  doc: '线性系统求解（LU）', ref: 'LAPACK dgesv；Golub & Van Loan §3.2',
  inv: 'A·x=b；残差 ‖Ax−b‖/‖b‖ < 1e-10',
  impl: (A, b) => {
    const { L, U, pivots } = luFactor(A);
    const n = L.shape[0];
    if (b.ndim !== 1 || b.shape[0] !== n) throw new ForgeError('E_SHAPE', 'solve b 长度不符');
    // 应用置换到 b
    const pb = b.data.slice();
    for (let k = 0; k < n; k++) { const p = pivots[k]; if (p !== k) { const t = pb[k]; pb[k] = pb[p]; pb[p] = t; } }
    const y = exec('linalg.forwardSubst', [L, new Tensor(pb, [n])]);
    return exec('linalg.backSubst', [U, y]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.solveSPD', arity: 2, sig: '(Tensor A[n,n], Tensor b[n]) -> Tensor x[n]',
  doc: 'SPD 系统求解（Cholesky）', ref: 'LAPACK dposv', inv: '对称正定 A·x = b 的唯一解（Cholesky 分解）',
  impl: (A, b) => {
    const L = exec('linalg.cholesky', [A]);
    const y = exec('linalg.forwardSubst', [L, b]);
    return exec('linalg.backSubst', [L.T(), y]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.solveT', arity: 2, sig: '(Tensor A[n,n], Tensor b[n]) -> Tensor x[n]',
  doc: '转置系统求解 Aᵀx=b', ref: 'LAPACK 变体', inv: 'Aᵀ·x=b',
  impl: (A, b) => exec('linalg.solve', [A.T(), b]),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.solveMultiple', arity: 2, sig: '(Tensor A[n,n], Tensor B[n,k]) -> Tensor X[n,k]',
  doc: '多右端系统求解 A·X=B', ref: 'LAPACK dgesv 多列', inv: '同时解 A·X = B 的多列右端项',
  impl: (A, B) => {
    A = mat2d(A, 'A'); B = mat2d(B, 'B');
    const n = A.shape[0], k = B.shape[1];
    const out = new Float64Array(n * k);
    for (let j = 0; j < k; j++) {
      const x = exec('linalg.solve', [A, new Tensor(B.col(j), [n])]);
      for (let i = 0; i < n; i++) out[i * k + j] = x.data[i];
    }
    return new Tensor(out, [n, k]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.tridiagSolve', arity: 4,
  sig: '(Tensor sub[n], Tensor diag[n], Tensor sup[n], Tensor rhs[n]) -> Tensor x[n]',
  doc: 'Thomas 算法（次/主/超对角 + 右端，O(n)）', ref: 'Thomas 1949；Conte & de Boor',
  inv: '重构三对角矩阵验证 A·x=rhs',
  impl: (sub, diag, sup, rhs) => {
    const n = diag.size;
    if (sub.size !== n || sup.size !== n || rhs.size !== n) throw new ForgeError('E_SHAPE', 'tridiagSolve 长度不符');
    const cp = new Float64Array(n), dp = new Float64Array(n);
    cp[0] = sup.data[0] / diag.data[0];
    dp[0] = rhs.data[0] / diag.data[0];
    for (let i = 1; i < n; i++) {
      const denom = diag.data[i] - sub.data[i] * cp[i - 1];
      cp[i] = sup.data[i] / denom;
      dp[i] = (rhs.data[i] - sub.data[i] * dp[i - 1]) / denom;
    }
    const x = new Float64Array(n);
    x[n - 1] = dp[n - 1];
    for (let i = n - 2; i >= 0; i--) x[i] = dp[i] - cp[i] * x[i + 1];
    return new Tensor(x, [n]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.toeplitzSolve', arity: 2,
  sig: '(Tensor r[n+1], Tensor b[n]) -> Tensor x[n]', doc: 'Toeplitz 系统求解（显式构造 + LU）',
  ref: 'Toeplitz 系统标准；Golub & Van Loan §4.7',
  inv: 'T[i,j]=r[|i-j|]；T·x=b',
  impl: (r, b) => {
    const n = b.size;
    if (r.size < n + 1) throw new ForgeError('E_VALUE', 'toeplitzSolve 需要 r[0..n]');
    const T = new Tensor(new Float64Array(n * n), [n, n]);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) T.set2(i, j, r.data[Math.abs(i - j)]);
    return exec('linalg.solve', [T, b]);
  },
});

// ============ 特征分解族（3 条） ============
defineOp({
  id: nid(), domain: D, name: 'linalg.eigenSym', arity: 1, sig: '(Tensor A[n,n]) -> object {values, vectors}',
  doc: '对称矩阵特征分解（循环 Jacobi 旋转）', ref: 'Jacobi 1845；Golub & Van Loan §8.4；LAPACK dsyevj',
  inv: 'A·vᵢ=λᵢvᵢ（残差‖Av−λv‖<1e-10）；V 正交；values 降序',
  impl: (A) => jacobiEigen(A, 200, 1e-14),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.eigenSymQR', arity: 1, sig: '(Tensor A[n,n]) -> object {values}',
  doc: '对称矩阵特征值（Wilkinson 位移 QR 迭代）', ref: 'Wilkinson 1968；Numerical Recipes §11.3',
  inv: '与 Jacobi 结果一致（1e-8）；仅特征值',
  impl: (A) => {
    A = mat2d(A, 'A');
    const n = A.shape[0];
    let a = A.clone();
    for (let iter = 0; iter < 200 * n; iter++) {
      // 收敛检查：非对角模
      let off = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a.get2(i, j) ** 2;
      if (Math.sqrt(off) < 1e-14 * Math.max(1, frob(a))) break;
      // 找最大非对角作为 sub-block 起点（简化：全矩阵 QR 迭代 + Wilkinson 位移）
      const mu = a.get2(n - 1, n - 1);
      for (let i = 0; i < n; i++) a.set2(i, i, a.get2(i, i) - mu);
      const { Q, R } = exec('linalg.qr', [a]);
      a = exec('linalg.matmul', [R, Q]);
      for (let i = 0; i < n; i++) a.set2(i, i, a.get2(i, i) + mu);
    }
    const values = new Float64Array(n);
    for (let i = 0; i < n; i++) values[i] = a.get2(i, i);
    // 降序
    const order = Array.from(values).map((v, i) => [v, i]).sort((p, q) => q[0] - p[0]);
    return { values: new Tensor(Array.from(order.map(o => o[0])), [n]) };
  },
});
function frob(A) { let s = 0; for (const v of A.data) s += v * v; return Math.sqrt(s); }

defineOp({
  id: nid(), domain: D, name: 'linalg.eigen2x2', arity: 1, sig: '(Tensor A[2,2]) -> object {values, vectors}',
  doc: '2×2 对称矩阵解析特征分解（测试基准）', ref: '解析解：λ=(tr±√(tr²−4det))/2',
  inv: '解析解；与 Jacobi 结果一致',
  impl: (A) => {
    A = mat2d(A, 'A');
    const [[a, b], [c, d]] = [A.toArray()[0], A.toArray()[1]];
    const tr = a + d, det = a * d - b * c;
    const disc = Math.sqrt(Math.max(0, tr * tr - 4 * det));
    const l1 = (tr + disc) / 2, l2 = (tr - disc) / 2;
    // 特征向量
    function vecFor(l) {
      if (Math.abs(b) > 1e-300) return norm2([b, l - a]);
      if (Math.abs(c) > 1e-300) return norm2([l - d, c]);
      return [1, 0];
    }
    const v1 = vecFor(l1), v2 = [-v1[1], v1[0]];
    const hi = l1 >= l2 ? 0 : 1;
    const values = hi === 0 ? [l1, l2] : [l2, l1];
    const vectors = hi === 0 ? [v1, v2] : [v2, v1];
    // 列堆叠：矩阵第 i 列为 vectors[i]
    const flat = [vectors[0][0], vectors[1][0], vectors[0][1], vectors[1][1]];
    return { values: new Tensor(values, [2]), vectors: new Tensor(flat, [2, 2]) };
    function norm2(v) { const n = Math.hypot(v[0], v[1]) || 1; return [v[0] / n, v[1] / n]; }
  },
});

function jacobiEigen(A, maxSweeps = 200, tol = 1e-14) {
  A = mat2d(A, 'A');
  const n = A.shape[0];
  if (A.shape[1] !== n) throw new ForgeError('E_SHAPE', 'eigenSym 需要方阵');
  let a = A.clone();
  const V = Tensor.eye(n);
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a.get2(p, q) ** 2;
    if (Math.sqrt(off) < tol * Math.max(1, frob(a))) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a.get2(p, q);
        if (Math.abs(apq) < 1e-300) continue;
        const app = a.get2(p, p), aqq = a.get2(q, q);
        const tau = (aqq - app) / (2 * apq);
        const t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          if (k === p || k === q) continue;
          const akp = a.get2(k, p), akq = a.get2(k, q);
          a.set2(k, p, c * akp - s * akq);
          a.set2(p, k, a.get2(k, p));
          a.set2(k, q, s * akp + c * akq);
          a.set2(q, k, a.get2(k, q));
        }
        a.set2(p, p, c * c * app - 2 * s * c * apq + s * s * aqq);
        a.set2(q, q, s * s * app + 2 * s * c * apq + c * c * aqq);
        a.set2(p, q, 0); a.set2(q, p, 0);
        for (let k = 0; k < n; k++) {
          const vkp = V.get2(k, p), vkq = V.get2(k, q);
          V.set2(k, p, c * vkp - s * vkq);
          V.set2(k, q, s * vkp + c * vkq);
        }
      }
    }
  }
  // 收集并降序排列
  const pairs = [];
  for (let i = 0; i < n; i++) pairs.push([a.get2(i, i), i]);
  pairs.sort((x, y) => y[0] - x[0]);
  const values = new Float64Array(n);
  const vectors = new Float64Array(n * n);
  for (let k = 0; k < n; k++) {
    values[k] = pairs[k][0];
    const src = pairs[k][1];
    for (let i = 0; i < n; i++) vectors[i * n + k] = V.get2(i, src);
  }
  return { values: new Tensor(values, [n]), vectors: new Tensor(vectors, [n, n]) };
}

// ============ SVD 族（3 条） ============
defineOp({
  id: nid(), domain: D, name: 'linalg.svd', arity: 1, sig: '(Tensor A[m,n]) -> object {U,S,V}',
  doc: 'SVD（单边 Jacobi 正交化）', ref: 'Hestenes 1958 单边 Jacobi；LAPACK dgesvj；Golub & Van Loan §8.6',
  inv: 'U·diag(S)·Vᵀ=A；U(m×k)/V(n×k) 列正交；S 降序非负',
  impl: (A) => {
    A = mat2d(A, 'A');
    const [m, n] = A.shape;
    const wide = m < n;
    // 统一按"高或方"处理：宽矩阵转置后求再交换
    const src = wide ? A.T() : A;
    const { U, S, V } = svdTall(src.data, src.shape[0], src.shape[1]);
    if (!wide) return { U: new Tensor(U, [m, n]), S: new Tensor(S, [n]), V: new Tensor(V, [n, n]) };
    // 宽矩阵：A = srcᵀ = V_s·Σ·U_sᵀ → U_A=V_s(m×m), V_A=U_sᵀ(m×n)
    const VAt = new Float64Array(m * n);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) VAt[i * n + j] = U[j * m + i];
    return { U: new Tensor(V, [m, m]), S: new Tensor(S, [m]), V: new Tensor(VAt, [m, n]) };
  },
});

// 单边 Jacobi SVD（m >= n）。返回平铺 U(m×n), S(n), V(n×n)
function svdTall(data, m, n, maxSweeps = 60, tol = 1e-13) {
  const U = Float64Array.from(data);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let converged = true;
    for (let i = 0; i < n - 1; i++) {
      for (let j = i + 1; j < n; j++) {
        let alpha = 0, beta = 0, gamma = 0;
        for (let k = 0; k < m; k++) {
          alpha += U[k * n + i] ** 2;
          beta += U[k * n + j] ** 2;
          gamma += U[k * n + i] * U[k * n + j];
        }
        if (Math.abs(gamma) <= tol * Math.sqrt(alpha * beta) || alpha === 0 || beta === 0) continue;
        converged = false;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = c * t;
        for (let k = 0; k < m; k++) {
          const uki = U[k * n + i], ukj = U[k * n + j];
          U[k * n + i] = c * uki - s * ukj;
          U[k * n + j] = s * uki + c * ukj;
        }
        for (let k = 0; k < n; k++) {
          const vki = V[k * n + i], vkj = V[k * n + j];
          V[k * n + i] = c * vki - s * vkj;
          V[k * n + j] = s * vki + c * vkj;
        }
      }
    }
    if (converged) break;
  }
  // 提取奇异值并归一化 U 列
  const S = new Float64Array(n);
  const cols = [];
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let k = 0; k < m; k++) s += U[k * n + j] ** 2;
    s = Math.sqrt(s);
    S[j] = s;
    cols.push({ s, j });
  }
  cols.sort((a, b) => b.s - a.s);
  const Uout = new Float64Array(m * n);
  const Vout = new Float64Array(n * n);
  const Sout = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const { s, j } = cols[k];
    Sout[k] = s;
    if (s > 0) for (let i = 0; i < m; i++) Uout[i * n + k] = U[i * n + j] / s;
    for (let i = 0; i < n; i++) Vout[i * n + k] = V[i * n + j];
  }
  return { U: Uout, S: Sout, V: Vout };
}

defineOp({
  id: nid(), domain: D, name: 'linalg.pinv', arity: 1, sig: '(Tensor A, number rcond=1e-12) -> Tensor',
  doc: 'Moore-Penrose 伪逆（SVD 截断）', ref: 'Golub & Van Loan §5.5；LAPACK dgelss',
  inv: 'A·A⁺·A=A；A⁺·A·A⁺=A⁺；满秩时等于逆',
  impl: (A, rcond = 1e-12) => {
    const { U, S, V } = exec('linalg.svd', [A]);
    const m = A.shape[0], n = A.shape[1];
    const k = Math.min(m, n);
    const smax = S.data[0] || 0;
    const cut = rcond * smax;
    const out = new Float64Array(n * m);
    for (let i = 0; i < k; i++) {
      const inv = S.data[i] > cut ? 1 / S.data[i] : 0;
      // A⁺ = V·Σ⁺·Uᵀ
      for (let p = 0; p < n; p++) {
        const vpi = V.get2(p, i);
        if (vpi === 0) continue;
        for (let q = 0; q < m; q++) out[p * m + q] += vpi * inv * U.get2(q, i);
      }
    }
    return new Tensor(out, [n, m]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.rank', arity: 1, sig: '(Tensor A, number tol?) -> number',
  doc: '数值秩（SVD 容差 max(m,n)·eps·σ₁）', ref: 'Golub & Van Loan §5.5；LAPACK 秩判定',
  inv: 'rank ≤ min(m,n)；满秩方阵 rank=n',
  impl: (A, tol) => {
    const { S } = exec('linalg.svd', [A]);
    const smax = S.data[0] || 0;
    const t = tol ?? Math.max(A.shape[0], A.shape[1]) * Number.EPSILON * smax;
    let r = 0;
    for (let i = 0; i < S.size; i++) if (S.data[i] > t) r++;
    return r;
  },
});

// ============ 范数与条件数（4 条） ============
defineOp({
  id: nid(), domain: D, name: 'linalg.normFro', sig: '(Tensor A) -> number', doc: 'Frobenius 范数 √ΣAᵢⱼ²',
  ref: 'Golub & Van Loan §2.2', inv: 'normFro(Aᵀ)=normFro(A)',
  impl: (A) => { let s = 0; for (const v of A.data) s += v * v; return Math.sqrt(s); },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.norm1', sig: '(Tensor A) -> number', doc: '1-范数（最大列和）',
  ref: 'Golub & Van Loan §2.2', inv: 'norm1(A)≥normFro(A)/√n',
  impl: (A) => {
    const [m, n] = A.shape;
    let best = 0;
    for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < m; i++) s += Math.abs(A.get2(i, j)); best = Math.max(best, s); }
    return best;
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.normInf', sig: '(Tensor A) -> number', doc: '∞-范数（最大行和）',
  ref: 'Golub & Van Loan §2.2', inv: 'normInf(A)=norm1(Aᵀ)',
  impl: (A) => {
    const [m, n] = A.shape;
    let best = 0;
    for (let i = 0; i < m; i++) { let s = 0; for (let j = 0; j < n; j++) s += Math.abs(A.get2(i, j)); best = Math.max(best, s); }
    return best;
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.normMax', sig: '(Tensor A) -> number', doc: '最大元范数 max|Aᵢⱼ|',
  ref: 'Golub & Van Loan §2.2', inv: 'normMax ≤ normFro ≤ √(mn)·normMax',
  impl: (A) => { let m2 = 0; for (const v of A.data) m2 = Math.max(m2, Math.abs(v)); return m2; },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.norm2', sig: '(Tensor A) -> number', doc: '2-范数（谱范数 σmax，SVD）',
  ref: 'Golub & Van Loan §2.2', inv: 'norm2=σmax；norm2(A)=norm2(Aᵀ)',
  impl: (A) => { const { S } = exec('linalg.svd', [A]); return S.data[0] || 0; },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.normNuc', sig: '(Tensor A) -> number', doc: '核范数 Σσᵢ',
  ref: '凸优化/矩阵补全标准（Candès & Recht 2009）', inv: 'normNuc ≥ norm2',
  impl: (A) => { const { S } = exec('linalg.svd', [A]); let s = 0; for (const v of S.data) s += v; return s; },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.cond2', sig: '(Tensor A) -> number', doc: '2-范数条件数 σmax/σmin',
  ref: 'Golub & Van Loan §2.2.4', inv: 'cond(I)=1；cond≥1；大=病态',
  impl: (A) => {
    const { S } = exec('linalg.svd', [A]);
    const smin = S.data[S.size - 1];
    if (!(smin > 0)) return Infinity;
    return S.data[0] / smin;
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.cond1', sig: '(Tensor A) -> number', doc: '1-范数条件数 ‖A‖₁·‖A⁻¹‖₁',
  ref: 'Golub & Van Loan §2.2.4', inv: 'cond1(A)=cond1(Aᵀ)',
  impl: (A) => {
    const inv = exec('linalg.inverse', [A]);
    return exec('linalg.norm1', [A]) * exec('linalg.norm1', [inv]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.gershgorin', sig: '(Tensor A) -> object',
  doc: 'Gershgorin 圆盘（特征值包含域估计）', ref: 'Gershgorin 1931',
  inv: '全部特征值落在圆盘并集内',
  impl: (A) => {
    A = mat2d(A, 'A');
    const n = A.shape[0];
    const disks = [];
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let j = 0; j < n; j++) if (j !== i) r += Math.abs(A.get2(i, j));
      disks.push({ center: A.get2(i, i), radius: r });
    }
    return { disks, lo: Math.min(...disks.map(d => d.center - d.radius)), hi: Math.max(...disks.map(d => d.center + d.radius)) };
  },
});

// ============ 迭代求解器（5 条） ============
function splitIter(A, b, maxIter, tol) {
  const n = A.shape[0];
  const x = new Float64Array(n);
  for (let it = 0; it < maxIter; it++) {
    let diff = 0;
    for (let i = 0; i < n; i++) {
      let s = b.data[i];
      for (let j = 0; j < n; j++) if (j !== i) s -= A.get2(i, j) * x[j];
      const xi = s / A.get2(i, i);
      diff = Math.max(diff, Math.abs(xi - x[i]));
      x[i] = xi;
    }
    if (diff < tol) return { x: new Tensor(x, [n]), iters: it + 1, converged: true };
  }
  return { x: new Tensor(x, [n]), iters: maxIter, converged: false };
}
defineOp({
  id: nid(), domain: D, name: 'linalg.jacobiIter', arity: 2,
  sig: '(Tensor A, Tensor b, number maxIter=1000, number tol=1e-10) -> object',
  doc: 'Jacobi 迭代（对角占优系统）', ref: 'Jacobi 1845；Saad《Iterative Methods》§4.2',
  inv: '严格对角占优必收敛',
  impl: (A, b, maxIter = 1000, tol = 1e-10) => splitIter(A, b, maxIter, tol),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.gaussSeidel', arity: 2,
  sig: '(Tensor A, Tensor b, number maxIter=1000, number tol=1e-10) -> object',
  doc: 'Gauss-Seidel 迭代（用最新分量，通常快于 Jacobi）', ref: 'Saad《Iterative Methods》§4.3',
  inv: 'SPD 或对角占优必收敛',
  impl: (A, b, maxIter = 1000, tol = 1e-10) => {
    const n = A.shape[0];
    const x = new Float64Array(n);
    for (let it = 0; it < maxIter; it++) {
      let diff = 0;
      for (let i = 0; i < n; i++) {
        let s = b.data[i];
        for (let j = 0; j < n; j++) if (j !== i) s -= A.get2(i, j) * x[j];
        const xi = s / A.get2(i, i);
        diff = Math.max(diff, Math.abs(xi - x[i]));
        x[i] = xi;
      }
      if (diff < tol) return { x: new Tensor(x, [n]), iters: it + 1, converged: true };
    }
    return { x: new Tensor(x, [n]), iters: maxIter, converged: false };
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.sor', arity: 2,
  sig: '(Tensor A, Tensor b, number maxIter=1000, number tol=1e-10, number omega=1.5) -> object',
  doc: 'SOR 超松弛迭代（ω∈(0,2)，ω=1 即 Gauss-Seidel）', ref: 'SOR 经典方法；Saad §4.4',
  inv: '最优 ω 收敛最快',
  impl: (A, b, maxIter = 1000, tol = 1e-10, omega = 1.5) => {
    const n = A.shape[0];
    const x = new Float64Array(n);
    for (let it = 0; it < maxIter; it++) {
      let diff = 0;
      for (let i = 0; i < n; i++) {
        let s = b.data[i];
        for (let j = 0; j < n; j++) if (j !== i) s -= A.get2(i, j) * x[j];
        const xi = (1 - omega) * x[i] + omega * s / A.get2(i, i);
        diff = Math.max(diff, Math.abs(xi - x[i]));
        x[i] = xi;
      }
      if (diff < tol) return { x: new Tensor(x, [n]), iters: it + 1, converged: true };
    }
    return { x: new Tensor(x, [n]), iters: maxIter, converged: false };
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.conjGradient', arity: 2,
  sig: '(Tensor A, Tensor b, number maxIter?, number tol=1e-10) -> object',
  doc: '共轭梯度法（SPD 系统，Krylov 子空间）', ref: 'Hestenes & Stiefel 1952；Saad §6.7；LAPACK 之外的经典',
  inv: '精确算术下 ≤n 步收敛；残差单调不增',
  impl: (A, b, maxIter, tol = 1e-10) => {
    const n = A.shape[0];
    const iters = maxIter ?? n;
    let x = new Float64Array(n);
    let r = Float64Array.from(b.data);
    let p = Float64Array.from(b.data);
    let rsold = r.reduce((s, v) => s + v * v, 0);
    const bnorm = Math.sqrt(b.data.reduce((s, v) => s + v * v, 0)) || 1;
    for (let it = 0; it < iters; it++) {
      if (Math.sqrt(rsold) / bnorm < tol) return { x: new Tensor(x, [n]), iters: it, converged: true };
      const Ap = new Float64Array(n);
      for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += A.get2(i, j) * p[j]; Ap[i] = s; }
      const pAp = p.reduce((s, v, i) => s + v * Ap[i], 0);
      if (pAp === 0) break;
      const alpha = rsold / pAp;
      for (let i = 0; i < n; i++) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
      const rsnew = r.reduce((s, v) => s + v * v, 0);
      const beta = rsnew / rsold;
      for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
      rsold = rsnew;
    }
    return { x: new Tensor(x, [n]), iters, converged: Math.sqrt(rsold) / bnorm < tol };
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.powerIteration', arity: 1,
  sig: '(Tensor A, number maxIter=1000, number tol=1e-12) -> object',
  doc: '幂迭代（按模最大特征对；负主特征时向量符号振荡，返回值取模）',
  ref: '幂法经典；Golub & Van Loan §8.2',
  inv: '|value|=max|λ|；残差 ‖Av−λv‖→0',
  impl: (A, maxIter = 1000, tol = 1e-12) => {
    const n = A.shape[0];
    const rng = createRng(1);
    let v = new Tensor(Float64Array.from({ length: n }, () => rng.normal()), [n]);
    let lambda = 0;
    let oscillate = 0;
    for (let it = 0; it < maxIter; it++) {
      const Av = exec('linalg.matmulVec', [A, v]);
      let norm = 0;
      for (const x of Av.data) norm += x * x;
      norm = Math.sqrt(norm);
      if (norm === 0) break;
      const vNew = Av.map((x) => x / norm);
      const lam = exec('math.dot', [vNew, exec('linalg.matmulVec', [A, vNew])]);
      // 符号振荡检测（负主特征）：连续量级稳定但异号
      if (Math.abs(lam + lambda) < tol * Math.max(1, Math.abs(lam))) oscillate++;
      else oscillate = 0;
      if (oscillate >= 3) return { value: Math.abs(lam), vector: vNew, iters: it + 1 };
      if (Math.abs(lam - lambda) < tol) { lambda = lam; v = vNew; break; }
      lambda = lam; v = vNew;
    }
    return { value: lambda, vector: v, iters: maxIter };
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.inverseIteration', arity: 1,
  sig: '(Tensor A, number mu=0, number maxIter=100) -> object',
  doc: '反迭代（求靠近位移 μ 的特征对）', ref: 'Golub & Van Loan §8.3',
  inv: '收敛到最接近 μ 的特征值',
  impl: (A, mu = 0, maxIter = 100) => {
    const n = A.shape[0];
    const shifted = A.map((v, i) => (i % (n + 1) === 0 ? v - mu : v));
    const rng = createRng(2);
    let v = new Tensor(Float64Array.from({ length: n }, () => rng.normal()), [n]);
    let norm = Math.sqrt(v.data.reduce((s, x) => s + x * x, 0));
    v = v.map((x) => x / norm);
    let lambda = 0;
    for (let it = 0; it < maxIter; it++) {
      const w = exec('linalg.solve', [shifted, v]);
      let wn = 0;
      for (const x of w.data) wn += x * x;
      wn = Math.sqrt(wn);
      if (!Number.isFinite(wn) || wn === 0) break;
      const vNew = w.map((x) => x / wn);
      const lam = exec('math.dot', [vNew, exec('linalg.matmulVec', [A, vNew])]);
      if (Math.abs(lam - lambda) < 1e-13) { lambda = lam; v = vNew; break; }
      lambda = lam; v = vNew;
    }
    return { value: lambda, vector: v };
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.rayleighQuotient', arity: 1, sig: '(Tensor A, Tensor v) -> number',
  doc: 'Rayleigh 商 vᵀAv/vᵀv', ref: 'Golub & Van Loan §8.2',
  inv: 'v 为特征向量时等于特征值',
  impl: (A, v) => {
    const Av = exec('linalg.matmulVec', [A, v]);
    return exec('math.dot', [v, Av]) / exec('math.dot', [v, v]);
  },
});

// ============ 矩阵函数与结构（6 条） ============
defineOp({
  id: nid(), domain: D, name: 'linalg.matrixPower', arity: 2, sig: '(Tensor A, number k) -> Tensor',
  doc: '矩阵幂 Aᵏ（快速幂）', ref: 'Numerical Recipes §11.2',
  inv: 'A⁰=I；A¹=A；AᵏAᵐ=Aᵏ⁺ᵐ',
  impl: (A, k) => {
    A = mat2d(A, 'A');
    if (k < 0) { A = exec('linalg.inverse', [A]); k = -k; }
    let result = Tensor.eye(A.shape[0]);
    let base = A.clone();
    let e = k;
    while (e > 0) {
      if (e & 1) result = exec('linalg.matmul', [result, base]);
      base = exec('linalg.matmul', [base, base]);
      e = Math.floor(e / 2);
    }
    return result;
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.matrixExp', arity: 1, sig: '(Tensor A[n,n]) -> Tensor',
  doc: '矩阵指数 e^A（缩放平方 + Taylor 级数）',
  ref: 'Higham 2005《Functions of Matrices》§10.3（缩放平方思想）',
  inv: 'e⁰=I；e^{tA}e^{sA}=e^{(t+s)A}；d/dt e^{tA}=A·e^{tA}',
  impl: (A) => {
    A = mat2d(A, 'A');
    const n = A.shape[0];
    // 缩放：使 ‖A/2^s‖∞ ≤ 0.5，Taylor 18 项后误差 ~1e-17
    const normA = exec('linalg.normInf', [A]);
    const s = Math.max(0, Math.ceil(Math.log2(normA || 1)) + 1);
    const As = A.map((v) => v / 2 ** s);
    let term = Tensor.eye(n);
    let X = Tensor.eye(n);
    for (let k = 1; k <= 18; k++) {
      term = exec('linalg.matmul', [term, As]).map((v) => v / k);
      X = exec('math.add', [X, term]);
    }
    for (let i = 0; i < s; i++) X = exec('linalg.matmul', [X, X]);
    return X;
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.symmetrize', sig: '(Tensor A) -> Tensor', doc: '对称化 (A+Aᵀ)/2',
  ref: 'Golub & Van Loan §1.2 (symmetric/Hermitian part)', inv: 'symmetrize(A)ᵀ=symmetrize(A)',
  impl: (A) => exec('math.scalarMul', [0.5, exec('math.add', [A, A.T()])]),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.whiten', arity: 1, sig: '(Tensor X[n,d], number eps=1e-5) -> object',
  doc: 'ZCA 白化（协方差特征分解）', ref: 'ZCA 白化标准（Bell & Sejnowski 1996）',
  inv: '白化后协方差 ≈ I；whiten 矩阵对称',
  impl: (X, eps = 1e-5) => {
    const mean = exec('math.meanAxis', [X, 0]);
    const Xc = exec('math.sub', [X, mean]);
    const cov = exec('linalg.gemm', [1 / (X.shape[0] - 1), Xc.T(), Xc, 0, null]);
    const { values, vectors } = exec('linalg.eigenSym', [cov]);
    const Dinv = values.map((v) => 1 / Math.sqrt(v + eps));
    // W = V·diag(Dinv)·Vᵀ（ZCA）
    const diagD = Tensor.zeros([values.size, values.size]);
    for (let i = 0; i < values.size; i++) diagD.set2(i, i, Dinv.data[i]);
    const W = exec('linalg.matmul', [exec('linalg.matmul', [vectors, diagD]), vectors.T()]);
    return { W, mean, Xw: exec('linalg.matmul', [Xc, W.T()]) };
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.nearestSPD', arity: 1, sig: '(Tensor A) -> Tensor',
  doc: '最近对称正定矩阵（对称化 + 特征值截断修正）', ref: 'Higham 2002 思想（交替投影的谱截断变体）',
  inv: '结果 SPD；对称',
  impl: (A) => {
    let B = exec('linalg.symmetrize', [A]);
    const n = B.shape[0];
    const eps = 1e-12 * Math.max(1, exec('linalg.normInf', [B]));
    const { values, vectors } = exec('linalg.eigenSym', [B]);
    const T = Tensor.zeros([n, n]);
    for (let i = 0; i < n; i++) T.set2(i, i, Math.max(values.data[i], eps));
    B = exec('linalg.symmetrize', [exec('linalg.matmul', [exec('linalg.matmul', [vectors, T]), vectors.T()])]);
    return B;
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.quadForm', arity: 2, sig: '(Tensor x, Tensor A) -> number', doc: '二次型 xᵀAx',
  ref: '线性代数标准', inv: 'A 对称时 quadForm(x,A)=λ 加权和',
  impl: (x, A) => exec('math.dot', [x, exec('linalg.matmulVec', [A, x])]),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.matVecT', arity: 2, sig: '(Tensor x[m], Tensor A[m,n]) -> Tensor y[n]',
  doc: 'xᵀ·A（行向量乘矩阵）', ref: 'BLAS dgemv(trans)', inv: 'matVecT(x,A)=matmulVec(A.T,x)',
  impl: (x, A) => exec('linalg.matmulVec', [A.T(), x]),
});

// ============ 结构化矩阵与构造（8 条） ============
defineOp({
  id: nid(), domain: D, name: 'linalg.vander', arity: 1, sig: '(Tensor x, number n?) -> Tensor',
  doc: 'Vandermonde 矩阵（列幂次 n-1..0）', ref: '数值分析标准构造',
  inv: 'vander(x,n) 列 = x 的幂',
  impl: (x, n) => {
    const m = x.size;
    const nn = n ?? m;
    const out = new Float64Array(m * nn);
    for (let i = 0; i < m; i++) for (let j = 0; j < nn; j++) out[i * nn + j] = x.data[i] ** (nn - 1 - j);
    return new Tensor(out, [m, nn]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.polyfit', arity: 3, sig: '(Tensor x, Tensor y, number deg) -> Tensor',
  doc: '最小二乘多项式拟合（Vandermonde + lstsq）', ref: 'Numerical Recipes §3.3',
  inv: '残差最小；deg≥n-1 时精确插值',
  impl: (x, y, deg) => {
    const V = exec('linalg.vander', [x, deg + 1]);
    const sol = exec('linalg.lstsq', [V, y]);
    return sol.x;
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.polyval', arity: 2, sig: '(Tensor coefs, Tensor x) -> Tensor',
  doc: 'Horner 多项式求值（coefs 高次→低次）', ref: 'Numerical Recipes §3.3',
  inv: '对单点退化为 num.horner',
  impl: (coefs, x) => x.map((v) => {
    let s = 0;
    for (let i = 0; i < coefs.size; i++) s = s * v + coefs.data[i];
    return s;
  }),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.lstsq', arity: 2, sig: '(Tensor A, Tensor b, number rcond=1e-12) -> object',
  doc: '最小二乘解（QR 高效路径；秩亏回退 SVD）', ref: 'LAPACK dgels/dgelss；Golub & Van Loan §5.3',
  inv: '‖Ax−b‖ 最小；满秩列时 x=A⁺b',
  impl: (A, b, rcond = 1e-12) => {
    A = mat2d(A, 'A'); b = expectTensor(b, 'b');
    const r = exec('linalg.rank', [A, rcond]);
    const n = A.shape[1];
    if (r === n && A.shape[0] >= n) {
      const { Q, R } = exec('linalg.qr', [A]);
      const m = A.shape[0];
      const qtb = exec('linalg.matmulVec', [Q.T(), b]);
      const R1 = new Tensor(R.data.subarray(0, n * n), [n, n]);
      const x = exec('linalg.backSubst', [R1, new Tensor(qtb.data.subarray(0, n), [n])]);
      return { x, rank: r };
    }
    const Ap = exec('linalg.pinv', [A, rcond]);
    return { x: exec('linalg.matmulVec', [Ap, b]), rank: r };
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.nullSpace', arity: 1, sig: '(Tensor A, number tol?) -> Tensor',
  doc: '零空间基（SVD 右奇异向量）', ref: 'Golub & Van Loan §5.5',
  inv: 'A·N=0；N 列正交',
  impl: (A, tol) => {
    const { V, S } = exec('linalg.svd', [A]);
    const smax = S.data[0] || 0;
    const t = tol ?? Math.max(A.shape[0], A.shape[1]) * Number.EPSILON * smax;
    const cols = [];
    for (let i = 0; i < S.size; i++) if (S.data[i] <= t) cols.push(i);
    if (cols.length === 0) return new Tensor(new Float64Array(0), [V.shape[0], 0]);
    const out = new Float64Array(V.shape[0] * cols.length);
    for (let k = 0; k < cols.length; k++) for (let i = 0; i < V.shape[0]; i++) out[i * cols.length + k] = V.get2(i, cols[k]);
    return new Tensor(out, [V.shape[0], cols.length]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.kron', arity: 2, sig: '(Tensor A[m,n], Tensor B[p,q]) -> Tensor[mp,nq]',
  doc: 'Kronecker 积', ref: 'Horn & Johnson《Topics in Matrix Analysis》',
  inv: 'kron(A,I)=块对角扩展',
  impl: (A, B) => {
    A = mat2d(A, 'A'); B = mat2d(B, 'B');
    const [m, n] = A.shape, [p, q] = B.shape;
    const out = new Float64Array(m * p * n * q);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
      const a = A.get2(i, j);
      for (let r = 0; r < p; r++) for (let c = 0; c < q; c++) {
        out[(i * p + r) * (n * q) + (j * q + c)] = a * B.get2(r, c);
      }
    }
    return new Tensor(out, [m * p, n * q]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.vec', sig: '(Tensor A[m,n]) -> Tensor[mn]', doc: '列堆叠向量化 vec(A)[j·m+i]=A[i,j]',
  ref: 'Horn & Johnson', inv: 'vec(Iₙ) = [1,0,..,0,1,..]；unvec(vec(A))=A',
  impl: (A) => {
    A = mat2d(A, 'A');
    const [m, n] = A.shape;
    const out = new Float64Array(m * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < m; i++) out[j * m + i] = A.get2(i, j);
    return new Tensor(out, [m * n]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.unvec', arity: 2, sig: '(Tensor x[mn], number m) -> Tensor[m,n]',
  doc: '列堆叠反向量化', ref: 'Horn & Johnson', inv: 'unvec(vec(A))=A',
  impl: (x, m) => {
    const n = x.size / m;
    const out = new Float64Array(x.size);
    for (let j = 0; j < n; j++) for (let i = 0; i < m; i++) out[i * n + j] = x.data[j * m + i];
    return new Tensor(out, [m, n]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.householder', arity: 1, sig: '(Tensor x) -> object {v, beta}',
  doc: 'Householder 向量（反射 Hx=±‖x‖e₁）', ref: 'Golub & Van Loan §5.1',
  inv: 'H=I−βvvᵀ；Hx 与 x 模长相同',
  impl: (x) => {
    let norm = 0;
    for (const v of x.data) norm += v * v;
    norm = Math.sqrt(norm);
    const alpha = x.data[0] > 0 ? -norm : norm;
    const v = Float64Array.from(x.data);
    v[0] -= alpha;
    let v2 = 0;
    for (const vv of v) v2 += vv * vv;
    return { v: new Tensor(v, x.shape), beta: v2 === 0 ? 0 : 2 / v2 };
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.givens', arity: 2, sig: '(number a, number b) -> object {c,s,r}',
  doc: 'Givens 旋转（消去 b）', ref: 'Golub & Van Loan §5.1',
  inv: 'c²+s²=1；[c s;-s c][a;b]=[r;0]（r=√(a²+b²)）',
  impl: (a, b) => {
    if (b === 0) return { c: a >= 0 ? 1 : -1, s: 0, r: Math.abs(a) };
    const r = Math.hypot(a, b);
    return { c: a / r, s: b / r, r };
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.convMatrix', arity: 2,
  sig: '(Tensor kernel[k], number n) -> Tensor[n,n]', doc: '1D 卷积矩阵（Toeplitz，same 语义）',
  ref: '信号处理标准构造', inv: 'convMatrix(k,n)·x = conv(x,k)',
  impl: (k, n) => {
    const kk = k.size;
    const pad = Math.floor((kk - 1) / 2);
    const out = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < kk; j++) {
      const col = i + j - pad;
      if (col >= 0 && col < n) out[i * n + col] = k.data[j];
    }
    return new Tensor(out, [n, n]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.circulantMatvec', arity: 2, sig: '(Tensor c[n], Tensor x[n]) -> Tensor',
  doc: '循环矩阵乘法（O(n²)，无需显式构造）', ref: '循环矩阵理论',
  inv: 'circulant 矩阵作用于向量 = 循环卷积',
  impl: (c, x) => {
    const n = c.size;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += c.data[(j - i + n) % n] * x.data[j];
      out[i] = s;
    }
    return new Tensor(out, [n]);
  },
});

// ============ 谓词族（7 条） ============
const PREDICATES = [
  ['isSymmetric', (A, n) => { for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (Math.abs(A.get2(i, j) - A.get2(j, i)) > 1e-12) return false; return true; }, '对称'],
  ['isUpperTri', (A) => { const [m, n] = A.shape; for (let i = 0; i < m; i++) for (let j = 0; j < Math.min(i, n); j++) if (Math.abs(A.get2(i, j)) > 1e-12) return false; return true; }, '上三角'],
  ['isLowerTri', (A) => { const [m, n] = A.shape; for (let i = 0; i < m; i++) for (let j = i + 1; j < n; j++) if (Math.abs(A.get2(i, j)) > 1e-12) return false; return true; }, '下三角'],
  ['isDiagonal', (A, n) => { for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j && Math.abs(A.get2(i, j)) > 1e-12) return false; return true; }, '对角'],
  ['isOrthogonal', (A, n) => { const P = exec('linalg.matmul', [A.T(), A]); return exec('num.allclose', [P, Tensor.eye(n), 1e-10, 1e-10]); }, '正交'],
  ['isSPD', (A, n) => { try { exec('linalg.cholesky', [A]); return true; } catch { return false; } }, '对称正定'],
  ['isIdempotent', (A, n) => exec('num.allclose', [exec('linalg.matmul', [A, A]), A, 1e-10, 1e-10]), '幂等（A²=A）'],
];
for (const [n, f, d] of PREDICATES) {
  defineOp({
    id: nid(), domain: D, name: `linalg.${n}`, sig: '(Tensor A) -> boolean', doc: `判定矩阵是否${d}`,
    ref: '矩阵性质标准判定', inv: `对${d}矩阵返回 true`,
    impl: (A) => { A = mat2d(A, 'A'); if (A.shape[0] !== A.shape[1] && n !== 'isUpperTri' && n !== 'isLowerTri') return false; return f(A, A.shape[0]); },
  });
}

// ============ like 族与形状操作（10 条） ============
defineOp({
  id: nid(), domain: D, name: 'linalg.eyeLike', sig: '(Tensor A) -> Tensor', doc: '与 A 同形状单位阵',
  ref: 'NumPy eye_like', inv: 'eyeLike(A) 与 A 同形', impl: (A) => Tensor.eye(A.shape[0], A.shape[1]),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.zerosLike', sig: '(Tensor A) -> Tensor', doc: '同形状零矩阵',
  ref: 'NumPy zeros_like', inv: '输出全零张量，形状与输入完全相同', impl: (A) => Tensor.zeros(A.shape),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.onesLike', sig: '(Tensor A) -> Tensor', doc: '同形状全一矩阵',
  ref: 'NumPy ones_like', inv: '输出全一张量，形状与输入完全相同', impl: (A) => Tensor.ones(A.shape),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.fullLike', arity: 2, sig: '(Tensor A, number v) -> Tensor', doc: '同形状常数矩阵',
  ref: 'NumPy full_like', inv: '输出全部填充标量 v，形状与输入相同', impl: (A, v) => Tensor.full(A.shape, v),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.concatRows', arity: 2, sig: '(Tensor A, Tensor B) -> Tensor',
  doc: '纵向拼接（列数须一致）', ref: 'NumPy vstack', inv: '沿行（轴0）拼接，输出行数等于各输入行数之和',
  impl: (A, B) => { A = mat2d(A, 'A'); B = mat2d(B, 'B'); if (A.shape[1] !== B.shape[1]) throw new ForgeError('E_SHAPE', 'concatRows 列数不符'); return exec('math.concat', [[A, B], 0]); },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.concatCols', arity: 2, sig: '(Tensor A, Tensor B) -> Tensor',
  doc: '横向拼接（行数须一致）', ref: 'NumPy hstack', inv: '沿列（轴1）拼接，输出列数等于各输入列数之和',
  impl: (A, B) => { A = mat2d(A, 'A'); B = mat2d(B, 'B'); if (A.shape[0] !== B.shape[0]) throw new ForgeError('E_SHAPE', 'concatCols 行数不符'); return exec('math.concat', [[A, B], 1]); },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.submatrixRows', arity: 3, sig: '(Tensor A, number r0, number r1) -> Tensor',
  doc: '取行区间 [r0,r1)', ref: 'NumPy 切片', inv: '行数=r1−r0',
  impl: (A, r0, r1) => { A = mat2d(A, 'A'); const [m, n] = A.shape; if (r0 < 0 || r1 > m || r0 >= r1) throw new ForgeError('E_VALUE', '行区间非法'); return new Tensor(A.data.slice(r0 * n, r1 * n), [r1 - r0, n]); },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.submatrixCols', arity: 3, sig: '(Tensor A, number c0, number c1) -> Tensor',
  doc: '取列区间 [c0,c1)', ref: 'NumPy 切片', inv: '列数=c1−c0',
  impl: (A, c0, c1) => {
    A = mat2d(A, 'A'); const [m, n] = A.shape;
    if (c0 < 0 || c1 > n || c0 >= c1) throw new ForgeError('E_VALUE', '列区间非法');
    const out = new Float64Array(m * (c1 - c0));
    for (let i = 0; i < m; i++) for (let j = c0; j < c1; j++) out[i * (c1 - c0) + (j - c0)] = A.get2(i, j);
    return new Tensor(out, [m, c1 - c0]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.center', sig: '(Tensor X[n,d]) -> Tensor', doc: '按列去均值',
  ref: 'PCA/统计标准', inv: '每列均值=0',
  impl: (X) => exec('math.sub', [X, exec('math.meanAxis', [X, 0])]),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.normalizeRows', sig: '(Tensor X) -> Tensor', doc: '行 L2 归一化',
  ref: '余弦相似度标准预处理', inv: '每行 L2 范数=1',
  impl: (X) => {
    const out = new Float64Array(X.size);
    const [m, n] = X.shape;
    for (let i = 0; i < m; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += X.get2(i, j) ** 2;
      s = Math.sqrt(s) || 1;
      for (let j = 0; j < n; j++) out[i * n + j] = X.get2(i, j) / s;
    }
    return new Tensor(out, [m, n]);
  },
});

// ============ 轴归约工具（6 条） ============
const ROWCOL = [
  ['rowSum', 'math.sumAxis', 1, '行和'], ['colSum', 'math.sumAxis', 0, '列和'],
  ['rowMean', 'math.meanAxis', 1, '行均值'], ['colMean', 'math.meanAxis', 0, '列均值'],
  ['rowMax', 'math.maxAxis', 1, '行最大'], ['colMax', 'math.maxAxis', 0, '列最大'],
  ['rowMin', 'math.minAxis', 1, '行最小'], ['colMin', 'math.minAxis', 0, '列最小'],
];
for (const [n, base, ax, d] of ROWCOL) {
  defineOp({
    id: nid(), domain: D, name: `linalg.${n}`, sig: '(Tensor A) -> Tensor', doc: `矩阵${d}`,
    ref: 'NumPy 轴归约', inv: `与 math 轴归约一致`, impl: (A) => exec(base, [A, ax]),
  });
}

// ============ 补充指令（15 条） ============
defineOp({
  id: nid(), domain: D, name: 'linalg.trace', sig: '(Tensor A) -> number', doc: '矩阵迹 ΣAᵢᵢ',
  ref: '线性代数标准', inv: 'trace(AB)=trace(BA)；trace(A)=Σλᵢ',
  impl: (A) => { A = mat2d(A, 'A'); let s = 0; for (let i = 0; i < A.shape[0]; i++) s += A.get2(i, i); return s; },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.transpose', sig: '(Tensor A) -> Tensor', doc: '矩阵转置（指令化入口）',
  ref: '线性代数标准', inv: 'transpose(transpose(A))=A',
  impl: (A) => { A = mat2d(A, 'A'); return A.T(); },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.diag', sig: '(Tensor A) -> Tensor', doc: '提取对角线',
  ref: 'NumPy diag', inv: 'diag(diagMat(v))=v',
  impl: (A) => { A = mat2d(A, 'A'); const n = Math.min(A.shape[0], A.shape[1]); const out = new Float64Array(n); for (let i = 0; i < n; i++) out[i] = A.get2(i, i); return new Tensor(out, [n]); },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.diagMat', arity: 1, sig: '(Tensor v, number n?) -> Tensor',
  doc: '对角线构造矩阵', ref: 'NumPy diag', inv: 'diagMat(diag(A))=A（方阵）',
  impl: (v, n) => { const nn = n ?? v.size; const out = Tensor.zeros([nn, nn]); for (let i = 0; i < Math.min(nn, v.size); i++) out.set2(i, i, v.data[i]); return out; },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.scaleMat', arity: 2, sig: '(number alpha, Tensor A) -> Tensor', doc: '数乘 αA',
  ref: 'BLAS dscal', inv: 'scaleMat(1,A)=A', impl: (alpha, A) => A.map((v) => alpha * v),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.addMat', arity: 2, sig: '(Tensor A, Tensor B) -> Tensor', doc: '矩阵加法（广播）',
  ref: 'NumPy add', inv: 'addMat(A,B)=addMat(B,A)', impl: (A, B) => exec('math.add', [A, B]),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.subMat', arity: 2, sig: '(Tensor A, Tensor B) -> Tensor', doc: '矩阵减法（广播）',
  ref: 'NumPy subtract', inv: 'subMat(A,A)=0', impl: (A, B) => exec('math.sub', [A, B]),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.hadamard', arity: 2, sig: '(Tensor A, Tensor B) -> Tensor', doc: 'Hadamard 积（逐元素）',
  ref: 'Horn & Johnson', inv: 'hadamard(A,B)=hadamard(B,A)',
  impl: (A, B) => exec('math.mul', [A, B]),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.triLower', sig: '(Tensor A, number k=0) -> Tensor', doc: '下三角部分（第 k 条对角以下）',
  ref: 'NumPy tril', inv: 'triLower(triLower(A))=triLower(A)',
  impl: (A, k = 0) => { A = mat2d(A, 'A'); const [m, n] = A.shape; const out = new Float64Array(m * n); for (let i = 0; i < m; i++) for (let j = 0; j <= Math.min(n - 1, i + k); j++) out[i * n + j] = A.get2(i, j); return new Tensor(out, [m, n]); },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.triUpper', sig: '(Tensor A, number k=0) -> Tensor', doc: '上三角部分（第 k 条对角以上）',
  ref: 'NumPy triu', inv: 'triUpper(triUpper(A))=triUpper(A)',
  impl: (A, k = 0) => { A = mat2d(A, 'A'); const [m, n] = A.shape; const out = new Float64Array(m * n); for (let i = 0; i < m; i++) for (let j = Math.max(0, i + k); j < n; j++) out[i * n + j] = A.get2(i, j); return new Tensor(out, [m, n]); },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.polar', arity: 1, sig: '(Tensor A) -> object {U, H}', doc: '极分解 A=U·H（U 正交，H SPD）',
  ref: 'Higham 1986；Golub & Van Loan §8.3', inv: 'U 正交；H SPD；U·H=A',
  impl: (A) => {
    const { U, S, V } = exec('linalg.svd', [A]);
    // A = U_A·Σ·Vᵀ → U_polar = U_A·Vᵀ，H = V·Σ·Vᵀ
    const Uout = exec('linalg.matmul', [U, V.T()]);
    const H = exec('linalg.matmul', [exec('linalg.matmul', [V, exec('linalg.diagMat', [S, V.shape[0]])]), V.T()]);
    return { U: Uout, H };
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.colSpace', arity: 1, sig: '(Tensor A, number tol?) -> Tensor',
  doc: '列空间正交基（SVD 左奇异向量）', ref: 'Golub & Van Loan §5.5',
  inv: '列正交；每列在 A 的列空间中',
  impl: (A, tol) => {
    const { U, S } = exec('linalg.svd', [A]);
    const smax = S.data[0] || 0;
    const t = tol ?? Math.max(A.shape[0], A.shape[1]) * Number.EPSILON * smax;
    const cols = [];
    for (let i = 0; i < S.size; i++) if (S.data[i] > t) cols.push(i);
    const m = A.shape[0];
    const out = new Float64Array(m * cols.length);
    for (let k = 0; k < cols.length; k++) for (let i = 0; i < m; i++) out[i * cols.length + k] = U.get2(i, cols[k]);
    return new Tensor(out, [m, cols.length]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.rowSpace', arity: 1, sig: '(Tensor A, number tol?) -> Tensor',
  doc: '行空间正交基（SVD 右奇异向量）', ref: 'Golub & Van Loan §5.5',
  inv: 'QR 正交化给出行空间的标准正交基', impl: (A, tol = undefined) => exec('linalg.colSpace', [A.T(), tol]),
});
defineOp({
  id: nid(), domain: D, name: 'linalg.blockDiag', arity: 1, sig: '(Tensor[] blocks) -> Tensor',
  doc: '块对角矩阵', ref: 'Horn & Johnson', inv: '对角块为输入，非块区为零',
  impl: (blocks) => {
    const total = blocks.reduce((s, b) => s + b.shape[0], 0);
    const out = new Float64Array(total * total);
    let off = 0;
    for (const B of blocks) {
      const n = B.shape[0];
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[(off + i) * total + (off + j)] = B.get2(i, j);
      off += n;
    }
    return new Tensor(out, [total, total]);
  },
});
defineOp({
  id: nid(), domain: D, name: 'linalg.flipLR', sig: '(Tensor A) -> Tensor', doc: '沿列轴翻转',
  ref: 'NumPy fliplr', inv: 'flipLR(flipLR(A))=A',
  impl: (A) => { A = mat2d(A, 'A'); const [m, n] = A.shape; const out = new Float64Array(m * n); for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[i * n + j] = A.get2(i, n - 1 - j); return new Tensor(out, [m, n]); },
});

// 文件底部自检
if (opsByDomain(D).length !== 102) {
  throw new Error(`域${D}(LINALG) 指令数 = ${opsByDomain(D).length}，应为 102`);
}
