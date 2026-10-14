// 域3 线性代数测试：分解不变量、求解残差、矩阵函数精度
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { exec, opsByDomain } from '../src/core/registry.mjs';
import { Tensor } from '../src/core/index.mjs';
import '../src/ops/d1-math.mjs';
import '../src/ops/d2-num.mjs';
import '../src/ops/d3-linalg.mjs';
import { approx, tensorClose, maxAbsDiff, randMat, randSPD, blobs } from './helpers.mjs';

const L = (name, ...args) => exec(name, args);

test('域3 恰好 102 条指令且 id 连续', () => {
  const ops = opsByDomain(3);
  assert.equal(ops.length, 102);
  const ids = ops.map(o => o.id).sort();
  assert.equal(ids[0], 'D3-001');
  assert.equal(ids[101], 'D3-102');
  assert.equal(new Set(ids).size, 102);
});

test('matmul 族 vs 朴素三重循环', () => {
  const rngA = randMat(5, 7, 11);
  const rngB = randMat(7, 4, 22);
  const C = L('linalg.matmul', rngA, rngB);
  assert.deepEqual(C.shape, [5, 4]);
  // 朴素参考
  for (let i = 0; i < 5; i++) for (let j = 0; j < 4; j++) {
    let s = 0;
    for (let k = 0; k < 7; k++) s += rngA.get2(i, k) * rngB.get2(k, j);
    assert.ok(approx(C.get2(i, j), s, 1e-10), `C[${i},${j}]`);
  }
  // 变体恒等式
  tensorClose(L('linalg.matmulAT', rngA.T(), rngB), C, 1e-10);
  tensorClose(L('linalg.matmulBT', rngA, rngB.T()), L('linalg.matmul', rngA, rngB), 1e-10);
  tensorClose(L('linalg.gemm', 2, rngA, rngB, 0, null), C.map((v) => 2 * v), 1e-10);
  const D = Tensor.zeros([5, 4]);
  tensorClose(L('linalg.gemm', 1, rngA, rngB, 3, D), C, 1e-10);
  // 单位阵
  tensorClose(L('linalg.matmul', rngA, Tensor.eye(7)), rngA, 1e-10);
  // matmulVec
  const x = Tensor.fromFlat([1, 2, 3, 4, 5, 6, 7], [7]);
  tensorClose(L('linalg.matmulVec', rngA, x), exec('linalg.matmul', [rngA, x.reshape(7, 1)]).reshape(5), 1e-10);
  // 形状错误
  assert.throws(() => L('linalg.matmul', rngA, randMat(3, 3, 1)), (e) => e.code === 'E_SHAPE');
});

test('LU：P·A=L·U 与 det/inverse/solve', () => {
  const A = randMat(6, 6, 33);
  const b = Tensor.fromFlat([1, 2, 3, 4, 5, 6], [6]);
  const { L: Lm, U: Um, pivots, sign } = L('linalg.lu', A);
  // 重构 P·A = L·U
  let PA = A.clone();
  for (let k = 0; k < pivots.length; k++) { const p = pivots[k]; if (p !== k) { for (let j = 0; j < 6; j++) { const t = PA.get2(k, j); PA.set2(k, j, PA.get2(p, j)); PA.set2(p, j, t); } } }
  tensorClose(L('linalg.matmul', Lm, Um), PA, 1e-10);
  // L 单位下三角，U 上三角
  assert.ok(L('linalg.isLowerTri', Lm));
  assert.ok(L('linalg.isUpperTri', Um));
  for (let i = 0; i < 6; i++) assert.equal(Lm.get2(i, i), 1);
  // det：与 2x2 解析解交叉验证 + 三角阵
  const tri = L('linalg.triUpper', randMat(4, 4, 9), 0);
  let detTri = 1;
  for (let i = 0; i < 4; i++) detTri *= tri.get2(i, i);
  assert.ok(approx(L('linalg.det', tri), detTri, 1e-10));
  const A22 = Tensor.fromNested([[1, 2], [3, 4]]);
  assert.equal(L('linalg.det', A22), -2);
  // inverse：A·A⁻¹=I
  const Ainv = L('linalg.inverse', A);
  tensorClose(L('linalg.matmul', A, Ainv), Tensor.eye(6), 1e-8);
  // solve：残差
  const x = L('linalg.solve', A, b);
  tensorClose(L('linalg.matmulVec', A, x), b, 1e-8);
  // solveSPD 与 solve 一致
  const S = randSPD(6, 5);
  const xs = L('linalg.solveSPD', S, b);
  tensorClose(L('linalg.matmulVec', S, xs), b, 1e-8);
  tensorClose(xs, L('linalg.solve', S, b), 1e-8);
  // solveT
  tensorClose(L('linalg.matmulVec', A.T(), L('linalg.solveT', A, b)), b, 1e-8);
  // solveMultiple
  const B2 = Tensor.fromNested([[1, 0], [0, 1], [1, 1], [2, 1], [0, 3], [1, 1]]);
  const X2 = L('linalg.solveMultiple', A, B2);
  tensorClose(L('linalg.matmul', A, X2), B2, 1e-8);
  // 奇异矩阵
  const sing = Tensor.fromNested([[1, 2], [2, 4]]);
  assert.throws(() => L('linalg.inverse', sing), (e) => e.code === 'E_NUMERICAL');
});

test('QR：Q 正交、Q·R=A', () => {
  const A = randMat(7, 5, 77);
  const { Q, R } = L('linalg.qr', A);
  assert.deepEqual(Q.shape, [7, 7]);
  assert.deepEqual(R.shape, [7, 5]);
  tensorClose(L('linalg.matmul', Q, R), A, 1e-10);
  tensorClose(L('linalg.matmul', Q.T(), Q), Tensor.eye(7), 1e-10);
  assert.ok(L('linalg.isUpperTri', R));
  // 方阵情形
  const S = randMat(5, 5, 78);
  const qr2 = L('linalg.qr', S);
  tensorClose(L('linalg.matmul', qr2.Q, qr2.R), S, 1e-10);
  assert.ok(approx(Math.abs(L('linalg.det', qr2.Q)), 1, 1e-10));
});

test('Cholesky 与 logdet', () => {
  const S = randSPD(6, 5);
  const Lm = L('linalg.cholesky', S);
  assert.ok(L('linalg.isLowerTri', Lm));
  for (let i = 0; i < 6; i++) assert.ok(Lm.get2(i, i) > 0);
  tensorClose(L('linalg.matmul', Lm, Lm.T()), S, 1e-10);
  assert.ok(L('linalg.isSPD', S));
  assert.ok(approx(Math.exp(L('linalg.logdet', S)), L('linalg.det', S), 1e-6));
  // 非 SPD 抛错
  assert.throws(() => L('linalg.cholesky', Tensor.fromNested([[1, 2], [2, 1]])), (e) => e.code === 'E_NUMERICAL');
  assert.equal(L('linalg.isSPD', Tensor.fromNested([[1, 2], [2, 1]])), false);
});

test('eigenSym：残差/正交/降序/交叉验证', () => {
  const A = randMat(8, 8, 42);
  const Sym = exec('linalg.symmetrize', [A]);
  const { values, vectors } = L('linalg.eigenSym', Sym);
  assert.deepEqual(values.shape, [8]);
  // 降序
  for (let i = 1; i < 8; i++) assert.ok(values.data[i - 1] >= values.data[i] - 1e-12);
  // 特征方程残差
  let maxRes = 0;
  for (let i = 0; i < 8; i++) {
    const v = new Tensor(vectors.col(i), [8]);
    const Av = L('linalg.matmulVec', Sym, v);
    for (let k = 0; k < 8; k++) maxRes = Math.max(maxRes, Math.abs(Av.data[k] - values.data[i] * v.data[k]));
  }
  assert.ok(maxRes < 1e-10, `特征残差 ${maxRes}`);
  // V 正交
  tensorClose(L('linalg.matmul', vectors.T(), vectors), Tensor.eye(8), 1e-10);
  // 重构 A = VΛVᵀ
  const recon = L('linalg.matmul', L('linalg.matmul', vectors, L('linalg.diagMat', values)), vectors.T());
  tensorClose(recon, Sym, 1e-9);
  // 与 2x2 解析解一致（特征值精确相等；向量符号可取反，比较投影一致性）
  const S2 = Tensor.fromNested([[4, 1], [1, 3]]);
  const e2 = L('linalg.eigen2x2', S2);
  const eJ = L('linalg.eigenSym', S2);
  tensorClose(eJ.values, e2.values, 1e-10);
  for (let i = 0; i < 2; i++) {
    const vJ = new Tensor(eJ.vectors.col(i), [2]);
    const v2 = new Tensor(e2.vectors.col(i), [2]);
    assert.ok(Math.abs(exec('math.dot', [vJ, v2])) > 1 - 1e-8, `特征向量${i}应平行`);
  }
  // QR 迭代与 Jacobi 一致
  const eQ = L('linalg.eigenSymQR', Sym);
  tensorClose(eQ.values, values, 1e-7);
  // trace = Σλ
  assert.ok(approx(L('linalg.trace', Sym), values.data.reduce((s, v) => s + v, 0), 1e-8));
});

test('SVD：重构/正交/奇异值/pinv/rank', () => {
  const A = randMat(6, 4, 99);
  const { U, S, V } = L('linalg.svd', A);
  assert.deepEqual(U.shape, [6, 4]);
  assert.deepEqual(S.shape, [4]);
  assert.deepEqual(V.shape, [4, 4]);
  // 重构
  const recon = L('linalg.matmul', U, L('linalg.matmul', L('linalg.diagMat', S, 4), V.T()));
  tensorClose(recon, A, 1e-9);
  // 正交
  tensorClose(L('linalg.matmul', U.T(), U), Tensor.eye(4), 1e-9);
  tensorClose(L('linalg.matmul', V.T(), V), Tensor.eye(4), 1e-9);
  // 奇异值非负降序
  for (let i = 0; i < 4; i++) { assert.ok(S.data[i] >= 0); if (i > 0) assert.ok(S.data[i - 1] >= S.data[i] - 1e-12); }
  // 范数
  assert.ok(approx(L('linalg.norm2', A), S.data[0], 1e-9));
  assert.ok(approx(L('linalg.normNuc', A), S.data.reduce((s, v) => s + v, 0), 1e-9));
  // 宽矩阵
  const W = randMat(3, 6, 55);
  const sw = L('linalg.svd', W);
  const reconW = L('linalg.matmul', sw.U, L('linalg.matmul', L('linalg.diagMat', sw.S, 3), sw.V));
  tensorClose(reconW, W, 1e-9);
  // 满秩方阵 pinv = inverse
  const Sq = randMat(5, 5, 66);
  tensorClose(L('linalg.pinv', Sq), L('linalg.inverse', Sq), 1e-7);
  // 秩亏
  const lowRank = L('linalg.matmul', randMat(6, 2, 7), randMat(2, 5, 8));
  assert.equal(L('linalg.rank', lowRank), 2);
  const pinvLR = L('linalg.pinv', lowRank);
  tensorClose(L('linalg.matmul', L('linalg.matmul', lowRank, pinvLR), lowRank), lowRank, 1e-7);
  // 零空间
  const NS = L('linalg.nullSpace', lowRank);
  assert.deepEqual(NS.shape, [5, 3]);
  tensorClose(L('linalg.matmul', lowRank, NS), Tensor.zeros([6, 3]), 1e-7);
});

test('条件数与范数', () => {
  assert.equal(L('linalg.cond2', Tensor.eye(4)), 1);
  const A = randMat(5, 5, 88);
  assert.ok(L('linalg.cond1', A) >= 1);
  assert.ok(L('linalg.cond2', A) >= 1);
  const diag = L('linalg.diagMat', Tensor.fromFlat([1, 2, 3], [3]));
  assert.ok(approx(L('linalg.cond2', diag), 3, 1e-12));
  // 范数一致性
  assert.ok(approx(L('linalg.normFro', A), Math.sqrt(A.data.reduce((s, v) => s + v * v, 0)), 1e-12));
  const At = A.T();
  assert.ok(approx(L('linalg.norm1', At), L('linalg.normInf', A), 1e-12));
  // Gershgorin 包含全部特征值
  const Sym = exec('linalg.symmetrize', [A]);
  const g = L('linalg.gershgorin', Sym);
  const { values } = L('linalg.eigenSym', Sym);
  for (const v of values.data) assert.ok(v >= g.lo - 1e-9 && v <= g.hi + 1e-9, `特征值 ${v} 应在 [${g.lo},${g.hi}]`);
});

test('迭代求解器', () => {
  // 对角占稳系统
  const n = 12;
  const A = Tensor.zeros([n, n]);
  for (let i = 0; i < n; i++) { A.set2(i, i, 10); if (i > 0) A.set2(i, i - 1, 1); if (i < n - 1) A.set2(i, i + 1, 1); }
  const xTrue = Tensor.fromFlat(Array.from({ length: n }, (_, i) => i + 1), [n]);
  const b = L('linalg.matmulVec', A, xTrue);
  const xExact = L('linalg.solve', A, b);
  tensorClose(xExact, xTrue, 1e-10);
  for (const name of ['linalg.jacobiIter', 'linalg.gaussSeidel', 'linalg.sor']) {
    const r = L(name, A, b, 2000, 1e-11);
    assert.ok(r.converged, `${name} 未收敛`);
    tensorClose(r.x, xTrue, 1e-6, `${name} 解`);
  }
  const rSor = L('linalg.sor', A, b, 2000, 1e-11, 1.4);
  assert.ok(rSor.converged);
  // CG 对 SPD
  const S = randSPD(10, 3);
  const b2 = Tensor.fromFlat(Array.from({ length: 10 }, (_, i) => i), [10]);
  const rCG = L('linalg.conjGradient', S, b2);
  assert.ok(rCG.converged, 'CG 未收敛');
  tensorClose(L('linalg.matmulVec', S, rCG.x), b2, 1e-7);
  // 幂迭代（收敛到按模最大特征；符号振荡时返回模）
  const Sym = exec('linalg.symmetrize', [randMat(8, 8, 4)]);
  const { values } = L('linalg.eigenSym', Sym);
  const pi = L('linalg.powerIteration', Sym, 500, 1e-12);
  const domByMod = Math.max(Math.abs(values.data[0]), Math.abs(values.data[values.size - 1]));
  assert.ok(approx(Math.abs(pi.value), domByMod, 1e-7), `幂迭代 |${pi.value}| vs |${domByMod}|`);
  // rayleigh
  const v0 = new Tensor(L('linalg.eigenSym', Sym).vectors.col(0), [8]);
  assert.ok(approx(L('linalg.rayleighQuotient', Sym, v0), values.data[0], 1e-8));
});

test('矩阵函数与结构', () => {
  // matrixExp：e⁰=I；对角矩阵精确
  const D = L('linalg.diagMat', Tensor.fromFlat([0.5, -1, 2], [3]));
  const E = L('linalg.matrixExp', D);
  tensorClose(E, L('linalg.diagMat', Tensor.fromFlat([Math.exp(0.5), Math.exp(-1), Math.exp(2)], [3])), 1e-10);
  // 半群性 e^{A}e^{A}=e^{2A}
  const A = Tensor.fromNested([[0.1, 0.2], [0, 0.3]]);
  tensorClose(L('linalg.matmul', L('linalg.matrixExp', A), L('linalg.matrixExp', A)), L('linalg.matrixExp', A.map((v) => 2 * v)), 1e-8);
  // matrixPower
  tensorClose(L('linalg.matrixPower', A, 0), Tensor.eye(2), 1e-12);
  tensorClose(L('linalg.matrixPower', A, 3), L('linalg.matmul', L('linalg.matmul', A, A), A), 1e-12);
  // kron
  const K = L('linalg.kron', Tensor.eye(2), Tensor.fromNested([[1, 2], [3, 4]]));
  assert.deepEqual(K.shape, [4, 4]);
  tensorClose(K, Tensor.fromNested([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 1, 2], [0, 0, 3, 4]]), 1e-12);
  // vec/unvec
  const M = randMat(3, 4, 12);
  tensorClose(L('linalg.unvec', L('linalg.vec', M), 3), M, 1e-12);
  // householder：反射保持范数
  const x = Tensor.fromFlat([3, 4, 0], [3]);
  const { v, beta } = L('linalg.householder', x);
  // 手动构造 H = I - beta v vᵀ
  const Hm = Tensor.zeros([3, 3]);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) Hm.set2(i, j, (i === j ? 1 : 0) - beta * v.data[i] * v.data[j]);
  const Hx = L('linalg.matmulVec', Hm, x);
  assert.ok(Math.abs(Hx.data[1]) < 1e-12 && Math.abs(Hx.data[2]) < 1e-12);
  assert.ok(approx(Math.abs(Hx.data[0]), 5, 1e-12));
  // givens
  const g = L('linalg.givens', 3, 4);
  assert.ok(approx(g.c * g.c + g.s * g.s, 1, 1e-12));
  assert.ok(approx(g.c * 3 + g.s * 4, 5, 1e-12));
  assert.ok(approx(-g.s * 3 + g.c * 4, 0, 1e-12));
  // tridiag/Toeplitz
  const sub = Tensor.fromFlat([0, -1, -1, -1], [4]);
  const diag = Tensor.fromFlat([2, 2, 2, 2], [4]);
  const sup = Tensor.fromFlat([-1, -1, -1, 0], [4]);
  const rhs = Tensor.fromFlat([1, 0, 0, 1], [4]);
  const xt = L('linalg.tridiagSolve', sub, diag, sup, rhs);
  const T = Tensor.fromNested([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]]);
  tensorClose(L('linalg.matmulVec', T, xt), rhs, 1e-10);
  tensorClose(L('linalg.toeplitzSolve', Tensor.fromFlat([2, -1, 0, 0, -1], [5]), rhs), xt, 1e-10);
  // convMatrix / circulant（same 卷积：out[i]=Σⱼk[j]·x[i+j−pad]）
  const k = Tensor.fromFlat([1, 2, 1], [3]);
  const CM = L('linalg.convMatrix', k, 5);
  const xs = Tensor.fromFlat([1, 0, 1, 0, 1], [5]);
  tensorClose(L('linalg.matmulVec', CM, xs), Tensor.fromFlat([2, 2, 2, 2, 2], [5]), 1e-12);
  // polar
  const A2 = randMat(4, 4, 21);
  const pol = L('linalg.polar', A2);
  tensorClose(L('linalg.matmul', pol.U, pol.H), A2, 1e-8);
  tensorClose(L('linalg.matmul', pol.U.T(), pol.U), Tensor.eye(4), 1e-7);
  assert.ok(L('linalg.isSPD', pol.H));
  // blockDiag
  const BD = L('linalg.blockDiag', [Tensor.eye(2), Tensor.ones([1, 1])]);
  assert.deepEqual(BD.shape, [3, 3]);
  assert.equal(BD.get2(2, 2), 1);
  // colSpace/rowSpace：行空间与零空间正交；列正交
  const LR = L('linalg.matmul', randMat(6, 2, 31), randMat(2, 5, 32));
  const CS = L('linalg.colSpace', LR);
  const RS = L('linalg.rowSpace', LR);
  const NS = L('linalg.nullSpace', LR);
  assert.deepEqual(CS.shape, [6, 2]);
  assert.deepEqual(RS.shape, [5, 2]);
  tensorClose(L('linalg.matmul', CS.T(), CS), Tensor.eye(2), 1e-7);
  tensorClose(L('linalg.matmul', RS.T(), NS), Tensor.zeros([2, 3]), 1e-7); // 行空间 ⊥ 零空间
});

test('polyfit/spline/whiten/nearestSPD', () => {
  // polyfit 精确恢复二次多项式
  const xs = Tensor.linspace(-2, 2, 21);
  const ys = xs.map((v) => 3 * v * v - 2 * v + 1);
  const coefs = L('linalg.polyfit', xs, ys, 2);
  tensorClose(coefs, Tensor.fromFlat([3, -2, 1], [3]), 1e-8);
  // 带噪声线性数据 R² 高
  const rngX = Tensor.linspace(-3, 3, 50);
  const noisy = rngX.map((v, i) => 2 * v + 1 + (i % 5 - 2) * 0.01);
  const cf = L('linalg.polyfit', rngX, noisy, 1);
  assert.ok(approx(cf.data[0], 2, 1e-2), `斜率 ${cf.data[0]}`);
  assert.ok(approx(cf.data[1], 1, 1e-2), `截距 ${cf.data[1]}`);
  tensorClose(L('linalg.polyval', cf, rngX), noisy, 0.05);
  // whiten：协方差 ≈ I
  const X = blobs({ n: 300, centers: [[0, 0], [4, 4]], std: 1, seed: 5 }).X;
  const { Xw } = L('linalg.whiten', X, 1e-8);
  const Xwc = L('linalg.center', Xw);
  const cov = L('linalg.gemm', 1 / (Xw.shape[0] - 1), Xwc.T(), Xwc, 0, null);
  tensorClose(cov, Tensor.eye(2), 1e-5);
  // nearestSPD：给半定矩阵
  const semi = Tensor.fromNested([[1, 2], [2, 1]]);
  const ns = L('linalg.nearestSPD', semi);
  assert.ok(L('linalg.isSPD', ns));
  assert.ok(L('linalg.isSymmetric', ns));
});

test('谓词与形状操作', () => {
  const A = randMat(4, 4, 3);
  assert.equal(L('linalg.isSymmetric', exec('linalg.symmetrize', [A])), true);
  assert.equal(L('linalg.isSymmetric', A), false);
  assert.equal(L('linalg.isOrthogonal', Tensor.eye(3)), true);
  assert.equal(L('linalg.isDiagonal', L('linalg.diagMat', Tensor.fromFlat([1, 2], [2]))), true);
  assert.equal(L('linalg.isIdempotent', Tensor.fromNested([[1, 0], [0, 0]])), true);
  // like 族
  assert.deepEqual(L('linalg.zerosLike', A).shape, [4, 4]);
  assert.deepEqual(L('linalg.fullLike', A, 7).toArray().flat().every((v) => v === 7), true);
  // 拼接/切片
  const B = randMat(2, 4, 4);
  assert.deepEqual(L('linalg.concatRows', A, B).shape, [6, 4]);
  assert.deepEqual(L('linalg.concatCols', A, randMat(4, 2, 5)).shape, [4, 6]);
  assert.deepEqual(L('linalg.submatrixRows', A, 1, 3).shape, [2, 4]);
  tensorClose(L('linalg.submatrixCols', A, 0, 2), Tensor.fromNested([Array.from(A.row(0).subarray(0, 2)), Array.from(A.row(1).subarray(0, 2)), Array.from(A.row(2).subarray(0, 2)), Array.from(A.row(3).subarray(0, 2))]), 1e-12);
  // 三角提取
  const T = randMat(4, 4, 6);
  assert.ok(L('linalg.isUpperTri', L('linalg.triUpper', T)));
  assert.ok(L('linalg.isLowerTri', L('linalg.triLower', T)));
  // flipLR
  const F = L('linalg.flipLR', Tensor.fromNested([[1, 2, 3]]));
  assert.deepEqual(F.toArray(), [[3, 2, 1]]);
  // 轴归约
  const M = Tensor.fromNested([[1, 2], [3, 4]]);
  assert.deepEqual(L('linalg.rowSum', M).toArray(), [3, 7]);
  assert.deepEqual(L('linalg.colSum', M).toArray(), [4, 6]);
  assert.deepEqual(L('linalg.colMean', M).toArray(), [2, 3]);
  // center/normalizeRows
  const C = L('linalg.center', M);
  tensorClose(exec('math.meanAxis', [C, 0]), Tensor.zeros([2]), 1e-12);
  assert.ok(approx(exec('math.l2', [new Tensor(L('linalg.normalizeRows', M).row(0), [2])]), 1, 1e-12));
});
