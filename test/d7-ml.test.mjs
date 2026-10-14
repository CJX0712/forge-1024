// 域7 机器学习测试：闭式解一致性、稀疏性、可分性、聚类可分离性、PCA 谱性质、
// 核矩阵半正定性、层次聚类单调链、预处理不变量，外加全部 fit/predict 对的冒烟。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { exec, opsByDomain, requiredParams } from '../src/core/registry.mjs';
import { Tensor, createRng } from '../src/core/index.mjs';
import '../src/ops/d1-math.mjs';
import '../src/ops/d2-num.mjs';
import '../src/ops/d3-linalg.mjs';
import '../src/ops/d6-prob.mjs';
import '../src/ops/d7-ml.mjs';
import { approx, blobs, linData, xorData, linSep } from './helpers.mjs';

const O = (name, args) => exec(name, args);
const V = (t) => Array.from(t.data);                       // Tensor -> 纯数组
const T1 = (a) => Tensor.fromFlat(a, [a.length]);
const accuracy = (pred, truth) => {
  const a = pred instanceof Tensor ? V(pred) : pred;
  const b = truth instanceof Tensor ? V(truth) : truth;
  let c = 0;
  for (let i = 0; i < a.length; i++) if (a[i] === b[i]) c++;
  return c / a.length;
};

// —— 测试数据 ——
const REG = linData(80, 7, 0.15);            // y = 2x + 1 + 噪声
const SEP = linSep(120, 11);                 // 线性可分二分类（标签 ±1）
const XOR = xorData(240, 3);                 // XOR 四团
const B3 = blobs({ n: 90, centers: [[0, 0], [8, 8], [0, 8]], std: 0.5, seed: 42 });
const B2 = blobs({ n: 80, centers: [[0, 0], [5, 5]], std: 0.7, seed: 17 });

// 相关三维数据（用于 PCA / 白化）
function correlated3(n = 90, seed = 3) {
  const rng = createRng(seed);
  const rows = [];
  for (let i = 0; i < n; i++) {
    const a = rng.normal(), b = rng.normal();
    rows.push([a, 2 * a + 0.1 * rng.normal(), 0.5 * b + a]);
  }
  return Tensor.fromNested(rows);
}

// 协方差矩阵（无偏，分母 n−1）
function covMat(T) {
  const [n, d] = T.shape;
  const rows = [];
  for (let i = 0; i < n; i++) rows.push(Array.from(T.data.subarray(i * d, (i + 1) * d)));
  const m = new Array(d).fill(0);
  for (const r of rows) for (let j = 0; j < d; j++) m[j] += r[j];
  for (let j = 0; j < d; j++) m[j] /= n;
  const C = [];
  for (let a = 0; a < d; a++) {
    const row = [];
    for (let b = 0; b < d; b++) {
      let s = 0;
      for (const r of rows) s += (r[a] - m[a]) * (r[b] - m[b]);
      row.push(s / (n - 1));
    }
    C.push(row);
  }
  return { C: Tensor.fromNested(C), mean: m };
}

// =====================================================================
test('1. 域7 恰好 103 条，id 为 D7-001..D7-103 且唯一', () => {
  const ops = opsByDomain(7);
  assert.equal(ops.length, 103);
  const ids = ops.map((o) => o.id).sort();
  assert.equal(ids[0], 'D7-001');
  assert.equal(ids[102], 'D7-103');
  assert.equal(new Set(ids).size, 103);
  // 元数据齐全 + arity 与 impl 形参个数严格一致
  for (const o of ops) {
    assert.ok(o.name.startsWith('ml.'), `${o.id} 名称前缀错误: ${o.name}`);
    assert.ok(o.doc && o.sig.includes('->') && o.ref.length >= 6 && o.inv.length >= 6, `${o.name} 元数据缺失`);
    assert.equal(o.arity, requiredParams(o.impl), `${o.name} arity 与必需形参数不符`);
  }
});

// =====================================================================
test('2. 线性回归闭式解：残差正交于设计矩阵（Xᵀ(y−Xŵ)≈0）', () => {
  const st = O('ml.linreg.fit', [REG.X, REG.y]);
  const pred = O('ml.linreg.predict', [st, REG.X]);
  const rows = REG.X.toArray();
  const yv = V(REG.y), pv = V(pred);
  // 含截距列：Σ_i (y_i − ŷ_i) ≈ 0 且 Σ_i x_ij (y_i − ŷ_i) ≈ 0
  let g0 = 0;
  const g = [0];
  for (let i = 0; i < yv.length; i++) {
    const r = yv[i] - pv[i];
    g0 += r;
    g[0] += rows[i][0] * r;
  }
  const scale = Math.sqrt(yv.reduce((s, v) => s + v * v, 0));
  assert.ok(Math.abs(g0) < 1e-8 * scale, `截距方向梯度过大: ${g0}`);
  assert.ok(Math.abs(g[0]) < 1e-8 * scale, `特征方向梯度过大: ${g[0]}`);
  // 与 linalg.lstsq 直接解一致
  const aug = Tensor.fromNested(rows.map((r) => [1, r[0]]));
  const ref = O('linalg.lstsq', [aug, REG.y]).x;
  assert.ok(approx(V(ref)[0], st.intercept, 1e-9), '截距与 linalg 解不一致');
  assert.ok(approx(V(ref)[1], st.coef[0], 1e-9), '斜率与 linalg 解不一致');
});

// =====================================================================
test('3. Ridge 随 λ→0 退化为普通最小二乘', () => {
  const ols = O('ml.linreg.fit', [REG.X, REG.y]);
  const r1 = O('ml.ridge.fit', [REG.X, REG.y, 1e-10]);
  const r2 = O('ml.ridge.fit', [REG.X, REG.y, 1.0]);
  assert.ok(approx(r1.coef[0], ols.coef[0], 1e-6), `λ≈0 时应等于 OLS: ${r1.coef[0]} vs ${ols.coef[0]}`);
  assert.ok(approx(r1.intercept, ols.intercept, 1e-6), '截距应等于 OLS');
  // λ 增大时系数绝对值单调收缩
  assert.ok(Math.abs(r2.coef[0]) < Math.abs(r1.coef[0]), 'λ 增大应使系数收缩');
});

// =====================================================================
test('4. Lasso 在高 λ 下产生稀疏解，λ=0 时退化为最小二乘', () => {
  const sparse = O('ml.lasso.fit', [REG.X, REG.y, 8, 2000]);
  const zeros = sparse.coef.filter((v) => v === 0).length;
  assert.ok(zeros > 0, `高 λ 下应出现精确零系数，实际 coef=${sparse.coef}`);
  assert.equal(sparse.nonzero, sparse.coef.length - zeros);
  const dense = O('ml.lasso.fit', [REG.X, REG.y, 0, 2000]);
  const ols = O('ml.linreg.fit', [REG.X, REG.y]);
  assert.ok(approx(dense.coef[0], ols.coef[0], 1e-6), 'λ=0 时应等于 OLS');
  // 弹性网：l1Ratio=1 退化为 Lasso，l1Ratio=0 退化为 Ridge
  const en1 = O('ml.elasticNet.fit', [REG.X, REG.y, 8, 1, 2000]);
  assert.ok(en1.coef.filter((v) => v === 0).length > 0, 'l1Ratio=1 应产生稀疏解');
  // 弹性网内部按标准化特征施加惩罚，岭回归按原始尺度；故在已标准化的数据上比对
  // （此时 ‖w‖² 惩罚等价，l1Ratio=0 必须与同 λ 的岭回归同解）。
  const XS = O('ml.scale.standard', [REG.X]).X;
  const en0 = O('ml.elasticNet.fit', [XS, REG.y, 1, 0, 5000]);
  const ridge = O('ml.ridge.fit', [XS, REG.y, 1]);
  assert.ok(approx(en0.coef[0], ridge.coef[0], 1e-6), `l1Ratio=0 应等于岭回归: ${en0.coef[0]} vs ${ridge.coef[0]}`);
  const en1s = O('ml.elasticNet.fit', [XS, REG.y, 1, 1, 5000]);
  const las = O('ml.lasso.fit', [XS, REG.y, 1, 5000]);
  assert.ok(approx(en1s.coef[0], las.coef[0], 1e-6), `l1Ratio=1 应等于 Lasso: ${en1s.coef[0]} vs ${las.coef[0]}`);
});

// =====================================================================
test('5. 逻辑回归在线性可分数据上训练准确率 = 1（含感知机、线性 SVM）', () => {
  const lr = O('ml.logreg.fit', [SEP.X, SEP.y, 200, 1e-8]);
  assert.equal(accuracy(O('ml.logreg.predict', [lr, SEP.X]), SEP.y), 1);
  const p = O('ml.logreg.predictProba', [lr, SEP.X]);
  // 说明：线性可分数据上 |wᵀx| 很大，σ 会饱和到端点 0 或 1，故断言闭区间 [0,1]
  for (const v of V(p)) assert.ok(v >= 0 && v <= 1, `概率应 ∈ [0,1]，实际 ${v}`);
  // 概率与硬预测一致（阈值 0.5）
  const hard = V(O('ml.logreg.predict', [lr, SEP.X]));
  const proba = V(p);
  V(SEP.y).forEach((lab, i) => {
    const isPos = proba[i] >= 0.5;
    assert.equal(hard[i], isPos ? lr.posLabel : lr.negLabel, '概率阈值与硬分类不一致');
    assert.equal(hard[i], lab, `第 ${i} 个样本误分`);
  });
  const pc = O('ml.perceptron.fit', [SEP.X, SEP.y, 0.1, 300]);
  assert.equal(accuracy(O('ml.perceptron.predict', [pc, SEP.X]), SEP.y), 1);
  const svm = O('ml.svm.fit', [SEP.X, SEP.y, 1, 400, 0.01]);
  assert.equal(accuracy(O('ml.svm.predict', [svm, SEP.X]), SEP.y), 1);
  const dec = V(O('ml.svm.decision', [svm, SEP.X]));
  const lab = V(SEP.y);
  dec.forEach((d, i) => {
    const want = lab[i] === svm.posLabel ? 1 : -1;
    assert.ok(d * want > 0, `决策函数符号错误: ${d} for label ${lab[i]}`);
  });
});

// =====================================================================
test('6. 高斯朴素贝叶斯在两类高斯数据上准确率 > 0.9（含 LDA/QDA/softmax）', () => {
  // 均值相距 4σ 的两类单位方差高斯
  const rng = createRng(2026);
  const rows = [], ys = [];
  for (let i = 0; i < 200; i++) {
    const c = i % 2;
    rows.push([c * 4 + rng.normal(), c * 4 + rng.normal()]);
    ys.push(c);
  }
  const X = Tensor.fromNested(rows), y = T1(ys);
  const nb = O('ml.nb.gaussianFit', [X, y]);
  assert.ok(accuracy(O('ml.nb.gaussianPredict', [nb, X]), y) > 0.9);
  const lda = O('ml.lda.fit', [X, y]);
  assert.ok(accuracy(O('ml.lda.predict', [lda, X]), y) > 0.9);
  const qda = O('ml.qda.fit', [X, y]);
  assert.ok(accuracy(O('ml.qda.predict', [qda, X]), y) > 0.9);
  // 三分类（softmax / 多项式 NB / 伯努利 NB）
  const sm = O('ml.softmax.fit', [B3.X, B3.y, 0.5, 600]);
  assert.equal(accuracy(O('ml.softmax.predict', [sm, B3.X]), B3.y), 1);
  const P = O('ml.softmax.predictProba', [sm, B3.X]);
  const [m, K] = P.shape;
  for (let i = 0; i < m; i++) {
    let s = 0;
    for (let k = 0; k < K; k++) s += P.data[i * K + k];
    assert.ok(approx(s, 1, 1e-10), `第 ${i} 行概率和 = ${s}`);
  }
  // 多项式 NB：真实的计数型数据（两类各有明显不同的多项分布）
  const rng2 = createRng(77);
  const counts = [], cys = [];
  for (let i = 0; i < 200; i++) {
    const c = i % 2;
    const p = c === 0 ? [0.75, 0.2, 0.05] : [0.05, 0.2, 0.75];
    const row = [0, 0, 0];
    for (let t = 0; t < 12; t++) {
      const u = rng2.next();
      row[u < p[0] ? 0 : u < p[0] + p[1] ? 1 : 2]++;
    }
    counts.push(row); cys.push(c);
  }
  const Xc = Tensor.fromNested(counts), yc = T1(cys);
  const mnb = O('ml.nb.multinomialFit', [Xc, yc, 1]);
  assert.ok(accuracy(O('ml.nb.multinomialPredict', [mnb, Xc]), yc) > 0.9,
    `多项式 NB 计数数据准确率应 > 0.9，实际 ${accuracy(O('ml.nb.multinomialPredict', [mnb, Xc]), yc)}`);
  // 伯努利 NB：二值特征，两类取反
  const bins = [], bys = [];
  for (let i = 0; i < 200; i++) {
    const c = i % 2;
    const p = c === 0 ? [0.9, 0.1] : [0.1, 0.9];
    bins.push(p.map((q) => (rng2.next() < q ? 1 : 0)));
    bys.push(c);
  }
  const Xb = Tensor.fromNested(bins), yb = T1(bys);
  const bnb = O('ml.nb.bernoulliFit', [Xb, yb, 1]);
  assert.ok(accuracy(O('ml.nb.bernoulliPredict', [bnb, Xb]), yb) > 0.9,
    `伯努利 NB 二值数据准确率应 > 0.9，实际 ${accuracy(O('ml.nb.bernoulliPredict', [bnb, Xb]), yb)}`);
});

// =====================================================================
test('7. k-NN：k=1 时训练集误差为 0，概率行和为 1', () => {
  const kn = O('ml.knn.fit', [B3.X, B3.y]);
  assert.equal(accuracy(O('ml.knn.predict', [kn, B3.X, 1]), B3.y), 1);
  const P = O('ml.knn.predictProba', [kn, B3.X, 5]);
  const [m, K] = P.shape;
  for (let i = 0; i < m; i++) {
    let s = 0;
    for (let k = 0; k < K; k++) s += P.data[i * K + k];
    assert.ok(approx(s, 1, 1e-10), `概率行和 = ${s}`);
  }
  // 回归：k=1 时预测值等于训练标签
  const kr = O('ml.knn.fit', [REG.X, REG.y]);
  const out = V(O('ml.knn.regress', [kr, REG.X, 1]));
  V(REG.y).forEach((v, i) => assert.ok(approx(out[i], v, 1e-12)));
});

// =====================================================================
test('8. CART 在 XOR 上训练误差为 0，depth/leafCount 与实测一致', () => {
  const t = O('ml.tree.fit', [XOR.X, XOR.y, 12, 2, 'auto']);
  assert.equal(accuracy(O('ml.tree.predict', [t, XOR.X]), XOR.y), 1);
  const depth = O('ml.tree.depth', [t]);
  const leaves = O('ml.tree.leafCount', [t]);
  // 独立复算：遍历 state.root 统计深度与叶数
  const walk = (node) => {
    if (node.leaf) return { d: 0, l: 1 };
    const a = walk(node.left), b = walk(node.right);
    return { d: 1 + Math.max(a.d, b.d), l: a.l + b.l };
  };
  const w = walk(t.root);
  assert.equal(depth, w.d, 'depth 与树结构实测不一致');
  assert.equal(leaves, w.l, 'leafCount 与树结构实测不一致');
  assert.ok(depth <= 12 && leaves >= 2 && leaves <= 2 ** depth, 'depth/leaves 关系越界');
  // 分类树概率：行和为 1 且 argmax 等于硬预测
  const Pb = O('ml.tree.predictProba', [t, XOR.X]);
  const hard = V(O('ml.tree.predict', [t, XOR.X]));
  const [m, K] = Pb.shape;
  for (let i = 0; i < m; i++) {
    let s = 0, bi = 0;
    for (let k = 0; k < K; k++) {
      s += Pb.data[i * K + k];
      if (Pb.data[i * K + k] > Pb.data[i * K + bi]) bi = k;
    }
    assert.ok(approx(s, 1, 1e-12), '叶概率行和应为 1');
    assert.equal(t.classes[bi], hard[i], '概率 argmax 与硬预测不一致');
  }
  // 特征重要性非负且和为 1
  const imp = V(O('ml.tree.importance', [t]));
  assert.equal(imp.length, 2);
  assert.ok(imp.every((v) => v >= 0));
  assert.ok(approx(imp.reduce((a, b) => a + b, 0), 1, 1e-10));
});

// =====================================================================
test('9. k-means 在 3 个明显分离的簇上 ARI = 1', () => {
  const km = O('ml.kmeans.fit', [B3.X, 3, 42, 100, 'kmeanspp']);
  const labels = T1(km.labels);
  const ari = O('ml.metric.ari', [B3.y, labels]);
  assert.ok(ari >= 0.99, `ARI 应为 1，实际 ${ari}`);
  // 惯量一致性与非负性
  assert.ok(km.inertia > 0);
  assert.ok(approx(O('ml.kmeans.inertia', [km, B3.X]), km.inertia, 1e-10));
  // predict 与 fit 标签一致（同一批数据）
  assert.deepEqual(V(O('ml.kmeans.predict', [km, B3.X])), km.labels);
  // k-means++ 初始化：质心取自样本
  const C = O('ml.kmeanspp.init', [B3.X, 3, 7]);
  assert.deepEqual(C.shape, [3, 2]);
  const pts = B3.X.toArray();
  C.toArray().forEach((c) => {
    assert.ok(pts.some((p) => approx(p[0], c[0], 1e-12) && approx(p[1], c[1], 1e-12)), '质心应取自训练样本');
  });
  // 谱聚类同样能分离三簇
  const sp = O('ml.spectral.fit', [B3.X, 3, 0.05]);
  assert.ok(O('ml.metric.ari', [B3.y, O('ml.spectral.labels', [sp])]) >= 0.99);
});

// =====================================================================
test('10. DBSCAN 识别噪声点（label = −1）并给出 ≥ 2 个簇', () => {
  // 两个紧致团 + 40 个均匀散布的噪声点
  const rng = createRng(9);
  const pts = blobs({ n: 60, centers: [[0, 0], [10, 10]], std: 0.35, seed: 5 }).X.toArray();
  for (let i = 0; i < 40; i++) pts.push([rng.range(-15, 25), rng.range(-15, 25)]);
  const X = Tensor.fromNested(pts);
  const db = O('ml.dbscan.fit', [X, 0.8, 4]);
  const labels = db.labels;
  assert.ok(labels.includes(-1), 'DBSCAN 应识别出噪声点 (−1)');
  assert.ok(db.nClusters >= 2, `应至少得到 2 个簇，实际 ${db.nClusters}`);
  // 团内点不应被判为噪声
  const coreLabels = labels.slice(0, 60);
  assert.ok(coreLabels.every((v) => v !== -1), '紧致团内不应出现噪声');
  // predict：团内点仍被指派，孤立远点判为 −1
  const pr = V(O('ml.dbscan.predict', [db, X]));
  assert.ok(pr.slice(0, 60).every((v) => v !== -1));
  assert.ok(pr.includes(-1), 'predict 应能给出 −1');
});

// =====================================================================
test('11. PCA：得分协方差非对角元 ≈ 0、explainedVar 之和 = 总方差、inverse∘transform ≈ id', () => {
  const X = correlated3();
  const d = 3, n = X.shape[0];
  const pca = O('ml.pca.fit', [X, d]);
  const Z = O('ml.pca.transform', [pca, X]);
  assert.deepEqual(Z.shape, [n, d]);
  // 得分协方差：非对角元 ≈ 0，对角元 = 特征值
  const { C } = covMat(Z);
  const ev = V(O('ml.pca.explainedVar', [pca]));
  for (let a = 0; a < d; a++) {
    for (let b = 0; b < d; b++) {
      const v = C.data[a * d + b];
      if (a === b) assert.ok(approx(v, ev[a], 1e-8), `得分方差应等于特征值 λ_${a}: ${v} vs ${ev[a]}`);
      else assert.ok(Math.abs(v) < 1e-10, `得分协方差非对角元应 ≈ 0，实际 ${v}`);
    }
  }
  // explainedVar（nComp = d）之和 = 总方差
  const total = covMat(X).C;
  let trace = 0;
  for (let j = 0; j < d; j++) trace += total.data[j * d + j];
  assert.ok(approx(ev.reduce((a, b) => a + b, 0), trace, 1e-9), '解释方差之和应等于总方差（协方差迹）');
  // 特征值降序
  for (let a = 1; a < d; a++) assert.ok(ev[a] <= ev[a - 1] + 1e-12, '解释方差应降序');
  // inverse(transform(x)) = x（nComp = d 时精确）
  const Xh = O('ml.pca.inverse', [pca, Z]);
  let md = 0;
  for (let i = 0; i < X.size; i++) md = Math.max(md, Math.abs(Xh.data[i] - X.data[i]));
  assert.ok(md < 1e-10, `重构误差过大: ${md}`);
  // 白化后协方差 ≈ 单位阵
  const W = O('ml.whiten', [X]);
  const cw = covMat(W).C;
  for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) {
    assert.ok(approx(cw.data[a * d + b], a === b ? 1 : 0, 1e-8),
      `白化协方差应为单位阵，[${a},${b}]=${cw.data[a * d + b]}`);
  }
});

// =====================================================================
test('12. PCA 特征值与 linalg.eigenSym（协方差矩阵）一致', () => {
  const X = correlated3();
  const pca = O('ml.pca.fit', [X, 3]);
  const eig = O('linalg.eigenSym', [covMat(X).C]);
  const ref = V(eig.values);
  const mine = V(O('ml.pca.explainedVar', [pca]));
  for (let k = 0; k < 3; k++) {
    assert.ok(approx(mine[k], ref[k], 1e-9), `第 ${k} 个特征值: ${mine[k]} vs ${ref[k]}`);
  }
  // 载荷向量是单位正交的特征向量
  const W = pca.components;
  for (let a = 0; a < W.length; a++) {
    assert.ok(approx(Math.hypot(...W[a]), 1, 1e-10), '载荷向量应为单位向量');
    for (let b = a + 1; b < W.length; b++) {
      let s = 0;
      for (let j = 0; j < 3; j++) s += W[a][j] * W[b][j];
      assert.ok(Math.abs(s) < 1e-10, `载荷向量应正交，内积 ${s}`);
    }
  }
  // 核 PCA：训练集自身投影的方差 ≈ 核特征值
  const kp = O('ml.kpca.fit', [X, 2, 0.5]);
  const Zk = O('ml.kpca.transform', [kp, X]);
  assert.deepEqual(Zk.shape, [X.shape[0], 2]);
  const ck = covMat(Zk).C;
  kp.eigenvalues.forEach((lam, k) => {
    assert.ok(approx(ck.data[k * 2 + k], lam, 1e-8), `核主成分方差应等于核特征值: ${ck.data[k * 2 + k]} vs ${lam}`);
  });
});

// =====================================================================
test('13. 标准化后每列均值 ≈ 0、标准差 ≈ 1；minmax 后 ∈ [0,1]', () => {
  const rng = createRng(4);
  const rows = [];
  for (let i = 0; i < 60; i++) rows.push([rng.range(-5, 15), rng.range(0, 3), rng.range(-1, 1)]);
  const X = Tensor.fromNested(rows);
  const d = 3, n = 60;
  // standard
  const st = O('ml.scale.standard', [X]);
  for (let j = 0; j < d; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += st.X.data[i * d + j];
    assert.ok(Math.abs(s / n) < 1e-12, `第 ${j} 列均值应 ≈ 0，实际 ${s / n}`);
    const m = s / n;
    let v = 0;
    for (let i = 0; i < n; i++) v += (st.X.data[i * d + j] - m) ** 2;
    assert.ok(approx(Math.sqrt(v / n), 1, 1e-12), `第 ${j} 列标准差应 ≈ 1`);
  }
  // minmax
  const mm = O('ml.scale.minmax', [X, 0, 1]);
  for (const v of V(mm.X)) assert.ok(v >= 0 && v <= 1, `minmax 后应 ∈ [0,1]，实际 ${v}`);
  for (let j = 0; j < d; j++) {
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < n; i++) {
      mn = Math.min(mn, mm.X.data[i * d + j]);
      mx = Math.max(mx, mm.X.data[i * d + j]);
    }
    assert.ok(approx(mn, 0, 1e-12) && approx(mx, 1, 1e-12), `第 ${j} 列应铺满 [0,1]`);
  }
  // robust：中位数 ≈ 0
  const rb = O('ml.scale.robust', [X]);
  for (let j = 0; j < d; j++) {
    const col = [];
    for (let i = 0; i < n; i++) col.push(rb.X.data[i * d + j]);
    col.sort((a, b) => a - b);
    const med = (col[29] + col[30]) / 2;
    assert.ok(Math.abs(med) < 1e-10, `稳健缩放后中位数应 ≈ 0，实际 ${med}`);
  }
  // maxabs：|x| ≤ 1 且每列最大绝对值为 1
  const ma = O('ml.scale.maxabs', [X]);
  for (let j = 0; j < d; j++) {
    let mx = 0;
    for (let i = 0; i < n; i++) mx = Math.max(mx, Math.abs(ma.X.data[i * d + j]));
    assert.ok(approx(mx, 1, 1e-12), `第 ${j} 列最大绝对值应为 1`);
  }
  // normalize：行 L2 范数为 1
  const nz = O('ml.scale.normalize', [X]);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < d; j++) s += nz.X.data[i * d + j] ** 2;
    assert.ok(approx(Math.sqrt(s), 1, 1e-12), '行范数应为 1');
  }
  // 编码与缺失值填补
  const oh = O('ml.encode.onehot', [T1([0, 1, 2, 1]), 3]);
  assert.deepEqual(oh.toArray(), [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]);
  const le = O('ml.encode.label', [T1([5, 5, -1, 2])]);
  assert.deepEqual(le.classes, [-1, 2, 5]);
  V(le.codes).forEach((c, i) => assert.equal(le.classes[c], [5, 5, -1, 2][i]));
  const Xn = X.clone();
  Xn.data[0] = NaN; Xn.data[7] = NaN;
  const im = O('ml.impute.mean', [Xn]);
  assert.ok(V(im.X).every(Number.isFinite), '均值填补后不应有 NaN');
  const im2 = O('ml.impute.median', [Xn]);
  assert.ok(V(im2.X).every(Number.isFinite), '中位数填补后不应有 NaN');
  assert.ok(approx(im.X.data[0], im.fill[0], 1e-15) && approx(im2.X.data[0], im2.fill[0], 1e-15));
});

// =====================================================================
test('14. 层次聚类三种 linkage 在 4 点上满足单调链性质且 single ≤ average ≤ complete', () => {
  const P = Tensor.fromNested([[0, 0], [3, 0], [0, 4], [9, 9]]);
  const S = O('ml.linkage.single', [P]);
  const Cp = O('ml.linkage.complete', [P]);
  const A = O('ml.linkage.average', [P]);
  for (const r of [S, Cp, A]) {
    assert.equal(r.Z.length, 3, 'n=4 时应有 3 次合并');
    assert.equal(r.n, 4);
    // 单调链：合并距离非减
    for (let i = 1; i < r.Z.length; i++) {
      assert.ok(r.Z[i][2] >= r.Z[i - 1][2] - 1e-12,
        `${r.method} 合并距离应单调不减: ${r.Z[i - 1][2]} -> ${r.Z[i][2]}`);
    }
    // 簇大小递增，最后一次为 4
    assert.equal(r.Z[2][3], 4);
    // scipy 约定：单点簇 id ∈ [0, n)
    r.Z.forEach((z) => { assert.ok(z[0] >= 0 && z[1] >= 0); });
  }
  for (let i = 0; i < 3; i++) {
    assert.ok(S.Z[i][2] <= A.Z[i][2] + 1e-12, `single ≤ average（第 ${i} 次合并）`);
    assert.ok(A.Z[i][2] <= Cp.Z[i][2] + 1e-12, `average ≤ complete（第 ${i} 次合并）`);
  }
  // 具体距离核对：最近两点距离 3，最远两点距离 √(81+81)
  assert.ok(approx(S.Z[0][2], 3, 1e-12), `单链接首次合并应为 3，实际 ${S.Z[0][2]}`);
  assert.ok(approx(Cp.Z[2][2], Math.sqrt(162), 1e-12), `全链接最后合并应为 √162，实际 ${Cp.Z[2][2]}`);
  // 切成 2 簇：远点 [9,9] 自成一簇
  const ag = O('ml.agglo.fit', [P, 2, 'complete']);
  const lab = V(O('ml.agglo.labels', [ag]));
  assert.equal(new Set(lab).size, 2);
  assert.notEqual(lab[3], lab[0], '远点应与前三点不同簇');
  assert.equal(lab[0], lab[1]);
});

// =====================================================================
test('15. RBF 核矩阵：对称、对角 = 1、半正定；kernel.center 行列和为 0', () => {
  const n = B2.X.shape[0];
  const K = O('ml.kernel.rbf', [B2.X, B2.X, 0.25]);
  assert.deepEqual(K.shape, [n, n]);
  for (let i = 0; i < n; i++) {
    assert.ok(approx(K.data[i * n + i], 1, 1e-15), `对角元应为 1，实际 ${K.data[i * n + i]}`);
    for (let j = 0; j < n; j++) {
      assert.ok(approx(K.data[i * n + j], K.data[j * n + i], 1e-15), '核矩阵应对称');
      assert.ok(K.data[i * n + j] > 0 && K.data[i * n + j] <= 1, 'RBF 元素应 ∈ (0,1]');
    }
  }
  const eig = O('linalg.eigenSym', [K]);
  const minEig = Math.min(...V(eig.values));
  assert.ok(minEig >= -1e-8, `核矩阵应半正定，最小特征值 ${minEig}`);
  // 中心化：行和列和均为 0
  const Kc = O('ml.kernel.center', [K]);
  for (let i = 0; i < n; i++) {
    let rs = 0, cs = 0;
    for (let j = 0; j < n; j++) { rs += Kc.data[i * n + j]; cs += Kc.data[j * n + i]; }
    assert.ok(Math.abs(rs) < 1e-10 && Math.abs(cs) < 1e-10, `中心化行列和应 ≈ 0: ${rs}, ${cs}`);
  }
  // 其余核的形状与基本性质
  assert.deepEqual(O('ml.kernel.linear', [B2.X, B2.X]).shape, [n, n]);
  assert.deepEqual(O('ml.kernel.poly', [B2.X, B2.X, 3, 1, 1]).shape, [n, n]);
  const Kl = O('ml.kernel.laplacian', [B2.X, B2.X, 0.25]);
  for (let i = 0; i < n; i++) assert.ok(approx(Kl.data[i * n + i], 1, 1e-15), '拉普拉斯核对角应为 1');
  const Kc2 = O('ml.kernel.cosine', [B2.X, B2.X]);
  for (let i = 0; i < n; i++) {
    assert.ok(approx(Kc2.data[i * n + i], 1, 1e-12), '余弦核对角应为 1');
    for (let j = 0; j < n; j++) assert.ok(Math.abs(Kc2.data[i * n + j]) <= 1 + 1e-12, '余弦值应 ∈ [−1,1]');
  }
  // 核 SVM 能拟合线性可分数据且 α ∈ [0,C]
  const ks = O('ml.svm.kernelFit', [SEP.X, SEP.y, 1, 0.5, 1e-3, 80]);
  assert.ok(ks.nSV > 0, '应存在支持向量');
  ks.svAlpha.forEach((a) => assert.ok(a >= -1e-9 && a <= ks.C + 1e-9, `α 应在 [0,C] 内，实际 ${a}`));
  assert.equal(accuracy(O('ml.svm.predict', [ks, SEP.X]), SEP.y), 1);
  const kd = V(O('ml.svm.decision', [ks, SEP.X]));
  V(SEP.y).forEach((lab, i) => {
    const want = lab === ks.posLabel ? 1 : -1;
    assert.ok(kd[i] * want > 0, `核 SVM 决策符号错误: ${kd[i]}`);
  });
});

// =====================================================================
test('16. 指标：r2 完美预测 = 1、均值基线 = 0；silhouette ∈ [−1,1]；ARI/CH 正确', () => {
  const y = REG.y;
  assert.equal(O('ml.metric.mse', [y, y]), 0);
  assert.equal(O('ml.metric.rmse', [y, y]), 0);
  assert.equal(O('ml.metric.mae', [y, y]), 0);
  assert.ok(approx(O('ml.metric.r2', [y, y]), 1, 1e-12), '完美预测的 R² 应为 1');
  // 常数基线（= 均值）预测时 R² = 0
  const yv = V(y), m = yv.reduce((a, b) => a + b, 0) / yv.length;
  assert.ok(approx(O('ml.metric.r2', [y, T1(yv.map(() => m))]), 0, 1e-12), '均值基线预测 R² 应为 0');
  // 一致性与量纲
  const pred = O('ml.linreg.predict', [O('ml.linreg.fit', [REG.X, REG.y]), REG.X]);
  const mse = O('ml.metric.mse', [y, pred]);
  assert.ok(approx(O('ml.metric.rmse', [y, pred]), Math.sqrt(mse), 1e-12), 'RMSE = √MSE');
  assert.ok(O('ml.metric.mae', [y, pred]) <= O('ml.metric.rmse', [y, pred]) + 1e-12, 'MAE ≤ RMSE');
  // 轮廓系数
  const km = O('ml.kmeans.fit', [B3.X, 3, 42, 100, 'kmeanspp']);
  const sil = O('ml.metric.silhouette', [B3.X, T1(km.labels)]);
  assert.ok(sil.score >= -1 && sil.score <= 1, `轮廓系数应 ∈ [−1,1]，实际 ${sil.score}`);
  assert.ok(sil.score > 0.5, `三簇分离良好时轮廓系数应较高，实际 ${sil.score}`);
  for (const s of V(sil.samples)) assert.ok(s >= -1 && s <= 1, `逐样本轮廓应 ∈ [−1,1]，实际 ${s}`);
  // ARI：自身 = 1；置换标签不变；单簇时退化为 0
  assert.ok(approx(O('ml.metric.ari', [B3.y, B3.y]), 1, 1e-12));
  const perm = T1(V(B3.y).map((v) => (v === 0 ? 7 : v === 1 ? 8 : 9)));
  assert.ok(approx(O('ml.metric.ari', [B3.y, perm]), 1, 1e-12), 'ARI 对标签重命名不变');
  assert.equal(O('ml.metric.ari', [B3.y, T1(B3.y.toArray().map(() => 0))]), 0);
  // Calinski-Harabasz：正确簇数下远大于 1
  const ch = O('ml.metric.calinski', [B3.X, T1(km.labels)]);
  assert.ok(ch > 1, `CH 指数应显著为正，实际 ${ch}`);
  assert.equal(O('ml.metric.calinski', [B3.X, T1(B3.y.toArray().map(() => 0))]), 0, '单簇时 CH 无定义（返回 0）');
  // 特征选择：与标签强相关的特征得分最高
  const ft = O('ml.select.fTest', [B3.X, B3.y]);
  assert.equal(ft.scores.shape[0], 2);
  assert.ok(V(ft.scores).every((v) => v >= 0), 'F 统计量应非负');
  assert.deepEqual(ft.order.length, 2);
  const mi = O('ml.select.mutualInfo', [B3.X, B3.y, 10]);
  assert.ok(V(mi.scores).every((v) => v >= -1e-12), '互信息应非负');
  const vs = O('ml.select.variance', [B3.X, 0.01]);
  assert.deepEqual(vs.index, [0, 1], '两列均非常数应被保留');
  // 常数列应被阈值剔除
  const Xc = Tensor.fromNested(Array.from({ length: 20 }, (_, i) => [i, 0]));
  assert.deepEqual(O('ml.select.variance', [Xc, 1e-9]).index, [0], '常数列应被剔除');
});

// =====================================================================
test('17. 工具：clone 深拷贝、getParams/setParams/describe 一致', () => {
  const km = O('ml.kmeans.fit', [B2.X, 2, 1, 50, 'kmeanspp']);
  const cp = O('ml.util.clone', [km]);
  assert.deepEqual(cp, km, '克隆应值相等');
  assert.notEqual(cp.centers, km.centers, '克隆不应共享引用');
  const params = O('ml.util.getParams', [km]);
  assert.equal(params.k, 2);
  assert.equal(params.kind, 'ml.kmeans');
  assert.ok(!('centers' in params), 'getParams 只返回标量参数');
  const patched = O('ml.util.setParams', [km, { k: 7 }]);
  assert.equal(patched.k, 7);
  assert.equal(km.k, 2, 'setParams 不应修改原 state');
  const desc = O('ml.util.describe', [km]);
  assert.equal(desc.kind, 'ml.kmeans');
  assert.deepEqual(desc.keys, Object.keys(km));
  assert.deepEqual(desc.params, params);
  assert.ok(desc.bytes > 0);
});

// =====================================================================
test('18. 集成方法冒烟：RF/GB/Ada 形状正确且 GB 训练损失单调不增', () => {
  const rf = O('ml.rf.fit', [XOR.X, XOR.y, 20, 10, 7]);
  const rp = O('ml.rf.predict', [rf, XOR.X]);
  assert.deepEqual(rp.shape, [XOR.X.shape[0]]);
  assert.ok(accuracy(rp, XOR.y) > 0.9, `随机森林在 XOR 上准确率应 > 0.9，实际 ${accuracy(rp, XOR.y)}`);
  const ri = V(O('ml.rf.importance', [rf]));
  assert.equal(ri.length, 2);
  assert.ok(ri.every((v) => v >= 0) && approx(ri.reduce((a, b) => a + b, 0), 1, 1e-10));

  const gb = O('ml.gb.fit', [REG.X, REG.y, 60, 0.1, 3]);
  const gp = O('ml.gb.predict', [gb, REG.X]);
  assert.deepEqual(gp.shape, [REG.X.shape[0]]);
  const loss = V(O('ml.gb.stagedLoss', [gb, REG.X, REG.y]));
  assert.equal(loss.length, 60);
  // 说明：梯度提升带 shrinkage 的贪心分裂在极小损失处可能出现 ≤1e-12 的数值平台，
  // 故以「不增 + 1e-12 容差」判定，并额外要求末值远小于初值。
  for (let i = 1; i < loss.length; i++) {
    assert.ok(loss[i] <= loss[i - 1] + 1e-12, `第 ${i} 轮损失上升: ${loss[i - 1]} -> ${loss[i]}`);
  }
  assert.ok(loss[loss.length - 1] < loss[0] * 0.1, 'GB 训练损失应显著下降');
  // 提升后拟合优度应优于常数基线
  assert.ok(O('ml.metric.r2', [REG.y, gp]) > 0.9);

  const ad = O('ml.ada.fit', [B2.X, B2.y, 30]);
  assert.deepEqual(O('ml.ada.predict', [ad, B2.X]).shape, [B2.X.shape[0]]);
  assert.ok(accuracy(O('ml.ada.predict', [ad, B2.X]), B2.y) > 0.9,
    `AdaBoost 在两团数据上准确率应 > 0.9，实际 ${accuracy(O('ml.ada.predict', [ad, B2.X]), B2.y)}`);
  assert.ok(ad.learners.length > 0 && ad.learners.every((l) => l.alpha > 0));
});

// =====================================================================
test('19. 全部 fit/predict 对冒烟：不抛异常且输出形状正确', () => {
  const n2 = B2.X.shape[0], n3 = B3.X.shape[0], nr = REG.X.shape[0], nx = XOR.X.shape[0];
  const fitPairs = [
    ['ml.linreg.fit', [REG.X, REG.y], 'ml.linreg.predict', nr, []],
    ['ml.ridge.fit', [REG.X, REG.y, 0.5], 'ml.ridge.predict', nr, []],
    ['ml.lasso.fit', [REG.X, REG.y, 0.1, 200], 'ml.lasso.predict', nr, []],
    ['ml.elasticNet.fit', [REG.X, REG.y, 0.1, 0.5, 200], 'ml.elasticNet.predict', nr, []],
    ['ml.poly.fit', [REG.X, REG.y, 3], 'ml.poly.predict', nr, []],
    ['ml.robust.fit', [REG.X, REG.y, 1.345, 25], 'ml.robust.predict', nr, []],
    ['ml.logreg.fit', [B2.X, B2.y, 100, 1e-8], 'ml.logreg.predict', n2, []],
    ['ml.softmax.fit', [B3.X, B3.y, 0.3, 200], 'ml.softmax.predict', n3, []],
    ['ml.perceptron.fit', [B2.X, B2.y, 0.1, 100], 'ml.perceptron.predict', n2, []],
    ['ml.lda.fit', [B3.X, B3.y], 'ml.lda.predict', n3, []],
    ['ml.qda.fit', [B3.X, B3.y], 'ml.qda.predict', n3, []],
    ['ml.nb.gaussianFit', [B3.X, B3.y], 'ml.nb.gaussianPredict', n3, []],
    ['ml.nb.multinomialFit', [B3.X, B3.y, 1], 'ml.nb.multinomialPredict', n3, []],
    ['ml.nb.bernoulliFit', [B3.X, B3.y, 1], 'ml.nb.bernoulliPredict', n3, []],
    ['ml.tree.fit', [XOR.X, XOR.y, 6, 2, 'auto'], 'ml.tree.predict', nx, []],
    ['ml.rf.fit', [XOR.X, XOR.y, 8, 6, 1], 'ml.rf.predict', nx, []],
    ['ml.gb.fit', [REG.X, REG.y, 10, 0.1, 3], 'ml.gb.predict', nr, []],
    ['ml.ada.fit', [B3.X, B3.y, 10], 'ml.ada.predict', n3, []],
    ['ml.svm.fit', [B2.X, B2.y, 1, 100, 0.01], 'ml.svm.predict', n2, []],
    ['ml.svm.kernelFit', [B2.X, B2.y, 1, 0.3, 1e-3, 40], 'ml.svm.predict', n2, []],
    ['ml.knn.fit', [B3.X, B3.y], 'ml.knn.predict', n3, [5]],
    ['ml.kmeans.fit', [B3.X, 3, 1, 50, 'kmeanspp'], 'ml.kmeans.predict', n3, []],
    ['ml.dbscan.fit', [B3.X, 1.5, 4], 'ml.dbscan.predict', n3, []],
    ['ml.agglo.fit', [B3.X, 3, 'average'], 'ml.agglo.labels', n3, []],
    ['ml.spectral.fit', [B3.X, 3, 0.05], 'ml.spectral.labels', n3, []],
    ['ml.pca.fit', [B3.X, 2], 'ml.pca.transform', n3, []],
  ];
  for (const [fit, fitArgs, pred, expectLen, extra] of fitPairs) {
    const st = O(fit, fitArgs);
    // state 必须为可 JSON 序列化的纯对象
    assert.equal(typeof st, 'object', `${fit} 应返回对象 state`);
    assert.doesNotThrow(() => JSON.stringify(st), `${fit} 的 state 必须可 JSON 序列化`);
    const arg = pred === 'ml.agglo.labels' || pred === 'ml.spectral.labels' ? [st] : [st, fitArgs[0]];
    const out = O(pred, arg.concat(extra));
    assert.ok(out instanceof Tensor, `${pred} 应返回 Tensor`);
    assert.equal(out.shape[0], expectLen, `${pred} 输出行数应为 ${expectLen}，实际 ${out.shape[0]}`);
    assert.ok(Array.from(out.data).every(Number.isFinite), `${pred} 输出应全部有限`);
  }
  // transform 型指令的形状冒烟
  const pca = O('ml.pca.fit', [B3.X, 2]);
  assert.deepEqual(O('ml.pca.transform', [pca, B3.X]).shape, [n3, 2]);
  assert.deepEqual(O('ml.pca.inverse', [pca, O('ml.pca.transform', [pca, B3.X])]).shape, [n3, 2]);
  assert.deepEqual(O('ml.pca.explainedVar', [pca]).shape, [2]);
  const kp = O('ml.kpca.fit', [B3.X, 2, 0.2]);
  assert.deepEqual(O('ml.kpca.transform', [kp, B3.X]).shape, [n3, 2]);
  const D = Tensor.fromNested([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]]);
  assert.deepEqual(O('ml.mds.classical', [D, 2]).shape, [3, 2]);
  assert.deepEqual(O('ml.whiten', [B3.X]).shape, [n3, 2]);
  // 预处理 / 特征选择 / 指标 / 工具
  for (const name of ['ml.scale.standard', 'ml.scale.robust', 'ml.scale.maxabs', 'ml.scale.normalize',
    'ml.impute.mean', 'ml.impute.median']) {
    const r = O(name, [B3.X]);
    assert.deepEqual(r.X.shape, [n3, 2], `${name} 输出形状错误`);
  }
  assert.deepEqual(O('ml.scale.minmax', [B3.X, 0, 1]).X.shape, [n3, 2]);
  assert.deepEqual(O('ml.encode.onehot', [B3.y, 3]).shape, [n3, 3]);
  assert.deepEqual(O('ml.select.variance', [B3.X, 0]).scores.shape, [2]);
  assert.deepEqual(O('ml.select.fTest', [B3.X, B3.y]).scores.shape, [2]);
  assert.deepEqual(O('ml.select.mutualInfo', [B3.X, B3.y, 8]).scores.shape, [2]);
  for (const name of ['ml.metric.mse', 'ml.metric.rmse', 'ml.metric.mae', 'ml.metric.r2']) {
    assert.equal(typeof O(name, [REG.y, REG.y]), 'number', `${name} 应返回 number`);
  }
  assert.equal(typeof O('ml.metric.calinski', [B3.X, B3.y]), 'number');
  assert.equal(typeof O('ml.metric.ari', [B3.y, B3.y]), 'number');
  assert.equal(typeof O('ml.tree.depth', [O('ml.tree.fit', [XOR.X, XOR.y, 4, 2, 'auto'])]), 'number');
  assert.equal(typeof O('ml.tree.leafCount', [O('ml.tree.fit', [XOR.X, XOR.y, 4, 2, 'auto'])]), 'number');
  assert.deepEqual(O('ml.tree.importance', [O('ml.tree.fit', [XOR.X, XOR.y, 4, 2, 'auto'])]).shape, [2]);
  assert.deepEqual(O('ml.tree.predictProba', [O('ml.tree.fit', [XOR.X, XOR.y, 4, 2, 'auto']), XOR.X]).shape, [nx, 2]);
  assert.deepEqual(O('ml.linkage.single', [B3.X]).Z.length, n3 - 1);
  assert.deepEqual(O('ml.linkage.complete', [B3.X]).Z.length, n3 - 1);
  assert.deepEqual(O('ml.linkage.average', [B3.X]).Z.length, n3 - 1);
  assert.deepEqual(O('ml.kmeanspp.init', [B3.X, 3, 1]).shape, [3, 2]);
  assert.equal(typeof O('ml.kmeans.inertia', [O('ml.kmeans.fit', [B3.X, 3, 1, 50, 'kmeanspp']), B3.X]), 'number');
  assert.deepEqual(O('ml.knn.predictProba', [O('ml.knn.fit', [B3.X, B3.y]), B3.X, 5]).shape, [n3, 3]);
  assert.deepEqual(O('ml.knn.regress', [O('ml.knn.fit', [REG.X, REG.y]), REG.X, 3]).shape, [nr]);
  assert.deepEqual(O('ml.logreg.predictProba', [O('ml.logreg.fit', [B2.X, B2.y, 50, 1e-8]), B2.X]).shape, [n2]);
  assert.deepEqual(O('ml.softmax.predictProba', [O('ml.softmax.fit', [B3.X, B3.y, 0.3, 100]), B3.X]).shape, [n3, 3]);
  assert.deepEqual(O('ml.svm.decision', [O('ml.svm.fit', [B2.X, B2.y, 1, 50, 0.01]), B2.X]).shape, [n2]);
  assert.deepEqual(O('ml.gb.stagedLoss', [O('ml.gb.fit', [REG.X, REG.y, 5, 0.1, 2]), REG.X, REG.y]).shape, [5]);
  assert.deepEqual(O('ml.rf.importance', [O('ml.rf.fit', [B3.X, B3.y, 5, 5, 1])]).shape, [2]);
  assert.deepEqual(O('ml.kernel.linear', [B2.X, B2.X]).shape, [n2, n2]);
  assert.deepEqual(O('ml.kernel.rbf', [B2.X, B2.X, 0.2]).shape, [n2, n2]);
  assert.deepEqual(O('ml.kernel.poly', [B2.X, B2.X, 2, 1, 1]).shape, [n2, n2]);
  assert.deepEqual(O('ml.kernel.laplacian', [B2.X, B2.X, 0.2]).shape, [n2, n2]);
  assert.deepEqual(O('ml.kernel.cosine', [B2.X, B2.X]).shape, [n2, n2]);
  assert.deepEqual(O('ml.kernel.center', [O('ml.kernel.rbf', [B2.X, B2.X, 0.2])]).shape, [n2, n2]);
  assert.equal(typeof O('ml.util.clone', [O('ml.kmeans.fit', [B2.X, 2, 1, 20, 'kmeanspp'])]), 'object');
  assert.equal(typeof O('ml.util.getParams', [O('ml.kmeans.fit', [B2.X, 2, 1, 20, 'kmeanspp'])]), 'object');
  assert.equal(typeof O('ml.util.setParams', [O('ml.kmeans.fit', [B2.X, 2, 1, 20, 'kmeanspp']), { k: 3 }]), 'object');
  assert.equal(typeof O('ml.util.describe', [O('ml.kmeans.fit', [B2.X, 2, 1, 20, 'kmeanspp'])]), 'object');
});

// =====================================================================
test('20. 可复现性：同种子结果完全一致；非法输入按错误分类学抛错', () => {
  const a = O('ml.kmeans.fit', [B3.X, 3, 2026, 60, 'kmeanspp']);
  const b = O('ml.kmeans.fit', [B3.X, 3, 2026, 60, 'kmeanspp']);
  assert.deepEqual(a.labels, b.labels, '同种子 k-means 结果应完全一致');
  assert.deepEqual(a.centers, b.centers);
  const r1 = O('ml.rf.fit', [B3.X, B3.y, 6, 5, 99]);
  const r2 = O('ml.rf.fit', [B3.X, B3.y, 6, 5, 99]);
  assert.deepEqual(V(O('ml.rf.predict', [r1, B3.X])), V(O('ml.rf.predict', [r2, B3.X])), '同种子 RF 应一致');
  // 错误分类学
  assert.throws(() => O('ml.kmeans.fit', [B3.X, 0, 1, 10, 'kmeanspp']), /E_VALUE/);
  assert.throws(() => O('ml.linreg.fit', [REG.X, T1([1, 2])]), /E_SHAPE/);
  const single = T1(V(B3.y).map(() => 0));           // 与 X 同长度的单类别标签
  assert.throws(() => O('ml.logreg.fit', [B3.X, single, 10, 1e-8]), /E_VALUE/, '单类别应报 E_VALUE');
  assert.throws(() => O('ml.lda.fit', [B3.X, single]), /E_VALUE/, '单类别 LDA 应报 E_VALUE');
  assert.throws(() => O('ml.metric.mse', [REG.y, T1([1, 2])]), /E_SHAPE/);
  assert.throws(() => O('ml.tree.fit', [XOR.X, XOR.y, 3, 2, 'bogus']), /E_VALUE/);
  // 省略必需参数应报 E_ARITY（可选参数 omission 合法，不算错误）
  assert.throws(() => O('ml.ridge.fit', [REG.X]), /E_ARITY/);
  assert.throws(() => O('ml.knn.predict', [O('ml.knn.fit', [B3.X, B3.y])]), /E_ARITY/);
});
