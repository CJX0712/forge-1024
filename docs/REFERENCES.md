# 参考文献（REFERENCES）

> 本文件由 `node tools/gen-ops-doc.mjs` 从指令元数据 `ref` 字段自动聚合生成。
> 共 754 条独立出处，覆盖 1024 条微指令。

每条微指令都携带 `ref`（论文 / 标准 / 权威实现）与 `inv`（可验证不变量），
使得任意一条实现都能被追溯到原始文献，并被独立的数学性质交叉验证。

| 出处 | 引用指令数 | 指令示例 |
|---|---|---|
| Baydin et al., Automatic Differentiation in ML, JMLR 18(153):1-43, 2018 | 31 | `ad.exp`、`ad.log`、`ad.log1p` 等 |
| IEEE 754 / ECMAScript Math | 28 | `math.sin`、`math.cos`、`math.tan` 等 |
| NumPy errstate / IEEE 754 域规则 | 24 | `num.guard.exp`、`num.guard.log`、`num.guard.log1p` 等 |
| NumPy ufunc 语义 | 20 | `math.addScalar`、`math.scalarAdd`、`math.subScalar` 等 |
| Devroye 1986, Non-Uniform Random Variate Generation（逆变换法） | 10 | `prob.uniform.sample`、`prob.exponential.sample`、`prob.laplace.sample` 等 |
| NumPy ufunc 语义 / IEEE 754 | 10 | `math.add`、`math.sub`、`math.mul` 等 |
| Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1 | 9 | `prob.exponential.cdf`、`prob.exponential.fit`、`prob.laplace.cdf` 等 |
| Kendall & Stuart, The Advanced Theory of Statistics, Vol.1 | 9 | `prob.stat.mean`、`prob.stat.variance`、`prob.stat.std` 等 |
| NumPy 轴归约 | 8 | `linalg.rowSum`、`linalg.colSum`、`linalg.rowMean` 等 |
| 对偶数一元算子 | 7 | `ad.dual.exp`、`ad.dual.log`、`ad.dual.sqrt` 等 |
| 矩阵性质标准判定 | 7 | `linalg.isSymmetric`、`linalg.isUpperTri`、`linalg.isLowerTri` 等 |
| Johnson, Kotz & Kemp, Discrete Univariate Distributions | 7 | `prob.bernoulli.cdf`、`prob.binomial.cdf`、`prob.poisson.cdf` 等 |
| double-double 算术标准 | 5 | `num.ddFromNumber`、`num.ddToNumber`、`num.ddAdd` 等 |
| Golub & Van Loan §2.2 | 5 | `linalg.normFro`、`linalg.norm1`、`linalg.normInf` 等 |
| IEEE 754 | 5 | `num.safeDiv`、`num.safeSqrt`、`num.countNaN` 等 |
| Kitzes et al. (2018) The Practice of Reproducible Research, UC Press | 5 | `train.repro.logAppend`、`train.repro.experimentCompare`、`train.repro.recordMetric` 等 |
| 对偶数前向模式（Griewank & Walther） | 4 | `ad.dual.add`、`ad.dual.sub`、`ad.dual.mul` 等 |
| 数值微分标准 | 4 | `num.gradCentral`、`num.gradForward`、`num.hessianDiag` 等 |
| Breiman et al., CART, 1984 | 4 | `ml.tree.predict`、`ml.tree.predictProba`、`ml.tree.depth` 等 |
| Demšar (2006) Statistical Comparisons of Classifiers over Multiple Data Sets, JMLR | 4 | `train.eval.resultTable`、`train.eval.compareTable`、`train.eval.rankRuns` 等 |
| Horn & Johnson | 4 | `linalg.vec`、`linalg.unvec`、`linalg.hadamard` 等 |
| Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 2 | 4 | `prob.uniform.fit`、`prob.logistic.cdf`、`prob.beta.cdf` 等 |
| Minkowski 距离族 | 4 | `math.manhattan`、`math.euclidean`、`math.sqeuclidean` 等 |
| NumPy axis reduction | 4 | `math.sumAxis`、`math.meanAxis`、`math.maxAxis` 等 |
| NumPy reduction | 4 | `math.sum`、`math.prod`、`math.min` 等 |
| Welford 1962 | 4 | `num.welfordInit`、`num.welfordUpdate`、`num.welfordFinalize` 等 |
| 回归评估标准 | 3 | `num.mae`、`num.mse`、`num.rmse` |
| 线性代数标准 | 3 | `linalg.quadForm`、`linalg.trace`、`linalg.transpose` |
| Golub & Van Loan §5.5 | 3 | `linalg.nullSpace`、`linalg.colSpace`、`linalg.rowSpace` |
| Numerical Recipes §4.1 | 3 | `num.trapz`、`num.simpson`、`num.midpoint` |
| Pedregosa et al. (2011) Scikit-learn: Machine Learning in Python, JMLR 12 | 3 | `train.data.minmaxApply`、`train.data.labelDecode`、`train.ckpt.load` |
| Perez-Riverol et al. (2016) Ten Simple Rules for Taking Advantage of Git and GitHub, PLOS CB | 3 | `train.repro.envInfo`、`train.repro.runSummary`、`train.repro.exportRun` |
| Robbins & Monro, A Stochastic Approximation Method, Ann. Math. Statist. 1951 | 3 | `opt.sgd`、`opt.sgd.step`、`dl.util.applyGrads` |
| Vaswani et al., Attention Is All You Need, NeurIPS 2017 | 3 | `dl.attn.scaledDot`、`dl.attn.multiHead`、`dl.model.transformerBlock` |
| You et al., Large Batch Training of Deep Networks (LAMB), 2017 | 3 | `opt.lamb`、`opt.lamb.step`、`opt.util.trustRatio` |
| 分类标准损失 | 2 | `ad.crossEntropy`、`ad.nll` |
| 数值分析定义 | 2 | `num.absErr`、`num.relErr` |
| 序列化约定：state 必须为纯对象+数组 | 2 | `opt.util.stateToJSON`、`opt.util.stateFromJSON` |
| Bernstein et al., signSGD: Compressed Optimisation for Non-Convex Problems, ICML 2018 | 2 | `opt.signSgd`、`opt.signSgd.step` |
| Chen et al., Symbolic Discovery of Optimization Algorithms (Lion), 2023 | 2 | `opt.lion`、`opt.lion.step` |
| Cortes & Vapnik 1995；Boser, Guyon & Vapnik 1992 | 2 | `ml.svm.predict`、`ml.svm.decision` |
| Cover & Hart 1967 | 2 | `ml.knn.predict`、`ml.knn.predictProba` |
| DeGroot, Optimal Statistical Decisions；Gelman et al., BDA ch.1 | 2 | `prob.bayes.posteriorPredictive`、`prob.bayes.evidence` |
| Dekker 1971 | 2 | `num.twoProd`、`num.quickTwoSum` |
| Devroye 1986, Non-Uniform Random Variate Generation | 2 | `prob.bernoulli.sample`、`prob.binomial.sample` |
| Dozat, Incorporating Nesterov Momentum into Adam, ICLR 2016 | 2 | `opt.nadam`、`opt.nadam.step` |
| Duchi, Hazan & Singer, Adaptive Subgradient Methods for Online Learning, JMLR 2011 | 2 | `opt.adagrad`、`opt.adagrad.step` |
| Evans & Rosenthal, Probability and Statistics（正态分布） | 2 | `prob.normal.pdf`、`prob.normal.cdf` |
| Evans & Rosenthal, Probability and Statistics（chi-square 分布） | 2 | `prob.chi2.pdf`、`prob.chi2.cdf` |
| Fike & Alonso, Derivative Analysis via Dual Numbers, 2011 | 2 | `ad.dual.primal`、`ad.dual.tangent` |
| FNV-1a（Fowler–Noll–Vo 哈希） | 2 | `sys.cfg.hash`、`sys.env.seed` |
| forge registry introspection (self-describing op catalog) | 2 | `opt.util.listOptimizers`、`opt.util.optimizerCount` |
| Golub & Van Loan §2.2.4 | 2 | `linalg.cond2`、`linalg.cond1` |
| Golub & Van Loan §3.1 | 2 | `linalg.forwardSubst`、`linalg.backSubst` |
| Golub & Van Loan §5.1 | 2 | `linalg.householder`、`linalg.givens` |
| Graves, Generating Sequences With Recurrent Neural Networks, 2013（RMSprop 形式） | 2 | `opt.rmsprop`、`opt.rmsprop.step` |
| He et al., Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet, ICCV 2015 | 2 | `dl.act.prelu`、`dl.init.heUniform` |
| Hoare (1969) An Axiomatic Basis for Computer Programming, CACM 12(10) | 2 | `train.repro.assertReproducible`、`train.util.coverageCheck` |
| Hopcroft & Ullman 1979 图算法基本量 | 2 | `sys.graph.nodeCount`、`sys.graph.edgeCount` |
| Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 2（uniform） | 2 | `prob.uniform.pdf`、`prob.uniform.cdf` |
| Kingma & Ba, Adam 论文 §7.2（Adamax 算法） | 2 | `opt.adamax`、`opt.adamax.step` |
| Kingma & Ba, Adam: A Method for Stochastic Optimization, ICLR 2015 | 2 | `opt.adam`、`opt.adam.step` |
| Loshchilov & Hutter, Decoupled Weight Decay Regularization (AdamW), ICLR 2019 | 2 | `opt.adamw`、`opt.adamw.step` |
| McCallum & Nigam, AAAI-98 | 2 | `ml.nb.multinomialPredict`、`ml.nb.bernoulliPredict` |
| Numerical Recipes §3.3 | 2 | `linalg.polyfit`、`linalg.polyval` |
| NumPy 切片 | 2 | `linalg.submatrixRows`、`linalg.submatrixCols` |
| NumPy concatenate | 2 | `math.concat`、`ad.concat` |
| NumPy diag | 2 | `linalg.diag`、`linalg.diagMat` |
| NumPy pad | 2 | `math.pad`、`ad.padConst` |
| NumPy stack | 2 | `math.stack`、`ad.stack` |
| NumPy where | 2 | `math.where`、`ad.where` |
| Polyak & Juditsky (1992) Acceleration of Stochastic Approximation by Averaging, SIAM J. Control Optim. | 2 | `train.ema.update`、`train.ema.init` |
| Polyak, Some Methods of Speeding Up the Convergence of Iteration Methods, 1964 | 2 | `opt.sgdMomentum`、`opt.sgdMomentum.step` |
| Prometheus TopK 查询语义 | 2 | `sys.tel.hotOps`、`sys.tel.topErrors` |
| PyTorch LogSoftmax | 2 | `num.logSoftmax`、`ad.logSoftmax` |
| Reddi, Kale & Kumar, On the Convergence of Adam and Beyond, ICLR 2018 | 2 | `opt.amsgrad`、`opt.amsgrad.step` |
| Riedmiller & Braun, A Direct Adaptive Method for Faster Backpropagation Learning, 1993 | 2 | `opt.rprop`、`opt.rprop.step` |
| Smith, A Disciplined Approach to Neural Network Hyper-Parameters, 2018 | 2 | `opt.lr.onecycle`、`opt.util.onecycleFactor` |
| Sokal & Michener 1958；Lance & Williams 1967 | 2 | `ml.linkage.complete`、`ml.linkage.average` |
| Sokolova & Lapalme (2009) A Systematic Analysis of Performance Measures for Classification Tasks, IP&M 45(4) | 2 | `train.eval.microAvg`、`train.eval.perClass` |
| Sutskever et al., On the Importance of Initialization and Momentum in Deep Learning, ICML 2013 | 2 | `opt.sgdNesterov`、`opt.sgdNesterov.step` |
| Zaheer et al., Adaptive Methods for Nonconvex Optimization, NeurIPS 2018 | 2 | `opt.yogi`、`opt.yogi.step` |
| Zeiler, ADADELTA: An Adaptive Learning Rate Method, 2012 | 2 | `opt.adadelta`、`opt.adadelta.step` |
|  lodash.pick 语义 / 配置最小权限原则 | 1 | `sys.cfg.pick` |
| 12-Factor App 配置约定（环境变量） | 1 | `sys.cfg.toEnv` |
| 3σ / z-score 离群处理：Iglewicz & Hoaglin (1993) How to Detect and Handle Outliers, ASQ Press | 1 | `train.data.clipOutliers` |
| 标签编码约定：Pedregosa et al. (2011) Scikit-learn, JMLR 12 | 1 | `train.data.labelEncode` |
| 标准 logistic 函数（Berkson 1944；深度学习通用门控） | 1 | `dl.act.sigmoid` |
| 标准对数似然；PyTorch nn.BCELoss | 1 | `dl.loss.bce` |
| 标准激活（LeCun et al. 1998 高效反向传播推崇 tanh） | 1 | `dl.act.tanh` |
| 表格化报告惯例：Demšar (2006) Statistical Comparisons of Classifiers over Multiple Data Sets, JMLR | 1 | `train.hist.toMarkdown` |
| 测试断言与训练早停 | 1 | `opt.util.assertConverged` |
| 测试隔离与状态复位：Hoare (1969) CACM 12(10) | 1 | `train.repro.resetState` |
| 常见混合精度与梯度累积实践（如 PyTorch GradScaler 逆缩放思想） | 1 | `opt.grad.scaleByBatch` |
| 常见训练脚本实践（如 Transformer 的 Noam 逆平方根的线性变体） | 1 | `opt.lr.linearDecay` |
| 常见训练实践（TensorFlow tf.clip_by_value） | 1 | `opt.grad.clipValue` |
| 成对求和 / BLAS 累加策略 | 1 | `num.pairwiseSum` |
| 调度器分派入口 | 1 | `opt.util.scheduleAt` |
| 对比学习惯例（Chen et al. SimCLR 2020 的投影头归一化） | 1 | `dl.norm.l2normalize` |
| 对标 PyTorch 的参数校验 | 1 | `opt.util.validateCfg` |
| 对标 PyTorch 训练循环的最小化抽象 | 1 | `opt.util.runOptimizer` |
| 对标 PyTorch optim 的 defaults 字典 | 1 | `opt.util.defaultCfg` |
| 对偶数标准构造 | 1 | `ad.dual.from` |
| 多项式/MARS 基展开：Friedman (1991) Multivariate Adaptive Regression Splines, Ann. Statist. | 1 | `train.data.polyExpand` |
| 二分类标准损失 | 1 | `ad.bce` |
| 泛函分析 / NumPy norm | 1 | `math.lp` |
| 泛化误差评估：Hastie, Tibshirani & Friedman (2009) The Elements of Statistical Learning, §7 | 1 | `train.loop.evaluate` |
| 方差稳定化变换：Box & Cox (1964) An Analysis of Transformations, JRSS-B；Tukey (1957) Ann. Math. Statist. | 1 | `train.data.logTransform` |
| 分层抽样：Kohavi (1995) A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection, IJCAI | 1 | `train.data.stratifiedBatches` |
| 分层交叉验证：Kohavi (1995) IJCAI；Sechidis, Tsoumakas & Vlahavas (2012) On the Stratification of Multi-Label Data, ECML | 1 | `train.data.stratifiedKFoldSplit` |
| 分类报告惯例：Sokolova & Lapalme (2009) A Systematic Analysis of Performance Measures for Classification Tasks, IP&M 45(4) | 1 | `train.eval.classificationReport` |
| 符号扫描（工程求根常用） | 1 | `num.rootScan` |
| 函数式数据流管线：Hughes (1984) Why Functional Programming Matters；scikit-learn Pipeline（Pedregosa et al. 2011） | 1 | `train.data.pipelineCompose` |
| 回归标准损失 | 1 | `ad.mse` |
| 混淆矩阵：Stehman (1997) Selecting and Interpreting Measures of Thematic Classification Accuracy, Remote Sens. Environ. | 1 | `train.eval.confusionReport` |
| 基准测试方法论（如 MLPerf 训练基准的最小化对照） | 1 | `opt.util.compareOptimizers` |
| 基准对比后处理 | 1 | `opt.util.bestOf` |
| 检查点/快照语义：Pedregosa et al. (2011) Scikit-learn, JMLR 12（模型持久化约定） | 1 | `train.ckpt.save` |
| 解析解：λ=(tr±√(tr²−4det))/2 | 1 | `linalg.eigen2x2` |
| 可复现构建与确定性摘要：Debian Reproducible Builds 项目（2013–）；Perez-Riverol et al. (2016) Ten Simple Rules for Taking Advantage of Git and GitHub, PLOS CB | 1 | `train.repro.fingerprint` |
| 链式法则 / Golub & Van Loan | 1 | `ad.matmul` |
| 幂法经典；Golub & Van Loan §8.2 | 1 | `linalg.powerIteration` |
| 模型参数量统计惯例（PyTorch numel 求和） | 1 | `dl.util.paramCount` |
| 前向模式归约 | 1 | `ad.dual.sum` |
| 前向模式矩阵法则 | 1 | `ad.dual.matmul` |
| 三向 holdout 惯例：Hastie, Tibshirani & Friedman (2009) The Elements of Statistical Learning, §7 | 1 | `train.loop.trainValTest` |
| 深度学习标准 | 1 | `ad.softmax` |
| 深度学习标准层 | 1 | `ad.dense` |
| 深度学习标准激活 | 1 | `ad.hardtanh` |
| 深度学习标准实践 / cs231n | 1 | `num.softmax` |
| 时间序列滞后特征与窗口：Box & Jenkins (1970) Time Series Analysis: Forecasting and Control | 1 | `train.data.windowIterator` |
| 收敛度量标准定义 | 1 | `opt.util.paramDistance` |
| 数据矩阵约定：Pedregosa et al. (2011) Scikit-learn: Machine Learning in Python, JMLR 12 | 1 | `train.data.dataset` |
| 数据清洗 / 梯度爆炸防护 | 1 | `num.sanitize` |
| 数据清洗常见操作 | 1 | `num.nanToZero` |
| 数据泄漏防范：Kohavi (1995) A Study of Cross-Validation and Bootstrap, IJCAI | 1 | `train.pipe.standardizeThenSplit` |
| 数据增强综述：Shorten & Khoshgoftaar (2019) A Survey on Image Data Augmentation for Deep Learning, Applied Intelligence | 1 | `train.aug.scaleJitter` |
| 数值分析标准 | 1 | `num.interpLagrange` |
| 数值分析标准构造 | 1 | `linalg.vander` |
| 数值分析复合梯形公式（误差 O(h²)） | 1 | `prob.util.trapz` |
| 缩尾（Winsorizing）：Tukey (1962) The Future of Data Analysis, Ann. Math. Statist.；Dixon (1960) Ann. Math. Statist. | 1 | `train.data.winsorize` |
| 特殊函数标准 | 1 | `num.beta` |
| 特征缩放与量纲一致性：Pedregosa et al. (2011) Scikit-learn, JMLR 12 | 1 | `train.data.featureScale` |
| 梯度累积（大有效批）：Huang et al. (2019) GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism, NeurIPS | 1 | `train.accum.step` |
| 梯度累积训练实践（等效放大 batch） | 1 | `opt.grad.accumulateAdd` |
| 梯度累积训练实践（TensorFlow/PyTorch accumulation） | 1 | `opt.grad.accumulateReset` |
| 通用函数逼近（Hornik 1991） | 1 | `ad.mlp` |
| 统计学标准（多元正态归一化） | 1 | `linalg.logdet` |
| 凸优化/矩阵补全标准（Candès & Recht 2009） | 1 | `linalg.normNuc` |
| 凸优化下降方法理论（Nocedal & Wright 第 3 章） | 1 | `opt.util.monotoneLoss` |
| 稳健回归损失 | 1 | `ad.mae` |
| 线性代数标准定义 | 1 | `math.cross` |
| 小批随机梯度下降的批处理惯例：Bottou (2010) Large-Scale Machine Learning with Stochastic Gradient Descent, COMPSTAT | 1 | `train.data.batchIterator` |
| 信号处理标准构造 | 1 | `linalg.convMatrix` |
| 信息检索标准度量 | 1 | `math.cosine` |
| 学习曲线分析：Perlich, Provost & Simonoff (2003) Tree Induction vs. Logistic Regression: A Learning-Curve Analysis, Machine Learning | 1 | `train.eval.learningCurve` |
| 循环矩阵理论 | 1 | `linalg.circulantMatvec` |
| 训练集统计量应用于测试集（避免数据泄漏）：Kohavi (1995) IJCAI | 1 | `train.data.standardizerApply` |
| 训练日志惯例：Goodfellow et al. (2016) Deep Learning, ch.8 | 1 | `train.hist.log` |
| 训练循环与早停：Prechelt (1998) Early Stopping — But When?; Goodfellow et al. (2016) Deep Learning, ch.8 | 1 | `train.loop.run` |
| 训练预算控制：Domhan et al. (2015) Speeding Up Automatic Hyperparameter Optimization of DNNs, ICML | 1 | `train.stop.maxEpochs` |
| 哑变量/独热编码约定：Pedregosa et al. (2011) Scikit-learn, JMLR 12 | 1 | `train.data.onehotEncode` |
| 验证曲线/模型选择：Hastie, Tibshirani & Friedman (2009) The Elements of Statistical Learning, §7.10 | 1 | `train.eval.validationCurve` |
| 优化测试函数标准形式（Jamil & Yang, A Literature Survey of Benchmark Functions, 2013） | 1 | `opt.util.sphereFn` |
| 优化基准标准形式（Nocedal & Wright《Numerical Optimization》第 3 章二次模型） | 1 | `opt.util.quadraticFn` |
| 优化终止准则（Nocedal & Wright《Numerical Optimization》§2.2） | 1 | `opt.util.converged` |
| 余弦相似度标准预处理 | 1 | `linalg.normalizeRows` |
| 与 opt.<name> 的 init 一致 | 1 | `opt.util.zeroState` |
| 预测评估标准 | 1 | `num.mape` |
| 预算受限训练与提前终止：Domhan et al. (2015) ICML；Real et al. (2017) Large-Scale Evolution of Image Classifiers | 1 | `train.stop.timeBudget` |
| 阈值扫描与工作点选择：Fawcett (2006) Pattern Recognition Letters 27(8) | 1 | `train.eval.thresholdSweep` |
| 早停与预算控制的组合策略：Domhan et al. (2015) ICML；Prechelt (1998) | 1 | `train.stop.criteria` |
| 增量解码 KV 缓存（Pope et al. 2022；HuggingFace generate 实现） | 1 | `dl.attn.kvCacheAppend` |
| 注册表内省（describeOp） | 1 | `opt.util.describeOptimizer` |
| 自回归推理 KV 缓存惯例（Pope et al., Efficiently Scaling Transformer Inference, 2022） | 1 | `dl.attn.kvCacheInit` |
| 组合预设：Bernstein et al., signSGD: Compressed Optimisation for Non-Convex Problems, ICML 2018 + PyTorch 文档：lr_scheduler.ConstantLR | 1 | `opt.signSgdConstant` |
| 组合预设：Chen et al., Symbolic Discovery of Optimization Algorithms (Lion), 2023 + Goyal et al., Accurate Large Minibatch SGD, 2017 + Loshchilov & Hutter, SGDR, ICLR 2017 | 1 | `opt.lionWarmupCosine` |
| 组合预设：Dozat, Incorporating Nesterov Momentum into Adam, ICLR 2016 + Smith, A Disciplined Approach to Neural Network Hyper-Parameters, 2018 | 1 | `opt.nadamOnecycle` |
| 组合预设：Duchi, Hazan & Singer, Adaptive Subgradient Methods for Online Learning, JMLR 2011 + PyTorch 文档：lr_scheduler.ExponentialLR | 1 | `opt.adagradExp` |
| 组合预设：Graves, Generating Sequences With Recurrent Neural Networks, 2013（RMSprop 形式） + PyTorch 文档：lr_scheduler.MultiStepLR | 1 | `opt.rmspropMultistep` |
| 组合预设：Kingma & Ba, Adam 论文 §7.2（Adamax 算法） + Goyal et al., Accurate Large Minibatch SGD, 2017（warmup 策略） | 1 | `opt.adamaxWarmup` |
| 组合预设：Kingma & Ba, Adam: A Method for Stochastic Optimization, ICLR 2015 + Loshchilov & Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts, ICLR 2017 | 1 | `opt.adamCosine` |
| 组合预设：Loshchilov & Hutter, Decoupled Weight Decay Regularization (AdamW), ICLR 2019 + Smith, A Disciplined Approach to Neural Network Hyper-Parameters, 2018 | 1 | `opt.adamwOnecycle` |
| 组合预设：Polyak, Some Methods of Speeding Up the Convergence of Iteration Methods, 1964 + 常见训练脚本实践（如 Transformer 的 Noam 逆平方根的线性变体） | 1 | `opt.momentumLinear` |
| 组合预设：Riedmiller & Braun, A Direct Adaptive Method for Faster Backpropagation Learning, 1993 + Smith, Cyclical Learning Rates for Training Neural Networks, 2017 | 1 | `opt.rpropTriangular` |
| 组合预设：Sutskever et al., On the Importance of Initialization and Momentum in Deep Learning, ICML 2013 + PyTorch 文档：lr_scheduler.StepLR | 1 | `opt.sgdNesterovStep` |
| 组合预设：You et al., Large Batch Training of Deep Networks (LAMB), 2017 + Loshchilov & Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts, ICLR 2017 | 1 | `opt.lambCosine` |
| 最速下降法固定步长形式（Nocedal & Wright《Numerical Optimization》§3.1） | 1 | `opt.line.fixedStep` |
| Abramowitz & Stegun 7.1.26；级数 erf(x)=2/√π·Σ(-1)ⁿx^{2n+1}/(n!(2n+1)) | 1 | `num.erf` |
| Acklam 算法（Peter Acklam）；Wichura AS241 为另一基准 | 1 | `num.normalQuantile` |
| Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, EMNLP 2023 | 1 | `dl.attn.groupedQuery` |
| Aitken 1926 / Numerical Recipes §5.1 | 1 | `num.aitken` |
| Ansible / SaltStack 配置合并语义 | 1 | `sys.cfg.merge` |
| Ansible / SaltStack defaults 语义 | 1 | `sys.cfg.defaults` |
| Armijo, Minimization of Functions Having Lipschitz Continuous First Partial Derivatives, 1966 | 1 | `opt.line.backtracking` |
| Arthur & Vassilvitskii, k-means++: The Advantages of Careful Seeding, SODA 2007 | 1 | `ml.kmeanspp.init` |
| Ba et al. 2016 | 1 | `ad.layerNorm` |
| Ba et al., Layer Normalization, 2016 | 1 | `dl.norm.layernorm` |
| Baydin et al. 2018 (tape/graph introspection) | 1 | `ad.graphSize` |
| Baydin et al. 2018 综述 | 1 | `ad.backward` |
| Beale 函数，标准优化测试函数（Jamil & Yang, 2013） | 1 | `opt.util.beale` |
| Bell & Sejnowski, The "Independent Components" of Natural Scenes, Vision Research, 1997；Hyvärinen et al., ICA, 2001 | 1 | `ml.whiten` |
| Bengio et al., A Neural Probabilistic Language Model, JMLR 2003；Mikolov et al. word2vec 2013 | 1 | `dl.layer.embedding` |
| Bentley & McIlroy 1993 差分算法思想 / 配置漂移检测 | 1 | `sys.cfg.diff` |
| Bergstra et al. 2009 | 1 | `ad.softsign` |
| Bergstra et al., Theano 激活函数库, 2010；Glorot & Bengio 2010 提及 | 1 | `dl.act.softsign` |
| Bernstein et al., signSGD, ICML 2018；Chen et al., Lion, 2023 | 1 | `opt.util.signUpdate` |
| Bishop (1995) Training with Noise is Equivalent to Tikhonov Regularization, Neural Computation 6(1) | 1 | `train.aug.gaussianNoise` |
| BLAS daxpy | 1 | `ad.scaleAdd` |
| BLAS dgemm | 1 | `linalg.gemm` |
| BLAS dgemm(transA,transB) | 1 | `linalg.matmulABT` |
| BLAS dgemm(transA) | 1 | `linalg.matmulAT` |
| BLAS dgemm(transB) | 1 | `linalg.matmulBT` |
| BLAS dgemv | 1 | `linalg.matmulVec` |
| BLAS dgemv(trans) | 1 | `linalg.matVecT` |
| BLAS dscal | 1 | `linalg.scaleMat` |
| BLAS gemm 浮点计数约定 / Golub & Van Loan《Matrix Computations》 | 1 | `sys.bench.matmulFlops` |
| BLAS transA | 1 | `ad.matmulAT` |
| Booth 函数，标准优化测试函数（Jamil & Yang, 2013） | 1 | `opt.util.booth` |
| Bottou (2010) Large-Scale Machine Learning with SGD, COMPSTAT（每轮重排） | 1 | `train.loop.shuffleEachEpoch` |
| Box & Muller 1958（正态变量生成） | 1 | `prob.normal.sample` |
| Box & Muller, Annals of Mathematical Statistics 29(2), 1958 | 1 | `prob.samp.boxMuller` |
| Breiman et al., CART, 1984；Breiman, Random Forests, Machine Learning, 2001 | 1 | `ml.tree.importance` |
| Breiman, Friedman, Olshen & Stone, Classification and Regression Trees, 1984 | 1 | `ml.tree.fit` |
| Breiman, Random Forests, 2001 | 1 | `ml.rf.predict` |
| Breiman, Random Forests, 2001；Louppe et al., Understanding Variable Importances in Forests, 2013 | 1 | `ml.rf.importance` |
| Breiman, Random Forests, Machine Learning, 2001；Ho, Random Decision Forests, 1995 | 1 | `ml.rf.fit` |
| Brent 1973, Algorithms for Minimization Without Derivatives；Numerical Recipes §9.3 | 1 | `num.brent` |
| Bridle 1990 | 1 | `ml.softmax.predict` |
| Bridle 1990；Bishop PRML §4.3.4 | 1 | `ml.softmax.predictProba` |
| Bridle, Probabilistic Interpretation of Feedforward Network Outputs, 1990；Bishop PRML §4.3.4 | 1 | `ml.softmax.fit` |
| Brier (1950) Monthly Weather Review 78(1); Murphy (1972) / proper scoring rules 综述 | 1 | `train.eval.logloss` |
| Brier (1950) Verification of Forecasts Expressed in Terms of Probability, Monthly Weather Review 78(1) | 1 | `train.eval.brier` |
| Brodersen et al. (2010) The Balanced Accuracy and Its Posterior Distribution, ICPR | 1 | `train.eval.balancedAcc` |
| Caliński & Harabasz, A Dendrite Method for Cluster Analysis, Communications in Statistics, 1974 | 1 | `ml.metric.calinski` |
| Chan, Golub & LeVeque 1979（并行方差合并） | 1 | `num.welfordMerge` |
| Chen et al., A Simple Framework for Contrastive Learning of Visual Representations (SimCLR), ICML 2020 | 1 | `dl.loss.ntxent` |
| Cho et al. 2014；PyTorch nn.GRU 前向语义 | 1 | `dl.seq.gruForward` |
| Cho et al., Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation, EMNLP 2014 | 1 | `dl.seq.gruCell` |
| Clevert et al. 2016 | 1 | `ad.elu` |
| Clevert et al., Fast and Accurate Deep Network Learning by Exponential Linear Units, ICLR 2016 | 1 | `dl.act.elu` |
| CLI 表格渲染惯例（GNU column 风格） | 1 | `sys.util.table` |
| Cochran, Sampling Techniques §5.2（分层抽样/JIPS） | 1 | `prob.samp.stratified` |
| Cohen (1960) A Coefficient of Agreement for Nominal Scales, Educational and Psychological Measurement 20(1) | 1 | `train.eval.cohenKappa` |
| Cortes & Vapnik, Support-Vector Networks, Machine Learning, 1995；Shalev-Shwartz et al., Pegasos, 2011 | 1 | `ml.svm.fit` |
| Courbariaux et al., BinaryNet, 2016；Howard et al., MobileNets V3（h-swish 同源）, 2019 | 1 | `dl.act.hardSigmoid` |
| Cover & Hart 1967；Altman, An Introduction to Kernel and Nearest-Neighbor Nonparametric Regression, 1992 | 1 | `ml.knn.regress` |
| Cover & Hart, Nearest Neighbor Pattern Classification, IEEE Trans. IT, 1967 | 1 | `ml.knn.fit` |
| Cover & Thomas, Elements of Information Theory §2.2 | 1 | `prob.info.crossEntropy` |
| Cover & Thomas, Elements of Information Theory §2.3；Shannon 1948 | 1 | `prob.info.mutualInfo` |
| Cover & Thomas, Elements of Information Theory（total variation） | 1 | `prob.info.tvd` |
| Cover & Thomas《Elements of Information Theory》2nd ed., 2006；Guyon & Elisseeff, JMLR 2003 | 1 | `ml.select.mutualInfo` |
| Cox 1958；Bishop, PRML, 2006 §4.3 | 1 | `ml.logreg.predict` |
| Cox 1958；Platt, Probabilistic Outputs for SVMs, 1999 | 1 | `ml.logreg.predictProba` |
| Cox, The Regression Analysis of Binary Sequences, JRSS-B, 1958；Nelder & Wedderburn, GLM, 1972 | 1 | `ml.logreg.fit` |
| CSV 规范：Shafranovich (2005) RFC 4180, IETF | 1 | `train.repro.tableToCsv` |
| Dao et al., FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, NeurIPS 2022 | 1 | `dl.attn.flashTiling` |
| Dark Knowledge / 常见 SGD 变体实践 | 1 | `opt.lr.inverse` |
| Dauphin et al., Language Modeling with Gated Convolutional Networks, ICML 2017 | 1 | `dl.act.glu` |
| Davies & Bouldin, A Cluster Separation Measure, IEEE TPAMI 1979 | 1 | `ml.metric.davies` |
| Davis & Goadrich (2006) The Relationship Between Precision-Recall and ROC Curves, ICML | 1 | `train.eval.prAnalysis` |
| Decandia et al. 2007 Dynamo 可重试错误分类 | 1 | `sys.err.isRetryable` |
| Decandia et al. 2007 Dynamo 重试与退避策略 | 1 | `sys.exec.retry` |
| DeGroot, Optimal Statistical Decisions；Gelman et al., Bayesian Data Analysis ch.2 | 1 | `prob.bayes.update` |
| Dekker 1971；Higham §4.2 | 1 | `num.decompose` |
| Devroye 1986, Non-Uniform Random Variate Generation（inverse CDF method） | 1 | `prob.samp.inverseCdf` |
| Devroye, Györfi & Lugosi (1996) A Probabilistic Theory of Pattern Recognition | 1 | `train.eval.stratifiedBaseline` |
| Devroye, Györfi & Lugosi (1996) A Probabilistic Theory of Pattern Recognition（零规则基线） | 1 | `train.eval.majorityBaseline` |
| Dietterich (1998) Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms, Neural Computation | 1 | `train.eval.repeatedCV` |
| Dietterich (1998) Neural Computation; Nadeau & Bengio (2003) Inference for the Generalization Error, NIPS | 1 | `train.eval.pairedCompare` |
| Dosovitskiy et al., An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, ICLR 2021 | 1 | `dl.layer.patching` |
| Duda & Hart 1973；Hastie et al., ESL 2009 §6.6.3 | 1 | `ml.nb.gaussianPredict` |
| Duda & Hart, Pattern Classification and Scene Analysis, 1973；Hastie et al., ESL 2009 §6.6.3 | 1 | `ml.nb.gaussianFit` |
| Dugas et al. 2001 | 1 | `ad.softplus` |
| Dugas et al. 2001（将 softplus 引入神经网络） | 1 | `num.softplus` |
| Dugas et al., Incorporating Second-Order Functional Knowledge for Better Option Pricing, NIPS 2000 | 1 | `dl.act.softplus` |
| Dumoulin & Visin, A Guide to Convolution Arithmetic for Deep Learning, 2016；ONNX ConvTranspose | 1 | `dl.layer.transposeConv2d` |
| ECMAScript Object.freeze / 不可变配置模式 | 1 | `sys.cfg.freeze` |
| ECMAScript SameValueZero 语义 / Liskov 替换原则 | 1 | `sys.util.deepEqual` |
| ECMAScript typeof 与 Object.prototype.toString 分类 | 1 | `sys.util.typeOf` |
| Efron (1979) Bootstrap Methods: Another Look at the Jackknife, Ann. Statist. 7(1); Efron & Tibshirani (1993) An Introduction to the Bootstrap | 1 | `train.eval.bootstrapCI` |
| Elman 1990；PyTorch nn.RNN 的前向语义 | 1 | `dl.seq.rnnForward` |
| Elman, Finding Structure in Time, Cognitive Science 1990；Rumelhart et al. 1986 | 1 | `dl.seq.rnnCell` |
| Epoch/小批训练惯例：Bottou (2010) COMPSTAT；Goodfellow et al. (2016) Deep Learning, ch.8 | 1 | `train.loop.epoch` |
| Ester et al., KDD 1996 | 1 | `ml.dbscan.predict` |
| Ester, Kriegel, Sander & Xu, A Density-Based Algorithm for Discovering Clusters, KDD 1996 | 1 | `ml.dbscan.fit` |
| Evans & Rosenthal, Probability and Statistics（极大似然估计） | 1 | `prob.normal.fit` |
| Evans & Rosenthal, Probability and Statistics（MLE） | 1 | `prob.bayes.mleNormal` |
| Evans & Rosenthal, Probability and Statistics（t 分布表示） | 1 | `prob.studentT.sample` |
| Fawcett (2006) An Introduction to ROC Analysis, Pattern Recognition Letters 27(8); Hanley & McNeil (1982) Radiology 143(1) | 1 | `train.eval.rocAnalysis` |
| Fisher 1925《Statistical Methods for Research Workers》；Guyon & Elisseeff, JMLR 2003 | 1 | `ml.select.fTest` |
| Fisher 1936；Hastie et al., ESL 2009 §4.3 | 1 | `ml.lda.predict` |
| Fisher-Yates 打乱 + PCG 流：O'Neill (2014) PCG: A Family of Simple Fast Space-Efficient Statistically Good RNGs | 1 | `train.data.shuffledBatches` |
| Fisher, The Use of Multiple Measurements in Taxonomic Problems, 1936；Hastie et al., ESL 2009 §4.3 | 1 | `ml.lda.fit` |
| fit/transform 分离与数据泄漏防范：Kohavi (1995) IJCAI；Pedregosa et al. (2011) JMLR | 1 | `train.data.pipelineFitTransform` |
| FNV-1a：Fowler/Noll/Vo (1991) | 1 | `train.repro.hashParams` |
| FNV-1a：Fowler/Noll/Vo (1991)；数据版本控制：Perez-Riverol et al. (2016) PLOS CB | 1 | `train.repro.hashData` |
| FNV-1a：Fowler/Noll/Vo (1991)；Perez-Riverol et al. (2016) PLOS CB | 1 | `train.repro.manifestHash` |
| FNV-1a（Fowler–Noll–Vo 哈希）/ MurmurHash3 | 1 | `sys.reg.catalogHash` |
| forge-1024 错误分类学 | 1 | `sys.err.codes` |
| forge-1024 错误分类学 / Rust error kinds | 1 | `sys.err.classify` |
| forge-1024 错误分类学（统一错误前缀） | 1 | `sys.err.format` |
| forge-1024 错误分类学（Rust error kinds 风格） | 1 | `sys.verify.errorTaxonomy` |
| forge-1024 错误分类学诊断建议 | 1 | `sys.err.suggest` |
| forge-1024 目录规约：每域恰好 102 条 | 1 | `sys.verify.domainCounts` |
| Freund & Schapire 1997；Zhu et al., SAMME, 2009 | 1 | `ml.ada.predict` |
| Freund & Schapire, A Decision-Theoretic Generalization of On-Line Learning, JCSS, 1997；Zhu et al., SAMME, 2009 | 1 | `ml.ada.fit` |
| Friedman 2001；Hastie et al., ESL 2009 §10.9-10.10 | 1 | `ml.gb.stagedLoss` |
| Friedman, Greedy Function Approximation: A Gradient Boosting Machine, 2001 | 1 | `ml.gb.predict` |
| Friedman, Greedy Function Approximation: A Gradient Boosting Machine, Annals of Statistics, 2001 | 1 | `ml.gb.fit` |
| Gauss 1809 / Legendre 1805；Golub & Van Loan《Matrix Computations》§5.3 | 1 | `ml.linreg.fit` |
| Gauss 1809 最小二乘；PyTorch nn.MSELoss | 1 | `dl.loss.mse` |
| Gelman et al., Bayesian Data Analysis ch.2；DeGroot, Optimal Statistical Decisions | 1 | `prob.bayes.conjugateNormal` |
| Gelman et al., Bayesian Data Analysis ch.2（可信区间） | 1 | `prob.bayes.credibleInterval` |
| Gelman et al., Bayesian Data Analysis ch.2（正态-正态共轭） | 1 | `prob.bayes.mapNormal` |
| Gershgorin 1931 | 1 | `linalg.gershgorin` |
| Glorot & Bengio, AISTATS 2010 | 1 | `dl.init.xavierNormal` |
| Glorot & Bengio, Understanding the Difficulty of Training Deep Feedforward Neural Networks, AISTATS 2010 | 1 | `dl.init.xavierUniform` |
| Golub & Van Loan §1.1；BLAS dgemm；Goto & van de Geijn 分块思想 | 1 | `linalg.matmul` |
| Golub & Van Loan §1.2 (symmetric/Hermitian part) | 1 | `linalg.symmetrize` |
| Golub & Van Loan §1.3 | 1 | `math.outer` |
| Golub & Van Loan §1.3 / BLAS ddot | 1 | `math.dot` |
| Golub & Van Loan §2.3；LAPACK dgetri | 1 | `linalg.inverse` |
| Golub & Van Loan §3.2 | 1 | `linalg.luNoPivot` |
| Golub & Van Loan §5.5；LAPACK 秩判定 | 1 | `linalg.rank` |
| Golub & Van Loan §5.5；LAPACK dgelss | 1 | `linalg.pinv` |
| Golub & Van Loan §8.2 | 1 | `linalg.rayleighQuotient` |
| Golub & Van Loan §8.3 | 1 | `linalg.inverseIteration` |
| Goodfellow et al. (2016) Deep Learning, §8.3（小批梯度下降） | 1 | `train.loop.batchStep` |
| Google Benchmark 报告格式 | 1 | `sys.bench.reportMarkdown` |
| Google Benchmark 对比报告 | 1 | `sys.bench.compare` |
| Google Benchmark 复杂度扫描（Big-O） | 1 | `sys.bench.scaling` |
| Goyal et al., Accurate Large Minibatch SGD, 2017 | 1 | `opt.util.warmupFactor` |
| Goyal et al., Accurate Large Minibatch SGD, 2017 + Loshchilov & Hutter, SGDR, ICLR 2017 | 1 | `opt.lr.warmupCosine` |
| Goyal et al., Accurate Large Minibatch SGD, 2017（warmup 策略） | 1 | `opt.lr.warmup` |
| Graphviz DOT 语言（Gansner & North） | 1 | `sys.graph.toDot` |
| Griewank & Walther, Evaluating Derivatives, 2008 | 1 | `ad.jacobianDiag` |
| Guyon & Elisseeff, An Introduction to Variable and Feature Selection, JMLR, 2003；scikit-learn VarianceThreshold | 1 | `ml.select.variance` |
| Hampel 1974, JASA 69:383-393 (robust scale estimator) | 1 | `num.medAbsErr` |
| Hampel 1974, JASA 69:383-393; Leys et al. 2013 JESP | 1 | `math.mad` |
| Hampel, Ronchetti, Rousseeuw & Stahel, Robust Statistics: The Approach Based on Influence Functions, 1986 | 1 | `math.median` |
| Hastie et al., ESL 2009 §2.4 | 1 | `ml.metric.rmse` |
| Hastie et al., ESL 2009 §2.4；Gauss 最小二乘传统 | 1 | `ml.metric.mse` |
| Hastie et al., ESL 2009 §2.4；Huber 1964（L1 损失） | 1 | `ml.metric.mae` |
| Hastie et al., ESL 2009 §2.4；scikit-learn LabelEncoder | 1 | `ml.encode.label` |
| Hastie et al., ESL 2009 §2.4（定性变量的哑变量编码） | 1 | `ml.encode.onehot` |
| Hastie et al., ESL 2009 §3.4；scikit-learn StandardScaler | 1 | `ml.scale.standard` |
| Hastie et al., ESL 2009 §4.3 | 1 | `ml.qda.predict` |
| Hastie et al., ESL 2009 §4.3；McLachlan《Discriminant Analysis and Statistical Pattern Recognition》1992 | 1 | `ml.qda.fit` |
| Hastie et al., ESL 2009 §7.2 | 1 | `ml.poly.predict` |
| Hastie et al., ESL 2009 §7.2（基函数展开）；Forsythe 1957（多项式拟合） | 1 | `ml.poly.fit` |
| Hastie, Tibshirani & Friedman, ESL 2nd ed., 2009, §3.2 | 1 | `ml.linreg.predict` |
| He et al. 2015 / Glorot & Bengio 2010；PyTorch nn.Sequential 的参数构造 | 1 | `dl.model.mlpInit` |
| He et al., Deep Residual Learning for Image Recognition, CVPR 2016 | 1 | `dl.model.residual` |
| He et al., ICCV 2015；PyTorch kaiming_normal_ | 1 | `dl.init.heNormal` |
| Hendrycks & Gimpel 2016 | 1 | `ad.gelu` |
| Hendrycks & Gimpel, Gaussian Error Linear Units (GELUs), 2016 | 1 | `dl.act.gelu` |
| Hestenes & Stiefel 1952；Saad §6.7；LAPACK 之外的经典 | 1 | `linalg.conjGradient` |
| Hestenes 1958 单边 Jacobi；LAPACK dgesvj；Golub & Van Loan §8.6 | 1 | `linalg.svd` |
| Higham 1986；Golub & Van Loan §8.3 | 1 | `linalg.polar` |
| Higham 2002 思想（交替投影的谱截断变体） | 1 | `linalg.nearestSPD` |
| Higham 2002, Accuracy and Stability of Numerical Algorithms, §1.2 | 1 | `num.maxAbsErr` |
| Higham 2005《Functions of Matrices》§10.3（缩放平方思想） | 1 | `linalg.matrixExp` |
| Himmelblau 函数，标准优化测试函数（Jamil & Yang, 2013） | 1 | `opt.util.himmelblau` |
| Hoare (1969) An Axiomatic Basis for Computer Programming, CACM 12(10)（不变量断言） | 1 | `train.util.assertSplitDisjoint` |
| Hoare (1969) An Axiomatic Basis for Computer Programming, CACM 12(10)（断言式不变量） | 1 | `train.repro.determinismCheck` |
| Hochreiter & Schmidhuber 1997；PyTorch nn.LSTM 前向语义 | 1 | `dl.seq.lstmForward` |
| Hochreiter & Schmidhuber, Long Short-Term Memory, Neural Computation 1997 | 1 | `dl.seq.lstmCell` |
| Hoerl & Kennard 1970；Hastie et al., ESL 2009 §3.4.1 | 1 | `ml.ridge.predict` |
| Hoerl & Kennard, Ridge Regression, 1970；Krogh & Hertz 1992 权重衰减 | 1 | `dl.reg.l2` |
| Hoerl & Kennard, Ridge Regression: Biased Estimation for Nonorthogonal Problems, 1970 | 1 | `ml.ridge.fit` |
| Holdout 验证：Devroye, Györfi & Lugosi (1996) A Probabilistic Theory of Pattern Recognition | 1 | `train.data.trainTestSplit` |
| Horn & Johnson《Topics in Matrix Analysis》 | 1 | `linalg.kron` |
| Horner 规则 / Numerical Recipes §5.3 | 1 | `num.horner` |
| Hornik et al. 1989 通用逼近定理；PyTorch nn.Sequential 前向语义 | 1 | `dl.model.mlpForward` |
| Howard et al. 2019（MobileNetV3） | 1 | `ad.hardswish` |
| Howard et al., MobileNets: Efficient Convolutional Neural Networks for Mobile Vision, 2017 | 1 | `dl.layer.depthwiseConv2d` |
| Huang et al. (2019) GPipe, NeurIPS | 1 | `train.accum.reset` |
| Huang et al., Deep Networks with Stochastic Depth, ECCV 2016 | 1 | `dl.reg.stochasticDepth` |
| Huber 1964 稳健统计 | 1 | `ad.huber` |
| Huber 1964；Hampel et al.《Robust Statistics》1986 | 1 | `ml.robust.predict` |
| Huber 1964；scikit-learn RobustScaler（IQR 定义同 Tukey 1977） | 1 | `ml.scale.robust` |
| Huber, Robust Estimation of a Location Parameter, Ann. Math. Statist. 1964 | 1 | `dl.loss.huber` |
| Huber, Robust Estimation of a Location Parameter, Ann. Math. Statist., 1964 | 1 | `ml.robust.fit` |
| Hyndman & Fan, Sample Quantiles in Statistical Packages, The American Statistician, 1996（类型 7） | 1 | `sys.bench.percentile` |
| Hypothesis 属性测试框架（不变量断言） | 1 | `sys.exec.assertOp` |
| Hypothesis 属性测试框架（不变量声明） | 1 | `sys.verify.invariants` |
| Hypothesis 属性测试框架（MacIver & Donaldson et al.） | 1 | `sys.verify.fuzz` |
| IEEE 754 双精度 | 1 | `num.eps` |
| IEEE 754 双精度指数定义域 | 1 | `num.stableExp` |
| IEEE 754 顺序累加 | 1 | `num.naiveSum` |
| IEEE 754 isFinite | 1 | `num.isfiniteAll` |
| IEEE 754 log | 1 | `num.safeLog` |
| IEEE 754 nextAfter / C99 nextafter | 1 | `num.nextafter` |
| IEEE 754 pow | 1 | `num.safePow` |
| IEEE 754 ulp 概念 | 1 | `num.ulpErr` |
| IEEE 754-2019 浮点字符串化 / ECMAScript Number.prototype.toFixed | 1 | `sys.util.formatNum` |
| IEEE 754-2019 确定性运算 / Hypothesis 可复现性要求 | 1 | `sys.verify.determinism` |
| Ioffe & Szegedy 2015 | 1 | `ad.batchnorm1d` |
| Ioffe & Szegedy 2015 的 running_mean / running_var 更新式 | 1 | `dl.norm.runningUpdate` |
| Ioffe & Szegedy 2015 的批统计；Welford 1962 数值稳定方差 | 1 | `dl.norm.batchStats` |
| Ioffe & Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, ICML 2015 | 1 | `dl.norm.batchnorm` |
| Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference, CVPR 2018 | 1 | `dl.util.quantizeInt8` |
| Jacobi 1845；Golub & Van Loan §8.4；LAPACK dsyevj | 1 | `linalg.eigenSym` |
| Jacobi 1845；Saad《Iterative Methods》§4.2 | 1 | `linalg.jacobiIter` |
| Jarque & Bera, International Statistical Review 55(2), 1987 | 1 | `prob.test.jarqueBera` |
| JAX grad | 1 | `ad.grad` |
| JAX lax.scan 语义；Hochreiter & Schmidhuber 1997 的时间展开抽象 | 1 | `dl.seq.scan` |
| JAX vjp | 1 | `ad.vjp` |
| Jeffreys 1946；Le Cam 1964（Hellinger 距离） | 1 | `prob.info.hellinger` |
| Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1（Cauchy） | 1 | `prob.cauchy.pdf` |
| Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1（exponential） | 1 | `prob.exponential.pdf` |
| Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1（gamma） | 1 | `prob.gamma.pdf` |
| Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1（Laplace） | 1 | `prob.laplace.pdf` |
| Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1（Pareto） | 1 | `prob.pareto.pdf` |
| Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1（Rayleigh） | 1 | `prob.rayleigh.pdf` |
| Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 1（Weibull） | 1 | `prob.weibull.pdf` |
| Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 2（beta 抽样） | 1 | `prob.beta.sample` |
| Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 2（beta） | 1 | `prob.beta.pdf` |
| Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 2（logistic） | 1 | `prob.logistic.pdf` |
| Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions, Vol. 2（Student t） | 1 | `prob.studentT.pdf` |
| Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions（各分布矩与熵闭式） | 1 | `prob.util.distSummary` |
| Johnson, Kotz & Kemp, Discrete Univariate Distributions（Bernoulli） | 1 | `prob.bernoulli.pmf` |
| Johnson, Kotz & Kemp, Discrete Univariate Distributions（binomial） | 1 | `prob.binomial.pmf` |
| Johnson, Kotz & Kemp, Discrete Univariate Distributions（categorical） | 1 | `prob.categorical.pmf` |
| Johnson, Kotz & Kemp, Discrete Univariate Distributions（geometric） | 1 | `prob.geometric.pmf` |
| Johnson, Kotz & Kemp, Discrete Univariate Distributions（Poisson） | 1 | `prob.poisson.pmf` |
| Jolliffe《Principal Component Analysis》2002 §2.6 | 1 | `ml.pca.explainedVar` |
| JSON Schema 草案 2020-12 校验语义 | 1 | `sys.cfg.validate` |
| K 折交叉验证：Stone (1974) Cross-Validatory Choice and Assessment of Statistical Predictions, JRSS-B | 1 | `train.data.kFoldSplit` |
| Kahan 1965 | 1 | `num.kahanSum` |
| Kahan 1965 / BLAS 补偿点积 | 1 | `num.dotKahan` |
| Kahn 1962 拓扑排序算法 | 1 | `sys.graph.toposort` |
| Kendall & Stuart, The Advanced Theory of Statistics, Vol.1（分位数定义） | 1 | `prob.stat.quantile` |
| Kendall & Stuart, The Advanced Theory of Statistics, Vol.1（稳健统计量） | 1 | `prob.stat.mad` |
| Kingma & Ba, ICLR 2015 | 1 | `opt.util.biasCorrection` |
| Kingma & Ba, ICLR 2015（偏差修正分析） | 1 | `opt.util.etaScale` |
| Klambauer et al., Self-Normalizing Neural Networks, NeurIPS 2017 | 1 | `dl.act.selu` |
| Knuth TAOCP Vol.2 §4.2.2 | 1 | `num.twoSum` |
| Knuth, TAOCP Vol.2 §3.4.1（Knuth 算法）；Devroye 1986 | 1 | `prob.poisson.sample` |
| Krogh & Hertz, A Simple Weight Decay Can Improve Generalization, 1992 | 1 | `opt.grad.weightDecay` |
| Kullback & Leibler 1951 | 1 | `ad.klDiv` |
| Kullback & Leibler 1951（KL 散度原始文献） | 1 | `prob.info.klDiv` |
| Kullback & Leibler, On Information and Sufficiency, Ann. Math. Statist. 1951 | 1 | `dl.loss.klDiv` |
| LAPACK 变体 | 1 | `linalg.solveT` |
| LAPACK dgels/dgelss；Golub & Van Loan §5.3 | 1 | `linalg.lstsq` |
| LAPACK dgeqrf；Golub & Van Loan §5.2 | 1 | `linalg.qr` |
| LAPACK dgesv 多列 | 1 | `linalg.solveMultiple` |
| LAPACK dgesv；Golub & Van Loan §3.2 | 1 | `linalg.solve` |
| LAPACK dgetrf 后取 ΠUᵢᵢ | 1 | `linalg.det` |
| LAPACK dgetrf；Golub & Van Loan §3.2 | 1 | `linalg.lu` |
| LAPACK dposv | 1 | `linalg.solveSPD` |
| LAPACK dpotrf；Golub & Van Loan §4.2 | 1 | `linalg.cholesky` |
| Laplace 1774 最小绝对偏差；PyTorch nn.L1Loss | 1 | `dl.loss.mae` |
| Laplace 连分数 + 级数 | 1 | `num.erfc` |
| Larsson et al., FractalNet: Ultra-Deep Neural Networks without Residuals, ICLR 2017；Huang et al., Deep Networks with Stochastic Depth, ECCV 2016 | 1 | `dl.reg.dropPath` |
| Lasso（Tibshirani 1996） | 1 | `ad.l1Reg` |
| LeCun et al. 1998；Lin et al., Network In Network, 2014（全局平均池化思想） | 1 | `dl.layer.avgpool1d` |
| LeCun et al. 1998；ONNX Operator Standard: Conv | 1 | `dl.layer.conv1d` |
| LeCun et al., Gradient-Based Learning Applied to Document Recognition, 1998；ONNX Conv | 1 | `dl.layer.conv2d` |
| Lin 1991（JS divergence）；Endres & Schindelin 2003 | 1 | `prob.info.jsDiv` |
| Lin et al., Focal Loss for Dense Object Detection, ICCV 2017 | 1 | `dl.loss.focal` |
| Lin et al., Network In Network, ICLR 2014 | 1 | `dl.layer.globalAvgPool` |
| Lin et al., Network In Network, ICLR 2014；He et al. ResNet 2015 | 1 | `dl.layer.avgpool2d` |
| Little & Rubin 2002；Tukey《Exploratory Data Analysis》1977 | 1 | `ml.impute.median` |
| Little & Rubin《Statistical Analysis with Missing Data》2nd ed., 2002 §4.2 | 1 | `ml.impute.mean` |
| Lloyd 1982 | 1 | `ml.kmeans.predict` |
| Lloyd 1982；Hastie et al., ESL 2009 §14.3 | 1 | `ml.kmeans.inertia` |
| Lloyd, Least Squares Quantization in PCM, 1982；Arthur & Vassilvitskii, k-means++, SODA 2007 | 1 | `ml.kmeans.fit` |
| LLVM benchmarking / Google Benchmark 计时模型 | 1 | `sys.bench.timeOp` |
| LLVM lit 冒烟测试框架 | 1 | `sys.verify.smoke` |
| LLVM Pass Manager 的可失败调度 / Rust Result 语义 | 1 | `sys.exec.try` |
| LLVM Pass Manager 结果缓存 / Bellman 1957 动态规划备忘 | 1 | `sys.exec.memo` |
| LLVM Pass Manager 流水线 / Unix 管道语义 | 1 | `sys.exec.pipe` |
| LLVM Pass Manager 批处理调度 | 1 | `sys.exec.batch` |
| LLVM Pass Manager 统一 pass 调度入口 | 1 | `sys.exec.call` |
| LLVM test-suite / Google Benchmark 批量注册 | 1 | `sys.bench.suite` |
| LLVM verifier 诊断输出 / 12-Factor 健康检查 | 1 | `sys.self.doctor` |
| log-sum-exp trick（数值计算标准技巧） | 1 | `num.logsumexp` |
| Logistic 函数 | 1 | `ad.sigmoid` |
| Logistic 函数标准实现 | 1 | `num.sigmoid` |
| Loshchilov & Hutter, SGDR, ICLR 2017 | 1 | `opt.util.cosineFactor` |
| Loshchilov & Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts, ICLR 2017 | 1 | `opt.lr.cosine` |
| Maas et al. 2013 | 1 | `ad.leakyRelu` |
| Maas et al., Rectifier Nonlinearities Improve Neural Network Acoustic Models, ICML 2013 | 1 | `dl.act.leakyRelu` |
| Marsaglia & Bray, CACM 7(1), 1964 | 1 | `prob.samp.marsagliaPolar` |
| Marsaglia & Tsang 2000（伽马抽样）；Evans & Rosenthal | 1 | `prob.chi2.sample` |
| Marsaglia & Tsang, ACM Trans. Math. Software 26(2), 2000 | 1 | `prob.gamma.sample` |
| Massey 1951（Kolmogorov 分布临界值）；Stephens 1974（EDF 统计量修正） | 1 | `prob.test.ks` |
| Matthews (1975) Biochimie 62(6); Gorodkin (2004) Correntropy…/K-correlation for multi-class, Pattern Recognition Letters | 1 | `train.eval.mcc` |
| McCallum & Nigam, A Comparison of Event Models for Naive Bayes Text Classification, AAAI-98 | 1 | `ml.nb.multinomialFit` |
| McCallum & Nigam, AAAI-98；Hastie et al., ESL 2009 §6.6.3 | 1 | `ml.nb.bernoulliFit` |
| McCormick, A Mathematical Programming Approach to the Global Optimization of Chemical Processes, 1972 | 1 | `opt.util.mccormick` |
| Mermaid 流程图语法 / LLVM Pass Manager 依赖图 | 1 | `sys.doc.mermaid` |
| Metropolis, Rosenbluth, Rosenbluth, Teller & Teller 1953 | 1 | `prob.samp.metropolis` |
| Milletari et al., V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation, 3DV 2016 | 1 | `dl.loss.dice` |
| Min-Max 归一化：Pedregosa et al. (2011) Scikit-learn, JMLR 12 | 1 | `train.data.minmaxFit` |
| Minka, "Estimating a Gamma Distribution", 2002；Thom 1968（初值近似） | 1 | `prob.gamma.fit` |
| Misra 2020 | 1 | `ad.mish` |
| Misra, Mish: A Self Regularized Non-Monotonic Neural Activation Function, 2019 | 1 | `dl.act.mish` |
| Miyato et al., Spectral Normalization for Generative Adversarial Networks, ICLR 2018 | 1 | `dl.norm.spectralnorm` |
| Nair & Hinton 2010；Glorot 2011 | 1 | `ad.relu` |
| Nair & Hinton, Rectified Linear Units Improve Restricted Boltzmann Machines, ICML 2010 | 1 | `dl.act.relu` |
| Neumaier 1974 | 1 | `num.neumaierSum` |
| Newton-Raphson / Numerical Recipes §9.4 | 1 | `num.newton` |
| Ng, Jordan & Weiss, NeurIPS 2001 | 1 | `ml.spectral.labels` |
| Ng, Jordan & Weiss, On Spectral Clustering: Analysis and an Algorithm, NeurIPS 2001；Shi & Malik, PAMI 2000 | 1 | `ml.spectral.fit` |
| Nocedal & Wright《Numerical Optimization》§2.2（梯度范数终止准则） | 1 | `opt.util.gradientNorm` |
| Nocedal & Wright《Numerical Optimization》§3.1（精确线搜索） | 1 | `opt.line.exactQuadratic` |
| Node.js process 对象（内置运行时内省） | 1 | `sys.env.info` |
| Node.js process.hrtime.bigint() 单调时钟 | 1 | `sys.env.nowMs` |
| Node.js process.memoryUsage() | 1 | `sys.env.memory` |
| Node.js process.platform / process.arch | 1 | `sys.env.platform` |
| Node.js process.uptime() | 1 | `sys.env.uptimeMs` |
| Node.js process.versions.node | 1 | `sys.env.nodeVersion` |
| Numerical Recipes §11.2 | 1 | `linalg.matrixPower` |
| Numerical Recipes §9.1 | 1 | `num.bisect` |
| Numerical Recipes §9.2 | 1 | `num.secant` |
| NumPy add | 1 | `linalg.addMat` |
| NumPy allclose | 1 | `num.allclose` |
| NumPy arange | 1 | `math.arange` |
| NumPy arange / Python range 语义 | 1 | `sys.util.range` |
| NumPy argmax | 1 | `math.argmax` |
| NumPy argmax(axis) | 1 | `math.argmaxAxis` |
| NumPy argmin | 1 | `math.argmin` |
| NumPy array_equal | 1 | `num.arrayEqual` |
| NumPy broadcast | 1 | `ad.broadcastTo` |
| NumPy clip | 1 | `math.clamp` |
| NumPy cumprod | 1 | `math.cumprod` |
| NumPy cumsum | 1 | `math.cumsum` |
| NumPy cumtrapz | 1 | `num.cumtrapz` |
| NumPy diff | 1 | `num.diff` |
| NumPy eye | 1 | `math.eye` |
| NumPy eye_like | 1 | `linalg.eyeLike` |
| NumPy fancy indexing | 1 | `ad.getItem` |
| NumPy fliplr | 1 | `linalg.flipLR` |
| NumPy full | 1 | `math.full` |
| NumPy full_like | 1 | `linalg.fullLike` |
| NumPy hstack | 1 | `linalg.concatCols` |
| NumPy interp | 1 | `num.interpLinear` |
| NumPy interp(kind=nearest) | 1 | `num.interpNearest` |
| NumPy isclose | 1 | `num.isclose` |
| NumPy linspace | 1 | `math.linspace` |
| NumPy maximum + 次梯度 | 1 | `ad.maximum` |
| NumPy mean | 1 | `math.mean` |
| NumPy mean(axis) | 1 | `ad.meanAxis` |
| NumPy minimum + 次梯度 | 1 | `ad.minimum` |
| NumPy norm | 1 | `math.linf` |
| NumPy norm / 泛函分析 | 1 | `math.l1` |
| NumPy norm / BLAS dnrm2 | 1 | `math.l2` |
| NumPy ones | 1 | `math.ones` |
| NumPy ones_like | 1 | `linalg.onesLike` |
| NumPy quantile | 1 | `math.quantile` |
| NumPy searchsorted | 1 | `num.searchSorted` |
| NumPy std | 1 | `math.std` |
| NumPy subtract | 1 | `linalg.subMat` |
| NumPy sum(axis) | 1 | `ad.sumAxis` |
| NumPy tile | 1 | `math.tile` |
| NumPy tril | 1 | `linalg.triLower` |
| NumPy triu | 1 | `linalg.triUpper` |
| NumPy var / 统计学定义 | 1 | `math.var` |
| NumPy var(axis) | 1 | `math.varAxis` |
| NumPy vstack | 1 | `linalg.concatRows` |
| NumPy zeros | 1 | `math.zeros` |
| NumPy zeros_like | 1 | `linalg.zerosLike` |
| ONNX 域级 checker | 1 | `sys.self.domainHealth` |
| ONNX checker / LLVM verifier | 1 | `sys.self.checkAll` |
| ONNX Operator Registry 出处可追溯性要求 | 1 | `sys.verify.refCoverage` |
| ONNX Operator Registry 存在性查询 | 1 | `sys.reg.exists` |
| ONNX Operator Registry 检索语义 | 1 | `sys.reg.search` |
| ONNX Operator Registry 命名唯一性约束 | 1 | `sys.verify.nameUnique` |
| ONNX Operator Registry 全量文档 | 1 | `sys.doc.catalogMarkdown` |
| ONNX Operator Registry 文档格式 | 1 | `sys.doc.opMarkdown` |
| ONNX Operator Registry 文档完备性要求 | 1 | `sys.verify.docCoverage` |
| ONNX Operator Registry 稳定标识 | 1 | `sys.reg.opIds` |
| ONNX Operator Registry 稳定标识约定 | 1 | `sys.verify.idContinuity` |
| ONNX Operator Registry 一致性检查（checker） | 1 | `sys.verify.catalog` |
| ONNX Operator Registry 域化查询 | 1 | `sys.reg.byDomain` |
| ONNX Operator Registry 域划分 | 1 | `sys.reg.domains` |
| ONNX Operator Registry 域文档 | 1 | `sys.doc.domainMarkdown` |
| ONNX Operator Registry op 文档 | 1 | `sys.reg.describe` |
| ONNX Operator Registry README 章节格式 | 1 | `sys.doc.readmeSection` |
| ONNX Operator Registry（opset 域化统计） | 1 | `sys.reg.count` |
| ONNX Operator Schema 签名清单 | 1 | `sys.doc.sigTable` |
| ONNX Operator Schema 元数校验；arity 由实现反推以避免元数据漂移 | 1 | `sys.verify.arityCheck` |
| ONNX Operator Schema（输入输出签名） | 1 | `sys.reg.signature` |
| ONNX opset 版本化约定 | 1 | `sys.reg.version` |
| ONNX Tensor 内存布局 / IEEE 754-2019 float64 | 1 | `sys.bench.memoryEstimate` |
| OpenTelemetry 错误追踪语义 | 1 | `sys.err.trace` |
| OpenTelemetry 指标规范（手动埋点） | 1 | `sys.tel.record` |
| Opitz & Burst (2019) Macro F1 and Macro F1+, arXiv:1911.03347; Sokolova & Lapalme (2009) IP&M 45(4) | 1 | `train.eval.macroAvg` |
| P.J. Acklam（Beasley-Springer-Moro 族逆正态算法）；Halley 迭代 | 1 | `prob.util.normalQuantile` |
| Pascanu, Mikolov & Bengio (2013) On the Difficulty of Training Recurrent Neural Models, ICML | 1 | `train.loop.withClipping` |
| Pascanu, Mikolov & Bengio, On the Difficulty of Training RNNs, ICML 2013 | 1 | `opt.grad.clipNorm` |
| PCA/统计标准 | 1 | `linalg.center` |
| PCG 随机流与命名派生：O'Neill (2014) PCG RNG; FNV-1a: Fowler/Noll/Vo (1991) | 1 | `train.repro.seedAll` |
| Pearson 1900（卡方准则）；Cochran 1952（GOF 适用条件） | 1 | `prob.test.chi2Gof` |
| Pearson 1901；Hotelling 1933；Jolliffe《Principal Component Analysis》2002 | 1 | `ml.pca.transform` |
| Pearson 1901；Jolliffe 2002 | 1 | `ml.pca.inverse` |
| Pearson, On Lines and Planes of Closest Fit, 1901；Hotelling, Analysis of a Complex of Statistical Variables, 1933 | 1 | `ml.pca.fit` |
| Pedregosa et al. (2011) Scikit-learn, JMLR 12 | 1 | `train.repro.restoreParams` |
| Pedregosa et al. (2011) Scikit-learn, JMLR 12（模型持久化） | 1 | `train.repro.snapshotParams` |
| Perez-Riverol et al. (2016) Ten Simple Rules for Taking Advantage of Git and GitHub, PLOS Comput. Biol. 12(7) | 1 | `train.repro.manifest` |
| Platt, Fast Training of Support Vector Machines using SMO, 1998；Boser, Guyon & Vapnik, 1992 | 1 | `ml.svm.kernelFit` |
| Polyak & Juditsky (1992) SIAM J. Control Optim. | 1 | `train.ema.apply` |
| Polyak & Juditsky, Averaging of Iterative Methods, 1992；深度学习常见 EMA 实践（如 Mean Teacher） | 1 | `opt.ema` |
| Polyak 动量 / EMA 标准形式 | 1 | `opt.util.momentumBlend` |
| Prechelt (1998) Early Stopping — But When? (Neural Networks: Tricks of the Trade); Prechelt & Probst (1996) 收敛监控 | 1 | `train.stop.earlyStopping` |
| Prechelt (1998) Early Stopping — But When?；验证集选模惯例：Hastie et al. (2009) ESL §7 | 1 | `train.ckpt.bestTracker` |
| Prechelt (1998) Early Stopping — But When?（最佳点选取） | 1 | `train.repro.bestEpoch` |
| Press et al., Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation, ICLR 2022 | 1 | `dl.attn.alibiBias` |
| Prometheus 比率指标（errors / total） | 1 | `sys.tel.errorRate` |
| Prometheus 健康报告 / ONNX checker 报告 | 1 | `sys.self.report` |
| Prometheus 健康摘要指标 | 1 | `sys.self.summary` |
| Prometheus 吞吐率指标 / Google Benchmark 时间预算模式 | 1 | `sys.bench.throughput` |
| Prometheus 直方图观测语义 | 1 | `sys.exec.timed` |
| Prometheus 指标模型（分组计数 + 占比） | 1 | `sys.reg.domainStats` |
| Prometheus 指标模型（counter / summary） | 1 | `sys.tel.summary` |
| Prometheus 指标重置语义 | 1 | `sys.tel.reset` |
| Prometheus counter 单调递增语义 | 1 | `sys.tel.totalCalls` |
| Prometheus counter 语义 | 1 | `sys.err.count` |
| Prometheus Summary 分位数语义 | 1 | `sys.tel.latencyStats` |
| Python zip / Haskell zipWith 语义 | 1 | `sys.util.zip` |
| PyTorch 文档：lr_scheduler.ConstantLR | 1 | `opt.lr.constant` |
| PyTorch 文档：lr_scheduler.ExponentialLR | 1 | `opt.lr.exp` |
| PyTorch 文档：lr_scheduler.MultiStepLR | 1 | `opt.lr.multistep` |
| PyTorch 文档：lr_scheduler.PolynomialLR | 1 | `opt.lr.polynomial` |
| PyTorch 文档：lr_scheduler.StepLR | 1 | `opt.lr.step` |
| PyTorch aten native_functions 名称索引 | 1 | `sys.reg.nameIndex` |
| PyTorch autograd 惯例 | 1 | `ad.tapeReset` |
| PyTorch detach | 1 | `ad.detach` |
| PyTorch embedding | 1 | `ad.gatherRows` |
| PyTorch FX 图变换中的逐节点映射 | 1 | `sys.exec.map` |
| PyTorch FX Graph 解释执行 / LLVM Pass 调度 | 1 | `sys.graph.execute` |
| PyTorch FX Graph 校验 / LLVM Pass Manager verifier | 1 | `sys.graph.validate` |
| PyTorch FX Graph IR / LLVM Pass Manager 依赖图 | 1 | `sys.graph.build` |
| PyTorch gradcheck + 投影 | 1 | `ad.gradcheckVec` |
| PyTorch gradcheck 方法论 | 1 | `ad.gradcheck` |
| PyTorch nn.BCEWithLogitsLoss（log-sum-exp 稳定化） | 1 | `dl.loss.bceWithLogits` |
| PyTorch nn.CrossEntropyLoss（LogSoftmax + NLLLoss 融合） | 1 | `dl.loss.ceWithLogits` |
| PyTorch nn.Flatten / Keras Flatten 层 | 1 | `dl.layer.flatten` |
| PyTorch nn.init.constant_；Gers et al. 2000 遗忘门偏置技巧 | 1 | `dl.init.constant` |
| PyTorch nn.init.normal_；Radford et al. GPT-2 的 0.02 缩放惯例 | 1 | `dl.init.normal` |
| PyTorch nn.init.ones_ / TensorFlow ones_initializer | 1 | `dl.init.ones` |
| PyTorch nn.init.uniform_；早期神经网络的对称破缺做法 | 1 | `dl.init.uniform` |
| PyTorch nn.init.zeros_ / TensorFlow zeros_initializer | 1 | `dl.init.zeros` |
| PyTorch nn.NLLLoss；Fisher 1912 极大似然 | 1 | `dl.loss.nll` |
| PyTorch nn.Unflatten / TensorFlow reshape 语义 | 1 | `dl.layer.unflatten` |
| PyTorch no_grad | 1 | `ad.noGrad` |
| PyTorch parameters_to_vector（torch.nn.utils） | 1 | `dl.util.flattenParams` |
| PyTorch scatter_add | 1 | `ad.scatterAdd` |
| Raffel et al., Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, JMLR 2020 | 1 | `dl.attn.relativeBias` |
| Ramachandran et al. 2017（Swish搜索）；Hendrycks & Gimpel 2016（GELU 前身） | 1 | `ad.silu` |
| Ramachandran et al., Searching for Activation Functions, 2017 | 1 | `dl.act.swish` |
| Rand, Objective Criteria for the Evaluation of Clustering Methods, 1971；Hubert & Arabie, Comparing Partitions, 1985 | 1 | `ml.metric.ari` |
| Ranzato et al. 2007；LeCun et al. 1998 的子采样层 | 1 | `dl.layer.maxpool1d` |
| Ranzato et al., Unsupervised Learning of Invariant Feature Hierarchies, 2007；ONNX MaxPool | 1 | `dl.layer.maxpool2d` |
| RFC 1866 HTML 2.0 / HTML5 表格元素 | 1 | `sys.doc.htmlExport` |
| RFC 4180 CSV 格式 | 1 | `sys.doc.csvExport` |
| RFC 6901 JSON Pointer（点路径简化版） | 1 | `sys.util.pick` |
| RFC 8259 JSON 交换格式 | 1 | `sys.doc.jsonExport` |
| Richardson 1910 / Numerical Recipes §4.2 | 1 | `num.richardson` |
| Ridder 1979 | 1 | `num.ridder` |
| Ridge/Tikhonov 正则化 | 1 | `ad.l2Reg` |
| Rockafellar 1970, Convex Analysis (subgradient theory) | 1 | `ad.sign` |
| Romberg 1955 / Numerical Recipes §4.3 | 1 | `num.romberg` |
| Rosenblatt 1958 | 1 | `ml.perceptron.predict` |
| Rosenblatt 1958 感知机；ONNX Operator Standard: Gemm | 1 | `dl.layer.dense` |
| Rosenblatt, The Perceptron: A Probabilistic Model for Information Storage, 1958 | 1 | `ml.perceptron.fit` |
| Rosenbrock, An Automatic Method for Finding the Greatest or Least Value of a Function, 1960 | 1 | `opt.util.rosenbrock` |
| Rousseeuw, Silhouettes: A Graphical Aid to the Interpretation of Cluster Analysis, 1987 | 1 | `ml.metric.silhouette` |
| runOptimizer 的轨迹切片视图 | 1 | `opt.util.lossTrajectory` |
| Russakovsky et al. (2015) ImageNet Large Scale Visual Recognition Challenge, IJCV 115(3) | 1 | `train.eval.topKAcc` |
| Rust slice::chunks / NumPy reshape 分块语义 | 1 | `sys.util.chunk` |
| Saad《Iterative Methods》§4.3 | 1 | `linalg.gaussSeidel` |
| Salimans & Kingma, Weight Normalization: A Simple Reparameterization to Accelerate Training, NIPS 2016 | 1 | `dl.norm.weightnorm` |
| Salton & McGill 1983；scikit-learn normalize | 1 | `ml.scale.normalize` |
| Schölkopf & Smola 2002；Salton & McGill《Introduction to Modern IR》1983（余弦测度） | 1 | `ml.kernel.cosine` |
| Schölkopf & Smola, Learning with Kernels, 2002 §2.1 | 1 | `ml.kernel.linear` |
| Schölkopf & Smola, Learning with Kernels, 2002 §4.4（L1 距离对应的平稳核） | 1 | `ml.kernel.laplacian` |
| Schölkopf & Smola, Learning with Kernels, 2002；Mercer 1909（正定性） | 1 | `ml.kernel.rbf` |
| Schölkopf & Smola, Learning with Kernels, 2002；Vapnik 1995 | 1 | `ml.kernel.poly` |
| Schölkopf, Smola & Müller 1998 | 1 | `ml.kpca.transform` |
| Schölkopf, Smola & Müller, Nonlinear Component Analysis as a Kernel Eigenvalue Problem, 1998 | 1 | `ml.kernel.center` |
| Schölkopf, Smola & Müller, Nonlinear Component Analysis as a Kernel Eigenvalue Problem, Neural Computation, 1998 | 1 | `ml.kpca.fit` |
| Schroff et al., FaceNet: A Unified Embedding for Face Recognition and Clustering, CVPR 2015 | 1 | `dl.loss.triplet` |
| Schuster & Paliwal, Bidirectional Recurrent Neural Networks, IEEE TSP 1997 | 1 | `dl.seq.bidirectional` |
| scikit-learn 估计器内省 API；Buitinck et al. 2013 | 1 | `ml.util.describe` |
| scikit-learn BaseEstimator.clone / get_params 设计；Gamma et al.《Design Patterns》原型模式 1994 | 1 | `ml.util.clone` |
| scikit-learn BaseEstimator.get_params 约定（Buitinck et al., API design for ML software, 2013） | 1 | `ml.util.getParams` |
| scikit-learn BaseEstimator.set_params 约定；Buitinck et al. 2013 | 1 | `ml.util.setParams` |
| scikit-learn class_weight='balanced'；King & Zeng, Logistic Regression in Rare Events Data, 2001 | 1 | `train.util.classWeights` |
| scikit-learn MaxAbsScaler；Hastie et al., ESL 2009 §3.4 | 1 | `ml.scale.maxabs` |
| scikit-learn MinMaxScaler；Han & Kamber《Data Mining》2006 §3.5 | 1 | `ml.scale.minmax` |
| Shannon 1948；注意力可解释性分析（ entropy 作为注意力塌缩指标） | 1 | `dl.attn.entropy` |
| Shannon 1948；Cover & Thomas, Elements of Information Theory §2.1 | 1 | `prob.info.entropy` |
| Shannon 1948；PyTorch 的 NLLLoss 与概率输入的交叉熵 | 1 | `dl.loss.ce` |
| Shazeer et al., Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer, ICLR 2017 | 1 | `dl.model.moeRouter` |
| Shazeer, GLU Variants Improve Transformer, 2020；Touvron et al., LLaMA, 2023 | 1 | `dl.model.swigluFFN` |
| Smirnov 1939；Massey 1951（两样本 KS 分布） | 1 | `prob.test.ks2Sample` |
| Smith (2017) Cyclical Learning Rates for Training Neural Networks (SGDR); Loshchilov & Hutter (2017) SGDR, ICLR | 1 | `train.lr.hook` |
| Smith, Cyclical Learning Rates for Training Neural Networks, 2017 | 1 | `opt.lr.triangular` |
| Sneath & Sokal 1973 | 1 | `ml.agglo.labels` |
| Sneath & Sokal《Numerical Taxonomy》1973；Sokal & Michener, A Statistical Method for Evaluating Systematic Relationships, 1958 | 1 | `ml.agglo.fit` |
| Sneath 1957；Lance & Williams, A General Theory of Classificatory Sorting Strategies, 1967 | 1 | `ml.linkage.single` |
| Sokolova & Lapalme (2009) IP&M 45(4); Cohen (1960); Matthews (1975); Brier (1950) | 1 | `train.eval.fullReport` |
| SOR 经典方法；Saad §4.4 | 1 | `linalg.sor` |
| Srivastava et al. (2014) Dropout: A Simple Way to Prevent Neural Networks from Overfitting, JMLR 15 | 1 | `train.aug.featureDropout` |
| Srivastava et al. 2014 | 1 | `ad.dropout` |
| Srivastava et al., Dropout: A Simple Way to Prevent Neural Networks from Overfitting, JMLR 2014 | 1 | `dl.reg.dropout` |
| Stirling 展开（DLMF 5.11.1） | 1 | `num.lgamma` |
| Stirling 展开（DLMF 5.11.1）；Numerical Recipes §6.1（递推策略） | 1 | `num.gamma` |
| Stone (1974) JRSS-B; Kohavi (1995) A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection, IJCAI | 1 | `train.eval.crossValidate` |
| Su et al., RoFormer, 2021；GPT-J/LLaMA 的 half-split 实现惯例 | 1 | `dl.attn.rotaryApply` |
| Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding, 2021 | 1 | `dl.attn.rotaryEmbed` |
| Szegedy et al. (2016) Rethinking the Inception Architecture for Computer Vision, CVPR; Müller et al. (2019) When Does Label Smoothing Help?, NeurIPS | 1 | `train.labelSmoothing` |
| Szegedy et al., Rethinking the Inception Architecture for Computer Vision, CVPR 2016 | 1 | `dl.loss.labelSmoothCE` |
| Tarjan 1972 强连通分量 / Hopcroft & Ullman 1979 图算法 | 1 | `sys.graph.cycleDetect` |
| TensorFlow tf.truncated_normal_initializer；Salimans & Kingma 2016 提及 | 1 | `dl.init.truncatedNormal` |
| Thomas 1949；Conte & de Boor | 1 | `linalg.tridiagSolve` |
| Tibshirani 1996；Hastie et al., ESL 2009 §3.4.2 | 1 | `ml.lasso.predict` |
| Tibshirani, Regression Shrinkage and Selection via the Lasso, JRSS-B 1996；深度学习权重衰减实践 | 1 | `dl.reg.l1` |
| Tibshirani, Regression Shrinkage and Selection via the Lasso, JRSS-B, 1996；Beck & Teboulle, FISTA, 2009 | 1 | `ml.lasso.fit` |
| Toeplitz 系统标准；Golub & Van Loan §4.7 | 1 | `linalg.toeplitzSolve` |
| Tong, The Multivariate Normal Distribution；Evans & Rosenthal | 1 | `prob.mvn.pdf` |
| Tong, The Multivariate Normal Distribution（Cholesky 抽样法） | 1 | `prob.mvn.sample` |
| Torgerson, Multidimensional Scaling: I. Theory and Method, Psychometrika, 1952；Mardia et al. 1979 | 1 | `ml.mds.classical` |
| Transformer 变长批处理惯例（Vaswani et al. 2017 的工程实现） | 1 | `dl.attn.paddingMask` |
| Ulyanov et al., Instance Normalization: The Missing Ingredient for Fast Stylization, 2016 | 1 | `dl.norm.instancenorm` |
| Unix 进程退出码约定 / LLVM lit 退出码 | 1 | `sys.self.exitCode` |
| Vallender 1974；Olkin & Pukelsheim 1982（正态间 W1 = \|μ1−μ2\|） | 1 | `prob.info.wasserstein1` |
| Van Rijsbergen (1979) Information Retrieval, 2nd ed.（F 度量）; Sokolova & Lapalme (2009) IP&M | 1 | `train.eval.fbeta` |
| Varma & Simon (2006) Bias in Error Estimation When Using Cross-Validation for Model Selection, BMC Bioinformatics 7:91; Stone (1974) JRSS-B | 1 | `train.eval.nestedCV` |
| Vaswani et al. 2017 的 encoder-decoder attention | 1 | `dl.attn.crossAttention` |
| Vaswani et al. 2017 的多头 reshape 惯例 | 1 | `dl.attn.headSplit` |
| Vaswani et al. 2017 的注意力核心（去掉线性投影的裸形式） | 1 | `dl.attn.selfAttention` |
| Vaswani et al. 2017 解码器掩码 | 1 | `ad.causalMask` |
| Vaswani et al. 2017（Attention is All You Need） | 1 | `ad.scaledDotProductAttention` |
| Vaswani et al., Attention Is All You Need, NeurIPS 2017（decoder 的 subsequent mask） | 1 | `dl.attn.causalMask` |
| von Neumann 1951（拒绝采样）；Marsaglia & Bray 1964（正态的 Laplace 包络） | 1 | `prob.samp.rejection` |
| Wilkinson 1968；Numerical Recipes §11.3 | 1 | `linalg.eigenSymQR` |
| Wolfe, Convergence Conditions for Ascent Methods, SIAM Review 1971；Nocedal & Wright《Numerical Optimization》§3.1 | 1 | `opt.line.wolfeLite` |
| Wright, Correlation and Causation, J. Agricultural Research, 1921；Hastie et al., ESL 2009 §3.2 | 1 | `ml.metric.r2` |
| Wu & He, Group Normalization, ECCV 2018 | 1 | `dl.norm.groupnorm` |
| Yong et al., Gradient Centralization for Better Training Performance, CVPR 2020 | 1 | `opt.grad.centralization` |
| Z-score 标准化与 3σ 离群准则：Iglewicz & Hoaglin (1993) How to Detect and Handle Outliers, ASQ Press | 1 | `train.data.standardizerFit` |
| ZCA 白化标准（Bell & Sejnowski 1996） | 1 | `linalg.whiten` |
| Zhang & Sennrich, Root Mean Square Layer Normalization, NeurIPS 2019 | 1 | `dl.norm.rmsnorm` |
| Zhang et al. (2018) mixup: Beyond Empirical Risk Minimization, ICLR | 1 | `train.mixup` |
| Zhang et al., mixup: Beyond Empirical Risk Minimization, ICLR 2018 | 1 | `dl.reg.mixup` |
| Zhang, Hinton et al., Lookahead Optimizer: k Steps Forward, 1 Step Back, NeurIPS 2019 | 1 | `opt.lookahead` |
| Zou & Hastie 2005 | 1 | `ml.elasticNet.predict` |
| Zou & Hastie, Regularization and Variable Selection via the Elastic Net, JRSS-B 2005 | 1 | `dl.reg.l1l2` |
| Zou & Hastie, Regularization and Variable Selection via the Elastic Net, JRSS-B, 2005 | 1 | `ml.elasticNet.fit` |
