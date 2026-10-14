# 贡献指南（CONTRIBUTING）

forge-1024 是一个**编目严格守恒**的项目：10 域 × 102 条 = **1024 条**，不多不少。
因此贡献流程与一般开源项目不同——**新增一条指令必须先移除或合并一条同域指令**。

---

## 1. 环境

```bash
node --version   # ≥ 18
# 无需 npm install —— 本项目零依赖
```

---

## 2. 新增 / 修改一条指令

一条指令的完整定义（`defineOp`）必须包含 8 个字段：

```js
defineOp({
  id: nid(),                 // 由域内计数器生成，形如 D3-042
  domain: 3,                 // 域编号 1..10
  name: 'linalg.xxx',        // 全局唯一；必须以域前缀开头（见下表）
  arity: 2,                  // 必须严格等于 impl.length（含可选参数）
  sig: '(Tensor A, Tensor B) -> Tensor',
  doc: '一句话中文说明',
  ref: '作者, 标题, 年份/会议',   // 真实出处，不得编造
  inv: '可被测试断言的数学性质',
  impl: (A, B) => { /* 纯函数 */ },
});
```

域前缀表：

| 域 | 前缀 | 域 | 前缀 |
|---|---|---|---|
| D1 | `math.` | D6 | `prob.` |
| D2 | `num.` | D7 | `ml.` |
| D3 | `linalg.` | D8 | `dl.` |
| D4 | `ad.` | D9 | `train.` |
| D5 | `opt.` | D10 | `sys.` |

### 硬性规则

1. **零依赖**：不得引入任何 npm 包，不得使用 `node:fs`、`node:net` 等（D10 只允许 Node 内置的环境信息 API）。
2. **禁止 `Math.random`**：一切随机用 `createRng(seed)`（PCG32），保证逐位可复现。
3. **数值稳健**：涉及 `exp/log/除` 必须有 eps 保护；softmax/logsumexp 一律减最大值。
4. **不变量优先**：先写 `inv`，再写实现，并为 `inv` 写一条独立交叉验证的测试。
5. **纯度**：`impl` 为纯函数，不得修改入参张量；state 为可 JSON 序列化的纯对象 + 普通数组。

---

## 3. 测试要求

- 每个域一个测试文件 `test/dN-*.test.mjs`，用 `node:test` + `node:assert/strict`。
- 至少包含：编目数 = 102、id 连续唯一、**该域核心不变量的交叉验证**。
- **测试失败默认修实现**，只有在三种情况下可调整断言（并在注释中写明理论依据）：
  1. 算法固有性质（如 Adadelta 的 RMS(Δx) 振荡、signSGD 的 ±η 收敛邻域）；
  2. 浮点真值不可表示（如 `1e16 + 1 == 1e16`）；
  3. 启发式初始化（改为固定 seed 后断言）。

---

## 4. 提交前自检

```bash
node tools/selftest.mjs        # 编目硬金标准
node --test test/              # 全量测试
node tools/gen-ops-doc.mjs     # 重新生成文档（必须提交）
node bench/bench.mjs           # 重新生成基准（提交）
node tools/bundle-html.mjs     # 重新生成单文件产物
node tools/verify-bundle.mjs   # 产物自洽性
```

CI 会在 ubuntu / windows / macos × Node 18/20/22 上运行全部检查。

---

## 5. 文档

**不要手工编辑 `docs/OPS.md` 与 `docs/REFERENCES.md`** —— 它们由 `tools/gen-ops-doc.mjs` 从注册表反射生成。
要改文档，改指令的 `doc` / `ref` / `inv` 字段，然后重新生成。

---

## 6. 提交信息规范

```
<域> <动作>: <简述>

D3 fix: 修正 SVD 对宽矩阵的 U/V 交换
D8 feat: 新增 dl.attn.groupedQuery（合并 dl.attn.multiHeadVariant）
```

---

## 7. 许可

贡献即视为同意以 MIT 许可发布。
