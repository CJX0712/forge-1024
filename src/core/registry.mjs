// 指令注册表（Op Registry）+ 执行器。
// 设计基准：ONNX Operator Standard（opset 25，170+ 算子，ai.onnx 域）的
// 「命名 + 签名 + 文档 + 版本化」组织方式；PyTorch aten native_functions 的注册模式。
// 每条微指令是一个纯函数 + 元数据（签名/文档/引用/不变量），可被 exec 统一调度、遥测、校验。
import { ForgeError } from './errors.mjs';
import * as telemetry from './telemetry.mjs';

export const KERNEL_VERSION = '1.0.0';
export const CATALOG_VERSION = '2026.10';

export const DOMAINS = {
  1: { code: 'MATH', name: '数学基元', desc: '逐元素运算、归约、范数、距离、构造器' },
  2: { code: 'NUM', name: '数值方法与稳定化', desc: '稳定softmax、补偿求和、误差度量、求根、积分、差分、特殊函数' },
  3: { code: 'LINALG', name: '线性代数', desc: '分解族（LU/QR/Cholesky/Jacobi/SVD）、求解器、迭代法、矩阵函数' },
  4: { code: 'AUTODIFF', name: '自动微分', desc: '反向模式 vjp、前向模式对偶数、梯度检验' },
  5: { code: 'OPT', name: '优化器', desc: '16 种经典优化器、学习率调度、梯度变换、线搜索' },
  6: { code: 'PROB', name: '概率统计', desc: '分布族、信息论、矩估计、贝叶斯推断、采样与拟合检验' },
  7: { code: 'ML', name: '机器学习', desc: '经典模型族、评估指标、数据预处理' },
  8: { code: 'DL', name: '深度学习', desc: '层、激活、注意力、循环网络、损失、初始化' },
  9: { code: 'TRAIN', name: '训练与评估', desc: '数据管线、训练循环、交叉验证、可复现性' },
  10: { code: 'SYS', name: '系统编排', desc: '注册表内省、遥测、组合、文档生成、基准、模糊测试' },
};

const registry = new Map(); // id -> spec
const byName = new Map();   // name -> spec

const REQUIRED = ['id', 'domain', 'name', 'sig', 'doc', 'ref', 'inv', 'impl'];

// —— arity 机器推导 ——
// 约定：arity = 「必需形参个数」（第一个带默认值 / rest 形参之前的形参数），
// 与 exec 的 `args.length >= arity` 语义严格对应。
// 之所以不让作者手写：手写值会与 impl 漂移（历史上一度出现 165 条缺失、51 条不符）。
// 由实现反推元数据，是唯一不可能漂移的做法。
export function requiredParams(fn) {
  if (typeof fn !== 'function') return 0;
  const src = fn.toString().trim();
  let list = null;
  let m = src.match(/^(?:async\s+)?function\s*[\w$]*\s*\(([\s\S]*?)\)/);
  if (m) list = m[1];
  if (list === null) {
    m = src.match(/^(?:async\s+)?\(([\s\S]*?)\)\s*=>/);
    if (m) list = m[1];
  }
  if (list === null) {
    // 单参数裸箭头：x => ...（无括号）
    if (/^(?:async\s+)?[\w$]+\s*=>/.test(src)) return 1;
    if (/^(?:async\s+)?\(\s*\)\s*=>/.test(src)) return 0;
    return 0;
  }
  // 按逗号切分，但跳过括号/花括号/方括号/引号内部的逗号（默认值里可能带对象或数组字面量）
  const parts = [];
  let depth = 0, cur = '', quote = null;
  for (const ch of list) {
    if (quote) { cur += ch; if (ch === quote) quote = null; continue; }
    if (ch === '"' || ch === "'" || ch === '`') { quote = ch; cur += ch; continue; }
    if (ch === '(' || ch === '[' || ch === '{') { depth++; cur += ch; continue; }
    if (ch === ')' || ch === ']' || ch === '}') { depth--; cur += ch; continue; }
    if (ch === ',' && depth === 0) { parts.push(cur.trim()); cur = ''; continue; }
    cur += ch;
  }
  if (cur.trim()) parts.push(cur.trim());
  // 函数体（形参表之后的部分），用于识别「实现自带兜底」的可选形参
  const body = src.slice(src.indexOf(list) + list.length + 1);
  const hasFallback = (name) => {
    const esc = name.replace(/\$/g, '\\$');
    // p ?? 默认值      —— 明确的空值合并兜底
    // p === undefined ? … : …  / p == null ? … : …  —— 三元兜底
    return new RegExp(`\\b${esc}\\s*\\?\\?`).test(body)
      || new RegExp(`\\b${esc}\\s*(?:===|==)\\s*undefined\\s*\\?`).test(body)
      || new RegExp(`\\b${esc}\\s*(?:===|==)\\s*null\\s*\\?`).test(body)
      // 守卫式可选：if (p !== undefined) / if (p != null) —— 实现自行处理缺省
      || new RegExp(`\\b${esc}\\s*!==?\\s*(?:undefined|null)\\b`).test(body);
  };
  let n = 0;
  for (const p of parts) {
    if (p.startsWith('...')) break;   // rest 形参及其后全部可选
    if (p.includes('=')) break;       // 第一个带默认值的形参起为可选
    const name = p.split(':')[0].trim();
    if (/^[A-Za-z_$][\w$]*$/.test(name) && hasFallback(name)) break; // 实现自带兜底 ⇒ 可选
    n++;
  }
  return n;
}

export function defineOp(spec) {
  for (const k of REQUIRED) {
    if (spec[k] === undefined || spec[k] === null) {
      throw new ForgeError('E_VALUE', `defineOp 缺少必填字段 "${k}"（op: ${spec.id || spec.name || '?'}）`);
    }
  }
  if (registry.has(spec.id)) throw new ForgeError('E_VALUE', `指令 id 重复: ${spec.id}`);
  if (byName.has(spec.name)) throw new ForgeError('E_VALUE', `指令 name 重复: ${spec.name}`);
  if (!DOMAINS[spec.domain]) throw new ForgeError('E_VALUE', `未知域: ${spec.domain}`);
  const full = {
    check: spec.check ?? null,
    test: spec.test ?? null,
    tags: spec.tags ?? [],
    ...spec,
    // arity 一律由 impl 形参表机器推导，放在展开之后以保证不被作者手写值覆盖
    // （历史教训：手写 arity 一度出现 165 条缺失、51 条与实现不符）。
    arity: requiredParams(spec.impl),
  };
  registry.set(full.id, full);
  byName.set(full.name, full);
  return full;
}

export function getOp(idOrName) {
  const op = registry.get(idOrName) || byName.get(idOrName);
  if (!op) throw new ForgeError('E_NOTFOUND', `指令不存在: ${idOrName}`);
  return op;
}

export function hasOp(x) { return registry.has(x) || byName.has(x); }

export function listOps({ domain } = {}) {
  const all = [...registry.values()];
  return domain == null ? all : all.filter(o => o.domain === domain);
}

export function countOps() { return registry.size; }

export function opsByDomain(d) { return [...registry.values()].filter(o => o.domain === d); }

export function searchOps(q) {
  const s = String(q).toLowerCase();
  return [...registry.values()].filter(o =>
    o.name.toLowerCase().includes(s) || o.doc.toLowerCase().includes(s) || o.id.toLowerCase().includes(s));
}

// 统一执行入口：校验 -> 调用 -> 遥测 -> 错误归一化。
// arity 语义 = 最少必需参数个数（带默认值的可选参数不计入）。
export function exec(idOrName, args = []) {
  const op = getOp(idOrName);
  if (op.arity != null && args.length < op.arity) {
    telemetry.recordError(op.id);
    throw new ForgeError('E_ARITY', `指令 ${op.id} (${op.name}) 至少需要 ${op.arity} 个参数，收到 ${args.length}`);
  }
  const t0 = telemetry.now();
  try {
    if (op.check) op.check(args);
    return op.impl(...args);
  } catch (e) {
    telemetry.recordError(op.id);
    if (e instanceof ForgeError) throw e;
    throw new ForgeError('E_NUMERICAL', `指令 ${op.id} (${op.name}) 执行失败: ${e.message}`);
  } finally {
    telemetry.record(op.id, t0);
  }
}

// 不抛错的执行：返回 {ok, value|error}，供上层编排与模糊测试使用。
export function tryExec(idOrName, args = []) {
  try { return { ok: true, value: exec(idOrName, args) }; }
  catch (e) { return { ok: false, error: e }; }
}

export function describeOp(x) {
  const op = getOp(x);
  return `${op.id} ${op.name} — ${op.doc} | ${op.sig} | 参考: ${op.ref} | 不变量: ${op.inv}`;
}
