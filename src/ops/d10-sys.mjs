// 域10 系统编排 SYS（102 条微指令）
// 对标对象：
//   · ONNX Operator Registry（opset 域化命名 + 签名 + 文档 + 版本化）→ sys.reg.*
//   · LLVM Pass Manager（pass 编排、依赖图、流水线调度）        → sys.graph.* / sys.exec.pipe
//   · PyTorch FX（符号图追踪与变换 IR）                          → sys.graph.* / sys.doc.mermaid
//   · Prometheus 指标模型（counter / summary / 分位数）          → sys.tel.*
//   · Hypothesis 属性测试框架（基于策略的随机反例搜索）          → sys.verify.fuzz / determinism
// 组织结构：12 注册表内省 + 8 遥测 + 9 执行组合 + 12 验证 + 9 文档生成 + 8 基准
//           + 8 图编排 + 8 配置 + 7 错误 + 7 环境 + 6 自检 + 8 工具 = 102 条。
// 分层依赖：域10 只依赖内核（core/registry + core/telemetry + core/errors），绝不 import d2..d9；
//           需要数学基元时由调用方注入 op 列表，避免循环依赖。
// 纯函数约定：sys.doc.* 全部返回字符串（不落盘）；随机一律走 createRng(seed)，禁用 Math.random。
import {
  defineOp, exec, tryExec, getOp, hasOp, listOps, countOps, opsByDomain, searchOps,
  describeOp, DOMAINS, KERNEL_VERSION, CATALOG_VERSION, telemetry,
  createRng, seedFrom, ForgeError, CODES, Tensor, expectNumber, expectTensor, requiredParams,
} from '../core/index.mjs';

const D = 10;
let _s = 0;
const nid = () => `D${D}-${String(++_s).padStart(3, '0')}`;

// =====================================================================
// 内部小工具（非指令）
// =====================================================================

// 高精度单调时钟（纳秒级），毫秒浮点。基准与遥测统一走这里。
const _now = () => {
  if (typeof process !== 'undefined' && process.hrtime && process.hrtime.bigint) {
    return Number(process.hrtime.bigint()) / 1e6;
  }
  if (typeof performance !== 'undefined' && performance.now) return performance.now();
  return 0;
};

// FNV-1a 64 位（Fowler–Noll–Vo）；以 BigInt 精确实现，避免 32 位碰撞。
const _M64 = (1n << 64n) - 1n;
const _P64 = 0x100000001b3n;
function fnv1a64(str) {
  let h = 0xcbf29ce484222325n;
  const s = String(str);
  for (let i = 0; i < s.length; i++) {
    h = (h ^ BigInt(s.charCodeAt(i) & 0xff)) & _M64;
    h = (h * _P64) & _M64;
  }
  return h;
}
const _hex64 = (h) => '0x' + h.toString(16).padStart(16, '0');

// 稳定序列化：用于确定性比较、配置哈希、CSV/JSON 导出。
//   -0 归一为 0；NaN 字面量化；Tensor 展开为 shape+data；对象键排序。
function _stable(v) {
  if (v === null) return 'null';
  if (v === undefined) return 'undef';
  const t = typeof v;
  if (t === 'number') return Object.is(v, -0) ? '0' : (Number.isNaN(v) ? 'NaN' : String(v));
  if (t === 'bigint') return v.toString() + 'n';
  if (t === 'boolean' || t === 'string') return JSON.stringify(v);
  if (t === 'function') return 'fn';
  if (v instanceof Tensor) return `T[${v.shape.join(',')}](${Array.from(v.data).map(_stable).join(',')})`;
  if (v instanceof Float64Array) return `F[${Array.from(v).map(_stable).join(',')}]`;
  if (Array.isArray(v)) return `[${v.map(_stable).join(',')}]`;
  const keys = Object.keys(v).sort();
  return `{${keys.map(k => JSON.stringify(k) + ':' + _stable(v[k])).join(',')}}`;
}

// 转成 JSON 安全结构（函数折叠为占位串，键排序），供 sys.doc.jsonExport 使用。
function _plain(v) {
  if (v === null || v === undefined) return v;
  const t = typeof v;
  if (t === 'number') return Number.isFinite(v) ? v : String(v);
  if (t === 'string' || t === 'boolean') return v;
  if (t === 'bigint') return v.toString();
  if (t === 'function') return '[function]';
  if (v instanceof Tensor) return { shape: v.shape.slice(), data: Array.from(v.data) };
  if (v instanceof Float64Array) return Array.from(v);
  if (Array.isArray(v)) return v.map(_plain);
  const out = {};
  for (const k of Object.keys(v).sort()) out[k] = _plain(v[k]);
  return out;
}

// op 元信息摘要（剔除 impl 等不可序列化字段）
const _sum = (op) => (op && typeof op === 'object')
  ? {
    id: op.id ?? null, name: op.name ?? null, domain: op.domain ?? null,
    arity: op.arity ?? null, sig: op.sig ?? '', doc: op.doc ?? '', ref: op.ref ?? '', inv: op.inv ?? '',
  }
  : op;
const _sums = (ops) => (Array.isArray(ops) ? ops : []).map(_sum);

// op 查找：先在入参 ops 中按 name/id 找，再回落到全局注册表。
function _findOp(ops, x) {
  if (x && typeof x === 'object') return x;
  const arr = Array.isArray(ops) ? ops : [];
  const hit = arr.find(o => o && (o.name === x || o.id === x));
  if (hit) return hit;
  if (typeof x === 'string' && hasOp(x)) return getOp(x);
  throw new ForgeError('E_NOTFOUND', `指令不存在: ${x}`);
}
const _nameOf = (id) => { try { return hasOp(id) ? getOp(id).name : id; } catch { return id; } };

// 本地观测表：telemetry 只暴露 record/recordError（语义为「已发生」），
// 域10 另维护一张可手动写入的观测表，供 sys.tel.record 使用。
const _obs = new Map();  // key -> {calls, errors}
const _lat = new Map();  // key -> number[]（毫秒）
const _memo = new Map(); // 记忆化缓存

function _obsGet(k) {
  if (!_obs.has(k)) _obs.set(k, { calls: 0, errors: 0 });
  return _obs.get(k);
}
function _latAdd(k, ms) {
  if (!_lat.has(k)) _lat.set(k, []);
  _lat.get(k).push(ms);
}
// 合并「内核遥测」与「本地观测」为统一行。
function _mergedRows() {
  const map = new Map();
  for (const r of telemetry.summary()) {
    map.set(r.id, { id: r.id, name: _nameOf(r.id), calls: r.calls, errors: r.errors, ms: r.ms });
  }
  for (const [k, v] of _obs) {
    const id = (typeof k === 'string' && hasOp(k)) ? getOp(k).id : k;
    if (!map.has(id)) map.set(id, { id, name: _nameOf(id), calls: 0, errors: 0, ms: 0 });
    const e = map.get(id);
    e.calls += v.calls; e.errors += v.errors;
  }
  return [...map.values()];
}
function _rowsByIdOrName(rows, x) {
  if (x === undefined || x === null || x === '') return rows;
  const id = (typeof x === 'string' && hasOp(x)) ? getOp(x).id : x;
  return rows.filter(r => r.id === id || r.name === x);
}

// —— 签名解析：'(Tensor a, number[] shape) -> Tensor' -> ['Tensor','number[]'] ——
function _sigTypes(sig) {
  const m = /\(([^)]*)\)/.exec(String(sig == null ? '' : sig));
  if (!m || m[1].trim() === '') return [];
  return m[1].split(',').map(s => s.trim()).filter(Boolean).map((s) => {
    const mm = /^([A-Za-z_$][\w$<>]*(?:\[\])*)/.exec(s);
    return mm ? mm[1] : 'any';
  });
}

// —— 按签名类型生成随机入参（Hypothesis 风格的策略推导）——
function _genOne(t, rng) {
  switch (t) {
    case 'Tensor': return Tensor.fromFlat([rng.range(-4, 4), rng.range(-4, 4), rng.range(-4, 4)], [3]);
    case 'Tensor[]': return [Tensor.fromFlat([rng.range(-4, 4), rng.range(-4, 4)], [2])];
    case 'number': return rng.range(-8, 8);
    case 'number[]': return [2, 2];
    case 'number[][]': return [[1, 1], [1, 1]];
    case 'string': return 'abc';
    case 'boolean': return true;
    case 'object': return {};
    case 'array': return [1, 2];
    default:
      if (/Tensor/.test(t)) return Tensor.fromFlat([1, 2, 3], [3]);
      return rng.range(-8, 8);
  }
}
function _genArgs(op, rng) {
  const types = _sigTypes(op && op.sig);
  const n = types.length ? types.length : Math.max(1, (op && op.arity) || 1);
  const out = [];
  for (let i = 0; i < n; i++) out.push(_genOne(types[i] || 'any', rng));
  return out;
}

// —— 图工具（Kahn / DFS 染色）——
function _nodesOf(g) {
  const nodes = Array.isArray(g) ? g : (g && Array.isArray(g.nodes) ? g.nodes : null);
  if (!nodes) throw new ForgeError('E_TYPE', 'graph 需为 {nodes:[...]} 或节点数组');
  return nodes;
}
function _adjOf(nodes) {
  const ids = nodes.map(n => n.id);
  const adj = new Map(ids.map(i => [i, []]));
  const indeg = new Map(ids.map(i => [i, 0]));
  for (const n of nodes) {
    for (const src of (n.inputs || [])) {
      if (!adj.has(src)) throw new ForgeError('E_NOTFOUND', `节点 ${n.id} 引用了不存在的上游: ${src}`);
      adj.get(src).push(n.id);
      indeg.set(n.id, indeg.get(n.id) + 1);
    }
  }
  return { ids, adj, indeg };
}
function _topo(nodes) {
  const { ids, adj, indeg } = _adjOf(nodes);
  const q = ids.filter(i => indeg.get(i) === 0);
  const order = [];
  while (q.length) {
    const cur = q.shift();
    order.push(cur);
    for (const nx of adj.get(cur)) {
      indeg.set(nx, indeg.get(nx) - 1);
      if (indeg.get(nx) === 0) q.push(nx);
    }
  }
  return { order, ok: order.length === ids.length };
}
function _findCycle(nodes) {
  const { ids, adj } = _adjOf(nodes);
  const color = new Map(ids.map(i => [i, 0]));
  const stack = [];
  let cycle = null;
  const dfs = (u) => {
    color.set(u, 1); stack.push(u);
    for (const v of adj.get(u)) {
      if (cycle) return;
      const c = color.get(v);
      if (c === 0) dfs(v);
      else if (c === 1) { cycle = stack.slice(stack.indexOf(v)).concat([v]); return; }
    }
    color.set(u, 2); stack.pop();
  };
  for (const i of ids) { if (!cycle && color.get(i) === 0) dfs(i); }
  return cycle; // null 或 [a,b,...,a]
}
const _edgesOf = (nodes) => {
  const es = [];
  for (const n of nodes) for (const s of (n.inputs || [])) es.push([s, n.id]);
  return es;
};

// —— 配置工具 ——
const _isPlainObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v) && !(v instanceof Tensor);
function _deepMerge(a, b) {
  if (!_isPlainObj(a) || !_isPlainObj(b)) return b === undefined ? a : b;
  const out = { ...a };
  for (const k of Object.keys(b)) {
    out[k] = (_isPlainObj(out[k]) && _isPlainObj(b[k])) ? _deepMerge(out[k], b[k]) : b[k];
  }
  return out;
}
function _deepFreeze(o) {
  if (!_isPlainObj(o) && !Array.isArray(o)) return o;
  for (const k of Object.keys(o)) {
    const v = o[k];
    if ((_isPlainObj(v) || Array.isArray(v)) && !Object.isFrozen(v)) _deepFreeze(v);
  }
  return Object.freeze(o);
}
function _walkDiff(a, b, path, acc) {
  if (_isPlainObj(a) && _isPlainObj(b)) {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    for (const k of keys) {
      const p = path ? `${path}.${k}` : k;
      if (!(k in a)) acc.added.push(p);
      else if (!(k in b)) acc.removed.push(p);
      else _walkDiff(a[k], b[k], p, acc);
    }
    return;
  }
  if (_stable(a) !== _stable(b)) acc.changed.push(path || '');
}
function _typeOk(v, t) {
  switch (t) {
    case 'number': return typeof v === 'number' && Number.isFinite(v);
    case 'int': return Number.isInteger(v);
    case 'string': return typeof v === 'string';
    case 'boolean': return typeof v === 'boolean';
    case 'object': return _isPlainObj(v);
    case 'array': return Array.isArray(v);
    case 'function': return typeof v === 'function';
    case 'tensor': return v instanceof Tensor;
    case 'any': return true;
    default: return false;
  }
}
const _esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const _csvCell = (v) => {
  const s = String(v == null ? '' : v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

// —— 错误分类学映射 ——
const ERR_CATEGORY = {
  E_ARITY: 'interface', E_TYPE: 'type', E_SHAPE: 'shape', E_VALUE: 'value',
  E_NUMERICAL: 'numerical', E_LIMIT: 'resource', E_NOTFOUND: 'lookup', E_STATE: 'state',
};
const ERR_RETRYABLE = { E_NUMERICAL: true, E_LIMIT: true };
const ERR_SUGGEST = {
  E_ARITY: '补齐参数到 op.arity 要求个数（可选参数也要占位传入）',
  E_TYPE: '核对签名类型：Tensor / number / string / object 是否与 op.sig 一致',
  E_SHAPE: '核对 shape 与广播规则，必要时先 reshape / 补齐前导维度',
  E_VALUE: '取值越界：检查正数、区间、非空集合等前置条件',
  E_NUMERICAL: '数值失败：缩放输入、换用稳定算法或提高迭代精度；属可重试类',
  E_LIMIT: '超出限额：增大迭代上限 / 时间预算后重试；属可重试类',
  E_NOTFOUND: '指令名拼写错误，或尚未 import 对应域文件完成注册',
  E_STATE: '状态机顺序错误：检查初始化与调用次序（如重复 backward）',
};
const _errCodeOf = (e) => (e && e.code) ? e.code : (e instanceof Error ? 'E_NUMERICAL' : 'E_UNKNOWN');

// =====================================================================
// 一、注册表内省 sys.reg.*（12 条）
// =====================================================================

defineOp({
  id: nid(), domain: D, name: 'sys.reg.count', arity: 1,
  sig: '(number? domain) -> number',
  doc: '指令总数；给定 domain 时返回该域条数',
  ref: 'ONNX Operator Registry（opset 域化统计）',
  inv: 'count()=countOps()；count(d)=opsByDomain(d).length；∑count(d)=count()',
  impl: (domain) => (domain === undefined || domain === null) ? countOps() : opsByDomain(Number(domain)).length,
});

defineOp({
  id: nid(), domain: D, name: 'sys.reg.domains', arity: 0,
  sig: '() -> object[]',
  doc: '列出全部域元信息（code/name/desc）及当前实际条数',
  ref: 'ONNX Operator Registry 域划分',
  inv: '返回 10 个域；每域 count = opsByDomain(d).length',
  impl: () => Object.keys(DOMAINS).map(Number).sort((a, b) => a - b).map((d) => ({
    domain: d, code: DOMAINS[d].code, name: DOMAINS[d].name, desc: DOMAINS[d].desc,
    count: opsByDomain(d).length,
  })),
});

defineOp({
  id: nid(), domain: D, name: 'sys.reg.byDomain', arity: 1,
  sig: '(number domain) -> op[]',
  doc: '按域取指令摘要列表（已剔除 impl，可安全序列化）',
  ref: 'ONNX Operator Registry 域化查询',
  inv: '结果全部 op.domain 等于入参；条数等于 sys.reg.count(domain)',
  impl: (domain) => {
    const d = Number(domain);
    if (!DOMAINS[d]) throw new ForgeError('E_VALUE', `未知域: ${d}`);
    return _sums(opsByDomain(d));
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.reg.search', arity: 1,
  sig: '(string q) -> op[]',
  doc: '按关键字检索指令（匹配 name / doc / id，忽略大小写）',
  ref: 'ONNX Operator Registry 检索语义',
  inv: '每条结果的 name/doc/id 之一包含小写化后的查询串',
  impl: (q) => {
    const s = String(q == null ? '' : q);
    return _sums(searchOps(s));
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.reg.describe', arity: 1,
  sig: '(string nameOrId) -> string',
  doc: '单条指令的完整一行描述（id / name / doc / sig / 参考 / 不变量）',
  ref: 'ONNX Operator Registry op 文档',
  inv: '输出包含 op.id 与 op.name，且与 describeOp 一致',
  impl: (nameOrId) => describeOp(String(nameOrId)),
});

defineOp({
  id: nid(), domain: D, name: 'sys.reg.exists', arity: 1,
  sig: '(string nameOrId) -> boolean',
  doc: '判断指令是否已注册（按 id 或 name）',
  ref: 'ONNX Operator Registry 存在性查询',
  inv: 'exists(x) 为真当且仅当 getOp(x) 不抛错',
  impl: (nameOrId) => (typeof nameOrId === 'string' ? hasOp(nameOrId) : false),
});

defineOp({
  id: nid(), domain: D, name: 'sys.reg.signature', arity: 1,
  sig: '(string nameOrId) -> string',
  doc: '取指令签名字符串',
  ref: 'ONNX Operator Schema（输入输出签名）',
  inv: 'signature(x) = getOp(x).sig',
  impl: (nameOrId) => _findOp(null, nameOrId).sig,
});

defineOp({
  id: nid(), domain: D, name: 'sys.reg.catalogHash', arity: 1,
  sig: '(boolean? includeDocs) -> string',
  doc: '全目录指纹：FNV-1a 64 位，按 id 排序后对 id|domain|name|arity|sig(+doc+ref+inv) 求哈希',
  ref: 'FNV-1a（Fowler–Noll–Vo 哈希）/ MurmurHash3',
  inv: '相同目录两次调用结果相同；任一 op 改名 / 改签名 / 新增一条都会改变哈希',
  impl: (includeDocs) => {
    const withDocs = includeDocs === undefined ? true : Boolean(includeDocs);
    const ops = listOps().slice().sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    const parts = ops.map(o => withDocs
      ? `${o.id}|${o.domain}|${o.name}|${o.arity}|${o.sig}|${o.doc}|${o.ref}|${o.inv}`
      : `${o.id}|${o.domain}|${o.name}|${o.arity}|${o.sig}`);
    return _hex64(fnv1a64(parts.join('\n')));
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.reg.version', arity: 0,
  sig: '() -> object',
  doc: '内核版本与目录版本',
  ref: 'ONNX opset 版本化约定',
  inv: 'version().kernel = KERNEL_VERSION；version().catalog = CATALOG_VERSION',
  impl: () => ({ kernel: KERNEL_VERSION, catalog: CATALOG_VERSION }),
});

defineOp({
  id: nid(), domain: D, name: 'sys.reg.nameIndex', arity: 0,
  sig: '() -> object',
  doc: 'name -> {id, domain, arity} 的索引表，供 O(1) 反查',
  ref: 'PyTorch aten native_functions 名称索引',
  inv: '索引键集合等于全部指令 name 集合，且每个 name 唯一',
  impl: () => {
    const out = {};
    for (const o of listOps()) out[o.name] = { id: o.id, domain: o.domain, arity: o.arity };
    return out;
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.reg.opIds', arity: 1,
  sig: '(number? domain) -> string[]',
  doc: '排序后的指令 id 列表（可选按域过滤）',
  ref: 'ONNX Operator Registry 稳定标识',
  inv: '列表按字典序升序；无重复；过滤后条数等于该域条数',
  impl: (domain) => {
    const ops = (domain === undefined || domain === null) ? listOps() : opsByDomain(Number(domain));
    return ops.map(o => o.id).sort();
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.reg.domainStats', arity: 0,
  sig: '() -> object[]',
  doc: '每个域的统计：条数、占比、首个/末个 id',
  ref: 'Prometheus 指标模型（分组计数 + 占比）',
  inv: '各域 count 之和 = countOps()；ratio ∈ [0,1] 且总和为 1',
  impl: () => {
    const total = countOps() || 1;
    return Object.keys(DOMAINS).map(Number).sort((a, b) => a - b).map((d) => {
      const ops = opsByDomain(d);
      const ids = ops.map(o => o.id).sort();
      return {
        domain: d, code: DOMAINS[d].code, name: DOMAINS[d].name,
        count: ops.length, ratio: ops.length / total,
        firstId: ids[0] ?? null, lastId: ids[ids.length - 1] ?? null,
      };
    });
  },
});

// =====================================================================
// 二、遥测 sys.tel.*（8 条）
// =====================================================================

defineOp({
  id: nid(), domain: D, name: 'sys.tel.summary', arity: 1,
  sig: '(number? k) -> row[]',
  doc: '遥测汇总：每行 {id, name, calls, errors, ms}，按调用次数降序；k 限定行数',
  ref: 'Prometheus 指标模型（counter / summary）',
  inv: '按 calls 降序；k 给定时行数 ≤ k；每行 calls ≥ errors ≥ 0',
  impl: (k) => {
    const rows = _mergedRows().sort((a, b) => b.calls - a.calls || (a.id < b.id ? -1 : 1));
    const n = (k === undefined || k === null) ? rows.length : Math.max(0, Math.floor(Number(k)));
    return rows.slice(0, n);
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.tel.hotOps', arity: 1,
  sig: '(number k) -> row[]',
  doc: '调用最频繁的 k 条指令（热点分析）',
  ref: 'Prometheus TopK 查询语义',
  inv: '返回行数 = min(k, 总行数)，且是 summary 的前缀',
  impl: (k) => {
    const rows = _mergedRows().sort((a, b) => b.calls - a.calls || (a.id < b.id ? -1 : 1));
    const n = Math.max(0, Math.floor(Number(k === undefined ? 10 : k)));
    return rows.slice(0, n);
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.tel.reset', arity: 0,
  sig: '() -> boolean',
  doc: '清空内核遥测与域10 本地观测表',
  ref: 'Prometheus 指标重置语义',
  inv: 'reset 后 totalCalls() 中来自内核的部分为 0',
  impl: () => { telemetry.reset(); _obs.clear(); _lat.clear(); return true; },
});

defineOp({
  id: nid(), domain: D, name: 'sys.tel.totalCalls', arity: 0,
  sig: '() -> number',
  doc: '累计调用次数（内核遥测 + 本地观测）',
  ref: 'Prometheus counter 单调递增语义',
  inv: 'totalCalls() = 各行 calls 之和；reset 后归零',
  impl: () => telemetry.totalCalls() + [..._obs.values()].reduce((a, v) => a + v.calls, 0),
});

defineOp({
  id: nid(), domain: D, name: 'sys.tel.record', arity: 2,
  sig: '(string opName, boolean ok) -> object',
  doc: '手动记录一次调用观测（成功/失败各计一次调用，失败额外计一次错误）',
  ref: 'OpenTelemetry 指标规范（手动埋点）',
  inv: 'ok=false 使该指令 errors+1；ok=true 仅使 calls+1',
  impl: (opName, ok) => {
    const key = String(opName);
    const e = _obsGet(key);
    e.calls += 1;
    if (!ok) e.errors += 1;
    return { op: key, calls: e.calls, errors: e.errors };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.tel.errorRate', arity: 1,
  sig: '(string? opName) -> number',
  doc: '错误率 = 错误数 / 调用数；无调用时返回 0',
  ref: 'Prometheus 比率指标（errors / total）',
  inv: 'errorRate ∈ [0,1]；分母为 0 时返回 0',
  impl: (opName) => {
    const rows = _rowsByIdOrName(_mergedRows(), opName);
    let c = 0, e = 0;
    for (const r of rows) { c += r.calls; e += r.errors; }
    return c === 0 ? 0 : e / c;
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.tel.latencyStats', arity: 0,
  sig: '() -> object',
  doc: '延迟统计（min/mean/max/p50/p95），样本来自 sys.exec.call / sys.exec.timed / sys.bench.*',
  ref: 'Prometheus Summary 分位数语义',
  inv: '样本升序后 p50 ≤ p95 ≤ max；无样本时全部为 0',
  impl: () => {
    const xs = [];
    for (const arr of _lat.values()) for (const v of arr) xs.push(v);
    for (const r of _mergedRows()) if (r.ms > 0) xs.push(r.ms);
    if (xs.length === 0) return { n: 0, min: 0, max: 0, mean: 0, p50: 0, p95: 0 };
    const s = xs.slice().sort((a, b) => a - b);
    const q = (p) => s[Math.min(s.length - 1, Math.max(0, Math.ceil(p * s.length) - 1))];
    const sum = s.reduce((a, v) => a + v, 0);
    return {
      n: s.length, min: s[0], max: s[s.length - 1],
      mean: sum / s.length, p50: q(0.5), p95: q(0.95),
    };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.tel.topErrors', arity: 1,
  sig: '(number k) -> row[]',
  doc: '错误次数最多的 k 条指令',
  ref: 'Prometheus TopK 查询语义',
  inv: '按 errors 降序；errors 为 0 的行排在末尾',
  impl: (k) => {
    const rows = _mergedRows().sort((a, b) => b.errors - a.errors || (a.id < b.id ? -1 : 1));
    const n = Math.max(0, Math.floor(Number(k === undefined ? 10 : k)));
    return rows.slice(0, n);
  },
});

// =====================================================================
// 三、执行与组合 sys.exec.*（9 条）
// =====================================================================

defineOp({
  id: nid(), domain: D, name: 'sys.exec.call', arity: 2,
  sig: '(string name, any[] args) -> any',
  doc: '统一调度执行：额外把本次耗时记入域10 延迟表',
  ref: 'LLVM Pass Manager 统一 pass 调度入口',
  inv: '返回值等于 exec(name, args)；耗时被记录到 latencyStats 样本集',
  impl: (name, args) => {
    const list = Array.isArray(args) ? args : (args === undefined ? [] : [args]);
    const t0 = _now();
    const v = exec(String(name), list);
    _latAdd(String(name), _now() - t0);
    return v;
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.exec.try', arity: 2,
  sig: '(string name, any[] args) -> object',
  doc: '不抛错执行，返回 {ok, value|error}；error 被折叠为 {code, message}',
  ref: 'LLVM Pass Manager 的可失败调度 / Rust Result 语义',
  inv: '永不抛出；ok=true 时含 value，ok=false 时含 error.code',
  impl: (name, args) => {
    const list = Array.isArray(args) ? args : (args === undefined ? [] : [args]);
    const r = tryExec(String(name), list);
    if (r.ok) return { ok: true, value: r.value, error: null };
    return { ok: false, value: null, error: { code: _errCodeOf(r.error), message: r.error.message } };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.exec.batch', arity: 1,
  sig: '(call[] calls) -> object[]',
  doc: '批量执行：calls 为 [[name, args], ...] 或 [{op, args}, ...]；逐条 try，不中断',
  ref: 'LLVM Pass Manager 批处理调度',
  inv: '输出长度等于输入长度；失败项 ok=false 且不影响其它项',
  impl: (calls) => {
    const list = Array.isArray(calls) ? calls : [];
    return list.map((c) => {
      let name, args;
      if (Array.isArray(c)) { name = c[0]; args = c[1]; }
      else if (c && typeof c === 'object') { name = c.op ?? c.name; args = c.args; }
      else throw new ForgeError('E_TYPE', `batch 元素无法解析: ${String(c)}`);
      const list2 = Array.isArray(args) ? args : (args === undefined ? [] : [args]);
      const t0 = _now();
      const r = tryExec(String(name), list2);
      const ms = _now() - t0;
      return r.ok
        ? { ok: true, op: String(name), value: r.value, ms, error: null }
        : { ok: false, op: String(name), value: null, ms, error: { code: _errCodeOf(r.error), message: r.error.message } };
    });
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.exec.pipe', arity: 2,
  sig: '(step[] steps, any x) -> any',
  doc: '管道串联：steps 为 [{op, args?}]，每步以「上一步输出 + args」为入参调用',
  ref: 'LLVM Pass Manager 流水线 / Unix 管道语义',
  inv: 'pipe([s1,s2], x) = exec(s2.op, [exec(s1.op, [x, ...s1.args]), ...s2.args])',
  impl: (steps, x) => {
    const list = Array.isArray(steps) ? steps : [];
    let cur = x;
    for (const st of list) {
      const name = typeof st === 'string' ? st : (st && (st.op ?? st.name));
      if (!name) throw new ForgeError('E_VALUE', 'pipe 步骤缺少 op 名');
      const extra = (st && st.args === undefined) ? [] : (Array.isArray(st.args) ? st.args : [st.args]);
      cur = exec(String(name), [cur, ...extra]);
    }
    return cur;
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.exec.map', arity: 2,
  sig: '(string opName, any[] xs) -> any[]',
  doc: '把单参指令映射到数组每个元素',
  ref: 'PyTorch FX 图变换中的逐节点映射',
  inv: '输出长度等于 xs 长度；第 i 项 = exec(opName, [xs[i]])',
  impl: (opName, xs) => {
    const list = Array.isArray(xs) ? xs : [];
    return list.map(v => exec(String(opName), [v]));
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.exec.timed', arity: 2,
  sig: '(string name, any[] args) -> object',
  doc: '带耗时返回：{value, ms}',
  ref: 'Prometheus 直方图观测语义',
  inv: 'ms ≥ 0；value 等于 exec(name, args)',
  impl: (name, args) => {
    const list = Array.isArray(args) ? args : (args === undefined ? [] : [args]);
    const t0 = _now();
    const v = exec(String(name), list);
    const ms = _now() - t0;
    _latAdd(String(name), ms);
    return { value: v, ms };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.exec.retry', arity: 3,
  sig: '(string opName, any[] args, object cfg) -> object',
  doc: '失败重试：只对可重试错误码（E_NUMERICAL / E_LIMIT）重试；cfg={retries, backoffMs}',
  ref: 'Decandia et al. 2007 Dynamo 重试与退避策略',
  inv: '不可重试错误码时 attempts=1；全部失败后 ok=false 且 attempts=retries+1',
  impl: (opName, args, cfg) => {
    const c = (cfg === undefined || cfg === null) ? {} : cfg;
    const retries = Math.max(0, Math.floor(c.retries === undefined ? 2 : Number(c.retries)));
    const backoff = Math.max(0, Number(c.backoffMs === undefined ? 0 : c.backoffMs));
    const list = Array.isArray(args) ? args : (args === undefined ? [] : [args]);
    let attempts = 0;
    let lastCode = null, lastMsg = null;
    for (let i = 0; i <= retries; i++) {
      attempts = i + 1;
      const r = tryExec(String(opName), list);
      if (r.ok) return { ok: true, value: r.value, attempts, error: null };
      lastCode = _errCodeOf(r.error);
      lastMsg = r.error.message;
      if (!ERR_RETRYABLE[lastCode]) break;
      if (backoff > 0 && i < retries) { const t0 = _now(); while (_now() - t0 < backoff) { /* 忙等退避 */ } }
    }
    return { ok: false, value: null, attempts, error: { code: lastCode, message: lastMsg } };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.exec.memo', arity: 2,
  sig: '(string opName, any[] args) -> object',
  doc: '记忆化：以 name + 稳定序列化(args) 为键缓存，返回 {value, hit}',
  ref: 'LLVM Pass Manager 结果缓存 / Bellman 1957 动态规划备忘',
  inv: '首次 hit=false；同参再次调用 hit=true 且 value 与首次稳定序列化相同',
  impl: (opName, args) => {
    const list = Array.isArray(args) ? args : (args === undefined ? [] : [args]);
    const key = `${String(opName)}#${_stable(list)}`;
    if (_memo.has(key)) return { value: _memo.get(key), hit: true };
    const v = exec(String(opName), list);
    _memo.set(key, v);
    return { value: v, hit: false };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.exec.assertOp', arity: 3,
  sig: '(string name, any[] args, function|string predicate) -> boolean',
  doc: '执行后断言：predicate 为函数（接收 value）或已注册指令名（以 [value] 调用）',
  ref: 'Hypothesis 属性测试框架（不变量断言）',
  inv: '返回布尔；predicate 抛错时按 E_VALUE 归一化后向上抛出',
  impl: (name, args, predicate) => {
    const list = Array.isArray(args) ? args : (args === undefined ? [] : [args]);
    const v = exec(String(name), list);
    let r;
    if (typeof predicate === 'function') r = predicate(v);
    else if (typeof predicate === 'string') r = exec(predicate, [v]);
    else throw new ForgeError('E_TYPE', 'predicate 需为函数或已注册指令名');
    return Boolean(r);
  },
});

// =====================================================================
// 四、验证 sys.verify.*（12 条）—— 全部以 ops 数组为第一入参
// =====================================================================

defineOp({
  id: nid(), domain: D, name: 'sys.verify.catalog', arity: 2,
  sig: '(op[] ops, object opt) -> object',
  doc: '目录级综合校验：名字唯一 / id 连续 / arity 一致 / doc 与 ref 齐备 / 域计数',
  ref: 'ONNX Operator Registry 一致性检查（checker）',
  inv: 'ok = (issues.length === 0)；total = ops.length；checks 覆盖 5 个子项',
  impl: (ops, opt) => {
    const list = Array.isArray(ops) ? ops : [];
    const o = (opt === undefined || opt === null) ? {} : opt;
    const minLen = o.minLen === undefined ? 4 : Number(o.minLen);
    const issues = [];
    const seen = new Map();
    for (const op of list) {
      const nm = op && op.name;
      if (seen.has(nm)) issues.push({ kind: 'dupName', name: nm }); else seen.set(nm, op);
      if (!op || !op.doc || String(op.doc).length < minLen) issues.push({ kind: 'doc', name: nm });
      if (!op || !op.ref || String(op.ref).length < minLen) issues.push({ kind: 'ref', name: nm });
      if (op && op.arity != null && typeof op.impl === 'function' && op.arity !== op.impl.length) {
        issues.push({ kind: 'arity', name: nm, arity: op.arity, impl: op.impl.length });
      }
    }
    const byDomain = {};
    for (const op of list) { const d = op && op.domain; byDomain[d] = (byDomain[d] || 0) + 1; }
    const ids = list.map(op => op && String(op.id || '')).sort();
    for (let i = 0; i < ids.length; i++) {
      if (i > 0 && ids[i] === ids[i - 1]) issues.push({ kind: 'dupId', id: ids[i] });
    }
    return { ok: issues.length === 0, total: list.length, byDomain, issues };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.verify.invariants', arity: 1,
  sig: '(op[] ops) -> object',
  doc: '检查每条指令都声明了非空不变量（长度 ≥ minLen），并统计未声明项',
  ref: 'Hypothesis 属性测试框架（不变量声明）',
  inv: 'checked = ops.length；ok 为真当且仅当每条 op.inv 非空且长度 ≥ 5',
  impl: (ops) => {
    const list = Array.isArray(ops) ? ops : [];
    const failures = [];
    for (const op of list) {
      const inv = op && op.inv;
      if (!inv || String(inv).length < 5) failures.push({ name: op && op.name, id: op && op.id });
    }
    return { checked: list.length, ok: failures.length === 0, failures };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.verify.determinism', arity: 4,
  sig: '(op[] ops, string opName, any[] args, number n) -> boolean',
  doc: '确定性检验：同参重复调用 n 次，稳定序列化逐位相同则为 true',
  ref: 'IEEE 754-2019 确定性运算 / Hypothesis 可复现性要求',
  inv: '对纯函数指令返回 true；任两次结果序列化不同即返回 false',
  impl: (ops, opName, args, n) => {
    _findOp(ops, opName);
    const list = Array.isArray(args) ? args : (args === undefined ? [] : [args]);
    const count = Math.max(1, Math.floor(Number(n === undefined ? 3 : n)));
    let base = null;
    for (let i = 0; i < count; i++) {
      const r = tryExec(String(opName), list);
      if (!r.ok) throw r.error;
      const s = _stable(r.value);
      if (base === null) base = s;
      else if (s !== base) return false;
    }
    return true;
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.verify.fuzz', arity: 2,
  sig: '(op[] ops, string opName, object cfg={}) -> object',
  doc: '模糊测试：按签名推导类型生成随机入参反复调用，捕获但不抛出异常',
  ref: 'Hypothesis 属性测试框架（MacIver & Donaldson et al.）',
  inv: '返回 {trials, failures}；同 seed 下 trials 序列可复现；健壮指令 failures 为空',
  impl: (ops, opName, cfg = undefined) => {
    const op = _findOp(ops, opName);
    const c = (cfg === undefined || cfg === null) ? {} : cfg;
    const trials = Math.max(1, Math.floor(Number(c.trials === undefined ? 20 : c.trials)));
    const seed = Number(c.seed === undefined ? 20261014 : c.seed);
    const rng = createRng(seed);
    const failures = [];
    for (let i = 0; i < trials; i++) {
      const args = _genArgs(op, rng);
      const r = tryExec(String(opName), args);
      if (!r.ok) failures.push({ trial: i, args: _plain(args), code: _errCodeOf(r.error), message: r.error.message });
    }
    return { op: String(opName), trials, failures, seed };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.verify.smoke', arity: 2,
  sig: '(op[] ops, object argsTable) -> object',
  doc: '冒烟：按 argsTable[name] 给出的入参逐条 tryExec，统计失败项',
  ref: 'LLVM lit 冒烟测试框架',
  inv: 'total = ops.length；ok 为真当且仅当 failures 为空；永不抛出',
  impl: (ops, argsTable) => {
    const list = Array.isArray(ops) ? ops : [];
    const tbl = (argsTable && typeof argsTable === 'object') ? argsTable : {};
    const failures = [];
    let okCount = 0;
    for (const op of list) {
      const nm = op && op.name;
      const args = Array.isArray(tbl[nm]) ? tbl[nm] : (tbl[nm] === undefined ? [] : [tbl[nm]]);
      const r = tryExec(String(nm), args);
      if (r.ok) okCount++;
      else failures.push({ name: nm, code: _errCodeOf(r.error), message: r.error.message });
    }
    return { total: list.length, ok: okCount, failures };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.verify.docCoverage', arity: 2,
  sig: '(op[] ops, number minLen) -> object',
  doc: '文档覆盖率：doc 非空且长度 ≥ minLen 视为已覆盖',
  ref: 'ONNX Operator Registry 文档完备性要求',
  inv: 'ratio = covered/total ∈ [0,1]；全覆盖时 missing 为空且 ok 为真',
  impl: (ops, minLen) => {
    const list = Array.isArray(ops) ? ops : [];
    const m = Math.max(0, Math.floor(Number(minLen === undefined ? 4 : minLen)));
    const missing = [];
    let covered = 0;
    for (const op of list) {
      const d = op && op.doc;
      if (d && String(d).length >= m) covered++;
      else missing.push(op && (op.name ?? op.id));
    }
    return { total: list.length, covered, ratio: list.length ? covered / list.length : 1, missing, ok: missing.length === 0 };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.verify.refCoverage', arity: 2,
  sig: '(op[] ops, number minLen) -> object',
  doc: '参考出处覆盖率：ref 非空且长度 ≥ minLen 视为已覆盖',
  ref: 'ONNX Operator Registry 出处可追溯性要求',
  inv: 'ratio = covered/total ∈ [0,1]；全覆盖时 missing 为空且 ok 为真',
  impl: (ops, minLen) => {
    const list = Array.isArray(ops) ? ops : [];
    const m = Math.max(0, Math.floor(Number(minLen === undefined ? 4 : minLen)));
    const missing = [];
    let covered = 0;
    for (const op of list) {
      const d = op && op.ref;
      if (d && String(d).length >= m) covered++;
      else missing.push(op && (op.name ?? op.id));
    }
    return { total: list.length, covered, ratio: list.length ? covered / list.length : 1, missing, ok: missing.length === 0 };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.verify.idContinuity', arity: 1,
  sig: '(op[] ops) -> object',
  doc: 'id 连续性：每域 id 形如 D<d>-NNN，且序号 001..N 无缺号无重复',
  ref: 'ONNX Operator Registry 稳定标识约定',
  inv: 'ok 为真当且仅当每域序号集合恰为 {1..count} 且前缀与 domain 一致',
  impl: (ops) => {
    const list = Array.isArray(ops) ? ops : [];
    const groups = new Map();
    const issues = [];
    for (const op of list) {
      const id = op && String(op.id || '');
      const m = /^D(\d+)-(\d+)$/.exec(id);
      if (!m) { issues.push({ kind: 'format', id }); continue; }
      const d = Number(m[1]);
      if (op.domain != null && Number(op.domain) !== d) issues.push({ kind: 'prefixMismatch', id, domain: op.domain });
      if (!groups.has(d)) groups.set(d, []);
      groups.get(d).push(Number(m[2]));
    }
    const byDomain = {};
    for (const [d, nums] of groups) {
      const s = nums.slice().sort((a, b) => a - b);
      const dup = s.filter((v, i) => i > 0 && v === s[i - 1]);
      if (dup.length) issues.push({ kind: 'dup', domain: d, seq: dup });
      for (let i = 0; i < s.length; i++) {
        if (s[i] !== i + 1) { issues.push({ kind: 'gap', domain: d, expected: i + 1, got: s[i] }); break; }
      }
      byDomain[d] = s.length;
    }
    return { ok: issues.length === 0, count: list.length, byDomain, issues };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.verify.nameUnique', arity: 1,
  sig: '(op[] ops) -> object',
  doc: '指令名唯一性检查',
  ref: 'ONNX Operator Registry 命名唯一性约束',
  inv: 'ok 为真当且仅当 name 集合无重复；dupes 列出全部重名',
  impl: (ops) => {
    const list = Array.isArray(ops) ? ops : [];
    const seen = new Map();
    const dupes = [];
    for (const op of list) {
      const nm = op && op.name;
      if (seen.has(nm)) dupes.push(nm);
      else seen.set(nm, true);
    }
    return { ok: dupes.length === 0, total: list.length, unique: seen.size, dupes: [...new Set(dupes)] };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.verify.arityCheck', arity: 1,
  sig: '(op[] ops) -> object',
  doc: 'arity 一致性：注册的 arity 必须等于由 impl 形参表推导出的「必需参数个数」',
  ref: 'ONNX Operator Schema 元数校验；arity 由实现反推以避免元数据漂移',
  inv: 'ok 为真当且仅当每条指令的 arity = requiredParams(impl)，且 0 ≤ arity ≤ impl.length',
  impl: (ops) => {
    const list = Array.isArray(ops) ? ops : [];
    const issues = [];
    let checked = 0, unchecked = 0;
    for (const op of list) {
      const fn = op && typeof op.impl === 'function' ? op.impl : null;
      if (!fn) { unchecked++; continue; }
      const req = requiredParams(fn);
      const n = fn.length;
      if (op.arity == null) { unchecked++; continue; }
      checked++;
      if (Number(op.arity) !== req) issues.push({ name: op.name, arity: op.arity, required: req, impl: n });
      else if (req > n) issues.push({ name: op.name, arity: op.arity, required: req, impl: n });
    }
    return { ok: issues.length === 0, checked, unchecked, issues };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.verify.domainCounts', arity: 2,
  sig: '(op[] ops, number|object expected) -> object',
  doc: '域计数校验：expected 为数字则每域都须等于它；为对象则逐域比对',
  ref: 'forge-1024 目录规约：每域恰好 102 条',
  inv: 'ok 为真当且仅当每个出现过的域条数都符合 expected',
  impl: (ops, expected) => {
    const list = Array.isArray(ops) ? ops : [];
    const exp = (expected === undefined || expected === null) ? 102 : expected;
    const counts = {};
    for (const op of list) { const d = op && op.domain; counts[d] = (counts[d] || 0) + 1; }
    const issues = [];
    for (const d of Object.keys(counts)) {
      const want = (typeof exp === 'number') ? exp : (exp[d] ?? exp[String(d)]);
      if (want == null) continue;
      if (counts[d] !== Number(want)) issues.push({ domain: Number(d), got: counts[d], want: Number(want) });
    }
    return { ok: issues.length === 0, counts, issues };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.verify.errorTaxonomy', arity: 1,
  sig: '(op[] ops) -> object',
  doc: '错误分类学自洽性：每个 CODES 都有类别与建议；分类映射无未知码',
  ref: 'forge-1024 错误分类学（Rust error kinds 风格）',
  inv: 'ok 为真当且仅当 codes 全覆盖且 classify 对每个码返回非 unknown 类别',
  impl: (ops) => {
    const list = Array.isArray(ops) ? ops : [];
    const codes = Object.keys(CODES);
    const unknown = [];
    for (const c of codes) {
      if (!ERR_CATEGORY[c] || !ERR_SUGGEST[c]) unknown.push(c);
    }
    return {
      ok: unknown.length === 0, count: codes.length, codes,
      categories: { ...ERR_CATEGORY }, retryable: { ...ERR_RETRYABLE },
      unknown, scanned: list.length,
    };
  },
});

// =====================================================================
// 五、文档生成 sys.doc.*（9 条）—— 全部返回字符串，不写文件
// =====================================================================

defineOp({
  id: nid(), domain: D, name: 'sys.doc.opMarkdown', arity: 2,
  sig: '(op[] ops, string nameOrId) -> string',
  doc: '单条指令的 Markdown 卡片',
  ref: 'ONNX Operator Registry 文档格式',
  inv: '输出以 "### " 开头，并包含该指令的 name / sig / doc / ref / inv',
  impl: (ops, nameOrId) => {
    const op = _findOp(ops, nameOrId);
    return [
      `### \`${op.name}\` · \`${op.id}\``,
      '',
      `- 域：D${op.domain}${DOMAINS[op.domain] ? ` (${DOMAINS[op.domain].code} ${DOMAINS[op.domain].name})` : ''}`,
      `- 元数：${op.arity}`,
      `- 签名：\`${op.sig}\``,
      `- 说明：${op.doc}`,
      `- 参考：${op.ref}`,
      `- 不变量：${op.inv}`,
      '',
    ].join('\n');
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.doc.domainMarkdown', arity: 2,
  sig: '(op[] ops, number domain) -> string',
  doc: '单个域的 Markdown 指令表',
  ref: 'ONNX Operator Registry 域文档',
  inv: '表格行数 = 该域条数 + 2（表头与分隔行）；标题含域编号',
  impl: (ops, domain) => {
    const d = Number(domain);
    const rows = (Array.isArray(ops) ? ops : []).filter(o => o && Number(o.domain) === d);
    const meta = DOMAINS[d];
    const head = `## 域${d} ${meta ? `${meta.code} ${meta.name}` : ''}${meta ? ` — ${meta.desc}` : ''}`;
    const out = [head, '', `共 ${rows.length} 条。`, '', '| id | name | 签名 | 说明 |', '| --- | --- | --- | --- |'];
    for (const o of rows) {
      out.push(`| \`${o.id}\` | \`${o.name}\` | \`${_csvCell(o.sig)}\` | ${_csvCell(o.doc)} |`);
    }
    out.push('');
    return out.join('\n');
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.doc.catalogMarkdown', arity: 2,
  sig: '(op[] ops, string title) -> string',
  doc: '全目录 Markdown（按域分组 + 计数）',
  ref: 'ONNX Operator Registry 全量文档',
  inv: '包含全部传入指令的 name；域标题数量等于出现的域个数',
  impl: (ops, title) => {
    const list = Array.isArray(ops) ? ops : [];
    const t = (title === undefined || title === null) ? 'forge-1024 指令目录' : String(title);
    const groups = new Map();
    for (const o of list) {
      const d = o && o.domain;
      if (!groups.has(d)) groups.set(d, []);
      groups.get(d).push(o);
    }
    const ds = [...groups.keys()].sort((a, b) => a - b);
    const out = [`# ${t}`, '', `内核 ${KERNEL_VERSION} · 目录 ${CATALOG_VERSION} · 共 ${list.length} 条`, ''];
    for (const d of ds) {
      const meta = DOMAINS[d];
      const rows = groups.get(d);
      out.push(`## 域${d} ${meta ? `${meta.code} ${meta.name}` : ''}（${rows.length} 条）`, '');
      for (const o of rows) out.push(`- \`${o.id}\` **${o.name}** \`${o.sig}\` — ${o.doc}`);
      out.push('');
    }
    return out.join('\n');
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.doc.jsonExport', arity: 2,
  sig: '(op[] ops, boolean pretty) -> string',
  doc: '目录 JSON 导出（可被 JSON.parse 还原）',
  ref: 'RFC 8259 JSON 交换格式',
  inv: 'JSON.parse(输出).ops.length = ops.length；pretty 为真时带 2 空格缩进',
  impl: (ops, pretty) => {
    const list = Array.isArray(ops) ? ops : [];
    const obj = {
      kernel: KERNEL_VERSION, catalog: CATALOG_VERSION, count: list.length,
      ops: _plain(_sums(list)),
    };
    return JSON.stringify(obj, null, (pretty === undefined ? true : Boolean(pretty)) ? 2 : 0);
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.doc.csvExport', arity: 2,
  sig: '(op[] ops, string delimiter) -> string',
  doc: '目录 CSV 导出（RFC 4180 引号转义）',
  ref: 'RFC 4180 CSV 格式',
  inv: '行数 = ops.length + 1；列数恒为 7；字段含分隔符时被双引号包裹',
  impl: (ops, delimiter) => {
    const list = Array.isArray(ops) ? ops : [];
    const sep = (delimiter === undefined || delimiter === null) ? ',' : String(delimiter);
    const lines = [['id', 'domain', 'name', 'arity', 'sig', 'doc', 'ref'].map(_csvCell).join(sep)];
    for (const o of _sums(list)) {
      lines.push([o.id, o.domain, o.name, o.arity, o.sig, o.doc, o.ref].map(_csvCell).join(sep));
    }
    return lines.join('\n');
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.doc.htmlExport', arity: 2,
  sig: '(op[] ops, string title) -> string',
  doc: '目录 HTML 导出（单文件，含内联样式）',
  ref: 'RFC 1866 HTML 2.0 / HTML5 表格元素',
  inv: '输出以 <html 开头且以 </html> 收尾；每行指令出现在 <tr> 中',
  impl: (ops, title) => {
    const list = Array.isArray(ops) ? ops : [];
    const t = (title === undefined || title === null) ? 'forge-1024 指令目录' : String(title);
    const out = [];
    out.push('<!doctype html>');
    out.push('<html lang="zh-CN">');
    out.push('<head><meta charset="utf-8">');
    out.push(`<title>${_esc(t)}</title>`);
    out.push('<style>body{font-family:system-ui,sans-serif;margin:2rem}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}th{background:#f3f4f6}</style>');
    out.push('</head><body>');
    out.push(`<h1>${_esc(t)}</h1>`);
    out.push(`<p>内核 ${_esc(KERNEL_VERSION)} · 目录 ${_esc(CATALOG_VERSION)} · 共 ${list.length} 条</p>`);
    out.push('<table><thead><tr><th>id</th><th>domain</th><th>name</th><th>元数</th><th>签名</th><th>说明</th></tr></thead><tbody>');
    for (const o of _sums(list)) {
      out.push(`<tr><td>${_esc(o.id)}</td><td>${_esc(o.domain)}</td><td><code>${_esc(o.name)}</code></td><td>${_esc(o.arity)}</td><td><code>${_esc(o.sig)}</code></td><td>${_esc(o.doc)}</td></tr>`);
    }
    out.push('</tbody></table></body></html>');
    return out.join('\n');
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.doc.mermaid', arity: 2,
  sig: '(op[] ops, string kind) -> string',
  doc: '域依赖图（Mermaid）：节点为域，边表示分层依赖 D1→D2→…→D10',
  ref: 'Mermaid 流程图语法 / LLVM Pass Manager 依赖图',
  inv: '输出以 "graph" 开头；每个出现的域各有一个节点声明',
  impl: (ops, kind) => {
    const list = Array.isArray(ops) ? ops : [];
    const dir = (kind === undefined || kind === null || kind === 'graph') ? 'TD' : String(kind);
    const ds = [...new Set(list.map(o => o && o.domain).filter(d => d != null))].sort((a, b) => a - b);
    const out = [`graph ${dir}`];
    for (const d of ds) {
      const meta = DOMAINS[d];
      const n = list.filter(o => Number(o.domain) === Number(d)).length;
      out.push(`  D${d}["D${d} ${meta ? meta.code : ''} ${meta ? meta.name : ''} (${n})"]`);
    }
    for (let i = 0; i + 1 < ds.length; i++) out.push(`  D${ds[i]} --> D${ds[i + 1]}`);
    out.push('');
    return out.join('\n');
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.doc.sigTable', arity: 2,
  sig: '(op[] ops, number? domain) -> string',
  doc: '签名表（Markdown）：name | 签名 | 元数',
  ref: 'ONNX Operator Schema 签名清单',
  inv: '行数 = 指令数 + 2；domain 给定时仅列该域',
  impl: (ops, domain) => {
    let list = Array.isArray(ops) ? ops : [];
    if (domain !== undefined && domain !== null) list = list.filter(o => o && Number(o.domain) === Number(domain));
    const out = ['| name | 签名 | 元数 |', '| --- | --- | --- |'];
    for (const o of list) out.push(`| \`${o.name}\` | \`${o.sig}\` | ${o.arity} |`);
    out.push('');
    return out.join('\n');
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.doc.readmeSection', arity: 2,
  sig: '(op[] ops, number domain) -> string',
  doc: '为 README 生成单域章节（简介 + 分组 + 签名表）',
  ref: 'ONNX Operator Registry README 章节格式',
  inv: '以 "## " 开头；含域描述与全部该域指令名',
  impl: (ops, domain) => {
    const d = Number(domain);
    const meta = DOMAINS[d];
    const rows = (Array.isArray(ops) ? ops : []).filter(o => o && Number(o.domain) === d);
    const prefix = [...new Set(rows.map(o => String(o.name || '').split('.').slice(0, 2).join('.')))].sort();
    const out = [
      `## 域${d} ${meta ? `${meta.code} ${meta.name}` : ''}`,
      '',
      meta ? meta.desc : '',
      '',
      `- 指令数：**${rows.length}**`,
      `- 分组：${prefix.map(p => `\`${p}.*\``).join('、')}`,
      '',
      '| name | 签名 |',
      '| --- | --- |',
    ];
    for (const o of rows) out.push(`| \`${o.name}\` | \`${o.sig}\` |`);
    out.push('');
    return out.join('\n');
  },
});

// =====================================================================
// 六、基准 sys.bench.*（8 条）
// =====================================================================

defineOp({
  id: nid(), domain: D, name: 'sys.bench.timeOp', arity: 4,
  sig: '(op[] ops, string name, any[] args, object cfg) -> object',
  doc: '微基准：warmup 预热后计时 n 次，返回 {ms, opsPerSec, n}',
  ref: 'LLVM benchmarking / Google Benchmark 计时模型',
  inv: 'n>0 时 ms>0；opsPerSec = n / (ms/1000) 有限；失败次数记入 failures',
  impl: (ops, name, args, cfg) => {
    _findOp(ops, name);
    const c = (cfg === undefined || cfg === null) ? {} : cfg;
    const warm = Math.max(0, Math.floor(Number(c.warmup === undefined ? 1 : c.warmup)));
    const n = Math.max(1, Math.floor(Number(c.n === undefined ? 10 : c.n)));
    const list = Array.isArray(args) ? args : (args === undefined ? [] : [args]);
    let failures = 0;
    for (let i = 0; i < warm; i++) { if (!tryExec(String(name), list).ok) failures++; }
    const t0 = _now();
    for (let i = 0; i < n; i++) { if (!tryExec(String(name), list).ok) failures++; }
    const ms = Math.max(_now() - t0, Number.EPSILON);
    _latAdd(String(name), ms);
    return {
      op: String(name), n, warmup: warm, failures,
      ms: +ms.toFixed(6), perCallMs: +(ms / n).toFixed(9),
      opsPerSec: n / (ms / 1000),
    };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.bench.compare', arity: 4,
  sig: '(op[] ops, string[] names, any[] args, object cfg) -> object[]',
  doc: '多指令横向计时对比，返回按 ms 升序的行',
  ref: 'Google Benchmark 对比报告',
  inv: '行数 = names.length；按 ms 升序；每行含 op/ms/opsPerSec',
  impl: (ops, names, args, cfg) => {
    const list = Array.isArray(names) ? names : [names];
    const rows = list.map(nm => {
      const r = exec('sys.bench.timeOp', [ops, nm, args, cfg]);
      return { op: nm, ms: r.ms, opsPerSec: r.opsPerSec, n: r.n, failures: r.failures };
    });
    return rows.sort((a, b) => a.ms - b.ms);
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.bench.throughput', arity: 4,
  sig: '(op[] ops, string name, any[] args, object cfg) -> object',
  doc: '吞吐量：在 budgetMs 时间预算内尽可能多跑，返回 {opsPerSec, n, ms}',
  ref: 'Prometheus 吞吐率指标 / Google Benchmark 时间预算模式',
  inv: 'n ≥ 1；ms ≥ budgetMs（跑满预算后收尾）；opsPerSec = n/(ms/1000)',
  impl: (ops, name, args, cfg) => {
    _findOp(ops, name);
    const c = (cfg === undefined || cfg === null) ? {} : cfg;
    const budget = Math.max(0, Number(c.budgetMs === undefined ? 20 : c.budgetMs));
    const cap = Math.max(1, Math.floor(Number(c.maxN === undefined ? 100000 : c.maxN)));
    const list = Array.isArray(args) ? args : (args === undefined ? [] : [args]);
    const t0 = _now();
    let n = 0;
    for (let i = 0; i < cap; i++) {
      tryExec(String(name), list);
      n++;
      if (_now() - t0 >= budget) break;
    }
    const ms = Math.max(_now() - t0, Number.EPSILON);
    return { op: String(name), n, ms: +ms.toFixed(6), budgetMs: budget, opsPerSec: n / (ms / 1000) };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.bench.matmulFlops', arity: 4,
  sig: '(op[] ops, number m, number n, number k) -> object',
  doc: '矩阵乘浮点吞吐量：理论 flops = 2·m·n·k，并用朴素三重循环实测 GFLOPS',
  ref: 'BLAS gemm 浮点计数约定 / Golub & Van Loan《Matrix Computations》',
  inv: 'flops === 2*m*n*k；gflops = flops/(ms*1e9) 且有限非负',
  impl: (ops, m, n, k) => {
    const M = Math.floor(Number(m)), N = Math.floor(Number(n)), K = Math.floor(Number(k));
    if (!(M > 0 && N > 0 && K > 0)) throw new ForgeError('E_VALUE', 'm/n/k 必须为正整数');
    const flops = 2 * M * N * K;
    const rng = createRng(20261014);
    const A = new Float64Array(M * K);
    const B = new Float64Array(K * N);
    for (let i = 0; i < A.length; i++) A[i] = rng.normal();
    for (let i = 0; i < B.length; i++) B[i] = rng.normal();
    const C = new Float64Array(M * N);
    const t0 = _now();
    for (let i = 0; i < M; i++) {
      for (let p = 0; p < K; p++) {
        const a = A[i * K + p];
        for (let j = 0; j < N; j++) C[i * N + j] += a * B[p * N + j];
      }
    }
    const ms = Math.max(_now() - t0, Number.EPSILON);
    return {
      m: M, n: N, k: K, flops, ms: +ms.toFixed(6),
      gflops: flops / (ms * 1e9), check: C[0],
    };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.bench.memoryEstimate', arity: 3,
  sig: '(op[] ops, number[] shape, number dtypeBytes) -> object',
  doc: '张量内存估算：元素数 × 字节数（dtypeBytes 默认 8 = float64）',
  ref: 'ONNX Tensor 内存布局 / IEEE 754-2019 float64',
  inv: 'elements = ∏shape；bytes = elements·dtypeBytes；mb = bytes/2^20',
  impl: (ops, shape, dtypeBytes) => {
    const sh = Array.isArray(shape) ? shape.map(Number) : [Number(shape)];
    if (sh.some(v => !Number.isFinite(v) || v <= 0)) throw new ForgeError('E_VALUE', `shape 必须为正数数组: [${sh}]`);
    const bytes = Number(dtypeBytes === undefined ? 8 : dtypeBytes);
    if (!(bytes > 0)) throw new ForgeError('E_VALUE', 'dtypeBytes 必须 > 0');
    let elements = 1;
    for (const v of sh) elements *= v;
    const total = elements * bytes;
    return { shape: sh, dtypeBytes: bytes, elements, bytes: total, kb: total / 1024, mb: total / 1048576 };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.bench.reportMarkdown', arity: 2,
  sig: '(op[] ops, object[] rows) -> string',
  doc: '把基准行渲染成 Markdown 报告表',
  ref: 'Google Benchmark 报告格式',
  inv: '表格行数 = rows.length + 2；rows 为空时仍输出表头',
  impl: (ops, rows) => {
    const list = Array.isArray(rows) ? rows : [];
    const out = ['# 基准报告', '', `内核 ${KERNEL_VERSION} · 目录 ${CATALOG_VERSION} · 样本 ${list.length} 项`, ''];
    out.push('| 指令 | 次数 n | 总耗时 ms | 单次 ms | 吞吐 ops/s |', '| --- | --- | --- | --- | --- |');
    for (const r of list) {
      const per = (r && r.perCallMs != null) ? r.perCallMs : (r && r.ms != null && r.n ? r.ms / r.n : 0);
      out.push(`| \`${r && r.op}\` | ${r && r.n} | ${r && r.ms} | ${(+per).toFixed(6)} | ${Math.round(r ? r.opsPerSec : 0)} |`);
    }
    out.push('');
    return out.join('\n');
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.bench.scaling', arity: 4,
  sig: '(op[] ops, string name, number[] sizes, object cfg) -> object[]',
  doc: '规模扫描：对每个 size 计时一次；cfg.buildArgs(size) 可自定义入参构造（默认 [size]）',
  ref: 'Google Benchmark 复杂度扫描（Big-O）',
  inv: '行数 = sizes.length；每行含 size/ms/opsPerSec/ok；入参构造失败不会抛出',
  impl: (ops, name, sizes, cfg) => {
    _findOp(ops, name);
    const list = Array.isArray(sizes) ? sizes : [sizes];
    const c = (cfg === undefined || cfg === null) ? {} : cfg;
    const out = [];
    for (const s of list) {
      const args = typeof c.buildArgs === 'function' ? c.buildArgs(s) : [s];
      const arr = Array.isArray(args) ? args : [args];
      const t0 = _now();
      const r = tryExec(String(name), arr);
      const ms = Math.max(_now() - t0, Number.EPSILON);
      out.push({ size: s, ok: r.ok, ms: +ms.toFixed(6), opsPerSec: 1000 / ms, code: r.ok ? null : _errCodeOf(r.error) });
    }
    return out;
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.bench.suite', arity: 2,
  sig: '(op[] ops, object cfg) -> object',
  doc: '基准套件：对前 limit 条指令按签名生成入参并各计时一次，汇总为行',
  ref: 'LLVM test-suite / Google Benchmark 批量注册',
  inv: 'rows.length = min(limit, ops.length)；每行含 op/ms/ok；永不抛出',
  impl: (ops, cfg) => {
    const list = Array.isArray(ops) ? ops : [];
    const c = (cfg === undefined || cfg === null) ? {} : cfg;
    const limit = Math.max(0, Math.floor(Number(c.limit === undefined ? 20 : c.limit)));
    const trials = Math.max(1, Math.floor(Number(c.trials === undefined ? 3 : c.trials)));
    const warm = Math.max(0, Math.floor(Number(c.warmup === undefined ? 0 : c.warmup)));
    const seed = Number(c.seed === undefined ? 20261014 : c.seed);
    const rng = createRng(seed);
    const chosen = limit === 0 ? list : list.slice(0, limit);
    const rows = [];
    for (const op of chosen) {
      const args = _genArgs(op, rng);
      for (let i = 0; i < warm; i++) tryExec(String(op.name), args);
      const t0 = _now();
      let okCount = 0;
      for (let i = 0; i < trials; i++) if (tryExec(String(op.name), args).ok) okCount++;
      const ms = Math.max(_now() - t0, Number.EPSILON);
      rows.push({
        op: op.name, id: op.id, n: trials, ok: okCount, failures: trials - okCount,
        ms: +ms.toFixed(6), perCallMs: +(ms / trials).toFixed(9), opsPerSec: trials / (ms / 1000),
      });
    }
    return { count: rows.length, limit, rows };
  },
});

// =====================================================================
// 七、图编排 sys.graph.*（8 条）—— 对标 PyTorch FX IR / LLVM Pass Manager
// =====================================================================

defineOp({
  id: nid(), domain: D, name: 'sys.graph.build', arity: 1,
  sig: '(node[] nodes) -> object',
  doc: '由图节点 {id, op?, inputs?, value?, args?} 构建图；校验 id 唯一与上游引用存在',
  ref: 'PyTorch FX Graph IR / LLVM Pass Manager 依赖图',
  inv: 'ids 与 nodes 一一对应；重复 id 或悬空 inputs 抛 E_VALUE / E_NOTFOUND',
  impl: (nodes) => {
    const list = Array.isArray(nodes) ? nodes : [];
    const ids = list.map(n => n && n.id);
    const dup = ids.filter((v, i) => ids.indexOf(v) !== i);
    if (dup.length) throw new ForgeError('E_VALUE', `图节点 id 重复: ${[...new Set(dup)].join(', ')}`);
    for (const n of list) {
      if (!n || n.id === undefined || n.id === null) throw new ForgeError('E_VALUE', '图节点缺少 id');
      for (const src of (n.inputs || [])) {
        if (!ids.includes(src)) throw new ForgeError('E_NOTFOUND', `节点 ${n.id} 的上游 ${src} 不存在`);
      }
    }
    const byId = {};
    for (const n of list) byId[n.id] = { id: n.id, op: n.op ?? null, inputs: (n.inputs || []).slice(), ...(n.value !== undefined ? { value: n.value } : {}), ...(n.args !== undefined ? { args: n.args } : {}) };
    return { nodes: list.map(n => byId[n.id]), ids: ids.slice(), byId };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.graph.toposort', arity: 1,
  sig: '(graph) -> string[]',
  doc: 'Kahn 拓扑排序；存在环时抛 E_STATE',
  ref: 'Kahn 1962 拓扑排序算法',
  inv: '输出中每条边的起点位置严格早于终点；无环时长度等于节点数',
  impl: (graph) => {
    const nodes = _nodesOf(graph);
    const { order, ok } = _topo(nodes);
    if (!ok) throw new ForgeError('E_STATE', `图中存在环，无法拓扑排序（已排出 ${order.length}/${nodes.length}）`);
    return order;
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.graph.execute', arity: 2,
  sig: '(graph, object inputs) -> object',
  doc: '按拓扑序执行图：叶子取 inputs[id] 或 node.value，其余调用 exec(node.op, 上游值 + node.args)',
  ref: 'PyTorch FX Graph 解释执行 / LLVM Pass 调度',
  inv: 'values 覆盖全部节点；叶子节点值 = inputs[id] ?? node.value',
  impl: (graph, inputs) => {
    const nodes = _nodesOf(graph);
    const { order, ok } = _topo(nodes);
    if (!ok) throw new ForgeError('E_STATE', '图中存在环，无法执行');
    const byId = new Map(nodes.map(n => [n.id, n]));
    const src = (graph && !Array.isArray(graph) && graph.byId) ? graph.byId : null;
    const values = {};
    for (const id of order) {
      const n = byId.get(id) || (src && src[id]);
      if (!n) throw new ForgeError('E_STATE', `拓扑序中出现未知节点: ${id}`);
      if (n.op === null || n.op === undefined) {
        const ext = inputs && typeof inputs === 'object' ? inputs : {};
        if (Object.prototype.hasOwnProperty.call(ext, id)) values[id] = ext[id];
        else if (n.value !== undefined) values[id] = n.value;
        else throw new ForgeError('E_VALUE', `叶子节点 ${id} 既无 value 也无外部输入`);
      } else {
        const args = (n.inputs || []).map(i => {
          if (!Object.prototype.hasOwnProperty.call(values, i)) {
            throw new ForgeError('E_STATE', `节点 ${id} 的上游 ${i} 尚未求值`);
          }
          return values[i];
        });
        const extra = n.args === undefined ? [] : (Array.isArray(n.args) ? n.args : [n.args]);
        values[id] = exec(String(n.op), args.concat(extra));
      }
    }
    return { order, values };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.graph.validate', arity: 1,
  sig: '(graph) -> object',
  doc: '图合法性检查：id 唯一、上游引用存在、op 已注册、无环',
  ref: 'PyTorch FX Graph 校验 / LLVM Pass Manager verifier',
  inv: 'ok = issues.length === 0；hasCycle 为真时必有一条 cycle 类问题',
  impl: (graph) => {
    const nodes = _nodesOf(graph);
    const issues = [];
    const ids = nodes.map(n => n && n.id);
    const dup = [...new Set(ids.filter((v, i) => ids.indexOf(v) !== i))];
    for (const d of dup) issues.push({ kind: 'dupId', id: d });
    for (const n of nodes) {
      for (const s of (n.inputs || [])) if (!ids.includes(s)) issues.push({ kind: 'dangling', node: n.id, ref: s });
      if (n.op !== null && n.op !== undefined && !hasOp(String(n.op))) {
        issues.push({ kind: 'unknownOp', node: n.id, op: n.op });
      }
    }
    let hasCycle = false;
    try {
      const cyc = _findCycle(nodes);
      if (cyc) { hasCycle = true; issues.push({ kind: 'cycle', cycle: cyc }); }
    } catch (e) {
      issues.push({ kind: 'unresolved', message: e.message });
    }
    return { ok: issues.length === 0, nodes: nodes.length, edges: _edgesOf(nodes).length, hasCycle, issues };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.graph.toDot', arity: 2,
  sig: '(graph, string title) -> string',
  doc: '导出 Graphviz DOT 有向图',
  ref: 'Graphviz DOT 语言（Gansner & North）',
  inv: '以 digraph 开头，以 } 收尾；每个节点与每条边各占一行',
  impl: (graph, title) => {
    const nodes = _nodesOf(graph);
    const t = (title === undefined || title === null) ? 'forge-graph' : String(title);
    const out = [`digraph "${_esc(t)}" {`, '  rankdir=TB;'];
    for (const n of nodes) {
      const label = n.op ? `${n.id}\\n${n.op}` : `${n.id}\\n(const)`;
      out.push(`  "${_esc(n.id)}" [label="${label}"];`);
    }
    for (const [s, d] of _edgesOf(nodes)) out.push(`  "${_esc(s)}" -> "${_esc(d)}";`);
    out.push('}', '');
    return out.join('\n');
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.graph.cycleDetect', arity: 1,
  sig: '(graph) -> object',
  doc: 'DFS 三色标记检测环，返回 {hasCycle, cycle}（cycle 首尾同节点）',
  ref: 'Tarjan 1972 强连通分量 / Hopcroft & Ullman 1979 图算法',
  inv: 'hasCycle 为真当且仅当 cycle 非空且 cycle[0] === cycle[cycle.length-1]',
  impl: (graph) => {
    const nodes = _nodesOf(graph);
    const cyc = _findCycle(nodes);
    return { hasCycle: Boolean(cyc), cycle: cyc || [] };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.graph.nodeCount', arity: 1,
  sig: '(graph) -> number',
  doc: '图节点数',
  ref: 'Hopcroft & Ullman 1979 图算法基本量',
  inv: 'nodeCount = |V| ≥ 0',
  impl: (graph) => _nodesOf(graph).length,
});

defineOp({
  id: nid(), domain: D, name: 'sys.graph.edgeCount', arity: 1,
  sig: '(graph) -> number',
  doc: '图边数（每个 inputs 引用计一条边）',
  ref: 'Hopcroft & Ullman 1979 图算法基本量',
  inv: 'edgeCount = Σ|node.inputs| ≥ 0',
  impl: (graph) => _edgesOf(_nodesOf(graph)).length,
});

// =====================================================================
// 八、配置 sys.cfg.*（8 条）
// =====================================================================

defineOp({
  id: nid(), domain: D, name: 'sys.cfg.merge', arity: 3,
  sig: '(object base, object over, boolean deep) -> object',
  doc: '配置合并：deep 为真时递归合并普通对象，数组与标量整体覆盖',
  ref: 'Ansible / SaltStack 配置合并语义',
  inv: '深合并时 base 未被覆盖的键全部保留；over 的同名标量键胜出',
  impl: (base, over, deep) => {
    const a = _isPlainObj(base) ? base : {};
    const b = _isPlainObj(over) ? over : {};
    const useDeep = deep === undefined ? true : Boolean(deep);
    if (!useDeep) return { ...a, ...b };
    return _deepMerge(a, b);
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.cfg.validate', arity: 2,
  sig: '(object cfg, object schema) -> object',
  doc: '按 schema 校验：值为类型名（number/int/string/boolean/object/array/function/tensor/any）或 {type, required, min, max, enum}',
  ref: 'JSON Schema 草案 2020-12 校验语义',
  inv: 'ok = issues.length === 0；缺 required 键或类型不符都会产生一条 issue',
  impl: (cfg, schema) => {
    const c = _isPlainObj(cfg) ? cfg : {};
    const s = _isPlainObj(schema) ? schema : {};
    const issues = [];
    for (const k of Object.keys(s)) {
      const spec = s[k];
      const rule = typeof spec === 'string' ? { type: spec } : (spec || {});
      const has = Object.prototype.hasOwnProperty.call(c, k);
      const v = c[k];
      if (!has) {
        if (rule.required) issues.push({ key: k, kind: 'missing' });
        continue;
      }
      if (rule.type && !_typeOk(v, rule.type)) issues.push({ key: k, kind: 'type', want: rule.type, got: typeof v });
      if (typeof v === 'number') {
        if (rule.min != null && v < rule.min) issues.push({ key: k, kind: 'min', value: v, bound: rule.min });
        if (rule.max != null && v > rule.max) issues.push({ key: k, kind: 'max', value: v, bound: rule.max });
      }
      if (Array.isArray(rule.enum) && !rule.enum.some(e => _stable(e) === _stable(v))) {
        issues.push({ key: k, kind: 'enum', value: v });
      }
    }
    return { ok: issues.length === 0, checked: Object.keys(s).length, issues };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.cfg.freeze', arity: 1,
  sig: '(object cfg) -> object',
  doc: '深度冻结配置（Object.freeze 递归）；严格模式下写入将抛 TypeError',
  ref: 'ECMAScript Object.freeze / 不可变配置模式',
  inv: '返回对象及其所有子对象 Object.isFrozen 为真；非严格模式下写入静默失败',
  impl: (cfg) => _deepFreeze(_isPlainObj(cfg) || Array.isArray(cfg) ? cfg : {}),
});

defineOp({
  id: nid(), domain: D, name: 'sys.cfg.diff', arity: 2,
  sig: '(object a, object b) -> object',
  doc: '配置差异：以点路径给出 added / removed / changed 三类键',
  ref: 'Bentley & McIlroy 1993 差分算法思想 / 配置漂移检测',
  inv: 'a 与 b 完全相同时三类列表均为空；diff(a,b) 与 diff(b,a) 的 added/removed 互换',
  impl: (a, b) => {
    const acc = { added: [], removed: [], changed: [] };
    _walkDiff(_isPlainObj(a) ? a : {}, _isPlainObj(b) ? b : {}, '', acc);
    return acc;
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.cfg.hash', arity: 1,
  sig: '(object cfg) -> string',
  doc: '配置指纹：FNV-1a 64 位，基于稳定序列化（键序无关）',
  ref: 'FNV-1a（Fowler–Noll–Vo 哈希）',
  inv: '键顺序不同的等价配置哈希相同；任一标量改动即改变哈希',
  impl: (cfg) => _hex64(fnv1a64(_stable(cfg))),
});

defineOp({
  id: nid(), domain: D, name: 'sys.cfg.pick', arity: 2,
  sig: '(object cfg, string[] keys) -> object',
  doc: '按键白名单抽取子配置（缺失键跳过）',
  ref: ' lodash.pick 语义 / 配置最小权限原则',
  inv: '结果的键集合 ⊆ keys；每个存在的键值与原配置相同',
  impl: (cfg, keys) => {
    const c = _isPlainObj(cfg) ? cfg : {};
    const ks = Array.isArray(keys) ? keys : [keys];
    const out = {};
    for (const k of ks) if (Object.prototype.hasOwnProperty.call(c, k)) out[k] = c[k];
    return out;
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.cfg.defaults', arity: 2,
  sig: '(object cfg, object defaults) -> object',
  doc: '填充默认值：递归补全缺失键，已存在的键保持不变',
  ref: 'Ansible / SaltStack defaults 语义',
  inv: 'defaults 的每个键都出现在结果中；cfg 已有键的值不被覆盖',
  impl: (cfg, defaults) => {
    const c = _isPlainObj(cfg) ? cfg : {};
    const d = _isPlainObj(defaults) ? defaults : {};
    const out = { ...c };
    for (const k of Object.keys(d)) {
      if (!Object.prototype.hasOwnProperty.call(out, k) || out[k] === undefined) { out[k] = d[k]; continue; }
      if (_isPlainObj(out[k]) && _isPlainObj(d[k])) out[k] = exec('sys.cfg.defaults', [out[k], d[k]]);
    }
    return out;
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.cfg.toEnv', arity: 2,
  sig: '(object cfg, string prefix) -> object',
  doc: '配置转环境变量风格键：prefix + 大写下划线路径（嵌套用双下划线）',
  ref: '12-Factor App 配置约定（环境变量）',
  inv: '键数 = cfg 的叶子键数；键形如 PREFIX_KEY，全大写且仅含 [A-Z0-9_]',
  impl: (cfg, prefix) => {
    const c = _isPlainObj(cfg) ? cfg : {};
    const p = (prefix === undefined || prefix === null || prefix === '') ? '' : String(prefix);
    const out = {};
    const walk = (o, path) => {
      for (const k of Object.keys(o)) {
        const v = o[k];
        const seg = String(k).replace(/([a-z])([A-Z])/g, '$1_$2').toUpperCase();
        const next = path ? `${path}__${seg}` : seg;
        if (_isPlainObj(v)) walk(v, next);
        else out[p ? `${p.toUpperCase()}_${next}` : next] = String(v);
      }
    };
    walk(c, '');
    return out;
  },
});

// =====================================================================
// 九、错误 sys.err.*（7 条）
// =====================================================================

defineOp({
  id: nid(), domain: D, name: 'sys.err.classify', arity: 1,
  sig: '(string code) -> string',
  doc: '把错误码映射到类别：interface/type/shape/value/numerical/resource/lookup/state，未知为 unknown',
  ref: 'forge-1024 错误分类学 / Rust error kinds',
  inv: '对每个 CODES 键返回预定义类别且不等于 unknown；未知码返回 unknown',
  impl: (code) => ERR_CATEGORY[String(code)] || 'unknown',
});

defineOp({
  id: nid(), domain: D, name: 'sys.err.format', arity: 2,
  sig: '(string code, string message) -> string',
  doc: '格式化错误行：`[CODE] message`；未知码统一写作 [E_UNKNOWN]',
  ref: 'forge-1024 错误分类学（统一错误前缀）',
  inv: '输出以 [ 开头且包含 message；码不在 CODES 中时前缀为 [E_UNKNOWN]',
  impl: (code, message) => {
    const c = ERR_CATEGORY[String(code)] ? String(code) : 'E_UNKNOWN';
    return `[${c}] ${message === undefined || message === null ? '' : String(message)}`;
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.err.codes', arity: 0,
  sig: '() -> object[]',
  doc: '列出全部错误码及其描述、类别、可否重试',
  ref: 'forge-1024 错误分类学',
  inv: '条数 = |CODES|；每条含 code/desc/category/retryable 四个键',
  impl: () => Object.keys(CODES).map(c => ({
    code: c, desc: CODES[c], category: ERR_CATEGORY[c] || 'unknown',
    retryable: Boolean(ERR_RETRYABLE[c]),
  })),
});

defineOp({
  id: nid(), domain: D, name: 'sys.err.isRetryable', arity: 1,
  sig: '(string code) -> boolean',
  doc: '是否可重试：仅 E_NUMERICAL（数值失败）与 E_LIMIT（资源限额）可重试',
  ref: 'Decandia et al. 2007 Dynamo 可重试错误分类',
  inv: 'isRetryable("E_NUMERICAL") = true；isRetryable("E_SHAPE") = false',
  impl: (code) => Boolean(ERR_RETRYABLE[String(code)]),
});

defineOp({
  id: nid(), domain: D, name: 'sys.err.count', arity: 1,
  sig: '(string? opName) -> number',
  doc: '错误计数：不给 opName 时返回全部指令的错误总数',
  ref: 'Prometheus counter 语义',
  inv: 'count(opName) ≤ count()；均为非负整数',
  impl: (opName) => {
    const rows = _rowsByIdOrName(_mergedRows(), opName);
    let s = 0;
    for (const r of rows) s += r.errors;
    return s;
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.err.trace', arity: 2,
  sig: '(string opName, any[] args) -> object',
  doc: '追踪一次调用的错误：成功返回 {ok:true, code:null}，失败返回 {ok:false, code, message}',
  ref: 'OpenTelemetry 错误追踪语义',
  inv: '永不抛出；ok=false 时 code 必为字符串',
  impl: (opName, args) => {
    const list = Array.isArray(args) ? args : (args === undefined ? [] : [args]);
    const r = tryExec(String(opName), list);
    if (r.ok) return { ok: true, op: String(opName), code: null, message: null };
    return { ok: false, op: String(opName), code: _errCodeOf(r.error), message: r.error.message };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.err.suggest', arity: 1,
  sig: '(string code) -> string',
  doc: '给出该错误码的修复建议（未知码给通用建议）',
  ref: 'forge-1024 错误分类学诊断建议',
  inv: '对每个 CODES 键返回非空建议串；未知码返回兜底建议',
  impl: (code) => ERR_SUGGEST[String(code)] || '未知错误码：请检查调用约定或补充该码到分类学表',
});

// =====================================================================
// 十、环境 sys.env.*（7 条）
// =====================================================================

defineOp({
  id: nid(), domain: D, name: 'sys.env.info', arity: 0,
  sig: '() -> object',
  doc: '运行环境快照：平台、架构、Node 版本、V8 版本、内核版本',
  ref: 'Node.js process 对象（内置运行时内省）',
  inv: 'platform 与 arch 取自 process；versions.node 非空字符串',
  impl: () => ({
    platform: process.platform, arch: process.arch,
    node: (process.versions && process.versions.node) || 'unknown',
    v8: (process.versions && process.versions.v8) || 'unknown',
    versions: { ...(process.versions || {}) },
    kernel: KERNEL_VERSION, catalog: CATALOG_VERSION,
  }),
});

defineOp({
  id: nid(), domain: D, name: 'sys.env.nodeVersion', arity: 0,
  sig: '() -> string',
  doc: 'Node 版本字符串（semver）',
  ref: 'Node.js process.versions.node',
  inv: '形如 major.minor.patch；等于 sys.env.info().node',
  impl: () => (process.versions && process.versions.node) || 'unknown',
});

defineOp({
  id: nid(), domain: D, name: 'sys.env.memory', arity: 0,
  sig: '() -> object',
  doc: '进程内存用量（MB，三位小数）',
  ref: 'Node.js process.memoryUsage()',
  inv: 'heapUsed ≤ heapTotal；各项均为有限非负数',
  impl: () => {
    const m = process.memoryUsage();
    const mb = (v) => +(v / 1048576).toFixed(3);
    return {
      rss: mb(m.rss), heapTotal: mb(m.heapTotal), heapUsed: mb(m.heapUsed),
      external: mb(m.external), arrayBuffers: mb(m.arrayBuffers || 0),
    };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.env.nowMs', arity: 0,
  sig: '() -> number',
  doc: '单调递增时钟当前毫秒（hrtime 纳秒精度）',
  ref: 'Node.js process.hrtime.bigint() 单调时钟',
  inv: '连续两次调用后者 ≥ 前者；单位毫秒',
  impl: () => _now(),
});

defineOp({
  id: nid(), domain: D, name: 'sys.env.seed', arity: 1,
  sig: '(string str) -> number',
  doc: '由字符串派生 32 位种子（FNV-1a），便于用文字种子复现实验',
  ref: 'FNV-1a（Fowler–Noll–Vo 哈希）',
  inv: '同串同种子；结果 ∈ [0, 2^32)',
  impl: (str) => seedFrom(String(str === undefined ? '' : str)),
});

defineOp({
  id: nid(), domain: D, name: 'sys.env.platform', arity: 0,
  sig: '() -> string',
  doc: '平台标识，形如 "platform-arch"',
  ref: 'Node.js process.platform / process.arch',
  inv: '等于 `${process.platform}-${process.arch}`',
  impl: () => `${process.platform}-${process.arch}`,
});

defineOp({
  id: nid(), domain: D, name: 'sys.env.uptimeMs', arity: 0,
  sig: '() -> number',
  doc: '进程已运行毫秒数',
  ref: 'Node.js process.uptime()',
  inv: '非负且单调不减；等于 process.uptime()*1000',
  impl: () => process.uptime() * 1000,
});

// =====================================================================
// 十一、自检 sys.self.*（6 条）
// =====================================================================

defineOp({
  id: nid(), domain: D, name: 'sys.self.checkAll', arity: 2,
  sig: '(op[] ops, object opt) -> object',
  doc: '全量自检：名字唯一 / id 连续 / arity 一致 / doc·ref 覆盖 / 域计数 / 不变量声明',
  ref: 'ONNX checker / LLVM verifier',
  inv: 'ok = issues.length === 0；byDomain 给出每个出现域的条数',
  impl: (ops, opt) => {
    const list = Array.isArray(ops) ? ops : [];
    const o = (opt === undefined || opt === null) ? {} : opt;
    const expected = o.expected === undefined ? 102 : o.expected;
    const minLen = o.minLen === undefined ? 4 : Number(o.minLen);
    const checks = {
      nameUnique: exec('sys.verify.nameUnique', [list]),
      idContinuity: exec('sys.verify.idContinuity', [list]),
      arityCheck: exec('sys.verify.arityCheck', [list]),
      docCoverage: exec('sys.verify.docCoverage', [list, minLen]),
      refCoverage: exec('sys.verify.refCoverage', [list, minLen]),
      invariants: exec('sys.verify.invariants', [list]),
      domainCounts: exec('sys.verify.domainCounts', [list, expected]),
    };
    const issues = [];
    for (const [k, r] of Object.entries(checks)) {
      if (r && r.ok === false) issues.push({ check: k, detail: r.issues || r.failures || r.missing || r.dupes || [] });
    }
    const byDomain = {};
    for (const op of list) { const d = op && op.domain; byDomain[d] = (byDomain[d] || 0) + 1; }
    return { ok: issues.length === 0, total: list.length, byDomain, checks, issues };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.self.summary', arity: 2,
  sig: '(op[] ops, object opt) -> object',
  doc: '自检摘要：版本、总数、各域条数、目录哈希、健康与否',
  ref: 'Prometheus 健康摘要指标',
  inv: 'total = ops.length；hash 等于 sys.reg.catalogHash 对同目录的结果',
  impl: (ops, opt) => {
    const list = Array.isArray(ops) ? ops : [];
    const byDomain = {};
    for (const op of list) { const d = op && op.domain; byDomain[d] = (byDomain[d] || 0) + 1; }
    const chk = exec('sys.self.checkAll', [list, opt]);
    return {
      kernel: KERNEL_VERSION, catalog: CATALOG_VERSION,
      total: list.length, byDomain, ok: chk.ok, issues: chk.issues.length,
      hash: exec('sys.reg.catalogHash', [true]),
    };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.self.exitCode', arity: 2,
  sig: '(op[] ops, object opt) -> number',
  doc: '自检退出码：全部通过为 0，否则为 1（供 CI 使用）',
  ref: 'Unix 进程退出码约定 / LLVM lit 退出码',
  inv: '返回值 ∈ {0,1}；checkAll().ok 为真时必为 0',
  impl: (ops, opt) => (exec('sys.self.checkAll', [ops, opt]).ok ? 0 : 1),
});

defineOp({
  id: nid(), domain: D, name: 'sys.self.domainHealth', arity: 2,
  sig: '(op[] ops, number domain) -> object',
  doc: '单域健康度：条数、名字唯一、id 连续、doc·ref 覆盖情况',
  ref: 'ONNX 域级 checker',
  inv: 'count = 该域条数；ok 为真当且仅当 issues 为空',
  impl: (ops, domain) => {
    const d = Number(domain);
    const rows = (Array.isArray(ops) ? ops : []).filter(o => o && Number(o.domain) === d);
    const issues = [];
    const nu = exec('sys.verify.nameUnique', [rows]);
    if (!nu.ok) issues.push({ check: 'nameUnique', detail: nu.dupes });
    const ic = exec('sys.verify.idContinuity', [rows]);
    if (!ic.ok) issues.push({ check: 'idContinuity', detail: ic.issues });
    const dc = exec('sys.verify.docCoverage', [rows, 4]);
    if (!dc.ok) issues.push({ check: 'docCoverage', detail: dc.missing });
    const rc = exec('sys.verify.refCoverage', [rows, 4]);
    if (!rc.ok) issues.push({ check: 'refCoverage', detail: rc.missing });
    return {
      domain: d, code: DOMAINS[d] ? DOMAINS[d].code : null, name: DOMAINS[d] ? DOMAINS[d].name : null,
      count: rows.length, ok: issues.length === 0, issues,
    };
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.self.doctor', arity: 1,
  sig: '(op[] ops) -> string',
  doc: '诊断报告（Markdown）：逐域健康度 + 问题清单 + 修复建议',
  ref: 'LLVM verifier 诊断输出 / 12-Factor 健康检查',
  inv: '以 "# 诊断报告" 开头；每域一行；末尾给出总体结论',
  impl: (ops) => {
    const list = Array.isArray(ops) ? ops : [];
    const chk = exec('sys.self.checkAll', [list, undefined]);
    const ds = [...new Set(list.map(o => o && o.domain).filter(d => d != null))].sort((a, b) => a - b);
    const out = ['# 诊断报告', '', `内核 ${KERNEL_VERSION} · 目录 ${CATALOG_VERSION} · 共 ${list.length} 条`, '', '## 各域健康度', ''];
    out.push('| 域 | 名称 | 条数 | 状态 |', '| --- | --- | --- | --- |');
    for (const d of ds) {
      const h = exec('sys.self.domainHealth', [list, d]);
      out.push(`| D${d} | ${h.code} ${h.name} | ${h.count} | ${h.ok ? 'OK' : 'FAIL'} |`);
    }
    out.push('', '## 问题清单', '');
    if (chk.issues.length === 0) out.push('无。');
    else for (const it of chk.issues) out.push(`- \`${it.check}\`：${JSON.stringify(it.detail).slice(0, 300)}`);
    out.push('', '## 结论', '', chk.ok ? '全部检查通过。' : `存在 ${chk.issues.length} 类问题，请按上述清单修复。`);
    out.push('');
    return out.join('\n');
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.self.report', arity: 2,
  sig: '(op[] ops, string title) -> string',
  doc: '自检报告（Markdown）：统计表 + 遥测热点 + 目录指纹',
  ref: 'Prometheus 健康报告 / ONNX checker 报告',
  inv: '以 "# " 开头；包含条目总数与目录哈希',
  impl: (ops, title) => {
    const list = Array.isArray(ops) ? ops : [];
    const t = (title === undefined || title === null) ? 'forge-1024 自检报告' : String(title);
    const sm = exec('sys.self.summary', [list, undefined]);
    const out = [`# ${t}`, '', `内核 ${sm.kernel} · 目录 ${sm.catalog} · 共 ${sm.total} 条 · 指纹 \`${sm.hash}\``, ''];
    out.push('## 域分布', '');
    out.push('| 域 | 名称 | 条数 |', '| --- | --- | --- |');
    for (const d of Object.keys(sm.byDomain).sort((a, b) => a - b)) {
      const meta = DOMAINS[d];
      out.push(`| D${d} | ${meta ? `${meta.code} ${meta.name}` : '-'} | ${sm.byDomain[d]} |`);
    }
    out.push('', '## 遥测热点', '');
    const hot = exec('sys.tel.hotOps', [5]);
    if (hot.length === 0) out.push('暂无样本。');
    else {
      out.push('| 指令 | 调用 | 错误 | 累计 ms |', '| --- | --- | --- | --- |');
      for (const r of hot) out.push(`| \`${r.name}\` | ${r.calls} | ${r.errors} | ${r.ms} |`);
    }
    out.push('', '## 结论', '', sm.ok ? '自检通过。' : `自检未通过：${sm.issues} 类问题。`);
    out.push('');
    return out.join('\n');
  },
});

// =====================================================================
// 十二、通用工具 sys.util.*（8 条）
// =====================================================================

defineOp({
  id: nid(), domain: D, name: 'sys.util.range', arity: 3,
  sig: '(number a, number b?, number step?) -> number[]',
  doc: '等差数列；单参为 [0, a)，双参为 [a, b)，step 默认 1',
  ref: 'NumPy arange / Python range 语义',
  inv: '长度 = ceil((b-a)/step) 且 ≥ 0；step=1 时首项 a、末项 < b',
  impl: (a, b, step) => {
    const A = Number(a);
    const st = (step === undefined || step === null) ? 1 : Number(step);
    if (!Number.isFinite(A)) throw new ForgeError('E_VALUE', `range 起点非有限: ${a}`);
    if (st === 0) throw new ForgeError('E_VALUE', 'range 的 step 不能为 0');
    const B = (b === undefined || b === null) ? 0 : Number(b);
    const start = (b === undefined || b === null) ? 0 : A;
    const stop = (b === undefined || b === null) ? A : B;
    const out = [];
    if (st > 0) { for (let v = start; v < stop; v += st) out.push(v); }
    else { for (let v = start; v > stop; v += st) out.push(v); }
    return out;
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.util.zip', arity: 2,
  sig: '(any[] a, any[] b) -> any[][]',
  doc: '拉链：按较短数组长度配对成 [[a0,b0], ...]',
  ref: 'Python zip / Haskell zipWith 语义',
  inv: '长度 = min(a.length, b.length)；第 i 项为 [a[i], b[i]]',
  impl: (a, b) => {
    const A = Array.isArray(a) ? a : [a];
    const B = Array.isArray(b) ? b : [b];
    const n = Math.min(A.length, B.length);
    const out = [];
    for (let i = 0; i < n; i++) out.push([A[i], B[i]]);
    return out;
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.util.pick', arity: 2,
  sig: '(object obj, string path) -> any',
  doc: '按点路径取值，如 "a.b.0"；路径不存在返回 undefined',
  ref: 'RFC 6901 JSON Pointer（点路径简化版）',
  inv: 'pick(o, "k") = o.k；路径中任一段缺失返回 undefined 而不抛错',
  impl: (obj, path) => {
    const p = String(path === undefined ? '' : path);
    if (p === '') return obj;
    let cur = obj;
    for (const seg of p.split('.')) {
      if (cur === null || cur === undefined) return undefined;
      if (seg === '') continue;
      cur = cur[seg];
    }
    return cur;
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.util.deepEqual', arity: 2,
  sig: '(any a, any b) -> boolean',
  doc: '深度相等：自反、对称、支持嵌套对象/数组/Tensor；-0 与 0 视为相等，NaN 视为等于自身',
  ref: 'ECMAScript SameValueZero 语义 / Liskov 替换原则',
  inv: 'deepEqual(a,a) 恒真（自反）；deepEqual(a,b) = deepEqual(b,a)（对称）',
  impl: (a, b) => {
    if (a === b) return true;
    if (a === null || b === null || a === undefined || b === undefined) return false;
    if (typeof a !== typeof b) return false;
    if (a instanceof Tensor || b instanceof Tensor) {
      if (!(a instanceof Tensor) || !(b instanceof Tensor)) return false;
      if (a.shape.length !== b.shape.length) return false;
      for (let i = 0; i < a.shape.length; i++) if (a.shape[i] !== b.shape[i]) return false;
      if (a.data.length !== b.data.length) return false;
      for (let i = 0; i < a.data.length; i++) {
        const x = a.data[i], y = b.data[i];
        if (!(x === y || (Number.isNaN(x) && Number.isNaN(y)))) return false;
      }
      return true;
    }
    if (Array.isArray(a) || Array.isArray(b)) {
      if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
      for (let i = 0; i < a.length; i++) if (!exec('sys.util.deepEqual', [a[i], b[i]])) return false;
      return true;
    }
    if (typeof a === 'number') return a === b || (Number.isNaN(a) && Number.isNaN(b));
    if (typeof a !== 'object') return false;
    const ka = Object.keys(a), kb = Object.keys(b);
    if (ka.length !== kb.length) return false;
    for (const k of ka) {
      if (!Object.prototype.hasOwnProperty.call(b, k)) return false;
      if (!exec('sys.util.deepEqual', [a[k], b[k]])) return false;
    }
    return true;
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.util.typeOf', arity: 1,
  sig: '(any v) -> string',
  doc: '精确类型名：null/undefined/array/tensor/typedarray/number/string/boolean/function/object/bigint/symbol',
  ref: 'ECMAScript typeof 与 Object.prototype.toString 分类',
  inv: 'typeOf(null)="null"；typeOf([])="array"；typeOf(Tensor)="tensor"',
  impl: (v) => {
    if (v === null) return 'null';
    if (v === undefined) return 'undefined';
    if (Array.isArray(v)) return 'array';
    if (v instanceof Tensor) return 'tensor';
    if (ArrayBuffer.isView(v)) return 'typedarray';
    if (v instanceof Map) return 'map';
    if (v instanceof Set) return 'set';
    if (v instanceof Date) return 'date';
    const t = typeof v;
    if (t === 'object') return 'object';
    return t;
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.util.formatNum', arity: 2,
  sig: '(number v, number digits) -> string',
  doc: '定点格式化（digits 位小数）；非有限数输出字面量',
  ref: 'IEEE 754-2019 浮点字符串化 / ECMAScript Number.prototype.toFixed',
  inv: '有限数输出不含指数记号且小数位数恰为 digits；NaN 输出 "NaN"',
  impl: (v, digits) => {
    const x = Number(v);
    const d = Math.max(0, Math.min(20, Math.floor(Number(digits === undefined ? 2 : digits))));
    if (Number.isNaN(x)) return 'NaN';
    if (!Number.isFinite(x)) return x > 0 ? 'Infinity' : '-Infinity';
    return x.toFixed(d);
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.util.table', arity: 2,
  sig: '(any[][] rows, string[] headers) -> string',
  doc: '对齐文本表（列宽取该列最长单元）',
  ref: 'CLI 表格渲染惯例（GNU column 风格）',
  inv: '行数 = rows.length + (headers?1:0)；同列所有行宽度一致',
  impl: (rows, headers) => {
    const body = (Array.isArray(rows) ? rows : []).map(r => (Array.isArray(r) ? r : [r]).map(c => String(c == null ? '' : c)));
    const head = Array.isArray(headers) ? headers.map(h => String(h == null ? '' : h)) : null;
    const all = head ? [head, ...body] : body;
    const cols = all.reduce((m, r) => Math.max(m, r.length), 0);
    const w = new Array(cols).fill(0);
    for (const r of all) for (let i = 0; i < cols; i++) w[i] = Math.max(w[i], (r[i] ?? '').length);
    const line = (r) => r.map((c, i) => (c ?? '').padEnd(w[i])).join(' | ').replace(/\s+$/, '');
    const out = [];
    if (head) { out.push(line(head)); out.push(w.map(n => '-'.repeat(n)).join('-+-')); }
    for (const r of body) out.push(line(r));
    out.push('');
    return out.join('\n');
  },
});

defineOp({
  id: nid(), domain: D, name: 'sys.util.chunk', arity: 2,
  sig: '(any[] arr, number size) -> any[][]',
  doc: '按固定长度切分数组，末尾不足一组时保留为短块',
  ref: 'Rust slice::chunks / NumPy reshape 分块语义',
  inv: '块数 = ceil(len/size)；除末块外每块长度均为 size；拼接还原原数组',
  impl: (arr, size) => {
    const list = Array.isArray(arr) ? arr : [arr];
    const s = Math.floor(Number(size));
    if (!(s > 0)) throw new ForgeError('E_VALUE', `chunk 的 size 必须为正整数，收到 ${size}`);
    const out = [];
    for (let i = 0; i < list.length; i += s) out.push(list.slice(i, i + s));
    return out;
  },
});


// 分位数（Hyndman & Fan 类型 7，即 NumPy/R 默认线性插值定义）：
//   h = (n−1)·p/100，q = x[⌊h⌋] + (h−⌊h⌋)·(x[⌊h⌋+1] − x[⌊h⌋])
// 基准报告中 p50/p95/p99 延迟分位数由此给出，比均值更抗离群。
defineOp({
  id: nid(), domain: 10, name: 'sys.bench.percentile', arity: 2,
  sig: '(number[] xs, number p) -> number',
  doc: '分位数（Hyndman-Fan 类型 7 线性插值，NumPy 默认口径）',
  ref: 'Hyndman & Fan, Sample Quantiles in Statistical Packages, The American Statistician, 1996（类型 7）',
  inv: 'percentile(xs,0)=min(xs)；percentile(xs,100)=max(xs)；关于 p 单调不减',
  impl: (xs, p) => {
    if (!Array.isArray(xs) && !(xs && typeof xs.length === 'number')) {
      throw new ForgeError('E_TYPE', 'xs: 期望数值数组');
    }
    expectNumber(p, 'p');
    if (p < 0 || p > 100) throw new ForgeError('E_VALUE', `p 必须 ∈ [0,100]，收到 ${p}`);
    const arr = Array.from(xs, Number).filter(Number.isFinite).sort((a, b) => a - b);
    if (arr.length === 0) throw new ForgeError('E_VALUE', 'xs 不能为空（或无有限值）');
    if (arr.length === 1) return arr[0];
    const h = (arr.length - 1) * (p / 100);
    const lo = Math.floor(h), hi = Math.min(lo + 1, arr.length - 1);
    return arr[lo] + (h - lo) * (arr[hi] - arr[lo]);
  },
});

// —— 文件底部自检：域10 必须恰好 103 条，否则 import 即失败 ——
if (opsByDomain(10).length !== 103) {
  throw new Error(`域10 指令数错误: 期望 103，实际 ${opsByDomain(10).length}`);
}
