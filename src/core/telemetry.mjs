// 遥测：指令调用计数、耗时、追踪、错误计数。
// 计时默认关闭（避免污染微基准）；bench 工具按需开启。
const calls = new Map();   // opId -> 次数
const errors = new Map();  // opId -> 次数
const times = new Map();   // opId -> 累计毫秒
let enabled = true;
let timing = false;
let tracing = false;
const trace = [];

export function now() {
  if (typeof process !== 'undefined' && process.hrtime && process.hrtime.bigint) {
    return Number(process.hrtime.bigint()) / 1e6; // 毫秒
  }
  if (typeof performance !== 'undefined' && performance.now) return performance.now();
  return 0;
}

export function record(id, t0) {
  if (!enabled) return;
  calls.set(id, (calls.get(id) || 0) + 1);
  if (timing) times.set(id, (times.get(id) || 0) + (now() - t0));
  if (tracing && trace.length < 10000) trace.push({ id, t: now() });
}

export function recordError(id) {
  if (!enabled) return;
  errors.set(id, (errors.get(id) || 0) + 1);
}

export function summary() {
  const rows = [];
  for (const [id, n] of calls) {
    rows.push({ id, calls: n, errors: errors.get(id) || 0, ms: +(times.get(id) || 0).toFixed(3) });
  }
  rows.sort((a, b) => b.calls - a.calls);
  return rows;
}

export function totalCalls() {
  let s = 0;
  for (const n of calls.values()) s += n;
  return s;
}

export function hotOps(k = 10) { return summary().slice(0, k); }

export function reset() { calls.clear(); errors.clear(); times.clear(); trace.length = 0; }

export function setEnabled(b) { enabled = b; }
export function setTiming(b) { timing = b; }
export function setTracing(b) { tracing = b; if (!b) trace.length = 0; }
export function getTrace() { return trace.slice(); }
