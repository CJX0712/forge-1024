// 伪随机数发生器：PCG-XSL-RR 32bit。
// 参考：M.E. O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically Good
// Algorithms for Random Number Generation", 2014（pcg-random.org）。
// 64bit 状态以 BigInt 精确实现；Box-Muller 正态带缓存；同种子序列完全确定（可复现）。
const MASK64 = (1n << 64n) - 1n;
const MASK32 = 0xffffffffn;
const MULT = 6364136223846793005n;

export function createRng(seed = 20261014) {
  let state = 0n;
  const inc = 1n; // 定序流，序列仅由种子决定
  state = (state * MULT + inc) & MASK64;
  state = (state + BigInt(seed >>> 0)) & MASK64;
  state = (state * MULT + inc) & MASK64;
  let normalCache = null;

  function nextU32() {
    const old = state;
    state = (old * MULT + inc) & MASK64;
    const xorshifted = Number(((old >> 18n) ^ old) >> 27n) & 0xffffffff;
    const rot = Number(old >> 59n);
    return ((xorshifted >>> rot) | (xorshifted << ((32 - rot) & 31))) >>> 0;
  }

  const rng = {
    // [0,1) 均匀
    next() { return nextU32() / 4294967296; },
    nextU32,
    // [a,b) 均匀实数
    range(a, b) { return a + (b - a) * rng.next(); },
    // [0,n) 均匀整数
    int(n) { return Math.floor(rng.next() * n) % n; },
    // 标准正态（Box-Muller，成对生成带缓存）
    normal() {
      if (normalCache !== null) { const v = normalCache; normalCache = null; return v; }
      let u1 = rng.next();
      if (u1 < 1e-300) u1 = 1e-300;
      const u2 = rng.next();
      const r = Math.sqrt(-2 * Math.log(u1));
      normalCache = r * Math.sin(2 * Math.PI * u2);
      return r * Math.cos(2 * Math.PI * u2);
    },
    // 正态 N(mu, sigma)
    normalScaled(mu, sigma) { return mu + sigma * rng.normal(); },
    bernoulli(p) { return rng.next() < p ? 1 : 0; },
    // Fisher-Yates，返回新数组（不改入参）
    shuffle(arr) {
      const a = arr.slice();
      for (let i = a.length - 1; i > 0; i--) {
        const j = rng.int(i + 1);
        [a[i], a[j]] = [a[j], a[i]];
      }
      return a;
    },
    pick(arr) { return arr[rng.int(arr.length)]; },
    // 不放回抽样 k 个
    sample(arr, k) { return rng.shuffle(arr).slice(0, k); },
    // 获取内部状态（用于序列持久化/复现）
    getState() { return { state: state.toString(16), hasCache: normalCache !== null }; },
  };
  return rng;
}

// 由字符串派生 32bit 种子（FNV-1a），便于用文字种子复现实验。
export function seedFrom(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}
