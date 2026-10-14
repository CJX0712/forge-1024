// 错误分类学（Error Taxonomy）
// 所有微指令错误统一携带 code，供 exec 层归类、测试断言与上层恢复策略使用。
// 设计对标：Rust 的 error kinds / Python 异常层级，但保持零依赖。

export const CODES = {
  E_ARITY: '参数个数不符',
  E_TYPE: '参数类型不符（应为 Tensor / number / string 等）',
  E_SHAPE: '张量形状不符或不可广播',
  E_VALUE: '参数取值非法（如负方差、k>n、空数据）',
  E_NUMERICAL: '数值失败（不收敛、奇异矩阵、溢出）',
  E_LIMIT: '超出资源限额（迭代次数 / 时间预算）',
  E_NOTFOUND: '指令不存在',
  E_STATE: '状态错误（未初始化 / 重复 backward / 越界访问）',
};

export class ForgeError extends Error {
  constructor(code, message) {
    super(`[${code}] ${message}`);
    this.name = 'ForgeError';
    this.code = code;
  }
}

export function forgeError(code, message) {
  return new ForgeError(code, message);
}
