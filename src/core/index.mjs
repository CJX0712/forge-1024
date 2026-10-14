// 内核门面：一条 import 拿到全部核心能力。
// 分层依赖规则（域间）：D1<-D2<-D3<-D4<-{D5,D6}<-D7<-D8<-D9<-D10；D10 仅依赖注册表。
import {
  KERNEL_VERSION, CATALOG_VERSION, DOMAINS,
  exec, tryExec, getOp, hasOp, listOps, countOps, opsByDomain, searchOps, describeOp,
} from './registry.mjs';
import * as telemetry from './telemetry.mjs';

export {
  Tensor, prod, stridesOf, sameShape, broadcastShapes,
  isTensor, expectTensor, expectNumber, elementwise, reduceTensor,
} from './tensor.mjs';
export { createRng, seedFrom } from './rng.mjs';
export { ForgeError, CODES, forgeError } from './errors.mjs';
export {
  defineOp, exec, tryExec, getOp, hasOp, listOps, countOps, opsByDomain, searchOps, describeOp,
  DOMAINS, KERNEL_VERSION, CATALOG_VERSION, requiredParams,
} from './registry.mjs';
export { telemetry };

export const kernel = {
  version: KERNEL_VERSION,
  catalog: CATALOG_VERSION,
  call: exec,
  tryCall: tryExec,
  get: getOp,
  has: hasOp,
  list: listOps,
  count: countOps,
  byDomain: opsByDomain,
  search: searchOps,
  describe: describeOp,
  stats: () => telemetry.summary(),
  hotOps: (k) => telemetry.hotOps(k),
  resetStats: () => telemetry.reset(),
  totalCalls: () => telemetry.totalCalls(),
  domains: DOMAINS,
};
