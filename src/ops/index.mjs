// 指令域汇总：导入即注册。每域恰好 102 条，合计 1024 条。
// 分层依赖：D1<-D2<-D3<-D4<-{D5,D6}<-D7<-D8<-D9<-D10
import './d1-math.mjs';
import './d2-num.mjs';
import './d3-linalg.mjs';
import './d4-autodiff.mjs';
import './d5-opt.mjs';
import './d6-prob.mjs';
import './d7-ml.mjs';
import './d8-dl.mjs';
import './d9-train.mjs';
import './d10-sys.mjs';
