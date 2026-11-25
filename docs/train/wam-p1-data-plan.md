# WAM P1 阶段数据补全与采集规范（历史规划归档）

> **状态说明（2026-11-23 已由 v2.0 替代）**：本文档为 2026-11-19 针对 Phase 0（18 个会话）阶段制定的早期 P1 数据补全草案。其中“优先用 D1 合成仿真按键数据入训”“双腕保留各机型原始米制”“各分支独立归一化”等早期方案已被实际闭环实验证伪并正式废弃。  
> **现行权威规范请直接查阅**：
> - 总体与逐数据集处理规范：[docs/design/data-processing.md](../design/data-processing.md)
> - 闭环问题诊断与纠偏依据：[docs/design/evaluation-diagnosis.md](../design/evaluation-diagnosis.md)
> - 当前 P0/P1 数据执行路线：[docs/train/p1-data-next-steps-guide.md](p1-data-next-steps-guide.md)

---

## 保留的历史事实与有效结论（供追溯）

1. **已完成的早期缺口补齐**：
   - 真实光学人体数据已从早期 18 个会话扩充至 `human-protocol` 175 个会话（`186,003` 窗口，详见 [2026-11-20 执行记录](wam-p1-data-progress-20261120.md)）；
   - WAM 已完成 4 帧历史本体状态与 `embodiment_id` 末端路由接入，并已接入真实 G1 遥操数据（OpenHLM 406 回合 + HIW-500 1,280 回合）与 Nymeria（3 + 32 录制）。
2. **D1 合成仿真入训已暂停**：
   - 早期试采的合成仿真按钮轨迹因动作生硬及手物同框门禁未过而被整批隔离，当前 `sim` 分支权重保持 `0.0`，Genie Sim 仅用作闭环评测平台。
3. **D4（Stage 3 WAM 回灌）依然保留**：
   - 待基于 v2.0 Canonical Human Z-up 五点数据重训的新版 WAM 通过离线与 G1/Q1 闭环门禁后，仍按 [WAM–Mimic 协同与桥接规范](../design/wam-mimic-bridge.md#3-wam-与-mimic-四阶段协同训练路线) 执行 Stage 3 WAM 预测回灌联训 Mimic，并开启 `kp_jitter` / `kp_drift`。
4. **通用工程底线延续不变**：
   - 严禁硬编码宿主机绝对路径，所有存储目录通过环境变量只读/读写挂载注入容器；
   - 严禁按帧随机拆分 `train/val/test`，必须按完整受试者、录制会话或独立场景隔离；
   - 任何新数据入训前必须经过自动化数值门禁、坐标往返校验与视觉联系表 / Rerun 3D 抽检。
