# WAM–Mimic 训练协同、跨构型桥接与闭环评估规范（v2.0）

> **本文档定位**：定义 **WAM（标准人体五点规划层）** 与 **Global-MorphMimic（跨机身全身运控层）** 之间的精确接口、双向重定向数学协议、Stage 1–4 协同训练策略以及闭环评测矩阵。彻底废除旧版 `q1_direct_targets()` 直接跨机型硬塞坐标与 `ckpt_12000` H=1 短视评测的错误做法。  
> **关联文档**：[系统总架构设计](design.md) · [评估问题诊断与根因分析](evaluation-diagnosis.md) · [统一数据处理方案](data-processing.md)

---

## 1. 双向跨构型桥接架构（对齐 MorphMimic + Anchor 1 $H_5'$）

本桥接设计严格遵循两篇参考论文的分工边界：
- **MorphMimic**（[morphmimic.pdf](https://lzyang2000.github.io/morphmimic/static/morphmimic.pdf) · `morphmimic/model/human.py`）：五点目标采用 **锚点 $t_0$ 水平朝向系（Yaw-Only Heading Frame）**，直接输入共享 Human Encoder（`goal_kp`），由形态条件化的共享隐空间映射至 Q1 / Bumi / G1 全身动作；局部轴通过可逆矩阵 $\mathbf{C} = \mathbf{R}_z(\text{HUMAN\_YAW\_OFFSET})\mathbf{R}_x(\text{HUMAN\_ROLL\_OFFSET})$ 显式传输；
- **Anchor 1**（[arXiv:2611.00193](https://arxiv.org/pdf/2611.00193)）：用 **结构保持五点几何标定（$H_5'$）** 替代在线关节 IK，并由 **Global + Ref.** 显式世界系腕踝误差反馈（`ref_feedback`）保障高精度末端操作与无漂移全局跟踪。

在 v2.0 架构中，WAM **永远只认识唯一的标准人体（Canonical Human / $H_5'$）Z-up FLU 五点空间**；所有与具体机器人身高、臂长、静立踝高相关的变换，全部集中封装在确定性的**双向几何标定桥接层**与 **Mimic 底层全身控制器** 中：

```text
                ┌────────────────────────────────────────────────────────────┐
                │   Cosmos-WAM (4~5 Hz, 严格运行于 Canonical Human 5pt 空间)   │
                │   输入: 4帧 RGB + Text + 4x75 标准人体历史 + 4x4 EE历史 + e  │
                │   输出: 10x45 标准人体五点目标 (Yaw-Only系) + 10xD_ee 末端    │
                └───────────────────────▲──────────────────────┬─────────────┘
                                        │                      │
         [上行桥接: Robot -> CanonHuman] │                      │ [下行桥接: CanonHuman -> Mimic 双路]
         Q1/G1/Bumi 实测 4 帧 FK        │                      ├─────────────────────────┐
         按本机 (H_leg, L_arm, C_rest)  │                      │                         │
         映射到标准人体 Yaw-Only 系      │                      ▼                         ▼
         用唯一全局 stats.json 归一化    │          (路 1) goal_kp 直通         (路 2) ref_feedback 重定向
                                        │          Canonical Human 5pt         Yaw-Only 反投影 + 按目标机型
                ┌───────────────────────┴──┐       (供共享 Human Encoder)      (H_leg, L_arm, 地面防穿模)
                │  物理仿真 / 真机状态回读   │               │                   生成目标机型世界 5 点
                │  (Q1 / Bumi / G1 @ 50Hz) │               ▼                         ▼
                └───────────────────────▲──┘      ┌────────────────────────────────────────────┐
                                        │         │  Global-MorphMimic WBC (50 Hz, 40k H=10)   │
                                        └─────────┤  共享 Human Encoder + Global+Ref. Decoder   │
                                     q_body_des   └────────────────────────────────────────────┘
```

---

## 2. 桥接层四条硬数学规范（彻底杜绝鞠躬失控与跨机型漂移）

### 2.1 上行桥接（目标机器人实测 FK → WAM 标准人体输入）
每个规划周期 $t_0$，从当前机器人（Q1 / Bumi / G1）的 50 Hz 状态历史中取过去 4 帧（$t_0 - 0.3\text{s}, -0.2\text{s}, -0.1\text{s}, 0$）实测五点世界 FLU 位姿，执行与训练数据处理（[data-processing.md §1.2](data-processing.md#12-标准人体canonical-human五点空间数学规范)）**完全同源的逆映射**：
1. 取 $t_0$ 骨盆水平朝向 $\mathbf{H}_0 = \mathbf{R}_z(\psi_{t_0})$（**仅用 Yaw，不用俯仰/横滚**）；
2. 下肢（骨盆、左踝、右踝）位置与线速度除以该机型实测静立骨盆-踝垂距 $H_{\text{leg}}^{(r)}$（Q1 `0.379 m`、G1 `0.745 m`，以当前仿真资产静立测得为准）；
3. 双腕位置与线速度拆分为骨盆平移项（乘 $H_{\text{leg}}^{\text{canon}}/H_{\text{leg}}^{(r)}$）+ 相对骨盆臂展项（乘 $L_{\text{arm}}^{\text{canon}}/L_{\text{arm}}^{(r)}$）；
4. 五点旋转去除机型零位偏置并相对 $\mathbf{H}_0$ 求 `rot6d` 与局部角速度；
5. 用全局唯一的 `canonical_human_5pt_stats.json` 做 Z-score 归一化，送入 WAM `proprio [4, 75]`。
- **验收效果**：无论在 Q1 还是 G1 上启动，静立第 0 步的 `proprio` 都稳定落在标准分布中心附近（`[-1σ, +1σ]`），彻底根除旧版 Q1 输入 `+7.3σ` OOD 问题。

### 2.2 下行桥接（WAM 标准人体输出 → Mimic 双路目标）
WAM 采样并经唯一全局 `canonical_human_5pt_stats.json` 反归一化，得到未来 10 步（10 Hz）标准人体五点 `block [10, 45]`：

1. **航向原点与有限前瞻（Bounded XY Lead + Standing Z Anchor）**：
   - 使用当前机器人骨盆的 **Yaw-Only 矩阵** $\mathbf{H}_0 = \mathbf{R}_z(\psi_{t_0})$ 反投影，**严禁使用带 pitch/roll 的全 3D 矩阵 `r0`**；
   - 水平原点 `origin_xy` 保留相对上一块参考轨迹最多 `max_lead_m = 0.08 m` 的有界前瞻（防止跟踪滞后造成“原地踏步锁定”或“无限漂移”）；
   - 垂直原点 `origin_z` 锚定在站立骨盆高度 `robot_rest_z`（由 WAM 预测的 $\Delta z_0$ 控制蹲起，防止轻微下蹲被累进锁死）。
2. **路 1：`goal_kp`（直通 Mimic 共享 Human Encoder）**：
   - 将标准人体局部五点经 $\mathbf{H}_0$ 还原至世界系，施加 Mimic 约定的 SMPL/Human 局部轴对齐矩阵后，经 50 Hz 样条插值（位置三次样条 + 旋转 Slerp）拼装 10 帧预览窗口送入 `pack_human_goal()`。
3. **路 2：`ref_feedback`（Global + Ref. 目标机型世界五点）**：
   - **必须按目标机型的真实运动学尺度把标准人体五点重定向为机器人五点**（见 [ref_feedback.py](../../runs/mimic-all-motion-resume12000-40000-20261122/source/src/morphmimic/model/ref_feedback.py) 的 `embodiment_retargeted_robot_trajectory` 契约）：
     - 下肢目标位置乘以目标机型实测站高 $H_{\text{leg}}^{(r)}$；
     - 双腕相对骨盆位移乘以目标机型臂长比 $L_{\text{arm}}^{(r)} / L_{\text{arm}}^{\text{canon}}$（Q1 约 `0.62`，G1 约 `0.86`）；
     - **地面防穿模保护**：双踝世界高度执行下限钳位 $z_{\text{ank}}^{\text{target}} \ge z_{\text{ank}}^{\text{rest},(r)} - 0.005\text{ m}$，禁止将脚踝目标压入地下。
   - 经 50 Hz 插值后取 `0, 20, 40, 60 ms` 四帧计算世界坐标腕踝误差（$e_t$ 144D + 有效位 $v_t$ 16D）送入 Mimic Actor Decoder。

### 2.3 双阶段手腕 6D 空间误差门控（末端执行器旁路）
当启用平行夹爪或灵巧手（`embodiment_id ∈ {1, 3, ...}`）时，末端指令 $\mathbf{a}_{\text{ee}}^{(e)}$ 不经过 WBC，由桥接层按单侧手独立执行双阶段门控：
- **预塑形阶段**（$\phi_{\text{close}} < \eta_{\text{close}}^{(e)}$）：直通下发预张开宽度或预抓手型；
- **闭合触发阶段**（$\phi_{\text{close}} \ge \eta_{\text{close}}^{(e)}$）：等待对应手腕跟踪误差同时满足 $\|\Delta p_{\text{wrist}}\| \le 2.5\text{ cm}$ 且 $\Delta\theta_{\text{wrist}} \le 8^\circ$（或超时 `600 ms`）后释放闭合，并短暂稳住当前五点目标 `150 ms` 防撞翻物体。

### 2.4 Mimic 基线版本强制要求
- **正式闭环强制使用 `ckpt_40000.pt`（H=10 十帧五点预览）**（当前位于 `runs/mimic-all-motion-resume12000-40000-20261122/{q1,bumi,g1}/ckpt_40000.pt`）。
- 废弃旧版 12,000 步 `H=1` 权重作为闭环默认基线（仅留作历史对照归档）。

---

## 3. WAM 与 Mimic 四阶段协同训练路线

| 阶段 | 核心目标 | 关键训练配置与数据契约 | 达标门禁（进入下一阶段前提） |
| :--- | :--- | :--- | :--- |
| **Stage 1 · Mimic 基础全身运控** | 掌握全动作库 + 标准人体/机器人五点 `H=10` 预览 + Global+Ref. 跟踪 | Q1 / Bumi / G1 三机身联合训练至 40k 步（**已完成**：`mimic-all-motion-resume12000-40000-20261122`） | 留出动捕集高存活率 + 参考五点闭环成功按下按钮（`button-retarget-ablation` 100% 通过） |
| **Stage 2 · WAM 统一标准人体多源训练** | 学会从单目 RGB + 文本 + 4 帧标准人体历史稳定预测未来 1.0 s 标准人体五点 + 末端动作 | 使用 [data-processing.md](data-processing.md) 重处理后的纯真实多源集（修完 `hiw500` 双目裁剪、全 Z-up FLU、唯一全局统计量、运动桶配比 50%/40%/10%）；解冻 DiT 末 2 Block，训练 **2–3 epoch（约 25k–40k 步，batch 64）** | 1. 各分支验证损失均衡（彻底消除 `nymeria` 1.80 异常）；<br>2. 真人测试/留出集 `ADE < 120 mm`、`0.2s < 45 mm`、旋转 `< 18°`；<br>3. G1 夹爪测试集 `ADE < 45 mm`、`EE MAE < 0.06` |
| **Stage 3 · WAM 预测回灌联训 Mimic** | 让 Mimic 免疫 WAM 5 Hz 重规划跳变、单目深度微偏与轻微动力学不一致 | 用 Stage 2 达标 WAM 在共享轨迹上离线 rollout，经 §2.2 Yaw 桥接生成含真实预测特征的五点窗口，与干净数据按 **50:50** 混合，开启 `kp_jitter` / `kp_drift` 续训 Mimic | 在相同 WAM 输入下，相比 Stage 1 Mimic，电机 Jerk 显著下降、零倾倒率提升、末端腕距误差收敛 |
| **Stage 4（可选）· 新末端少样本热插拔** | 扩展 6/7-DoF 灵巧手而不重训主干与 Mimic | 冻结 WAM 视觉/动作主干与 Stage 3 Mimic，仅训练新增 `embodiment_id` 的末端编解码头与嵌入 | 新末端留出集 `EE MAE` 达标且保持五点导航不退化 |

---

## 4. 分层闭环评估协议（G1 上界探针 + Q1 跨构型迁移）

为避免“把跨构型桥接问题误判成 WAM 能力问题”，所有闭环评估必须同时报告以下三层对照：

| 评估层级 | 评测内容与意义 | 固定基准任务 |
| :--- | :--- | :--- |
| **Tier 1 · WAM 离线开环** | 检验统一标准人体空间下的预测精度（均在标准人体米制与测地角下汇报） | `human_protocol_test`、`human_operations`（只读回归）、`nymeria40_test`、`gripper_openhlm_test`、`hiw500_test` |
| **Tier 2 · 原生 G1 同分布上界探针** | 消除跨机身差异，检验 WAM 视觉-动作规划本身能否驱动人形稳定走近并触碰/操作目标 | Genie Sim `openhlm-red-can-g1`（白桌 + 红罐 + 橙垫，`OpenHLM task_0` 文本，G1 1.0× 尺度 + Mimic G1 40k） |
| **Tier 3 · Q1 / Bumi 跨构型闭环迁移** | 检验“Canonical Human 五点 → 双向桥接 → Mimic 40k”的跨身高零样本泛化能力 | 1. 同分布场景：`openhlm-red-can-q1`；<br>2. 零样本 OOD 场景：`press-button`、`matching-block-contact`；<br>3. 必配对照：`reference_retargeted`（验证 Mimic 与场景物理本身无问题） |

### 闭环量化验收指标
每次闭环评测（每场景至少 3 个固定种子，正式回归扩展至 10–100 回合）须同步记录并公开：
- **稳定性**：存活步数占比、`anchor_ori / anchor_pos` 摔倒率（目标 `< 5%`）、物理溢出位数（必须为 `0`）；
- **移动能力**：骨盆净前进位移 `pelvis_net_dx_m`、双踝净位移与最大抬脚高度 `ankle_max_z_m`（确认真迈步而非原地弯腰）；
- **操作能力**：最近腕距 `min_wrist_distance_m`、物理手物接触帧数 `hand_contact_steps`、按钮行程 `button_max_travel_m`（按键成功门禁 `≥ 3 mm`）及最终任务成功率 `SR_task`；
- **可视化留证**：WebRTC / MP4 回放（含头相机 RGB 画中画、世界五点目标球标、实时腕距与步数），严禁仅凭离线 ADE 宣称闭环达标。
