# WAM 训练数据与训练执行手册（v2.0 Canonical Human 口径）

> **重要更新（2026-11-23）**：本手册已全面对齐 **v2.0 架构**（见 [系统总架构设计](../design/design.md)、[评估问题诊断](../design/evaluation-diagnosis.md)、[统一数据处理方案](../design/data-processing.md)、[WAM–Mimic 协同与桥接规范](../design/wam-mimic-bridge.md)）。旧版“各分支保留各自坐标系/米制双腕/独立统计量”“全 3D 骨盆系反投影”等错误约定已作废。  
> **数据策略**：在合成仿真动作自然度达标前继续禁用 `sim` 分支入训（权重 `0.0`，仅作闭环评测环境）；全面聚焦纯真实人类 Ego 与真机遥操数据，并**统一重定向至标准人体（Canonical Human / $H_5'$）Z-up FLU 五点空间**。  
> **路径约束**：所有宿主机根目录均通过环境变量 `${P1_RUN_ROOT}`、`${RAW_DATA_ROOT}`、`${PRETRAINED_ROOT}`、`${RUNS_ROOT}` 注入容器，严禁硬编码。

---

## 1. 训练前必做的存量数据纠偏（已完成 ✅）

> **状态更新（2026-11-23）**：以下四项存量修复、不合格旧数据清理（已删除旧坐姿 `nymeria`、原始 VRS、`sim` 预览与旧错裁 `hiw500`）、唯一全局统计量拟合与全量六项 QA 门禁**均已完成并通过验收**。详细数据资产清点、QA 实测指标与训练/桥接代码用法见 **[Canonical Human v2.0 合格数据集使用手册](canonical-v2-data-usage.md)**。

在启动下一轮 WAM 训练前，各分支的 v2.0 纠偏项与目标训练权重如下：

| 数据分支 | 目录（相对 `${P1_RUN_ROOT}/processed`） | `embodiment_tag` (`id`) | 当前存量规模 | 必须先完成的 v2.0 纠偏项 | 推荐目标分支权重 |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **RoboMoCap 光学** (`human-protocol`) | `human-protocol/protocol.json`<br>（`human-base` + `human-new`） | `human_5pt_only` (`2`) | 175 会话 / `186,003` 窗 | **① Y-up → Z-up FLU 旋转对齐**；② Yaw-Only 系 + 臂长标准化；③ 标出定脚任务并降权（视频/文本潜码直接复用） | **25%** |
| **Nymeria 惯导** (`nymeria-40`) | `nymeria-40/`（隔离旧 3 条坐姿） | `human_5pt_only` (`2`) | 32 受试者 / `~292k` 窗 | ① 隔离旧 3 条坐姿子集；② 重导出 Yaw-Only 标准人体五点 + 双标尺（视频/文本潜码直接复用） | **25%** |
| **OpenHLM G1 实机** (`gripper`) | `gripper/wam_lerobot_v2/g1_parallel_gripper` | `bimanual_parallel_gripper` (`1`) | 406 回合 / `131,944` 窗 | **① G1 FK → 标准人体五点重定向**（修正踝高 `-1.095 → -1.00` 与臂长）；② 可选增补去黑边内切视口潜码 | **25%** |
| **HIW-500 G1 实机** (`hiw500`) | `hiw500/wam_lerobot_v2/g1_hiw500` | `bimanual_parallel_gripper` (`1`) | 1,280 回合 / `~556k` 窗 | **① 修复 `480×1280` 双目跨缝裁剪 Bug（改裁左目 `480×480`）并重跑视频与 VAE 潜码**；② G1 FK → 标准人体五点；③ 原地静止窗降采样 | **25%** |
| **Genie Sim 仿真** (`sim`) | — | `bimanual_stub` (`0`) | 仅保留闭环评测环境 | 权重固定 `0.0`，不入训 | **0%** |

> **只读回归集**：`processed/human-operations/`（12 会话 / 12,185 窗口）同步执行 Y-up → FLU 转换供离线评估，严禁并入训练集。  
> **运动分桶采样**：在分支权重之上叠加运动桶配比（Bucket A 行走/移动接近 `50%~55%`、Bucket B 主动操作 `35%~40%`、Bucket C 静立 `≤10%`），彻底消除原地定脚偏置。

---

## 2. 统一张量契约（v2.0）

所有分支经重处理后严格共享以下格式与**唯一全局五点统计量 `canonical_human_5pt_stats.json`**（仅由各活跃分支 `train` split 联合拟合）：

| 字段 | 形状 / 类型 | 说明 |
| :--- | :---: | :--- |
| `video_latents` | `[16, 5, 28, 28]` (`BF16`) | 17 帧 10 Hz RGB 经固定版 Cosmos-Predict2.5 VAE 编码；**仅前 2 个时间潜码作为因果观测条件** |
| `text_embeddings` | `[512, 100352]` (`BF16`) | 固定版 Cosmos-Reason1-7B 28 层逐 token 归一化拼接特征 |
| `proprio` | `[4, 75]` (`FP32`) | 过去 4 帧（`t0-0.3s .. t0`）严格因果 **Canonical Human Z-up FLU（Yaw-Only `{P_t0}`）** 五点位姿+线速+角速，使用**全局统一统计量**归一化 |
| `ee_hist` | `[4, 4]` (`FP32`) | 过去 4 帧 `[ee_meas(2), ee_res(2)]`；真人分支置零并由 `ee_mask=0` 屏蔽 |
| `actions` | `[10, 47]` (`FP32`) | 未来 `t0+0.1s .. t0+1.0s`（10 Hz）：前 `45D` 为**标准人体五点**（使用**全局统一统计量**归一化），后 `2D` 为末端开合（`0=开, 1=闭`） |
| `embodiment_id` | `int64` | **仅路由末端执行器编解码头**：`0=stub`、`1=parallel_gripper`、`2=human_5pt_only` |

---

## 3. 模型与训练超参规范

- **模型主干**（`wammimic/training/model.py`）：
  - `Cosmos-Predict2.5-2B`（BF16 冻结前缀，解冻末 2 层 DiT Block 使用 FP32 主权重，`lr=1e-5`）；
  - 共享 `MoT Action Expert`（`width=384, layers=4`，`lr=3e-4`, `warmup=200`, cosine decay）；
  - 四分组均衡流匹配损失（下肢 `[0:9]`、双腕 `[9:15]`、旋转 `[15:45]`、带 `ee_mask` 的末端 `[45:47]`）。
- **训练充分度要求**：
  - 此前 5,000 步 × batch 64 仅覆盖约 `0.31 epoch`（严重欠拟合）。新版训练目标为 **2–3 epoch**（约 `25,000–40,000` 步，`effective_batch=64`，建议 4×4090 DDP 加速），每 2,000 步记录各分支验证损失与离线 ADE。
- **训练健康度硬监控**：
  - 所有活跃分支验证流匹配损失须同步收敛在同一量级（如 `0.25–0.40`），严禁出现类似旧版 `nymeria = 1.807` 的异常离群分支。

---

## 4. 验收门禁（离线 + G1 上界探针 + Q1 跨构型闭环）

1. **Tier 1 · 离线开环留出集**（在标准人体米制与测地角口径下报告）：
   - `human_protocol_test`、`human_operations`（只读回归）、`nymeria40_test`：位置 ADE **`< 120 mm`**，`0.2s` 误差 **`< 45 mm`**，旋转测地角 **`< 18°`**，退化旋转 `= 0%`；
   - `gripper_openhlm_test`、`hiw500_test`：位置 ADE **`< 45 mm`**，夹爪 `EE MAE < 0.06`。
2. **Tier 2 · 原生 G1 同分布闭环探针**（使用 [evaluate_geniesim_indist.py](../../scripts/evaluate_geniesim_indist.py) + **Mimic G1 40k H=10**）：
   - `openhlm-red-can-g1` 稳定直立前行接近桌台并自然抬臂触碰红罐，零 `anchor_ori` 倾倒。
3. **Tier 3 · Q1 跨构型闭环评估**（双向 Canonical Human ↔ Q1 桥接 + **Mimic Q1 40k H=10**）：
   - 同时评估同分布 `openhlm-red-can-q1` 与零样本 OOD `press-button` / `matching-block-contact`，报告存活率、骨盆/双踝净位移、抬脚高度、最近腕距与任务成功率，并通过 WebRTC 回放目视核验。
