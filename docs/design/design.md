# WAM-Mimic 跨构型人形机器人全身移动操作总体架构设计（v2.0）

> **文档版本**：v2.0（2026-11-23，确立 **Canonical Human Z-up FLU 五点统一空间**、**Yaw-Only 骨盆参考系** 与 **双向跨构型桥接** 正确路线，全面清理 v1.x 中导致 G1→Q1 失配与鞠躬失控的错误约定）  
> **基准算力**：单机 4 × NVIDIA RTX 4090（24 GB/卡，PCIe 4.0）  
> **目标机身构型**：启元 PrimeBot Q1（`0.88 m`）、松延动力 Noetix Bumi（`0.94 m`）、宇树 Unitree G1（`1.32 m`）  
> **支持末端构型**：球头盲端（Stub）、双手平行夹爪（Parallel Gripper）、多规格双灵巧手（6-DoF / 7-DoF 及可扩展槽位）

## 文档导航（拆分设计体系）

为避免单文档冗余与前后口径冲突，设计体系拆分为四份职责清晰的权威文档：

| 文档 | 核心职责 |
| :--- | :--- |
| **[系统总体架构设计](design.md)**（本文） | 系统分层定位、核心数学契约（含 Global + Ref. 反馈与双源五点契约）、模型主干与 4×4090 容器化部署规范 |
| **[评估问题诊断与根因分析](evaluation-diagnosis.md)** | 复盘 `wam-real-5000-b64-dit2` 在 Genie Sim Q1/G1 上的闭环故障、同分布对照实验证据与纠偏依据 |
| **[统一数据处理方案](data-processing.md)** | **先总后分**：总体 Canonical Human Z-up 五点处理逻辑、运动分桶与六项 QA 门禁 + 逐数据集具体修复与处理清单 |
| **[WAM–Mimic 协同与桥接规范](wam-mimic-bridge.md)** | 上/下行双向跨构型运动学桥接、`goal_kp` + `ref_feedback` 双路对齐、Stage 1–4 联训与三级闭环评估矩阵 |

---

## 1. 核心架构准则（v2.0 纠偏定型）

1. **规划与运控解耦，WAM 严格收敛在“标准人体（Canonical Human）五点空间”**：
   - **Cosmos-WAM** 不学习、也不输出某一台具体机器人的物理尺寸，而是**永远在唯一统一的 Canonical Human Z-up FLU 五点空间**（标准站高 $H_{\text{leg}}^{\text{canon}} = 0.85\text{ m}$、标准臂长 $L_{\text{arm}}^{\text{canon}} = 0.58\text{ m}$）中，根据第一视角单目 RGB + 任务文本 + 4 帧标准人体本体历史，预测未来 `1.0 s`（`10 × 45D`）的五点运动趋势及末端执行器指令 $\mathbf{a}_{\text{ee}}^{(e)}$。
   - 无论训练数据来自真人动捕（RoboMoCap / Nymeria / GIMO / EgoHumanoid）还是真机遥操（OpenHLM G1 / HIW-500 G1），在 ETL 阶段均**统一重定向至同一个标准人体五点空间**并共享唯一全局五点归一化统计量（详见 [data-processing.md](data-processing.md)）。
2. **`t0` 水平朝向骨盆系（Yaw-Only Heading Frame）解耦俯仰失稳**：
   - WAM 的 4 帧历史与 10 帧未来目标均锚定在观测时刻 $t_0$ 骨盆的**水平朝向系** $\{P_{t_0}\}$（仅取偏航角 $\mathbf{H}_0 = \mathbf{R}_z(\psi_{t_0})$，不含 pitch/roll）。重力永远沿 $-Z$，骨盆真实俯仰/横滚保留在 `pelvis_rot6d` 中，彻底杜绝弯腰前倾时把双踝目标反投影到身后半空的“鞠躬正反馈”（详见 [evaluation-diagnosis.md §2.1](evaluation-diagnosis.md#21-wam-与-mimic-的架构偏离与桥接失配直接触发鞠躬正反馈)）。
3. **GR00T N2 风格 `EmbodimentTag` 路由专注末端执行器差异**：
   - 共享 `Cosmos-Predict2.5-2B` 主干 + `MoT Action Expert` 处理跨构型通用的标准人体 45D 空间推理；`embodiment_id`（`CategorySpecificMLP` + 条件嵌入）**仅用于路由异构末端执行器状态与动作段**（`human_5pt_only` / `bimanual_stub` / `bimanual_parallel_gripper` / `bimanual_dex_*`），不再用于强行隔离开裂的五点坐标系。
4. **双向跨构型桥接 + `Global-MorphMimic` 双路驱动**：
   - 闭环运行时，上行桥接把目标机器人（Q1 / Bumi / G1）4 帧实测 FK 映射到标准人体五点送入 WAM；下行桥接将 WAM 输出的标准人体五点**一路直通 Mimic 共享 Human Encoder（`goal_kp`）**，**另一路经 Yaw-Only 反投影并按目标机型腿长/臂长/地面防穿模约束精确重定向为机器人五点，送入 Actor Decoder 的必需主路 Global + Ref. 腕踝反馈（`ref_feedback`）**（详见 [wam-mimic-bridge.md](wam-mimic-bridge.md)）。
5. **同源数据 + WAM 回灌混合联训（Stage 3）**：
   - Stage 1 产出的 `Global-MorphMimic`（40k 步 `H=10` 预览）提供高质量全身运动先验；Stage 2 完成统一标准人体 WAM 训练后，通过 Stage 3 将 WAM 真实预测序列组装回灌与干净参考混合训练 Mimic，消除开环/闭环分布落差。

---

## 2. 系统总体闭环拓扑

```text
+-----------------------------------------------------------------------------------------+
|          System 1/2: Cosmos-WAM Policy Server (4~5 Hz, Canonical Human 5pt 空间)          |
|  [Text Emb] + [Monocular RGB (4x224x224)] + [CanonHuman 4x75] + [EE Hist 4x4] + [Tag e] |
|                 |                                                      |                |
|   WAN2.1 VAE Enc + Cosmos-Predict2.5-2B DiT (仅前 2 因果潜码)    ProprioEnc + EE_Enc[e]   |
|                 +------------------> Shared MoT Action Expert (Rectified Flow)          |
|                                      ├─ Shared Body Head  -> 10 x 45D (CanonHuman 5pt)  |
|                                      └─ EE Head[e]        -> 10 x D_ee^(e)              |
+----------------------------------------------+------------------------------------------+
                                               |
+----------------------------------------------v------------------------------------------+
|     Bidirectional Cross-Embodiment Bridge (50 Hz 插值、重定向与 6D 门控)                  |
|  1. 上行: 目标机型 (Q1/Bumi/G1) 4帧实测 FK -> Canonical Human Yaw-Only 4x75 -> WAM       |
|  2. 下行路 1 (goal_kp): Canonical Human 5pt -> Yaw-Only 还原 + 50Hz 插值 -> Human Enc   |
|  3. 下行路 2 (ref_feedback): Canonical Human -> 目标机型 (H_leg, L_arm, 地面保护) 重定向  |
|  4. 末端旁路: 双阶段手腕 6D 误差门控 (<=2.5 cm & <=8°) 释放夹爪/灵巧手闭合                 |
+----------------------+--------------------------------------------------+---------------+
                       | goal_kp (H=10) + 目标机型 ref_feedback (4x4x10)   | Gated q_ee_des
                       v                                                  | (Bypass WBC)
+----------------------------------------------------+                    |
|     System 0: Global-MorphMimic WBC (50 Hz, H=10)  |                    |
|  - 共享 Human/Five-Point Encoder -> motion latent  |                    |
|  - Global + Ref. 腕踝位姿误差 e_t (144D + 16D)      |                    |
|  - 本体状态历史 o_t -> Shared Actor Decoder         |                    |
+----------------------+-----------------------------+                    |
                       | 全身关节目标 q_body_des                           v
+----------------------v--------------------------------------------------+---------------+
|           物理仿真引擎 (MuJoCo CPU/Warp · Genie Sim 场景资产) / 真机执行层                 |
+-----------------------------------------------------------------------------------------+
```

---

## 3. 核心接口与数学契约

### 3.1 WAM 单帧 `45D + D_ee^(e)` 标准人体输出格式

WAM 输出未来 `1.0 s`（`10 Hz`，共 10 步）动作块 $A_{\text{chunk}}^{(e)} \in \mathbb{R}^{10 \times (45 + D_{\text{ee}}^{(e)})}$，锚定在 $t_0$ **标准人体水平朝向骨盆系 $\{P_{t_0}\}$（FLU Z-up，仅用 $\mathbf{R}_z(\psi_{t_0})$）**：

```python
# WAM Single-Frame Output Schema (v2.0 Canonical Human FLU Z-up, Yaw-Only {P_t0}):
CanonicalHumanFivePoint45D = {
    # --- 15D Position Block (5 points x 3D xyz in FLU: +X 前, +Y 左, +Z 上) ---
    "pelvis_norm_xyz":    [0, 1, 2],    # 骨盆相对 t0 骨盆位置 / 实测站立骨盆-踝垂距 H_leg
    "l_ankle_norm_xyz":   [3, 4, 5],    # 左踝相对 t0 骨盆位置 / H_leg (直立基准 z ≈ -1.00, y > 0)
    "r_ankle_norm_xyz":   [6, 7, 8],    # 右踝相对 t0 骨盆位置 / H_leg (直立基准 z ≈ -1.00, y < 0)
    "l_wrist_canon_xyz":  [9, 10, 11],  # 左腕映射至标准人体米制工作空间 (L_arm_canon = 0.58 m)
    "r_wrist_canon_xyz":  [12, 13, 14], # 右腕映射至标准人体米制工作空间 (L_arm_canon = 0.58 m)

    # --- 30D Orientation Block (5 points x 6D continuous rotation rot6d, 相对 H_0) ---
    "pelvis_rot6d":       [15, 16, 17, 18, 19, 20],  # 包含骨盆真实 pitch/roll 与相对 yaw
    "l_ankle_rot6d":      [21, 22, 23, 24, 25, 26],
    "r_ankle_rot6d":      [27, 28, 29, 30, 31, 32],
    "l_wrist_rot6d":      [33, 34, 35, 36, 37, 38],
    "r_wrist_rot6d":      [39, 40, 41, 42, 43, 44],
}
```

末端构型路由表（`embodiment_id` 仅区分末端通道，不隔离 45D 五点空间）：

| `embodiment_tag` | `id` | 用途说明 | $D_{\text{ee}}^{(e)}$ | 归一化区间 | `ee_mask` |
| :--- | :---: | :--- | :---: | :---: | :---: |
| `bimanual_stub` | `0` | 球头/盲端接触触发（保留槽位） | `2` | `[0, 1]` | `1` |
| `bimanual_parallel_gripper` | `1` | 双手平行夹爪（OpenHLM / HIW-500 G1） | `2` | `[0, 1]`（`0=开, 1=闭`） | `1` |
| `human_5pt_only` | `2` | 纯真人五点（RoboMoCap / Nymeria / GIMO / EgoHumanoid） | `0`（填零） | — | `0` |
| `bimanual_dex_6dof` / `dex3_7dof` | `3+` | 后续灵巧手扩展槽位（Stage 4 热插拔） | `12 / 14` | `[0, 1]` / `[-1, 1]` | `1` |

### 3.2 虚拟骨盆 FLU 坐标系对齐协议

所有进入 System 0 的机身 IMU 状态统一折算至遵循 ROS REP-103 前左上（FLU：$+X$ 前、$+Y$ 左、$+Z$ 上）规范的虚拟骨盆坐标系 $\{P\}$：

$$\mathbf{g}_{\text{pelvis}} = \mathbf{R}_P^I(\mathbf{q}_w)\,\mathbf{g}_{\text{imu}}, \qquad \boldsymbol{\omega}_{\text{pelvis}} = \mathbf{R}_P^I(\mathbf{q}_w)\,\boldsymbol{\omega}_{\text{imu}} - \mathbf{J}_w(\mathbf{q}_w)\dot{\mathbf{q}}_w$$

### 3.3 四分组均衡流匹配损失（WAM Stage 2）

对每个构型 $e$，将流匹配速度场 MSE 拆分为下肢（`[0:9]`）、双腕（`[9:15]`）、五点旋转（`[15:45]`）与带掩码的归一化末端段四组均衡加权，防止 30D 旋转稀释 15D 位置精度，并对无末端标签的真人分支严格屏蔽末端梯度（`ee_mask = 0`）：

$$\mathcal{L}_{\text{FM}}^{(e)} = \lambda_{\text{lpos}}\mathcal{L}_{\text{MSE}}(\hat{\mathbf{v}}_{0:9}, \mathbf{v}^*_{0:9}) + \lambda_{\text{wpos}}\mathcal{L}_{\text{MSE}}(\hat{\mathbf{v}}_{9:15}, \mathbf{v}^*_{9:15}) + \lambda_{\text{rot}}\mathcal{L}_{\text{MSE}}(\hat{\mathbf{v}}_{15:45}, \mathbf{v}^*_{15:45}) + m_{\text{ee}}\frac{\lambda_{\text{ee}}}{D_{\text{ee}}^{(e)}}\left\|\hat{\mathbf{v}}_{\text{ee}}^{(e)} - \mathbf{v}_{\text{ee}}^{*,(e)}\right\|_2^2$$

### 3.5 共享 Actor Decoder 的 Global + Ref. 腕踝反馈

共享 Actor Decoder 在每个 50 Hz 控制步联合使用五点编码潜码 $z_t$、本体历史 $o_t$、构型条件 $m$ 和 **Global + Ref. 腕踝位姿误差 $e_t$**（144D + 16D 有效位）输出全身关节目标：

1. **目标来源与尺度**：取 $0、20、40、60\text{ ms}$ 四帧及四个末端（左踝、右踝、左腕、右腕）。**目标必须已按目标机身完成腿长、臂长与局部轴标定**（在 WAM 闭环中由 [wam-mimic-bridge.md §2.2](wam-mimic-bridge.md#22-下行桥接wam-标准人体输出--mimic-双路目标) 从标准人体五点重定向生成）。
2. **误差定义**：令 $R_P^W(t)$ 为当前实测骨盆到世界系的旋转，$i \in \{\text{LA, RA, LW, RW}\}$，$\tau \in \{0, 20, 40, 60\}\text{ ms}$：

   $$e_{i,\tau}(t) = \left[\,R_P^W(t)^\top\!\left(p_i^{\mathrm{ref},W}(t+\tau)-p_i^{\mathrm{FK},W}(t)\right) \;\middle\|\; \operatorname{rot6d}\!\left(R_i^{\mathrm{FK},W}(t)^\top R_i^{\mathrm{ref},W}(t+\tau)\right)\,\right]$$

   $$\mathbf{q}^{\mathrm{des}}_{\mathrm{body},t} = D_{\mathrm{shared}}\!\left(z_t,\ o_t,\ m,\ e_t,\ v_t\right)$$

3. **契约校验**：训练（Stage 1/3）、离线评估、闭环评测与 WebRTC Preview 必须通过 `verify_ref_contract()` 共享同一反馈版本与时序定义。

### 3.6 原始人体与机器人参考五点的共同数据契约（Mimic 侧）

Mimic 侧保持对两类来源五点的兼容支持（对应 `mimic-all-motion-resume12000-40000-20261122`，`H=10` 十帧预览）：

| 来源标记 | 存储与用途 |
| :--- | :--- |
| `original_human` | 配对的 SMPL/SOMA/BVH 或标准人体五点，进入共享 Human Encoder（`goal_kp`） |
| `robot_reference` | 从目标机型 50 Hz NPZ 提取的具名 body 五点，经固定局部轴矩阵 `C = Rz(HUMAN_YAW_OFFSET) Rx(HUMAN_ROLL_OFFSET)` 对齐后共同参与五点编码训练；`ref_feedback` 主路始终使用目标机型 FK 尺度的世界坐标腕踝目标 |

---

## 4. 代码仓库与容器化部署规范

### 4.1 核心目录映射（与当前仓库一致）

| 目录 / 文件 | 职责 |
| :--- | :--- |
| `conf/data/pipeline.yaml` | 数据源相对子目录、清单与来源注册（标准库 JSON 兼容 YAML） |
| `wammimic/data_pipeline/` | 统一标准人体五点 ETL、各数据源适配器（`human_ego.py`, `nymeria.py`, `openhlm.py`, `hiw500.py`）、潜码缓存与 QA 校验 |
| `wammimic/training/` | `CosmosWAM` 模型（`model.py`）、损失、离线评估（`evaluation.py`）与可视化 |
| `scripts/train_wam_p1.py` | WAM 多源联合训练入口（待更新为全局标准人体统计量 + 运动分桶采样） |
| `scripts/eval_wam_offline.py` | WAM 离线多留出集评估（标准人体米制 ADE / 0.2s 误差 / 测地角 / EE MAE） |
| `scripts/evaluate_geniesim_wam_mimic.py` / `scripts/evaluate_geniesim_indist.py` | Genie Sim Q1 / G1 闭环评估与同分布对照验证入口 |
| `scripts/serve_wam_preview.py` + `scripts/wam_preview.html` | 闭环回放 WebRTC 可视化服务 |
| `runs/` | 训练与评估运行目录（配置、日志、检查点、回放 MP4/JSONL） |

### 4.2 容器化与环境变量解耦规范

所有容器（`docker/Dockerfile.data`、`docker/Dockerfile.train`、预览/评测镜像 `blinkmorphmimic:preview-*`）遵循统一卷挂载原则，严禁硬编码宿主机路径：

| 环境变量 | 容器挂载点 | 权限 | 用途 |
| :--- | :--- | :---: | :--- |
| `RAW_DATA_ROOT` | `/mnt/raw_data` | `ro` | 原始多源数据（只读，防误写） |
| `PROCESSED_DATA_ROOT` / `P1_RUN_ROOT` | `/mnt/processed_data` | `rw`（处理）/ `ro`（训练） | 标准化 Parquet、潜码缓存、全局统计量与 QA 报告 |
| `PRETRAINED_ROOT` | `/mnt/pretrained` | `ro` | `Cosmos-Predict2.5-2B`、VAE 与 `Cosmos-Reason1-7B` 固定修订权重 |
| `RUNS_ROOT` | `/mnt/runs` | `rw` | WAM / Mimic 检查点、TensorBoard 与闭环评测回放 |

### 4.3 4 × RTX 4090 算力使用原则
- **离线潜码预缓存**：始终离线运行 `Cosmos-Reason1-7B` 文本编码与 `WAN2.1` VAE 视频编码落盘，训练期不加载编码器，单卡节省 4–6 GB 显存；
- **Stage 2 WAM 训练**：使用 BF16 混合精度 + FP32 可训主参数；单卡可跑 `micro_batch=32`（配合梯度累积 `effective_batch=64`），4 卡 DDP 可加速至 **2–3 epoch（25k–40k 步）** 充分收敛；
- **闭环评测**：使用固定物理种子与 MuJoCo 零溢出诊断，必须同时运行原生 G1 同分布探针与 Q1 跨构型闭环评测，并通过 WebRTC 页面留存回放证据。
