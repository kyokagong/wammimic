# WAM P1 训练数据与训练执行指南

> **数据策略更新（2026-11-21 UTC）**：鉴于当前合成仿真数据（`sim`）在绕障与桌面交互动作自然度上仍存在明显失真，**暂时停止将仿真数据加入 WAM 训练集**（`sim` 采样权重置为 `0.0`，现有仿真目录仅保留作为离线参考与 Genie Sim 闭环评测环境）。WAM 训练集全面聚焦 **真实人类第一视角全身数据（RoboMoCap、Nymeria、GIMO、EgoHumanoid、HumanVerse-500）** 与 **真实机器人遥操数据（OpenHLM G1）**。  
> **路径约束**：本文档所有宿主机路径均通过环境变量 `${P1_RUN_ROOT}`、`${RAW_DATA_ROOT}`、`${PRETRAINED_ROOT}`、`${RUNS_ROOT}` 注入容器，严禁在配置或脚本中硬编码宿主机绝对路径。

---

## 1. WAM 真实训练集架构总览（零仿真入训）

所有训练分支统一对齐至 **`t0` 完整 3D 骨盆坐标系**、**4 帧因果历史五点本体状态** 与 **Cosmos-Predict2.5-2B** 预缓存视频/文本潜码（`[16, 5, 28, 28]` BF16 + `[512, 100352]` BF16）。

### 1.1 当前已就绪真实训练分支

| 数据分支 | 目录（相对 `${P1_RUN_ROOT}/processed`） | `embodiment_tag` (`id`) | 当前规模 (`train / val / test`) | 本体历史状态 | 动作目标 | 坐标往返最大误差 | 角色与纯真实采样权重 |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **D2a · RoboMoCap 光学协议** (`human-protocol`) | `human-protocol/protocol.json`<br>（合并 `human-base` + `human-new`） | `human_5pt_only` (`2`) | **175** 会话 / **186,003** 窗口<br>(`149,835 / 16,670 / 19,498`) | `proprio [4, 75]` | `10 × 45`<br>(`ee_mask = 0`) | `1.26e-7 m`<br>`3.07e-6°` | **当前主干（55%）**：真实人体光学动捕 + 头相机，覆盖避障行走与桌面操作 |
| **D3 · OpenHLM G1 实机遥操** (`gripper`) | `gripper/wam_lerobot_v2/g1_parallel_gripper` | `bimanual_parallel_gripper` (`1`) | **406** 回合 / **131,944** 窗口<br>(`106,823 / 10,774 / 14,347`) | `proprio_5pt [4, 79]`<br>(`4×75` + `4×4` 夹爪实测/残差) | `10 × 47`<br>(`45D` + `2D` 夹爪 `0=开,1=闭`) | `6.86e-8 m`<br>`2.96e-6°` | **机器人真机先验（25%）**：G1 真实双臂夹爪抓放与行走操作 |
| **D2b · Nymeria 惯导参考** (`nymeria` + `nymeria-40`) | `nymeria/` + `nymeria-40/` | `human_5pt_only` (`2`) | **3 + 32** 录制 / **~289,530** 窗口<br>(`nymeria-40` 32 名受试者按 `26/3/3` 隔离) | `proprio [4, 75]` | `10 × 45`<br>(`ee_mask = 0`) | `1.23e-7 m`<br>`4.36e-6°` | **真人扩量主力（30%）**：Aria 第一视角 + Xsens 240 Hz 全身真值 |
| **D3b · HIW-500 G1 开放世界** (`hiw500`) | `hiw500/wam_lerobot_v2/g1_hiw500` | `bimanual_parallel_gripper` (`1`) | **~100 GB 精选子集**（覆盖 11 类开放世界移动操作任务，只下头相机 + Parquet） | `proprio_5pt [4, 79]`<br>（G1 29-DoF FK + WBC 根速度/夹爪） | `10 × 47`<br>(`45D` + `2D` 夹爪) | `< 1 mm`<br>`< 0.1°` | **真机 G1 主力增量（20%）**：`BitRobot/HIW-500-LeRobot` 真实开放世界多场景 G1 第一视角移动操作 |
| **D1 · Genie Sim 仿真** (`sim`) | — | `bimanual_stub` (`0`) | 已清理无用训练缓存 | — | — | — | **暂停入训（权重 `0%`）**：仅保留 Genie Sim 闭环评测环境 |

> **回归留出集**：`processed/human-operations/`（12 会话 / 12,185 窗口）保持独立只读，严禁并入训练集。

### 1.2 下一阶段重点接入的四大真人 Ego 全身数据集

为重点补齐 **“室内真实避障行走 → 走到桌台/开关前 → 自然伸手按键/触碰/操作”** 能力，后续训练集扩展全面围绕以下四个真人第一视角 + 3D 全身数据源推进（全部统一映射至 `human_5pt_only` 协议，`proprio [4, 75]` + `actions [10, 45]`）：

| 数据集 | 第一视角 RGB | 全身 3D 位姿来源 | 与“室内绕障 + 桌面/开关交互”的契合点 | 标签来源标识 (`label_provenance`) | 接入方式与优先级 |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Nymeria** (Meta ECCV 2024) | Project Aria RGB | Xsens MVN Link 240 Hz 全身真值 | **P0（立即可扩）**：300h / 1,200 序列，大量真实室内绕开家具/行人走向台面、门、开关、冰箱并伸手交互，自带分层自然语言描述 | `xsens_inertial_mocap_reference` | **已打通解码管线**；按语言标注定向筛选绕行+交互序列，采用“流式下载→解码缓存→删 VRS”扩容 |
| **GIMO** (Stanford ECCV 2022) | 头戴 Ego RGB + 眼动 | 3D 室内场景扫描 + SMPL-X 全身动作 | **P1（场景-交互强对齐）**：专门面向多家具障碍室内导航与目标交互（接近、触碰、按压、拿取、落座） | `smplx_indoor_ego_reference` | 申请学术非商用下载权限；新增 `gimo` 适配器提取 SMPL-X 骨盆/双踝/双腕五点并过往返校验 |
| **EgoHumanoid / V2** (OpenDriveLab) | ZED / 头戴 Ego RGB | PICO VR / 穿戴式全身 3D 追踪 | **P1（面向人形移动操作设计）**：直接以 LeRobot 格式对齐真人第一视角与人形全身移动操作动作空间 | `vr_ego_humanoid_reference` | 复用开源 `eh2-collect-human` 与对齐脚本；支持接入其公开子集及同协议定向实采（如“绕障→按键”专项 50–100 条） |
| **HumanVerse-500** ($\lambda_0$ / OpenLambda) | 穿戴式 Ego RGB | 穿戴式全身 + 手部 3D 动捕 | **P2（大规模开放世界增量）**：500h / 32,000 回合 / 827 开放世界任务，涵盖走向家具、伸手与物体交互 | `wearable_openworld_human_reference` | 跟进 `LambdaHumanoid/openlambda` 与 Hugging Face 公开切片及对齐工具，按统一五点契约增量接入 |

---

## 2. 统一张量契约与坐标系规范

### 2.1 单样本张量定义
所有真人与实机分支共享以下核心字段，读取器按各自 `train` split 统计独立归一化（不同 `label_provenance` 的归一化统计严禁混用）：

| 字段 | 形状 / 类型 | 说明 |
| :--- | :---: | :--- |
| `video_latents` | `[16, 5, 28, 28]` (`BF16`) | 17 帧 10 Hz RGB（4 帧历史 + 12 帧未来）经 Cosmos-Predict2.5 VAE 编码；**仅前 2 个时间潜码作为动作分支观测条件**，后 3 帧仅用于未来视频辅助损失 |
| `video_condition_mask` | `[1, 5, 28, 28]` (`FP32`) | 时间维前 2 帧为 `1`，后 3 帧为 `0`，严格防止未来画面泄漏 |
| `text_embeddings` | `[512, 100352]` (`BF16`) | 固定版本 Cosmos-Reason1 28 层拼接特征，按各数据包根目录下 `prompt_bank_bf16.safetensors` 索引 |
| `proprio` / `proprio_5pt` | `[4, 75 + D_ee_obs]` (`FP32`) | **过去 4 帧（`t0-0.3s .. t0`）严格因果历史状态**：每帧 `45D` 五点位姿 + `15D` 线速度 + `15D` 角速度（`75D` 共享主干）；`gripper` 末尾拼接 `4D` 夹爪历史（`79D`），真人分支为 `75D` |
| `actions` | `[10, 45 + D_ee]` (`FP32`) | 未来 `t0+0.1s .. t0+1.0s`（10 Hz）动作块。前 `45D` 为 `15D` 五点位置 + `30D` `rot6d`；真人分支 `ee_action_mask = 0` |
| `embodiment_id` | `int64` | `0 = bimanual_stub`（当前停训保留）、`1 = bimanual_parallel_gripper`、`2 = human_5pt_only` |

### 2.2 统一坐标与尺度约定（与 P0 桥接器严格一致）
- **参考系**：`actions[:, :45]` 与 `proprio[:, :75]` 全部锚定在观测时刻 `t0` 的**完整 3D 骨盆坐标系**（含 pitch/roll，与 `anchor_quaternion_wxyz` 对齐）。
- **下肢尺度**：骨盆与左右踝位置/线速度除以该构型/受试者实测腿长（`Q1 = 0.44 m`、`Bumi = 0.48 m`、`G1 = 0.68 m`、人类取各自 `human_leg_length`）；左右腕位置/线速度保持米制。
- **往返校验硬门禁**：每个新接入的真人数据集（GIMO / EgoHumanoid / HumanVerse-500 及 Nymeria 扩容批次）必须通过 `human_features -> block_targets` 往返校验（位置误差 **`< 1 mm`**，旋转误差 **`< 0.1°`**）且严格按受试者/会话划分 `train / val / test`。

---

## 3. 纯真实多源训练启动方式（禁用仿真分支）

使用 [scripts/train_wam_p1.py](../../scripts/train_wam_p1.py)，通过将 `sim` 对应的第二位采样权重置为 `0.0`（例如 `--branch-weights 0.55 0.00 0.25 0.20`），即可在完全不采样仿真数据的前提下执行纯真实数据训练（待 GIMO / EgoHumanoid / HumanVerse-500 接入后，按独立 `human_5pt_only` 分支并入权重列表）：

```bash
export P1_RUN_ROOT=<你的P1运行目录>
export PRETRAINED_ROOT=<你的预训练权重根目录>
export RUNS_ROOT=<你的训练输出根目录>

docker run --rm --pull never --network none --gpus '"device=0"' --shm-size 16g \
  -v "$(pwd):/workspace:ro" \
  -v "${P1_RUN_ROOT}/processed:/mnt/p1_processed:ro" \
  -v "${PRETRAINED_ROOT}:/mnt/pretrained:ro" \
  -v "${RUNS_ROOT}:/mnt/runs" \
  -w /workspace \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint /opt/isaaclab-venv/bin/python \
  wammimic:train-pilot scripts/train_wam_p1.py \
  --p1-processed /mnt/p1_processed \
  --backbone /mnt/pretrained/nvidia/Cosmos-Predict2.5-2B \
  --backbone-manifest /mnt/pretrained/cosmos_backbone_manifest.json \
  --run /mnt/runs/wam-real-only-b32 \
  --steps 10000 --micro-batch 32 --accumulation 1 \
  --branch-weights 0.55 0.00 0.25 0.20 \
  --checkpoint-every 1000 --keep-checkpoints 2
```

> **禁止事项**：严禁按帧重新随机拆分 `train/val/test`；严禁把不同 `label_provenance`（如光学动捕 `robomocap`、惯导 `nymeria`、SMPL-X `gimo`、VR `egohumanoid`）的原始归一化统计混合计算；在 `sim` 质量达标前严禁将低质仿真轨迹并入训练权重。

---

## 4. 模型架构与训练超参建议

- **架构（已落地于 `wammimic/training/model.py`）**：
  - 冻结 `Cosmos-Predict2.5-2B` 前缀（BF16），仅取前 2 个时间潜码作为因果观测条件；
  - 4 帧历史本体编码器（`proprio [4, 75]` + 末端历史 `ee_hist [4, 4]`）与视觉/文本条件拼接进入 Action Expert；
  - GR00T N2 风格 `embodiment_id` 条件路由，真人分支自动屏蔽末端损失（`ee_mask = 0`）。
- **推荐超参**：
  - 单卡 RTX 4090：`micro_batch=32, accumulation=1`（`effective_batch=32`），`AdamW`（`lr=3e-4`, `warmup=200`, cosine decay），`--keep-checkpoints 2`；
  - 多卡扩展（4×4090 DDP）：可同步解冻 DiT 末 2 Block（`lr=1e-5`）并扩展至 `effective_batch=64`、`20,000–30,000` 步。
- **监控**：TensorBoard 日志写入 `${RUNS_ROOT}/<run_name>/tensorboard`，重点跟踪 `lower_position_flow_loss`、`wrist_flow_loss`、`rotation_flow_loss`、`ee_flow_loss` 及各真人验证集 ADE。

---

## 5. 验收门禁（纯真实训练 → 零样本闭环迁移）

1. **真实人类留出集（`human-protocol` test + `nymeria` test + 固定回归集 `human-operations` + 后续 `gimo/egohumanoid` test）**：
   - 位置 ADE **`< 140 mm`**（显著优于 `training_mean` 基线 `~235 mm`），`0.2s` 短时误差 **`< 50 mm`**，旋转测地角误差 **`< 20°`**，退化旋转比例 `= 0%`。
2. **真机夹爪留出集（`gripper` test）**：
   - 五点位置 ADE **`< 50 mm`**，双夹爪开合预测 MAE **`< 0.08`**。
3. **Genie Sim 零样本闭环评估（仅评测、不入训）**：
   - 使用 [evaluate_geniesim_wam_mimic.py](../../scripts/evaluate_geniesim_wam_mimic.py) 验证零摔倒稳定追踪，以及从真人 Ego 数据迁移到仿真场景的避障接近与手部操作泛化能力。

更多四大真人 Ego 数据集的筛选、流式解包、适配与接入细则见 [P1 下一步数据收集与处理执行指南](p1-data-next-steps-guide.md)。
