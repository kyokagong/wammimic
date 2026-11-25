# WAM Canonical Human v2.0 合格数据集使用手册

> **生效日期**：2026-11-23  
> **数据口径**：严格遵循 [统一数据处理方案（Canonical Human 五点契约 v2.0）](../design/data-processing.md)  
> **路径约定**：本文档所有命令均通过环境变量 `${P1_RUN_ROOT}`、`${RAW_DATA_ROOT}`、`${PRETRAINED_ROOT}`、`${RUNS_ROOT}` 注入，严禁在代码或配置中硬编码宿主机绝对路径（当前已完成数据位于 `runs/p1-data-20261120/processed`）。

---

## 1. 当前保留的合格数据总览

所有过去不合格的数据（旧版 3 条坐姿 `processed/nymeria` 12 GB、已解码的 `raw/nymeria` 53 GB、旧版仿真预览 `qa_previews/sim*` 以及存在双目跨缝裁剪缺陷的旧 `hiw500` 视频/潜码/Parquet 89 GB）已全部删除，`${P1_RUN_ROOT}/processed/` 下**仅保留当前通过 v2.0 规范处理并验收合格的数据**：

| 目录（相对 `${P1_RUN_ROOT}/processed`） | 数据来源 | `embodiment_tag` (`id`) | 回合数 | 训练窗 (`train`) | 磁盘占用 | 角色与状态 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `human-base/` + `human-new/` + `human-protocol/` | RoboMoCap 真人光学（NOKOV → Z-up FLU） | `human_5pt_only` (`2`) | 175 | 149,835 | 35.1 GB | **活跃训练分支（权重 25%）** · 6/6 门禁通过 |
| `gripper/` | OpenHLM G1 实机遥操（G1 FK → 标准人体） | `bimanual_parallel_gripper` (`1`) | 406 | 106,823 | 18 GB | **活跃训练分支（权重 25%）** · 6/6 门禁通过 |
| `nymeria-40/` | Nymeria 真人惯导（32 受试者 26/3/3 隔离） | `human_5pt_only` (`2`) | 32 | 217,510 | 141 GB | **活跃训练分支（权重 25%）** · 6/6 门禁通过 |
| `hiw500/` | HIW-500 G1 实机（左目无拼缝重裁 + 新 VAE 潜码） | `bimanual_parallel_gripper` (`1`) | 2,243 | 782,521 | 122 GB | **活跃训练分支（权重 25%）** · 6/6 门禁通过 |
| `human-operations/` | RoboMoCap 独立操作回归集（已转 Z-up FLU） | `human_5pt_only` (`2`) | 12 | —（只读留出） | 2.2 GB | **只读离线回归集**（严禁入训） |
| `canonical_human_5pt_stats.json` | 唯一全局五点统计量（四分支等权 `0.25` 联合拟合） | — | — | 1,256,689 | 24 KB | 所有训练与评估共用 |
| `canonical_v2_qa_summary.json` | 六项 QA 门禁汇总报告 | — | — | — | 8 KB | 验收留证 |

---

## 2. 六项 QA 门禁实测结果与运动分桶统计

完整报告见各分支 `qa_reports/<branch>_canonical_ready.json`、总报告 `canonical_v2_qa_summary.json` 及 20 窗视觉联系表 `${P1_RUN_ROOT}/qa_previews/<branch>_rgb_contact.png`：

| 分支 | Gate 1 轴向/踝高 (`L/R_ank_z`) | Gate 1 双腕 (`L/R_wri_y`, `wri_x`) | Gate 2 往返/正交 | Gate 3 拼缝异常 | Gate 4/5 因果/隔离 | Gate 6 全局 Z-score | 运动分桶 (`train`: A行走 / B操作 / C静立) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `human-base` | `-0.983 / -0.988` ✅ | `+0.285 / -0.268`, `+0.210` ✅ | `<1mm / <0.1°` ✅ | `0` ✅ | ✅ / ✅ | `1.348σ` ✅ | `9,506` / `3,488` / `3,385` |
| `human-new` | `-0.989 / -0.987` ✅ | `+0.212 / -0.199`, `+0.241` ✅ | `<1mm / <0.1°` ✅ | `0` ✅ | ✅ / ✅ | `0.801σ` ✅ | `71,696` / `32,852` / `28,908` |
| `gripper` | `-0.995 / -0.999` ✅ | `+0.250 / -0.267`, `+0.219` ✅ | `<1mm / <0.1°` ✅ | `0` ✅ | ✅ / ✅ | `0.871σ` ✅ | `27,000` / `16,233` / `64,244` |
| `nymeria-40` | `-0.978 / -0.979` ✅ | `+0.159 / -0.183`, `+0.206` ✅ | `<1mm / <0.1°` ✅ | `0` ✅ | ✅ / ✅ | `0.934σ` ✅ | `92,279` / `63,371` / `61,860` |
| `hiw500` | `-0.991 / -0.991` ✅ | `+0.203 / -0.200`, `+0.270` ✅ | `<1mm / <0.1°` ✅ | `0` ✅ | ✅ / ✅ | `1.122σ` ✅ | `59,826` / `91,456` / `631,239` |
| `human-operations` (只读留出) | `-0.993 / -0.979` ✅ | `+0.328 / -0.280`, `+0.192` ✅ | `<1mm / <0.1°` ✅ | `0` ✅ | ✅ / ✅ | `1.503σ`（未入训自然偏移） | 仅用于离线评估 |

---

## 3. 张量与 Parquet 字段契约（v2.0）

所有分支的 Parquet（`data/chunk-000/episode_XXXXXX.parquet`）统一使用 **Canonical Human（$H_{\text{leg}}^{\text{canon}} = 0.85\text{ m}, L_{\text{arm}}^{\text{canon}} = 0.58\text{ m}$）Z-up FLU、Yaw-Only `{P_t0}`** 表达，并新增 4 个运动学分桶字段：

- `action`：`[10, 45]`（真人）或 `[10, 47]`（机器人，后 2 维为夹爪目标 `[0, 1]^2`）
- `observation.state`（真人）或 `observation.state_5pt_hist`（机器人）：`[4, 75]` 或 `[4, 79]`（前 75D 为 45D 位姿 + 15D 线速 + 15D 角速，后 4D 为 `[ee_meas(2), ee_res(2)]`）
- `pelvis_disp_1s`（`float32`）：未来 1.0 s 骨盆水平位移（标准人体米制）
- `ankle_max_lift_1s`（`float32`）：未来 1.0 s 双踝最大抬脚高度（标准人体米制）
- `wrist_disp_1s`（`float32`）：未来 1.0 s 双腕相对骨盆最大位移（标准人体米制）
- `motion_bucket`（`int8`）：`0 = locomotion`（Bucket A）、`1 = manipulation`（Bucket B）、`2 = idle`（Bucket C）

---

## 4. 如何在训练与评估中使用

### 4.1 直接启动 WAM v2.0 训练（`scripts/train_wam_p1.py`）

[train_wam_p1.py](../../scripts/train_wam_p1.py) 已原生支持 v2.0 契约：
- **自动加载唯一全局统计量**：检测到 `${P1_RUN_ROOT}/processed/canonical_human_5pt_stats.json` 后自动为所有 4 个活跃分支统一施加该全局归一化（也可用 `--global-stats` 显式指定）；
- **默认分支配比**：`--branch-weights 0.25 0.0 0.25 0.25 0.25`（依次对应 `human_protocol`、`sim=0.0`、`gripper`、`nymeria-40`、`hiw500`）；
- **默认运动分桶采样**：`--bucket-weights 0.50 0.40 0.10`（Bucket A 行走/迈步 50%、Bucket B 近场主动操作 40%、Bucket C 静立 10%），彻底消除 `hiw500` 80% 静止窗的稀释效应。

示例启动命令（容器内挂载只读模型与已处理数据）：

```bash
docker run --rm --gpus '"device=0"' \
  -v "${PWD}:/workspace" \
  -v "${P1_RUN_ROOT}/processed:/mnt/p1_processed:ro" \
  -v "${PRETRAINED_ROOT}:/mnt/models:ro" \
  -v "${RUNS_ROOT}:/mnt/runs" \
  -w /workspace -e PYTHONPATH=/workspace \
  --entrypoint /opt/isaaclab-venv/bin/python \
  wammimic:train-pilot \
  scripts/train_wam_p1.py \
    --p1-processed-root /mnt/p1_processed \
    --backbone /mnt/models/nvidia/Cosmos-Predict2.5-2B \
    --backbone-manifest /mnt/models/cosmos_backbone_20261118.json \
    --run-root /mnt/runs/wam-v2-canonical-30k \
    --steps 30000 \
    --micro-batch 32 --accumulation 2 \
    --unfreeze-dit-blocks 2 --dit-lr 1e-5 --learning-rate 3e-4 \
    --branch-weights 0.25 0.0 0.25 0.25 0.25 \
    --bucket-weights 0.50 0.40 0.10
```

### 4.2 在 Python 代码中直接读取数据集

```python
from scripts.train_wam_p1 import MultiEmbodimentP1Dataset

ds = MultiEmbodimentP1Dataset(
    "/mnt/p1_processed",
    split="train",
    weights=(0.25, 0.0, 0.25, 0.25, 0.25),
    bucket_weights=(0.50, 0.40, 0.10),
)
sample = ds[0]
# sample keys:
#   video_latents:   [16, 5, 28, 28] BF16 (仅前 2 帧潜码作因果视觉条件)
#   text_embeddings: [512, 100352] BF16
#   proprio:         [4, 75] FP32 (全局统计量归一化后的 Canonical Human 历史)
#   ee_hist:         [4, 4] FP32
#   actions:         [10, 47] FP32 (前 45D 全局归一化 Canonical Human，后 2D 夹爪)
#   embodiment_id:   int64 (1=bimanual_parallel_gripper, 2=human_5pt_only)
#   ee_mask:         float32 (1.0=有末端监督, 0.0=真人无末端)
```

### 4.3 推理 / 闭环桥接（Canonical Human ↔ Q1 / G1 / Bumi）

在推理或接 Mimic 40k H=10 闭环评估时，直接调用 [canonical_human.py](../../wammimic/data_pipeline/canonical_human.py) 中的双向确定性接口，确保与训练数据 100% 同源：

```python
import json, numpy as np
from wammimic.data_pipeline.canonical_human import (
    ROBOT_GEOMETRY,
    world_flu_to_canonical_history,
    canonical_5pt_to_target_world,
)

gstats = json.loads(open("/mnt/p1_processed/canonical_human_5pt_stats.json").read())
geom = ROBOT_GEOMETRY["q1"]  # 或 "g1" / "bumi"

# 1. 上行桥接：机器人 50 Hz 历史 FK -> [1, 4, 75] Canonical Human -> 全局 Z-score 归一化
proprio_raw = world_flu_to_canonical_history(poses_50hz, times_50hz, hist_idx, geom["leg_height"], geom["arm_length"])
proprio_norm = (proprio_raw - np.array(gstats["proprio_5pt"]["mean"])) / np.maximum(gstats["proprio_5pt"]["std"], 1e-6)

# 2. 下行桥接：WAM 预测反归一化 -> 目标机型世界 FLU 五点 [10, 5, 7]（带地面防穿模）
pred_45 = pred_norm[:, :45] * np.array(gstats["action_5pt"]["std"]) + np.array(gstats["action_5pt"]["mean"])
target_world_5pt = canonical_5pt_to_target_world(
    pred_45, anchor_world_5pt, geom["leg_height"], geom["arm_length"],
    ankle_min_z=geom["ankle_rest_z"] - 0.005,
)
```

---

## 5. 后续新增数据或复现管线的标准命令

1. **单元测试**：
   ```bash
   python -m pytest tests/test_canonical_human.py -v
   ```
2. **刷新无需重算潜码的分支**（RoboMoCap / OpenHLM / Nymeria-40）：
   ```bash
   python scripts/refresh_canonical_v2.py --processed-root "${P1_RUN_ROOT}/processed" --raw-root "${RAW_DATA_ROOT}" --workers 16
   ```
3. **多卡并行缓存新视频 VAE 潜码**（如后续接入新视频源，支持按需指定 `--gpus 0,1` 断点续跑）：
   ```bash
   python scripts/cache_hiw500_4gpu.py \
     --model-root "${PRETRAINED_ROOT}" \
     --output-root "${P1_RUN_ROOT}/processed/hiw500" \
     --log-dir "${P1_RUN_ROOT}/control" \
     --gpus 0,1 --batch-size 32
   ```
4. **重拟合唯一全局统计量并跑全量六项 QA 门禁**：
   ```bash
   python scripts/finalize_canonical_v2.py --processed-root "${P1_RUN_ROOT}/processed"
   ```
