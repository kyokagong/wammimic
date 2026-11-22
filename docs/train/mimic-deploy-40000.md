# Mimic 40000 steps 部署模型

交付目录：`/data/kk/artifact/mimic-40000-onnx-20261122/`。
来源为本仓库最新完成的 `runs/mimic-all-motion-resume12000-40000-20261122`，
Q1、Bumi、G1 均使用固定 `ckpt_40000.pt` 的 **EMA、float32、batch=1、ONNX opset 18**。

## 模型与关键接口变化

每个机型目录包含 `motion.onnx`、`human_h10.onnx`、`metadata.json`、`manifest.json`、
`SHA256SUMS` 及 `verification/` 原图和数值测试数据。归一化、形态和固定五点掩码已经固化，
四层 FiLM 及两处 morph 拼接完成折叠。机器人状态和参考反馈仍为动态输入。

| 模型 | float32 输入，均未归一化 | 输出 |
|---|---|---|
| motion | `obs_history[1,930]`、`goal_motion[1,670]`、`ref_feedback[1,160]` | `action_motion[1,29]` |
| human_h10 | `obs_history[1,930]`、`goal_human[1,2160]`、`human_heading[1,60]`、`ref_feedback[1,160]` | `action_human[1,29]` |

**Human 预览从旧版 H1 改为 H10**：SMPL 五点槽仍是 `[0,7,8,20,21]`，
但第 0 至 9 帧全部启用，heading=True。数组宽度与旧模型相同，时间语义已经改变。
调用方需要连续十个 50 Hz 目标帧；实时输入只有当前帧时，不应视为已经满足本模型的预览契约。

`obs_history` 为十帧实测观测，由旧到新，每帧
`[q_rel29, qd29, imu_gyro3, gravity3, previous_action29]`。
Motion 目标为十帧 `[ref_q29, ref_qd29, anchor_position3, relative_anchor_rot6d6]`。
人体特征为十帧、24 槽、每槽位置3+rot6d6，先在人体骨盆首帧 heading 坐标系表达，再由图内归一化和掩码。
`human_heading` 相对当前实测机器人 anchor 计算，并补偿 SMPL rest-frame。
rot6d 统一为矩阵前两列按行展平 `[m00,m01,m10,m11,m20,m21]`，四元数 wxyz，米/弧度。
冻结的特征定义位于 `source-reference/`，供核对移植，它不是完整执行器。

两个入口都需要 `ref_feedback`：当前实测 FK 对比同机型参考的 0/20/40/60ms 目标，
按左踝、右踝、左腕、右腕排列。前 144D 为位置/旋转误差，末 16D 为有效位。
位置差表达在当前完整骨盆坐标系，旋转差表达在当前末端坐标系。
当前四末端任一无效时模型保留 NaN 输出守卫，执行器须拒绝非有限输出。
原生关节映射、IMU/anchor、默认位、缩放和 PD 参数都取**对应机型**的 `metadata.json`。

输出是 canonical action，实际角度按有效槽映射：

```text
n = perm[c]  # 跳过 n=-1
target[n] = native_default[n] + sign[c] * action[c] * native_action_scale[n]
```

## 动作是否需要更新

详见 `action-update-analysis.json`：

- 旧 34,621 条机器人参考及原始人体载荷的 manifest 哈希逐条一致，无删除或替换。
- 前次精简包的 21 条参考或中立位来源均用当前实际文件复查，未变化，可继续复用
  `/data/kk/artifact/mimic-12000-compact-20261121/references/`。
- 新训练增加一条 **Q1 官方起身** `official_q1_getup`，本包附增量参考；
  当前 Preview 的 Q1 躺卧快捷入口选择它。旧坐姿、半蹲和其他机型起身窗口保持原样。
- MotionDecode 新增机器人 FK 来源的五点标签。完整训练库共有 1,378 份机器人参考五点
  （1,377 份 MotionDecode 加 Q1 官方起身），属于机器人参考，不是真实人体标注。

没有重新打包整个动作库。本包只附与原精简范围相关的增量：

```text
action-updates/q1/official_q1_getup.npz                 500 帧、50 Hz
action-updates/robot_reference50/q1/official_q1_getup.npz
action-updates/robot_reference50/{q1,bumi,g1}/motiondecode_baji_WP_Baji_Quan_17851016.npz
action-updates/updates.json                           路径、来源、哈希及与旧包的关联
action-updates/official-q1-validation.json             FK、地面适配和原始来源收据
```

官方起身从帧 0 开始，帧 350 起为 150 帧站立保持；保留该参考本身的姿态与朝向，
不要套用旧中立位的末端世界位置或把全部起身入口都切换到它。
它在导入时仅做地面净空的竖直平移，原始官方关节姿态保持。
八极拳仍复用旧机器人参考，新配套 NPZ 是 24 槽 SMPL 传输约定的参考五点侧车，
按当前 `human_features_from_goal/heading_from_goal` 构造 H10 特征。
它不是可直接传入模型的 `[1,2160]` 张量，也不是人体传感器数据。

## 校验、复核与复现

在包根目录运行（Python 3.11+、NumPy、ONNX Runtime）：

```bash
sha256sum -c SHA256SUMS
python export_mimic_deploy.py --check q1
python export_mimic_deploy.py --check bumi
python export_mimic_deploy.py --check g1
```

每图 260 组输入，原始 ONNX、优化 ONNX、float32 PyTorch 三方比较，阈值 `1e-4`。
四组留出参考覆盖真实人体与机器人参考五点，另加 256 组 std=0.2 的原始特征扰动，
有效位保持二值。还检查缺失关节精确零、无效当前参考触发 NaN。
最终实测见 `validation-summary.json`；源 checkpoint、数据、源码和模型 SHA 均可追溯。

重新导出时使用原冻结源码、对应机器人资产及训练环境；从仓库根运行：

```bash
python scripts/export_mimic_deploy.py \
  --source runs/mimic-all-motion-resume12000-40000-20261122/source \
  --checkpoint runs/mimic-all-motion-resume12000-40000-20261122/q1/ckpt_40000.pt \
  --data-root runs/mimic-official-recovery-training-20261122/data \
  --output runs/new-40000-export/q1
```

对 Bumi/G1 改 checkpoint 和输出路径。脚本从 checkpoint 自动选择 H1/H10，拒绝不支持的预览长度；
所有数值验证通过后才发布新目录，不覆盖已有包。模型浮点精度不变，未做 INT8、FP16 或 TensorRT。
`prior-checkpoint-evaluation/` 是同权重的既有 PyTorch 闭环评估，未作为本次 ONNX 闭环结果。
本次未重新进行 ONNX 闭环或实机验收；实际部署沿用本体校准、限位、故障保护和急停。

## 本次导出实测

六张部署图均通过 260 组验证及独立复核；六项回归测试通过，旧 H1 包独立复核兼容。

| 机型 | 模型 | 三方比较最大绝对误差 | 节点数（原→优化） | 图大小 MiB |
|---|---|---:|---:|---:|
| q1 | motion | 9.54e-06 | 234→70 | 36.06 |
| q1 | human_h10 | 8.58e-06 | 303→84 | 48.45 |
| bumi | motion | 1.55e-05 | 234→70 | 36.06 |
| bumi | human_h10 | 2.05e-05 | 303→84 | 48.45 |
| g1 | motion | 9.07e-06 | 234→70 | 36.06 |
| g1 | human_h10 | 1.31e-05 | 303→84 | 48.45 |
