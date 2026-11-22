# Mimic 12000 steps ONNX 部署指南

交付目录：`/data/kk/artifact/mimic-12000-onnx-20261121/`。模型来自
`/data/kk/wammimic/runs/mimic-flip-12000-20261119/{q1,bumi,g1}/ckpt_12000.pt`，
均选择 **EMA、float32、batch=1、ONNX opset 18**。按机型选择子目录，不能混用常量和关节映射。

## 文件与模型选择

每个机型子目录包含：

- `motion.onnx`：机器人原生参考轨迹接口。
- `human_h1.onnx`：人体骨盆、双踝、双腕五点，只有当前一帧，heading 有效。
- `metadata.json`：checkpoint 哈希、输入约定、机器人关节顺序、映射、默认位、动作缩放、PD 参数、归一化和验证结果。
- `manifest.json`、`SHA256SUMS`：产物及来源哈希；两个分支均通过才发布目录。
- `verification/*.original.onnx`：折叠前、同一机器人常量与掩码的原始图。
- `verification/inputs.npz`：260 组输入及 float32 PyTorch 期望输出。
- `export_mimic_deploy.py`：可在边缘环境独立数值复核的同版本脚本。

交付根目录另外包含本指南、`validation-summary.json`、主机 CPU 复核/耗时报告、
导出环境版本与复现脚本，以及同 checkpoint 的既有完整 val 评估报告副本。
`source-reference/` 保存特征构造与反馈的冻结源码供移植核对；它是供阅读的源码子集，不是可安装的运行包。

## 图边界：比旧版增加 160D 实时反馈

所有输入为 **未归一化的 float32 特征**，输出为 29 维 canonical 动作。
归一化已在图内，禁止上游重复归一化。所有维度固定，不能输入 batch>1。

| 模型 | 输入 | 输出 |
|---|---|---|
| `motion.onnx` | `obs_history[1,930]`、`goal_motion[1,670]`、`ref_feedback[1,160]` | `action_motion[1,29]` |
| `human_h1.onnx` | `obs_history[1,930]`、`goal_human[1,2160]`、`human_heading[1,60]`、`ref_feedback[1,160]` | `action_human[1,29]` |

Q1 原生 22 关节、Bumi 21、G1 29；三个模型的输出宽度仍统一为 29，缺失槽精确为零。
`human_h1` 的 H1 是单帧 horizon，不是机器人型号。

九项常量 `morph`、`obs_mean/var`、`goal_mean/var`、`human_mean/var`、`heading_mean/var`
来自该机型 checkpoint，固化到相应图中；未使用的分支常量随依赖清理。
Human 掩码固定为 24 点中的 `[0,7,8,20,21]`、10 帧中的 `[0]`，heading=True。
没有运行时 morphology 或掩码开关，也没有冻结实际机器人姿态或参考目标。

## 观测历史

50 Hz（20 ms）更新，每帧 93 维，连续十帧从旧到新展平，最新一帧是当前反馈：

```text
[q_rel(29), qd(29), imu_gyro(3), projected_gravity(3), previous_action(29)]
```

- `q_rel`：实测 canonical 关节角减 canonical 默认位，rad；`qd` 为 rad/s。
- `gyro`、单位重力方向在 `metadata.json` 的 `imu_body` 坐标系内；按正确 IMU 安装外参处理。
- `previous_action` 是上一控制步的 canonical 网络输出，动作缩放之前；缺失关节槽为零。
- 复位时遵循原生 `ObsHistory`：以当前观测填满十帧，随后逐步滚动。
- 位置映射为 `q_c = sign[c] * (q_native[perm[c]] - offset[c])`，速度同符号转换。
  具体数组取本机型元数据；不要把原生/SDK 顺序直接送入模型。

归一化使用 `(x-mean)/(sqrt(var)+0.01)`，观测按单帧 93D 统计重复处理。

## Motion 目标

采用 goal version 2、`goal_anchor=reference`。未来十个 50 Hz 连续参考帧逐帧拼接：

```text
[reference_q(29), reference_qd(29), anchor_position(3), relative_anchor_rot6d(6)]
```

参考 q 是绝对 canonical 关节角。anchor 位置在第一目标帧参考骨盆的 heading 坐标系内：
原点为该参考骨盆 XY，Z 保留世界高度；姿态为参考 torso 相对当前实测机器人 torso 的完整旋转。
rot6d 一律取旋转矩阵前两列并按行展平：`[m00,m01,m10,m11,m20,m21]`。
片段末尾重复最后参考帧。遵循冻结源码 `model/rewards_torch.py::goal_features`，不要替换为全世界 XYZ 或 Euler 角。

## Human H1 特征

输入不是原始 5×7 位姿。先按 SMPL 24 点槽把人体骨盆、左右踝、左右腕放入
`[0,7,8,20,21]`，四元数为 wxyz，单位米；坐标系与训练人体预处理一致。
`model/human.py::human_features_from_goal` 生成 `[10,24,9]`：每点位置 3D 与 rot6d 6D，
在第 0 帧人体骨盆 yaw 坐标系中表达，原点取骨盆 XY、Z 保留世界高度，再展平为 2160。

`human_heading` 使用 `heading_from_goal`：五点配置以人体骨盆作为 heading 源，
相对**实测机器人 torso**计算，补偿训练的 SMPL rest-frame 变换：
`inv(R_robot_torso) @ R_human_pelvis @ Rx(-pi/2) @ Rz(-pi/2)`，逐帧 rot6d 展平为 60。
不要改成人体朝向自身、相对机器人 pelvis 或仅 yaw 差。

图内先归一化，再把除第 0 帧、五点之外的特征清零，并拼入固定有效位。
调用方仍须提供完整有限值数组，可在原始位姿阶段重复当前帧到十帧，未跟踪槽使用有限占位。
五个必需点或 heading 无效时应由执行器暂停该接口；此图不能动态关闭缺失点。
该版本不消费 WAM 的完整未来十帧人体预览；需要十帧输入时应重新选择掩码并导出、复评。

## Global + Ref. 实时反馈：必须实现

训练时两个接口都使用 `model/ref_feedback.py::global_ref_feedback`。
它比较当前实测机器人 FK 与**按当前机型重定向的机器人参考目标**，不直接比较人体骨长。
Human H1 也需要这一路机器人参考，只有人体五点和本体历史不足以运行这一版模型。

- 时刻：当前物理参考帧的 `[0,20,40,60] ms`；原生 motion command 通常指向下一步，
  因此用 `feedback_frame` 做减一帧、片头钳制。不能用下一帧替换当前帧。
- 每帧四末端顺序：左踝、右踝、左腕、右腕。
- 每末端 9D：目标位置减当前实测位置，旋转到**当前完整骨盆坐标系**（3D）；
  `inv(R_actual_effector) @ R_target_effector` 的 rot6d（6D）。
- 前 144D 为 `[4帧,4末端,9误差]` 展平；最后 16D 为 `[4帧,4末端]` 有效位，0/1。
  当前四个有效位位于索引 `144:148`。所有未来帧都与同一个当前实测 FK 比较。
- 参考与实测 FK 位姿在同一回合的固定世界坐标系。世界位置需要一致的状态估计与参考对齐；
  旧模型的“无需全局机器人 XYZ”说明不能直接套用到本版反馈链路。
- 超出片尾、目标或实测位姿无效时，对应误差置零、有效位清零，不可伪造有效参考。
  当前四末端任一无效时，模型保留训练行为：输出 NaN。执行器必须检查有限值并进入既定停机/保持流程。

完整契约保存在 `metadata.json.ref_feedback_contract`。导出优化没有移除该守卫。

实际 FK 使用各机型的原生 body frame；契约中的语义顺序保持一致：

| 机型 | 骨盆 / anchor | 左右踝 | 左右腕 |
|---|---|---|---|
| Q1/G1 | `pelvis` / `torso_link` | `left/right_ankle_roll_link` | `left/right_wrist_yaw_link` |
| Bumi | `base_link` / `waist_yaw_link` | `l/r_ankle_roll_link` | `l/r_hand_frame` |

Bumi 的 hand frame 是各自 `elbow_pitch_link` 下局部位置 `[0,0,-0.18702218] m` 的刚性子帧，
不能直接用肘关节原点替代。`source-reference/robots/` 保存了本次冻结的三机型映射供核对。

## 原生动作与控制

输出不是电机角度或力矩。按照元数据，针对每个 `perm[c]>=0` 的 canonical 槽计算：

```python
n = perm[c]
native_target[n] = native_default[n] + sign[c] * action[c] * native_action_scale[n]
```

`perm[c]==-1` 必须跳过；其余原生关节保持默认位。PD 增益、力矩上限和动作缩放取
`native_control`，关节名称取 `native_joint_order`。Q1 腰关节排列、Bumi 肘关节符号和
Bumi MJCF/SDK 顺序差异均应按名称映射。图内没有电机安全限幅、速度限制、延迟补偿或急停。
保持执行器已有的校准、限位、slew、姿态保护与故障处理，不能仅凭 ONNX 动作直接下发驱动。

## 离线复核与最小推理示例

只运行现成 ONNX 需要 NumPy、ONNX Runtime，无需 PyTorch、MuJoCo 或完整训练环境。

```bash
cd /data/kk/artifact/mimic-12000-onnx-20261121/q1
sha256sum -c SHA256SUMS
python export_mimic_deploy.py --check . --benchmark 1000
```

```python
from pathlib import Path
import numpy as np
import onnxruntime as ort

bundle = Path('/data/kk/artifact/mimic-12000-onnx-20261121/q1')
opts = ort.SessionOptions()
opts.intra_op_num_threads = opts.inter_op_num_threads = 1
runtime = ort.InferenceSession(str(bundle / 'motion.onnx'), opts,
                               providers=['CPUExecutionProvider'])
with np.load(bundle / 'verification/inputs.npz', allow_pickle=False) as samples:
    feed = {x.name: samples[x.name][0:1] for x in runtime.get_inputs()}
    action = runtime.run(None, feed)[0]
    np.testing.assert_allclose(action, samples['motion_expected'][0:1], atol=1e-4, rtol=0)
assert action.shape == (1, 29) and np.isfinite(action).all()
```

上例只读验证输入，不连接硬件。实际执行前需用实时历史、目标、heading 和反馈替换 feed。
每次先检查输入有限、shape/dtype、有效位，再检查输出有限。切换接口时延续同一物理历史和上一动作。

## 优化、数值验证与边界

两个接口分别导出等价单分支图，自动不包含另一编码器的依赖；未先生成一张共享双输出图。
保持原生 float32 推理与归一化，在 ONNX 上做静态常量链计算、四层 FiLM 仿射折叠、
decoder 与 reference_head 首层末尾 154D morph 拼接折叠，再按目标输出提取依赖并检查 ONNX。
指定 Gemm 的转置/缩放、单消费者、常量与拼接布局不匹配时直接报错。

每图 260 组：四条配对 val 轨迹的运动学输入（真实参考历史，previous_action=0，无物理闭环），
加 256 组原始特征高斯扰动，std=0.2，seed=0，有效位保持二值。
同时比较原始 ONNX、优化 ONNX、checkpoint float32 PyTorch，最大绝对误差阈值 `1e-4`；
四组基础 Human 输入还与原生 `act_human` 的原始人体特征路径复核。
检查有限输出、形状、缺失关节精确零及无效当前参考触发 NaN。
来源哈希在导出结束时复查；不同机型整包分别发布，根级清单在全部完成后生成。

默认 ORT 图优化、CPUExecutionProvider、单线程，保留 ORT 加载优化。
未做 INT8、FP16、权重剪枝、重训练、CUDA 或 TensorRT；归一化未额外手工折叠进线性层。
`validation-summary.json` 和各 CPU 报告记录本次实测误差、图大小、节点数和耗时。
耗时为本主机热身后同步 `session.run`，不含传感、FK、调度、传输与电机执行，
超过 20ms 次数仅是单次推理耗时统计，不是周期调度超时统计；未做 ORT profiling。

既有 val 报告属于原始 PyTorch checkpoint 的闭环评估，交付时按 SHA256 核对到同一权重。
此次没有重跑 ONNX 闭环、sim2sim 或实机评测；数值等价和主机耗时不能替代硬件部署验收。

## 重新导出

仓库入口：`/data/kk/wammimic/scripts/export_mimic_deploy.py`，运行命令见同目录
`mimic.md` 的部署导出部分。交付的 `reproduce.sh` 固定了本次基础镜像 SHA256，
从冻结源码、只读资产和数据重建新目录，checkpoint 只读，派生输出写入新目录。
原源码目录若已移动，需要传入新的 `--source`；完整训练依赖与机器人资产仍需可用。
源文件哈希、checkpoint 哈希、软件版本和硬件信息都可从交付元数据与报告追溯。

## 本次导出实测（2026-11-21）

主机 AMD EPYC 7K62，ORT 1.23.2 CPU 单线程；每图 260 组等价验证、1000 次耗时采样。

| 机型 | 模型 | 三方比较最大绝对误差 | 节点数（原→优化） | 大小 MiB（原→优化） | 优化 P50 / P95 / P99 ms | P50 加速比 | >20ms |
|---|---|---:|---:|---:|---:|---:|---:|
| q1 | motion | 6.91e-06 | 234→70 | 41.97→36.06 | 2.345 / 2.552 / 2.860 | 1.054× | 0 |
| q1 | human_h1 | 1.05e-05 | 303→84 | 54.37→48.45 | 3.117 / 3.355 / 3.957 | 1.023× | 0 |
| bumi | motion | 1.86e-05 | 234→70 | 41.97→36.06 | 2.366 / 2.799 / 3.043 | 1.035× | 0 |
| bumi | human_h1 | 2.57e-05 | 303→84 | 54.37→48.45 | 3.058 / 3.439 / 3.986 | 1.029× | 0 |
| g1 | motion | 8.11e-06 | 234→70 | 41.97→36.06 | 2.365 / 2.632 / 2.906 | 1.043× | 0 |
| g1 | human_h1 | 1.5e-05 | 303→84 | 54.37→48.45 | 3.135 / 3.496 / 3.888 | 1.038× | 0 |

六张图均通过 ONNX 检查、数值阈值、缺失关节零输出及无效当前反馈拒绝检查；四项相关回归测试通过。主机 P50 改善幅度较小，不能据此推断 Q1 板端或完整控制周期速度。
