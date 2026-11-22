# Mimic 训练数据使用手册

后续 Mimic 训练使用 **`mimic-official-recovery-training-20261122/data`**。该版在双来源五点库上追加 Q1 官方起身，共 34,622 条 50 Hz 原生机器人参考，保留原先 MotionDecode 数据及全部原始划分。历史数据和已有冻结训练保持不变。训练从数据根目录取数，容器内保留相同绝对路径并只读挂载。

```text
/data/kk/wammimic/runs/mimic-official-recovery-training-20261122/data
```

## 取数路径与划分

相对于上述数据根目录读取以下文件。

| 路径 | 用途 |
| --- | --- |
| `manifest.json` | 全量清单。用 `(robot, name)` 定位轨迹，从 `output` 读取原生机器人 NPZ 的绝对路径。 |
| `{q1,bumi,g1}/train.yaml`、`{q1,bumi,g1}/val.yaml` | 原生训练器和评估器使用的机型划分。沿用清单中的顺序和划分。 |
| `contact-floor.json` | 与该版清单绑定的地面校准记录，交给原生训练器。 |
| `human50/` | 同步人体参考。按清单的 `human_available` 和 `human_sidecar` 取数。 |
| `standing-review.json`、`in-place-validation.json` | 中立站姿基准、来源与参考轨迹校验记录。 |
| `windows.json` | 三类原地起身的重置区间、站稳区间和固定回归用例。 |

| 机型 | 原生关节数 | Train 轨迹 | Val 轨迹 |
| --- | ---: | ---: | ---: |
| `q1` | 22 | 10,394 | 1,139 |
| `bumi` | 21 | 10,400 | 1,139 |
| `g1` | 29 | 10,411 | 1,139 |

有配对人体参考的轨迹可用于人体五点接口。离线五点文件在同级 `/data/kk/blinkmorphmimic/prepared/inplace-recovery-five-point-20261117-v2`，通过其 `manifest.json` 的 `clips[].output` 相对路径读取 `human_pos_w[T,5,3]`、`human_quat_w[T,5,4]` 与 `time_s[T]`。上述旧版数据中的 MotionDecode 只用于机器人动作接口。设计 v1.9 的新数据版本通过具名机器人 FK 位姿生成 MotionDecode 机器人参考五点，使用独立来源标记加入五点训练。WAM 预测接入仍需按设计做人体到机器人尺度、轴向和时间对齐。 [WAM 取数手册](wam.md)

双来源五点使用 `scripts/prepare_mimic_robot_five.py --parent <旧数据根> --output <新数据根>` 处理。当前输出为 `runs/mimic-dual-five-20261121/data`。原始人体保存在 `human50`，MotionDecode 机器人参考保存在 `robot_reference50`，默认训练混合入口为 `keypoints50`。清单的 `human_available` 保留真实人体来源含义，新增 `five_point_source` 等字段绑定五点来源及哈希。`scripts/run_mimic.py prepare --data-root <新数据根> --run <运行目录>` 自动选择混合入口和十帧预览。`scripts/evaluate_mimic.py --five-point-source original_human|robot_reference|mixed` 按来源选择完整留出集并使用 checkpoint 配置的预览长度。数据转换和坐标验收详见 [设计 §3.6](../design/design.md#36-原始人体与机器人参考五点的共同数据契约)。

本版已生成每机型 MotionDecode 训练 396、验证 63 份参考五点，三机型合计 1,377 份。`five-point-validation.json` 保存全量哈希、来源、划分及旋转往返校验，`config.json` 与 `freeze.json` 绑定下一轮训练入口。实际新训练尚未启动。Preview 的 `12,000 · 机器人参考五点 · H1 兼容诊断` 使用旧 EMA 检查新输入，原始人体与 PICO 入口继续可选。新增运行代码相对 PICO Preview 的改动保存在 `patches/blinkmorphmimic-dual-five.patch`。

## 中立站姿与起始朝向

中立位是**双脚着地、双腿站立、身体直立、双臂自然下垂，双手位于大腿两侧**。三台机器人都取 LAFAN1 `walk2_subject4` 在 50 Hz 重采样后的第 `11446` 帧（索引从 0 开始，228.92 秒），各用自己的原生关节角和根高度。 [中立站姿正侧视图](/data/kk/blinkmorphmimic/prepared/inplace-recovery-neutral-20261117-v2/lafan-E-robots.png)

- 除新增官方起身外，其余参考轨迹的第 `0` 帧根节点水平位置都是 `(0, 0)`，机器人正前方统一为世界 **`+X`、yaw = 0**。起始中立位静止至少 50 帧（1 秒），区间见 `neutral_contract.entry_hold_end_frame`。同一机型使用同一个站姿，跨机型共享朝向。官方起身保留官方躺姿和朝向，直接从第 0 帧开始。
- 要求仿真的普通回合也严格同向时，给各环境分配轨迹并用 `sampling="start"`，从轨迹第 `0` 帧初始化。关闭重置时的根位姿、速度和关节扰动，以及失败回跳。现有无扰动评估环境可用 `morphmimic_env_cfg(..., eval=True)`，经 `MultiMotionCommand.assign_clips` 分配轨迹后重置。训练采样若选到中间帧，重置姿态取该帧。
- 普通移动轨迹在动作终点恢复同机型的中立姿态，保留终点的世界位置和朝向。原地起身轨迹回到第 `0` 帧的完整中立位，含根位置、朝向和关节姿态。坐姿、躺姿、半蹲的专项重置保留选中低姿态帧的完整根四元数和关节状态。对比起身方向时关闭随机 yaw，统一以世界 `+X` 的站立目标为基准。

## 加载方式

原生物理训练在 `/data/kk/blinkmorphmimic` 的训练环境中运行。常规训练入口 `python -m scripts.train_optimized` 的 `--data-root` 指向本版数据根目录，入口会配置三个机型的 `train.yaml`、`manifest.json`、`human50/` 和 `contact-floor.json`。按实际设备填写 `--gpu-ids`，先使用 `--dry-run` 核对展开的配置，再按原生训练说明冻结数据和运行配置。 [BlinkMorphMimic 启动说明](/data/kk/blinkmorphmimic/README.md:23)

### Global + Ref. 10,000 步训练与可视化

运行目录为 `/data/kk/wammimic/runs/mimic-global-ref-10000-20261119`。`scripts/run_mimic.py` 用原生优化配置从随机权重训练 Q1、Bumi、G1，每卡 2,048 个环境。一个 step 对应原生训练器的一次迭代。起身窗口占重置样本的 20%，在三类姿态间平衡取样。`config.json` 和 `freeze.json` 记录运行参数及数据、源码、镜像身份。源码可从上一轮运行的 `source/` 快照复制，在副本应用 `patch -p1 < patches/blinkmorphmimic-global-ref.patch`，再把仓库的 `scripts/run_mimic.py` 复制到副本的 `scripts/`。冻结前运行 `python -m scripts.check_ref_feedback` 检查平移、旋转、有效位和 Actor 动作梯度。

按 [设计 §3.5](../design/design.md#35-共享-actor-decoder-的-global--ref-腕踝反馈)，共享 Actor Decoder 每步拼入 **144 维腕踝位姿误差和 16 维有效位**。左踝、右踝、左腕、右腕各使用 0、20、40、60 ms 的固定世界坐标目标。原生动作命令指向下一次物理步，腕踝反馈以当前物理帧为 0 ms。物理采样、留出评估和 Preview 使用当前仿真 FK，可微展开使用当前预测状态。所有入口调用 `global_ref_feedback`。训练和推理沿用原生 Motion 与 Human 五点编码器。Human 五点从人体 sidecar 生成潜码，两种入口的物理反馈目标都取同一条按机型重定向的机器人 NPZ 轨迹。该运行覆盖原生 Mimic，WAM `465D` 回灌及独立人体参考标定按设计另接入。

每 1,000 次迭代保存一次 checkpoint。训练结束时每台机器人保留 `ckpt_7000.pt`、`ckpt_8000.pt`、`ckpt_9000.pt`、`ckpt_10000.pt`。`latest.pt` 是最新 checkpoint 的硬链接。

- 新训练的回报、损失、吞吐、留出跟踪和起身结果发布到 `global-ref/`，按机型查看。`global-ref/baseline/` 显示旧权重的配对复评。公网服务只挂载 TensorBoard 事件文件。 [公网 TensorBoard](http://1.94.0.106:31776/)
- 通过最终权重的反馈验收及 Preview 冒烟检查后，公网 Preview 切换到本次第 10,000 步权重。支持单本体参考叠加、三本体同屏、两种编码入口及坐姿、躺姿和半蹲重置。浏览器共享同一个仿真会话。 [公网评估 Preview](http://1.94.0.106:31777/)

`scripts/finish_mimic.sh` 在训练容器成功退出后运行评估和 Preview 检查。`evaluation/` 保存各机型的 `val.yaml` 跟踪结果和固定起身回归。全局指标在参考与仿真的共同起始坐标系中计算，逐回合测量骨盆 XY、腕踝世界位置与相对转角误差，以及 10 cm、1 rad、25 cm 门限下的完整轨迹成功率。人体五点评估使用有真人配对的片段。起身成功要求实际物理状态连续站稳 3 秒。若传入上一轮运行作为基线，`evaluation/baseline/` 用同一留出片段和评分逻辑复评旧权重。`feedback-verification.json` 记录固定参考的平移检验、已训练 Actor 的动作变化以及评估与 Preview 的逐步一致性。

### Flip 专项续训与 WebRTC Preview

续训目录为 `/data/kk/wammimic/runs/mimic-flip-12000-20261119`。从上一轮 `ckpt_10000.pt` 恢复共享策略、世界模型、优化器和 EMA，保留原学习率调度，训练至第 12,000 步。将 `patches/blinkmorphmimic-flip-continuation.patch` 应用到 10,000 步源码快照的副本，再复制 `scripts/run_mimic.py`。用 `scripts.run_mimic prepare --parent-run <父运行目录>` 生成冻结配置，用 `scripts.run_mimic train --resume <父运行目录>/q1/ckpt_10000.pt` 启动三机型续训。两个命令都需要 `--data-root` 和 `--run`。`patches/blinkmorphmimic-flip-continuation-tests.patch` 更新旧版回归用例，使其检查重要性采样与两种专项窗口的叠加。

- 重置分布按期望分配为 **72% 常规重要性采样、8% flip 专项、20% 起身窗口**。常规采样沿用按数据来源平衡的基础分布，以及起始区间的在线终止率和跟踪误差。Flip 专项在每机型四条训练参考之间均匀选取，再从其冻结窗口抽帧。
- `flip-windows.json` 的起跳代理为参考根节点达到高度峰值前的最快上升帧。专项窗口从该帧前 0.5 秒到参考高度峰值后约 0.3 秒，限制在原始动作段内。两条留出 flip 不参与窗口生成。
- `scripts/evaluate_mimic_flips.py` 用同两条留出轨迹复评 10,000 与 12,000 步。`evaluation/flips/` 保存整段跟踪结果、原生提前终止项以及逐控制步遥测。离地高度取双脚足底碰撞几何最低点到地面的距离。在整机无地面接触且足底净空大于 2 mm 的帧记录离地，另报连续离地时长和骨盆最大上升量。
- 公网 Preview 使用 BlinkMorphMimic 的原生 WebRTC `<video>` 和 H.264/NVENC 流。TCP 信令沿用 31777，媒体复用现有 UDP 31870–31889 中继。部署脚本 `scripts/launch_mimic_webrtc.sh` 通过 `blinkmorph-preview` 的容器网络接入 UDP 中继。若原容器被重建，需重新启动本 Preview 以加入它的新网络命名空间。 [公网 Preview](http://1.94.0.106:31777/)
- Human 五点模式在 WebRTC 画面中以橙色显示当前帧的骨盆、双腕和双踝，保留原始人体尺度。单机视图中绿色为机型参考，三机同屏中五点随各机型所在位置绘制。部署时将 `patches/blinkmorphmimic-preview-five.patch` 应用于续训源码的副本，再用 `MIMIC_PREVIEW_SOURCE` 指定该副本。
- PICO 上传复用 BlinkMorphMimic 的五点 JSONL 格式与坐标标定，支持 32 MB、最长 10 分钟，转换为 50 Hz 原始人体五点后用当前 Human 模型播放并叠加橙色点位。上传目录通过 `MIMIC_PICO_DATA_ROOT` 指定，录制独立于固定模型与训练数据。先应用五点补丁，再应用 `patches/blinkmorphmimic-preview-pico.patch`，通过 `MIMIC_PREVIEW_SOURCE` 指定副本。当前部署副本为本轮运行的 `preview-pico-source`，录制目录为 `pico-recordings`。PICO 的机器人参考沿用模板首帧，供初始化和监测，人体输入取自提交的五点录制。上游 `blinkmorph-preview` 被重建后，应重新创建本服务容器以加入新网络命名空间，并核对公网 TCP 转发目标。

直接读取参考轨迹时，按下例加载。NPZ 中的身体位置在世界坐标系下，四元数按 `wxyz` 存储。用 `root_body` 从 `body_names` 找到根节点，用 `joint_names` 映射仿真关节列。`joint_pos`、`joint_vel` 为 `[T,J]`，`body_pos_w`、`body_quat_w` 为 `[T,14,3]`、`[T,14,4]`。

```python
import json
from pathlib import Path

import numpy as np

root = Path("/data/kk/blinkmorphmimic/prepared/inplace-recovery-neutral-20261117-v2")
robot = "g1"
subset = json.loads((root / robot / "train.yaml").read_text())
manifest = json.loads((root / "manifest.json").read_text())
windows = json.loads((root / "windows.json").read_text())

# 当前版本的 YAML 划分文件采用 JSON 语法。
train_paths = {Path(subset["root"]) / name for name in subset["clips"]}
by_clip = {(row["robot"], row["name"]): row for row in manifest["clips"]}
case = next(row for row in windows["evaluation_cases"]
            if row["robot"] == robot and row["category"] == "seated")
row = by_clip[robot, case["clip"]]
path = Path(row["output"])
assert row["split"] == "train" and path in train_paths

start, stand, end = (case[k] for k in ("start_frame", "stand_frame", "hold_end_frame"))
with np.load(path, allow_pickle=False) as ref:
    root_idx = ref["body_names"].tolist().index(str(ref["root_body"]))
    assert int(ref["fps"]) == 50 and 0 <= start < stand < end <= row["frames"]
    root_pos = ref["body_pos_w"][start:end, root_idx]
    root_quat = ref["body_quat_w"][start:end, root_idx]
    joint_pos = ref["joint_pos"][start:end]
    joint_vel = ref["joint_vel"][start:end]
    assert end - stand == 150
print(path, root_pos.shape, root_quat.shape, joint_pos.shape, joint_vel.shape)
```

WAM-Mimic 仓库中的 `MimicDataset` 接收旧版 `morph_5pt_store` 的 NPY 布局。本版用 BlinkMorphMimic 原生 NPZ 加载器。后续如需接入该仓库的 `465D` 读取链路，按设计做格式适配，并沿用本版清单的轨迹划分、重置索引和朝向。

## 坐姿、躺姿、半蹲起身

| `windows.json` 的 `category` | Q1 | Bumi | G1 | 参考动作来源 |
| --- | ---: | ---: | ---: | --- |
| `seated` 坐姿起身 | 7 | 8 | 11 | 从骨架识别的坐姿开始，保留原始起身片段，接回统一站姿。 |
| `lying` 躺姿起身 | 14 | 17 | 18 | 原有片段接回统一站姿；Q1 新增一条官方起身，保留官方末姿。 |
| `half_squat` 半蹲起身 | 1 | 3 | 10 | 半蹲起点来自记录的姿态，到中立站姿的原地起身由运动学生成。 |

- **训练重置。** 从 `windows[]` 按机型和类别取窗口，在同一轨迹的 `[start_frame, stop_frame)` 内采样重置帧。帧号从 0 开始，`stop_frame` 是重置候选的上界，轨迹继续播放到 `hold_end_frame`。专项采样应按类别平衡，避免躺起数量压过半蹲。原生训练器支持在 `TrainCfg` 中设置 `recovery_windows=<数据根>/windows.json`、`recovery_windows_sha256=<该文件的 SHA256>`、`recovery_reset_share` 为 `(0, 1)` 内的比例，并使用 `importance_sampling=False` 的 `return` 重置采样。训练器会校验清单和各机型 `train.yaml` 与窗口绑定的哈希。
- **固定回归。** 从 `evaluation_cases[]` 取固定 `start_frame`，在无扰动评估环境用 `assign_clips(..., start_frames=...)` 指定片段内帧号后重置。预览可取 `preview_poses[category][robot]`。持续运行策略和物理，参考在 `[stand_frame, hold_end_frame)` 保持最终站姿 150 帧（3 秒）。从实际仿真状态判断是否连续站稳 3 秒，按机型和类别记录成功率与起身耗时。原有 88 条加 Q1 官方起身共 89 条，均来自 `train`，这些用例用于回归，独立泛化成绩应在留出的起身轨迹上统计。

`in-place-validation.json` 校验的是参考运动学。坐姿标注来自骨架几何，座椅接触、站立平衡和实际起身成功率需由闭环仿真验证。

### Q1 官方躺卧起身参考

2026-11-22 修正 Preview 的 Q1 单机「躺卧」入口。旧默认是 LAFAN1 的 `fallAndGetUp2_subject3__neutral_lying_016` 第 139 帧，髋约 -1.6 rad、膝约 2 rad，属于动作中途的折腿姿态。新入口采用 [PrimebotTech 的 q1_getUp.npz](https://github.com/PrimebotTech/primebot_rl_mjlab/blob/2153a6c561b3b755ae5bcca89683cb66d5ad5852/src/assets/motions/q1/q1_getUp.npz) 第 0 帧。原文件 SHA256 为 `133e79d897eef3d7a2988019245f75b460e5cfac2d7571b4447345aa71db4344`，50 Hz、350 帧、22 个原生关节。350 帧的原生 FK 与官方文件体位姿最大位置差低于 `1.6e-7 m`，关节角保持原值。

当前碰撞模型下原始参考最低穿地约 12.14 cm。`scripts/prepare_q1_official_recovery.py` 仅对每帧根与所有刚体作相同的竖直平移，留 2 mm 地面间隙，并同步修正竖直线速度。初帧抬高约 2.21 cm，最大抬高约 12.34 cm；末帧追加 150 帧目标保持。此适配不证明动力学可行性。机器人参考五点由同一机器人参考提取，沿用五点坐标转换，标为 `robot_reference`；没有伪造原始人体五点。

产物和验证记录在 `runs/mimic-q1-official-recovery-20261122/`。12,000 步权重未重新训练。现已通过 `scripts/append_q1_official_recovery.py` 将该动作追加到 `runs/mimic-official-recovery-training-20261122/data` 的 Q1 `train.yaml` 和混合五点入口，验证集保持不变。Preview 保留原始人体、MotionDecode、PICO 和其他机型入口；Q1 单机躺卧按钮按当前接口选择官方 Motion 或机器人五点 H1 参考。

### 40,000 步全动作训练

当前运行位于 `runs/mimic-all-motion-resume12000-40000-20261122`，从 `runs/mimic-flip-12000-20261119` 的三个 `ckpt_12000.pt` 恢复完整训练状态，继续到 40,000 步，实际新增 28,000 步。Q1、Bumi、G1 分别使用 GPU 0、2、3，每卡 2048 个环境，保留共享模型和 Global + Ref. 主路。最初的 4096 环境尝试在 Bumi 采样图预热申请 412876800 字节时触发 Warp CUDA OOM；2048环境下的上次零初始化尝试随后因参考重建只接受H1、Actor却配置H10而退出。两次失败日志均保留，零步检查点不用于此次续训。新重建损失支持Actor相同的预览长度，旧参考载荷和真实人体五点不变。Muon/Adam、EMA、归一化及失败率和跟踪误差统计均从12,000步父模型迁移。Actor EMA 系数 0.995。每 5000 次迭代保存检查点，各机型只保留最近两次，最终为 35,000 和 40,000 步。

本次取消起身、flip、probe 专项配额。重要性采样以逐动作均匀权重为基底，片段内均匀分配到 bin；30% 基底覆盖加 70% 由实际失败率和跟踪误差驱动的自适应采样，优先级封顶沿用 8 倍。`importance_source_balance=False` 避免只有一条的官方 Q1 数据集被分配整组权重。Bumi、G1 没有新增官方轨迹，现有起身来源保留，检查范围和证据见 `runs/mimic-official-recovery-training-20261122/official-source-audit.json`。

完整评估使用每机型 1139 条留出动作，分别运行 Motion 和混合来源五点，共 6834 回合，并按数据源分别统计 Survival、平均归一化回报及全局跟踪误差；另外跑训练集固定起身回归和 Q1 官方起身用例。起身回归沿用 checkpoint 中的五点预览长度，按真实仿真状态判断站稳。评估和 Preview 沿用同一份机器人参考、来源掩码和 Global + Ref. 反馈计算，完成验证后更新公网 WebRTC Preview。

### 每 5000 步的完整评估曲线

`scripts/watch_mimic_evaluations.py` 监视本轮三机型检查点，依次评估 15,000、20,000、25,000、30,000、35,000、40,000 步。使用余下的 GPU 1 串行评估，每个点均使用相同的 6,834 回合完整留出协议和 seed 0，并另外记录起身回归。训练仍在 GPU 0、2、3 上运行。

监视器先为原子写完的三个检查点建立临时硬链接，避免评估排队期间被“保留最近两次”的训练清理机制删除。完成后释放临时链接，只保存评估结果及权重SHA256。每步的 JSON、CSV 和失败动作表位于本轮运行的 `evaluation/checkpoints/<step>/`。汇总曲线保存为 `evaluation/checkpoint-curve.csv`、`.json`、`.png`，包含各机型及合并留出集的 Survival 和平均归一化回报。

TensorBoard 的 `resume12000-to40000/evaluation/{q1,bumi,g1,aggregate}` 按真实 checkpoint 步数持续追加评估点。`periodic-evaluation/status.json` 记录正在评估、排队、完成或失败的步骤；服务名为 `wammimic-mimic-periodic-evaluation.service`。

训练结束后，`wammimic-mimic-final-evaluation-40000.service` 使用已释放的 GPU 0、2、3 分别评估 Q1、Bumi、G1 的 40,000 步 EMA 检查点，沿用相同的 6,834 回合留出协议和固定起身回归。`final-evaluation-status.json` 记录最终评估状态；核对六份报告的 EMA 类型、训练清单、检查点 SHA256、动作数和物理诊断后，生成最终 `curve-point.json`，允许发布服务进行 WebRTC 检查和公网切换。GPU 1 的中间检查点评估继续运行，其后的曲线汇总会纳入已完成的 40,000 步结果；最终 Preview 发布不再等待较早检查点排队结束。

主结果按 **Motion / Human × Q1 / Bumi / G1** 六行展示，指标仅为 **整段完成率 Survival** 和 **平均归一化回报**。各检查点的 `evaluation/checkpoints/<step>/paper-metrics.md`、`.json` 随完成报告更新，未完成项标记“评估中”，不填零。公网 TensorBoard 的 Text 页签可查看 `evaluation/table` 下的同一张表。

上述正式曲线使用 **EMA 权重**，表格标题与 JSON 必须标注权重类型；非 EMA 当前策略的对照结果须分开保存，不能填入同一条 EMA 曲线。续训加载优化器状态后，仍会逐轮按当前 `lr_decay_iters` 重算 LR；延长训练步数时不得默认把衰减终点一起延长并据此声称 LR 连续。当前运行已观测到 12k 到 15k 的有效 LR 从 6e-6 抬高至约 1.40e-4，以及该点 EMA 与非 EMA 表现的明显差异；详见运行内 `evaluation/12000-vs15000-audit/report.md`。

Preview 的 `official_q1_recovery` 表示动作库包含 Q1 官方参考，仅在 **Q1 + 躺卧** 时走官方动作快捷入口。坐姿、半蹲和其他机型继续通过 `standup` 请求读取 `windows.json` 的原有 `preview_poses`，不能仅根据该标记把所有姿态都路由到 `official_q1_getup`。本轮前端修复通过 `preview-static/app.js` 只读覆盖部署，保留冻结训练源码和 40k EMA；公网转发服务为 `wammimic-mimic-webrtc-40000-half-squat.service`，验证凭据为运行内 `half-squat-fix-verification.json`。

Survival 为从首帧开始、未提前终止且到达末帧的动作数除以全部动作数。每动作归一化回报为 `Σr / [R_MAX × (L−1)]`，`R_MAX=0.1`，然后对动作等权取平均；提前终止仍用完整动作长度作分母。训练短窗口回报、世界模型想象存活率、起身回归成功率、全局跟踪阈值成功率均不替代这两个主指标。

**范围必须标注：** 当前队列使用各机型 1,139 条留出动作，Human 为原始人体五点和机器人参考五点混合来源，H10。这是本地留出评估。MorphMimic Table I 使用其共同训练动作库、五点加十帧预览；仅公式一致不代表已复现论文的动作集合或五点来源。全动作共同库评估与当前留出集须分别报告，禁止合并或沿用同一曲线名称。

两种接口各一次实际 Warp 策略仿真均运行 499 步、9.98 秒，以 `clip_end` 结束，无提前终止及已监测的物理溢出。Motion 末帧图像接近直立，五点 H1 末帧仍屈膝、躯干前倾。`probe-{motion,human}.json` 及初末帧 PNG 保存实际结果；本次没有测量连续三秒站稳，也未测多种子起身成功率或真实硬件表现。

## 12000 steps Global + Ref. 部署导出

使用 `scripts/export_mimic_deploy.py` 导出本轮 EMA，完整接口、构造特征、原生关节映射和复核方式见 [部署指南](mimic-deploy.md)。该入口适配保存的 160D Global + Ref. 契约，每机型输出固定 batch=1 的 `motion.onnx` 与五点单帧 `human_h1.onnx`。两者都保留动态 `ref_feedback[1,160]`。它们不能直接替换没有该输入的旧 Q1 模型。

在冻结源码所用的 Python 环境（含 `onnx`、`onnxruntime` 和机器人资产）中，从仓库根目录运行：

```bash
python scripts/export_mimic_deploy.py \
  --source runs/mimic-flip-12000-20261119/source \
  --checkpoint runs/mimic-flip-12000-20261119/q1/ckpt_12000.pt \
  --data-root /data/kk/blinkmorphmimic/prepared/inplace-recovery-neutral-20261117-v2 \
  --output runs/new-mimic-export/q1
```

输出目录必须不存在；Bumi/G1 分别改 checkpoint 与输出子目录。四层 FiLM 和两处 morph 拼接在严格布局检查后折叠，九项机器人常量及 Human H1 掩码固定。每张图与原始 ONNX、float32 PyTorch 比较 260 组输入，误差阈值 `1e-4`，并验证缺失关节动作精确为零、无效当前参考仍输出 NaN。验证集来自四条配对 val 轨迹的运动学特征和 256 组扰动，属于导出数值验证。两图通过后才发布完整目录，manifest 最后生成；SHA256 绑定图、元数据、验证输入、源码及 checkpoint，结束时复查来源。

独立复核只需要 NumPy 和 ONNX Runtime：

```bash
python scripts/export_mimic_deploy.py --check runs/new-mimic-export/q1 --benchmark 1000
```

`--benchmark` 报告单线程主机 CPU 的 P50/P95/P99、最大耗时、超过 20ms 次数和 P50 加速比，不包含传感、FK、网络传输与电机执行。未做 INT8、FP16、剪枝、重训练或 TensorRT；归一化常量未手工折叠进线性层。

### 模型与完整动作库一并交付

当前交付以 [精简部署包](mimic-compact.md) 为准：只保留八极拳、`dance1_subject1`、`dance1_subject2`、三类站起与静态中立位，每机型 7 条参考。

40,000 步模型更新见 [40000 steps 部署指南](mimic-deploy-40000.md)。导出器按 checkpoint 选择 H1/H10；本轮生成 `motion.onnx` 和 `human_h10.onnx`，继续使用动态 160D 实测参考反馈。旧动作载荷与原精简包一致，新增 Q1 官方起身与机器人参考五点仅按小型增量交付，不再重复打包动作库。

按 `blinkmorph_20000` 的代表动作、起身窗口和完整动作索引结构整理的 12000 steps 完整包见 [模型与动作交付指南](mimic-delivery.md)。`scripts/package_mimic_deploy.py` 将已验证 ONNX 与当前 34,621 条机器人参考、33,244 条人体侧车、33,244 条人体五点一并整理，导出真实数据归档并逐个核对载荷 SHA256。
