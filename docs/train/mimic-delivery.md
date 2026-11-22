# Mimic 12000 steps：模型与当前完整动作库交付指南

交付目录：`/data/kk/artifact/mimic-12000-deploy-20261121/`。
按照 `/data/kk/UFO/deliverables/blinkmorph_20000` 的动作交付结构整理，模型固定为本项目
`mimic-flip-12000-20261119` 的 12000 steps EMA。原始 ONNX 包与旧 20000 交付目录均保留。

## 1. 文件结构

```text
model/                         已验证的 12000 steps ONNX 原始交付包
  q1/, bumi/, g1/               每机型 motion.onnx、human_h1.onnx、metadata.json
  部署指南.md                   详细特征公式、关节映射和数值/CPU 报告
references/
  q1/, bumi/, g1/               每机型 16 条代表机器人参考，共 48 条
  human50/{q1,bumi,g1}/         对应代表动作的真实人体 24 点，共 42 条
  five_point/{q1,bumi,g1}/      对应代表动作的真实人体五点，共 42 条
  catalog.json                 代表动作目录、哈希、fps、帧数、train/val
  recovery_windows.json        三机型各躺/坐/半蹲入口，共 9 个窗口
  full-preview-index.json      当前完整动作库的可迁移相对路径与哈希
motion-library.tar             完整当前机器人、人体及五点数据，真实载荷，无外部软链接
validation/
  library-audit.json           本次全量归档校验及数量记录
  standing-review.json         中立站姿来源
  windows.json                 当前全部起身窗口
  flip-windows-12000.json      本轮训练的 flip 专项窗口
licenses/                      保留原模型和数据的归属、许可证与使用条款
package_mimic_deploy.py        可复现打包、独立校验脚本
SHA256SUMS_ALL, manifest.json  整包哈希及来源身份
```

代表动作沿用旧包的选择，但逐文件从**当前数据根目录**读取并对照当前 manifest 校验。
每台机器人包含八极拳、LAFAN1 舞蹈、SEED 舞蹈、flip、三类原地起身及中立站姿来源。
它们是参考轨迹，不是实机录制的策略输出，也不是按本模型逐条筛选的实机合格列表。

## 2. 当前完整动作库

完整归档包含以下两个目录及各自 manifest、train/val 划分和来源记录：

```text
inplace-recovery-neutral-20261117-v2/
inplace-recovery-five-point-20261117-v2/
```

总计 **34,621 条机器人动作、33,244 条人体 24 点侧车、33,244 条人体五点**，
合计 **101,109 个索引载荷**，均为 50 Hz。
其中 1,377 条 MotionDecode 动作没有真实人体配对，只支持 motion 参考，未伪造人体五点。
每个机器人/动作是一条独立记录，数量包含 train 和 val；不是独立动作名称去重后的数量。

该版 native manifest SHA256 为：

```text
8e77139fa7ddfef4cb2bb7fb9977d1f0fdcad30c94778ecc16c919969253547e
```

它与当前 12000 steps 训练冻结记录及旧 `blinkmorph_20000` 动作目录一致。
本次重新生成完整归档，未依赖旧交付中已缺失的 `blinkmorph_20000_preview_library.tar`。
`library-audit.json` 记录实际全量载荷数、归档大小和哈希；根级 SHA 同时保护归档本身。

## 3. 校验与解包

校验脚本需要 Python 3.11+ 与 NumPy；仅核对模型/数据文件时不依赖训练框架。
在交付包根目录执行：

```bash
sha256sum -c SHA256SUMS_ALL
python package_mimic_deploy.py --check . --full-archive
mkdir -p /path/to/mimic-library
tar -xf motion-library.tar -C /path/to/mimic-library
```

第一步核对整包所有文件，第二步额外逐个核对 TAR 中 101,109 个索引载荷及代表动作/起身窗口；
两步都会顺序读取大归档。确保解包位置具有足够空间，解包数据不应写回原始训练数据根目录。

源 manifest 和 subset YAML 的绝对路径保持原始字节，供训练来源追溯。
换机器后用 `references/full-preview-index.json` 的相对路径定位，不要假定原始绝对路径可用。
例如读取第一条完整参考及配对五点：

```python
import hashlib
import json
from pathlib import Path
import numpy as np

bundle = Path('/data/kk/artifact/mimic-12000-deploy-20261121')
library = Path('/path/to/mimic-library')
index = json.loads((bundle / 'references/full-preview-index.json').read_text())
row = next(r for r in index['clips'] if r['robot'] == 'q1' and 'five_file' in r)
reference = library / row['reference_file']
with reference.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == row['reference_sha256']
with np.load(reference, allow_pickle=False) as ref:
    assert int(ref['fps']) == 50
    joint_names, joint_pos = ref['joint_names'], ref['joint_pos']
with np.load(library / row['five_file'], allow_pickle=False) as human:
    human_pos_w, human_quat_w = human['human_pos_w'], human['human_quat_w']
    time_s = human['time_s']
```

## 4. 与 12000 steps ONNX 配对使用

本包模型位于 `model/{q1,bumi,g1}`；只使用对应机型的模型、元数据和参考轨迹。
详细接口见 `model/部署指南.md`，其中原 ONNX 包示例路径可替换为本包 `model/` 路径。

| 入口 | 动态输入 | 输出 |
|---|---|---|
| motion | `obs_history[1,930]`、`goal_motion[1,670]`、`ref_feedback[1,160]` | `action_motion[1,29]` |
| human_h1 | `obs_history[1,930]`、`goal_human[1,2160]`、`human_heading[1,60]`、`ref_feedback[1,160]` | `action_human[1,29]` |

NPZ 中 `joint_pos/joint_vel` 为原生关节顺序，按名称和模型元数据转 canonical。
`body_pos_w/body_quat_w` 在世界坐标系中，四元数为 wxyz，位置为米、关节角为弧度。
运动参考取当前帧到未来第 9 帧，片尾钳制；`goal_motion` 仍需按模型源码构造，不能直接展平 NPZ。
人体五点顺序为骨盆、左踝、右踝、左腕、右腕，SMPL 槽 `[0,7,8,20,21]`。
五点 NPZ 也不是 `goal_human[2160]`，需先构造对应 24 点槽、heading 和特征。

**两个入口都必须提供实时 Global + Ref. 反馈**：当前实测机器人 FK 对比同机型重定向目标，
使用 0/20/40/60ms 的四帧、左踝/右踝/左腕/右腕的误差及有效位。
Human H1 的人体目标只启用当前一帧，但仍需要这一路机器人参考。
不能把 20000 包的旧无反馈 ONNX 调用方法直接套用到这六张模型。

模型独立数值复核（需要 NumPy 和 ONNX Runtime）：

```bash
python model/export_mimic_deploy.py --check model/q1
python model/export_mimic_deploy.py --check model/bumi
python model/export_mimic_deploy.py --check model/g1
```

9 个代表起身窗口提供 `start_frame/hold_frame`。起身后继续策略和物理更新，参考钳制在 hold_frame；
静立目标与经审核的 `walk2_subject4` 第 11446 帧一致。数据中的保持姿态不是机器人实际站稳结果。

## 5. 复现本次整理

从 `/data/kk/wammimic` 根目录，用一个尚不存在的新输出目录运行：

```bash
python scripts/package_mimic_deploy.py \
  --models /data/kk/artifact/mimic-12000-onnx-20261121 \
  --template /data/kk/UFO/deliverables/blinkmorph_20000 \
  --data-root /data/kk/blinkmorphmimic/prepared/inplace-recovery-neutral-20261117-v2 \
  --five-root /data/kk/blinkmorphmimic/prepared/inplace-recovery-five-point-20261117-v2 \
  --run /data/kk/wammimic/runs/mimic-flip-12000-20261119 \
  --guide docs/train/mimic-delivery.md \
  --output runs/new-mimic-full-delivery
```

打包会检查模型哈希与当前 checkpoint、训练数据身份、全部归档载荷 SHA、代表数据格式/有限值、
原生关节顺序与 9 个起身保持目标。所有来源保持原样，输出目录必须不存在，manifest 最后生成。
打包程序不会重新训练、重新导出 ONNX 或改写原始动作。模型的 260 组/图数值报告保留在 `model/`。

本次全量归档检查验证文件身份；未重跑全动作 ONNX 闭环或实机验收。
保留各数据来源许可证与非商业等使用条款，详见 `licenses/`；模型许可不替代动作数据许可。
