# Nymeria WAM 数据处理

Nymeria 使用独立的 `xsens_inertial_mocap_reference` 来源协议，数据集名为 `nymeria_ego`。它提供真人惯导骨架参考，不能当作 NOKOV 光学真值、机器人关节轨迹或末端执行器标签。

## 自动管线

设置 `RAW_DATA_ROOT`、`PROCESSED_DATA_ROOT`、`PRETRAINED_ROOT`、`EXISTING_RAW_DATA_ROOT`、`P1_CONTROL_ROOT` 后运行：

```bash
python -m scripts.run_p1_pipeline nymeria --gpu 3
```

原始文件、模型、控制文件均只读挂载，处理容器禁网。来源相对子路径在 `conf/data/pipeline.yaml` 注册；控制目录包含私有下载清单、选择记录和已安装的 `nymeria-runtime/site`。本轮使用官方 `projectaria-tools==2.4.0` CPython 3.12 wheel，下载的 12,361,075 bytes 计入已有 1 GB 软件/元数据预留。后处理和断点续跑不新增网络下载。

顺序为 `nymeria → cache-nymeria-text → cache-nymeria-video → validate-nymeria`。各阶段日志及状态写入控制目录。每条录制保存输出哈希和收据；重跑复核来源与输出后复用已完成工作，来源改变则拒绝覆盖。

## 输入与转换

每条录制使用完整头部 RGB VRS、`body_motion.zip` 中的 `body/xdata.npz`、官方 activity summarization CSV 和元数据。先对全部下载文件核验官方大小与 SHA-1，再提取 NPZ，保留原始文件。

- 240 Hz XSens 骨架直接读取原始时间戳和世界系位置、四元数，提取骨盆、左右脚、左右手。不补齐四元数，不修复时钟，不插值缺失测量。
- RGB 的设备时钟通过 VRS 记录的同步关系转换成 TIME_CODE，与骨架统一；叙述的设备时间也使用同一映射。
- 建立 10 Hz 时间网格；取网格之前最近的骨架和 RGB。骨架采样年龄上限约 5.17 ms，RGB 约 68.67 ms；这是离散采样年龄门槛，不能解释为硬件同步误差保证。
- RGB 顺时针旋转 90° 后缩放为 224×224，保留鱼眼画面，不镜像、不去畸变。
- 叙述交界处重叠的时间段标为歧义；任何窗口跨越歧义、标注边界、未标注区、时间间隔异常或骨架突跳时均排除。检查覆盖整个 240 Hz 测量区间，而非只检查抽样帧。
- `action[10,45]` 表示未来 0.1–1.0 秒五点位置和 6D 旋转；`observation.state[4,75]` 表示过去四帧五点位姿、线速度和角速度。状态按 10 Hz 网格选取，速度来自紧邻过去的 240 Hz 实测。两者统一到 t0 完整 3D 骨盆坐标系；下肢位置和线速度按该参与者腿长缩放，手的位置保留米制。
- 每个窗口的 17 帧 RGB 经固定版本 Cosmos-Predict2.5 VAE 生成 `video_latents[16,5,28,28]`，仅前两帧潜码作为条件；实际编码检查未来画面不会污染条件潜码。官方叙述通过固定版本 Cosmos-Reason1 生成文本潜码，属于活动叙述监督，不伪造操作指令。

三名参与者各占一个 train/val/test 分区，不跨区。归一化统计仅由 train 的有效窗口计算。本轮仅三条录制，录制、任务和参与者互相混杂，不能据此证明跨人泛化。

## 输出和验证

输出根目录中的 `wam_lerobot_v2/nymeria_ego/` 包含 Parquet、RGB MP4、原始抽样标签、元数据和视频潜码；文本潜码位于根目录。训练按以下契约取数：

```python
from wammimic.data_pipeline.datasets import NymeriaWAMDataset
from torch.utils.data import DataLoader

reader = NymeriaWAMDataset(processed_root, split="train")
batch = next(iter(DataLoader(reader, batch_size=2)))
# video_latents: [B,16,5,28,28]
# text_embeddings: [B,512,100352]
# actions: [B,10,45], proprio: [B,4,75]
```

`HumanWAMDataset` 仍只接受光学来源；Nymeria 不自动加入光学训练协议。当前 WAM 主干与动作头可以使用 RGB/text/action 契约，额外历史状态需模型显式接入。本轮不训练或更新任何已有模型。

最终 `qa_reports/nymeria_wam_training_ready.json` 只有在来源/派生物哈希、全量标签重算、坐标逆变换、潜码形状与有限性、文本映射、参与者划分、train 统计、三个分区的批量读取和现有 WAM RGB/text 基线有限梯度检查全部通过后才写入 `ready=true`。这项梯度检查不执行优化器更新。

自动导出 20 个可复现随机窗口的 Rerun 回放（RGB、五点坐标轴、手轨迹）和联系表，位于 `qa_previews/`。真人来源没有校准的机器人网格或地面，不伪造这些可视化要素。该 QA 不代表机器人闭环或完整 P1 门禁通过。

## 本轮验证结果（2026-11-21）

三条录制、三名佩戴者，共 25,492 个配对网格点，23,245 个有效且已缓存窗口；train/val/test 分别为 8,320 / 8,590 / 6,335。70 条不同的官方活动叙述均已生成文本潜码。当前子集约 42.49 分钟配对数据，尚未达到计划的 50–100 小时目标。

自动四阶段管线已完成，`qa_reports/nymeria_wam_training_ready.json` 为 `ready=true`。全量重算与逆变换的最大位置误差为 `1.22818e-7 m`，最大旋转误差为 `4.36058e-6°`；这是特征转换一致性，不是惯导参考的物理精度。有效窗口 t0 的最大 RGB 采样年龄为 `66.665 ms`，骨架为 `4.200 ms`，两者均为因果取样，不能视为硬件同步误差。

额外使用现有 5,000 步 Action Expert 检查点和真实 Cosmos-Predict2.5-2B 主干，对三个分区分别完成前向、未来潜码替换不改变条件特征、以及有限梯度检查。结果在 `qa_reports/nymeria_cosmos_wam_smoke.json`，优化器更新次数为 0；此项不评估任务成功率或预测精度。

20 个 RGB 样本及对应的五点坐标轴/轨迹联系表已查看，抽检未见明显朝向或左右点颠倒。证据为 `qa_previews/` 和 `qa_reports/nymeria_rgb_visual_review.json`；不将此称为校准机器人网格、接触或闭环验收。官方原始 metadata 与选择清单、处理结果的佩戴者身份已逐条核对，证据在 `qa_reports/nymeria_participant_source_check.json`。

53 项回归测试通过。中断续跑已实际复核三条导入收据并复用文本缓存。预算收据确认本轮预留 99,927,923,367 bytes ≤ 100,000,000,000 bytes，SDK 计入已有软件预留。未提交 Git，未启动新训练。
