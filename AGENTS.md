# WAM-Mimic 项目指南

本指南适用于整个仓库。项目使用 Cosmos-Predict2.5-2B 从第一视角观测预测骨盆、左右踝和左右腕的未来位姿，并研究如何用这些参考驱动 Q1、Bumi 和 Unitree G1 的全身控制。Python 要求 3.11 或更高版本，训练和 Nymeria 解码的可选依赖分别在 `pyproject.toml` 的 `train` 和 `nymeria` 分组中。

## 目录与入口

| 路径 | 用途 |
| --- | --- |
| `conf/data/pipeline.yaml` | P1 数据源、相对子目录、清单和标签来源。沿用可由标准库 JSON 解析的 YAML 格式。 |
| `wammimic/data_pipeline/` | 人体动捕、OpenHLM、Nymeria、仿真与 Mimic 数据转换，以及坐标变换、缓存、Dataset 和验证。 |
| `wammimic/training/` | Cosmos 主干与动作专家、训练数据划分、损失、评估和可视化。 |
| `wammimic/_vendor/` | 固定版本的上游组件。更新时同步核对许可证、NOTICE 和来源哈希。 |
| `scripts/` | 运行入口。P1 数据处理用 `run_p1_pipeline.py`，多构型 WAM 训练用 `train_wam_p1.py`，Mimic 训练入口为 `run_mimic.py`，仿真联合评估见 `evaluate_geniesim_wam_mimic.py`。 |
| `docker/`、`patches/` | 数据和训练镜像，以及应用于外部 BlinkMorphMimic 源码快照的补丁。 |
| `tests/` | 数据协议、缓存、下载鉴权、P1 审计和 WAM 训练回归测试。 |
| `runs/` | 各次运行的配置、日志、检查点和评估产物。引用结果时注明具体运行目录。 |
| `docs/` | [架构设计](docs/design/design.md)、[Phase 0 数据管线](docs/data-pipeline.md)、[早期人体 WAM 试验](docs/wam-training.md)及各阶段训练手册。 |

## 训练与数据契约

- **WAM P1** 使用人体光学、Nymeria 惯导、Q1 仿真球头和 G1 遥操夹爪四个分支。历史状态取截至观测时刻 `t0` 的 4 帧，动作覆盖未来 10 步（10 Hz），整块共用 `t0` 的完整三维骨盆参考系。五点为 15D 位置加 30D 旋转，下肢位置按腿长缩放，双腕保留米制。人体输出 `10×45`，两个机器人分支输出 `10×47`，后两维分别表示接触和夹爪闭合度。修改读取器或动作头时保持各分支的标签、掩码和旧数据接口。[WAM 训练手册](docs/train/wam.md)
- **Mimic** 使用 BlinkMorphMimic 的 50 Hz 原生 NPZ 参考库及对应加载器。仓库内的 `MimicDataset` 读取旧版 NPY 库。接入设计中的 465D 窗口时须适配格式，并保留原清单的轨迹划分、朝向和重置帧。[Mimic 训练手册](docs/train/mimic.md)
- **划分与因果性** 按数据源协议以完整会话或回合划分，归一化统计只用训练集。旧操作回归集独立只读。动作预测仅使用因果历史，五个视频时间潜码中只有前两个可作为动作条件。保留光学、惯导、机器人 FK 与仿真标签的来源，缺失的末端或接触标注用掩码表达。

## 运行与验证

- 从仓库根目录运行。数据、模型、输出根目录通过对应入口的参数或环境变量传入，例如 `RAW_DATA_ROOT`、`PRETRAINED_ROOT`、`P1_RUN_ROOT` 和 `RUNS_ROOT`。原始数据和预训练权重只读挂载，派生数据与训练输出写入各自目录。
- 下载按所属阶段的清单、版本、哈希和预算核算，保留断点分片的占用记录。签名 URL 和访问令牌只存于私有凭据文件，避免写入仓库、日志和共享报告。
- 修改后可在已安装对应依赖的环境中运行 `python -m pytest tests/`，或指定受影响的测试文件。P1 数据就绪状态查 `${P1_RUN_ROOT}/control/progress.json` 的 `p1_training_ready` 及各分支 QA 报告。报告结果时写明数据划分和评估层级，机器人稳定性与任务成功率须使用相应的闭环记录。
