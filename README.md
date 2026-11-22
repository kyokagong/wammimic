# wammimic

Cosmos-Predict2.5-2B WAM 与三机身 Mimic 的数据准备。

WAM 主线使用人体第一视角视频及同步的光学动捕五点骨架参考（骨盆、左右踝、左右腕），由 `HumanWAMDataset` 读取 `10×45` 动作。现有机器人遥操 WAM 属于独立辅助分支；Mimic 继续使用已准备的 Q1/Bumi/G1 数据。

开始训练时可查阅 [WAM 数据使用手册](docs/train/wam.md) 和 [Mimic 数据使用手册](docs/train/mimic.md)。

数据转换、潜码/文本缓存、自动下载后处理和训练读取器见 [Phase 0 使用说明](docs/data-pipeline.md)。
操作类划分、Cosmos 主干微调、真实预测回放与试验命令见 [WAM 训练与评估](docs/wam-training.md)。
原始输入只读；训练就绪状态由数据验证报告确认。首轮新增数据和模型下载预算为 100 GB；2026-11-20 用户为 P1 另增本轮 100 GB，按独立清单核算，见 [P1 执行记录](docs/train/wam-p1-data-progress-20261120.md)。
