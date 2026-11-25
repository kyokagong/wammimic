# 人体 WAM 小规模训练与可视化评估（Phase 0 历史试验归档）

> **历史归档提示（2026-11-23）**：本文档仅作为 Phase 0 早期人体开环小规模试验（2026-11-18，无本体历史输入、未做 Y-up → FLU 转换、无机器人端到端闭环）的复现记录保留。**当前正式 WAM 训练、数据处理与闭环桥接请严格遵循**：[系统总架构设计](design/design.md)、[统一数据处理方案](design/data-processing.md)、[WAM–Mimic 协同与桥接规范](design/wam-mimic-bridge.md) 与 [WAM 训练手册](train/wam.md)。

本轮训练 `Cosmos-Predict2.5-2B` 最后一个 Transformer block 的原始参数和人体五点动作专家。输入是人体第一视角的历史视频潜码和任务文本，输出未来 10 帧、每帧 45D 五点位姿。没有机器人本体输入、夹爪或手指伪标签。现有 Mimic 目录始终只读。

## 数据划分

基础人体包为 `/data/kk/data/wammimic_human_ego_20261117`。新增操作原始数据为 `/data/kk/data/robomocap_operations_raw_20261118`，处理结果为 `/data/kk/data/wammimic_operations_processed_20261118`。超过同步门槛或 BVH/NPZ 不一致的候选排除，保留在下载核算及审计记录中。

`protocol.json` 合并两个已验证的数据包，固定完整会话划分。源动捕采集目录、相同视频哈希及相同 BVH 哈希均不能跨 split。行走保持官方按参与者隔离的 val/test；新增环境交互、双人移动操作、移动搬运、长程任务和桌面操作覆盖 train/val/test。**操作部分验证的是跨会话泛化，参与者可能在训练中出现，不能据此宣称跨人或跨场景泛化。**

所有归一化统计只从最终训练会话重算，读取器覆盖源数据包的旧统计；源数据和原有划分文件保持不变。评估窗口至少间隔 16 帧，避免四帧历史加十二帧未来相互重叠。小规模试验每个训练会话均匀选 8 个窗口，每个评估会话选 4 个窗口；报告同时保存完整数据规模与实际试验样本量。

## 模型与损失

主干位于 `/data/kk/models/nvidia/Cosmos-Predict2.5-2B`，固定 NVIDIA 官方 `diffusers/base/pre-trained` 修订，逐文件校验哈希并严格加载约 20.59 亿参数。编码器组件在 `/data/kk/models/wammimic_cosmos25_encoders_20261118` 通过硬链接复用原有组件，未重复下载。Diffusers 固定为 0.37.1。

动作分支仅把前两个因果潜码送入 DiT，取最后一个 block 的特征，供三层、宽度 256 的动作专家交叉注意力读取。动作块内部使用带时间条件的流匹配；未来动作仅作为扩散训练的带噪目标。另一次主干前向承担未来视频流匹配辅助损失，其特征不进入动作分支。这使未来视频像素和带噪未来视频都无法泄漏到动作条件中。

动作流按 `a(t)=(1-t)ε+t a_gt`，回归 `a_gt-ε`。三个损失组分别为下肢位置 `[0:9]`、腕部位置 `[9:15]`、旋转 `[15:45]`，取组均值后等权平均。时间采样为 `s=sigmoid(N(0,1)); t=5s/(1+4s)`。视频遵循官方 sigma 方向，回归 `noise-clean`，只计算未来三个潜码，辅助权重 0.1。

首轮参数：3 epoch、micro batch 1、梯度累积 4、专家学习率 3e-4、主干最后一个 block 学习率 1e-5。冻结前缀为 BF16，最后一个 block 和动作专家使用 FP32 主参数、BF16 自动混合精度，以保留小更新。检查真实主干梯度、冻结边界、参数变化及实际峰值显存。

同输入轻量 MLP 基线读取完全相同的历史视频潜码及文本，使用相同训练窗口和更新次数。WAM 与基线各自用验证集选择最优 checkpoint，选择指标为按录制会话平均的位置 ADE；旋转另行报告。测试集不参与选模型。

## 评估

位置先撤销训练统计归一化，下肢乘回每条人体骨架的实测腿长，双腕直接使用米制，计算五点 ADE/FDE、P95 和 0.2/0.5/1.0 秒误差。所有未来帧共用同一个 t0 参考系，不逐帧重锚定或做 Procrustes 对齐。姿态将 rot6d 正交化到 SO(3)，计算测地角；退化旋转记为 180° 并报告失败比例，非有限预测直接使评估失败，不丢弃失败样本改善均值。

报告按操作会话留出/行走跨人留出、动作家族、录制会话和参与者分组。三个固定采样种子分别报告，不能挑选最接近动捕参考的样本当主指标。训练集均值是简单基线；常速度参考需要过去动捕，是特权运动学参考，不是与 RGB/text WAM 同输入的公平对照。

另对相邻 t0 帧用固定零初始动作噪声推理，将两个动作块恢复到统一世界坐标，比较九个重叠未来时刻的一致性。这是时间一致性诊断，不等同机器人稳定性。推理耗时包含 DiT 条件提取与动作积分，输入编码已缓存，未计在线 RGB/text 编码耗时。

## 运行

从仓库根目录构建并运行：

```bash
docker build -f docker/Dockerfile.data -t wammimic:data-phase0 .
docker build -f docker/Dockerfile.train -t wammimic:train-pilot .
python -m scripts.run_wam_pilot --data-root /data/kk/data --model-root /data/kk/models --epochs 3 --gpu 0
```

监督程序等待新增操作包验证就绪后，依次执行 `protocol / train / evaluate / visualize`。容器均禁用网络；原始数据、原有 Mimic、模型底座只读，训练运行目录和新划分目录单独可写。数据首次下载和缓存需先完成；下载沿用累计 100 GB 控制清单，HF 凭据通过私有临时文件传入，不写入仓库、清单或运行结果。

默认输出：

- 划分与统计：`/data/kk/data/wammimic_wam_protocol_20261118/protocol.json`
- 训练与评估：`/data/kk/models/wammimic_runs/human-ego-pilot-20261118`
- `training_summary.json`：真实更新数、主干参数变化、显存和所选 epoch。
- `evaluation_{val,test}_seed{0,1,2}.json`：固定种子的完整指标；seed 0 同时保存预测 NPZ。
- `best/{wam,baseline}.safetensors`：各自最优的可训练参数。WAM 需要同时保留固定修订的底座，不能把增量当作完整 2B 权重。
- `tensorboard/`：本地训练曲线；可用 `tensorboard --logdir <运行目录>/tensorboard` 查看。
- `visualizations/report.html`、MP4、PNG、RRD：真实 checkpoint 的固定片段和最差片段。MP4 同步展示参考 RGB、五点位置/腕部姿态、俯视轨迹与误差；RRD 可用 Rerun 旋转和逐帧检查。未做未经相机标定验证的像素投影叠加。

本轮是小规模人体五点开环试验。源动捕骨架是测量解算参考，人体局部参考系尚需适配才能接入机器人 Mimic；没有地面、接触和手指真值，也没有机器人闭环成功率。
