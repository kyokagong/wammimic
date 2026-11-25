# 数据管线基础契约与 Cosmos 潜码缓存说明

> **现行权威规范（v2.0）**：面向 WAM-Mimic 正式训练的**总体数据处理逻辑、Canonical Human（$H_5'$）Z-up FLU 五点坐标规范、运动分桶与逐数据集处理清单**，请直接以 [统一数据处理方案](design/data-processing.md) 为准。本文档仅保留所有阶段共用的 **Cosmos-Predict2.5 离线潜码缓存规范**、**Mimic 底层参考库边界** 与 **Phase 0 基础设施约定**；早期 Phase 0 中与 v2.0 冲突的“支撑脚地面投影”“双腕各构型原米制”“无本体历史输入”“各分支独立统计量”等旧规则已移除。

---

## 1. 基础容器与存储边界

- 原始数据（`RAW_DATA_ROOT`）与预训练权重（`PRETRAINED_ROOT`）始终以**只读**方式挂载；派生标准化数据写入独立的 `PROCESSED_DATA_ROOT` / `P1_RUN_ROOT`，训练产物写入 `RUNS_ROOT`。
- 处理容器全程禁网；数据源相对子路径在 `conf/data/pipeline.yaml`（JSON 兼容 YAML，标准库解析）中注册。
- 下载严格按固定清单、SHA256 哈希与预算执行，保留断点收据；签名 URL 与访问令牌仅存于私有凭据文件，不写入仓库或共享报告。

---

## 2. Cosmos-Predict2.5 离线潜码缓存格式（全阶段共用）

WAM 主干固定为 **Cosmos-Predict2.5-2B**，配套固定版本的视频 tokenizer（Wan2.1 VAE）与文本编码器（Cosmos-Reason1-7B），严禁按名称随意互换其他版本。

- **视频潜码**：`latents/chunk-000/episode_*.safetensors`，每样本为 BF16 **`[16, 5, 28, 28]`**，附带对应 `frame_index`：
  - 每个有效样本由 4 帧历史 `[t-3, t-2, t-1, t]`（首帧重复一次构成 5 帧前缀）+ 12 帧真实未来视频共 **17 帧**组成，满足 `4k+1` 时间编码结构；
  - **严格因果性**：前 **2 个时间潜码**只包含历史观测并作为 WAM 动作条件，后 3 个潜码仅用于未来视频辅助损失；仅保留满足 `t+12 < length` 的样本，绝不补帧伪造未来视频；编码抽检必须验证“替换后 12 帧未来像素时前 2 个条件潜码严格不变”；
  - 固定 VAE SHA256：`38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`（实现复用 `wammimic/_vendor/`）。
- **文本潜码**：输出根目录 `prompt_bank_bf16.safetensors`，按 `task_N` 索引，每条为 BF16 **`[512, 100352]`**（Cosmos-Reason1 28 层 × 3584 维，逐 token 归一化拼接）：
  - 固定文本编码器修订：`3210bec0495fdc7a8d3dbb8d58da5711eab4b423`。
- **潜码复用规则**：当仅更新五点坐标转换（如 Y-up → FLU、G1 FK → 标准人体五点）而不改变 224×224 视频帧和文本时，**可直接复用既有视频与文本潜码**；若视频本身存在裁剪错误（如 `hiw500` 双目跨缝 Bug），必须重转 MP4 并重跑视频 VAE 潜码。

---

## 3. Mimic 参考库边界

- **正式 Mimic 训练（40k H=10）**：使用 BlinkMorphMimic 的 **50 Hz 原生 NPZ 参考库**（含 `human50` 原始人体五点 sidecar 与 `robot_reference50` 机器人参考五点，见 [Mimic 训练手册](train/mimic.md) 与 [设计 §3.6](design/design.md#36-原始人体与机器人参考五点的共同数据契约mimic-侧)）。
- **仓内旧 `morph_5pt_store/`（mmap NPY）**：保留作为 50 Hz 世界 FLU 五点/关节归档（例如 OpenHLM 406 条回合的 `morph_5pt_store/g1/openhlm_*/poses.npy` 可直接用于重算 v2.0 标准人体五点 Parquet，无需重跑原始 30 Hz FK）。
- **地面与划分保留**：保留源清单的分组、受试者/会话划分与 `contact-floor.json` 地面校验标记；`reference_floor_valid=false` 的片段不得充当足底接触真值。

---

## 4. 非五点存量数据隔离清单

以下存量数据源因缺失配对的第一视角 RGB 或缺失全身五点参考，**继续隔离，不进入 WAM 五点训练集**：
- `egowalk_raw`：5 Hz RGB 与相机里程计，缺少全身五点骨架；
- `oakink2` 当前子集：缺少图像流与可靠下肢参考；
- 纯动捕无配对第一视角视频归档（如 `MotionDecode` 纯动作文件）；
- 固定底座机械臂数据（`ds_*`、`so101_data`）。

关于各现役数据源的具体处理步骤与修复清单，见 [统一数据处理方案](design/data-processing.md) 与 [P1 数据纠偏与扩展执行指南](train/p1-data-next-steps-guide.md)。
