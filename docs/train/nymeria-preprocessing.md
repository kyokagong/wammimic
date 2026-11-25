# Nymeria WAM 数据处理与 v2.0 对齐说明

> **v2.0 契约更新（2026-11-23）**：本文档保留 2026-11-21 首轮 Nymeria 解码、VRS/Xsens/叙述时钟对齐与 QA 事实记录。面向后续 WAM 正式训练，Nymeria 的坐标输出与归一化已统一收敛至 [统一数据处理方案](../design/data-processing.md)：
> 1. 参考系由旧版“t0 完整 3D 骨盆系”升级为 **Canonical Human Z-up FLU（Yaw-Only 骨盆朝向系 `{P_t0}`）**，且双腕同步按受试者实测臂长标准化至标准人体尺度；
> 2. 移除“独立使用本分支统计量训练”的旧规则，并入**唯一全局 `canonical_human_5pt_stats.json`**（`embodiment_id = 2`，`ee_mask = 0`）；
> 3. 旧版 3 条 `nymeria/` 录制以沙发坐姿写白板为主（`L_ank z ≈ -0.454`），在正式训练中降权/隔离，主用 **`nymeria-40/`（32 名独立受试者，26/3/3 隔离）** 及后续流式扩容序列。

---

## 1. 原始解码与因果时间对齐规范（保持有效）

每条录制使用完整头部 RGB VRS、`body_motion.zip` 中的 `body/xdata.npz`、官方 activity summarization CSV 和元数据，处理容器禁网且原始目录只读挂载：

- **完整性校验**：对全部原始文件核验官方大小与 SHA-1，再提取 NPZ；`projectaria-tools==2.4.0` 离线运行。
- **240 Hz Xsens 骨架**：直接读取原始时间戳与世界 FLU 位置/四元数，提取骨盆、左右踝、左右腕五点；不补齐四元数、不修复时钟、不插值缺失测量。
- **统一时钟（TIME_CODE）**：通过 VRS 同步映射将 RGB 设备时钟与叙述时间统一折算到 TIME_CODE，与 Xsens 骨架严格对齐。
- **严格因果 10 Hz 网格**：在 10 Hz 网格取网格之前最近的骨架（采样年龄上限 `5.17 ms`）与 RGB（上限 `68.67 ms`）；历史速度仅由紧邻过去的 240 Hz 实测帧后向差分计算。
- **Aria RGB 方向校正**：顺时针旋转 `90°` 后缩放为 `224×224`，保持顶部朝上、不镜像。
- **严格异常窗口剔除**：跨越叙述重叠歧义、标注边界、未标注区、时间间隔异常或 240 Hz 骨架突跳的窗口一律排除（覆盖整个 240 Hz 测量区间）。
- **潜码复用**：既有 `nymeria-40` 的 `video_latents [16, 5, 28, 28]`（仅前 2 潜码为因果条件）与 `prompt_bank_bf16.safetensors` 在升级至 v2.0 标准人体五点 Parquet 时**100% 复用，无需重下载 VRS 或重跑编码**。

---

## 2. 流式扩容规范（用于后续 60–100 条增量）

由于单小时原始 Aria VRS 约 `51 GB`，后续扩容统一采用流式处理：
1. 先按官方语言元数据关键词（`walk around / avoid / table / counter / switch / button / press / cabinet / reach`）筛选高价值室内移动操作序列；
2. 单条下载 VRS + `xdata.npz` + 标注 → 导出 v2.0 Canonical Human Z-up Yaw-Only Parquet 与 Cosmos 视频/文本潜码 → 通过 [六项 QA 门禁](../design/data-processing.md#15-入训前六项硬性-qa-门禁缺一不可)；
3. 写入哈希收据后**立即删除该条原始 VRS**，仅保留压缩后处理产物与潜码。

---

## 3. 首轮验证历史记录（2026-11-21 · 3 条 `nymeria/` 验证包）

- 首轮 3 条录制（3 名佩戴者，23,245 有效窗口，70 条叙述）跑通 `nymeria → cache-nymeria-text → cache-nymeria-video → validate-nymeria` 全链路，坐标往返最大误差 `1.228e-7 m`、`4.361e-6°`，因果替换未来像素对前 2 条件潜码零影响，20 条 RGB/五点抽检无左右或朝向颠倒。
- 后续扩展批次 `nymeria-40/` 已完成 32 名受试者（`26/3/3` 隔离、~29.2 万窗口）潜码缓存，待按 [统一数据处理方案](../design/data-processing.md#数据集-4nymeria-真人惯导第一视角nymeria--nymeria-40--后续扩容) 重导出 v2.0 标准人体五点 Parquet 后并入新版训练。
