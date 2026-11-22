# Phase 0 首批数据处理

原始数据和机器人资产只读挂载，输出写入独立的 `PROCESSED_DATA_ROOT`。不修改或提交已有数据。

`docker/Dockerfile.data` 复用本机已安装的 `wbwam-repro:latest` 数据依赖，不重新下载训练框架。

## Mimic

`morph_5pt_store/{q1,bumi,g1}` 存储每条轨迹的 `poses.npy`（50 Hz、五点世界位置和 wxyz 四元数）、`joints.npy`（关节位置和速度）及哈希收据。两者均可 mmap 加载。

`mimic_window(poses, index, leg_length)` 返回未归一化的 465D 输入。每个窗口的所有未来帧使用同一个当前骨盆参考；下肢采用腿长标尺，双腕为米制。训练加载器应先归一化 450D，再根据末尾 15D 掩码重新清零无效块。

下肢原点使用支撑脚的地面投影，腕部位置相对当前骨盆。缺少接触力标注时，以踝部高度判断支撑脚；两踝高度差不超过 2 cm 时用双脚投影中心。这是推定支撑状态，实际滑移和双脚腾空需后续物理 QA。

直接复用已配对机器人 FK 五点；不将未完成尺度适配的人体世界坐标冒充机器人五点。保留源清单的分组和 train/val 划分，不重新随机拆分镜像或三机身同源动作。

同时校验源 `contact-floor.json` 与源清单、各轨迹 SHA256 的绑定，将地面偏移和可信标记写入成品清单。mmap 保留原始世界坐标；读取器用该偏移计算下肢相对地面高度，并把返回的 `root_pose` 高度移至地面零平面。`reference_floor_valid=false` 表示没有可信地面估计，此类轨迹不能充当足底接触/滑移监督真值。

运行环境变量：`MORPH_MANIFEST`、`MORPH_SOURCE_PREFIX`（清单记录的原路径前缀）、`MORPH_PREPARED_ROOT`（容器内对应挂载点）、`PROCESSED_DATA_ROOT`。命令：`python -m wammimic.data_pipeline.prepare morph --workers 4`。

## WAM 人体 ego 主线

这批数据使用 [RoboMoCap 官方发布](https://huggingface.co/datasets/BitRobot/robomocap)，固定修订 `20adb80c474c163fe451b5c4f994b665b3234024`。原始 30 Hz 头部左前视角 RGB 与 120 Hz NOKOV 骨架处于官方校准的共同时间轴；每三帧视频取一帧，并配对最近的实际动捕帧，配对量化误差上限为约 4.17 ms。官方校时离散度另行保存，并要求不超过 5 ms；这两项都不等于真实同步误差的保证。失效测量不插值，不跳过后用其他骨架填补。

用户已接受光学动捕系统解算的骨架作为参考。五点从官方 `Hips / LeftFoot / RightFoot / LeftHand / RightHand` 提取，对应骨盆、左右踝、左右腕。保留世界坐标、原始姿态、视频和动捕帧号、时间戳及有效掩码；只在同一有效动作标注段内、历史和未来均完整时构建训练窗口。每条数据以三个实际有效帧的独立 BVH 前向计算复核 NPZ 的单位、骨架身份和旋转约定。这里的骨架解算是动捕参考，不是无误差的真实人体解剖位姿。

官方 READY 和原始有效掩码不能保证没有追踪突跳。另保存 `quality_valid`，按完整 120 Hz 骨架检查相邻测量：骨盆/踝/腕速度分别超过 10/15/20 m/s，或段姿态角速度超过 60 rad/s 时，排除该跳变及前后 0.1 秒。训练窗口必须覆盖的每一帧原始骨架都有效，避免漏掉两帧采样视频之间的失效。该门槛仅排除明显异常，不构成生物力学真实性认证；原始有效位、位姿和官方 sanity 警告均保留，不平滑或修改参考。

下载 24 个候选会话，23 个通过校验，覆盖 14 位参与者及 8 个动作家族，遵守官方按参与者隔离的 train/val/test（通过 16/3/4 会话）；val/test 仅有 locomotion，不能代表操作任务泛化。超过同步门槛的候选已替换，不降低门槛。另一条 `P01/session_024` 的原始 BVH 骨骼偏移全为零，与发布的 NPZ 位置不一致，已隔离而不补造骨长。下载包约 2.20 GB，包含原始骨架/BVH、标记点、头部刚体、相机标定和时间戳以及单路视频；没有下载或使用 SMPL-X 拟合、视觉手指、SLAM 或场景估计充当人体标签。

输出 `wam_lerobot_v2/human_ego`，完整动作块为 `10×45`，排列仍为 15D 位置 + 30D rot6d。**人体预训练坐标协议与设计中的机器人地面参考协议不同**：所有位置锚定 t0 人体骨盆原点和完整三维朝向，下肢除以该人体骨架的实测腿长，双腕保留米制；旋转保留源骨骼轴并相对 t0 骨盆。源 NOKOV 世界零点并非物理地面，因此不生成地面高度、支撑脚或接触真值。接入现有 Mimic 前仍需人体到机器人尺度、骨骼轴和地面参考系适配；原始世界位姿及锚点索引保留供重新导出，不可将这批局部 45D 直接视为现有 Mimic 的指令。

`HumanWAMDataset` 仅返回视频潜码、因果条件掩码、文本缓存及归一化 `10×45` 动作，不注入人体动捕状态作输入，不添加 G1 本体状态、机器人构型标签或虚构夹爪/手指动作。现阶段可用于人体五点规划预训练；完整设计的机器人 `45D+EE` 训练仍需要对应的真实标签和适配。动作标签按原始厂商标注逐帧绑定 instruction，不把一个会话内不同动作错误合并成同一句提示词。统计量只使用 train。

当前已有数据检查结果写入人体输出的 `qa_reports/source_audit.json`：EgoWalk 的相机里程计与文字缺少人体五点；当前 OakInk2 子集缺图像和可用光学下肢参考；已有 BVH/动作归档尚无验证的配对 ego 视频；MotionDecode 当前公开文件为 G1 重定向结果；Nymeria 尚无录制文件。上述数据均未混入人体五点监督。

宿主机运行：

```bash
export RAW_DATA_ROOT=/data/kk/data
export PROCESSED_DATA_ROOT=/data/kk/data/wammimic_human_ego_20261117
export PRETRAINED_ROOT=/data/kk/data/wammimic_model_cache_20261116
python -m scripts.run_human_wam
```

监督程序等待固定下载清单全部校验完成，依次执行 `human-ego`、`cache-human-text`、`cache-human-video`、`validate-human-wam`。处理容器无网络，整个原始目录（包含旧 Mimic）只读，只有新的人体输出可写。最后验证所有已保存五点与源实测帧逐项一致、动作和潜码窗口掩码、参与者隔离、完整文件哈希、真实 DataLoader 和有限梯度，并复核旧 Mimic 清单及全部轨迹文件哈希。

人体就绪依据为 `qa_reports/human_wam_training_ready.json`。缓存复用现有 Cosmos-Predict2.5-2B 组件，格式见下文。旧数据根目录 `/data/kk/data/wammimic_processed_20261116` 保留。

## WAM 机器人遥操辅助分支

OpenHLM 子集下载完成后运行 `python -m wammimic.data_pipeline.prepare openhlm`，另设置 `RAW_DATA_ROOT` 和 `OPENHLM_DOWNLOAD_MANIFEST`。

`wam_lerobot_v2/g1_parallel_gripper` 包含 Parquet、10 Hz 224×224 头部视频、任务/构型/模态元数据、训练集统计量和收据。每一行保存 **10×47 的完整未来动作块**，明确使用观测时刻 t0 的参考系，避免把十个各自重锚定的单帧拼接成错误窗口。此处采用预组块存储；使用 LeRobot/GR00T 训练器时应直接读取此动作块，不能再沿时间轴叠加第二个动作窗口。

`action.is_pad` 标识越界目标；`valid_for_training` 只允许至少四帧历史且未来十帧完整的行。统计量仅使用 train 中的有效行。动作的 45D 部分由未来实测关节轨迹 FK 派生，末端部分来自原始命令。额外的 50 Hz 同源关节/五点写入 Mimic 库，以 `traj_id` 关联。

未来目标经过抗混叠低通处理；每个动作块的参考原点与朝向使用未低通的 t0 观测位姿，与本体状态保持一致，避免未来滤波结果改变观测参考系。
本体关节速度与角速度仅由当前及过去的原始 30 Hz 观测估计。根轨迹先在原始时钟上求支撑里程计，再以位置 PCHIP、姿态 Slerp 对齐到 50 Hz；避免插值斜率把未来观测带入本体输入。

用 `meta/splits.json` 的 episode 列表显式选择 train/val/test，再过滤 `valid_for_training`。不要默认把全部回合用于训练。视频文件索引与 Parquet 行一一对应。

OpenHLM 的根平移没有直接真值，使用支撑脚运动学里程计估计，并在每条收据中标记。它存在滑移和累计漂移，不能作为全局定位真值。关节顺序按 OpenHLM 官方 OpenPI 格式解析，不沿用旧项目错误的 body-first 假设。异常轨迹进入 QA 报告。

## 缓存与验证边界

EgoWalk 官方处理版为 5 Hz，缺少全身关节；当前 OakInk2 子集缺少图像和可靠的光学下肢参考。两者不进入人体五点全监督集合。Nymeria 官方链接到位后还需单独核验其参考来源，不能默认当作光学真值。

WAM 主干已确认是 **Cosmos-Predict2.5-2B**。视频 tokenizer 和文本编码器是它的配套组件，不是替代主干。官方源码在 `predict2/tokenizers/cosmos.py` 中注册 Wan2.1/Wan2.2 tokenizer，并在 `predict2/text_encoders/text_encoder.py` 中提供 Reason1 系文本编码器；实际缓存必须跟随选定 checkpoint 的配置，不能按名称互换已有的 Wan2.2 或 Reason2。

缓存实现直接复用固定版本的官方 VAE 核心；来源、版权和校验信息位于 `wammimic/_vendor/`，没有安装整个分布式推理 SDK。使用公开上游 VAE 的前提是其 SHA256 与 Cosmos `tokenizer.pth` 完全相同。

### 缓存格式

- 视频：`latents/chunk-000/episode_*.safetensors`，每个样本为 BF16 `[16,5,28,28]`，附带对应的原始 `frame_index`。
- 文本：输出根目录的 `prompt_bank_bf16.safetensors`，按 `task_N` 索引。每条为 BF16 `[512,100352]`：官方 Reason1 28 层、每层 3584 维，按官方规则逐 token 归一化后拼接。
- 权重固定为官方 checkpoint 注册表指定的 Reason1 修订 `3210bec0495fdc7a8d3dbb8d58da5711eab4b423`；VAE SHA256 为 `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`。

四帧历史 `[t-3,t-2,t-1,t]` 在最前重复一帧，形成 5 帧编码前缀，再接 12 帧真实未来视频，共 17 帧，满足 tokenizer 的 `4k+1` 时间结构。前两个潜码只包含历史，后续三个为视频预测目标。动作仍预测未来 10 步（1 秒）；视频辅助目标覆盖 1.2 秒，多出的两帧用于时间编码对齐。缓存只保留满足 `t+12 < length` 的样本，绝不把不足的未来视频补为真值。编码首批样本时，会改变未来像素并验证前两个条件潜码保持不变。

`HumanWAMDataset` 返回人体视频潜码、条件掩码、文本特征和 `10×45` 动作。`WAMDataset` 的默认旧分支返回机器人视频潜码、条件掩码、文本特征、归一化本体状态和 `10×47` 动作。`MimicDataset` 按机身读取 mmap 数据，返回未归一化的 465D 窗口及真实关节参考；训练时的 Running Normalizer 应保留末尾掩码并在归一化后重新清零无效块。

`validate-mimic` / `validate-wam` 校验文件哈希、分组隔离、形状、旋转及动作范围，并通过真实 DataLoader 批次和一个轻量消费头的有限梯度检查。仅在这些检查通过后发布训练数据就绪报告。这不是完整 Cosmos/Mimic 网络的训练测试，也不是动态物理稳定性认证。完整 Cosmos 主干权重仍需要 Hugging Face 仓库授权；生成缓存不需要加载该主干。

Rerun 抽检和完整物理评估未在此流程中执行。面向 P1 阶段（解决 WAM 泛化瓶颈、桌面操作域差与夹爪/灵巧手监督缺失）的新增数据清单与接入规范见 [WAM P1 阶段数据补全与采集规范](train/wam-p1-data-plan.md)，基于当前进度的下一步数据收集与处理指导见 [P1 下一步数据收集与处理执行指南](train/p1-data-next-steps-guide.md)。

参考：[官方 tokenizer](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/cosmos_predict2/_src/predict2/tokenizers/cosmos.py)、[官方文本编码器](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/cosmos_predict2/_src/predict2/text_encoders/text_encoder.py)。

## 旧机器人分支后台执行

构建镜像后，宿主机设置 `RAW_DATA_ROOT`、`PROCESSED_DATA_ROOT`、`MORPH_PREPARED_ROOT`、`MORPH_SOURCE_PREFIX`、`PRETRAINED_ROOT`，执行 `python scripts/run_phase0.py`。已完成且来源一致的标准化不会重复运行。监督程序自动重启退出但未完成的受预算限制下载，生成 WAM 标准数据、文本缓存、视频潜码，并运行最终数据验证。处理容器禁用网络，模型权重只读挂载，避免编码器自行下载未列入预算的文件。

状态位于 `pipeline_status.json`，各阶段独立写日志。最终依据是 `qa_reports/mimic_training_ready.json` 和 `qa_reports/wam_training_ready.json`，而不是进程仍在运行或文件已经出现。

下载采用有界 HTTP 分片，合并并校验完整文件 SHA 后才发布，完成收据绑定固定下载清单。只按文件大小匹配不会触发标准化。编码任务选择剩余显存最多且容量足够的单张 GPU，不足时等待。

新增下载总上限为十进制 100 GB，**累计计算原始数据、模型组件和保留的旧部分下载**。加入人体子集后，已选数据、模型及保留片段预算约 74.29 GB；另保留 Nymeria 20 GB 和软件/元数据余量 5 GB，合计约 99.29 GB。实际固定清单与保守核算见 `/data/kk/data/wammimic_downloads_20261116/budget_100g.json`；不能把各单包限额相加当作还可新增的下载额度。标准化数据与潜码属于本地产生的派生产物，另占磁盘空间。此流程不下载 Nymeria、不启动正式训练。
