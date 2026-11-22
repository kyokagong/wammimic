# Mimic 12000 steps 精简部署包

交付目录：`/data/kk/artifact/mimic-12000-compact-20261121/`。
包含 Q1、Bumi、G1 的 12000 steps EMA 部署模型及指定参考动作。

## 动作范围

每台机器人仅保留以下 **7 条参考**，三台合计 21 条：

| 类别 | 动作 |
|---|---|
| 八极拳 | `motiondecode_baji_WP_Baji_Quan_17851016` |
| 舞蹈 1 | `dance1_subject1` |
| 舞蹈 2 | `dance1_subject2` |
| 站起 | 躺姿、坐姿、半蹲各一条，文件和入口帧见 `recovery_windows.json` |
| 中立位 | `neutral.npz`，50 帧、50 Hz、相同站姿，速度全部为零 |

中立位取审核通过的 `walk2_subject4` 第 11446 帧的关节角与全身位姿，
只导出这一个姿态并重复为 1 秒参考，不附带完整的站姿来源片段。
持续站立时钳制参考末帧，继续更新实时观测、策略和控制。

两个舞蹈和三条站起各附真实人体 24 点及五点数据，每机型各 5 组，三台共 15 组。
八极拳没有真实人体配对；中立位是机器人静态目标。二者使用 motion 接口。

## 文件

```text
model/{q1,bumi,g1}/            motion.onnx、human_h1.onnx、metadata.json、数值复核文件
model/部署指南.md             完整模型输入、特征公式、原生关节映射与 CPU 报告
references/{q1,bumi,g1}/       每机型 7 条机器人参考
references/human50/           15 组配对人体 24 点
references/five_point/        15 组配对人体五点
references/catalog.json       本精简包完整动作清单、相对路径、帧数及 SHA256
references/recovery_windows.json  9 个站起窗口的 start_frame / hold_frame
validation/motion-audit.json  动作范围与格式、静态中立位和起身目标的校验结果
licenses/                    模型及数据的原始许可证和归属
package_mimic_deploy.py       打包与校验脚本
manifest.json, SHA256SUMS_ALL 整包文件哈希
```

所有动作直接位于 `references/`，读取 `catalog.json` 的 `file/human_file/five_file` 相对路径即可。
数据是 50 Hz 原生 NPZ，四元数 wxyz，位置米、关节角弧度；按 `joint_names` 与同机型元数据映射关节。
站起应从窗口的 `start_frame` 读取，达到 `hold_frame` 后持续保持该参考，不能通过覆盖实际状态代替起身。

## 模型接入与验证

六张部署图保持原 12000 steps 优化版本，四层 FiLM、morph 拼接及常量已折叠/冻结。
两个入口均必须保留动态 `ref_feedback[1,160]`：当前实测 FK 对比同机型参考的 0/20/40/60ms 目标。
`obs_history` 是十帧实时观测，不能用参考数据代替。NPZ 也不能直接展平后充当模型目标特征。
模型构造特征、heading、动作缩放及保护要求见 `model/部署指南.md`；其中示例目录替换为本包的 `model/`。

在本包根目录校验（Python 3.11+、NumPy）：

```bash
sha256sum -c SHA256SUMS_ALL
python package_mimic_deploy.py --check .
```

模型独立数值复核额外需要 ONNX Runtime：

```bash
python model/export_mimic_deploy.py --check model/q1
python model/export_mimic_deploy.py --check model/bumi
python model/export_mimic_deploy.py --check model/g1
```

重新生成相同范围的精简包，从 `/data/kk/wammimic` 根目录运行：

```bash
python scripts/package_mimic_deploy.py \
  --compact-from /data/kk/artifact/mimic-12000-deploy-20261121 \
  --guide docs/train/mimic-compact.md \
  --output runs/new-mimic-compact
```

输出目录须不存在。校验包括模型/动作 SHA256、准确动作范围、关节顺序、有限值、
中立位恒定且速度为零，以及 9 个站起保持帧与中立位一致。
本次未重新训练或运行实机验收；静态参考和数值等价不代表真实机器人已稳定站立。
数据使用条款见 `licenses/`。
