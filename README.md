# 图文匹配实验与论文配置索引

本仓库保存苹果与三作物消融，以及 Mean Teacher、FixMatch、SoftMatch、SimMatch、FreeMatch 的**代码和实验定义**。不包含图片、caption、训练/验证配对表、样本 ID、模型权重、教师目标、预测分数或原始实验结果。远端是公开仓库；请勿将这些输入或输出直接放入 Git。上传边界见 [`docs/PUBLIC_UPLOAD_POLICY.md`](docs/PUBLIC_UPLOAD_POLICY.md)。

2026-10-04三机补充：[实验/论文配置索引](docs/THREE_HOST_EXPERIMENTS_20261004.md)、[126个唯一科学配置视图](docs/experiment_catalog_20261004.json)、[源码版本/hash映射](docs/source_manifest_three_hosts_20261004.json)。新增本机66消融、原Windows+本机唯一40 MT/FM、Linux20 FreeMatch；126是计划配置数，不是完工数。可用 `python tools/find_experiment_config.py --crop banana --method fixmatch --budget 10` 查到具体配置和代码。原SoftMatch/SimMatch自管服务器40不纳入本次结果收集，已有代码及提交原样保留。

新增 [`experiments/multicrop_itm_soft_simmatch_v1`](experiments/multicrop_itm_soft_simmatch_v1)：苹果、木薯、水稻、香蕉 × 五预算 × 两算法。单种子、仅验证集、SoftMatch2200步 / SimMatch2400步，非等算力；是ITM适配，不是病害分类，也不是官方分类任务精确复现。源码含原始与加速执行层，复现限制详见该目录。下方 G1–G4 定义保持原样，不与这轮算法表混合。

| 组别 | 方法 | 预算 | 初始化 |
| --- | --- | --- | --- |
| G1 | 图文匹配 BCE | 1%、5%、10%、20%、30%、100% | ALBEF 预训练权重 |
| G2 | BCE + 自定义 Pair-USA | 同上 | 对应预算 G1 的最佳权重，再训练 1600 步 |
| G3 | BCE + OT | 1%、5%、10%、20%、30% | 对应预算 G1 的最佳权重，再训练 1600 步 |
| G4 | BCE + 自定义 Pair-USA + OT | 同 G3 | 对应预算 G1 的最佳权重，再训练 1600 步 |

每个阶段 1600 次成功更新；100% 预算无 U，G3/G4 不适用。G1/G2 只使用有标签 L，G3/G4 使用 U 的 blind caption，但不读取 U 的隐藏匹配来源标签。这里的 BCE 是**图文对匹配二分类**，不是六病害多标签 BCE。G1 比其他三组少一个训练阶段，因此四组并非完全等算力。负例是固定的随机异病例完整 caption 错配，未经逐图语义审核，可能存在假负例。

## 仓库内容

- [`experiments/apple_itm_pairusa_random_v2/code`](experiments/apple_itm_pairusa_random_v2/code)：原 A/B 训练器、图文匹配模型、随机负例构造与工程核验。历史 B 为独立初始化的 Pair-USA 学生，不计入上述 G1–G4 主表；其每预算教师被 G2/G4 复用。
- [`experiments/apple_itm_pairusa_warmstart_v1/code`](experiments/apple_itm_pairusa_warmstart_v1/code)：G2 的 A-best 续训逻辑。
- [`experiments/apple_itm_usa_ot_ablation_v1/code`](experiments/apple_itm_usa_ot_ablation_v1/code)：G3/G4 的 OT 目标、训练与数值恢复。最先完成的 5%/20% 四组使用早期求解器，其源码单独归档在 `archive/pre_recovery_solver/`；后续组使用当前求解器。不能把两者称作同一代码指纹。
- [`experiments/apple_itm_g1_g4_multiseed_v1`](experiments/apple_itm_g1_g4_multiseed_v1)：新增 `20260826`、`20260827` 的代码与计划。**这两个种子尚未运行，也未做真实 GPU 工程核验。**
- [`src/albef_ssl`](src/albef_ssl)：本实验使用的 ALBEF/LoRA 模型实现及必要的上游 ViT、xBERT 文件。上游许可见 [`third_party/ALBEF_LICENSE.txt`](third_party/ALBEF_LICENSE.txt)。
- [`tests/test_ot_public.py`](tests/test_ot_public.py)：不依赖数据与历史失败张量的 CPU 数值检查。原内部 OT 测试依赖未公开的故障重放制品，因此不直接放入此代码仓库。

## 复现边界

这是**本机历史实现的源码归档**，不是跨机器开箱即跑的软件包。历史 Python 源码和计划保留了原运行环境的绝对路径；配置、输入与断点的来源哈希相互绑定。擅自替换路径、修改旧配置或复写断点会破坏旧实验的来源核验。另见 [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) 和 [`docs/DATA.md`](docs/DATA.md)。

基础 ALBEF 权重、BERT tokenizer 和冻结的输入表需由使用者在获得相应权利后另行准备，并校验哈希。仓库不提供凭空再生成旧实验数值的承诺，也不提供独立测试集模型评估。

主实验历史训练种子为 `20260825`。新增两种子仅改变训练随机流；固定的数据构造和负例配对仍来自 `20260825`。固定 seed 不代表跨硬件逐位一致。

无数据的快速检查：`python -m unittest discover -s tests -p 'test_*.py'`。这不能替代真实数据、GPU 初始化和断点恢复核验。
