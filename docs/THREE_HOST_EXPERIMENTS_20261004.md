# 三机实验与论文配置索引（2026-10-04）

本次补充**实验定义、源码和去部署化配置视图**，不是数值结果发布或完工宣告。当前主实验仍在运行；最终结果HTML在全部完成、完整核验之后另行本地交付。这里的126是已登记学生配置数，不能引用为“已完成126组”。

| 范围 | 作物/预算 | 学生配置 | 成功步/组 | 配置入口 |
| --- | --- | ---: | ---: | --- |
| 本机三作物消融 | 木薯、水稻、香蕉；1/5/10/20/30/100% | 66（各22） | 1600 | 下方三个消融目录 |
| Windows服务器+本机联合主实验 | 苹果、木薯、水稻、香蕉；1/5/10/20/30% | 40（MT20+FM20） | MT1800 / FM2000 | [协议](../experiments/multicrop_itm_mt_fixmatch_v1/configs/protocol.public.json) |
| 当前新Linux FreeMatch | 同上四作物×五预算 | 20 | 2600 | [协议](../experiments/multicrop_itm_freematch_v1/configs/protocol.public.json) |

原Windows28和本机12是同一40逻辑配置的两种实际执行环境，**不能计为额外40或52组**。发布前已逐项匹配本机与Windows归档的17项原科学source及40原config。三作物消融已经按其原冻结完成格式终验；MT/FM及FreeMatch没有因源码发布而停止、换版本、改配置或重跑。

用户自管另一服务器40组不在这次结果收集/运行范围；仓库已存在的SoftMatch/SimMatch代码归档原样保留，只作既有方法背景，不读取该服务器数据、结果或推定其完成情况。

## 找某一组的配置

完整机器可读清单：[experiment_catalog_20261004.json](experiment_catalog_20261004.json)。它含126个唯一run，逐个映射到公开配置文件、训练代码目录、原科学配置SHA256和成功步预算。

例如：

```sh
python tools/find_experiment_config.py --crop banana --method fixmatch --budget 10
python tools/find_experiment_config.py --crop rice --stage S4 --budget 30
python tools/find_experiment_config.py --run-id apple_001_freematch_s20260825
```

索引不读取数据、不启动训练、不生成新的实验配置。JSON中的原配置hash用于查找私有历史来源，**不能把删去绑定后的public.json冒称原配置或拿它恢复旧断点**。

## 消融目录与阶段链

- [木薯](../experiments/cassava_itm_g1_g4_v1/README.md)：保留旧5%与后18组fast执行区别，OT双边残差门槛分别1e-5和1e-4；公开视图逐组保留原物理参数。
- [水稻](../experiments/rice_itm_s1_s4_v1/README.md)：22组继承fast定义；全文guard400，9853主集，480待复核隔离，group-aware划分。
- [香蕉](../experiments/banana_itm_s1_s4_v1/README.md)：22组继承fast定义；全文guard384，category-qualified IDs及派生raw路径适配；不公开实际ID。

S1 = ALBEF4M→BCE；S2 = 本预算S1-best→BCE+Pair-USA；S3 = 本预算S1-best→BCE+OT；S4 = 本预算S3-best→BCE+Pair-USA+OT，复用同预算S2关系教师。100%无U，仅S1/S2。每阶段独立1600步及优化器/LR日程；不是把阶段初始化当正式断点恢复。

这里的后续三作物S4从S3-best启动，**不同于既有苹果历史G4从A-best启动**，不能混淆。教师训练和父阶段计算量另计，不把教师/工程步混进66学生阶段。木薯12个历史S1/S2冻结checkpoint没有RNG/best_snapshot，代码新增格式不能追溯补造原状态。

## 前两主方法与第五算法

- [Mean Teacher / FixMatch](../experiments/multicrop_itm_mt_fixmatch_v1/README.md)：独立公共34键零步初始化，不从BCE-best或USA教师启动。MT前40%MSE ramp；FM硬阈值0.95，全32个U归一化。早期FM没有通过伪标签不是故障。
- [FreeMatch](../experiments/multicrop_itm_freematch_v1/README.md)：ICLR2023 SAT+SAF图文pair二类适配。统计只在成功optimizer更新后提交一次；2600步。当前speed_v4为原reference实现至多两个不同run的有界并行，不是retained单worker优化。

共同seed20260825；L16正锚图→32图文对/U32，原全文blind caption及固定随机异病例完整caption弱负例；不读取guided/私有U标签。ALBEF4M及公共34键零步来源一致，不代表跨OS/GPU长程逐位相同。每100步固定400 Validation图/800pairs，主方法EMA选模和阈值，无新Test推理。

主实验文本guard按作物分别256/384/400/384，全文不截断；不能用统一短文本上限替代长作物输入。FP16初始scale1024、同一步溢出退避，LR80暖启后余弦，EMA0.999；物理16仅同一步OOM退8/4。消融旧版本参数以逐组配置与版本说明为准。

## 版本、环境与发布边界

[环境公开快照](environments_20261004.public.json)记录三种主执行环境及本机消融环境，删去了真实机器/SSH/GPU UUID和路径。Windows原28、独立本机12、Linux FreeMatch的Python/CUDA等不同，不能称完全同环境或等计算。

[源码清单](source_manifest_three_hosts_20261004.json)记录原SHA256、公开副本SHA256、目录映射和变换。新增文件仅做LF规范化、去标识路径/部署字面量或配置私有绑定剔除；未改活跃原科学文件。算法源码不是自动可运行的软件包，真实输入合同/官方资产/准入/所有权/冻结fingerprint仍由使用者独立准备，仓库不附真实passed或复用旧准入。

公开视图保留模型、优化器、学习率、批次、阶段、增强、阈值和步数；删除图片/配对路径、teacher/parent结果绑定、安装路径、launch flags和实际运行身份。没有上传数据/caption/划分样本ID、预测、权重/教师张量、checkpoint、SQLite/回执、日志、result/status/HTML或连接凭据。

单seed、随机假负例、Validation选模型/阈值、不等步数/算力、教师及多阶段成本、图文pair适配与执行版本/环境限制必须如实写进论文。CPU公共测试只能验证合同、方程及归档，不能代替完整数据、真实GPU重放、长程完成或科学有效性。

无数据CPU检查：

```sh
python -m unittest discover -s tests -p "test_*.py"
python tools/verify_source_archive.py --manifest docs/source_manifest_three_hosts_20261004.json
```
