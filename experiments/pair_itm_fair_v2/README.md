# 公平比较 v2：统一总预算、模块消融、三个训练种子

这是**新的科学协议和共同训练代码**，不是对历史结果重新命名。没有启动新的训练，也没有真实 GPU passed、数值结果或完成声明。历史苹果/三作物消融、MT/FM、SoftMatch/SimMatch、FreeMatch 的冻结源码、配置、哈希清单和断点不修改、不接管、不续接。

## 固定主方案

- 训练种子固定为 **20260825、20260826、20260827**。数据构造种子仍为 **20260825**；三次训练不重新生成划分、负例或 Validation。不同种子的共同骨干/ITM 头可以有不同初始化；同一作物/预算/种子内，各方法共同部分必须完全相同。
- 每个学生运行总计 **3200 次成功 optimizer 更新**；OOM/AMP 失败不计步，但成本计入。共同逻辑 L16 锚图→32pairs、U32 抽样机会；不声称各方法实际使用了相同数量的 U 或相同计算量。
- 共同模型、优化器、LR、图像/文本输入合同、每步采样和增强流。每种方法保留必要的独有模块、阈值、统计、EMA 教师或记忆库，不把不同算法强行改成同一损失。
- 每100成功步固定400图/800pairs Validation，所有方法以 **EMA** 选模，paired accuracy优先、AUROC次之；学生模型仅诊断，不能按方法挑“学生/EMA中更好者”。共32次选模机会。阈值用 Validation 最大 macro-F1，同时报告固定0.5；无新的 Test 推理。
- 全局80步LR预热后按3200上限余弦衰减。不会随早停结果重定义目标、重置LR，或从旧best多获得一个阶段。

主表为 BCE、Mean Teacher、FixMatch、SoftMatch、SimMatch、FreeMatch、BCE+Pair-USA+OT：四作物×五预算（1/5/10/20/30%）×三种子，共420配置。这是**等学生更新预算**，不是等算力或原论文分类任务的原样复现。

## 四组受控消融

| 分支 | 前1600步 | 后1600步 |
| --- | --- | --- |
| BCE | BCE | BCE |
| BCE+USA | BCE | BCE+Pair-USA |
| BCE+OT | BCE | BCE+OT |
| BCE+USA+OT | BCE | BCE+Pair-USA+OT |

四组从同一个公共零步来源出发，在1600步固定终点具有相同的共同学生、EMA、优化器、scaler、RNG完整状态。**不选父best、不让S4继承S3、不增加阶段或重置优化器/LR。** 共同状态哈希进入结果并在对照汇总时检查。独立运行可重复计算预热；如果以后缓存共享终点，仍须计入每种方法的1600步来源成本，且不得只复制学生权重。

USA/OT在分支第101–200步升至0.1（USA线性，OT余弦）。OT全作物采用相同epsilon0.1、最多100次、双边残差1e-5，使用同一全文blind caption和RandAugment+Cutout强视图。此定义有意不同于历史各阶段/增强/求解器版本。

USA按同一作物/预算/种子固定复用一个新关系教师，只在该L正负对训练；教师最大20epoch、patience3。真实canonical/flip目标、来源哈希和完整准备成本必需，不能复用其他种子/预算或把教师当免费。教师Validation及描述符成本属于完整方法成本，并非32次学生选模机会的一部分。

100%标签预算没有U，仅BCE和BCE+USA，不造OT/SSL占位结果。四作物消融共264配置；BCE及完整方法与主表交叉登记但只运行一次。固定方案去重后**564个学生配置**，不是完成564组，也不包含额外教师作业。自适应补充使用另一套运行ID和预算策略，不混入这564个固定预算结果。

## 自适应补充

与固定方案相同的3200上限及LR日程；至少1600步后开始观察，paired accuracy改善达到0.005才重置耐心，连续8次验证无足够改善停止。AUROC仍可打破best选模平局，但不重置早停耐心。早停状态完整保存、恢复后不清零。该规则已统一冻结为v2定义，尚无真实数据证据证明它最优；不能根据哪种方法赢再单独更改它。

只用验证指标停止，不能跨方法比较训练loss或以伪标签通过率低为故障。提前停止的结果只进入adaptive补充，不冒称固定3200步完成。三种子均齐全时报告均值和**样本标准差（n=3）**；不是自动显著性声明，也不是独立Test表现。

## 代码入口

- [统一协议](configs/protocol.public.json)：三种子、预算、模块、优化器、评估和早停。
- [共同训练器](../../src/fair_benchmark/runner.py)：成功步计数、选模、完整断点和成本账本。
- [九种方法适配](../../src/fair_benchmark/torch_backend.py)：四个消融分支和五种SSL；复用历史纯数学/模型组件，不导入旧队列或注册器。
- [私有输入绑定](../../src/fair_benchmark/data.py)：逐文件SHA256、L/U/Validation隔离、U盲字段、教师身份和全文caption检查。
- [关系教师准备](../../src/fair_benchmark/teacher.py)：显式准备新同格教师目标，返回私有制品；不写公开仓库或伪造准入。
- [汇总检查](../../src/fair_benchmark/report.py)：拒绝重复ID、缺种子、变更数据、初始化/预热不匹配、策略混算或不完整断点计数。

无数据、无Torch、无GPU的计划/源码指纹命令（输出stdout）：

```sh
python tools/fair_benchmark.py plan --counts-only
python tools/fair_benchmark.py plan --dataset banana --budget 10 --method pairusa_ot
python tools/fair_benchmark.py plan --policy adaptive --counts-only
python tools/fair_benchmark.py source-fingerprint
```

这些只生成公开定义，没有launch enabled、真实机器身份或passed。源码指纹覆盖新共同代码及实际复用的模型/数值/缓存组件；不是历史source fingerprint。

私有结果汇总：

```sh
python tools/fair_benchmark.py summarize /private/results/seed1.json /private/results/seed2.json /private/results/seed3.json
```

只输出聚合统计，不输出样本、caption、部署身份或provenance。结果与汇总均显式标记`simulation_only`，禁止将CPU替身与真实实验混表；同格USA两个分支必须绑定相同教师目标SHA256。保存结果仍须遵守仓库结果不公开的边界。

## 部署前的真实条件

仓库仍不附合法数据、ALBEF4M/tokenizer、私有输入绑定、教师张量或GPU准入。训练API不能替代真实所有权移交、环境注册、GPU工程和独立冷缓存全状态重放。**本次没有提供自动launch/接管旧服务器的命令，不能把CPU测试当成训练准入。**

独立私有工作区必须在仓库外。`BoundPairData(root, config, binding)`要求`fair_private_inputs_v2`：当前protocol SHA256、固定data_construction_seed、各作物manifest的path/SHA256、逐个model_assets的path/SHA256。manifest沿用纯字段L/U/Validation和asset index合同，路径必须相对且不越界。USA额外要求`teachers`按`crop_budget_s<seed>`登记新目标文件path/SHA256；教师必须匹配该L/Validation/protocol，包含两视图256维目标和实测完整成本。

在**新的独立worker进程**中、真实准入完成后，`TorchBackend(config, bound_data)`与`FairTrainer(config, backend, real_provenance)`组成共同训练循环：每次`advance()`后在`evaluation_due`调用`evaluate()`；选中的完整`snapshot()`保存为best，并按相同周期保存last。`restore()`只接受完整fair-v2来源，`result()`只有在固定上限或注册早停条件满足、末次验证完成后才返回。私有调度器仍需负责GPU/唯一run锁、现场保留和原子制品保存，不能并发重复同一配置。

`real_provenance`必须绑定真实输入、源码、环境、硬件及新GPU准入回执，其中`inputs`必须等于`bound_data.input_identity`；不得填示例passed或复用旧回执。成本账本包括学生（含重试）、验证（EMA及学生诊断）、初始化、记忆库和教师上游时间；CUDA计时边界同步，前/反向pairs计数是**可训练tail的调用/尝试数，不是完整模型FLOPs估计**。时间效率比较还必须同硬件、同环境、串行无争用；异机并行时间不用于速度排名。

仅在更新前发生的OOM允许缩小物理批量并重试同一步。优化器/EMA提交中途失败可能已变动活状态，后端会拒绝再次训练、评估或保存；私有调度器须保留证据，并经授权在新worker中恢复last完整断点，不能把失败状态冒称成功或自动接管历史进程。私有输入检查同时拒绝改名/复制的相同图片跨L/U/Validation泄漏、额外CSV单元格和空U。

CPU合成检查：

```sh
python -B -m unittest discover -s tests -p "test_fair*.py" -v
python -B -m unittest discover -s tests -p "test_*.py" -v
```

CPU替身仅验证控制、数学和完整重放；实际ALBEF构造、真实输入、GPU精度/显存、所有权及3200步收敛均需独立实机验证。三作物历史方案和旧126配置索引继续原样保留，不能与fair-v2混表。
