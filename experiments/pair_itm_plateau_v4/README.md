# v4：BCE 验证平台期 → 模块适配 → 等追加步数 BCE 对照

这是本仓库默认的两阶段训练协议，替代固定1600步BCE起点。主实验与消融均从**同格的同一个BCE平台期完整检查点**开始；第二阶段保留有标签BCE，加入各方法自身的损失，并分别记录原始、加权及总损失。数据构造、模型初始化与训练随机化设置统一由[公开协议](configs/protocol.public.json)定义。

v4 的阶段规则、私有输入合同和断点格式独立登记。v2/v3与历史实现保留用于溯源；v3接口需显式使用`--version v3`，其断点、数据合同和结果不与v4混用。

## 两阶段规则

1. 第一阶段只训练图文对匹配BCE，从同一官方ALBEF初始化开始。每100次成功更新验证固定400锚图/800pairs，只用**EMA Validation paired accuracy**判断平台期。最少800步后开始计耐心，改善达到0.005才重置，连续8次没有足够改善结束；AUROC只记录best诊断，不重置耐心。
2. BCE安全上限6000步。达到上限但未检测到平台期，状态为`bce_cap_requires_review`，**不能自动分叉、不能称作收敛、不能进入论文平台期Test参照**；需先审查并另行注册新协议，不能临时提高上限或挑best点。
3. 平台期时冻结**终点student、终点EMA、optimizer/scaler/RNG等完整状态和来源回执**，不选历史best。终点EMA是评价模型；适配继续各自的终点student/EMA，而不是悄悄将best或EMA覆盖student。BCE父状态和输入/初始化SHA256在所有分支核对。
4. 第二阶段包含BCE继续训练、BCE+Pair-USA、BCE+OT、BCE+Pair-USA+OT，以及五种SSL的BCE+自身损失。各分支统一**新optimizer、scaler、阶段LR及物理批量**，继承同一父student/EMA和RNG，独有头按同seed规则冷初始化；不继承其他模块分支。早停耐心独立重置。
5. 第二阶段同格共同最少步数为`max(1600, ceil(L锚图数/16)+100)`，安全上限4000。共同grace覆盖Mean Teacher内部ramp和SimMatch实例预热；若grace超上限拒绝该格，不能为某方法缩短预热。相同100步Validation、0.005改善、8次耐心。触及上限标`phase_cap_exhausted`，不冒充平台期。

最少步数与安全上限是明确的工程/方法约束，**不是保证收敛、最优超参数或已排除审稿质疑的证据**。这些规则必须在新实验前冻结，不可看Test后修改。验证“平台期”也不等于数学意义的收敛。

每阶段LR重新80步预热，随后余弦衰减至对应安全上限，最低倍率0.1而非0，避免因固定日程降到零形成假平台。所有适配及匹配BCE的LR都按4000步日程；不能按各自提前停止步数重新缩放。

USA/OT在适配第101–200步升至0.1。Mean Teacher按4000步适配上限计算40%内部ramp。SimMatch记忆库在**导入父状态后的第一适配更新前**用当前EMA初始化，初始化成本单独记录；实例预热以适配时钟计算。FreeMatch/SoftMatch统计只随适配更新推进。阶段1/2采样流及增强时钟分离，所有同格分支和匹配BCE共享阶段2采样/增强。

## 如何区分模块收益和继续训练收益

每个非BCE模块分支停止后、访问Test前，导出其Validation停止回执。另一个BCE对照从**同一父状态**出发，按相同的第二阶段LR/采样/增强，跑到该模块实际追加步数，忽略自身早停；不得传任意整数、换种子/父状态或用Test选择匹配步数。匹配对照仍每100步验证，但不据此改变终点。

| 对比 | 可解释的量 |
| --- | --- |
| 自适应BCE − 冻结BCE平台期参照 | 继续BCE训练的增益 |
| 自适应模块 − 自适应BCE | 完整自适应方案的差异，步数可能不同 |
| 模块 − 对应等追加步数BCE | 在相同追加学生更新数下的模块增量 |
| 模块 − 冻结BCE参照 | 总体增益，不能全部归因于模块 |

冻结BCE是单独参照，不能和训练成本不同的适配分支混作等预算排名。相同追加学生步数**仍不是等总算力**：teacher、记忆库、实际U前反向和参数量都不同。匹配BCE可能尚未到自身平台期，这恰好是“相同步数”的控制，不冒称双方同等收敛。

四作物×五主预算×七主方法×三种子，主表420配置；四组消融含100%下适用的BCE/USA，264配置，角色共享不重跑。全计划去重后为 **72个BCE父格＋564个自适应分支＋492个匹配BCE＝1128个逻辑配置**。这不是完成数，也不包含教师准备次数。匹配目标执行前未知；只允许私有协调器在完整回执和各终点状态均保存时物理复用相同BCE轨迹，不能省略逻辑配置、成本或按匹配对照表现挑模块。

## 私有输入与实际训练 API

真正训练前需另行准备合法输入、此前未用于选择的独立Test、官方模型/tokenizer资产、同格L-only新关系教师及GPU工程准入。数据、权重、教师、回执、日志、结果、checkpoint都放在公开仓库外。v4使用`PlateauPairData`：

- `fair_private_inputs_v4`绑定protocol SHA256、固定构造seed、作物manifest、全部模型/tokenizer资产及holdout登记。
- `fair_holdout_registration_v4`绑定每作物manifest/Validation/Test/index哈希及Test数量；要求训练前登记、此前未用于训练或选择，来源仅official/new preregistered holdout。其真实使用历史由可信外部记录核查，JSON声明不能自证；不能把旧Validation重命名为Test。
- manifest的L/U/Validation/Test/index字段和逐文件校验沿用[v3数据合同](../pair_itm_fair_v3/README.md#独立-test-私有合同)。U只含blind字段；L/U/Validation/Test不能同ID或同图像字节重叠。训练前只检查Test字节hash及无标签成员索引，不解析Test caption/标签。
- USA目标为`fair_pairusa_targets_v4`，同作物/预算/seed的USA与完整方法复用一个L-only教师。使用`fair_benchmark.teacher.prepare_teacher_targets(cfg, data)`显式准备并在仓库外保存/绑定。最多20epoch、Validation AUROC耐心3，完整教师/描述符成本不视为免费；不允许历史v3教师改名。

核心类支持真实Torch训练，不只是计划器。下面是**私有协调器**的调用顺序；`input_root`、`binding`、真实provenance和原子存储/日志回调须由部署方准备，公开CLI不伪造准入、不自动启动任何训练：

```python
from plateau_benchmark.spec import make_config
from plateau_benchmark.data import PlateauPairData
from plateau_benchmark.torch_backend import PlateauBackend
from plateau_benchmark.runner import PlateauTrainer, train_until_stop

parent_cfg = make_config("banana", 10, "bce", 20260825, stage="bce")
parent_data = PlateauPairData(input_root, parent_cfg, binding)
parent_backend = PlateauBackend(parent_cfg, parent_data)
parent_trainer = PlateauTrainer(parent_cfg, parent_backend, parent_provenance)
train_until_stop(parent_trainer, persist_checkpoint=save_private_atomic,
                 log_record=append_private_jsonl)
parent_bundle = parent_trainer.export_parent()  # 只有验证平台期可以导出
save_parent_bundle_private_atomic(parent_bundle)

# 以下分支各在独立worker中运行，从同一完整bundle导入。
module_cfg = make_config("banana", 10, "ot", 20260825)
module_data = PlateauPairData(input_root, module_cfg, binding)
module = PlateauTrainer(module_cfg, PlateauBackend(module_cfg, module_data),
                        module_provenance, parent=parent_bundle)
train_until_stop(module, persist_checkpoint=save_private_atomic,
                 log_record=append_private_jsonl)
matching = module.export_match_receipt()  # 必须先于Test
save_match_receipt_private_atomic(matching)

matched_cfg = make_config("banana", 10, "bce", 20260825,
                          stage="matched_bce", match_method="ot")
matched_data = PlateauPairData(input_root, matched_cfg, binding)
matched = PlateauTrainer(matched_cfg, PlateauBackend(matched_cfg, matched_data),
                         matched_provenance, parent=parent_bundle,
                         match_receipt=matching)
train_until_stop(matched, persist_checkpoint=save_private_atomic,
                 log_record=append_private_jsonl)
```

其他主方法及消融同样构造；不能仅跑上面OT示例就称完成主表。`provenance["inputs"]`必须等于该worker的`data.input_identity`；真实worker的`source_sha256`必须是本次`source_fingerprint()["source_sha256"]`，父与所有分支代码指纹一致。计时比較还需真实硬件/环境SHA256、串行无争用证明，以及`full_pipeline_same_hardware_serial_execution=true`的可信记录（含教师等所有计账操作）。父阶段硬件/环境和当前阶段不一致时不做时间排名。每个新worker持有自己数据/模型，不在同一Python进程并发调用全局RNG；对照不与其他GPU训练争用。

`snapshot()/restore()`保存预算历史、父与匹配回执、student/EMA/optimizer/scaler/RNG、算法统计/记忆库、账本和Test状态。先按原cfg创建新worker（分支重新导入同一父bundle），再`restore(完整v4快照)`；不能加载v3/历史断点或更换match目标。预提交OOM可在同一步按16→8→4微批重试；optimizer/EMA部分提交失败会污染live worker，拒绝继续或保存，须从最后完整checkpoint构造新worker，不能将部分提交计为成功。

逐步记录`losses_raw/weighted`：BCE、USA、无标签一致性或OT、SimMatch instance、SAF，以及总loss、权重、筛选U数量/平均mask、总梯度范数、阶段步数和物理批量。USA/OT延迟期间的零权重不是模块生效证据；不同损失数值大小也不等于贡献大小。模块因果贡献以匹配对照为主，不由训练loss单独证明。

## 独立 Test 与账本

先完成/持久化所有父、模块、自适应BCE及匹配控制训练并冻结全组目标，再进入Test。`seal_for_test()`锁定终点EMA、Validation阈值、整个停止历史、输入合同和父/阶段零步证据；`evaluate_test(persist_intent=...)`先原子登记意图再打开Test，真实路径无回调拒绝访问。只用EMA，不比较student、不搜索阈值或选步数；同时报告Validation锁定阈值及固定0.5。

部署方必须用**唯一run锁＋持久化一次性意图日志**校验全组训练目标已冻结，拒绝旧pre-Test快照、重复run或副本重试。对象标记不能独自保证跨进程全局一次访问，也不能替代真实准入。回调/访问失败不退款；训练、匹配回执生成与Test均不提供自动事故重试。BCE父bundle在平台期验证时就冻结，之后的Test/恢复成本不重写其训练起点。

每个逻辑分支`compute = upstream_bce_compute + phase_compute`，即使父训练物理复用也计完整BCE训练成本；USA额外计完整教师成本。冻结BCE自身Test成本归其参照，分支自身Test成本归各分支，不混入父训练。记录student/共同/独有参数、教师资源、CUDA峰值及恢复成本；CPU峰值为None。计数不是整个模型FLOPs，不声明等算力或等U使用。

汇总按阶段/角色分离，要求三种子齐全，核对同父、同阶段零步状态、教师、输入与匹配来源；缺对照和缺方法明确标注。输出n=3均值、样本标准差及配对差值，不自动宣称统计显著，也不输出私有provenance或逐样本内容。自适应主方法和消融为受控ITM适配，不能宣称原论文分类任务原样复现。没有新增HPO/pilot；历史实验已存在，并非“从未参考旧结果”。

## 无数据的只读命令

```sh
python -B tools/fair_benchmark.py plan --counts-only
python -B tools/fair_benchmark.py plan --stage bce --dataset banana --budget 10
python -B tools/fair_benchmark.py plan --stage adaptation --table main --method fixmatch
python -B tools/fair_benchmark.py plan --stage matched_bce --match-method ot
python -B tools/fair_benchmark.py source-fingerprint
python -B tools/fair_benchmark.py --version v3 plan --counts-only
python -B tools/fair_benchmark.py summarize /private/seed1.json /private/seed2.json /private/seed3.json
python -B tools/fair_benchmark.py summarize /private/seed1.json /private/seed2.json /private/seed3.json --view validation_diagnostic
python -B -m unittest discover -s tests -p "test_*.py" -v
```

默认汇总必须有Test，Validation-only只能显式诊断；需要提供各方法/对照三种子结果文件，不能把plan喂给summary。源码指纹只是代码证据，不是GPU准入。CPU测试含微型模型真实更新、阶段边界probe和多数更新被模拟的快速控制器；它们不代表完整ALBEF训练、收敛或真实GPU精度重放。真实GPU、合法独立Test、所有权/一次访问登记和方法表现仍需另行验证。

入口：[协议](configs/protocol.public.json)、[训练循环](../../src/plateau_benchmark/runner.py)、[Torch后端](../../src/plateau_benchmark/torch_backend.py)、[停止预算](../../src/plateau_benchmark/budget.py)、[回执](../../src/plateau_benchmark/contracts.py)、[数据](../../src/plateau_benchmark/data.py)、[汇总](../../src/plateau_benchmark/report.py)。遵守[公开上传白名单](../../docs/PUBLIC_UPLOAD_POLICY.md)，不上传实验产物。
