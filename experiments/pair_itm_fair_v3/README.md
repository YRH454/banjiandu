# 公平比较 v3：固定终点、独立 Test、预热对照与三种子

这是新的实验协议及共同训练实现，不是历史结果的重新包装。训练种子为 **20260825、20260826、20260827**；数据构造种子仍为 **20260825**，不因训练种子或方法重新生成划分、负例和评估集。旧科学源码、配置索引、数据、结果与 checkpoint 不改动、不续接。v2 的协议文件保留，但其数据绑定和断点不兼容 v3。

**当前只有代码、公开计划和 CPU 合成测试，没有启动新训练，没有真实 ALBEF/GPU 准入或 v3 数值结果。** 下面的配置数量均为计划数，不是完成数；公开 CLI 不提供 launch 或旧进程接管。

## 主要评价与公平性边界

- 固定方案所有学生均为 **3200 次成功 optimizer 更新**，主结果统一使用**最后第3200步的 EMA**。Validation-best 只记录作诊断，不能作为主要模型，避免模块启用之前的 best 取代完整方法。
- 共同初始化、ALBEF/LoRA 拓扑、逻辑批量、采样流、增强、优化器和学习率日程。同作物/预算/种子内，共同骨干和 ITM 头必须一致；独有模块单独初始化并报告参数量。
- 全局80步 LR 预热，随后按3200步上限余弦衰减；分支、恢复或自适应早停不重新预热、不重置优化器、不改变 LR 上限。
- L16锚图产生32个正负 pairs；非100%预算提供U32抽样机会。BCE不使用U，其他方法的实际U使用与前反向调用不同。**相同学生步数不等于相同总算力、相同U使用量或完全相同算法机制。**
- 每100成功步验证固定400锚图/800pairs，共32次固定预算验证。EMA负责评估和停止，student仅作 Validation 诊断。独立 Test 上不再比较 student/EMA、不选 checkpoint、不搜索阈值。
- 主指标来自此前未用于训练、调参或选择的 **独立 Test**。终点 Validation 最大 macro-F1 阈值先冻结，再用于 Test；同时报告固定0.5结果、paired accuracy、AUROC、accuracy、macro-F1。

这里比较的是同一图文匹配任务上的受控 ITM 适配，不是各算法原论文分类基准的精确复现，也不保证某方法已获得最优超参数。

## 主表、模块消融和统一 BCE 起点对照

主表含 BCE、Mean Teacher、FixMatch、SoftMatch、SimMatch、FreeMatch、BCE+Pair-USA+OT，覆盖四作物的1/5/10/20/30%标签预算。五种SSL保留各自机制，从公共初始化开始：Mean Teacher自身无监督ramp不等于额外BCE阶段；各方法共同的80步**学习率预热**也不等于1600步**BCE训练起点**。

消融统一为以下方案，不从父 best 或 S3 继承，不多获得训练阶段：

| 分支 | 第1–1600步 | 第1601–3200步 |
| --- | --- | --- |
| BCE | BCE | BCE |
| BCE+Pair-USA | BCE | BCE+Pair-USA |
| BCE+OT | BCE | BCE+OT |
| BCE+Pair-USA+OT | BCE | BCE+Pair-USA+OT |

四组在1600步固定终点必须具有相同的共同student/EMA、优化器、scaler、RNG完整状态，并核对其哈希。Pair-USA/OT在分支第101–200步升至0.1；第一阶段及分支最初100步不启用模块损失。100%标签预算没有U，仅BCE和BCE+Pair-USA适用。

为检查主表中不同训练起点的影响，预先固定一个单独的 **warmup_control 补充表**：四作物 × **1%及10%** × 七主方法 × 三种子，全部采用1600步BCE+1600步各自分支。范围不是按新结果中谁胜出挑选；这个补充只检验已注册范围，不能外推为其他预算已通过相同控制。BCE与完整方法复用原运行，不重跑。

补充SSL的前1600步不处理U损失或推进算法统计；分支不重置优化器/LR。Mean Teacher按1600步分支时钟进行40%内部ramp；SimMatch在第一SSL更新之前用**当时的EMA**初始化记忆库；FreeMatch统计只随成功SSL步提交。该对照有意改变SSL训练安排，因此不能标作原算法标准训练，也不与native主表混算。

| 计划视图 | 配置数 | 说明 |
| --- | ---: | --- |
| 固定主表 | 420 | 四作物×五预算×七方法×三种子 |
| 固定消融 | 264 | 含100%预算下适用的两方法 |
| 固定BCE起点补充 | 168 | 含复用的BCE与完整方法 |
| 固定方案去重 | **684** | 表间共享角色不是额外重复；新增120个SSL起点对照 |
| 自适应独立计划 | **564** | 不包含SSL起点补充，不与固定方案合并 |

数量不包括显式准备的新关系教师。是否执行这些计划需要另外的真实准入与用户授权。

## 自适应及超参数注册

adaptive 使用相同3200步上限及全局LR日程，至少1600步后按 EMA Validation paired accuracy 停止；改善达到0.005才重置耐心，连续8次无足够改善停止。AUROC不重置早停耐心。完整恢复停止状态，终点EMA仍是唯一主要模型；不能把短运行标成固定3200步完成。

v3 注册为**不做新增超参数搜索**：每方法0新搜索试验、0新数据pilot，Test用于调参次数为0。已有历史实验，设置来源于归档数学组件和明确的ITM适配；这不是“从未看过历史结果”的声明，也不是所有方法参数均最优的证据。若以后搜索，应先定义新协议，为各方法注册相同搜索预算和合法Validation流程，不能看Test后修改本协议。

## 独立 Test 私有合同

实际数据、caption、样本ID、模型权重、教师、预测和结果都保存在仓库外。`BoundPairData(root, config, binding)`要求新`fair_private_inputs_v3`绑定，包括protocol SHA256、固定数据构造种子、各作物manifest、全部模型/tokenizer资产和`holdout_registration`的相对path/SHA256。

每作物manifest包括：

- `asset_index`、`token_guard`、`budgets["010"]["l"/"u"]`、`validation`，沿用纯字段L/U CSV合同；100%预算不依赖U。
- `test`：与L相同字段的冻结正负caption CSV的path/SHA256；`test_anchors`：可信正整数。
- `test_index`：path/SHA256，内容只含`image_id`、`image_path`、`image_sha256`，没有caption、标签或隐藏来源。

训练前校验文件digest和无标签Test成员索引，拒绝同ID或相同图片字节跨L/U/Validation/Test泄漏；不解析Test caption/标签行，不允许其进入训练batch。终点冻结后只允许一次打开Test CSV，成员及顺序必须与索引完全一致。文件digest校验并不是Test模型推理。

`holdout_registration`采用`fair_holdout_registration_v3`，绑定protocol和每作物的manifest、Validation、Test、index SHA256及数量；`origin`只能为`official_holdout`或`new_preregistered_holdout`，并要求`registered_before_training`与`test_previously_unused_for_training_or_selection`为true。

**这些JSON声明不能自己证明真实的使用历史。** 必须由可信外部记录核查此前是否用于训练、选模、调参或人工观察；不能把旧Validation改名当Test，也不能通过改SHA256洗掉此前使用记录。不能取得合法、此前未使用的独立holdout时，不进行最终Test、不把Validation诊断写成论文主要泛化结果；代码不自动重新划分或修改历史数据。

## 训练与一次性 Test API

真实准入、输入、权重及同格教师准备完成后，在新的独立worker中组合`TorchBackend(config, bound_data)`和`FairTrainer(config, backend, real_provenance)`。循环调用`advance()`，到`evaluation_due`调用`evaluate()`；按统一周期原子保存完整`snapshot()`，best如保留只作诊断。`restore()`只接受完整v3来源，不接受v2/历史断点。末次验证结束后：

1. `seal_for_test()`冻结终点EMA哈希、完整预算/停止历史、Validation指标哈希、Validation阈值、配置/协议和Test合同。
2. 私有调度器持有**唯一run锁**，核查其持久化Test意图日志。
3. `evaluate_test(persist_intent=...)`先调用回调，原子持久化其完整意图snapshot，再打开Test。真实路径未提供回调会拒绝访问。回调必须阻止日志已有意图的同一run再次访问；成功返回后再原子保存结果。

训练结束尚无Test时`result()`是`awaiting_test`；Test意图已消费但未成功是`test_intent_pending_or_failed`；完整Test成功才是`completed`。访问失败或日志回调失败不会自动获得第二次机会，同一对象不能用旧快照退还Test次数，恢复后的已登记意图也不能再次调用Test。

**进程内标记不能独自保证跨进程、旧pre-Test快照或副本全局只访问一次。** 唯一锁、原子日志、结果持久化和人工事故审计是私有部署方必须实现的职责；本仓库没有假装用一个Python对象替代这些职责，也没有发布自动部署/测试集重试CLI。协议冻结机制用于约束正常运行流程，不是对恶意代码或有权直接读取数据者的安全隔离。

## 成本、资源与统计

Pair-USA每作物/预算/种子准备一个新L-only关系教师，USA及完整方法绑定相同教师目标；最多20epoch，Validation AUROC耐心3，不接触Test。每epoch覆盖全部L，计入更新、L前向、反向、800pairs验证、canonical/flip描述符和目标提取成本。即使物理复用，完整教师成本仍计到每个使用它的方法，不冒称免费。

学生账本包含初始化、实际前反向pairs、重试、记忆库、EMA及student验证、Test推理和新worker恢复构造开销。计数是可训练tail的调用/尝试数，**不是整个模型FLOPs**；冻结前缀缓存等初始化成本主要记录为时间。CUDA计时同步，记录总/共同/独有可训练参数、教师参数和峰值allocated/reserved显存。CPU峰值为`None`，不是GPU实测值。

汇总按table/crop/budget/method/policy隔离，要求每方法三个种子齐全，检查数据、共同初始化、BCE终点与教师指纹；输出均值和样本标准差（n=3），并按相同种子给出相对BCE差值。缺方法标为不完整对照。没有自动显著性声明；结果结构通过检查也不能自动认证科学有效性。时间效率比较要求对应对照方法及三种子均同硬件、同环境、串行无争用，不能用异机并行墙钟时间排名。

## 只读命令与测试

以下命令只输出stdout，不需要数据、Torch或GPU，也不会启动训练：

```sh
python tools/fair_benchmark.py plan --counts-only
python tools/fair_benchmark.py plan --table main --dataset banana --budget 10 --method pairusa_ot
python tools/fair_benchmark.py plan --table warmup_control --dataset banana --budget 10 --method fixmatch
python tools/fair_benchmark.py plan --policy adaptive --counts-only
python tools/fair_benchmark.py source-fingerprint
```

私有结果汇总默认只接受完整独立Test；无Test必须显式请求Validation诊断，不能写作论文Test表：

```sh
python tools/fair_benchmark.py summarize /private/results/seed1.json /private/results/seed2.json /private/results/seed3.json --table main
python tools/fair_benchmark.py summarize /private/results/seed1.json /private/results/seed2.json /private/results/seed3.json --view validation_diagnostic --table ablation
python -B -m unittest discover -s tests -p "test_*.py" -v
```

源码入口：[协议](configs/protocol.public.json)、[训练控制](../../src/fair_benchmark/runner.py)、[九方法后端](../../src/fair_benchmark/torch_backend.py)、[私有绑定](../../src/fair_benchmark/data.py)、[教师准备](../../src/fair_benchmark/teacher.py)、[评价检查](../../src/fair_benchmark/evaluation.py)、[汇总](../../src/fair_benchmark/report.py)。汇总不输出样本、caption或私有部署provenance，仍须遵守[公开上传边界](../../docs/PUBLIC_UPLOAD_POLICY.md)。

CPU合成测试只验证控制、数学、输入合同、完整状态重放和负向边界；小模型、假文件字节及阶段边界probe不是真实数据/GPU执行证据。真实ALBEF初始化、GPU精度/显存、独立冷缓存重放、合法独立Test、所有权注册和3200步收敛仍需另行验证。
