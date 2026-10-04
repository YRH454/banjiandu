# 图文匹配实验：BCE 平台期与半监督学习

面向苹果、木薯、水稻和香蕉的图文匹配二分类实验，包含 BCE、Pair-USA、OT，以及 Mean Teacher、FixMatch、SoftMatch、SimMatch、FreeMatch 的统一训练与对照代码。这里的 BCE 判断图像与文本是否匹配，不是病害多标签分类。

仓库保存源码与实验定义；图片、caption、配对表、模型权重、教师目标、预测、日志和实验结果保存在仓库外。上传规则见 [`docs/PUBLIC_UPLOAD_POLICY.md`](docs/PUBLIC_UPLOAD_POLICY.md)。

## 默认实验配置

当前默认协议为 **`pair_itm_plateau_v4`**：[公开配置](experiments/pair_itm_plateau_v4/configs/protocol.public.json) · [完整训练说明](experiments/pair_itm_plateau_v4/README.md)。模型、优化器、增强、停止规则和实验网格以该配置为准。

训练分为两个自适应阶段：先用 BCE 训练到验证平台期，再从同格的共同 BCE 终点派生主实验和消融分支。第二阶段继续保留 BCE，并加入各方法的模块损失；每项原始损失、加权损失和总损失分别记录。训练不再采用固定的“1600 步起点 + 1600 步续训”。

| 实验范围 | 作物 | 标注预算 |
| --- | --- | --- |
| 主实验 | 苹果、木薯、水稻、香蕉 | 1%、5%、10%、20%、30% |
| 消融实验 | 同上 | 1%、5%、10%、20%、30%、100% |

100% 预算没有无标签池 U，仅使用 BCE 和 BCE + Pair-USA；不生成 OT 或其他半监督方法的占位结果。图文负例采用固定异病例完整 caption 错配，属于可能包含假负例的弱负例。

## BCE 起点与消融分支

第一阶段从公共 ALBEF 初始化出发，只训练图文匹配 BCE。达到验证平台期后冻结终点 student、EMA、优化器、scaler 和随机状态，作为所有同格分支的共同父检查点。采用平台期终点，不采用历史 best checkpoint。

| 组别 / 用途 | 第二阶段损失 | 初始化与训练规则 |
| --- | --- | --- |
| 平台期参照 | 不追加训练 | 冻结的 BCE 终点 EMA |
| G1：BCE 续训 | BCE | 从共同父检查点继续，自适应停止 |
| G2：Pair-USA | BCE + Pair-USA | 同一父检查点，自适应停止 |
| G3：OT | BCE + OT | 同一父检查点，自适应停止 |
| G4：完整方法 | BCE + Pair-USA + OT | 同一父检查点，自适应停止，不继承 G3 |
| 等追加步数对照 | BCE | 从同一父检查点训练到对应模块的实际追加步数 |

各分支继承相同的父 student / EMA 与随机状态，并统一重建阶段优化器、scaler 和学习率日程。USA 与完整方法复用同格的 L-only 关系教师；OT 使用 U 的图文视图，不读取 U 的隐藏匹配标签。

## 半监督主实验

主表包含 BCE 续训、以下五种半监督方法，以及 BCE + Pair-USA + OT。所有方法均从共同的 BCE 平台期父检查点出发，第二阶段保留有标签 BCE。

| 方法 | 附加模块 |
| --- | --- |
| Mean Teacher | EMA 教师一致性损失 |
| FixMatch | 置信度筛选的硬伪标签一致性损失 |
| SoftMatch | 自适应软权重伪标签损失 |
| SimMatch | 语义一致性与实例一致性损失 |
| FreeMatch | 自适应阈值伪标签损失与 SAF 正则项 |

## 自适应停止与主要参数

| 配置项 | 当前设置 |
| --- | --- |
| 验证间隔 | 每 100 次成功更新；固定 400 锚图 / 800 图文对 |
| 平台期监控 | EMA Validation Paired Accuracy |
| 改善门槛 / 耐心 | 0.005；连续 8 次验证无足够改善 |
| BCE 阶段 | 最少 800 步；安全上限 6000 步 |
| 模块阶段共同最少步数 | `max(1600, ceil(L锚图数 / 16) + 100)` |
| 模块阶段安全上限 | 4000 步；覆盖统一的内部预热约束 |
| 学习率 | 每阶段 80 步预热，随后余弦衰减；最低倍率 0.1 |
| LoRA / head 学习率 | `1e-4` / `2e-4` |
| 图像大小 / EMA 衰减 | 384 / 0.999 |
| 逻辑批量 | L：16 正锚图 / 32 配对；无标签分支 U：32 配对 |
| 物理微批 | 16 对；同一步 OOM 时可降至 8 或 4，逻辑批量不变 |
| USA / OT 权重 | 模块阶段第 101–200 步升至 0.1 |

触及安全上限会单独记录，不将其视为验证平台期；BCE 未满足平台期条件时，不自动进入模块阶段。AUROC 和 Validation-best 仅用于诊断，不替代终点评价模型，也不重置平台期耐心。

## 区分继续训练与模块收益

每个模块停止后，先用 Validation 停止回执确定其追加步数，再构造对应的等追加步数 BCE 对照。对照采用相同父状态、阶段学习率、采样和增强，不能用 Test 表现决定匹配步数。

| 对比 | 解释 |
| --- | --- |
| BCE 续训 − 平台期参照 | 继续训练的收益 |
| 模块分支 − 对应等追加步数 BCE | 相同追加学生更新数下的模块增量 |
| 模块分支 − 自适应 BCE 续训 | 完整自适应训练策略的差异 |
| 模块分支 − 平台期参照 | 总体增益，包含继续训练因素 |

等追加学生步数不等于等总算力。父阶段、教师、记忆库、U 前反向、恢复与验证等成本分别计账；时间比较需要同硬件、同环境和无并行争用。

主评价使用终点 EMA 的独立 Test 指标。模型、停止回执和 Validation 阈值先冻结，再进行一次性 Test 访问；同时报告锁定阈值与固定 0.5，Test 不用于选模型、阈值或训练步数。

## 代码与配置入口

| 入口 | 用途 |
| --- | --- |
| [`experiments/pair_itm_plateau_v4/configs/protocol.public.json`](experiments/pair_itm_plateau_v4/configs/protocol.public.json) | 默认实验协议 |
| [`src/plateau_benchmark`](src/plateau_benchmark) | 两阶段训练、平台期停止、共同父状态、匹配回执与汇总 |
| [`src/fair_benchmark`](src/fair_benchmark) | 图文数据合同、模型后端、算法组件、教师与独立评价 |
| [`src/albef_ssl`](src/albef_ssl) | ALBEF / LoRA 模型实现 |
| [`tools/fair_benchmark.py`](tools/fair_benchmark.py) | 查询配置、查看代码指纹与汇总私有结果 |
| [`tests`](tests) | CPU 与合成输入检查 |

查询当前配置，不启动训练：

```sh
python -B tools/fair_benchmark.py plan --counts-only
python -B tools/fair_benchmark.py plan --stage bce --dataset banana --budget 10 --counts-only
python -B tools/fair_benchmark.py plan --stage adaptation --table main --method fixmatch --counts-only
python -B tools/fair_benchmark.py plan --stage matched_bce --match-method ot --counts-only
python -B tools/fair_benchmark.py source-fingerprint
```

训练 API、私有输入绑定及检查点恢复方式见 [v4 训练说明](experiments/pair_itm_plateau_v4/README.md)。基础模型、tokenizer、训练数据和合法独立 Test 由使用者另行准备，不能把旧 Validation 重命名为 Test。

代码检查：

```sh
python -B -m unittest discover -s tests -p "test_*.py"
```

旧版实验实现与配置索引保留在 `experiments` 历史目录和 [`docs/THREE_HOST_EXPERIMENTS_20261004.md`](docs/THREE_HOST_EXPERIMENTS_20261004.md)，仅用于溯源。旧 v3 接口需显式指定 `--version v3`；不同协议的配置、断点和结果不混用。上游许可见 [`third_party`](third_party)。
