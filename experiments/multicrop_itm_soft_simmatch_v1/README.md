# 四作物 SoftMatch / SimMatch 图文匹配源码归档

这是服务器实验的**源码、科学定义和执行版本归档**，不是新的训练任务，也不是开箱即跑的发布包。原有苹果 G1–G4 消融保持不变。本目录不公开实验分数、HTML结果报告、图片、caption、样本ID、权重、逐组运行配置或审计制品。

## 实验定义

| 项目 | 定义 |
| --- | --- |
| 任务 | ALBEF 图文匹配二分类（ITM），不是作物/病害分类 |
| 配置网格 | 苹果、木薯、水稻、香蕉 × 1%、5%、10%、20%、30% × 两算法，共40配置 |
| 随机种子 | `20260825`，单训练种子，不提供多种子显著性结论 |
| 更新预算 | SoftMatch 每组2200次成功更新；SimMatch 每组2400次；不声称等算力 |
| 共同初始化 | 官方 ALBEF4M + 精确登记的34个 step-zero 可训练张量；不加载 BCE best，不增加辅助预训练 |
| 模型 | 384×384；视觉 LoRA 层10–11，rank4/alpha8；融合 cross-attention 层6–11，rank8/alpha16；Pair-USA关闭 |
| 逻辑批量 | 16个不同L锚图形成32正负图文对；另抽32个U图文对；初始物理批量16，冻结前缀批量4 |
| 文本与增强 | 完整 blind caption、不截断；每数据集 token guard 来自未公开manifest。弱增强为384方形resize+确定性翻转；强增强叠加RandAugment与Cutout，不做crop |
| 评估 | 每数据集固定400验证锚图/800图文对；两算法均用EMA；不做Test推理 |
| 选择 | best按paired accuracy优先、AUROC其次；分类阈值在验证集最大化macro-F1，平局靠近0.5；另记录固定0.5 |

优化器、学习率、EMA、算法参数详见 [`configs/protocol.public.json`](configs/protocol.public.json)。该文件是移除部署字段后的科学子集，**不是原始protocol文件，不保有历史配置哈希，不可直接复制为旧运行配置**。

SoftMatch使用当前学生弱视图的原始argmax伪标签；均匀分布对齐仅参与置信度/Gaussian权重。U损失按全部32对归一化，不只除以被选中的对数。SimMatch使用EMA弱视图、pair-ID记忆库及128维投影；同一图的正负caption是两个记忆节点。记忆库仅以L训练对做无优化器前向初始化，不读取Validation/Test；instance warmup为`ceil(L锚图数/16)`且包含在2400步内，初始化前向成本不能忽略。

算法机制参照 Microsoft USB 指定提交，并做了二分类图文匹配适配；**不等于官方图像分类任务的精确复现**。参考代码仅供核对，不是完整`semilearn`包，也不会被本训练器导入。

## 文件与版本映射

- `code/`：12份原始Python源码，保留服务器执行字节。`base_train.py`虽保留Mean Teacher/FixMatch的历史基类逻辑，本轮仅由`train.py`扩展为SoftMatch/SimMatch；不能据此宣称跑了其他算法。
- `verification/`：原始初始化导出及**首次串行部署阶段**核验脚本。导出器需要历史参考环境和资产；`verify_running.py`只核验当时指定首组，并会写审计回执，不是最终40组验收或通用只读监控器。
- `archive/speedup_20261002/`：原始性能worker、缓存、接管调度器和CPU缓存测试；对应原项目`audit/speedup_20261002/`。只移动归档路径，不改算法代码。
- [`../../src/albef_ssl`](../../src/albef_ssl)：复用仓库已有核心实现，7份文件与服务器`core/albef_ssl`逐字节相同；没有复制或覆盖历史模型。
- `references/usb/`：指定上游提交的机制参考；仅统一CRLF为LF并补末尾换行，原字节哈希也登记在manifest。MIT条款在[`USB_LICENSE.txt`](../../third_party/USB_LICENSE.txt)，其正文已核对上游。ALBEF许可继续使用原仓库文件。
- [`source_manifest.json`](source_manifest.json)：33份源文件/依赖清单的公开路径和SHA256，包含上述7份共享核心；不包含数据指纹、逐样本信息或运行审计JSON。

`code/bundle.py`不发布：它原用于内部迁移，会打包审计、结果和初始化权重，不适合作为公开仓库打包入口。旧benchmark混用优化器状态的比较不能作有效加速证据，因此不发布`benchmark_speed.py`；部署绑定的历史监控脚本也不发布。

## 性能策略与科学协议分开

[`configs/execution.public.json`](configs/execution.public.json)描述`gpu_parallel_perf_20261002_v1`：按算法分两条队列，组内串行，最多两个GPU worker；8线程解码预取、CPU特征LRU、关闭激活重算；物理批量仍16、前缀批量仍4。数据、逻辑批量、种子和目标更新数不变。更大batch未通过一致性容差，未纳入采用策略。

原科学协议记录`activation_checkpointing=true`及串行部署，优化是另登记的执行层，不覆写原科学配置。切换时原SoftMatch worker被接管而非重启，部分组使用原始执行层，后续组使用性能worker；**不能把这轮视作40组统一的执行代码指纹**，也不能将并行组耗时相加当作整体壁钟，或直接据此宣称算法速度优势。

`parallel_dispatcher.py`是**历史接管工具，不是通用新实验启动器**；它检查旧调度器PID/start_ticks并发信号，依赖私有`bench_v2/benchmark.json`和审计状态。即使`--dry-run`也会登记policy和锁。不要对已完成项目重跑`--takeover-original`。公开执行描述不是`accepted_policy.json`回执，不能拿它绕过新的GPU一致性与完整重放gate。

## 复现边界与CPU检查

需要另行获得并登记合法数据、完整caption、冻结L/U/Validation表、asset index、manifest、官方ALBEF4M、本地tokenizer、精确34键初始化张量、模型资产回执和逐组配置。U loader只接受公开的pair字段，不接受隐藏标签/来源列。这里“公开字段”指训练接口权限，不代表数据可公开分发。

若在新的独立私有工作区组装运行环境，须将共享`src/albef_ssl`放入其`core/albef_ssl`，恢复原相对目录，并为新部署生成真实配置及回执；不要修改服务器旧目录、复制旧哈希或覆盖旧checkpoint。公开归档删除了内部打包工具、运行配置和回执，来源集合不同，因此**不能直接恢复历史checkpoint，也不能保证生成历史实验数值**。

`configs/requirements.txt`保留服务器Python依赖快照；基础运行环境还包括Python3.11、PyTorch`2.5.1+cu124`及torchvision`0.20.1+cu124`，需相应CUDA wheel来源。它不是跨机器依赖锁；本次上传没有做干净安装或新的GPU训练验证。

从仓库根目录进行不依赖私有数据的检查：

```text
python tools/verify_source_archive.py
python experiments/multicrop_itm_soft_simmatch_v1/code/unit_tests.py
python experiments/multicrop_itm_soft_simmatch_v1/archive/speedup_20261002/test_perf_cache.py
python -m unittest discover -s tests -p "test_*.py"
```

这些检查仅覆盖哈希、CPU数值/状态/缓存、指标和报告渲染，不替代真实数据与GPU gate。

## 私有结果的阅读工具

[`tools/render_result_report.py`](../../tools/render_result_report.py)只发布生成器源码，没有嵌入本轮数值。它验证40个唯一配置及混淆矩阵一致性，生成离线中文HTML，支持筛选、阈值切换、详情和CSV导出；只提取聚合指标，不嵌入provenance、样本、caption或部署信息。

命令`python tools/render_result_report.py <私有summary.json>`将UTF-8 HTML输出到stdout；将输出保存到Git仓库之外。HTML和从页面导出的CSV仍属于实验结果，不能上传。checkpoint选择和阈值选择都发生在同一验证集，报告不能替代独立测试表现。
