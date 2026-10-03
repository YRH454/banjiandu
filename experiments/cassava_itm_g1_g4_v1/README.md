# cassava S1–S4 单种子消融（22学生阶段）

预算1/5/10/20/30%各S1–S4，100%仅S1/S2，共22。每阶段1600成功步，seed20260825，只固定400图/800pairs Validation，无新Test。教师成本另计。

S1从ALBEF4M初始化；S2从本预算S1-best启动BCE+Pair-USA；S3从S1-best启动BCE+OT；S4从S3-best启动BCE+Pair-USA+OT，复用本预算S2关系教师。阶段初始化不是恢复前一阶段optimizer。完整blind caption和固定异病例caption弱负例，逻辑L16锚图32pairs/U32，不使用私有U标签。

[逐组科学配置视图](configs/runs)保留原模型/阶段/批次/优化器/辅助权重/ramp/OT参数，并删去私有输入、教师/父checkpoint和状态绑定；[原计划公开视图](configs/plan.public.json)不包含当前完成状态。阶段配置和同名数据池不可跨作物替换。

原5%四阶段只读保留，其OT双边门槛1e-5；后18个fast版本为1e-4、物理16，同一步OOM退8/4。其他参数以逐组视图为准。原12个S1/S2 checkpoint未保存RNG/best_snapshot，不补造，不声称完整可恢复。三项候选工程优化未部署；fast不代表已证明显著提速。

木薯全文guard384，固定嵌套L/U及负例；旧5%与fast执行版本不可强并为完全相同协议。

公开源码保留模型、Pair-USA、OT、阶段包装和数据准备逻辑；去标识绝对路径不是真实部署值。没有上传实际数据/配对/样本ID/教师/checkpoints/指纹/失败payload/锁或result。公开视图不是原运行配置，不能用于接受旧fingerprint或从父best清零冒称恢复。模型架构历史依赖见仓库src/albef_ssl；未经独立绑定/测试，不声称开箱复现。

[三机索引](../../docs/THREE_HOST_EXPERIMENTS_20261004.md) · [原/公开源hash清单](../../docs/source_manifest_three_hosts_20261004.json)
