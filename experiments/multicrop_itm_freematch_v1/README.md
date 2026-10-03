# 四作物 FreeMatch（第五算法，20×2600成功步）

机制参考[FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning, ICLR2023](https://arxiv.org/abs/2205.07246)，USB固定commit1ef4cbebcc0b368158315aeb425053858cf6c845。[参考代码](references/usb/freematch)及MIT许可保留。这是图文pair二类适配，不是原图像分类论文原样复现。

四作物×五预算×seed20260825，共20唯一run。配置见[configs/runs](configs/runs)，共同定义见[protocol.public.json](configs/protocol.public.json)；SAT/SAF见[code/freematch_algorithm.py](code/freematch_algorithm.py)，正式训练器见[code/train.py](code/train.py)。

SAT三个统计初始0.5、EMA0.999，以current weak confidence均值，无quantile或0.95 clipping；hardlabel且除全32U。SAF为作者negative CE/histogram modulation、epsilon1e-12、系数0.01，按全32计算。统计只在成功optimizer后提交一次，同一步失败重试不推进。原bounded-memory两遍strong logit/gradient重放产生每run额外83200 strong前向访问，不能声称与其他方法等计算。

当前实际加速版本是[orchestration/speed_v4](orchestration/speed_v4)：reference、激活checkpoint开启，至多2个不同完整run并行；每run独立cold/gate，wave完全idle才串行工程准入。原serial dispatcher也保留源码历史。retained候选未准入，v5未部署；目录中候选源码的存在不表示任何方案科学或工程通过。原failed证据/实际passed/nonce/注册/部署身份只私有保留，不公开或补造。不得自行运行/更换旧服务器controller。

固定2600成功optimizer/SAT步、26次400图800pairs EMA Validation、不Test；全文blind caption及随机异病例弱负例、公共34键零步值、逻辑L16/U32和原FP16/EMA/LR/OOM规则均不变。成功步、工程重放步、额外前向成本分别计算。真实Linux Python3.12.13/torch2.5.1+cu121，不伪装成Windows来源。

公开副本只去部署标识/LF规范化，原源及配置hash映射在[源码清单](../../docs/source_manifest_three_hosts_20261004.json)。真实profile、GPU UUID/host、环境绑定/输入/权重/passed缺省，因此不是直接可运行或可恢复历史断点的项目。候选CPU测试/源码归档不替代真实GPU和正式完整终验。
