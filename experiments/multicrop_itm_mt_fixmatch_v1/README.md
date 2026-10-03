# 四作物 Mean Teacher / FixMatch（原40逻辑配置）

20 MT×1800成功步 + 20 FixMatch×2000成功步，seed20260825。四作物apple/cassava/rice/banana×预算001/005/010/020/030；原Windows28和独立本机12联合唯一40，不新增实验。

- 逐组参数见[configs/runs](configs/runs)；共同协议见[protocol.public.json](configs/protocol.public.json)。
- 算法/训练器见[code/train.py](code/train.py)、[code/losses.py](code/losses.py)；独立模型架构副本见[core](core)。
- 仅Validation，每100步400图/800pairs，用EMA；MT前40%MSE ramp，FixMatch阈值0.95且全32U归一。无BCE-best/USA教师初始化，公共34键零步值匹配原来源。
- 固定逻辑L16锚图/32pairs/U32，全文blind caption，随机异病例弱负例；激活checkpoint、FP16scale1024、同一步OOM16→8→4、80步LR预热余弦。
- 原Windows Python3.11.5/cu124与本机Python3.12.14/cu121不同；公开环境快照不提供安装成功或数值一致性的承诺。

这是科学源码/配置视图归档，不是直接运行器。原bundle/make_configs、私有execution_profile、authority/claim/lock状态、实际输入/权重及passed不分发。public.json剔除了原协议指纹及launch binding，公开源码可能含去标识占位路径；不得将公开hash冒称原执行fingerprint或修改旧断点接受它。公开CPU测试不启动训练或GPU。

[全部三机配置索引](../../docs/THREE_HOST_EXPERIMENTS_20261004.md)。上游ALBEF/Apache源码许可见[third_party](../../third_party)。
