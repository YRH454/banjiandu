# 公开上传白名单

该仓库是公开源码归档，上传以明确文件清单为准，不递归打包服务器项目，也不使用强制加入绕过忽略规则。

| 类别 | 处理 |
| --- | --- |
| 训练/模型/指标源码、无真实数据的CPU测试 | 可上传；保留历史代码，新增实验独立目录 |
| 科学协议与性能定义 | 上传公开子集，删除SSH/GPU/部署字段；不是可直接启动的私有配置 |
| 通用HTML生成器 | 上传源码；生成报告留在仓库外 |
| 上游参考代码、许可、源码SHA256 | 保留来源与许可，标明换行规范化/目录映射 |
| 图片、caption、固定配对/划分表、样本ID、U隐藏标签 | 不上传 |
| 权重、step-zero初始化张量、teacher/记忆向量、checkpoint、tokenizer资产 | 不上传 |
| 原始运行配置、模型/数据资产回执、审计JSON、benchmark原始制品 | 不上传；2026-10-04授权新增126份去部署化科学配置视图，独立标记public且不复用原运行fingerprint |
| result/status/summary、逐样本预测、训练JSONL、日志、HTML/CSV结果 | 不上传 |
| SSH凭据、端点、GPU UUID、环境秘密、进程状态、缓存、安装运行时 | 不上传 |

服务器`assets/data/outputs/reports/logs/audit/runtime/pip-cache/incoming/locks`不是公开上传源。仅从`audit`人工挑选无部署值的性能**源码**，放在`archive`并登记原位置；审计制品不跟随。

本轮也不上传会打包私有制品的`bundle.py`，不发布失效benchmark作为提速证据。所有结果、checkpoint和服务器源码保持原状。既有仓库里的历史本机路径按原归档约定保留；这不代表批准新增任何凭据或服务器部署值。

提交前必须检查显式暂存路径、所有JSON字段、源码指纹、文件大小、凭据/部署值和`git diff --cached`。`.gitignore`是辅助防线，不是对文件内容的授权。不得使用`git add .`或`git add -f`把运行副本整体加入。
