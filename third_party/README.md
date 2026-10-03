# 第三方源码

`src/albef_ssl/vendor/albef/` 中的 ViT、xBERT 及 BERT 配置源自 [Salesforce ALBEF](https://github.com/salesforce/ALBEF)。原许可证保存在 [`ALBEF_LICENSE.txt`](ALBEF_LICENSE.txt)。本仓库的实验代码对上游实现做了集成与 LoRA 适配；请勿将上游代码或权重误称为本项目原创。预训练权重本仓库未分发。

四作物 SoftMatch / SimMatch 的机制参考来自 [Microsoft USB / Semi-supervised-learning](https://github.com/microsoft/Semi-supervised-learning/tree/1ef4cbebcc0b368158315aeb425053858cf6c845)，固定提交`1ef4cbebcc0b368158315aeb425053858cf6c845`。参考子集保存在`experiments/multicrop_itm_soft_simmatch_v1/references/usb/`，不是完整可安装的semilearn包；版权头保留，CRLF/末尾换行规范化记录在源码清单。MIT条款见[`USB_LICENSE.txt`](USB_LICENSE.txt)，正文核对该提交的上游LICENSE.txt。`code/ssl_algorithms.py`为这些机制的ITM适配，不应误称为完全原创算法或上游分类实验精确复现。
