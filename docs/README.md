# 文档索引

> 最后更新：2026-09-02 00:03 CST（UTC+08:00）

## 推荐阅读顺序

1. [主交付文档](./provenance_delivery.md)：官网契约、社区源码架构、完整调用链、NPU
   扩展点和验收边界。
2. [新手入门](./beginner_guide.md)：背景概念、环境、用法和源码导读。
3. [`triton_experimental` 交付说明](./triton_experimental/README.md)：当前正式范围、
   实现、验证与演示。
4. [技术参考](./technical_reference.md)：需求变更前后的详细技术研究。
5. [历史研究摘要](./history_summary.md)：CPU、早期普通 NPU、cache、FlexAttention 和
   默认 BlockMask 的历史结论。

## 当前交付资料

| 类型 | 入口 | 用途 |
| --- | --- | --- |
| 主交付 | [`provenance_delivery.md`](./provenance_delivery.md) | 对照官网和社区源码说明设计与 NPU 对齐结论 |
| 总体说明 | [`triton_experimental/README.md`](./triton_experimental/README.md) | 范围、实现和验收 |
| 复现脚本 | [`triton_experimental/scripts/`](./triton_experimental/scripts/README.md) | 静态、timeline、rsplit、combo、Llama 与 A/B 探针 |
| 验收产物 | [`triton_experimental/artifacts/`](./triton_experimental/artifacts/README.md) | HTML、mapping、trace 与结构化结果 |
| forward 演示 | [三栏 HTML](./triton_experimental/artifacts/llama_swiglu/provenance_tracking_forward.html) | 完整 pre-grad→post-grad→代码联动 |
| backward 演示 | [三栏 HTML](./triton_experimental/artifacts/llama_swiglu/provenance_tracking_backward.html) | backward post-grad→kernel 与社区边界 |

## 状态口径

- “通过”表示在记录的 PyTorch、torch_npu、Triton Ascend、CANN 和 910B2 环境中实测通过。
- “社区边界”表示行为与社区 PyTorch 当前实现一致，不额外合成缺失的 backward
  `from_node` 关系。
- “历史研究”不属于当前 `triton_experimental` 验收范围，不能作为当前后端 PASS 证据。
- “后端不支持”表示门禁已经真实进入目标 codegen，但在 provenance 产生最终 mapping 前
  被后端自身错误阻断；必须同时给出关闭/开启 provenance 的 A/B 证据。
- HTML、JSON、Python 源码中的英文标识属于工具格式或代码接口，不做翻译。

## 社区基线

- [PyTorch 2.13 Provenance Tracking 官网文档](https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_inductor_provenance.html)
- [PyTorch 2.14 社区文档源文件](https://github.com/pytorch/pytorch/blob/release/2.14/docs/source/user_guide/torch_compiler/torch.compiler_inductor_provenance.md)
- [tlparse 社区仓](https://github.com/pytorch/tlparse)
