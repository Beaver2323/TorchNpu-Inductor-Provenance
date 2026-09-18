# 文档索引

> 最后更新：2026-09-18（CST，UTC+08:00）

先看[当前状态与工作记录](./work_records/README.md)：源码 PR !46073 已关联需求 #4909，
HEAD `195830924`，基线 `ec693356f`。本文中的历史验证不替代当前 CI。

## 推荐阅读顺序

1. [主交付文档](./provenance_delivery.md)：官网契约、社区源码架构、完整调用链、NPU
   扩展点和验收边界。
2. [新手入门](./beginner_guide.md)：背景概念、环境、用法和源码导读。
3. [`triton_experimental` 交付说明](./triton_experimental/README.md)：当前正式范围、
   实现、验证与演示。
4. [PR diff 逐段讲解](./pr_diff_walkthrough.md)：固定 BASE/HEAD 的改动解释、代码框、
   框架调用链与模型源码栈，完整覆盖 21 个非测试文件；测试仅保留覆盖导航。
5. [技术参考](./technical_reference.md)：需求变更前后的详细技术研究。
6. [历史研究摘要](./history_summary.md)：CPU、早期普通 NPU、cache、FlexAttention 和
   默认 BlockMask 的历史结论。

只想了解扩大模块验证时遇到了什么问题、解决到了哪里，先看
[模块验证工作总结（通俗版）](./module_validation_summary.md)，文末附可直接用于工作汇报的总结。

## 当前交付资料

| 类型 | 入口 | 用途 |
| --- | --- | --- |
| 主交付 | [`provenance_delivery.md`](./provenance_delivery.md) | 对照官网和社区源码说明设计与 NPU 对齐结论 |
| diff 导读 | [`pr_diff_walkthrough.md`](./pr_diff_walkthrough.md) | 结合实际源码解释每项功能修改和调用栈 |
| 非测试完整补丁 | [当前 PR](./diffs/triton_experimental_provenance_ec693356f_195830924_non_tests.patch) / [历史讲解版](./diffs/triton_experimental_provenance_f030beadb_dbc0db52f_non_tests.patch) | 均覆盖 21 个非测试文件，含 PNG 二进制内容；分别固定 ec693356f→195830924 和 f030beadb→dbc0db52f |
| 总体说明 | [`triton_experimental/README.md`](./triton_experimental/README.md) | 范围、实现和验收 |
| 模块验证总结 | [`module_validation_summary.md`](./module_validation_summary.md) | 用通俗语言说明实际难点、处理结果和未完成范围 |
| CI / rebase 核查 | [`pr_46073_ci_20260916.md`](./pr_46073_ci_20260916.md) | 9 月 16 日三组失败及后续进展；9 月 18 日 #68280 已触发，尚无最终通过结论 |
| 失败引入来源 | [`pr_46073_origin_analysis.md`](./pr_46073_origin_analysis.md) | 最新 master rebase、三个问题的引入提交、CI wheel 源码与三版本签名对照 |
| 本 PR 最小修复 | [`pr_46073_api_fix.md`](./pr_46073_api_fix.md) | 公开 API 元数据已修复并推送；聚焦回归通过、完整本地检查仍有既有失败 |
| 工作记录 | [`work_records/README.md`](./work_records/README.md) | Tracking 有效记录归档、需求 issue、版本时间线和原始证据索引 |
| CPU 入门基线 | [`cpu/README.md`](./cpu/README.md) | CPU 脚本、独立三栏 HTML、mapping / stack 及页面阅读说明 |
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
