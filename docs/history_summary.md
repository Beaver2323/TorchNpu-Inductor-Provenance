# 需求变更前的历史研究摘要

> 最后更新：2026-09-18（CST，UTC+08:00）；实验日期仍以各条记录为准。
>
> 本文仅保存历史结论，不属于当前 `triton_experimental` 交付验收。

## 合并说明

早期工作曾按 CPU、普通 NPU、cache、FlexAttention template、默认 BlockMask 和
backward 调查分别维护多篇演示文档。需求收束后，这些页面与新手指南、技术参考存在
大量重复，因此合并为本摘要。被合并文档和旧探针可从 Git 提交 `5ace897` 恢复。

## 历史结论

| 日期 | 方向 | 结论 | 当前定位 |
| --- | --- | --- | --- |
| 2026-08-20 | CPU `tlparse` | CPU Inductor 可生成 pre-grad/post-grad/生成代码三栏联动页面 | 背景基线 |
| 2026-08-20 | 修改前普通 NPU | 图级 pre/post 映射存在，kernel 映射为空，证明 NPU codegen 缺少 hook | 缺口证据 |
| 2026-08-20 | 修改后普通 NPU | `add→relu→mul` 融合 kernel 建立双向 mapping，并在 910B2 上通过 | 已由当前静态 smoke 覆盖 |
| 2026-08-20 | FX Graph cache | cache miss/hit 的 mapping 与 stack 逐字节一致 | 社区 cache 机制可复用 |
| 2026-08-20 | FlexAttention template | forward template kernel 可关联 `flex_attention`、score 与 mask 来源节点 | 历史专项 |
| 2026-08-21 | 默认 BlockMask | `block_mask=None` 的 forward 数值和 provenance 页面通过 | 历史专项 |
| 2026-08-21 | 默认 BlockMask backward | 反向图和 dK/dV 候选生成；哨兵限界修复后仍阻塞于 BishengIR 长编译 | 未完成，不计入 PASS |
| 2026-08-25 | extern/aclnn 双 mm | 两次底层同名 mm 通过不同 wrapper / flow 关联到各自源码栈 | 历史专项，不等于当前 extern 已验收 |
| 2026-08-25 | MLIR 双融合 kernel | 设备来源栈从 0/2 到 2/2，分别指向不同模型源码段 | 已退出当前范围 |
| 2026-08-25 | DVM mlir_fusion | 两个 dvm_* kernel 的来源栈从 0/2 到 2/2 | 已退出当前范围 |
| 2026-08-25 | DVM graph_fusion | 逻辑 key、host launch 与真实 AIVEC 名称通过 flow 桥接，2/2 栈 | 已退出当前范围 |
| 2026-08-25 | CATLASS 双 mm | 同一个底层模板符号的两次调用通过 handle / marker 区分，2/2 栈 | 已退出当前范围 |
| 2026-08-26 | DVM matmul template | 调试模式区分两次模板 launch 的运行时名称，消除第二次误取第一次 stack | 已退出当前范围 |

上述 8 月 25—26 日记录使用 PyTorch `2.14.0a0+git8e86e0a`、当时的 torch_npu
`83cc452` 工作树和 Ascend910B2，属于需求收束前的实验，不能外推为当前
`triton_experimental` 的能力。
完整工作时间线见 [Tracking 交接快照](./work_records/tracking_handoff_20260918.md)，
各旧文档的合并去向见[归档来源清单](./work_records/source_inventory.json)。
CPU 入门的独立脚本、HTML 和 mapping 已精选保留在[CPU 演示](./cpu/README.md)。

## 为什么不再保留分散页面

当前正式需求只要求 `torch_npu/_inductor/triton_experimental`。CATLASS、MLIR/AKG、
DVM、multistream extern 和 FlexAttention backward 均已退出验收范围。继续在文件树中
并列展示这些页面容易造成“全部后端均已交付”的误解，也会重复维护环境、命令和结论。

仍有学习价值的背景知识已保留在：

- [新手入门](./beginner_guide.md)
- [技术参考](./technical_reference.md)
- [当前 `triton_experimental` 交付说明](./triton_experimental/README.md)

## 历史边界

FlexAttention backward 当时用于探索更复杂的 AOTAutograd、template 和融合路径，并非
provenance 只支持 backward，也不是当前功能验收的必要条件。任何能产生真实 forward、
backward kernel 并覆盖同一关联链路的稳定模型都可以验证基础 contract；当前交付采用
Llama 风格 RMSNorm + SwiGLU 模块完成这项验证。
