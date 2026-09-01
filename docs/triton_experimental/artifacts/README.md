# 验收产物索引

> 最后更新：2026-09-02 00:03 CST（UTC+08:00）
>
> 原始实测日期为 2026-08-27 至 2026-09-01，以各结果 JSON 为准；更新时间不代表重跑全部历史用例。

## 目录

| 目录 | 内容 | 状态 |
| --- | --- | --- |
| [`llama_swiglu/`](./llama_swiglu/) | forward/backward HTML、mapping、kernel stack、timeline 和综合结果 | 当前主演示 |
| [`static_smoke/`](./static_smoke/) | 三操作静态 level 1/2 结果和三栏页面 | 最小静态证据 |
| [`timeline/`](./timeline/) | forward/backward 与 rsplit 的 result/trace | 最小运行时证据 |
| [`validation/`](./validation/) | 代表性模型矩阵、provenance A/B 与 ComboKernel 门禁 | 边界证据 |

## Llama 主演示

- [forward 三栏页面](./llama_swiglu/provenance_tracking_forward.html)
- [backward 三栏页面](./llama_swiglu/provenance_tracking_backward.html)
- [综合验证结果](./llama_swiglu/llama_swiglu_result.json)
- [静态节点映射](./llama_swiglu/llama_swiglu_node_mappings.json)
- [运行时 kernel 源码栈](./llama_swiglu/llama_swiglu_kernel_stacks.json)
- [Perfetto trace](./llama_swiglu/llama_swiglu_timeline_trace.json)

完全重复的 `provenance_tracking.html` 兼容副本已经移除，只保留名称明确的 forward 和
backward 页面。自动生成 HTML 和 trace 保持原始工具格式，不翻译内部字段。

## ComboKernel 边界证据

- [level 0 结果](./validation/combo_level0_result.json)：关闭 provenance 仍因未定义 `x0` 编译失败。
- [level 1 结果](./validation/combo_level1_result.json)：开启 provenance 后同位置、同错误失败。
- [level 1 生成代码](./validation/combo_level1_output_code.py)：保留单个 combo kernel、三个分支和 provenance handle。

两组结果共同证明 ComboKernel 尚未进入 provenance mapping 验收，阻断点属于实验后端
codegen，而不是 provenance 功能。
