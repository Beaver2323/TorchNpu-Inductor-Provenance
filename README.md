# TorchNPU Inductor 来源追踪

> 最后更新：2026-09-14（CST，UTC+08:00）

Provenance Tracking（来源追踪）帮助你查看模型中的操作在编译后对应哪些图节点和 kernel。
本文参照 [PyTorch 2.13 官方 provenance 文档][official]，介绍
`torch.compile` 使用 `triton_experimental` NPU 后端时的操作方法与社区功能对齐情况。

`tlparse` 将来源关系显示为三栏：

```text
输入 GraphModule / pre-grad 图  ↔  post-grad 图  ↔  Inductor 生成代码
```

页面中的**粗体行**表示工具覆盖的节点或 kernel；黄色高亮显示当前选择的来源关系。
一个 kernel 可以对应多个图节点，例如 `add → relu → mul` 融合后只生成一次 kernel 调用。
先看效果可下载 [Llama 前向演示 HTML](docs/triton_experimental/artifacts/llama_swiglu/provenance_tracking_forward.html)
并在浏览器打开。

本仓提供中文文档与演示证据；使用功能不依赖本仓脚本。适配源码见
[Ascend/pytorch PR !46073](https://gitcode.com/Ascend/pytorch/merge_requests/46073)，
本文对应提交 `4845c9289`。

## 用户使用方法

### 1. 准备包含该功能的环境

需要 PyTorch、torch_npu 和 Triton Ascend；已安装的 PyTorch 必须包含社区 provenance
能力，torch_npu 必须包含本 PR 的适配。仅安装普通 wheel 或下载本文档仓，不会自动获得
尚未包含在 wheel 中的功能。

项目的历史验证环境是 PyTorch `release/2.14`、匹配的 torch_npu wheel、
Triton Ascend `release/3.2.2`、CANN 9.0.1 和 Ascend 910B2。
环境检查示例：

```bash
python -c '
from torch._inductor import config
from torch_npu.profiler import inductor_trace_handler
print("provenance level:", config.trace.provenance_tracking_level)
print("NPU timeline handler: OK")
'
```

该检查确认配置和 timeline 入口存在；实际编译后的 mapping / trace 才能确认链路可用。

### 2. 安装 tlparse

按照官网流程，先安装 [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html)，
再执行：

```bash
cargo install tlparse
```

本项目历史实测使用 `tlparse 0.4.8`。

### 3. 为自己的程序启用来源追踪

在现有程序中选择 NPU Inductor 后端：

```python
compiled_model = torch.compile(
    model,
    backend="inductor",
    options={"npu_backend": "triton_experimental"},
)
```

如果没有现成程序，可保存以下完整示例为 `your_program.py`：

```python
import torch
import torch_npu


class Model(torch.nn.Module):
    def forward(self, x):
        return torch.relu(torch.sin(x)) * x


model = Model().npu().train()
x = torch.randn(64, 128, device="npu", requires_grad=True)
compiled_model = torch.compile(
    model,
    backend="inductor",
    options={"npu_backend": "triton_experimental"},
    fullgraph=True,
)
compiled_model(x).sum().backward()
torch.npu.synchronize()
```

在启动 Python 前设置社区环境变量，生成结构化编译日志：

```bash
TORCH_TRACE=/tmp/my_inductor_trace INDUCTOR_PROVENANCE=1 python your_program.py
```

`TORCH_TRACE` 指定日志目录，建议每次实验使用新的目录。`INDUCTOR_PROVENANCE=1`
开启 normal 来源追踪。首次前向和首次 `.backward()` 可能分别触发编译，反向图也可产生
来源记录；无需为反向再单独调用一次 `torch.compile`。

### 4. 生成并阅读三栏高亮页面

选择日志目录中一个具体的 `.log` 文件，替换下面的 `your_log.log`：

```bash
tlparse /tmp/my_inductor_trace/your_log.log \
  --inductor-provenance \
  -o /tmp/my_tlparse_output \
  --no-browser
```

打开输出目录的 `index.html`，点击 **Provenance Tracking** 链接。
点击粗体节点或 kernel，观察另外两栏中对应的黄色高亮。

- 将具体日志文件交给 `tlparse`；官网提示 `tlparse parse <目录>` 可能无法生成高亮页面。
- 多个日志文件分别解析。一个日志里也可能包含多个编译图，因此可以出现多个 HTML 页面。
- 不加 `--inductor-provenance` 时，仍可在索引中读取 mapping JSON；该参数用于生成高亮页面。
- 反向图的 FX 入口也可能叫 `def forward`，这是 GraphModule 的统一命名，不代表它是模型前向。

三栏页面使用的主要编译产物与官网一致：

| 产物 | 用途 |
| --- | --- |
| `before_pre_grad_graph.txt` | 输入 / pre-grad 图 |
| `after_post_grad_graph.txt` | post-grad 图 |
| `inductor_output_code.txt` | JIT Inductor 生成代码 |
| `inductor_aot_wrapper_code.txt` | 社区 AOTInductor wrapper；本轮 NPU AOTI 未验收 |
| `inductor_provenance_tracking_node_mappings.json` | 图节点与 kernel 的双向关系 |

文件名可能带有 tlparse 添加的编号；JIT 和 AOT 产物不必同时出现。

## 查看每个 kernel 对应的源码

启用 `INDUCTOR_PROVENANCE=1` 后，在 tlparse 索引中找到
`inductor_provenance_tracking_kernel_stack_traces.json`，点击旁边的 **readable_html**，
即可查看 kernel 对应的模型源码栈。这与官网的 kernel 源码查看方式一致。

例如下面的 key（来自本仓[静态演示](docs/triton_experimental/artifacts/static_smoke/kernel_stack_traces.json)）：

```text
triton_unk_fused_add_mul_relu_0:1
```

`:1` 是 debug handle，用来区分生成代码中的 kernel 调用位置；它不是耗时或执行次数。
生成代码的注释中也能找到相同 handle，从而与 mapping、源码栈对应。
一个融合 kernel 可以对应多条源码栈。

## NPU 扩展：在 profiler timeline 中查看源码栈

三栏 HTML 描述编译期来源关系。timeline 则在实际执行的设备 kernel 事件上回填来源栈，
便于结合耗时定位模型代码。该能力基于本项目配套的 PyTorch 2.14 处理器与 torch_npu
adapter，属于官网 2.13 静态页面用法之外的补充。

要运行完整示例，保留上面 `your_program.py` 的 import 和 `Model` 定义，
将 `model = ...` 及之后的代码替换为下列内容，保存为 `your_profile_program.py`：

```python
from torch_npu.profiler import inductor_trace_handler


model = Model().npu().train()
compiled_model = torch.compile(
    model,
    backend="inductor",
    options={"npu_backend": "triton_experimental"},
    fullgraph=True,
)


def make_input():
    return torch.randn(64, 128, device="npu", requires_grad=True)


# 在 profiler 外完成首次前向和反向编译。
warmup_x = make_input()
compiled_model(warmup_x).sum().backward()
torch.npu.synchronize()
model.zero_grad(set_to_none=True)

handler = inductor_trace_handler("/tmp/my_npu_timeline", worker_name="rank0")
profile_x = make_input()
with torch_npu.profiler.profile(on_trace_ready=handler):
    compiled_model(profile_x).sum().backward()
    torch.npu.synchronize()
```

在编译和采样前开启 timeline 配置：

```bash
TORCH_COMPILE_DEBUG_EXTEND=1 TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 \
  python your_profile_program.py
```

将 `/tmp/my_npu_timeline/*.pt.trace.json` 中的具体文件载入 [Perfetto](https://ui.perfetto.dev/)，
选择 NPU device kernel 事件，查看 `args.stack`。这是编译期模型源码栈的回填，
不是在设备执行现场采集 Python 栈。
如需同时生成静态 HTML 所需日志，在上述启动命令中再设置 `TORCH_TRACE` 和
`INDUCTOR_PROVENANCE=1`，然后按前面的步骤运行 tlparse。

## 环境变量与 Python 配置

以下配置已按本项目配套的 [PyTorch 源码][config-source]核对，均应在 `import torch`
之前设置。它们不保证在所有旧版 PyTorch wheel 中存在。

| 环境变量 | 默认值 | 对应作用 |
| --- | --- | --- |
| `INDUCTOR_PROVENANCE=0/1/2` | `0` | `trace.provenance_tracking_level`：关闭 / normal / basic |
| `TORCH_TRACE=/path/to/logs` | 未设置 | 结构化编译日志目录，供 tlparse 使用 |
| `TORCH_COMPILE_DEBUG_EXTEND=1` | `0` | `trace.provenance_tracking_to_timeline=True` |
| `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1` | `1` | `triton.unique_kernel_names=True`，辅助关联编译信息与 profiler 事件 |
| `TORCH_COMPILE_DEBUG_MAX_EVENTS=<N>` | `500000` | timeline 后处理的最大事件数；`0` 表示不限制 |
| `TORCH_COMPILE_DEBUG=1` | `0` | 综合编译调试；未设置 `INDUCTOR_PROVENANCE` 时，level 回退为 1 |

timeline 开启时，有效 provenance level 至少为 1，因此仅做 timeline 可以不单独设置
`INDUCTOR_PROVENANCE`。环境变量启用回填能力，实际 NPU trace 导出仍需
`inductor_trace_handler`。它也支持 `use_gzip=True` 输出压缩 trace。

也可以用 Python 设置配置，作用域须覆盖编译、预热、采样和导出回调：

```python
from torch._inductor import config

with config.patch({
    "trace.provenance_tracking_level": 1,
    "trace.provenance_tracking_to_timeline": True,
    "triton.unique_kernel_names": True,
}):
    # 在这里执行上面的模型编译、预热与 profiler 代码。
    ...
```

## 与社区功能的对齐矩阵

“已验证”指本仓原始实测基线，主要产物生成于 2026-08-27 至 2026-09-01。
源码提交 `4845c9289` 在 2026-09-11 rebase 后完成静态检查，尚无该 HEAD 的完整 NPU
端到端复测结果。以下不把历史产物当作最新 PR CI 的通过证明。

| 功能 | 社区依据 / 能力 | NPU `triton_experimental` 状态 | 证据或限制 |
| --- | --- | --- | --- |
| 三栏节点与代码高亮 | 官网输入图、post-grad 图、生成代码三栏 | 已对齐、已验证 | [Llama 前向 HTML](docs/triton_experimental/artifacts/llama_swiglu/provenance_tracking_forward.html) |
| `INDUCTOR_PROVENANCE=1` + `TORCH_TRACE` | 官网标准启用方式 | 已对齐、已验证 | [静态结果](docs/triton_experimental/artifacts/static_smoke/static_level1_result.json) |
| `tlparse --inductor-provenance` 与 mapping JSON | 官网可视化及机器可读产物 | 已对齐、已验证；复用社区工具和 schema | [mapping JSON](docs/triton_experimental/artifacts/static_smoke/node_mappings.json) |
| Triton kernel 来源 | 官网覆盖 Triton kernel | 已对齐、已验证 | [最小三栏页面](docs/triton_experimental/artifacts/static_smoke/provenance_tracking.html) |
| kernel 源码栈与 debug handle | 官网 `readable_html`、`kernel:handle` | 已对齐、已验证 | [kernel stacks](docs/triton_experimental/artifacts/static_smoke/kernel_stack_traces.json) |
| level 1 / 2 | 配套 PyTorch 2.14 的 normal / basic 配置 | 已对齐、已验证 | [level 2 结果](docs/triton_experimental/artifacts/static_smoke/static_level2_result.json) |
| backward 来源关系 | 配套社区实现；不保证每个节点都有完整 `from_node` | post-grad→kernel 已验证；左栏缺失遵循社区边界 | [Llama 反向 HTML](docs/triton_experimental/artifacts/llama_swiglu/provenance_tracking_backward.html) |
| profiler timeline 源码栈 | 配套 PyTorch 2.14 处理器；非官网 2.13 该页的操作流程 | 已适配、已验证前向和反向 | [trace/result](docs/triton_experimental/artifacts/timeline/) |
| rsplit partial / combine | NPU 后端两次 launch，共享社区来源登记机制 | 两个 kernel 均已验证；不等于 ComboKernel | [rsplit 结果](docs/triton_experimental/artifacts/timeline/rsplit_result.json) |
| ComboKernel | 官网明确覆盖社区 combo kernel | 未对齐；历史测试被 NPU codegen 错误阻断 | [level 0](docs/triton_experimental/artifacts/validation/combo_level0_result.json) / [level 1](docs/triton_experimental/artifacts/validation/combo_level1_result.json) 均缺少 `x0/x0mask` 定义 |
| C++ kernel 来源 | 官网覆盖社区 C++ kernel | 不适用本轮 NPU Triton 后端 | 社区 C++ 支持不能计作 NPU 已验证 |
| AOTInductor provenance | 官网展示 AOT 三栏；社区实现另有 `kernel_information.json` | 未验收 | NPU AOTI 设备、lazy 初始化与 ABI 前提见[交付指南](docs/triton_experimental/README.md#123-为什么本轮不能把-aotinductor-标为完成) |

本轮范围仅包括 `triton_experimental`。Llama 风格 RMSNorm + SwiGLU 已在历史基线完成
两组动态形状的前反向、输入/参数梯度和 provenance 验证，详见
[模型结果](docs/triton_experimental/artifacts/llama_swiglu/llama_swiglu_result.json)。
其他后端与 FlexAttention 的早期研究见历史资料，不纳入本矩阵的当前验收范围。

## 进一步阅读

- [主交付文档](docs/provenance_delivery.md)：官网契约、设计与验收边界。
- [PR diff 逐段讲解](docs/pr_diff_walkthrough.md)：修改前后代码框、调用栈、测试与原始补丁。
- [新手入门](docs/beginner_guide.md)：FX、Inductor IR 与静态 / 运行时追踪基础。
- [演示产物索引](docs/triton_experimental/artifacts/README.md)：HTML、mapping、源码栈和 timeline。
- [文档总索引](docs/README.md)：开发者复现方法、技术参考与历史研究。
- [PyTorch 官方 provenance 文档][official]与 [tlparse 社区仓](https://github.com/pytorch/tlparse)。

[official]: https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_inductor_provenance.html
[config-source]: https://github.com/pytorch/pytorch/blob/8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b/torch/_inductor/config.py
