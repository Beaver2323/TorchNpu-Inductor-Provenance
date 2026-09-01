# TorchNPU Inductor 来源追踪

> 最后更新：2026-09-02 01:23 CST（UTC+08:00）

本仓库存放 TorchInductor Provenance Tracking（来源追踪）在昇腾 NPU 上的调研文档、
复现脚本和验收产物。当前正式范围只覆盖
`torch_npu/_inductor/triton_experimental`；其他 NPU 后端和 FlexAttention 内容仅作为
需求变更前的历史研究，不属于本轮验收。

## 仓库定位

本仓采用文档交付：以 PyTorch 官网公开用法和社区源码实现为基线，用中文主交付文档
说明设计、调用链、NPU 扩展点、已验证范围和未验收边界；演示 HTML、静态 mapping、
timeline trace/result 与复现脚本作为配套证据。本仓不是源码交付仓。

- 官方源码目标仓：`https://gitcode.com/Ascend/pytorch`
- 开发 fork：`https://gitcode.com/gcw_3ffySSwy/pytorch`
- 源码交付分支：`codex/triton-experimental-provenance-delivery`
- 源码交付提交：`6ca3af211`，基于官方提交 `83cc45248`
- 工作流参考：`https://gitcode.com/AllenGuanC/inductor-meta-worktree`

## 当前结论

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| CPU Inductor 静态来源追踪 | 已验证 | 可生成 `tlparse` 三栏页面 |
| NPU `triton_experimental` 静态 level 1/2 | 已验证 | kernel 与 post-grad 节点双向映射有效 |
| NPU forward/backward 运行时来源追踪 | 已验证 | profiler 设备 kernel 可回填源码栈 |
| rsplit partial/combine | 已验证 | 一次调度产生的两个 kernel 均有独立来源 |
| NPU ComboKernel | 后端不支持 | level 0/1 均生成缺失 `x0/x0mask` 定义的 kernel，与 provenance 无关 |
| Llama 风格 RMSNorm + SwiGLU | 已验证 | 两组动态形状、前反向数值、输入/参数梯度、静态映射和 timeline 均通过 |
| backward pre-grad→生成代码全覆盖 | 社区边界 | 社区 PyTorch 的 backward 节点可能缺少 `from_node`，不承诺每个节点都形成完整三段链 |
| AOTInductor `kernel_information.json` | 未验收 | 当前 910B2 和共享 NPU AOTI/lazy/ABI 基线不满足验收前提 |

验证环境为 PyTorch `release/2.14`、匹配的 `torch_npu` wheel、Triton Ascend
`release/3.2.2`、CANN 9.0.1 和 Ascend 910B2。

## 用户使用方法

使用 provenance 不需要本仓的演示脚本。用户只需要一个已安装 PyTorch、
torch_npu 和 Triton Ascend 的 NPU 环境，然后在自己原有的 `torch.compile`
程序上开启追踪。前提是已安装的 PyTorch/torch_npu wheel 包含本功能；
本文档仓本身不会把功能动态注入普通 wheel。

### 1. 检查环境是否包含该功能

在自己的 PyTorch + torch_npu 环境中执行：

```bash
python -c '
from torch._inductor import config
from torch_npu.profiler import inductor_trace_handler
print("provenance level:", config.trace.provenance_tracking_level)
print("NPU timeline handler: OK")
'
```

如果 `provenance_tracking_level` 或 `inductor_trace_handler` 不存在，表示当前 wheel
未包含对应的 PyTorch 社区能力或 torch_npu 适配，需先升级到包含交付提交的
wheel，而不是下载本文档仓的脚本。

### 2. 用自己的 `torch.compile` 程序生成静态来源记录

用户现有程序只需保证使用 `inductor` 和 `triton_experimental`：

```python
compiled_model = torch.compile(
    model,
    backend="inductor",
    options={"npu_backend": "triton_experimental"},
)
```

如果手头没有程序，下面是一个完整的 `your_program.py`。它是普通用户程序，
不导入本仓任何内容：

```python
import torch
import torch_npu  # 注册 NPU 设备和 Inductor 后端


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

在启动 Python 之前设置两个社区 PyTorch 环境变量：

```bash
export TORCH_TRACE=/tmp/my_inductor_trace
export INDUCTOR_PROVENANCE=1
python your_program.py
```

- `TORCH_TRACE` 指定 PyTorch 结构化编译日志目录。每次测试建议使用新的空目录。
- `INDUCTOR_PROVENANCE=1` 开启完整来源追踪；可改为 `2` 使用较轻量的 basic
  模式。`0` 表示关闭。
- 环境变量必须在 `import torch` 前生效，因为 Inductor 在导入时读取它们。
- forward 和 backward 都只需正常调用；backward 在首次 `.backward()` 时由
  AOTAutograd 编译并记录。

### 3. 用 tlparse 生成三栏 HTML

`tlparse` 是独立的可视化工具，不是本仓脚本。首次使用时安装：

```bash
cargo install tlparse
```

本项目实测过 `tlparse 0.4.8`。用上一步生成的某一个具体 `.log`
文件生成页面：

```bash
tlparse /tmp/my_inductor_trace/<log_file_name>.log \
  --inductor-provenance \
  -o /tmp/my_tlparse_output \
  --no-browser
```

不要把日志目录作为 `tlparse parse` 子命令参数。若目录中有多个
`.log`，应分别解析；单进程普通运行通常只产生一个。打开输出目录的
`index.html`，进入 **Provenance Tracking** 链接即可看到：

```text
pre-grad FX  ↔  post-grad FX  ↔  Inductor 生成代码
```

黄色高亮表示当前选中节点/kernel 的来源关系。同一输出目录中的
`inductor_provenance_tracking_node_mappings*.json` 是对应的机器可读映射。

### 4. 在 NPU profiler timeline 中查看运行时源码栈

静态 HTML 只需环境变量。如果还需要把 Python 源码栈回填到 NPU profiler
device kernel 事件，需把原有 profiling 代码的 `on_trace_ready` 换成
torch_npu 提供的 handler：

```python
import torch
import torch_npu
from torch._inductor import config
from torch_npu.profiler import inductor_trace_handler


with config.patch(
    {
        "trace.provenance_tracking_level": 1,
        "trace.provenance_tracking_to_timeline": True,
        "triton.unique_kernel_names": True,
    }
):
    compiled_model = torch.compile(
        model,
        backend="inductor",
        options={"npu_backend": "triton_experimental"},
    )

    # 先在 profiler 外完成 forward/backward 首次编译。
    warmup_x = make_input()
    compiled_model(warmup_x).sum().backward()
    torch.npu.synchronize()

    handler = inductor_trace_handler(
        "/tmp/my_npu_timeline", worker_name="rank0"
    )
    profile_x = make_input()
    with torch_npu.profiler.profile(on_trace_ready=handler):
        compiled_model(profile_x).sum().backward()
        torch.npu.synchronize()
```

这里的 `model` 和 `make_input()` 都是用户自己的对象，不来自本仓。输出的
`/tmp/my_npu_timeline/*.pt.trace.json` 仍是标准 Ascend Chrome trace，可在 Perfetto
中打开。选中 NPU device kernel 事件后，在 `args.stack` 中查看回填的
Python 源码栈。

### 5. 已知边界

- 当前交付只验收 `triton_experimental` 后端。
- ComboKernel 会因 NPU 后端缺少 `x0/x0mask` 定义而编译失败，该问题与
  provenance 开关无关。
- backward 页面的 FX `GraphModule` 仍显示 `def forward`，这是 FX 的统一入口
  命名，不表示它是模型前向图。
- 验收脚本、专用 wheel target 和 Tracking 绝对路径只用于本项目开发回归，
  不是用户接口。需要复现交付验收时，再阅读
  [`triton_experimental` 交付说明](docs/triton_experimental/README.md)。

## 从哪里开始

1. [主交付文档](docs/provenance_delivery.md)：对照官网契约和社区源码理解设计、调用链、
   NPU 适配点与验收边界。
2. [文档总索引](docs/README.md)：了解全部交付内容和推荐阅读顺序。
3. [新手入门](docs/beginner_guide.md)：理解 pre-grad、post-grad、Inductor IR、kernel
   以及静态/运行时来源追踪。
4. [`triton_experimental` 交付说明](docs/triton_experimental/README.md)：查看实现范围、
   复现命令和验收结论。
5. [技术参考](docs/technical_reference.md)：查看需求变更前后的完整技术研究。
6. [历史研究摘要](docs/history_summary.md)：了解已退出当前范围的早期结论。

## 核心演示

- [Llama forward 三栏页面](docs/triton_experimental/artifacts/llama_swiglu/provenance_tracking_forward.html)
- [Llama backward 三栏页面](docs/triton_experimental/artifacts/llama_swiglu/provenance_tracking_backward.html)
- [Llama 验证结果](docs/triton_experimental/artifacts/llama_swiglu/llama_swiglu_result.json)
- [Llama 静态节点映射](docs/triton_experimental/artifacts/llama_swiglu/llama_swiglu_node_mappings.json)
- [Llama Perfetto trace](docs/triton_experimental/artifacts/llama_swiglu/llama_swiglu_timeline_trace.json)
- [代表性模型验证矩阵](docs/triton_experimental/artifacts/validation/model_validation_result.json)
- [ComboKernel level 0/1 A/B 结果](docs/triton_experimental/artifacts/validation/combo_level1_result.json)
- [全部产物索引](docs/triton_experimental/artifacts/README.md)

HTML 需要下载到本地浏览器打开。timeline trace JSON 可载入 Perfetto。forward 与
backward 必须分别阅读：两个页面的 FX `GraphModule` 都显示 `def forward`，这是 FX 的
统一图入口命名，不表示 backward 页面执行的是模型前向。

## 文件树

```text
.
├── README.md
└── docs
    ├── README.md
    ├── provenance_delivery.md
    ├── beginner_guide.md
    ├── technical_reference.md
    ├── history_summary.md
    └── triton_experimental
        ├── README.md
        ├── scripts
        │   ├── README.md
        │   └── *.py
        └── artifacts
            ├── README.md
            ├── llama_swiglu
            ├── static_smoke
            ├── timeline
            └── validation
```

文件树按“学习文档、当前交付、复现脚本、验收产物”收束。完全重复的 Llama forward
兼容 HTML 已删除；需求变更前的分散演示文档合并到历史摘要，原始细节仍可从 Git 历史
提交 `5ace897` 恢复。

## 文档语言与原始产物

说明性 Markdown 文档统一使用中文。源码标识、配置项、kernel 名、JSON schema 和
`tlparse` 自动生成 HTML 保留社区原始英文格式，以保证证据可复现并与社区工具对齐。
