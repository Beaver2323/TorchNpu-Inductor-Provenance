# TorchNPU Inductor 来源追踪

> 最后更新：2026-09-02 01:10 CST（UTC+08:00）

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

## 快速使用

下面先给出可直接复制的最短路径。详细原理、wheel 安装和全部探针参数见
[`triton_experimental` 交付说明](docs/triton_experimental/README.md)。

### 1. 不跑程序，直接看现成演示

克隆本仓后，用浏览器打开下列页面：

- [Llama forward 三栏页面](docs/triton_experimental/artifacts/llama_swiglu/provenance_tracking_forward.html)
- [Llama backward 三栏页面](docs/triton_experimental/artifacts/llama_swiglu/provenance_tracking_backward.html)
- [三操作最小页面](docs/triton_experimental/artifacts/static_smoke/provenance_tracking.html)

如果浏览器不允许直接读取本地 HTML，可以启动一个只读静态服务：

```bash
cd /home/z50063656/tmp
python -m http.server 8000 \
  --directory /home/z50063656/TorchNpu-Inductor-Provenance
```

然后访问
`http://127.0.0.1:8000/docs/triton_experimental/artifacts/llama_swiglu/provenance_tracking_forward.html`。

### 2. 在当前 Tracking 环境实测静态 provenance

所有测试从 `/home/z50063656/tmp` 启动，不要在 `torch_npu` 源码树内导入
`torch`。下列命令复用本项目已验证的独立 wheel target，不会修改现有
conda 环境：

```bash
cd /home/z50063656/tmp
source /home/z50063656/Tracking/activate_tracking.sh

export PYTHONPATH=/home/z50063656/Tracking/triton_experimental_delivery/wheel_target_20260827_v10:$PYTHONPATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=7
export DEMO_ROOT=/home/z50063656/TorchNpu-Inductor-Provenance/docs/triton_experimental

export TORCH_TRACE=/home/z50063656/Tracking/triton_experimental_delivery/readme_static_trace_001
python "$DEMO_ROOT/scripts/static_probe.py" \
  --output-dir /home/z50063656/Tracking/triton_experimental_delivery/readme_static_run_001 \
  --level 1 \
  --expect-mapped
```

成功时，脚本会输出 `result.json`，其中 `kernel_to_post` 应将一个
`triton_*:debug_handle` 映射到 `add/relu/mul`，`max_abs_diff` 应为 0。
脚本要求 `--output-dir` 与 `TORCH_TRACE` 目录事先不存在；重跑时请换一个新后缀，
例如把 `_001` 改成 `_002`。

### 3. 将本次记录生成 tlparse 三栏 HTML

确认 `tlparse --version` 可用；本项目实测版本为 `0.4.8`。如尚未安装，可执行
`cargo install --locked --version 0.4.8 tlparse`。在上一步同一个 shell 中继续：

```bash
tlparse -i "$TORCH_TRACE"/*.log \
  -o /home/z50063656/Tracking/triton_experimental_delivery/readme_tlparse_001 \
  --no-browser
```

打开输出目录中的 `index.html`，再进入
`provenance_tracking_<compile-id>.html`。三栏从左到右分别是 pre-grad FX、post-grad
FX 和生成的 Triton/Python wrapper 代码。

### 4. 实测 forward/backward 运行时 timeline

保持第 2 步的环境变量，执行：

```bash
python "$DEMO_ROOT/scripts/timeline_probe.py" \
  --output-dir /home/z50063656/Tracking/triton_experimental_delivery/readme_timeline_run_001
```

成功后会生成 `*.pt.trace.json` 和 `result.json`。将 trace JSON 载入
`https://ui.perfetto.dev/`，选中 NPU device kernel 事件，在事件的 `args.stack`
中查看回填的 Python 源码栈。

注意：ComboKernel 当前会在后端代码生成阶段因缺少 `x0/x0mask` 定义而失败，
不应当作 provenance 成功演示；请使用上述 `static_probe.py`、
`timeline_probe.py` 或 `llama_swiglu_demo.py`。

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
