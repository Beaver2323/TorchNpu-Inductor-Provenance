# NPU TorchInductor Provenance 可视化实测

## 结果

2026-08-20 已在真实 Ascend 910B2 上完成普通 NPU Triton provenance 的完整链路：

```text
torch.compile(NPU)
  -> NPU Triton 融合 kernel
  -> TORCH_TRACE structured log
  -> tlparse 0.4.8
  -> pre-grad / post-grad / generated code 三栏联动高亮
```

本轮融合 kernel 为 `triton_poi_fused_add_mul_relu_0:1`，它同时对应 post-grad 图中的 `add`、`relu` 和 `mul`。真实设备回归结果为：

```text
Ran 3 tests in 27.289s
OK
```

demo 输出为：

```text
torch=2.14.0a0+git8e86e0a
torch_npu=2.14.0a0+git83cc452
device=Ascend910B2
checksum=9206.284180
```

## 本次报告入口

- [当前 `triton_experimental` 三栏页面](./inductor_provenance_demo/triton_experimental/provenance_tracking.html)
- [当前静态节点映射 JSON](./inductor_provenance_demo/triton_experimental/node_mappings.json)
- [当前 kernel stack JSON](./inductor_provenance_demo/triton_experimental/kernel_stack_traces.json)

需求变更前的普通 NPU 演示产物没有发布到本仓；其原始路径
`npu_provenance_verified_20260820/` 仅作为历史实验记录，不再提供会跳转到 404 的链接。

打开总入口后找到 `Provenance Tracking`，进入 `provenance_tracking_-_0_0_0`。也可以直接打开上面的三栏页面。

## 如何自行重跑

所有测试必须从 `/home/z50063656/tmp` 启动，不能在 torch_npu 源码树内导入 `torch`。先用 `npu-smi info` 重新选择当时空闲的物理设备；下面的 `5` 只是本次验证使用的设备。

```bash
cd /home/z50063656/tmp
source /home/z50063656/Tracking/activate_tracking.sh

RUN_DIR=$(mktemp -d /home/z50063656/tmp/tracking-npu-provenance.XXXXXX)
export ASCEND_RT_VISIBLE_DEVICES=5
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export INDUCTOR_PROVENANCE=1
export TORCH_COMPILE_DEBUG=1
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCH_TRACE="$RUN_DIR/trace"
export TORCH_COMPILE_DEBUG_DIR="$RUN_DIR/debug"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/cache"

python /home/z50063656/Tracking/npu_provenance_demo.py
```

生成可视化报告：

```bash
source /home/z50063656/.cargo/env
TRACE_FILE=$(find "$RUN_DIR/trace" -maxdepth 1 -name '*.log' -print -quit)

tlparse "$TRACE_FILE" \
  --inductor-provenance \
  --no-browser \
  -o "$RUN_DIR/tlparse"
```

本轮输出为 `Stats { ok: 139 }`。事件数可能随 PyTorch 日志内容变化，不应把固定数字作为唯一断言；真正的验收目标是 mapping、stack 和三栏行号关系都存在。

## 三栏分别表示什么

| 区域 | 内容 | 本例节点/代码 |
| --- | --- | --- |
| 左栏 `preGradGraph` | 接近用户模型的输入 FX 图 | `added`、`activated`、`mul` |
| 中栏 `postGradGraph` | AOTAutograd/Inductor 优化后的 ATen 图 | `add.Tensor`、`relu.default`、`mul.Tensor` |
| 右栏 `generatedCode` | NPU Python wrapper 和 Triton `.run()` 调用 | `triton_poi_fused_add_mul_relu_0.run(...)` |

本例完整关系为：

```text
用户模型                  post-grad ATen 图              NPU Triton kernel

added = x + y       ---> add.Tensor       ┐
activated = relu()  ---> relu.default     ├---> triton_poi_fused_add_mul_relu_0:1
return activated*2  ---> mul.Tensor       ┘
```

推荐按以下顺序阅读：

1. 点击左栏的 `added`，中栏应定位到 `aten.add.Tensor`。
2. 点击中栏的 `relu`，左栏应定位到 `activated`，右栏定位到同一个融合 kernel 调用。
3. 点击右栏第 132 行的 `triton_poi_fused_add_mul_relu_0.run(...)`，中栏的 `add`、`relu`、`mul` 会同时高亮，并继续关联到左栏三个来源节点。

一个 kernel 同时对应三个节点是预期行为：Inductor 把三个逐元素操作融合进一次 NPU kernel 发射。当前 provenance 的追踪粒度是 kernel 调用，不是 Triton kernel 内部的每条指令。

## 原始 mapping 如何阅读

PyTorch 产生的原始 artifact 为：

```json
{
  "cppCodeToPost": {
    "triton_poi_fused_add_mul_relu_0:1": ["mul", "relu", "add"]
  },
  "postToCppCode": {
    "add": ["triton_poi_fused_add_mul_relu_0:1"],
    "mul": ["triton_poi_fused_add_mul_relu_0:1"],
    "relu": ["triton_poi_fused_add_mul_relu_0:1"]
  },
  "postToPre": {
    "add": ["added"],
    "mul": ["mul"],
    "relu": ["activated"]
  },
  "preToPost": {
    "activated": ["relu"],
    "added": ["add"],
    "mul": ["mul"]
  },
  "version": 2.0
}
```

`cppCodeToPost` 是 PyTorch 数据协议中的历史字段名，并不表示这个 kernel 必须由 C++ 实现。NPU Triton、CATLASS、MLIR 和 DVM kernel 都继续使用该字段，避免引入 NPU 专属 schema。

kernel key 可以拆成：

```text
triton_poi_fused_add_mul_relu_0:1
│      │                        └─ debug handle，本次调用的唯一编号
│      └────────────────────────── 融合节点和 kernel 序号
└───────────────────────────────── Triton pointwise kernel
```

同一个 key 还出现在 wrapper 注释和 stack artifact 中：

```python
# [Provenance debug handles] triton_poi_fused_add_mul_relu_0:1
triton_poi_fused_add_mul_relu_0.run(
    arg0_1, arg1_1, buf0, 64, 128, stream=raw_stream0
)
```

stack artifact 可反查到 `npu_provenance_demo.py` 的三条 `forward` 源码：第 9 行 add、第 10 行 relu、第 11 行 mul。

## 为什么 HTML 中两个 cpp 字段为空仍然能联动

这是最容易误读的地方。tlparse 会把原始的 kernel/node mapping 再转换为页面行号，并按生成代码载体拆成两组：

| tlparse HTML 字段 | 对应代码载体 | 本轮结果 |
| --- | --- | --- |
| `pyCodeToPost` / `postToPyCode` | JIT `inductor_output_code` Python wrapper | 非空，实际用于本页联动 |
| `cppCodeToPost` / `postToCppCode` | `inductor_aot_wrapper_code` AOT C++ wrapper | 空，本轮没有 AOT C++ wrapper |

本轮 HTML 内嵌行号映射为：

```json
{
  "pyCodeToPost": {"132": [10, 7, 4]},
  "postToPyCode": {"10": [132], "7": [132], "4": [132]},
  "cppCodeToPost": {},
  "postToCppCode": {}
}
```

`tlparse 0.4.8/src/lib.rs::convert_node_mappings_to_line_numbers()` 分别调用 `build_python_kernel_to_lines_map()` 和 `build_cpp_kernel_to_lines_map()` 构造这两组表；`src/provenance.js::findCorrespondingLines()` 在存在 Python `codeData` 时明确读取 `pyCodeToPost`/`postToPyCode`。因此两个 cpp 行号字段为空不是 NPU 兼容问题。

CPU JIT 页面也是相同结构，只是对应行号为 84。这一对照进一步证明，NPU 不需要为了填充 AOT C++ 字段而移动注释或修改 tlparse。

## 自动验收结果

从本轮 HTML 中解析出的结果为：

```text
pre_post_highlight=pass {'10': [7], '13': [10], '7': [4]}
kernel_post_highlight=pass {'132': [10, 7, 4]}
post_kernel_highlight=pass {'10': [132], '4': [132], '7': [132]}
aot_cpp_mapping=not_applicable {} {}
```

这同时验证了：

1. pre-grad 与 post-grad 双向关联存在；
2. NPU `.run()` 行可定位三个 post-grad 节点；
3. 三个 post-grad 节点都能反向定位同一 `.run()` 行；
4. 空的 AOT C++ 行号表不会影响 JIT Python wrapper 页面。

## 当前边界

普通 NPU Triton pointwise/reduction 路径已经完成端到端验证。代码还接入了 template、combo、FlexAttention dK/dV、CATLASS、Meta/MLIR、DVM 和多流 extern 路径，但这些分支仍需要各自的专项实机用例。

Profiler timeline provenance 是另一项能力：它要把源码栈回填到 Ascend/CANN 时间线，不能由本次静态三栏页面通过来替代。当前结论仅覆盖编译期静态 provenance。
