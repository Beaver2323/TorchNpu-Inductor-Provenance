# CPU TorchInductor Provenance 可视化实测

## 结果

2026-08-20 已在 Tracking 环境完成 CPU provenance 的完整链路：

```text
torch.compile(CPU)
  -> TORCH_TRACE structured log
  -> tlparse 0.4.8
  -> index.html
  -> pre-grad / post-grad / generated code 三栏联动高亮
```

实测融合 kernel 为 `cpp_fused_add_mul_relu_0:1`，对应 `add`、`relu` 和
`mul` 三个 post-grad 节点。

## 安装版本

```text
rustc 1.97.1 (8bab26f4f 2026-07-14)
cargo 1.97.1 (c980f4866 2026-06-30)
tlparse 0.4.8
host: aarch64-unknown-linux-gnu
```

Rust/Cargo 安装在 `/home/z50063656/.cargo` 和
`/home/z50063656/.rustup`。新 shell 会通过 `.bashrc`/`.profile` 加载 Cargo
PATH；当前 shell 也可以执行：

```bash
source /home/z50063656/.cargo/env
```

## 生成 trace

测试必须从 `/home/z50063656/tmp` 启动：

```bash
cd /home/z50063656/tmp
source /home/z50063656/Tracking/activate_tracking.sh

RUN_DIR=$(mktemp -d /home/z50063656/tmp/tracking-cpu-provenance.XXXXXX)
export INDUCTOR_PROVENANCE=1
export TORCH_TRACE="$RUN_DIR/trace"
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR="$RUN_DIR/debug"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/cache"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1

python /home/z50063656/Tracking/cpu_provenance_demo.py
```

## 生成可视化报告

必须把具体 `.log` 文件交给 `tlparse`，不能把 trace 目录直接作为 provenance
输入：

```bash
TRACE_FILE=$(find "$RUN_DIR/trace" -name '*.log' -print -quit)
tlparse "$TRACE_FILE" \
  --inductor-provenance \
  --no-browser \
  -o "$RUN_DIR/tl_out"
```

成功标志：

```text
Stats { ok: 142 }
```

## 本次报告入口

> 本节记录的是早期 CPU 研究产物路径，产物目录未发布到本仓。当前已发布且可点击的
> NPU 演示入口见 [`triton_experimental` 产物索引](./inductor_provenance_demo/triton_experimental/README.md)。

- tlparse 总入口：`cpu_provenance_tlparse/index.html`
- Provenance 三栏高亮器：`cpu_provenance_tlparse/provenance_tracking_-_0_0_0.html`
- Kernel stack readable 页面：`cpu_provenance_tlparse/-_0_0_0/inductor_provenance_tracking_kernel_stack_traces_14_readable.html`
- 原始节点映射 JSON：`cpu_provenance_tlparse/-_0_0_0/inductor_provenance_tracking_node_mappings_13.json`

在总入口中找到 **Provenance Tracking**，点击
`provenance_tracking_-_0_0_0`。三栏分别是：

1. 输入/pre-grad GraphModule；
2. post-grad GraphModule；
3. Inductor 生成的 Python wrapper 与 CPU C++ kernel。

点击第三栏的 provenance handle 调用行，页面会高亮第二栏的 `add`、`relu`、
`mul`；再通过双向映射联动到第一栏的 `added`、`activated`、`mul`。

## 如何阅读三栏页面

建议按“左栏 -> 中栏 -> 右栏”阅读：原始图节点经过编译优化后，最终进入了
哪个 kernel。

| 区域 | 内容 | 本例 |
| --- | --- | --- |
| 左栏 `preGradGraph` | 接近用户模型的输入 FX 图 | `added`、`activated`、`mul` |
| 中栏 `postGradGraph` | AOTAutograd/Inductor 优化后的 ATen 图 | `aten.add.Tensor`、`aten.relu.default`、`aten.mul.Tensor` |
| 右栏 `generatedCode` | Inductor 生成的 Python wrapper 和 CPU C++ kernel | `cpp_fused_add_mul_relu_0` |

本例的完整来源关系：

```text
用户模型                  post-grad ATen 图             最终 kernel

added = x + y       ---> add.Tensor       ┐
activated = relu()  ---> relu.default     ├---> cpp_fused_add_mul_relu_0:1
return activated*2  ---> mul.Tensor       ┘
```

页面中粗体表示该行存在 provenance 映射，黄色表示当前选择行及其关联来源；
三个栏可以独立滚动。

推荐按以下顺序操作：

1. 点击左栏 `added = l_x_ + l_y_`，中栏应高亮
   `torch.ops.aten.add.Tensor(...)`。
2. 点击中栏 `relu`，左栏应高亮 `activated`，右栏仍对应同一个融合 kernel。
3. 点击右栏的 handle 注释或紧随其后的调用：

   ```python
   # [Provenance debug handles] cpp_fused_add_mul_relu_0:1
   cpp_fused_add_mul_relu_0(arg0_1, arg1_1, buf0)
   ```

   中栏的 `add`、`relu`、`mul` 应同时高亮。这是 kernel 到 post-grad 的
   反向追踪。

kernel 名可以拆解为：

```text
cpp_fused_add_mul_relu_0:1
│   │                 │ └─ debug handle，本次调用的唯一编号
│   │                 └── 第 0 个此类生成 kernel
│   └──────────────────── 融合了 add、mul、relu
└──────────────────────── CPU C++ Inductor kernel
```

右栏 C++ kernel 中的核心计算为：

```cpp
auto tmp2 = float(tmp0 + tmp1);        // add
auto tmp3 = std::max(tmp2, 0);         // relu
auto tmp5 = float(tmp3 * 2.0);         // mul
out_ptr0[x0] = tmp5;
```

循环范围是 8192，来自输入形状 `64 * 128`。三个逐元素操作被合并进一次
循环，可以减少中间 Tensor 和 kernel 调用开销。

当前 provenance 的最小追踪粒度是 kernel，而不是 kernel 内部的单条 C++
指令。因此点击该 kernel 时三个 post-grad 节点会一起高亮，不会把
`tmp2`、`tmp3`、`tmp5` 分别映射到不同节点。

`cpp_fused_add_mul_relu_0:1` 中的 `:1` 是 debug handle，不是版本号。原始
映射 JSON、生成代码注释和 kernel stack artifact 都用这个 key 关联同一次
kernel 调用。

## 自动验收结果

本次从生成 HTML 中解析出的行映射：

```text
index_provenance_section=pass
pre_post_highlight=pass
kernel_post_highlight=pass {'84': [10, 7, 4]}
post_kernel_highlight=pass {'10': [84], '4': [84], '7': [84]}
```

这里第三栏第 84 行是：

```python
# [Provenance debug handles] cpp_fused_add_mul_relu_0:1
cpp_fused_add_mul_relu_0(arg0_1, arg1_1, buf0)
```
