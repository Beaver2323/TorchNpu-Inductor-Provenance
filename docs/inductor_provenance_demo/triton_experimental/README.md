# triton_experimental Inductor provenance 新手指南

本文说明 `torch_npu/_inductor/triton_experimental` 后端的 Inductor
provenance（来源追踪）是什么、如何实现、如何运行，以及当前已经验证到什么程度。
本文中的结论和产物均来自 Tracking 项目，不使用 `Pass` 项目的环境或结果。

### 仓库与交付关系

- [官方目标仓](https://gitcode.com/Ascend/pytorch)
- [开发 fork](https://gitcode.com/gcw_3ffySSwy/pytorch)
- 当前 worktree 的 `origin` 指向官方仓，`fork` 指向开发 fork；交付时应把分支 push
  到 `fork`，再向 `origin` 发起 PR。
- 源码已推送到开发 fork 的 `codex/triton-experimental-provenance-delivery` 分支，
  当前提交为 `bb356bffb`，官方基线为 `83cc45248`。
- [架构师个人预合入/历史参考仓](https://gitcode.com/rmch/npu_inductor_2.13.0)
  是历史参考，
  不是目标仓或 fork。
- [本 GitHub 仓](https://github.com/Beaver2323/TorchNpu-Inductor-Provenance)
  是前期研究与文档仓，
  不是源码交付目标。
- [工作流参考仓](https://gitcode.com/AllenGuanC/inductor-meta-worktree)
  只用于复用工作流程。

## 1. 当前结论

本分支已经为 `triton_experimental` 打通两类 provenance：

- 静态 provenance：在编译产物中建立“用户代码 / FX 节点 / NPU Triton kernel”映射，
  并可由 tlparse 展示为三栏联动页面。
- 运行时 provenance：在 NPU profiler Chrome trace 中，把编译期保存的 Python
  源码栈回填到实际执行的 NPU Triton kernel 事件。

已在 Ascend910B2 上用独立安装的 torch_npu wheel 验证：

- 普通前向融合 kernel 的静态映射正确，数值误差为 0。
- 普通前向与 backward kernel 都带有正确源码栈。
- rsplit reduction 产生的 partial 和 combine 两个 kernel 都带有正确源码栈。
- Llama 风格 RMSNorm + SwiGLU 残差块在两组动态形状上的前向、反向、输入梯度和
  参数梯度均正确；9 个 timeline kernel 事件都能回溯到模块源码。
- ConvNeXt 块的两组动态前向，以及 TransformerEncoderLayer 的基准前反向和第二形状
  动态前向均通过模块级 provenance 验证；超出该范围的后端限制单独记录，不计为通过。
- NPU 原始 trace 的事件根、名称、分类和 flow 没有被适配器的临时格式污染。
- tlparse 使用 `--inductor-provenance` 成功解析最小案例的 137 条、Llama 案例的 250 条
  结构化日志记录。

本任务范围只包括 `triton_experimental`。默认 NPU Inductor、MLIR、DVM、AKG、
CATLASS 和 torchair 不属于本交付范围。

## 2. 先理解 torch.compile 编译链路

可以把一次 `torch.compile(..., backend="inductor")` 简化为：

```text
Python 模型
  -> TorchDynamo 捕获 FX graph（pre-grad）
  -> AOTAutograd 拆分 forward / backward
  -> Inductor 图优化（post-grad）
  -> Scheduler 融合多个节点
  -> triton_experimental 生成 wrapper 和 NPU Triton kernel
  -> Triton Ascend 编译并在 NPU 上执行
```

优化过程中，一个用户操作可能被分解成多个 FX 节点；多个 FX 节点也可能融合成一个
kernel。如果只看最终 kernel 名，很难回答下面的问题：

- 这个 kernel 来自模型的哪几行代码？
- 它包含 post-grad graph 中的哪些节点？
- 某个 FX 节点最后进入了哪个 kernel？
- profiler 时间线上很慢的 kernel 应该回到哪段模型代码定位？

provenance 的作用就是保存并展示这些关系。

## 3. 社区 PyTorch 已提供什么

PyTorch 社区的 provenance 主体位于：

- `torch/_inductor/config.py`：开关和级别。
- `torch/fx/traceback.py`、图变换观察器和 `compile_fx.py`：在图变换中保留来源元数据。
- `torch/_inductor/debug.py`：给 kernel 调用分配 debug handle，汇总节点映射和源码栈。
- 各 codegen scheduler 的 `codegen_comment(...)`：在 kernel 发射前登记来源。
- `torch/_inductor/profiler.py`：把编译期 kernel 信息关联到 CUDA/Kineto timeline。
- tlparse：把结构化 `TORCH_TRACE` 日志转换为三栏高亮页面。

社区文档说明静态高亮当前覆盖 Triton kernel、C++ kernel 和 combo kernel，并支持
TorchInductor JIT 与 AOTInductor 产物。社区的完整能力还包括 extern kernel、
`kernel_information.json`、cache 一致性和 profiler timeline 等测试。

本次 NPU 交付复用社区的图来源追踪、debug handle、artifact 和 timeline 处理算法；
只在 `triton_experimental` 的 kernel 发射点补齐登记，并在 torch_npu profiler 一侧
适配 Ascend trace 格式。当前真实验收覆盖 JIT Triton、forward/backward 和 rsplit，
不能据此宣称 AOTInductor、extern 或其他 NPU 后端已经完成验收。

## 4. 配置级别

`torch._inductor.config.trace.provenance_tracking_level` 的取值为：

| 级别 | 含义 | 建议用途 |
| --- | --- | --- |
| 0 | 关闭，默认值 | 普通运行，开销最低 |
| 1 | normal，保留完整图变换来源关系 | tlparse 三栏映射、详细问题定位 |
| 2 | basic，保留 kernel 的基础来源/源码栈，但跳过一部分昂贵的图变换观察 | 只关心 kernel 源码栈时降低编译开销 |

环境变量 `INDUCTOR_PROVENANCE=1` 等价于打开 level 1。开启
`trace.provenance_tracking_to_timeline=True`（环境变量
`TORCH_COMPILE_DEBUG_EXTEND=1`）时，有效级别至少会提升到 1。

为了让 profiler 中的实际 kernel 名能和编译期信息稳定匹配，还应打开：

```python
"triton.unique_kernel_names": True
```

## 5. 静态 provenance 与运行时 provenance

### 5.1 静态 provenance

静态链路发生在编译期间，不需要 profiler。`triton_experimental` 在每次真正发射
kernel 调用之前执行：

```python
self.codegen_comment(node_schedule, kernel_name)
```

社区实现随后完成三件事：

1. 从 scheduler nodes 收集 post-grad origins 和 Python stack traces。
2. 为本次 kernel 调用分配唯一 debug handle，例如
   `triton_unk_fused_add_mul_relu_0:1`。
3. 把映射写入 JSON，并把同一个 handle 写进生成 wrapper 的注释。

主要 artifact 是：

- `inductor_provenance_tracking_node_mappings.json`
- `inductor_provenance_tracking_kernel_stack_traces.json`
- `output_code.py`
- `TORCH_TRACE` 里的 pre-grad、post-grad 和生成代码 artifact

优点是关系确定、易于复现，不受 profiler 采样和 runtime flow 影响。难点是必须在图
变换和融合过程中持续保存来源，并在“每个实际 kernel 调用”的正确位置登记。

### 5.2 运行时 provenance

运行时链路先正常编译，再用 NPU profiler 导出 Chrome trace。处理器根据：

- `Torch-Compiled Region` / `CompiledFunctionBackward` 的执行区间；
- CPU kernel launch；
- `torch_to_npu` host-to-device flow；
- NPU device kernel；
- 编译期保存的 kernel 名与源码栈；

找到实际执行的 device kernel，并给它的 `args.stack` 写入 Python 源码栈。这样可在
Perfetto/Chrome trace 中从耗时 kernel 直接回到模型源码。

运行时适配更难，因为它依赖编译期名称和 profiler 事件之间的稳定关联，且 NPU trace
与社区 CUDA/Kineto trace 的 schema 不同：

- NPU trace 顶层可以直接是事件列表，而社区处理器接收 `traceEvents` 字典。
- NPU 使用 `torch_to_npu`，社区算法识别 `ac2g`。
- NPU flow 端点可能集中在文件尾部，社区算法要求 flow 紧跟关联事件。
- NPU 的 `ts`/`dur` 可能是字符串。
- backward 的编译期名称可能是 `triton_unk_*`，真实 device 名却是 `k_*`。

适配器在深拷贝上临时规范化这些差异，调用社区 `_InductorTraceProcessor`，然后只把
最终 `stack` 拷回原始 NPU trace。临时的 `ac2g`、`uid`、`cat=kernel` 和逻辑别名
不会泄漏到导出的原始 trace。

## 6. 为什么必须单独验证 backward 和 rsplit

普通 forward 通过只能证明单一名称、单一 launch 的基础路径。

backward 值得单独验证，是因为它由 AOTAutograd 延迟编译，使用
`CompiledFunctionBackward` 区域，而且本环境中真实 device kernel 使用 `k_*` 名称。
通过 backward 验证可证明：

- forward/backward 图信息没有串线；
- backward graph key 能被找到；
- `k_*` 到编译期 `triton_unk_*` 的受限别名匹配有效；
- 最终 trace 保留真实 `k_*` 名称，同时获得正确 stack。

不要求一定使用 FlexAttention backward。任何能稳定进入
`triton_experimental`、产生真实 backward Triton kernel，并覆盖相同关联链路的模型
都可验证基础 contract。本交付保留小型 `sin + relu + mul` 作为快速 smoke test，并把
在社区 provenance 能力边界内通过的 Llama 风格 RMSNorm + SwiGLU 残差块作为远端主演示。
这里的“通过”包括数值/梯度、timeline 以及 forward 三栏映射；backward 验证
post-grad 到 kernel 的映射，但不承诺每个 pre-grad 节点都有到生成代码的完整传递链。
FlexAttention backward 只能作为更复杂的模板/融合专项，不能替代基础链路验证。

rsplit reduction 一次 scheduler 计划会发射两个运行时 kernel：partial 先写 workspace，
combine 再归并。如果只在整个 schedule 末尾登记一次，两个 launch 中至少一个没有独立
handle。当前实现在每次 launch 前分别调用 `codegen_comment`。rsplit 通过证明“一份
node schedule 对应多个 runtime kernel”时也能逐 kernel 追踪。

## 7. 源码改动

### 7.1 triton_experimental codegen

文件：`torch_npu/_inductor/triton_experimental/codegen/triton.py`

- 普通路径：在 `final_kernel.call_kernel(...)` 前登记带 kernel 名的 provenance。
- rsplit 路径：把 `node_schedule` 传给 `_npu_call_rsplit_kernels`，在 partial 和
  combine 两次 `generate_kernel_call(...)` 前分别登记。

登记必须靠近实际 launch；如果提前对整个 schedule 只登记一次，kernel 名、debug
handle 和调用位置就可能错位。

### 7.2 NPU profiler adapter

文件：`torch_npu/profiler/_inductor_profiler.py`

- 支持事件列表和 `traceEvents` 字典两种顶层格式。
- 规范化 flow 顺序和 `torch_to_npu -> ac2g`。
- 只在编译期确实存在对应信息时，临时匹配 backward `k_*` 名称。
- 复用社区 `_InductorTraceProcessor`，不复制其核心关联算法。
- 支持 gzip、最大事件数保护和编译期临时状态清理。
- 处理结束后保持原始 NPU trace schema，只增加 `args.stack`。

公开入口：

```python
from torch_npu.profiler import inductor_trace_handler
```

## 8. 从 wheel 运行验证

所有测试都从 `/home/z50063656/tmp` 启动，避免在 torch_npu 源码树内导入导致级联污染。
下列绝对路径记录原验证环境；在其他机器复现时，应替换为本仓和目标源码仓的实际路径。

本轮验证 wheel：

```text
/home/z50063656/Tracking/triton_experimental_delivery/wheels/
torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl
```

SHA256：

```text
fe8a90dec309a3d6089dd7807a56d3bc8a4f7f0bd886c755c5199f92237dc22d
```

这是面向本功能验证的 wheel，构建时使用项目提供的
`DISABLE_INSTALL_TORCHAIR=TRUE`，因此不应作为包含 torchair 的完整发布 wheel。
源码提交/PR 才是正式代码交付；wheel 用来证明这些源码可构建、可安装、可在 NPU
运行。

为了不改变现有 conda 环境，本轮安装到独立 target：

```bash
cd /home/z50063656/tmp
source /home/z50063656/Tracking/activate_tracking.sh

python -m pip install --no-deps \
  --target /home/z50063656/Tracking/triton_experimental_delivery/wheel_target_20260827_v10 \
  /home/z50063656/Tracking/triton_experimental_delivery/wheels/torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl

export PYTHONPATH=/home/z50063656/Tracking/triton_experimental_delivery/wheel_target_20260827_v10:$PYTHONPATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=7
```

静态映射用例：

```bash
python /home/z50063656/Tracking/worktrees/torch_npu_triton_provenance_delivery/test/_inductor/test_triton_experimental_enable.py \
  TestTritonExperimentalProvenance.test_kernel_maps_to_post_grad_nodes -v
```

无需 NPU 的 rsplit 调用顺序单测：

```bash
python /home/z50063656/Tracking/worktrees/torch_npu_triton_provenance_delivery/test/_inductor/test_triton_experimental_enable.py \
  TestTritonExperimentalProvenance.test_rsplit_maps_each_runtime_kernel -v
```

profiler schema 适配单测：

```bash
python /home/z50063656/Tracking/worktrees/torch_npu_triton_provenance_delivery/test/profiler/test_inductor_profiler.py -v
```

代表性模块的结构化验证范围见
[model_validation_result.json](./model_validation_result.json)。其中 Llama 的数值、梯度、
timeline、forward 三栏映射和 backward post-grad→kernel 映射通过；backward
pre-grad→生成代码的覆盖遵循社区 PyTorch 的现有边界。ConvNeXt 和 Transformer 的后端
边界未混入 PASS 计数。

## 9. 生成静态 tlparse 页面

```bash
cd /home/z50063656/tmp
source /home/z50063656/Tracking/activate_tracking.sh
export PYTHONPATH=/home/z50063656/Tracking/triton_experimental_delivery/wheel_target_20260827_v10:$PYTHONPATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=7

export TORCH_TRACE=/home/z50063656/Tracking/triton_experimental_delivery/my_torch_trace
export DEMO_ROOT=/path/to/TorchNpu-Inductor-Provenance/docs/inductor_provenance_demo/triton_experimental
python "$DEMO_ROOT/static_probe.py" \
  --output-dir /home/z50063656/Tracking/triton_experimental_delivery/my_static_run \
  --expect-mapped

tlparse -i "$TORCH_TRACE"/*.log \
  -o /home/z50063656/Tracking/triton_experimental_delivery/my_tlparse \
  --no-browser
```

远端主演示使用 Llama 风格 RMSNorm + SwiGLU 残差块，同时覆盖动态形状、forward、
backward、输入/参数梯度、静态映射和运行时 timeline：

```bash
export TORCH_TRACE=/home/z50063656/Tracking/triton_experimental_delivery/my_llama_trace
python "$DEMO_ROOT/llama_swiglu_demo.py" \
  --output-dir /home/z50063656/Tracking/triton_experimental_delivery/my_llama_run

tlparse -i "$TORCH_TRACE"/*.log \
  -o /home/z50063656/Tracking/triton_experimental_delivery/my_llama_tlparse \
  --no-browser
```

探针会要求输出目录尚不存在，并在一个运行中验证 `[2, 32, 256]` 与
`[3, 24, 256]` 两组形状。

验证 basic level 2 时给 probe 增加 `--level 2`；本轮真实 NPU 结果仍映射到同一个
`triton_unk_fused_add_mul_relu_0:1 -> add/relu/mul`，数值误差为 0。

不要把目录传给旧版 `tlparse parse` 子命令；当前实测命令是把具体 `.log` 文件作为
位置参数，并使用 `-i/--inductor-provenance`。

## 10. 如何阅读三栏 HTML

主演示拆成两个独立页面：

- [Llama forward 三栏 HTML](./llama_swiglu/provenance_tracking_forward.html)，页面中的
  AOT ID 是 `0_forward`；
- [Llama backward 三栏 HTML](./llama_swiglu/provenance_tracking_backward.html)，页面中的
  AOT ID 是 `0_backward`。

[兼容入口](./llama_swiglu/provenance_tracking.html)指向内容完全相同的 forward 页面，
因此此前关注的左栏第 33 行可以继续用原 URL 查看。每个页面只表示一个编译单元，
不再把一个 HTML 描述为同时包含 forward 和 backward。原来的
[三操作 smoke 页面](./provenance_tracking.html)继续保留，便于第一次阅读时快速定位。

三栏含义：

| 面板 | 内容 | 回答的问题 |
| --- | --- | --- |
| 左栏 | pre-grad/input FX graph | 最接近用户模型的操作是什么 |
| 中栏 | post-grad FX graph | 分解、融合和优化后还有哪些节点 |
| 右栏 | Inductor 生成的 Python wrapper / kernel call | 最终由哪个 kernel 执行 |

阅读方法：

1. 在右栏找到带 `[Provenance debug handles]` 的 kernel 调用。
2. 点击或悬停该调用行，页面会把对应的 post-grad 节点标黄。
3. 再观察左栏的关联高亮，回溯到原始操作。
4. 加粗表示该行至少存在一条可交互的相邻映射；它不保证
   pre-grad→post-grad→生成代码三段一定全部连通。普通文本也不代表错误。
5. kernel 名末尾的 `:1` 是一次调用的 debug handle，不是 Triton launch 参数。

forward 页面中，左栏第 33 行的 `linear_2` 会映射到中栏第 34～38 行，其中第 36～38
行继续映射到右栏第 222、350、355 行。因此这个页面可完整演示三栏联动。

Llama forward 页面中，右栏可看到多个 Triton 与 extern kernel handle。例如：

```text
triton_unk_fused_mean_mul_pow_rsqrt_0:1
  <-> post-grad: mean, pow, rsqrt, add, mul ...
  <-> 用户模型: RMSNorm variance / normalization

triton_unk_fused_mul_silu_view_1:4
  <-> post-grad: mul, div, exp, neg, view ...
  <-> 用户模型: SwiGLU gate/up projection

triton_unk_fused_add_addmm_view_2:6
  <-> post-grad: add, addmm, view
  <-> 用户模型: down projection + residual
```

backward 页面包含 `silu_backward`、reduction 和 RMSNorm gradient 映射。该页左栏第 33
行 `linear_2` 只直接映射到中栏 `permute_2`，而 `permute_2` 没有独立生成 kernel，因而
右栏不会随之高亮。真正调用 `extern_kernels.mm:8` 的中栏 `mm` 节点有 post-grad→代码
映射，但社区 backward graph 当前没有足够的 `from_node` 元数据把该 `mm` 再关联回这个
pre-grad `linear_2`。这属于社区 PyTorch 的现有静态 provenance 覆盖边界，不是
`triton_experimental` 后端丢失 kernel handle。完整原始关系见
[llama_swiglu_node_mappings.json](./llama_swiglu/llama_swiglu_node_mappings.json)。

这与社区源码的实现一致：`torch/_inductor/debug.py` 中
`create_mapping_pre_post_grad_nodes()` 只递归消费 `from_node`；
`torch/_inductor/utils.py` 在 backward 分支也明确记录 backward node 当前可能没有
`from_node`。本交付没有额外合成社区不存在的跨图关系。

两个页面里的 FX `GraphModule` 都显示 `def forward` 是正常现象。FX 对任意已捕获的计算图
统一生成 `forward` 入口；在 backward 页面里，这个函数的输入是保存的张量和 tangent，
输出是各输入/参数的梯度，所以它仍然代表反向图，而不是又执行一次模型前向。

JIT 模式的第三栏是 Python wrapper，因此 tlparse 内部有效行号映射是
`pyCodeToPost/postToPyCode`。原始 JSON 为兼容历史格式仍使用
`cppCodeToPost/postToCppCode` 保存 kernel key；AOT C++ 第三栏为空不表示 NPU 映射
失败。

tlparse 会生成多个 HTML，因为它们职责不同：`index.html` 是导航页，
`failures_and_restarts.html` 汇总重编译/失败，`compilation_metrics_*.html` 展示指标，
`provenance_tracking_<compile-id>.html` 才是某次编译图的三栏页面。文件名中的
`-_0_0_0` 是 tlparse 编译标识，不表示生成了四份相同结果。

本次 Llama trace 的 forward/backward 记录使用同一个可见 compile id；把整份 trace 一次
交给当前社区 `tlparse` 时，同名 highlighter 最终展示后出现的 backward 记录。为了让两种
图都可审阅，本仓使用同一份原始 trace 分别生成 forward-only highlighter，并把完整 trace
生成的 backward highlighter 原样保留。这只是拆分展示，页面格式和映射语义没有超出社区
`tlparse`。

## 11. 生成运行时时间线

先把本仓演示目录设置为 `DEMO_ROOT`：

```bash
export DEMO_ROOT=/path/to/TorchNpu-Inductor-Provenance/docs/inductor_provenance_demo/triton_experimental
```

普通 forward/backward：

```bash
python "$DEMO_ROOT/timeline_probe.py" \
  --output-dir /home/z50063656/Tracking/triton_experimental_delivery/my_timeline_forward_backward
```

Llama 动态形状前反向主演示：

```bash
python "$DEMO_ROOT/llama_swiglu_demo.py" \
  --output-dir /home/z50063656/Tracking/triton_experimental_delivery/my_llama_run
```

rsplit 双 kernel：

```bash
python "$DEMO_ROOT/rsplit_timeline_probe.py" \
  --output-dir /home/z50063656/Tracking/triton_experimental_delivery/my_timeline_rsplit
```

得到的 `*.pt.trace.json` 可载入 Perfetto。选中以 `triton_` 或 `k_` 开头的 device
kernel，在事件参数中查看 `stack`。仓库内提供三份可复现 trace：

- [Llama forward/backward trace](./llama_swiglu/llama_swiglu_timeline_trace.json)
- [普通 forward/backward trace](./timeline_forward_backward_trace.json)
- [rsplit partial/combine trace](./timeline_rsplit_trace.json)

## 12. 本轮验收证据

| 验证项 | 结果 | 关键证据 |
| --- | --- | --- |
| wheel 构建 | PASS | 34 MiB，zip 完整性通过 |
| wheel 源码一致性 | PASS | 三个产品文件与 staged 源码 SHA256 一致 |
| profiler 适配单测 | PASS | 12/12（含同名防串线、长名称恢复、模块前反向） |
| rsplit 发射顺序单测 | PASS | 1/1 |
| NPU 静态映射套件 | PASS | 5/5（level 0/1/2、动态形状 mapping 隔离） |
| 静态 probe | PASS | Ascend910B2，max abs diff = 0 |
| level 2 静态 probe | PASS | 同一 kernel→add/relu/mul 映射，max abs diff = 0 |
| tlparse | PASS | Stats `{ ok: 137 }` |
| forward/backward timeline | PASS | 两个 device kernel 均有 stack |
| rsplit timeline | PASS | partial/combine 两个 device kernel 均有 stack |
| 代表性模型/模块套件 | PASS | NPU 6 串行回归 3/3，无 skip，238.232 秒 |
| Llama 远端主演示 | PASS（社区边界内） | 两组动态形状、9 个 timeline kernel 事件；forward-only tlparse 202 条、完整 trace 250 条；forward 三栏映射完整，backward post-grad→kernel 通过，部分 pre-grad→backward 代码链不完整 |
| AOTInductor 可行性门禁 | BLOCKED | 当前硬件/基线不满足 NPU AOTI 验证前提，详见 12.2 节 |

静态 probe 中 `fxgraph_cache_hit=0`、`fxgraph_cache_bypass=1` 的原始 trace 原因是
`Unsupported post grad custom pass`：`triton_experimental/fx_passes.py` 把普通
`_composed` callable 安装到 `post_grad_custom_post_pass`，没有提供缓存键所需的
`CustomGraphPass.uuid()`。这是自定义 post-grad pass 的 cache identity 限制，不是
provenance 映射失败；本轮 trace 没有把 pre-grad pass 记录为直接 bypass 原因。

### 12.1 代表性模块 provenance A/B 因果排除

2026-08-28 在同一 Ascend910B2、物理 NPU 6 上关闭编译缓存，对两个失败边界做了独立
A/B。A 组为 level 0 且关闭 timeline provenance，B 组为 level 2 且打开 timeline
provenance；两组在相同位置生成相同失败 kernel。

- ConvNeXt 两组的失败源码 SHA256 都是
  `4347d006bc6bffc75c1f1707ee680208933ecf49e8c3b85159144766f34cf506`，并在同一行
  生成非法 Python 赋值。
- Transformer 两组失败 kernel 源码区域 SHA256 都是
  `f994c3761019a9f80abd49b444875b939c97a05e24a18c54f72e5899cdeb5722`，同样因
  `tl.store` mask 无法广播而耗尽所有 Triton config。

因此两个边界均为 `NOT_CAUSED_BY_PROVENANCE`。完整证据见
[provenance_ab_result.json](./provenance_ab_result.json)，复现入口为
[provenance_ab_probe.py](./provenance_ab_probe.py)。

### 12.2 为什么本轮不能把 AOTInductor 标为完成

社区 AOTInductor 的 provenance 会在 `.pt2` 包内生成
`model/data/aotinductor/model/kernel_information.json`。它与 JIT 静态 JSON 使用相同的
来源登记信息，但还必须完整通过 NPU C++ wrapper、Triton 二进制打包、C++ runtime ABI、
`.pt2` 加载和数值运行。因此，JIT 的 `node_mappings.json` 成功不能替代 AOTI 验收。

2026-08-27 做了两组真实 NPU 可行性诊断，诊断原型没有保留为交付代码：

1. 当前 `triton_experimental` 只注册 Python wrapper，查询
   `get_wrapper_codegen_for_device("npu", cpp_wrapper=True)` 返回 `None`。原型补入 NPU
   C++ wrapper 后，编译继续暴露 `DEVICE_TO_ATEN` 映射和所选 Triton 二进制未写入
   `CudaKernelParamCache` 两个实验后端缺口。
2. 原型补齐这两个缺口后已经进入最终 C++ 编译，但当前 PyTorch 2.14 生成代码与 wheel
   内 NPU AOTI runtime header 的 ABI 不匹配，包括 model container 构造函数、
   `did_call_load_constants` 和 `LazyKernelCompileResult`。同一 wheel 的默认 NPU AOTI
   对照也在 lazy AOTI 基础设施处失败（`KeyError: 'GridNpu'`），证明这不是 provenance
   登记点单独可以解决的问题。

此外，仓库现有 `torch_npu/_inductor/docs/feature/aoti/overview.md` 的设备支持说明只列出
Atlas A5；本机 `npu-smi` 显示为 Ascend910B2。继续修改共享 AOTI runtime/header 或默认
后端会超出“只负责 `triton_experimental`”的需求边界，所以本轮撤回未验证原型，继续以
已完整验收的 v10 JIT wheel 作为交付基线。

后续重新开启 AOTI 验证至少需要同时满足：

- 使用文档声明支持的 A5 设备；
- PyTorch、torch_npu wheel 以及 NPU AOTI runtime header 属于同一兼容版本；
- 同环境下默认 NPU AOTI 的最小 `.pt2` 编译、加载和运行先通过；
- 然后再为 `triton_experimental` 补 C++ wrapper 与二进制缓存发布，并验证包内
  `kernel_information.json` 的 kernel 名、stack、pre/post-grad nodes 以及加载数值。

## 13. 产物索引

- [Llama forward 三栏 HTML](./llama_swiglu/provenance_tracking_forward.html)：`0_forward` 独立页面。
- [Llama backward 三栏 HTML](./llama_swiglu/provenance_tracking_backward.html)：`0_backward` 独立页面。
- [Llama 三栏兼容入口](./llama_swiglu/provenance_tracking.html)：内容与 forward 页面相同。
- [Llama 验证结果](./llama_swiglu/llama_swiglu_result.json)：两组形状、数值/梯度和 timeline 摘要。
- [Llama node mappings](./llama_swiglu/llama_swiglu_node_mappings.json)：forward/backward 静态映射。
- [Llama kernel stacks](./llama_swiglu/llama_swiglu_kernel_stacks.json)：运行时 kernel 源码栈。
- [Llama timeline trace](./llama_swiglu/llama_swiglu_timeline_trace.json)：可载入 Perfetto。
- [llama_swiglu_demo.py](./llama_swiglu_demo.py)：主演示独立复现脚本。
- [model_validation_result.json](./model_validation_result.json)：三种代表性模型/模块验证矩阵。
- [provenance_ab_result.json](./provenance_ab_result.json)：level 0/2 因果对照。
- [provenance_ab_probe.py](./provenance_ab_probe.py)：边界 A/B 独立探针。
- [static_result.json](./static_result.json)：wheel 静态 level 1 运行摘要。
- [static_level2_result.json](./static_level2_result.json)：basic level 2 的真实 NPU 静态运行摘要。
- [node_mappings.json](./node_mappings.json)：kernel 与 post/pre-grad 节点关系。
- [kernel_stack_traces.json](./kernel_stack_traces.json)：kernel 与 Python 源码栈关系。
- [provenance_tracking.html](./provenance_tracking.html)：下载后可离线打开的 tlparse 三栏页面。
- [timeline_forward_backward_result.json](./timeline_forward_backward_result.json)：普通前后向摘要。
- [timeline_forward_backward_trace.json](./timeline_forward_backward_trace.json)：普通前后向 Perfetto trace。
- [timeline_rsplit_result.json](./timeline_rsplit_result.json)：rsplit 摘要。
- [timeline_rsplit_trace.json](./timeline_rsplit_trace.json)：rsplit Perfetto trace。
- [static_probe.py](./static_probe.py)：静态 provenance 独立复现实验。
- [timeline_probe.py](./timeline_probe.py)：普通 forward/backward timeline 独立复现实验。
- [rsplit_timeline_probe.py](./rsplit_timeline_probe.py)：rsplit timeline 独立复现实验。

## 14. 尚未完成与风险

- 尚未在稳定独占窗口把 wheel 安装进长期 Tracking conda 环境；当前使用独立 target，
  避免与其他进程修改环境发生冲突。
- `triton_experimental` 的 AOTInductor `kernel_information.json` 尚未验收；本轮已完成
  可行性门禁，确认当前 910B2 + PyTorch/torch_npu 2.14 基线同时受设备支持范围和共享
  AOTI lazy/ABI 问题阻塞。不能用已通过的 JIT JSON 代替，恢复条件见 12.2 节。
- level 2 静态映射、gzip、事件上限/状态清理和 list/dict trace root 均已完成专项
  回归；level 2 runtime timeline 尚未单独重复采集，因为 timeline 开关会把有效级别
  至少提升到 1，运行时关联算法与显式 level 1 相同。
- 当前验证 wheel 禁用了 torchair；如需要完整发布 wheel，应在标准发布构建环境中重新
  启用 torchair，而不是把本验证 wheel 当成正式发行包。
- ConvNeXt backward 和 Transformer 第二动态形状 backward 的边界均已用 level 0/2
  A/B 排除 provenance 因果；它们没有被包装成通过案例。
- 本地验证 wheel 的版本标签来自基线 commit `83cc452`；功能源码随后以提交
  `bb356bffb` 推送到开发 fork。判断验证 wheel 内容时仍应结合本页记录的 wheel SHA
  和文件哈希，不能只看版本字符串。
- 本页、演示 HTML、静态映射、timeline trace/result 和复现脚本已同步到前期研究与
  文档仓；正式源码交付仍以 GitCode 分支及后续面向官方仓的 PR 为准。
