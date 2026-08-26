# NPU FlexAttention template provenance 可视化演示

更新时间：2026-08-20

## 结论

NPU Triton template provenance 已在真实 Ascend 910B2 上跑通。验证模型使用
`FlexAttention` forward template，数值结果与
`torch.nn.functional.scaled_dot_product_attention` 一致，并生成如下映射：

```json
{
  "triton_flex_attention_fwd_mask_in:1": [
    "flex_attention",
    "sdpa_score0",
    "sdpa_mask0"
  ]
}
```

生成 wrapper 的第 606～608 行依次为来源说明、provenance handle 和真实调用：

```python
# Topologically Sorted Source Nodes: [flex_attention], Original ATen: [flex_attention]
# [Provenance debug handles] triton_flex_attention_fwd_mask_in:1
triton_flex_attention_fwd_mask_in.run(...)
```

这验证了 `NPUTritonScheduling.codegen_template()` 必须把最终的
`kernel_name` 传给 provenance hook；只写普通 source-node 注释不足以建立 kernel
与 post-grad 节点的双向关系。

## 本轮环境与结果

- PyTorch：`2.14.0a0+git8e86e0a`
- torch_npu：`2.14.0a0+git83cc452`
- 设备：Ascend 910B2
- 输入：`[1, 1, 128, 64]`，`float32`
- checksum：`-37.427276611328125`
- tlparse：`0.4.8`
- tlparse 结果：`Stats { ok: 282 }`

关键文件：

- [可重复运行脚本](../examples/npu_template_provenance_demo.py)
- 运行结果：`npu_template_provenance_verified_20260820/run/result.json`
- 运行日志：`npu_template_provenance_verified_20260820/demo.log`
- 原始 mapping：`npu_template_provenance_verified_20260820/run/debug/torchinductor/model__0_inference_0.0/inductor_provenance_tracking_node_mappings.json`
- 原始 output code：`npu_template_provenance_verified_20260820/run/debug/torchinductor/model__0_inference_0.0/output_code.py`
- tlparse 总入口：`npu_template_provenance_verified_20260820/tlparse/index.html`
- 三栏 provenance 页面：`npu_template_provenance_verified_20260820/tlparse/provenance_tracking_-_0_0_0.html`
- tlparse mapping：`npu_template_provenance_verified_20260820/tlparse/-_0_0_0/inductor_provenance_tracking_node_mappings_14.json`
- kernel stack 页面：`npu_template_provenance_verified_20260820/tlparse/-_0_0_0/inductor_provenance_tracking_kernel_stack_traces_15_readable.html`

> 上述 template 产物目录未随本仓发布；当前可点击演示见
> [`triton_experimental` 产物索引](./inductor_provenance_demo/triton_experimental/README.md)。

## 如何复现

所有测试从 `/home/z50063656/tmp` 启动。先用 `npu-smi info` 重新选择当时空闲的
物理设备，不要把下面的设备号 5 当成固定分配。

```bash
cd /home/z50063656/tmp
source /home/z50063656/Tracking/activate_tracking.sh

export RUN_ROOT=/home/z50063656/Tracking/npu_template_provenance_recheck
export ASCEND_RT_VISIBLE_DEVICES=5
export TORCH_TRACE="$RUN_ROOT/trace"

python /home/z50063656/Tracking/npu_template_provenance_demo.py \
  --output-dir "$RUN_ROOT/run"
```

然后生成 HTML：

```bash
source /home/z50063656/.cargo/env
TRACE_FILE=$(find "$RUN_ROOT/trace" -maxdepth 1 -name '*.log' -print -quit)

tlparse "$TRACE_FILE" \
  --inductor-provenance \
  --no-browser \
  -o "$RUN_ROOT/tlparse"
```

脚本要求 `--output-dir` 是一个新目录，以免旧产物混入本轮结论。

## 如何阅读三栏页面

打开 `provenance_tracking_-_0_0_0.html` 后：

1. 右栏生成代码定位到第 608 行，即
   `triton_flex_attention_fwd_mask_in.run(...)`。
2. 点击第 608 行，中央 post-grad 图会联动高亮第 6、4、5 行。
3. 这三行对应 `flex_attention`、`sdpa_score0` 和 `sdpa_mask0`。
4. 反向点击中央任一节点，也会回到右栏第 608 行。

页面内转换后的关系为：

```json
{
  "pyCodeToPost": {"608": [6, 4, 5]},
  "postToPyCode": {
    "4": [608],
    "5": [608],
    "6": [608]
  },
  "cppCodeToPost": {},
  "postToCppCode": {}
}
```

这里右栏是 JIT Python wrapper，所以 tlparse 使用 `pyCodeToPost`；两个 C++ 行号表
为空是正常现象。原始 PyTorch artifact 中的历史字段仍叫 `cppCodeToPost`，其中的
kernel key 是 `triton_flex_attention_fwd_mask_in:1`，不要混淆这两层字段语义。

## 为什么原演示显式构造 BlockMask

第一次实测把 `block_mask=None` 留在编译函数内，FlexAttention 会在被捕获的图中执行
`_create_empty_block_mask()`。当前 torch_npu 会用自己的
`torch_npu._inductor.lowering.make_reduction` 覆盖上游实现；该函数尚未接收 PyTorch
2.14 新增的 `strict_sum` 参数，因此在 template codegen 前报错：

```text
TypeError: make_reduction() got an unexpected keyword argument 'strict_sum'
```

因此当时的演示脚本在编译图外调用公开 API `create_block_mask()`，再把得到的
`BlockMask` 传入模型，用于先隔离验证 template provenance。

该限制现已解除：`strict_sum` 签名、缺失 full metadata、PyTorch 2.14 scheduler hook
和默认 mask 的 template grid 均已完成兼容，`block_mask=None` 已在真实 NPU 跑通。
新代码可直接使用默认参数；完整原理、复现和 HTML 阅读方法见
[默认 BlockMask 演示](./npu_default_block_mask_provenance_demo.md)。保留本显式 BlockMask
演示仍有价值，因为它提供了更小、更容易理解的 3 节点 template mapping。

首次失败产物保留在：

`/home/z50063656/Tracking/npu_template_provenance_p1_20260820`

## 当前覆盖边界

本轮已经覆盖真实 NPU FlexAttention forward template，显式和默认 BlockMask 均已
通过。FlexAttention backward dK/dV
的 legacy、tasklist、tasklist-no-split、reduce 四个运行时候选分支已通过 wrapper 契约
测试，确认每个 handle 紧邻自己的 `.run()`；但四个分支的真实 backward NPU 分派仍需
分别构造输入继续验证。combo 已完成调度契约，cache miss/hit 已完成真实 NPU 验证；
CATLASS、MLIR/DVM、multistream、AOTI 和 combo 真实执行仍属于后续专项覆盖。
