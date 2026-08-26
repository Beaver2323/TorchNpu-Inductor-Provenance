# NPU 默认 BlockMask provenance 演示

更新时间：2026-08-21

## 结论

`flex_attention(..., block_mask=None)` 已在真实 Ascend 910B2 上通过
`torch.compile(backend="inductor", fullgraph=True)` 端到端验证。它不再需要用户为了
绕开 torch_npu 兼容问题而提前构造 `BlockMask`，并且成功生成可交互的 tlparse
provenance 页面。

这里“端到端”特指 forward。2026-08-21 已开始训练 backward 专项：反向图可以捕获并
进入 dK/dV template 候选编译，默认 mask 的十亿级稀疏块哨兵也已在 lowering 阶段按
真实 tile 数归一化；但 `bishengir-compile` 观察 11 分 48 秒仍未返回，所以 backward
数值、provenance JSON 和 tlparse 页面尚未生成。完整证据见
[默认 BlockMask backward 编译调查](./npu_default_block_mask_backward_investigation.md)。

本轮结果：

```text
shape=[1, 1, 128, 64]
dtype=float32
checksum=-37.427276611328125
tlparse=Stats { ok: 258 }
kernel=triton_flex_attention_fwd_mask_in:12
```

checksum 与显式 `create_block_mask(..., BLOCK_SIZE=128)` 的既有基线完全一致，输出也已
通过 `torch.testing.assert_close` 与
`torch.nn.functional.scaled_dot_product_attention` 对比。

backward 限界修复后的最新 provenance、lowering 和 scheduler 三组完整契约回归为：

```text
provenance: Ran 5 tests, OK
lowering:   Ran 18 tests, OK
scheduler:  Ran 10 tests, OK
total:      33 tests, OK
```

## 为什么这个用例重要

`block_mask=None` 看似“没有 mask”，但 PyTorch 实际会在捕获图内调用
`_create_empty_block_mask()`。它会建立普通 KV/Q metadata，并用 `1 << 30` 作为覆盖
全序列的稀疏块大小哨兵。因此该用例一次覆盖了：

```text
Dynamo 捕获
  -> 默认 BlockMask 构造
  -> sum reduction lowering
  -> FlexAttention template 选择
  -> scheduler 融合
  -> NPU Triton 编译与运行
  -> provenance JSON
  -> tlparse 三栏页面
```

显式在图外创建 BlockMask 的旧基线主要验证 template provenance；默认 BlockMask
基线还验证了 PyTorch 2.14 与 torch_npu 的前置 lowering、metadata 和 scheduler
兼容性。

## 为何此前会失败

本轮逐层暴露并修复了四个真正影响原生 NPU Inductor 的兼容缺口：

1. PyTorch 2.14 调用 `make_reduction(..., strict_sum=...)`，torch_npu override 的旧签名
   不接收该关键字。修复后 `strict_sum` 被原样传给 `Reduction.create`。
2. 默认 BlockMask 的 `full_kv_num_blocks/full_kv_indices` 是 `None`。NPU template
   参数必须是具名 IR buffer，所以 lowering 现在创建同 shape、同 dtype、全零且已
   realize 的 metadata；backward 的 full-KV/full-Q 也同步处理。
3. PyTorch 2.14 的 `CUDACombinedScheduling` 新增 NVGEMM reduction-epilogue hook。
   NPU scheduler 现在明确返回 `False`，不再访问不存在的 CUDA-only 成员。
4. NPU forward template 原来用
   `NUM_SPARSE_Q_BLOCKS * SPARSE_Q_MULTIPLE` 决定 query tile 数。默认 mask 的
   `1 << 30` 哨兵会把 `[1,1,128,64]` 错误放大到 8,388,608 个 query tile，最终产生
   MTE 越界。现在 grid 使用真实的 `tl.cdiv(Q_LEN, BLOCK_M)`，与上游按真实 Q 长度
   发射 program 的语义一致。

失败阶段均保留了独立目录。前三类 codegen 前失败只有 FX 图、没有
`output_code.py`；设备越界阶段已经有 `output_code.py`，由其中的
`SPARSE_Q_BLOCK_SIZE=1073741824` 和巨大 `NUM_Q_TILES` 直接定位根因。

## 如何运行

所有测试必须从 `/home/z50063656/tmp` 启动。设备号需先根据 `npu-smi info` 重新选择，
不要假设物理 NPU 5 永远空闲。

```bash
cd /home/z50063656/tmp
source /home/z50063656/Tracking/activate_tracking.sh

export RUN_ROOT=/home/z50063656/Tracking/npu_default_block_mask_recheck
export ASCEND_RT_VISIBLE_DEVICES=5
export TORCH_COMPILE_DEBUG=1
export TORCH_TRACE="$RUN_ROOT/trace"

python /home/z50063656/Tracking/npu_default_block_mask_provenance_probe.py \
  --output-dir "$RUN_ROOT/run"
```

`--output-dir` 必须是新目录，避免旧 artifact 混入结论。再生成 HTML：

```bash
source /home/z50063656/.cargo/env
TRACE_FILE=$(find "$RUN_ROOT/trace" -maxdepth 1 -name '*.log' -print -quit)

tlparse "$TRACE_FILE" \
  --inductor-provenance \
  --no-browser \
  -o "$RUN_ROOT/tlparse"
```

原生 torch_npu Triton/Inductor 测试不要导入 `torch_npu.contrib.transfer_to_npu`。
该兼容层的 `_patch_has_triton()` 当前固定返回 `False`，会出现误导性的
“triton-ascend is not installed”以及 `Device npu not supported`。

## 如何阅读 HTML

打开：

`npu_default_block_mask_provenance_grid_verified_20260820/tlparse/provenance_tracking_-_0_0_0.html`

右栏第 898 行是最终 template 调用：

```python
# [Provenance debug handles] triton_flex_attention_fwd_mask_in:12
triton_flex_attention_fwd_mask_in.run(...)
```

点击第 898 行，中央 post-grad 图会高亮 10 行：

```json
{"898": [58, 4, 27, 69, 52, 49, 46, 45, 55, 5]}
```

其中：

- 第 58 行：最终 `flex_attention` HOP；
- 第 4、5、69 行：score/mask 子图与默认恒真 mask；
- 第 27、45、46、49、52、55 行：默认 BlockMask 的 counts、indices、transpose、
  reduction、dtype conversion 和零索引构造。

这比显式 BlockMask 基线映射的 3 个节点更多，因为默认 BlockMask 的构造本身也在
被编译图中。右栏仍是 JIT Python wrapper，因此有效字段是
`pyCodeToPost/postToPyCode`；AOT C++ 字段为空并不是错误。

## 关键文件

- [可重复运行探针](../examples/npu_default_block_mask_provenance_probe.py)
- 结果 JSON：`npu_default_block_mask_provenance_grid_verified_20260820/run/result.json`
- 运行日志：`npu_default_block_mask_provenance_grid_verified_20260820.log`
- 原始 mapping：`npu_default_block_mask_provenance_grid_verified_20260820/run/debug/torchinductor/model__0_inference_0.0/inductor_provenance_tracking_node_mappings.json`
- 生成代码：`npu_default_block_mask_provenance_grid_verified_20260820/run/debug/torchinductor/model__0_inference_0.0/output_code.py`
- tlparse 入口：`npu_default_block_mask_provenance_grid_verified_20260820/tlparse/index.html`
- 三栏 provenance 页面：`npu_default_block_mask_provenance_grid_verified_20260820/tlparse/provenance_tracking_-_0_0_0.html`
- tlparse mapping：`npu_default_block_mask_provenance_grid_verified_20260820/tlparse/-_0_0_0/inductor_provenance_tracking_node_mappings_14.json`
- kernel stack 页面：`npu_default_block_mask_provenance_grid_verified_20260820/tlparse/-_0_0_0/inductor_provenance_tracking_kernel_stack_traces_15_readable.html`

> 上述 BlockMask 产物目录未随本仓发布；当前可点击演示见
> [`triton_experimental` 产物索引](./inductor_provenance_demo/triton_experimental/README.md)。

## 当前覆盖边界

本轮已完成默认 BlockMask 的 forward、数值、代码生成、真实 NPU 执行和 provenance
可视化。backward 已不再只是“尚未执行”：专用探针、反向 FX 图、候选 template 和
哨兵限界修复都已完成，阻塞已精确到 dK/dV 的 BishengIR 编译阶段。由于失败发生在
backward `output_code.py` 生成前，当前没有反向 kernel mapping 或可供 tlparse 阅读的
反向 HTML。动态 shape、非 128 对齐序列和 AOTI 也仍需要专项覆盖。

## Backward 继续入口

- [backward 探针](../examples/npu_default_block_mask_backward_provenance_probe.py)
- [调查、artifact 与复现命令](./npu_default_block_mask_backward_investigation.md)
- 修复前 dK/dV artifact：`npu_default_block_mask_backward_device1_20260821/`（未发布）
- 修复后 dK/dV artifact：`npu_default_block_mask_backward_lowering_bound_20260821/`（未发布）
