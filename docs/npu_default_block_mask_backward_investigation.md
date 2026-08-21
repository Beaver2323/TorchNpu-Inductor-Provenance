# NPU 默认 BlockMask backward 编译调查

更新时间：2026-08-21

## 当前结论

`flex_attention(..., block_mask=None)` 的 forward 已端到端跑通；backward 探针、
前向数值基准、反向 FX 图和 dK/dV 候选 kernel 均已建立，但本轮还没有得到完整的
backward 数值、provenance JSON 或 tlparse HTML。

阻塞位置已经收敛到 Triton Ascend 后端的 `bishengir-compile`：修复默认 BlockMask
哨兵值泄漏后，一个 dK/dV `ttadapter` 已从包含 `1073741824`、`8388608` 的异常循环边界，
缩小为只按真实 128 序列长度寻址的正常规模；编译进程仍持续占用约一个 CPU 核，
观察 11 分 48 秒仍未返回。关闭 torch_npu FlexAttention 的 backward 多消费者融合选项
也没有消除该长编译。

因此当前准确状态是：

```text
forward E2E                     已完成
backward FX/AOTAutograd 捕获     已完成
默认哨兵的 dK/dV 稀疏倍数归一化 已实现并通过契约测试
backward output_code.py          未生成
backward NPU 数值/provenance     未完成
剩余阻塞                         BishengIR dK/dV kernel 编译性能
```

## 背景：为什么 forward 通过不代表 backward 通过

训练图由 AOTAutograd 拆为独立的 forward 和 backward 编译单元。forward template
负责输出和 log-sum-exp 等保存量；backward 至少还要生成 dQ 与 dK/dV kernel。两者可以
选择不同模板、不同 tile 配置，并分别调用 Triton Ascend 编译器。

调试产物也按编译单元生成：

```text
model__0_forward_1.0/
  fx_graph_*.py
  output_code.py
  provenance JSON

model__0_backward_3.1/
  fx_graph_*.py
  output_code.py          # 只有 codegen/候选编译完成后才出现
  provenance JSON         # 只有 dump 阶段到达后才出现
```

本轮各次失败中，forward 目录都有 `output_code.py`，backward 目录只有
`fx_graph_readable.py`、`fx_graph_transformed.py` 和 `fx_graph_runnable.py`。这不是
artifact 丢失，而是 backward 尚未越过 template 候选预编译，所以也没有可供 tlparse
转换成行号的 backward wrapper。

## 默认 BlockMask 的哨兵为何会影响 backward

PyTorch 用 `1 << 30` 表示默认 mask 覆盖完整 Q/KV 序列。这个值是“逻辑上无限大”的
BlockMask metadata 哨兵，并不意味着长度 128 的输入真的有十亿个元素。

旧 dK/dV template 直接计算：

```text
SPARSE_Q_MULTIPLE = SPARSE_Q_BLOCK_SIZE // BLOCK_M1
q_start = q_block * SPARSE_Q_BLOCK_SIZE + ...
```

当 `SPARSE_Q_BLOCK_SIZE = 1073741824`、`BLOCK_M1 = 128` 时，倍数变成
`8388608`。这些常量进入 MLIR 的循环、除法、取模和地址计算，给后端造成巨大且不符合
真实张量形状的优化问题。

现在在 Python lowering 阶段把倍数限制到实际 tile 数：

```python
sparse_q_multiple = min(
    SPARSE_Q_BLOCK_SIZE // BLOCK_M1,
    ceil(Q_LEN / BLOCK_M1),
)
sparse_kv_multiple = min(
    SPARSE_KV_BLOCK_SIZE // BLOCK_N1,
    ceil(KV_LEN / BLOCK_N1),
)
```

并把 `SPARSE_Q_MULTIPLE`、`SPARSE_KV_MULTIPLE` 与
`EFFECTIVE_SPARSE_Q_BLOCK_SIZE` 作为编译期 option 注入 template。显式 BlockMask
不大于真实序列时，其原有行为不变；默认哨兵只被归一化为本次张量真正需要的 tile 数。

## 调查过程与证据

### 1. 原始实现：异常常量进入 dK/dV MLIR

目录：

`npu_default_block_mask_backward_device1_20260821`

- forward 编译成功并生成 `output_code.py`；
- backward FX 图成功生成；
- dK/dV 的 `bishengir-compile` 持续约 100% CPU，超过 11 分钟未返回；
- 保留的 `ttadapter` 含 `1073741824` 和 `8388608`，后者用于循环边界及除法/取模；
- 本次运行由我们主动中止，没有把超时误报成数值失败。

代表性旧 artifact：

`npu_default_block_mask_backward_device1_20260821/cache/tmpzso8x_c_/triton/0tyMPV3I8PxmyOD1TOLWWhC-q7uJXTM8kPgBFf-vfHQ/triton_flex_attention_bwd_dkdv_mask_in.ttadapter`

### 2. 第一版模板内限界：迅速失败并暴露错误层次

目录：

`npu_default_block_mask_backward_bounded_20260821`

第一版尝试在 `@triton.jit` kernel 内用 Python/Triton 混合条件计算有效倍数，66.54 秒后
所有 11 个 dK/dV 候选都被拒绝，外层表现为：

```text
NoValidChoicesError:
No compilable choices found for flex_attention_backward_dkdv_only
```

单独预编译保留候选后得到真正的内层错误：

```text
ValueError: Did you forget to add @triton.jit ?
(_builder argument must be provided outside of JIT functions.)
```

这说明 shape 常量归一化应在 Python lowering/模板实例化阶段完成，不能在 JIT 函数内
写普通 Python 条件表达式。本版没有保留为最终实现。

### 3. lowering 阶段限界：异常常量已从 dK/dV MLIR 消失

目录：

`npu_default_block_mask_backward_lowering_bound_20260821`

最终实现把上述三个有效参数在 `flex_attention.py` 中按每个候选配置计算，再注入
`flexattention_template.py`。代表性修复后 artifact：

`npu_default_block_mask_backward_lowering_bound_20260821/cache/tmpdb7szf9i/triton/XSbBlQZfmpSfDzEpSNpnMRT42gBVZSaSiXmaMNTpmWY/triton_flex_attention_bwd_dkdv_mask_in.ttadapter`

该文件为 189 行、25446 字节；检查结果为：

```text
1073741824    不存在
8388608       不存在
Q block base 128
```

这证明默认哨兵泄漏已经修复。随后 `bishengir-compile` 仍以约 99.7% CPU 运行，观察
11 分 48 秒未完成。修复后的 IR 规模已经接近显式 128 BlockMask 的 dK/dV kernel，
因此剩余长编译更可能属于 dK/dV 模板或 BishengIR pass 的共性问题，而不再是默认
BlockMask 特有的十亿级循环边界。

### 4. 关闭 backward 多消费者融合：未改变结论

目录：

`npu_default_block_mask_backward_no_hfusion_20260821`

探针的 `--disable-backward-hfusion` 会设置：

```python
npu_config.flex_attention.hfusion_enable_multiple_consumer_fusion = False
```

候选源码 metadata 证明该选项生效，但 `bishengir-compile` 仍持续约 99.9% CPU，3 分
51 秒无结果后主动停止。编译命令中仍出现的 `--enable-hfusion-compile=true` 属于更底层
BishengIR 通用开关，和上述 torch_npu 候选 option 不是同一个控制层。这个诊断选项
不会修改产品默认配置。

### 5. 设备竞争说明

首轮运行开始时物理 NPU 5 空闲，随后另一个分布式任务占用了物理 NPU 2～5，NPU 5
显存升至约 51 GB。为避免把资源争用当成代码问题，本任务转到当时空闲的物理 NPU 1，
并让它映射为进程内 `npu:0`。后续复现仍应先执行 `npu-smi info`，不要固定假设设备号。

## 当前代码修改

- `src/torch_npu/torch_npu/_inductor/kernel/flex_attention.py`
  - 在 `make_bwd_dkdv_kernel_options()` 中按 Q/KV 实际 tile 数归一化稀疏倍数；
  - 向 legacy/tasklist dK/dV template 注入有效倍数与有效 Q block size；
  - dK/dV dispatch spec 复用同一有效 KV 倍数。
- `src/torch_npu/torch_npu/_inductor/kernel/flexattention_template.py`
  - 删除 JIT 内对十亿级哨兵的直接除法；
  - legacy 与 tasklist 都使用注入的有效倍数；
  - Q 起始地址使用 `EFFECTIVE_SPARSE_Q_BLOCK_SIZE`。
- `src/torch_npu/test/_inductor/test_scheduling_contract.py`
  - 增加默认 BlockMask backward 稀疏倍数限界契约；
  - 同时防止旧的 JIT 内 `RAW_*` 计算和直接 `q_block * SPARSE_Q_BLOCK_SIZE` 回归。
- `npu_default_block_mask_backward_provenance_probe.py`
  - 比较 compiled FlexAttention 与 SDPA 的 forward、dQ、dK、dV；
  - 要求 forward/backward 分别产生 FlexAttention provenance mapping；
  - 提供仅用于定位的 `--disable-backward-hfusion`。

## 构建与回归

本轮没有重新编译 PyTorch，只重新执行 torch_npu editable build。最终构建日志：

`/tmp/tracking_torch_npu_backward_sparse_bound_build_2.log`

源码与 `build/packages` 的最终 SHA256 一致：

```text
flex_attention.py
2f950761c8cf8356a65c6be7176f9724976c8a918c8f32a867d5a3502a5ed795

flexattention_template.py
39709a66093591f955ac9b5d43dd0b6dcc8241826cb68ccb4242330958173eaa
```

从 `/home/z50063656/tmp` 启动的契约回归结果：

```text
test_provenance_tracing.py         Ran 5 tests   OK
test_lowering_device_dispatch.py   Ran 18 tests  OK
test_scheduling_contract.py        Ran 10 tests  OK
合计                               33 tests       OK
```

新增的精确契约单测单独执行也通过：`Ran 1 test in 0.064s, OK`。修改范围通过
`lintrunner`、`git diff --check` 和 Python 语法检查。

曾有一次把绝对文件路径错误地传给 `python -m unittest`，导致三个 unittest loader
错误；该命令没有加载任何测试，随后已经用正确入口完成上述 33 项回归，不能把它解释
为实现回归失败。

## 如何复现当前状态

所有命令必须从 `/home/z50063656/tmp` 启动：

```bash
cd /home/z50063656/tmp
source /home/z50063656/Tracking/activate_tracking.sh
npu-smi info

export ASCEND_RT_VISIBLE_DEVICES=1
export RUN_ROOT=/home/z50063656/Tracking/npu_default_block_mask_backward_recheck
export TORCH_TRACE="$RUN_ROOT/trace"
export TORCH_COMPILE_DEBUG=1

python /home/z50063656/Tracking/npu_default_block_mask_backward_provenance_probe.py \
  --output-dir "$RUN_ROOT/run"
```

`RUN_ROOT` 必须是不存在的新目录。若只验证多消费者融合假设，可追加
`--disable-backward-hfusion`；它不是推荐的长期修复。

当前预期是 forward 完成，backward 进入 dK/dV 候选编译后长时间停留。运行未完成前
不要执行 tlparse；因为没有 backward `output_code.py` 和 provenance JSON，HTML 不会
包含要验证的 backward kernel 映射。

## 下一步工作

1. 从修复后 cache 选取一个 dK/dV 候选，固定 shape/config，绕开 11 候选调度，建立
   单 kernel 的可重复 `bishengir-compile` 基线。
2. 保存完整编译命令和各阶段 IR；若工具支持 pass timing，定位是哪个 BishengIR pass
   占用时间。
3. 将显式 `BlockMask(BLOCK_SIZE=128)` 与默认 BlockMask 归一化后的同配置 IR 做结构化
   diff，确认剩余差异是否来自 full-block metadata、tasklist 或 hfusion 属性。
4. 逐项缩减 dK/dV kernel：先固定单候选，再分别关闭 tasklist/full-block 分支或拆分
   dK/dV，找出触发长编译的最小结构。
5. 只有 backward 成功生成 `output_code.py` 后，才继续数值、kernel mapping、stack、
   tlparse 三栏和四个 dK/dV dispatch 分支的 E2E 验证。

## 安全检查点

当前九个实现文件和三个测试文件已按白名单保存并重新应用到工作树：

```text
stash@{0}: codex: NPU provenance P1 backward compiler investigation
commit: 39677751f7cf15fce81903a58fc7adafcee81c5e
```

工作树还有其他进程生成的大量构建文件和第三方子模块状态。不要执行 `git clean`、
`git reset --hard` 或宽范围回退。
