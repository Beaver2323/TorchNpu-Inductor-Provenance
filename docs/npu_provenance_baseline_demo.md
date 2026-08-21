# NPU TorchInductor Provenance 基线演示

> 本文保留的是修改前 baseline，用来证明原始缺口。修改后的真实 NPU 三栏演示见 [npu_provenance_visualization_demo.md](./npu_provenance_visualization_demo.md)。

## 1. 演示结论

2026-08-20 已在 Ascend 910B2 上成功执行带 provenance 开关的 NPU
`torch.compile` 模型。模型数值正确，FX 图级映射完整，但未适配的 torch_npu
没有生成 kernel 级映射：

```text
preToPost/postToPre       非空  -> PyTorch 通用图追踪链路可复用
cppCodeToPost/postToCppCode 为空 -> NPU codegen 缺少 kernel provenance hook
```

这是一条可重复的“修改前”基线，用于和 torch_npu 工作分支的修改后结果做
A/B 对比。

## 2. 最小模型

演示脚本：[npu_provenance_demo.py](../examples/npu_provenance_demo.py)

核心计算为：

```python
def forward(self, x):
    added = x + 1
    activated = torch.relu(added)
    return activated * 2
```

Inductor 会把 `add -> relu -> mul` 融合为一个 NPU Triton kernel。这正好适合
演示“一条 kernel 对应多个 FX 节点”的 provenance 关系。

## 3. 运行命令

所有测试从 `/home/z50063656/tmp` 启动，避免在 torch_npu 源码树内导入
`torch`。

```bash
cd /home/z50063656/tmp
source /home/z50063656/Tracking/activate_tracking.sh

RUN_DIR=$(mktemp -d /home/z50063656/tmp/tracking-provenance.XXXXXX)
export ASCEND_RT_VISIBLE_DEVICES=6
export INDUCTOR_PROVENANCE=1
export TORCH_COMPILE_DEBUG=1
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCH_TRACE="$RUN_DIR/trace"
export TORCH_COMPILE_DEBUG_DIR="$RUN_DIR/debug"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/cache"

python /home/z50063656/Tracking/npu_provenance_demo.py
```

设备编号 `6` 是本轮使用的空闲物理设备；重跑前应通过 `npu-smi info` 重新
选择空闲设备。

## 4. 本次实际输出

```text
torch=2.14.0a0+git8e86e0a
torch_npu=2.14.0
device=Ascend910B2
checksum=9206.284180
TORCH_TRACE=/home/z50063656/tmp/tracking-provenance-baseline.bpyBfR/trace
TORCH_COMPILE_DEBUG_DIR=/home/z50063656/tmp/tracking-provenance-baseline.bpyBfR/debug
```

`checksum` 成功输出说明 FX 捕获、Inductor lowering、Triton Ascend 编译、
launcher 构建和 NPU kernel 执行都已完成。

## 5. Provenance 产物

本轮结构化 trace 文件：

```text
/home/z50063656/tmp/tracking-provenance-baseline.bpyBfR/trace/
dedicated_log_torch_trace_cq5_tzob.log
```

其中的节点映射为：

```json
{
  "preToPost": {
    "added": ["add"],
    "activated": ["relu"],
    "mul": ["mul"]
  },
  "postToPre": {
    "add": ["added"],
    "relu": ["activated"],
    "mul": ["mul"]
  },
  "cppCodeToPost": {},
  "postToCppCode": {},
  "version": 2.0
}
```

四个字段的含义：

| 字段 | 回答的问题 | 本轮结果 |
| --- | --- | --- |
| `preToPost` | 用户/输入 FX 节点变成了哪些 post-grad 节点？ | 正常 |
| `postToPre` | post-grad 节点来自哪个输入 FX 节点？ | 正常 |
| `cppCodeToPost` | 最终 kernel 包含哪些 post-grad 节点？ | 空，待适配 |
| `postToCppCode` | post-grad 节点进入了哪个 kernel？ | 空，待适配 |

字段名 `cppCodeToPost` 是历史命名，NPU Triton、CATLASS、MLIR 和 DVM kernel
仍应使用该字段，不应另造 NPU 专属 schema。

## 6. 修改后的验收目标

普通 Triton 路径适配成功后，后两组映射应呈现类似结构；实际 kernel 名和
handle 数字由本轮编译决定：

```json
{
  "cppCodeToPost": {
    "triton_poi_fused_add_mul_relu_0:1": ["add", "relu", "mul"]
  },
  "postToCppCode": {
    "add": ["triton_poi_fused_add_mul_relu_0:1"],
    "relu": ["triton_poi_fused_add_mul_relu_0:1"],
    "mul": ["triton_poi_fused_add_mul_relu_0:1"]
  }
}
```

同时必须满足：

1. compiled 输出与 eager 输出一致。
2. kernel key 中包含唯一 debug handle。
3. kernel stack artifact 使用同一个 key，且包含模型 `forward` 来源。
4. `tlparse <log-file> --inductor-provenance` 能把 pre-grad、post-grad 和生成
   kernel 三栏关联起来。

第 1～3 项由 torch_npu 回归测试和 raw structured trace 验证；第 4 项需要
本机提供 Rust `tlparse` CLI 后执行。
