# NPU Inductor provenance cache miss/hit 验证

更新时间：2026-08-20

## 结论

NPU Inductor provenance 已通过跨进程 FX Graph cache miss/hit 验证。两个独立 Python
进程复用同一个 `TORCHINDUCTOR_CACHE_DIR`：第一次完成编译和写缓存，第二次明确记录
`fx_graph_cache_hit` 与 `aotautograd_cache_hit`。两次运行的数值 checksum 都是
`9206.284180`。

最关键的断言结果：

```text
mapping_cmp=0
stack_cmp=0
```

即 cache miss 和 cache hit trace 中的 provenance node mapping 与 kernel stack JSON
逐字节一致。cache hit 的三栏页面仍包含：

```json
{
  "pyCodeToPost": {"132": [10, 7, 4]},
  "postToPyCode": {
    "10": [132],
    "4": [132],
    "7": [132]
  }
}
```

因此，上游 FX Graph cache 的 artifact 保存和重放机制可以直接服务 NPU provenance，
无需为 NPU 另建 cache schema。

## 运行方法

所有测试从 `/home/z50063656/tmp` 启动。运行前用 `npu-smi info` 选择空闲设备。下面
的物理设备 5 只是本轮示例。

```bash
cd /home/z50063656/tmp
source /home/z50063656/Tracking/activate_tracking.sh

export ROOT=/home/z50063656/Tracking/npu_provenance_cache_hit_recheck
export ASCEND_RT_VISIBLE_DEVICES=5
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache"
export TORCH_COMPILE_DEBUG=1
export TORCHINDUCTOR_PROVENANCE_TRACKING_LEVEL=1
mkdir -p "$ROOT/cache" "$ROOT/miss" "$ROOT/hit"
```

第一次运行，制造 cache miss：

```bash
TORCH_COMPILE_DEBUG_DIR="$ROOT/miss/debug" \
TORCH_TRACE="$ROOT/miss/trace" \
python /home/z50063656/Tracking/npu_provenance_demo.py
```

第二次运行，复用完全相同的 cache：

```bash
TORCH_COMPILE_DEBUG_DIR="$ROOT/hit/debug" \
TORCH_TRACE="$ROOT/hit/trace" \
python /home/z50063656/Tracking/npu_provenance_demo.py
```

分别用 `tlparse --inductor-provenance --no-browser` 解析两份 trace，再比较其中的
mapping 和 stack JSON。

## 本轮证据

- [cache miss 运行日志](./npu_provenance_cache_hit_verified_20260820/miss/run.log)
- [cache hit 运行日志](./npu_provenance_cache_hit_verified_20260820/hit/run.log)
- [cache miss tlparse](./npu_provenance_cache_hit_verified_20260820/miss/tlparse/index.html)
- [cache hit tlparse](./npu_provenance_cache_hit_verified_20260820/hit/tlparse/index.html)
- [cache hit 三栏页面](./npu_provenance_cache_hit_verified_20260820/hit/tlparse/provenance_tracking_-_0_0_0.html)
- [cache hit 事件](./npu_provenance_cache_hit_verified_20260820/hit/tlparse/-_0_0_0/fx_graph_cache_hit_13.json)
- [cache hit mapping](./npu_provenance_cache_hit_verified_20260820/hit/tlparse/-_0_0_0/inductor_provenance_tracking_node_mappings_10.json)
- [cache hit stack](./npu_provenance_cache_hit_verified_20260820/hit/tlparse/-_0_0_0/inductor_provenance_tracking_kernel_stack_traces_11.json)
- [cache hit stack 可读页面](./npu_provenance_cache_hit_verified_20260820/hit/tlparse/-_0_0_0/inductor_provenance_tracking_kernel_stack_traces_11_readable.html)

本轮 tlparse 统计为：

```text
cache miss: Stats { ok: 144 }
cache hit:  Stats { ok: 63 }
```

cache hit 的事件更少是正常现象，因为它不再执行完整 Inductor codegen。

## 为什么 hit 的 debug 目录没有 output_code.py

cache miss 会执行 codegen，因此 `TORCH_COMPILE_DEBUG_DIR` 下能看到 `output_code.py`、
mapping 等直接 debug artifact。cache hit 直接加载缓存结果，不会重新走 debug formatter，
所以 hit 的普通 debug 目录只有 Dynamo/Inductor 日志。

这不等于 provenance 丢失。PyTorch 把缓存条目中保存的 output code、post-grad graph、
node mapping 和 kernel stack 重放进 `TORCH_TRACE`；tlparse 再从结构化 trace 还原文件和
三栏页面。cache hit 验收应检查 trace/tlparse，而不是错误地要求 hit debug 目录重新
生成一套文件。

## 当前覆盖边界

本轮验证的是 JIT Python wrapper 的 FX Graph cache。AOTInductor 打包后的
`kernel_information.json`、AOTI 加载，以及 provenance level 改变时 cache key/重编译
行为仍需单独验证。
