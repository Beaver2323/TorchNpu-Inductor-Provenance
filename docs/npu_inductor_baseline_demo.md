# PyTorch 2.14 NPU Inductor 基线演示

## 1. 演示目标

本演示用于确认当前 `Tracking` 隔离环境具备后续 TorchInductor provenance 适配所需的基础能力：

1. PyTorch、torch_npu、Triton Ascend 和源码提交相互匹配。
2. CPU eager 和 NPU eager 可以正确执行。
3. `torch.compile(..., backend="inductor")` 可以分别在 CPU 和 Ascend NPU 上完成编译、执行和数值校验。

本基线只证明 NPU Inductor 环境可用，不代表 provenance kernel 映射已经适配完成。

## 2. 环境基线

| 组件 | 版本或提交 |
| --- | --- |
| Python | 3.11.15 |
| PyTorch | `2.14.0a0+git8e86e0a` |
| PyTorch commit | `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b` |
| torch_npu | `2.14.0` |
| torch_npu commit | `83cc452480c3546fd5cccf853bfe3a360ce9dbfc` |
| triton-ascend | `3.2.2` |
| CANN | 9.0.1 |
| NPU | Ascend 910B2 |
| Conda 环境 | `/home/z50063656/envs/Tracking` |

源码目录：

- PyTorch：`/home/z50063656/Tracking/src/pytorch`
- torch_npu：`/home/z50063656/Tracking/src/torch_npu`
- Triton Ascend：`/home/z50063656/Tracking/src/triton-ascend`

## 3. 一键验证

所有测试必须从 `/home/z50063656/tmp` 启动，不能在 torch_npu 源码目录内导入 `torch`。

```bash
cd /home/z50063656/tmp
source /home/z50063656/Tracking/activate_tracking.sh

export ASCEND_RT_VISIBLE_DEVICES=6
export TORCH_COMPILE_DEBUG=1

python /home/z50063656/.codex/skills/zyf-env-2-14/scripts/verify_env.py
```

设备编号 `6` 只是本次验证所用的空闲物理设备。重新执行前应先用 `npu-smi info` 选择当时空闲的设备。

## 4. 验证程序的核心逻辑

CPU 和 NPU 使用相同计算：

```python
import torch
import torch_npu


def run(device):
    fn = torch.compile(
        lambda x: x.sin() + x.cos(),
        backend="inductor",
    )
    return fn(torch.ones(16, device=device)).cpu()


print(run("cpu")[:2])
print(run("npu")[:2])
```

输入为全 1 Tensor，因此每个输出元素应接近：

```text
sin(1) + cos(1) = 1.3817732338905338
```

正式验证脚本使用 `rtol=1e-4, atol=1e-4` 做数值检查，不只检查程序是否退出。

## 5. 本次实际输出

2026-08-20 的实机运行结果：

```text
python=3.11.15
torch=2.14.0a0+git8e86e0a
torch_git=8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b
torch_npu=2.14.0
torch_npu_git=83cc452480c3546fd5cccf853bfe3a360ce9dbfc
triton_ascend=3.2.2
pytorch_head=8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b
torch_npu_head=83cc452480c3546fd5cccf853bfe3a360ce9dbfc
inductor_cpu=tensor([1.3818, 1.3818])
inductor_npu=tensor([1.3818, 1.3818])
status=pass
```

结果说明：

- 安装包记录的 Git commit 与源码 checkout 完全一致。
- CPU Inductor 编译和数值检查通过。
- NPU eager 初始化和执行通过。
- NPU Triton/Inductor 完成首次编译、launcher 构建、设备执行和结果回传。
- 旧混合环境中的 `ATen/ATen.h` launcher 错误在该隔离环境中不再出现。

## 6. 调试产物

本次启用了 `TORCH_COMPILE_DEBUG=1`，调试目录为：

```text
/home/z50063656/tmp/tracking-baseline.BGw54a/debug/torch_compile_debug/
```

由于基线测试通过，不需要用调试产物诊断失败。后续 provenance 专项验证仍应设置独立的 `TORCH_TRACE`、`TORCH_COMPILE_DEBUG_DIR` 和 `TORCHINDUCTOR_CACHE_DIR`，避免复用本次 cache。

## 7. 与 provenance 任务的关系

环境基线通过后，下一步才是验证：

```text
用户 FX 节点
  -> post-grad FX 节点
  -> NPU Triton/CATLASS/MLIR/DVM kernel
  -> provenance debug handle
  -> structured trace / tlparse 高亮
```

当前 torch_npu 源码仍有多个 `codegen_comment(schedule)` 未传 `kernel_name` 的调用点。因此预期现状是模型可以在 NPU Inductor 上正确执行，但 kernel-level provenance mapping 不完整。该差异将由下一阶段专项 baseline 和代码适配验证。

专项的修改前 provenance 命令、真实 JSON 和修改后验收目标见
[npu_provenance_baseline_demo.md](./npu_provenance_baseline_demo.md)。
