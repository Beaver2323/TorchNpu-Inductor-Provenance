# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
from torch_npu._inductor.triton_experimental import get_current_raw_stream as get_raw_stream
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import get_current_raw_stream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
import torch_npu
empty_strided_npu = torch_npu._C._empty_strided_npu


# kernel path: /tmp/torchinductor_root/tmpy65ym09i/u7/cu75zioe7bx7y3w5n7fiqhzmvfwcplaahhqwrvdosqncr72zkwrc.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.pointwise(
    size_hints={'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 3, 'min_blocks': 0, 'autotune_grouping': True, 'block_arg_names': ('XBLOCK',), 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': 100, 'no_x_dim_1': False, 'xnumel_1': 400, 'no_x_dim_2': False, 'xnumel_2': 100}, 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': True, 'backend_hash': 'D0A5764714F6964B7DDC8949E7DA556771FD1CABBAA741D79495E9BB4C7A1FB0', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(100, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(400, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(100, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        xnumel = 100
        r0_numel = 1
        tmp0 = tl.load(in_ptr0 + (x0), x0mask)
        tmp1 = tl.full([1], 0, tl.int32)
        tmp2 = tl.maximum(tmp1, tmp0)
        tl.store(out_ptr0 + (x0), tmp2, x0mask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        xnumel = 400
        r0_numel = 1
        tmp3 = tl.load(in_ptr1 + (x1), x1mask)
        tmp4 = tl.sigmoid(tmp3)
        tl.store(out_ptr1 + (x1), tmp4, x1mask)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        xnumel = 100
        r0_numel = 1
        tmp5 = tl.load(in_ptr2 + (x2), x2mask)
        tmp6 = libdevice.tanh(tmp5)
        tl.store(out_ptr2 + (x2), tmp6, x2mask)
    else:
        pass
''', device_str='npu')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            arg2_1 = copy_if_misaligned(arg2_1)
            buf0 = empty_strided_npu((10, 10), (10, 1), torch.float32)
            buf1 = empty_strided_npu((20, 20), (20, 1), torch.float32)
            buf2 = empty_strided_npu((10, 10), (10, 1), torch.float32)
            # Topologically Sorted Source Nodes: [relu, sigmoid, tanh], Original ATen: [aten.relu, aten.sigmoid, aten.tanh]
            # [Provenance debug handles] triton_poi_fused_0:1
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_0.run(arg0_1, arg1_1, arg2_1, buf0, buf1, buf2, stream=raw_stream0)
            del arg0_1
            del arg1_1
            del arg2_1
        return (buf0, buf1, buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((10, 10), (10, 1), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((20, 20), (20, 1), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((10, 10), (10, 1), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
