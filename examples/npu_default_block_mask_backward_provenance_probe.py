#!/usr/bin/env python3
"""Verify default FlexAttention BlockMask backward and provenance on NPU."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu
from torch._inductor import config
from torch._inductor.debug import (
    reset_inductor_kernel_provenance_debug_handle,
)
from torch.nn.attention.flex_attention import flex_attention
from torch_npu._inductor import config as npu_config


class DefaultBlockMaskFlexAttention(torch.nn.Module):
    def forward(self, query, key, value):
        return flex_attention(query, key, value, block_mask=None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="A new directory for debug artifacts and result.json.",
    )
    parser.add_argument(
        "--disable-backward-hfusion",
        action="store_true",
        help=(
            "Diagnostic only: disable the dK/dV multiple-consumer fusion "
            "compiler option."
        ),
    )
    return parser.parse_args()


def _make_leaf(tensor):
    return tensor.detach().clone().requires_grad_(True)


def _read_flex_kernel_mappings(debug_dir):
    mapping_paths = sorted(
        debug_dir.rglob("inductor_provenance_tracking_node_mappings.json")
    )
    mappings = []
    for mapping_path in mapping_paths:
        mapping = json.loads(mapping_path.read_text())
        kernel_to_post = mapping["cppCodeToPost"]
        flex_kernels = {
            kernel: post_nodes
            for kernel, post_nodes in kernel_to_post.items()
            if "flex_attention" in kernel
        }
        mappings.append(
            {
                "mapping_path": str(mapping_path),
                "flex_kernels": flex_kernels,
            }
        )
    return mappings


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    debug_dir = output_dir / "debug"

    if args.disable_backward_hfusion:
        npu_config.flex_attention.hfusion_enable_multiple_consumer_fusion = (
            False
        )

    torch.manual_seed(0)
    shape = (1, 1, 128, 64)
    query_base = torch.randn(shape, device="npu", dtype=torch.float32)
    key_base = torch.randn(shape, device="npu", dtype=torch.float32)
    value_base = torch.randn(shape, device="npu", dtype=torch.float32)
    grad_output = torch.randn(shape, device="npu", dtype=torch.float32)

    expected_inputs = tuple(
        _make_leaf(tensor) for tensor in (query_base, key_base, value_base)
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        *expected_inputs
    )
    expected.backward(grad_output)
    expected_grads = tuple(tensor.grad.detach().clone() for tensor in expected_inputs)

    actual_inputs = tuple(
        _make_leaf(tensor) for tensor in (query_base, key_base, value_base)
    )
    model = DefaultBlockMaskFlexAttention().npu().train()

    reset_inductor_kernel_provenance_debug_handle()
    torch._dynamo.reset()
    with config.patch(
        {
            "trace.enabled": True,
            "trace.debug_dir": str(debug_dir),
            "trace.provenance_tracking_level": 1,
            "force_disable_caches": True,
        }
    ):
        actual = torch.compile(
            model,
            backend="inductor",
            fullgraph=True,
        )(*actual_inputs)
        actual.backward(grad_output)
    torch.npu.synchronize()

    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
    for actual_input, expected_grad in zip(actual_inputs, expected_grads):
        torch.testing.assert_close(
            actual_input.grad,
            expected_grad,
            rtol=5e-2,
            atol=5e-2,
        )

    mappings = _read_flex_kernel_mappings(debug_dir)
    if len(mappings) < 2:
        raise AssertionError(
            "expected separate forward and backward provenance mappings, "
            f"found {len(mappings)}"
        )
    all_flex_kernels = {
        kernel
        for mapping in mappings
        for kernel in mapping["flex_kernels"]
    }
    forward_kernels = sorted(
        kernel for kernel in all_flex_kernels if "fwd" in kernel
    )
    backward_kernels = sorted(
        kernel
        for kernel in all_flex_kernels
        if "bwd" in kernel or "backward" in kernel
    )
    if not forward_kernels:
        raise AssertionError(f"no FlexAttention forward mapping: {mappings}")
    if not backward_kernels:
        raise AssertionError(f"no FlexAttention backward mapping: {mappings}")

    result = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": torch.npu.get_device_name(0),
        "shape": list(shape),
        "block_mask": None,
        "backward_hfusion": not args.disable_backward_hfusion,
        "output_checksum": float(actual.float().sum()),
        "grad_checksums": {
            name: float(tensor.grad.float().sum())
            for name, tensor in zip(("query", "key", "value"), actual_inputs)
        },
        "forward_kernels": forward_kernels,
        "backward_kernels": backward_kernels,
        "mappings": mappings,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
