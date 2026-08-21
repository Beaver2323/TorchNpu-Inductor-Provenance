#!/usr/bin/env python3
"""Verify default FlexAttention BlockMask lowering and provenance on NPU."""

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
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    debug_dir = output_dir / "debug"

    torch.manual_seed(0)
    shape = (1, 1, 128, 64)
    model = DefaultBlockMaskFlexAttention().npu().eval()
    query = torch.randn(shape, device="npu", dtype=torch.float32)
    key = torch.randn(shape, device="npu", dtype=torch.float32)
    value = torch.randn(shape, device="npu", dtype=torch.float32)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key, value
    )

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
        )(query, key, value)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)

    mapping_paths = list(
        debug_dir.rglob("inductor_provenance_tracking_node_mappings.json")
    )
    if len(mapping_paths) != 1:
        raise AssertionError(
            f"expected one provenance mapping, found {len(mapping_paths)}"
        )
    mapping_path = mapping_paths[0]
    mapping = json.loads(mapping_path.read_text())
    kernel_to_post = mapping["cppCodeToPost"]
    template_kernels = {
        kernel: post_nodes
        for kernel, post_nodes in kernel_to_post.items()
        if "flex_attention" in kernel
    }
    if not template_kernels:
        raise AssertionError(
            "no FlexAttention template kernel found in cppCodeToPost: "
            f"{kernel_to_post}"
        )

    result = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": torch.npu.get_device_name(0),
        "shape": list(shape),
        "block_mask": None,
        "checksum": float(actual.float().sum()),
        "mapping_path": str(mapping_path),
        "template_kernels": template_kernels,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
