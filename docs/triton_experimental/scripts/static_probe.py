#!/usr/bin/env python3
"""Validate static provenance for the in-tree triton_experimental backend."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu
from torch._inductor import config
from torch._inductor.debug import (
    reset_inductor_kernel_provenance_debug_handle,
)
from torch._dynamo.utils import counters


class ProvenanceModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        added = x + 1
        activated = torch.relu(added)
        return activated * 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--level",
        type=int,
        choices=(1, 2),
        default=1,
        help="Inductor provenance tracking level",
    )
    parser.add_argument(
        "--expect-mapped",
        action="store_true",
        help="require one triton kernel to map to add/relu/mul",
    )
    parser.add_argument(
        "--expect-cache-hit",
        action="store_true",
        help="require an Inductor FX graph cache hit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    debug_dir = output_dir / "compile_debug"

    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available")

    torch.manual_seed(20260827)
    model = ProvenanceModel().npu().eval()
    x = torch.randn(4096, device="npu")
    expected = model(x)

    reset_inductor_kernel_provenance_debug_handle()
    torch._dynamo.reset()
    counters["inductor"].clear()
    with config.patch(
        {
            "trace.enabled": True,
            "trace.debug_dir": str(debug_dir),
            "trace.provenance_tracking_level": args.level,
            "triton.unique_kernel_names": True,
        }
    ):
        compiled = torch.compile(
            model,
            backend="inductor",
            options={"npu_backend": "triton_experimental"},
            fullgraph=True,
        )
        actual = compiled(x)
        torch.npu.synchronize()

    torch.testing.assert_close(actual, expected)
    mapping_paths = sorted(
        debug_dir.rglob("inductor_provenance_tracking_node_mappings.json")
    )
    stack_paths = sorted(
        debug_dir.rglob(
            "inductor_provenance_tracking_kernel_stack_traces.json"
        )
    )
    output_code_paths = sorted(debug_dir.rglob("output_code.py"))
    if len(mapping_paths) != 1:
        raise AssertionError(
            f"expected one mapping artifact, got {mapping_paths}"
        )
    if len(stack_paths) > 1:
        raise AssertionError(
            f"expected at most one stack artifact, got {stack_paths}"
        )
    if len(output_code_paths) != 1:
        raise AssertionError(
            f"expected one output_code.py, got {output_code_paths}"
        )

    mapping = json.loads(mapping_paths[0].read_text())
    stacks = json.loads(stack_paths[0].read_text()) if stack_paths else {}
    output_code = output_code_paths[0].read_text()
    kernel_to_post = mapping.get("cppCodeToPost", {})
    kernel_keys = sorted(kernel_to_post)
    result = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "torch_npu_module": str(Path(torch_npu.__file__).resolve()),
        "device": torch.npu.get_device_name(0),
        "provenance_level": args.level,
        "max_abs_diff": float((actual - expected).abs().max().cpu()),
        "experimental_marker": (
            "triton_experimental import npu_triton_heuristics" in output_code
        ),
        "mapping_path": str(mapping_paths[0]),
        "stack_path": str(stack_paths[0]) if stack_paths else None,
        "output_code_path": str(output_code_paths[0]),
        "kernel_keys": kernel_keys,
        "kernel_to_post": kernel_to_post,
        "post_to_kernel": mapping.get("postToCppCode", {}),
        "stack_keys": sorted(stacks),
        "fxgraph_cache_hit": counters["inductor"]["fxgraph_cache_hit"],
        "inductor_counters": dict(counters["inductor"]),
    }

    if not result["experimental_marker"]:
        raise AssertionError("generated code did not use triton_experimental")
    if args.expect_mapped:
        if len(kernel_keys) != 1:
            raise AssertionError(
                f"expected one mapped kernel, got {kernel_keys}"
            )
        post_nodes = set(kernel_to_post[kernel_keys[0]])
        if post_nodes != {"add", "relu", "mul"}:
            raise AssertionError(f"unexpected post-grad nodes: {post_nodes}")
        if (
            not kernel_keys[0].startswith("triton_")
            or ":" not in kernel_keys[0]
        ):
            raise AssertionError(f"unexpected kernel key: {kernel_keys[0]}")
        if f"[Provenance debug handles] {kernel_keys[0]}" not in output_code:
            raise AssertionError(
                f"wrapper debug handle is missing: {kernel_keys[0]}"
            )
    if args.expect_cache_hit and result["fxgraph_cache_hit"] < 1:
        raise AssertionError(f"expected an FX graph cache hit: {result}")

    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
