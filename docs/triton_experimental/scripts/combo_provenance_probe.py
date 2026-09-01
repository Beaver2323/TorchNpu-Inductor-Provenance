#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import torch_npu
from torch._dynamo.utils import counters
from torch._inductor import config, metrics
from torch._inductor.debug import reset_inductor_kernel_provenance_debug_handle


class ComboModel(torch.nn.Module):
    def forward(self, a, b, c):
        return torch.relu(a), torch.sigmoid(b), torch.tanh(c)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--level", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    debug_dir = output_dir / "compile_debug"

    if not torch.npu.is_available():
        raise RuntimeError("NPU is unavailable")

    torch.manual_seed(20260901)
    model = ComboModel().npu().eval()
    inputs = (
        torch.randn(10, 10, device="npu"),
        torch.randn(20, 20, device="npu"),
        torch.randn(10, 10, device="npu"),
    )
    expected = model(*inputs)

    reset_inductor_kernel_provenance_debug_handle()
    torch._dynamo.reset()
    counters["inductor"].clear()
    metrics.reset()
    compile_error = None
    actual = None
    try:
        with config.patch(
            {
                "trace.enabled": True,
                "trace.debug_dir": str(debug_dir),
                "trace.provenance_tracking_level": args.level,
                "force_disable_caches": True,
                "combo_kernels": True,
                "benchmark_combo_kernel": False,
                "triton.unique_kernel_names": True,
            }
        ):
            compiled = torch.compile(
                model,
                backend="inductor",
                options={"npu_backend": "triton_experimental"},
                fullgraph=True,
            )
            actual = compiled(*inputs)
            torch.npu.synchronize()
    except Exception as error:
        compile_error = error

    max_abs_diff = None
    if actual is not None:
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor)
        max_abs_diff = max(
            float((actual_tensor - expected_tensor).abs().max().cpu())
            for actual_tensor, expected_tensor in zip(actual, expected, strict=True)
        )

    mapping_paths = sorted(
        debug_dir.rglob("inductor_provenance_tracking_node_mappings.json")
    )
    output_paths = sorted(debug_dir.rglob("output_code.py"))
    stack_paths = sorted(
        debug_dir.rglob("inductor_provenance_tracking_kernel_stack_traces.json")
    )
    if len(output_paths) != 1:
        raise AssertionError(
            f"unexpected artifacts: mappings={mapping_paths}, output={output_paths}"
        )

    mapping = json.loads(mapping_paths[0].read_text()) if mapping_paths else {}
    output_code = output_paths[0].read_text()
    kernel_to_post = mapping.get("cppCodeToPost", {})
    combo_markers = {
        "foreach_decorator": "@triton_heuristics.foreach(" in output_code,
        "sequential_grid": "SequentialComboKernelGrid" in output_code,
        "round_robin_grid": "RoundRobinComboKernelGrid" in output_code,
        "pid_dispatch": "num_xblocks_0" in output_code,
    }
    mapped_sets = [set(nodes) for nodes in kernel_to_post.values()]
    expected_nodes = {"relu", "sigmoid", "tanh"}
    if compile_error is None:
        mapping_pass = args.level == 0 or (
            len(kernel_to_post) == 1 and mapped_sets == [expected_nodes]
        )
        status = (
            "PASS"
            if any(combo_markers.values())
            and max_abs_diff == 0.0
            and mapping_pass
            else "FAIL"
        )
    elif any(combo_markers.values()) and "x0 is not defined" in str(compile_error):
        status = "UNSUPPORTED"
    else:
        status = "FAIL"
    result = {
        "status": status,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "torch_npu_module": str(Path(torch_npu.__file__).resolve()),
        "device": torch.npu.get_device_name(),
        "backend": "triton_experimental",
        "combo_kernels": True,
        "provenance_level": args.level,
        "combo_markers": combo_markers,
        "max_abs_diff": max_abs_diff,
        "generated_kernel_count": metrics.generated_kernel_count,
        "kernel_to_post": kernel_to_post,
        "mapping_path": str(mapping_paths[0]) if mapping_paths else None,
        "output_code_path": str(output_paths[0]),
        "stack_path": str(stack_paths[0]) if stack_paths else None,
        "compile_error_type": type(compile_error).__name__ if compile_error else None,
        "compile_error_marker": (
            "NameError: x0 is not defined"
            if compile_error and "x0 is not defined" in str(compile_error)
            else None
        ),
        "inductor_counters": dict(counters["inductor"]),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if result["status"] == "FAIL":
        raise AssertionError("ComboKernel gate produced an unexpected result")


if __name__ == "__main__":
    main()
