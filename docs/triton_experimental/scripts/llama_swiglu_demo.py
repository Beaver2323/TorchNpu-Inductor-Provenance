#!/usr/bin/env python3
"""Generate a persistent Llama-style provenance demo on Ascend NPU."""

import argparse
import json
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu
from torch._inductor import config
from torch._inductor.debug import (
    get_kernel_information_jsons,
    reset_inductor_kernel_provenance_debug_handle,
)
from torch_npu.profiler import inductor_trace_handler


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps)
        return normalized * self.weight


class LlamaSwiGLUBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm(256)
        self.gate_proj = torch.nn.Linear(256, 512)
        self.up_proj = torch.nn.Linear(256, 512)
        self.down_proj = torch.nn.Linear(512, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        normalized = self.norm(x)
        gated = F.silu(self.gate_proj(normalized)) * self.up_proj(normalized)
        return self.down_proj(gated) + residual


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _parameter_grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _max_abs_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.detach() - expected.detach()).abs().max().cpu())


def _parameter_grad_max_abs_diff(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> float:
    return max(_max_abs_diff(actual[name], expected[name]) for name in expected)


def _trace_events(trace: object) -> list[dict[str, object]]:
    if isinstance(trace, list):
        return [event for event in trace if isinstance(event, dict)]
    if isinstance(trace, dict):
        events = trace.get("traceEvents", [])
        if isinstance(events, list):
            return [event for event in events if isinstance(event, dict)]
    raise TypeError(f"unsupported trace root: {type(trace).__name__}")


def _is_generated_kernel_name(name: str) -> bool:
    return name.startswith(("triton_", "k_")) or (
        len(name) == 49 and ("fused" in name or "softmax" in name)
    )


def _summarize_trace(path: Path) -> dict[str, object]:
    trace = json.loads(path.read_text())
    events = _trace_events(trace)
    flows = [event for event in events if event.get("ph") in ("s", "f")]
    device_kernels = [
        event
        for event in events
        if _is_generated_kernel_name(str(event.get("name", "")))
        and str((event.get("args") or {}).get("Task Type", "")).startswith(
            "KERNEL_"
        )
    ]
    cpu_launch_names = [
        str(event["name"])
        for event in events
        if event.get("cat") == "cpu_op"
        and _is_generated_kernel_name(str(event.get("name", "")))
    ]
    device_names = [str(event["name"]) for event in device_kernels]
    stack_names = [
        str(event["name"])
        for event in device_kernels
        if (event.get("args") or {}).get("stack")
    ]
    stack_text = str(
        [(event.get("args") or {}).get("stack") for event in device_kernels]
    )
    required_source_lines = [
        "variance = x.pow(2).mean(dim=-1, keepdim=True)",
        "gated = F.silu(self.gate_proj(normalized))",
        "return self.down_proj(gated) + residual",
    ]
    summary = {
        "trace_root": type(trace).__name__,
        "event_count": len(events),
        "device_kernel_names": device_names,
        "device_kernel_stack_names": stack_names,
        "device_kernel_stacks": [
            {
                "name": str(event["name"]),
                "stack": (event.get("args") or {}).get("stack"),
            }
            for event in device_kernels
        ],
        "cpu_launch_names": cpu_launch_names,
        "required_source_lines": required_source_lines,
        "required_source_lines_found": {
            line: line in stack_text for line in required_source_lines
        },
        "compiled_forward_region": any(
            event.get("name") == "CompiledFunction" for event in events
        ),
        "compiled_backward_region": any(
            event.get("name") == "CompiledFunctionBackward" for event in events
        ),
        "torch_to_npu_flow_count": sum(
            event.get("name") == "torch_to_npu" for event in flows
        ),
        "ac2g_flow_count": sum(
            event.get("name") == "ac2g" for event in flows
        ),
        "uid_event_count": sum("uid" in event for event in events),
        "temporary_kernel_category_count": sum(
            event.get("cat") == "kernel" for event in events
        ),
    }
    if len(device_names) < 6:
        raise AssertionError(f"expected at least six generated kernels: {summary}")
    if stack_names != device_names:
        raise AssertionError(f"device kernel stacks are incomplete: {summary}")
    if cpu_launch_names != device_names:
        raise AssertionError(f"CPU/device kernel names differ: {summary}")
    if not all(summary["required_source_lines_found"].values()):
        raise AssertionError(f"module source lines are missing: {summary}")
    if not summary["compiled_forward_region"]:
        raise AssertionError(f"compiled forward region is missing: {summary}")
    if not summary["compiled_backward_region"]:
        raise AssertionError(f"compiled backward region is missing: {summary}")
    if summary["torch_to_npu_flow_count"] == 0:
        raise AssertionError(f"torch_to_npu flows are missing: {summary}")
    if (
        summary["ac2g_flow_count"] != 0
        or summary["uid_event_count"] != 0
        or summary["temporary_kernel_category_count"] != 0
    ):
        raise AssertionError(f"temporary normalized schema leaked: {summary}")
    return summary


def _read_graph_artifacts(
    paths: list[Path], debug_dir: Path
) -> list[dict[str, object]]:
    return [
        {
            "source": str(path.relative_to(debug_dir)),
            "data": json.loads(path.read_text()),
        }
        for path in paths
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available")

    torch.manual_seed(20260831)
    model = LlamaSwiGLUBlock().npu().train()
    base = torch.randn(2, 32, 256, device="npu")
    alternate = torch.randn(3, 24, 256, device="npu")
    grad_base = torch.randn_like(base)
    grad_alternate = torch.randn_like(alternate)

    expected_input = base.detach().clone().requires_grad_(True)
    expected = model(expected_input)
    expected.backward(grad_base)
    expected_input_grad = expected_input.grad.detach().clone()
    expected_parameter_grads = _parameter_grads(model)
    model.zero_grad(set_to_none=True)

    reset_inductor_kernel_provenance_debug_handle()
    torch._dynamo.reset()
    with tempfile.TemporaryDirectory() as temporary_debug_dir:
        debug_dir = Path(temporary_debug_dir)
        with config.patch(
            {
                "trace.enabled": True,
                "trace.debug_dir": str(debug_dir),
                "trace.provenance_tracking_level": 1,
                "trace.provenance_tracking_to_timeline": True,
                "triton.unique_kernel_names": True,
                "force_disable_caches": True,
            }
        ):
            compiled = torch.compile(
                model,
                backend="inductor",
                options={"npu_backend": "triton_experimental"},
                fullgraph=True,
                dynamic=True,
            )

            actual_input = base.detach().clone().requires_grad_(True)
            actual = compiled(actual_input)
            actual.backward(grad_base)
            actual_input_grad = actual_input.grad.detach().clone()
            actual_parameter_grads = _parameter_grads(model)
            model.zero_grad(set_to_none=True)

            expected_alt_input = alternate.detach().clone().requires_grad_(True)
            expected_alternate = model(expected_alt_input)
            expected_alternate.backward(grad_alternate)
            expected_alt_input_grad = expected_alt_input.grad.detach().clone()
            expected_alt_parameter_grads = _parameter_grads(model)
            model.zero_grad(set_to_none=True)

            actual_alt_input = alternate.detach().clone().requires_grad_(True)
            actual_alternate = compiled(actual_alt_input)
            actual_alternate.backward(grad_alternate)
            actual_alt_input_grad = actual_alt_input.grad.detach().clone()
            actual_alt_parameter_grads = _parameter_grads(model)
            model.zero_grad(set_to_none=True)
            torch.npu.synchronize()

            handler = inductor_trace_handler(
                str(output_dir), worker_name="llama_swiglu"
            )
            prof_input = base.detach().clone().requires_grad_(True)
            with torch_npu.profiler.profile(on_trace_ready=handler):
                prof_output = compiled(prof_input)
                prof_output.backward(grad_base)
                torch.npu.synchronize()
            prof_input_grad = prof_input.grad.detach().clone()
            prof_parameter_grads = _parameter_grads(model)

        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(
            actual_alternate, expected_alternate, rtol=5e-3, atol=5e-3
        )
        torch.testing.assert_close(
            prof_output, expected, rtol=5e-3, atol=5e-3
        )
        torch.testing.assert_close(
            actual_input_grad, expected_input_grad, rtol=5e-3, atol=5e-3
        )
        torch.testing.assert_close(
            actual_alt_input_grad,
            expected_alt_input_grad,
            rtol=5e-3,
            atol=5e-3,
        )
        torch.testing.assert_close(
            prof_input_grad, expected_input_grad, rtol=5e-3, atol=5e-3
        )
        for name in expected_parameter_grads:
            torch.testing.assert_close(
                actual_parameter_grads[name],
                expected_parameter_grads[name],
                rtol=5e-3,
                atol=5e-3,
            )
            torch.testing.assert_close(
                prof_parameter_grads[name],
                expected_parameter_grads[name],
                rtol=5e-3,
                atol=5e-3,
            )
            torch.testing.assert_close(
                actual_alt_parameter_grads[name],
                expected_alt_parameter_grads[name],
                rtol=5e-3,
                atol=5e-3,
            )

        mapping_paths = sorted(
            debug_dir.rglob("inductor_provenance_tracking_node_mappings.json")
        )
        stack_paths = sorted(
            debug_dir.rglob(
                "inductor_provenance_tracking_kernel_stack_traces.json"
            )
        )
        if len(mapping_paths) != 2 or len(stack_paths) not in (0, 2):
            raise AssertionError(
                "expected forward/backward mapping and stack artifacts, got "
                f"{len(mapping_paths)} mappings and {len(stack_paths)} stacks"
            )
        mappings = _read_graph_artifacts(mapping_paths, debug_dir)
        stacks = _read_graph_artifacts(stack_paths, debug_dir)

    exported = sorted(output_dir.glob("*.pt.trace.json"))
    if len(exported) != 1:
        raise AssertionError(f"expected one profiler trace, got {exported}")
    trace_path = output_dir / "llama_swiglu_timeline_trace.json"
    exported[0].replace(trace_path)
    trace_summary = _summarize_trace(trace_path)

    mapping_document = {
        "schema_version": 1,
        "model": "Llama-style RMSNorm + SwiGLU residual block",
        "graphs": mappings,
    }
    stack_document = {
        "schema_version": 1,
        "model": "Llama-style RMSNorm + SwiGLU residual block",
        "compile_graphs": stacks,
        "timeline_kernel_stacks": trace_summary["device_kernel_stacks"],
    }
    _write_json(output_dir / "llama_swiglu_node_mappings.json", mapping_document)
    _write_json(output_dir / "llama_swiglu_kernel_stacks.json", stack_document)

    result = {
        "schema_version": 1,
        "date": "2026-08-29",
        "model": "Llama-style RMSNorm + SwiGLU residual block",
        "backend": "triton_experimental",
        "device": torch.npu.get_device_name(0),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "provenance_tracking_level": 1,
        "dynamic": True,
        "base_shape": list(base.shape),
        "alternate_shape": list(alternate.shape),
        "mapping_graph_count": len(mappings),
        "validation": {
            "base_forward_max_abs_diff": _max_abs_diff(actual, expected),
            "base_input_grad_max_abs_diff": _max_abs_diff(
                actual_input_grad, expected_input_grad
            ),
            "base_parameter_grad_max_abs_diff": _parameter_grad_max_abs_diff(
                actual_parameter_grads, expected_parameter_grads
            ),
            "alternate_forward_max_abs_diff": _max_abs_diff(
                actual_alternate, expected_alternate
            ),
            "alternate_input_grad_max_abs_diff": _max_abs_diff(
                actual_alt_input_grad, expected_alt_input_grad
            ),
            "alternate_parameter_grad_max_abs_diff": (
                _parameter_grad_max_abs_diff(
                    actual_alt_parameter_grads, expected_alt_parameter_grads
                )
            ),
            "profiler_forward_max_abs_diff": _max_abs_diff(
                prof_output, expected
            ),
            "profiler_input_grad_max_abs_diff": _max_abs_diff(
                prof_input_grad, expected_input_grad
            ),
            "profiler_parameter_grad_max_abs_diff": (
                _parameter_grad_max_abs_diff(
                    prof_parameter_grads, expected_parameter_grads
                )
            ),
        },
        "timeline": trace_summary,
        "artifacts": {
            "node_mappings": "llama_swiglu_node_mappings.json",
            "kernel_stacks": "llama_swiglu_kernel_stacks.json",
            "timeline_trace": "llama_swiglu_timeline_trace.json",
        },
        "status": "PASS",
    }
    if get_kernel_information_jsons():
        raise AssertionError("profiler compile information was not cleared")
    _write_json(output_dir / "llama_swiglu_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
