#!/usr/bin/env python3
"""Capture a triton_experimental forward/backward NPU profiler timeline."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch_npu
from torch._inductor import config
from torch._inductor.debug import reset_inductor_kernel_provenance_debug_handle
from torch_npu.profiler import inductor_trace_handler


class TimelineDemo(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = torch.relu(torch.sin(x))
        return activated * x


def trace_events(trace: object) -> list[dict[str, object]]:
    if isinstance(trace, list):
        return [event for event in trace if isinstance(event, dict)]
    if isinstance(trace, dict):
        events = trace.get("traceEvents", [])
        if isinstance(events, list):
            return [event for event in events if isinstance(event, dict)]
    raise TypeError(f"unsupported trace root: {type(trace).__name__}")


def summarize_trace(path: Path) -> dict[str, object]:
    trace = json.loads(path.read_text())
    events = trace_events(trace)
    flows = [event for event in events if event.get("ph") in ("s", "f")]
    device_triton = [
        event
        for event in events
        if str(event.get("name", "")).startswith(("triton_", "k_"))
        and str((event.get("args") or {}).get("Task Type", "")).startswith(
            "KERNEL_"
        )
    ]
    stack_events = [
        event for event in events if (event.get("args") or {}).get("stack")
    ]
    cpu_triton = [
        event
        for event in events
        if event.get("cat") == "cpu_op"
        and str(event.get("name", "")).startswith(("triton_", "k_"))
    ]
    compiled_forward = [
        event for event in events if event.get("name") == "CompiledFunction"
    ]
    compiled_backward = [
        event
        for event in events
        if event.get("name") == "CompiledFunctionBackward"
    ]

    def contained(event: dict[str, object], region: dict[str, object]) -> bool:
        if event.get("tid") != region.get("tid"):
            return False
        event_start = float(event["ts"])
        event_end = event_start + float(event.get("dur", 0))
        region_start = float(region["ts"])
        region_end = region_start + float(region.get("dur", 0))
        return region_start <= event_start and event_end <= region_end

    forward_launches = [
        event["name"]
        for event in cpu_triton
        if any(contained(event, region) for region in compiled_forward)
    ]
    backward_launches = [
        event["name"]
        for event in cpu_triton
        if any(contained(event, region) for region in compiled_backward)
    ]
    return {
        "trace_root": type(trace).__name__,
        "event_count": len(events),
        "flow_names": Counter(
            str(event.get("name", "")) for event in flows
        ).most_common(),
        "device_triton_names": [event["name"] for event in device_triton],
        "device_triton_stack_names": [
            event["name"]
            for event in device_triton
            if (event.get("args") or {}).get("stack")
        ],
        "device_triton_stacks": {
            event["name"]: (event.get("args") or {}).get("stack")
            for event in device_triton
        },
        "forward_cpu_triton_launches": forward_launches,
        "backward_cpu_triton_launches": backward_launches,
        "stack_event_count": len(stack_events),
        "torch_to_npu_flow_count": sum(
            event.get("name") == "torch_to_npu" for event in flows
        ),
        "ac2g_flow_count": sum(event.get("name") == "ac2g" for event in flows),
        "uid_event_count": sum("uid" in event for event in events),
        "temporary_kernel_category_count": sum(
            event.get("cat") == "kernel" for event in events
        ),
    }


def validate_trace(summary: dict[str, object]) -> None:
    device_names = summary["device_triton_names"]
    stack_names = summary["device_triton_stack_names"]
    if len(device_names) < 2:
        raise AssertionError(
            f"expected forward/backward Triton kernels: {summary}"
        )
    if stack_names != device_names:
        raise AssertionError(f"device Triton stacks are incomplete: {summary}")
    if not summary["forward_cpu_triton_launches"]:
        raise AssertionError(f"forward Triton launch is missing: {summary}")
    if not summary["backward_cpu_triton_launches"]:
        raise AssertionError(f"backward Triton launch is missing: {summary}")
    if summary["torch_to_npu_flow_count"] == 0:
        raise AssertionError(f"torch_to_npu flows are missing: {summary}")
    if (
        summary["ac2g_flow_count"] != 0
        or summary["uid_event_count"] != 0
        or summary["temporary_kernel_category_count"] != 0
    ):
        raise AssertionError(
            f"internal normalized schema leaked to output: {summary}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available")

    torch.manual_seed(20260827)
    device = torch.device("npu:0")
    model = TimelineDemo().to(device).train()
    base = torch.randn(64, 128, device=device)
    reset_inductor_kernel_provenance_debug_handle()
    torch._dynamo.reset()

    with config.patch(
        {
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
        )

        warmup_input = base.detach().clone().requires_grad_(True)
        warmup_grad = torch.randn_like(warmup_input)
        compiled(warmup_input).backward(warmup_grad)
        torch.npu.synchronize()

        handler = inductor_trace_handler(
            str(output_dir), worker_name="triton_experimental"
        )
        prof_input = base.detach().clone().requires_grad_(True)
        prof_grad = torch.randn_like(prof_input)
        with torch_npu.profiler.profile(on_trace_ready=handler):
            output = compiled(prof_input)
            output.backward(prof_grad)
            torch.npu.synchronize()

    exported = sorted(output_dir.glob("*.pt.trace.json"))
    if len(exported) != 1:
        raise AssertionError(f"expected one trace, got {exported}")
    trace_summary = summarize_trace(exported[0])
    validate_trace(trace_summary)
    result = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": torch.npu.get_device_name(0),
        "backend": "triton_experimental",
        "output_checksum": float(output.detach().float().sum().cpu()),
        "grad_checksum": float(prof_input.grad.float().sum().cpu()),
        "trace": str(exported[0]),
        "summary": trace_summary,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
