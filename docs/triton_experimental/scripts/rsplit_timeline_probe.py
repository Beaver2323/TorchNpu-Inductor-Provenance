#!/usr/bin/env python3
"""Capture provenance for the two kernels emitted by an NPU rsplit sum."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu
from torch._inductor import config
from torch._inductor.debug import reset_inductor_kernel_provenance_debug_handle
from torch_npu.profiler import inductor_trace_handler


class RsplitDemo(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = torch.relu(torch.sin(x))
        return (activated * x).sum()


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
    device_kernels = [
        event
        for event in events
        if str(event.get("name", "")).startswith(("triton_", "k_"))
        and str((event.get("args") or {}).get("Task Type", "")).startswith(
            "KERNEL_"
        )
    ]
    cpu_launches = [
        event
        for event in events
        if event.get("cat") == "cpu_op"
        and str(event.get("name", "")).startswith(("triton_", "k_"))
    ]
    flows = [event for event in events if event.get("ph") in ("s", "f")]
    return {
        "trace_root": type(trace).__name__,
        "event_count": len(events),
        "device_kernel_names": [event["name"] for event in device_kernels],
        "device_kernel_stack_names": [
            event["name"]
            for event in device_kernels
            if (event.get("args") or {}).get("stack")
        ],
        "device_kernel_stacks": {
            event["name"]: (event.get("args") or {}).get("stack")
            for event in device_kernels
        },
        "cpu_launch_names": [event["name"] for event in cpu_launches],
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
    device_names = summary["device_kernel_names"]
    if len(device_names) != 2:
        raise AssertionError(f"expected two rsplit device kernels: {summary}")
    if summary["device_kernel_stack_names"] != device_names:
        raise AssertionError(f"rsplit device stacks are incomplete: {summary}")
    if summary["cpu_launch_names"] != device_names:
        raise AssertionError(f"rsplit CPU launches do not match: {summary}")
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

    torch.manual_seed(20260828)
    device = torch.device("npu:0")
    model = RsplitDemo().to(device).eval()
    x = torch.randn(64, 128, device=device)
    expected = model(x)
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
        compiled(x)
        torch.npu.synchronize()

        handler = inductor_trace_handler(
            str(output_dir), worker_name="triton_experimental_rsplit"
        )
        with torch_npu.profiler.profile(on_trace_ready=handler):
            output = compiled(x)
            torch.npu.synchronize()

    torch.testing.assert_close(output, expected)
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
        "output": float(output.detach().cpu()),
        "trace": str(exported[0]),
        "summary": trace_summary,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
