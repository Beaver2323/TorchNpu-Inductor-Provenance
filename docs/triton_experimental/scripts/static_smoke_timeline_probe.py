"""Capture matching static provenance and a Perfetto trace for relu(x + 1) * 2."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tlparse", required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["TORCH_TRACE"] = str(output_dir / "torch_trace")

    import torch
    import torch_npu
    import triton
    from torch._inductor import config
    from torch._inductor.debug import reset_inductor_kernel_provenance_debug_handle
    from torch_npu.profiler import inductor_trace_handler

    class Model(torch.nn.Module):
        def forward(self, x):
            added = x + 1
            activated = torch.relu(added)
            return activated * 2

    print("runtime:", torch.__version__, torch_npu.__version__, flush=True)
    print("torch_npu:", torch_npu.__file__, flush=True)
    torch.manual_seed(20260916)
    model = Model().npu().eval()
    x = torch.randn(4096, device="npu")
    expected = model(x)
    torch._dynamo.reset()
    reset_inductor_kernel_provenance_debug_handle()
    with torch.no_grad(), config.patch({
        "trace.enabled": True,
        "trace.debug_dir": str(output_dir / "compile_debug"),
        "trace.provenance_tracking_level": 1,
        "trace.provenance_tracking_to_timeline": True,
        "triton.unique_kernel_names": True,
        "force_disable_caches": True,
    }):
        compiled = torch.compile(
            model, backend="inductor", fullgraph=True,
            options={"npu_backend": "triton_experimental"},
        )
        print("compiling warmup", flush=True)
        torch.testing.assert_close(compiled(x), expected)
        torch.npu.synchronize()
        print("capturing NPU trace", flush=True)
        handler = inductor_trace_handler(str(output_dir), worker_name="static_smoke")
        with torch_npu.profiler.profile(on_trace_ready=handler):
            actual = compiled(x)
            torch.npu.synchronize()
        torch.testing.assert_close(actual, expected)

    exported = list(output_dir.glob("*.pt.trace.json"))
    if len(exported) != 1:
        raise AssertionError(f"expected one profiler trace, got {exported}")
    trace_path = output_dir / "trace.json"
    exported[0].rename(trace_path)
    trace = json.loads(trace_path.read_text())
    events = trace if isinstance(trace, list) else trace["traceEvents"]
    kernels = [event for event in events if isinstance(event, dict)
               and str(event.get("name", "")).startswith(("triton_", "k_"))
               and str((event.get("args") or {}).get("Task Type", "")).startswith("KERNEL_")]
    if len(kernels) != 1 or not kernels[0].get("args", {}).get("stack"):
        raise AssertionError(f"expected one device kernel with stack: {kernels}")

    mapping_paths = list((output_dir / "compile_debug").rglob(
        "inductor_provenance_tracking_node_mappings.json"))
    if len(mapping_paths) != 1:
        raise AssertionError(f"expected one static graph: {mapping_paths}")
    mapping = json.loads(mapping_paths[0].read_text())
    kernel_mapping = mapping.get("cppCodeToPost", {})
    if len(kernel_mapping) != 1 or set(next(iter(kernel_mapping.values()))) != {"add", "relu", "mul"}:
        raise AssertionError(f"unexpected static mapping: {mapping}")
    shutil.copyfile(mapping_paths[0], output_dir / "node_mappings.json")
    stack_paths = list((output_dir / "compile_debug").rglob(
        "inductor_provenance_tracking_kernel_stack_traces.json"))
    if len(stack_paths) == 1:
        shutil.copyfile(stack_paths[0], output_dir / "kernel_stack_traces.json")

    logs = list((output_dir / "torch_trace").glob("*.log"))
    if len(logs) != 1:
        raise AssertionError(f"expected one structured log: {logs}")
    subprocess.run([
        args.tlparse, str(logs[0]), "--inductor-provenance",
        "--no-browser", "-o", str(output_dir / "tlparse"),
    ], check=True)
    # tlparse may also emit an unattributed page with an empty mapping.
    pages = list((output_dir / "tlparse").rglob("provenance_tracking*.html"))
    htmls = [path for path in pages
             if all(key in path.read_text() for key in kernel_mapping)]
    if len(htmls) != 1:
        raise AssertionError(f"expected one page matching the kernel handles: {htmls}")
    shutil.copyfile(htmls[0], output_dir / "provenance_tracking.html")
    result = {
        "status": "PASS", "time_utc": datetime.now(timezone.utc).isoformat(),
        "model": "relu(x + 1) * 2", "scope": "inference forward only",
        "shape": [4096], "backend": "triton_experimental",
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "triton": triton.__version__, "torch_module": torch.__file__,
        "torch_npu_module": torch_npu.__file__,
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "device": torch.npu.get_device_name(0), "force_disable_caches": True,
        "max_abs_diff": float((actual - expected).abs().max().cpu()),
        "kernel_to_post": kernel_mapping, "event_count": len(events),
        "device_kernel_name": kernels[0]["name"],
        "device_kernel_stack": kernels[0]["args"]["stack"],
        "trace": "trace.json", "html": "provenance_tracking.html",
        "tlparse_pages": [path.name for path in pages],
        "selected_tlparse_page": htmls[0].name,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Existing isolated validation wheel; not latest PR full regression.",
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
