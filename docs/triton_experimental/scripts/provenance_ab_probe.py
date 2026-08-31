import argparse
import json
import os
import re
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from torch._inductor import config
from torch._inductor.debug import (
    reset_inductor_kernel_provenance_debug_handle,
)


class ConvNeXtBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(
            64, 64, kernel_size=7, padding=3, groups=64
        )
        self.norm = torch.nn.LayerNorm(64)
        self.pointwise1 = torch.nn.Linear(64, 256)
        self.pointwise2 = torch.nn.Linear(256, 64)
        self.layer_scale = torch.nn.Parameter(torch.full((64,), 1e-6))

    def forward(self, x):
        residual = x
        hidden = self.depthwise(x)
        hidden = hidden.permute(0, 2, 3, 1)
        hidden = self.norm(hidden)
        hidden = F.gelu(self.pointwise1(hidden))
        hidden = self.pointwise2(hidden) * self.layer_scale
        hidden = hidden.permute(0, 3, 1, 2)
        return hidden + residual


class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x):
        return self.layer(x)


def _compiled(model):
    return torch.compile(
        model,
        backend="inductor",
        options={"npu_backend": "triton_experimental"},
        fullgraph=True,
        dynamic=True,
    )


def _run_convnext():
    torch.manual_seed(20260901)
    model = ConvNeXtBlock().npu().train()
    value = torch.randn(2, 64, 16, 16, device="npu", requires_grad=True)
    grad = torch.randn_like(value)
    output = _compiled(model)(value)
    output.backward(grad)
    torch.npu.synchronize()


def _run_transformer():
    torch.manual_seed(20260902)
    model = TransformerEncoderBlock().npu().train()
    compiled = _compiled(model)

    base = torch.randn(2, 32, 256, device="npu", requires_grad=True)
    base_output = compiled(base)
    base_output.backward(torch.randn_like(base_output))
    torch.npu.synchronize()
    model.zero_grad(set_to_none=True)

    alternate = torch.randn(3, 24, 256, device="npu", requires_grad=True)
    alternate_output = compiled(alternate)
    alternate_output.backward(torch.randn_like(alternate_output))
    torch.npu.synchronize()


def _kernel_names(text):
    return sorted(
        set(
            re.findall(
                r"(?:Name=)?((?:triton|k)_[A-Za-z0-9_]+)",
                text,
            )
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("convnext", "transformer"), required=True)
    parser.add_argument("--level", type=int, choices=(0, 2), required=True)
    parser.add_argument("--timeline", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("TORCH_COMPILE_DEBUG") != "1":
        raise RuntimeError("TORCH_COMPILE_DEBUG=1 is required")
    if args.timeline != (args.level == 2):
        raise ValueError("use timeline off for level 0 and on for level 2")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    compile_debug = args.output_dir / "compile_debug"
    reset_inductor_kernel_provenance_debug_handle()
    torch._dynamo.reset()

    result = {
        "case": args.case,
        "configured_provenance_level": args.level,
        "timeline_provenance": args.timeline,
        "torch_compile_debug": True,
        "device": "npu",
    }
    try:
        with config.patch(
            {
                "trace.enabled": True,
                "trace.debug_dir": str(compile_debug),
                "trace.provenance_tracking_level": args.level,
                "trace.provenance_tracking_to_timeline": args.timeline,
                "triton.unique_kernel_names": True,
                "force_disable_caches": True,
            }
        ):
            result["effective_provenance_level"] = (
                config.effective_provenance_tracking_level()
            )
            if args.case == "convnext":
                _run_convnext()
            else:
                _run_transformer()
    except Exception as error:
        formatted_traceback = traceback.format_exc()
        result.update(
            {
                "outcome": "FAIL",
                "exception_type": (
                    f"{type(error).__module__}.{type(error).__qualname__}"
                ),
                "exception_message": str(error),
                "kernel_names_in_exception": _kernel_names(
                    formatted_traceback
                ),
                "traceback": formatted_traceback,
            }
        )
    else:
        result["outcome"] = "PASS"

    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "case",
                    "configured_provenance_level",
                    "effective_provenance_level",
                    "timeline_provenance",
                    "outcome",
                    "exception_type",
                    "kernel_names_in_exception",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
