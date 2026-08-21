import os

import torch
import torch_npu


class ProvenanceDemo(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        added = x + y
        activated = torch.relu(added)
        return activated * 2.0


def main() -> None:
    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available")

    torch.manual_seed(0)
    device = torch.device("npu:0")
    model = ProvenanceDemo().to(device)
    x = torch.randn(64, 128, device=device)
    y = torch.randn(64, 128, device=device)

    expected = model(x, y)
    compiled_model = torch.compile(model, backend="inductor", fullgraph=True)
    actual = compiled_model(x, y)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected)

    print(f"torch={torch.__version__}")
    print(f"torch_npu={torch_npu.__version__}")
    print(f"device={torch.npu.get_device_name(0)}")
    print(f"checksum={actual.float().sum().item():.6f}")
    print(f"TORCH_TRACE={os.environ.get('TORCH_TRACE')}")
    print(f"TORCH_COMPILE_DEBUG_DIR={os.environ.get('TORCH_COMPILE_DEBUG_DIR')}")


if __name__ == "__main__":
    main()
