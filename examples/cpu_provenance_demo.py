import os

import torch


class ProvenanceDemo(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        added = x + y
        activated = torch.relu(added)
        return activated * 2.0


def main() -> None:
    torch.manual_seed(0)
    model = ProvenanceDemo().eval()
    x = torch.randn(64, 128)
    y = torch.randn(64, 128)

    expected = model(x, y)
    actual = torch.compile(model, backend="inductor", fullgraph=True)(x, y)
    torch.testing.assert_close(actual, expected)

    print(f"torch={torch.__version__}")
    print("device=cpu")
    print(f"checksum={actual.float().sum().item():.6f}")
    print(f"TORCH_TRACE={os.environ.get('TORCH_TRACE')}")
    print(f"TORCH_COMPILE_DEBUG_DIR={os.environ.get('TORCH_COMPILE_DEBUG_DIR')}")


if __name__ == "__main__":
    main()
