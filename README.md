# TorchNpu Inductor Provenance

Research notes, runnable probes, and NPU adaptation work for TorchInductor
provenance tracking on Ascend NPU.

This first public snapshot focuses on documentation and reproducible examples.
The scoped `torch_npu` implementation patch and generated trace/HTML artifacts
will be published separately after they are cleaned up for review.

## Current status

- CPU Inductor provenance and `tlparse` visualization: verified.
- Ascend NPU Triton provenance: verified end to end.
- FlexAttention forward template provenance: verified end to end.
- Default `BlockMask=None` forward provenance: verified end to end.
- FX graph cache miss/hit provenance: verified.
- Default `BlockMask=None` backward: under investigation. The oversized sparse
  sentinel has been bounded to the real tile count, while dK/dV compilation is
  still blocked in `bishengir-compile` before backward `output_code.py` is
  generated.

The validated environment used PyTorch `release/2.14`, an editable `torch_npu`
build, Triton Ascend `release/3.2.2`, and Ascend 910B2 devices.

## Start here

1. [Beginner guide](docs/inductor_provenance_npu_beginner_guide.md)
2. [Full research notes](docs/inductor_provenance_npu_research.md)
3. [CPU visualization demo](docs/cpu_provenance_visualization_demo.md)
4. [NPU visualization demo](docs/npu_provenance_visualization_demo.md)
5. [FlexAttention template demo](docs/npu_template_provenance_visualization_demo.md)
6. [Default BlockMask demo](docs/npu_default_block_mask_provenance_demo.md)
7. [Default BlockMask backward investigation](docs/npu_default_block_mask_backward_investigation.md)

## Repository layout

```text
docs/       Research notes, environment baselines, and visualization guides
examples/   Minimal CPU/NPU provenance probes
```

Run NPU examples from a directory outside a `torch_npu` source tree to avoid
source-tree import contamination. Select an idle device with `npu-smi info`
before setting `ASCEND_RT_VISIBLE_DEVICES`.

Some documents retain absolute paths and links to generated artifacts from the
original experiment host. Those paths are evidence and reproduction references;
large caches, structured traces, and generated HTML are intentionally excluded
from this initial snapshot.

## Scope

This repository concerns static compiler provenance:

```text
pre-grad FX nodes <-> post-grad FX nodes <-> generated kernels
```

Ascend profiler timeline attribution is a separate follow-up because it depends
on CANN/Ascend trace schemas rather than the CUDA/Kineto-specific adapter.

