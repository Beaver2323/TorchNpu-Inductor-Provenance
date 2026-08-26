# TorchNpu Inductor Provenance

Research notes, runnable probes, and NPU adaptation work for TorchInductor
provenance tracking on Ascend NPU.

## Repository role

This GitHub repository was used for early research and documentation. It is not
the source-code delivery target for the current implementation.

- Official target: `https://gitcode.com/Ascend/pytorch`
- Development fork: `https://gitcode.com/gcw_3ffySSwy/pytorch`
- Historical architect staging/reference repository:
  `https://gitcode.com/rmch/npu_inductor_2.13.0` (not a delivery target or fork)
- Workflow reference: `https://gitcode.com/AllenGuanC/inductor-meta-worktree`

The active delivery scope is only
`torch_npu/_inductor/triton_experimental`. The source implementation is pushed
to the GitCode development fork on branch
`codex/triton-experimental-provenance-delivery` at commit `bb356bffb`, based on
official commit `83cc45248`; the official repository remains the PR target.
This repository continues to retain useful background material and runnable
examples; older multi-backend sections are historical research rather than the
current acceptance scope.

## Current status

- CPU Inductor provenance and `tlparse` visualization: verified.
- `triton_experimental` static provenance level 1/2: verified on NPU.
- `triton_experimental` forward/backward runtime timeline attribution: verified.
- `triton_experimental` rsplit partial/combine timeline attribution: verified.
- NPU trace list/dict roots, gzip output, event limits, and state cleanup:
  covered by focused tests.
- AOTInductor `kernel_information.json`: not accepted on the current baseline;
  it is blocked by NPU AOTI device/support and shared lazy/ABI prerequisites.
- FlexAttention and other backend findings remain historical evidence and are
  not part of the current delivery.

The final validation used PyTorch `release/2.14`, an isolated install of the
matching `torch_npu` validation wheel, Triton Ascend `release/3.2.2`, and Ascend
910B2 devices.

## Current `triton_experimental` demo

1. [Guide and artifact index](docs/inductor_provenance_demo/triton_experimental/README.md)
2. [Static three-panel provenance HTML](docs/inductor_provenance_demo/triton_experimental/provenance_tracking.html)
3. Static JSON: [node mappings](docs/inductor_provenance_demo/triton_experimental/node_mappings.json),
   [kernel stacks](docs/inductor_provenance_demo/triton_experimental/kernel_stack_traces.json),
   [level 1 result](docs/inductor_provenance_demo/triton_experimental/static_result.json), and
   [level 2 result](docs/inductor_provenance_demo/triton_experimental/static_level2_result.json)
4. Forward/backward timeline: [result](docs/inductor_provenance_demo/triton_experimental/timeline_forward_backward_result.json)
   and [Perfetto trace](docs/inductor_provenance_demo/triton_experimental/timeline_forward_backward_trace.json)
5. Rsplit timeline: [result](docs/inductor_provenance_demo/triton_experimental/timeline_rsplit_result.json)
   and [Perfetto trace](docs/inductor_provenance_demo/triton_experimental/timeline_rsplit_trace.json)
6. Reproduction scripts: [static](docs/inductor_provenance_demo/triton_experimental/static_probe.py),
   [forward/backward timeline](docs/inductor_provenance_demo/triton_experimental/timeline_probe.py), and
   [rsplit timeline](docs/inductor_provenance_demo/triton_experimental/rsplit_timeline_probe.py)

Download the HTML to open the interactive three-panel view locally. Load the
timeline trace JSON files into Perfetto.

## Historical research

- [Beginner guide](docs/inductor_provenance_npu_beginner_guide.md)
- [Full research notes](docs/inductor_provenance_npu_research.md)
- [CPU visualization research](docs/cpu_provenance_visualization_demo.md)
- [Earlier NPU visualization research](docs/npu_provenance_visualization_demo.md)
- [FlexAttention template research](docs/npu_template_provenance_visualization_demo.md)
- [Default BlockMask research](docs/npu_default_block_mask_provenance_demo.md)
- [Default BlockMask backward investigation](docs/npu_default_block_mask_backward_investigation.md)

## Repository layout

```text
docs/       Research notes, guides, curated HTML/JSON evidence, and NPU probes
examples/   Earlier minimal CPU/NPU provenance probes
```

Run NPU examples from a directory outside a `torch_npu` source tree to avoid
source-tree import contamination. Select an idle device with `npu-smi info`
before setting `ASCEND_RT_VISIBLE_DEVICES`.

Some documents retain absolute paths and links to generated artifacts from the
original experiment host. Those paths are evidence and reproduction references;
the curated `triton_experimental` HTML, static mappings, timeline traces/results,
and reproduction scripts are included. Wheels, build logs, caches, and temporary
profiler directories remain excluded.

## Scope

The current delivery concerns both static compiler provenance and runtime
timeline attribution for `triton_experimental`:

```text
pre-grad FX nodes <-> post-grad FX nodes <-> generated kernels
```

Runtime attribution uses an NPU adapter for CANN/Ascend trace schemas while
reusing the community Inductor trace processor. Default Triton, CATLASS, MLIR,
DVM, AKG, torchair, and AOTInductor acceptance remain outside this delivery.
