> 归档时间：2026-09-18（CST）。9 月 16 日只读归因阶段快照；后续基线已修复，当前状态见上级 README。
>
> 原始来源：`Tracking/triton_experimental_delivery/pr_origin_20260916.7sterg/README.md`。

# PR !46073 引入来源核查

更新时间：2026-09-16 20:04（CST）。仅 rebase、诊断，不修复、不推送。

- BASE：`2a9b559fb237dde85fce6bb72cba931fd6d38548`。
- HEAD：`7b8795bb9045e83ce293cfb682eb98aab1fd3a8f`。
- 功能提交：`52cf72e6c3738d1128be79198e5bcdd8f2a4fa4f`。
- 两个提交 range-diff 均为 `=`，原工作树子模块 dirty 状态保留。
- API 导出缺陷来自原 PR `00256261d`。
- indexing 参数变更来自社区 #192410；CI nightly 引入快照 `461e8d2656f2`。
- profiler step 对齐已由社区 #193042 删除；CI nightly 引入快照 `644e799e34e0`，同时删除旧测试。
- 已核对 CI wheel git_version `3c73a854`，提取的三个 Python 文件与 Git 源码完全相同。
- 三份 NPU indexing AST 完全相同，传入 CI 新参数都发生 TypeError；这是源码契约复现，不是完整 NPU 模型实测。
- wheel 只下载、提取源码，没有安装。共享 Python / PyTorch / torch_npu 环境未修改。

详细分析文档：
`/home/z50063656/TorchNpu-Inductor-Provenance/docs/pr_46073_origin_analysis.md`。

复核命令（必须从 tmp 运行）：

```bash
cd /home/z50063656/tmp
/home/z50063656/envs/Tracking/bin/python \
  /home/z50063656/Tracking/triton_experimental_delivery/pr_origin_20260916.7sterg/check_origins.py
```

`origin_contracts.json` 中主线 PR 的祖先检查为 false、nightly 快照检查为 true 是预期结果：
CI wheel 使用 nightly 快照链，不应把不同历史中的 SHA 混为直接祖先。
