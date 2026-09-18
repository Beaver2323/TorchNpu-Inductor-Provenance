> 归档时间：2026-09-18（CST）。9 月 16 日构建与旧流水阶段快照，不是当前待办；后续修复及推送见上级 README。
>
> 原始来源：`Tracking/triton_experimental_delivery/pr_verify_20260916.b2PVBM/README.md`。

# PR !46073 rebase 后验证记录

更新时间：2026-09-16（CST，UTC+08:00）。状态：进行中，尚无本轮 NPU PASS 结论。

20:07 后续进展：`de350aa` 的隔离 wheel 已构建成功，但用户要求先确认引入来源，
本轮没有安装或运行该 wheel。当前开发分支已再次 rebase 到 master `2a9b559fb`，
HEAD `7b8795bb9`。本目录保留第一次 rebase 的构建/CI 留证，新的引入来源核查在
`../pr_origin_20260916.7sterg/`，详细说明见文档仓 `docs/pr_46073_origin_analysis.md`。

- 原 PR HEAD：`dbc0db52fa3384575fd83cc4db23c15a298f802d`。
- 拉取的官方 master：`c01b3d31a28f7b7056087ad005c23fa44d49339e`。
- 本地 rebase 后 HEAD：`de350aa0fbe0dc925e0da02314e05870c635b95b`。
- 两个提交 range-diff 均为 `=`，无冲突，`git diff --check` 通过。
- 备份分支：`codex/provenance-backup-before-rebase-20260916`。
- 未推送；原工作树的 Tensorpipe、DVM、op-plugin、torchair 状态未清理、未覆盖。
- 验证从 `/home/z50063656/tmp` 启动。独立构建副本采用各子模块的精确 gitlink 版本；
  torchair 与 torch-mlir 不在功能范围，不参与构建。共享 Python 环境保持不变。

## 远端流水线

GitCode 只读 API 确认最新回报为 `PR-pipeline_pytorch#67808`，运行提交 `dbc0db52`，
评论时间 `2026-09-16T09:52:22+08:00`。

- Build_X86、Build_ARM、Build_X86_213、Build_ARM_213：COMPLETED。
- codecheck_pre-commit、check_error、lintrunner、Antipoison、SCA：COMPLETED。
- UT_ARM_A2_Part_01、UT_ARM_A2_Select_Part_01：FAILED。
- UT_inductor_Part_213：COMPLETED。
- UT_inductor_Part_01/02/03/04：IGNORED，不算通过。
- 已获取两项失败任务的公开日志及分页信息，存于 `remote_ci/`。
- `test_correct_module_names`：新增公开 API `inductor_trace_handler.__module__` 仍指向私有模块，1 FAIL。
- `test_inductor_profiler.py`：8 PASS、4 ERROR；`test_inductor_provenance_models.py`：3 ERROR。
  7 个 ERROR 都是 `indexing()` 不接受 `allow_reduction_invariant_indexing`，发生在 kernel 代码生成阶段。
- 既有 `test_profiler.py`：147 项，2 FAIL、80 SKIP、65 PASS；两项失败均为 CPU annotation overlap，归因待核查。
- 远端安装 PyTorch `2.15.0.dev20260902+cpu`，本地为 `2.14.0a0+git8e86e0a`，不能混用验收结论。
- CI 除 PR HEAD 还执行自动合并 master；原始日志保留了检出/merge 步骤，不能把它当纯 HEAD 的本地复现。

详细中文说明已写入
`/home/z50063656/TorchNpu-Inductor-Provenance/docs/pr_46073_ci_20260916.md`。

## 独立构建进度

- 私有 build-only torchgen 副本补齐 packaged 数据，并隔离 `tools` 包名冲突；没有修改共享 torch 源码。
- Tensorpipe 5 个嵌套子模块已按精确 gitlink 准备。
- `build_retry3.log` 构建完成，wheel 位于 `wheels/torch_npu-2.14.0a0+gitde350aa-cp311-cp311-linux_aarch64.whl`；
  没有使用旧 83cc452 或这个 de350aa 安装冒充第二次 rebase 后的 7b8795bb9。
- 产品代码未修复；已向用户询问是否允许修复公开 API 导出与 2.15 参数兼容问题，暂不推送。
- `cpu_annotation_torch214.log`：原生社区 2.14 CPU annotation 用例 1 PASS、exit code 0；
  禁用后端自动加载、未导入 torch_npu，不计入 PR NPU 回归或 CI 2.15 对照。

[流水线详情](https://www.openlibing.com/apps/pipelineDetail?projectId=4&pipelineId=9ab4a8ad25bd4afeb808647efc5fc2f1&pipelineRunId=3c4da1bc2e5a41fcad0c60a9039594f1&codeHostingPlatformFlag=gitcode)

## 待执行

1. 独立构建并隔离安装 rebase 后版本，核对导入路径和源码哈希。
2. 顺序执行 profiler、静态映射、模块测试；失败先记录分析，SKIP 不计 PASS。
3. 生成新的前向/反向 timeline 与静态演示，保留原始结果。
4. 更新本记录；若构建或环境受阻，明确区分环境阻断与功能失败。
