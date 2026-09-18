# 项目工作记录与当前状态

更新时间：2026-09-18（CST，UTC+08:00）。本页为工作状态入口；阶段快照中的旧状态不覆盖本页。

## 当前交付状态

| 对象 | 状态 / 版本 |
| --- | --- |
| 文档交付仓 | [Beaver2323/TorchNpu-Inductor-Provenance](https://github.com/Beaver2323/TorchNpu-Inductor-Provenance)，`main` |
| 正式后端范围 | 仅 `torch_npu/_inductor/triton_experimental` |
| 需求 | [issue #4909](https://gitcode.com/Ascend/pytorch/issues/4909)，已正式关联 PR |
| 源码交付 | [PR !46073](https://gitcode.com/Ascend/pytorch/pull/46073)，open；源码不在本仓提交 |
| PR HEAD / 基线 | `195830924122a99e8f97bb85bb6be514d87af6f0` / `ec693356f4644c5f50c990e78b8fd9ba7737a4d3` |
| 最新同步 | 9 月 18 日 rebase 无冲突，3 个提交的 range-diff 均为 `=`，已精确 force-with-lease 推送 |
| 基线兼容修复 | 已包含上游 `13570cf9d`（!46394）的 indexing / expand 新参数转发 |
| 本 PR 公开 API 缺陷 | 已修复并推送，聚焦回归 PASS；完整本地规范检查仍有既有失败 |
| 完整 CI | [#68280](https://www.openlibing.com/apps/pipelineDetail?projectId=4&pipelineId=9ab4a8ad25bd4afeb808647efc5fc2f1&pipelineRunId=352c489602d544728778ef44b5ca8508&codeHostingPlatformFlag=gitcode) 已针对 `19583092` 启动，9 月 18 日 10:38 只读核查仍显示运行中；不是 PASS |
| 其他合入门禁 | 当次查询 CLA 已通过、`needs-issue` 已移除；仍有 `stat/needs-squash`，没有自动 squash 或宣称已可合入 |

共享环境和已有 dirty 子模块未清理。最新 PR 没有在本机重建完整 wheel 或完成全量 2.15
NPU 回归；历史演示、静态映射成功、本地元数据验证都不能替代这项验收。

## 推荐阅读与证据索引

| 问题 | 文档 / 原始记录 |
| --- | --- |
| 现在只做哪些后端？ | [需求变更原文](./requirements_change.md)、[需求 issue 正文](./20260918/provenance_requirement_issue.md) |
| 从头到现在做过什么？ | [Tracking 主线程交接快照](./tracking_handoff_20260918.md)，先看顶部最新章节；旧章节是历史 |
| 当年为什么有多后端 / Pass 记录？ | [历史研究摘要](../history_summary.md)、[迁移说明](./migration_note.md)；不纳入当前验收 |
| 旧 CI 为什么失败？ | [CI 分析](../pr_46073_ci_20260916.md)、[引入来源分析](../pr_46073_origin_analysis.md) |
| 构建与只读归因如何进行？ | [9/16 构建阶段](./20260916/build_and_ci.md)、[归因阶段](./20260916/origin_investigation.md)、[CI wheel 源码清单](./20260916/origin/ci_wheel_manifest.json)、[源码契约对照](./20260916/origin/origin_contracts.json) |
| 本 PR 修了什么？ | [API 修复说明](../pr_46073_api_fix.md)、[复现报告](./api_public_binding/复现报告.md)、[根因](./api_public_binding/根因分析.md)、[验证报告](./api_public_binding/修复验证报告.md)、[合入描述](./api_public_binding/代码合入描述.md) |
| 修复前后能直接对照吗？ | [修复前聚焦日志](./20260916/api_fix/before_metadata.txt)、[修复后聚焦日志](./20260916/api_fix/after_metadata.txt)、[修复前完整检查](./20260916/api_fix/before_public.txt)、[修复后完整检查](./20260916/api_fix/after_public.txt)、[失败集合比较](./20260916/api_fix/result.json)、[修复补丁](./20260916/api_fix/fix.patch) |
| 第一次推送的记录？ | [9/16 修复运行记录](./20260916/api_fix_run.md)、[推送记录](./20260916/api_fix/push_record.json) |
| 新基线 rebase、CI 与 issue 是否真正执行？ | [9/18 工作记录](./20260918/README.md)、[结果](./20260918/result.json)、[range-diff](./20260918/range_diff.txt)、[推送](./20260918/push.txt)、[触发 CI](./20260918/trigger_ci.txt)、[建 issue](./20260918/create_issue.txt)、[正式关联](./20260918/link_issue.txt) |
| 新 HEAD 的公开 API 复验？ | [聚焦日志](./20260918/pre_push_metadata.txt)、[完整原用例日志](./20260918/pre_push_public.txt) |
| 代码 diff 是否完整？ | [逐段讲解与新增修复说明](../pr_diff_walkthrough.md)、[当前完整非测试补丁](../diffs/triton_experimental_provenance_ec693356f_195830924_non_tests.patch) |
| 演示能否下载？ | [CPU 三栏入门](../cpu/README.md)、[NPU 演示索引](../triton_experimental/artifacts/README.md)、[同次采集的 HTML 与 Perfetto trace](../triton_experimental/artifacts/static_smoke/timeline_20260916/README.md) |

## 本地 46 项失败的准确含义

这是 **1 个测试用例** `TestPublicBindings.test_correct_module_names` 列出的 **46 个导出名称**，
不是 46 个 NPU 算子、模型或 provenance 功能失败。

该测试同时扫描 `torch` 和 `torch_npu`。这 46 项均位于本地 `torch.*`：模块没有
`__all__`，导入的非下划线名称被认为是公开接口，但对象的 `__module__` 不属于该模块。
例如模块导入 `collections.defaultdict` 后，检查器将 `defaultdict` 误作该模块拟公开的名字，
而它的真实所属模块仍为 `collections`，因而报告规范不一致。测试不执行它们的数值计算。

| 类别 | 数量 | 例子 |
| --- | --- | --- |
| 实验性量化模块 | 14 | Any、Callable、FakeQuantize、DataLoader |
| 分布式示例模块 | 31 | Future、DTensor、DeviceMesh、lru_cache |
| FX 实验性形状推导 | 1 | defaultdict |

修复前有 47 项；修复后仅消除了本 PR 的 `torch_npu.profiler.inductor_trace_handler`，
其余 46 项不变。9 月 18 日复验仍为同一集合，新增 0；完整用例 exit 1，明确记录为
`FAIL_EXISTING`，没有过滤断言、增加 allowlist 或把失败记为通过。
“本地既有”只针对记录中的 PyTorch 2.14 源码环境，不表示最新 2.15 CI 同样失败。

## 下一步与未验收项

1. 查看 #68280 最终结果，分别检查公开 API、provenance 端到端和 annotation；不要重复触发
   流水，也不要用旧 HEAD 的 #67808 代替新结果。
2. 需要改代码、调整测试范围或再次 squash / 推送源码时，另按用户指示执行。
3. ComboKernel、AOTInductor 及已排除后端保持原边界，不因基线升级自动记为已支持。

## 归档规则与去重

- 本次来源包括 `Tracking` 和文档仓现有待提交文档；[来源清单](./source_inventory.json)
  记录来源路径、处理方式、SHA256 与去重说明。
- [文档发布检查](./publication_checks_20260918.md)记录链接、格式、补丁一致性和敏感信息检查范围。
- 当前指南继续维护在文档仓；Tracking 中同主题的新手指南 / 技术研究不再复制第二套。
  早期多后端演示的有用结论归入[历史摘要](../history_summary.md)和主线程交接快照，
  不作为当前支持矩阵的证据。
- 阶段原始报告保留原时间、版本和绝对路径，并加归档提示。路径是原机证据位置，
  不是读者 clone 文档仓后自然存在的运行环境。
- 原始运行日志用 `.txt` 后缀发布，以明确区分精选证据与被 `.gitignore` 排除的大量临时日志。
  JSON、trace、HTML 和源码接口名称不翻译、不修改历史版本信息。
- 不发布环境、wheel、完整源码树、编译缓存、凭据、临时认证脚本或误建 Pass 补丁。
  未整包上传远端 CI 日志；已在分析文档中保留公开流水地址及对应失败证据。

本文是带日期的交付快照；当前实时状态请以链接到的 PR、issue 和流水页面为准。
