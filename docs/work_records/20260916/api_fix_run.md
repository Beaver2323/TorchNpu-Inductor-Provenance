> 归档时间：2026-09-18（CST）。9 月 16 日 API 修复阶段快照；后续 rebase 见 ../20260918/README.md。
>
> 原始来源：`Tracking/triton_experimental_delivery/pr_api_fix_20260916.ebZYKd/README.md`。

# 本 PR 公开 API 一行修复

更新时间：2026-09-16 21:54（CST）。已推送源码 HEAD `5efe8a254`，基于 master `0095e028c`。

- 产品修复：`torch_npu/profiler/__init__.py` 设置 handler 的公开 `__module__`。
- 回归测试：新增 `test_inductor_trace_handler_public_api`，修复前 FAIL、修复后 PASS。
- 完整原始公开 API 测试前后均 FAIL；条目 47 → 46，唯一消除项是本 PR handler，新增 0。
- 证据 `result.json` 与 `fix.patch`，全部未过滤日志 `before_*.log`、`after_*.log`。
- 仅 Python 元数据验证，native wheel 为 de350aa，不是当前完整 HEAD / CI 2.15 的验收。
- runtime 是本轮单独安装副本，未修改共享环境。
- 未修改 indexing、旧 annotation 测试、allowlist 或 skip。
- 用户授权后已 commit、无冲突 rebase、显式 force-with-lease 推送到原 PR !46073。
- 推送前再次聚焦回归 PASS，日志 `pre_push_metadata.log`；三提交 range-diff 均为 `=`。
- 远端 Git / PR API 均确认 HEAD `5efe8a2541fcc5eab415d4ef4303949b85c61db3`。
- 21:54 尚未确认新 HEAD 的流水启动或结果，不能复用旧 HEAD 的 #67808 作为新结果。
- `result.json`、`fix.patch` 保留 20:49 修复时的原始证据；推送记录见 `push_record.json`。
- 本地文档已同步状态，GitHub 文档仓本次未提交或推送。

文档仓摘要：`docs/pr_46073_api_fix.md`。
完整验证报告：`Tracking/issues/test_correct_module_names_inductor_trace_handler/修复验证报告.md`。
