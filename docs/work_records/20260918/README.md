> 归档时间：2026-09-18（CST）。工作记录快照，保留原始命令和本机路径；本仓日志使用 .txt 后缀，见上级证据索引。
>
> 原始来源：`Tracking/triton_experimental_delivery/pr_rebase_20260918.xErqjW/README.md`。

# PR !46073：同步已修复基线并重触发流水

更新时间：2026-09-18 10:19（CST，UTC+08:00）。

## 需求 issue 关联（10:19）

用户确认是 provenance 功能需求，而非本地 46 项公开 API 检查失败。
已以用户账号创建 [#4909](https://gitcode.com/Ascend/pytorch/issues/4909)，
并通过正式 API 关联 [PR !46073](https://gitcode.com/Ascend/pytorch/pull/46073)。
关联查询确认成功；没有改 PR 正文、关闭选项或重复触发 CI。
正文备份：`provenance_requirement_issue.md`；证据：`create_issue.log`、`link_issue.log`。

## 操作结果

- 原 PR HEAD：`5efe8a2541fcc5eab415d4ef4303949b85c61db3`。
- 最新 master：`ec693356f4644c5f50c990e78b8fd9ba7737a4d3`。
- rebase 后已推送 HEAD：`195830924122a99e8f97bb85bb6be514d87af6f0`。
- 三个 PR 提交：`05933d054` 功能、`0eba096b8` 文档截图、`195830924` API 修复。
- 无冲突，三个 range-diff 均为 `=`，diff whitespace 检查通过。
- 使用绑定原远端 SHA 的 `--force-with-lease` 推送，远端 Git 已核对一致。
- 备份分支：`codex/provenance-before-rebase-20260918-0954`。
- 上游修复 `13570cf9d`（PR !46394）现为当前 HEAD 的祖先，包含 indexing / expand
  新关键字参数转发。没有在我们的 PR 中另加重复补丁。
- 官方工作流匹配 `^compile$`；已通过用户账号发送 `compile`，远端确认评论 ID
  `190330352`，时间 09:59:05。`trigger_ci.log` 保存发送响应。
- 已确认 [完整流水 #68280](https://www.openlibing.com/apps/pipelineDetail?projectId=4&pipelineId=9ab4a8ad25bd4afeb808647efc5fc2f1&pipelineRunId=352c489602d544728778ef44b5ca8508&codeHostingPlatformFlag=gitcode)
  正在运行，测试提交 `19583092`。机器人评论 ID `190330600`，09:59:48 发布。
  截至 10:01，尚无最终结果；不能将运行中或 docs CI 跳过标成完整 CI 通过。

## 本地验证边界

| 项目 | 结果 |
| --- | --- |
| 公开 API 聚焦回归 | PASS，exit 0，无 skip |
| 完整公开 API 原用例 | FAIL_EXISTING，exit 1，仍为原有 46 项 |
| 与 9 月 16 日失败集合比较 | 完全一致，新增 0、消除 0；目标 handler 不在失败集合中 |
| profiler 初始化 / 处理器文件与隔离 runtime | 两个文件 SHA256 均一致 |
| 当前完整 HEAD 的本地构建 / NPU 模型验证 | 未执行，交由远端完整流水验证 |

使用既有隔离 `de350aa` native wheel + 精确 Python API 修复，PyTorch 2.14 /
Python 3.11；不是完整 `195830924` 或 PyTorch 2.15 的验收。
旧 runner 的 scope 文本保留历史 SHA，但日志中实际测试路径是当前交付工作树。
完整原用例仍有既有失败，按验证技能停止扩测；按用户明确授权继续交付与 CI 触发。
没有修改 allowlist、skip、annotation 测试、共享安装或四个原有 dirty 子模块。

## 完整复跑命令

```bash
cd /home/z50063656/tmp
source /usr/local/Ascend/cann9.0.1/cann-9.0.1/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TORCHINDUCTOR_NPU_BACKEND=triton_experimental
export ASCEND_RT_VISIBLE_DEVICES=6
export PYTHONPATH=/home/z50063656/Tracking/triton_experimental_delivery/pr_api_fix_20260916.ebZYKd/runtime
export TORCH_COMPILE_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR=/home/z50063656/Tracking/triton_experimental_delivery/pr_rebase_20260918.xErqjW/cache
set -o pipefail
/home/z50063656/envs/Tracking/bin/python \
  /home/z50063656/Tracking/triton_experimental_delivery/pr_api_fix_20260916.ebZYKd/run_case.py metadata \
  2>&1 | tee /home/z50063656/Tracking/triton_experimental_delivery/pr_rebase_20260918.xErqjW/pre_push_metadata.log
/home/z50063656/envs/Tracking/bin/python \
  /home/z50063656/Tracking/triton_experimental_delivery/pr_api_fix_20260916.ebZYKd/run_case.py public \
  2>&1 | tee /home/z50063656/Tracking/triton_experimental_delivery/pr_rebase_20260918.xErqjW/pre_push_public.log
```

记录包括 `range_diff.log`、两份完整原始测试日志、`push.log`、`trigger_ci.log`。
本次只推送源码 PR；GitHub 文档仓仅本地更新，未提交或推送。
