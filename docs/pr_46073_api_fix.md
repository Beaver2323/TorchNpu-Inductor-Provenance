# PR !46073：仅修复本 PR 的公开 API 问题

更新时间：2026-09-18 10:01（CST，UTC+08:00）。

状态：**已提交并推送到 PR !46073，聚焦回归通过；不是全量 CI 通过。**

## 最新进展：2026-09-18

- 已 rebase 最新 master `ec693356f` 并推送，PR HEAD 为 `195830924`，三个补丁内容均不变。
- 上游 !46394 / `13570cf9d` 已修复 indexing / expand 的 2.15 参数兼容，本轮已包含。
- 公开 API 聚焦回归再次 PASS；完整原用例仍为本地既有 46 项失败，集合不变，新增 0。
- 验证仍使用下述 2.14 隔离 runtime，不冒充完整新 HEAD 或 2.15 验收。
- 用户授权后已发送 `compile`，机器人确认完整流水 #68280 针对 `19583092` 运行中；
  [新流水入口](https://www.openlibing.com/apps/pipelineDetail?projectId=4&pipelineId=9ab4a8ad25bd4afeb808647efc5fc2f1&pipelineRunId=352c489602d544728778ef44b5ca8508&codeHostingPlatformFlag=gitcode)，最终结果待确认。
- 原始日志和复跑命令：`Tracking/triton_experimental_delivery/pr_rebase_20260918.xErqjW/`。
- 本页与原始证据已纳入 9 月 18 日文档仓归档；下方历史阶段保留当时的推送状态。

## 2026-09-16 提交与推送状态（历史）

- 用户授权后，先提交本 PR 的公开 API 修复，再 rebase 到本次拉取的 master
  `0095e028c54ccd86e83dcf3e9acccfca147e6084`，无冲突。
- 当前远端 PR HEAD：`5efe8a2541fcc5eab415d4ef4303949b85c61db3`；该提交就是 API 修复。
- 提交作者：`gcw_3ffySSwy <1305321851@qq.com>`。
- 三个 PR 提交的 rebase 前后 range-diff 均为 `=`；`git diff --check` 通过。
- 使用绑定旧远端 `dbc0db52fa3384575fd83cc4db23c15a298f802d` 的
  `--force-with-lease` 推送，未覆盖其他人的新提交。
- Git 远端和 [PR API](https://api.gitcode.com/api/v5/repos/Ascend/pytorch/pulls/46073)
  均确认新 HEAD；PR 为 open，`mergeable=true` 仅表示无合并冲突，不表示 CI / 审批通过。
- 21:54 检查时，评论区最新可见流水仍是旧 HEAD `dbc0db52` 的 #67808；
  尚未确认新 HEAD 的流水启动或结果，没有代发 `compile` 命令。
- 本页及本地交接记录已更新；本次仅推送源码 PR，未提交或推送另一个 GitHub 文档仓的待提交修改。

## 改了什么

只在 `torch_npu/profiler/__init__.py` 增加一行有效代码：

```python
inductor_trace_handler.__module__ = __name__
```

函数现在声明自己属于公开模块 `torch_npu.profiler`，而不是私有实现模块。
用户原有导入方式、函数签名、函数身份、trace 处理逻辑都不变。

在本 PR 已新增的 `test/profiler/test_inductor_profiler.py` 中补充
`test_inductor_trace_handler_public_api`，检查公开列表、对象同一性、模块名及 pickle 往返。
本轮源码共修改这两个文件，其他用户修改保留。

## 实际验证结果

| 检查 | 修复前 | 修复后 |
| --- | --- | --- |
| 新增聚焦回归用例 | FAIL，exit 1 | PASS，exit 0 |
| 未修改的完整公开 API 原用例 | FAIL，47 个问题项 | FAIL_EXISTING，46 个问题项 |
| 本 PR 的 handler 是否仍在失败清单 | 是 | 否 |
| 新增失败项 | — | 0 |
| `git diff --check` | — | 通过 |

完整检查唯一消除的条目是 `torch_npu.profiler.inductor_trace_handler`，其余 46 个
`torch.*` 条目与修复前相同。不修改它们，也不把完整检查标成 PASS。

测试从 `/home/z50063656/tmp` 启动，使用隔离 wheel 副本并核对修复文件哈希。
原生 wheel 基于 `de350aa`，Python 修复来自当前 `7b8795bb9` 工作树，
涉及的 profiler 文件与原始检查文件已确认一致。环境为 PyTorch 2.14 / Python 3.11 /
aarch64，**不能冒充完整当前 HEAD 或远端 2.15 CI 的验收**。

最终 rebase 到 `5efe8a254` 后，推送前再次运行聚焦用例：PASS，exit 0。
日志为 `pre_push_metadata.log`，当前源码与隔离 runtime 的 profiler 初始化文件
SHA256 均为 `1ec815a37c9d44ac3a6eb813a323fc3d6ebf13c746d79fc8ef538e6eb79d9295`。
该复验仍仅覆盖 Python 元数据修复，不扩大前述 wheel / 环境验证边界。

## 明确没有修改

- `NPUTritonKernel.indexing` 的 PyTorch 2.15 参数兼容问题。
- CI 使用旧 annotation overlap 测试的问题。
- 任何 allowlist、disabled 测试列表或原始公开 API 检查断言。
- 共享 PyTorch / torch_npu 安装、其他进程的源码修改。

因此，推送这次修复后，远端流水仍可能被另外两项问题阻断。
它们的引入来源见[归因文档](./pr_46073_origin_analysis.md)。

## 证据与交接

本轮证据目录：

```text
/home/z50063656/Tracking/triton_experimental_delivery/pr_api_fix_20260916.ebZYKd/
```

包含修复前后四份原始日志、`result.json`、`fix.patch` 和可重跑的脚本。
完整复现、根因、验证报告及合入描述草案位于：

```text
/home/z50063656/Tracking/issues/test_correct_module_names_inductor_trace_handler/
```

验证技能要求完整原用例仍失败时停止扩测；本轮保留既有失败，未扩修其他问题。
源码修复已提交并推送；源码工作树仅保留原有四个子模块的 dirty 状态。
