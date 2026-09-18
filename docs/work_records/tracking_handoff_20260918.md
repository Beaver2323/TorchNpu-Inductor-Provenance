> 归档时间：2026-09-18（CST）。快照；下方旧阶段仅供历史参考，当前状态请看本目录 README。
>
> 原始来源：`Tracking/MAIN_THREAD_HANDOFF.md`。

# Main thread 交接：Inductor provenance NPU 适配

更新时间：2026-09-18 10:19（CST，UTC+08:00）

## 2026-09-18 10:19 已新建并正式关联需求 issue（最新）

- 用户明确要求为 provenance 功能新建需求单，不是为本地 46 项公开 API 失败建单。
- 已以 `gcw_3ffySSwy` 创建 [issue #4909](https://gitcode.com/Ascend/pytorch/issues/4909)：
  `[需求] 为 triton_experimental 后端补齐 Inductor provenance 来源追踪`。
- 已通过 GitCode 正式关联 API 将 #4909 关联到 PR !46073，并用关联查询接口核实成功。
- issue 包含静态映射、timeline、rsplit、公开 API、测试与中文文档交付要求，
  明确其他后端 / ComboKernel / AOTI 边界，不预先标记全量验收通过。
- 本轮未修改源码或 PR 正文，未改合并后自动关闭设置，未重复触发流水。
- 正文备份和操作结果：`Tracking/triton_experimental_delivery/pr_rebase_20260918.xErqjW/`
  下的 `provenance_requirement_issue.md`、`create_issue.log`、`link_issue.log`。

## 2026-09-18 10:01 已同步新基线，完整流水 #68280 运行中（最新）

- 用户授权执行 rebase → 推送 → 触发完整流水。
- 最新 master：`ec693356f4644c5f50c990e78b8fd9ba7737a4d3`。
- 新本地 / 远端 HEAD：`195830924122a99e8f97bb85bb6be514d87af6f0`。
- 三个 PR 提交为 `05933d054`、`0eba096b8`、`195830924`，range-diff 全部为 `=`，无冲突。
- 现已包含上游 `13570cf9d`（!46394）的 indexing / expand 2.15 参数兼容修复。
- 精确 lease 绑定旧远端 `5efe8a254`，推送成功；备份分支
  `codex/provenance-before-rebase-20260918-0954` 保留旧 HEAD。
- 公开 API 聚焦复验 PASS；完整原用例仍有相同 46 项本地 torch.* 既有失败，新增 0，
  目标 handler 未失败。只验证 Python 元数据，不是当前完整 HEAD 的本地构建或 2.15 验收。
- 已由用户账号发送 `compile`（评论 `190330352`，09:59:05）；机器人确认完整流水
  `PR-pipeline_pytorch#68280` 针对 `19583092` 运行中（评论 `190330600`）。
  run ID：`352c489602d544728778ef44b5ca8508`，pipeline ID 沿用
  `9ab4a8ad25bd4afeb808647efc5fc2f1`，projectId=4。截至 10:01 尚无最终结果。
- 共享环境、原有四个 dirty 子模块保持原样；没有修改 annotation、skip 或 allowlist。
- 本地文档同步；GitHub 文档仓仍未推送。
- 本轮证据：`Tracking/triton_experimental_delivery/pr_rebase_20260918.xErqjW/`。

下一步：检查 #68280 最终结果，尤其公开 API、provenance 端到端和 annotation；
不要把此前旧提交的失败当作本轮结果，也不要重复发送 compile。

## 2026-09-16 21:54 已授权并完成推送（历史）

- 用户明确同意推送，仅提交本 PR 的公开 API 元数据修复及对应测试。
- 当前 master 基线：`0095e028c54ccd86e83dcf3e9acccfca147e6084`。
- 当前本地 / 远端 PR HEAD：`5efe8a2541fcc5eab415d4ef4303949b85c61db3`。
- 当前三个 PR 提交：`0aab41b79` 功能、`7b5c6b20d` 文档截图、`5efe8a254` API 修复。
- 作者：`gcw_3ffySSwy <1305321851@qq.com>`。rebase 无冲突，三个 range-diff 均为 `=`。
- 推送使用精确 lease `dbc0db52fa3384575fd83cc4db23c15a298f802d`，成功更新既有 fork 分支。
- 备份：`codex/provenance-before-push-20260916-2148` 指向 rebase 前 `8acf294d0`。
- 推送前聚焦回归再次 PASS，源码 / 私有 runtime 文件哈希一致；仍不代表完整 HEAD 构建
  或远端 PyTorch 2.15 验收。未修改共享环境、indexing、旧 annotation 用例。
- 源码工作树只保留原有四个 dirty 子模块；其他用户修改未清理。
- PR API 确认 open / mergeable=true；21:54 最新可见 CI 评论仍为旧 HEAD 的 #67808。
  尚未确认新 HEAD 流水启动或结果，未代发 `compile`。不要把无冲突当成 CI / 审批通过。
- 本地报告和交接已更新；GitHub 文档仓有其他待提交修改，本次未提交 / 推送该仓。
- 证据：`Tracking/triton_experimental_delivery/pr_api_fix_20260916.ebZYKd/`
  的 `pre_push_metadata.log`、`push_record.json`。

下一步：查看新 HEAD 的 CI 状态；是否手动触发、是否修复基线问题需依用户新指示执行。

## 2026-09-16 20:49 仅本 PR 修复（历史）

用户随后授权“只修复本 PR 的问题”。现已在当前 `7b8795bb9` 工作树完成：

- `torch_npu/profiler/__init__.py` 增加一行 `inductor_trace_handler.__module__ = __name__`。
- `test/profiler/test_inductor_profiler.py` 增加公开导出 / 模块名 / pickle 同一性回归。
- 聚焦用例 FAIL → PASS；未修改的完整公开 API 检查仍 FAIL_EXISTING，条目 47 → 46，
  唯一消除本 PR handler，其余 46 个 torch.* 既有项不变、无新增项。
- 仅 Python 修复，独立 runtime 使用已构建 de350aa 原生 wheel，修复文件与源码哈希一致。
  不宣称完整 7b8795bb9 或远端 2.15 CI 已通过。
- 源码两个文件有未提交修改；原有四个子模块 dirty 状态保留；**未 commit/push**。
- 用户明确排除后端 indexing 兼容与 CI 旧 annotation 测试，本轮没有修改。

最新文档：`/home/z50063656/TorchNpu-Inductor-Provenance/docs/pr_46073_api_fix.md`。
证据：`Tracking/triton_experimental_delivery/pr_api_fix_20260916.ebZYKd/`。
完整报告：`Tracking/issues/test_correct_module_names_inductor_trace_handler/`。
下方 20:07 的“未授权修复”仅是历史阶段；本节覆盖该状态，但仍未获推送授权。

## 2026-09-16 PR !46073 最新状态（优先于下方历史记录）

用户最新要求：先 rebase，确认失败由什么引入；**本轮未授权实施修复或推送**。

- 正式源码工作树仍为 `Tracking/worktrees/torch_npu_triton_provenance_delivery`。
- 第二次 fetch/rebase 的 master：`2a9b559fb237dde85fce6bb72cba931fd6d38548`。
- 本地 HEAD：`7b8795bb9045e83ce293cfb682eb98aab1fd3a8f`，功能提交 `52cf72e6c`。
- 两个提交 range-diff 均为 `=`，无冲突、未推送；原工作树四个子模块 dirty 状态保留。
- 远端失败流水 #67808 是原 PR HEAD `dbc0db52` 加 CI merge，不能算作上述新 HEAD 的测试。
- 公开 API 导出缺陷来自本 PR；indexing 参数兼容缺口由上游 #192410 新接口与 NPU 旧签名
  共同触发；annotation overlap 是上游 #193042 删除旧行为后，CI 仍运行旧测试的错配。
- 已下载 CI 同版 2.15 wheel **只提取源码、未安装**，核对 git_version `3c73a854` 及
  两个 nightly 快照的祖先关系。三份 NPU indexing 源码完全相同，参数绑定均复现相同错误。
- 前一轮 `de350aa` 的 2.14 wheel 已构建完成，未计入第二次 rebase 或 CI 2.15 的回归。
- 共享环境未修改；本机为 aarch64，测试一律从 `/home/z50063656/tmp` 启动。

最新报告：`/home/z50063656/TorchNpu-Inductor-Provenance/docs/pr_46073_origin_analysis.md`。
本轮证据：`Tracking/triton_experimental_delivery/pr_origin_20260916.7sterg/`。
远端日志及上一轮构建：`Tracking/triton_experimental_delivery/pr_verify_20260916.b2PVBM/`。

下一步需按用户新指示决定是否修复；不要自动改代码、加 skip 或推送。下方 8 月状态仅供历史参考。

## 2026-08-27 需求变更后的主线（优先阅读）

本节覆盖下方旧的多后端“建议继续步骤”。当前需求以
`/home/z50063656/Tracking/需求变更.md` 为准：只负责
`torch_npu/_inductor/triton_experimental`，默认 NPU Inductor、MLIR、DVM、AKG、
CATLASS 和 torchair 均不属于当前交付范围。下方大段多后端记录仅作为历史研究保留，
不得再按其中的 AKG/multistream/FlexAttention 优先级继续开发。

正确项目和环境：

- 项目根：`/home/z50063656/Tracking`
- 环境入口：`/home/z50063656/Tracking/activate_tracking.sh`
- 测试启动目录：`/home/z50063656/tmp`
- PyTorch：`2.14.0a0+git8e86e0a`
- torch_npu 基线：`2.14.0a0+git83cc452`
- 设备：Ascend910B2；主要证据使用物理 NPU 7，最终 v10 wheel 的静态和
  forward/backward 复验因 NPU 7 被其他进程占满而改用空闲物理 NPU 4

仓库角色（2026-08-27 用户最终确认）：

- 官方目标仓：`https://gitcode.com/Ascend/pytorch`
- 开发 fork：`https://gitcode.com/gcw_3ffySSwy/pytorch`
- 当前交付 worktree 已配置 `origin` 指向官方仓、`fork` 指向开发 fork；正常交付应
  push 到 `fork`，再向 `origin` 发起 PR，不能直接把个人参考仓当作目标仓。
- `https://gitcode.com/rmch/npu_inductor_2.13.0` 是架构师个人预合入/历史参考仓，
  不是目标仓或开发 fork。
- `https://github.com/Beaver2323/TorchNpu-Inductor-Provenance` 是前期研究和文档仓，
  不是源码交付目标；其本地文档副本位于
  `/home/z50063656/TorchNpu-Inductor-Provenance`。
- `https://gitcode.com/AllenGuanC/inductor-meta-worktree` 只提供工作流程参考。

旧的 58 文件多后端实现继续保存在：

- 仓库：`/home/z50063656/Tracking/src/torch_npu`
- 分支：`codex/inductor-provenance`
- 本地提交：`3a0ff2ce2`

不要继续修改或整体清理该工作树。曾在 `Pass` 项目中形成的实验补丁只保存在
`/home/z50063656/Tracking/triton_experimental_migration/pass_experiment_reference.patch`
作为只读迁移参考，不能作为 Tracking 的测试证据。

### 当前正式交付工作树

- 路径：
  `/home/z50063656/Tracking/worktrees/torch_npu_triton_provenance_delivery`
- 分支：`codex/triton-experimental-provenance-delivery`
- 基线：`83cc452480c3546fd5cccf853bfe3a360ce9dbfc`
- 状态：产品代码、测试、probe 和真实证据已 staged；尚未 commit/push。
- 新手文档：
  `docs/inductor_provenance_demo/triton_experimental/README.md`

实现范围：

1. `triton_experimental/codegen/triton.py` 在普通 Triton launch 前登记 kernel
   provenance；rsplit partial/combine 两次 launch 分别登记。
2. `torch_npu/profiler/_inductor_profiler.py` 适配 NPU list/dict trace、尾置 flow、
   `torch_to_npu`、字符串时间戳以及 backward `k_*` 名称，并复用社区
   `_InductorTraceProcessor`。
3. `torch_npu.profiler.inductor_trace_handler` 作为 NPU runtime provenance 公开入口。
4. 测试覆盖静态 mapping、rsplit 发射顺序、NPU profiler schema、普通
   forward/backward 和真实 rsplit 双 kernel。

### 2026-08-27 wheel 构建与验收

为了避免其他进程修改活动环境，本轮没有安装进 Tracking conda site-packages；而是
完成 wheel 构建，并 `pip --target` 安装到隔离目录做真实回归。

- wheel：
  `/home/z50063656/Tracking/triton_experimental_delivery/wheels/torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl`
- SHA256：
  `fe8a90dec309a3d6089dd7807a56d3bc8a4f7f0bd886c755c5199f92237dc22d`
- 隔离安装：
  `/home/z50063656/Tracking/triton_experimental_delivery/wheel_target_20260827_v10`
- 完整构建日志：
  `/home/z50063656/Tracking/triton_experimental_delivery/wheel_build_20260827_v10.log`

wheel 压缩完整性通过，且以下三个 wheel 内文件与 staged 源码 SHA256 完全一致：

- `torch_npu/_inductor/triton_experimental/codegen/triton.py`
- `torch_npu/profiler/__init__.py`
- `torch_npu/profiler/_inductor_profiler.py`

构建使用项目原生 `DISABLE_INSTALL_TORCHAIR=TRUE`。原因是 torchair 子模块的
`configure` 强制使用 PyTorch 源码目录而非已安装 libtorch，导致唯一剩余构建目标
失败；torchair 不在当前需求范围。该 wheel 是功能验收 wheel，不是包含 torchair 的
完整发布 wheel。正式代码交付仍应为 source commit/PR。

wheel 级实测结果：

- profiler schema 单测：PASS，4/4；覆盖 list/dict trace root、尾置 flow、gzip、
  事件上限和编译期状态清理。
- rsplit 发射顺序单测：PASS，1/1。
- 真实 NPU 静态 mapping 单测：PASS；final v10 于 2026-08-27 收尾时在物理 NPU 7
  再次重跑 1/1，36.707 秒。
- 静态 probe：PASS，最大绝对误差 0，mapping 为
  `triton_unk_fused_add_mul_relu_0:1 -> add/relu/mul`。
- level 2 静态 probe：PASS；同一 kernel 仍映射到 `add/relu/mul`，最大绝对误差 0。
- tlparse：PASS，`Stats { ok: 137 }`。
- 普通 forward/backward timeline：PASS；`triton_unk_*` forward 与真实 `k_*`
  backward device kernel 都有用户源码 stack，且原始 NPU schema 无临时字段泄漏。
- rsplit timeline：PASS；`triton_unk_fused_mul_relu_sin_sum_0` 和 `_1` 两个
  device kernel 都有相同的正确来源 stack。

### 2026-08-27 AOTInductor 可行性门禁

已按“JIT 结果不能代替 AOTI”原则做真实 NPU 诊断，但没有把未跑通的原型留在正式
staged 产品代码中，最终 wheel 仍回退并校验为 v10 SHA256
`fe8a90dec309a3d6089dd7807a56d3bc8a4f7f0bd886c755c5199f92237dc22d`。

结论是当前环境不能验收 `triton_experimental` 的
`kernel_information.json`：

- 实验后端当前只注册 Python wrapper，AOTI C++ wrapper 尚未接入。
- 诊断原型依次补入 C++ wrapper、NPU AOTI 基础 hooks 和 Triton 二进制缓存发布后，
  已从早期 codegen 错误推进到最终 C++ 编译。
- 最终 C++ 编译显示 PyTorch 2.14 生成代码与当前 wheel 内 NPU AOTI runtime header
  ABI 不匹配（model container、`did_call_load_constants`、
  `LazyKernelCompileResult`）。
- 同 wheel 默认 NPU AOTI 对照也失败于 PyTorch 2.14 lazy AOTI 的
  `KeyError: 'GridNpu'`，因此共享基线本身尚未形成可用对照。
- 仓库 `torch_npu/_inductor/docs/feature/aoti/overview.md` 只声明 Atlas A5 支持，
  当前物理设备是 Ascend910B2。

诊断日志保存在
`/home/z50063656/Tracking/triton_experimental_delivery/wheel_aoti_npu7_20260827_v11*.log`
和 `default_aoti_npu7_20260827_v11.log`。诊断 v11 wheel 已归档为
`wheels/archive/torch_npu-2.14.0a0+git83cc452-v11-aoti-prototype-cp311-cp311-linux_aarch64.whl`
（SHA256 `a0614aa910b3a62b787ff539572868807fdfced475429fd2befdd2c320451793`），
不得作为通过件。

恢复 AOTI 工作的门禁顺序：A5 设备 -> 匹配的 PyTorch/torch_npu/AOTI header ->
默认 NPU 最小 `.pt2` 编译加载通过 -> 实验后端 C++ wrapper/二进制缓存接入 ->
`kernel_information.json` 结构、来源和加载数值验收。共享 runtime/default backend
不在当前需求范围，不能为了本任务直接扩线修改。

隔离证据根目录：
`/home/z50063656/Tracking/triton_experimental_delivery`。仓库内已保存可推送的
`provenance_tracking.html`、两份 Perfetto trace、对应 result JSON、mapping、stack
和三个 probe。

### 当前剩余步骤

1. 对 staged 文件运行最终 lint、语法检查和 diff 审查。
2. level 2、gzip、max-events、状态清理和 list/dict root 专项均已完成；后续不扩展到
   其他后端。
3. AOTInductor 已完成可行性门禁；只有满足上一节 A5/兼容基线条件后才恢复，且不可用
   JIT 结果代替 `kernel_information.json` 验收。
4. 用户授权后才创建 commit；另行获得远端与分支授权后再 push/建 PR。
5. 只有在环境独占窗口且用户明确要求时，才把最终 wheel 安装到长期 Tracking 环境。

## 历史多后端结论（仅供追溯）

普通 NPU Triton provenance 已在真实 Ascend 910B2 上完整跑通，包括源码改造、
torch_npu editable 构建、真实设备回归、demo、kernel/node/stack artifact 和
`tlparse 0.4.8` 三栏页面联动。此前怀疑的 tlparse 兼容性问题已经排除。P1 已进一步
完成真实 NPU FlexAttention forward template E2E，以及 FlexAttention dK/dV 四分支
wrapper 契约测试。2026-08-20 夜间继续修复了 PyTorch 2.14 新增的
`make_reduction(strict_sum=...)`、缺失 full-block metadata、scheduler 新 hook 和
template grid 兼容。默认 `BlockMask=None` forward 已在真实 NPU 完整跑通数值、
provenance 与 tlparse 三栏页面。2026-08-21 已继续建立默认 BlockMask backward 探针，
修复 dK/dV template 把 `1 << 30` 哨兵放大成 `8388608` 个 tile 的问题，并通过 33 项
契约回归；但 backward 仍在生成 `output_code.py` 之前卡于 BishengIR dK/dV 编译，
观察 11 分 48 秒仍未完成。因此 backward 数值、provenance 和 tlparse 不能标记为跑通。
这里特指 FlexAttention backward，不代表通用 backward 不可用。2026-08-24 已用
pointwise/reduction 模型在物理 NPU 2 跑通通用 AOTAutograd backward：forward/backward
独立编译单元、数值、Triton Ascend 执行、output-code handle 和 kernel stack 均通过，
梯度最大绝对误差为 `1.1920928955078125e-07`，tlparse 为 `Stats { ok: 220 }`。

该用例同时确认社区边界：standalone backward 没有 pre-grad graph，静态 mapping
artifact 只有 `{"version": 2.0}`，因此 backward 三栏页没有黄色 node 联动；源码 stack
和 handle 仍完整。通用 backward profiler timeline、普通 MLIR、DVM `mlir_fusion`、
DVM `graph_fusion`、DVM matmul template 与 CATLASS 双 kernel timeline 均已完成；
FlexAttention dK/dV 长编译继续作为独立专项隔离，后续再做 AKG 和 multistream 等专项覆盖。

2026-08-25 已完成 profiler timeline 第二阶段的最小 NPU forward/backward 链路。
torch_npu 新增专用 `inductor_trace_handler`：按 `(pid, tid, ts)` 关联文件尾部的
`torch_to_npu`/`fwdbwd` flow，在内存副本中适配社区处理器，再只把 `args.stack` 合并
回原始 Ascend trace。物理 NPU 6 实测 107 个事件、8 个 `torch_to_npu` endpoint；两个
forward Triton kernel 和一个 backward Triton kernel 均有用户源码 stack。最终文件
保持顶层 list、字符串时间戳和原始 flow，`ac2g=0`、内部 `uid=0`。

同日继续完成 extern/aclnn timeline。单个 `mm` 会被社区“单候选 stack”兜底掩盖，
所以最终用两个不同源码行的 `torch.mm` 做强验证。静态 key 是
`extern_kernels.mm:1/:2`，两个设备事件却同名
`aclnnMm_MatMulCommon_MatMulV2`。修复前双算子为 0/2 stack；NPU adapter 增加唯一
wrapper alias 后真实 NPU 7 达到 2/2，并分别指向 first/second 源码行，数值误差为 0。
最终 345 个事件、4 个 `torch_to_npu` endpoint，仍保持 `ac2g=0`、`uid=0`。

同日又完成普通 MLIR backend timeline 强验证。Tracking 环境补装 Ascend 官方
`aarch64/cp311 torch-mlir 0.0.1` wheel（`--no-deps`，未改动 torch/torch_npu）。一个
编译图生成 `mlir_fused_add_mul_sin_0:1` 与
`mlir_fused_add_cos_mul_1:2` 两个不同融合核。社区处理器只识别 `triton_*`，修复前真实
设备事件为 0/2 stack；torch_npu adapter 增加 `mlir_*` 精确名称识别后达到 2/2，且
分别指向 first/second 源码段。数值误差为 0，最终 363 个事件、4 个
`torch_to_npu` endpoint，`ac2g=0`、`uid=0`。单 MLIR kernel 会被社区单候选兜底掩盖，
因此本轮双核结果才是有效的强验证。

同日继续完成 DVM `mlir_fusion` timeline。环境门禁确认现有 Torch-MLIR wheel 和
torch_npu `_C.dvm` 扩展可直接工作，无需新增依赖。双核静态 key 为
`dvm_fused_add_mul_relu_0:1` 与 `dvm_fused_add_exp_mul_1:2`，真实 CANN device event
正好是去掉 handle 的同名 `dvm_*`。修复前为 0/2 stack；NPU adapter 加入 `dvm_*`
精确名称识别后为 2/2，分别指向 first/second 源码段。数值误差 0，最终 343 个事件、
4 个 `torch_to_npu` endpoint，`ac2g=0`、`uid=0`。本结论只覆盖 DVM
`mlir_fusion`，不包含 `graph_fusion`、template、dynamic shape 或 backward。

同日继续完成 DVM `graph_fusion` timeline。该路径是显式
`DvmGraphFusionPatch` post-grad pass，不是 `npu_backend=dvm`。修复前静态 registry 只有
`torch.ops.dvm.fused_graph_2_1.default:<handle>`，真实设备事件为
`DvmAddMaximumMul`/`DvmSubExpAdd`，且没有稳定逻辑 launch 可关联，因此为 0/2 stack。
修复后 codegen 生成 `dvm_graph_fused_0:2`/`dvm_graph_fused_1:4` provenance key，并通过
`k.set_kernel_info` 生成稳定 host launch 和 `torch_to_npu` flow；profiler adapter 沿 flow
临时使用逻辑名匹配、写入 stack 后恢复设备原名。最终 2/2 设备事件分别指向 first/second
用户源码，数值误差 0，343 个事件，`torch_to_npu=4`、`HostToDevice=4`、`ac2g=0`、
`uid=0`。范围仍不含 graph_fusion dynamic shape、backward、matmul/reduction 或多输出。

2026-08-25 夜间继续完成 CATLASS timeline。以隔离源码方式浅克隆官方 CATLASS
`v2.0.0`，通过 `TORCHINDUCTOR_NPU_CATLASS_DIR` 使用，没有向 conda 环境安装包。首次
真实编译已选中 CATLASS，但 CANN 9.0.1 BiSheng 链接报
`undefined symbol: g_opSystemRunCfg`；CATLASS 模板补入与 CANN op compiler 等价的 Atlas
A2 L2 cache runtime config 后，两个 `torch.mm` 均由 CATLASS 执行，最大误差
`2.288818359375e-05`。两个调用复用同一个底层 `KernelAdapter` mangled name，修复前为
0/2 stack；codegen 在 timeline 模式为每次 ctypes launch 生成完整
`catlass_fused_mm_0:<handle>` RecordFunction，profiler adapter 沿 flow 精确映射并恢复设备
原名后达到 2/2。最终 383 个事件，`torch_to_npu=4`、`HostToDevice=4`、`ac2g=0`、
`uid=0`。

2026-08-26 继续完成 DVM matmul template timeline。强 probe 在同一编译图中放置两个
shape、epilogue 完全相同但源码位置不同的 `torch.mm`；DVM 正常复用同一个 builder，
所以编译期虽登记 `dvm_fused_add_mm_mul_relu_0:1/:2`，两个同名设备事件却都被社区
`_stack_for_kernel()` 的首次命中规则回填为第一段源码。修复只在
`provenance_tracking_to_timeline` 调试模式关闭该 template source 复用，使两次 launch
分别命名为 `_0` 和 `_1`；普通模式仍保留原有复用和性能行为。最终 raw 0/2、processed
2/2 且源码栈不同，数值误差 0，344 个事件，`torch_to_npu=4`、`HostToDevice=4`、
`ac2g=0`、`uid=0`。该结论覆盖静态 A2 float16 `mm + add + relu + mul` 基线；
`bmm/addmm/baddbmm` 的 timeline、dynamic shape 与 backward 仍属于扩展覆盖。

2026-08-24 已把 CPU、普通 NPU Triton、FlexAttention forward、默认 BlockMask
forward、FX graph cache miss/hit 共 6 套 tlparse 演示发布到 GitCode fork 的
`codex/inductor-provenance-demo` 分支。upstream `Ascend/pytorch` 拒绝当前账号创建
分支，因此没有向 upstream 强推。纯演示提交为 `6c651b392`；包含实现、测试和演示的
完整本地提交为 `3a0ff2ce2`，目前仍未推送。通用 backward 是第 7 套本地演示，HTML
已加入仓库工作树，但尚未提交或推送。

PyTorch 已经编译、安装并可用，本轮没有重新编译 PyTorch。此前静态 provenance 修改
已完整构建并 editable 安装 torch_npu；本轮 timeline 是纯 Python 新增文件，因并发
环境清掉 PyTorch 源码树的 `torchgen/packaged`，完整 editable rebuild 在 codegen
metadata 阶段被阻塞。本轮已把两个 Python 文件精确同步到现有
`build/packages/torch_npu`，源码/运行副本 SHA256 一致并完成真实 NPU 验证。正式交付
前应在稳定构建树重新执行完整 torch_npu editable build。最新 graph_fusion 与 profiler
源码也已同步到 editable `build/packages`，两组 source/build SHA256 分别一致。

## 运行环境

- PyTorch：`2.14.0a0+git8e86e0a`
- torch_npu：`2.14.0a0+git83cc452`
- Torch-MLIR：`0.0.1`，Ascend 官方 `aarch64/cp311` wheel，以 `--no-deps` 安装
- CATLASS：官方 `v2.0.0` 源码，commit `769cd40a8716b28650b6bebb08db4834eea4462f`，
  通过 `TORCHINDUCTOR_NPU_CATLASS_DIR` 隔离引用，未 pip 安装
- torch_npu 安装方式：源码 editable install
- torch_npu 运行路径：
  `/home/z50063656/Tracking/src/torch_npu/build/packages/torch_npu/__init__.py`
- 设备：Ascend 910B2；早期验证使用物理 NPU 6，forward 最终回归与 P1 template 使用
  当时空闲的物理 NPU 5；2026-08-21 backward 调查改用物理 NPU 1，均映射为进程内
  设备 0；2026-08-24 通用 backward 使用当时空闲的物理 NPU 2。另一个分布式任务曾
  在调查中途占用物理 NPU 2～5，复现前必须重新检查设备
- 环境入口：`/home/z50063656/Tracking/activate_tracking.sh`
- 所有测试必须从 `/home/z50063656/tmp` 启动，不能在 torch_npu 源码目录内导入
  `torch`。

## 已完成工作

1. 已实现 NPU Inductor provenance 接入，涉及：
   - 普通 NPU Triton kernel
   - template kernel
   - combo kernel
   - FlexAttention dK/dV 路径
   - wrapper provenance 注释及 multistream
   - CATLASS、Meta/MLIR、DVM 路径
2. torch_npu 已成功构建并安装。
3. NPU provenance 测试已在真实设备执行，结果为：

   ```text
   Ran 3 tests in 27.723s
   OK
   ```

4. NPU demo 已成功运行：

   ```text
   torch=2.14.0a0+git8e86e0a
   torch_npu=2.14.0a0+git83cc452
   device=Ascend910B2
   checksum=9206.284180
   ```

5. provenance JSON 已正确建立同一个 NPU Triton fused kernel 与 post-grad
   `add`、`relu`、`mul` 节点的双向关系。
6. tlparse 成功解析 trace：`Stats { ok: 144 }`，并生成四个 HTML 页面。
7. 2026-08-20 最终复测再次通过：

   ```text
   Ran 3 tests in 27.289s
   OK
   Stats { ok: 139 }
   ```

8. 最终页面自动验收通过：

   ```text
   pre_post_highlight=pass
   kernel_post_highlight=pass {'132': [10, 7, 4]}
   post_kernel_highlight=pass {'10': [132], '4': [132], '7': [132]}
   ```

9. 新增 FlexAttention dK/dV wrapper 契约测试，验证 tasklist、no-split、reduce、
   legacy 四个候选 kernel 的 handle 均紧邻各自 `.run()`；该测试通过：

   ```text
   Ran 1 test in 0.028s
   OK
   ```

10. 真实 NPU FlexAttention forward template 已通过数值和 provenance 验证：

    ```text
    triton_flex_attention_fwd_mask_in:1
      -> flex_attention, sdpa_score0, sdpa_mask0
    checksum=-37.427276611328125
    ```

11. template trace 经 tlparse 0.4.8 解析成功：`Stats { ok: 282 }`。三栏页面中
    `pyCodeToPost={"608":[6,4,5]}`，第 608 行为真实 template `.run()`。
12. 首轮 `block_mask=None` 失败已定位为独立兼容问题：torch_npu 自定义
    `make_reduction` 未接收 PyTorch 2.14 的 `strict_sum` 参数。把公开 API
    `create_block_mask()` 放在编译图外后，template provenance 正常通过。
13. 新增契约测试后完整回归再次在真实 NPU 通过：

    ```text
    Ran 4 tests in 20.774s
    OK
    ```
14. 新增 combo kernel 调度契约测试，确认 provenance 使用 `combo_kernel_node.snodes`
    且发生在真实 `call_kernel` 之前；最新完整回归为：

    ```text
    Ran 5 tests in 19.817s
    OK
    ```

15. 跨进程 NPU FX Graph cache miss/hit 已通过。hit trace 明确包含
    `fx_graph_cache_hit`，miss/hit 的 mapping 与 stack JSON 均逐字节一致；hit 页面仍有
    `pyCodeToPost={"132":[10,7,4]}`。
16. `torch_npu._inductor.lowering.make_reduction` 已对齐上游 PyTorch 2.14：新增关键字
    参数 `strict_sum=False` 并透传 `Reduction.create`；源码与构建产物 SHA256 一致，
    运行时签名检查和新增契约单测均通过：

    ```text
    make_reduction(..., *, strict_sum: bool = False)
    Ran 1 test in 0.027s
    OK
    ```

17. 默认 `BlockMask=None` 的真实 NPU 重试确认原始 `strict_sum` 异常已消失，并到达
    FlexAttention mask-in template 候选编译。随后已完成 full metadata、scheduler 和
    template grid 三层修复，最终结果为：

    ```text
    checksum=-37.427276611328125
    triton_flex_attention_fwd_mask_in:12 -> 10 个 post-grad 来源节点
    tlparse: Stats { ok: 258 }
    ```

    tlparse 页面的最终调用位于右栏第 898 行，
    `pyCodeToPost["898"]=[58,4,27,69,52,49,46,45,55,5]`。
18. `transfer_to_npu` 不适用于本项目的原生 Triton/Inductor 验证：其
    `_patch_has_triton()` 固定返回 `False`，会导致 `Device npu not supported`。原生
    torch_npu Inductor 测试只需 `import torch_npu` 完成设备后端注册。
19. 最终正式 editable 构建成功，`lowering.py`、`flex_attention.py`、
    `flexattention_template.py`、`npu_combined_scheduling.py` 的源码/构建产物 SHA256
    均逐对一致；provenance、lowering、scheduler 三组完整回归通过：

    ```text
    Ran 32 tests in 20.724s
    OK
    ```
20. 已增加默认 BlockMask backward 探针并完成编译器分层调查：原始 dK/dV MLIR 中的
    `1073741824`/`8388608` 已通过 lowering 阶段稀疏倍数限界消除，修复后代表性
    `ttadapter` 为 189 行、25446 字节，只按真实 Q 长度 128 寻址。随后
    `bishengir-compile` 仍持续约 99.7% CPU，11 分 48 秒未返回；关闭 backward
    多消费者融合后 3 分 51 秒仍无改善。没有 backward `output_code.py`，说明阻塞在
    template 候选编译而非 provenance dump 或运行时。新增精确契约及完整回归为：

    ```text
    provenance: Ran 5 tests, OK
    lowering:   Ran 18 tests, OK
    scheduler:  Ran 10 tests, OK
    total:      33 tests, OK
    ```
21. 6 套 tlparse 演示已整理为 `docs/inductor_provenance_demo/`：149 个实际页面依赖
    文件、约 2.7 MiB，排除了 `raw.log`、`raw.jsonl`、payload、Inductor cache 和设备
    二进制。12 个入口/三栏页面的本地引用检查为 0 断链，HTTP 实测全部返回 200；
    GitCode hooks 通过，远端 SHA 与本地一致：

    ```text
    fork:   https://gitcode.com/gcw_3ffySSwy/pytorch.git
    branch: codex/inductor-provenance-demo
    commit: 6c651b392ca9d57482c3ee58d8eb583217783edf
    ```
22. 通用 NPU backward provenance E2E 已完成。模型包含 `sin`、`relu`、乘法和
    reduction，AOTAutograd 生成独立 forward/backward 编译单元；backward 实际执行：

    ```text
    triton_poi_fused_add_cos_expand_mul_relu_sin_t_0:1
    loss=2529.457275390625
    grad_checksum=3339.64013671875
    grad_max_abs_diff=1.1920928955078125e-07
    tlparse: Stats { ok: 220 }
    ```

    backward `output_code.py` 中 handle 紧邻 `.run()`，结构化 trace 中有 1 个非空
    backward stack kernel。社区 standalone backward mapping 只含 schema version，
    这是无 pre-grad graph 的上游行为，不是 NPU hook 缺失。第 7 套演示已复制到
    `docs/inductor_provenance_demo/html/npu_backward/`，排除了 raw trace、cache 和设备
    二进制，尚未提交或推送。
23. NPU profiler timeline provenance 最小 E2E 已完成。新增专用 handler 和合成
    forward/backward 单测；真实 Ascend trace 证明顶层为 list、`torch_to_npu` flow 位于
    文件尾、时间戳为字符串、设备事件无 `kernel` category。adapter 在副本中完成社区
    schema 转换，最终只合并 stack：

    ```text
    events=107
    torch_to_npu endpoints=8
    device Triton kernels with stack=3/3
    ac2g leaked to output=0
    uid leaked to output=0
    unit test: Ran 1 test, OK
    lintrunner: ok No lint issues.
    ```

    最终 trace 和说明已加入
    `docs/inductor_provenance_demo/timeline/`，尚未提交或推送。
24. extern/aclnn timeline 的重复同名 kernel E2E 已完成。真实生成代码含两个唯一
    wrapper `extern_kernels_extern_kernels_mm_0/_1`，而两个 CANN 设备事件都显示为
    `aclnnMm_MatMulCommon_MatMulV2`。adapter 按同一 extern base 内的 wrapper suffix 与
    debug handle 单调顺序建立临时 alias，再复用社区时间包围和 flow 处理：

    ```text
    raw device stacks=0/2
    processed device stacks=2/2
    first stack -> first = torch.mm(lhs0, rhs0)
    second stack -> second = torch.mm(lhs1, rhs1)
    max_abs_diff=0.0
    events=345, torch_to_npu=4, ac2g=0, uid=0
    unit test: Ran 2 tests, OK
    lintrunner: ok No lint issues.
    ```

    处理后 trace 已加入
    `docs/inductor_provenance_demo/timeline/npu_extern_mm_pair_trace.json`，说明见
    `npu_extern_provenance_timeline_demo.md`，尚未提交或推送。
25. 普通 MLIR backend 的双融合 kernel timeline E2E 已完成。单 kernel 会被社区
    “单候选 stack”兜底掩盖，所以 probe 在同一图中生成两个不同 shape 的 MLIR 融合核。
    社区处理器只识别 `triton_*`，修复前两个设备事件均没有 stack；NPU 处理器增加
    `mlir_*` 精确名称匹配后分别关联到不同用户源码段：

    ```text
    mlir_fused_add_mul_sin_0:1 -> first/first_product
    mlir_fused_add_cos_mul_1:2 -> second/second_sum
    raw device stacks=0/2
    processed device stacks=2/2
    max_abs_diff=0.0
    events=363, torch_to_npu=4, ac2g=0, uid=0
    unit test: Ran 3 tests, OK
    lintrunner: ok No lint issues.
    ```

    处理后 trace 已加入
    `docs/inductor_provenance_demo/timeline/npu_mlir_pair_trace.json`，完整依赖、原理、
    复现和阅读说明见 `npu_mlir_provenance_timeline_demo.md`。
26. DVM `mlir_fusion` 双融合 kernel timeline E2E 已完成。真实 device event 与静态
    key 去掉 handle 后完全同名，因此复用与 MLIR 相同的精确关联，不需要 wrapper alias：

    ```text
    dvm_fused_add_mul_relu_0:1 -> first/first_product
    dvm_fused_add_exp_mul_1:2 -> second/second_sum
    raw device stacks=0/2
    processed device stacks=2/2
    max_abs_diff=0.0
    events=343, torch_to_npu=4, ac2g=0, uid=0
    unit test: Ran 4 tests, OK
    lintrunner: ok No lint issues.
    ```

    处理后 trace 已加入
    `docs/inductor_provenance_demo/timeline/npu_dvm_pair_trace.json`，说明见
    `npu_dvm_provenance_timeline_demo.md`。
27. DVM `graph_fusion` 双 component timeline E2E 已完成。该路径的编译期逻辑名、host
    launch 和底层设备名分属三层，不能只靠前缀白名单：

    ```text
    dvm_graph_fused_0:2 -> host dvm_graph_fused_0 -> DvmAddMaximumMul
    dvm_graph_fused_1:4 -> host dvm_graph_fused_1 -> DvmSubExpAdd
    raw device stacks=0/2
    processed device stacks=2/2
    max_abs_diff=0.0
    events=343, torch_to_npu=4, HostToDevice=4, ac2g=0, uid=0
    profiler regression: Ran 5 tests, OK
    graph_fusion targeted: Ran 3 tests, OK
    lintrunner: ok No lint issues.
    ```

    `graph_fusion.py` 现在登记稳定逻辑 key 并发出 `set_kernel_info`；profiler adapter 沿
    flow 临时把逻辑名用于社区精确匹配，再恢复设备原名。处理后 trace 已加入
    `docs/inductor_provenance_demo/timeline/npu_dvm_graph_pair_trace.json`，详细说明见
    `npu_dvm_graph_fusion_provenance_timeline_demo.md`。
28. CATLASS 双 `mm` timeline E2E 已完成。官方 `v2.0.0` 的 `catlass_cppgen` 成功由
    torch_npu 直接导入，CANN 9.0.1 的 L2 cache runtime config 链接门槛已补齐：

    ```text
    static keys=catlass_fused_mm_0:1, catlass_fused_mm_0:2
    device names=同一个 Catlass::KernelAdapter<...BasicMatmulTla...> mangled name
    raw device stacks=0/2
    processed device stacks=2/2，且两条用户栈不同
    max_abs_diff=2.288818359375e-05
    events=383, torch_to_npu=4, HostToDevice=4, ac2g=0, uid=0
    CATLASS codegen targeted: Ran 2 tests, OK
    CATLASS profiler targeted: Ran 1 test, OK
    provenance full regression: Ran 7 tests, OK
    profiler full regression: Ran 6 tests, OK
    lintrunner: ok No lint issues.
    ```

    timeline codegen marker 使用完整 `kernel_name:debug_handle`，避免同一个生成 kernel
    被调用两次时都命中首个静态条目。处理后 trace 已加入
    `docs/inductor_provenance_demo/timeline/npu_catlass_mm_pair_trace.json`，详细说明见
    `npu_catlass_provenance_timeline_demo.md`。
29. DVM matmul template 的重复同构 `mm` timeline E2E 已完成。该强 probe 专门覆盖
    “同一个 template builder 被不同源码调用两次”的歧义场景：

    ```text
    baseline static keys=dvm_fused_add_mm_mul_relu_0:1, dvm_fused_add_mm_mul_relu_0:2
    baseline processed stacks=2/2，但两条都错误指向 first stack
    fixed static keys=dvm_fused_add_mm_mul_relu_0:1, dvm_fused_add_mm_mul_relu_1:2
    fixed device names=dvm_fused_add_mm_mul_relu_0, dvm_fused_add_mm_mul_relu_1
    raw device stacks=0/2
    processed device stacks=2/2，且分别指向 first/second
    max_abs_diff=0.0
    events=344, torch_to_npu=4, HostToDevice=4, ac2g=0, uid=0
    normal mm/bmm/addmm/baddbmm regression: Ran 4 tests, OK
    new targeted regression: Ran 1 test, OK
    provenance full regression: Ran 7 tests, OK
    lintrunner: ok No lint issues.
    ```

    根因是社区 `_stack_for_kernel()` 按 base name 检索静态 key 并在第一个结果处停止；
    按运行时出现次序分配 handle 会被多图和循环破坏，因此没有采用。当前修复仅在
    timeline 调试开关启用时强制每个 DVM template 调用生成唯一 kernel 名，普通执行仍
    复用源码。处理后 trace 已加入
    `docs/inductor_provenance_demo/timeline/npu_dvm_template_pair_trace.json`，详细说明见
    `npu_dvm_template_provenance_timeline_demo.md`。

## 主要源代码修改

- `/home/z50063656/Tracking/src/torch_npu/torch_npu/_inductor/codegen/scheduling.py`
- `/home/z50063656/Tracking/src/torch_npu/torch_npu/_inductor/codegen/wrapper.py`
- `/home/z50063656/Tracking/src/torch_npu/torch_npu/_inductor/codegen/catlass/catlass_scheduling.py`
- `/home/z50063656/Tracking/src/torch_npu/torch_npu/_inductor/codegen/catlass/catlass_template.py`
- `/home/z50063656/Tracking/src/torch_npu/torch_npu/_inductor/dvm/mlir_fusion.py`
- `/home/z50063656/Tracking/src/torch_npu/torch_npu/_inductor/dvm/graph_fusion.py`
- `/home/z50063656/Tracking/src/torch_npu/torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/npu/codegen/meta_kernel.py`
- `/home/z50063656/Tracking/src/torch_npu/torch_npu/_inductor/lowering.py`
- `/home/z50063656/Tracking/src/torch_npu/torch_npu/_inductor/kernel/flex_attention.py`
- `/home/z50063656/Tracking/src/torch_npu/torch_npu/_inductor/kernel/flexattention_template.py`
- `/home/z50063656/Tracking/src/torch_npu/torch_npu/_inductor/codegen/npu_combined_scheduling.py`
- `/home/z50063656/Tracking/src/torch_npu/test/_inductor/test_provenance_tracing.py`
- `/home/z50063656/Tracking/src/torch_npu/test/_inductor/test_lowering_device_dispatch.py`
- `/home/z50063656/Tracking/src/torch_npu/test/_inductor/test_scheduling_contract.py`
- `/home/z50063656/Tracking/src/torch_npu/torch_npu/profiler/_inductor_profiler.py`
- `/home/z50063656/Tracking/src/torch_npu/torch_npu/profiler/__init__.py`
- `/home/z50063656/Tracking/src/torch_npu/test/profiler/test_inductor_profiler.py`
- `/home/z50063656/Tracking/src/torch_npu/test/_inductor/test_dvm_graph_fusion.py`
- `/home/z50063656/Tracking/src/torch_npu/test/_inductor/test_dvm_mlir_fusion.py`

测试文件包含 `wrapper.comment = "#"`，修正使用 `object.__new__` 构造测试对象时缺少
实例属性的问题，并新增 FlexAttention dK/dV 四分支、combo、`strict_sum` 和 scheduler
调用位置断言。最新 P1 十二文件版本保存于 scoped `stash@{0}`（提交对象
`39677751f7cf15fce81903a58fc7adafcee81c5e`）；上一默认 BlockMask forward checkpoint
为 `stash@{1}`，combo/cache checkpoint 为 `stash@{2}`，template checkpoint 为
`stash@{3}`，P0 为 `stash@{4}`。最新 P1 内容已提交到本地
`codex/inductor-provenance` 的 `3a0ff2ce2`；stash 本身仍保留为安全副本。
editable 安装的 `build/packages/torch_npu` 与本轮九个实现源文件一致；对四个新增兼容
修复文件另做了 source/build SHA256 核对，结果全部相等。

## 关键产物

- 新手入门与源码导读：
  `/home/z50063656/Tracking/inductor_provenance_npu_beginner_guide.md`
- 通用 NPU backward 演示文档：
  `/home/z50063656/Tracking/npu_backward_provenance_demo.md`
- 通用 NPU backward 脚本与结果：
  `/home/z50063656/Tracking/npu_backward_provenance_demo.py`
  `/home/z50063656/Tracking/npu_backward_provenance_verified_20260824`
- 仓库内通用 backward 演示：
  `/home/z50063656/Tracking/src/torch_npu/docs/inductor_provenance_demo/html/npu_backward/index.html`
- 仓库内演示索引：
  `/home/z50063656/Tracking/src/torch_npu/docs/inductor_provenance_demo/README.md`
- NPU profiler timeline 演示文档：
  `/home/z50063656/Tracking/npu_provenance_timeline_demo.md`
- NPU profiler timeline probe：
  `/home/z50063656/Tracking/npu_provenance_timeline_probe.py`
- NPU profiler timeline 最终产物：
  `/home/z50063656/Tracking/npu_provenance_timeline_verified_20260825_v3`
- NPU extern/aclnn timeline 演示文档与 probe：
  `/home/z50063656/Tracking/npu_extern_provenance_timeline_demo.md`
  `/home/z50063656/Tracking/npu_extern_provenance_timeline_probe.py`
- NPU extern/aclnn timeline 最终产物：
  `/home/z50063656/Tracking/npu_extern_timeline_verified_20260825`
- NPU MLIR timeline 演示文档与 probe：
  `/home/z50063656/Tracking/npu_mlir_provenance_timeline_demo.md`
  `/home/z50063656/Tracking/npu_mlir_provenance_timeline_probe.py`
- NPU MLIR timeline 最终产物：
  `/home/z50063656/Tracking/npu_mlir_timeline_pair_fixed_20260825/artifacts`
- NPU DVM timeline 演示文档与 probe：
  `/home/z50063656/Tracking/npu_dvm_provenance_timeline_demo.md`
  `/home/z50063656/Tracking/npu_dvm_provenance_timeline_probe.py`
- NPU DVM timeline 最终产物：
  `/home/z50063656/Tracking/npu_dvm_timeline_pair_fixed_20260825/artifacts`
- NPU DVM graph_fusion timeline 演示文档与 probe：
  `/home/z50063656/Tracking/npu_dvm_graph_fusion_provenance_timeline_demo.md`
  `/home/z50063656/Tracking/npu_dvm_graph_fusion_provenance_timeline_probe.py`
- NPU DVM graph_fusion timeline 最终产物：
  `/home/z50063656/Tracking/npu_dvm_graph_timeline_fixed4_20260825/artifacts`
- NPU DVM matmul template timeline 演示文档与 probe：
  `/home/z50063656/Tracking/npu_dvm_template_provenance_timeline_demo.md`
  `/home/z50063656/Tracking/npu_dvm_template_provenance_timeline_probe.py`
- NPU DVM matmul template timeline baseline 与最终产物：
  `/home/z50063656/Tracking/npu_dvm_template_timeline_baseline_20260826`
  `/home/z50063656/Tracking/npu_dvm_template_timeline_fixed_20260826`
- NPU CATLASS timeline 演示文档与 probe：
  `/home/z50063656/Tracking/npu_catlass_provenance_timeline_demo.md`
  `/home/z50063656/Tracking/npu_catlass_provenance_timeline_probe.py`
- NPU CATLASS timeline 最终产物：
  `/home/z50063656/Tracking/npu_catlass_timeline_fixed3_20260825`
- 仓库内 Perfetto trace：
  `/home/z50063656/Tracking/src/torch_npu/docs/inductor_provenance_demo/timeline/npu_forward_backward_trace.json`
  `/home/z50063656/Tracking/src/torch_npu/docs/inductor_provenance_demo/timeline/npu_extern_mm_pair_trace.json`
  `/home/z50063656/Tracking/src/torch_npu/docs/inductor_provenance_demo/timeline/npu_mlir_pair_trace.json`
  `/home/z50063656/Tracking/src/torch_npu/docs/inductor_provenance_demo/timeline/npu_dvm_pair_trace.json`
  `/home/z50063656/Tracking/src/torch_npu/docs/inductor_provenance_demo/timeline/npu_dvm_graph_pair_trace.json`
  `/home/z50063656/Tracking/src/torch_npu/docs/inductor_provenance_demo/timeline/npu_catlass_mm_pair_trace.json`
  `/home/z50063656/Tracking/src/torch_npu/docs/inductor_provenance_demo/timeline/npu_dvm_template_pair_trace.json`
- GitCode 纯演示分支：
  `https://gitcode.com/gcw_3ffySSwy/pytorch/tree/codex/inductor-provenance-demo`
- NPU demo：`/home/z50063656/Tracking/npu_provenance_demo.py`
- demo 输出：`/home/z50063656/Tracking/npu_provenance_run/demo.log`
- trace：
  `/home/z50063656/Tracking/npu_provenance_run/trace/dedicated_log_torch_trace_k5286vfc.log`
- provenance mapping：
  `/home/z50063656/Tracking/npu_provenance_run/debug/torch_compile_debug/run_2026_08_20_19_01_49_534952-pid_2837624/torchinductor/model__0_inference_0.0/inductor_provenance_tracking_node_mappings.json`
- tlparse 主页面：
  `/home/z50063656/Tracking/npu_provenance_tlparse/index.html`
- 目标 provenance 页面：
  `/home/z50063656/Tracking/npu_provenance_tlparse/provenance_tracking_-_0_0_0.html`
- tlparse 解析后的 mapping：
  `/home/z50063656/Tracking/npu_provenance_tlparse/-_0_0_0/inductor_provenance_tracking_node_mappings_15.json`
- tlparse 解析后的 stack：
  `/home/z50063656/Tracking/npu_provenance_tlparse/-_0_0_0/inductor_provenance_tracking_kernel_stack_traces_16.json`
- 最终 NPU 演示文档：
  `/home/z50063656/Tracking/npu_provenance_visualization_demo.md`
- 最终独立运行目录：
  `/home/z50063656/Tracking/npu_provenance_verified_20260820`
- 最终 tlparse 页面：
  `/home/z50063656/Tracking/npu_provenance_verified_20260820/tlparse/provenance_tracking_-_0_0_0.html`
- 最终 tlparse mapping：
  `/home/z50063656/Tracking/npu_provenance_verified_20260820/tlparse/-_0_0_0/inductor_provenance_tracking_node_mappings_14.json`
- 最终 tlparse stack：
  `/home/z50063656/Tracking/npu_provenance_verified_20260820/tlparse/-_0_0_0/inductor_provenance_tracking_kernel_stack_traces_15.json`
- FlexAttention template 演示文档：
  `/home/z50063656/Tracking/npu_template_provenance_visualization_demo.md`
- FlexAttention template 运行脚本：
  `/home/z50063656/Tracking/npu_template_provenance_demo.py`
- FlexAttention template 最终产物：
  `/home/z50063656/Tracking/npu_template_provenance_verified_20260820`
- FlexAttention template 三栏页面：
  `/home/z50063656/Tracking/npu_template_provenance_verified_20260820/tlparse/provenance_tracking_-_0_0_0.html`
- 首次失败产物（`strict_sum` 兼容问题）：
  `/home/z50063656/Tracking/npu_template_provenance_p1_20260820`
- cache miss/hit 演示文档：
  `/home/z50063656/Tracking/npu_provenance_cache_hit_demo.md`
- cache miss/hit 完整产物：
  `/home/z50063656/Tracking/npu_provenance_cache_hit_verified_20260820`
- 默认 BlockMask 独立探针：
  `/home/z50063656/Tracking/npu_default_block_mask_provenance_probe.py`
- 默认 BlockMask 演示文档：
  `/home/z50063656/Tracking/npu_default_block_mask_provenance_demo.md`
- 默认 BlockMask 最终产物与 tlparse 页面：
  `/home/z50063656/Tracking/npu_default_block_mask_provenance_grid_verified_20260820`
- 默认 BlockMask backward 独立探针：
  `/home/z50063656/Tracking/npu_default_block_mask_backward_provenance_probe.py`
- backward 编译调查与复现说明：
  `/home/z50063656/Tracking/npu_default_block_mask_backward_investigation.md`
- 修复前超长编译 artifact：
  `/home/z50063656/Tracking/npu_default_block_mask_backward_device1_20260821`
- 修复后仍超长编译 artifact：
  `/home/z50063656/Tracking/npu_default_block_mask_backward_lowering_bound_20260821`
- 关闭 backward 多消费者融合的诊断 artifact：
  `/home/z50063656/Tracking/npu_default_block_mask_backward_no_hfusion_20260821`
- `strict_sum` 修复的成功构建日志：`/tmp/tracking_torch_npu_strict_sum_build.log`
- 默认 BlockMask 最终正式构建日志：
  `/tmp/tracking_torch_npu_default_mask_final_build.log`
- 最终 32 项回归日志：`/tmp/tracking_default_mask_full_contract_regression.log`
- backward 稀疏倍数修复最终构建日志：
  `/tmp/tracking_torch_npu_backward_sparse_bound_build_2.log`
- 完整构建失败日志：`/tmp/tracking_torch_npu_editable_build_retry.log`
- 成功的增量构建日志：`/tmp/tracking_torch_npu_editable_build_incremental.log`

## tlparse 调查结论

不存在 NPU 三栏联动缺陷。此前只查看了 AOT C++ 行号字段，遗漏了 JIT Python wrapper
字段。PyTorch 原始 artifact 使用历史名称 `cppCodeToPost`/`postToCppCode` 保存所有后端
kernel key；tlparse 转成页面行号时再拆分为：

```text
pyCodeToPost/postToPyCode    -> inductor_output_code Python wrapper
cppCodeToPost/postToCppCode  -> inductor_aot_wrapper_code C++ wrapper
```

本轮是 JIT Python wrapper，所以有效关系是
`pyCodeToPost = {"132": [10, 7, 4]}`；第 132 行就是 NPU Triton `.run()` 调用。
`tlparse/src/provenance.js::findCorrespondingLines()` 在 Python `codeData` 存在时读取该组
字段。AOT C++ 行号字段为空是正常结果，CPU JIT 页面也采用同一结构。

## 建议继续步骤

1. NPU profiler timeline 的普通 Triton forward/backward、重复 extern/aclnn、普通
   MLIR、DVM `mlir_fusion`/`graph_fusion`、DVM matmul template 和 CATLASS 双 `mm` 已完成。
   下一步优先覆盖 AKG，再做 multistream extern；DVM template 后续扩展
   `bmm/addmm/baddbmm`、dynamic shape 和 backward，CATLASS 后续扩展 dynamic shape、
   workspace、epilogue 和 backward，不与基础闭环混为一项。
2. provenance、lowering、scheduler 的 33 项回归和最终正式 editable 构建已完成；
   后续增加新后端用例时继续纳入该回归集合。
3. 独立固定一个 FlexAttention dK/dV 候选，保存完整 `bishengir-compile` 命令和分阶段
   IR，定位长耗时 pass；阻塞解除后完成 dQ/dK/dV 数值、stack、handle 和四分支 E2E。
   Standalone backward 的完整三栏 node mapping 不是当前社区 contract，不作为 NPU
   单方验收项。
4. CATLASS 与 DVM template 基础双 `mm` 已完成；下一块为 AKG 增加专项真实 NPU 用例，
   随后确认多流 extern 的实际事件名和 flow 结构。
5. cache hit 已完成；继续补 AOTI `kernel_information.json` 验证。
6. timeline adapter 已采用“torch_npu schema 适配 + 社区处理算法复用”，继续保持
   PyTorch 社区文件不出现 Ascend 私有名称。
7. 不需要为普通 NPU Triton/template 修改 tlparse 或移动现有 provenance 注释。

## Git 与构建注意事项

- 分支：`codex/inductor-provenance`
- 完整本地提交：`3a0ff2ce2`，包含 provenance 实现、测试和演示；由于本次远端授权
  仅覆盖演示内容，且 upstream 无建分支权限，该提交尚未推送。
- 纯演示分支：`codex/inductor-provenance-demo`，提交 `6c651b392`，已推送到
  `fork`（`https://gitcode.com/gcw_3ffySSwy/pytorch.git`）。
- `stash@{0}`：`codex: NPU provenance P1 backward compiler investigation`，最新十二文件
  checkpoint，提交对象 `39677751f7cf15fce81903a58fc7adafcee81c5e`；内容已进入本地
  提交 `3a0ff2ce2`，stash 继续保留。
- `stash@{1}`：`codex: NPU Inductor provenance P1 default BlockMask verified`。
- `stash@{2}`：`codex: NPU Inductor provenance P1 template combo cache verified`。
- `stash@{3}`：`codex: NPU Inductor provenance P1 template verified`。
- `stash@{4}`：`codex: NPU Inductor provenance verified final`，P0 安全副本。
- `stash@{5}`：`codex: NPU Inductor provenance adaptation WIP`。
- `stash@{6}`：`codex: NPU Inductor provenance checkpoint before build`。
- 工作树中有大量构建生成文件和第三方子模块状态，不要执行 `git clean`、
  `git reset --hard` 或整体回退。
- 2026-08-25 完整 torch_npu editable rebuild 两次在环境/构建树阶段失败：首次为
  PyTorch `tools` 包遮蔽 torch_npu namespace，临时 overlay 解除后又发现 PyTorch
  源码缺少生成的 `torchgen/packaged`。不要把这两个失败归因于 timeline 源码；正式
  构建前先恢复稳定 PyTorch build tree。当前运行副本已用纯 Python 精确同步验证。
- torchair 旧构建缓存已从源目录可恢复地移动到：
  `/tmp/tracking-torchair-build-cache-from-benchmark-20260820`
