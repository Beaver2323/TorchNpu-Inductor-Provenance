> 归档时间：2026-09-18（CST）。已发布需求 issue #4909 正文快照：https://gitcode.com/Ascend/pytorch/issues/4909 。
>
> 原始来源：`Tracking/triton_experimental_delivery/pr_rebase_20260918.xErqjW/provenance_requirement_issue.md`。

# [需求] 为 triton_experimental 后端补齐 Inductor provenance 来源追踪

创建时间：2026-09-18（CST，UTC+08:00）。

对应实现：[PR !46073](https://gitcode.com/Ascend/pytorch/pull/46073)。

## 背景与目标

PyTorch 社区的 Inductor provenance tracking 可以关联输入图、post-grad 图和生成代码，
帮助用户理解编译融合后某个 kernel 来自哪些模型操作。
NPU 的 `triton_experimental` 有自己的 kernel 发射路径和 profiler trace 格式，
需要补齐来源登记及运行时 trace 适配，才能复用社区的可视化和来源分析能力。

本需求以 [PyTorch 官方 provenance 文档](https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_inductor_provenance.html)
的静态用法为基线，并结合匹配 PyTorch 版本的 timeline 处理器，
让用户在自己的模型上使用标准配置、tlparse 和 NPU profiler 获取来源关系，
不依赖开发者的本地演示脚本。

## 需求范围

仅覆盖 `torch_npu/_inductor/triton_experimental` 后端。

1. **静态来源映射与可视化**
   - 在普通 Triton kernel launch 前登记 provenance debug handle。
   - 支持社区 `INDUCTOR_PROVENANCE=1`、`TORCH_TRACE` 和
     `tlparse --inductor-provenance` 使用流程，复用社区 mapping schema 和三栏 HTML。
   - 在匹配版本上验证 provenance level 1 / 2；关闭时不额外生成相应 kernel 来源映射。
   - 覆盖前向、反向及动态形状重编译，避免不同编译图的来源映射混淆。
2. **NPU 运行时 timeline 来源回填**
   - 提供公开入口 `torch_npu.profiler.inductor_trace_handler`，配合
     `trace.provenance_tracking_to_timeline` 使用。
   - 适配 Ascend `torch_to_npu` flow、list/dict trace 根结构、尾置 flow 和字符串时间戳，
     临时归一化到社区处理器所需 schema，再将 kernel source stack 回填到原始 NPU trace。
   - 兼容运行时 `k_*` 名称及长 kernel 名截断；无法唯一判断来源时不能猜测并写入错误来源。
   - 保留原 trace 的必要字段和结构，使结果能用 Perfetto 等 Chrome trace 查看器读取。
3. **NPU 特殊发射路径**
   - 对 rsplit partial / combine 两次 launch 分别登记来源，并验证两个运行时 kernel。
   - 相同 kernel 名在不同编译区域出现时，各区域来源栈不能互相覆盖。
4. **公开接口、测试和文档交付**
   - 公开入口的导出列表、模块元数据、对象身份和 pickle 行为符合仓库规范。
   - 提供静态映射、trace 转换、forward/backward、rsplit 和代表性模型测试。
   - 中文文档包含独立使用示例、可用环境变量、与社区功能对齐矩阵，以及
     HTML、mapping JSON、timeline trace/result 和复现脚本。

## 验收标准

- [ ] 包含该功能的匹配环境中，用户可按文档对自己的模型生成 tlparse 三栏页面，
  点击覆盖的节点或 kernel 能查看正确来源关系。
- [ ] 前向 / backward 的 NPU kernel timeline 能回填来源栈；保留社区本身的映射边界，
  不为缺失的 pre-grad 来源人为合成关系。
- [ ] rsplit 两个运行时 kernel 均有来源登记；跨图同名、截断歧义、动态重编译等场景有回归检查。
- [ ] 相关数值 / 梯度验证与来源关系验证分开记录；SKIP、编译失败、历史验证均不冒充当前提交通过。
- [ ] 公开 API 检查、专项测试及 PR CI 对最终提交给出可追溯结果；文档和演示产物随 PR 交付。

## 不属于本需求的范围及已知边界

- 不扩展到默认 NPU Inductor、DVM、MLIR、AKG、CATLASS 或 torchair 后端。
- rsplit 双 launch 不等同于社区 ComboKernel。历史 ComboKernel 测试在 provenance
  关闭 / 开启时均被 NPU codegen 缺少 `x0/x0mask` 定义的问题阻断，不计为本需求已支持。
- 不将社区 C++ kernel 支持计为本轮 NPU Triton 支持；AOTInductor 编译、打包、加载和
  运行不在本轮已验收范围，extern 专项和 cache 一致性也不作为独立已通过能力。
- backward 输入图来源覆盖遵循社区 `from_node` 现有边界，不承诺每一行都有跨三栏高亮。
- 不借本需求扩大修复范围到无关的 PyTorch 基线或 CI 测试准备问题。

## 实现与验证入口

- [PR !46073](https://gitcode.com/Ascend/pytorch/pull/46073) 已提交实现，当前记录的 HEAD 为
  `195830924122a99e8f97bb85bb6be514d87af6f0`，基于 master `ec693356f`。
- [用户文档与对齐矩阵](https://gitcode.com/Ascend/pytorch/blob/195830924122a99e8f97bb85bb6be514d87af6f0/torch_npu/_inductor/triton_experimental/docs/inductor_provenance_demo/triton_experimental/README.md)。
- 测试：`test/_inductor/test_triton_experimental_provenance.py`、
  `test/profiler/test_inductor_profiler.py`、`test/profiler/test_inductor_provenance_models.py`。
- 历史 wheel 的 NPU 实测产物保留原版本、日期和设备信息；不能代替最新 rebase 提交的验收。
- 9 月 18 日 10:01 查询时，[完整流水 #68280](https://www.openlibing.com/apps/pipelineDetail?projectId=4&pipelineId=9ab4a8ad25bd4afeb808647efc5fc2f1&pipelineRunId=352c489602d544728778ef44b5ca8508&codeHostingPlatformFlag=gitcode)
  针对 `19583092` 运行中，最终结果以流水页面为准。本需求单不预先宣称全量验收通过。
