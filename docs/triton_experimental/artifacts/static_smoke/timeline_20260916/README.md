# 最小静态 HTML 与 Perfetto trace 配对演示

> 实测：2026-09-16 00:35 CST（UTC+08:00）

本次重跑与上一级静态演示相同的计算式 `relu(x + 1) * 2`，输入为 NPU 上的
4096 个 float32 元素。先完成编译和预热，再对同一个编译模型采样；下面两份产物来自
同一次实验。旧的 `../provenance_tracking.html` 不覆盖，配套阅读请使用本目录的新 HTML。

- [Perfetto trace：trace.json](./trace.json)：载入 Perfetto，选择设备事件
  `triton_unk_fused_add_mul_relu_0`，展开 `stack` 查看三处模型源码。
- [静态三栏 HTML](./provenance_tracking.html)：下载后用浏览器打开，点击节点或粗体 kernel 行。
- [静态节点映射](./node_mappings.json)：`triton_unk_fused_add_mul_relu_0:1` 对应 add/relu/mul。
- [本次结果与实际版本](./result.json)：不是 Perfetto 输入文件。

结果：PASS；与 eager 的最大绝对误差为 **0**，trace 共 **47** 个事件，包含 **1** 个
Triton 设备 kernel，带有 **3** 条模型源码栈，分别指向 add、relu、mul。
静态 key 的 `:1` 是 debug handle，运行事件名不带它；本次两者的 kernel 名匹配。

运行环境：Tracking Python、既有隔离 v10 wheel、物理 NPU 6（Ascend910B2）。
PyTorch 为 `2.14.0a0+git8e86e0a`，torch_npu 为 `2.14.0a0+git83cc452`，
Triton 自报版本为 `3.2.0`。没有安装依赖或修改共享环境。这是既有验收环境的
**推理前向**演示，不包含 backward，不是最新源码 PR 的全量回归或性能测试。

## 为什么本次 tlparse 有两个页面

原始输出同时包含 `provenance_tracking_-_0_0_0.html` 和
`provenance_tracking_-_-_-_-.html`。前者有本模型的 kernel handle 和三栏行号映射；
后者的 `lineMappings` 为空。脚本按静态 kernel handle 选择前者，保存为本目录的
`provenance_tracking.html`，不是单凭 HTML 数量判断成功与否。

第一次采样已通过数值和设备 stack 检查，但收尾时的“只能有一个 HTML”断言失败。
修正页面选择逻辑后重新运行，当前文件全部来自成功的第二次运行，没有混用两次结果。

## 复现与原始记录

脚本：[static_smoke_timeline_probe.py](../../../scripts/static_smoke_timeline_probe.py)。
使用匹配且含该功能的环境，从 `/home/z50063656/tmp` 启动，输出目录必须尚不存在：

```bash
cd /home/z50063656/tmp
python /home/z50063656/TorchNpu-Inductor-Provenance/docs/triton_experimental/scripts/static_smoke_timeline_probe.py \
  --output-dir /tmp/static_smoke_timeline_new \
  --tlparse /home/z50063656/.cargo/bin/tlparse
```

本次实际解释器为 `/home/z50063656/envs/Tracking/bin/python`，加载 CANN 9.0.1 后设置
`PYTHONPATH=/home/z50063656/Tracking/triton_experimental_delivery/wheel_target_20260827_v10`、
`ASCEND_RT_VISIBLE_DEVICES=6`，使用独立临时编译缓存。脚本自行在导入 torch 前设置
`TORCH_TRACE`，并在编译、采样和导出期间开启 provenance/timeline 配置。

原始日志：`/home/z50063656/Tracking/triton_experimental_delivery/static_smoke_timeline_20260916_r2.log`。
完整编译日志、debug 文件和 tlparse 页面保存在同名前缀的 `_r2/` 目录。
运行日志中的默认 profiler schedule 和 autotune 告警不作为性能结论；本次验收仅检查
数值、静态映射、设备 kernel stack 和配套页面。
