# 复现脚本

> 最后更新：2026-09-02 00:03 CST（UTC+08:00）

所有测试必须从 `/home/z50063656/tmp` 启动，避免在 torch_npu 源码树中导入 `torch`。

| 脚本 | 用途 |
| --- | --- |
| `static_probe.py` | 最小静态 provenance level 1/2 验证 |
| `timeline_probe.py` | 最小 forward/backward timeline 验证 |
| `rsplit_timeline_probe.py` | rsplit partial/combine 独立来源验证 |
| `llama_swiglu_demo.py` | Llama 风格动态形状、前反向、静态与 timeline 综合演示 |
| `provenance_ab_probe.py` | level 0/2 因果 A/B 边界验证 |
| `combo_provenance_probe.py` | ComboKernel level 0/1 可行性与 provenance A/B 门禁 |

完整环境变量和命令见上一级[交付说明](../README.md)。脚本输出目录必须不存在，由脚本
自行创建，避免旧产物污染本轮结果。
