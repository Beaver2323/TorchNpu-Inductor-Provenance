> 归档时间：2026-09-18（CST）。历史迁移说明；错误项目补丁未纳入文档仓，不能作为正式验收证据。
>
> 原始来源：`Tracking/triton_experimental_migration/README.md`。

# triton_experimental 迁移暂存

`pass_experiment_reference.patch` 是 2026-08-26 误在独立 `Pass` 项目中形成的 15 文件参考补丁。

- SHA256：`ed48403feba79e290bb2594275df2dc946e60e4ad261ad2f7e6b2d1284f2c06c`
- 正确目标项目：`/home/z50063656/Tracking`
- 状态：仅供迁移审阅，不能作为 Tracking 的正式补丁或验收结果
- 原因：补丁中的文档路径、构建记录和 wheel 验收均来自错误的 Pass 项目环境

迁移时只能选择与 `triton_experimental` 需求相关的产品代码和测试，并必须使用
`/home/z50063656/Tracking/activate_tracking.sh` 重新构建、运行和生成证据。
