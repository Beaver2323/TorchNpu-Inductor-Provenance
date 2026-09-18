# 从小例子到模型模块：遇到了什么问题，解决到了哪一步

> 整理时间：2026-09-16（CST，UTC+08:00）。范围：Tracking 项目的 `triton_experimental` 后端。
> 本文总结的是 2026-08-28 的模块验证和 2026-08-29 的 Llama 演示记录，不代表今天重新跑过全部模块。

## 先用一句话说明这项工作

我们要让用户看见：**NPU 上执行的这段计算，到底来自模型里的哪几行代码。**

小例子只有加法、乘法，容易对应；换成模型模块后，一行代码可能被拆成多段计算，
多行代码也可能被合成一段，还会出现反向求梯度、输入尺寸变化等情况。因此，扩大验证
不只是“多跑几个模型”，还要确认来源没有漏掉、没有串到别的计算上。

这里说的 kernel，就是 NPU 实际执行的一段计算程序。验证用的是 Llama 风格计算块、
ConvNeXt 计算块和 Transformer 编码层，不是三个完整大模型的端到端训练。

## 1. 最先卡住的，可能不是追踪功能，而是模型本身编译不过

ConvNeXt 的反向计算遇到了后端生成错误代码的问题。例如，它生成过这样的赋值：

```python
x2 + 64*y3 = x2 + 64*y0 + 16384*y1
```

等号左边是一个表达式，不能被赋值，所以这段代码不能执行；后续备用路径又遇到了
NPU 不支持该内存排列方式的问题。

Transformer 则是第一种输入尺寸能做反向计算，换成第二种尺寸后，生成代码中控制
写入位置的条件与数据形状对不上，尝试的编译配置全部失败。

难点在于：这是原本就有的后端问题，还是加了来源追踪以后弄坏的？不能凭报错位置判断。
我们在同一环境、同一设备上禁用缓存，分别关闭和开启来源追踪做对照：两边都失败，
ConvNeXt 的出错 kernel 源码一致，Transformer 的出错代码片段也一致。

**处理结果：确认这两个具体失败不是来源追踪引起的，但没有在本任务里修好后端。**
ConvNeXt 反向和 Transformer 第二种尺寸的反向，都没有计入通过范围。
证据见[开关对照结果](./triton_experimental/artifacts/validation/provenance_ab_result.json)。

## 2. 计算确实执行了，但记录里的名字对不上

模型融合的计算越多，生成的 kernel 名字越长。Llama 演示中出现过：

```text
编译时：triton_unk_fused_add_div_expand_mul_pow_sum_view_4
运行时：riton_unk_fused_add_div_expand_mul_pow_sum_view_4
```

运行记录只保留了长名字末尾的 49 个字符，这个例子刚好丢掉开头的 `t`。
直接按名字查找，就会出现“时间线上有计算，但查不到它来自哪行代码”。

处理办法是用编译时登记的完整名字核对：只有能唯一确认时才补全匹配；如果两个不同
名字被截短后一样，就不猜。短名字别名也必须得到编译记录确认。

此外，Ascend 记录“CPU 发起计算、NPU 执行计算”的格式与社区处理器要求不同。
适配层在副本里转换格式、关联来源，最后只把来源调用栈写回原记录，保留原来的事件
名字和时间线格式。调用栈就是“在哪个文件、哪一行调用了这段计算”。

**处理结果：Llama 演示中检查到的 9 次 Triton 设备计算事件全部带上了来源调用栈，
包括截短名字的事件。** 这里是 9 次执行事件，不是 9 个不同 kernel，更不代表覆盖了
所有外部算子库的计算。
证据见[Llama 演示结果](./triton_experimental/artifacts/llama_swiglu/llama_swiglu_result.json)；
实现见 [名字匹配](./pr_diff_walkthrough.md#schema-kernel-name)和
[副本处理与写回](./pr_diff_walkthrough.md#schema-copy-back)。

## 3. 不光要“找得到来源”，还要防止“找错来源”

模型会有前向图、反向图；输入尺寸变化还可能触发重新编译。不同图里可能出现相同
kernel 名字。如果只保存一张“名字对应代码行”的总表，后一次编译就可能干扰前一次。

这部分是扩大验证时必须防住的风险，不能把它说成已经复现过的模型串图故障。
实现上复用社区按编译区域关联的逻辑；写回运行记录时，还核对事件名称、类型、进程、
线程和时间，不能只凭名字写回。

**处理结果：补充了“两个编译区域出现同名 kernel”和“动态尺寸重新编译后映射隔离”
的检查。** 这证明了对应测试场景没有串用来源，不等于所有并发、缓存和多次采集组合
都已经验证完。
对应源码是 [profiler 测试][profiler-tests]中的
`TestInductorProfiler.test_same_kernel_name_in_two_regions_keeps_distinct_stacks`
和[静态映射测试][static-tests]中的
`TestTritonExperimentalProvenance.test_dynamic_shape_recompile_keeps_mappings_isolated`。

## 4. 页面上没有高亮，不一定是 NPU 适配漏了

Llama 反向页面有一个实际例子：左栏的 `linear_2` 能对应到中栏的 `permute_2`，
但不能一路高亮到右栏真正执行矩阵乘法的代码。

原因是：社区 PyTorch 保存的反向图来源信息在这里没有连完整。中栏矩阵乘法 `mm`
到右栏 `extern_kernels.mm:8` 的关系存在，但缺少把它接回左栏 `linear_2` 的信息。
所以“中栏能找到执行代码”和“左栏每一行都能一路找到执行代码”是两种不同的覆盖范围。
时间线上有调用栈，也不能证明三栏页面里的每一条关系都完整。

**处理结果：保留真实关系，明确标注社区已有的限制，没有人为补一条看起来合理的连线。**
这符合本项目“与社区对齐”的要求，但不能写成“反向来源链全部补齐”。
原始证据见[Llama 映射 JSON](./triton_experimental/artifacts/llama_swiglu/llama_swiglu_node_mappings.json)。

展示时还遇到一个问题：这次 Llama 的前向、反向记录用了同一个可见编译编号，当前
tlparse 一次处理完整记录时，同名页面最终显示后面的反向图。交付时从同一份原始记录
分别保留了[前向页](./triton_experimental/artifacts/llama_swiglu/provenance_tracking_forward.html)
和[反向页](./triton_experimental/artifacts/llama_swiglu/provenance_tracking_backward.html)。
这是为了方便查看，不是额外实现了一套来源推断规则。

## 5. “能运行、能打开 HTML”还不够算通过

页面有内容，不代表计算结果正确；有来源文字，也不代表对应到了正确的模型代码。
因此，模块验证同时检查了两件事：

- 计算是否正确：编译前后输出是否接近；纳入反向验证的路径，还比较输入梯度和每个
  参数的梯度；采集时间线的那次运行也要比较。
- 追踪是否有效：静态映射不是空的；检查的设备计算带有调用栈；能找到预期模型代码；
  CPU 发起计算与 NPU 执行计算的记录对应；临时转换字段没有混入最终时间线。

操作上，先在采集范围外完成首次编译，再采集实际执行；每次使用新的输入，及时清空
参数梯度，并等待 NPU 完成计算，避免把梯度累积或尚未执行完的结果误判成追踪问题。

**处理结果：验收从“有文件”变成了“结果正确，而且文件里的来源关系可核对”。**
具体检查在[模块测试][model-tests]的
`TestInductorProvenanceModels._check_model()` 中。

## 最后到底验证到了哪里

下表依据[模块验证记录](./triton_experimental/artifacts/validation/model_validation_result.json)，
只描述记录中的环境和输入范围。

| 模块 | 已经验证通过 | 不能算已完成的部分 |
| --- | --- | --- |
| Llama 风格 RMSNorm + SwiGLU 计算块 | 两种尺寸的前向、输入及参数梯度；前向三栏映射；反向中栏到代码的映射；前反向时间线来源 | 反向左栏到右栏的来源链不是逐节点全覆盖，存在社区限制 |
| ConvNeXt 计算块 | 两种尺寸的前向；静态映射；推理时间线来源 | 反向被后端错误阻断 |
| Transformer 编码层 | 第一种尺寸的前向及输入、参数梯度；第二种尺寸的前向；静态映射；第一种尺寸的前反向时间线来源 | 第二种尺寸的反向被后端错误阻断 |

记录里的“3 个测试通过”，指三个**按上述范围编写的测试**通过，不是三个模块的所有
训练路径都通过。本次整理没有重新跑模块测试，也没有给出性能收益或完整大模型训练结论。

## 可以直接用于工作汇报的总结

将 NPU 来源追踪从简单算子扩展到三个代表性模型模块进行验证，主要解决了运行记录中
kernel 名称被截短、Ascend 记录格式与社区处理器不一致导致的来源关联问题，并增加了
同名 kernel、多图和动态尺寸下的关联检查。验证不只检查页面生成，还核对输出、梯度、
静态映射和运行时调用栈。同时通过关闭/开启追踪的对照实验，确认了两处后端反向失败
并非本功能引入，明确保留了社区反向来源链不完整的限制。最终形成了可演示、可核对、
也明确说明未覆盖范围的模块级验证资料。

## 想看代码时，从这里开始

代码链接固定到交付快照 `dbc0db52f`，避免后续分支变化影响核对。

- 名字补全：[torch_npu/profiler/_inductor_profiler.py][profiler-source] 的
  `_experimental_kernel_name()`、`_truncated_kernel_name_map()`。
- 记录转换与来源写回：同文件的 `_normalize_trace_for_inductor()`、
  `_add_inductor_provenance()`、`_copy_stacks_to_origin()`；
  [完整调用过程和 diff](./pr_diff_walkthrough.md#schema-conversion)有展开说明。
- 反向静态来源限制：社区 `torch/_inductor/debug.py::create_mapping_pre_post_grad_nodes()`
  按已有 `from_node` 信息建立关系；本项目的[交付指南第 10 节](./triton_experimental/README.md)
  结合上述 Llama 页面解释了为什么会缺一段关系。
- 验收范围：[模块测试][model-tests]以及上面的结构化结果，二者结合阅读，避免把未测路径算作通过。

[profiler-source]: https://gitcode.com/gcw_3ffySSwy/pytorch/blob/dbc0db52fa3384575fd83cc4db23c15a298f802d/torch_npu/profiler/_inductor_profiler.py
[profiler-tests]: https://gitcode.com/gcw_3ffySSwy/pytorch/blob/dbc0db52fa3384575fd83cc4db23c15a298f802d/test/profiler/test_inductor_profiler.py
[static-tests]: https://gitcode.com/gcw_3ffySSwy/pytorch/blob/dbc0db52fa3384575fd83cc4db23c15a298f802d/test/_inductor/test_triton_experimental_provenance.py
[model-tests]: https://gitcode.com/gcw_3ffySSwy/pytorch/blob/dbc0db52fa3384575fd83cc4db23c15a298f802d/test/profiler/test_inductor_provenance_models.py
