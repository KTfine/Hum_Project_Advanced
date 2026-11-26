# 尖括号与 JSON 指令格式对照

日期：2026-11-26（Asia/Shanghai）

## 结论

在 30 个冻结请求、60 次真实 DeepSeek Flash 独立调用中，项目的尖括号协议显著减少了模型输出 token，同时保持了与 JSON 相同的完整指令语义准确率。

- 双方都完全正确的 27 个配对中，尖括号平均 14.67 completion tokens，JSON 平均 23.70，尖括号节省 **38.13%**。
- 每个配对的 `JSON − 尖括号` 平均为 9.04 token；10,000 次配对 bootstrap 的 95% CI 为 **[8.33, 9.81]**，不跨 0。
- 两种格式的完整指令序列准确率均为 **28/30（93.33%）**。
- 尖括号严格解析率为 28/30，JSON 为 30/30。因此本实验支持“更短且任务正确率相同”，不支持“尖括号的格式解析率更高”。

## 方法

- 30 个中文请求分为 single、parameterized、speech、tachie、multi 五类，每类 6 条。
- 每条请求分别要求尖括号和紧凑 JSON；奇偶样本交换调用顺序。
- 模型：`deepseek-flash`；thinking off；服务指纹 `bc19ba97cbe016479903222bb1578db6`。
- 两边提供相同动作集合、参数规则和立绘字段枚举；JSON 使用常规 `{"commands":[{"type":"..."}]}`，没有刻意添加冗余包装。
- completion tokens 直接读取 API `usage.completion_tokens`，不是本地估算。
- 严格解析后统一规范化为有序 `{kind, params}` 序列，与冻结预期自动精确比较；不使用人工或 LLM judge。
- 失败保留在30条可靠性分母中。`<sit>` 与文档规定等价的 `<sit:distance=near>`在语义比较中统一，其余错误不放宽。

## 汇总结果

| 格式 | 平均输出 token | 中位输出 token | 严格解析率 | 完整语义准确率 | 每个正确指令的 token 成本 |
|---|---:|---:|---:|---:|---:|
| 尖括号 | **15.90** | **11.0** | 93.33% | 93.33% | **12.89** |
| JSON | 24.67 | 21.0 | **100.00%** | 93.33% | 20.00 |

每100个 completion tokens，尖括号完成7.76个正确指令，JSON完成5.00个。

### 分类别结果

| 类别 | 尖括号正确/6 | JSON正确/6 | 尖括号平均输出 token | JSON平均输出 token |
|---|---:|---:|---:|---:|
| single | 6 | 6 | 4.00 | 11.00 |
| parameterized | 4 | 4 | 20.17 | 27.33 |
| speech | 6 | 6 | 9.50 | 17.83 |
| tachie | 6 | 6 | 29.50 | 38.50 |
| multi | 6 | 6 | 16.33 | 28.67 |

两种格式的错误都集中在 parameterized 类。尖括号有两条输出错误：一条重复传送且在标签间加入空格，一条把扶墙请求误写为无效传送与立绘组合。JSON有两条语义错误：把最远座位写成 `(0,0)` 坐标，以及把扶墙写成立绘加传送；两条 JSON 都能解析，但执行语义不正确。

## 输入 token 与适用边界

本实验的预定主指标是输出 token。为保证两种格式的协议约束完整，尖括号系统提示在本次独立请求中平均产生 308.87 prompt tokens，JSON 为 285.87；连同输出后的平均 total tokens 分别是 324.77 和 310.53。因此不能用本实验声称“每次独立请求的总计费 token 更少”。

尖括号的优势体现在输出带宽、生成长度、流式闭合和每个正确执行指令的输出成本。在真实长会话中系统提示可以被历史复用或缓存，但本实验没有测量缓存后的总费用，也没有发现可归因于格式的延迟优势（平均约 0.81 秒，基本相同）。结论只适用于当前模型、提示和动作集合。

## 方法修正

第一次 pilot 使用了容易被误读的 `speak:text` 写法、遗漏了立绘合法枚举，并把文档明确等价的近座位写法判为不同。pilot 中双方正确的16对已显示45.4%的输出 token 节省，但可靠性比较受到上述设计缺陷影响。修正先通过回归测试固定，然后删除 pilot 正式目录并从零执行本记录的60次调用；pilot 不进入最终统计。

## 证据

- 运行目录：`D:\AI_cache\hum_project_evaluation\runs\20261126-command-format-formal`
- manifest SHA-256：`d9635d80aedf40d0fd806d2d76785bd679e39921f96763a7cef5370138e992b6`
- predictions JSONL SHA-256：`67a932bf3e038a081cf5ae33ccd6ae99af8b46b1e7933e7d91d5710e4afac599`
- metrics SHA-256：`bd7a6802681dfc40c16b4b66a90344b84959f329629a39894f81726fe019c967`
- 实验设计：[2026-11-26-command-format-design.md](../plans/2026-11-26-command-format-design.md)
- 实验脚本：[command_format_comparison.py](../scripts/command_format_comparison.py)

## 既有人体理解案例图片

按自动规则保留了 activity、pose、emotion 各2个成功与2个失败案例，共12张图片及索引，总计约0.67 MiB。activity 与 emotion 使用八帧联系图；pose 使用单帧骨架叠加图。分类案例按冻结结果的正确/错误和样本ID选择；pose按 PCK@0.10 最高/最低选择，没有人工挑图。

四张 pose 图固定取八个等距采样帧中的第5帧（slot 4），并叠加两套13关节骨架：绿色为 Penn Action ground truth，洋红色为 DeepSeek 预测。固定帧位避免人工挑选最好看的帧。高分图展示两套骨架大致重合；低分图仍可看到完整、结构合理的 DeepSeek 人体骨架，但关键点与 ground truth 存在明显位置偏差，用于说明通用模型能够预测 pose，只是定位精度不及专用模型。

[案例索引与图片](../examples/2026-11-26-human-understanding/README.md) · [机器可读选择清单](../examples/2026-11-26-human-understanding/manifest.json)
