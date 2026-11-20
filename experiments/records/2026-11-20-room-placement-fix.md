# 平视自习室的站位修正与双模式复测

2026-11-20，Asia/Shanghai。使用同机位、同人物的两张生成图，分别模拟人物站在木柜旁和玻璃隔断旁。输入图和此前测量方法见[首次测试](2026-11-20-room-interaction.md)。

## 指导模式

最终版本共发出 6 次真实图文请求，全部返回 HTTP 200，模型为 `deepseek-flash`，服务指纹为 `bc19ba97cbe016479903222bb1578db6`。本次运行的系统提示 SHA-256 为 `73b26f3c24e2cba6a87e56a96746eee966676dbe206c3dccc3ddb7c2fcbc3cc6`。六条回复均使用原生标签。另有两次没有待办的送帧，客户端直接略过。测试逐轮更换静态图，核查文字回复、观察时机和落盘照片。[六轮原始响应](../runtime/interaction-20261120/guidance-20261120-221902-910784/guidance-results.json) · [事件和各请求元数据](../runtime/interaction-20261120/guidance-20261120-221902-910784/guidance-events.jsonl)

| 要求或画面 | 实际回应与动作 |
|---|---|
| 询问身边有什么，想要一个小调整，先别拍 | 认出储物柜，建议“靠近柜子的那只手，轻轻搭在柜边上就好”。只给了一个动作。 |
| 纠正为不挪步、不靠家具，只调整上半身 | 回答“好，轻轻放下肩膀就行。” |
| 要求到玻璃隔断旁时说一声 | 登记 `<wait:feedback>`。人物仍在木柜旁时返回 `<blank>`。 |
| 换为人物已站到玻璃隔断旁的画面 | 回答“看到了，你已经到玻璃隔断旁边了。”待办结束，下一帧未再触发请求。 |
| 明确要求现在拍 | 回答“好，拍了。”并执行一次快门。保存的 PNG 与玻璃旁人物帧逐像素相同。[实际照片](../runtime/interaction-20261120/guidance-20261120-221902-910784/photos/photo-20261120-221939-026254-885e3dws.png) |

首次测试给了多项身体调整，还从图中推断光从右侧透入。第一轮修改去掉了光向判断，模型仍把肩、肘和手臂放进同一条建议。第二轮把两次建议各减到一个动作，但提示中的例子直接提到了测试场景的书架，模型也多次写出 `<speak:内容。</speak>`。最终提示使用通用的肩部例子。客户端另增加了混合闭合标签的流式回归测试，字幕只保留正文，适配记录可追溯。[首次回复](../runtime/interaction-20261120/guidance-20261120-215735-518335/guidance-results.json) · [第一轮修改](../runtime/interaction-20261120/guidance-20261120-221325-661422/guidance-results.json) · [第二轮修改](../runtime/interaction-20261120/guidance-20261120-221459-891698/guidance-results.json)

## 合照模式

同机位合照分四段独立会话，分别发出 8、3、2、4 次真实请求。加上指导模式 6 次，本轮最终复测共 23 次。使用当前客户端、实际流式解析与动作分发、同帧分割和 PNG 快门，回放分辨率为 640 × 480。四段合照会话的系统提示 SHA-256 均为 `9e4389b918edc58ef690607e060979e27c2f4da3579f137b6de780e07cc7465c`。四张合照快门文件分别与受检输出 PNG 的 SHA-256 相同。真实图文请求使用合成场景，历史真人照片仅在本机核对体型。

| 要求 | 画面和执行结果 |
|---|---|
| 用户入镜，再要求小白龙从肩后露脸 | 全身入场为并排正站。收到肩后请求后改为探身。立绘脸框内 910 个不透明像素全部可见，手描人物轮廓与立绘相交的 6,322 个像素中，96.3% 保留原人物。真人脸保持完整。[肩后画面](../runtime/interaction-fix-20261120/co-photo-20261120-222831-944907/behind.png) |
| 用户从左侧书柜移到右侧玻璃旁，再叫她跟过来 | 没有口令时保持脚点 (197, 421)，这个回合没有发起模型请求。收到指令后脚点移至 (389, 428)，与人物鞋底高度相差约 0.7 px，近处白桌处于角色身后，两张脸都可见。[完整会话与动作](../runtime/interaction-fix-20261120/co-photo-20261120-222831-944907/manifest.json) |
| 简单说“坐下来”，再指定近处白桌前沿 | 第一次就选近桌，切成支撑在桌面前沿、双腿下垂的坐姿。模型第二次确认时保持了已经正确的位置。支撑点距独立标注的桌沿约 2.8 px。[近桌坐姿](../runtime/interaction-fix-20261120/co-photo-20261120-222831-944907/seat-table.png) |
| 指定远桌，然后换到近桌 | 模型先发 `<sit:table=far>`，再发 `<sit:table=near>`。实测支撑点从 (319.6, 240.7) 移至 (318.4, 295.1)，角色的等效站高从画高约 42% 变为约 61%。远桌坐姿的下半身被近桌部分遮住，两次脸部均完整。只在最后要求拍摄时保存一张。[远桌](../runtime/interaction-fix-20261120/tables-20261120-222831-811420/far-table.png) · [近桌快门](../runtime/interaction-fix-20261120/tables-20261120-222831-811420/photos/photo-20261120-222839-305575-6o5ize3c.png) · [逐轮指令](../runtime/interaction-fix-20261120/tables-20261120-222831-811420/manifest.json) |
| 起立后去书柜空地，回来并排拍全身合照 | 客户端执行起立和指定落点，角色站在左边空地 (102, 422)。随后按指令回到人旁边，等到拍摄请求才保存。独立会话单独说“跟过来”也到达正确位置。[书柜空地](../runtime/interaction-fix-20261120/co-photo-20261120-222831-944907/destination.png) · [全身合照快门](../runtime/interaction-fix-20261120/standing-20261120-222831-867283/photos/photo-20261120-222839-434616-8ob5tdu5.png) |
| 保持半身肩后位置拍摄 | 入场与快门的脚点同为 (370, 512)，鞋底在画面下方 32 px，露脸区域完整。完整体型仍使用原标定。[半身合照快门](../runtime/interaction-fix-20261120/portrait-20261120-222833-668101/photos/photo-20261120-222838-872075-twiofvmr.png) |

本轮的前几次真实对话暴露了三个问题。按坐标猜近桌时会猜中远桌，现由前后深度选择。模型在选择远桌后有时会重复发送同一坐姿立绘，客户端现会保留已选桌面。还有一次拍照前的重复探身会让半身合照换边，现会先检查当前位置能否保住露脸和遮挡。最终四段合照会话的客户端均未跳过动作。独立全身会话入场时模型使用了异常语音闭合标签，实测终端显示了标签残片。按同一原始响应逐字符回放最终解析器后，输出为“好，我来。”，随后正常切换模式。[选桌问题发生时的原始会话](../runtime/interaction-fix-20261120/tables-20261120-222634-974483/manifest.json) · [最终独立全身会话](../runtime/interaction-fix-20261120/standing-20261120-222831-867283/manifest.json)

## 景深与体型

两张平视人物图均识别出前后两个桌组和六把椅子。构图和合成共用同一曝光的家具轮廓与相对逆深度。能看清桌椅支脚时，使用紧挨支脚的地面校正前后界。在空场景里，小白龙从白桌后移到桌前，等效站高从 212.21 px 增至 317.09 px，桌椅遮挡影响的角色像素由 7,891 降至 0。回到远处脚点后，成片逐像素恢复。本机也复测了最初的俯拍原图，其中连在一起的拼桌被识别成一个桌组。[平视前后对照](../runtime/interaction-fix-20261120/level-empty/comparison.png) · [原图测量](../runtime/interaction-fix-20261120/original-empty/measurements.json)

历史全身、半身、头肩合照的完整等效站高，修正前后依次均为 283.96、306.44 和 717.95 px。站姿脚点与地平线的距离决定前后移动的投影比例。坐姿另用立绘内的桌面支撑点，坐姿脚底不会用来重新标定身体大小。单目深度使用同帧相对尺度。[历史照片标定与测量](../runtime/interaction-20261120/standing-summary.md)

本轮使用固定机位的两张静态人物图和其中一张的半身裁切，验证了口令前后的最终位置与照片。连续走动、动态模糊和真实摄像头帧率仍需现场验证。Windows 全量回归共 174 项，168 项通过。4 个旧肩位几何用例与 2 个旧入场用例也在改动前的对照中失败。[本轮全量日志](../runtime/interaction-fix-20261120/verified-regression.log) · [原基线日志](../runtime/scene-unittest-base.log)

## 复现

从应用根目录执行，使用本机配置的 DeepSeek 凭据。各次会话使用新的带时间戳输出目录。两张测试图及手描对照线取自首次测试。

```powershell
$inputDir = 'experiments/runtime/interaction-20261120'
$outputDir = 'experiments/runtime/interaction-fix-20261120'
python -m experiments.scripts.guidance_interaction $inputDir
python -m experiments.scripts.co_photo_interaction "$inputDir/person-left.png" "$inputDir/person-right.png" $outputDir
python -m experiments.scripts.co_photo_interaction "$inputDir/person-left.png" "$inputDir/person-right.png" $outputDir --portrait-entry
python -m experiments.scripts.co_photo_interaction "$inputDir/person-left.png" "$inputDir/person-right.png" $outputDir --standing-pair
python -m experiments.scripts.co_photo_interaction "$inputDir/person-left.png" "$inputDir/person-right.png" $outputDir --table-choice
```

可打开各目录中的 `geometry-metrics.json` 复查投影与原始人物轮廓，`manifest.json` 保存当轮动作、模型响应和实际脚点。单目分割和投影均由本地模型产生。
