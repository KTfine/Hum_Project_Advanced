# 肩后探身时真人走开

2026-11-21，Asia/Shanghai。上一轮合照中，用户换边后只叫小白龙跟过来。她走到身旁正站，口头却说还要从肩后露头。这次更新了合照提示词：默认肩后探身在真人离开原位后由客户端自动转为原地站立，被问到姿势时参考最新画面和 `current_tachie`，收到跟随要求时描述本次动作。

## 两段指定图片会话

| 输入 | 客户端画面 | 模型实际答复 |
|---|---|---|
| 半身图自动探身入场，原人物位置连续三帧空出。随后用户问“你现在是什么姿势？” | `placement_source` 从“自动侧探”变为“原地站立”，脚点、倍率和标定站高均相同。 | “我现在是站着的，表情是高冷范儿，手自然垂在身侧，正着站。”只发了说话标签。 |
| 全身图问“你从我肩后探个头吧。” | 收到 `<peek>` 后到肩后探身。 | “好，我从你肩后探出来。” |
| 同一会话换到另一侧后只说“跟过来。” | 收到 `<follow>` 后在新人位旁并排正站。 | “来啦，站你旁边。” |

第一段覆盖客户端默认安排的探身。用户显式发起 `<peek>` 得到新的肩后落点，或另选立绘，均不保证离开三帧后自动转正。第二段检验的是用户开口后跟到身边。此次只修改合照 prompt 和说明，站位与离场检测沿用了现有实现。

## 复核条件

- 基于 `a9fafa9130bca914d739a998aefbb32ac79fe8cc` 后的当前工作区，产品修改在 `photo_assistant/prompts.py`。使用默认小白龙性格和立绘，两段实际发送的 system SHA-256 均为 `aa13bc0a65145a55d7b5e9edfdf588887f36027ebb72601cfe109fc4b79a42cc`。
- 摄像头回放使用 `interaction-20261120/person-left.png` 和 `person-right.png`。半身用同一裁切框，全身使用完整画面。每段独立开始，跟随组保留上一轮探身的对话和动作历史。
- 三次实际图文请求均 HTTP 200，返回 `deepseek-flash`，thinking 为 `low`，上限 4096 tokens，指纹 `bc19ba97cbe016479903222bb1578db6`。三次回复均含可解析的原生标签。测试使用固定机位的图片与文字输入。
- 本地回归 `python -m unittest tests.test_prompt tests.test_prompt_state_handoff tests.test_peek_feet tests.test_output_adaptation` 共 31 项通过。Windows 沙箱内的 asyncio 测试因本机回环连接阻塞而中止，完整结果来自沙箱外重跑。

原始证据保存在测试机的 `experiments/runtime/departure-prompt-20261121/`。本轮运行目录 `004413-276106/` 内，`departure/manifest.json` 记录自动站直的逐帧数据，`follow/manifest.json` 记录探头和随后跟随的原始回复。两段回放共用测试目录中的 `replay.py`。
