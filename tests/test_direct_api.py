"""Exercise the live client's direct provider transport without devices or a provider key."""
import asyncio
import base64
import json
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from photo_assistant.api.session import DeepSeekClient, DirectSession, ENDPOINT


async def read_until(session, kind):
    async with asyncio.timeout(2):
        while True:
            event = await session.recv()
            if event["type"] == kind:
                return event


class DirectApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_observations_keep_latest_action_or_speech_without_using_user_quota(self):
        calls = []
        move = "<teleport:x=0.4,y=0.8>"
        shot = "<capture><speak:拍好了。>"
        replies = iter(("<speak:好。>", move, "<blank>", shot, "<speak:是的。>"))

        async def complete(messages, *, on_delta):
            calls.append(messages)
            reply = next(replies)
            on_delta(reply)
            return SimpleNamespace(text=reply, metadata={})

        provider = AsyncMock()
        provider.complete.side_effect = complete
        session = DirectSession("co-photo", provider, max_turns=2)
        self.addAsyncCleanup(session.close)
        jpeg = b"\xff\xd8\xffA"
        self.assertFalse(session.can_observe)
        self.assertFalse(session.submit_observation(jpeg))
        session.submit_frame(jpeg, "看看我与窗户的位置。")
        self.assertFalse(session.can_observe)
        self.assertNotIn("turn_source", await read_until(session, "response.done"))
        original = session.handoff_messages()
        for i in range(3):
            self.assertTrue(session.can_observe)
            self.assertFalse(session.submit_observation(b""))
            self.assertTrue(session.submit_observation(jpeg, placement={"x": .25}))
            delta = await read_until(session, "response.output.delta")
            self.assertEqual(delta["turn_source"], "observation")
            # The provider has finished, but the frontend has not drained this response yet.
            self.assertFalse(session.can_observe)
            self.assertFalse(session.submit_observation(jpeg))
            done = await read_until(session, "response.done")
            self.assertEqual(done["turn_source"], "observation")
            self.assertEqual(done["response_id"], delta["response_id"])
            self.assertEqual(session.handoff_messages(), original + (
                {"role": "user", "content": "【定时画面观察】"},
                {"role": "assistant", "content": move if i < 2 else shot},
            ))
            blocks = calls[-1][-1]["content"]
            self.assertEqual(blocks[0]["text"], "【定时画面观察】")
            self.assertEqual(blocks[1]["type"], "image_url")
            self.assertIn('"x": 0.25', blocks[2]["text"])
            if i:
                self.assertEqual(calls[-1][-2]["content"], move)
        self.assertEqual(len(calls[2]), len(calls[3]))
        session.submit_frame(None, "刚才说的什么？")
        self.assertNotIn("turn_source", await read_until(session, "response.done"))
        self.assertEqual(calls[-1][-2], {"role": "assistant", "content": shot})
        self.assertEqual(session.handoff_messages()[0]["content"], "刚才说的什么？")
        await session.close()
        self.assertFalse(session.can_observe)
        self.assertFalse(session.submit_observation(jpeg))

    async def test_observation_mode_switch_keeps_real_user_goal_and_replaces_previous_observation(self):
        user_exchange = ({"role": "user", "content": "准备好以后和我合照。"},
                         {"role": "assistant", "content": "<speak:好。>"})
        marker = {"role": "user", "content": "【定时画面观察】"}
        previous_observation = (marker, {"role": "assistant", "content": "<speak:等你准备好。>"})
        previous = user_exchange + previous_observation
        provider = AsyncMock()
        mode_reply = "<mode:co-photo><stand>"
        provider.complete.return_value = SimpleNamespace(text=mode_reply, metadata={})
        session = DirectSession("guidance", provider, handoff_messages=previous)
        self.addAsyncCleanup(session.close)
        self.assertEqual(session.handoff_messages(), previous)
        self.assertTrue(session.submit_observation(b"\xff\xd8\xffB", placement={"x": .25}))
        await read_until(session, "response.done")
        messages, = provider.complete.await_args.args
        self.assertEqual(messages[1:-1], list(previous))
        self.assertEqual(session.handoff_messages(), user_exchange + (
            marker, {"role": "assistant", "content": mode_reply},
        ))

    async def test_user_interrupts_observation_started_from_mode_handoff(self):
        started, cancelled = asyncio.Event(), asyncio.Event()
        calls = []
        async def complete(messages, *, on_delta):
            calls.append(messages)
            if len(calls) == 1:
                on_delta("<speak:过时反馈>")
                started.set()
                try:
                    await asyncio.Future()
                finally:
                    cancelled.set()
            on_delta("<speak:好。>")
            return SimpleNamespace(text="<speak:好。>", metadata={})

        provider = AsyncMock()
        provider.complete.side_effect = complete
        handoff = ({"role": "user", "content": "过来和我合照。"},
                   {"role": "assistant", "content": "<mode:co-photo>"})
        session = DirectSession("co-photo", provider, handoff_messages=handoff)
        self.addAsyncCleanup(session.close)
        self.assertTrue(session.can_observe)
        self.assertTrue(session.submit_observation(b"\xff\xd8\xffB"))
        await asyncio.wait_for(started.wait(), 2)
        session.submit_frame(None, "先不说构图。")
        interrupted = await session.recv()
        self.assertEqual((interrupted["type"], interrupted["turn_source"]), ("response.interrupted", "observation"))
        delta = await session.recv()
        self.assertNotIn("turn_source", delta)
        self.assertEqual(delta["text"], "<speak:好。>")
        await read_until(session, "response.done")
        self.assertTrue(cancelled.is_set())
        self.assertEqual(calls[1][1:-1], list(handoff))

    async def test_close_cancels_inflight_request(self):
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def complete(messages, *, on_delta):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        provider = AsyncMock()
        provider.complete.side_effect = complete
        session = DirectSession("guidance", provider)
        self.addAsyncCleanup(session.close)
        jpeg = b"\xff\xd8\xffX"
        session.submit_frame(jpeg, "指导我摆姿势")
        await asyncio.wait_for(started.wait(), 2)
        await session.close()
        self.assertTrue(cancelled.is_set())
        provider.__aexit__.assert_awaited_once()

    async def test_new_mode_uses_last_exchange_only_on_next_user_input(self):
        provider = AsyncMock()
        provider.complete.return_value = SimpleNamespace(text="<capture>", metadata={})
        previous = ({"role": "user", "content": "和我合照。"},
                    {"role": "assistant", "content": "<mode:co-photo>"})
        session = DirectSession("co-photo", provider, handoff_messages=previous)
        self.addAsyncCleanup(session.close)
        jpeg = b"\xff\xd8\xffA"
        for text in (None, "", " "):
            session.submit_frame(jpeg, text)
        await asyncio.sleep(0)
        provider.complete.assert_not_awaited()
        self.assertEqual(session.handoff_messages(), previous)
        session.submit_frame(jpeg, "拍吧。")
        await read_until(session, "response.done")
        with patch("time.monotonic", return_value=time.monotonic() + 60):
            session.submit_frame(jpeg)
        await asyncio.sleep(0)
        provider.complete.assert_awaited_once()
        messages, = provider.complete.await_args.args
        self.assertEqual(messages[1:3], list(previous))
        self.assertEqual(session.handoff_messages(), (
            {"role": "user", "content": "拍吧。"}, {"role": "assistant", "content": "<capture>"},
        ))

    async def test_single_remote_request_streams_the_goal_and_latest_jpeg(self):
        requests = []

        def provider(request):
            self.assertEqual(str(request.url), ENDPOINT)
            self.assertEqual(request.headers["Authorization"], "Bearer test-only")
            requests.append(json.loads(request.content))
            chunks = ({"delta": {"content": "<mo"}},
                      {"delta": {"content": "de:co-photo>"}, "finish_reason": "stop"})
            body = "".join("data: " + json.dumps({"choices": [chunk]}) + "\n\n"
                           for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

        http = DeepSeekClient("test-only", transport=httpx.MockTransport(provider))
        session = DirectSession("guidance", http)
        self.addAsyncCleanup(session.close)
        first, latest = b"\xff\xd8\xffA", b"\xff\xd8\xffB"
        session.submit_frame(first)
        session.submit_frame(latest)
        await asyncio.sleep(0)
        self.assertEqual(requests, [])
        session.submit_frame(None, "和我合照")
        first_delta = await read_until(session, "response.output.delta")
        second_delta = await read_until(session, "response.output.delta")
        done = await read_until(session, "response.done")
        self.assertEqual([first_delta["text"], second_delta["text"]], ["<mo", "de:co-photo>"])
        self.assertEqual(first_delta["response_id"], done["response_id"])
        self.assertEqual(done["text"], "<mode:co-photo>")
        self.assertEqual(len(requests), 1)
        system, user = requests[0]["messages"]
        self.assertEqual(system["content"], "guidance")
        self.assertEqual(user["content"][0]["text"], "和我合照")
        self.assertEqual(user["content"][1]["image_url"]["url"],
                         "data:image/jpeg;base64," + base64.b64encode(latest).decode())

    async def test_new_goal_cancels_stream_drops_queued_output_and_keeps_user_history(self):
        started, cancelled = asyncio.Event(), asyncio.Event()
        calls = []

        class Provider:
            async def complete(self, messages, on_delta):
                calls.append(messages)
                if len(calls) == 1:
                    on_delta("<mode:co-photo>")
                    started.set()
                    try:
                        await asyncio.Future()
                    finally:
                        cancelled.set()
                on_delta("<speak:调整好了。>")
                return SimpleNamespace(text="<speak:调整好了。>", metadata={})

            async def __aexit__(self, *exc):
                pass

        session = DirectSession("guidance", Provider())
        self.addAsyncCleanup(session.close)
        session.submit_frame(None, "第一条要求")
        await asyncio.wait_for(started.wait(), 2)
        session.submit_frame(None, "换个姿势")
        self.assertEqual((await session.recv())["type"], "response.interrupted")
        event = await read_until(session, "response.output.delta")
        self.assertEqual(event["text"], "<speak:调整好了。>")
        await read_until(session, "response.done")
        self.assertTrue(cancelled.is_set())
        self.assertEqual([m["content"] for m in calls[1]],
                         ["guidance", "第一条要求", "换个姿势"])

    async def test_readiness_question_after_camera_loss_clears_cached_image(self):
        provider = AsyncMock()
        provider.complete.return_value = SimpleNamespace(text="<speak:我现在看不到画面。>", metadata={})
        session = DirectSession("co-photo", provider)
        self.addAsyncCleanup(session.close)
        session.submit_frame(b"\xff\xd8\xffA", placement={"avatar_visible": True})
        question = "你就位了吗"
        session.submit_frame(None, question, placement={"camera_frame_available": False, "avatar_visible": False})
        await read_until(session, "response.done")

        provider.complete.assert_awaited_once()
        messages, = provider.complete.await_args.args
        blocks = messages[-1]["content"]
        self.assertEqual([block["type"] for block in blocks], ["text", "text"])
        self.assertEqual(blocks[0]["text"], question)
        self.assertIn("没有可用的摄像头画面", blocks[1]["text"])
        self.assertEqual(session.handoff_messages()[0], {"role": "user", "content": question})

    async def test_each_user_request_sends_placement_for_its_own_frame(self):
        requests = []

        def provider(request):
            requests.append(json.loads(request.content)["messages"])
            event = {"choices": [{"delta": {"content": "<blank>"}, "finish_reason": "stop"}]}
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n")

        http = DeepSeekClient("test-only", transport=httpx.MockTransport(provider))
        session = DirectSession("co-photo", http)
        self.addAsyncCleanup(session.close)
        first, next_frame = b"\xff\xd8\xffA", b"\xff\xd8\xffB"
        initial = {"x": 0.63, "y": 0.87, "scale": 1.0, "standing_height_ratio": 0.62,
                   "scene_objects": [{"label": "dining table", "center_x": 0.22, "behind_y": 0.54},
                                     {"label": "dining table", "center_x": 0.68, "behind_y": 0.67}]}
        updated = {"x": 0.63, "y": 0.87, "scale": 1.5, "standing_height_ratio": 0.93,
                   "scene_objects": []}
        session.submit_frame(first, placement=initial)
        session.submit_frame(None, "原地变大一点")
        await read_until(session, "response.done")
        session.submit_frame(next_frame, placement=updated)
        session.submit_frame(None, "现在多大？")
        await read_until(session, "response.done")
        # A fresh frame without a character must clear the previous placement.
        session.submit_frame(first, "看这张")
        await read_until(session, "response.done")

        goal, followup, last = (messages[-1]["content"] for messages in requests)
        self.assertEqual(goal[0]["text"], "原地变大一点")
        self.assertEqual(followup[0]["text"], "现在多大？")
        for blocks, jpeg, expected in ((goal, first, initial), (followup, next_frame, updated)):
            self.assertEqual(blocks[1]["image_url"]["url"],
                             "data:image/jpeg;base64," + base64.b64encode(jpeg).decode())
            self.assertEqual(blocks[2]["type"], "text")
            prefix, data = blocks[2]["text"].split("：")
            self.assertEqual(prefix, "随图的角色与场景信息")
            self.assertEqual(json.loads(data), expected)
        self.assertEqual(len(last), 2)
        self.assertEqual(requests[-1][1]["content"], goal)
        self.assertEqual(requests[-1][3]["content"], followup)
