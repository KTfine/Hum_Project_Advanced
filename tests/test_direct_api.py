"""Exercise the live client's direct provider transport without devices or a provider key."""
import asyncio
import base64
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from photo_assistant.api.session import DeepSeekClient, DirectSession, ENDPOINT
from photo_assistant.prompts import CO_PHOTO_COMMAND_PROMPT, OBSERVATION_MARKER, PHOTO_GUIDANCE_PROMPT


async def read_until(session, kind):
    async with asyncio.timeout(2):
        while True:
            event = await session.recv()
            if event["type"] == kind:
                return event


class DirectApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_observation_starts_only_after_a_goal_and_close_cancels_it(self):
        started, cancelled = asyncio.Event(), asyncio.Event()
        messages_seen = []

        async def complete(messages, *, on_delta):
            messages_seen.append(messages)
            if len(messages_seen) == 1:
                return SimpleNamespace(text="<speak:往左一点。>", metadata={})
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
        session.submit_frame(jpeg)
        session.submit_frame(jpeg, "指导我摆姿势")
        await read_until(session, "response.done")
        session.submit_frame(jpeg)
        self.assertEqual(len(messages_seen), 1)
        # Advance the observation clock without waiting on real model latency.
        with patch("photo_assistant.api.session.time", monotonic=lambda: float("inf")):
            session.submit_frame(jpeg)
        await asyncio.wait_for(started.wait(), 2)
        self.assertEqual([m["role"] for m in messages_seen[1]],
                         ["system", "user", "assistant", "user"])
        self.assertFalse(messages_seen[0][-1]["content"][0]["text"].startswith(OBSERVATION_MARKER))
        self.assertTrue(messages_seen[1][-1]["content"][0]["text"].startswith(OBSERVATION_MARKER))
        for template in (PHOTO_GUIDANCE_PROMPT, CO_PHOTO_COMMAND_PROMPT):
            self.assertIn(OBSERVATION_MARKER, template)
        await session.close()
        self.assertTrue(cancelled.is_set())
        provider.__aexit__.assert_awaited_once()

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

    async def test_cached_and_observed_frames_send_their_own_placement(self):
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
        initial = {"x": 0.63, "y": 0.87, "scale": 1.0}
        updated = {"x": 0.63, "y": 0.87, "scale": 1.5}
        session.submit_frame(first, placement=initial)
        session.submit_frame(None, "原地变大一点")
        await read_until(session, "response.done")
        with patch("photo_assistant.api.session.time", monotonic=lambda: float("inf")):
            session.submit_frame(next_frame, placement=updated)
        await read_until(session, "response.done")
        # A fresh frame without a character must clear the previous placement.
        session.submit_frame(first, "看这张")
        await read_until(session, "response.done")

        goal, observation, last = (messages[-1]["content"] for messages in requests)
        self.assertEqual(goal[0]["text"], "原地变大一点")
        self.assertTrue(observation[0]["text"].startswith(OBSERVATION_MARKER))
        for blocks, jpeg, expected in ((goal, first, initial), (observation, next_frame, updated)):
            self.assertEqual(blocks[1]["image_url"]["url"],
                             "data:image/jpeg;base64," + base64.b64encode(jpeg).decode())
            self.assertEqual(blocks[2]["type"], "text")
            prefix, data = blocks[2]["text"].split("：")
            self.assertEqual(prefix, "本帧角色脚点和倍率")
            self.assertEqual(json.loads(data), expected)
        self.assertEqual(len(last), 2)
        self.assertEqual(requests[-1][1]["content"], goal)
        self.assertEqual(requests[-1][3]["content"], observation)
