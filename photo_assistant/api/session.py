"""Stream photo-assistant turns directly to the provider from the running client."""
from __future__ import annotations

import asyncio
import json
import os
import uuid

from .client import ENDPOINT, ApiError, DeepSeekClient, Image
from photo_assistant.protocol import StreamingCommandParser


_OBSERVATION = ("这是定时送来的当前画面。沿用用户最近的拍摄目标，有新的、有用的反馈时用 <speak:内容> 简短说出，"
                "避免重复之前的建议。无需回应时输出 <blank>。")


async def open_session(
    system_prompt: str, *, api_key: str | None = None,
    handoff_messages: tuple[dict[str, str], ...] = (),
) -> DirectSession:
    """Open a new mode with its own prompt and the preceding exchange, if any."""
    client = DeepSeekClient(
        api_key=api_key,
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
        thinking=os.environ.get("DEEPSEEK_THINKING", "low"),
    )
    await client.__aenter__()
    return DirectSession(system_prompt, client, handoff_messages=handoff_messages)


class DirectSession:
    # The frontend can drain every delta before handing off to a new prompt.
    emits_response_done = True

    def __init__(self, system_prompt, client, *, max_turns=20, handoff_messages=()):
        self.session_id = uuid.uuid4().hex
        self._system = {"role": "system", "content": system_prompt}
        self._client = client
        self._max_turns = max_turns
        self._user_turns = 0
        self._handoff = tuple(handoff_messages)
        self._history = []
        self._last_feedback = ""
        self._latest_frame: tuple[bytes, dict[str, float | str] | None] | None = None
        self._events = asyncio.Queue()
        self._response_id = None
        self._active = None
        self._active_is_observation = False
        self._tasks = set()
        self._request_lock = asyncio.Lock()
        self._closed = False

    @property
    def can_observe(self) -> bool:
        return (not self._closed and self._response_id is None
                and (self._user_turns > 0 or any(m["role"] == "user" for m in self._handoff)))

    def submit_observation(self, jpeg: bytes, *, placement: dict | None = None) -> bool:
        """Offer a fresh camera frame without interrupting a conversation turn."""
        if not self.can_observe or not jpeg:
            return False
        if placement is not None and placement.get("camera_frame_available") is False:
            return False
        frame = (jpeg, None if placement is None else placement.copy())
        self._start_turn(_OBSERVATION, frame=frame, observation=True)
        self._latest_frame = frame
        return True

    def submit_frame(self, jpeg: bytes | None, text: str | None = None, *,
                     placement: dict | None = None) -> None:
        """Cache the latest frame and start a request when the user speaks or types."""
        if self._closed:
            raise RuntimeError("Photo session is closed")
        if jpeg is not None:
            self._latest_frame = (jpeg, None if placement is None else placement.copy())
        elif placement is not None and placement.get("camera_frame_available") is False:
            self._latest_frame = None
        if not text or not text.strip():
            return
        if self._response_id is not None:
            self._active.cancel()
            interrupted = {
                "type": "response.interrupted", "response_id": self._response_id, "reason": "new_input",
            }
            if self._active_is_observation:
                interrupted["turn_source"] = "observation"
            self._events.put_nowait((None, interrupted))
            self._response_id = None
        if self._user_turns >= self._max_turns:
            self._events.put_nowait((None, {
                "type": "error",
                "error": {
                    "code": "history_limit",
                    "message": "Conversation history limit reached; start a new session.",
                },
            }))
            return

        no_camera = placement is not None and placement.get("camera_frame_available") is False
        self._start_turn(text, frame=self._latest_frame, no_camera=no_camera)

    def _start_turn(self, text, *, frame, observation=False, no_camera=False):
        content = text
        if frame is not None:
            image, frame_placement = frame
            content = [{"type": "text", "text": text}, Image(image).block()]
            if frame_placement is not None:
                content.append({"type": "text", "text": "随图的角色与场景信息：" + json.dumps(frame_placement, ensure_ascii=False)})
        elif no_camera:
            content = [{"type": "text", "text": text}, {"type": "text", "text": "当前没有可用的摄像头画面。"}]
        user_message = {"role": "user", "content": content}
        messages = [self._system, *self._handoff, *self._history]
        if self._last_feedback:
            messages.extend((
                {"role": "user", "content": "【上一次画面观察】"},
                {"role": "assistant", "content": self._last_feedback},
            ))
        messages.append(user_message)
        if not observation:
            # Keep the user's goal even when its answer is interrupted.
            self._history.append(user_message)
            self._user_turns += 1
            self._last_feedback = ""
        self._active_is_observation = observation
        self._response_id = uuid.uuid4().hex
        self._active = asyncio.create_task(self._complete(messages, self._response_id, observation))
        self._tasks.add(self._active)
        self._active.add_done_callback(self._tasks.discard)

    async def _complete(self, messages, response_id, observation):
        def emit(event):
            if not self._closed and response_id == self._response_id:
                if observation:
                    event["turn_source"] = "observation"
                self._events.put_nowait((response_id, {**event, "response_id": response_id}))

        def delta(chunk):
            emit({"type": "response.output.delta", "kind": "text", "text": chunk})

        try:
            # A replacement waits until the previous HTTP stream releases its connection.
            async with self._request_lock:
                if self._closed or response_id != self._response_id:
                    return
                result = await self._client.complete(messages, on_delta=delta)
        except Exception as exc:
            if self._closed or response_id != self._response_id:
                return
            detail = type(exc).__name__
            if isinstance(exc, ApiError):
                status = exc.result.metadata.get("http_status")
                detail = f"HTTP {status}" if status else str(exc)
            emit({"type": "error", "error": {
                "code": "provider_error", "message": f"DeepSeek API request failed ({detail})",
            }})
            return
        if self._closed or response_id != self._response_id:
            return
        if not observation:
            self._history.append({"role": "assistant", "content": result.text})
        emit({"type": "response.done", "text": result.text, "metadata": result.metadata})

    def handoff_messages(self) -> tuple[dict[str, str], ...]:
        """Keep the last user turn and reply as history, without images or old frame coordinates."""
        last_user = next((i for i in range(len(self._history) - 1, -1, -1)
                          if self._history[i]["role"] == "user"), None)
        if last_user is None:
            return self._handoff
        messages = []
        for message in self._history[last_user:]:
            content = message["content"]
            # submit_frame stores the utterance in the first block, before the camera data.
            text = content if isinstance(content, str) else content[0]["text"]
            messages.append({"role": message["role"], "content": text})
        return tuple(messages)

    async def recv(self) -> dict:
        while True:
            response_id, event = await self._events.get()
            if response_id is None or response_id == self._response_id:
                if event["type"] == "response.done" and event.get("turn_source") == "observation":
                    commands = StreamingCommandParser(allow_plain_speech=True).feed(response_id, event["text"])
                    feedback = "".join(command.params["text"] for command in commands if command.kind == "speak")
                    if feedback:
                        self._last_feedback = feedback
                if event["type"] in {"response.done", "error"} and response_id == self._response_id:
                    self._response_id = None
                return event

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._response_id = None
        self._events.put_nowait((None, {"type": "session.closed", "reason": "client_stop"}))
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await self._client.__aexit__(None, None, None)
