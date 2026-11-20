"""Stream photo-assistant turns directly to the provider from the running client."""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

from .client import ENDPOINT, ApiError, DeepSeekClient, Image
from photo_assistant.prompts import OBSERVATION_MARKER, OBSERVATION_MESSAGE
from photo_assistant.protocol import CHARACTER_ACTIONS, StreamingCommandParser


_OBSERVATION_TASKS = {
    "feedback": OBSERVATION_MARKER + "只检查用户原话委托的那项调整。条件还未达到就输出 <blank>；"
                "看清变化后给一次具体反馈，或完成切换模式前用户指定的动作。此轮没有快门任务。",
    "capture": OBSERVATION_MARKER + "只检查用户为这张照片指定的条件。以用户原话为准。"
               "尚未到位时输出 <blank>，到位时输出一次 <capture>。等待期间保持安静。",
}


async def open_session(
    system_prompt: str, *, api_key: str | None = None, observe_on_start: bool = False,
    handoff_messages: tuple[dict[str, str], ...] = (),
    managed_observations: bool = False, pending_observation: str | None = None,
) -> DirectSession:
    """Open a new mode with its own prompt and the preceding exchange, if any."""
    client = DeepSeekClient(
        api_key=api_key,
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
        thinking=os.environ.get("DEEPSEEK_THINKING", "low"),
    )
    await client.__aenter__()
    return DirectSession(system_prompt, client, observe_on_start=observe_on_start,
                         handoff_messages=handoff_messages, managed_observations=managed_observations,
                         pending_observation=pending_observation)


class DirectSession:
    # The frontend can drain every delta before handing off to a new prompt.
    emits_response_done = True

    def __init__(self, system_prompt, client, *, visual_interval=8.0, max_turns=20,
                 observe_on_start=False, handoff_messages=(),
                 managed_observations=False, pending_observation=None):
        self.session_id = uuid.uuid4().hex
        self._system = {"role": "system", "content": system_prompt}
        self._client = client
        self._visual_interval = visual_interval
        self._max_turns = max_turns
        self._user_turns = 0
        self._handoff = handoff_messages
        self._history = []
        self._last_observation = []
        self._latest_frame: tuple[bytes, dict[str, float | str] | None] | None = None
        self._last_request = float("-inf")
        self._observe = observe_on_start
        self._managed_observations = managed_observations
        self.pending_observation = pending_observation if managed_observations else None
        if self.pending_observation is not None and self.pending_observation not in _OBSERVATION_TASKS:
            raise ValueError("Unknown pending observation task")
        self._resume_user_goal = bool((observe_on_start or self.pending_observation) and handoff_messages)
        self._wait_for_placement = bool(observe_on_start and self.pending_observation and handoff_messages)
        self._awaiting_delivery = False
        self._events = asyncio.Queue()
        self._response_id = None
        self._interrupted = set()
        self._active = None
        self._tasks = set()
        self._request_lock = asyncio.Lock()
        self._closed = False

    def submit_frame(self, jpeg: bytes | None, text: str | None = None, *,
                     placement: dict | None = None) -> None:
        """Accept the latest frame, giving new human input priority over observation."""
        if self._closed:
            raise RuntimeError("Photo session is closed")
        if jpeg is not None:
            self._latest_frame = (jpeg, None if placement is None else placement.copy())
        human_turn = bool(text and text.strip())
        if human_turn:
            if self._managed_observations:
                self.pending_observation = None
                self._last_observation.clear()
            if self._active is not None and (not self._active.done() or self._awaiting_delivery):
                self._active.cancel()
                self._interrupted.add(self._response_id)
                self._events.put_nowait((None, {
                    "type": "response.interrupted", "response_id": self._response_id, "reason": "new_input",
                }))
                self._response_id = None
                self._awaiting_delivery = False
            if self._user_turns >= self._max_turns:
                self._events.put_nowait((None, {
                    "type": "error",
                    "error": {
                        "code": "history_limit",
                        "message": "Conversation history limit reached; start a new session.",
                    },
                }))
                return
        else:
            if self._managed_observations:
                if (self.pending_observation is None or self._awaiting_delivery
                        or (self._wait_for_placement and (placement is None or not placement.get("avatar_visible", True)))):
                    return
                text = _OBSERVATION_TASKS[self.pending_observation]
            elif self._observe:
                text = OBSERVATION_MESSAGE
            else:
                return
            if (jpeg is None or self._visual_interval <= 0
                    or (self._active is not None and not self._active.done())
                    or time.monotonic() - self._last_request < self._visual_interval):
                return

        content = text
        if self._latest_frame is not None:
            image, frame_placement = self._latest_frame
            content = [{"type": "text", "text": text}, Image(image).block()]
            if frame_placement is not None:
                content.append({"type": "text", "text": "本帧角色脚点和倍率：" + json.dumps(frame_placement, ensure_ascii=False)})
        user_message = {"role": "user", "content": content}
        messages = [self._system, *self._handoff, *self._history, *self._last_observation, user_message]
        if human_turn:
            # Keep the user's goal even when its answer is interrupted.
            self._history.append(user_message)
            self._user_turns += 1
        self._response_id = uuid.uuid4().hex
        self._last_request = time.monotonic()
        source = "user" if human_turn else "handoff" if self._resume_user_goal else "observation"
        self._resume_user_goal = False
        self._wait_for_placement = False
        self._awaiting_delivery = self._managed_observations
        self._active = asyncio.create_task(
            self._complete(messages, user_message, human_turn, self._response_id, source, self.pending_observation)
        )
        self._tasks.add(self._active)
        self._active.add_done_callback(self._tasks.discard)

    async def _complete(self, messages, user_message, human_turn, response_id, source, task):
        def emit(event):
            if not self._closed and response_id == self._response_id:
                self._events.put_nowait((response_id, {**event, "response_id": response_id}))

        def delta(chunk):
            event = {"type": "response.output.delta", "kind": "text", "text": chunk, "turn_source": source}
            if task is not None:
                event["observation_task"] = task
            emit(event)

        try:
            # A replacement waits until the previous HTTP stream releases its connection.
            async with self._request_lock:
                if self._closed or response_id != self._response_id:
                    return
                result = await self._client.complete(messages, on_delta=delta)
        except Exception as exc:
            if self._closed or response_id != self._response_id:
                return
            self._observe = False
            self.pending_observation = None
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
        assistant = {"role": "assistant", "content": result.text}
        if human_turn:
            self._history.append(assistant)
            self._last_observation = []
        else:
            self._last_observation = [user_message, assistant]
        if self._managed_observations:
            self._finish_observation_turn(result.text, source)
        self._observe = True
        emit({"type": "response.done", "text": result.text, "metadata": result.metadata})

    def _finish_observation_turn(self, text, source):
        """Only a completed human exchange can request the next camera observation."""
        commands = StreamingCommandParser(allow_plain_speech=True).feed("turn", text)
        kinds = {command.kind for command in commands}
        if source == "user":
            if "capture" not in kinds:
                self.pending_observation = next(
                    (command.params["task"] for command in reversed(commands) if command.kind == "wait"), None)
        elif self.pending_observation == "capture" and "capture" in kinds:
            self.pending_observation = None
        elif self.pending_observation == "feedback":
            if "speak" in kinds or (source == "handoff" and kinds & CHARACTER_ACTIONS):
                self.pending_observation = None

    def handoff_messages(self) -> tuple[dict[str, str], ...]:
        """Keep the last user turn and reply as history, without images or old frame coordinates."""
        last_user = next((i for i in range(len(self._history) - 1, -1, -1)
                          if self._history[i]["role"] == "user"), None)
        # In a mode with only automatic observations, retain the earlier human goal.
        turn = self._handoff[:2] if last_user is None else self._history[last_user:]
        if not turn:
            return ()
        messages = []
        for message in (*turn, *self._last_observation):
            content = message["content"]
            # submit_frame stores the utterance in the first block, before the camera data.
            text = content if isinstance(content, str) else content[0]["text"]
            messages.append({"role": message["role"], "content": text})
        return tuple(messages)

    async def recv(self) -> dict:
        while True:
            response_id, event = await self._events.get()
            if response_id is None or (not self._closed and response_id not in self._interrupted):
                if response_id == self._response_id and event["type"] in {"response.done", "error"}:
                    self._awaiting_delivery = False
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
