"""Stream the selected microphone to AssemblyAI and submit completed user turns."""

import asyncio
from collections.abc import Callable
from contextlib import suppress
import os
import queue
import threading
import time


_ENDPOINT = "wss://streaming.assemblyai.com/v3/ws"
_MODEL = "assemblyai-stt-3-7-realtime-agent"


class _ConnectionLost(ConnectionError):
    """A transient server closure for which a fresh streaming session is appropriate."""


def _message_data(message):
    from aiohttp import WSMsgType

    if message.type == WSMsgType.TEXT:
        data = message.json()
        if data.get("type") == "Error" or data.get("error"):
            raise RuntimeError(f"AssemblyAI 拒绝识别请求：{data.get('error') or data.get('message')}")
        return data

    if message.type == WSMsgType.CLOSE:
        reason = message.extra or "服务端关闭连接"
        if message.data in {1008, 3006, 3007, 3009}:
            raise RuntimeError(f"AssemblyAI 关闭连接（{message.data}）：{reason}")
        raise _ConnectionLost(f"AssemblyAI 关闭连接（{message.data}）：{reason}")
    raise _ConnectionLost("AssemblyAI 语音连接已断开。")


async def _stream_turns(ws, sd, device, info, speaking, on_text, on_status):
    sample_rate = int(info["default_samplerate"])
    audio = queue.Queue(maxsize=30)
    generation = 0  # Only the capture callback advances this after an audio discontinuity.

    def capture(indata, frames, timing, status):
        nonlocal generation
        if status:
            generation += 1
            return
        muted = speaking is not None and speaking.is_set()
        # Preserve the audio clock for remote endpointing, without transmitting speaker echo.
        pcm = bytes(frames * 2) if muted else bytes(indata)
        try:
            audio.put_nowait((generation, pcm, muted))
        except queue.Full:
            generation += 1

    async def upload(stream):
        last_audio = time.monotonic()
        reported_generation = 0
        silent_blocks = 0
        warned_silence = False
        while True:
            try:
                block_generation, pcm, muted = await asyncio.to_thread(audio.get, True, 0.25)
            except queue.Empty:
                if not stream.active or time.monotonic() - last_audio >= 5:
                    raise RuntimeError("设备已停止提供音频，请检查系统麦克风权限或通过 --microphone-device 选择输入。")
                continue
            last_audio = time.monotonic()
            if generation != reported_generation:
                reported_generation = generation
                on_status("麦克风音频丢帧，本句将跳过，请稍后重复。")
            if block_generation != generation:
                continue
            if not warned_silence and not muted:
                silent_blocks = 0 if any(pcm) else silent_blocks + 1
                if silent_blocks >= 50:
                    warned_silence = True
                    on_status("此输入设备已连续 5 秒提供全零音频，请检查静音、系统输入音量或 --microphone-device。")
            async with asyncio.timeout(8):
                await ws.send_bytes(pcm)

    async def receive():
        last_final = -1
        accepted_generation = 0
        while True:
            data = _message_data(await ws.receive())
            kind = data.get("type")
            if kind == "SpeechStarted":
                on_status("检测到说话，正在识别。")
            elif kind == "Termination":
                raise _ConnectionLost("AssemblyAI 已结束听写会话。")
            elif kind == "Turn" and data.get("end_of_turn"):
                order = data["turn_order"]
                if order <= last_final:
                    continue
                last_final = order
                if accepted_generation != generation:
                    accepted_generation = generation
                    on_status("已跳过不完整的语音，请重新说一遍。")
                    continue
                text = data.get("transcript", "").strip()
                if text:
                    on_text(text)

    with sd.RawInputStream(device=device, samplerate=sample_rate, blocksize=sample_rate // 10,
                           channels=1, dtype="int16", callback=capture) as stream:
        on_status(f"AssemblyAI 已连接，正在监听 {info['name']}（{sample_rate} Hz）。可以说中文或英文。")
        tasks = [asyncio.create_task(upload(stream)), asyncio.create_task(receive())]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def _listen(api_key, device, speaking, on_text, on_status):
    import aiohttp
    import sounddevice as sd

    info = sd.query_devices(device, "input")
    params = {"sample_rate": int(info["default_samplerate"]), "speech_model": _MODEL, "mode": "balanced"}
    timeout = aiohttp.ClientTimeout(total=12)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as client:
        for attempt in range(3):
            try:
                on_status(f"正在为 {info['name']} 连接 AssemblyAI。")
                async with client.ws_connect(_ENDPOINT, params=params, headers={"Authorization": api_key},
                                             heartbeat=20) as ws:
                    try:
                        async with asyncio.timeout(12):
                            first = _message_data(await ws.receive())
                        if first.get("type") != "Begin":
                            raise RuntimeError("AssemblyAI 返回了未知的会话初始化事件。")
                        await _stream_turns(ws, sd, device, info, speaking, on_text, on_status)
                        return
                    finally:
                        if not ws.closed:
                            # Close the billable session even on cancellation or a device error.
                            with suppress(TimeoutError, ConnectionError, aiohttp.ClientError):
                                async with asyncio.timeout(1):
                                    await ws.send_json({"type": "Terminate"})
                                    while not ws.closed:
                                        message = await ws.receive()
                                        if message.type != aiohttp.WSMsgType.TEXT or message.json().get("type") == "Termination":
                                            break
            except aiohttp.WSServerHandshakeError as exc:
                if exc.status in {401, 403}:
                    raise RuntimeError(f"AssemblyAI 认证失败（HTTP {exc.status}），请检查 API Key 和账户状态。") from exc
                if exc.status < 500 or attempt == 2:
                    raise RuntimeError(f"AssemblyAI 语音连接失败（HTTP {exc.status}）。") from exc
            except (aiohttp.ClientConnectionError, TimeoutError, _ConnectionLost) as exc:
                if attempt == 2:
                    raise RuntimeError(f"AssemblyAI 语音连接中断，请检查网络。{exc}") from exc
            on_status("AssemblyAI 语音连接中断，正在重新连接。")
            await asyncio.sleep(2 ** attempt)


async def recognize_microphone(
    stop_event: threading.Event, on_text: Callable[[str], None], on_status: Callable[[str], None],
    *, speaking: threading.Event | None = None, api_key: str | None = None, device: int | str | None = None,
) -> None:
    if stop_event.is_set():
        return
    key = (api_key or os.environ.get("ASSEMBLYAI_API_KEY") or "").strip()
    if not key:
        on_status("未配置 AssemblyAI API Key，无法开始听写。")
        return

    def deliver(text):
        if not stop_event.is_set():
            on_text(text)

    worker = asyncio.create_task(_listen(key, device, speaking, deliver, on_status))
    try:
        while not worker.done() and not stop_event.is_set():
            await asyncio.wait([worker], timeout=0.2)
        if worker.done():
            await worker
    except Exception as exc:
        if not stop_event.is_set():
            # Provider diagnostics are visible in the terminal. Never echo the credential.
            detail = str(exc).replace(key, "***")
            on_status(f"麦克风识别不可用，可以继续打字。{detail}")
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
