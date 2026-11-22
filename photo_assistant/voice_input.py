"""Recognize Mandarin locally with Vosk and the default microphone."""

import asyncio
from collections.abc import Callable
import json
import os
from pathlib import Path
import queue
import re
import threading
import time

from .paths import PROJECT_ROOT


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "vosk-model-small-cn-0.22"
_HAN_WORD_BOUNDARY = re.compile(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])")


def _transcript(result_json: str) -> str:
    text = json.loads(result_json)["text"].strip()
    return _HAN_WORD_BOUNDARY.sub("", text)


def _listen(
    stop_event: threading.Event, on_text: Callable[[str], None], on_status: Callable[[str], None],
) -> None:
    model_path = Path(os.environ.get("HUM_VOSK_MODEL") or DEFAULT_MODEL_PATH).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"找不到 Vosk 中文模型目录 {model_path}，请安装模型或设置 HUM_VOSK_MODEL。")
    try:
        import sounddevice as sd
        import vosk
    except ImportError as exc:
        raise RuntimeError("请在项目 Python 环境中安装 vosk 和 sounddevice。") from exc

    vosk.SetLogLevel(-1)
    model = vosk.Model(str(model_path))
    if stop_event.is_set():
        return
    sample_rate = int(sd.query_devices(kind="input")["default_samplerate"])
    recognizer = vosk.KaldiRecognizer(model, sample_rate)
    audio = queue.Queue(maxsize=30)  # Three seconds at 100 ms per block.
    generation = 0  # Only the audio callback advances this on a discontinuity.

    def capture(indata, frames, timing, status):
        nonlocal generation
        if status:
            generation += 1
            return
        try:
            audio.put_nowait((generation, bytes(indata)))
        except queue.Full:
            generation += 1

    with sd.RawInputStream(samplerate=sample_rate, blocksize=sample_rate // 10,
                           channels=1, dtype="int16", callback=capture) as stream:
        on_status("正在用 Vosk 听普通话，说完一句稍作停顿。")
        last_audio = time.monotonic()
        decoded_generation = 0
        awaiting_boundary = False
        while not stop_event.is_set():
            try:
                block_generation, block = audio.get(timeout=0.2)
            except queue.Empty:
                if not stream.active or time.monotonic() - last_audio > 5:
                    raise RuntimeError("麦克风已停止提供音频，请检查默认输入设备。")
                continue
            last_audio = time.monotonic()
            if block_generation != generation:
                continue
            if block_generation != decoded_generation:
                recognizer.Reset()
                decoded_generation = block_generation
                if not awaiting_boundary:
                    on_status("麦克风音频丢帧，请停顿片刻，等待听写恢复。")
                awaiting_boundary = True
            if not recognizer.AcceptWaveform(block):
                continue
            text = _transcript(recognizer.Result())
            if block_generation != generation or stop_event.is_set():
                continue
            if awaiting_boundary:
                # This endpoint may contain only the tail of an interrupted command.
                awaiting_boundary = False
                on_status("听写已恢复，请重复刚才的指令。")
            elif text:
                on_text(text)


async def recognize_microphone(
    stop_event: threading.Event, on_text: Callable[[str], None], on_status: Callable[[str], None],
) -> None:
    if stop_event.is_set():
        return
    loop = asyncio.get_running_loop()
    worker_stop = threading.Event()
    worker = asyncio.create_task(asyncio.to_thread(
        _listen, worker_stop,
        lambda text: loop.call_soon_threadsafe(on_text, text),
        lambda message: loop.call_soon_threadsafe(on_status, message),
    ))
    try:
        while not worker.done() and not stop_event.is_set():
            await asyncio.wait([worker], timeout=0.2)
    finally:
        worker_stop.set()
        try:
            await worker
        except Exception as exc:
            if not stop_event.is_set():
                on_status(f"麦克风识别不可用，可以继续打字。{exc}")
