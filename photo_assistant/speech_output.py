"""Stream English with AssemblyAI and speak Mandarin with Edge in clause order."""

import asyncio
from collections.abc import Callable
from queue import SimpleQueue
import re
import threading

import httpx


CHINESE_VOICE = "zh-CN-XiaoyiNeural"
ENGLISH_VOICE = "aai-audrey"
_ASSEMBLYAI_URL = "https://tts.us.assemblyai.com/v1/speech"
_SAMPLE_RATE = 24000
_HAN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\U00020000-\U000323af]")
_ENGLISH_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")


def _voice_for(text: str, previous: str) -> str:
    if _HAN.search(text):
        # AssemblyAI's standalone TTS does not speak Mandarin.
        return CHINESE_VOICE
    if _ENGLISH_WORD.search(text):
        return ENGLISH_VOICE
    return previous  # Countdowns and other number-only clauses follow the speaker.


async def _assembly_audio(client: httpx.AsyncClient, text: str, api_key: str):
    payload = {"text": text, "voice_id": ENGLISH_VOICE, "language": "english",
               "encoding": "pcm_s16le", "sample_rate": _SAMPLE_RATE, "container": "raw", "stream": True}
    async with client.stream("POST", _ASSEMBLYAI_URL, json=payload,
                             headers={"Authorization": api_key}) as response:
        response.raise_for_status()
        tail = b""
        async for packet in response.aiter_bytes():
            # A network read can end between the two bytes of a PCM sample.
            data = tail + packet
            end = len(data) - len(data) % 2
            if end:
                yield data[:end]
            tail = data[end:]
        if tail:
            raise ValueError("Incomplete 16-bit PCM sample")


async def _synthesize(text: str, voice: str) -> bytes:
    import edge_tts
    import miniaudio

    request = edge_tts.Communicate(text, voice, connect_timeout=6, receive_timeout=12)
    mp3 = bytearray()
    async with asyncio.timeout(20):
        async for packet in request.stream():
            if packet["type"] == "audio":
                mp3.extend(packet["data"])
    audio = miniaudio.decode(bytes(mp3), output_format=miniaudio.SampleFormat.SIGNED16,
                             nchannels=1, sample_rate=_SAMPLE_RATE)
    return audio.samples.tobytes()


class SpeechOutput:
    """Fetch the next clause while the preceding clause plays on the speaker."""

    def __init__(self, on_status: Callable[[str], None], *, api_key: str | None = None):
        self.speaking = threading.Event()
        self._on_status = on_status
        self._api_key = api_key
        self._closed = threading.Event()
        self._requests: SimpleQueue[str | None] = SimpleQueue()
        self._thread: threading.Thread | None = None
        self._active: tuple[asyncio.AbstractEventLoop, asyncio.Task] | None = None

    def start(self) -> None:
        if self._thread is None and not self._closed.is_set():
            self._thread = threading.Thread(target=self._run, name="speech-output", daemon=True)
            self._thread.start()

    def say(self, text: str) -> None:
        if not self._closed.is_set() and any(char.isalnum() for char in text):
            self._requests.put(text.strip())

    def close(self) -> None:
        self._closed.set()
        self._requests.put(None)
        active = self._active
        if active is not None:
            loop, task = active
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass  # The worker finished between reading _active and cancelling.
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    def _run(self) -> None:
        try:
            asyncio.run(self._process())
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if not self._closed.is_set():
                self._on_status(f"语音播报不可用，回复会继续显示在终端。{exc}")
        finally:
            self._active = None
            self.speaking.clear()
            self._closed.set()

    async def _process(self) -> None:
        import sounddevice

        self._active = asyncio.get_running_loop(), asyncio.current_task()
        if self._closed.is_set():
            return
        english = f"AssemblyAI {ENGLISH_VOICE}" if self._api_key else "输入 AssemblyAI API Key 后启用"
        self._on_status(f"在线朗读已启用。中文 Edge {CHINESE_VOICE}，英文{english}。")
        ready = asyncio.Queue(maxsize=1)
        playback = asyncio.create_task(self._play_ready(ready, sounddevice))
        voice = CHINESE_VOICE
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=6.0)) as client:
                while not self._closed.is_set():
                    text = await asyncio.to_thread(self._requests.get)
                    if text is None or self._closed.is_set():
                        break
                    voice = _voice_for(text, voice)
                    try:
                        if voice == CHINESE_VOICE:
                            await ready.put(await _synthesize(text, voice))
                        elif self._api_key:
                            async for samples in _assembly_audio(client, text, self._api_key):
                                await ready.put(samples)
                    except Exception as exc:
                        # Avoid exposing service URLs, headers or the original reply.
                        reason = (f"HTTP {exc.response.status_code}" if isinstance(exc, httpx.HTTPStatusError)
                                  else type(exc).__name__)
                        if not self._closed.is_set():
                            self._on_status(f"这段语音合成失败（{voice}，{reason}），已跳过。")
        finally:
            playback.cancel()
            await asyncio.gather(playback, return_exceptions=True)

    async def _play_ready(self, ready: asyncio.Queue, device) -> None:
        while True:
            pcm = await ready.get()
            try:
                # Keep the same device stream across PCM packets and adjacent clauses.
                with device.RawOutputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="int16") as stream:
                    self.speaking.set()
                    while not self._closed.is_set():
                        writer = asyncio.create_task(asyncio.to_thread(self._write, stream, pcm))
                        try:
                            await asyncio.shield(writer)
                        except asyncio.CancelledError:
                            # PortAudio must finish its active write before the stream closes.
                            await writer
                            stream.abort()
                            raise
                        try:
                            pcm = await asyncio.wait_for(ready.get(), timeout=0.2)
                        except TimeoutError:
                            break  # Release the microphone after the acoustic tail.
            except Exception as exc:
                if not self._closed.is_set():
                    self._on_status(f"扬声器播放失败。{exc}")
            finally:
                self.speaking.clear()

    def _write(self, stream, pcm: bytes) -> None:
        # The blocking device call covers at most 50 ms of mono int16 audio.
        block_bytes = (_SAMPLE_RATE // 20) * 2
        for start in range(0, len(pcm), block_bytes):
            if self._closed.is_set():
                return
            stream.write(pcm[start:start + block_bytes])
