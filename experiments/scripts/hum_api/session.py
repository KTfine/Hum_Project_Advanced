"""Prompt assembly, successful-turn history and incremental command callbacks."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import time

from photo_assistant.avatar.pose import AvatarPose, WHITE_DRAGON_CATALOG, load_pose_catalog
from photo_assistant.persona import read_character_persona
from photo_assistant.prompts import CO_PHOTO_COMMAND_PROMPT, PHOTO_GUIDANCE_PROMPT, character_name_for
from photo_assistant.protocol import StreamingCommandParser
from .client import ApiError, Image

MODES = ('guidance', 'co-photo')


def _prompt_and_catalog(mode, *, character_name=None, character_persona=None,
                        character_state=None, recent_context=None, sprite_catalog=None):
    if mode not in MODES:
        raise ValueError('Unknown photo mode')
    if sprite_catalog is None:
        sprite_catalog = WHITE_DRAGON_CATALOG
    shared = dict(
        character_name=character_name or character_name_for(sprite_catalog),
        character_persona=read_character_persona() if character_persona is None else character_persona,
        response_language='中文', recent_context=recent_context or '',
        extra_instructions='图文输入中，画面来自用户提供的真人图片。',
    )
    if mode == 'guidance':
        return PHOTO_GUIDANCE_PROMPT.format(**shared), None

    catalog = load_pose_catalog(sprite_catalog)
    state = {'x': 0.5, 'y': 0.95, 'scale': 1.0, 'rotation': 0.0,
             'pose': asdict(catalog.entry_pose), **(character_state or {})}
    return CO_PHOTO_COMMAND_PROMPT.format(
        **shared, pose_catalog=catalog.describe(), tachie_fields=catalog.entry_pose.command_fields(),
        entry_context='起始姿势由调用方提供。',
        current_character_state=json.dumps(state, ensure_ascii=False),
    ), catalog


def build_prompt(mode='guidance', **prompt_options):
    system, _ = _prompt_and_catalog(mode, **prompt_options)
    return system


class PhotoSession:
    """One sequential conversation. Device modules supply images and consume commands.

    Callbacks run synchronously on the event loop: enqueue UI work and return quickly.
    A failed stream may already have emitted commands; those effects cannot be rolled back.
    """
    def __init__(self, client, *, mode='guidance', system_prompt=None,
                 max_turns=20, allow_plain_speech=False, **prompt_options):
        if max_turns < 1 or mode not in MODES:
            raise ValueError('Invalid history limit or mode')
        self.client, self.mode = client, mode
        self.custom_prompt = system_prompt
        self.last_system_prompt: str | None = None
        self.prompt_options = prompt_options
        if system_prompt is None and prompt_options.get('character_persona') is None:
            self.prompt_options['character_persona'] = read_character_persona()
        self.max_turns, self.allow_plain_speech = max_turns, allow_plain_speech
        self._history = []
        self._busy = False

    def _prepare_prompt(self):
        if self.custom_prompt is not None:
            return self.custom_prompt, None
        return _prompt_and_catalog(self.mode, **self.prompt_options)

    @property
    def system_prompt(self):
        system, _ = self._prepare_prompt()
        return system

    @property
    def history(self):
        return deepcopy(self._history)

    def reset(self, mode=None):
        if self._busy:
            raise RuntimeError('Wait for or cancel the active turn before resetting')
        if mode is not None:
            if self.custom_prompt is not None:
                raise ValueError('Custom prompts use explicit replacement, not built-in mode switching')
            if mode not in MODES:
                raise ValueError('Unknown photo mode')
            self.mode = mode
        self._history.clear()

    async def ask(self, text: str, *, images=(), observations=None, on_delta=None, on_command=None):
        images = tuple(images)
        if self._busy:
            raise RuntimeError('A PhotoSession accepts one turn at a time')
        if len(self._history) // 2 >= self.max_turns:
            raise ValueError('History limit reached; reset explicitly before starting a new goal')
        if not text.strip() and not images:
            raise ValueError('Provide text or an image')
        content = [image.block() for image in images]
        content.append({'type': 'text', 'text': text})
        if observations is not None:
            content.append({'type': 'text', 'text': '客户端观测（可能不完整）：' + json.dumps(observations, ensure_ascii=False)})
        user = {'role': 'user', 'content': content}
        system, active_catalog = self._prepare_prompt()
        messages = [{'role': 'system', 'content': system}, *deepcopy(self._history), user]
        parser = StreamingCommandParser(allow_plain_speech=self.allow_plain_speech)
        commands, first_command = [], None
        started = time.monotonic()
        mode_before = self.mode

        def receive(chunk):
            nonlocal first_command, active_catalog
            if on_delta:
                on_delta(chunk)
            for command in parser.feed('turn', chunk):
                if command.kind == 'tachie':
                    if active_catalog is None:
                        path = self.prompt_options.get('sprite_catalog') or WHITE_DRAGON_CATALOG
                        active_catalog = load_pose_catalog(path)
                    if AvatarPose.from_tags(command.params) not in active_catalog.entries:
                        continue
                commands.append(command)
                if first_command is None:
                    first_command = time.monotonic() - started
                if on_command:
                    on_command(command)

        def annotate(result):
            result.commands = list(commands)
            result.metadata.update(
                mode_before=mode_before, mode_after=self.mode, custom_prompt=self.custom_prompt is not None,
                first_command_s=first_command, plain_speech_fragments=parser.plain_speech_count,
                image_sha256=[hashlib.sha256(image.data).hexdigest() for image in images],
                history_messages=len(messages) - 2,
                commands=[asdict(c) for c in commands],
            )

        self._busy = True
        self.last_system_prompt = system
        try:
            result = await self.client.complete(messages, on_delta=receive)
            self._history.extend([user, {'role': 'assistant', 'content': result.text}])
            # The last complete mode request wins, and only after a successful response.
            requested = [c.params['mode'] for c in commands if c.kind == 'mode']
            if requested and self.custom_prompt is None and requested[-1] != self.mode:
                self.mode = requested[-1]
                self._history.clear()
            annotate(result)
            return result
        except ApiError as exc:
            annotate(exc.result)
            raise
        except asyncio.CancelledError as exc:
            if hasattr(exc, 'result'):
                annotate(exc.result)
            raise
        finally:
            self._busy = False
