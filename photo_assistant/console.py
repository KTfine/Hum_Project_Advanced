"""Line-oriented output shared by the live transcript and the terminal editor."""

import os
import sys
import unicodedata


def _cell_width(char: str) -> int:
    if unicodedata.combining(char) or unicodedata.category(char) in {"Mn", "Me", "Cf"}:
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1


class TerminalOutput:
    """All display calls run synchronously on the client's asyncio thread."""

    def __init__(self) -> None:
        self._speaking = False
        self._editing = False
        self._draft_width = 0
        self._draft_clipped = False
        self._visible_draft = ""
        self._pending: list[str] = []

    def _write(self, text: str, file=None) -> None:
        stream = sys.stdout if file is None else file
        if not self._editing:
            print(text, end="", file=stream, flush=True)
            return
        complete, newline, tail = text.rpartition("\n")
        if newline:
            # Completed speech and status lines can appear above a draft
            # immediately. Only unfinished speech stays buffered.
            self._erase_draft()
            self._flush_pending()
            print(complete, file=stream, flush=True)
            if self._visible_draft:
                print(self._visible_draft, end="", flush=True)
        if tail:
            self._pending.append(tail)

    def _flush_pending(self) -> None:
        if self._pending:
            print("".join(self._pending), end="", flush=True)
            self._pending.clear()

    def _erase_draft(self) -> None:
        if self._draft_width:
            print("\r" + " " * self._draft_width + "\r", end="", flush=True)

    def speech(self, text: str) -> None:
        if text:
            self._write(("" if self._speaking else "[speak] ") + text)
            self._speaking = True

    def end_speech(self) -> None:
        if self._speaking:
            self._write("\n")
            self._speaking = False

    def line(self, text: str, *, file=None) -> None:
        self.end_speech()
        self._write(text + "\n", file)

    def update_draft(self, text: str) -> None:
        if not self._editing:
            self.end_speech()
            self._editing = True
        if sys.stdout.isatty():
            # Keep the draft on one row, even with double-width Chinese
            # characters. Enter prints the entire line when it was clipped.
            try:
                columns = os.get_terminal_size(sys.stdout.fileno()).columns
            except (OSError, ValueError):
                columns = 80
            available = max(0, columns - 1)
            width = 0
            start = len(text)
            while start:
                next_width = width + _cell_width(text[start - 1])
                if next_width > available:
                    break
                width = next_width
                start -= 1
            self._draft_clipped = start > 0
            self._draft_width = min(self._draft_width, available)
            self._erase_draft()
            self._visible_draft = text[start:]
            print(self._visible_draft, end="", flush=True)
            self._draft_width = width
        if not text:
            self.finish_draft()

    def finish_draft(self, submitted: str = "") -> None:
        if not self._editing:
            return
        if submitted:
            if self._draft_clipped or not sys.stdout.isatty():
                print(("\r" if self._draft_width else "") + submitted, flush=True)
            else:
                print(flush=True)
        else:
            self._erase_draft()
        self._editing = False
        self._draft_width = 0
        self._draft_clipped = False
        self._visible_draft = ""
        self._flush_pending()
