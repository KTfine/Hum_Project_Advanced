"""Load the personality text shared by guidance and co-photo sessions."""

from pathlib import Path

from .paths import PROJECT_ROOT


DEFAULT_PERSONA_FILE = PROJECT_ROOT / "角色性格.txt"


def read_character_persona(path: Path = DEFAULT_PERSONA_FILE) -> str:
    try:
        return path.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"无法读取 UTF-8 角色性格文件 {path}：{exc}") from exc


def add_persona_options(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--character-persona", help="one-run personality text")
    group.add_argument("--character-persona-file", type=Path, default=DEFAULT_PERSONA_FILE,
                       help="UTF-8 personality file (default: %(default)s)")


def load_persona_option(parser, args):
    if args.character_persona is None:
        try:
            args.character_persona = read_character_persona(args.character_persona_file)
        except ValueError as exc:
            parser.error(str(exc))
