from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


def normalize_name(value: str) -> str:
    value = value.strip().strip("'\"`’‘“”")
    value = unicodedata.normalize("NFKC", value)
    value = value.casefold()
    value = re.sub(r"[\s_\-.,;:|/\\]+", " ", value)
    return value.strip()


class TeamResolver:
    def __init__(self, alias_path: Path):
        self.alias_path = alias_path
        self.aliases = self._load_aliases()

    def _load_aliases(self) -> dict[str, str]:
        raw = json.loads(self.alias_path.read_text(encoding="utf-8"))
        aliases: dict[str, str] = {}
        for canonical, values in raw.items():
            aliases[normalize_name(canonical)] = canonical
            for value in values:
                aliases[normalize_name(value)] = canonical
        return aliases

    def resolve(self, value: str) -> str:
        normalized = normalize_name(value)
        return self.aliases.get(normalized, value.strip().strip("'\"`’‘“”"))


def parse_matchup(text: str, resolver: TeamResolver) -> tuple[str, str]:
    parts = [part.strip() for part in re.split(r"\s*(?:,| vs | v | - |—|–)\s*", text, maxsplit=1, flags=re.I)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError('Введите пару команд, например: "Англия, Гана"')
    return resolver.resolve(parts[0]), resolver.resolve(parts[1])
