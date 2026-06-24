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


def has_cyrillic(value: str) -> bool:
    return any("а" <= char.casefold() <= "я" or char in "ёЁ" for char in value)


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
        resolved, _ = self.resolve_with_status(value)
        return resolved

    def resolve_with_status(self, value: str) -> tuple[str, bool]:
        normalized = normalize_name(value)
        if normalized in self.aliases:
            return self.aliases[normalized], True
        return value.strip().strip("'\"`’‘“”"), False


def parse_matchup(text: str, resolver: TeamResolver) -> tuple[str, str]:
    parts = [part.strip() for part in re.split(r"\s*(?:,| vs | v | - |—|–)\s*", text, maxsplit=1, flags=re.I)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError('Введите пару команд, например: "Англия, Гана"')
    resolved: list[str] = []
    unknown: list[str] = []
    for part in parts:
        team, known = resolver.resolve_with_status(part)
        if not known and has_cyrillic(part):
            unknown.append(part)
        resolved.append(team)
    if unknown:
        raise ValueError(
            "Не распознал сборную: "
            + ", ".join(unknown)
            + ". Используйте название из списка участников ЧМ или добавьте алиас в data/team_aliases.json."
        )
    return resolved[0], resolved[1]
