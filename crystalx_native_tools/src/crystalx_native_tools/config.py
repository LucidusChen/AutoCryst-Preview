"""The local tool configuration file (``config.local.json``, git-ignored at the repository root).

Example::

    {
      "tools": {"shelxt": "/opt/xtal/arm64/shelxt", "platon_special": "toolbox/platon_special"},
      "toolbox": "toolbox",
      "search_dirs": ["/opt/xtal/bin"],
      "checkdef": "/usr/local/bin/check.def",
      "strict_native": true,
      "include_olex2": true,
      "include_known_locations": true
    }

Every key is optional. Relative paths are resolved against the directory that holds the file.
Unknown keys and unknown tool names are reported as problems, never silently used; a file that is
not valid JSON is a problem too, and the defaults apply. Nothing in this file is executed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crystalx_core.domain.tools import ToolKind

__all__ = ["ToolsConfig", "load_config", "default_config_path", "CONFIG_FILE_NAME"]

CONFIG_FILE_NAME = "config.local.json"
_KNOWN_KEYS = {"schema", "tools", "toolbox", "search_dirs", "checkdef", "strict_native", "include_olex2",
               "include_known_locations"}
_TOOL_NAMES = {kind.value for kind in ToolKind}


@dataclass(frozen=True)
class ToolsConfig:
    source: Path | None = None
    tools: dict[str, Path] = field(default_factory=dict)
    toolbox: Path | None = None
    search_dirs: tuple[Path, ...] = ()
    checkdef: Path | None = None
    strict_native: bool | None = None
    """``None`` when the file does not say; the caller's default then applies."""

    include_olex2: bool = True
    include_known_locations: bool = True
    problems: tuple[str, ...] = ()

    def with_overrides(self, *, toolbox: Path | None = None, strict_native: bool | None = None) -> "ToolsConfig":
        return ToolsConfig(
            source=self.source,
            tools=dict(self.tools),
            toolbox=_absolute(toolbox) if toolbox is not None else self.toolbox,
            search_dirs=self.search_dirs,
            checkdef=self.checkdef,
            strict_native=self.strict_native if strict_native is None else strict_native,
            include_olex2=self.include_olex2,
            include_known_locations=self.include_known_locations,
            problems=self.problems,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "source": str(self.source) if self.source else None,
            "tools": {name: str(path) for name, path in sorted(self.tools.items())},
            "toolbox": str(self.toolbox) if self.toolbox else None,
            "search_dirs": [str(d) for d in self.search_dirs],
            "checkdef": str(self.checkdef) if self.checkdef else None,
            "strict_native": self.strict_native,
            "include_olex2": self.include_olex2,
            "include_known_locations": self.include_known_locations,
            "problems": list(self.problems),
        }


def _absolute(path: Path | str, base: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (base or Path.cwd()) / candidate
    return Path(os.path.abspath(candidate))


def default_config_path(repo_root: Path | None = None) -> Path:
    return (repo_root or Path.cwd()) / CONFIG_FILE_NAME


def load_config(path: Path | str | None) -> ToolsConfig:
    """Read the configuration. A missing file gives the defaults with no problems."""
    if path is None:
        return ToolsConfig()
    source = _absolute(path)
    if not source.is_file():
        return ToolsConfig(source=None)
    base = source.parent
    problems: list[str] = []
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return ToolsConfig(source=source, problems=(f"{source.name} is not readable JSON: {exc}",))
    if not isinstance(raw, dict):
        return ToolsConfig(source=source, problems=(f"{source.name} must hold a JSON object",))

    for key in sorted(set(raw) - _KNOWN_KEYS):
        problems.append(f"unknown key {key!r} ignored")

    tools: dict[str, Path] = {}
    raw_tools = raw.get("tools", {})
    if isinstance(raw_tools, dict):
        for name, value in raw_tools.items():
            if name not in _TOOL_NAMES:
                problems.append(f"unknown tool {name!r} ignored")
            elif not isinstance(value, str) or not value.strip():
                problems.append(f"tool {name!r} needs a path string")
            else:
                tools[name] = _absolute(value, base)
    else:
        problems.append("'tools' must be an object of tool name to path")

    def path_value(key: str) -> Path | None:
        value = raw.get(key)
        if value is None:
            return None
        if isinstance(value, str) and value.strip():
            return _absolute(value, base)
        problems.append(f"{key!r} must be a path string")
        return None

    search: list[Path] = []
    raw_dirs = raw.get("search_dirs", [])
    if isinstance(raw_dirs, list) and all(isinstance(d, str) for d in raw_dirs):
        search = [_absolute(d, base) for d in raw_dirs if d.strip()]
    else:
        problems.append("'search_dirs' must be a list of path strings")

    def flag(key: str, default: bool | None) -> bool | None:
        value = raw.get(key, default)
        if value is None or isinstance(value, bool):
            return value
        problems.append(f"{key!r} must be true or false")
        return default

    return ToolsConfig(
        source=source,
        tools=tools,
        toolbox=path_value("toolbox"),
        search_dirs=tuple(search),
        checkdef=path_value("checkdef"),
        strict_native=flag("strict_native", None),
        include_olex2=bool(flag("include_olex2", True)),
        include_known_locations=bool(flag("include_known_locations", True)),
        problems=tuple(problems),
    )
