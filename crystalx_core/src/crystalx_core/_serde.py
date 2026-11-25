"""Minimal, dependency-free serialisation for the frozen dataclasses in ``crystalx_core.domain``.

Why hand-rolled rather than pydantic/attrs: ``crystalx_core`` must install and run with the standard
library alone, on any platform, so that the workflow core is never the reason an environment fails
to build.

Supported types: dataclasses, ``Enum``, ``NewType`` over a supported base, ``str``/``int``/``float``/
``bool``/``None``, ``Path``, ``datetime`` (timezone-aware, serialised as ISO-8601), ``list[T]``,
``tuple[T, ...]``, ``dict[str, T]``, and ``T | None``.

Deliberately unsupported: unions of dataclasses. Model a discriminated choice with an explicit
``kind`` enum plus optional fields instead, so that the on-disk record stays self-describing.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import json
import types
import typing
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

__all__ = ["to_jsonable", "from_jsonable", "dumps", "loads", "canonical_json", "DataclassT"]

DataclassT = TypeVar("DataclassT")

_NONE_TYPE = type(None)


def _unwrap_newtype(tp: Any) -> Any:
    """Return the supertype of a ``NewType``, recursively; other types pass through."""
    while hasattr(tp, "__supertype__"):
        tp = tp.__supertype__
    return tp


def _is_optional(tp: Any) -> tuple[bool, Any]:
    """Return ``(True, inner)`` for ``Optional[inner]``, else ``(False, tp)``."""
    origin = get_origin(tp)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in get_args(tp) if a is not _NONE_TYPE]
        if len(args) == 1 and len(get_args(tp)) == 2:
            return True, args[0]
    return False, tp


# --------------------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------------------


def to_jsonable(value: Any) -> Any:
    """Convert a domain object into plain JSON-compatible Python data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, _dt.datetime):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    raise TypeError(f"Cannot serialise object of type {type(value)!r}")


# --------------------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------------------


def from_jsonable(target_type: type[DataclassT] | Any, data: Any) -> Any:
    """Rebuild a domain object of ``target_type`` from plain JSON data."""
    target_type = _unwrap_newtype(target_type)

    is_optional, inner = _is_optional(target_type)
    if is_optional:
        if data is None:
            return None
        return from_jsonable(inner, data)

    if target_type is Any or target_type is None:
        return data

    origin = get_origin(target_type)
    if origin is not None:
        args = get_args(target_type)
        if origin in (list, set, frozenset):
            built = [from_jsonable(args[0], item) for item in data]
            return origin(built) if origin is not list else built
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(from_jsonable(args[0], item) for item in data)
            return tuple(from_jsonable(arg, item) for arg, item in zip(args, data))
        if origin is dict:
            return {str(k): from_jsonable(args[1], v) for k, v in data.items()}
        raise TypeError(f"Cannot deserialise generic type {target_type!r}")

    if isinstance(target_type, type):
        if issubclass(target_type, enum.Enum):
            return target_type(data)
        if issubclass(target_type, Path):
            return Path(data)
        if issubclass(target_type, _dt.datetime):
            return _dt.datetime.fromisoformat(data)
        if issubclass(target_type, bool):
            return bool(data)
        if dataclasses.is_dataclass(target_type):
            hints = get_type_hints(target_type, include_extras=False)
            kwargs: dict[str, Any] = {}
            for field in dataclasses.fields(target_type):
                if field.name not in data:
                    if (
                        field.default is dataclasses.MISSING
                        and field.default_factory is dataclasses.MISSING  # type: ignore[misc]
                    ):
                        raise KeyError(
                            f"Missing required field {field.name!r} for {target_type.__name__}"
                        )
                    continue
                kwargs[field.name] = from_jsonable(hints[field.name], data[field.name])
            return target_type(**kwargs)

    return data


# --------------------------------------------------------------------------------------
# json helpers
# --------------------------------------------------------------------------------------


def dumps(value: Any, *, indent: int | None = 2) -> str:
    """Serialise a domain object to a JSON string."""
    return json.dumps(to_jsonable(value), indent=indent, ensure_ascii=False, sort_keys=False)


def loads(target_type: type[DataclassT], text: str) -> DataclassT:
    """Parse a JSON string back into ``target_type``."""
    return typing.cast(DataclassT, from_jsonable(target_type, json.loads(text)))


def canonical_json(value: Any) -> str:
    """Deterministic JSON rendering, for hashing.

    Keys sorted, no insignificant whitespace, so that the same logical object always produces the
    same bytes and therefore the same digest.
    """
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
