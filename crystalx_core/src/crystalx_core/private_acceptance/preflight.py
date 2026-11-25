"""Fail-closed, read-only preflight for a *selected* private real-run toolchain.

This does not resolve alternative programs, probe launches, repair the environment, or certify
an OS signature. The optional native-tools inspector supplies dependency/host facts; a stricter
bounded structural check closes known inventory gaps without changing its shared implementation.
Keep these records under the private run root: executable paths and hashes are local evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from crystalx_core.app.models import RunMode
from crystalx_core.domain.tools import ResolutionSource, ToolKind, ToolResolution
from crystalx_core.workspace.hashing import sha256_file

if TYPE_CHECKING:
    from crystalx_core.app.toolchain import Toolchain

__all__ = ["PreflightPolicy", "PreflightReport", "evaluate_toolchain", "recheck_before_dispatch"]

_MAX_COMMAND_BYTES = 4 * 1024 * 1024
_MAX_SIGNATURE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class PreflightPolicy:
    strict_native: bool = True
    required_tools: tuple[ToolKind, ...] = (ToolKind.SHELXT, ToolKind.SHELXL)
    require_platon: bool = True
    checkdef_path: Path | None = None
    # A resource on disk is not necessarily a resource in the real process's working directory.
    # Only the harness that binds/copies this exact digest into that directory may set this.
    validation_resource_binding_verified: bool = False


@dataclass(frozen=True)
class PreflightReport:
    status: str
    tools: tuple[dict[str, Any], ...]
    blockers: tuple[str, ...]
    host: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""
    policy: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "READY_STATIC" and not self.blockers

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "static": {"status": self.status, "blockers": list(self.blockers)},
            "launch": {"status": "NOT_TESTED", "real_invocations": 0},
            "validation": {"status": "NOT_VERIFIED"},
            "host": self.host,
            "policy": self.policy,
            "tools": list(self.tools),
            "fingerprint": self.fingerprint,
            "limitations": [
                "Read-only static evidence; no program has been launched.",
                "Signature container structure is checked; cryptographic and Gatekeeper approval are not.",
                "A dispatch recheck narrows replacement races; it is not an atomic OS executable pin.",
                "Local dependencies are inspected recursively within a finite bound; dyld cache content and dynamic loader search overrides are not certified.",
            ],
        }


def _policy_record(policy: PreflightPolicy) -> dict[str, Any]:
    return {
        "strict_native": policy.strict_native,
        "required_tools": [kind.value for kind in policy.required_tools],
        "require_platon": policy.require_platon,
        "checkdef_path": str(policy.checkdef_path) if policy.checkdef_path is not None else None,
        "validation_resource_binding_verified": policy.validation_resource_binding_verified,
    }


def _report(tools: list[dict[str, Any]], blockers: list[str], host: dict[str, Any],
            policy: PreflightPolicy) -> PreflightReport:
    body = {"tools": tools, "host": host, "policy": _policy_record(policy)}
    fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return PreflightReport("BLOCKED" if blockers else "READY_STATIC", tuple(tools),
                           tuple(dict.fromkeys(blockers)), host, fingerprint, body["policy"])


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "realpath": str(path.resolve(strict=True)), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "mode": stat.st_mode, "device": stat.st_dev,
            "inode": stat.st_ino, "sha256": sha256_file(path)}


def evaluate_toolchain(toolchain: Toolchain, *, policy: PreflightPolicy | None = None) -> PreflightReport:
    """Inspect precisely the resolutions used by ``toolchain``; never run a binary.

    Importing core alone does not require ``crystalx_native_tools``. If that optional local
    inspector is unavailable, dedicated real acceptance is BLOCKED instead of being skipped.
    The caller must use this same toolchain instance for the subsequent RunService dispatch.
    """
    policy = policy or PreflightPolicy()
    blockers: list[str] = []
    if not policy.required_tools and not policy.require_platon:
        blockers.append("no_required_stages")
    if any(kind not in (ToolKind.SHELXT, ToolKind.SHELXL) for kind in policy.required_tools):
        blockers.append("unsupported_stage: only implemented SHELXT/SHELXL and selected PLATON are eligible")
    if toolchain.mode != RunMode.REAL:
        return _report([], ["real_toolchain_required"], {}, policy)
    try:
        from crystalx_native_tools.host import detect_host
        from crystalx_native_tools.inspect import inspect_file
    except ImportError:
        return _report([], ["optional_native_inspector_unavailable"], {}, policy)
    try:
        host = detect_host()
        host_record = host.to_record()
        availability = toolchain.availability()
        selected = [availability.get(kind) for kind in policy.required_tools]
        if policy.require_platon:
            selected.append(toolchain.platon())
    except (OSError, ValueError, RuntimeError) as exc:
        return _report([], [f"selection_or_host_check_failed: {type(exc).__name__}"], {}, policy)
    if host.system != "Darwin":
        blockers.append("host_format_not_supported: private real preflight currently validates macOS Mach-O only")
    if not host.macos_version:
        blockers.append("host_macos_version_unverified")
    runner = getattr(toolchain, "runner", None)
    environment = os.environ if runner is None else getattr(runner, "base_env", None)
    if environment is None:
        blockers.append("runner_environment_unverified")
        host_record["loader_environment"] = {"status": "UNVERIFIED"}
    else:
        loader = {key: hashlib.sha256(str(value).encode()).hexdigest()
                  for key, value in environment.items()
                  if value and key.startswith(("DYLD_", "LD_"))}
        host_record["loader_environment"] = {"variables_sha256": loader}
        if loader:
            blockers.append("loader_environment_override_unverified: " + ", ".join(sorted(loader)))
    tools: list[dict[str, Any]] = []
    names = [kind.value for kind in policy.required_tools] + (["platon"] if policy.require_platon else [])
    for name, resolution in zip(names, selected):
        item = _evaluate_resolution(name, resolution, host, policy, inspect_file)
        tools.append(item)
        blockers.extend(f"{item['tool']}: {reason}" for reason in item["blockers"])
    return _report(tools, blockers, host_record, policy)


def _evaluate_resolution(name: str, resolution: ToolResolution | None, host: Any,
                         policy: PreflightPolicy, inspect_file: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"tool": name, "static": "BLOCKED", "launch": "NOT_TESTED",
                            "validation": "NOT_VERIFIED", "blockers": [], "selection": None}
    reasons: list[str] = item["blockers"]
    if resolution is None:
        reasons.append("selected_resolution_missing")
        return item
    item["tool"] = resolution.kind.value
    if (name == "platon" and resolution.kind not in (ToolKind.PLATON, ToolKind.PLATON_SPECIAL)
            or name != "platon" and resolution.kind.value != name):
        reasons.append("selected_resolution_kind_mismatch")
    item["resolver"] = {"status": resolution.diagnostic.kind.value,
                         "message": resolution.diagnostic.message}
    identity = resolution.identity
    if not resolution.is_usable:
        reasons.append("selected_resolution_unusable")
    if identity is None:
        reasons.append("selected_identity_missing")
        return item
    item["selection"] = {"path": str(identity.path), "realpath": str(identity.realpath),
                         "sha256": identity.sha256, "source": identity.source.value,
                         "architecture": identity.arch.value,
                         "size": identity.size_bytes}
    if identity.source == ResolutionSource.SYNTHETIC_DEMO:
        reasons.append("synthetic_source_not_permitted")
    if identity.kind != resolution.kind:
        reasons.append("selected_identity_kind_mismatch")
    try:
        current = _file_fingerprint(identity.path)
        item["file"] = current
        if (current["sha256"] != identity.sha256 or current["size"] != identity.size_bytes
                or current["realpath"] != str(identity.realpath)):
            reasons.append("selected_identity_changed")
        inspection = inspect_file(identity.path, host, check_dependencies=True)
        item["inspection"] = inspection.to_record()
        if not inspection.regular or not inspection.executable:
            reasons.append("not_regular_executable")
        if inspection.quarantined is not False:
            reasons.append("quarantined" if inspection.quarantined else "quarantine_state_unverified")
        if inspection.format not in ("mach_o", "mach_o_fat") or inspection.macho is None:
            reasons.append("unsupported_executable_format: scripts/interpreters and unknown formats are unverified")
        else:
            macho = inspection.macho
            chosen = macho.slice_for(host.hardware_arch)
            if chosen is None:
                reasons.append("native_architecture_missing" if policy.strict_native else "translation_launch_unverified")
            if macho.problems or inspection.problems:
                reasons.extend("native_inspection: " + str(problem) for problem in inspection.problems)
                reasons.extend("macho: " + str(problem) for problem in macho.problems)
            if chosen is not None:
                item["execution_architecture"] = chosen.arch
                item["translation"] = False
                item["minimum_macos"] = chosen.min_os
                if chosen.filetype != "execute":
                    reasons.append("macho_not_executable")
                if chosen.platform != "macos":
                    reasons.append("macho_platform_not_macos")
                if not chosen.has_entry_point:
                    reasons.append("macho_entry_point_missing")
                if not chosen.min_os:
                    reasons.append("minimum_macos_unverified")
                reasons.extend("macho_slice: " + problem for problem in chosen.problems)
            reasons.extend("macho_structure: " + problem for problem in _macho_structure(identity.path))
        item["dependency_files"] = _dependency_evidence(inspection, host, inspect_file, reasons)
        # Check that hashing and inspecting one pathname did not observe two different files.
        if _file_fingerprint(identity.path) != current or inspection.sha256 != current["sha256"]:
            reasons.append("executable_changed_during_inspection")
    except (OSError, ValueError, struct.error) as exc:
        reasons.append("static_inspection_failed: " + type(exc).__name__)
    if resolution.kind in (ToolKind.PLATON, ToolKind.PLATON_SPECIAL):
        item["resources"] = _check_resources(policy)
        reasons.extend(item["resources"]["blockers"])
    item["static"] = "BLOCKED" if reasons else "READY_STATIC"
    return item


def _dependency_evidence(inspection: Any, host: Any, inspect_file: Any,
                         reasons: list[str]) -> list[dict[str, Any]]:
    """Inspect local linked dylibs as data, with a hard bound and cycle protection.

    Native inspector's @executable_path expansion uses the subject directory. For a transitive
    dylib this is not generally the executable directory; refuse that uncertain context rather
    than claim another path would be loaded. The dyld shared cache remains OS-supplied evidence.
    """
    pending = [(dep, 0) for dep in inspection.dependencies]
    visited: set[str] = set()
    evidence: list[dict[str, Any]] = []
    while pending:
        dep, depth = pending.pop(0)
        if dep.kind == "weak" and dep.status == "missing_weak":
            continue
        if dep.status == "system_cache":
            continue
        if dep.blocks_launch or dep.status != "present" or dep.arch_ok is not True or not dep.resolved:
            reasons.append(f"dependency_unverified: {dep.name} ({dep.status})")
            continue
        if depth and dep.name.startswith("@executable_path"):
            reasons.append(f"transitive_loader_context_unverified: {dep.name}")
            continue
        path = Path(dep.resolved)
        realpath = str(path.resolve())
        if realpath in visited:
            continue
        if len(visited) >= 128:
            reasons.append("dependency_graph_limit_exceeded")
            break
        visited.add(realpath)
        before = _file_fingerprint(path)
        library = inspect_file(path, host, check_dependencies=True)
        record = {"file": before, "inspection": library.to_record()}
        evidence.append(record)
        if library.quarantined is not False:
            reasons.append(f"dependency_quarantine_unverified: {path}")
        chosen = library.macho.slice_for(host.hardware_arch) if library.macho else None
        if not library.regular or chosen is None or chosen.filetype != "dylib":
            reasons.append(f"dependency_not_native_dylib: {path}")
        elif chosen.platform != "macos" or not chosen.min_os:
            reasons.append(f"dependency_platform_or_minos_unverified: {path}")
        # inspect_file evaluates a *main executable*. A verified native MH_DYLIB is the
        # expected dependency type and need not have execute permission. Ignore only those
        # context-specific findings, never malformed/platform/signature/dependency findings.
        # Older optional inspector versions have no structured findings, so retain the exact
        # existing permission-message compatibility path as well.
        library_context = library.regular and chosen is not None and chosen.filetype == "dylib"
        main_only = tuple(f for f in getattr(library, "findings", ())
                          if library_context and f.code in ("not_executable", "not_an_executable"))
        ignored_messages = {f.message for f in main_only}
        if library_context:
            ignored_messages.add("not marked executable")
        record["inspection_context"] = "linked_dynamic_library"
        record["main_executable_only_findings"] = [f.code for f in main_only]
        problems = [p for p in library.problems if p not in ignored_messages]
        if problems or (library.macho and library.macho.problems):
            reasons.append(f"dependency_inspection_rejected: {path}")
        structure = _macho_structure(path)
        record["structure_problems"] = structure
        if structure:
            reasons.append(f"dependency_structure_rejected: {path}")
        if before != _file_fingerprint(path) or before["sha256"] != library.sha256:
            reasons.append(f"dependency_changed_during_inspection: {path}")
        pending.extend((child, depth + 1) for child in library.dependencies)
    return evidence


def _check_resources(policy: PreflightPolicy) -> dict[str, Any]:
    evidence: dict[str, Any] = {"checkdef": None, "blockers": [], "compatibility": "NOT_VERIFIED"}
    if policy.checkdef_path is None:
        evidence["blockers"].append("required_checkdef_missing: no automatic creation may be assumed")
        return evidence
    try:
        path = Path(policy.checkdef_path)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("not a nonempty regular file")
        evidence["checkdef"] = _file_fingerprint(path)
    except (OSError, ValueError):
        evidence["blockers"].append("required_checkdef_missing_or_unreadable")
    if not policy.validation_resource_binding_verified:
        evidence["blockers"].append("checkdef_process_binding_unverified: pipeline unsets CHECKDEF")
    return evidence


def recheck_before_dispatch(toolchain: Toolchain, previous_report: PreflightReport, *,
                            policy: PreflightPolicy | None = None) -> PreflightReport:
    """Re-read selected executables/resources immediately before dispatch; changes block launch.

    Rechecking is still read-only. It does not itself authorise a real run, claim a launch, or
    remove the small check-to-exec race. The adapter's own executable digest guard remains active.
    """
    if policy is None:
        saved = previous_report.policy
        policy = PreflightPolicy(
            strict_native=saved.get("strict_native", True),
            required_tools=tuple(ToolKind(name) for name in saved.get("required_tools", ("shelxt", "shelxl"))),
            require_platon=saved.get("require_platon", True),
            checkdef_path=Path(saved["checkdef_path"]) if saved.get("checkdef_path") else None,
            validation_resource_binding_verified=saved.get("validation_resource_binding_verified", False),
        )
    current = evaluate_toolchain(toolchain, policy=policy)
    blockers = list(current.blockers)
    if not previous_report.ready:
        blockers.append("previous_preflight_not_ready")
    if current.fingerprint != previous_report.fingerprint:
        blockers.append("dispatch_evidence_changed: selected identity, dependencies, policy, host, or resource changed")
    if blockers:
        return PreflightReport("BLOCKED", current.tools, tuple(dict.fromkeys(blockers)),
                               current.host, current.fingerprint, current.policy)
    return current


def _macho_structure(path: Path) -> list[str]:
    """Bounded independent structure check, including complete load and signature containers."""
    with path.open("rb") as stream:
        size = path.stat().st_size

        def read(offset: int, count: int) -> bytes:
            if offset < 0 or count < 0 or offset + count > size:
                raise ValueError("range outside file")
            stream.seek(offset)
            data = stream.read(count)
            if len(data) != count:
                raise ValueError("file changed or truncated")
            return data

        magic = read(0, 4)
        slices: list[tuple[int, int]] = []
        if magic in (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"):
            count = struct.unpack(">I", read(4, 4))[0]
            if not 0 < count <= 32:
                return ["invalid universal slice count"]
            entry_size = 32 if magic[-1] == 0xBF else 20
            table_end = 8 + count * entry_size
            table = read(8, count * entry_size)
            for index in range(count):
                chunk = table[index * entry_size:(index + 1) * entry_size]
                if entry_size == 32:
                    _cpu, _sub, offset, length, align, _reserved = struct.unpack(">iiQQII", chunk)
                else:
                    _cpu, _sub, offset, length, align = struct.unpack(">iiIII", chunk)
                if offset < table_end or length < 28 or offset + length > size or align > 31:
                    return ["invalid universal slice bounds"]
                if offset % (1 << align):
                    return ["unaligned universal slice"]
                if any(offset < prev + length_prev and prev < offset + length for prev, length_prev in slices):
                    return ["overlapping universal slices"]
                slices.append((offset, length))
        else:
            slices.append((0, size))
        problems: list[str] = []
        for offset, length in slices:
            problems.extend(_slice_structure(read, offset, length))
        return problems


def _slice_structure(read: Any, offset: int, length: int) -> list[str]:
    magic = read(offset, 4)
    if magic in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe"):
        endian = "<"
    elif magic in (b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce"):
        endian = ">"
    else:
        return ["invalid slice magic"]
    is64 = magic in (b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf")
    header_size = 32 if is64 else 28
    if length < header_size:
        return ["truncated slice header"]
    header = read(offset, header_size)
    _cpu, _sub, _type, ncmds, sizeofcmds, _flags = struct.unpack_from(endian + "iiIIII", header, 4)
    if sizeofcmds > _MAX_COMMAND_BYTES or sizeofcmds > length - header_size or ncmds > sizeofcmds // 8:
        return ["load command count or extent invalid"]
    commands = read(offset + header_size, sizeofcmds)
    cursor = 0
    signature_count = 0
    platforms: set[str] = set()
    for _ in range(ncmds):
        if cursor + 8 > len(commands):
            return ["truncated load command list"]
        cmd, command_size = struct.unpack_from(endian + "II", commands, cursor)
        if command_size < 8 or command_size % (8 if is64 else 4) or cursor + command_size > len(commands):
            return ["malformed or unaligned load command"]
        body = commands[cursor:cursor + command_size]
        if cmd in (0xC, 0x80000018, 0x8000001F, 0x20, 0x80000023, 0x8000001C):
            minimum = 12 if cmd == 0x8000001C else 24
            if command_size < minimum:
                return ["truncated dependency/rpath command"]
            string_offset = struct.unpack_from(endian + "I", body, 8)[0]
            if not minimum <= string_offset < len(body) or b"\0" not in body[string_offset:]:
                return ["dependency/rpath string outside command"]
        elif cmd == 0x32:
            if command_size < 24:
                return ["truncated build-version command"]
            platform, _minos, _sdk, ntools = struct.unpack_from(endian + "IIII", body, 8)
            if command_size != 24 + ntools * 8:
                return ["truncated build-version tool records"]
            platforms.add("macos" if platform == 1 else "other")
        elif cmd == 0x24:
            if command_size != 16:
                return ["truncated minimum-macOS command"]
            platforms.add("macos")
        elif cmd == 0x80000028 and command_size != 24:
            return ["truncated main entry command"]
        elif cmd == 0x1D:
            signature_count += 1
            if command_size != 16 or signature_count > 1:
                return ["invalid or duplicate signature command"]
            dataoff, datasize = struct.unpack_from(endian + "II", body, 8)
            if (dataoff < header_size + sizeofcmds or datasize < 12
                    or datasize > _MAX_SIGNATURE_BYTES or dataoff + datasize > length):
                return ["signature extent invalid or truncated"]
            problem = _signature_structure(read(offset + dataoff, datasize))
            if problem:
                return [problem]
        cursor += command_size
    if cursor != sizeofcmds:
        return ["load command byte count does not match header"]
    if platforms != {"macos"}:
        return ["platform absent, unsupported, or inconsistent"]
    return []


def _signature_structure(blob: bytes) -> str | None:
    # Versioned field extents follow Apple's public XNU cs_blobs.h CodeDirectory layout:
    # https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/cs_blobs.h
    magic, declared, count = struct.unpack_from(">III", blob)
    if magic != 0xFADE0CC0 or declared > len(blob) or declared < 12 or count > 64:
        return "signature container invalid or truncated"
    table_end = 12 + count * 8
    if table_end > declared:
        return "signature index table truncated"
    directories = 0
    for index in range(count):
        slot, start = struct.unpack_from(">II", blob, 12 + index * 8)
        if start < table_end or start + 8 > declared:
            return "signature child offset invalid"
        submagic, child_length = struct.unpack_from(">II", blob, start)
        if child_length < 8 or start + child_length > declared:
            return "signature child container truncated"
        if slot == 0:
            directories += 1
            if submagic != 0xFADE0C02 or child_length < 44:
                return "signature code directory invalid"
            version, _flags, hash_offset, identifier_offset, special, code = struct.unpack_from(">IIIIII", blob, start + 8)
            hash_size = blob[start + 36]
            minimum_header = 44
            for introduced, extent in ((0x20100, 48), (0x20200, 52), (0x20300, 64),
                                       (0x20400, 88), (0x20500, 96), (0x20600, 108)):
                if version >= introduced:
                    minimum_header = extent
            if version > 0x20600:
                return "signature code directory version unsupported"
            if child_length < minimum_header:
                return "signature code directory header truncated"
            child = blob[start:start + child_length]
            if not minimum_header <= identifier_offset < child_length or b"\0" not in child[identifier_offset:]:
                return "signature identifier outside directory"
            if special or code:
                if not hash_size or hash_offset - special * hash_size < minimum_header or hash_offset + code * hash_size > child_length:
                    return "signature hash table outside directory"
            if version >= 0x20200:
                team = struct.unpack_from(">I", child, 48)[0]
                if team and (not minimum_header <= team < child_length or b"\0" not in child[team:]):
                    return "signature team identifier outside directory"
    if directories != 1:
        return "signature requires exactly one primary code directory"
    return None
