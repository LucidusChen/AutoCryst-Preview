"""Independent, local PLATON diagnostics on an already sealed candidate.

This is not the IUCr online checkCIF service. No HTTP client or upload is used.
Solving and reconstruction never call this module implicitly. A diagnostic
failure or an ALERT leaves the sealed raw candidate available for inspection.
The existing PLATON parser remains format-unverified; no zero-alert PASS or
scientific-acceptance claim is manufactured from process success.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path

from crystalx_core._serde import to_jsonable
from crystalx_core.app.models import RunMode
from crystalx_core.app.toolchain import Toolchain
from crystalx_core.private_acceptance.sealing import (
    SealError, read_private_reference, read_sealed_output, verify_seal,
)
from crystalx_core.tools.runner import CancelToken, ResourceLimits
from crystalx_core.validation.platon import PlatonReport, PlatonSettings, parse_chk_text, validation_prerequisites


class CheckCifStatus(str, Enum):
    NOT_EVALUATED = "NOT_EVALUATED"
    FAILED = "FAILED"
    ALERTS_RECORDED = "ALERTS_RECORDED"


@dataclass(frozen=True)
class PinnedArtifact:
    path: Path
    sha256: str


@dataclass(frozen=True)
class CheckCifResult:
    status: CheckCifStatus
    reasons: tuple[str, ...]
    candidate_available: bool
    seal_sha256: str | None
    candidate_sha256: str | None = None
    output_directory: Path | None = None
    report: PlatonReport | None = None
    process_verified: bool = False
    invocations: int = 0  # Adapter attempts; process_verified is separate startup/output evidence.
    program: str = ""
    executable_sha256: str = ""
    scientifically_reviewed: bool = False
    official_iucr_online_check: bool = False

    @property
    def counts(self) -> dict[str, int]:
        return self.report.counts if self.report else {key: 0 for key in "ABCG"}

    def to_record(self) -> dict:
        result = to_jsonable(asdict(self))
        result["status"] = self.status.value
        result["alert_counts"] = self.counts
        return result


def _fresh_private_directory(path: Path, private_root: Path) -> Path:
    target, root = Path(path).absolute(), Path(private_root).absolute()
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise SealError("Independent diagnostic outputs must remain private") from exc
    if not relative.parts or ".." in target.parts or root.resolve(strict=True) != root:
        raise SealError("Independent diagnostic root is unsafe")
    if target.parent.resolve(strict=True) != target.parent:
        raise SealError("Independent diagnostic parent is unsafe")
    target.mkdir(mode=0o700, exist_ok=False)
    return target


def run_local_checkcif(run_root: Path, *, private_root: Path, expected_seal_sha256: str,
        output_dir: Path, toolchain: Toolchain, settings: PlatonSettings | None = None,
        model_role: str = "model", cif_role: str = "cif", fcf_role: str = "fcf",
        checkdef: PinnedArtifact | None = None, allow_builtin_checkdef: bool = False,
        execute: bool = False, cancel: CancelToken | None = None) -> CheckCifResult:
    """Inspect or explicitly run a separate, bounded local diagnostic.

    Candidate/CIF/FCF must be in the same externally pinned seal. This establishes
    file identity, not their scientific compatibility. No answer is accepted.
    ``execute=True`` is the call boundary for the owner's applicable execution
    authority; it is never inferred from the presence of an executable. Supplying
    check.def or accepting documented built-in generation is explicit. Version
    compatibility and output dialect remain unverified until actually reviewed.
    """
    settings = settings or PlatonSettings()
    if (isinstance(settings.timeout_s, bool) or not math.isfinite(settings.timeout_s)
            or not 0 < settings.timeout_s <= 3600):
        raise ValueError("Independent PLATON wall-clock bound must be in (0, 3600]")
    cancel = cancel or CancelToken()
    try:
        seal = verify_seal(run_root, private_root=private_root, expected_seal_sha256=expected_seal_sha256)
        model = read_sealed_output(seal, private_root=private_root, role=model_role)
        if not model:
            raise SealError("Empty candidate")
        model_sha = hashlib.sha256(model).hexdigest()
    except (SealError, OSError):
        return CheckCifResult(CheckCifStatus.NOT_EVALUATED, ("sealed_candidate_identity_failed",), False, None)

    def result(status, reasons, *, directory=None, report=None, verified=False, calls=0, adapter=None, available=True):
        return CheckCifResult(status, tuple(reasons), available, seal.seal_sha256, model_sha, directory,
            report, verified, calls, adapter.kind.value if adapter else "",
            adapter.identity.sha256 if adapter else "")

    try:
        cif = read_sealed_output(seal, private_root=private_root, role=cif_role)
        fcf = read_sealed_output(seal, private_root=private_root, role=fcf_role)
    except (SealError, OSError):
        return result(CheckCifStatus.NOT_EVALUATED, ("sealed_cif_and_fcf_required_no_uncontrolled_shelxl_regeneration",))
    if not cif or not fcf:
        return result(CheckCifStatus.NOT_EVALUATED, ("nonempty_cif_and_fcf_required",))
    if not execute:
        return result(CheckCifStatus.NOT_EVALUATED, ("independent_local_validation_not_explicitly_requested",))
    if toolchain.mode is not RunMode.REAL:
        return result(CheckCifStatus.NOT_EVALUATED, ("simulated_toolchain_cannot_establish_local_platon_validation",))
    if cancel.cancelled:
        return result(CheckCifStatus.NOT_EVALUATED, ("cancelled_before_independent_validation",))
    if checkdef is None and not allow_builtin_checkdef:
        return result(CheckCifStatus.NOT_EVALUATED, ("explicit_checkdef_or_accepted_builtin_resource_policy_required",))
    adapter = toolchain.platon_adapter()
    if adapter is None:
        return result(CheckCifStatus.NOT_EVALUATED, ("selected_platon_executable_not_usable",))
    try:
        checkdef_bytes = None
        if checkdef is not None:
            checkdef_bytes, actual = read_private_reference(checkdef.path, case_root=checkdef.path.parent)
            if actual != checkdef.sha256 or not checkdef_bytes:
                raise SealError("check.def pin differs")
        directory = _fresh_private_directory(output_dir, private_root)
        for name, content in (("candidate.cif", cif), ("candidate.fcf", fcf)):
            with (directory / name).open("xb") as stream:
                stream.write(content)
        inputs = {"candidate.cif": hashlib.sha256(cif).hexdigest(), "candidate.fcf": hashlib.sha256(fcf).hexdigest()}
        if checkdef_bytes is not None:
            with (directory / "check.def").open("xb") as stream:
                stream.write(checkdef_bytes)
            inputs["check.def"] = checkdef.sha256
        if validation_prerequisites(directory, "candidate"):
            raise SealError("Validation prerequisites disappeared")
        verify_seal(run_root, private_root=private_root, expected_seal_sha256=seal.seal_sha256)
    except (SealError, OSError, ValueError):
        return result(CheckCifStatus.NOT_EVALUATED, ("independent_staging_or_resource_identity_failed",), adapter=adapter)
    limits = settings.limits or ResourceLimits(cpu_seconds=max(1, math.ceil(settings.timeout_s)),
                                               max_file_bytes=64 * 1024 * 1024, max_open_files=128)
    plan = adapter.build_plan(args=("-u", "candidate.cif"), cwd=directory,
        timeout_s=settings.timeout_s, expected_outputs=("candidate.chk",), declared_inputs=tuple(inputs),
        env_unset=("DISPLAY", "CHECKDEF"), label="independent-local-checkcif", limits=limits)
    (directory / "command.json").write_text(json.dumps(to_jsonable(plan.to_record()), sort_keys=True, indent=2))
    report = None
    try:
        outcome, record = adapter.invoke(plan, log_dir=directory, cancel=cancel)
        (directory / "invocation.json").write_text(json.dumps(to_jsonable(record), sort_keys=True, indent=2))
        verify_seal(run_root, private_root=private_root, expected_seal_sha256=seal.seal_sha256)
        if (record.input_hashes != inputs or record.executable_sha256 != adapter.identity.sha256
                or record.tool != adapter.kind or record.argv != plan.argv or record.cwd != str(directory)):
            raise SealError("Invocation input identity differs")
        for name, expected in inputs.items():
            _, actual = read_private_reference(directory / name, case_root=directory)
            if actual != expected:
                raise SealError("Validation changed a staged input")
        chk = directory / "candidate.chk"
        if chk.exists() or chk.is_symlink():
            content, chk_sha = read_private_reference(chk, case_root=directory)
            if record.output_hashes.get("candidate.chk") != chk_sha:
                raise SealError("Report identity differs from retained process output")
            report = parse_chk_text(content.decode("utf-8", errors="replace"), source=chk)
        verified = bool(outcome.succeeded and record.succeeded and not outcome.cancelled and not record.cancelled)
        if not verified:
            answer = result(CheckCifStatus.FAILED, ("independent_platon_process_failed_or_cancelled",),
                            directory=directory, report=report, calls=1, adapter=adapter)
        elif report is None or not report.lines_read:
            answer = result(CheckCifStatus.FAILED, ("independent_platon_report_missing_or_empty",),
                            directory=directory, calls=1, verified=True, adapter=adapter)
        elif report.alerts:
            answer = result(CheckCifStatus.ALERTS_RECORDED,
                ("alerts_require_expert_review", "local_chk_parser_format_remains_unverified"),
                directory=directory, report=report, calls=1, verified=True, adapter=adapter)
        else:
            answer = result(CheckCifStatus.NOT_EVALUATED,
                ("no_recognized_alerts_is_not_a_validation_pass", "local_chk_parser_format_remains_unverified"),
                directory=directory, report=report, calls=1, verified=True, adapter=adapter)
    except Exception:
        # Program exceptions can include private CIF text. Retain local files,
        # but do not copy exception details into ordinary progress summaries.
        available = True
        try:
            verify_seal(run_root, private_root=private_root, expected_seal_sha256=seal.seal_sha256)
        except (SealError, OSError):
            available = False
        answer = result(CheckCifStatus.FAILED, ("independent_validation_evidence_or_process_failed",),
                        directory=directory, report=report, calls=1, adapter=adapter, available=available)
    (directory / "result.json").write_text(json.dumps(answer.to_record(), sort_keys=True, indent=2))
    return answer
