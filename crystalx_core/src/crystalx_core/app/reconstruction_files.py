"""File-backed refinement/density boundary, exercised by a fixed synthetic subprocess.

No real SHELXL binding is enabled here. The controlled child writes SHELXL-shaped RES/LST
artifacts to exercise the existing adapter/runner and parsers, not least-squares mathematics.
The default assessment leaves scientific checks and convergence unevaluated. Q peaks are a
partial list tied to one exact output and accepted model, never a complete density map.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Callable

from crystalx_core._serde import to_jsonable
from crystalx_core.app.demo import demo_digests
from crystalx_core.domain.tools import (
    BinaryArch, BinaryFormat, BinaryProbe, ResolutionSource, ToolDiagnostic,
    ToolDiagnosticKind, ToolIdentity, ToolKind, ToolResolution,
)
from crystalx_core.io.shelx_res import read_res
from crystalx_core.io.shelxl_lst import ShelxlListing, parse_lst_text
from crystalx_core.refinement.locks import apply_locks, normalize_managed_output
from crystalx_core.refinement.shelxl import RefinementSettings, prepare_refinement_ins
from crystalx_core.tools.adapter import ExternalToolAdapter
from crystalx_core.tools.runner import ProcessRunner, ResourceLimits
from crystalx_core.workspace.hashing import sha256_file
from crystalx_core.workflow.reconstruction import (
    Assessment, DensityObservation, DensityPeak, ModelSnapshot, ReconstructionBlocked,
    RefinementOutcome, RefinementTrial, digest, read_snapshot,
)


# Constant stdlib-only fixture, launched through the same ProcessRunner as external programs.
# No user-controlled program, import path, shell, environment override or command is accepted.
_PROGRAM = r'''
from pathlib import Path
import hashlib, sys
stem, expected_hkl = sys.argv[1:]
hkl = Path(stem + ".hkl")
assert hashlib.sha256(hkl.read_bytes()).hexdigest() == expected_hkl
text = Path(stem + ".ins").read_text()
assert "CRYSTALX-SYNTHETIC-DEMO" in text
cycles = int(next(line.split()[1] for line in text.splitlines() if line.startswith("L.S. ")))
sfac = next(line.split()[1:] for line in text.splitlines() if line.startswith("SFAC "))
atoms = []
for line in text.splitlines():
    row = line.split()
    if row and row[0] == "HKLF":
        break
    if len(row) != 7:
        continue
    try:
        index = int(row[1])
        values = tuple(float(v) for v in row[2:])
    except ValueError:
        continue
    if 1 <= index <= len(sfac):
        atoms.append((row[0], sfac[index - 1]))
assert atoms
bad = any(label.startswith("X") and element == "O" for label, element in atoms)
r = .05 if bad else .45 - .04 * sum(e == "N" for _, e in atoms) - .03 * (len(atoms) - 2)
w, g, n = 2 * r, 1 + r / 10, 4 * len(atoms)
summary = (f"R1 = {r:.6f} for 800 Fo > 4sig(Fo) and {r:.6f} for all 1000 data\n"
           f"wR2 = {w:.6f}, GooF = S = {g:.6f}, Restrained GooF = {g:.6f} for all data\n")
rem = "".join("REM " + line + "\n" for line in summary.splitlines())
rem += f"REM {n} parameters refined using 0 restraints\n"
peak = "" if any(label.startswith("X") for label, _ in atoms) else "Q1 1 0.45 0.50 0.35 11 0.05 2.0\n"
Path(stem + ".res").write_text(text + rem + peak)
Path(stem + ".lst").write_text("SYNTHETIC SOFTWARE FIXTURE: no least-squares calculation\n"
    + summary + f"{n} / {n} parameters\nLeast-squares cycle {cycles}\n"
    + "Mean shift/su = 0.001 Maximum = 0.002 for C1\nHighest peak 2.0 Deepest hole -0.5\n")
'''

_CHECKS = ("geometry", "coordination", "occupancy_adp", "chirality_symmetry", "restraints",
           "heavy_site_drift", "residual_distribution")
_MAX_OUTPUT_BYTES = 8 * 1024 * 1024


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _regular(path: Path, root: Path, *, maximum: int = _MAX_OUTPUT_BYTES) -> bytes:
    """Read one bounded regular file; refuse symlinks, traversal and mutable read results."""
    if path.parent.resolve(strict=True) != root.resolve(strict=True):
        raise ReconstructionBlocked("refinement_file_outside_trial_directory")
    named = path.lstat()
    if not stat.S_ISREG(named.st_mode) or named.st_nlink != 1 or named.st_size > maximum:
        raise ReconstructionBlocked("refinement_file_must_be_bounded_unaliased_regular_file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0))
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum
                or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)):
            raise ReconstructionBlocked("refinement_file_must_be_bounded_unaliased_regular_file")
        data = source.read(maximum + 1)
    after = path.lstat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or not stat.S_ISREG(after.st_mode) or after.st_nlink != 1 or len(data) > maximum):
        raise ReconstructionBlocked("refinement_file_changed_during_read")
    return data


@dataclass(frozen=True)
class FileRefinementEvidence:
    model_sha256: str
    basis_key: str
    hkl_path: Path
    hkl_sha256: str
    raw_res_path: Path
    raw_res_sha256: str
    lst_path: Path
    lst_sha256: str
    invocation_path: Path
    invocation_sha256: str
    normalized_path: Path
    peaks: tuple[DensityPeak, ...]
    synthetic: bool = True

    @property
    def source_sha256(self) -> str:
        return digest((self.raw_res_sha256, self.lst_sha256, self.invocation_sha256,
                       self.hkl_sha256, self.model_sha256, self.peaks))

    def verify(self) -> None:
        root = self.raw_res_path.parent
        if self.synthetic is not True or root.is_symlink() or root.resolve(strict=True) != root.absolute():
            raise ReconstructionBlocked("controlled_file_evidence_cannot_be_relabelled_or_relocated")
        for path, expected in ((self.raw_res_path, self.raw_res_sha256), (self.lst_path, self.lst_sha256),
                               (self.invocation_path, self.invocation_sha256),
                               (self.normalized_path, self.model_sha256), (self.hkl_path, self.hkl_sha256)):
            if _hash_bytes(_regular(path, root)) != expected:
                raise ReconstructionBlocked("registered_refinement_evidence_changed")
        parsed = read_res(self.raw_res_path)
        actual = tuple(DensityPeak(p.label, (p.x, p.y, p.z), p.height) for p in parsed.q_peaks)
        model = read_snapshot(self.normalized_path, reflection_sha256=self.hkl_sha256)
        if (parsed.sha256 != self.raw_res_sha256 or actual != self.peaks
                or model.sha256 != self.model_sha256 or model.basis_key != self.basis_key):
            raise ReconstructionBlocked("registered_density_must_match_raw_peaks_and_normalized_model")


class FileEvidenceStore:
    """Exact model lookup only; no latest-file search or alternative trial substitution."""

    def __init__(self):
        self._by_model: dict[tuple[str, str], FileRefinementEvidence] = {}

    def register(self, evidence: FileRefinementEvidence) -> None:
        evidence.verify()
        self._by_model[(evidence.model_sha256, evidence.basis_key)] = evidence

    def for_model(self, model: ModelSnapshot) -> FileRefinementEvidence:
        evidence = self._by_model.get((model.sha256, model.basis_key))
        if evidence is None:
            raise ReconstructionBlocked("density_has_no_registered_output_for_this_model")
        evidence.verify()
        return evidence


class QPeakDensity:
    def __init__(self, evidence: FileEvidenceStore):
        self.evidence = evidence

    def observe(self, model: ModelSnapshot) -> DensityObservation:
        source = self.evidence.for_model(model)
        observation = DensityObservation(model.sha256, model.basis_key, source.source_sha256,
            "SYNTHETIC_PEAK_LIST" if source.synthetic else "Q_PEAK_LIST", source.peaks, "partial")
        observation.validate_for(model, synthetic=source.synthetic)
        return observation


SyntheticAssessor = Callable[[ModelSnapshot, ShelxlListing, FileRefinementEvidence], Assessment]


class ShelxlFileRefinement:
    """Use real subprocess/file/constraint contracts with an explicitly synthetic program.

    Real mode refuses before opening inputs. A future real binding needs verified SHELXL output
    syntax and independently validated scientific evaluators; this class grants neither.
    """

    def __init__(self, evidence: FileEvidenceStore, *, synthetic_assessor: SyntheticAssessor | None = None,
                 runner: ProcessRunner | None = None):
        self.evidence, self.synthetic_assessor = evidence, synthetic_assessor
        executable = Path(sys.executable).resolve(strict=True)
        info = executable.stat()
        identity = ToolIdentity(ToolKind.SHELXL, executable, executable, ResolutionSource.SYNTHETIC_DEMO,
            info.st_size, sha256_file(executable), info.st_mtime,
            BinaryProbe(BinaryFormat.UNKNOWN, BinaryArch.UNKNOWN),
            version="controlled Python fixture; not SHELXL")
        self._adapter = ExternalToolAdapter(ToolResolution(ToolKind.SHELXL, identity,
            ToolDiagnostic(ToolDiagnosticKind.OK, "Controlled synthetic file fixture")), runner=runner,
            strict_regular_files=True)

    def refine(self, request: RefinementTrial) -> RefinementOutcome:
        if not request.synthetic:
            raise ReconstructionBlocked("real_file_refinement_dispatch_disabled")
        try:
            return self._refine_synthetic(request)
        except Exception as error:
            # Retain a local, nonsensitive stop even when strict snapshots reject an unsafe
            # filesystem entry before an invocation record can be assembled. Never read it.
            root = request.work_directory
            if root.is_dir() and not root.is_symlink() and root.resolve(strict=True) == root.absolute():
                target = root / "file-failure.json"
                try:
                    with target.open("x", encoding="utf-8") as destination:
                        json.dump({"exception_type": type(error).__name__, "synthetic": True,
                                   "scientifically_reviewed": False}, destination)
                except OSError:
                    pass
            raise

    def _refine_synthetic(self, request: RefinementTrial) -> RefinementOutcome:
        if request.cancel.cancelled:
            raise ReconstructionBlocked("cancelled")
        if (isinstance(request.timeout_s, bool) or not math.isfinite(request.timeout_s)
                or request.timeout_s <= 0 or type(request.cycles) is not int or not 1 <= request.cycles <= 50):
            raise ReconstructionBlocked("invalid_refinement_resource_bounds")
        root = request.work_directory
        if root.is_symlink() or root.resolve(strict=True) != root.absolute():
            raise ReconstructionBlocked("trial_directory_must_be_canonical")
        if request.input_path != root / "trial.ins" or request.hkl_path != root / "trial.hkl":
            raise ReconstructionBlocked("unexpected_refinement_input_names")
        ins_data = _regular(request.input_path, root)
        hkl_data = _regular(request.hkl_path, root)
        hkl_sha = _hash_bytes(hkl_data)
        if hkl_sha != demo_digests()[1]:
            raise ReconstructionBlocked("synthetic_refinement_requires_builtin_generated_reflections")
        baseline, _, _, _ = prepare_refinement_ins(request.proposed_model.text,
            RefinementSettings(cycles=request.cycles, timeout_s=request.timeout_s))
        application = apply_locks(baseline, request.active_locks)
        if _hash_bytes(ins_data) != application.output_sha256:
            raise ReconstructionBlocked("refinement_ins_differs_from_explicit_model_and_lock_ledger")
        for name in ("trial.res", "trial.lst", "file-invocation.json", "lock-free-model.res", "file-evidence.json"):
            if (root / name).exists() or (root / name).is_symlink():
                raise ReconstructionBlocked("refinement_outputs_must_be_fresh")
        if sha256_file(self._adapter.identity.path) != self._adapter.identity.sha256:
            raise ReconstructionBlocked("controlled_interpreter_changed")
        loader_environment = tuple(sorted({key for key in os.environ if key.startswith(("DYLD_", "LD_"))}
            | {"LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH"}))
        plan = self._adapter.build_plan(args=("-I", "-S", "-c", _PROGRAM, "trial", hkl_sha), cwd=root,
            timeout_s=request.timeout_s, expected_outputs=("trial.res", "trial.lst"),
            declared_inputs=("trial.ins", "trial.hkl"), label="synthetic-refinement-file-fixture",
            env_unset=loader_environment,
            limits=ResourceLimits(cpu_seconds=max(1, math.ceil(request.timeout_s)),
                                  max_file_bytes=_MAX_OUTPUT_BYTES, max_open_files=128))
        outcome, record = self._adapter.invoke(plan, log_dir=root / "process-logs", cancel=request.cancel)
        invocation = root / "file-invocation.json"
        invocation.write_text(json.dumps(to_jsonable(record), sort_keys=True, indent=2))
        if (sha256_file(self._adapter.identity.path) != self._adapter.identity.sha256
                or _regular(request.input_path, root) != ins_data or _regular(request.hkl_path, root) != hkl_data):
            raise ReconstructionBlocked("refinement_process_changed_pinned_inputs_or_interpreter")
        if request.cancel.cancelled or record.cancelled:
            raise ReconstructionBlocked("cancelled")
        if not outcome.succeeded or not record.succeeded:
            raise ReconstructionBlocked("refinement_subprocess_failed:" + record.diagnostic)
        raw, listing_path = root / "trial.res", root / "trial.lst"
        raw_data, lst_data = _regular(raw, root), _regular(listing_path, root)
        expected_inputs = {"trial.ins": _hash_bytes(ins_data), "trial.hkl": hkl_sha}
        if (record.input_hashes != expected_inputs
                or record.output_hashes.get("trial.res") != _hash_bytes(raw_data)
                or record.output_hashes.get("trial.lst") != _hash_bytes(lst_data)):
            raise ReconstructionBlocked("refinement_invocation_file_identity_mismatch")
        normalized = normalize_managed_output(raw_data.decode("utf-8"), application,
            baseline_ins_text=baseline, active_locks=request.active_locks)
        model_path = root / "lock-free-model.res"
        model_path.write_text(normalized.text)
        model = read_snapshot(model_path, reflection_sha256=hkl_sha)
        if (model.basis_key != request.proposed_model.basis_key or model.sfac != request.proposed_model.sfac
                or tuple((s.site_id, s.element) for s in model.sites)
                != tuple((s.site_id, s.element) for s in request.proposed_model.sites)):
            raise ReconstructionBlocked("refinement_output_changed_model_identity_or_settings")
        listing = parse_lst_text(lst_data.decode("utf-8"), source=listing_path)
        parsed = read_res(raw)
        if parsed.sha256 != _hash_bytes(raw_data) or listing.has_fatal_error:
            raise ReconstructionBlocked("refinement_output_changed_or_fatal_listing")
        for key in ("r1_observed", "r1_all", "n_observed", "n_all", "wr2", "goof"):
            first, second = getattr(parsed.r_factors, key), getattr(listing, key)
            if first is None or second is None or not math.isfinite(first) or abs(first - second) > 1e-8:
                raise ReconstructionBlocked("refinement_metrics_missing_or_disagree:" + key)
        if (not listing.parameters_refined or parsed.r_factors.n_parameters != listing.parameters_refined
                or not listing.n_all or not listing.n_observed):
            raise ReconstructionBlocked("refinement_parameter_or_reflection_counts_missing_or_disagree")
        if listing.cycles_run != request.cycles:
            raise ReconstructionBlocked("synthetic_listing_cycle_count_differs_from_requested_bound")
        peaks = tuple(DensityPeak(p.label, (p.x, p.y, p.z), p.height) for p in parsed.q_peaks)
        if len(peaks) > 200:
            raise ReconstructionBlocked("refinement_peak_list_size_exceeded")
        evidence = FileRefinementEvidence(model.sha256, model.basis_key, request.hkl_path, hkl_sha,
            raw, _hash_bytes(raw_data), listing_path, _hash_bytes(lst_data), invocation,
            sha256_file(invocation), model_path, peaks)
        DensityObservation(model.sha256, model.basis_key, evidence.source_sha256,
            "SYNTHETIC_PEAK_LIST", peaks).validate_for(model, synthetic=True)
        (root / "file-evidence.json").write_text(json.dumps(to_jsonable({"evidence": evidence,
            "listing": listing.to_record(), "normalization": normalized,
            "program_sha256": _hash_bytes(_PROGRAM.encode()), "scientifically_reviewed": False}),
            indent=2, sort_keys=True))
        assessment = Assessment(model.basis_key, listing.r1_observed, listing.wr2, listing.goof,
            (), listing.n_all, listing.parameters_refined,
            tuple((name, "NOT_EVALUATED") for name in _CHECKS), evidence.source_sha256,
            requires_review=True, least_squares_converged=None, model_sha256=model.sha256)
        if self.synthetic_assessor is not None:
            assessment = self.synthetic_assessor(model, listing, evidence)
            if (not isinstance(assessment, Assessment) or assessment.scientific_validation
                    or assessment.model_sha256 != model.sha256 or assessment.basis_key != model.basis_key
                    or assessment.source_sha256 != evidence.source_sha256
                    or (assessment.r1, assessment.wr2, assessment.goof, assessment.n_reflections, assessment.n_parameters)
                    != (listing.r1_observed, listing.wr2, listing.goof, listing.n_all, listing.parameters_refined)):
                raise ReconstructionBlocked("synthetic_assessor_changed_file_facts_or_claimed_science")
        # Evaluators are another mutation boundary, including on an exception (no registration).
        evidence.verify()
        if _regular(request.input_path, root) != ins_data or request.cancel.cancelled:
            raise ReconstructionBlocked("cancelled" if request.cancel.cancelled else "assessor_changed_refinement_input")
        self.evidence.register(evidence)
        return RefinementOutcome(normalized.text, assessment, "synthetic",
                                 review_required=assessment.requires_review)
