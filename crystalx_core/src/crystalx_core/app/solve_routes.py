"""Initial-solution routes, using the application's existing tool adapters.

This module does not run CrystalX assignment or a refinement loop. Protein experimental phasing
requires explicitly supplied experiment inputs; an ordinary small-molecule HKL is not promoted to
SAD/MAD data. SHELXE's trace is a PDB, while its SHELXD input RES is only a substructure.

Command/input contracts were checked against the vendor's primary documentation:
https://shelx.uni-goettingen.de/shelxc_keywords.php
https://shelx.uni-goettingen.de/shelxe_keywords.php
https://shelx.uni-goettingen.de/shelxs_keywords.php
https://shelx.uni-goettingen.de/shelxd_sm_keywords.php
These checks establish command semantics, not validation against the owner's installations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import json
import math
from pathlib import Path
import re
import shutil
import stat
import time
from typing import Any, Callable, Mapping

from crystalx_core.app.staging import safe_stem
from crystalx_core.app.toolchain import Toolchain
from crystalx_core.domain.chemistry import ATOMIC_NUMBERS
from crystalx_core.domain.symmetry import ChiralityCompatibility, analyse_symmetry
from crystalx_core.domain.tools import ToolKind
from crystalx_core.errors import CrystalXError
from crystalx_core.io.shelx_reader import logical_lines, read_ins_header, scan_hkl
from crystalx_core.io.shelx_res import read_res
from crystalx_core.tools.runner import CancelToken, ResourceLimits
from crystalx_core.tools.adapter import ExternalToolAdapter
from crystalx_core.refinement.shelxl import _is_atom_line, _logical_from_text
from crystalx_core.workspace.hashing import sha256_file

__all__ = ["InitialStrategy", "PhasingDataset", "ProteinPhasingConfig", "InitialRouteConfig",
           "RouteStep", "InitialRoutePlan", "InitialRouteResult", "ProteinConversionRequest",
           "plan_initial_route", "execute_initial_route", "RouteConfigError"]

MAX_GENERATED_ARTIFACT_BYTES = 32 * 1024 * 1024


class RouteConfigError(ValueError):
    pass


class InitialStrategy(str, Enum):
    SHELXT = "shelxt"
    PROTEIN_CDE = "protein_cde"
    SMALL_MOLECULE = "small_molecule"


def _strict(data: Any, keys: set[str], required: set[str] = frozenset()) -> dict:
    if not isinstance(data, dict) or set(data) - keys or required - set(data):
        raise RouteConfigError("strategy configuration has missing or unknown fields")
    return data


def _positive(value, label, *, integer=False):
    if (type(value) not in ({int} if integer else {int, float})
            or not math.isfinite(value) or value <= 0):
        raise RouteConfigError(f"{label} must be finite and positive")


@dataclass(frozen=True)
class PhasingDataset:
    role: str
    path: Path
    format: str = "SHELX_INTENSITIES"
    friedel_pairs_retained: bool = False

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path).absolute())
        if self.role not in {"SAD", "PEAK", "INFL", "HREM", "LREM", "NAT"}:
            raise RouteConfigError("unsupported experimental dataset role")
        if self.format not in {"SHELX_INTENSITIES", "SHELX_AMPLITUDES", "XDS_ASCII", "SCALEPACK", "MTZ"}:
            raise RouteConfigError("unsupported experimental input format")
        if type(self.friedel_pairs_retained) is not bool:
            raise RouteConfigError("Friedel-pair declaration must be boolean")

    def to_dict(self):
        return {**asdict(self), "path": str(self.path)}

    @classmethod
    def from_dict(cls, data):
        return cls(**_strict(data, set(cls.__dataclass_fields__), {"role", "path"}))


@dataclass(frozen=True)
class ProteinPhasingConfig:
    protein_confirmed: bool = False
    experiment: str = "SAD"
    space_group: str = ""
    find_sites: int | None = None
    heavy_element: str = ""
    solvent_fraction: float | None = None
    datasets: tuple[PhasingDataset, ...] = ()
    trace_cycles: int = 3
    density_cycles: int = 20
    heavy_atoms_in_native: bool = True
    preferred_hand: str = "auto"

    def __post_init__(self):
        object.__setattr__(self, "datasets", tuple(self.datasets))
        if type(self.protein_confirmed) is not bool or type(self.heavy_atoms_in_native) is not bool:
            raise RouteConfigError("protein/native-heavy-atom declarations must be boolean")
        if self.experiment not in {"SAD", "MAD"}:
            raise RouteConfigError("this adapter implements explicit SAD/MAD configurations only")
        if not isinstance(self.space_group, str) or any(c in self.space_group for c in "\r\n\x00"):
            raise RouteConfigError("space group must be a single instruction value")
        if self.find_sites is not None:
            _positive(self.find_sites, "find_sites", integer=True)
        if self.heavy_element and self.heavy_element not in ATOMIC_NUMBERS:
            raise RouteConfigError("unknown heavy element")
        if self.solvent_fraction is not None and (type(self.solvent_fraction) not in {float, int}
                or not math.isfinite(self.solvent_fraction) or not 0 < self.solvent_fraction < 1):
            raise RouteConfigError("solvent_fraction must lie strictly between zero and one")
        _positive(self.trace_cycles, "trace_cycles", integer=True)
        _positive(self.density_cycles, "density_cycles", integer=True)
        if self.preferred_hand not in {"auto", "original", "inverted"}:
            raise RouteConfigError("invalid SHELXE hand selection")
        if any(not isinstance(dataset, PhasingDataset) for dataset in self.datasets):
            raise RouteConfigError("experimental datasets require typed declarations")

    def to_dict(self):
        return {**asdict(self), "datasets": [dataset.to_dict() for dataset in self.datasets]}

    @classmethod
    def from_dict(cls, data):
        obj = dict(_strict(data, set(cls.__dataclass_fields__)))
        obj["datasets"] = tuple(PhasingDataset.from_dict(item) for item in obj.get("datasets", []))
        return cls(**obj)


@dataclass(frozen=True)
class InitialRouteConfig:
    strategy: InitialStrategy = InitialStrategy.SHELXT
    small_solver: str = "shelxs"
    small_method: str = "direct"
    trials: int = 100
    find_sites: int | None = None
    plop: tuple[int, ...] = ()
    protein: ProteinPhasingConfig | None = None
    timeout_s: float = 600.0

    def __post_init__(self):
        object.__setattr__(self, "strategy", InitialStrategy(self.strategy))
        object.__setattr__(self, "plop", tuple(self.plop))
        if self.small_solver not in {"shelxs", "shelxd"} or self.small_method not in {"direct", "patterson"}:
            raise RouteConfigError("invalid small-molecule solver/method")
        _positive(self.trials, "trials", integer=True)
        _positive(self.timeout_s, "timeout_s")
        if self.find_sites is not None:
            _positive(self.find_sites, "find_sites", integer=True)
        if len(self.plop) > 10:
            raise RouteConfigError("SHELXD accepts at most ten PLOP counts")
        for count in self.plop:
            _positive(count, "PLOP count", integer=True)
        if self.protein is not None and not isinstance(self.protein, ProteinPhasingConfig):
            raise RouteConfigError("protein configuration must be typed")

    def to_dict(self):
        return {**asdict(self), "strategy": self.strategy.value,
                "plop": list(self.plop), "protein": self.protein.to_dict() if self.protein else None}

    @classmethod
    def from_dict(cls, data):
        obj = dict(_strict(data, set(cls.__dataclass_fields__)))
        if obj.get("protein") is not None:
            obj["protein"] = ProteinPhasingConfig.from_dict(obj["protein"])
        return cls(**obj)


@dataclass(frozen=True)
class RouteStep:
    name: str
    tool: ToolKind
    args: tuple[str, ...]
    declared_inputs: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    stdin_script: str | None = None

    def to_dict(self):
        return {**asdict(self), "tool": self.tool.value}


@dataclass(frozen=True)
class InitialRoutePlan:
    config: InitialRouteConfig
    ins_path: Path
    hkl_path: Path
    work_dir: Path
    stem: str
    input_hashes: tuple[tuple[str, str], ...]
    staged_sources: tuple[tuple[str, Path, str], ...]
    prepared_ins: str | None
    steps: tuple[RouteStep, ...]
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self):
        return {"schema": "crystalx.initial_route_plan/1", "strategy": self.config.to_dict(),
                "ins_path": str(self.ins_path), "hkl_path": str(self.hkl_path), "work_dir": str(self.work_dir),
                "stem": self.stem, "input_hashes": dict(self.input_hashes),
                "staged_sources": [{"name": name, "source": str(path), "sha256": sha} for name, path, sha in self.staged_sources],
                "prepared_ins": self.prepared_ins, "steps": [step.to_dict() for step in self.steps],
                "blockers": list(self.blockers), "warnings": list(self.warnings),
                "assignment_performed": False, "refinement_loop_performed": False}


@dataclass(frozen=True)
class ProteinConversionRequest:
    pdb_path: Path
    paired_hkl_path: Path
    original_ins_path: Path
    work_dir: Path
    hand: str
    declared_space_group: str
    deadline: float | None = None
    cancel: CancelToken | None = None


@dataclass(frozen=True)
class InitialRouteResult:
    status: str
    strategy: InitialStrategy
    seed_path: Path | None = None
    seed_format: str | None = None
    paired_hkl_path: Path | None = None
    seed_sha256: str | None = None
    paired_hkl_sha256: str | None = None
    invocations: tuple[dict[str, Any], ...] = ()
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_calls(self) -> int:
        return int(self.evidence.get("attempted_invocations", len(self.invocations)))

    @property
    def duration_s(self) -> float:
        return sum(float(item.get("duration_s", 0)) for item in self.invocations)

    def to_dict(self):
        return {**asdict(self), "strategy": self.strategy.value,
                "seed_path": str(self.seed_path) if self.seed_path else None,
                "paired_hkl_path": str(self.paired_hkl_path) if self.paired_hkl_path else None,
                "assignment_performed": False, "refinement_loop_performed": False}


def _source(path: Path, name: str):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise RouteConfigError("input must be an explicit regular local file")
    return name, path, sha256_file(path)


def _small_instructions(text: str, config: InitialRouteConfig) -> str:
    controls = {"TREF", "PATT", "TEXP", "FIND", "PLOP", "NTRY", "PATS", "GROP", "L.S.", "CGLS"}
    lines = [line for logical in _logical_from_text(text.splitlines())
             if logical.instruction not in controls for line in logical.raw_lines]
    location = next((n for n, line in enumerate(lines) if line.split() and line.split()[0].upper() == "HKLF"), None)
    if location is None:
        raise RouteConfigError("small-molecule input needs an explicit HKLF instruction")
    if config.small_solver == "shelxs":
        inserted = [f"{'PATT' if config.small_method == 'patterson' else 'TREF'} {config.trials}"]
    else:
        inserted = (["PATS"] if config.small_method == "patterson" else [])
        inserted += [f"FIND {config.find_sites}", "PLOP " + " ".join(map(str, config.plop)), f"NTRY {config.trials}"]
    return "\n".join([*lines[:location], *inserted, *lines[location:]]) + "\n"


def plan_initial_route(config: InitialRouteConfig, *, ins_path: Path, hkl_path: Path,
                       work_dir: Path) -> InitialRoutePlan:
    """Read explicit inputs and form a serializable plan; do not write or resolve/run programs."""
    ins_path, hkl_path, work_dir = Path(ins_path).absolute(), Path(hkl_path).absolute(), Path(work_dir).absolute()
    sources = [_source(ins_path, "original.ins"), _source(hkl_path, "original.hkl")]
    header = read_ins_header(ins_path)
    initial_controls = {"FIND", "PLOP", "PATS", "GROP", "NTRY", "PSMF", "TEST", "ESEL", "MIND"}
    logical_input = tuple(logical_lines(ins_path))
    if (any((_is_atom_line(line.text) and line.instruction not in initial_controls)
            or line.instruction in {"ATOM", "HETATM"} for line in logical_input)
            or any(line.instruction.startswith("+") for line in logical_input)):
        raise RouteConfigError("initial solving needs atom-free instructions without external file includes")
    stem = "solve_" + safe_stem(ins_path.name)
    blockers, warnings, steps = [], [], []
    prepared = None
    if header.cell is None:
        blockers.append("unit_cell_missing")
    elif not all(math.isfinite(value) for value in (header.cell.a, header.cell.b, header.cell.c,
                                                    header.cell.alpha, header.cell.beta, header.cell.gamma)):
        blockers.append("unit_cell_nonfinite")
    if config.strategy is InitialStrategy.SHELXT:
        prepared = ins_path.read_text(encoding="utf-8")
        steps.append(RouteStep("shelxt", ToolKind.SHELXT, (stem,), (stem + ".ins", stem + ".hkl"),
                               (stem + "_a.res", stem + "_a.hkl")))
    elif config.strategy is InitialStrategy.SMALL_MOLECULE:
        if header.latt is None:
            blockers.append("fixed_space_group_instructions_missing")
        if not header.sfac or not header.unit or len(header.sfac) != len(header.unit):
            blockers.append("declared_cell_contents_missing")
        if config.small_solver == "shelxd" and (config.find_sites is None or not config.plop):
            blockers.append("shelxd_small_requires_explicit_find_and_plop_counts")
        if header.hklf_format not in {3, 4}:
            blockers.append("small_solver_requires_explicit_hklf3_or4")
        if not blockers:
            prepared = _small_instructions(ins_path.read_text(encoding="utf-8"), config)
        steps.append(RouteStep(config.small_solver, ToolKind(config.small_solver), (stem,),
                               (stem + ".ins", stem + ".hkl"), (stem + ".res",)))
        warnings.append("small_molecule_direct_methods_do_not_establish_absolute_hand")
    else:
        protein = config.protein
        if protein is None:
            blockers.append("protein_phasing_configuration_missing")
        else:
            if not protein.protein_confirmed:
                blockers.append("protein_context_not_declared")
            if not protein.space_group:
                blockers.append("protein_space_group_missing")
            if analyse_symmetry(header.latt, header.symm).compatibility is ChiralityCompatibility.CONTAINS_IMPROPER:
                blockers.append("protein_experimental_phasing_requires_sohncke_space_group")
            if protein.find_sites is None or not protein.heavy_element:
                blockers.append("protein_substructure_sites_and_element_required")
            if protein.solvent_fraction is None:
                blockers.append("protein_solvent_fraction_missing")
            roles = [dataset.role for dataset in protein.datasets]
            if len(set(roles)) != len(roles):
                blockers.append("duplicate_experimental_dataset_role")
            anomalous_roles = set(roles) - {"NAT"}
            if protein.experiment == "SAD" and anomalous_roles != {"SAD"}:
                blockers.append("sad_requires_explicit_anomalous_dataset")
            if protein.experiment == "MAD" and (not 2 <= len(anomalous_roles) <= 4
                    or not anomalous_roles <= {"PEAK", "INFL", "HREM", "LREM"}
                    or not anomalous_roles.intersection({"PEAK", "INFL"})):
                blockers.append("mad_requires_two_to_four_wavelength_datasets_including_peak_or_inflection")
            instructions, names = [], []
            anomalous_hashes = []
            if header.cell:
                cell = header.cell
                instructions.append("CELL " + " ".join(map(str, (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma))))
            instructions.append("SPAG " + protein.space_group)
            for n, dataset in enumerate(protein.datasets):
                if dataset.format == "MTZ":
                    blockers.append("mtz_to_supported_reflections_conversion_required")
                    continue
                suffix = ".sca" if dataset.format == "SCALEPACK" else ".hkl"
                name = f"phasing_{n:02d}" + suffix
                if not dataset.path.is_file():
                    blockers.append("declared_experimental_dataset_missing:" + dataset.role)
                    continue
                sources.append(_source(dataset.path, name))
                if dataset.role != "NAT":
                    anomalous_hashes.append(sources[-1][2])
                names.append(name)
                if dataset.role != "NAT":
                    if not dataset.friedel_pairs_retained:
                        blockers.append("anomalous_friedel_pairs_not_declared:" + dataset.role)
                    if dataset.format.startswith("SHELX") and scan_hkl(dataset.path).n_friedel_pairs == 0:
                        blockers.append("anomalous_friedel_pairs_not_present:" + dataset.role)
                flag = " -f" if dataset.format == "SHELX_AMPLITUDES" else ""
                instructions.append(dataset.role + " " + name + flag)
            if protein.experiment == "MAD" and len(set(anomalous_hashes)) != len(anomalous_hashes):
                blockers.append("mad_requires_distinct_wavelength_measurements")
            if protein.find_sites:
                instructions.append(f"FIND {protein.find_sites}")
            if protein.heavy_element:
                instructions.append("SFAC " + protein.heavy_element)
            instructions.append(f"NTRY {config.trials}")
            steps.extend((RouteStep("shelxc", ToolKind.SHELXC, (stem,), tuple(names),
                                   (stem + ".hkl", stem + "_fa.ins", stem + "_fa.hkl"), "\n".join(instructions) + "\n"),
                          RouteStep("shelxd", ToolKind.SHELXD, (stem + "_fa",),
                                   (stem + "_fa.ins", stem + "_fa.hkl"), (stem + "_fa.res",))))
            common = (stem, stem + "_fa", f"-s{protein.solvent_fraction}", f"-m{protein.density_cycles}", f"-a{protein.trace_cycles}", "-z")
            if protein.heavy_atoms_in_native:
                common += ("-h",)
            for hand, suffix, flags in (("original", "", ()), ("inverted", "_i", ("-i",))):
                steps.append(RouteStep("shelxe_" + hand, ToolKind.SHELXE, (*common, *flags),
                    (stem + ".hkl", stem + "_fa.hkl", stem + "_fa.res"),
                    (stem + suffix + ".phs", stem + suffix + ".pdb")))
            warnings.append("protein_trace_requires_explicit_pdb_to_shelxl_conversion_before_refinement")
            warnings.append("experimental_dataset_roles_are_owner_declarations_not_signal_validation")
    return InitialRoutePlan(config, ins_path, hkl_path, work_dir, stem,
                            (("ins", sources[0][2]), ("hkl", sources[1][2])), tuple(sources),
                            prepared, tuple(steps), tuple(dict.fromkeys(blockers)), tuple(warnings))


def _write_json(path: Path, data: Mapping[str, Any]):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(data), stream, sort_keys=True, allow_nan=False, indent=2)


def _trace_cc(path: Path, *, require_trace_context: bool = False) -> float | None:
    # Narrow explicit trace score, retained as an unverified dialect parser until locally checked.
    lines = path.read_text(errors="replace").splitlines()
    if require_trace_context:
        lines = [line for line in lines if re.search(r"partial\s+structure|trac(?:e|ing)", line, re.I)]
    matches = re.findall(r"(?:CC|correlation coefficient)[^\r\n=]*=\s*([+-]?\d+(?:\.\d+)?)\s*%?", "\n".join(lines), re.I)
    values = [float(value) for value in matches if -100 <= float(value) <= 100]
    return values[-1] if values else None


def _artifact(path: Path, root: Path) -> None:
    """Generated files must be bounded, contained ordinary files, not aliases to other roles."""
    try:
        relative = path.relative_to(root)
        if any(part in {".", ".."} for part in relative.parts):
            raise RouteConfigError("generated artifact path traverses its working directory")
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise RouteConfigError("generated artifact uses a symlink alias")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RouteConfigError("generated artifact is not an unaliased regular file")
        if info.st_size > MAX_GENERATED_ARTIFACT_BYTES:
            raise RouteConfigError("generated artifact exceeds the 32 MiB limit")
    except (ValueError, OSError) as exc:
        raise RouteConfigError("generated artifact is absent or outside its working directory") from exc


def execute_initial_route(plan: InitialRoutePlan, *, toolchain: Toolchain,
                          cancel: CancelToken | None = None, limits: ResourceLimits | None = None,
                          protein_converter: Callable[[ProteinConversionRequest], Path] | None = None,
                          deadline: float | None = None,
                          clock: Callable[[], float] = time.monotonic) -> InitialRouteResult:
    """Execute only this finite initial route; leave AI assignment and loops to the shared service."""
    config, work, stem = plan.config, plan.work_dir, plan.stem
    cancel = cancel or CancelToken()
    records, warnings = [], list(plan.warnings)
    owned_work = False
    attempted = 0
    if deadline is not None and (type(deadline) not in {int, float} or not math.isfinite(deadline)):
        raise RouteConfigError("initial-route deadline must be a finite monotonic time")

    def result(status, *, seed=None, paired=None, blockers=(), evidence=None, fmt=None):
        value = InitialRouteResult(status, config.strategy, seed, fmt, paired,
            sha256_file(seed) if seed else None, sha256_file(paired) if paired else None,
            tuple(records), tuple(blockers), tuple(warnings), {**(evidence or {}), "attempted_invocations": attempted})
        if owned_work and not (work / "route-result.json").exists():
            _write_json(work / "route-result.json", value.to_dict())
        return value

    if plan.blockers:
        return result("BLOCKED", blockers=plan.blockers)
    if cancel.cancelled:
        return result("CANCELLED", blockers=("cancelled_before_initial_route",))
    if deadline is not None and clock() >= deadline:
        return result("BLOCKED", blockers=("initial_route_deadline_exhausted",))
    adapters = {}
    for kind in dict.fromkeys(step.tool for step in plan.steps):
        adapter = toolchain.adapter(kind)
        adapters[kind] = (ExternalToolAdapter(adapter.resolution, runner=adapter.runner,
                                              strict_regular_files=True) if adapter else None)
    missing = tuple("tool_unavailable:" + kind.value for kind, adapter in adapters.items() if adapter is None)
    if missing:
        return result("BLOCKED", blockers=missing)
    limits = limits or ResourceLimits()
    limits = replace(limits, max_file_bytes=min(limits.max_file_bytes or MAX_GENERATED_ARTIFACT_BYTES,
                                              MAX_GENERATED_ARTIFACT_BYTES))
    if work.is_symlink() or work.exists() and (not work.is_dir() or any(work.iterdir())):
        raise RouteConfigError("initial route needs a fresh empty work directory")
    work.mkdir(parents=True, exist_ok=True)
    owned_work = True
    _write_json(work / "route-plan.json", plan.to_dict())
    for name, source, digest in plan.staged_sources:
        if sha256_file(source) != digest:
            return result("BLOCKED", blockers=("source_input_changed_before_staging",))
        target = work / name
        shutil.copyfile(source, target)
        if sha256_file(target) != digest or sha256_file(source) != digest:
            return result("BLOCKED", blockers=("input_changed_while_staging",))
    if plan.prepared_ins is not None:
        (work / (stem + ".ins")).write_text(plan.prepared_ins)
        shutil.copyfile(work / "original.hkl", work / (stem + ".hkl"))
    completed, written = {}, {}
    for index, step in enumerate(plan.steps):
        if cancel.cancelled:
            return result("CANCELLED", blockers=("cancelled_during_initial_route",))
        remaining = deadline - clock() if deadline is not None else config.timeout_s
        if remaining <= 0:
            return result("BLOCKED", blockers=("initial_route_deadline_exhausted",))
        adapter = adapters[step.tool]
        command = adapter.build_plan(args=step.args, cwd=work, timeout_s=min(config.timeout_s, remaining),
            stdin_script=step.stdin_script, expected_outputs=step.expected_outputs,
            declared_inputs=step.declared_inputs, label=step.name, limits=limits)
        _write_json(work / f"step-{index:02d}-command.json", command.to_record())
        attempted += 1
        try:
            outcome, record = adapter.invoke(command, log_dir=work, cancel=cancel)
            for name in step.expected_outputs:
                if (work / name).exists() or (work / name).is_symlink():
                    _artifact(work / name, work)
        except (OSError, ValueError, CrystalXError):
            return result("FAILED", blockers=("unsafe_or_unreadable_initial_solver_artifact",))
        from crystalx_core._serde import to_jsonable
        stored = to_jsonable(record)
        _write_json(work / f"step-{index:02d}-provenance.json", stored)
        records.append(stored)
        for output in stored.get("outputs", []):
            if output.get("role") in {"output", "modified"}:
                written[output["relative_path"]] = output.get("sha256")
        completed[step.name] = outcome.succeeded
        if outcome.cancelled:
            return result("CANCELLED", blockers=("cancelled_during_initial_route",))
        if deadline is not None and clock() >= deadline:
            return result("FAILED", blockers=("initial_route_deadline_exhausted",))
        if not outcome.succeeded:
            if step.tool is ToolKind.SHELXE:
                warnings.append(step.name + ":" + outcome.diagnostic.kind.value)
                continue
            return result("FAILED", blockers=(step.name + ":" + outcome.diagnostic.kind.value,))
    if config.strategy is InitialStrategy.PROTEIN_CDE:
        choices, scores, score_sources = {}, {}, {}
        for hand, suffix in (("original", ""), ("inverted", "_i")):
            pdb = work / (stem + suffix + ".pdb")
            if completed.get("shelxe_" + hand) and pdb.is_file() and any(line.startswith(("ATOM  ", "HETATM")) for line in pdb.read_text(errors="replace").splitlines()):
                choices[hand] = pdb
                scores[hand] = _trace_cc(pdb)
                score_sources[hand] = pdb.name if scores[hand] is not None else None
                if scores[hand] is None:
                    invocation = next((entry for entry in records if entry.get("label") == "shelxe_" + hand), {})
                    stdout = Path(invocation["stdout_path"]) if invocation.get("stdout_path") else None
                    if stdout is not None:
                        try:
                            _artifact(stdout, work)
                            log_entry = next((entry for entry in invocation.get("outputs", [])
                                              if entry.get("relative_path") == str(stdout.relative_to(work))), {})
                            if log_entry.get("sha256") == sha256_file(stdout):
                                scores[hand] = _trace_cc(stdout, require_trace_context=True)
                                score_sources[hand] = stdout.name if scores[hand] is not None else None
                        except (OSError, ValueError):
                            warnings.append("shelxe_trace_log_unavailable:" + hand)
        if not choices:
            return result("FAILED", blockers=("shelxe_produced_no_readable_trace",))
        preferred = config.protein.preferred_hand
        if preferred == "auto":
            scored = [(value, hand) for hand, value in scores.items() if value is not None]
            hand = max(scored)[1] if scored else ("original" if "original" in choices else "inverted")
            basis = "reported_trace_cc" if scored else "provisional_available_hand_no_trace_cc"
            if len(scored) != 2 or len({value for value, _ in scored}) != 2:
                warnings.append("shelxe_hand_comparison_incomplete_or_tied")
        else:
            if preferred not in choices:
                return result("FAILED", blockers=("explicit_shelxe_hand_failed",))
            hand, basis = preferred, "explicit_configuration"
        seed, paired = choices[hand], work / (stem + ".hkl")
        if (written.get(seed.name) != sha256_file(seed)
                or written.get(paired.name) != sha256_file(paired)):
            return result("FAILED", blockers=("protein_initial_artifact_changed_after_generation",))
        evidence = {"source": "SHELXE_autotraced_PDB", "hand": hand, "hand_selection": basis,
                    "trace_cc": scores, "cc_parser_format_verified": False,
                    "trace_cc_sources": score_sources, "hand_selection_is_provisional": True,
                    "alternatives": {key: str(value) for key, value in choices.items()},
                    "paired_reflections_source": "SHELXC_native_HKLF4_output",
                    "substructure_is_not_full_seed": True}
        if protein_converter is None:
            return result("INITIAL_ARTIFACTS_READY", seed=seed, paired=paired, fmt="pdb",
                          blockers=("pdb_to_shelxl_conversion_required",), evidence=evidence)
        if deadline is not None and clock() >= deadline:
            return result("FAILED", blockers=("initial_route_deadline_exhausted",), evidence=evidence)
        trace = seed
        try:
            converted = Path(protein_converter(ProteinConversionRequest(seed, paired, work / "original.ins",
                work, hand, config.protein.space_group, deadline, cancel))).absolute()
        except Exception as exc:
            evidence["conversion_error_type"] = type(exc).__name__
            return result("FAILED", blockers=("protein_converter_failed",), evidence=evidence)
        if cancel.cancelled:
            return result("CANCELLED", blockers=("cancelled_during_protein_conversion",), evidence=evidence)
        if deadline is not None and clock() >= deadline:
            return result("FAILED", blockers=("initial_route_deadline_exhausted",), evidence=evidence)
        try:
            _artifact(converted, work)
        except RouteConfigError:
            return result("FAILED", blockers=("protein_converter_output_not_contained",), evidence=evidence)
        if (written.get(trace.name) != sha256_file(trace) or written.get(paired.name) != sha256_file(paired)):
            return result("FAILED", blockers=("protein_converter_changed_initial_artifacts",), evidence=evidence)
        seed = converted
        evidence["conversion_provider_used"] = True
    elif config.strategy is InitialStrategy.SHELXT:
        seed, paired = work / (stem + "_a.res"), work / (stem + "_a.hkl")
        evidence = {"source": "SHELXT_primary_solution", "paired_reflections_source": "SHELXT_matching_HKL_output"}
    else:
        seed, paired = work / (stem + ".res"), work / (stem + ".hkl")
        evidence = {"source": config.small_solver, "paired_reflections_source": "documented_same_setting_input_HKL",
                    "original_reflections_preserved": sha256_file(paired) == dict(plan.input_hashes)["hkl"]}
    if config.strategy is not InitialStrategy.PROTEIN_CDE:
        expected_pair = (written.get(paired.name) if config.strategy is InitialStrategy.SHELXT
                         else dict(plan.input_hashes)["hkl"])
        if written.get(seed.name) != sha256_file(seed) or expected_pair != sha256_file(paired):
            return result("FAILED", blockers=("initial_model_or_paired_reflections_changed",), evidence=evidence)
    try:
        model = read_res(seed)
    except (OSError, ValueError, CrystalXError):
        return result("FAILED", blockers=("initial_model_unreadable",), evidence=evidence)
    if model.n_atoms == 0 and not model.q_peaks:
        return result("FAILED", blockers=("initial_output_contains_no_atom_or_peak_sites",), evidence=evidence)
    evidence["n_atoms"] = model.n_atoms
    evidence["n_peaks"] = len(model.q_peaks)
    evidence["element_assignments_are_provisional"] = True
    return result("INITIAL_MODEL_READY", seed=seed, paired=paired, fmt="shelx", evidence=evidence)
