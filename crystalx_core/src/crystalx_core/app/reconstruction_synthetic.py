"""Supported reconstruction fixtures for the built-in generated demo only.

Inject ``toolchain_factory`` and ``assigner_factory`` into ReconstructionBackendService.
These providers perform no crystallography, execute no binaries and load no weights. Both
factories and every subsequent operation check the original demo bytes. Staged instructions,
reflections and the small generated model are checked again at their point of consumption.
Only current-label assignment and the XT/XS/D -> XL software routes are supported; protein
datasets and measured inputs are deliberately outside this provider's contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import TYPE_CHECKING

from crystalx_core.app.demo import DEMO_CELL, DEMO_MARKER, demo_atoms, demo_digests
from crystalx_core.app.mock_toolchain import _mock_identity
from crystalx_core.app.models import RunMode
from crystalx_core.app.reconstruction_demo import SyntheticInference, SyntheticPreparer, synthetic_seed_text
from crystalx_core.app.reconstruction_runtime_inference import RuntimeAssignment
from crystalx_core.app.solve_routes import InitialStrategy, plan_initial_route
from crystalx_core.app.toolchain import Toolchain
from crystalx_core.domain.tools import (
    BinaryArch, HostPlatform, ToolAvailabilityReport, ToolDiagnostic, ToolDiagnosticKind,
    ToolKind, ToolResolution,
)
from crystalx_core.io.shelx_reader import logical_lines
from crystalx_core.io.shelx_res import read_res
from crystalx_core.refinement.cycle import cell_values
from crystalx_core.tools.runner import ProcessRunner, RawResult
from crystalx_core.workflow.reconstruction import (
    DensityObservation, ReconstructionBlocked, digest, read_snapshot,
)

if TYPE_CHECKING:
    from crystalx_core.app.reconstruction_backend import ReconstructionRequest

__all__ = ["toolchain_factory", "assigner_factory"]

_BANNER = f"{DEMO_MARKER}: SYNTHETIC software fixture; no scientific calculation or trained weights"
_TOOLS = (ToolKind.SHELXT, ToolKind.SHELXS, ToolKind.SHELXD, ToolKind.SHELXL)
_ELEMENTS = ("C", "H", "N", "O", "Zn")
_MAX_BYTES = 2 * 1024 * 1024
_MODEL_INSTRUCTIONS = frozenset((
    "TITL", "REM", "CELL", "LATT", "SYMM", "SFAC", "UNIT", "FVAR", "HKLF", "END",
    "L.S.", "PLAN", "LIST", "ACTA", "FMAP",
))


def _regular(path: Path, expected: str | None = None) -> bytes:
    """Bounded reads reject aliases before opening, and detect replacement during a read."""
    try:
        path = Path(path).absolute()
        if path.parent.resolve(strict=True) != path.parent:
            raise ValueError("noncanonical parent")
        named = path.lstat()
        if not stat.S_ISREG(named.st_mode) or named.st_nlink != 1 or named.st_size > _MAX_BYTES:
            raise ValueError("not a bounded unaliased regular file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        with os.fdopen(os.open(path, flags), "rb") as source:
            opened = os.fstat(source.fileno())
            data = source.read(_MAX_BYTES + 1)
            after = os.fstat(source.fileno())
        identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if (len(data) > _MAX_BYTES or identity(named) != identity(opened)
                or identity(named) != identity(after) or identity(named) != identity(path.lstat())
                or expected is not None and hashlib.sha256(data).hexdigest() != expected):
            raise ValueError("changed or unrecognized bytes")
        return data
    except (OSError, ValueError) as error:
        raise ReconstructionBlocked("synthetic_provider_input_is_not_pinned_builtin_data") from error


@dataclass(frozen=True)
class _Binding:
    ins: Path
    hkl: Path
    ins_sha256: str
    hkl_sha256: str
    stem: str
    prepared_ins: bytes
    initial_tool: ToolKind
    metal_only: bool

    def verify(self):
        _regular(self.ins, self.ins_sha256)
        _regular(self.hkl, self.hkl_sha256)


def _bind(request: ReconstructionRequest) -> _Binding:
    from crystalx_core.app.reconstruction_backend import ReconstructionRequest
    if not isinstance(request, ReconstructionRequest) or request.mode is not RunMode.MOCK_DEMO:
        raise ReconstructionBlocked("synthetic_provider_requires_mock_demo_mode")
    if request.heavy_preparation != "current_loop":
        raise ReconstructionBlocked("synthetic_provider_supports_current_loop_preparation_only")
    if request.route.strategy is InitialStrategy.PROTEIN_CDE or request.route.protein is not None:
        raise ReconstructionBlocked("synthetic_provider_does_not_accept_protein_or_extra_datasets")
    if (request.cycle.refinement_method != "ls" or request.cycle.early_cgls_rounds or request.cycle.damp_factor is not None
            or request.cycle.fourier_map!=2 or request.cycle.anisotropic_metals or request.cycle.metal_ripple_exclusion_angstrom):
        raise ReconstructionBlocked("synthetic_provider_supports_default_refinement_controls_only")
    if request.cycle.inference_protocol != "current_labels":
        raise ReconstructionBlocked("synthetic_provider_supports_current_labels_only")
    if request.checkpoint is not None or request.checkpoint_sha256 is not None or request.device not in ("auto", "cpu"):
        raise ReconstructionBlocked("synthetic_provider_does_not_load_weights_or_use_accelerators")
    if request.candidate_elements and tuple(request.candidate_elements) != _ELEMENTS:
        raise ReconstructionBlocked("synthetic_provider_requires_builtin_candidate_elements")
    if set(request.anchor_elements) - {"Zn"}:
        raise ReconstructionBlocked("synthetic_provider_has_only_the_builtin_zinc_anchor")
    ins_sha, hkl_sha = demo_digests()
    _regular(request.ins_path, ins_sha)
    _regular(request.hkl_path, hkl_sha)
    plan = plan_initial_route(request.route, ins_path=request.ins_path, hkl_path=request.hkl_path,
                              work_dir=request.ins_path.parent / "unused-synthetic-preview")
    if plan.blockers or plan.prepared_ins is None or len(plan.steps) != 1 or plan.steps[0].tool not in _TOOLS[:3]:
        raise ReconstructionBlocked("synthetic_provider_requires_a_supported_small_molecule_route")
    binding = _Binding(request.ins_path, request.hkl_path, ins_sha, hkl_sha, plan.stem,
                       plan.prepared_ins.encode("utf-8"), plan.steps[0].tool, request.metal_only)
    binding.verify()
    return binding


def _model(path: Path, *, encoded_metal: bool = False):
    data = _regular(path)
    parsed = read_res(path)
    expected_sites = {atom.label: atom.frac for atom in (demo_atoms()[0], demo_atoms()[2])}
    if (parsed.sha256 != hashlib.sha256(data).hexdigest() or _regular(path) != data
            or DEMO_MARKER not in parsed.header.title or parsed.header.cell is None
            or cell_values(parsed.header.cell) != DEMO_CELL or parsed.header.wavelength_a != .71073
            or parsed.header.latt != -1 or parsed.header.symm != ("-X,Y+1/2,-Z",)
            or parsed.header.sfac != _ELEMENTS or parsed.header.unit != (10, 28, 4, 12, 2)
            or parsed.header.hklf_format != 4 or parsed.q_peaks or not parsed.atoms
            or len({a.label for a in parsed.atoms}) != len(parsed.atoms)):
        raise ReconstructionBlocked("synthetic_provider_model_is_not_a_builtin_derivative")
    atom_ids = {a.label.upper() for a in parsed.atoms}
    if any(line.instruction not in _MODEL_INSTRUCTIONS | atom_ids for line in logical_lines(path)):
        raise ReconstructionBlocked("synthetic_provider_model_has_unsupported_instructions")
    for atom in parsed.atoms:
        xyz = (atom.x, atom.y, atom.z)
        expected = expected_sites.get(atom.label)
        if encoded_metal and atom.label == "Zn1" and all(v > 10 for v in xyz):
            xyz = tuple(value - 10 for value in xyz)
        allowed = {"Zn"} if atom.label == "Zn1" else {"C", "N"}
        if (expected is None or any(not math.isclose(v, ref, abs_tol=1e-8, rel_tol=0)
                                   for v, ref in zip(xyz, expected))
                or parsed.element_of(atom) not in allowed or atom.occupancy_code != 11
                or atom.u_values != (.03,) or atom.part not in (None, 0)
                or atom.residue is not None or atom.afix not in (None, 0)):
            raise ReconstructionBlocked("synthetic_provider_model_is_not_a_builtin_derivative")
    return parsed, data


def _write(path: Path, data: str | bytes):
    with path.open("xb") as output:
        output.write(data.encode("utf-8") if isinstance(data, str) else data)


class _Executor:
    supports_control = True

    def __init__(self, binding: _Binding):
        self.binding = binding
        self.calls = []

    def execute(self, plan, env, *, cancel=None, stdout_path=None, stderr_path=None):
        if cancel is not None and cancel.cancelled:
            return RawResult(None, _BANNER + "\n", "", cancelled=True)
        try:
            binding = self.binding
            binding.verify()
            kind, root = plan.tool_kind, plan.cwd
            if (kind not in _TOOLS or root.resolve(strict=True) != root.absolute()
                    or plan.stdin_script is not None or plan.env_overrides
                    or plan.argv[0] != str(_mock_identity(kind).path)
                    or plan.tool_sha256 != _mock_identity(kind).sha256):
                raise ReconstructionBlocked("synthetic_provider_invocation_not_supported")
            if kind is ToolKind.SHELXL:
                if (plan.argv[1:] != ("trial",) or plan.declared_inputs != ("trial.ins", "trial.hkl")
                        or plan.expected_outputs != ("trial.res", "trial.lst")):
                    raise ReconstructionBlocked("synthetic_provider_invocation_not_supported")
                _regular(root / "trial.hkl", binding.hkl_sha256)
                parsed, text = _model(root / "trial.ins", encoded_metal=binding.metal_only)
                summary = ("R1 = 0.300000 for 800 Fo > 4sig(Fo) and 0.300000 for all 1000 data\n"
                           "wR2 = 0.600000, GooF = S = 1.030000, Restrained GooF = 1.030000 for all data\n")
                params = 4 * len(parsed.atoms)
                rem = "".join("REM " + line + "\n" for line in summary.splitlines())
                outputs = {
                    "trial.res": text + (f"REM {_BANNER}\n" + rem
                        + f"REM {params} parameters refined using 0 restraints\n").encode(),
                    "trial.lst": _BANNER + "\n" + summary + f"{params} / {params} parameters\n",
                    "trial.cif": f"data_synthetic\n_audit_creation_method '{_BANNER}'\n",
                    "trial.fcf": f"data_synthetic\n_audit_creation_method '{_BANNER}'\n",
                }
            else:
                stem = binding.stem
                expected_outputs = ((stem + "_a.res", stem + "_a.hkl") if kind is ToolKind.SHELXT
                                    else (stem + ".res",))
                if (kind is not binding.initial_tool or plan.argv[1:] != (stem,)
                        or plan.declared_inputs != (stem + ".ins", stem + ".hkl")
                        or plan.expected_outputs != expected_outputs):
                    raise ReconstructionBlocked("synthetic_provider_invocation_not_supported")
                _regular(root / "original.ins", binding.ins_sha256)
                _regular(root / "original.hkl", binding.hkl_sha256)
                if _regular(root / (stem + ".ins")) != binding.prepared_ins:
                    raise ReconstructionBlocked("synthetic_provider_staged_instructions_changed")
                reflections = _regular(root / (stem + ".hkl"), binding.hkl_sha256)
                seed = synthetic_seed_text().replace("\nCELL", f"\nREM {_BANNER}\nCELL", 1)
                outputs = {expected_outputs[0]: seed}
                if kind is ToolKind.SHELXT:
                    outputs[expected_outputs[1]] = reflections
            binding.verify()
            if cancel is not None and cancel.cancelled:
                return RawResult(None, _BANNER + "\n", "", cancelled=True)
            for name, data in outputs.items():
                if (root / name).exists() or (root / name).is_symlink():
                    raise ReconstructionBlocked("synthetic_provider_outputs_must_be_fresh")
            for name, data in outputs.items():
                _write(root / name, data)
            # HKL bytes stay exact; their synthetic identity is recorded here and in provenance.
            _write(root / "synthetic-provider.json", json.dumps({
                "synthetic": True, "actual_pretrained_weights_executed": False,
                "scientifically_validated": False, "banner": _BANNER,
                "original_ins_sha256": binding.ins_sha256, "original_hkl_sha256": binding.hkl_sha256,
                "outputs": {name: hashlib.sha256(_regular(root / name)).hexdigest() for name in outputs},
                "metrics_kind": "fixed_software_fixture_values_not_calculated",
            }, indent=2) + "\n")
            self.calls.append(plan)
            return RawResult(0, _BANNER + "\n", "")
        except (OSError, ValueError) as error:
            detail = str(error) if isinstance(error, ReconstructionBlocked) else "synthetic_provider_artifact_rejected"
            return RawResult(64, _BANNER + "\n", detail + "\n")


class _Toolchain(Toolchain):
    mode = RunMode.MOCK_DEMO

    def __init__(self, binding: _Binding):
        self.executor = _Executor(binding)

    def availability(self):
        return ToolAvailabilityReport(HostPlatform("synthetic", BinaryArch.UNKNOWN), {
            kind.value: ToolResolution(kind, _mock_identity(kind) if kind in _TOOLS else None,
                ToolDiagnostic(ToolDiagnosticKind.OK if kind in _TOOLS else ToolDiagnosticKind.NOT_FOUND,
                               _BANNER if kind in _TOOLS else "Not supported by reconstruction demo"))
            for kind in ToolKind
        })

    def _runner(self):
        return ProcessRunner(self.executor, base_env={})


class _Assigner:
    def __init__(self, binding: _Binding):
        self.binding = binding
        self.inference = SyntheticInference()

    def assign(self, path, *, candidate_elements, include_q_peaks=True,
               input_protocol="current_labels", original_ins_path=None, initial_labels=None, protocol_seed=42):
        self.binding.verify()
        if input_protocol != "current_labels":
            raise ReconstructionBlocked("synthetic_provider_supports_current_labels_only")
        if tuple(candidate_elements) != _ELEMENTS or type(include_q_peaks) is not bool:
            raise ReconstructionBlocked("synthetic_provider_requires_builtin_candidate_elements")
        if original_ins_path is not None:
            _regular(Path(original_ins_path), self.binding.ins_sha256)
        parsed, data = _model(Path(path))
        model = read_snapshot(Path(path), reflection_sha256=self.binding.hkl_sha256, coordinate_bound=5.)
        observation = DensityObservation(model.sha256, model.basis_key, model.sha256,
                                         "SYNTHETIC_PEAK_LIST", ())
        prepared = SyntheticPreparer().prepare(model, observation)
        result = self.inference.predict_elements(prepared.request)
        self.binding.verify()
        if _regular(Path(path)) != data:
            raise ReconstructionBlocked("synthetic_provider_model_changed_during_assignment")
        return RuntimeAssignment(prepared.request, result, parsed.sha256, prepared.request.site_ids, (), {
            "synthetic": True, "actual_pretrained_weights_executed": False,
            "input_protocol": "current_labels", "banner": _BANNER,
            "original_ins_sha256": self.binding.ins_sha256, "original_hkl_sha256": self.binding.hkl_sha256,
            "input_protocol_record": {"protocol": "current_labels", "prediction_passes": 1,
                                      "forced_element_coverage": False},
        })


def toolchain_factory(request: ReconstructionRequest) -> Toolchain:
    """Build an XT/XS/D + XL software fixture for exact built-in demo inputs only."""
    return _Toolchain(_bind(request))


def assigner_factory(request: ReconstructionRequest):
    """Build deterministic synthetic assignment; no checkpoint or ML runtime is loaded."""
    return _Assigner(_bind(request))
