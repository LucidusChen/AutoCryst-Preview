"""Stand-ins for SHELXT, SHELXL and PLATON, for the application demo only.

**Nothing here is crystallography.** The mock programs write files in the shapes the real ones
write, so that the whole application -- staging, provenance, cancellation, parsing, the viewer,
downloads -- can be exercised on a machine that has no SHELX or PLATON. Every file they write
carries ``MOCK`` and the demo marker; every alert code they write starts with ``MOCK`` so it cannot
be confused with a PLATON test; and their figures of merit are fixed placeholders, not
calculations.

They also **refuse to run on anything except the built-in synthetic demo case**, checked by exact
content digest of the reflection file and by the demo marker in the instruction file. A mock result
can therefore never be produced for a user's own data, whatever a frontend does.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from crystalx_core.app.demo import DEMO_MARKER, demo_atoms, demo_digests
from crystalx_core.app.models import RunMode
from crystalx_core.app.toolchain import PIPELINE_TOOLS, Toolchain
from crystalx_core.domain.tools import (
    BinaryArch,
    BinaryFormat,
    BinaryProbe,
    HostPlatform,
    ResolutionSource,
    ToolAvailabilityReport,
    ToolDiagnostic,
    ToolDiagnosticKind,
    ToolIdentity,
    ToolKind,
    ToolResolution,
)
from crystalx_core.tools.runner import CancelToken, ExecutionPlan, ProcessRunner, RawResult

__all__ = ["MockToolchain", "MockToolchainExecutor", "MOCK_TOOLS", "MOCK_BANNER"]

MOCK_BANNER = "MOCK -- synthetic output written by the CrystalX demo toolchain; no real program ran"
MOCK_TOOLS = (ToolKind.SHELXT, ToolKind.SHELXL, ToolKind.PLATON_SPECIAL)

_TICK_S = 0.05
_U_ISO = {"Zn": 0.021, "O": 0.034, "N": 0.031, "C": 0.038}


def _mock_identity(kind: ToolKind) -> ToolIdentity:
    path = Path(f"<mock-{kind.value}>")
    return ToolIdentity(
        kind=kind,
        path=path,
        realpath=path,
        source=ResolutionSource.SYNTHETIC_DEMO,
        size_bytes=0,
        sha256=hashlib.sha256(f"crystalx-mock-{kind.value}-v1".encode()).hexdigest(),
        mtime_epoch=0.0,
        probe=BinaryProbe(
            fmt=BinaryFormat.UNKNOWN,
            arch=BinaryArch.UNKNOWN,
            note="synthetic demo stand-in; not an executable",
        ),
        version="mock-1",
        version_probe_note="not a program: the CrystalX demo toolchain",
    )


@dataclass
class MockToolchainExecutor:
    """Executes nothing. Writes synthetic outputs into the plan's working directory."""

    step_seconds: float = 0.6
    fail_tool: ToolKind | None = None
    fail_mode: str = "nonzero"
    """``nonzero`` (exit 1), ``no_output`` (exit 0, expected file missing) or ``hang`` (runs
    until cancelled or timed out)."""

    omit_suffixes: tuple[str, ...] = ()
    """Output suffixes the stand-ins do not write, e.g. ``(".fcf",)`` to exercise the PLATON
    prerequisite check."""

    shelxt_centrosymmetric: bool = False
    """Write the SHELXT stand-in's solution with ``LATT 1`` (an inversion centre), to exercise the
    chirality check on SHELXT's chosen space group."""

    shelxl_listing_extra: tuple[str, ...] = ()
    """Extra lines appended to the SHELXL stand-in's listing, e.g. an error-marked line."""

    shelxt_write_hkl: bool = True
    """Write ``_a.hkl`` beside the solution. ``False`` exercises the 'Orientation as input'
    reflection contract (see :mod:`crystalx_core.app.reflections`)."""

    shelxt_orientation_statement: bool = True
    """Put "Orientation as input" in the solution's REM lines. The wording imitates SHELXT's but the
    real format is unverified: tests built on it prove the contract, not SHELXT's output format."""

    shelxt_cell: tuple[float, float, float, float, float, float] | None = None
    """Replace the solution's CELL parameters (the wavelength is kept), to exercise the cell check."""

    supports_control = True

    def execute(
        self,
        plan: ExecutionPlan,
        env: Mapping[str, str],
        *,
        cancel: CancelToken | None = None,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
    ) -> RawResult:
        kind = plan.tool_kind
        log: list[str] = [f"[{MOCK_BANNER}]", f"[mock] program: {kind.value if kind else '?'}"]
        err: list[str] = []

        def flush() -> bool:
            if stdout_path is None or stderr_path is None:
                return False
            stdout_path.write_text("\n".join(log) + "\n", encoding="utf-8")
            stderr_path.write_text("\n".join(err) + ("\n" if err else ""), encoding="utf-8")
            return True

        def finish(**kwargs) -> RawResult:
            written = flush()
            return RawResult(
                stdout="\n".join(log), stderr="\n".join(err), logs_written=written, **kwargs
            )

        refusal = self._refusal(plan)
        if refusal:
            err.append(f"[mock] refusing to run: {refusal}")
            return finish(exit_code=64)

        hang = kind is not None and kind is self.fail_tool and self.fail_mode == "hang"
        started = time.monotonic()
        ticks = max(1, int(round(self.step_seconds / _TICK_S)))
        tick = 0
        while hang or tick < ticks:
            tick += 1
            if cancel is not None:
                cancel.wait(_TICK_S)
            else:
                time.sleep(_TICK_S)
            if cancel is not None and cancel.cancelled:
                log.append("[mock] stopped on request")
                return finish(exit_code=None, cancelled=True)
            if time.monotonic() - started > plan.timeout_s:
                log.append("[mock] wall-clock budget exhausted")
                return finish(exit_code=None, timed_out=True)
            if tick % 4 == 0:
                log.append(f"[mock] working ({tick * _TICK_S:.1f} s)")
                flush()

        if kind is not None and kind is self.fail_tool:
            if self.fail_mode == "nonzero":
                err.append("[mock] injected failure: the demo was asked to fail at this step")
                return finish(exit_code=1)
            if self.fail_mode == "no_output":
                log.append("[mock] injected failure: exiting 0 without writing the expected output")
                return finish(exit_code=0)

        writer = {
            ToolKind.SHELXT: self._shelxt,
            ToolKind.SHELXL: self._shelxl,
            ToolKind.PLATON_SPECIAL: self._platon,
            ToolKind.PLATON: self._platon,
        }.get(kind)  # type: ignore[arg-type]
        if writer is None:
            err.append(f"[mock] no stand-in exists for {kind}")
            return finish(exit_code=2)
        log.extend(writer(plan))
        log.append("[mock] done")
        return finish(exit_code=0)

    # ------------------------------------------------------------------ refusal
    def _refusal(self, plan: ExecutionPlan) -> str:
        """Why this plan is not the demo, or an empty string when it is."""
        cwd = plan.cwd
        kind = plan.tool_kind
        _, hkl_digest = demo_digests()
        if kind in (ToolKind.SHELXT, ToolKind.SHELXL):
            if len(plan.argv) < 2:
                return "no stem given"
            stem = plan.argv[1]
            ins = cwd / f"{stem}.ins"
            hkl = cwd / f"{stem}.hkl"
            if not ins.is_file() or not hkl.is_file():
                return f"{stem}.ins/{stem}.hkl not found in the working directory"
            if DEMO_MARKER not in ins.read_text(encoding="utf-8", errors="replace"):
                return "the instruction file is not the synthetic demo case"
            if hashlib.sha256(hkl.read_bytes()).hexdigest() != hkl_digest:
                return "the reflection file is not the synthetic demo case"
            return ""
        if kind in (ToolKind.PLATON, ToolKind.PLATON_SPECIAL):
            cif_name = next((a for a in plan.argv[1:] if a.lower().endswith(".cif")), "")
            cif = cwd / cif_name if cif_name else None
            if cif is None or not cif.is_file():
                return "no CIF given"
            if DEMO_MARKER not in cif.read_text(encoding="utf-8", errors="replace"):
                return "the CIF is not from the synthetic demo"
            return ""
        return f"unsupported program {kind}"

    # ------------------------------------------------------------------ writers
    def _shelxt(self, plan: ExecutionPlan) -> list[str]:
        cwd, stem = plan.cwd, plan.argv[1]
        header = _header_lines(cwd / f"{stem}.ins")
        rng = random.Random(7)
        atoms = []
        for atom in demo_atoms():
            jitter = [round(rng.uniform(-0.004, 0.004), 4) for _ in range(3)]
            x, y, z = (atom.frac[i] + jitter[i] for i in range(3))
            sfac = _sfac_index(header, atom.element)
            atoms.append(f"{atom.label:<5}{sfac:3d} {x:9.5f} {y:9.5f} {z:9.5f}  11.00000   0.05000")
        group = "P2(1)/m" if self.shelxt_centrosymmetric else "P2(1)"
        lines = [
            f"TITL {stem}_a in {group}  [{MOCK_BANNER}] {DEMO_MARKER}",
            "REM MOCK: the next three lines imitate SHELXT's summary format with synthetic numbers.",
            f"REM SHELXT solution in {group}: R1 0.087, Rweak 0.031, Alpha 0.000",
            "REM <I/s> 0.180 for 6 systematic absences"
            + (", Orientation as input" if self.shelxt_orientation_statement else ""),
            "REM Formula found by SHELXT: C5 N2 O6 Zn",
            *[
                self._solution_header_line(line)
                for line in header
                if not line.upper().startswith(("TITL", "HKLF", "END"))
            ],
            "L.S. 10",
            "PLAN 20",
            "FMAP 2",
            *atoms,
            "HKLF 4",
            "END",
        ]
        (cwd / f"{stem}_a.res").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if self.shelxt_write_hkl:
            shutil.copyfile(cwd / f"{stem}.hkl", cwd / f"{stem}_a.hkl")
        (cwd / f"{stem}.lxt").write_text(
            f"[{MOCK_BANNER}]\nMOCK listing: no space-group search was performed.\n",
            encoding="utf-8",
        )
        if not self.shelxt_write_hkl:
            return ["[mock] wrote one trial structure (_a.res) and no reflection file"]
        return ["[mock] wrote one trial structure (_a.res) and its reflection file (_a.hkl)"]

    def _solution_header_line(self, line: str) -> str:
        upper = line.upper()
        if self.shelxt_centrosymmetric and upper.startswith("LATT"):
            return "LATT 1"
        if self.shelxt_cell is not None and upper.startswith("CELL"):
            wavelength = line.split()[1]
            return "CELL " + " ".join([wavelength, *(f"{v:g}" for v in self.shelxt_cell)])
        return line

    def _shelxl(self, plan: ExecutionPlan) -> list[str]:
        cwd, stem = plan.cwd, plan.argv[1]
        text = (cwd / f"{stem}.ins").read_text(encoding="utf-8", errors="replace")
        cycles_match = re.search(r"^L\.S\.\s+(\d+)", text, re.MULTILINE)
        cycles = int(cycles_match.group(1)) if cycles_match else 4
        plan_match = re.search(r"^PLAN\s+(\d+)", text, re.MULTILINE)
        n_peaks = min(int(plan_match.group(1)) if plan_match else 20, 12)
        write_cif = re.search(r"^ACTA\b", text, re.MULTILINE) is not None
        write_fcf = re.search(r"^LIST\s+4\b", text, re.MULTILINE) is not None

        header = _header_lines(cwd / f"{stem}.ins")
        truth = {a.label: a for a in demo_atoms()}
        atom_lines: list[str] = []
        atom_sites: list[tuple[str, str, tuple[float, float, float], float]] = []
        for line in text.splitlines():
            tokens = line.split()
            if len(tokens) >= 6 and tokens[0] in truth:
                atom = truth[tokens[0]]
                u = _U_ISO.get(atom.element, 0.04)
                x, y, z = atom.frac
                atom_lines.append(
                    f"{atom.label:<5}{int(tokens[1]):3d} {x:9.5f} {y:9.5f} {z:9.5f}  11.00000 {u:9.5f}"
                )
                atom_sites.append((atom.label, atom.element, atom.frac, u))

        r1, r1_all, n_obs, n_all, wr2, goof, params = 0.0412, 0.0465, 1180, 1328, 0.1023, 1.045, 57
        peaks = _q_peaks(atom_sites, n_peaks)
        res_lines = [
            f"TITL {stem}  [{MOCK_BANNER}] {DEMO_MARKER}",
            "REM MOCK: refinement figures below are fixed placeholders, not a least-squares result.",
            *[
                line
                for line in header
                if not line.upper().startswith(("TITL", "HKLF", "END", "L.S.", "PLAN", "LIST", "FMAP", "ACTA"))
            ],
            f"L.S. {cycles}",
            "PLAN 20",
            "FMAP 2",
            "ACTA",
            "WGHT    0.050000",
            "FVAR       1.23456",
            *atom_lines,
            "HKLF 4",
            "",
            f"REM  {stem} in P2(1)",
            f"REM wR2 = {wr2:.4f}, GooF = S = {goof:.3f}, Restrained GooF = {goof:.3f} for all data",
            f"REM R1 = {r1:.4f} for {n_obs} Fo > 4sig(Fo) and {r1_all:.4f} for all {n_all} data",
            f"REM {params} parameters refined using 0 restraints",
            "",
            "END",
            "",
            "WGHT      0.0500      0.0000",
            "REM Highest difference peak  0.512,  deepest hole -0.433,  1-sigma level  0.098",
            *[
                f"Q{i + 1:<4}1 {x:9.5f} {y:9.5f} {z:9.5f}  11.00000  0.05    {h:.2f}"
                for i, ((x, y, z), h) in enumerate(peaks)
            ],
        ]
        (cwd / f"{stem}.res").write_text("\n".join(res_lines) + "\n", encoding="utf-8")

        lst = [f" [{MOCK_BANNER}]", f" {DEMO_MARKER}", ""]
        shift = 0.25
        for cycle in range(1, cycles + 1):
            lst.append(f" Least-squares cycle  {cycle}   (MOCK: no least squares was performed)")
            lst.append(f" Mean shift/su = {shift:.3f} Maximum = {shift * 4:.3f} for  x Zn1")
            shift *= 0.3
        lst += [
            "",
            f" R1 = {r1:.4f} for {n_obs} Fo > 4sig(Fo) and {r1_all:.4f} for all {n_all} data",
            f" wR2 = {wr2:.4f}, GooF = S = {goof:.3f}, Restrained GooF = {goof:.3f} for all data",
            f" {params} / {params} parameters",
            " Highest peak    0.51  at  0.2512  0.3012  0.2034  [  0.05 A from ZN1 ]",
            " Deepest hole   -0.43  at  0.2488  0.2989  0.1968  [  0.07 A from ZN1 ]",
        ]
        lst += list(self.shelxl_listing_extra)
        (cwd / f"{stem}.lst").write_text("\n".join(lst) + "\n", encoding="utf-8")

        if ".cif" in self.omit_suffixes:
            write_cif = False
        if ".fcf" in self.omit_suffixes:
            write_fcf = False
        if write_cif:
            (cwd / f"{stem}.cif").write_text(_mock_cif(stem, atom_sites, r1, wr2, goof), encoding="utf-8")
        if write_fcf:
            (cwd / f"{stem}.fcf").write_text(
                f"# [{MOCK_BANNER}] {DEMO_MARKER}\ndata_{stem}\n_shelx_refln_list_code 4\n"
                "loop_\n_refln_index_h\n_refln_index_k\n_refln_index_l\n_refln_F_squared_calc\n"
                "_refln_F_squared_meas\n_refln_F_squared_sigma\n_refln_observed_status\n"
                "   1   0   0   12.0   11.6   0.8 o\n",
                encoding="utf-8",
            )
        return [f"[mock] wrote {stem}.res/.lst after {cycles} pretend cycles"]

    def _platon(self, plan: ExecutionPlan) -> list[str]:
        cif_name = next(a for a in plan.argv[1:] if a.lower().endswith(".cif"))
        stem = Path(cif_name).stem
        lines = [
            f"# [{MOCK_BANNER}] {DEMO_MARKER}",
            "# PLATON did not run. The codes below start with MOCK and correspond to no PLATON",
            "# test; they exist to show how each level is displayed.",
            "MOCK001_ALERT_3_B Synthetic demo alert used to show how a level-B alert looks.",
            "MOCK002_ALERT_4_C Synthetic demo alert: hydrogen atoms are not modelled in the demo.",
            "MOCK003_ALERT_2_G Synthetic demo alert carrying general information only.",
        ]
        (plan.cwd / f"{stem}.chk").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return [f"[mock] wrote {stem}.chk with three synthetic alerts"]


def _header_lines(ins: Path) -> list[str]:
    """Header records of an instruction file, up to the first atom, control or HKLF record."""
    keep = ("TITL", "REM", "CELL", "ZERR", "LATT", "SYMM", "SFAC", "UNIT", "DISP", "TEMP", "SIZE")
    lines = []
    for line in ins.read_text(encoding="utf-8", errors="replace").splitlines():
        word = line.split()[0].upper() if line.split() else ""
        if word in keep:
            lines.append(line.rstrip())
        elif word in ("HKLF", "END"):
            break
    return lines


def _sfac_index(header: list[str], element: str) -> int:
    for line in header:
        tokens = line.split()
        if tokens and tokens[0].upper() == "SFAC":
            symbols = [t.capitalize() for t in tokens[1:]]
            if element in symbols:
                return symbols.index(element) + 1
    return 1


def _q_peaks(sites, count: int):
    """Peaks about 0.95 A from light atoms, in descending height: where hydrogens would be."""
    from crystalx_core.app.demo import DEMO_CELL
    from crystalx_core.geometry.model import frac_to_cart, orthogonalisation_matrix

    m = orthogonalisation_matrix(DEMO_CELL)
    rng = random.Random(11)
    peaks = []
    for label, element, frac, _u in sites:
        if element not in ("C", "N", "O") or len(peaks) >= count:
            continue
        cart = frac_to_cart(m, frac)
        theta, phi = rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi)
        d = (math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta))
        target = tuple(cart[i] + 0.95 * d[i] for i in range(3))
        z = target[2] / m[2][2]
        y = (target[1] - m[1][2] * z) / m[1][1]
        x = (target[0] - m[0][1] * y - m[0][2] * z) / m[0][0]
        peaks.append(((x, y, z), round(0.62 - 0.03 * len(peaks), 2)))
    return peaks


def _mock_cif(stem: str, sites, r1: float, wr2: float, goof: float) -> str:
    from crystalx_core.app.demo import DEMO_CELL

    a, b, c, alpha, beta, gamma = DEMO_CELL
    rows = "\n".join(
        f"{label} {element} {x:.5f} {y:.5f} {z:.5f} {u:.4f} Uiso 1"
        for label, element, (x, y, z), u in sites
    )
    return (
        f"# [{MOCK_BANNER}] {DEMO_MARKER}\n"
        "# This is not a SHELXL CIF and must not be deposited or reused.\n"
        f"data_{stem}\n"
        "_audit_creation_method 'CrystalX demo toolchain (MOCK) -- not SHELXL'\n"
        f"_cell_length_a {a:.4f}\n_cell_length_b {b:.4f}\n_cell_length_c {c:.4f}\n"
        f"_cell_angle_alpha {alpha:.3f}\n_cell_angle_beta {beta:.3f}\n_cell_angle_gamma {gamma:.3f}\n"
        "_space_group_name_H-M_alt 'P 21'\n"
        f"_refine_ls_R_factor_gt {r1:.4f}\n_refine_ls_wR_factor_ref {wr2:.4f}\n"
        f"_refine_ls_goodness_of_fit_ref {goof:.3f}\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
        "_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_U_iso_or_equiv\n"
        "_atom_site_adp_type\n_atom_site_occupancy\n"
        f"{rows}\n"
    )


@dataclass
class MockToolchain(Toolchain):
    """The demo toolchain: SHELXT, SHELXL and ``platon_special`` stand-ins, nothing else."""

    step_seconds: float = 0.6
    fail_tool: ToolKind | None = None
    fail_mode: str = "nonzero"
    omit_suffixes: tuple[str, ...] = ()
    shelxt_centrosymmetric: bool = False
    shelxl_listing_extra: tuple[str, ...] = ()
    shelxt_write_hkl: bool = True
    shelxt_orientation_statement: bool = True
    shelxt_cell: tuple[float, float, float, float, float, float] | None = None
    _report: ToolAvailabilityReport | None = field(default=None, init=False, repr=False)

    mode = RunMode.MOCK_DEMO

    def availability(self) -> ToolAvailabilityReport:
        if self._report is None:
            resolutions: dict[str, ToolResolution] = {}
            for kind in PIPELINE_TOOLS:
                if kind in MOCK_TOOLS:
                    resolutions[kind.value] = ToolResolution(
                        kind=kind,
                        identity=_mock_identity(kind),
                        diagnostic=ToolDiagnostic(
                            kind=ToolDiagnosticKind.OK,
                            message=f"{kind.value}: synthetic demo stand-in (MOCK), not the real program",
                        ),
                    )
                else:
                    resolutions[kind.value] = ToolResolution(
                        kind=kind,
                        identity=None,
                        diagnostic=ToolDiagnostic(
                            kind=ToolDiagnosticKind.NOT_FOUND,
                            message=f"{kind.value} is not part of the demo toolchain",
                        ),
                    )
            self._report = ToolAvailabilityReport(
                host=HostPlatform(system="demo", arch=BinaryArch.UNKNOWN), resolutions=resolutions
            )
        return self._report

    def _runner(self) -> ProcessRunner:
        return ProcessRunner(
            MockToolchainExecutor(
                step_seconds=self.step_seconds,
                fail_tool=self.fail_tool,
                fail_mode=self.fail_mode,
                omit_suffixes=self.omit_suffixes,
                shelxt_centrosymmetric=self.shelxt_centrosymmetric,
                shelxl_listing_extra=self.shelxl_listing_extra,
                shelxt_write_hkl=self.shelxt_write_hkl,
                shelxt_orientation_statement=self.shelxt_orientation_statement,
                shelxt_cell=self.shelxt_cell,
            ),
            base_env={},
        )
