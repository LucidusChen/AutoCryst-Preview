"""Bounded, local-only measurements against the actual upstream CPU model.

This module accepts no saved reference predictions. It starts our fixed worker
in an isolated Python process, which executes hash-pinned upstream sources with
real PyG/scatter/cluster imports. Missing weights or dependencies are BLOCKED.
Neither completed measurements nor caller-attested checkpoint provenance confer
scientific acceptance. There is deliberately no numerical pass threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MappingProxyType
from uuid import uuid4

from crystalx_core.domain.inference import ElementInferenceRequest

from .source import AUDITED_SHA256, SourceRevisionError


PROTOCOL = "crystalx-local-parity-v1"
UPSTREAM_SHA256 = MappingProxyType({
    **{f"crystalx_infer/models/{name}": digest for name, digest in AUDITED_SHA256.items()},
    "crystalx_infer/common/chem.py": "7684cb06d250b44c0d00aebef237965d3b40efa947930e460a5ca886ec8c6205",
    "crystalx_infer/common/heavy.py": "3490522fcf92817b9b412d18875438e87cefabfaa60d56c7707e2aff5560717d",
})


@dataclass(frozen=True)
class ParityCase:
    """An explicit neutral request; no answer structures or file parsing."""

    case_id: str
    request: ElementInferenceRequest

    def __post_init__(self):
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be nonempty")
        if not isinstance(self.request, ElementInferenceRequest):
            raise TypeError("request must be an ElementInferenceRequest")
        if len(self.request.site_ids) > 128:
            raise ValueError("Parity cases are bounded to 128 total sites including context")


def fixed_cases() -> tuple[ParityCase, ...]:
    """Hand-authored software probes, never a representative trained dataset.

    All 41 sites of the dense Fe environment lie within 5 Angstrom of one
    another, exercising the cap of 32 (including self loops). Context masking
    also checks that site order survives the two passes.
    """
    sparse = ElementInferenceRequest(
        ("C1", "N1", "O1", "Fe1"), (6, 7, 8, 26),
        ((0., 0., 0.), (1.2, 0., 0.), (0., 1.3, 0.), (8., 0., 0.)),
        (6, 7, 8, 26),
    )
    positions = [(0., 0., 0.)]
    # Two offset rings avoid coincident positions and put >32 sites in range.
    for ring, z in ((0, -.45), (1, .45)):
        for i in range(20):
            angle = 2 * math.pi * (i + .25 * ring) / 20
            positions.append((1.7 * math.cos(angle), 1.7 * math.sin(angle), z))
    dense = ElementInferenceRequest(
        ("Fe-center",) + tuple(f"ligand-{i:02d}" for i in range(40)),
        (26,) + (6, 7, 8, 6) * 10, tuple(positions), (6, 7, 8, 26),
        main_site_count=21,
    )
    return (ParityCase("hand-authored-sparse-v1", sparse),
            ParityCase("hand-authored-dense-Fe-cap32-context-v1", dense))


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_upstream_sources(source_root: Path) -> dict[str, bytes]:
    """Read and verify every source before any upstream source executes."""
    contents = {}
    for name, expected in UPSTREAM_SHA256.items():
        path = Path(source_root) / name
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise SourceRevisionError(f"Cannot read pinned upstream source: {path}") from exc
        if hashlib.sha256(data).hexdigest() != expected:
            raise SourceRevisionError(f"Unsupported upstream source revision: {path}")
        contents[name] = data
    return contents


def _run_worker(payload: dict, python: Path, timeout_s: float) -> dict:
    """No shell, PYTHONPATH injection, caller-selected module or reference JSON."""
    worker = Path(__file__).with_name("_parity_worker.py")
    with tempfile.TemporaryDirectory(prefix="crystalx-parity-") as scratch:
        completed = subprocess.run(
            [str(python), "-I", str(worker)],
            input=json.dumps(payload, allow_nan=False), capture_output=True, text=True,
            timeout=timeout_s, cwd=scratch, check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"Isolated {payload['mode']} worker exited {completed.returncode}: "
                           + completed.stderr[-2000:])
    try:
        result = json.loads(completed.stdout)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("Isolated worker did not return the parity JSON protocol") from exc
    if (not isinstance(result, dict) or result.get("protocol") != PROTOCOL
            or result.get("nonce") != payload["nonce"] or result.get("mode") != payload["mode"]):
        raise ValueError("Worker protocol, run nonce or mode differs from this invocation")
    if result.get("status") not in ("MEASURED", "BLOCKED"):
        raise ValueError("Invalid worker status")
    return result


def _validate_observation(observation: dict, case: ParityCase) -> None:
    """Fail closed on stale/misaligned/nonfinite observations before scoring."""
    request = case.request
    candidates = sorted(request.candidate_atomic_numbers or set(request.atomic_numbers))
    count = request.main_site_count or len(request.site_ids)
    if (observation.get("case_id") != case.case_id
            or observation.get("input_sha256") != _digest(asdict(request))
            or observation.get("site_ids") != list(request.site_ids[:count])
            or observation.get("candidate_atomic_numbers") != candidates):
        raise ValueError("Worker case identity, site ordering or candidate set differs")
    probabilities = observation.get("probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != count:
        raise ValueError("Probability row count differs from main site count")
    for row in probabilities:
        if (not isinstance(row, list) or len(row) != len(candidates)
                or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in row)
                or not math.isclose(sum(row), 1., abs_tol=2e-6)):
            raise ValueError("Invalid full conditional probability distribution")
    first = observation.get("first_pass_labels_all_sites")
    if (not isinstance(first, list) or len(first) != len(request.site_ids)
            or any(type(label) is not int or label not in candidates for label in first)):
        raise ValueError("Invalid first-pass label ordering")
    passes = observation.get("graphs")
    if not isinstance(passes, list) or len(passes) != 2:
        raise ValueError("Expected one observed graph per model pass")
    for graph in passes:
        for kind in ("radius_edges", "effective_edges"):
            edges = graph.get(kind)
            if (not isinstance(edges, list)
                    or any(not isinstance(edge, list) or len(edge) != 2
                           or any(type(index) is not int or not 0 <= index < len(request.site_ids)
                                  for index in edge) for edge in edges)
                    or len({tuple(edge) for edge in edges}) != len(edges)):
                raise ValueError("Invalid observed directed graph")
        if not set(map(tuple, graph["effective_edges"])) <= set(map(tuple, graph["radius_edges"])):
            raise ValueError("Effective edges are not a subset of radius edges")
        eligible = graph.get("eligible_neighbor_counts")
        if (not isinstance(eligible, list) or len(eligible) != len(request.site_ids)
                or any(type(n) is not int or not 1 <= n <= len(request.site_ids) for n in eligible)):
            raise ValueError("Invalid eligible neighbor counts")


def _compare_case(reference: dict, native: dict, case: ParityCase) -> dict:
    for observation in (reference, native):
        _validate_observation(observation, case)
    candidates = reference["candidate_atomic_numbers"]
    sites = []
    for site_id, a, b in zip(reference["site_ids"], reference["probabilities"], native["probabilities"]):
        ra = sorted(range(len(candidates)), key=lambda i: (-a[i], candidates[i]))
        rb = sorted(range(len(candidates)), key=lambda i: (-b[i], candidates[i]))
        sites.append({
            "site_id": site_id,
            "reference_top": candidates[ra[0]], "native_top": candidates[rb[0]],
            "reference_runner_up": candidates[ra[1]], "native_runner_up": candidates[rb[1]],
            "top_changed": ra[0] != rb[0], "runner_up_changed": ra[1] != rb[1],
            "reference_probabilities": a, "native_probabilities": b,
            "max_absolute_probability_delta": max(abs(x - y) for x, y in zip(a, b)),
            "l1_probability_delta": sum(abs(x - y) for x, y in zip(a, b)),
        })
    graphs = []
    for pass_index, (a, b) in enumerate(zip(reference["graphs"], native["graphs"]), 1):
        drift = {"pass": pass_index}
        for key in ("radius_edges", "effective_edges"):
            old, new = set(map(tuple, a[key])), set(map(tuple, b[key]))
            drift[key] = {
                "reference_count": len(old), "native_count": len(new),
                "ordered_equal": a[key] == b[key],
                "removed": sorted(old - new), "added": sorted(new - old),
            }
        for label, observed in (("reference", a), ("native", b)):
            degrees = [0] * len(case.request.site_ids)
            for _, center in observed["radius_edges"]:
                degrees[center] += 1
            counts = observed["eligible_neighbor_counts"]
            drift[label + "_eligible_neighbor_counts"] = counts
            drift[label + "_observed_neighbor_counts"] = degrees
            drift[label + "_truncated_sites"] = [case.request.site_ids[i]
                                                       for i, (n, d) in enumerate(zip(counts, degrees)) if n > d]
        graphs.append(drift)
    return {"case_id": case.case_id, "site_index_to_id": list(case.request.site_ids),
            "candidate_atomic_numbers": candidates, "sites": sites, "graphs": graphs,
            "first_pass_label_changes": [
                {"site_id": site_id, "reference": a, "native": b}
                for site_id, a, b in zip(case.request.site_ids, reference["first_pass_labels_all_sites"],
                                         native["first_pass_labels_all_sites"]) if a != b
            ],
            "top_change_count": sum(site["top_changed"] for site in sites),
            "runner_up_change_count": sum(site["runner_up_changed"] for site in sites),
            "max_absolute_probability_delta": max(site["max_absolute_probability_delta"] for site in sites)}


def measure_parity(
    source_root: Path,
    checkpoint: Path | None = None,
    *,
    expected_sha256: str | None = None,
    checkpoint_kind: str = "official_local",
    provenance: str = "",
    cases: tuple[ParityCase, ...] | None = None,
    devices: tuple[str, ...] = ("cpu", "mps"),
    reference_python: Path | None = None,
    timeout_s: float = 120.,
) -> dict:
    """Return measurements or explicit BLOCKED reasons, never a science pass.

    ``official_local`` requires the owner's provenance and independent expected
    digest; their statement is recorded as an attestation, not authenticated
    publisher provenance. ``synthetic_fixture`` can exercise plumbing only.
    No downloads or installs are attempted. An alternate reference interpreter
    must already contain this adapter/core and actual upstream dependencies.
    """
    if checkpoint_kind not in ("official_local", "synthetic_fixture"):
        raise ValueError("Unknown checkpoint kind")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("Explicit local checkpoint provenance is required")
    if (expected_sha256 is not None and
            (len(expected_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected_sha256))):
        raise ValueError("expected_sha256 must be a SHA-256 hexadecimal digest")
    if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s) or not 0 < timeout_s <= 600):
        raise ValueError("timeout_s must be finite and in (0, 600]")
    devices = tuple(devices)
    if not devices or len(set(devices)) != len(devices) or any(d not in ("cpu", "mps") for d in devices):
        raise ValueError("Choose distinct native cpu/mps devices")
    cases = tuple(fixed_cases() if cases is None else cases)
    if (not 1 <= len(cases) <= 8 or any(not isinstance(case, ParityCase) for case in cases)
            or len({case.case_id for case in cases}) != len(cases)):
        raise ValueError("Provide 1 to 8 uniquely named ParityCases")
    source_root = Path(source_root).expanduser().resolve()
    verify_upstream_sources(source_root)
    suite = [{"case_id": case.case_id, "request": asdict(case.request)} for case in cases]
    report = {
        "protocol": PROTOCOL, "status": "BLOCKED", "blocked_reasons": [],
        "checkpoint_kind": checkpoint_kind, "checkpoint_provenance": provenance,
        "official_checkpoint_claim": "caller_attested_not_publisher_authenticated" if checkpoint_kind == "official_local" else "synthetic_only",
        "provenance_verified_by_publisher": False,
        "source_sha256": dict(UPSTREAM_SHA256), "case_suite_sha256": _digest(suite),
        "cases": suite, "workers": {}, "comparisons": {},
        "scientific_validation_performed": False, "probabilities_calibrated": False,
        "upstream_prediction_parity_verified": False,
        "trained_comparison_measured": False,
        "scientific_tolerance": None, "scientific_decision": "OWNER_REVIEW_REQUIRED",
        "scope": "Model/graph comparison on explicit Cartesian requests; no upstream preprocessing, coverage adjustment or structure acceptance.",
    }
    if checkpoint is None or not Path(checkpoint).expanduser().is_file():
        report["blocked_reasons"].append("Local heavy checkpoint is absent; no trained comparison was run.")
    if expected_sha256 is None:
        report["blocked_reasons"].append("An independently supplied expected checkpoint SHA-256 is required.")
    if report["blocked_reasons"]:
        return report
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    if _file_digest(checkpoint) != expected_sha256.lower():
        raise ValueError("Checkpoint SHA-256 differs from the expected digest")
    report["checkpoint_sha256"] = expected_sha256.lower()
    payload = {"protocol": PROTOCOL, "nonce": uuid4().hex, "source_root": str(source_root),
               "checkpoint": str(checkpoint), "expected_sha256": expected_sha256.lower(), "cases": suite}
    modes = [("upstream_cpu", reference_python or Path(sys.executable))]
    modes += [("native_" + device, Path(sys.executable)) for device in devices]
    for mode, python in modes:
        try:
            result = _run_worker({**payload, "mode": mode}, Path(python), timeout_s)
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            result = {"status": "BLOCKED", "reason": str(exc), "mode": mode}
        report["workers"][mode] = result
        if result["status"] == "BLOCKED":
            report["blocked_reasons"].append(f"{mode}: {result.get('reason', 'unavailable')}")
            continue
        if (result.get("checkpoint_sha256") != expected_sha256.lower()
                or result.get("source_sha256") != dict(UPSTREAM_SHA256)
                or result.get("case_suite_sha256") != report["case_suite_sha256"]
                or result.get("device_used") != ("cpu" if mode == "upstream_cpu" else mode[7:])
                or result.get("fell_back") is not False):
            raise ValueError("Worker checkpoint/source/input/device identity differs")
        if len(result.get("observations", [])) != len(cases):
            raise ValueError("Worker case count differs")
        for observation, case in zip(result["observations"], cases):
            _validate_observation(observation, case)
    reference = report["workers"]["upstream_cpu"]
    if reference["status"] == "MEASURED":
        for device in devices:
            native = report["workers"]["native_" + device]
            if native["status"] == "MEASURED":
                report["comparisons"][device] = [
                    _compare_case(a, b, case)
                    for a, b, case in zip(reference["observations"], native["observations"], cases)
                ]
    if len(report["comparisons"]) == len(devices):
        report["status"] = "MEASURED"
    report["trained_comparison_measured"] = bool(report["comparisons"]) and checkpoint_kind == "official_local"
    return report
