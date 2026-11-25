"""Framework-neutral contracts for experimental element-inference backends.

Coordinates must already be Cartesian Angstrom coordinates, with any symmetry
context supplied explicitly. Model probabilities are not experimental evidence.
This module deliberately imports neither torch nor an engine implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Protocol


class InferenceDevice(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"


@dataclass(frozen=True)
class ElementInferenceRequest:
    site_ids: tuple[str, ...]
    atomic_numbers: tuple[int, ...]
    positions_angstrom: tuple[tuple[float, float, float], ...]
    candidate_atomic_numbers: tuple[int, ...] | None = None
    main_site_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "site_ids", tuple(self.site_ids))
        object.__setattr__(self, "atomic_numbers", tuple(self.atomic_numbers))
        object.__setattr__(self, "positions_angstrom", tuple(tuple(p) for p in self.positions_angstrom))
        count = len(self.site_ids)
        if not count or len(self.atomic_numbers) != count or len(self.positions_angstrom) != count:
            raise ValueError("Nonempty site IDs, atomic numbers and positions must have equal lengths")
        if any(not isinstance(s, str) or not s.strip() for s in self.site_ids):
            raise ValueError("Site IDs must be nonempty strings")
        if len(set(self.site_ids)) != count:
            raise ValueError("Site IDs must be unique, including supplied symmetry context")
        if any(type(z) is not int or not 1 <= z <= 118 for z in self.atomic_numbers):
            raise ValueError("Atomic numbers must be integers in [1, 118]")
        if any(len(p) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                 or not math.isfinite(v) for v in p)
               for p in self.positions_angstrom):
            raise ValueError("Positions must be finite Cartesian triples in Angstrom")
        if self.main_site_count is not None and (type(self.main_site_count) is not int
                                                or not 1 <= self.main_site_count <= count):
            raise ValueError("main_site_count must include at least one and at most all sites")
        if self.candidate_atomic_numbers is not None:
            values = tuple(self.candidate_atomic_numbers)
            if (len(values) < 2 or len(set(values)) != len(values)
                    or any(type(z) is not int or not 1 <= z <= 118 for z in values)):
                raise ValueError("At least two distinct candidate atomic numbers are required")
            object.__setattr__(self, "candidate_atomic_numbers", values)


@dataclass(frozen=True)
class ClassProbability:
    atomic_number: int
    probability: float


@dataclass(frozen=True)
class ElementPrediction:
    site_id: str
    topk: tuple[ClassProbability, ...]

    @property
    def runner_up(self) -> ClassProbability:
        return self.topk[1]


@dataclass(frozen=True)
class ModelIdentity:
    weights_sha256: str
    source_files: tuple[tuple[str, str], ...]
    hidden_channels: int
    num_classes: int
    num_layers: int
    num_rbf: int
    architecture: str = "CrystalX inference TorchMD_ET/EquivariantScalar"
    num_heads: int = 8
    cutoff_angstrom: float = 5.0
    max_num_neighbors: int = 32


@dataclass(frozen=True)
class ElementInferenceResult:
    predictions: tuple[ElementPrediction, ...]
    model_identity: ModelIdentity
    device_requested: InferenceDevice
    device_used: InferenceDevice
    fell_back: bool
    fallback_reason: str
    neighbor_policy: str
    truncated_nodes: int
    probability_kind: str
    warnings: tuple[str, ...]
    wall_clock_s: float
    input_sha256: str
    candidate_atomic_numbers: tuple[int, ...]
    torch_version: str
    probabilities_calibrated: bool = False
    scientific_validation_performed: bool = False
    upstream_prediction_parity_verified: bool = False


class InferenceBackend(Protocol):
    name: str

    def predict_elements(
        self,
        request: ElementInferenceRequest,
        *,
        device: InferenceDevice = InferenceDevice.AUTO,
        allow_cpu_fallback: bool = True,
    ) -> ElementInferenceResult: ...
