"""Load a locally supplied, hash-pinned CrystalX inference snapshot.

No upstream source is copied, edited or imported as ``crystalx_infer``. Imports
are rewritten in memory inside per-load private namespaces. Only the inference
OutputModel/EquivariantScalar output head is loaded; ASE-dependent molecular
property heads, training models, TorchScript and full-object pickle loading are
outside this adapter's scope. This is compatibility plumbing, not a sandbox for
arbitrary code: all four source hashes must match before *any* snapshot executes.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Callable, Mapping
from uuid import uuid4


AUDITED_SHA256 = MappingProxyType({
    "torchmd_utils.py": "fe0b366af8c7f29dd8317b2a00a89b863216245af07284905cd3ae8b5d3f859d",
    "torchmd_et.py": "01750a2aa941e01a2f689558058fcea0bdee70a2a1fb2fb14f54997b910a0c9c",
    "torchmd_net.py": "b84a5fb6097f5a4a41fd6683f1ea85aed1bfe4dad8b503a6b1a2df3f2aff82cd",
    "noise_output_model.py": "aeff577743fc8d1c9641bc28c435e6c899e5c67d8498d64bce8a4dad409a9b5e",
})


class SourceRevisionError(ValueError):
    """The supplied inference sources do not match the audited snapshot."""


@dataclass(frozen=True)
class LoadedModels:
    TorchMD_ET: type
    TorchMD_Net: type
    EquivariantScalar: type
    identity: Mapping[str, str]
    source_root: Path


class _ScopedImports(ast.NodeTransformer):
    _single = {
        "torch_geometric.nn": {"MessagePassing": "_cx_MessagePassing"},
        "torch_scatter": {"scatter": "_cx_scatter"},
        "torch_cluster": {"radius_graph": "_cx_radius_graph"},
    }

    def visit_ImportFrom(self, node):
        if node.module == "crystalx_infer.models.torchmd_utils":
            values = {alias.name: ast.Attribute(value=ast.Name(id="_cx_utils", ctx=ast.Load()),
                                               attr=alias.name, ctx=ast.Load()) for alias in node.names}
        elif node.module in self._single:
            mapping = self._single[node.module]
            if any(alias.name not in mapping for alias in node.names):
                raise SourceRevisionError(f"Unsupported external symbol in {node.module}")
            values = {alias.name: ast.Name(id=mapping[alias.name], ctx=ast.Load()) for alias in node.names}
        else:
            return node
        replacements = []
        for alias in node.names:
            if alias.name == "*":
                raise SourceRevisionError("Wildcard snapshot imports are unsupported")
            replacements.append(ast.copy_location(ast.Assign(
                targets=[ast.Name(id=alias.asname or alias.name, ctx=ast.Store())],
                value=values[alias.name],
            ), node))
        return replacements


def _tree(data: bytes, filename: str):
    tree = ast.parse(data, filename=filename)
    if filename.endswith("noise_output_model.py"):
        # Preserve only the two required class definitions and their imports.
        # The other heads use ASE masses and are unrelated to atom classification.
        tree.body = [node for node in tree.body if (
            isinstance(node, ast.ImportFrom)
            or isinstance(node, ast.Import) and all(alias.name != "ase" for alias in node.names)
            or isinstance(node, ast.ClassDef) and node.name in {"OutputModel", "EquivariantScalar"}
        )]
    tree = _ScopedImports().visit(tree)
    return ast.fix_missing_locations(tree)


def load_model_classes(source_root: Path, *, radius_graph_fn: Callable | None = None) -> LoadedModels:
    """Load only the audited inference model, with per-load graph dependencies.

    ``source_root`` is the upstream ``inference_repo`` directory. Every required
    file is read and hashed before execution; missing/changed files fail closed.
    A graph callback is scoped to the returned classes, never installed globally.
    Callers should use weights-only state_dict checkpoints, not pickled classes.
    """
    source_root = Path(source_root).expanduser().resolve()
    model_dir = source_root / "crystalx_infer" / "models"
    contents = {}
    identity = {}
    for filename, expected in AUDITED_SHA256.items():
        path = model_dir / filename
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise SourceRevisionError(f"Cannot read required inference source: {path}") from exc
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise SourceRevisionError(f"Unsupported source revision: {path}; expected {expected}, got {actual}")
        contents[filename] = data
        identity[filename] = actual
    if radius_graph_fn is not None and not callable(radius_graph_fn):
        raise TypeError("radius_graph_fn must be callable")

    # Import optional tensor dependencies only after source identity is checked.
    from .message_passing import MessagePassing
    from .ops import radius_graph, scatter

    graph = radius_graph if radius_graph_fn is None else radius_graph_fn
    prefix = f"_crystalx_apple_snapshot_{uuid4().hex}"
    modules = {}
    for filename, data in contents.items():
        name = filename[:-3]
        module = ModuleType(f"{prefix}.{name}")
        module.__file__ = str(model_dir / filename)
        module.__dict__.update({
            "_cx_MessagePassing": MessagePassing,
            "_cx_scatter": scatter,
            "_cx_radius_graph": graph,
            "_cx_utils": modules.get("torchmd_utils"),
        })
        exec(compile(_tree(data, str(model_dir / filename)), str(model_dir / filename), "exec"), module.__dict__)
        modules[name] = module
    return LoadedModels(
        TorchMD_ET=modules["torchmd_et"].TorchMD_ET,
        TorchMD_Net=modules["torchmd_net"].TorchMD_Net,
        EquivariantScalar=modules["noise_output_model"].EquivariantScalar,
        identity=MappingProxyType(identity),
        source_root=source_root,
    )
