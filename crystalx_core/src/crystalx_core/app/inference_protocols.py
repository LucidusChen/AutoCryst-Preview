"""Pinned descriptor and coverage protocol helpers; no model execution at import time."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from crystalx_core.domain.chemistry import atomic_number


HEAVY_HELPER_SHA256 = "3490522fcf92817b9b412d18875438e87cefabfaa60d56c7707e2aff5560717d"
INPUT_PROTOCOLS = ("current_labels", "initial_labels", "upstream_ins")


def pinned_upstream_helper(source_root: Path, name: str):
    """Load only one exact audited function, without importing upstream packages.

    This executes the unchanged function AST from the already installed upstream
    helper. It does not import RDKit, training modules, or checkpoint download code.
    Core atomic-number lookup replaces the function's simple symbol lookup only.
    """
    if name not in ("build_sorted_init_z_from_ratio", "enforce_coverage_by_prob"):
        raise ValueError("Unsupported pinned heavy helper")
    path=Path(source_root)/"crystalx_infer/common/heavy.py"
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=HEAVY_HELPER_SHA256:
        raise ValueError("Upstream heavy protocol helper SHA-256 changed")
    definitions=[node for node in ast.parse(data,filename=str(path)).body
                 if isinstance(node,ast.FunctionDef) and node.name==name]
    if len(definitions)!=1:
        raise ValueError("Pinned upstream function unavailable")
    scope={"atomic_num_from_symbol":atomic_number,"__builtins__":__builtins__}
    if name=="build_sorted_init_z_from_ratio":
        import numpy
        scope["np"]=numpy
    else:
        import torch
        scope["torch"]=torch
    module=ast.fix_missing_locations(ast.Module(body=definitions,type_ignores=[]))
    exec(compile(module,str(path),"exec"),scope)
    return scope[name]
