"""External-tool resolution and safe invocation.

SHELX and PLATON are user-supplied, separately licensed programs. Nothing here downloads, compiles,
bundles, or redistributes a binary.
"""

from __future__ import annotations

from crystalx_core.tools.adapter import (
    ExternalToolAdapter,
    FileRecord,
    ToolInvocationRecord,
    snapshot_directory,
)
from crystalx_core.tools.binfmt import probe_binary, probe_bytes
from crystalx_core.tools.registry import (
    PLATON_PREFERENCE,
    TOOL_SPECS,
    ToolSpec,
    known_install_dirs,
    olex2_candidate_dirs,
    spec_for,
)
from crystalx_core.tools.resolver import (
    ROSETTA_MARKER_PATHS,
    ROSETTA_REMEDY,
    ResolverConfig,
    ToolResolver,
    detect_host_platform,
    rosetta_is_installed,
)
from crystalx_core.tools.runner import (
    CommandExecutor,
    ExecutionOutcome,
    ExecutionPlan,
    ProcessRunner,
    RawResult,
    SubprocessExecutor,
    plan_for_tool,
)

__all__ = [
    "PLATON_PREFERENCE",
    "ROSETTA_MARKER_PATHS",
    "ROSETTA_REMEDY",
    "TOOL_SPECS",
    "CommandExecutor",
    "ExecutionOutcome",
    "ExternalToolAdapter",
    "FileRecord",
    "ToolInvocationRecord",
    "snapshot_directory",
    "ExecutionPlan",
    "ProcessRunner",
    "RawResult",
    "ResolverConfig",
    "SubprocessExecutor",
    "ToolResolver",
    "ToolSpec",
    "detect_host_platform",
    "known_install_dirs",
    "olex2_candidate_dirs",
    "plan_for_tool",
    "probe_binary",
    "probe_bytes",
    "rosetta_is_installed",
    "spec_for",
]
