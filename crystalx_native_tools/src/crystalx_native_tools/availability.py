"""What is known about official native builds and source access, per tool, with the evidence.

Researched 2026-10-22. ``unknown`` means that no authoritative source this project could read
establishes the fact -- not that the thing does not exist. The official SHELX and PLATON websites
refuse automated reading (robots exclusion), so their download areas were not inspected; the owner
can check them directly. Local header facts (what the copies in ``toolbox/`` are) are measured by
:mod:`crystalx_native_tools.inventory`, not asserted here.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Evidence", "OfficialAvailability", "availability_for", "RESEARCHED_ON"]

RESEARCHED_ON = "2026-10-22"

_SPEK_2026 = (
    "A. L. Spek, Acta Cryst. (2026) C82, 595-602, in the collection 'A Tribute to George "
    "Sheldrick' (https://journals.iucr.org/c/issues/2026/10/00/gz3274/gz3274.pdf)"
)
_LINDEN_2026 = (
    "A. Linden, 'Installing PLATON on a Mac and getting Olex2 to use it', version 1.5.2026 "
    "(local copy: platon_instruction/Platon_installation_on_a_Mac.pdf)"
)
_PLATON_NOTICE = "PLATON-INSTALLATION-NOTICE and README (local copies in platon_instruction/)"
_WIKIPEDIA = "https://en.wikipedia.org/wiki/SHELX (secondary; cites the SHELX homepage)"
_SHELX_SITE = "https://shelx.uni-goettingen.de/ (robots exclusion: not read automatically)"
_PLATON_SITE = "https://www.platonsoft.nl/xraysoft/Mac-OSX/platon/ (robots exclusion: not read automatically)"
_MITSUDO = "https://mitsudo.net/en/building-a-crystal-structure-analysis-environment-on-a-mac/"


@dataclass(frozen=True)
class Evidence:
    statement: str
    status: str
    """``verified`` (an authoritative source says so), ``secondary``, or ``unknown``."""

    source: str


@dataclass(frozen=True)
class OfficialAvailability:
    tool: str
    native_arm64_build: str
    """``unknown``, or ``built_locally_from_source`` for PLATON."""

    source_code: str
    """``distributed_as_executables`` (current SHELX: how it is distributed, not proof that no
    source-access arrangement exists), ``available_free_of_charge`` (PLATON) or ``unknown``."""

    evidence: tuple[Evidence, ...]
    next_step: str

    def to_record(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "native_arm64_build": self.native_arm64_build,
            "source_code": self.source_code,
            "researched_on": RESEARCHED_ON,
            "evidence": [
                {"statement": e.statement, "status": e.status, "source": e.source} for e in self.evidence
            ],
            "next_step": self.next_step,
        }


_SHELX_EVIDENCE = (
    Evidence(
        "Current SHELX versions are supplied as executables; earlier versions were distributed as "
        "source code. This describes the distribution; it does not establish whether source can be "
        "obtained under some other arrangement.",
        "verified",
        _SPEK_2026,
    ),
    Evidence(
        "Executables are provided for Windows, Linux and macOS; the current software depends on "
        "libraries for Intel-based hardware, described as a possible future issue.",
        "verified",
        _SPEK_2026,
    ),
    Evidence(
        "No accessible authoritative source states whether a native arm64 (Apple Silicon) macOS "
        "build of this program exists. The official download area was not inspected.",
        "unknown",
        _SHELX_SITE,
    ),
    Evidence(
        "Free of charge for academic institutions; commercial users need a licence. Downloads "
        "require registration.",
        "secondary",
        _WIKIPEDIA,
    ),
    Evidence(
        "The maintenance of the SHELX software is credited to Isabel Uson and Regine Herbst-Irmer.",
        "verified",
        _SPEK_2026,
    ),
)

_SHELX_NEXT = (
    "Check the registered SHELX download area for an Apple Silicon build of this program and "
    "record what it offers. Without one, the x86_64 build needs Rosetta 2 (the machine owner's "
    "decision). A wrapper cannot remove that dependency; only a native executable, or source whose "
    "terms allow a local build, can."
)

_PLATON_EVIDENCE = (
    Evidence(
        "The PLATON Fortran source code is available free of charge and runs natively on macOS "
        "once compiled.",
        "verified",
        _SPEK_2026,
    ),
    Evidence(
        "No statically linked compiled versions are provided for the Mac; users compile PLATON "
        "themselves (gfortran; the full build links XQuartz X11).",
        "verified",
        _LINDEN_2026,
    ),
    Evidence(
        "A stripped, Fortran-only 'special' version without graphics exists for validation jobs; "
        "its build recipe is on the PLATON site, which was not read.",
        "verified",
        _LINDEN_2026 + "; " + _PLATON_SITE,
    ),
    Evidence(
        "Validation ('platon -u name.cif') needs check.def in the working directory or at $CHECKDEF; "
        "check.def carries a yymmdd version that should match the executable.",
        "verified",
        _PLATON_NOTICE,
    ),
    Evidence(
        "When PLATON cannot find check.def it creates one in the current working directory and uses "
        "it ('platon -z2' also creates it).",
        "verified",
        _LINDEN_2026 + "; " + _PLATON_NOTICE,
    ),
    Evidence(
        "A common Mac set-up copies platon and check.def to /usr/local/bin and sets CHECKDEF.",
        "secondary",
        _MITSUDO,
    ),
)

_PLATON_NEXT = (
    "Reuse the local arm64 build. Its launch and a real validation run are separate checks: run "
    "the opt-in launch probe, then an authorised validation case."
)

_SHELX_FAMILY = ("shelxt", "shelxl", "shelxs", "shelxc", "shelxd", "shelxe", "anode", "ciftab", "shredcif")


def availability_for(tool: str) -> OfficialAvailability | None:
    if tool in _SHELX_FAMILY:
        return OfficialAvailability(
            tool=tool,
            native_arm64_build="unknown",
            source_code="distributed_as_executables",
            evidence=_SHELX_EVIDENCE,
            next_step=_SHELX_NEXT,
        )
    if tool in ("platon", "platon_special"):
        return OfficialAvailability(
            tool=tool,
            native_arm64_build="built_locally_from_source",
            source_code="available_free_of_charge",
            evidence=_PLATON_EVIDENCE,
            next_step=_PLATON_NEXT,
        )
    return None
