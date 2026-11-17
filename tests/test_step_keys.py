"""What every protocol a shipped pipeline writes has to hold, read off the whole corpus.

No two steps under one key. A key is a step's handle: the page anchors it there and the bench's
check mark is kept under it, so two steps sharing one would share a tick. A page must still
render, so the renderer numbers a repeat rather than refusing it; this is where a duplicate is
caught instead, in the pipeline that wrote it, the day it lands.

Every citation resolves, and every troubleshooting row of the four cloning methods carries one
bar the rows named below. `Protocol.audit` sees a key naming no source; nothing but this sees a
row naming no key at all.

Every pipeline is planned on inputs small enough to cost the gate little, since what is asked of
each plan is only its steps and the sources they name. One test takes one pipeline, and a
pipeline is planned once a module, so no test pays for the whole corpus.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from mbio.cloning.goldengate import Plan
    from mbio.protocol import Protocol
    from mbio.sequence import SequenceRecord

#: Three part lists of two members each, which is a whole iGGA build the gate can afford.
PARTS = (
    {"N_a": "MKTAEK", "N_b": "MKTCEK"},
    {"bZIP_a": "WQAFAK", "bZIP_b": "WQAYAK"},
    {"C_a": "MKTHGK", "C_b": "MKTWGK"},
)

#: The orthogonal primer set this repo ships, which is what an iGGA pool is amplified by.
PRIMERS = Path(__file__).parents[1] / "docs" / "examples" / "ap1-library" / "primers.tsv"


def igga_protocols(directory: Path) -> tuple[Protocol, ...]:
    """Every protocol of an iGGA run, with a read-back page for each route a well is read by.

    The build asks for the primer plates and both read-backs, so the run is the whole chain and
    not the shorter one a plainer build writes, and the chain writes a validation protocol a
    route.
    """
    from mbio.sequence import SequenceRecord
    from mbio.snapgene import write_dna
    from synbio.dmx import ROUTES
    from synbio.igga.method import IGGA
    from synbio.igga.plan import plan_igga
    from synbio.igga.project import Build, PrimerPlates

    pad = ("TA" * 80)[:80]
    (directory / "parts.fasta").write_text(
        "".join(f">{name} a part\n{seq}\n" for one in PARTS for name, seq in one.items()),
        encoding="utf-8",
    )
    write_dna(
        SequenceRecord(pad + IGGA.internal_stuffer + pad, topology="circular", name="carrier"),
        directory / "carrier.dna",
    )
    build = Build(
        "library",
        positions=("N", "bZIP", "C"),
        parts=directory / "parts.fasta",
        vector=directory / "carrier.dna",
        host="e-coli-k12",
        oligo_length=350,
        batch_size=96,
        completeness=0.99,
        primers=PRIMERS,
        primer_plates=PrimerPlates(nanomoles=25, stock_um=100, working_ul=100),
        validate_from=0,
        routes=tuple(ROUTES),
    )
    return plan_igga(build, parts=PARTS).chain().protocols


def _protocols(
    pipeline: str,
    plan: Plan,
    puc19: SequenceRecord,
    gfp: SequenceRecord,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Protocol, ...]:
    """Every protocol this one pipeline writes."""
    from mbio.cloning.gateway import plan_gateway
    from mbio.cloning.gibson import plan_gibson
    from mbio.cloning.restriction import plan_restriction

    from .cloning.gateway.records import (
        attb_insert,
        destination_vector,
        donor_vector,
        entry_clone,
    )

    match pipeline:
        case "goldengate":
            return (plan.protocol(),)
        case "gibson":
            return (plan_gibson(puc19, gfp).protocol(),)
        case "restriction":
            return (plan_restriction(puc19, gfp).protocol(),)
        case "gateway":
            return (plan_gateway(entry_clone(gfp.sequence), destination_vector()).protocol(),)
        # Gateway's own second route, which runs two reactions and so builds its stop and its
        # transformation twice.
        case "gateway, BP then LR":
            return (
                plan_gateway(
                    attb_insert(gfp.sequence), destination_vector(), donor=donor_vector()
                ).protocol(),
            )
        case "igga":
            return igga_protocols(tmp_path_factory.mktemp("igga"))
        case _:
            raise AssertionError(f"no shipped pipeline is called {pipeline!r}")


#: The cloning methods, each the way one plan runs it.
CLONING = ("goldengate", "gibson", "restriction", "gateway", "gateway, BP then LR")

#: Every shipped pipeline, by the name its protocols are asked for under.
SHIPPED = (*CLONING, "igga")


@pytest.fixture(scope="module")
def shipped(
    plan: Plan, puc19: SequenceRecord, gfp: SequenceRecord, tmp_path_factory: pytest.TempPathFactory
) -> Callable[[str], tuple[Protocol, ...]]:
    """Read one pipeline's protocols, planned when first asked for and once a module.

    One plan a pipeline, rather than the whole corpus in one fixture: a test then pays for the
    pipeline it names, and the tests after it on that pipeline pay nothing.
    """
    held: dict[str, tuple[Protocol, ...]] = {}

    def read(pipeline: str) -> tuple[Protocol, ...]:
        if pipeline not in held:
            held[pipeline] = _protocols(pipeline, plan, puc19, gfp, tmp_path_factory)
        return held[pipeline]

    return read


@pytest.mark.parametrize("pipeline", SHIPPED)
def test_no_shipped_pipeline_writes_two_steps_under_one_key(
    shipped: Callable[[str], tuple[Protocol, ...]], pipeline: str
) -> None:
    counted = {
        f"{pipeline}: {protocol.title}": Counter(step.key for step in protocol.steps)
        for protocol in shipped(pipeline)
    }
    # Every protocol holds steps, so nothing here can pass by having found none to count.
    assert all(counted.values())
    assert not {
        where: sorted(key for key, seen in keys.items() if seen > 1)
        for where, keys in counted.items()
        if any(seen > 1 for seen in keys.values())
    }


#: The cloning troubleshooting rows no document states, by the pipeline that writes them: each
#: is arithmetic on this run's own volumes, or on how much of the outgrowth was plated.
RUN_FACTS = {
    "goldengate": frozenset({"A lawn", "Too dilute to fit in the reaction"}),
    "gibson": frozenset({"A lawn", "Too dilute to fit in the reaction"}),
    "restriction": frozenset({"A lawn", "Too dilute to fit in the reaction"}),
    "gateway": frozenset({"A lawn", "The DNA does not fit the reaction volume"}),
    "gateway, BP then LR": frozenset(
        {
            "A lawn",
            "The DNA does not fit the reaction volume",
            "The prep is too dilute for the reaction",
        }
    ),
}


@pytest.mark.parametrize("pipeline", SHIPPED)
def test_every_citation_a_shipped_protocol_renders_resolves(
    shipped: Callable[[str], tuple[Protocol, ...]], pipeline: str
) -> None:
    assert [
        check.detail
        for protocol in shipped(pipeline)
        for check in protocol.audit()
        if check.name == "sources" and check.status != "pass"
    ] == []


@pytest.mark.parametrize("pipeline", CLONING)
def test_every_cloning_troubleshooting_row_names_its_source(
    shipped: Callable[[str], tuple[Protocol, ...]], pipeline: str
) -> None:
    rows = [
        row
        for protocol in shipped(pipeline)
        for step in protocol.steps
        for row in step.troubleshooting
    ]
    # Every method writes rows, so nothing here passes by having found none.
    assert rows
    assert {row.problem for row in rows if row.citation is None} == RUN_FACTS[pipeline]
