"""Design a primer, or a pair, by choosing the annealing region a placement allows."""

import heapq
import math
from collections.abc import Callable, Container, Iterator
from dataclasses import dataclass

from mbio.checks import STATUSES, Check, Status, worst
from mbio.primers.evaluation import (
    _THERMO_MAX,
    PairReport,
    PrimerReport,
    evaluate_primer,
    pair_checks,
    screen_primer,
)
from mbio.primers.placement import Placement
from mbio.primers.polymerase import Q5, Polymerase
from mbio.primers.thresholds import TARGET_TM, THRESHOLDS, Thresholds
from mbio.sequence import (
    BindingSite,
    Primer,
    Segment,
    SequenceRecord,
    Strand,
    reverse_complement,
)

#: How a candidate ranks: its worst status, how many checks did not pass, whether its length
#: lies outside the band a length passes on, how far its Tm sits from the target, how far its
#: 5' end sits from the position asked for, its length, and that end, which settles a tie.
type _Rank = tuple[int, int, int, float, int, int, int]

#: A pair's rank, or the best one a pair could still reach: the same keys, summed.
type _PairRank = tuple[float, ...]

#: A judged pair: its rank, the order it was judged in, which settles a tie, and what it scored.
type _Judged = tuple[_PairRank, int, _Option, _Option, tuple[Check, ...]]


def design_primer(
    template: SequenceRecord,
    position: int,
    strand: Strand,
    *,
    placement: Placement | None = None,
    tail: str = "",
    name: str = "",
    target_tm: float = TARGET_TM,
    polymerase: Polymerase = Q5,
    thresholds: Thresholds = THRESHOLDS,
) -> Primer:
    """Design a primer annealing at a position on a template, or anywhere a `Placement` allows.

    A forward primer's annealing region starts at `position` and reads towards higher
    coordinates; a reverse primer's ends there and reads back. Either crosses the origin of a
    circular template. `placement` widens the search to every binding site it allows, and
    `position` stays the 5' end asked for. Each candidate, at every length `Thresholds.length`
    does not fail, is judged whole by `evaluate_primer` on the template, structures, runs and
    off-target sites included, so a designed primer is worse than a pass only where nothing the
    placement allows passes. Candidates rank by worst status, then by how many checks did not
    pass, then by the band a length passes on, then by the Tm nearest `target_tm`, then by
    nearness to `position`, and the shorter primer breaks a tie. `tail` joins its 5' end and
    stays out of the binding site. The choice is the one judging every candidate whole makes,
    but a candidate `screen_primer` already ranks below the best is never aligned.

    Raises
    ------
    ValueError
        If no annealing region fits the template there, or fits the placement, or if
        `Thresholds.length` lets a candidate run past the 60 nt primer3 aligns. Such a band is
        refused whole, never cut short, so a design tries every length it was given.

    Examples
    --------
    >>> template = SequenceRecord(
    ...     "GGCGTAATCATGGTCATAGCTGTTTCCTGTGTGAAATTGTTATCCGCT"
    ...     "TACAGGATCCACTAGTAACGGCCGCCAGTGTGCTGGAATTCGCCCTTA"
    ... )
    >>> design_primer(template, 0, Strand.FORWARD, name="MCS fwd").sequence
    'GGCGTAATCATGGTCATAGC'

    Anchored, where a tail has to join the 5' end at one place and only the length may vary:

    >>> anchored = Placement(five_prime=Segment(48, 49))
    >>> design_primer(template, 48, Strand.FORWARD, placement=anchored, tail="TTGGTCTCA").sequence
    'TTGGTCTCATACAGGATCCACTAGTAACGG'

    Near a target, where the 3' end has to land 10 to 21 bases before a junction:

    >>> junction = 48
    >>> near = Placement(three_prime=Segment(junction - 21, junction - 9))
    >>> primer = design_primer(template, junction - 40, Strand.FORWARD, placement=near)
    >>> primer.sequence, junction - primer.binding_sites[0].end
    ('ATCATGGTCATAGCTGTTTCC', 21)

    In a region, where both ends are free inside a flank:

    >>> region = Placement(five_prime=Segment(0, 6), three_prime=Segment(20, 40))
    >>> design_primer(template, 0, Strand.FORWARD, placement=region).sequence
    'CGTAATCATGGTCATAGCTGTT'
    """
    candidates = _candidates(
        template, position, strand, placement, tail, name, target_tm, polymerase, thresholds
    )
    floors = sorted(_fitted(candidates, position, placement), key=lambda one: one.floor)
    best = _judged(floors[0], template, target_tm, position, polymerase, thresholds)
    for candidate in floors[1:]:
        # In order of floor, so once a floor is no better than the best, none after it can be.
        if candidate.floor >= best.rank:
            break
        judged = _judged(candidate, template, target_tm, position, polymerase, thresholds)
        best = min(best, judged, key=lambda option: option.rank)
    return best.primer


def design_pair(
    template: SequenceRecord,
    start: int,
    end: int,
    *,
    forward_placement: Placement | None = None,
    reverse_placement: Placement | None = None,
    forward_tail: str = "",
    reverse_tail: str = "",
    forward_name: str = "",
    reverse_name: str = "",
    target_tm: float = TARGET_TM,
    polymerase: Polymerase = Q5,
    thresholds: Thresholds = THRESHOLDS,
    exclude: Container[BindingSite] = (),
) -> tuple[Primer, Primer]:
    """Design a pair amplifying `start` to `end`, judged on what a pair adds as well.

    `end` passes the length of a circular template when the amplicon crosses the origin. Each
    candidate is judged as in `design_primer`, and every pair of them adds `pair_checks`: the
    gap between the two Tms, the heterodimers and the products. The pair chosen ranks best on
    its worst status, then on the same keys summed over both primers, so a pair worse than a
    pass means nothing here passes; `ranked_pairs` gives the rest in that order. A placement
    widens either primer's search; the two bound the amplicon between them, which is why
    nothing takes an amplicon size. Neither primer takes a binding site in `exclude`.

    Raises
    ------
    ValueError
        If no annealing region fits the template at either end, or fits that end's placement,
        or `exclude` holds every one at an end, or `Thresholds.length` is refused as in
        `design_primer`.

    Examples
    --------
    >>> template = SequenceRecord(
    ...     "GGCGTAATCATGGTCATAGCTGTTTCCTGTGTGAAATTGTTATCCGCT"
    ...     "TACAGGATCCACTAGTAACGGCCGCCAGTGTGCTGGAATTCGCCCTTA"
    ... )
    >>> [primer.sequence for primer in design_pair(template, 0, len(template))]
    ['GGCGTAATCATGGTCATAGC', 'TAAGGGCGAATTCCAGCA']

    In a region, where each primer is free inside its own flank and the two bound the amplicon:

    >>> forward = Placement(five_prime=Segment(0, 6), three_prime=Segment(15, 35))
    >>> reverse = Placement(five_prime=Segment(90, 96), three_prime=Segment(60, 80))
    >>> pair = design_pair(
    ...     template,
    ...     0,
    ...     len(template),
    ...     forward_placement=forward,
    ...     reverse_placement=reverse,
    ... )
    >>> [primer.sequence for primer in pair]
    ['CGTAATCATGGTCATAGCTGTT', 'CGAATTCCAGCACACTGG']
    """
    best = next(
        ranked_pairs(
            template,
            start,
            end,
            forward_placement=forward_placement,
            reverse_placement=reverse_placement,
            forward_tail=forward_tail,
            reverse_tail=reverse_tail,
            forward_name=forward_name,
            reverse_name=reverse_name,
            target_tm=target_tm,
            polymerase=polymerase,
            thresholds=thresholds,
            exclude=exclude,
        )
    )
    return best.forward.primer, best.reverse.primer


def ranked_pairs(
    template: SequenceRecord,
    start: int,
    end: int,
    *,
    forward_placement: Placement | None = None,
    reverse_placement: Placement | None = None,
    forward_tail: str = "",
    reverse_tail: str = "",
    forward_name: str = "",
    reverse_name: str = "",
    target_tm: float = TARGET_TM,
    polymerase: Polymerase = Q5,
    thresholds: Thresholds = THRESHOLDS,
    exclude: Container[BindingSite] = (),
) -> Iterator[PairReport]:
    """Yield every pair `design_pair` chooses between, best first, judged as by `evaluate_pair`.

    The first is the pair `design_pair` chooses. A pair is judged only once no pair still to
    judge could rank above it, and a candidate is judged whole only once the search reaches it,
    so taking the first few judges few. A pair holding a binding site
    in `exclude` is passed over, and `exclude` is read as pairs are taken: a site added to it
    between two pairs stays out of every later one.

    Raises
    ------
    ValueError
        As `design_pair` does, on the call rather than when the first pair is taken.

    Examples
    --------
    >>> template = SequenceRecord(
    ...     "GGCGTAATCATGGTCATAGCTGTTTCCTGTGTGAAATTGTTATCCGCT"
    ...     "TACAGGATCCACTAGTAACGGCCGCCAGTGTGCTGGAATTCGCCCTTA"
    ... )
    >>> excluded = set()
    >>> pairs = ranked_pairs(template, 0, len(template), exclude=excluded)
    >>> best = next(pairs)
    >>> [primer.sequence for primer in (best.forward.primer, best.reverse.primer)]
    ['GGCGTAATCATGGTCATAGC', 'TAAGGGCGAATTCCAGCA']

    A site excluded once a pair is taken stays out of every pair after it:

    >>> excluded.add(best.reverse.primer.binding_sites[0])
    >>> after = next(pairs)
    >>> [primer.sequence for primer in (after.forward.primer, after.reverse.primer)]
    ['GGCGTAATCATGGTCATAGC', 'TAAGGGCGAATTCCAGCAC']
    """
    forwards = _candidates(
        template,
        start,
        Strand.FORWARD,
        forward_placement,
        forward_tail,
        forward_name,
        target_tm,
        polymerase,
        thresholds,
    )
    reverses = _candidates(
        template,
        end,
        Strand.REVERSE,
        reverse_placement,
        reverse_tail,
        reverse_name,
        target_tm,
        polymerase,
        thresholds,
    )
    _fitted(forwards, start, forward_placement)
    _fitted(reverses, end, reverse_placement)
    for candidates in (forwards, reverses):
        if all(candidate.site in exclude for candidate in candidates):
            strand = candidates[0].site.strand.name.lower()
            raise ValueError(f"every {strand} binding site is excluded")
    by_rank = (
        _ByRank(
            forwards, lambda one: _judged(one, template, target_tm, start, polymerase, thresholds)
        ),
        _ByRank(
            reverses, lambda one: _judged(one, template, target_tm, end, polymerase, thresholds)
        ),
    )
    return _in_rank_order(*by_rank, template, polymerase, thresholds, exclude)


@dataclass(frozen=True, slots=True)
class _Option:
    """One candidate binding site: the primer it makes, what it scored, how it ranks."""

    primer: Primer
    report: PrimerReport
    rank: _Rank

    @property
    def site(self) -> BindingSite:
        """Return the binding site."""
        return self.primer.binding_sites[0]


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One binding site a placement allows, ranked before it is judged whole.

    `floor` is its rank on `screen_primer` alone. Judging it whole only adds verdicts, so it
    never ranks better than that.
    """

    primer: Primer
    five: int
    floor: _Rank

    @property
    def site(self) -> BindingSite:
        """Return the binding site."""
        return self.primer.binding_sites[0]


class _ByRank:
    """Candidates in the order of their rank, each judged whole only when the order reaches it.

    The candidates wait in order of floor. The least is judged and waits again under its rank,
    until the least waiting is one already judged: no candidate still waiting can rank above it,
    so it is the next in rank order.
    """

    def __init__(
        self, candidates: list[_Candidate], judge: Callable[[_Candidate], _Option]
    ) -> None:
        self._judge = judge
        self._waiting: list[tuple[_Rank, int, _Candidate | _Option]] = [
            (candidate.floor, order, candidate) for order, candidate in enumerate(candidates)
        ]
        heapq.heapify(self._waiting)
        self._ranked: list[_Option] = []

    def at(self, index: int) -> _Option | None:
        """Return the option at this place in rank order, or ``None`` past the last."""
        while len(self._ranked) <= index and self._waiting:
            _, order, entry = heapq.heappop(self._waiting)
            if isinstance(entry, _Option):
                self._ranked.append(entry)
            else:
                judged = self._judge(entry)
                heapq.heappush(self._waiting, (judged.rank, order, judged))
        return self._ranked[index] if index < len(self._ranked) else None

    def run(self, status: int) -> "_Run":
        """Return the options of one status, which follow every better one in rank order."""
        start = 0
        while (option := self.at(start)) is not None and option.rank[0] < status:
            start += 1
        return _Run(self, start, status)


@dataclass(frozen=True, slots=True)
class _Run:
    """The options of one status: a stretch of a rank order, read as far as it is asked."""

    order: _ByRank
    start: int
    status: int

    def at(self, index: int) -> _Option | None:
        """Return the option at this place in the run, or ``None`` past its end."""
        option = self.order.at(self.start + index)
        return option if option is not None and option.rank[0] == self.status else None


def _candidates(
    template: SequenceRecord,
    position: int,
    strand: Strand,
    placement: Placement | None,
    tail: str,
    name: str,
    target_tm: float,
    polymerase: Polymerase,
    thresholds: Thresholds,
) -> list[_Candidate]:
    """Return every binding site the placement allows, each ranked on what needs no alignment.

    A band allowing a site longer than primer3 aligns is refused here, before any is judged.
    """
    if _longest(thresholds) > _THERMO_MAX:
        raise ValueError(
            f"primer3 aligns at most {_THERMO_MAX} nt, and this length band allows "
            f"{_longest(thresholds)}: {thresholds.length}"
        )
    candidates = []
    for site in _sites(template, position, strand, placement, thresholds):
        bases = template.bases(site.start, site.end)
        sequence = bases if strand is Strand.FORWARD else reverse_complement(bases)
        primer = Primer(name, tail + sequence, binding_sites=(site,))
        five = site.start if strand is Strand.FORWARD else site.end
        screened = PrimerReport(
            primer, screen_primer(primer, polymerase=polymerase, thresholds=thresholds)
        )
        floor = _rank(screened, target_tm, five, position, template)
        candidates.append(_Candidate(primer, five, floor))
    return candidates


def _judged(
    candidate: _Candidate,
    template: SequenceRecord,
    target_tm: float,
    position: int,
    polymerase: Polymerase,
    thresholds: Thresholds,
) -> _Option:
    """Return a candidate judged whole by `evaluate_primer`, and ranked on that."""
    primer = candidate.primer
    report = evaluate_primer(primer, template, polymerase=polymerase, thresholds=thresholds)
    return _Option(primer, report, _rank(report, target_tm, candidate.five, position, template))


def _sites(
    template: SequenceRecord,
    position: int,
    strand: Strand,
    placement: Placement | None,
    thresholds: Thresholds,
) -> Iterator[BindingSite]:
    """Yield every binding site a placement allows, at every length the band does not fail."""
    length = len(template)
    circular = template.topology == "circular"
    shortest = int(_finite(thresholds.length.warn_low, thresholds.length.low))
    longest = min(_longest(thresholds), length)
    anchors, five = _anchors(position, placement)
    for anchor in anchors:
        for size in range(shortest, longest + 1):
            # An anchor is one end: the start of the site where it is the 5' end of a forward
            # primer or the 3' end of a reverse one, and a length back from it otherwise.
            start = anchor if (strand is Strand.FORWARD) == five else anchor - size
            if circular:
                start %= length
            if not template.fits(start, start + size):
                continue
            site = BindingSite(start, start + size, strand)
            if placement is None or placement.allows(site, template):
                yield site


def _longest(thresholds: Thresholds) -> int:
    """Return the longest annealing region a design tries: the longest the band does not fail."""
    return int(_finite(thresholds.length.warn_high, thresholds.length.high))


def _anchors(position: int, placement: Placement | None) -> tuple[list[int], bool]:
    """Return the positions the end a placement bounds may take, and whether that end is 5'."""
    if placement is None:
        return [position], True
    five = _positions(placement.five_prime)
    return (five, True) if five else (_positions(placement.three_prime), False)


def _positions(span: Segment | None) -> list[int]:
    """Return every position a span holds, past the length where it runs across the origin."""
    return [] if span is None else list(range(span.start, span.end))


def _fitted(
    candidates: list[_Candidate], position: int, placement: Placement | None
) -> list[_Candidate]:
    """Return the candidates, refusing a position or a placement holding no annealing region."""
    if not candidates:
        where = (
            f"this template at {position}" if placement is None else f"this placement: {placement}"
        )
        raise ValueError(f"no annealing region fits {where}")
    return candidates


def _rank(
    report: PrimerReport, target_tm: float, five: int, position: int, template: SequenceRecord
) -> _Rank:
    return (
        STATUSES.index(report.status),
        sum(check.status not in (None, "pass") for check in report.checks),
        int(report["length"].status != "pass"),
        abs(report["tm"].value - target_tm),
        _apart(five, position, template),
        int(report["length"].value),
        five % len(template),
    )


def _apart(position: int, wanted: int, template: SequenceRecord) -> int:
    """Return how far a 5' end lies from the position asked for, the short way round a circle."""
    if template.topology != "circular":
        return abs(position - wanted)
    gap = (position - wanted) % len(template)
    return min(gap, len(template) - gap)


def _in_rank_order(
    forwards: _ByRank,
    reverses: _ByRank,
    template: SequenceRecord,
    polymerase: Polymerase,
    thresholds: Thresholds,
    exclude: Container[BindingSite],
) -> Iterator[PairReport]:
    """Yield every pair best first, judging pairs in the order of their bounds.

    No pair ranks above its bound, so a judged pair ranking above the next bound ranks above
    every pair still to judge, and is yielded before that one is judged. `exclude` may grow
    while a pair is yielded, so it is read after every yield.
    """
    judged: list[_Judged] = []
    pairs = _by_bound(forwards, reverses)
    for order, (bound, forward, reverse) in enumerate(pairs):
        yield from _taken(judged, bound, polymerase, exclude)
        if _excluded(forward, reverse, exclude):
            continue
        checks = pair_checks(forward.report, reverse.report, template, thresholds=thresholds)
        rank = _pair_rank(bound, forward, reverse, checks)
        heapq.heappush(judged, (rank, order, forward, reverse, checks))
    yield from _taken(judged, None, polymerase, exclude)


def _taken(
    judged: list[_Judged],
    bound: _PairRank | None,
    polymerase: Polymerase,
    exclude: Container[BindingSite],
) -> Iterator[PairReport]:
    """Yield, best first, every judged pair ranking above a bound, or every one without it.

    A pair holding an excluded site is dropped instead.
    """
    while judged and (bound is None or judged[0][0] < bound):
        _, _, forward, reverse, checks = heapq.heappop(judged)
        if not _excluded(forward, reverse, exclude):
            yield _report(forward, reverse, checks, polymerase)


def _excluded(forward: _Option, reverse: _Option, exclude: Container[BindingSite]) -> bool:
    return forward.site in exclude or reverse.site in exclude


def _report(
    forward: _Option, reverse: _Option, checks: tuple[Check, ...], polymerase: Polymerase
) -> PairReport:
    """Return what `evaluate_pair` reports, from what a design has judged already."""
    length = int(checks[-1].value) or None
    return PairReport(
        forward.report,
        reverse.report,
        checks,
        polymerase.annealing_temperature(forward.report["tm"].value, reverse.report["tm"].value),
        length,
        None if length is None else polymerase.extension_seconds(length),
    )


def _pair_rank(
    bound: _PairRank, forward: _Option, reverse: _Option, checks: tuple[Check, ...]
) -> _PairRank:
    verdicts: list[Status] = [check.status for check in checks if check.status is not None]
    status = worst((forward.report.status, reverse.report.status, *verdicts))
    return (
        STATUSES.index(status),
        bound[1] + sum(one != "pass" for one in verdicts),
        *bound[2:],
        forward.rank[-1],
    )


def _by_bound(forwards: _ByRank, reverses: _ByRank) -> Iterator[tuple[_PairRank, _Option, _Option]]:
    """Yield every pair of options, lowest bound first, judging none of the pairs.

    A bound leads with the worse status of its pair, so every pair of one worse status comes
    before any of the next, and the runs of a status are read only once the search reaches it.
    """
    statuses = range(len(STATUSES))
    for worse in statuses:
        runs = [
            (forwards.run(one), reverses.run(other))
            for one in statuses
            for other in statuses
            if max(one, other) == worse
        ]
        groups = [_summed(*pair) for pair in runs if all(run.at(0) is not None for run in pair)]
        yield from heapq.merge(*groups, key=lambda entry: entry[0])


def _summed(forwards: _Run, reverses: _Run) -> Iterator[tuple[_PairRank, _Option, _Option]]:
    """Yield pairs of one status each, in rising order of their bound.

    The bounds of a run rise with either index, so the next pair is always a step from one
    already taken: the walk of a sorted matrix, and it reaches the far corner only if asked.
    """
    heap: list[tuple[_PairRank, int, int, _Option, _Option]] = []
    seen: set[tuple[int, int]] = set()

    def reach(one: int, other: int) -> None:
        forward, reverse = forwards.at(one), reverses.at(other)
        if forward is not None and reverse is not None and (one, other) not in seen:
            seen.add((one, other))
            heapq.heappush(heap, (_bound(forward, reverse), one, other, forward, reverse))

    reach(0, 0)
    while heap:
        bound, one, other, forward, reverse = heapq.heappop(heap)
        yield bound, forward, reverse
        reach(one + 1, other)
        reach(one, other + 1)


def _bound(forward: _Option, reverse: _Option) -> _PairRank:
    """Return the best rank a pair could reach before what only a pair adds is judged.

    Those checks can only raise the worst status or add to the checks that did not pass, so no
    pair ranks better than this.
    """
    return (
        max(forward.rank[0], reverse.rank[0]),
        *(one + other for one, other in zip(forward.rank[1:], reverse.rank[1:], strict=True)),
    )


def _finite(value: float, fallback: float) -> float:
    return value if math.isfinite(value) else fallback
