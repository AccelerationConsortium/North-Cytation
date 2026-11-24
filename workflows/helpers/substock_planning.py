"""Plan substocks for a list of target concentrations, lowest error first.

Pure code, no hardware. One material (e.g. SDS) per call.

Rule (decided 2026-11-24, docs/plans/surfactant_target_samples_plan.md):
  1. Every target should be made within max_error (relative, e.g. 0.03 = 3%).
  2. Existing sources (stock, leftover substocks) are used where they meet the limit.
  3. Targets left over get the FEWEST new substocks that bring them under the limit.
     Concentrations are chosen from the targets (no fixed ladder); among candidates
     covering the same targets, the one with the lowest error wins.
  4. A target no source can reach (too dilute) gets an intermediate dilution first.
  5. A target that can be made but not within max_error is still made, by its lowest-error
     route (within OVER_LIMIT_SLACK, so nearby ones can share a substock), and flagged.
  6. After all substocks are chosen, each target uses its lowest-error source.

Error of one transfer comes from transfer_error(volume_ul) -> relative SD (or None if
that volume has no validated estimate; such volumes are not used). Errors along a
chain (stock -> substock -> well) are combined in quadrature, i.e. ASSUMED independent.
Water added to make up volume is ASSUMED exact (reservoir, as in sample tracking).
"""
import math

import numpy as np

GRID_POINTS = 400        # candidate concentrations searched per round
OVER_LIMIT_SLACK = 0.10  # an over-limit target accepts up to 10% above its best error
_REL = 1e-9


def make_transfer_error(uncertainty, liquid, tips):
    """transfer_error(volume_ul) from a DispenseUncertainty.

    tips: [(tip_name, max_ul)] in order of preference, e.g. small_tip up to 195 uL then
    large_tip; the first tip whose max_ul fits the volume is the one the robot would use.
    Volume above the largest validated volume of the last tip is split into equal parts;
    random errors add in quadrature, shared errors add linearly (they move together).
    Returns relative SD, or None when the volume has no validated estimate.
    """
    largest = max((v for v, r, s in uncertainty.table(liquid, tips[-1][0]) if r is not None),
                  default=None)

    def single(volume_ul):
        for tip, max_ul in tips:
            if volume_ul <= max_ul + _REL:
                return uncertainty.lookup(liquid, tip, volume_ul)
        return None

    def transfer_error(volume_ul):
        if volume_ul <= 0:
            return None
        parts = 1
        if largest is not None and volume_ul > largest + _REL:
            parts = math.ceil(volume_ul / largest - _REL)
        found = single(volume_ul / parts)
        if found is None:
            return None
        random_ul, shared_ul = found
        sd = math.sqrt(parts * random_ul ** 2 + (parts * shared_ul) ** 2)
        return sd / volume_ul

    return transfer_error


def _quad(*errors):
    return math.sqrt(sum(e * e for e in errors))


class _Source:
    def __init__(self, name, conc, error, available_ml=None, parent=None, transfer_ul=None):
        self.name, self.conc, self.error = name, float(conc), float(error)
        self.available_ml, self.parent, self.transfer_ul = available_ml, parent, transfer_ul
        self.new = parent is not None


def plan_substocks(targets, sources, settings, transfer_error, max_ul=None):
    """Plan sources for every target.

    targets:  {sample_id: in-well concentration} (same units as source conc). 0 = absent.
    sources:  [{"name", "conc", "error" (relative SD of conc), "available_ml" or None}]
              existing stock and substocks with known contents.
    settings: well_ul (total well volume), well_min_ul, well_max_ul,
              substock_ml (final volume of a new substock), substock_min_ul,
              dead_ml (unusable volume per vial), max_error, name_prefix.
    max_ul:   optional {sample_id: largest volume of this material in that well}, below
              well_max_ul (wells holding several materials share the well).
    Returns {"wells", "substocks", "usage", "infeasible", "sources"}. "infeasible" holds only
    targets no route can make (above the strongest source, or no validated volume at all);
    wells over max_error are planned and flagged over_limit.
    """
    s = settings
    pool = [_Source(x["name"], x["conc"], x["error"], x.get("available_ml")) for x in sources]
    if not pool:
        raise ValueError("At least one source (the stock) is required")
    max_transfer_ul = s["substock_ml"] * 1000
    active = {k: float(t) for k, t in targets.items() if t and t > 0}
    cap = {k: min(s["well_max_ul"], (max_ul or {}).get(k, s["well_max_ul"])) for k in active}
    limit = {k: s["max_error"] for k in active}   # raised for targets that can't meet max_error
    impossible = set()

    def well_error(source, k):
        v = active[k] * s["well_ul"] / source.conc
        if not (s["well_min_ul"] - _REL <= v <= cap[k] + _REL):
            return None
        e = transfer_error(v)
        return None if e is None else _quad(source.error, e)

    def best_existing(k):
        options = [(err, src) for src in pool if (err := well_error(src, k)) is not None]
        return min(options, key=lambda o: o[0]) if options else None

    def make(conc, parent):
        """Recipe for conc from parent within transfer limits: (error, transfer_ul) or None."""
        transfer = conc / parent.conc * max_transfer_ul
        if not (s["substock_min_ul"] - _REL <= transfer <= max_transfer_ul - _REL):
            return None
        e = transfer_error(transfer)
        return None if e is None else (_quad(parent.error, e), transfer)

    def best_recipe(conc):
        options = [(r, p) for p in pool if p.conc > conc * (1 + _REL) and (r := make(conc, p))]
        return min(options, key=lambda o: o[0][0]) if options else None

    def trial(conc):
        recipe = best_recipe(conc)
        if recipe is None:
            return None
        (err, transfer), parent = recipe
        return _Source("?", conc, err, parent=parent, transfer_ul=transfer)

    def conc_range(keys, points):
        """Log grid of source concentrations that put these targets' well volumes in range."""
        lo = min(active[k] * s["well_ul"] / cap[k] for k in keys)
        hi = max(active[k] * s["well_ul"] / s["well_min_ul"] for k in keys)
        return np.geomspace(lo, hi, points) if hi > lo * (1 + _REL) else np.array([lo])

    def uncovered():
        out = []
        for k in active:
            if k in impossible:
                continue
            b = best_existing(k)
            if b is None or b[0] > limit[k] + _REL:
                out.append(k)
        return out

    count = 0
    for _ in range(20 * len(active) + 20):
        left = uncovered()
        if not left:
            break
        best = None
        for conc in conc_range(left, GRID_POINTS):
            new = trial(conc)
            if new is None:
                continue
            errors = [e for k in left if (e := well_error(new, k)) is not None and e <= limit[k] + _REL]
            if not errors:
                continue
            score = (len(errors), -sum(errors) / len(errors))
            if best is None or score > best[0]:
                best = (score, new)
        if best is not None:
            count += 1
            best[1].name = f"{s['name_prefix']}{count}"
            pool.append(best[1])
            continue
        # No new substock helps. Targets no source can reach (too dilute) get an
        # intermediate dilution first.
        unreachable = [k for k in left if not any(best_recipe(c) for c in conc_range([k], 50))]
        if unreachable and _add_intermediate(unreachable, active, cap, pool, s, best_recipe,
                                             max_transfer_ul, count + 1):
            count += 1
            continue
        # Targets that can be made, but not within max_error: accept their lowest-error route.
        # Targets with no route at all are impossible.
        for k in left:
            options = [b[0]] if (b := best_existing(k)) else []
            options += [e for c in conc_range([k], GRID_POINTS)
                        if (n := trial(c)) is not None and (e := well_error(n, k)) is not None]
            if options:
                limit[k] = max(limit[k], min(options) * (1 + OVER_LIMIT_SLACK))
            else:
                impossible.add(k)
    else:
        raise RuntimeError("Substock planning did not converge")

    # Final assignment: lowest-error source per target.
    wells, infeasible = [], []
    for k, t in active.items():
        b = None if k in impossible else best_existing(k)
        if b is None:
            infeasible.append(dict(sample_id=k, target=t, reason=_reason(t, cap[k], pool, s)))
            continue
        err, src = b
        wells.append(dict(sample_id=k, target=t, source=src.name, volume_ul=t * s["well_ul"] / src.conc,
                          error=err, over_limit=err > s["max_error"] + _REL))
    return _finish(wells, infeasible, pool, s)


def _add_intermediate(unreachable, active, cap, pool, s, best_recipe, max_transfer_ul, number):
    """Add one intermediate dilution toward the most dilute unreachable target. True if added."""
    target_conc = min(active[k] * s["well_ul"] / cap[k] for k in unreachable)
    parents = [p for p in pool if p.conc > target_conc * (1 + _REL)]
    if not parents:
        return False
    parent = min(parents, key=lambda p: p.conc)
    steps = max(2, math.ceil(math.log(parent.conc / target_conc)
                             / math.log(max_transfer_ul / s["substock_min_ul"])))
    conc = parent.conc * (target_conc / parent.conc) ** (1 / steps)
    recipe = best_recipe(conc)
    if recipe is None or any(math.isclose(p.conc, conc, rel_tol=1e-6) for p in pool):
        return False
    (err, transfer), parent = recipe
    pool.append(_Source(f"{s['name_prefix']}{number}", conc, err, parent=parent, transfer_ul=transfer))
    return True


def _reason(target, cap_ul, pool, s):
    top = max(p.conc for p in pool)
    if target * s["well_ul"] / top > cap_ul + _REL:
        return f"above max: needs > {cap_ul:.0f} uL of the strongest source ({top:g})"
    return "no route gives a well volume with a validated error estimate"


def _finish(wells, infeasible, pool, s):
    """Drop unused substocks, count fills, total volume per source."""
    use_ml = {p.name: 0.0 for p in pool}
    for w in wells:
        use_ml[w["source"]] += w["volume_ul"] / 1000
    usable = s["substock_ml"] - s["dead_ml"]
    fills = {}

    def depth(p):
        return 0 if p.parent is None else 1 + depth(p.parent)

    # Children before parents: deeper (more dilute) substocks first.
    for p in sorted([p for p in pool if p.new], key=depth, reverse=True):
        if use_ml[p.name] <= 0:
            continue
        fills[p.name] = math.ceil(use_ml[p.name] / usable - _REL)
        use_ml[p.parent.name] += fills[p.name] * p.transfer_ul / 1000
    substocks = []
    for p in sorted([p for p in pool if p.name in fills], key=depth):
        substocks.append(dict(name=p.name, conc=p.conc, error=p.error, parent=p.parent.name,
                              transfer_ul=p.transfer_ul,
                              water_ul=s["substock_ml"] * 1000 - p.transfer_ul,
                              fills=fills[p.name], used_ml=use_ml[p.name]))
    usage = {}
    for p in pool:
        if p.new and p.name not in fills:
            continue
        need = use_ml[p.name]
        short = (p.available_ml is not None and need > p.available_ml - s["dead_ml"] + _REL)
        usage[p.name] = dict(need_ml=need, available_ml=p.available_ml, short=short)
    kept = [p for p in pool if not p.new or p.name in fills]
    return dict(wells=wells, substocks=substocks, usage=usage, infeasible=infeasible,
                sources={p.name: dict(conc=p.conc, error=p.error) for p in kept})
