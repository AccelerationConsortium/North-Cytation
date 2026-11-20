"""Turn target well contents into dispense volumes and substock recipes.

Pure functions, no hardware, so a plan can be checked before anything moves.

Input: one row per well with the target concentrations over the total well volume
(surfactant_mM, dye_uM) and a matrix:
  "water"   - water + surfactant + a FIXED volume of dye solution (dye_ul). The dye
              source is chosen so that volume gives the target, so every well has
              the same dye-solvent fraction. Dye 0 -> the same volume of pure solvent.
  "solvent" - solvent + dye only (e.g. the dye stock check); the dye volume varies.

Variable-volume components (surfactant; dye in solvent wells) use the source that
gives the LARGEST well volume within [minimum, well_max_ul], and the rest is
made up with water or solvent. The minimum is well_min_ul, or a component's
preferred_min_ul if set (higher -> larger, more accurate volumes but more
substocks). New substocks are added only when no existing source fits, as few
as possible.

Substocks are made from the MOST concentrated source that still needs at least
substock_min_transfer_ul (fewest dilution steps, no tiny transfers of stock). A
substock no source can make gets an intermediate dilution first.

Refills: a substock is made to its fill volume (substock_ml) and topped up, in
the recipe's ratio, whenever drawing from it would take it below its refill level
(substock_refill_ml). So one substock can serve any number of wells; demand_ml
counts every draw, and parents are charged for the first make plus every top-up
(top_up() gives the amounts). Each fill gives (fill - refill level) mL to use.

Existing substocks (from an earlier batch on the same plate) are passed in and
reused, so a plate can be planned and run in parts.
"""
import math

import pandas as pd

COMPONENTS = ("surfactant", "dye")
_REL_TOL = 1e-9


def _new_source(name, component, conc, kind, volume_ml=None, **recipe):
    return dict(name=name, component=component, conc=float(conc), kind=kind,
                volume_ml=volume_ml, parent=None, parent_ml=0.0, diluent_ml=0.0,
                new=False, **recipe)


def stock_sources(rules):
    """Stocks, the pure-solvent vial (dye blank) and water, from rules."""
    stocks = rules["stocks"]
    return [
        _new_source(rules["water_vial"], "water", 0.0, "water"),
        _new_source(stocks["surfactant"]["vial"], "surfactant", stocks["surfactant"]["conc"], "stock"),
        _new_source(stocks["dye"]["vial"], "dye", stocks["dye"]["conc"], "stock"),
        _new_source(rules["solvent_vial"], "dye", 0.0, "blank"),
    ]


def _same(a, b):
    return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=0.0)


def _diluent_vial(rules, component):
    """Water for surfactant substocks; for dye, solvent_diluent_vial (bulk) if set, else solvent_vial."""
    if component == "surfactant":
        return rules["water_vial"]
    return rules.get("solvent_diluent_vial") or rules["solvent_vial"]


def _substock_ml(rules, component):
    return rules["substock_ml"][component]


def _add_substock(conc, component, sources, rules):
    """Add a substock at conc, made from the most concentrated source with a
    transfer >= substock_min_transfer_ul. Adds intermediates if needed."""
    volume_ml = _substock_ml(rules, component)
    min_ml = rules["substock_min_transfer_ul"] / 1000
    parents = sorted((s for s in sources if s["component"] == component and s["conc"] > conc),
                     key=lambda s: s["conc"], reverse=True)
    if not parents:
        raise ValueError(f"{component} {conc:.4g} is above every available source")
    for parent in parents:
        transfer = volume_ml * conc / parent["conc"]
        if transfer >= min_ml - 1e-12:
            break
    else:
        # Even the most dilute parent needs too little: make an intermediate
        # halfway (geometrically) between it and the target, then dilute from that.
        # If the intermediate is itself too dilute, this repeats one level down.
        weakest = parents[-1]
        _add_substock(math.sqrt(weakest["conc"] * conc), component, sources, rules)
        return _add_substock(conc, component, sources, rules)
    # Next free number: substocks kept from earlier runs may leave gaps (e.g. _1 and _3).
    taken = {s["name"] for s in sources}
    number = 1
    while f"{rules['substock_prefix'][component]}_{number}" in taken:
        number += 1
    source = _new_source(f"{rules['substock_prefix'][component]}_{number}", component, conc,
                         "substock", volume_ml=volume_ml)
    source.update(parent=parent["name"], parent_ml=transfer, diluent_ml=volume_ml - transfer,
                  diluent=_diluent_vial(rules, component), new=True)
    sources.append(source)
    return source


def top_up(fill_ml, ratio, current_ml, min_parent_ml, capacity_ml):
    """(parent_ml, diluent_ml) to bring a substock from current_ml back to fill_ml
    with parent:total = ratio (the recipe's). A parent part below min_parent_ml is
    raised to it, with the diluent scaled to keep the ratio (the vial then ends
    above fill_ml); error if that would overfill the vial."""
    add = fill_ml - current_ml
    if add <= 0:
        return 0.0, 0.0
    parent = add * ratio
    if parent < min_parent_ml:
        add = min_parent_ml / ratio
        parent = min_parent_ml
    if current_ml + add > capacity_ml + 1e-9:
        raise ValueError(f"Top-up from {current_ml:.2f} mL needs {add:.2f} mL to keep a "
                         f"{min_parent_ml * 1000:.0f} uL parent transfer; that overfills a "
                         f"{capacity_ml:g} mL vial. Lower the refill level or the minimum transfer")
    return parent, add - parent


def _refill_draws(s, draw_ml, rules):
    """(top-ups, parent_ml, diluent_ml, left_ml) a substock needs to supply draw_ml,
    counting the first make if it is new, or what an earlier batch left in it
    (left_ml: usable mL above the refill level). Assumes each top-up starts at the
    refill level."""
    fill, refill = s["volume_ml"], rules["substock_refill_ml"][s["component"]]
    usable = fill - refill
    if usable <= 0:
        raise ValueError(f"{s['name']}: fill volume {fill:g} mL must be above its refill level {refill:g} mL")
    first = usable if s["new"] else s.get("left_ml", 0.0)
    topups = max(0, math.ceil((draw_ml - first) / usable - 1e-9))
    left = first + topups * usable - draw_ml
    parent = s["parent_ml"] if s["new"] else 0.0
    diluent = s["diluent_ml"] if s["new"] else 0.0
    if topups:
        p, d = top_up(fill, s["parent_ml"] / fill, refill, rules["substock_min_transfer_ul"] / 1000,
                      rules["vial_capacity_ml"])
        parent += topups * p
        diluent += topups * d
    return topups, parent, diluent, left


def _fixed_source(conc, sources, rules):
    """Dye source at exactly conc (for the fixed dye volume)."""
    for s in sources:
        if s["component"] == "dye" and _same(s["conc"], conc):
            return s
    return _add_substock(conc, "dye", sources, rules)


def _volume_ul(target, source_conc, total_ul):
    return target * total_ul / source_conc


def _min_ul(rules, component):
    """Smallest well volume for a component: its preferred minimum if set, else well_min_ul."""
    preferred = rules.get("preferred_min_ul", {}).get(component)
    return rules["well_min_ul"] if preferred is None else max(preferred, rules["well_min_ul"])


def _variable_source(target, component, sources, max_ul, rules):
    """Source giving the largest well volume in [minimum, max_ul], or None."""
    min_ul = _min_ul(rules, component)
    best = None
    for s in sources:
        if s["component"] != component or s["conc"] <= 0:
            continue
        ul = _volume_ul(target, s["conc"], rules["total_ul"])
        if min_ul - 1e-9 <= ul <= max_ul + 1e-9 and (best is None or ul > best[1]):
            best = (s, ul)
    return best


def _cover_variable(targets, component, sources, max_ul, rules):
    """Add the fewest substocks so every target has a source (greedy, high to low)."""
    for target in sorted({t for t in targets if t > 0}, reverse=True):
        if _variable_source(target, component, sources, max_ul, rules) is None:
            # Put this target at max_ul: covers it and as many lower targets as possible.
            conc = target * rules["total_ul"] / max_ul
            stock = next(s for s in sources if s["component"] == component and s["kind"] == "stock")
            if conc > stock["conc"] * (1 + _REL_TOL):
                raise ValueError(
                    f"{component} {target:.4g} needs {_volume_ul(target, stock['conc'], rules['total_ul']):.1f} uL "
                    f"of stock, above the {max_ul:.1f} uL limit; use a stronger stock or a higher well_max_ul")
            _add_substock(conc, component, sources, rules)


def plan_wells(targets, rules, existing=None, later_targets=None):
    """Return (wells, sources, recipes).

    targets: DataFrame with matrix, surfactant_mM, dye_uM (other columns are kept).
    rules: see stock_sources() and the keys used below; volumes in uL unless named _ml.
    existing: sources from earlier batches (list of dicts from a previous call);
              None means start from the stocks only.
    later_targets: targets of later batches, if known. Their fixed-volume dye
              substocks are made now, so this batch can use them too instead of
              making dilutions of its own.
    wells: targets plus source and volume columns (uL) for every component.
    sources: every source (stocks, blank, old and new substocks) with demand_ml
             (everything drawn from it) and, for substocks, topups.
    recipes: the new substocks only, in the order to make them.
    Demand counts wells, substock making and top-ups; water for substocks may
    come from the reservoir instead of the water vial (the workflow decides).
    """
    total, dye_ul = rules["total_ul"], rules["dye_ul"]
    sources = [dict(s, new=False) for s in existing] if existing else stock_sources(rules)
    bad = set(targets["matrix"]) - {"water", "solvent"}
    if bad:
        raise ValueError(f"Unknown matrix {sorted(bad)}; use water or solvent")
    if (targets[["surfactant_mM", "dye_uM"]] < 0).any().any():
        raise ValueError("Concentrations must be >= 0")
    water = targets["matrix"] == "water"
    if (targets.loc[~water, "surfactant_mM"] > 0).any():
        raise ValueError("Solvent wells cannot contain surfactant")

    # 1. Exact dye sources for the fixed dye volume (water wells, this batch and later ones).
    fixed = targets if later_targets is None or later_targets.empty else pd.concat([targets, later_targets])
    fixed_conc = fixed.loc[fixed["matrix"] == "water", "dye_uM"] * total / dye_ul
    for conc in sorted(set(fixed_conc[fixed_conc > 0]), reverse=True):
        _fixed_source(conc, sources, rules)
    # 2. Variable dye (solvent wells) and surfactant (water wells).
    well_max = rules["well_max_ul"]
    _cover_variable(targets.loc[~water, "dye_uM"], "dye", sources, min(well_max, total), rules)
    _cover_variable(targets.loc[water, "surfactant_mM"], "surfactant", sources,
                    min(well_max, total - dye_ul), rules)

    rows = []
    blank = next(s["name"] for s in sources if s["kind"] == "blank")
    for _, t in targets.iterrows():
        row = t.to_dict()
        if t["matrix"] == "water":
            if t["dye_uM"] > 0:
                dye_source = _fixed_source(t["dye_uM"] * total / dye_ul, sources, rules)["name"]
            else:
                dye_source = blank
            row.update(dye_source=dye_source, dye_ul=dye_ul)
            surf_ul, surf_source = 0.0, None
            if t["surfactant_mM"] > 0:
                s, surf_ul = _variable_source(t["surfactant_mM"], "surfactant", sources,
                                              min(well_max, total - dye_ul), rules)
                surf_source = s["name"]
            row.update(surfactant_source=surf_source, surfactant_ul=surf_ul,
                       water_ul=total - dye_ul - surf_ul, solvent_ul=0.0)
        else:
            dye_ul_well, dye_source = 0.0, None
            if t["dye_uM"] > 0:
                s, dye_ul_well = _variable_source(t["dye_uM"], "dye", sources, min(well_max, total), rules)
                dye_source = s["name"]
            row.update(dye_source=dye_source, dye_ul=dye_ul_well, surfactant_source=None,
                       surfactant_ul=0.0, water_ul=0.0, solvent_ul=total - dye_ul_well)
        rows.append(row)
    wells = pd.DataFrame(rows)

    # Demand per source: well volumes, then substock makes and top-ups. Children
    # come after their parents in sources, so go backwards to charge parents for
    # their children's top-ups.
    demand = {s["name"]: 0.0 for s in sources}
    demand.update({_diluent_vial(rules, c): 0.0 for c in COMPONENTS if _diluent_vial(rules, c) not in demand})
    for column, volume in (("dye_source", "dye_ul"), ("surfactant_source", "surfactant_ul")):
        for name, ul in wells.groupby(column)[volume].sum().items():
            demand[name] += ul / 1000
    demand[blank] += wells["solvent_ul"].sum() / 1000
    water_name = next(s["name"] for s in sources if s["kind"] == "water")
    demand[water_name] += wells["water_ul"].sum() / 1000
    for s in reversed(sources):
        if s["kind"] == "substock":
            s["topups"], parent_ml, diluent_ml, s["left_ml"] = _refill_draws(s, demand[s["name"]], rules)
            demand[s["parent"]] += parent_ml
            demand[s["diluent"]] += diluent_ml
    for s in sources:
        s["demand_ml"] = demand[s["name"]]
    bulk = rules.get("solvent_diluent_vial")
    if bulk and bulk not in {s["name"] for s in sources}:
        sources.append(dict(_new_source(bulk, "dye", 0.0, "bulk"), demand_ml=demand[bulk]))
    recipes = pd.DataFrame([s for s in sources if s["new"]],
                           columns=["name", "component", "conc", "parent", "parent_ml",
                                    "diluent", "diluent_ml", "volume_ml"])
    return wells, sources, recipes


def after_batch(sources):
    """Sources for the next batch: the same vials, already made (refilled as needed)."""
    return [dict(s, new=False, demand_ml=0.0, topups=0) for s in sources]
