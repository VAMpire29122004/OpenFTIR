"""
Shared clustering: groups peaks whose wavenumbers land close together into
one unit, rather than letting each contributing reference-library rule
render as its own separate label. Used by both interactive.py (Plotly
preview) and plotting.py (matplotlib publication export) — extracted here
so the two can never silently diverge on how overlapping candidates are
grouped. A researcher's edits in the interactive view must render
identically in the exported PNG; two separate clustering implementations
would risk exactly that kind of drift.

ANCHORED, not chained: every cluster's span is measured from its first
(lowest-wavenumber) member, not from whichever point was added last.
Chaining lets cluster width grow unboundedly as long as consecutive points
are each under min_dist apart — verified on real data, this merged an
11-class, 55 cm-1-wide span into a single marker. Anchoring caps every
cluster's width at min_dist by construction.
"""

TIER_PRIORITY = ["very_strong", "strong", "medium", "weak", "manual"]


def cluster_points(points: list, min_dist: float = 30.0) -> list:
    """
    points: list of dicts, each with at least a 'wn' key.
    Returns a list of clusters (each a list of the same dicts).
    """
    if not points:
        return []
    ordered = sorted(points, key=lambda p: p["wn"])
    clusters = [[ordered[0]]]
    for p in ordered[1:]:
        anchor = clusters[-1][0]["wn"]
        if p["wn"] - anchor < min_dist:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return clusters


def cluster_label(cluster: list, max_names: int = 4) -> str:
    names = sorted({p["cls"] for p in cluster})
    if len(names) > max_names:
        return ", ".join(names[:max_names]) + f" (+{len(names) - max_names} more)"
    return " / ".join(names)


def cluster_tier(cluster: list) -> str:
    """Highest-confidence tier present in the cluster, for color/legend
    purposes when members span multiple tiers."""
    tiers_present = {p["tier"] for p in cluster}
    for tier in TIER_PRIORITY:
        if tier in tiers_present:
            return tier
    return "weak"
