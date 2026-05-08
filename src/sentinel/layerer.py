import datetime
import json
import pathlib
from collections import defaultdict


def _percentile(data: list[float], p: float) -> float:
    """Linear interpolation percentile (numpy Type 7), p in [0, 1], clamped to data range."""
    sorted_d = sorted(data)
    n = len(sorted_d)
    k = p * (n - 1)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    return sorted_d[lo] + (k - lo) * (sorted_d[hi] - sorted_d[lo])


class Layerer:
    def tag(self, papers: list[dict]) -> list[dict]:
        current_year = datetime.date.today().year

        # Global Foundational threshold: 75th percentile of citation_count
        global_citations = [
            p["citation_count"] for p in papers
            if p.get("citation_count") is not None
        ]
        if len(global_citations) >= 2:
            foundational_threshold = _percentile(global_citations, 0.75)
        else:
            foundational_threshold = None

        # Group papers by cluster_id for per-cluster thresholds
        by_cluster: dict[int, list[dict]] = defaultdict(list)
        for p in papers:
            by_cluster[p.get("cluster_id", -1)].append(p)

        # Per-cluster Spearhead threshold: 85th percentile of trend_score among papers < 5 yrs old
        spearhead_thresholds: dict[int, float | None] = {}
        for cid, group in by_cluster.items():
            eligible = [
                p["trend_score"] for p in group
                if p.get("year") is not None
                and (current_year - p["year"]) < 5
                and p.get("trend_score") is not None
            ]
            if len(eligible) >= 2:
                spearhead_thresholds[cid] = _percentile(eligible, 0.85)
            else:
                spearhead_thresholds[cid] = None

        # Per-cluster Theory Extender threshold: 75th percentile of local_citation_count > 0
        extender_thresholds: dict[int, float | None] = {}
        for cid, group in by_cluster.items():
            eligible = [
                p.get("local_citation_count", 0) for p in group
                if p.get("local_citation_count", 0) > 0
            ]
            if len(eligible) >= 2:
                extender_thresholds[cid] = _percentile(eligible, 0.75)
            else:
                extender_thresholds[cid] = None

        enriched: list[dict] = []
        for paper in papers:
            p = dict(paper)
            year = p.get("year")
            citation_count = p.get("citation_count")
            lci = p.get("local_citation_count", 0)
            trend = p.get("trend_score")
            cid = p.get("cluster_id", -1)
            age = (current_year - year) if year is not None else None

            tag = "Standard"

            # Spearhead: highest precedence
            if age is not None and age < 5 and trend is not None:
                thresh = spearhead_thresholds.get(cid)
                if thresh is not None and trend >= thresh:
                    tag = "Spearhead"

            # Foundational
            if tag == "Standard" and age is not None and age >= 5 and citation_count is not None:
                if foundational_threshold is not None and citation_count >= foundational_threshold:
                    tag = "Foundational"

            # Theory Extender
            if tag == "Standard" and lci > 0:
                thresh = extender_thresholds.get(cid)
                if thresh is not None and lci >= thresh:
                    tag = "Theory Extender"

            p["layer_tag"] = tag
            enriched.append(p)

        return enriched

    def save(
        self,
        papers: list[dict],
        output_dir: str | pathlib.Path = "data/sentinel",
    ) -> pathlib.Path:
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        path = out / "papers.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(papers, indent=2), encoding="utf-8")
        tmp.replace(path)

        return path

    def save_cluster_summary(
        self,
        papers: list[dict],
        output_dir: str | pathlib.Path = "data/sentinel",
    ) -> pathlib.Path:
        by_cluster: dict[int, list[dict]] = defaultdict(list)
        for p in papers:
            by_cluster[p.get("cluster_id", -1)].append(p)

        summary: list[dict] = []
        for cid in sorted(by_cluster.keys()):
            group = by_cluster[cid]
            core = max(
                group,
                key=lambda p: (
                    p.get("local_citation_count", 0),
                    p.get("citation_count") or 0,
                ),
            )
            summary.append({
                "cluster_id": cid,
                "paper_count": len(group),
                "local_core": core.get("paper_id"),
                "local_core_title": core.get("title"),
                "local_core_lci": core.get("local_citation_count", 0),
            })

        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        path = out / "cluster_summary.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tmp.replace(path)

        return path
