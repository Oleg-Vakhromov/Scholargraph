import json
import pathlib
from collections import defaultdict
from typing import Optional


class Sentinel:
    async def tier(
        self,
        papers: list[dict],
        ranking: str = "preloaded",
        oa_client=None,
    ) -> list[dict]:
        enriched: list[dict] = []

        for paper in papers:
            p = dict(paper)

            # Auto mode: re-determine is_top_tier via OpenAlex citation percentile
            if ranking == "auto" and oa_client is not None and p.get("doi"):
                try:
                    result = await oa_client._get(
                        f"/works/doi:{p['doi']}",
                        {"select": "cited_by_percentile_year"},
                    )
                    percentile = (result.get("cited_by_percentile_year") or {}).get("max")
                    if percentile is not None and percentile >= 90:
                        p["is_top_tier"] = True
                except Exception:
                    pass  # leave is_top_tier unchanged on any API error

            # Assign tier
            citation_count = p.get("citation_count")
            is_top_tier = p.get("is_top_tier", False)
            has_citations = citation_count is not None and citation_count > 0

            if has_citations and is_top_tier:
                p["tier"] = "high"
            elif not has_citations and not is_top_tier:
                p["tier"] = "low"
            else:
                p["tier"] = "mid"

            enriched.append(p)

        return enriched

    def save_journal_metrics(
        self,
        papers: list[dict],
        output_dir: str | pathlib.Path = "data/sentinel",
    ) -> pathlib.Path:
        groups: dict[str, list[dict]] = defaultdict(list)
        for paper in papers:
            key = paper.get("journal_name") or "_unknown"
            groups[key].append(paper)

        metrics: list[dict] = []
        for journal_name, group in groups.items():
            paper_count = len(group)
            top_tier_count = sum(1 for p in group if p.get("is_top_tier"))
            high_tier_count = sum(1 for p in group if p.get("tier") == "high")
            velocities = [p["citation_velocity"] for p in group if p.get("citation_velocity") is not None]
            avg_velocity = round(sum(velocities) / len(velocities), 4) if velocities else None
            metrics.append({
                "journal_name": journal_name,
                "paper_count": paper_count,
                "top_tier_count": top_tier_count,
                "high_tier_count": high_tier_count,
                "avg_citation_velocity": avg_velocity,
            })

        metrics.sort(key=lambda x: x["paper_count"], reverse=True)

        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "journal_metrics.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        tmp.replace(path)
        return path

    def validate(
        self,
        papers: list[dict],
        edges: list[dict],
        expected_paper_count: int,
    ) -> dict:
        dois = [p["doi"].strip().lower() for p in papers if p.get("doi")]
        duplicates = len(dois) - len(set(dois))
        duplicate_doi_rate = round(duplicates / len(papers), 4) if papers else 0.0

        high_papers = [p for p in papers if p.get("tier") == "high"]
        required = ("doi", "title", "year", "citation_count", "journal_name")
        complete = sum(1 for p in high_papers if all(p.get(f) is not None for f in required))
        high_metadata_completeness = round(complete / len(high_papers), 4) if high_papers else 1.0

        paper_count = len(papers)
        paper_count_ok = abs(paper_count - expected_paper_count) / max(expected_paper_count, 1) <= 0.5

        missing_doi_papers = [p["paper_id"] for p in papers if not p.get("doi") and p.get("paper_id")]

        pass_all = (
            duplicate_doi_rate == 0.0
            and high_metadata_completeness >= 0.95
            and paper_count_ok
        )
        return {
            "duplicate_doi_rate": duplicate_doi_rate,
            "high_metadata_completeness": high_metadata_completeness,
            "paper_count_ok": paper_count_ok,
            "paper_count": paper_count,
            "expected_paper_count": expected_paper_count,
            "missing_doi_papers": missing_doi_papers,
            "pass_all": pass_all,
        }

    def save_report(
        self,
        report: dict,
        output_dir: str | pathlib.Path = "data/sentinel",
    ) -> pathlib.Path:
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "validation_report.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        tmp.replace(path)
        return path
