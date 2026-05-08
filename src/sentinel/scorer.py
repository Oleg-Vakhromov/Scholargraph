import datetime
import json
import pathlib
from collections import Counter


class Scorer:
    def score(
        self,
        papers: list[dict],
        edges: list[dict],
    ) -> list[dict]:
        current_year = datetime.date.today().year

        in_degree: Counter = Counter(e["target"] for e in edges if e.get("target"))

        enriched: list[dict] = []
        for paper in papers:
            p = dict(paper)
            pid = p.get("paper_id", "")

            p["local_citation_count"] = in_degree.get(pid, 0)

            citation_count = p.get("citation_count")
            year = p.get("year")
            if citation_count is not None and year is not None:
                p["trend_score"] = round(citation_count / (current_year - year + 1), 4)
            else:
                p["trend_score"] = None

            enriched.append(p)

        enriched.sort(
            key=lambda x: (
                -(x["local_citation_count"] or 0),
                -(x.get("citation_count") or 0),
            )
        )

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
