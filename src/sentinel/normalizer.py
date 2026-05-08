import datetime
import json
import pathlib
import re
from collections import Counter
from typing import Optional

from src.sentinel.constants import ABS_4_JOURNALS, BEHAV_ECON_JOURNALS, FT50_JOURNALS

_TOP_TIER = FT50_JOURNALS | ABS_4_JOURNALS | BEHAV_ECON_JOURNALS
_DOI_URL_PREFIX = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.IGNORECASE)


def _normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    return _DOI_URL_PREFIX.sub("", doi).strip().lower() or None


class Normalizer:
    def normalize(
        self,
        papers: list[dict],
        edges: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        id_map: dict[str, str] = {}

        # DOI dedup pass (primary)
        doi_seen: dict[str, str] = {}  # normalized_doi → canonical paper_id
        for paper in papers:
            pid = paper.get("paper_id")
            if not pid:
                continue
            norm_doi = _normalize_doi(paper.get("doi"))
            if not norm_doi:
                continue
            if norm_doi in doi_seen:
                id_map[pid] = doi_seen[norm_doi]
            else:
                doi_seen[norm_doi] = pid

        # Title+Year dedup pass (secondary — only papers not already merged)
        ty_seen: dict[tuple, str] = {}  # (title_norm, year) → canonical paper_id
        for paper in papers:
            pid = paper.get("paper_id")
            if not pid or pid in id_map:
                continue
            if _normalize_doi(paper.get("doi")):
                continue  # has DOI — handled in DOI pass
            title = paper.get("title") or ""
            year = paper.get("year")
            if not title:
                continue
            key = (title.lower().strip(), year)
            if key in ty_seen:
                id_map[pid] = ty_seen[key]
            else:
                ty_seen[key] = pid

        # Build survivor list
        survivors = [p for p in papers if p.get("paper_id") not in id_map]

        # Remap and clean edges
        remapped: list[dict] = []
        seen_edges: set[tuple] = set()
        for edge in edges:
            src = id_map.get(edge.get("source", ""), edge.get("source", ""))
            tgt = id_map.get(edge.get("target", ""), edge.get("target", ""))
            if src == tgt or not src or not tgt:
                continue
            key = (src, tgt)
            if key not in seen_edges:
                seen_edges.add(key)
                remapped.append({"source": src, "target": tgt})

        # Compute local_citation_count: in-degree count per paper_id
        in_degree: Counter = Counter(e["target"] for e in remapped)

        # Metadata enrichment
        current_year = datetime.date.today().year
        enriched: list[dict] = []
        for paper in survivors:
            p = dict(paper)
            journal_name = p.get("venue") or None
            is_top_tier = bool(journal_name and journal_name.lower() in _TOP_TIER)

            citation_count = p.get("citation_count")
            year = p.get("year")
            if citation_count is not None and year is not None:
                trend_score = round(citation_count / (current_year - year + 1), 4)
            else:
                trend_score = None

            pid = p.get("paper_id", "")
            p["journal_name"] = journal_name
            p["is_top_tier"] = is_top_tier
            p["trend_score"] = trend_score
            p["local_citation_count"] = in_degree.get(pid, 0)
            enriched.append(p)

        return enriched, remapped

    def save(
        self,
        papers: list[dict],
        edges: list[dict],
        output_dir: str | pathlib.Path = "data/sentinel",
    ) -> tuple[pathlib.Path, pathlib.Path]:
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        papers_path = out / "papers.json"
        edges_path = out / "edges.json"

        for path, data in [(papers_path, papers), (edges_path, edges)]:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(path)

        return papers_path, edges_path
