import json
import pathlib
import re
from collections import Counter, defaultdict

_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "and",
    "or", "is", "are", "be", "was", "were", "has", "have", "by", "from",
    "this", "that", "which", "via", "using", "based", "towards", "toward",
    "its", "their", "into", "as", "not", "we", "our", "new", "two",
    "study", "analysis", "effect", "effects", "role", "evidence", "review",
    "impact", "approach", "model", "models", "between", "across", "within",
    "among", "how", "does", "can", "use", "used", "also", "more", "than",
    "over", "under", "no", "vs", "do", "paper", "research", "results",
    "data", "method", "methods", "approach", "approaches", "framework",
    "1", "2", "3", "10",
})


class ClusterLabeler:
    def label(self, papers: list[dict]) -> list[dict]:
        by_cluster: dict[int, list[dict]] = defaultdict(list)
        for p in papers:
            cid = p.get("cluster_id")
            if cid is None or cid == -1:
                continue
            by_cluster[cid].append(p)

        summary: list[dict] = []
        for cid in sorted(by_cluster.keys()):
            group = by_cluster[cid]

            local_core = max(
                group,
                key=lambda p: (
                    p.get("local_citation_count", 0),
                    p.get("citation_count") or 0,
                ),
            )
            local_core_id = local_core.get("paper_id")

            spearhead_ids = {
                p["paper_id"]
                for p in group
                if p.get("layer_tag") == "Spearhead" and p.get("paper_id")
            }

            pool: list[str] = []
            for p in group:
                text = (p.get("title") or "") + " " + (p.get("abstract") or "")[:300]
                tokens = [
                    t for t in re.split(r"[^a-z]+", text.lower())
                    if len(t) >= 3 and t not in _STOPWORDS
                ]
                multiplier = (
                    2
                    if (p.get("paper_id") == local_core_id or p.get("paper_id") in spearhead_ids)
                    else 1
                )
                pool.extend(tokens * multiplier)

            top_terms = [w for w, _ in Counter(pool).most_common(3)]
            label = ", ".join(top_terms) if top_terms else f"cluster-{cid}"

            summary.append({
                "cluster_id": cid,
                "paper_count": len(group),
                "local_core": local_core_id,
                "local_core_title": local_core.get("title"),
                "local_core_lci": local_core.get("local_citation_count", 0),
                "label": label,
            })

        return summary

    def save(
        self,
        cluster_summary: list[dict],
        output_dir: str | pathlib.Path = "data/sentinel",
    ) -> pathlib.Path:
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        path = out / "cluster_summary.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cluster_summary, indent=2), encoding="utf-8")
        tmp.replace(path)

        return path
