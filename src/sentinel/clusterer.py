import json
import pathlib

import community
import networkx as nx


class Clusterer:
    def cluster(
        self,
        papers: list[dict],
        edges: list[dict],
    ) -> list[dict]:
        valid_ids = {p["paper_id"] for p in papers if p.get("paper_id")}

        G = nx.Graph()
        G.add_nodes_from(valid_ids)

        for edge in edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if src in valid_ids and tgt in valid_ids:
                G.add_edge(src, tgt)

        partition = community.best_partition(G, random_state=42)

        enriched: list[dict] = []
        for paper in papers:
            p = dict(paper)
            pid = p.get("paper_id", "")
            p["cluster_id"] = partition.get(pid, -1)
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
