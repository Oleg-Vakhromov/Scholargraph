import json
import pathlib

import pandas as pd


class SentinelDataLoader:
    def load(
        self,
        sentinel_dir: str | pathlib.Path = "data/sentinel",
    ) -> dict:
        base = pathlib.Path(sentinel_dir)
        papers_path = base / "papers.json"
        edges_path = base / "edges.json"

        if not papers_path.exists():
            raise FileNotFoundError(f"Sentinel papers not found: {papers_path}")
        if not edges_path.exists():
            raise FileNotFoundError(f"Sentinel edges not found: {edges_path}")

        papers_df = pd.DataFrame(json.loads(papers_path.read_text(encoding="utf-8")))
        if "cluster_id" in papers_df.columns:
            papers_df["cluster_id"] = pd.to_numeric(papers_df["cluster_id"], errors="coerce").astype("Int64")

        citations_df = pd.DataFrame(json.loads(edges_path.read_text(encoding="utf-8")))

        partition: dict[str, int] = {}
        if "paper_id" in papers_df.columns and "cluster_id" in papers_df.columns:
            for row in papers_df[["paper_id", "cluster_id"]].itertuples(index=False):
                if pd.notna(row.cluster_id):
                    partition[row.paper_id] = int(row.cluster_id)

        return {"papers_df": papers_df, "citations_df": citations_df, "partition": partition}
