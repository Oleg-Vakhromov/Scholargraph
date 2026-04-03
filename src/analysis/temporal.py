from typing import Dict, Optional

import pandas as pd


class TemporalAnalyzer:
    """
    Stateless temporal analysis of research cluster evolution.

    Operates on papers_df that has already been enriched with cluster_id
    by ClusterEngine.add_cluster_assignments(). No Louvain re-runs per year —
    uses the single full-corpus partition and slices by publication year.

    Usage:
        ce = ClusterEngine()
        partition = ce.detect_communities(G)
        papers_df = ce.add_cluster_assignments(corpus.papers_df, partition)
        labels = ce.label_clusters(papers_df, partition)

        ta = TemporalAnalyzer()
        evolution = ta.cluster_evolution(papers_df, labels)
        pivot = ta.evolution_pivot(evolution)
    """

    # ------------------------------------------------------------------
    # Per-year cluster membership counts
    # ------------------------------------------------------------------

    def cluster_evolution(
        self,
        papers_df: pd.DataFrame,
        cluster_labels: Dict[int, str],
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Count papers per cluster per publication year.

        Args:
            papers_df:      papers_df with cluster_id (Int64) and year (Int64)
                            columns. Rows with NA in either column are excluded.
            cluster_labels: Dict {community_id: label_string} from
                            ClusterEngine.label_clusters().
            min_year:       Inclusive lower bound on year (optional).
            max_year:       Inclusive upper bound on year (optional).

        Returns:
            DataFrame with columns: year, cluster_id, label, paper_count.
            Sorted by year ascending, then paper_count descending.
            Returns empty DataFrame with those columns if no rows survive
            filtering.
        """
        _cols = ["year", "cluster_id", "label", "paper_count"]

        df = papers_df.dropna(subset=["cluster_id", "year"])

        if min_year is not None:
            df = df[df["year"] >= min_year]
        if max_year is not None:
            df = df[df["year"] <= max_year]

        if df.empty:
            return pd.DataFrame(columns=_cols)

        counts = (
            df.groupby(["year", "cluster_id"], as_index=False)
            .size()
            .rename(columns={"size": "paper_count"})
        )

        counts["label"] = counts["cluster_id"].map(
            lambda cid: cluster_labels.get(int(cid), f"cluster-{int(cid)}")
        )

        counts = counts[_cols]

        return (
            counts
            .sort_values(["year", "paper_count"], ascending=[True, False])
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Pivot table for charting
    # ------------------------------------------------------------------

    def evolution_pivot(self, evolution_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot cluster evolution data into a year × cluster_id matrix.

        Produces a DataFrame suitable for a multi-line time-series chart in
        Phase 5 (one line per cluster, x-axis = year, y-axis = paper_count).

        Args:
            evolution_df: DataFrame from cluster_evolution() with columns
                          year, cluster_id, paper_count.

        Returns:
            DataFrame with year as index, cluster_ids as columns, paper counts
            as values (int, 0-filled). Index sorted ascending by year.
            Returns empty DataFrame if evolution_df is empty.
        """
        if evolution_df.empty:
            return pd.DataFrame()

        pivot = evolution_df.pivot_table(
            index="year",
            columns="cluster_id",
            values="paper_count",
            aggfunc="sum",
            fill_value=0,
        )

        pivot.columns.name = None
        pivot = pivot.astype(int)
        pivot = pivot.sort_index()

        return pivot
