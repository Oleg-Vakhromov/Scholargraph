from collections import Counter
from itertools import combinations
from typing import Dict, Iterable, Set

import pandas as pd


class CoCitationAnalyzer:
    """
    Stateless co-citation analysis for research corpus papers.

    Two corpus papers are "co-cited" when the same external (or internal)
    paper cites both. Co-citation frequency is a proxy for intellectual
    proximity — frequently co-cited papers address related topics even when
    they do not cite each other directly.

    All methods accept DataFrames and return new objects — no instance state,
    no side effects on inputs. Follows the stateless pattern of ClusterEngine
    and InfluenceAnalyzer.

    Usage:
        ca = CoCitationAnalyzer()
        coc_df = ca.build_cocitation_matrix(corpus.citations_df, set(papers_df["paper_id"]))
        clusters = ca.cluster_papers(coc_df, list(papers_df["paper_id"]), n_clusters=5)
    """

    # ------------------------------------------------------------------
    # Co-citation matrix
    # ------------------------------------------------------------------

    def build_cocitation_matrix(
        self,
        citations_df: pd.DataFrame,
        paper_ids: Set[str],
    ) -> pd.DataFrame:
        """
        Build a co-citation frequency table from citation edges.

        For each citing paper (source), collect the subset of its references
        that are corpus members (present in paper_ids). Every pair within that
        subset contributes one co-citation count.

        Args:
            citations_df: Citation edges DataFrame with "source" and "target" columns.
            paper_ids:    Set of corpus paper IDs (from papers_df["paper_id"]).

        Returns:
            DataFrame with columns ["paper_a", "paper_b", "cocitation_count"],
            sorted by cocitation_count descending. Pairs are canonical
            (paper_a < paper_b lexicographically) to avoid duplicates.
            Returns empty DataFrame with those columns if no in-corpus pairs exist
            or citations_df is empty.
        """
        _empty = pd.DataFrame(columns=["paper_a", "paper_b", "cocitation_count"])

        if citations_df.empty or not paper_ids:
            return _empty

        # Group targets by citing paper; keep only in-corpus targets
        pair_counts: Counter = Counter()
        for source, group in citations_df.groupby("source")["target"]:
            in_corpus = [t for t in group if t in paper_ids]
            if len(in_corpus) < 2:
                continue
            for a, b in combinations(sorted(in_corpus), 2):
                pair_counts[(a, b)] += 1

        if not pair_counts:
            return _empty

        rows = [
            {"paper_a": a, "paper_b": b, "cocitation_count": count}
            for (a, b), count in pair_counts.items()
        ]
        return (
            pd.DataFrame(rows, columns=["paper_a", "paper_b", "cocitation_count"])
            .sort_values("cocitation_count", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster_papers(
        self,
        cocitation_df: pd.DataFrame,
        paper_ids: Iterable[str],
        n_clusters: int = 5,
    ) -> Dict[str, int]:
        """
        Cluster papers using AgglomerativeClustering on the co-citation matrix.

        Papers absent from the co-citation matrix (zero co-citations with any
        corpus paper) receive cluster_id = -1.

        Args:
            cocitation_df: Output of build_cocitation_matrix().
            paper_ids:     All corpus paper IDs to include in result.
            n_clusters:    Number of clusters (must be >= 2).

        Returns:
            Dict mapping paper_id -> int cluster_id.
            Papers in the co-citation matrix: cluster_id in [0, n_clusters).
            Papers outside the matrix: cluster_id = -1.
        """
        all_ids = list(paper_ids)
        fallback = {pid: 0 for pid in all_ids}

        if cocitation_df.empty or n_clusters < 2:
            return fallback

        # Collect papers that appear in the co-citation matrix
        matrix_ids = sorted(
            set(cocitation_df["paper_a"]) | set(cocitation_df["paper_b"])
        )
        n = len(matrix_ids)

        if n < n_clusters:
            # Fewer matrix papers than requested clusters — clamp
            n_clusters = max(2, n)

        if n < 2:
            return fallback

        # Build index
        idx = {pid: i for i, pid in enumerate(matrix_ids)}

        # Initialise distance matrix (unknown pairs → max distance 1.0)
        import numpy as np
        dist = np.ones((n, n), dtype=float)
        np.fill_diagonal(dist, 0.0)

        for _, row in cocitation_df.iterrows():
            a, b, count = row["paper_a"], row["paper_b"], row["cocitation_count"]
            if a in idx and b in idx:
                d = 1.0 / (1.0 + count)
                i, j = idx[a], idx[b]
                dist[i, j] = d
                dist[j, i] = d

        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
        labels = model.fit_predict(dist)

        result: Dict[str, int] = {pid: -1 for pid in all_ids}
        for pid, label in zip(matrix_ids, labels):
            result[pid] = int(label)

        return result
