from collections import Counter
from itertools import combinations
from typing import Dict, Iterable, Set

import pandas as pd


class BibliographicCoupler:
    """
    Stateless bibliographic coupling analysis for research corpus papers.

    Two corpus papers are "bibliographically coupled" when they both cite the
    same paper. Coupling strength = number of shared references. This is the
    inverse direction of co-citation: co-citation looks at who cites a paper;
    bibliographic coupling looks at what papers cite.

    All methods accept DataFrames and return new objects — no instance state,
    no side effects on inputs. Follows the stateless pattern of CoCitationAnalyzer.

    Usage:
        bc = BibliographicCoupler()
        coup_df = bc.build_coupling_matrix(corpus.citations_df, set(papers_df["paper_id"]))
        clusters = bc.cluster_papers(coup_df, list(papers_df["paper_id"]), n_clusters=5)
    """

    # ------------------------------------------------------------------
    # Coupling matrix
    # ------------------------------------------------------------------

    def build_coupling_matrix(
        self,
        citations_df: pd.DataFrame,
        paper_ids: Set[str],
    ) -> pd.DataFrame:
        """
        Build a bibliographic coupling strength table from citation edges.

        For each paper that is cited (target), collect the subset of corpus
        papers that cite it (in-corpus sources). Every pair within that subset
        shares one reference, contributing one unit of coupling strength.

        Args:
            citations_df: Citation edges DataFrame with "source" and "target" columns.
            paper_ids:    Set of corpus paper IDs (from papers_df["paper_id"]).

        Returns:
            DataFrame with columns ["paper_a", "paper_b", "coupling_strength"],
            sorted by coupling_strength descending. Pairs are canonical
            (paper_a < paper_b lexicographically) to avoid duplicates.
            Returns empty DataFrame with those columns if no in-corpus pairs exist
            or citations_df is empty.
        """
        _empty = pd.DataFrame(columns=["paper_a", "paper_b", "coupling_strength"])

        if citations_df.empty or not paper_ids:
            return _empty

        # Group citing papers (sources) by the paper they cite (target)
        # Keep only corpus members as sources
        pair_counts: Counter = Counter()
        for target, group in citations_df.groupby("target")["source"]:
            in_corpus = [s for s in group if s in paper_ids]
            if len(in_corpus) < 2:
                continue
            for a, b in combinations(sorted(in_corpus), 2):
                pair_counts[(a, b)] += 1

        if not pair_counts:
            return _empty

        rows = [
            {"paper_a": a, "paper_b": b, "coupling_strength": count}
            for (a, b), count in pair_counts.items()
        ]
        return (
            pd.DataFrame(rows, columns=["paper_a", "paper_b", "coupling_strength"])
            .sort_values("coupling_strength", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster_papers(
        self,
        coupling_df: pd.DataFrame,
        paper_ids: Iterable[str],
        n_clusters: int = 5,
    ) -> Dict[str, int]:
        """
        Cluster papers using AgglomerativeClustering on the coupling matrix.

        Papers absent from the coupling matrix (no shared references with any
        corpus paper) receive cluster_id = -1.

        Args:
            coupling_df: Output of build_coupling_matrix().
            paper_ids:   All corpus paper IDs to include in result.
            n_clusters:  Number of clusters (must be >= 2).

        Returns:
            Dict mapping paper_id -> int cluster_id.
            Papers in the coupling matrix: cluster_id in [0, n_clusters).
            Papers outside the matrix: cluster_id = -1.
        """
        all_ids = list(paper_ids)
        fallback = {pid: 0 for pid in all_ids}

        if coupling_df.empty or n_clusters < 2:
            return fallback

        # Collect papers that appear in the coupling matrix
        matrix_ids = sorted(
            set(coupling_df["paper_a"]) | set(coupling_df["paper_b"])
        )
        n = len(matrix_ids)

        if n < n_clusters:
            n_clusters = max(2, n)

        if n < 2:
            return fallback

        idx = {pid: i for i, pid in enumerate(matrix_ids)}

        import numpy as np
        dist = np.ones((n, n), dtype=float)
        np.fill_diagonal(dist, 0.0)

        for _, row in coupling_df.iterrows():
            a, b, strength = row["paper_a"], row["paper_b"], row["coupling_strength"]
            if a in idx and b in idx:
                d = 1.0 / (1.0 + strength)
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
