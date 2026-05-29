from collections import Counter
from itertools import combinations
from typing import Dict, Iterable, Set, Tuple

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
        clusters = bc.cluster_papers(coup_df, list(papers_df["paper_id"]), resolution=0.5)
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
        resolution: float = 0.5,
    ) -> Dict[str, int]:
        """
        Cluster papers using AgglomerativeClustering on the coupling matrix.

        Papers absent from the coupling matrix (no shared references with any
        corpus paper) receive cluster_id = -1.

        Args:
            coupling_df: Output of build_coupling_matrix().
            paper_ids:   All corpus paper IDs to include in result.
            resolution:  Clustering granularity (0.1–1.0). Higher values produce
                         more, smaller clusters. Converted to a dendrogram distance
                         threshold: distance_threshold = 1.0 - resolution.

        Returns:
            Dict mapping paper_id -> int cluster_id.
            Papers outside the matrix: cluster_id = -1.
        """
        all_ids = list(paper_ids)
        fallback = {pid: 0 for pid in all_ids}

        if coupling_df.empty:
            return fallback

        # Collect papers that appear in the coupling matrix
        matrix_ids = sorted(
            set(coupling_df["paper_a"]) | set(coupling_df["paper_b"])
        )
        n = len(matrix_ids)

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
            n_clusters=None,
            distance_threshold=1.0 - resolution,
            compute_full_tree=True,
            metric="precomputed",
            linkage="average",
        )
        labels = model.fit_predict(dist)

        result: Dict[str, int] = {pid: -1 for pid in all_ids}
        for pid, label in zip(matrix_ids, labels):
            result[pid] = int(label)

        return result

    # ------------------------------------------------------------------
    # Spring-layout graph for visualization
    # ------------------------------------------------------------------

    def build_spring_graph(
        self,
        coupling_df: pd.DataFrame,
        title_lookup: dict,
        decile: float = 0.99,
    ) -> Tuple:
        """
        Build a NetworkX Graph for matplotlib spring-layout visualization.

        Filters edges to the top (1 - decile) fraction by coupling_strength,
        applies Girvan-Newman community detection (first split), and returns
        the graph with a community membership dict.

        Args:
            coupling_df:  Output of build_coupling_matrix().
            title_lookup: Dict mapping paper_id -> title.
            decile:       Quantile threshold for edge filtering (default 0.99).

        Returns:
            Tuple (G, community_dict, filtered_df).
        """
        import networkx as nx

        _empty_G = nx.Graph()
        _empty_dict: dict = {}

        if coupling_df.empty:
            return _empty_G, _empty_dict, coupling_df

        threshold = coupling_df["coupling_strength"].quantile(decile)
        filtered = coupling_df[coupling_df["coupling_strength"] >= threshold]

        if filtered.empty:
            return _empty_G, _empty_dict, filtered

        G = nx.Graph()
        for _, row in filtered.iterrows():
            a, b, w = row["paper_a"], row["paper_b"], row["coupling_strength"]
            G.add_node(a, label=(title_lookup.get(a) or a)[:40])
            G.add_node(b, label=(title_lookup.get(b) or b)[:40])
            G.add_edge(a, b, weight=int(w))

        if G.number_of_edges() == 0:
            return G, {n: 0 for n in G.nodes()}, filtered

        communities_gen = nx.algorithms.community.girvan_newman(G)
        top_level = tuple(sorted(c) for c in next(communities_gen))
        community_dict = {
            node: i for i, comm in enumerate(top_level) for node in comm
        }

        return G, community_dict, filtered
