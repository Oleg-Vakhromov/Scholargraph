import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import community as community_louvain
import networkx as nx
import pandas as pd


_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "and",
    "or", "is", "are", "be", "was", "were", "has", "have", "by", "from",
    "this", "that", "which", "via", "using", "based", "towards", "toward",
    "its", "their", "into", "as", "not", "we", "our", "new", "two",
})


class ClusterEngine:
    """
    Stateless graph clustering operations for research community detection.

    All methods accept DataFrames / dicts and return new objects — no instance
    state, no side effects on inputs. Follows the same design pattern as
    GraphEngine (Phase 3 Plan 01).

    Usage:
        engine = GraphEngine()
        G = engine.build_graph(corpus.papers_df, corpus.citations_df)
        ce = ClusterEngine()
        partition = ce.detect_communities(G)
        papers_df = ce.add_cluster_assignments(corpus.papers_df, partition)
        labels = ce.label_clusters(papers_df, partition, top_n=5)
        summary = ce.cluster_summary_df(papers_df, labels)
    """

    # ------------------------------------------------------------------
    # Community detection
    # ------------------------------------------------------------------

    def detect_communities(
        self, G: nx.DiGraph, resolution: float = 1.0
    ) -> Dict[str, int]:
        """
        Detect research communities using Louvain algorithm.

        Converts the directed citation graph to undirected before applying
        Louvain — standard practice since citation direction is less relevant
        for topical clustering than co-citation proximity.

        Args:
            G:          Directed citation graph from GraphEngine.build_graph().
            resolution: Louvain resolution parameter. Higher values produce more,
                        smaller communities. Default 1.0 is standard.

        Returns:
            Dict mapping node (paperId) → community_id (non-negative int).
            All nodes in G are covered. Returns {} if G has no nodes.
        """
        if G.number_of_nodes() == 0:
            return {}

        UG = G.to_undirected()
        return community_louvain.best_partition(UG, resolution=resolution)

    # ------------------------------------------------------------------
    # Cluster merging
    # ------------------------------------------------------------------

    def merge_small_clusters(
        self,
        partition: Dict[str, int],
        papers_df: pd.DataFrame,
        citations_df: pd.DataFrame,
        min_size: int,
    ) -> Dict[str, int]:
        """
        Merge clusters smaller than min_size into their best-connected neighbour.

        For each small cluster, counts citation edges (in either direction) to
        every other cluster and reassigns all its papers to the cluster with the
        most connections. Falls back to the largest non-small cluster when no
        citation edges exist.

        One pass is sufficient: merging only grows target clusters, so a cluster
        that was large enough at the start of the pass remains large.

        Args:
            partition:    Dict mapping paper_id → cluster_id (from detect_communities).
            papers_df:    Corpus paper DataFrame (paper_id required).
            citations_df: Citation edges DataFrame (source, target required).
            min_size:     Clusters with fewer than this many papers are merged.
                          Values <= 1 return a copy of partition unchanged.

        Returns:
            New Dict[str, int] with the same keys as partition. Does not mutate input.
        """
        if min_size <= 1 or not partition:
            return dict(partition)

        result = dict(partition)

        # Count papers per cluster
        cluster_sizes: Dict[int, int] = {}
        for cid in result.values():
            cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1

        # Identify small clusters sorted smallest-first
        small_clusters = sorted(
            [cid for cid, size in cluster_sizes.items() if size < min_size],
            key=lambda cid: cluster_sizes[cid],
        )

        if not small_clusters:
            return result

        small_set = set(small_clusters)

        # Build inter-cluster edge counts from citations_df
        edge_counts: Dict[tuple, int] = {}
        if not citations_df.empty:
            for _, row in citations_df.iterrows():
                src, tgt = row["source"], row["target"]
                if src not in result or tgt not in result:
                    continue
                c_src, c_tgt = result[src], result[tgt]
                if c_src == c_tgt:
                    continue
                key = (min(c_src, c_tgt), max(c_src, c_tgt))
                edge_counts[key] = edge_counts.get(key, 0) + 1

        # Largest non-small cluster (fallback)
        large_clusters = {cid: sz for cid, sz in cluster_sizes.items() if cid not in small_set}
        fallback_cluster = max(large_clusters, key=large_clusters.get) if large_clusters else None

        for small_cid in small_clusters:
            # Find edges from this small cluster to non-small clusters
            connections: Dict[int, int] = {}
            for (ca, cb), count in edge_counts.items():
                if ca == small_cid and cb not in small_set:
                    connections[cb] = connections.get(cb, 0) + count
                elif cb == small_cid and ca not in small_set:
                    connections[ca] = connections.get(ca, 0) + count

            if connections:
                target = max(connections, key=connections.get)
            elif fallback_cluster is not None:
                target = fallback_cluster
            else:
                continue  # no valid target (entire corpus is tiny)

            # Reassign all papers in the small cluster
            for pid, cid in result.items():
                if cid == small_cid:
                    result[pid] = target

            # Update sizes and remove from small_set
            cluster_sizes[target] = cluster_sizes.get(target, 0) + cluster_sizes.pop(small_cid, 0)
            small_set.discard(small_cid)

            # Update fallback if it has grown
            if fallback_cluster is not None:
                large_clusters[target] = cluster_sizes[target]
                fallback_cluster = max(large_clusters, key=large_clusters.get)

        return result

    # ------------------------------------------------------------------
    # Corpus enrichment
    # ------------------------------------------------------------------

    def add_cluster_assignments(
        self, papers_df: pd.DataFrame, partition: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Enrich papers_df with a cluster_id column.

        Returns a copy of papers_df with cluster_id as Int64 (nullable integer).
        Papers absent from the partition (e.g., added after graph build) receive
        pd.NA for cluster_id.

        Args:
            papers_df: Corpus paper DataFrame (must have "paper_id" column).
            partition: Dict from detect_communities().

        Returns:
            New DataFrame with "cluster_id" column (Int64). Original unchanged.
        """
        df = papers_df.copy()
        df["cluster_id"] = df["paper_id"].map(partition).astype("Int64")
        return df

    # ------------------------------------------------------------------
    # Cluster labeling
    # ------------------------------------------------------------------

    def label_clusters(
        self,
        papers_df: pd.DataFrame,
        partition: Dict[str, int],
        top_n: int = 5,
    ) -> Dict[int, str]:
        """
        Generate human-readable labels for each cluster from paper titles.

        For each community, collects the titles of corpus papers in that
        community, tokenizes them, removes common stopwords, and returns the
        top_n most frequent words as a comma-separated label.

        Args:
            papers_df: Corpus paper DataFrame (must have "paper_id", "title").
            partition: Dict from detect_communities().
            top_n:     Number of top keywords per cluster label.

        Returns:
            Dict mapping community_id → label string.
            All community_ids in partition.values() are covered.
            Clusters with no corpus titles fall back to "cluster-{id}".
        """
        if not partition:
            return {}

        # Build {community_id: [paper_ids]}
        cluster_papers: Dict[int, List[str]] = defaultdict(list)
        for node_id, cid in partition.items():
            cluster_papers[cid].append(node_id)

        # Build title lookup for corpus papers only
        title_lookup: Dict[str, str] = dict(
            zip(papers_df["paper_id"], papers_df["title"])
        )

        labels: Dict[int, str] = {}
        for cid, paper_ids in cluster_papers.items():
            titles = [
                title_lookup[pid]
                for pid in paper_ids
                if pid in title_lookup and title_lookup[pid]
            ]

            if not titles:
                labels[cid] = f"cluster-{cid}"
                continue

            # Tokenize: lowercase, split on non-alphanumeric
            words: List[str] = []
            for title in titles:
                tokens = re.split(r"[^a-z0-9]+", title.lower())
                words.extend(t for t in tokens if t and t not in _STOPWORDS)

            top_words = [w for w, _ in Counter(words).most_common(top_n)]
            labels[cid] = ", ".join(top_words) if top_words else f"cluster-{cid}"

        return labels

    # ------------------------------------------------------------------
    # Cluster summary
    # ------------------------------------------------------------------

    def cluster_summary_df(
        self,
        papers_df: pd.DataFrame,
        cluster_labels: Dict[int, str],
    ) -> pd.DataFrame:
        """
        Build a per-cluster summary DataFrame.

        Args:
            papers_df:      papers_df with "cluster_id" column (from
                            add_cluster_assignments()).
            cluster_labels: Dict from label_clusters().

        Returns:
            DataFrame with columns: cluster_id, label, paper_count,
            top_cited_paper — sorted by paper_count descending.
            Returns empty DataFrame with those columns if no cluster_id
            column exists or all values are pd.NA.
        """
        _cols = ["cluster_id", "label", "paper_count", "top_cited_paper"]

        if "cluster_id" not in papers_df.columns:
            return pd.DataFrame(columns=_cols)

        assigned = papers_df.dropna(subset=["cluster_id"])
        if assigned.empty:
            return pd.DataFrame(columns=_cols)

        records = []
        for cid, group in assigned.groupby("cluster_id"):
            cid_int = int(cid)
            top_idx = group["citation_count"].idxmax()
            records.append(
                {
                    "cluster_id": cid_int,
                    "label": cluster_labels.get(cid_int, f"cluster-{cid_int}"),
                    "paper_count": len(group),
                    "top_cited_paper": group.loc[top_idx, "title"],
                }
            )

        return (
            pd.DataFrame(records, columns=_cols)
            .sort_values("paper_count", ascending=False)
            .reset_index(drop=True)
        )
