import networkx as nx
import pandas as pd
from typing import Dict, List


class GraphEngine:
    """
    Stateless graph operations for corpus expansion.

    All methods accept DataFrames and return new objects — no instance state,
    no side effects on inputs.

    Usage (Phase 3 loop):
        engine = GraphEngine()
        G = engine.build_graph(corpus.papers_df, corpus.citations_df)
        scores = engine.compute_pagerank(G)
        corpus.papers_df = engine.add_pagerank_scores(corpus.papers_df, scores)
        candidates = engine.get_expansion_candidates(
            corpus.papers_df, corpus.citations_df, scores, top_k=100
        )
    """

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(self, papers_df: pd.DataFrame, citations_df: pd.DataFrame) -> nx.DiGraph:
        """
        Build a directed citation graph from corpus data.

        Nodes: all paper_ids in papers_df (corpus members)
        Edges: source→target pairs from citations_df (deduplicated)

        Out-of-corpus targets are added as nodes automatically by NetworkX
        when edges are added — this is intentional for PageRank computation
        over the full referenced neighbourhood.

        Args:
            papers_df:    Corpus paper metadata (must have "paper_id" column)
            citations_df: Citation edges (must have "source", "target" columns)

        Returns:
            nx.DiGraph
        """
        G = nx.DiGraph()

        # Add all corpus papers as nodes
        G.add_nodes_from(papers_df["paper_id"])

        # Add citation edges (deduplicated)
        if not citations_df.empty:
            edges = (
                citations_df[["source", "target"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            G.add_edges_from(edges)

        return G

    # ------------------------------------------------------------------
    # PageRank
    # ------------------------------------------------------------------

    def compute_pagerank(self, G: nx.DiGraph, alpha: float = 0.85) -> Dict[str, float]:
        """
        Compute PageRank scores for all nodes in G.

        Args:
            G:     Directed citation graph
            alpha: Damping factor (default 0.85, per README)

        Returns:
            Dict mapping node (paperId) → PageRank score.
            Scores sum to approximately 1.0.

        Edge case: if G has no edges, returns uniform scores (1/N per node)
        so downstream code always gets valid float scores.
        """
        if G.number_of_nodes() == 0:
            return {}

        if G.number_of_edges() == 0:
            uniform = 1.0 / G.number_of_nodes()
            return {node: uniform for node in G.nodes()}

        return nx.pagerank(G, alpha=alpha)

    # ------------------------------------------------------------------
    # Corpus enrichment
    # ------------------------------------------------------------------

    def add_pagerank_scores(
        self, papers_df: pd.DataFrame, pagerank_scores: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Enrich papers_df with a pagerank_score column.

        Returns a copy of papers_df sorted by pagerank_score descending.
        Papers absent from the graph receive score 0.0.

        Args:
            papers_df:       Corpus paper DataFrame
            pagerank_scores: Dict from compute_pagerank()

        Returns:
            New DataFrame with "pagerank_score" column (float64), sorted descending.
        """
        df = papers_df.copy()
        df["pagerank_score"] = df["paper_id"].map(pagerank_scores).fillna(0.0)
        df = df.sort_values("pagerank_score", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Expansion candidate selection
    # ------------------------------------------------------------------

    def get_expansion_candidates(
        self,
        papers_df: pd.DataFrame,
        citations_df: pd.DataFrame,
        pagerank_scores: Dict[str, float],
        top_k: int = 100,
    ) -> List[str]:
        """
        Identify top-K expansion candidates: papers referenced by the corpus
        but not yet in the corpus, ranked by PageRank score.

        Implements README section 4.6 — Step 6.

        Args:
            papers_df:       Current corpus (has "paper_id" column)
            citations_df:    Citation edges (has "target" column)
            pagerank_scores: Dict from compute_pagerank()
            top_k:           Maximum candidates to return

        Returns:
            List of paper IDs sorted by pagerank_score descending.
            Never includes IDs already in papers_df.
            Returns [] if citations_df is empty or all refs are already in corpus.
        """
        if citations_df.empty:
            return []

        corpus_ids = set(papers_df["paper_id"])
        referenced_ids = set(citations_df["target"].dropna())
        candidates = referenced_ids - corpus_ids

        if not candidates:
            return []

        sorted_candidates = sorted(
            candidates,
            key=lambda cid: pagerank_scores.get(cid, 0.0),
            reverse=True,
        )

        return sorted_candidates[:top_k]
