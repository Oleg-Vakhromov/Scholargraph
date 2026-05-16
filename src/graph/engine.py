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
            corpus.papers_df, corpus.citations_df, scores
        )
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _canonical_map(self, papers_df: pd.DataFrame) -> Dict[str, str]:
        """
        Build a paper_id → canonical_id map for graph node identity.

        Canonical ID is the DOI when available (since one DOI may have multiple
        Semantic Scholar paper IDs), otherwise the paper_id itself.
        """
        if "doi" not in papers_df.columns:
            return dict(zip(papers_df["paper_id"], papers_df["paper_id"]))
        result = {}
        for pid, doi in zip(papers_df["paper_id"], papers_df["doi"]):
            if doi and isinstance(doi, str) and doi.strip():
                result[pid] = doi.strip()
            else:
                result[pid] = pid
        return result

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(self, papers_df: pd.DataFrame, citations_df: pd.DataFrame) -> nx.DiGraph:
        """
        Build a directed citation graph from corpus data.

        Nodes: canonical IDs for corpus papers (DOI when available, else paper_id).
               Papers sharing a DOI are merged into a single node.
        Edges: source→target pairs from citations_df, translated to canonical IDs
               for corpus members; out-of-corpus IDs are kept as-is.

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

        canon = self._canonical_map(papers_df)

        # Add corpus nodes using canonical IDs (deduplicates papers sharing a DOI)
        G.add_nodes_from(set(canon.values()))

        # Add citation edges translated to canonical IDs and deduplicated
        if not citations_df.empty:
            src = citations_df["source"].apply(lambda x: canon.get(x, x))
            tgt = citations_df["target"].apply(lambda x: canon.get(x, x))
            edges = (
                pd.DataFrame({"source": src, "target": tgt})
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
        canon = self._canonical_map(papers_df)
        score_map = {pid: pagerank_scores.get(cid, 0.0) for pid, cid in canon.items()}
        df["pagerank_score"] = df["paper_id"].map(score_map).fillna(0.0)
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
        strategy: str = "pagerank",
        min_citations: int = 1,
    ) -> List[str]:
        """
        Identify expansion candidates: papers referenced by the corpus but not yet
        in the corpus, filtered by a minimum citation count and a 75th-percentile
        threshold, then sorted descending.

        Both strategies require each candidate to appear as a citation target at least
        `min_citations` times in the corpus before any further filtering:

        - "pagerank": among candidates meeting min_citations, keep those at or above
          the 75th percentile of their PageRank scores, sorted by score.
        - "citation_count": threshold = max(min_citations, 75th percentile of
          in-corpus citation counts), sorted by count.

        Args:
            papers_df:     Current corpus (has "paper_id" column)
            citations_df:  Citation edges (has "source", "target" columns)
            pagerank_scores: Dict from compute_pagerank()
            strategy:      "pagerank" (default) or "citation_count"
            min_citations: Minimum number of in-corpus papers that must cite a
                           candidate for it to qualify (default 1).

        Returns:
            List of qualifying paper IDs sorted descending by strategy score.
            Never includes IDs already in papers_df.
            Returns [] if citations_df is empty or no candidates qualify.
        """
        if citations_df.empty:
            return []

        canon = self._canonical_map(papers_df)
        corpus_canonicals = set(canon.values())

        # Translate citations through the canonical map so that corpus papers
        # with multiple S2 IDs (same DOI) are treated as the same endpoint
        norm_src = citations_df["source"].apply(lambda x: canon.get(x, x))
        norm_tgt = citations_df["target"].apply(lambda x: canon.get(x, x))

        referenced_ids = set(norm_tgt.dropna()) | set(norm_src.dropna())
        candidates = referenced_ids - corpus_canonicals

        if not candidates:
            return []

        # In-corpus citation count using DOI-normalized targets
        counts = norm_tgt[norm_tgt.isin(candidates)].value_counts()
        candidates = {cid for cid in candidates if counts.get(cid, 0) >= min_citations}

        if not candidates:
            return []

        if strategy == "citation_count":
            counts = counts[counts.index.isin(candidates)]
            threshold = int(max(min_citations, counts.quantile(0.75))) if not counts.empty else min_citations
            qualifying = counts[counts >= threshold].index.tolist()
            return sorted(qualifying, key=lambda cid: counts[cid], reverse=True)
        else:
            candidate_scores = {
                cid: pagerank_scores.get(cid, 0.0) for cid in candidates
            }
            score_series = pd.Series(candidate_scores)
            threshold = float(score_series.quantile(0.75)) if not score_series.empty else 0.0
            qualifying = score_series[score_series >= threshold].index.tolist()
            return sorted(qualifying, key=lambda cid: candidate_scores[cid], reverse=True)
