import networkx as nx
import pandas as pd


class InfluenceAnalyzer:
    """
    Stateless influence metric computation for corpus papers.

    Replicates the influential-paper analysis from the reference notebook
    (Bibliometrics_06032024.ipynb) adapted for Semantic Scholar data.

    Methods:
        compute_isc()         — in-sample citations, ISC ratio, sample relevance
        compute_betweenness() — betweenness centrality on undirected citation graph
    """

    def compute_isc(
        self,
        papers_df: pd.DataFrame,
        citations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute in-sample citation metrics for each corpus paper.

        In-sample citations (ISC): number of corpus papers that cite this paper.
        Only edges where the citing paper (source) is also in papers_df count —
        citations from outside the corpus are excluded.

        Adds columns:
            isc              (int)   — in-corpus citation count
            isc_ratio        (float) — isc / citation_count * 100, capped at 100
            sample_relevance (float) — isc / len(papers_df) * 100

        Args:
            papers_df:    Corpus paper DataFrame (paper_id, citation_count required)
            citations_df: Citation edges DataFrame (source, target required)

        Returns:
            New DataFrame (copy) with three columns appended.
        """
        df = papers_df.copy()
        total_papers = len(df)

        if citations_df.empty or total_papers == 0:
            df["isc"] = 0
            df["isc_ratio"] = 0.0
            df["sample_relevance"] = 0.0
            return df

        corpus_ids = set(df["paper_id"])

        # Count in-corpus citations per target paper
        in_corpus_edges = citations_df[citations_df["source"].isin(corpus_ids)]
        isc_counts = in_corpus_edges.groupby("target").size()

        df["isc"] = df["paper_id"].map(isc_counts).fillna(0).astype(int)

        def _isc_ratio(row):
            if row["citation_count"] == 0:
                return 0.0
            return round(min(row["isc"] / row["citation_count"] * 100, 100.0), 2)

        df["isc_ratio"] = df.apply(_isc_ratio, axis=1)
        df["sample_relevance"] = (df["isc"] / total_papers * 100).round(2)

        return df

    def compute_betweenness(
        self,
        G: nx.DiGraph,
        papers_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute betweenness centrality for corpus papers.

        Converts G to undirected before computing — consistent with
        the reference notebook and standard bibliometric practice.
        Only corpus papers (present in papers_df) receive their score;
        out-of-corpus ghost nodes are excluded from the result.

        Adds column:
            betweenness_centrality (float) — normalised [0, 1]

        Args:
            G:         Directed citation graph from GraphEngine.build_graph()
            papers_df: Corpus paper DataFrame (paper_id required)

        Returns:
            New DataFrame (copy) with betweenness_centrality column appended.
        """
        df = papers_df.copy()

        if G.number_of_nodes() == 0:
            df["betweenness_centrality"] = 0.0
            return df

        scores = nx.betweenness_centrality(G.to_undirected())
        df["betweenness_centrality"] = (
            df["paper_id"].map(scores).fillna(0.0).round(4)
        )
        return df
