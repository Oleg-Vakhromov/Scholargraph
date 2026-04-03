from typing import Callable, Optional, Tuple

import pandas as pd

from src.corpus.builder import CorpusBuilder
from src.expansion.filter import RelevanceFilter
from src.graph.engine import GraphEngine
from src.models.paper import Citation, Paper, citations_to_df, papers_to_df


class CorpusExpander:
    """
    Iteratively expands a paper corpus using PageRank-ranked candidates
    filtered by semantic relevance.

    Each iteration:
      1. Build citation graph + compute PageRank
      2. Select top-K out-of-corpus candidates
      3. Fetch candidate metadata
      4. Filter by cosine similarity to query
      5. Fetch references for passing papers
      6. Append new papers + citations to corpus (deduplicated)
      7. Check convergence

    Usage:
        expander = CorpusExpander(client, GraphEngine(), RelevanceFilter())
        papers_df, citations_df = expander.expand(
            corpus, query="knowledge graph", max_iterations=5
        )
    """

    def __init__(
        self,
        client,
        graph_engine: GraphEngine,
        relevance_filter: RelevanceFilter,
    ) -> None:
        """
        Args:
            client:           SemanticScholarClient instance.
            graph_engine:     GraphEngine instance (stateless — shared safely).
            relevance_filter: RelevanceFilter instance (model loaded once).
        """
        self._client = client
        self._engine = graph_engine
        self._filter = relevance_filter

    def expand(
        self,
        corpus: CorpusBuilder,
        query: str,
        max_iterations: int = 5,
        top_k_candidates: int = 100,
        relevance_threshold: float = 0.3,
        min_new_papers: int = 1,
        allowed_domains: list[str] | None = None,
        on_iteration: Callable[[int, int, int, int], None] | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Iteratively expand corpus with topically relevant, structurally
        important papers until convergence.

        Mutates corpus.papers_df and corpus.citations_df in place, then
        returns both DataFrames for convenience.

        Args:
            corpus:               CorpusBuilder with seed papers_df + citations_df.
            query:                Original search query (used for relevance scoring).
            max_iterations:       Hard cap on expansion iterations.
            top_k_candidates:     Candidates per iteration passed to GraphEngine.
            relevance_threshold:  Minimum cosine similarity for a paper to be added.
            min_new_papers:       Stop if fewer than this many papers added in an
                                  iteration (convergence criterion).

        Returns:
            (papers_df, citations_df) — final expanded DataFrames from corpus.
        """
        for iteration in range(max_iterations):
            seed_count = len(corpus.papers_df)

            # 1. Graph operations
            G = self._engine.build_graph(corpus.papers_df, corpus.citations_df)
            scores = self._engine.compute_pagerank(G)
            corpus.papers_df = self._engine.add_pagerank_scores(corpus.papers_df, scores)
            candidate_ids = self._engine.get_expansion_candidates(
                corpus.papers_df,
                corpus.citations_df,
                scores,
                top_k=top_k_candidates,
            )

            if not candidate_ids:
                break

            # 2. Fetch candidate metadata
            raw = self._client.get_papers_batch(candidate_ids)
            candidate_dicts = [d for d in raw if d and d.get("paperId")]

            if not candidate_dicts:
                break

            # Domain filter (when allowed_domains is set)
            if allowed_domains is not None:
                domain_set = set(allowed_domains)
                candidate_dicts = [
                    d for d in candidate_dicts
                    if d.get("fieldsOfStudy") and domain_set.intersection(d["fieldsOfStudy"])
                ]

            if not candidate_dicts:
                break

            # 3. Relevance filter
            relevant_dicts = self._filter.filter(
                query, candidate_dicts, threshold=relevance_threshold
            )

            if not relevant_dicts:
                break

            # 4. Build new Paper objects + DataFrame
            new_papers = [Paper.from_api_dict(d) for d in relevant_dicts]
            new_papers_df = papers_to_df(new_papers)

            # 5. Fetch references for each new paper
            all_new_citations = []
            for paper in new_papers:
                refs = self._client.get_references(paper.paper_id)
                for r in refs:
                    c = Citation.from_api_dict(r)
                    if c.source and c.target:
                        all_new_citations.append(c)

            new_citations_df = citations_to_df(all_new_citations)

            # 6. Append + deduplicate
            corpus.papers_df = (
                pd.concat([corpus.papers_df, new_papers_df], ignore_index=True)
                .drop_duplicates(subset=["paper_id"])
                .reset_index(drop=True)
            )

            if not new_citations_df.empty:
                corpus.citations_df = (
                    pd.concat([corpus.citations_df, new_citations_df], ignore_index=True)
                    .drop_duplicates(subset=["source", "target"])
                    .reset_index(drop=True)
                )

            new_paper_count = len(corpus.papers_df) - seed_count

            if on_iteration is not None:
                on_iteration(
                    iteration + 1,
                    seed_count,
                    len(new_citations_df),
                    new_paper_count,
                )

            # 7. Convergence check
            if len(new_papers_df) < min_new_papers:
                break

        return corpus.papers_df, corpus.citations_df
