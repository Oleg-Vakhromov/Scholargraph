import os
from typing import Optional, Tuple

import pandas as pd

from src.models.paper import Citation, Paper, citations_to_df, papers_to_df


class CorpusBuilder:
    """
    Builds and manages the initial paper corpus.

    Usage:
        cache = DiskCache(cache_dir="cache"); cache.load()
        client = SemanticScholarClient(cache=cache)
        corpus = CorpusBuilder(client)
        df = corpus.seed("knowledge graph", limit=500)
        corpus.save_papers("data/papers.csv")
    """

    def __init__(self, client) -> None:
        """
        Args:
            client: SemanticScholarClient instance (or None for load-only use).
        """
        self._client = client
        self.papers_df: pd.DataFrame = pd.DataFrame()
        self.citations_df: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Corpus construction
    # ------------------------------------------------------------------

    def seed(
        self,
        query: str,
        limit: int = 500,
        year_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """
        Build initial corpus from a keyword query.

        Implements README sections 4.2–4.3:
          Step 1 — search_bulk → (paperId, title, year, citationCount)
          Step 2 — get_papers_batch → full metadata enrichment

        Args:
            query:      Keyword search string
            limit:      Max papers to retrieve (default 500)
            year_range: Optional (year_min, year_max) tuple — inclusive filter

        Returns:
            papers_df with enriched metadata; also stored as self.papers_df
        """
        # Step 1: Initial retrieval
        search_results = self._client.search_bulk(
            query,
            limit=limit,
            fields="paperId,title,year,citationCount",
        )

        paper_ids = [p["paperId"] for p in search_results if p.get("paperId")]

        if not paper_ids:
            self.papers_df = pd.DataFrame()
            return self.papers_df

        # Step 2: Metadata enrichment
        raw_papers = self._client.get_papers_batch(paper_ids)

        papers = []
        for d in raw_papers:
            if not d or not d.get("paperId"):
                continue
            papers.append(Paper.from_api_dict(d))

        df = papers_to_df(papers)

        # Optional year filter
        if year_range is not None and not df.empty:
            year_min, year_max = year_range
            mask = df["year"].notna() & df["year"].between(year_min, year_max)
            df = df[mask].reset_index(drop=True)

        self.papers_df = df
        return self.papers_df

    def extract_domains(self) -> list[str]:
        """
        Return a sorted list of unique field-of-study strings across all seed papers.

        Each row's `fields_of_study` value is either None or a list of strings.
        Returns [] if the corpus is empty or no domain data is present.
        """
        if self.papers_df.empty:
            return []

        collected: set[str] = set()
        for value in self.papers_df["fields_of_study"]:
            if not value:
                continue
            for domain in value:
                if domain:
                    collected.add(domain)

        return sorted(collected)

    def apply_domain_filter(
        self,
        selected_domains: list[str],
        max_papers: int | None = None,
    ) -> pd.DataFrame:
        """
        Filter papers_df to papers belonging to at least one selected domain,
        then optionally cap to the top-N by citation count.

        Mutates self.papers_df in place and returns it.

        Args:
            selected_domains: List of domain strings to keep.
            max_papers:       If set, retain only the top-N papers by citation_count
                              after domain filtering.

        Returns:
            Filtered (and optionally capped) papers_df.
        """
        if self.papers_df.empty or not selected_domains:
            self.papers_df = pd.DataFrame()
            return self.papers_df

        domain_set = set(selected_domains)

        mask = [
            bool(value and domain_set.intersection(value))
            for value in self.papers_df["fields_of_study"]
        ]

        filtered_df = self.papers_df[mask].reset_index(drop=True)

        if max_papers is not None and len(filtered_df) > max_papers:
            filtered_df = (
                filtered_df
                .sort_values("citation_count", ascending=False)
                .head(max_papers)
                .reset_index(drop=True)
            )

        self.papers_df = filtered_df
        return self.papers_df

    def fetch_references(self, paper_ids=None) -> pd.DataFrame:
        """
        Fetch references for corpus papers and build citations_df.

        Implements README section 4.4 — Step 3: Build Citation Graph.

        Args:
            paper_ids: List of paper IDs to fetch refs for.
                       Defaults to all IDs in self.papers_df.

        Returns:
            citations_df with columns [source, target, title, year];
            also stored as self.citations_df.
        """
        if paper_ids is None:
            if self.papers_df.empty:
                self.citations_df = pd.DataFrame()
                return self.citations_df
            paper_ids = list(self.papers_df["paper_id"])

        all_citations = []
        for paper_id in paper_ids:
            for r in self._client.get_references(paper_id):
                c = Citation.from_api_dict(r)
                if c.source and c.target:
                    all_citations.append(c)
            for r in self._client.get_citations(paper_id):
                c = Citation.from_api_dict(r)
                if c.source and c.target:
                    all_citations.append(c)

        self.citations_df = citations_to_df(all_citations)
        return self.citations_df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_citations(self, path: str) -> None:
        """
        Persist citations_df to CSV.

        Raises:
            ValueError: if citations_df is empty (fetch_references() not yet called)
        """
        if self.citations_df.empty:
            raise ValueError("No citations to save — call fetch_references() first")

        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        self.citations_df.to_csv(path, index=False)

    def load_citations(self, path: str) -> pd.DataFrame:
        """
        Load citations_df from a previously saved CSV.

        Restores Int64 dtype for year column.

        Returns:
            Restored citations_df; also stored as self.citations_df
        """
        df = pd.read_csv(path)

        if "year" in df.columns:
            df["year"] = df["year"].astype("Int64")

        self.citations_df = df
        return self.citations_df

    def save_papers(self, path: str) -> None:
        """
        Persist papers_df to CSV.

        Raises:
            ValueError: if corpus is empty (seed() not yet called)
        """
        if self.papers_df.empty:
            raise ValueError("No corpus to save — call seed() first")

        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        self.papers_df.to_csv(path, index=False)

    def load_papers(self, path: str) -> pd.DataFrame:
        """
        Load papers_df from a previously saved CSV.

        Restores correct dtypes (Int64 for nullable integer columns).

        Returns:
            Restored papers_df; also stored as self.papers_df
        """
        df = pd.read_csv(path)

        # Restore nullable integer dtypes lost during CSV serialisation
        for col in ("year", "num_authors", "reference_count"):
            if col in df.columns:
                df[col] = df[col].astype("Int64")

        if "citation_count" in df.columns:
            df["citation_count"] = df["citation_count"].fillna(0).astype("int64")

        # Ensure doi/doi_url columns exist for CSVs saved before Phase 15
        for col in ("doi", "doi_url"):
            if col not in df.columns:
                df[col] = None

        self.papers_df = df
        return self.papers_df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """
        Return summary statistics for the current corpus.

        Returns:
            dict with total_papers, year_min, year_max, avg_citation_count
            (or {"total_papers": 0} if corpus is empty)
        """
        if self.papers_df.empty:
            return {"total_papers": 0}

        year_col = self.papers_df["year"]
        has_years = year_col.notna().any()

        return {
            "total_papers": len(self.papers_df),
            "year_min": int(year_col.min()) if has_years else None,
            "year_max": int(year_col.max()) if has_years else None,
            "avg_citation_count": round(
                float(self.papers_df["citation_count"].mean()), 1
            ),
        }
