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

        _cache = getattr(self._client, "_cache", None)
        if _cache is not None:
            _cache.save()

        self.papers_df = df
        return self.papers_df

    def seed_scopus(
        self,
        scopus_client,
        query: str,
        limit: int = 500,
        year_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """
        Build initial corpus from Scopus search results.

        After fetching from Scopus, tries to resolve each paper's DOI to a
        Semantic Scholar paper ID so that reference fetching works normally.
        Papers with no S2 match keep their Scopus EID as paper_id (references
        will be recovered later via OpenAlex enrichment).

        Returns papers_df; also stored as self.papers_df.
        """
        raw = scopus_client.search(query, limit=limit)
        if not raw:
            self.papers_df = pd.DataFrame()
            return self.papers_df

        papers = [Paper.from_api_dict(d) for d in raw if d.get("paperId")]

        df = papers_to_df(papers)

        if year_range is not None and not df.empty:
            year_min, year_max = year_range
            mask = df["year"].notna() & df["year"].between(year_min, year_max)
            df = df[mask].reset_index(drop=True)

        self.papers_df = df
        return self.papers_df

    def seed_both(
        self,
        scopus_client,
        query: str,
        limit: int = 500,
        year_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """
        Seed corpus from both Semantic Scholar and Scopus, merged by DOI.

        S2 results are fetched first (up to limit). Scopus results are fetched
        next (up to limit), then merged: papers already in the S2 set (matched
        by DOI) are skipped; new papers are appended. The merged set is
        capped at limit.

        Returns papers_df; also stored as self.papers_df.
        """
        # Step 1: S2 seed
        self.seed(query, limit=limit, year_range=year_range)
        s2_dois = set(
            str(d).lower()
            for d in self.papers_df["doi"]
            if d is not None and not (isinstance(d, float) and pd.isna(d))
        )

        # Step 2: Scopus seed — resolve S2 IDs, deduplicate
        raw_scopus = scopus_client.search(query, limit=limit)
        if raw_scopus:
            new_papers = []
            for d in raw_scopus:
                doi = d.get("doi")
                if doi and doi.lower() in s2_dois:
                    continue  # already in S2 results
                p = Paper.from_api_dict(d)
                if p.paper_id:
                    new_papers.append(p)

            if new_papers:
                new_df = papers_to_df(new_papers)
                if year_range is not None:
                    year_min, year_max = year_range
                    mask = new_df["year"].notna() & new_df["year"].between(year_min, year_max)
                    new_df = new_df[mask].reset_index(drop=True)

                self.papers_df = (
                    pd.concat([self.papers_df, new_df], ignore_index=True)
                    .drop_duplicates(subset=["paper_id"])
                    .head(limit)
                    .reset_index(drop=True)
                )

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
        if self.papers_df.empty:
            return self.papers_df
        if not selected_domains:
            return self.papers_df

        domain_set = set(selected_domains)
        mask = [
            bool(value and domain_set.intersection(value))
            for value in self.papers_df["fields_of_study"]
        ]
        filtered_df = self.papers_df[mask].reset_index(drop=True)

        if max_papers is not None:
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
            # Scopus EIDs (2-s2.0-...) are not valid S2 paper IDs — skip them;
            # their intra-corpus edges are recovered by enrich_references() via OpenAlex.
            if str(paper_id).startswith("2-s2.0-"):
                continue
            try:
                refs = self._client.get_references(paper_id)
            except Exception:
                continue
            for r in refs:
                c = Citation.from_api_dict(r)
                if c.source and c.target:
                    all_citations.append(c)

        _cache = getattr(self._client, "_cache", None)
        if _cache is not None:
            _cache.save()

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
    # Enrichment
    # ------------------------------------------------------------------

    def enrich_dois(self, crossref_client) -> tuple[int, int]:
        """
        Back-fill missing DOIs in papers_df using CrossRef.

        Iterates rows where doi is None, queries CrossRef by title,
        and updates doi + doi_url columns in-place.

        Returns (filled, failed) counts for rows that were attempted.
        """
        if self.papers_df.empty:
            return 0, 0

        filled = 0
        failed = 0
        for idx, row in self.papers_df.iterrows():
            if not pd.isna(self.papers_df.at[idx, "doi"]) and self.papers_df.at[idx, "doi"] is not None:
                continue
            title = row.get("title") if isinstance(row.get("title"), str) else None
            if not title or not title.strip():
                continue
            year = row.get("year")
            year = int(year) if pd.notna(year) else None
            result = crossref_client.lookup(title, year=year)
            if result is None:
                failed += 1
                continue
            raw_doi = result.get("DOI")
            if not raw_doi:
                failed += 1
                continue
            doi = raw_doi.strip()
            self.papers_df.at[idx, "doi"] = doi
            self.papers_df.at[idx, "doi_url"] = f"https://doi.org/{doi}"
            filled += 1
            _cache = getattr(self._client, "_cache", None)
            if _cache is not None:
                pid = row.get("paper_id")
                if pid and _cache.has_paper(pid):
                    cached = dict(_cache.get_paper(pid))
                    cached["doi"] = doi
                    _cache.set_paper(pid, cached)

        if filled and getattr(self._client, "_cache", None) is not None:
            self._client._cache.save()

        return filled, failed

    def enrich_references(self, openalex_client) -> tuple[int, int]:
        """
        Recover missing citation edges for papers that have reference_count > 0
        but no source rows in citations_df, using OpenAlex as the reference source.

        Only intra-corpus edges are added: both source and target must exist in
        papers_df (matched by normalized DOI). Self-loops are excluded.
        Idempotent — duplicate (source, target) pairs are never added.

        Returns (edges_added, papers_failed) where papers_failed is the count
        of candidate papers for which OpenAlex returned no usable edges.
        """
        if self.papers_df.empty:
            return 0

        # Build normalized DOI → paper_id index
        doi_index: dict = {}
        for _, row in self.papers_df.iterrows():
            raw_doi = row.get("doi")
            if raw_doi is None or (isinstance(raw_doi, float) and pd.isna(raw_doi)):
                continue
            doi_str = str(raw_doi).strip()
            if not doi_str:
                continue
            doi_str = doi_str.lstrip("https://doi.org/").lstrip("http://doi.org/").lower()
            if doi_str:
                doi_index[doi_str] = row["paper_id"]

        # Find zero-reference candidates
        if not self.citations_df.empty and "source" in self.citations_df.columns:
            existing_sources = set(self.citations_df["source"])
        else:
            existing_sources = set()

        candidates = self.papers_df[
            (~self.papers_df["paper_id"].isin(existing_sources))
            & (self.papers_df["reference_count"].notna())
            & (self.papers_df["reference_count"] > 0)
        ]

        if candidates.empty:
            return 0, 0

        # Existing (source, target) pairs for dedup
        if not self.citations_df.empty and "target" in self.citations_df.columns:
            existing_pairs: set = set(
                zip(self.citations_df["source"], self.citations_df["target"])
            )
        else:
            existing_pairs = set()

        new_rows = []
        seen_pairs: set = set()
        papers_failed = 0

        for _, row in candidates.iterrows():
            paper_id = row["paper_id"]
            doi = row.get("doi")
            if doi is not None and (isinstance(doi, float) and pd.isna(doi)):
                doi = None
            title = row.get("title")

            refs = openalex_client.get_references(doi=doi, title=title)
            edges_before = len(new_rows)
            for ref in refs:
                raw_ref_doi = ref.get("doi") or ""
                norm_doi = raw_ref_doi.strip().lstrip("https://doi.org/").lstrip("http://doi.org/").lower()
                if not norm_doi or norm_doi not in doi_index:
                    continue
                target_id = doi_index[norm_doi]
                if target_id == paper_id:
                    continue  # no self-loops
                pair = (paper_id, target_id)
                if pair in existing_pairs or pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                new_rows.append({
                    "source": paper_id,
                    "target": target_id,
                    "title": ref.get("title"),
                    "year": ref.get("year"),
                })
            if len(new_rows) == edges_before:
                papers_failed += 1

        if not new_rows:
            return 0, papers_failed

        new_df = pd.DataFrame(new_rows)
        new_df["year"] = new_df["year"].astype("Int64")

        if self.citations_df.empty:
            self.citations_df = new_df.reset_index(drop=True)
        else:
            self.citations_df = pd.concat(
                [self.citations_df, new_df], ignore_index=True
            )

        return len(new_rows), papers_failed

    def enrich_venues(self, crossref_client) -> tuple[int, int]:
        """
        Back-fill missing journal/venue names in papers_df using CrossRef.

        For rows where journal is None or empty:
          - If doi is set: use crossref_client.lookup_by_doi(doi) — deterministic
          - Else if title is set: use crossref_client.lookup(title, year) — fuzzy match
        Extracts container-title[0] from the CrossRef work dict and updates
        the journal column in-place.

        Returns (filled, failed) counts for rows that were attempted.
        """
        if self.papers_df.empty:
            return 0, 0

        filled = 0
        failed = 0
        for idx, row in self.papers_df.iterrows():
            current = self.papers_df.at[idx, "journal"]
            if not (pd.isna(current) or current is None or str(current).strip() == ""):
                continue

            result = None
            doi = row.get("doi")
            if doi and pd.notna(doi) and str(doi).strip():
                result = crossref_client.lookup_by_doi(str(doi).strip())

            if result is None:
                title = row.get("title")
                if not title or not str(title).strip():
                    continue
                year = row.get("year")
                year = int(year) if pd.notna(year) else None
                result = crossref_client.lookup(str(title), year=year)

            if result is None:
                failed += 1
                continue

            container_titles = result.get("container-title") or []
            if not container_titles:
                failed += 1
                continue
            venue = container_titles[0].strip()
            if not venue:
                failed += 1
                continue

            self.papers_df.at[idx, "journal"] = venue
            filled += 1
            _cache = getattr(self._client, "_cache", None)
            if _cache is not None:
                pid = row.get("paper_id")
                if pid and _cache.has_paper(pid):
                    cached = dict(_cache.get_paper(pid))
                    cached["journal"] = venue
                    _cache.set_paper(pid, cached)

        if filled and getattr(self._client, "_cache", None) is not None:
            self._client._cache.save()

        return filled, failed

    def enrich_from_scopus(self, scopus_client) -> tuple[int, int]:
        """
        Fill missing venue, abstract, and citation count from Scopus by DOI.

        Only queries Scopus for rows that have a DOI and are missing at least
        one of: journal, abstract. Citation count is updated if Scopus reports
        a higher value (Scopus counts are generally more complete).

        Returns (filled, failed) where filled = rows where at least one field
        was updated.
        """
        if self.papers_df.empty:
            return 0, 0

        filled = 0
        failed = 0

        for idx, row in self.papers_df.iterrows():
            doi = row.get("doi")
            if not doi or (isinstance(doi, float) and pd.isna(doi)):
                continue

            needs_venue = pd.isna(row.get("journal")) or not str(row.get("journal") or "").strip()
            needs_abstract = pd.isna(row.get("abstract")) or not str(row.get("abstract") or "").strip()
            scopus_citations = int(row.get("citation_count") or 0)

            if not (needs_venue or needs_abstract):
                continue

            result = scopus_client.get_by_doi(str(doi).strip())
            if not result:
                failed += 1
                continue

            updated = False
            if needs_venue and result.get("venue"):
                self.papers_df.at[idx, "journal"] = result["venue"]
                updated = True
            if needs_abstract and result.get("abstract"):
                self.papers_df.at[idx, "abstract"] = result["abstract"]
                updated = True
            scopus_count = result.get("citationCount") or 0
            if scopus_count > scopus_citations:
                self.papers_df.at[idx, "citation_count"] = scopus_count
                updated = True

            if updated:
                filled += 1
            else:
                failed += 1

        return filled, failed

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
