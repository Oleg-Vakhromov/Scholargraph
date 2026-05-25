import re
import time
import requests
from typing import List, Optional


BASE_URL = "https://api.semanticscholar.org/graph/v1"
_RATE_LIMIT_UNAUTH = 1.0   # seconds between requests — unauthenticated
_RATE_LIMIT_AUTH   = 0.1   # seconds between requests — authenticated (10 req/s)
_BATCH_SIZE = 500            # max paper IDs per /paper/batch call

_DOI_URL_PREFIXES = re.compile(
    r"^https?://(?:dx\.)?doi\.org/",
    re.IGNORECASE,
)


class SemanticScholarClient:
    def __init__(self, cache=None, api_key: Optional[str] = None) -> None:
        """
        Args:
            cache:   Optional DiskCache instance.
            api_key: Semantic Scholar API key. When provided, requests include
                     the x-api-key header and the rate limit drops to 0.1 s
                     (10 req/s). Defaults to the S2_API_KEY environment variable.
        """
        self._last_request_time: float = 0.0
        self._cache = cache
        self._api_key: Optional[str] = api_key
        self._rate_limit: float = _RATE_LIMIT_AUTH if api_key else _RATE_LIMIT_UNAUTH

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)

    def _headers(self) -> dict:
        if self._api_key:
            return {"x-api-key": self._api_key}
        return {}

    @staticmethod
    def _should_retry(status_code: int) -> bool:
        return status_code in (429, 500, 502, 503, 504)

    def _get(self, path: str, params: dict) -> dict:
        delay = 10
        response = None
        for attempt in range(5):
            self._wait_for_rate_limit()
            response = requests.get(f"{BASE_URL}{path}", params=params, headers=self._headers())
            self._last_request_time = time.time()
            if not self._should_retry(response.status_code):
                break
            if attempt < 4:
                time.sleep(delay)
                delay = min(delay * 2, 120)

        if not response.ok:
            raise RuntimeError(
                f"GET {path} failed [{response.status_code}]: {response.text}"
            )
        return response.json()

    def _post(self, path: str, params: dict, body: dict) -> list:
        delay = 10
        response = None
        for attempt in range(5):
            self._wait_for_rate_limit()
            response = requests.post(f"{BASE_URL}{path}", params=params, json=body, headers=self._headers())
            self._last_request_time = time.time()
            if not self._should_retry(response.status_code):
                break
            if attempt < 4:
                time.sleep(delay)
                delay = min(delay * 2, 120)

        if not response.ok:
            raise RuntimeError(
                f"POST {path} failed [{response.status_code}]: {response.text}"
            )
        return response.json()

    @staticmethod
    def _normalize_doi(raw: Optional[str]) -> Optional[str]:
        """
        Normalize a raw DOI string to a bare token.

        - Strips https://doi.org/, http://doi.org/, https://dx.doi.org/, etc.
        - Strips surrounding whitespace and trailing punctuation.
        - Returns None if result is empty.
        """
        if not raw:
            return None
        normalized = _DOI_URL_PREFIXES.sub("", raw).strip().rstrip(".")
        return normalized if normalized else None

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def search_bulk(
        self,
        query: str,
        limit: int = 100,
        fields: Optional[str] = None,
        sort: Optional[str] = None,
    ) -> List[dict]:
        """
        GET /paper/search (offset-paginated)

        Returns up to `limit` papers matching the query using the standard
        search endpoint, which applies semantic/relevance ranking and matches
        the result set shown on the Semantic Scholar website.

        The /paper/search/bulk endpoint uses strict keyword matching and often
        returns far fewer results for natural-language queries.

        Default fields: paperId, title, year, citationCount
        Not cached — searches should always return fresh results.
        """
        if fields is None:
            fields = "paperId,title,year,citationCount"

        results: List[dict] = []

        if limit > 1000:
            # /paper/search/bulk — cursor-paginated, no offset cap
            params: dict = {"query": query, "fields": fields}
            if sort:
                params["sort"] = sort
            while len(results) < limit:
                data = self._get("/paper/search/bulk", params)
                batch = data.get("data", [])
                results.extend(batch)
                token = data.get("token")
                if not token or not batch:
                    break
                params["token"] = token
        else:
            # /paper/search — semantic relevance ranking, offset + limit < 1000
            offset = 0
            page_size = min(100, limit)
            while len(results) < limit:
                page = min(page_size, limit - len(results), 999 - offset)
                if page <= 0:
                    break
                params: dict = {"query": query, "fields": fields, "limit": page, "offset": offset}
                if sort:
                    params["sort"] = sort
                data = self._get("/paper/search", params)
                batch = data.get("data", [])
                results.extend(batch)
                if not batch or len(batch) < page:
                    break
                offset += len(batch)

        return results[:limit]

    def get_papers_batch(
        self,
        paper_ids: List[str],
        fields: Optional[str] = None,
    ) -> List[dict]:
        """
        POST /paper/batch

        Fetches full metadata for a list of paper IDs.
        Chunks requests into batches of 500 (API limit).

        When a DiskCache is attached, already-cached IDs are served from cache
        without any API request. Results are stored to cache after each batch.

        Default fields: paperId, title, abstract, authors, year, journal,
                        publicationVenue, venue, externalIds,
                        citationCount, referenceCount, fieldsOfStudy

        Note: 'externalIds' is requested (not 'doi') — the standalone 'doi'
        field causes a 400 error from the Semantic Scholar API.

        Authors are flattened to a comma-separated string in `authors_str`.
        Author count is stored in `num_authors`.
        None entries from the API (unknown paper IDs) are replaced with {}.
        """
        if fields is None:
            fields = (
                "paperId,title,abstract,authors,year,journal,publicationVenue,venue,"
                "externalIds,citationCount,referenceCount,fieldsOfStudy"
            )

        # Split into cached vs uncached when cache is available
        if self._cache is not None:
            cached_ids = [pid for pid in paper_ids if self._cache.has_paper(pid)]
            uncached_ids = [pid for pid in paper_ids if not self._cache.has_paper(pid)]
        else:
            cached_ids = []
            uncached_ids = paper_ids

        # Fetch only uncached IDs
        fetched: dict = {}  # paperId → processed paper dict

        for i in range(0, len(uncached_ids), _BATCH_SIZE):
            chunk = uncached_ids[i: i + _BATCH_SIZE]
            batch = self._post(
                "/paper/batch",
                params={"fields": fields},
                body={"ids": chunk},
            )

            for paper in batch:
                if paper is None:
                    continue

                # Flatten authors list → comma-separated string + count
                raw_authors = paper.get("authors") or []
                paper["authors_str"] = ", ".join(
                    a.get("name", "") for a in raw_authors if a.get("name")
                ) or None
                paper["num_authors"] = len(raw_authors) if raw_authors else None

                # Normalize journal: journal.name → publicationVenue.name → venue (legacy string)
                _journal = paper.get("journal") or {}
                _pub_venue = paper.get("publicationVenue") or {}
                _legacy_venue = paper.get("venue")
                paper["venue"] = (
                    _journal.get("name") or _pub_venue.get("name") or _legacy_venue or None
                )

                # Extract DOI from externalIds (do NOT request 'doi' field directly — causes 400)
                ext_ids = paper.get("externalIds") or {}
                raw_doi = ext_ids.get("DOI") or ext_ids.get("doi") or None
                normalized_doi = self._normalize_doi(raw_doi)
                paper["doi"] = normalized_doi
                paper["doi_url"] = f"https://doi.org/{normalized_doi}" if normalized_doi else None

                pid = paper.get("paperId")
                if pid:
                    fetched[pid] = paper
                    if self._cache is not None:
                        self._cache.set_paper(pid, paper)

            if i + _BATCH_SIZE < len(uncached_ids):
                time.sleep(self._rate_limit)

        # Cache entries are written above; caller is responsible for cache.save()

        # Reconstruct result list in original input order
        all_results: List[dict] = []
        for pid in paper_ids:
            if pid in fetched:
                all_results.append(fetched[pid])
            elif self._cache is not None and self._cache.has_paper(pid):
                all_results.append(self._cache.get_paper(pid))
            else:
                all_results.append({})

        return all_results

    def search_match(
        self,
        title: str,
        fields: Optional[str] = None,
    ) -> Optional[dict]:
        """
        GET /paper/search/match

        Finds the single best-matching paper for an exact title query.
        Returns the raw API dict (with a `matchScore` key), or None if no
        match is found or the request fails.

        Default fields: paperId,title,journal,publicationVenue,venue,externalIds
        """
        if not title:
            return None

        if fields is None:
            fields = "paperId,title,journal,publicationVenue,venue,externalIds"

        try:
            data = self._get("/paper/search/match", params={"query": title, "fields": fields})
        except RuntimeError:
            return None

        return data if data.get("paperId") else None

    def get_references(self, paper_id: str) -> List[dict]:
        """
        GET /paper/{paperId}/references  (paginated)

        Returns a list of citation dicts with keys:
            source  — the requesting paper's ID
            target  — the referenced paper's ID
            title   — title of the referenced paper (may be None)
            year    — publication year of the referenced paper (may be None)

        Paginates using offset until all pages are exhausted (limit=1000 per page).
        When a DiskCache is attached, returns cached result immediately if available.
        Returns an empty list if paper_id is None or the paper has no references.
        """
        if not paper_id:
            return []

        if self._cache is not None and self._cache.has_references(paper_id):
            return self._cache.get_references(paper_id)

        citations = []
        offset = 0

        while True:
            data = self._get(
                f"/paper/{paper_id}/references",
                params={"fields": "paperId,title,year", "limit": 1000, "offset": offset},
            )
            batch = data.get("data") or []
            for item in batch:
                ref = item.get("citedPaper") or {}
                target_id = ref.get("paperId")
                if not target_id:
                    continue
                citations.append({
                    "source": paper_id,
                    "target": target_id,
                    "title": ref.get("title"),
                    "year": ref.get("year"),
                })
            next_offset = data.get("next")
            if not next_offset or not batch:
                break
            offset = next_offset

        if self._cache is not None:
            self._cache.set_references(paper_id, citations)

        return citations

    def get_citations(self, paper_id: str) -> List[dict]:
        """
        GET /paper/{paperId}/citations  (paginated)

        Returns papers that cite this paper (incoming citations).
        Each dict has keys:
            source  — the citing paper's ID
            target  — this paper's ID
            title   — title of the citing paper (may be None)
            year    — publication year of the citing paper (may be None)

        Paginates using offset until all pages are exhausted (limit=1000 per page).
        When a DiskCache is attached, returns cached result immediately if available.
        Returns an empty list if paper_id is None or the paper has no citations.
        """
        if not paper_id:
            return []

        if self._cache is not None and self._cache.has_incoming_citations(paper_id):
            return self._cache.get_incoming_citations(paper_id)

        citations = []
        offset = 0

        while True:
            data = self._get(
                f"/paper/{paper_id}/citations",
                params={"fields": "paperId,title,year", "limit": 1000, "offset": offset},
            )
            batch = data.get("data") or []
            for item in batch:
                citing = item.get("citingPaper") or {}
                source_id = citing.get("paperId")
                if not source_id:
                    continue
                citations.append({
                    "source": source_id,
                    "target": paper_id,
                    "title": citing.get("title"),
                    "year": citing.get("year"),
                })
            next_offset = data.get("next")
            if not next_offset or not batch:
                break
            offset = next_offset

        if self._cache is not None:
            self._cache.set_incoming_citations(paper_id, citations)
            self._cache.save()

        return citations
