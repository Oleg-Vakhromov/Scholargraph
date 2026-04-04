import time
import requests
from typing import List, Optional


BASE_URL = "https://api.semanticscholar.org/graph/v1"
_RATE_LIMIT_INTERVAL = 1.0  # seconds between requests (unauthenticated tier)
_BATCH_SIZE = 500            # max paper IDs per /paper/batch call


class SemanticScholarClient:
    def __init__(self, cache=None) -> None:
        """
        Args:
            cache: Optional DiskCache instance. When provided, get_papers_batch()
                   and get_references() skip already-cached IDs. search_bulk()
                   is never cached — searches should always return fresh results.
        """
        self._last_request_time: float = 0.0
        self._cache = cache

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < _RATE_LIMIT_INTERVAL:
            time.sleep(_RATE_LIMIT_INTERVAL - elapsed)

    def _get(self, path: str, params: dict) -> dict:
        delay = 60
        for attempt in range(4):
            self._wait_for_rate_limit()
            response = requests.get(f"{BASE_URL}{path}", params=params)
            self._last_request_time = time.time()
            if response.status_code != 429:
                break
            if attempt < 3:
                time.sleep(delay)
                delay *= 2

        if not response.ok:
            raise RuntimeError(
                f"GET {path} failed [{response.status_code}]: {response.text}"
            )
        return response.json()

    def _post(self, path: str, params: dict, body: dict) -> list:
        delay = 60
        for attempt in range(4):
            self._wait_for_rate_limit()
            response = requests.post(f"{BASE_URL}{path}", params=params, json=body)
            self._last_request_time = time.time()
            if response.status_code != 429:
                break
            if attempt < 3:
                time.sleep(delay)
                delay *= 2

        if not response.ok:
            raise RuntimeError(
                f"POST {path} failed [{response.status_code}]: {response.text}"
            )
        return response.json()

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
        GET /paper/search/bulk

        Returns up to `limit` papers matching the query.
        Handles pagination via the `token` cursor parameter.

        Default fields: paperId, title, year, citationCount
        Not cached — searches should always return fresh results.
        """
        if fields is None:
            fields = "paperId,title,year,citationCount"

        params: dict = {"query": query, "fields": fields}
        if sort:
            params["sort"] = sort

        results: List[dict] = []

        while len(results) < limit:
            data = self._get("/paper/search/bulk", params)
            batch = data.get("data", [])
            results.extend(batch)

            token = data.get("token")
            if not token or not batch:
                break

            params["token"] = token

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

        Default fields: paperId, title, abstract, authors, year, journal, publicationVenue, venue,
                        citationCount, referenceCount, fieldsOfStudy

        Authors are flattened to a comma-separated string in `authors_str`.
        Author count is stored in `num_authors`.
        None entries from the API (unknown paper IDs) are replaced with {}.
        """
        if fields is None:
            fields = (
                "paperId,title,abstract,authors,year,journal,publicationVenue,venue,"
                "citationCount,referenceCount,fieldsOfStudy"
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

                pid = paper.get("paperId")
                if pid:
                    fetched[pid] = paper
                    if self._cache is not None:
                        self._cache.set_paper(pid, paper)

            if i + _BATCH_SIZE < len(uncached_ids):
                time.sleep(_RATE_LIMIT_INTERVAL)

        # Fallback: for papers with no venue, try /paper/search/match by title
        for pid, paper in fetched.items():
            if paper.get("venue") is not None:
                continue
            title = paper.get("title")
            if not title:
                continue
            match = self.search_match(title)
            if match is None:
                continue
            _journal = match.get("journal") or {}
            _pub_venue = match.get("publicationVenue") or {}
            _legacy_venue = match.get("venue")
            resolved = _journal.get("name") or _pub_venue.get("name") or _legacy_venue or None
            if resolved:
                paper["venue"] = resolved
                if self._cache is not None:
                    self._cache.set_paper(pid, paper)

        # Persist new entries
        if self._cache is not None and fetched:
            self._cache.save()

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

        Default fields: paperId,title,journal,publicationVenue,venue
        """
        if not title:
            return None

        if fields is None:
            fields = "paperId,title,journal,publicationVenue,venue"

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
            self._cache.save()

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
