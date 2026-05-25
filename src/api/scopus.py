import time
import requests
from typing import List, Optional


_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"
_ABSTRACT_URL = "https://api.elsevier.com/content/abstract"
_RATE_LIMIT = 0.1   # 10 req/s with API key
_PAGE_SIZE = 25     # Scopus developer-key maximum per request


class ScopusClient:
    """
    Elsevier Scopus API client.

    Requires a personal API key from dev.elsevier.com.
    Institutional access (campus IP) gives full metadata; personal developer
    keys give title, author, abstract, DOI, venue, citation count, keywords.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        return {"X-ELS-APIKey": self._api_key, "Accept": "application/json"}

    def _wait(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < _RATE_LIMIT:
            time.sleep(_RATE_LIMIT - elapsed)

    def _get(self, url: str, params: dict) -> Optional[dict]:
        self._wait()
        try:
            resp = requests.get(url, headers=self._headers(), params=params, timeout=30)
            self._last_request_time = time.time()
            if not resp.ok:
                return None
            return resp.json()
        except Exception:
            return None

    @staticmethod
    def _normalize(entry: dict) -> dict:
        """Map a Scopus search-result entry to Paper.from_api_dict-compatible keys."""
        eid = entry.get("eid") or entry.get("dc:identifier") or ""

        cover_date = entry.get("prism:coverDate") or ""
        year_str = cover_date[:4]
        year = int(year_str) if year_str.isdigit() else None

        doi = entry.get("prism:doi") or None
        if doi:
            doi = doi.strip()

        raw_kw = entry.get("authkeywords") or ""
        fields = [k.strip() for k in raw_kw.replace(" | ", "|").split("|") if k.strip()] or None

        return {
            "paperId": eid,
            "title": entry.get("dc:title") or "",
            "abstract": entry.get("dc:description"),
            "authors_str": entry.get("dc:creator"),
            "num_authors": None,
            "year": year,
            "venue": entry.get("prism:publicationName"),
            "citationCount": int(entry.get("citedby-count") or 0),
            "referenceCount": None,
            "fieldsOfStudy": fields,
            "doi": doi,
            "doi_url": f"https://doi.org/{doi}" if doi else None,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 100) -> List[dict]:
        """
        Search Scopus for papers matching query.

        Returns normalized paper dicts compatible with Paper.from_api_dict.
        Paginates automatically up to limit.
        """
        results: List[dict] = []
        start = 0

        while len(results) < limit:
            count = min(_PAGE_SIZE, limit - len(results))
            data = self._get(_SEARCH_URL, {
                "query": query,
                "field": (
                    "eid,dc:identifier,dc:title,prism:doi,prism:publicationName,"
                    "prism:coverDate,citedby-count,dc:creator,authkeywords,dc:description"
                ),
                "count": count,
                "start": start,
            })
            if data is None:
                break

            search_results = data.get("search-results", {})
            entries = search_results.get("entry", [])

            # Scopus signals "empty" with a single error entry
            if not entries or (len(entries) == 1 and entries[0].get("error")):
                break

            for entry in entries:
                if entry.get("error"):
                    continue
                results.append(self._normalize(entry))

            total = int(search_results.get("opensearch:totalResults") or 0)
            start += len(entries)
            if start >= total:
                break

        return results[:limit]

    def get_by_doi(self, doi: str) -> Optional[dict]:
        """
        Fetch a single paper's metadata from Scopus by DOI.

        Returns a normalized paper dict (same keys as search()) or None.
        Used for enrichment — fills venue, abstract, citation count.
        """
        data = self._get(f"{_ABSTRACT_URL}/doi/{doi}", {"field": "coredata"})
        if data is None:
            return None

        core = data.get("abstracts-retrieval-response", {}).get("coredata", {})
        if not core:
            return None

        # coredata uses the same field names as search results
        cover_date = core.get("prism:coverDate") or ""
        year_str = cover_date[:4]
        year = int(year_str) if year_str.isdigit() else None

        raw_doi = core.get("prism:doi") or doi
        raw_doi = raw_doi.strip()

        return {
            "paperId": core.get("dc:identifier") or f"scopus:{doi}",
            "title": core.get("dc:title") or "",
            "abstract": core.get("dc:description"),
            "authors_str": core.get("dc:creator"),
            "num_authors": None,
            "year": year,
            "venue": core.get("prism:publicationName"),
            "citationCount": int(core.get("citedby-count") or 0),
            "referenceCount": None,
            "fieldsOfStudy": None,
            "doi": raw_doi,
            "doi_url": f"https://doi.org/{raw_doi}",
        }
