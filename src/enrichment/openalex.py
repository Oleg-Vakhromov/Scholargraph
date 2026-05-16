import re
import time
import requests
from typing import Optional


_OA_API = "https://api.openalex.org"


class OpenAlexReferenceClient:
    """
    Sync OpenAlex REST client for fetching a paper's reference list.

    Uses the polite pool (mailto param) for higher rate limits.
    get_references() returns a list of {doi, title, year} dicts for
    referenced works that have a DOI — never raises.
    """

    def __init__(
        self,
        mailto: str = "oleg.vakhromov@gmail.com",
        rate_limit: float = 0.05,
    ) -> None:
        self._mailto = mailto
        self._rate_limit = rate_limit
        self._last_request_time: float = 0.0

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)

    @staticmethod
    def _normalize_title(title: str) -> str:
        t = title.lower()
        t = re.sub(r"[^a-z0-9\s]", "", t)
        return re.sub(r"\s+", " ", t).strip()

    def _titles_match(self, a: str, b: str, threshold: float = 0.7) -> bool:
        na = self._normalize_title(a)
        nb = self._normalize_title(b)
        if na == nb:
            return True
        words_a = set(na.split())
        words_b = set(nb.split())
        if not words_a or not words_b:
            return False
        jaccard = len(words_a & words_b) / len(words_a | words_b)
        return jaccard >= threshold

    def _lookup_work(
        self, doi: Optional[str] = None, title: Optional[str] = None
    ) -> Optional[dict]:
        """Fetch a work from OpenAlex, returning the raw work dict or None."""
        try:
            self._wait_for_rate_limit()
            if doi and doi.strip():
                bare_doi = doi.strip().lstrip("https://doi.org/").lstrip("http://doi.org/")
                url = f"{_OA_API}/works/doi:{bare_doi}"
                params = {"select": "referenced_works", "mailto": self._mailto}
                response = requests.get(url, params=params, timeout=10)
                self._last_request_time = time.time()
                if not response.ok:
                    return None
                return response.json()

            elif title and title.strip():
                params = {
                    "search": title.strip(),
                    "per_page": 1,
                    "select": "title,referenced_works",
                    "mailto": self._mailto,
                }
                response = requests.get(
                    f"{_OA_API}/works", params=params, timeout=10
                )
                self._last_request_time = time.time()
                if not response.ok:
                    return None
                results = response.json().get("results") or []
                if not results:
                    return None
                top = results[0]
                candidate_titles = top.get("title") or ""
                if not self._titles_match(title, candidate_titles or ""):
                    return None
                return top

        except Exception:
            return None

        return None

    def _resolve_works(self, oa_ids: list) -> list:
        """
        Batch-fetch doi/title/year for a list of OpenAlex work IDs.

        Chunks to 50 per request. Returns list of {doi, title, year} for
        items that have a non-empty DOI.
        """
        results = []
        bare_ids = []
        for raw_id in oa_ids:
            if not raw_id:
                continue
            # "https://openalex.org/W123" → "W123"
            bare = raw_id.rstrip("/").split("/")[-1]
            if bare:
                bare_ids.append(bare)

        chunk_size = 50
        for i in range(0, len(bare_ids), chunk_size):
            chunk = bare_ids[i : i + chunk_size]
            filter_val = "|".join(chunk)
            try:
                self._wait_for_rate_limit()
                params = {
                    "filter": f"ids.openalex:{filter_val}",
                    "select": "doi,title,publication_year",
                    "per_page": chunk_size,
                    "mailto": self._mailto,
                }
                response = requests.get(
                    f"{_OA_API}/works", params=params, timeout=10
                )
                self._last_request_time = time.time()
                if not response.ok:
                    continue
                for item in response.json().get("results") or []:
                    raw_doi = item.get("doi") or ""
                    if not raw_doi:
                        continue
                    # Strip https://doi.org/ prefix and normalize
                    doi = re.sub(r"^https?://doi\.org/", "", raw_doi).strip().lower()
                    if not doi:
                        continue
                    title_val = item.get("title") or ""
                    year_val = item.get("publication_year")
                    results.append({"doi": doi, "title": title_val, "year": year_val})
            except Exception:
                continue

        return results

    def get_references(
        self, doi: Optional[str] = None, title: Optional[str] = None
    ) -> list:
        """
        Fetch the reference list for a paper identified by DOI or title.

        Returns a list of dicts: {"doi": str, "title": str, "year": Optional[int]}
        Only includes references that have a DOI.
        Never raises — returns [] on any error or when nothing found.
        """
        if not doi and not title:
            return []

        try:
            work = self._lookup_work(doi=doi, title=title)
            if work is None:
                return []
            referenced_works = work.get("referenced_works") or []
            if not referenced_works:
                return []
            return self._resolve_works(referenced_works)
        except Exception:
            return []
