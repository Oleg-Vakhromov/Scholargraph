import re
import time
import requests
from typing import Optional


_CROSSREF_API = "https://api.crossref.org/works"


class CrossRefClient:
    """
    Sync CrossRef REST client for DOI and venue back-fill.

    Uses the polite pool (mailto header) for higher rate limits.
    lookup() returns the raw CrossRef work dict on a good title match,
    None otherwise — callers can extract DOI, container-title, etc.
    """

    def __init__(
        self,
        mailto: str = "oleg.vakhromov@gmail.com",
        rate_limit: float = 0.1,
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

    def _titles_match(self, a: str, b: str) -> bool:
        na = self._normalize_title(a)
        nb = self._normalize_title(b)
        if na == nb:
            return True
        words_a = set(na.split())
        words_b = set(nb.split())
        if not words_a or not words_b:
            return False
        jaccard = len(words_a & words_b) / len(words_a | words_b)
        return jaccard >= 0.85

    def lookup_by_doi(self, doi: str) -> Optional[dict]:
        """
        Fetch the CrossRef work record for a known DOI.

        Returns the raw CrossRef work dict (contains container-title, DOI,
        title, etc.) on HTTP 200, otherwise None.

        Deterministic — no title matching needed.
        Never raises — all exceptions are caught and return None.
        """
        if not doi or not doi.strip():
            return None

        try:
            self._wait_for_rate_limit()
            url = f"{_CROSSREF_API}/{doi.strip()}"
            headers = {
                "User-Agent": f"Bibliometrics/1.1 (mailto:{self._mailto})",
            }
            response = requests.get(url, headers=headers, timeout=10)
            self._last_request_time = time.time()

            if not response.ok:
                return None

            return response.json().get("message")

        except Exception:
            return None

    def lookup(self, title: str, year: Optional[int] = None) -> Optional[dict]:
        """
        Query CrossRef for the best-matching work by title.

        Returns the raw CrossRef work dict (contains DOI, title,
        container-title, published, etc.) if the top result passes
        title similarity validation, otherwise None.

        Never raises — all exceptions are caught and return None.
        """
        if not title or not title.strip():
            return None

        try:
            self._wait_for_rate_limit()
            params = {
                "query.bibliographic": title,
                "rows": 3,
                "select": "DOI,title,container-title,published",
            }
            headers = {
                "User-Agent": f"Bibliometrics/1.1 (mailto:{self._mailto})",
            }
            response = requests.get(
                _CROSSREF_API, params=params, headers=headers, timeout=10
            )
            self._last_request_time = time.time()

            if not response.ok:
                return None

            items = response.json().get("message", {}).get("items", [])
            if not items:
                return None

            top = items[0]
            candidate_titles = top.get("title") or []
            candidate_title = candidate_titles[0] if candidate_titles else ""

            if not self._titles_match(title, candidate_title):
                return None

            return top

        except Exception:
            return None
