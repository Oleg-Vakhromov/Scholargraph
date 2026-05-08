import asyncio
import os
import re
import time
from typing import Optional

import aiohttp

_BASE_URL = "https://api.openalex.org"
_RATE_LIMIT_INTERVAL = 0.1  # 10 req/sec polite pool
_DOI_PREFIX = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.IGNORECASE)


class OpenAlexClient:
    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time: float = 0.0
        self._email: Optional[str] = os.getenv("OPENALEX_EMAIL")

    async def __aenter__(self) -> "OpenAlexClient":
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args) -> None:
        if self._session:
            await self._session.close()

    async def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _RATE_LIMIT_INTERVAL:
            await asyncio.sleep(_RATE_LIMIT_INTERVAL - elapsed)

    async def _get(self, path: str, params: dict) -> dict:
        if self._email:
            params = {**params, "mailto": self._email}

        delay = 1
        last_status: Optional[int] = None
        for attempt in range(4):  # initial + 3 retries (delays: 1s → 2s → 4s)
            await self._wait_for_rate_limit()
            async with self._session.get(f"{_BASE_URL}{path}", params=params) as response:
                self._last_request_time = time.monotonic()
                if response.status != 429 and response.status < 500:
                    if not response.ok:
                        raise RuntimeError(
                            f"OpenAlex GET {path} failed [{response.status}]: {await response.text()}"
                        )
                    return await response.json()
                last_status = response.status
            if attempt < 3:
                await asyncio.sleep(delay)
                delay *= 2

        raise RuntimeError(f"OpenAlex GET {path} failed after 3 retries: {last_status}")

    @staticmethod
    def _normalize_doi(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        normalized = _DOI_PREFIX.sub("", raw).strip().rstrip(".")
        return normalized if normalized else None

    @staticmethod
    def _extract_journal(work: dict) -> Optional[str]:
        try:
            return work["primary_location"]["source"]["display_name"]
        except (KeyError, TypeError):
            return None

    async def search_papers(self, query: str, limit: int = 100) -> list[dict]:
        results: list[dict] = []
        params: dict = {
            "search": query,
            "per-page": min(25, limit),
            "select": "id,doi,title,publication_year,cited_by_count,primary_location,cited_by_percentile_year",
            "cursor": "*",
        }

        while len(results) < limit:
            data = await self._get("/works", params)
            for work in data.get("results", []):
                percentile_data = work.get("cited_by_percentile_year") or {}
                results.append({
                    "openalex_id": work.get("id"),
                    "title": work.get("title"),
                    "doi": self._normalize_doi(work.get("doi")),
                    "year": work.get("publication_year"),
                    "citation_count": work.get("cited_by_count", 0),
                    "citation_percentile": percentile_data.get("max"),
                    "journal": self._extract_journal(work),
                })
            meta = data.get("meta", {})
            next_cursor = meta.get("next_cursor")
            if not next_cursor or not data.get("results"):
                break
            params["cursor"] = next_cursor

        return results[:limit]

    async def get_citation_percentile(self, openalex_id: str) -> Optional[float]:
        clean_id = openalex_id.replace("https://openalex.org/", "")
        data = await self._get(f"/works/{clean_id}", {"select": "cited_by_percentile_year"})
        percentile = data.get("cited_by_percentile_year")
        if isinstance(percentile, dict):
            return percentile.get("max")
        return None
