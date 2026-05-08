import asyncio
import re
import time
from typing import Optional

import aiohttp

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_RATE_LIMIT_INTERVAL = 1.0  # 1 req/sec unauthenticated tier
_BATCH_SIZE = 500
_DOI_URL_PREFIXES = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.IGNORECASE)


class SemanticScholarAsyncClient:
    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time: float = 0.0

    async def __aenter__(self) -> "SemanticScholarAsyncClient":
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
        delay = 60
        for attempt in range(4):  # matches existing sync client: 60s → 120s → 240s
            await self._wait_for_rate_limit()
            async with self._session.get(f"{_BASE_URL}{path}", params=params) as response:
                self._last_request_time = time.monotonic()
                if response.status != 429:
                    if not response.ok:
                        raise RuntimeError(
                            f"GET {path} failed [{response.status}]: {await response.text()}"
                        )
                    return await response.json()
            if attempt < 3:
                await asyncio.sleep(delay)
                delay *= 2

        raise RuntimeError(f"GET {path} failed after 4 attempts: 429")

    async def _post(self, path: str, params: dict, body: dict) -> list:
        delay = 60
        for attempt in range(4):
            await self._wait_for_rate_limit()
            async with self._session.post(
                f"{_BASE_URL}{path}", params=params, json=body
            ) as response:
                self._last_request_time = time.monotonic()
                if response.status != 429:
                    if not response.ok:
                        raise RuntimeError(
                            f"POST {path} failed [{response.status}]: {await response.text()}"
                        )
                    return await response.json()
            if attempt < 3:
                await asyncio.sleep(delay)
                delay *= 2

        raise RuntimeError(f"POST {path} failed after 4 attempts: 429")

    @staticmethod
    def _normalize_doi(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        normalized = _DOI_URL_PREFIXES.sub("", raw).strip().rstrip(".")
        return normalized if normalized else None

    async def get_references(self, paper_id: str) -> list[dict]:
        if not paper_id:
            return []

        citations: list[dict] = []
        offset = 0

        while True:
            data = await self._get(
                f"/paper/{paper_id}/references",
                {"fields": "paperId,title,year", "limit": 1000, "offset": offset},
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

        return citations

    async def get_papers_batch(
        self,
        paper_ids: list[str],
        fields: Optional[str] = None,
    ) -> list[dict]:
        if fields is None:
            fields = (
                "paperId,title,abstract,authors,year,journal,publicationVenue,venue,"
                "externalIds,citationCount,referenceCount,fieldsOfStudy"
            )

        results: dict[str, dict] = {}

        for i in range(0, len(paper_ids), _BATCH_SIZE):
            chunk = paper_ids[i : i + _BATCH_SIZE]
            batch = await self._post(
                "/paper/batch",
                params={"fields": fields},
                body={"ids": chunk},
            )

            for paper in batch:
                if paper is None:
                    continue

                raw_authors = paper.get("authors") or []
                paper["authors_str"] = (
                    ", ".join(a.get("name", "") for a in raw_authors if a.get("name")) or None
                )
                paper["num_authors"] = len(raw_authors) if raw_authors else None

                _journal = paper.get("journal") or {}
                _pub_venue = paper.get("publicationVenue") or {}
                _legacy_venue = paper.get("venue")
                paper["venue"] = (
                    _journal.get("name") or _pub_venue.get("name") or _legacy_venue or None
                )

                ext_ids = paper.get("externalIds") or {}
                raw_doi = ext_ids.get("DOI") or ext_ids.get("doi") or None
                normalized_doi = self._normalize_doi(raw_doi)
                paper["doi"] = normalized_doi
                paper["doi_url"] = f"https://doi.org/{normalized_doi}" if normalized_doi else None

                pid = paper.get("paperId")
                if pid:
                    results[pid] = paper

        return [results.get(pid, {}) for pid in paper_ids]

    async def search_match(
        self,
        title: str,
        fields: Optional[str] = None,
    ) -> Optional[dict]:
        if not title:
            return None

        if fields is None:
            fields = "paperId,title,journal,publicationVenue,venue,externalIds"

        try:
            data = await self._get(
                "/paper/search/match",
                {"query": title, "fields": fields},
            )
        except RuntimeError:
            return None

        return data if data.get("paperId") else None
