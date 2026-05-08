import os
import re
from typing import Optional

from pyzotero import zotero

_DOI_PREFIX = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.IGNORECASE)


class ZoteroClient:
    def __init__(self) -> None:
        user_id = os.environ.get("ZOTERO_USER_ID")
        api_key = os.environ.get("ZOTERO_API_KEY")
        if not user_id or not api_key:
            raise RuntimeError("ZOTERO_USER_ID and ZOTERO_API_KEY env vars required")
        self._zot = zotero.Zotero(user_id, "user", api_key)

    @staticmethod
    def _normalize_doi(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        normalized = _DOI_PREFIX.sub("", raw).strip().rstrip(".")
        return normalized if normalized else None

    def fetch_collection(self, collection_key: str) -> list[dict]:
        items = self._zot.collection_items(collection_key)
        results: list[dict] = []
        for item in items:
            data = item.get("data", {})
            if data.get("itemType") in ("attachment", "note"):
                continue
            title = data.get("title")
            if not title:
                continue
            doi = self._normalize_doi(data.get("DOI"))
            year: Optional[int] = None
            raw_date = data.get("date", "")
            if raw_date:
                try:
                    year = int(str(raw_date)[:4])
                except ValueError:
                    pass
            results.append({"title": title, "doi": doi, "year": year})
        return results
