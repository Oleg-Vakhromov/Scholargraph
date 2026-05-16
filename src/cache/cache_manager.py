import json
import os
import time
from typing import Optional

_CONSOLIDATED_FILE = "semantic_scholar_cache.json"
_LEGACY_FILES = ("papers.json", "references.json", "incoming_citations.json")


class DiskCache:
    """
    Two-bucket disk-backed cache for Semantic Scholar API responses.

    Buckets:
        _papers              — paperId → paper metadata dict
        _references          — paperId → list of reference dicts
        _incoming_citations  — paperId → list of incoming citation dicts

    Persistence:
        Single consolidated file: cache/semantic_scholar_cache.json
        Written atomically via temp-file rename (Windows/OneDrive-safe).

    Lifecycle:
        cache = DiskCache(cache_dir="cache")
        cache.load()   # call once at startup — silently skips if files absent
        ...
        cache.save()   # call after each batch to persist
    """

    def __init__(self, cache_dir: str = "cache") -> None:
        self._cache_dir = cache_dir
        self._papers: dict = {}
        self._references: dict = {}
        self._incoming_citations: dict = {}

    # ------------------------------------------------------------------
    # Paper cache
    # ------------------------------------------------------------------

    def has_paper(self, paper_id: str) -> bool:
        return paper_id in self._papers

    def get_paper(self, paper_id: str) -> Optional[dict]:
        return self._papers.get(paper_id)

    def set_paper(self, paper_id: str, data: dict) -> None:
        self._papers[paper_id] = data

    # ------------------------------------------------------------------
    # Reference cache
    # ------------------------------------------------------------------

    def has_references(self, paper_id: str) -> bool:
        return paper_id in self._references

    def get_references(self, paper_id: str) -> Optional[list]:
        return self._references.get(paper_id)

    def set_references(self, paper_id: str, refs: list) -> None:
        self._references[paper_id] = refs

    # ------------------------------------------------------------------
    # Incoming citations cache
    # ------------------------------------------------------------------

    def has_incoming_citations(self, paper_id: str) -> bool:
        return paper_id in self._incoming_citations

    def get_incoming_citations(self, paper_id: str) -> Optional[list]:
        return self._incoming_citations.get(paper_id)

    def set_incoming_citations(self, paper_id: str, citations: list) -> None:
        self._incoming_citations[paper_id] = citations

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load cache from disk.

        Reads from the consolidated semantic_scholar_cache.json if present.
        Falls back to legacy separate files (papers.json, references.json,
        incoming_citations.json) for backward compatibility — migrates them
        into memory but does NOT write the old files on save().
        Silently skips if no cache files are found.
        """
        consolidated_path = os.path.join(self._cache_dir, _CONSOLIDATED_FILE)

        if os.path.exists(consolidated_path):
            with open(consolidated_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._papers = data.get("papers", {})
            self._references = data.get("references", {})
            self._incoming_citations = data.get("incoming_citations", {})
            return

        # Legacy fallback: load separate files and migrate into memory
        papers_path = os.path.join(self._cache_dir, "papers.json")
        refs_path = os.path.join(self._cache_dir, "references.json")
        incoming_path = os.path.join(self._cache_dir, "incoming_citations.json")

        if os.path.exists(papers_path):
            with open(papers_path, "r", encoding="utf-8") as f:
                self._papers = json.load(f)

        if os.path.exists(refs_path):
            with open(refs_path, "r", encoding="utf-8") as f:
                self._references = json.load(f)

        if os.path.exists(incoming_path):
            with open(incoming_path, "r", encoding="utf-8") as f:
                self._incoming_citations = json.load(f)

    def save(self) -> None:
        """
        Persist all buckets to a single consolidated JSON file atomically.

        Writes to a temp file first, then renames to the target path.
        Retries up to 3 times (1s delay each) on transient OS lock errors
        (PermissionError / OSError) before raising.
        """
        os.makedirs(self._cache_dir, exist_ok=True)

        target_path = os.path.join(self._cache_dir, _CONSOLIDATED_FILE)
        tmp_path = target_path + ".tmp"

        data = {
            "papers": self._papers,
            "references": self._references,
            "incoming_citations": self._incoming_citations,
        }

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                os.replace(tmp_path, target_path)
                return
            except (PermissionError, OSError):
                if attempt < max_retries:
                    time.sleep(1)
                else:
                    raise

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "papers": len(self._papers),
            "references": len(self._references),
            "incoming_citations": len(self._incoming_citations),
        }
