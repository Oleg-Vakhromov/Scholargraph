import re
from typing import Dict, List, Optional

import pandas as pd


def _normalize(name: str) -> str:
    """Lowercase, strip non-alphanumeric chars, collapse spaces."""
    return re.sub(r' +', ' ', re.sub(r'[^a-z0-9 ]', '', name.lower().strip())).strip()


class ScimagoLoader:
    """
    Loads a SCImago journal rankings CSV and maps paper venues to quartiles.

    The SCImago CSV is semicolon-separated and can be downloaded from
    scimagojr.com/journalrank.php (select year → Download → CSV).
    Expected columns: Title, SJR Best Quartile.

    Usage:
        sl = ScimagoLoader()
        sl.load("data/scimago.csv")
        papers_df = sl.enrich_papers(papers_df)
        quartiles = sl.available_quartiles(papers_df)
    """

    def __init__(self) -> None:
        self._lookup: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: str) -> bool:
        """
        Load SCImago CSV and build normalized name → quartile lookup.

        Args:
            path: Path to the SCImago CSV file (semicolon-separated).

        Returns:
            True if loaded successfully, False if file not found or unreadable.
        """
        try:
            df = pd.read_csv(path, sep=";", encoding="utf-8", dtype=str)
        except FileNotFoundError:
            self._lookup = {}
            return False
        except Exception:
            self._lookup = {}
            return False

        # Normalise column names (strip whitespace)
        df.columns = [c.strip() for c in df.columns]

        if "Title" not in df.columns or "SJR Best Quartile" not in df.columns:
            self._lookup = {}
            return False

        valid_quartiles = {"Q1", "Q2", "Q3", "Q4"}
        lookup: Dict[str, str] = {}

        for _, row in df.iterrows():
            title = row.get("Title")
            quartile = row.get("SJR Best Quartile")
            if not isinstance(title, str) or not isinstance(quartile, str):
                continue
            quartile = quartile.strip()
            if quartile not in valid_quartiles:
                continue
            key = _normalize(title)
            if key:
                lookup[key] = quartile

        self._lookup = lookup
        return bool(lookup)

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def enrich_papers(self, papers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a `scimago_quartile` column to a copy of papers_df.

        Maps each paper's `journal` field through the normalized lookup.
        Papers with no journal or unmatched journal receive None.

        Args:
            papers_df: Corpus paper DataFrame (journal column expected).

        Returns:
            New DataFrame with `scimago_quartile` column (str or None).
            Returns copy with all-None column if loader not loaded or df empty.
        """
        df = papers_df.copy()

        if not self._lookup or df.empty or "journal" not in df.columns:
            df["scimago_quartile"] = None
            return df

        def _lookup_quartile(journal) -> Optional[str]:
            if not isinstance(journal, str) or not journal.strip():
                return None
            return self._lookup.get(_normalize(journal))

        df["scimago_quartile"] = df["journal"].apply(_lookup_quartile)
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def available_quartiles(self, papers_df: pd.DataFrame) -> List[str]:
        """
        Return sorted list of non-None unique quartile values in papers_df.

        Args:
            papers_df: DataFrame with `scimago_quartile` column.

        Returns:
            Sorted list e.g. ["Q1", "Q2", "Q3"]. Empty list if column absent.
        """
        if "scimago_quartile" not in papers_df.columns:
            return []
        values = papers_df["scimago_quartile"].dropna().unique().tolist()
        return sorted(v for v in values if v in {"Q1", "Q2", "Q3", "Q4"})

    @property
    def is_loaded(self) -> bool:
        """True if the lookup table was successfully populated."""
        return bool(self._lookup)
