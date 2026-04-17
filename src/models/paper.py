from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: Optional[str]
    authors: Optional[str]           # comma-separated author names
    num_authors: Optional[int]
    year: Optional[int]
    journal: Optional[str]
    citation_count: int
    reference_count: Optional[int]
    fields_of_study: Optional[List[str]]
    doi: Optional[str]               # bare DOI token, e.g. "10.1234/example"
    doi_url: Optional[str]           # full URL, e.g. "https://doi.org/10.1234/example"

    @classmethod
    def from_api_dict(cls, d: dict) -> Paper:
        """
        Construct a Paper from a Semantic Scholar API response dict.

        Handles camelCase → snake_case mapping and missing/None fields gracefully.
        Expects `authors_str` and `num_authors` keys as pre-processed by the client.
        Expects `doi` and `doi_url` keys pre-processed by the client from externalIds.
        """
        return cls(
            paper_id=d.get("paperId") or "",
            title=d.get("title") or "",
            abstract=d.get("abstract"),
            authors=d.get("authors_str"),
            num_authors=d.get("num_authors"),
            year=d.get("year"),
            journal=d.get("venue"),
            citation_count=d.get("citationCount") or 0,
            reference_count=d.get("referenceCount"),
            fields_of_study=d.get("fieldsOfStudy"),
            doi=d.get("doi"),
            doi_url=d.get("doi_url"),
        )


@dataclass
class Citation:
    source: str           # citing paperId
    target: str           # cited paperId
    title: Optional[str]
    year: Optional[int]

    @classmethod
    def from_api_dict(cls, d: dict) -> Citation:
        return cls(
            source=d.get("source") or "",
            target=d.get("target") or "",
            title=d.get("title"),
            year=d.get("year"),
        )


def papers_to_df(papers: List[Paper]) -> pd.DataFrame:
    """
    Convert a list of Paper dataclasses to a pandas DataFrame.

    Columns (in order):
        paper_id, title, abstract, authors, num_authors, year, journal,
        citation_count, reference_count, fields_of_study, doi, doi_url

    dtypes:
        year            → Int64  (pandas nullable integer — supports NaN)
        citation_count  → int64
        all others      → object
    """
    records = [
        {
            "paper_id": p.paper_id,
            "title": p.title,
            "abstract": p.abstract,
            "authors": p.authors,
            "num_authors": p.num_authors,
            "year": p.year,
            "journal": p.journal,
            "citation_count": p.citation_count,
            "reference_count": p.reference_count,
            "fields_of_study": p.fields_of_study,
            "doi": p.doi,
            "doi_url": p.doi_url,
        }
        for p in papers
    ]

    df = pd.DataFrame(
        records,
        columns=[
            "paper_id", "title", "abstract", "authors", "num_authors",
            "year", "journal", "citation_count", "reference_count",
            "fields_of_study", "doi", "doi_url",
        ],
    )

    df["year"] = df["year"].astype("Int64")
    df["citation_count"] = df["citation_count"].astype("int64")

    return df


def citations_to_df(citations: List[Citation]) -> pd.DataFrame:
    """
    Convert a list of Citation dataclasses to a pandas DataFrame.

    Columns: source, target, title, year
    dtypes: year → Int64 (nullable integer)
    """
    records = [
        {
            "source": c.source,
            "target": c.target,
            "title": c.title,
            "year": c.year,
        }
        for c in citations
    ]

    df = pd.DataFrame(records, columns=["source", "target", "title", "year"])
    df["year"] = df["year"].astype("Int64")
    return df
