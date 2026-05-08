import argparse
import asyncio
import pathlib
import sys
from typing import Optional

# Ensure project root is on sys.path when script is run directly (python src/sentinel/cli.py)
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from src.sentinel.api.semantic_scholar_async import SemanticScholarAsyncClient
from src.sentinel.clusterer import Clusterer
from src.sentinel.harvester import Harvester
from src.sentinel.labeler import ClusterLabeler
from src.sentinel.layerer import Layerer
from src.sentinel.normalizer import Normalizer
from src.sentinel.scorer import Scorer
from src.sentinel.sentinel import Sentinel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="Scholargraph Data Integrity Sentinel — Phase 1 Harvester",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Citation crawl depth: hops from seed papers (default: 2)",
    )
    parser.add_argument(
        "--ranking",
        choices=["auto", "preloaded"],
        default="preloaded",
        help=(
            "Journal ranking source: "
            "auto=OpenAlex top-10%% citation percentile, "
            "preloaded=FT50/ABS/BehavEcon lists (default: preloaded)"
        ),
    )
    return parser.parse_args()


def prompt_seed() -> tuple[str, str]:
    print("\nSelect seed method:")
    print("  [1] Zotero Collection Key")
    print("  [2] DOI list (comma-separated)")
    print("  [3] Natural language query")

    while True:
        choice = input("Choice [1/2/3]: ").strip()
        if choice in ("1", "2", "3"):
            break
        print("  Please enter 1, 2, or 3.")

    prompts = {
        "1": ("zotero", "Enter Zotero collection key: "),
        "2": ("doi", "Enter DOIs (comma-separated): "),
        "3": ("query", "Enter search query: "),
    }
    method, prompt_text = prompts[choice]
    raw_input = input(prompt_text).strip()
    return method, raw_input


async def resolve_seeds(
    method: str,
    raw_input: str,
    s2_client: SemanticScholarAsyncClient,
) -> list[dict]:
    seeds: list[dict] = []

    if method == "zotero":
        from src.sentinel.api.zotero import ZoteroClient
        zot_papers = ZoteroClient().fetch_collection(raw_input)
        doi_ids = [f"DOI:{p['doi']}" for p in zot_papers if p.get("doi")]
        if doi_ids:
            batch = await s2_client.get_papers_batch(doi_ids)
            for paper in batch:
                pid = paper.get("paperId") if paper else None
                if pid:
                    seeds.append({"paper_id": pid, "title": paper.get("title")})

    elif method == "doi":
        dois = [d.strip() for d in raw_input.split(",") if d.strip()]
        doi_ids = [f"DOI:{doi}" for doi in dois]
        batch = await s2_client.get_papers_batch(doi_ids)
        for paper in batch:
            pid = paper.get("paperId") if paper else None
            if pid:
                seeds.append({"paper_id": pid, "title": paper.get("title")})

    elif method == "query":
        data = await s2_client._get(
            "/paper/search/bulk",
            {"query": raw_input, "fields": "paperId,title", "limit": "20"},
        )
        for item in (data.get("data") or []):
            pid = item.get("paperId")
            if pid:
                seeds.append({"paper_id": pid, "title": item.get("title")})

    return seeds


def main() -> None:
    args = parse_args()
    method, raw_input = prompt_seed()

    async def run() -> None:
        async with SemanticScholarAsyncClient() as s2:
            print(f"\nResolving seeds via [{method}]...")
            seeds = await resolve_seeds(method, raw_input, s2)
            if not seeds:
                print("No seed papers found. Exiting.")
                return
            print(f"Seeded with {len(seeds)} papers. Crawling {args.depth} hop(s)...")
            harvester = Harvester()
            papers, edges = await harvester.harvest(seeds, args.depth, s2)
            print(f"\nHarvested {len(papers)} papers, {len(edges)} edges (raw).")
            norm_papers, norm_edges = Normalizer().normalize(papers, edges)
            print(f"Normalized: {len(norm_papers)} papers, {len(norm_edges)} edges.")
            papers_path, edges_path = Normalizer().save(norm_papers, norm_edges)
            print(f"Output: {papers_path}")
            print(f"Output: {edges_path}")

            scored_papers = Scorer().score(norm_papers, norm_edges)
            Scorer().save(scored_papers)
            if scored_papers:
                top = scored_papers[0]
                print(
                    f"Scored: {len(scored_papers)} papers "
                    f"(top lci={top.get('local_citation_count', 0)}, "
                    f"trend={top.get('trend_score')})"
                )

            clustered_papers = Clusterer().cluster(scored_papers, norm_edges)
            Clusterer().save(clustered_papers)
            n_clusters = len(set(p["cluster_id"] for p in clustered_papers))
            print(f"Clustered: {len(clustered_papers)} papers → {n_clusters} clusters")

            tagged_papers = Layerer().tag(clustered_papers)
            Layerer().save(tagged_papers)
            labeled_summary = ClusterLabeler().label(tagged_papers)
            ClusterLabeler().save(labeled_summary)
            top = max(labeled_summary, key=lambda c: c["paper_count"], default=None)
            if top:
                print(f"Labeled {len(labeled_summary)} clusters (top: cluster-{top['cluster_id']} = '{top['label']}')")
            from collections import Counter as _Counter
            tag_counts = _Counter(p["layer_tag"] for p in tagged_papers)
            print(
                f"Tagged: {len(tagged_papers)} papers — "
                f"Spearhead={tag_counts['Spearhead']} "
                f"Foundational={tag_counts['Foundational']} "
                f"TheoryExtender={tag_counts['Theory Extender']} "
                f"Standard={tag_counts['Standard']}"
            )

            if args.ranking == "auto":
                from src.sentinel.api.openalex import OpenAlexClient
                async with OpenAlexClient() as oa:
                    tiered_papers = await Sentinel().tier(tagged_papers, "auto", oa)
            else:
                tiered_papers = await Sentinel().tier(tagged_papers, "preloaded")

            from collections import Counter
            tier_counts = Counter(p.get("tier") for p in tiered_papers)
            print(f"Tiers: high={tier_counts['high']}, mid={tier_counts['mid']}, low={tier_counts['low']}")
            metrics_path = Sentinel().save_journal_metrics(tiered_papers)
            print(f"Output: {metrics_path}")

            expected_count = len(seeds) * (2 ** args.depth)
            report = Sentinel().validate(tiered_papers, norm_edges, expected_count)
            report_path = Sentinel().save_report(report)
            print(
                f"Validation: duplicate_doi_rate={report['duplicate_doi_rate']} "
                f"high_meta={report['high_metadata_completeness']} "
                f"paper_count_ok={report['paper_count_ok']}"
            )
            print(f"Output: {report_path}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
