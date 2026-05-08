from src.sentinel.api.semantic_scholar_async import SemanticScholarAsyncClient


class Harvester:
    async def harvest(
        self,
        seeds: list[dict],
        crawl_depth: int,
        s2_client: SemanticScholarAsyncClient,
    ) -> tuple[list[dict], list[dict]]:
        seen_ids: set[str] = set()
        all_papers: list[dict] = []
        all_edges: list[dict] = []

        # Initialise with seed papers
        frontier: list[str] = []
        for seed in seeds:
            pid = seed.get("paper_id")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_papers.append(seed)
                frontier.append(pid)

        for _ in range(crawl_depth):
            if not frontier:
                break

            next_frontier: list[str] = []

            for paper_id in frontier:
                refs = await s2_client.get_references(paper_id)
                for ref in refs:
                    target = ref.get("target")
                    if not target:
                        continue
                    all_edges.append({"source": paper_id, "target": target})
                    if target not in seen_ids:
                        seen_ids.add(target)
                        next_frontier.append(target)

            if not next_frontier:
                break

            batch = await s2_client.get_papers_batch(next_frontier)
            for paper in batch:
                if not paper:
                    continue
                pid = paper.get("paperId")
                if not pid:
                    continue
                all_papers.append({
                    "paper_id": pid,
                    "title": paper.get("title"),
                    "doi": paper.get("doi"),
                    "year": paper.get("year"),
                    "citation_count": paper.get("citationCount"),
                    "venue": paper.get("venue"),
                })

            frontier = next_frontier

        return all_papers, all_edges
