# Scholargraph — User Guide

Scholargraph is a literature mapping tool for academic researchers. You supply a keyword query; the tool builds a structured corpus from Semantic Scholar, maps citation relationships, detects research clusters, and produces export-ready outputs for further analysis in Gephi, Zotero, or R/Python.

The core design premise is that a keyword search retrieves papers ranked by platform relevance, not by their structural role in the literature. Scholargraph treats the keyword results as a starting point — a seed — and then follows citation links outward, using PageRank and semantic similarity to surface papers that sit at the structural core of the literature but would never appear on page one of a keyword search. The output is a citation graph with cluster labels, influence metrics, and temporal trends, not a ranked list.

---

## How the Search Works

The pipeline runs in two phases: **seeding** and **expansion**. Understanding both phases helps you choose the right parameter settings for your research question.

### Phase 1 — Seeding

When you click **▶ Run Pipeline**, Scholargraph sends your query to the Semantic Scholar bulk search endpoint. This returns up to **Seed limit** papers ranked by Semantic Scholar's default relevance ordering. Those papers are enriched with metadata (abstract, authors, venue, DOI, fields of study) via a batch API call.

At this point the corpus is a flat keyword-search result — the same papers you would get from searching Semantic Scholar directly, minus the web interface.

**What you control in this phase:**

- **Query** — free-text keyword search. Supports quoted phrases (e.g., `"meme stocks"`), plain keywords, or a combination. The query is also used later as the reference embedding for relevance filtering during expansion.
- **Seed limit** — how many papers to retrieve from the initial keyword search (10–2000, default: 200). A larger seed gives broader initial coverage but increases runtime. For exploratory runs, 50–100 is usually sufficient.
- **Year from / Year to + Apply year filter** — restricts the keyword search results to a publication year window. The checkbox must be enabled for the year bounds to take effect. Year filtering applies only to the seed; expansion may pull in papers outside the range if they are structurally well-connected.

After seeding, the app pauses at the **Domain Picker** before continuing.

### Phase 1b — Domain and Journal Filtering

Before expansion begins, you can narrow the corpus along two dimensions.

**Domain filter** — Semantic Scholar tags papers with fields of study (e.g., Economics, Computer Science, Psychology). The domain picker lists all fields found in the seed papers. Uncheck any that are off-topic. Papers not matching any selected field are dropped. If no field-of-study data is available for the seed, this step is skipped automatically.

**Journal quartile filter** — when SCImago data is available, a second filter lets you restrict the corpus to journals in selected SCImago quartiles (Q1, Q2, Q3, Q4). Papers with no SCImago quartile data are excluded when any quartile is selected. Use this if you want to restrict the corpus to peer-reviewed journals of a particular standing.

**Max seed papers (after domain filter)** — caps the corpus size after domain filtering (10–2000, default: 200). If more papers survive the domain filter than this cap, the lowest-cited papers are dropped first. This controls corpus size before expansion begins.

Clicking **Confirm domains & continue** triggers the rest of the pipeline.

### Phase 2 — Expansion

Expansion iteratively adds papers that are structurally important to the corpus but were not returned by the keyword search. Each iteration follows this sequence:

1. Build a citation graph from all corpus papers and their references.
2. Rank papers that are referenced by the corpus but not yet in it (candidates) using the selected strategy.
3. Filter candidates that do not meet the minimum in-corpus citation threshold.
4. Optionally filter candidates by cosine similarity to the original query (relevance filter).
5. Fetch full metadata and references for papers that pass all filters.
6. Add passing papers to the corpus; deduplicate.
7. Repeat until the iteration cap is reached, no candidates remain, or no new papers were added.

**What you control in this phase:**

**Ranking strategy** — determines how candidates are scored and selected:

- **PageRank (graph-based)** (default) — computes PageRank over the full citation neighbourhood, including papers not yet in the corpus. A high PageRank score means the paper is structurally central to this specific literature: many paths in the citation network pass through it, regardless of its global citation count. Candidates must exceed the 75th-percentile PageRank threshold to qualify.
- **In-sample citation count** — ranks candidates by how many corpus papers already cite them. This is the classic bibliometric approach — papers most cited within your corpus are added first. Candidates must be cited by at least max(3, 75th-percentile ISC) corpus papers to qualify.

Use PageRank when you want to surface papers that bridge sub-fields or are structurally important but may not be the most-cited in any single cluster. Use in-sample citation count when you want the expansion to closely track what your existing corpus is already citing.

**Max iterations** — hard cap on expansion passes (1–20, default: 3). More iterations produce a larger, denser corpus. Three passes is a reasonable starting point for a corpus of 200 seed papers. The loop stops earlier if convergence is reached (no qualifying candidates, no new papers added).

**Min. in-corpus citations** — minimum number of corpus papers that must already cite a candidate for it to qualify for expansion (1–20, default: 1). Raising this to 2–3 makes the expansion more conservative: only papers that multiple corpus papers independently reference will be added, reducing the risk of pulling in tangential work.

**Apply relevance filter** — when enabled (default), each candidate's title and abstract are embedded using the selected sentence-transformer model and compared against the original query by cosine similarity. Candidates below the **Relevance threshold** are discarded. Disable this if you want purely structure-driven expansion — every candidate that meets the citation threshold is added regardless of topical similarity to the query. Useful when the query is narrow but the literature spans multiple terminologies.

**Relevance threshold** — cosine similarity cutoff (0.0–1.0, default: 0.3). This controls how strictly candidates must match the query embedding. 0.3 is permissive and appropriate for broad queries. Raise to 0.5–0.7 if the corpus drifts into adjacent fields; lower toward 0.1–0.2 if expansion is too conservative and stalls.

**Embedding model** — the sentence-transformer model used to embed paper text during relevance filtering:

- **Fast (~90 MB)** — `all-MiniLM-L6-v2`. Default. Adequate for most English-language academic queries.
- **Balanced / High quality (~420 MB)** — `all-mpnet-base-v2`. Higher semantic accuracy; noticeably slower to load on first use.
- **Long-context multilingual (~2.2 GB)** — `BAAI/bge-m3`. Best for multilingual corpora or queries with long abstracts. Large download on first use.

Models are downloaded and cached locally on first use. Switching models during a session loads the new model from cache without re-downloading.

---

## Prerequisites

- Python 3.10 or higher
- pip
- Internet access (Semantic Scholar API + first-run model download)

---

## Installation

1. Download or clone the repository into a local folder.
2. Open a terminal in that folder and install dependencies:

```
pip install -r requirements.txt
```

---

## Running the App

```
python -m streamlit run app.py
```

Streamlit opens the app at `http://localhost:8501`. The page loads immediately; the pipeline only runs when you click **▶ Run Pipeline**.

---

## Quick Start

1. Enter a keyword query in the **Query** field (e.g., `retail investor sentiment`).
2. Leave all other settings at their defaults.
3. Click **▶ Run Pipeline**.
4. After seeding completes, a **domain picker** appears. Deselect any irrelevant fields of study, then click **Confirm domains & continue**.

Results appear below once the full pipeline finishes. For a fast test run, set **Seed limit** to 50 before clicking Run.

---

## Results Panels

**Metrics row**
Total papers in the corpus and total citation edges — a quick sanity check before diving into the details.

**Papers table**
All corpus papers with title, year, journal, citation count, authors, and DOI. Sorted by citation count descending by default.

**Influential Papers**
Papers ranked by in-corpus influence metrics:

- **isc** — in-sample citation count: how many corpus papers cite this paper
- **isc_ratio** — isc as a percentage of total Semantic Scholar citations (capped at 100 %). A high ratio means this paper is especially central to this specific literature relative to its global prominence
- **sample_relevance** — isc as a percentage of total corpus size: what fraction of the corpus cites this paper
- **betweenness_centrality** — how often this paper lies on the shortest path between other papers in the citation graph; high betweenness identifies papers that bridge otherwise-disconnected research clusters
- **DOI** — clickable link to the publisher page when available

A **⬇ Download influential papers** button exports the current view to CSV.

**Graph Display Settings**
Shared controls for all three network graphs below:

- **Node size metric** — selects which value drives node size: `isc` (default), `citation_count`, `isc_ratio`, `sample_relevance`, or `betweenness_centrality`
- **＋ Size / － Size** — scale all nodes up or down proportionally

**Co-citation Analysis**
Papers that are frequently cited together by the same sources cluster together here, even without a direct citation link between them. Controls: number of clusters, minimum co-citation count for an edge to appear.

**Bibliographic Coupling**
Papers that cite the same sources share intellectual foundations. This network connects papers by shared references. Controls: number of clusters, minimum shared references for an edge to appear.

**Clusters**
Research clusters detected via Louvain community detection on the citation graph. Each cluster is labeled by its most frequent non-stopword title words. A single partition is computed for the entire corpus and reused consistently across the cluster summary, graph colouring, and temporal chart — re-running community detection per year would produce inconsistent cluster IDs.

**Knowledge Graph**
Interactive citation network of the corpus. Nodes are papers; edges are citation links. The **Max nodes** slider limits the display to the top N papers by citation count.

Each of the three network graphs has two buttons:
- **Open in new tab ↗** — opens a self-contained HTML version in a new browser tab
- **⬇ Export view as CSV** — downloads the currently visible edges with columns `cluster_from`, `from_paper_name`, `cluster_to`, `to_paper_name`

**Temporal Evolution**
A line chart showing papers per cluster published each year, using the single corpus-wide Louvain partition. Use this to identify which research streams are growing, stable, or declining, and when new streams emerged.

**Export**
Four download formats:
- **papers.csv** — full corpus metadata (one row per paper)
- **citations.csv** — citation edge list (source → target)
- **graph.graphml** — citation graph compatible with Gephi, Cytoscape, and NetworkX
- **papers.bib** — BibTeX export. Each entry carries a `keywords` field in the form `cluster:<label>` (e.g., `cluster:behavioral finance`). Import into Zotero and filter by keyword to recreate the cluster structure as Zotero collections.

---

## Tips

**Start with a small run.** Set Seed limit to 50 and Max iterations to 1 before committing to a full run. A full pipeline with 200 seed papers and 3 iterations can take several minutes — most of that time is rate-limited API calls, not computation.

**Use the domain picker to focus the corpus before expansion.** Dropping off-topic fields at the domain filtering step prevents irrelevant papers from proliferating through PageRank during expansion. For a query like `retail investor sentiment`, deselect Computer Science or Mathematics if they appear.

**Prefer PageRank for interdisciplinary topics.** PageRank surfaces papers that bridge sub-fields — papers that a pure citation-count approach would underrank because their influence is distributed across multiple clusters rather than concentrated in one.

**Use in-sample citation count for well-defined fields.** When the literature has a clear canon, in-sample citation count expansion stays closer to what researchers in that field already recognise as foundational.

**Raise Min. in-corpus citations to 2–3 for conservative expansion.** The default of 1 means any paper cited by even one corpus paper is a candidate. Raising this threshold ensures candidates have multiple independent endorsers in the corpus before being added.

**Tune the relevance threshold if expansion drifts.** If the corpus picks up papers from adjacent fields, raise the threshold to 0.5–0.6. If expansion stalls and the corpus is smaller than expected, lower it to 0.15–0.2.

**Disable the relevance filter for terminologically diverse literatures.** A multi-disciplinary topic may use very different vocabulary across sub-fields. Cosine similarity against a single query string will penalise papers that are structurally central but use different terminology. Disabling the filter lets structure alone drive expansion.

**Use ISC to size nodes in internal analyses.** The default node size metric `isc` reflects centrality within this corpus. Switch to `citation_count` to emphasise globally prominent works.

**Compare co-citation, bibliographic coupling, and Louvain clusters.** The three methods capture different proximity signals. Convergence across methods — a cluster that appears in all three — is a strong signal of a genuine, coherent research community.

**Raise the edge threshold in dense networks.** For a corpus of several hundred papers, set Min co-citations or Min shared references to 3–5 to reduce visual clutter.

**The cache speeds up repeated runs.** Paper metadata and references are cached to `cache/semantic_scholar_cache.json`. Re-running the same query is significantly faster after the first run. The search results themselves are always fetched fresh.

**Year filter applies to the seed only.** Expansion may pull in papers outside the year range if they are structurally well-connected. For strict year bounds, filter `papers.csv` after export.

---

## Changelog

### April 2026 — Graph display and export improvements

- **Graph backgrounds** — All three network visualisations now use a light-grey background for better readability.
- **Node size metric** — A new **Graph Display Settings** panel lets you choose which metric drives node size across all graphs: `isc` (default), `citation_count`, `isc_ratio`, `sample_relevance`, or `betweenness_centrality`.
- **Node size scaling** — **＋ Size / － Size** buttons scale all nodes proportionally.
- **Open graph in new tab** — Each graph has an **Open in new tab ↗** button that opens a self-contained HTML version.
- **Export graph view as CSV** — Each graph has an **⬇ Export view as CSV** button.
- **Download influential papers** — A download button below the Influential Papers table exports it to CSV.

### Earlier — Cache, DOI enrichment, embedding models, BibTeX export (Anton Moiseev)

- **Persistent disk cache** — Paper metadata and reference lists cached to `cache/semantic_scholar_cache.json` with atomic writes and retry logic.
- **DOI enrichment** — DOI extracted from the Semantic Scholar batch response; title-match fallback if missing.
- **Clickable DOI links** — Papers and Influential Papers tables include a **DOI ↗** link column.
- **Embedding model selector** — Three sentence-transformer models for relevance filtering.
- **BibTeX export with Zotero cluster tags** — Each BibTeX entry carries a `keywords` field with its cluster label.

---

## Pipeline Architecture

### Stage 1 — Seeding

Sends the query to `GET /paper/search/bulk`. Enriches returned paper IDs via `POST /paper/batch` (chunks of 500). Falls back to `GET /paper/search/match` by title for missing journal names or DOIs. Applies year filter if enabled.

### Stage 1b — Domain and Journal Filtering

Extracts fields of study from seed papers; presents domain picker. Applies domain filter, journal quartile filter (if SCImago data is loaded), and the max-seed-papers cap. No API calls.

### Stage 2 — Reference Fetching

Calls `GET /paper/{id}/references` for each corpus paper (one call per paper, minimum 1-second gap between calls). Builds `citations_df` — the directed citation edge list.

### Stage 3 — Expansion

Iterative loop: build graph → compute scores (PageRank or in-sample citation count) → filter candidates → fetch metadata → relevance filter → fetch references → append. Stops at max iterations, convergence, or no passing candidates.

### Stage 4 — Cluster Analysis

Louvain community detection on the undirected citation graph. Labels clusters from top non-stopword title words. One partition, reused for all downstream steps.

### Stage 5 — Influence Analysis

Computes isc, isc_ratio, sample_relevance, and betweenness_centrality from the final corpus.

### Stage 6 — Co-citation and Bibliographic Coupling

Co-citation: pairs of corpus papers cited together accumulate strength. Bibliographic coupling: pairs of corpus papers sharing references accumulate strength. Both matrices are converted to distance matrices and clustered with AgglomerativeClustering (average linkage).

### Stage 7 — Temporal Analysis

Counts papers per cluster per year. Papers with no recorded publication year are excluded.

### Cache and Rate Limiting

1-second minimum gap between API calls. HTTP 429 triggers exponential backoff (up to 4 retries at 60 s, 120 s, 240 s). Cache writes are atomic (temp file + rename).
