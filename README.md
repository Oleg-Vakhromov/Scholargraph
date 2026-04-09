<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/271ab175-f927-430b-a6db-cca29550de67" />

# Scholargraph — User Guide

Scholargraph is a local app that takes a single keyword and maps out an academic field. Most tools like Google Scholar or Semantic Scholar hand you a ranked list and call it done. Scholargraph treats that as a starting point. It follows citation links outward, using PageRank and semantic similarity to surface papers that sit at the structural core of the literature but would never appear on page one of a keyword search.

The output isn't a list; it's a graph. Papers that cite each other heavily cluster together. Co-citation groups show which works share the same intellectual roots. Bibliographic coupling finds parallel research threads with no direct citation link between them at all. Influence metrics identify the papers that bridge sub-fields — not just the ones with the biggest citation counts. Everything exports in standard formats and caches locally, so iterative exploration is fast and the results don't depend on any proprietary platform to reproduce.

---

## Prerequisites

- Python 3.10 or higher
- pip
- Internet access (Semantic Scholar API + first-run model download)

On first run, the sentence-transformers library downloads the `all-MiniLM-L6-v2` embedding model (~90 MB). This happens once and is cached locally afterward.

---

## Installation

1. Download or clone the repository into a local folder.
2. Open a terminal in that folder and install dependencies:

```
pip install -r requirements.txt
```

That installs: requests, pandas, networkx, python-louvain, sentence-transformers, streamlit, pyvis, and scikit-learn.

---

## Running the App

```
python -m streamlit run app.py
```

Streamlit opens the app at `http://localhost:8501` in your default browser. The page loads immediately; the pipeline only runs when you click **▶ Run Pipeline**.

---

## Quick Start

1. Enter a keyword query in the **Query** field (e.g., `retail investor sentiment`).
2. Leave all other settings at their defaults.
3. Click **▶ Run Pipeline**.
4. After seeding completes, a **domain picker** appears. Deselect any irrelevant fields of study, then click **Confirm domains & continue**.

Results appear below once the full pipeline finishes. For a fast test run, set **Seed limit** to 50 before clicking Run.

---

## UI Walkthrough

### Sidebar Parameters

All pipeline settings live in the left sidebar.

**Query**
Free-text keyword search sent to the Semantic Scholar bulk search endpoint. Supports quoted phrases (e.g., `"meme stocks"`) and plain keywords. This is the only required field.

**Seed limit**
Maximum number of papers returned by the initial search (range: 10–2000, default: 200). Larger values give broader initial coverage but make the pipeline slower. Start with 50–100 for exploratory runs.

**Year from / Year to**
Publication year bounds applied to the initial search results. Only active when the **Apply year filter** checkbox is checked. Has no effect when the checkbox is off.

**Apply year filter**
When unchecked (default), year bounds are ignored entirely and all publication years are included. Check this to restrict the corpus to a specific time window.

**Max seed papers (after domain filter)**
Maximum number of papers to keep after the domain filter is applied (range: 10–2000, default: 200). Use this to cap the corpus size before expansion begins.

**Max iterations**
How many expansion passes to run after the initial seed (range: 1–20, default: 3). Each pass fetches new papers from outside the current corpus based on PageRank scores and relevance. More iterations = larger, denser corpus. Three is a reasonable starting point.

**Top-K candidates**
How many high-PageRank candidate papers are evaluated per expansion iteration (range: 10–500, default: 50). These are papers referenced by the corpus but not yet in it. A higher value casts a wider net each iteration.

**Relevance threshold**
Cosine similarity cutoff used to filter expansion candidates (range: 0.0–1.0, default: 0.3). Candidates whose title+abstract embedding scores below this threshold against the original query are discarded. Raise it (e.g., 0.5–0.7) if the expanded corpus feels off-topic; lower it if expansion is too conservative.

**▶ Run Pipeline**
Starts Stage 1 (seeding). After seeding completes, a domain picker appears before the pipeline continues.

---

### Domain Picker

After seeding, the app pauses to show a **Select relevant domains** panel. This lists all fields of study found in the seed papers.

- Deselect domains that are clearly off-topic for your query.
- Papers not matching any selected domain are dropped from the corpus.
- If no field-of-study data is available, the step is skipped automatically.

Click **Confirm domains & continue** to run the remaining pipeline stages: reference fetching, expansion, and analysis.

---

### Results Panels

Results appear below the domain picker after the full pipeline completes. They are computed once and persist until you run the pipeline again.

**Metrics row**
Two numbers at the top: total papers in the corpus and total citation edges. A quick sanity check before diving into the details.

**Papers table**
A sortable table listing every paper in the corpus with title, year, citation count, and authors. Sorted by citation count descending by default — the most globally-cited papers appear first.

**Influential Papers**
Papers ranked by in-sample citation metrics — how often they are cited by other papers *within this corpus*, not just globally. Columns:

- **isc** — in-sample citation count: how many corpus papers cite this paper
- **isc_ratio** — isc as a percentage of the paper's total Semantic Scholar citations (capped at 100 %). A high ratio means this paper is especially central to this specific literature
- **sample_relevance** — isc as a percentage of total corpus size: what fraction of the corpus cites this paper
- **betweenness_centrality** — how often this paper lies on the shortest path between other papers in the citation graph. High betweenness means the paper bridges otherwise-disconnected research clusters

**Co-citation Analysis**
Papers that are frequently cited together by the same sources are likely intellectually related, even if they do not cite each other directly. This panel shows a network where:

- Nodes are corpus papers that appear in at least one co-citation pair
- Edges connect paper pairs cited together, with edge thickness proportional to co-citation count
- Node colour reflects cluster membership from AgglomerativeClustering applied to the co-citation similarity matrix

Two controls adjust the view:
- **Co-citation clusters** — number of clusters to detect (default: matches the Louvain cluster count, capped at 20)
- **Min co-citations (edge threshold)** — minimum times two papers must be co-cited for an edge to appear (default: 2). Raise this to declutter dense networks

**Bibliographic Coupling**
Papers that cite the same sources share intellectual foundations and are likely working in the same research area. This panel shows a network where:

- Nodes are corpus papers that share at least one reference with another corpus paper
- Edges connect paper pairs with shared references, with edge thickness proportional to the number of shared references
- Node colour reflects cluster membership from AgglomerativeClustering applied to the coupling strength matrix

Two controls adjust the view:
- **Coupling clusters** — number of clusters to detect
- **Min shared references (edge threshold)** — minimum shared references required for an edge to appear (default: 2)

**Clusters**
Research clusters detected automatically via Louvain community detection on the citation graph. Displays a summary table with:
- Cluster ID
- Keyword label (top 5 most common non-stopword title words across papers in the cluster)
- Paper count
- Most-cited paper in the cluster

Clusters represent topical sub-communities within the corpus. A cluster with many papers and high citation counts for its top paper is likely a core research stream.

**Knowledge Graph**
An interactive node-link diagram of the corpus. Each node is a paper; edges are citation links (A → B means A cites B). Nodes are sized by citation count and colored by Louvain cluster. Use the **Max nodes** slider above the graph to control how many papers are shown (top N by citation count). Pan and zoom with the mouse.

**Temporal Evolution**
A line chart showing how many papers per cluster were published in each year. Each line is a cluster, labeled by its keyword label. Use this to spot which research streams are growing, stable, or declining — and when new streams emerged.

**Export**
Three download buttons for taking the corpus elsewhere:
- **papers.csv** — full corpus metadata (one row per paper)
- **citations.csv** — citation edge list (source → target)
- **graph.graphml** — the citation graph in GraphML format, compatible with Gephi, Cytoscape, and NetworkX

---

## Tips

- **Start small.** Set Seed limit to 50 and Max iterations to 1 for a quick test run before committing to a full pipeline execution. A full run with limit=200 and 3 iterations can take several minutes.

- **Use the domain picker to focus the corpus.** Dropping off-topic fields before expansion prevents irrelevant papers from proliferating into the corpus via PageRank. For a query like `retail investor sentiment`, you would typically deselect domains like Computer Science or Mathematics if they appear.

- **Tune the relevance threshold if results feel off-topic.** The default (0.3) is permissive. Raise it to 0.5 or 0.6 if the expanded corpus is drifting away from your query.

- **Compare co-citation and bibliographic coupling clusters to Louvain clusters.** The three clustering methods capture different notions of proximity. Louvain uses direct citation links; co-citation groups papers cited together; bibliographic coupling groups papers that cite the same sources. Convergence across methods is a strong signal that a cluster represents a genuine research community.

- **Raise the edge threshold in dense networks.** If the co-citation or bibliographic coupling network is too tangled to read, increase the **Min co-citations** or **Min shared references** slider. Start at 3–5 for a corpus of several hundred papers.

- **The cache speeds up repeated runs.** Paper metadata and references are cached to disk in the `cache/` folder. Re-running the same query with the same papers will be significantly faster on subsequent runs.

- **Use GraphML for deeper analysis.** The exported `graph.graphml` file can be opened in Gephi for advanced layout algorithms, filtering, and centrality calculations not available in the built-in graph view.

- **Year filter applies to the seed only.** The expansion phase may pull in papers outside the year range if they are highly connected to the corpus. If strict year bounds matter, post-filter `papers.csv` after export.

---

## Pipeline Logic

This section explains what the tool does at each stage of the pipeline.

### Stage 1: Seeding the Corpus

When you click **▶ Run Pipeline**, the tool sends your query to the Semantic Scholar bulk search endpoint (`GET /paper/search/bulk`). This returns up to **Seed limit** papers ranked by Semantic Scholar's default relevance ordering, with basic metadata: paper ID, title, year, and citation count. Results are paginated using a cursor token and fetched until the limit is reached.

The returned paper IDs are then sent to the batch metadata endpoint (`POST /paper/batch`) in chunks of up to 500 IDs at a time. This enriches each paper with its abstract, authors, venue, reference count, and fields of study.

For papers where the batch endpoint returns no journal name, the pipeline makes an additional call to the title-match endpoint (`GET /paper/search/match`) using the paper's title. The journal name returned by that lookup is used to fill the gap. This fallback runs once per paper with a missing journal, so the number of extra API calls depends on how many papers lack venue data in the batch response.

If **Apply year filter** is checked, papers outside the specified year range are dropped at this point.

**Output:** `papers_df` — one row per paper. Key columns: `paper_id`, `title`, `year`, `citation_count`, `abstract`, `authors`, `venue`, `reference_count`, `fields_of_study`.

### Stage 1b: Domain Filtering

After seeding, the app extracts the unique fields of study from all seed papers and presents them in the domain picker. When you confirm a selection:

1. Papers not matching any selected domain are removed from `papers_df`.
2. The corpus is further capped at **Max seed papers** by dropping lowest-cited papers beyond the cap.

This step has no API calls — it operates entirely on the data already fetched in Stage 1.

**Output:** `papers_df` trimmed to domain-matching, capped corpus.

### Stage 2: Fetching References

For each paper in the corpus, the pipeline calls `GET /paper/{paperId}/references` to retrieve the list of papers it cites. Each reference record becomes a directed citation edge: **source** (the corpus paper) → **target** (the paper it cites).

These target papers may or may not be in the corpus. Papers referenced by the corpus but not yet in it are the raw material for the expansion stage.

**Output:** `citations_df` — one row per citation edge. Columns: `source`, `target`, `title` (of target), `year` (of target).

**Note on speed:** Reference fetching makes one API call per corpus paper, with a minimum 1-second gap between calls (unauthenticated API tier). A seed corpus of 200 papers will take roughly 3–4 minutes for this stage alone on first run. Subsequent runs are faster because fetched references are cached to disk.

### Stage 3: Expansion Engine

After seeding and reference fetching, the pipeline runs up to **Max iterations** expansion passes. Each pass follows the same sequence:

**1. Build the citation graph**

A directed graph is constructed: nodes are papers in the corpus, edges are citation links from `citations_df`. Papers referenced by the corpus but not yet in it are also added as nodes. This allows PageRank to propagate across the full citation neighbourhood.

**2. Compute PageRank**

PageRank (damping factor α = 0.85) is computed over the full graph. A high PageRank score means a paper is structurally central to this literature — not just globally popular, but important within the specific citation neighbourhood of your corpus.

**3. Select expansion candidates**

All papers referenced by the corpus but not yet in it are candidates. They are ranked by PageRank and the top **Top-K candidates** are selected.

**4. Filter by domain**

If domains were selected in the domain picker, candidates whose fields of study do not intersect the selected domains are discarded before the relevance filter.

**5. Filter by relevance**

Each candidate's title and abstract are embedded using `all-MiniLM-L6-v2`. Cosine similarity is computed against the original query embedding. Candidates scoring below the **Relevance threshold** are discarded.

**6. Fetch references for new papers**

For each paper that passes both filters, its references are fetched from the API. New citation edges are added to `citations_df`.

**7. Append and deduplicate**

Passing papers are appended to `papers_df`. Duplicate paper IDs and citation edges are removed.

**Convergence:** The loop stops early when Max iterations is reached, no candidates remain, no candidates pass the filters, or fewer than 1 new paper was added in the last iteration.

### Stage 4: Cluster Analysis

After expansion, Louvain community detection groups papers into research clusters.

The citation graph is converted to undirected before Louvain runs — direction is less useful for finding topical communities than the raw co-citation and coupling signal captured by the undirected edges.

**Cluster labels** are generated from the most frequent non-stopword title words across all papers in each cluster.

**One partition is computed for the entire corpus and reused for all downstream steps** — graph coloring, the cluster summary table, and the temporal chart. Re-running Louvain per year would produce inconsistent cluster IDs, making year-over-year comparison impossible.

**Output:** `cluster_id` column added to `papers_df`; cluster summary table.

### Stage 5: Influence Analysis

Three metrics are computed per paper from the final corpus:

- **isc** — count of corpus papers that cite this paper (in-corpus citation count)
- **isc_ratio** — isc / total Semantic Scholar citations × 100, capped at 100 %
- **sample_relevance** — isc / corpus size × 100
- **betweenness_centrality** — normalised betweenness on the undirected citation graph

These are computed once on first render and cached in session state.

### Stage 6: Co-citation and Bibliographic Coupling

Two complementary proximity analyses are computed from `citations_df`:

**Co-citation:** For each paper that cites two or more corpus papers, those corpus papers accumulate one unit of co-citation strength. The result is a symmetric pair-count matrix across all corpus papers.

**Bibliographic coupling:** For each corpus paper that is cited by two or more corpus papers, those citing corpus papers accumulate one unit of coupling strength. Equivalently: two corpus papers are coupled by every shared reference they both cite.

Both matrices are converted to distance matrices (`d = 1 / (1 + strength)`) and clustered using AgglomerativeClustering with average linkage. Papers with no co-citation or coupling relationships receive cluster ID −1 and are shown in grey.

Both matrices are computed once and cached; cluster assignments are recalculated on each render when the cluster count control is changed.

### Stage 7: Temporal Analysis

For each combination of publication year and cluster ID in `papers_df`, the pipeline counts how many papers belong to that cluster and were published in that year. The result is a year × cluster matrix used as the line chart. Papers with no recorded publication year are excluded.

### Cache and Rate Limiting

The pipeline enforces a minimum 1-second gap between API requests to stay within the Semantic Scholar unauthenticated rate limit (~1 req/sec). If the API returns HTTP 429 (rate limit exceeded), the client retries with exponential backoff: up to 4 attempts at 60 s, 120 s, and 240 s intervals.

Paper metadata and reference lists are cached to disk in the `cache/` folder after each API call. On subsequent runs, cached papers and references are served from disk with no API request. The search query itself is never cached — search results are always fetched fresh. Journal names resolved via the title-match fallback are stored in the paper's cache entry, so the fallback call is not repeated on re-runs.
