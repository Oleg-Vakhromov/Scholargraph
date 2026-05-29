import pandas as pd
import streamlit as st
from pathlib import Path

from src.analysis.scimago import ScimagoLoader

_scimago = ScimagoLoader()
_scimago.load("data/scimago.csv")


def _generate_bibtex(papers_df, cluster_labels=None):
    """Generate BibTeX string for all papers in the corpus."""
    import re
    entries = []
    labels = cluster_labels or {}

    for _, row in papers_df.iterrows():
        # BibTeX key: paper_id sanitized to alphanumeric + underscores
        key = re.sub(r"[^A-Za-z0-9_]", "_", str(row.get("paper_id") or "unknown"))

        # Cluster tag
        cid = row.get("cluster_id")
        if cid is not None and not pd.isna(cid):
            cid_int = int(cid)
            label = labels.get(cid_int, f"cluster_{cid_int}")
            keywords = f"cluster:{label}"
        else:
            keywords = ""

        def _bib_escape(s):
            if not s:
                return ""
            return str(s).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

        fields = [f"  title = {{{_bib_escape(row.get('title'))}}}"]
        if row.get("authors"):
            fields.append(f"  author = {{{_bib_escape(row.get('authors'))}}}")
        if row.get("year") and not (isinstance(row.get("year"), float) and pd.isna(row.get("year"))):
            fields.append(f"  year = {{{int(row['year'])}}}")
        if row.get("journal"):
            fields.append(f"  journal = {{{_bib_escape(row.get('journal'))}}}")
        if row.get("doi"):
            fields.append(f"  doi = {{{_bib_escape(row.get('doi'))}}}")
        if row.get("doi_url"):
            fields.append(f"  url = {{{_bib_escape(row.get('doi_url'))}}}")
        if keywords:
            fields.append(f"  keywords = {{{keywords}}}")

        entry = f"@article{{{key},\n" + ",\n".join(fields) + "\n}"
        entries.append(entry)

    return "\n\n".join(entries)


def _graph_tab_button(net, html, col, file_name):
    """Render an 'Open in new tab' button using window.open + document.write.
    HTML is base64-encoded inside a <script> tag to avoid attribute quoting issues
    and Chrome's data: URI navigation restriction."""
    import base64 as _b64
    import streamlit.components.v1 as _components
    b64 = _b64.b64encode(html.encode("utf-8")).decode("ascii")
    fn = "openGraph_" + file_name.replace(".", "_").replace("-", "_")
    btn_html = (
        f"<script>function {fn}(){{"
        f"var bytes=Uint8Array.from(atob('{b64}'),function(c){{return c.charCodeAt(0);}});"
        f"var h=new TextDecoder('utf-8').decode(bytes);"
        f"var w=window.open('','_blank');"
        f"if(w){{w.document.open();w.document.write(h);w.document.close();}}"
        f"}}</script>"
        f"<button onclick='{fn}()' style='padding:4px 12px;border-radius:4px;"
        f"border:1px solid #999;cursor:pointer;background:#fff;"
        f"font-size:14px;font-family:sans-serif;color:#333'>"
        f"Open in new tab ↗</button>"
    )
    with col:
        _components.html(btn_html, height=44)

# ---------------------------------------------------------------------------
# Cached resource loaders — loaded once, reused across reruns
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_filter(model_name: str = "all-MiniLM-L6-v2"):
    from src.expansion.filter import RelevanceFilter
    return RelevanceFilter(model_name=model_name)


@st.cache_resource
def _get_client():
    from src.api.semantic_scholar import SemanticScholarClient
    import os
    _cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    try:
        from src.cache.cache_manager import DiskCache
        cache = DiskCache(cache_dir=_cache_dir)
        cache.load()
    except Exception:
        cache = None
    api_key = os.environ.get("S2_API_KEY") or None
    return SemanticScholarClient(cache=cache, api_key=api_key)



# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Bibliometrics", layout="wide")
st.title("Bibliometrics")
st.caption("Semantic Scholar corpus builder + research cluster explorer")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "papers_df" not in st.session_state:
    st.session_state["papers_df"] = None
if "citations_df" not in st.session_state:
    st.session_state["citations_df"] = None
if "corpus" not in st.session_state:
    st.session_state["corpus"] = None
if "partition" not in st.session_state:
    st.session_state["partition"] = None
if "cluster_labels" not in st.session_state:
    st.session_state["cluster_labels"] = None
if "cluster_summary" not in st.session_state:
    st.session_state["cluster_summary"] = None
if "seed_done" not in st.session_state:
    st.session_state["seed_done"] = False
if "available_domains" not in st.session_state:
    st.session_state["available_domains"] = []
if "selected_domains" not in st.session_state:
    st.session_state["selected_domains"] = []
if "pipeline_log" not in st.session_state:
    st.session_state["pipeline_log"] = []
if "influence_df" not in st.session_state:
    st.session_state["influence_df"] = None
if "cocitation_df" not in st.session_state:
    st.session_state["cocitation_df"] = None
if "cocitation_clusters" not in st.session_state:
    st.session_state["cocitation_clusters"] = None
if "cluster_params" not in st.session_state:
    st.session_state["cluster_params"] = None
if "available_quartiles" not in st.session_state:
    st.session_state["available_quartiles"] = []
if "selected_quartiles" not in st.session_state:
    st.session_state["selected_quartiles"] = []
if "scimago_loaded" not in st.session_state:
    st.session_state["scimago_loaded"] = False
if "bibcoupling_df" not in st.session_state:
    st.session_state["bibcoupling_df"] = None
if "bibcoupling_clusters" not in st.session_state:
    st.session_state["bibcoupling_clusters"] = None
if "node_size_scale" not in st.session_state:
    st.session_state["node_size_scale"] = 1.0

# ---------------------------------------------------------------------------
# Sidebar — pipeline parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Search")

    query = st.text_input("Query", placeholder="e.g. knowledge graph embedding")

    limit = st.number_input("Seed limit", min_value=10, max_value=2000, value=200, step=50)

    col1, col2 = st.columns(2)
    year_min = col1.number_input("Year from", min_value=1900, max_value=2030, value=2010, step=1)
    year_max = col2.number_input("Year to", min_value=1900, max_value=2030, value=2024, step=1)
    use_year_filter = st.checkbox("Apply year filter", value=False)

    st.divider()
    max_seed_papers = st.number_input(
        "Max seed papers (after domain filter)", min_value=10, max_value=2000, value=200, step=50
    )

    st.divider()
    st.subheader("Expansion")

    _STRATEGY_OPTIONS = {
        "PageRank (graph-based)": "pagerank",
        "In-sample citation count": "citation_count",
    }
    selected_strategy_label = st.selectbox(
        "Ranking strategy",
        options=list(_STRATEGY_OPTIONS.keys()),
        index=0,
        help="Both strategies filter candidates by a 75th-percentile threshold rather than a top-K cap. "
             "PageRank thresholds on graph score; In-sample citation count thresholds on ≥ max(3, 75th-percentile) "
             "in-corpus citations, matching the classic bibliometric approach.",
    )
    selected_strategy = _STRATEGY_OPTIONS[selected_strategy_label]

    max_iterations = st.number_input("Max iterations", min_value=1, max_value=20, value=3, step=1)

    with st.expander("Advanced settings", expanded=False):
        min_citations = st.number_input(
            "Min. in-corpus citations",
            min_value=1, max_value=20, value=1, step=1,
            help="Minimum number of corpus papers that must cite a candidate for it to qualify. "
                 "Applies to both strategies before the 75th-percentile threshold.",
        )

        apply_relevance_filter = st.checkbox(
            "Apply relevance filter",
            value=True,
            help="Filter expansion candidates by cosine similarity to the query. "
                 "Disable to add all structurally or citation-qualified candidates regardless of topical similarity.",
        )
        relevance_threshold = st.slider(
            "Relevance threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05,
            disabled=not apply_relevance_filter,
        )

        if apply_relevance_filter:
            st.divider()
            st.subheader("Embedding Model")
            _MODEL_OPTIONS = {
                "Fast (~90 MB)": "all-MiniLM-L6-v2",
                "Balanced / High quality (~420 MB)": "all-mpnet-base-v2",
                "Long-context multilingual (~2.2 GB)": "BAAI/bge-m3",
            }
            selected_model_label = st.selectbox(
                "Embedding model",
                options=list(_MODEL_OPTIONS.keys()),
                index=0,
                help="Controls relevance filtering during expansion. Faster models use less memory; larger models improve filtering quality for broad or multilingual queries.",
            )
            selected_model = _MODEL_OPTIONS[selected_model_label]
        else:
            selected_model = "all-MiniLM-L6-v2"

    run_button = st.button("▶ Run Pipeline", type="primary", width='stretch')

    if not _scimago.is_loaded:
        st.caption("ℹ SCImago data not found — quartile filter unavailable")

# ---------------------------------------------------------------------------
# Stage 1 — Seed corpus only
# ---------------------------------------------------------------------------

if run_button:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        # Reset all state on new run (AC-5)
        st.session_state["seed_done"] = False
        st.session_state["available_domains"] = []
        st.session_state["selected_domains"] = []
        st.session_state["papers_df"] = None
        st.session_state["citations_df"] = None
        st.session_state["corpus"] = None
        st.session_state["partition"] = None
        st.session_state["cluster_labels"] = None
        st.session_state["cluster_summary"] = None
        st.session_state["cluster_params"] = None
        st.session_state["available_quartiles"] = []
        st.session_state["selected_quartiles"] = []
        st.session_state["scimago_loaded"] = False
        st.session_state["pipeline_log"] = []
        st.session_state["influence_df"] = None
        st.session_state["cocitation_df"] = None
        st.session_state["cocitation_clusters"] = None
        st.session_state["bibcoupling_df"] = None
        st.session_state["bibcoupling_clusters"] = None
        st.session_state["node_size_scale"] = 1.0

        year_range = (int(year_min), int(year_max)) if use_year_filter else None

        with st.status("Seeding corpus...", expanded=True) as status:
            from src.corpus.builder import CorpusBuilder
            client = _get_client()
            corpus = CorpusBuilder(client)
            try:
                corpus.seed(query.strip(), limit=int(limit), year_range=year_range)
            except RuntimeError as _e:
                _msg = str(_e)
                if "429" in _msg:
                    st.error(
                        "Semantic Scholar rate limit hit. Wait 1–2 minutes and try again, "
                        "or apply for an API key at semanticscholar.org/product/api for higher limits."
                    )
                else:
                    st.error(f"Seed failed: {_msg}")
                status.update(label="Seed failed — API error", state="error")
                st.stop()
            if _scimago.is_loaded and not corpus.papers_df.empty:
                corpus.papers_df = _scimago.enrich_papers(corpus.papers_df)
                st.session_state["available_quartiles"] = _scimago.available_quartiles(corpus.papers_df)
                st.session_state["scimago_loaded"] = True
            if corpus.papers_df.empty:
                st.error(
                    f'Seed returned 0 papers for "{query.strip()}". '
                    "The Semantic Scholar API may not have results "
                    "for this query — try rephrasing or broadening the search terms."
                )
                status.update(label="Seed failed — no papers found", state="error")
            else:
                st.write(f"Seed complete — {len(corpus.papers_df)} papers")
                status.update(label="Seed complete", state="complete")

        if not corpus.papers_df.empty:
            st.session_state["corpus"] = corpus
            st.session_state["available_domains"] = corpus.extract_domains()
            st.session_state["seed_done"] = True

# ---------------------------------------------------------------------------
# Stage 2 — Domain picker and pipeline continuation
# ---------------------------------------------------------------------------

if st.session_state.get("seed_done") and st.session_state.get("corpus") is not None:
    available = st.session_state["available_domains"]

    if available:
        st.subheader("Select relevant domains")
        st.caption("Only papers belonging to the selected fields of study will be kept. Uncheck domains that are not relevant to your query.")
        chosen = st.multiselect(
            "Fields of study",
            options=available,
            default=available,
            key="domain_multiselect",
        )
    else:
        chosen = []
        st.info("No field-of-study data available for seed papers — proceeding without domain filter.")

    if st.session_state.get("scimago_loaded") and st.session_state.get("available_quartiles"):
        st.caption("Restrict corpus to journals in selected SCImago quartiles. Papers with no quartile data are excluded when any quartile is selected.")
        selected_quartiles = st.multiselect(
            "Journal quartile filter",
            options=st.session_state["available_quartiles"],
            default=[],
            key="quartile_multiselect",
        )
    else:
        selected_quartiles = []

    confirm_button = st.button("Confirm domains & continue", type="primary")

    if confirm_button:
        corpus = st.session_state["corpus"]
        selected = chosen if chosen else available  # fall back to all if none chosen

        def _log(msg):
            st.session_state["pipeline_log"].append(msg)
            st.write(msg)
            print(msg, flush=True)

        with st.status("Running pipeline...", expanded=True) as status:
            from src.expansion.expander import CorpusExpander
            from src.graph.engine import GraphEngine

            # Apply domain filter + citation cap
            if selected:
                corpus.apply_domain_filter(selected, max_papers=int(max_seed_papers))
                _log(f"Domain filter applied — {len(corpus.papers_df)} papers retained")

            # Apply quartile filter
            if selected_quartiles:
                mask = corpus.papers_df["scimago_quartile"].isin(selected_quartiles)
                corpus.papers_df = corpus.papers_df[mask].reset_index(drop=True)
                _log(f"Quartile filter applied — {len(corpus.papers_df)} papers retained ({', '.join(selected_quartiles)})")
            st.session_state["selected_quartiles"] = selected_quartiles

            st.session_state["selected_domains"] = selected

            # Fetch references
            _log("Fetching references...")
            corpus.fetch_references(
                on_progress=lambda n, t: _log(f"Fetching references… {n}/{t}")
            )
            _log(f"References fetched — {len(corpus.citations_df)} citation edges")

            # Expand
            _log("Expanding corpus...")
            _log("Loading embedding model…")
            expander = CorpusExpander(corpus._client, GraphEngine(), _load_filter(selected_model))

            from src.enrichment.crossref import CrossRefClient
            from src.enrichment.openalex import OpenAlexReferenceClient
            _crossref = CrossRefClient()
            _openalex = OpenAlexReferenceClient()

            def _report_iteration(iteration, seed_count, refs_extracted, new_papers):
                _log(
                    f"Iteration {iteration}: "
                    f"{seed_count} seed papers | "
                    f"{refs_extracted} references extracted | "
                    f"{new_papers} new papers added"
                )
                _doi_filled, _doi_failed = corpus.enrich_dois(_crossref)
                _log(f"  DOIs: {_doi_filled} filled, {_doi_failed} failed")
                _venue_filled, _venue_failed = corpus.enrich_venues(_crossref)
                _log(f"  Venues: {_venue_filled} filled, {_venue_failed} failed")
                _refs_added, _refs_failed = corpus.enrich_references(_openalex)
                _log(f"  References: {_refs_added} edges recovered, {_refs_failed} papers yielded nothing")

            expander.expand(
                corpus,
                query=query.strip(),
                max_iterations=int(max_iterations),
                relevance_threshold=float(relevance_threshold),
                expansion_strategy=selected_strategy,
                apply_relevance_filter=apply_relevance_filter,
                min_citations=int(min_citations),
                allowed_domains=st.session_state.get("selected_domains") or None,
                on_iteration=_report_iteration,
            )
            _log(f"Expansion complete — {len(corpus.papers_df)} papers total")

            if corpus.papers_df.empty:
                st.error(
                    "Corpus is empty after filtering. All papers were removed by the "
                    "domain or quartile filter — try selecting more domains or relaxing filters."
                )
                status.update(label="Pipeline failed — empty corpus", state="error")
                st.stop()

            Path("data").mkdir(exist_ok=True)
            corpus.save_papers("data/papers.csv")
            if not corpus.citations_df.empty:
                corpus.save_citations("data/citations.csv")
            _log("Saved to data/papers.csv and data/citations.csv")

            st.session_state["papers_df"] = corpus.papers_df
            st.session_state["citations_df"] = corpus.citations_df
            st.session_state["corpus"] = corpus
            st.session_state["seed_done"] = False  # hide picker after completion
            from src.sentinel.scorer import Scorer
            _pre_score_input = corpus.papers_df.to_dict("records")
            _pre_score_edges = (
                corpus.citations_df[["source", "target"]].to_dict("records")
                if not corpus.citations_df.empty else []
            )
            _pre_scored = Scorer().score(_pre_score_input, _pre_score_edges)
            st.session_state["pre_scored_df"] = pd.DataFrame(_pre_scored)[
                ["paper_id", "local_citation_count", "trend_score"]
            ]
            status.update(label="Pipeline complete!", state="complete")

# ---------------------------------------------------------------------------
# Pipeline log — persists after status widget collapses
# ---------------------------------------------------------------------------

if st.session_state.get("pipeline_log"):
    with st.expander("Pipeline run log", expanded=True):
        for msg in st.session_state["pipeline_log"]:
            st.write(msg)

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

if st.session_state["papers_df"] is not None:
    papers_df = st.session_state["papers_df"]
    citations_df = st.session_state["citations_df"]

    _PALETTE = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4",
    ]

    _LAYER_COLORS = {
        "Spearhead":       "#e74c3c",
        "Foundational":    "#3498db",
        "Theory Extender": "#9b59b6",
        "Standard":        "#95a5a6",
    }

    _mc1, _mc2, _mc3 = st.columns([2, 2, 1])
    _mc1.metric("Total papers", len(papers_df))
    _mc2.metric("Total citations", len(citations_df) if citations_df is not None else 0)
    if _mc3.button("✕ Clear results"):
        for _k in [
            "papers_df", "citations_df", "corpus", "partition", "cluster_labels",
            "cluster_summary", "influence_df", "cocitation_df", "cocitation_clusters",
            "bibcoupling_df", "bibcoupling_clusters", "graph", "pre_scored_df",
            "cluster_params",
        ]:
            st.session_state[_k] = None
        st.session_state["seed_done"] = False
        st.session_state["available_domains"] = []
        st.session_state["selected_domains"] = []
        st.session_state["available_quartiles"] = []
        st.session_state["selected_quartiles"] = []
        st.session_state["pipeline_log"] = []
        st.rerun()

    if "local_citation_count" in papers_df.columns:
        with st.sidebar:
            st.divider()
            st.subheader("Ranked List")
            st.caption("Sorted by local citation index (LCI), then global citations.")
            _ranked = (
                papers_df[["title", "local_citation_count", "citation_count"]]
                .sort_values(["local_citation_count", "citation_count"], ascending=[False, False])
                .reset_index(drop=True)
                .head(20)
            )
            _ranked.index = _ranked.index + 1
            _ranked.columns = ["Title", "LCI", "Global Citations"]
            _ranked["Title"] = _ranked["Title"].apply(lambda t: (str(t) or "")[:40])
            st.dataframe(_ranked, width='stretch')

    st.subheader("Papers")
    st.caption("All papers collected during seeding and expansion, ranked by citation count. Higher citation counts indicate greater influence in the field.")
    display_cols = [
        c for c in ["title", "year", "journal", "citation_count", "trend_score", "layer_tag", "authors", "doi_url"]
        if c in papers_df.columns
    ]
    st.dataframe(
        papers_df[display_cols].sort_values("citation_count", ascending=False),
        width='stretch',
        hide_index=True,
        column_config={
            "doi_url": st.column_config.LinkColumn("DOI", display_text="↗"),
        },
    )

    # -----------------------------------------------------------------------
    # Influential Papers
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Influential Papers")
    st.caption(
        "Papers ranked by in-sample citations (ISC) — how often they are cited "
        "by other papers within this corpus. ISC ratio shows what fraction of a "
        "paper's total Semantic Scholar citations come from within the corpus."
    )

    if st.session_state.get("influence_df") is None:
        from src.analysis.influence import InfluenceAnalyzer

        ia = InfluenceAnalyzer()
        _inf_df = ia.compute_isc(papers_df, st.session_state["citations_df"])
        st.session_state["influence_df"] = _inf_df

    _inf_display = st.session_state["influence_df"]
    _inf_cols = [
        c for c in [
            "title", "year", "journal", "citation_count",
            "isc", "isc_ratio", "sample_relevance", "doi_url",
        ]
        if c in _inf_display.columns
    ]
    st.dataframe(
        _inf_display[_inf_cols].sort_values("isc", ascending=False),
        width='stretch',
        hide_index=True,
        column_config={
            "doi_url": st.column_config.LinkColumn("DOI", display_text="↗"),
        },
    )
    _inf_csv = _inf_display[_inf_cols].sort_values("isc", ascending=False).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download influential papers",
        data=_inf_csv,
        file_name="influential_papers.csv",
        mime="text/csv",
    )

    # -----------------------------------------------------------------------
    # Graph tabs — Co-citation, Bibliographic Coupling, Knowledge Graph
    # -----------------------------------------------------------------------
    st.divider()
    _graph_tab1, _graph_tab2, _graph_tab3 = st.tabs(
        ["Co-citation Analysis", "Bibliographic Coupling", "Knowledge Graph"]
    )

    with _graph_tab1:
        st.caption(
            "Papers frequently cited together by the same sources are likely intellectually "
            "related, even without a direct citation link. Communities detected via "
            "Girvan-Newman; edges filtered to top 1% by co-citation count."
        )

        _coc_decile_col, _ = st.columns(2)
        _coc_decile = _coc_decile_col.slider(
            "Edge density (decile threshold)",
            min_value=0.90, max_value=1.0, value=0.99, step=0.01,
            key="cocit_decile",
        )

        if st.session_state.get("cocitation_df") is None:
            from src.analysis.cocitation import CoCitationAnalyzer
            _ca = CoCitationAnalyzer()
            st.session_state["cocitation_df"] = _ca.build_cocitation_matrix(
                citations_df, set(papers_df["paper_id"])
            )

        _coc_df = st.session_state["cocitation_df"]
        _coc_title_lookup = dict(zip(papers_df["paper_id"], papers_df["title"]))

        from src.analysis.cocitation import CoCitationAnalyzer
        _coc_G, _coc_communities, _coc_filtered = CoCitationAnalyzer().build_spring_graph(
            _coc_df, _coc_title_lookup, decile=_coc_decile
        )

        if _coc_G.number_of_nodes() == 0:
            st.info("No co-citation pairs meet the density threshold — try lowering the decile.")
        else:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import networkx as nx

            _coc_fig, _coc_ax = plt.subplots(figsize=(14, 10))
            _coc_pos = nx.spring_layout(_coc_G, k=0.8, seed=42)
            _coc_cmap = plt.cm.tab10
            _coc_colors = [_coc_cmap(_coc_communities.get(n, 0) % 10) for n in _coc_G.nodes()]
            nx.draw_networkx_nodes(_coc_G, _coc_pos, node_color=_coc_colors,
                                   node_size=300, edgecolors="black", ax=_coc_ax)
            nx.draw_networkx_edges(_coc_G, _coc_pos, alpha=0.5, ax=_coc_ax)
            _coc_labels = nx.get_node_attributes(_coc_G, "label")
            nx.draw_networkx_labels(_coc_G, _coc_pos, labels=_coc_labels, font_size=7, ax=_coc_ax)
            _coc_edge_labels = nx.get_edge_attributes(_coc_G, "weight")
            nx.draw_networkx_edge_labels(_coc_G, _coc_pos, edge_labels=_coc_edge_labels,
                                         font_size=6, ax=_coc_ax)
            _coc_ax.axis("off")
            st.pyplot(_coc_fig)
            plt.close(_coc_fig)

            _coc_export_rows = [
                {
                    "community_from": _coc_communities.get(_e["paper_a"], -1),
                    "from_paper_name": _coc_title_lookup.get(_e["paper_a"], _e["paper_a"]),
                    "community_to": _coc_communities.get(_e["paper_b"], -1),
                    "to_paper_name": _coc_title_lookup.get(_e["paper_b"], _e["paper_b"]),
                    "cocitation_count": _e["cocitation_count"],
                }
                for _, _e in _coc_filtered.iterrows()
            ]
            st.download_button(
                "⬇ Export view as CSV",
                data=pd.DataFrame(_coc_export_rows).to_csv(index=False).encode("utf-8"),
                file_name="cocitation_view.csv",
                mime="text/csv",
                key="cocit_export_csv",
            )

    with _graph_tab2:
        st.caption(
            "Papers that cite the same sources share intellectual foundations and are "
            "likely working in the same research area, even without directly referencing "
            "each other. Communities detected via Girvan-Newman; edges filtered to top 1% "
            "by shared-reference count."
        )

        _bib_decile_col, _ = st.columns(2)
        _bib_decile = _bib_decile_col.slider(
            "Edge density (decile threshold)",
            min_value=0.90, max_value=1.0, value=0.99, step=0.01,
            key="bib_decile",
        )

        if st.session_state.get("bibcoupling_df") is None:
            from src.analysis.bibcoupling import BibliographicCoupler
            _bc = BibliographicCoupler()
            st.session_state["bibcoupling_df"] = _bc.build_coupling_matrix(
                citations_df, set(papers_df["paper_id"])
            )

        _bib_df = st.session_state["bibcoupling_df"]
        _bib_title_lookup = dict(zip(papers_df["paper_id"], papers_df["title"]))

        from src.analysis.bibcoupling import BibliographicCoupler
        _bib_G, _bib_communities, _bib_filtered = BibliographicCoupler().build_spring_graph(
            _bib_df, _bib_title_lookup, decile=_bib_decile
        )

        if _bib_G.number_of_nodes() == 0:
            st.info("No bibliographic coupling pairs meet the density threshold — try lowering the decile.")
        else:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import networkx as nx

            _bib_fig, _bib_ax = plt.subplots(figsize=(14, 10))
            _bib_pos = nx.spring_layout(_bib_G, k=0.8, seed=42)
            _bib_cmap = plt.cm.tab10
            _bib_colors = [_bib_cmap(_bib_communities.get(n, 0) % 10) for n in _bib_G.nodes()]
            nx.draw_networkx_nodes(_bib_G, _bib_pos, node_color=_bib_colors,
                                   node_size=300, edgecolors="black", ax=_bib_ax)
            nx.draw_networkx_edges(_bib_G, _bib_pos, alpha=0.5, ax=_bib_ax)
            _bib_labels = nx.get_node_attributes(_bib_G, "label")
            nx.draw_networkx_labels(_bib_G, _bib_pos, labels=_bib_labels, font_size=7, ax=_bib_ax)
            _bib_edge_labels = nx.get_edge_attributes(_bib_G, "weight")
            nx.draw_networkx_edge_labels(_bib_G, _bib_pos, edge_labels=_bib_edge_labels,
                                         font_size=6, ax=_bib_ax)
            _bib_ax.axis("off")
            st.pyplot(_bib_fig)
            plt.close(_bib_fig)

            _bib_export_rows = [
                {
                    "community_from": _bib_communities.get(_e["paper_a"], -1),
                    "from_paper_name": _bib_title_lookup.get(_e["paper_a"], _e["paper_a"]),
                    "community_to": _bib_communities.get(_e["paper_b"], -1),
                    "to_paper_name": _bib_title_lookup.get(_e["paper_b"], _e["paper_b"]),
                    "coupling_strength": _e["coupling_strength"],
                }
                for _, _e in _bib_filtered.iterrows()
            ]
            st.download_button(
                "⬇ Export view as CSV",
                data=pd.DataFrame(_bib_export_rows).to_csv(index=False).encode("utf-8"),
                file_name="bibcoupling_view.csv",
                mime="text/csv",
                key="bib_export_csv",
            )

    with _graph_tab3:
        st.caption("Citation network. Node size reflects the selected metric; node colour reflects cluster or layer membership. Arrows point from citing paper to cited paper.")

        # Node size controls — only apply to this graph
        _gds_col1, _gds_col2, _gds_col3 = st.columns([4, 1, 1])
        _node_size_metric = _gds_col1.selectbox(
            "Node size metric",
            options=["citation_count", "isc", "isc_ratio", "sample_relevance"],
            index=0,
            key="node_size_metric",
        )
        _gds_col2.markdown('<div style="padding-top:27px"></div>', unsafe_allow_html=True)
        if _gds_col2.button("－ Size"):
            st.session_state["node_size_scale"] = max(st.session_state["node_size_scale"] / 1.3, 0.1)
        _gds_col3.markdown('<div style="padding-top:27px"></div>', unsafe_allow_html=True)
        if _gds_col3.button("＋ Size"):
            st.session_state["node_size_scale"] = min(st.session_state["node_size_scale"] * 1.3, 10.0)
        _node_scale = st.session_state["node_size_scale"]

        max_nodes = st.slider(
            "Max nodes", min_value=20, max_value=500, value=50, step=20,
            key="graph_max_nodes",
        )

        papers_for_graph = (
            st.session_state["papers_df"]
            .fillna({"citation_count": 0})
            .sort_values("citation_count", ascending=False)
            .head(max_nodes)
            .reset_index(drop=True)
        )
        node_ids = set(papers_for_graph["paper_id"])

        citations_df_kg = st.session_state["citations_df"]
        _visible_edges = pd.DataFrame()
        if citations_df_kg is not None and not citations_df_kg.empty:
            _visible_edges = citations_df_kg[
                citations_df_kg["source"].isin(node_ids) & citations_df_kg["target"].isin(node_ids)
            ]

        # Node sizes: use citation_count from papers_for_graph as always-available base;
        # merge influence metrics when available
        _size_series = papers_for_graph["citation_count"].fillna(0)
        _inf_df = st.session_state["influence_df"]
        if _inf_df is not None and _node_size_metric in _inf_df.columns:
            _size_merged = papers_for_graph[["paper_id"]].merge(
                _inf_df[["paper_id", _node_size_metric]], on="paper_id", how="left"
            )
            _size_series = _size_merged[_node_size_metric].fillna(0)
        elif _node_size_metric == "citation_count":
            _size_series = papers_for_graph["citation_count"].fillna(0)
        _size_max = float(max(_size_series.max(), 1e-9))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx

        G_kg = nx.DiGraph()
        _kg_node_colors = []
        _kg_node_sizes = []
        for i, row in papers_for_graph.iterrows():
            pid = row["paper_id"]
            G_kg.add_node(pid, label=(row["title"] or "")[:35])
            if "layer_tag" in papers_for_graph.columns:
                _kg_node_colors.append(_LAYER_COLORS.get(row.get("layer_tag"), "#888888"))
            else:
                cid = int(row["cluster_id"]) if pd.notna(row.get("cluster_id")) else -1
                _kg_node_colors.append(_PALETTE[cid % len(_PALETTE)] if cid >= 0 else "#888888")
            _val = float(_size_series.iloc[i])
            _kg_node_sizes.append(max(50.0, (_val / _size_max) * 600.0 * _node_scale + 50.0))

        for _, edge in _visible_edges.iterrows():
            G_kg.add_edge(edge["source"], edge["target"])

        _kg_fig, _kg_ax = plt.subplots(figsize=(14, 10))
        _kg_pos = nx.spring_layout(G_kg, k=0.8, seed=42)
        nx.draw_networkx_nodes(G_kg, _kg_pos, node_color=_kg_node_colors,
                               node_size=_kg_node_sizes, edgecolors="black", ax=_kg_ax)
        nx.draw_networkx_edges(G_kg, _kg_pos, alpha=0.4, arrows=True,
                               arrowstyle="-|>", arrowsize=12, ax=_kg_ax)
        _kg_labels = nx.get_node_attributes(G_kg, "label")
        nx.draw_networkx_labels(G_kg, _kg_pos, labels=_kg_labels, font_size=6, ax=_kg_ax)
        _kg_ax.axis("off")
        st.pyplot(_kg_fig)
        plt.close(_kg_fig)

        _kg_pid_cluster = dict(zip(papers_for_graph["paper_id"], papers_for_graph.get("cluster_id", pd.Series(dtype=int)).fillna(-1).astype(int)))
        _kg_pid_title = dict(zip(papers_for_graph["paper_id"], papers_for_graph["title"]))
        _kg_export_rows = [
            {
                "cluster_from": _kg_pid_cluster.get(edge["source"], -1),
                "from_paper_name": _kg_pid_title.get(edge["source"], edge["source"]),
                "cluster_to": _kg_pid_cluster.get(edge["target"], -1),
                "to_paper_name": _kg_pid_title.get(edge["target"], edge["target"]),
            }
            for _, edge in _visible_edges.iterrows()
        ] if not _visible_edges.empty else []
        if _kg_export_rows:
            st.download_button(
                "⬇ Export view as CSV",
                data=pd.DataFrame(_kg_export_rows).to_csv(index=False).encode("utf-8"),
                file_name="knowledge_graph_view.csv",
                mime="text/csv",
                key="kg_export_csv",
            )

    # -----------------------------------------------------------------------
    # Cluster analysis
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Clusters")
    st.caption(
        "Research clusters detected via Louvain community detection on the citation graph. "
        "Lower resolution produces fewer, broader clusters. Min cluster size merges small "
        "noise clusters into their most-connected neighbour."
    )

    _cl_col1, _cl_col2 = st.columns(2)
    cluster_resolution = _cl_col1.slider(
        "Resolution", min_value=0.1, max_value=2.0, value=0.5, step=0.1,
        key="cluster_resolution",
    )
    min_cluster_size = _cl_col2.number_input(
        "Min cluster size", min_value=1, max_value=50, value=3, step=1,
        key="cluster_min_size",
    )

    _current_params = (float(cluster_resolution), int(min_cluster_size))

    if st.session_state.get("cluster_params") != _current_params or "cluster_id" not in papers_df.columns:
        from src.analysis.cluster import ClusterEngine

        _G_cl = st.session_state.get("graph")
        if _G_cl is None:
            from src.graph.engine import GraphEngine
            _G_cl = GraphEngine().build_graph(papers_df, citations_df)
            st.session_state["graph"] = _G_cl
        _ce = ClusterEngine()
        _partition = _ce.detect_communities(_G_cl, resolution=float(cluster_resolution))
        _partition = _ce.merge_small_clusters(_partition, papers_df, citations_df, min_size=int(min_cluster_size))
        papers_df = _ce.add_cluster_assignments(papers_df, _partition)
        _cluster_labels = _ce.label_clusters(papers_df, _partition)
        _cluster_summary = _ce.cluster_summary_df(papers_df, _cluster_labels)

        from src.sentinel.layerer import Layerer

        papers_df = papers_df.drop(
            columns=[c for c in ["layer_tag", "local_citation_count", "trend_score"] if c in papers_df.columns]
        )
        _pre_scored_df = st.session_state.get("pre_scored_df")
        if _pre_scored_df is not None:
            papers_df = papers_df.merge(_pre_scored_df, on="paper_id", how="left")
        else:
            from src.sentinel.scorer import Scorer
            _score_input = papers_df.copy()
            if "cluster_id" in _score_input.columns:
                _score_input["cluster_id"] = _score_input["cluster_id"].fillna(-1).astype(int)
            _papers_list = _score_input.to_dict("records")
            _edges_list = (
                citations_df[["source", "target"]].to_dict("records")
                if citations_df is not None and not citations_df.empty else []
            )
            _pre_scored_df = pd.DataFrame(Scorer().score(_papers_list, _edges_list))[
                ["paper_id", "local_citation_count", "trend_score"]
            ]
            papers_df = papers_df.merge(_pre_scored_df, on="paper_id", how="left")
        _papers_list_for_layerer = papers_df.copy()
        if "cluster_id" in _papers_list_for_layerer.columns:
            _papers_list_for_layerer["cluster_id"] = _papers_list_for_layerer["cluster_id"].fillna(-1).astype(int)
        _tagged = Layerer().tag(_papers_list_for_layerer.to_dict("records"))
        _tag_df = pd.DataFrame(_tagged)[["paper_id", "layer_tag"]]
        papers_df = papers_df.merge(_tag_df, on="paper_id", how="left")

        st.session_state["papers_df"] = papers_df
        st.session_state["partition"] = _partition
        st.session_state["cluster_labels"] = _cluster_labels
        st.session_state["cluster_summary"] = _cluster_summary
        st.session_state["cluster_params"] = _current_params

    if st.session_state["cluster_summary"] is not None:
        st.dataframe(
            st.session_state["cluster_summary"],
            width='stretch',
            hide_index=True,
        )

    if "layer_tag" in papers_df.columns:
        _tag_counts = (
            papers_df["layer_tag"]
            .value_counts()
            .rename_axis("layer_tag")
            .reset_index(name="count")
        )
        st.dataframe(_tag_counts, width='stretch', hide_index=True)

    _cl_labels = st.session_state.get("cluster_labels") or {}
    if _cl_labels:
        _cl_options = {f"Cluster {cid}: {label}": cid for cid, label in sorted(_cl_labels.items())}
        _cl_selected_label = st.selectbox("Inspect cluster", list(_cl_options.keys()), key="cluster_inspect_select")
        _cl_selected_id = _cl_options[_cl_selected_label]
        _cl_papers = papers_df[papers_df["cluster_id"] == _cl_selected_id].copy()
        _cl_show_cols = [c for c in ["title", "year", "layer_tag", "citation_count", "local_citation_count", "journal"] if c in _cl_papers.columns]
        st.dataframe(
            _cl_papers[_cl_show_cols].sort_values("citation_count", ascending=False).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------------------------
    # Temporal Evolution
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Temporal Evolution")
    st.caption("Number of papers per cluster published each year. Shows how different research directions grew or declined over time. Uses a single community partition applied across all years for consistent cluster identity.")

    from src.analysis.temporal import TemporalAnalyzer

    _labels = st.session_state["cluster_labels"] or {}
    evolution = TemporalAnalyzer().cluster_evolution(
        st.session_state["papers_df"], _labels
    )

    if not evolution.empty:
        pivot = TemporalAnalyzer().evolution_pivot(evolution)
        pivot.columns = [
            f"#{int(c)} {_labels.get(int(c), f'cluster-{int(c)}')}"
            for c in pivot.columns
        ]
        st.line_chart(pivot)
    else:
        st.info("No publication year data available for temporal chart.")

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Export")
    st.caption("Download the full corpus for offline analysis. GraphML format is compatible with Gephi, Cytoscape, and NetworkX.")

    import io
    import networkx as nx

    exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)

    papers_csv = st.session_state["papers_df"].to_csv(index=False).encode("utf-8")
    exp_col1.download_button(
        "⬇ papers.csv",
        data=papers_csv,
        file_name="papers.csv",
        mime="text/csv",
    )

    _citations_df_export = st.session_state["citations_df"]
    if _citations_df_export is not None:
        citations_csv = _citations_df_export.to_csv(index=False).encode("utf-8")
        exp_col2.download_button(
            "⬇ citations.csv",
            data=citations_csv,
            file_name="citations.csv",
            mime="text/csv",
        )

    _G = st.session_state.get("graph")
    if _G is None:
        from src.graph.engine import GraphEngine
        _G = GraphEngine().build_graph(
            st.session_state["papers_df"],
            st.session_state["citations_df"],
        )
        st.session_state["graph"] = _G
    _graphml_buf = io.BytesIO()
    nx.write_graphml(_G, _graphml_buf)
    exp_col3.download_button(
        "⬇ graph.graphml",
        data=_graphml_buf.getvalue(),
        file_name="graph.graphml",
        mime="application/xml",
    )

    _bibtex_str = _generate_bibtex(
        st.session_state["papers_df"],
        st.session_state.get("cluster_labels"),
    )
    exp_col4.download_button(
        "⬇ papers.bib",
        data=_bibtex_str.encode("utf-8"),
        file_name="papers.bib",
        mime="text/plain",
    )

else:
    st.info("Enter a query and click ▶ Run Pipeline to start.")
