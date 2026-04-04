import pandas as pd
import streamlit as st
from pathlib import Path

from src.analysis.scimago import ScimagoLoader

_scimago = ScimagoLoader()
_scimago.load("data/scimago.csv")

# ---------------------------------------------------------------------------
# Cached resource loaders — loaded once, reused across reruns
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_filter():
    from src.expansion.filter import RelevanceFilter
    return RelevanceFilter()


@st.cache_resource
def _get_client():
    from src.api.semantic_scholar import SemanticScholarClient
    from src.cache.cache_manager import DiskCache
    cache = DiskCache(cache_dir="cache")
    cache.load()
    return SemanticScholarClient(cache=cache)


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

# ---------------------------------------------------------------------------
# Sidebar — pipeline parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Pipeline Parameters")

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

    max_iterations = st.number_input("Max iterations", min_value=1, max_value=20, value=3, step=1)
    top_k_candidates = st.number_input("Top-K candidates", min_value=10, max_value=500, value=50, step=10)
    relevance_threshold = st.slider(
        "Relevance threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05
    )

    run_button = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

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

        year_range = (int(year_min), int(year_max)) if use_year_filter else None

        with st.status("Seeding corpus...", expanded=True) as status:
            from src.corpus.builder import CorpusBuilder
            client = _get_client()
            corpus = CorpusBuilder(client)
            corpus.seed(query.strip(), limit=int(limit), year_range=year_range)
            if _scimago.is_loaded and not corpus.papers_df.empty:
                corpus.papers_df = _scimago.enrich_papers(corpus.papers_df)
                st.session_state["available_quartiles"] = _scimago.available_quartiles(corpus.papers_df)
                st.session_state["scimago_loaded"] = True
            st.write(f"Seed complete — {len(corpus.papers_df)} papers")
            status.update(label="Seed complete", state="complete")

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
            corpus.fetch_references()
            _log(f"References fetched — {len(corpus.citations_df)} citation edges")

            # Expand
            _log("Expanding corpus...")
            expander = CorpusExpander(corpus._client, GraphEngine(), _load_filter())

            def _report_iteration(iteration, seed_count, refs_extracted, new_papers):
                _log(
                    f"Iteration {iteration}: "
                    f"{seed_count} seed papers | "
                    f"{refs_extracted} references extracted | "
                    f"{new_papers} new papers added"
                )

            expander.expand(
                corpus,
                query=query.strip(),
                max_iterations=int(max_iterations),
                top_k_candidates=int(top_k_candidates),
                relevance_threshold=float(relevance_threshold),
                allowed_domains=st.session_state.get("selected_domains") or None,
                on_iteration=_report_iteration,
            )
            _log(f"Expansion complete — {len(corpus.papers_df)} papers total")

            Path("data").mkdir(exist_ok=True)
            corpus.save_papers("data/papers.csv")
            if not corpus.citations_df.empty:
                corpus.save_citations("data/citations.csv")
            _log("Saved to data/papers.csv and data/citations.csv")

            st.session_state["papers_df"] = corpus.papers_df
            st.session_state["citations_df"] = corpus.citations_df
            st.session_state["corpus"] = corpus
            st.session_state["seed_done"] = False  # hide picker after completion
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

    col1, col2 = st.columns(2)
    col1.metric("Total papers", len(papers_df))
    col2.metric("Total citations", len(citations_df) if citations_df is not None else 0)

    st.subheader("Papers")
    st.caption("All papers collected during seeding and expansion, ranked by citation count. Higher citation counts indicate greater influence in the field.")
    display_cols = [
        c for c in ["title", "year", "journal", "citation_count", "authors"]
        if c in papers_df.columns
    ]
    st.dataframe(
        papers_df[display_cols].sort_values("citation_count", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------------------------
    # Influential Papers
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Influential Papers")
    st.caption(
        "Papers ranked by in-sample citations (ISC) — how often they are cited "
        "by other papers within this corpus. ISC ratio shows what fraction of a "
        "paper's total Semantic Scholar citations come from within the corpus. "
        "Betweenness centrality identifies papers that act as bridges between "
        "research clusters."
    )

    if st.session_state.get("influence_df") is None:
        from src.analysis.influence import InfluenceAnalyzer
        from src.graph.engine import GraphEngine

        ia = InfluenceAnalyzer()
        _inf_df = ia.compute_isc(papers_df, st.session_state["citations_df"])
        _G_inf = GraphEngine().build_graph(papers_df, st.session_state["citations_df"])
        _inf_df = ia.compute_betweenness(_G_inf, _inf_df)
        st.session_state["influence_df"] = _inf_df

    _inf_display = st.session_state["influence_df"]
    _inf_cols = [
        c for c in [
            "title", "year", "journal", "citation_count",
            "isc", "isc_ratio", "sample_relevance", "betweenness_centrality",
        ]
        if c in _inf_display.columns
    ]
    st.dataframe(
        _inf_display[_inf_cols].sort_values("isc", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------------------------
    # Co-citation Analysis
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Co-citation Analysis")
    st.caption(
        "Papers frequently cited together by the same sources are likely intellectually "
        "related, even without a direct citation link. AgglomerativeClustering groups "
        "co-cited papers into themes; node colour reflects cluster membership."
    )

    _cocit_col1, _cocit_col2 = st.columns(2)
    n_cocit_clusters = _cocit_col1.number_input(
        "Co-citation clusters",
        min_value=2,
        max_value=20,
        value=min(len(set(st.session_state["partition"].values())), 20) if st.session_state.get("partition") else 5,
        step=1,
        key="cocit_n_clusters",
    )
    min_cocitations = _cocit_col2.slider(
        "Min co-citations (edge threshold)",
        min_value=1,
        max_value=10,
        value=2,
        key="cocit_threshold",
    )

    # Build co-citation matrix once; re-cluster on n_clusters change (fast)
    if st.session_state.get("cocitation_df") is None:
        from src.analysis.cocitation import CoCitationAnalyzer
        _ca = CoCitationAnalyzer()
        st.session_state["cocitation_df"] = _ca.build_cocitation_matrix(
            citations_df, set(papers_df["paper_id"])
        )

    _coc_df = st.session_state["cocitation_df"]

    from src.analysis.cocitation import CoCitationAnalyzer
    _coc_clusters = CoCitationAnalyzer().cluster_papers(
        _coc_df, list(papers_df["paper_id"]), n_clusters=int(n_cocit_clusters)
    )

    # Filter edges by threshold
    _coc_filtered = _coc_df[_coc_df["cocitation_count"] >= min_cocitations] if not _coc_df.empty else _coc_df

    if _coc_filtered.empty:
        st.info("No paper pairs meet the minimum co-citation threshold.")
    else:
        from pyvis.network import Network
        import streamlit.components.v1 as components

        _title_lookup = dict(zip(papers_df["paper_id"], papers_df["title"]))

        _coc_net = Network(
            height="500px",
            width="100%",
            bgcolor="#0e1117",
            font_color="white",
            directed=False,
        )
        _coc_net.barnes_hut()

        # Collect nodes from filtered edges
        _coc_node_ids = set(_coc_filtered["paper_a"]) | set(_coc_filtered["paper_b"])
        for _pid in _coc_node_ids:
            _cid = _coc_clusters.get(_pid, -1)
            _color = _PALETTE[_cid % len(_PALETTE)] if _cid >= 0 else "#888888"
            _label = (_title_lookup.get(_pid) or "")[:40]
            _coc_net.add_node(
                _pid,
                label=_label,
                title=_title_lookup.get(_pid) or _pid,
                color=_color,
                size=15,
            )

        for _, _edge in _coc_filtered.iterrows():
            _width = min(1 + _edge["cocitation_count"] * 0.5, 10)
            _coc_net.add_edge(
                _edge["paper_a"],
                _edge["paper_b"],
                value=float(_width),
            )

        _coc_html = _coc_net.generate_html()
        components.html(_coc_html, height=510, scrolling=False)

    # -----------------------------------------------------------------------
    # Bibliographic Coupling
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Bibliographic Coupling")
    st.caption(
        "Papers that cite the same sources share intellectual foundations and are "
        "likely working in the same research area, even without directly referencing "
        "each other. Edge weight reflects the number of shared references."
    )

    _bib_col1, _bib_col2 = st.columns(2)
    n_bib_clusters = _bib_col1.number_input(
        "Coupling clusters",
        min_value=2,
        max_value=20,
        value=min(len(set(st.session_state["partition"].values())), 20) if st.session_state.get("partition") else 5,
        step=1,
        key="bib_n_clusters",
    )
    min_coupling = _bib_col2.slider(
        "Min shared references (edge threshold)",
        min_value=1,
        max_value=10,
        value=2,
        key="bib_threshold",
    )

    if st.session_state.get("bibcoupling_df") is None:
        from src.analysis.bibcoupling import BibliographicCoupler
        _bc = BibliographicCoupler()
        st.session_state["bibcoupling_df"] = _bc.build_coupling_matrix(
            citations_df, set(papers_df["paper_id"])
        )

    _bib_df = st.session_state["bibcoupling_df"]

    from src.analysis.bibcoupling import BibliographicCoupler
    _bib_clusters = BibliographicCoupler().cluster_papers(
        _bib_df, list(papers_df["paper_id"]), n_clusters=int(n_bib_clusters)
    )

    _bib_filtered = _bib_df[_bib_df["coupling_strength"] >= min_coupling] if not _bib_df.empty else _bib_df

    if _bib_filtered.empty:
        st.info("No paper pairs meet the minimum shared-reference threshold.")
    else:
        from pyvis.network import Network
        import streamlit.components.v1 as components

        _title_lookup_bib = dict(zip(papers_df["paper_id"], papers_df["title"]))

        _bib_net = Network(
            height="500px",
            width="100%",
            bgcolor="#0e1117",
            font_color="white",
            directed=False,
        )
        _bib_net.barnes_hut()

        _bib_node_ids = set(_bib_filtered["paper_a"]) | set(_bib_filtered["paper_b"])
        for _pid in _bib_node_ids:
            _cid = _bib_clusters.get(_pid, -1)
            _color = _PALETTE[_cid % len(_PALETTE)] if _cid >= 0 else "#888888"
            _label = (_title_lookup_bib.get(_pid) or "")[:40]
            _bib_net.add_node(
                _pid,
                label=_label,
                title=_title_lookup_bib.get(_pid) or _pid,
                color=_color,
                size=15,
            )

        for _, _edge in _bib_filtered.iterrows():
            _width = min(1 + _edge["coupling_strength"] * 0.5, 10)
            _bib_net.add_edge(
                _edge["paper_a"],
                _edge["paper_b"],
                value=float(_width),
            )

        _bib_html = _bib_net.generate_html()
        components.html(_bib_html, height=510, scrolling=False)

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
        from src.graph.engine import GraphEngine

        _G_cl = GraphEngine().build_graph(papers_df, citations_df)
        _ce = ClusterEngine()
        _partition = _ce.detect_communities(_G_cl, resolution=float(cluster_resolution))
        _partition = _ce.merge_small_clusters(_partition, papers_df, citations_df, min_size=int(min_cluster_size))
        papers_df = _ce.add_cluster_assignments(papers_df, _partition)
        _cluster_labels = _ce.label_clusters(papers_df, _partition)
        _cluster_summary = _ce.cluster_summary_df(papers_df, _cluster_labels)

        st.session_state["papers_df"] = papers_df
        st.session_state["partition"] = _partition
        st.session_state["cluster_labels"] = _cluster_labels
        st.session_state["cluster_summary"] = _cluster_summary
        st.session_state["cluster_params"] = _current_params

    if st.session_state["cluster_summary"] is not None:
        st.dataframe(
            st.session_state["cluster_summary"],
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------------------------
    # Knowledge Graph
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Knowledge Graph")
    st.caption("Citation network visualisation. Node size reflects citation count; node colour reflects cluster membership. Edges point from citing paper to cited paper.")

    max_nodes = st.slider(
        "Max nodes", min_value=20, max_value=500, value=100, step=20,
        key="graph_max_nodes",
    )

    papers_for_graph = (
        st.session_state["papers_df"]
        .sort_values("citation_count", ascending=False)
        .head(max_nodes)
    )
    node_ids = set(papers_for_graph["paper_id"])

    from pyvis.network import Network
    import streamlit.components.v1 as components

    net = Network(
        height="650px", width="100%",
        bgcolor="#0e1117", font_color="white",
        directed=True,
    )
    net.barnes_hut()

    max_cit = max(papers_for_graph["citation_count"].max(), 1)

    for _, row in papers_for_graph.iterrows():
        cid = int(row["cluster_id"]) if pd.notna(row.get("cluster_id")) else -1
        color = _PALETTE[cid % len(_PALETTE)] if cid >= 0 else "#888888"
        size = 10 + 30 * (row["citation_count"] / max_cit)
        label = (row["title"] or "")[:40]
        net.add_node(
            row["paper_id"],
            label=label,
            title=f"{row['title']}\nYear: {row.get('year', '?')}\nCitations: {row['citation_count']}",
            color=color,
            size=float(size),
        )

    citations_df = st.session_state["citations_df"]
    if citations_df is not None and not citations_df.empty:
        for _, edge in citations_df.iterrows():
            if edge["source"] in node_ids and edge["target"] in node_ids:
                net.add_edge(edge["source"], edge["target"])

    html = net.generate_html()
    components.html(html, height=660, scrolling=False)

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
            _labels.get(int(c), f"cluster-{int(c)}")
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
    from src.graph.engine import GraphEngine

    exp_col1, exp_col2, exp_col3 = st.columns(3)

    papers_csv = st.session_state["papers_df"].to_csv(index=False).encode("utf-8")
    exp_col1.download_button(
        "⬇ papers.csv",
        data=papers_csv,
        file_name="papers.csv",
        mime="text/csv",
    )

    citations_csv = st.session_state["citations_df"].to_csv(index=False).encode("utf-8")
    exp_col2.download_button(
        "⬇ citations.csv",
        data=citations_csv,
        file_name="citations.csv",
        mime="text/csv",
    )

    _G = GraphEngine().build_graph(
        st.session_state["papers_df"],
        st.session_state["citations_df"],
    )
    _graphml_buf = io.BytesIO()
    nx.write_graphml(_G, _graphml_buf)
    exp_col3.download_button(
        "⬇ graph.graphml",
        data=_graphml_buf.getvalue(),
        file_name="graph.graphml",
        mime="application/xml",
    )

else:
    st.info("Enter a query and click ▶ Run Pipeline to start.")
