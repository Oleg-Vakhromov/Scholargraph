import pandas as pd
import streamlit as st
from pathlib import Path


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
    st.subheader("Expansion")

    max_iterations = st.number_input("Max iterations", min_value=1, max_value=20, value=3, step=1)
    top_k_candidates = st.number_input("Top-K candidates", min_value=10, max_value=500, value=50, step=10)
    relevance_threshold = st.slider(
        "Relevance threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05
    )

    run_button = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run button handler
# ---------------------------------------------------------------------------

if run_button:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        year_range = (int(year_min), int(year_max)) if use_year_filter else None

        with st.status("Running pipeline...", expanded=True) as status:
            from src.corpus.builder import CorpusBuilder
            from src.expansion.expander import CorpusExpander
            from src.graph.engine import GraphEngine

            client = _get_client()
            corpus = CorpusBuilder(client)

            st.write("Seeding corpus...")
            corpus.seed(query.strip(), limit=int(limit), year_range=year_range)
            st.write(f"Seed complete — {len(corpus.papers_df)} papers")

            st.write("Fetching references...")
            corpus.fetch_references()
            st.write(f"References fetched — {len(corpus.citations_df)} citation edges")

            st.write("Expanding corpus...")
            expander = CorpusExpander(client, GraphEngine(), _load_filter())
            expander.expand(
                corpus,
                query=query.strip(),
                max_iterations=int(max_iterations),
                top_k_candidates=int(top_k_candidates),
                relevance_threshold=float(relevance_threshold),
            )
            st.write(f"Expansion complete — {len(corpus.papers_df)} papers total")

            Path("data").mkdir(exist_ok=True)
            corpus.save_papers("data/papers.csv")
            if not corpus.citations_df.empty:
                corpus.save_citations("data/citations.csv")
            st.write("Saved to data/papers.csv and data/citations.csv")

            st.session_state["papers_df"] = corpus.papers_df
            st.session_state["citations_df"] = corpus.citations_df
            st.session_state["corpus"] = corpus
            status.update(label="Pipeline complete!", state="complete")

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

if st.session_state["papers_df"] is not None:
    papers_df = st.session_state["papers_df"]
    citations_df = st.session_state["citations_df"]

    col1, col2 = st.columns(2)
    col1.metric("Total papers", len(papers_df))
    col2.metric("Total citations", len(citations_df) if citations_df is not None else 0)

    st.subheader("Papers")
    display_cols = [
        c for c in ["title", "year", "citation_count", "authors"]
        if c in papers_df.columns
    ]
    st.dataframe(
        papers_df[display_cols].sort_values("citation_count", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------------------------
    # Cluster analysis
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Clusters")

    if "cluster_id" not in papers_df.columns:
        from src.analysis.cluster import ClusterEngine
        from src.graph.engine import GraphEngine

        G = GraphEngine().build_graph(papers_df, st.session_state["citations_df"])
        ce = ClusterEngine()
        partition = ce.detect_communities(G)
        papers_df = ce.add_cluster_assignments(papers_df, partition)
        cluster_labels = ce.label_clusters(papers_df, partition)
        cluster_summary = ce.cluster_summary_df(papers_df, cluster_labels)

        st.session_state["papers_df"] = papers_df
        st.session_state["partition"] = partition
        st.session_state["cluster_labels"] = cluster_labels
        st.session_state["cluster_summary"] = cluster_summary

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

    _PALETTE = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4",
    ]

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
