"""Query interface UI component."""

import streamlit as st

from api_client import RAGClient


def render_query_tab(client: RAGClient) -> None:
    """Render the query tab with workflow execution and RAG search."""

    # --- Workflow Query ---
    st.header("Query the RAG System")

    query = st.text_area("Enter your question", height=100, key="workflow_query")

    with st.expander("Options"):
        workflow_type = st.radio(
            "Workflow",
            ["simple", "full"],
            horizontal=True,
            help="'simple' is faster; 'full' includes a verification loop",
        )
        web_search = st.checkbox("Enable web search")
        data_path = st.text_input("Data path (optional)", help="Path to a data file for code execution")

    if st.button("Run Query", key="run_workflow") and query:
        with st.spinner("Running workflow..."):
            try:
                result = client.query_workflow(
                    query=query,
                    workflow_type=workflow_type,
                    data_path=data_path or None,
                    web_search_enabled=web_search,
                )
                st.markdown("### Summary")
                st.markdown(result["summary"])
                st.caption(
                    f"Workflow: {result['workflow_used']} | "
                    f"Time: {result['execution_time_seconds']}s"
                )
            except Exception as e:
                st.error(f"Workflow failed: {e}")

    # --- Direct RAG Search ---
    st.divider()
    st.header("Direct RAG Search")

    rag_query = st.text_input("Search query", key="rag_search_query")
    n_results = st.slider("Max results", 1, 50, 10, key="rag_n_results")

    if st.button("Search RAG", key="run_rag_search") and rag_query:
        with st.spinner("Searching..."):
            try:
                result = client.rag_search(query=rag_query, n_results=n_results)
                st.caption(f"Found {result['total_results']} results")
                for i, r in enumerate(result["results"], 1):
                    with st.expander(f"Result {i} -- {r['metadata'].get('title', 'N/A')}"):
                        st.text(r["text"][:2000])
                        st.json(r["metadata"])
            except Exception as e:
                st.error(f"Search failed: {e}")
