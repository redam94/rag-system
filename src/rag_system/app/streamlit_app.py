"""RAG System Demo -- Streamlit application.

Run with:
    uv run streamlit run src/rag_system/app/streamlit_app.py
"""

import streamlit as st

from api_client import RAGClient
from components.sidebar import render_sidebar
from components.document_upload import render_upload_tab
from components.query_interface import render_query_tab

st.set_page_config(page_title="RAG System Demo", layout="wide")


def _get_client() -> RAGClient:
    """Get or create the API client stored in session state."""
    if "client" not in st.session_state:
        base_url = st.session_state.get("api_base_url", "http://127.0.0.1:8000")
        st.session_state.client = RAGClient(base_url=base_url)
    client: RAGClient = st.session_state.client

    # Ensure we have a session
    if client.session_id is None:
        try:
            client.create_session()
        except Exception:
            pass  # Will show as disconnected in sidebar
    return client


def main() -> None:
    st.title("RAG System Demo")
    client = _get_client()

    # Sidebar
    render_sidebar(client)

    # Main content tabs
    tab_upload, tab_query = st.tabs(["Upload Documents", "Query"])

    with tab_upload:
        render_upload_tab(client)

    with tab_query:
        render_query_tab(client)


if __name__ == "__main__":
    main()
else:
    # When run via `streamlit run`, Streamlit executes the module top-level
    main()
