"""Document upload UI component."""

import streamlit as st

from api_client import RAGClient


def render_upload_tab(client: RAGClient) -> None:
    """Render the document upload tab."""
    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose files to add to the RAG store",
        type=["txt", "md", "pdf", "pptx"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Upload"):
        for f in uploaded_files:
            with st.spinner(f"Uploading {f.name}..."):
                try:
                    result = client.upload_document(f.name, f.getvalue())
                    st.success(
                        f"{result['filename']} -- "
                        f"{result['source_type']} -- "
                        f"{result['chunks_added']} chunks"
                    )
                except Exception as e:
                    st.error(f"Failed to upload {f.name}: {e}")

    # --- Document list ---
    st.divider()
    st.subheader("Documents in RAG Store")

    if st.button("Refresh List"):
        st.rerun()

    try:
        data = client.list_documents()
        docs = data.get("documents", [])
        if docs:
            st.table(docs)
        else:
            st.info("No documents uploaded yet.")
    except Exception as e:
        st.warning(f"Could not fetch document list: {e}")
