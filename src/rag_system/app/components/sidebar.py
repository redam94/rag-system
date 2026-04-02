"""Sidebar component: LLM configuration, API key management, connection status."""

import streamlit as st

from api_client import RAGClient


PROVIDERS = ["ollama", "openai", "anthropic", "google_vertexai"]


def render_sidebar(client: RAGClient) -> None:
    """Render the sidebar with LLM config and API key controls."""
    st.sidebar.header("LLM Configuration")

    # Fetch current config
    try:
        config = client.get_config()
    except Exception:
        config = {
            "provider": "ollama",
            "model": "",
            "code_model": "",
            "vision_model": "",
            "base_url": "",
            "api_key_set": False,
        }

    # Provider
    current_idx = PROVIDERS.index(config["provider"]) if config["provider"] in PROVIDERS else 0
    provider = st.sidebar.selectbox("Provider", PROVIDERS, index=current_idx)

    # Model names
    model = st.sidebar.text_input("Model", value=config.get("model", ""))
    code_model = st.sidebar.text_input("Code Model", value=config.get("code_model", ""))
    vision_model = st.sidebar.text_input("Vision Model", value=config.get("vision_model", ""))

    # Base URL (relevant for ollama / openai)
    if provider in ("ollama", "openai"):
        base_url = st.sidebar.text_input("Base URL", value=config.get("base_url", ""))
    else:
        base_url = None

    # Apply config
    if st.sidebar.button("Apply Configuration"):
        try:
            result = client.update_config(
                provider=provider,
                model=model or None,
                code_model=code_model or None,
                vision_model=vision_model or None,
                base_url=base_url,
            )
            st.sidebar.success(f"Config updated: {result['provider']} / {result['model']}")
        except Exception as e:
            st.sidebar.error(f"Failed to update config: {e}")

    # --- API Key ---
    st.sidebar.divider()
    st.sidebar.subheader("API Key")

    if config.get("api_key_set"):
        st.sidebar.success("API key is set")
    else:
        st.sidebar.warning("No API key set")

    api_key = st.sidebar.text_input("API Key", type="password", key="api_key_input")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Set Key", key="set_key_btn"):
            if api_key:
                try:
                    client.set_api_key(api_key)
                    st.sidebar.success("Key set")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(str(e))
            else:
                st.sidebar.warning("Enter a key first")
    with col2:
        if st.button("Clear Key", key="clear_key_btn"):
            try:
                client.clear_api_key()
                st.sidebar.info("Key cleared")
                st.rerun()
            except Exception as e:
                st.sidebar.error(str(e))

    # --- Connection status ---
    st.sidebar.divider()
    try:
        h = client.health()
        st.sidebar.success(f"Connected | RAG: {'on' if h['rag_enabled'] else 'off'}")
    except Exception:
        st.sidebar.error("Cannot reach API server")
