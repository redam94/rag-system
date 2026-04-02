"""HTTP client wrapper for the RAG System FastAPI backend."""

from typing import Any, Dict, List, Optional

import httpx


class RAGClient:
    """Thin wrapper around httpx for communicating with the RAG API."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", session_id: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id

    def _headers(self) -> Dict[str, str]:
        if self.session_id:
            return {"X-Session-ID": self.session_id}
        return {}

    # --- Session ---

    def create_session(self) -> Dict[str, Any]:
        resp = httpx.post(f"{self.base_url}/api/config/session", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data["session_id"]
        return data

    # --- Health ---

    def health(self) -> Dict[str, Any]:
        resp = httpx.get(f"{self.base_url}/api/health", headers=self._headers(), timeout=5)
        resp.raise_for_status()
        return resp.json()

    # --- LLM Config ---

    def get_config(self) -> Dict[str, Any]:
        resp = httpx.get(f"{self.base_url}/api/config/llm", headers=self._headers(), timeout=5)
        resp.raise_for_status()
        return resp.json()

    def update_config(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        code_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if provider is not None:
            body["provider"] = provider
        if model is not None:
            body["model"] = model
        if code_model is not None:
            body["code_model"] = code_model
        if vision_model is not None:
            body["vision_model"] = vision_model
        if base_url is not None:
            body["base_url"] = base_url
        if api_key is not None:
            body["api_key"] = api_key
        resp = httpx.post(
            f"{self.base_url}/api/config/llm",
            json=body,
            headers=self._headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    # --- API Key ---

    def set_api_key(self, api_key: str) -> Dict[str, Any]:
        resp = httpx.post(
            f"{self.base_url}/api/config/api-key",
            json={"api_key": api_key},
            headers=self._headers(),
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json()

    def clear_api_key(self) -> Dict[str, Any]:
        resp = httpx.delete(
            f"{self.base_url}/api/config/api-key",
            headers=self._headers(),
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json()

    # --- Documents ---

    def upload_document(self, filename: str, content: bytes) -> Dict[str, Any]:
        resp = httpx.post(
            f"{self.base_url}/api/documents/upload",
            files={"file": (filename, content)},
            headers=self._headers(),
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def list_documents(self) -> Dict[str, Any]:
        resp = httpx.get(
            f"{self.base_url}/api/documents",
            headers=self._headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    # --- Query ---

    def rag_search(
        self,
        query: str,
        n_results: int = 10,
        doc_types: Optional[List[str]] = None,
        document_titles: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"query": query, "n_results": n_results}
        if doc_types:
            body["doc_types"] = doc_types
        if document_titles:
            body["document_titles"] = document_titles
        resp = httpx.post(
            f"{self.base_url}/api/query/rag",
            json=body,
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def query_workflow(
        self,
        query: str,
        workflow_type: str = "simple",
        data_path: Optional[str] = None,
        web_search_enabled: bool = False,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "query": query,
            "workflow_type": workflow_type,
            "web_search_enabled": web_search_enabled,
        }
        if data_path:
            body["data_path"] = data_path
        resp = httpx.post(
            f"{self.base_url}/api/query/workflow",
            json=body,
            headers=self._headers(),
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()
