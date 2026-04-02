"""
LLM Provider Factory - Centralized LLM construction for multiple providers.

Supports:
- Ollama (local models via langchain-ollama)
- OpenAI (GPT models via langchain-openai)
- Anthropic (Claude models via langchain-anthropic)
- Google Vertex AI (Gemini models via langchain-google-vertexai)

All functions return BaseChatModel, the common LangChain interface.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider instance."""
    provider: str = "ollama"
    model: str = "qwen3:30b"
    temperature: float = 0.0
    base_url: Optional[str] = None
    api_key: Optional[str] = None


# Module-level cache keyed by (provider, model, base_url, temperature)
_llm_cache: Dict[str, BaseChatModel] = {}


def _cache_key(config: ProviderConfig) -> str:
    return f"{config.provider}:{config.model}:{config.base_url}:{config.temperature}"


def create_llm(config: ProviderConfig) -> BaseChatModel:
    """
    Create a new BaseChatModel instance for the given provider config.

    Provider packages are imported lazily so only the selected provider
    needs to be installed.
    """
    match config.provider:
        case "ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError:
                raise ImportError(
                    "langchain-ollama is required for the ollama provider. "
                    "Install with: pip install langchain-ollama"
                )
            kwargs = {
                "model": config.model,
                "temperature": config.temperature,
            }
            if config.base_url:
                kwargs["base_url"] = config.base_url
            return ChatOllama(**kwargs)

        case "openai":
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError(
                    "langchain-openai is required for the openai provider. "
                    "Install with: pip install langchain-openai"
                )
            kwargs = {
                "model": config.model,
                "temperature": config.temperature,
            }
            if config.api_key:
                kwargs["api_key"] = config.api_key
            if config.base_url:
                kwargs["base_url"] = config.base_url
            return ChatOpenAI(**kwargs)

        case "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError:
                raise ImportError(
                    "langchain-anthropic is required for the anthropic provider. "
                    "Install with: pip install langchain-anthropic"
                )
            kwargs = {
                "model": config.model,
                "temperature": config.temperature,
            }
            if config.api_key:
                kwargs["api_key"] = config.api_key
            return ChatAnthropic(**kwargs)

        case "google_vertexai":
            try:
                from langchain_google_vertexai import ChatVertexAI
            except ImportError:
                raise ImportError(
                    "langchain-google-vertexai is required for the google_vertexai provider. "
                    "Install with: pip install langchain-google-vertexai"
                )
            kwargs = {
                "model": config.model,
                "temperature": config.temperature,
            }
            return ChatVertexAI(**kwargs)

        case _:
            raise ValueError(
                f"Unknown LLM provider: '{config.provider}'. "
                f"Supported: ollama, openai, anthropic, google_vertexai"
            )


def get_llm(config: ProviderConfig) -> BaseChatModel:
    """Get or create a cached BaseChatModel instance."""
    key = _cache_key(config)
    if key not in _llm_cache:
        logger.info(f"Creating {config.provider} LLM: {config.model}")
        _llm_cache[key] = create_llm(config)
    return _llm_cache[key]


def get_llm_from_deps(deps: dict, model_key: str = "llm") -> BaseChatModel:
    """
    Convenience: build a ProviderConfig from a Deps dict and return a cached LLM.

    Reads provider, model name, base_url, and api_key from deps,
    falling back to DEFAULTS for any missing values.
    """
    from .state import DEFAULTS

    config = ProviderConfig(
        provider=deps.get("provider", DEFAULTS["provider"]),
        model=deps.get(model_key, DEFAULTS.get(model_key, DEFAULTS["llm"])),
        base_url=deps.get("base_url", DEFAULTS.get("base_url")),
        api_key=deps.get("api_key", DEFAULTS.get("api_key", "")),
    )
    return get_llm(config)


def clear_cache():
    """Clear the LLM instance cache."""
    _llm_cache.clear()
