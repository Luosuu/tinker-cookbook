"""Pluggable web-search / browse providers for the search-tool recipes.

A provider exposes two async primitives:
  - ``search(query, n_results)`` -> list of {title, snippet, url}
  - ``browse(url, max_chars)``   -> cleaned page text (truncated)

Providers are selected by name via :func:`get_provider`, reading the relevant
API key from the environment. Keeping keys out of the objects' identity (they
read env at call time via the factory, but store the key as a plain field)
means the higher-level tools stay pickleable as long as they reconstruct the
provider inside ``make_envs``.

Currently supported (verified live):
  - ``serper``   : Serper.dev  (google.serper.dev/search + scrape.serper.dev)
  - ``tavily``   : Tavily      (/search + /extract)
  - ``parallel`` : Parallel AI (/v1/search + /v1/extract)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import httpx

# ---------------------------------------------------------------------------
# Result type + protocol
# ---------------------------------------------------------------------------


@dataclass
class SearchHit:
    title: str
    snippet: str
    url: str


class SearchProvider(Protocol):
    """A web-search backend with search + browse primitives."""

    name: str

    async def search(self, query: str, n_results: int) -> list[SearchHit]: ...

    async def browse(self, url: str, max_chars: int) -> str: ...


# ---------------------------------------------------------------------------
# Serper.dev
# ---------------------------------------------------------------------------


class SerperProvider:
    name = "serper"
    _SEARCH_URL = "https://google.serper.dev/search"
    _SCRAPE_URL = "https://scrape.serper.dev"

    def __init__(self, api_key: str, timeout: float = 30.0):
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

    async def search(self, query: str, n_results: int) -> list[SearchHit]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self._SEARCH_URL, headers=self._headers(), json={"q": query, "num": n_results}
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            SearchHit(r.get("title", ""), r.get("snippet", ""), r.get("link", ""))
            for r in data.get("organic", [])[:n_results]
        ]

    async def browse(self, url: str, max_chars: int) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self._SCRAPE_URL, headers=self._headers(), json={"url": url}
            )
            resp.raise_for_status()
            data = resp.json()
        return (data.get("text", "") or "")[:max_chars]


# ---------------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------------


class TavilyProvider:
    name = "tavily"
    _SEARCH_URL = "https://api.tavily.com/search"
    _EXTRACT_URL = "https://api.tavily.com/extract"

    def __init__(self, api_key: str, timeout: float = 40.0):
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def search(self, query: str, n_results: int) -> list[SearchHit]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self._SEARCH_URL,
                headers=self._headers(),
                json={"query": query, "max_results": n_results},
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            SearchHit(r.get("title", ""), r.get("content", ""), r.get("url", ""))
            for r in data.get("results", [])[:n_results]
        ]

    async def browse(self, url: str, max_chars: int) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self._EXTRACT_URL,
                headers=self._headers(),
                json={"urls": [url], "format": "text"},
            )
            resp.raise_for_status()
            data = resp.json()
        results = data.get("results", [])
        if not results:
            return ""
        return (results[0].get("raw_content", "") or "")[:max_chars]


# ---------------------------------------------------------------------------
# Parallel AI  (/v1/search + /v1/extract)
# ---------------------------------------------------------------------------


class ParallelProvider:
    name = "parallel"
    _SEARCH_URL = "https://api.parallel.ai/v1/search"
    _EXTRACT_URL = "https://api.parallel.ai/v1/extract"

    def __init__(self, api_key: str, timeout: float = 60.0):
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key, "Content-Type": "application/json"}

    async def search(self, query: str, n_results: int) -> list[SearchHit]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self._SEARCH_URL,
                headers=self._headers(),
                json={"objective": query, "search_queries": [query]},
            )
            resp.raise_for_status()
            data = resp.json()
        hits: list[SearchHit] = []
        for r in data.get("results", [])[:n_results]:
            excerpts = r.get("excerpts", []) or []
            snippet = " ".join(excerpts)[:500]
            hits.append(SearchHit(r.get("title", ""), snippet, r.get("url", "")))
        return hits

    async def browse(self, url: str, max_chars: int) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self._EXTRACT_URL,
                headers=self._headers(),
                json={"urls": [url], "advanced_settings": {"full_content": True}},
            )
            resp.raise_for_status()
            data = resp.json()
        results = data.get("results", [])
        if not results:
            return ""
        r = results[0]
        # full_content holds the whole page; excerpts is a fallback.
        text = r.get("full_content") or " ".join(r.get("excerpts", []) or [])
        return (text or "")[:max_chars]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDER_ENV = {
    "serper": "SERPER_API_KEY",
    "tavily": "TAVILY_API_KEY",
    "parallel": "PARALLEL_API_KEY",
}

_PROVIDER_CLS = {
    "serper": SerperProvider,
    "tavily": TavilyProvider,
    "parallel": ParallelProvider,
}


def available_providers() -> list[str]:
    return list(_PROVIDER_CLS.keys())


def provider_env_var(name: str) -> str:
    """Return the environment-variable name holding ``name``'s API key."""
    key_name = _PROVIDER_ENV.get(name.lower())
    if key_name is None:
        raise ValueError(
            f"Unknown search provider {name!r}; choose from {available_providers()}"
        )
    return key_name


def get_provider(name: str, api_key: str | None = None) -> SearchProvider:
    """Construct a provider by name. Reads the key from the provider's env var
    when ``api_key`` is not given."""
    name = name.lower()
    if name not in _PROVIDER_CLS:
        raise ValueError(
            f"Unknown search provider {name!r}; choose from {available_providers()}"
        )
    key = api_key or os.environ.get(_PROVIDER_ENV[name])
    if not key:
        raise RuntimeError(
            f"Provider {name!r} needs {_PROVIDER_ENV[name]} set in the environment."
        )
    return _PROVIDER_CLS[name](key)  # type: ignore[abstract]
