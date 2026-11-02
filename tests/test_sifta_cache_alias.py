"""A gemini-3.8-flash request finds the response cached for the same gemini-3.7-flash request."""
from types import SimpleNamespace

from dspy.clients.cache import Cache

import workflow_sifta  # noqa: F401 -- aliases dspy.cache on import
import dspy
from workflow_sifta.runtime.usage import CACHE_KEY_IGNORED, alias_cache_keys, cache_request


def _memory_cache() -> Cache:
    return Cache(enable_disk_cache=False, enable_memory_cache=True, disk_cache_dir="",
                 disk_size_limit_bytes=0, memory_max_entries=100)


def _request(model: str) -> dict:
    return cache_request({"model": model, "messages": [{"role": "user", "content": "hi"}], "temperature": 1.0})


def test_a_38_request_is_keyed_as_the_37_request():
    cache = _memory_cache()
    alias_cache_keys(cache)
    k37 = cache.cache_key(_request("gemini/gemini-3.7-flash"), CACHE_KEY_IGNORED)
    assert cache.cache_key(_request("gemini/gemini-3.8-flash"), CACHE_KEY_IGNORED) == k37
    assert cache.cache_key(_request("gemini/gemini-2.5-flash"), CACHE_KEY_IGNORED) != k37
    assert cache.cache_key(_request("openai/gpt-4.1-mini"), CACHE_KEY_IGNORED) != k37


def test_a_38_request_gets_the_stored_37_response():
    cache = _memory_cache()
    alias_cache_keys(cache)
    cache.put(_request("gemini/gemini-3.7-flash"), SimpleNamespace(usage={}, text="stored"), CACHE_KEY_IGNORED)
    assert cache.get(_request("gemini/gemini-3.8-flash"), CACHE_KEY_IGNORED).text == "stored"
    assert cache.get(_request("gemini/gemini-2.5-flash"), CACHE_KEY_IGNORED) is None


def test_the_alias_does_not_change_the_request():
    cache = _memory_cache()
    alias_cache_keys(cache)
    request = _request("gemini/gemini-3.8-flash")
    cache.cache_key(request, CACHE_KEY_IGNORED)
    assert request["model"] == "gemini/gemini-3.8-flash"


def test_the_package_aliases_the_dspy_cache():
    key = dspy.cache.cache_key
    assert key(_request("gemini/gemini-3.8-flash"), CACHE_KEY_IGNORED) == key(_request("gemini/gemini-3.7-flash"),
                                                                              CACHE_KEY_IGNORED)
