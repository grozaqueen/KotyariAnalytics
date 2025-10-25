import httpx
from openai import OpenAI
from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, ALL_PROXY

_client = None

def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        http_client = httpx.Client(
            proxy=ALL_PROXY, timeout=40.0, trust_env=(ALL_PROXY is None)
        )
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, http_client=http_client)
    return _client

def generate(messages: list[dict], temperature: float = 0.4, max_tokens: int = 800) -> str:
    cli = _client_singleton()
    resp = cli.chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return (resp.choices[0].message.content or "").strip()
