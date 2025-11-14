
from typing import Optional, List, Dict
from openai import OpenAI
from etl.common.grok_client import get_grok_client, GROK_MODEL, get_current_proxy

_client: Optional[OpenAI] = None
_DEFAULT_MODEL = GROK_MODEL or "grok-3-mini"


def _client_singleton() -> OpenAI:

    global _client
    if _client is None:
        _client = get_grok_client()
    print(get_current_proxy())
    return _client


def generate(
    messages: List[Dict],
    temperature: float = 0.4,
    max_tokens: int = 800,
    model: Optional[str] = None,
) -> str:
    cli = _client_singleton()
    resp = cli.chat.completions.create(
        model=model or _DEFAULT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()
