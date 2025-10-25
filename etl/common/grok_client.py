import os
import socket
from typing import Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")
GROK_MODEL = os.getenv("GROK_MODEL")

_HTTP_CLIENT: Optional[httpx.Client] = None
_GROK_CLIENT: Optional[OpenAI] = None


def _is_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def _detect_proxy_url() -> Optional[str]:
    if _is_port_open("127.0.0.1", 53425):
        return "socks5h://127.0.0.1:53425"
    if _is_port_open("127.0.0.1", 53424):
        return "http://127.0.0.1:53424"
    return os.getenv("ALL_PROXY")


def _get_http_client() -> httpx.Client:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        proxy = _detect_proxy_url()
        _HTTP_CLIENT = httpx.Client(
            proxy=proxy,
            timeout=30.0,
            trust_env=False,
            follow_redirects=True,
        )
    return _HTTP_CLIENT


def get_grok_client() -> OpenAI:
    global _GROK_CLIENT
    if _GROK_CLIENT is None:
        if not XAI_API_KEY:
            raise RuntimeError("XAI_API_KEY is not set in environment")
        _GROK_CLIENT = OpenAI(
            api_key=XAI_API_KEY,
            base_url="https://api.x.ai/v1",
            http_client=_get_http_client(),
        )
    return _GROK_CLIENT


__all__ = ["get_grok_client", "XAI_API_KEY", "GROK_MODEL"]
