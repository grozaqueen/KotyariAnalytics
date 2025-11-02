import os
import socket
from typing import Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-beta")

_GROK_CLIENT: Optional[OpenAI] = None


def _is_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def _detect_proxy_url():
    return "http://xray-proxy:8100"


def get_grok_client() -> OpenAI:
    global _GROK_CLIENT
    if _GROK_CLIENT is None:
        proxy_url = _detect_proxy_url()

        if not proxy_url:
            raise RuntimeError(
                "Grok API недоступен без прокси в вашем регионе. "
                "Установите VLESS_PROXY_URL или ALL_PROXY в .env, "
                "или запустите локальный socks5/http прокси на порту 53425/53424/8100."
            )

        print(f"[grok_client] using proxy: {proxy_url}")

        http_client = httpx.Client(
            proxy=proxy_url,
            timeout=40.0,
            trust_env=False,
            follow_redirects=True,
        )

        _GROK_CLIENT = OpenAI(
            api_key=XAI_API_KEY,
            base_url="https://api.x.ai/v1",
            http_client=http_client,
        )

    return _GROK_CLIENT


__all__ = ["get_grok_client", "XAI_API_KEY", "GROK_MODEL"]
