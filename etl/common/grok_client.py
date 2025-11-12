from typing import Dict, Optional
import httpx
from openai import OpenAI
from dataclasses import dataclass
import threading

from dotenv import load_dotenv
import os
load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")
GROK_MODEL = os.getenv("GROK_MODEL")

@dataclass
class ProxyServerConfig:
    """Конфигурация прокси сервера"""
    host: str
    port: int


class ProxyConfig:
    """Конфигурация прокси с валидацией"""

    def __init__(self, config: Dict):
        proxy_server = config.get('proxy_server', {})
        self.proxy_server = ProxyServerConfig(
            host=proxy_server.get('host', ''),
            port=proxy_server.get('port', 0)
        )

    def validate(self) -> None:
        """Валидация конфигурации прокси"""
        if not 1 <= self.proxy_server.port <= 65535:
            raise ValueError(f"Invalid API port: {self.proxy_server.port}")

        if not self.proxy_server.host:
            raise ValueError("proxy host should be presented in config")


class GrokClient:
    """Клиент для работы с Grok API через прокси (Singleton)"""

    # Константы для Grok API
    GROK_BASE_URL = "https://api.x.ai/v1"
    DEFAULT_MODEL = "grok-3-mini"
    DEFAULT_TIMEOUT = 30.0

    _instance: Optional['GrokClient'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Thread-safe singleton implementation"""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
            self,
            api_key: str,
            proxy_config: ProxyConfig,
            timeout: float = DEFAULT_TIMEOUT,
            model: str = DEFAULT_MODEL
    ):
        """
        Инициализация клиента Grok с прокси

        Args:
            api_key: API ключ для Grok
            proxy_config: Конфигурация прокси
            timeout: Таймаут запросов в секундах
            model: Модель по умолчанию
        """
        # Проверка, что инициализация выполняется только один раз
        if hasattr(self, '_initialized'):
            return

        # Валидация конфигурации
        proxy_config.validate()

        self.api_key = api_key
        self.model = model
        self.proxy_config = proxy_config

        # Формирование URL прокси
        proxy_url = f"socks5://{proxy_config.proxy_server.host}:{proxy_config.proxy_server.port}"

        # Создание httpx клиента с прокси
        self._http_client = httpx.Client(
            proxy=proxy_url,
            timeout=timeout
        )

        self._openai_client = OpenAI(
            api_key=api_key,
            base_url=self.GROK_BASE_URL,
            http_client=self._http_client
        )

        self._initialized = True

    def get_client(self) -> OpenAI:
        """
        Возвращает настроенный OpenAI клиент для работы с Grok API

        Returns:
            OpenAI: Клиент с настроенным прокси и base_url
        """
        return self._openai_client

    def close(self):
        """Закрытие HTTP клиента"""
        if hasattr(self, '_http_client'):
            self._http_client.close()

    @classmethod
    def reset_instance(cls):
        """Сброс синглтона (полезно для тестирования)"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
                cls._instance = None


_grok_client_instance: Optional[GrokClient] = None
_instance_lock = threading.Lock()


def initialize_grok_client(
        api_key: str,
        proxy_config: ProxyConfig,
        timeout: float = GrokClient.DEFAULT_TIMEOUT,
        model: str = GrokClient.DEFAULT_MODEL
) -> None:
    """
    Инициализация глобального экземпляра GrokClient

    Args:
        api_key: API ключ для Grok
        proxy_config: Конфигурация прокси
        timeout: Таймаут запросов в секундах
        model: Модель по умолчанию
    """
    global _grok_client_instance

    with _instance_lock:
        if _grok_client_instance is None:
            _grok_client_instance = GrokClient(
                api_key=api_key,
                proxy_config=proxy_config,
                timeout=timeout,
                model=model
            )


def get_grok_client() -> OpenAI:
    """
    Возвращает OpenAI клиент из глобального синглтона

    Returns:
        OpenAI: Настроенный клиент для работы с Grok API
    """
    if _grok_client_instance is None:
        config = {
            'proxy_server': {
                'host': 'xray-proxy',
                'port': 8100
            }
        }

        proxy_config = ProxyConfig(config)

        # Инициализация глобального клиента (выполняется один раз)
        initialize_grok_client(
            api_key=XAI_API_KEY,
            proxy_config=proxy_config,
            timeout=30.0
        )

    return _grok_client_instance.get_client()

