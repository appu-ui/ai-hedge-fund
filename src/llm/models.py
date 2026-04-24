import os
from enum import Enum
from typing import List, Tuple

from groq import Groq
from pydantic import BaseModel


class ModelProvider(str, Enum):
    """Enum for supported LLM providers"""
    GROQ = "Groq"


class LLMModel(BaseModel):
    """Represents an LLM model configuration"""

    display_name: str
    model_name: str
    provider: ModelProvider

    def to_choice_tuple(self) -> Tuple[str, str, str]:
        """Convert to format needed for questionary choices"""
        return (self.display_name, self.model_name, self.provider.value)

    def is_custom(self) -> bool:
        return self.model_name == "-"

    def has_json_mode(self) -> bool:
        return True

DEFAULT_MODEL_NAME = "llama3-8b-8192"
DEFAULT_PROVIDER = ModelProvider.GROQ

AVAILABLE_MODELS: List[LLMModel] = [
    LLMModel(
        display_name="Llama 3 8B (Groq)",
        model_name=DEFAULT_MODEL_NAME,
        provider=DEFAULT_PROVIDER,
    )
]
OLLAMA_MODELS: List[LLMModel] = []
LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]
OLLAMA_LLM_ORDER: List[Tuple[str, str, str]] = []


def get_model_info(model_name: str, model_provider: str) -> LLMModel | None:
    """Get model information by model_name"""
    provider_value = model_provider.value if hasattr(model_provider, "value") else str(model_provider)
    provider_is_groq = str(provider_value).lower() == DEFAULT_PROVIDER.value.lower()
    if model_name == DEFAULT_MODEL_NAME and provider_is_groq:
        return AVAILABLE_MODELS[0]
    return None


def find_model_by_name(model_name: str) -> LLMModel | None:
    """Find a model by its name across all available models."""
    return AVAILABLE_MODELS[0] if model_name == DEFAULT_MODEL_NAME else None


def get_models_list():
    """Get the list of models for API responses."""
    return [
        {
            "display_name": model.display_name,
            "model_name": model.model_name,
            "provider": model.provider.value
        }
        for model in AVAILABLE_MODELS
    ]


def get_model(model_name: str, model_provider: ModelProvider, api_keys: dict = None) -> Groq:
    """Return a Groq SDK client. Other providers are intentionally disabled."""
    del model_name, model_provider
    api_key = (api_keys or {}).get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Groq API key not found. Please set GROQ_API_KEY in your environment.")
    return Groq(api_key=api_key)
