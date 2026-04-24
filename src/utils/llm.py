"""Helper functions for LLM"""

import json
import time
from pydantic import BaseModel
from src.llm.models import DEFAULT_MODEL_NAME, DEFAULT_PROVIDER, get_model
from src.utils.progress import progress
from src.graph.state import AgentState


def call_llm(
    prompt: any,
    pydantic_model: type[BaseModel],
    agent_name: str | None = None,
    state: AgentState | None = None,
    max_retries: int = 1,
    default_factory=None,
) -> BaseModel:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates and model config extraction
        state: Optional state object to extract agent-specific model configuration
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model
    """
    
    # Groq-only defaults
    model_name = DEFAULT_MODEL_NAME
    model_provider = DEFAULT_PROVIDER.value

    # Extract API keys from state if available
    api_keys = None
    if state:
        request = state.get("metadata", {}).get("request")
        if request and hasattr(request, 'api_keys'):
            api_keys = request.api_keys

    client = get_model(model_name, model_provider, api_keys)
    messages = _prompt_to_groq_messages(prompt)
    retries = max(1, max_retries + 1)
    runtime_model_name = DEFAULT_MODEL_NAME

    # Call the LLM with retries
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=runtime_model_name,
                messages=messages,
                temperature=0.2,
            )
            content = completion.choices[0].message.content or ""
            parsed_result = extract_json_from_response(content)
            if parsed_result is not None:
                return pydantic_model(**_normalize_signal_payload(parsed_result, pydantic_model))
            raise ValueError("Failed to parse structured JSON from Groq response")

        except Exception as e:
            err_text = str(e).lower()
            if "rate_limit" in err_text or "429" in err_text:
                # single backoff before final fallback
                time.sleep(6)
            if "model_decommissioned" in err_text or "decommissioned" in err_text:
                runtime_model_name = "llama-3.1-8b-instant"
            if agent_name:
                progress.update_status(agent_name, None, f"LLM error - retry {attempt + 1}/{retries}")

            if attempt == retries - 1:
                # Enforce structured fallback for all agents
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)


def create_default_response(model_class: type[BaseModel]) -> BaseModel:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        key = field_name.lower()
        if key == "signal":
            default_values[field_name] = _get_supported_signal_value(field, preferred="HOLD")
            continue
        if key == "confidence":
            default_values[field_name] = 10.0
            continue
        if key == "reasoning":
            default_values[field_name] = "LLM failed, fallback used"
            continue
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


def extract_json_from_response(content: str) -> dict | None:
    """Extracts JSON from markdown-formatted response."""
    try:
        if not content:
            return None
        content = content.strip()
        if content.startswith("{") and content.endswith("}"):
            return json.loads(content)
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7 :]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
        # Fallback: grab first JSON object in plain text
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start : end + 1])
    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
    return None


def get_agent_model_config(state, agent_name):
    """
    Get model configuration for a specific agent from the state.
    Falls back to global model configuration if agent-specific config is not available.
    Always returns valid model_name and model_provider values.
    """
    request = state.get("metadata", {}).get("request")
    
    if request and hasattr(request, 'get_agent_model_config'):
        # Get agent-specific model configuration
        model_name, model_provider = request.get_agent_model_config(agent_name)
        # Ensure we have valid values
        if model_name and model_provider:
            return model_name, model_provider.value if hasattr(model_provider, 'value') else str(model_provider)
    
    # Fall back to global configuration (system defaults)
    model_name = state.get("metadata", {}).get("model_name") or "gpt-4.1"
    model_provider = state.get("metadata", {}).get("model_provider") or "OPENAI"
    
    # Convert enum to string if necessary
    if hasattr(model_provider, 'value'):
        model_provider = model_provider.value
    
    return model_name, model_provider


def _prompt_to_groq_messages(prompt: any) -> list[dict[str, str]]:
    """Convert LangChain prompt objects or strings into Groq SDK messages."""
    if hasattr(prompt, "to_messages"):
        messages = []
        for m in prompt.to_messages():
            role = getattr(m, "type", "human")
            if role == "human":
                role = "user"
            elif role not in ("system", "assistant", "user"):
                role = "user"
            content = getattr(m, "content", "")
            if isinstance(content, list):
                content = " ".join(str(part) for part in content)
            messages.append({"role": role, "content": str(content)})
        return messages
    return [{"role": "user", "content": str(prompt)}]


def _normalize_signal_payload(payload: dict, model_class: type[BaseModel]) -> dict:
    """Normalize BUY/SELL/HOLD signals to each model's allowed literal values."""
    normalized = dict(payload)
    for field_name, field in model_class.model_fields.items():
        if field_name.lower() != "signal" or field_name not in normalized:
            continue
        value = str(normalized[field_name]).strip()
        upper = value.upper()
        alias_map = {
            "BUY": "BUY",
            "SELL": "SELL",
            "HOLD": "HOLD",
            "BULLISH": "BUY",
            "BEARISH": "SELL",
            "NEUTRAL": "HOLD",
        }
        canonical = alias_map.get(upper, upper)
        normalized[field_name] = _map_canonical_signal_to_model(field, canonical)
    return normalized


def _map_canonical_signal_to_model(field, canonical: str):
    allowed = list(getattr(field.annotation, "__args__", []) or [])
    if not allowed:
        return canonical
    allowed_str = [str(a) for a in allowed]
    lower_lookup = {a.lower(): a for a in allowed_str}
    mapping = {
        "BUY": ["buy", "bullish"],
        "SELL": ["sell", "bearish"],
        "HOLD": ["hold", "neutral"],
    }
    for candidate in mapping.get(canonical, [canonical.lower()]):
        if candidate in lower_lookup:
            return lower_lookup[candidate]
    return allowed_str[0]


def _get_supported_signal_value(field, preferred: str = "HOLD"):
    allowed = list(getattr(field.annotation, "__args__", []) or [])
    if not allowed:
        return preferred
    allowed_str = [str(a) for a in allowed]
    lower_lookup = {a.lower(): a for a in allowed_str}
    if preferred.lower() in lower_lookup:
        return lower_lookup[preferred.lower()]
    for alt in ("neutral", "hold", "bullish", "bearish", "buy", "sell"):
        if alt in lower_lookup:
            return lower_lookup[alt]
    return allowed_str[0]
