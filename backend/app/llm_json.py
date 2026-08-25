"""Small compatibility layer for DeepSeek JSON-mode structured responses.

DeepSeek's thinking models reject OpenAI function/tool calling for this use
case.  The agent therefore asks for JSON and validates it with Pydantic rather
than silently losing the model planner and falling back to keyword rules.
"""

import json
from typing import Any, TypeVar

from pydantic import BaseModel


ModelT = TypeVar("ModelT", bound=BaseModel)


def _text_content(content: Any) -> str:
    """Return text from either a normal or a block-based LangChain response."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return ""


def invoke_json_model(
    llm: Any,
    schema: type[ModelT],
    messages: list[Any],
    *,
    attempts: int = 2,
) -> ModelT | None:
    """Call an OpenAI-compatible model in JSON mode and validate its result.

    Returning ``None`` keeps the existing deterministic safety path available
    only when the model is genuinely unavailable or returns invalid JSON.
    """
    for _ in range(attempts):
        try:
            response = llm.bind(response_format={"type": "json_object"}).invoke(messages)
            content = _text_content(response.content).strip()
            # Be tolerant of an occasional prose prefix while still validating
            # the complete object with the Pydantic model.
            start, end = content.find("{"), content.rfind("}")
            if start < 0 or end < start:
                continue
            payload = json.loads(content[start : end + 1])
            return schema.model_validate(payload)
        except Exception:  # noqa: BLE001 - caller has a safe fallback plan
            continue
    return None
