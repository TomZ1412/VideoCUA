import base64
import os
import time
from typing import Dict, List, Optional, Tuple

from google import genai
from google.genai import errors, types


RETRY_SLEEP_SECONDS = float(os.getenv("OSWORLD_GEMINI_RETRY_SLEEP", "5"))


def _content_part_to_gemini(part: Dict) -> Optional[types.Part]:
    part_type = part.get("type")
    if part_type == "text":
        return types.Part.from_text(text=part.get("text", ""))

    if part_type == "image_url":
        url = ((part.get("image_url") or {}).get("url")) or ""
        if "," not in url:
            return None
        header, data = url.split(",", 1)
        mime_type = "image/png"
        if header.startswith("data:") and ";" in header:
            mime_type = header[len("data:"):header.index(";")]
        return types.Part.from_bytes(data=base64.b64decode(data), mime_type=mime_type)

    return None


def _messages_to_gemini_contents(messages: List[Dict]) -> Tuple[Optional[str], List[types.Content]]:
    system_parts: List[str] = []
    contents: List[types.Content] = []

    for message in messages:
        role = message.get("role")
        parts = []
        for part in message.get("content", []) or []:
            if role == "system" and isinstance(part, dict) and part.get("type") == "text":
                system_parts.append(part.get("text", ""))
                continue
            gemini_part = _content_part_to_gemini(part) if isinstance(part, dict) else None
            if gemini_part is not None:
                parts.append(gemini_part)

        if not parts:
            continue

        contents.append(
            types.Content(
                role="model" if role == "assistant" else "user",
                parts=parts,
            )
        )

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def _extract_response_text(response) -> str:
    try:
        parts = response.candidates[0].content.parts
    except Exception:
        text = getattr(response, "text", None)
        return text or ""

    thought_parts = []
    visible_parts = []
    for part in parts or []:
        part_text = getattr(part, "text", None)
        if not part_text:
            continue
        if getattr(part, "thought", False) is True:
            thought_parts.append(part_text)
        else:
            visible_parts.append(part_text)

    visible_text = "".join(visible_parts)
    thought_text = "\n".join(text.strip() for text in thought_parts if text.strip())

    if thought_text:
        return f"<think>\n{thought_text}\n</think>\n\n{visible_text.lstrip()}"
    if visible_text:
        return visible_text

    text = getattr(response, "text", None)
    return text or ""


def call_gemini(
    payload: Dict,
    model: str,
    *,
    api_key: Optional[str],
    base_url: Optional[str],
    default_max_tokens: int,
    default_temperature: float,
    default_top_p: float,
    media_resolution: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    include_thoughts: bool = False,
    logger=None,
) -> str:
    resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_AK")
    if not resolved_api_key:
        raise ValueError("Gemini API key is required. Pass --api_key or set GEMINI_API_KEY/GEMINI_AK.")

    http_options = None
    if base_url:
        http_options = types.HttpOptions(
            api_version=os.environ.get("GEMINI_API_VERSION", "v1"),
            base_url=base_url,
            timeout=int(float(os.environ.get("OSWORLD_GEMINI_TIMEOUT", "120")) * 1000),
        )

    client = genai.Client(api_key=resolved_api_key, http_options=http_options)
    system_instruction, contents = _messages_to_gemini_contents(payload["messages"])

    thinking_config = None
    if thinking_budget is not None or include_thoughts:
        thinking_config = types.ThinkingConfig(
            thinkingBudget=thinking_budget,
            includeThoughts=include_thoughts,
        )

    media_resolution_value = None
    if media_resolution:
        enum_name = f"MEDIA_RESOLUTION_{str(media_resolution).strip().upper()}"
        media_resolution_value = getattr(types.MediaResolution, enum_name, None)
        if media_resolution_value is None:
            raise ValueError(
                f"Unsupported media_resolution: {media_resolution}. "
                "Expected one of low|medium|high."
            )

    config = types.GenerateContentConfig(
        systemInstruction=system_instruction,
        temperature=payload.get("temperature", default_temperature),
        topP=payload.get("top_p", default_top_p),
        maxOutputTokens=payload.get("max_tokens", default_max_tokens),
        thinkingConfig=thinking_config,
        mediaResolution=media_resolution_value,
    )

    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return _extract_response_text(response)
        except errors.APIError as exc:
            status_code = getattr(exc, "code", None)
            if status_code in {400, 401, 403, 404}:
                logger.error(
                    f"⚠️[GeminiAgent] Unrecoverable API error: {exc}"
                )
                raise
            if logger:
                logger.warning(
                    "⏰[GeminiAgent] recoverable API error on attempt %d: %s. Retrying...",
                    attempt,
                    exc,
                )
            time.sleep(RETRY_SLEEP_SECONDS)
        except Exception as exc:
            if logger:
                logger.warning(
                    "⏰[GeminiAgent] network or unknown error on attempt %d: %s. Retrying...",
                    attempt,
                    exc,
                )
            time.sleep(RETRY_SLEEP_SECONDS)
