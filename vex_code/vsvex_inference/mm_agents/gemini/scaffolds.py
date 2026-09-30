from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from .actions import parse_hybrid_macro_response, parse_internal_response
from .history import ensure_empty_think_prefix
from .prompts import (
    build_hybrid_macro_system_prompt,
    build_hybrid_macro_tools_def,
    build_internal_system_prompt,
    build_internal_tools_def,
)


BuildToolsDef = Callable[[int, int, str], Dict]
BuildSystemPrompt = Callable[[Dict, str], str]
ParseResponse = Callable[..., Tuple[str, List[str]]]
ResponseTransform = Callable[[str], str]


@dataclass(frozen=True)
class GeminiScaffoldProfile:
    name: str
    build_tools_def: BuildToolsDef
    build_system_prompt: BuildSystemPrompt
    parse_response: ParseResponse
    response_transform: ResponseTransform


GEMINI_SCAFFOLD_PROFILES = {
    "qwen_internal": GeminiScaffoldProfile(
        name="qwen_internal",
        build_tools_def=build_internal_tools_def,
        build_system_prompt=build_internal_system_prompt,
        parse_response=parse_internal_response,
        response_transform=ensure_empty_think_prefix,
    ),
    "hybrid_macro": GeminiScaffoldProfile(
        name="hybrid_macro",
        build_tools_def=build_hybrid_macro_tools_def,
        build_system_prompt=build_hybrid_macro_system_prompt,
        parse_response=parse_hybrid_macro_response,
        response_transform=ensure_empty_think_prefix,
    ),
}


def get_gemini_scaffold_profile(name: str) -> GeminiScaffoldProfile:
    try:
        return GEMINI_SCAFFOLD_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(GEMINI_SCAFFOLD_PROFILES))
        raise ValueError(f"Unknown Gemini scaffold profile: {name}. Available profiles: {available}") from exc


def list_gemini_scaffold_profiles() -> List[str]:
    return sorted(GEMINI_SCAFFOLD_PROFILES)
