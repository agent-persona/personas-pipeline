from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    transport: str
    example_models: tuple[str, ...]
    twin_supported: bool = False


PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        name="anthropic",
        transport="anthropic",
        example_models=("claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"),
        twin_supported=True,
    ),
    "gemini": ProviderSpec(
        name="gemini",
        transport="openai",
        example_models=("gemini-2.5-flash", "gemini-2.5-pro"),
    ),
    "kimi": ProviderSpec(
        name="kimi",
        transport="openai",
        example_models=("kimi-k2-0711-preview", "kimi-k2-turbo-preview"),
    ),
    "zai": ProviderSpec(
        name="zai",
        transport="anthropic",
        example_models=("glm-4.7", "glm-4.5"),
        twin_supported=True,
    ),
    "minimax": ProviderSpec(
        name="minimax",
        transport="anthropic",
        example_models=("MiniMax-M2.7", "MiniMax-M2.5"),
        twin_supported=True,
    ),
}


def normalize_provider(provider: str | None) -> str:
    value = (provider or "anthropic").strip().lower()
    aliases = {
        "google": "gemini",
        "moonshot": "kimi",
        "z.ai": "zai",
        "glm": "zai",
    }
    return aliases.get(value, value)


def get_provider_spec(provider: str | None) -> ProviderSpec:
    normalized = normalize_provider(provider)
    if normalized not in PROVIDER_SPECS:
        supported = ", ".join(sorted(PROVIDER_SPECS))
        raise ValueError(f"Unsupported provider '{normalized}'. Supported: {supported}")
    return PROVIDER_SPECS[normalized]


def validate_provider_model(provider: str | None, model: str | None, *, label: str) -> None:
    spec = get_provider_spec(provider)
    selected = (model or "").strip()
    if not selected:
        raise ValueError(f"{label} model is empty for provider '{spec.name}'")

    lowered = selected.lower()
    if spec.name == "anthropic" and "claude" not in lowered:
        raise ValueError(f"{label} model '{selected}' does not look like an Anthropic model")
    if spec.name == "gemini" and "gemini" not in lowered:
        raise ValueError(f"{label} model '{selected}' does not look like a Gemini model")
    if spec.name == "kimi" and "kimi" not in lowered:
        raise ValueError(f"{label} model '{selected}' does not look like a Kimi model")
    if spec.name == "zai" and "glm" not in lowered:
        raise ValueError(f"{label} model '{selected}' does not look like a Z.AI GLM model")
    if spec.name == "minimax" and "minimax" not in lowered:
        raise ValueError(f"{label} model '{selected}' does not look like a MiniMax model")
