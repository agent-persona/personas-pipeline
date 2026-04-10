from __future__ import annotations

import copy
import json
import logging
import re
from typing import Any

from synthesis.provider_registry import normalize_provider

logger = logging.getLogger(__name__)

_TOP_LEVEL_LIST_FIELDS = (
    "goals",
    "pains",
    "motivations",
    "objections",
    "channels",
    "vocabulary",
    "decision_triggers",
    "sample_quotes",
)
_DEMOGRAPHIC_LIST_FIELDS = ("location_signals",)
_FIRMOGRAPHIC_LIST_FIELDS = ("role_titles", "tech_stack_signals")
_TOP_LEVEL_ALIASES = {
    "evidence": "source_evidence",
    "citations": "source_evidence",
    "supporting_evidence": "source_evidence",
    "quotes": "sample_quotes",
    "pain_points": "pains",
    "motivators": "motivations",
    "concerns": "objections",
    "preferred_channels": "channels",
    "decision_factors": "decision_triggers",
    "demographic": "demographics",
}
_DEMOGRAPHIC_ALIASES = {
    "age_range": "age_range",
    "gender_distribution": "gender_distribution",
    "location_signals": "location_signals",
    "education_level": "education_level",
    "income_bracket": "income_bracket",
}
_FIRMOGRAPHIC_ALIASES = {
    "company_size": "company_size",
    "industry": "industry",
    "role_titles": "role_titles",
    "role_title": "role_titles",
    "titles": "role_titles",
    "job_titles": "role_titles",
    "tech_stack_signals": "tech_stack_signals",
    "tech_stack": "tech_stack_signals",
    "technographic_signals": "tech_stack_signals",
    "technologies": "tech_stack_signals",
}
_PROVIDER_TOP_LEVEL_ALIASES: dict[str, dict[str, str]] = {
    "minimax": {
        "firmographic": "firmographics",
        "firmographics_signals": "firmographics",
        "journey": "journey_stages",
        "journey_stage": "journey_stages",
        "journey_map": "journey_stages",
        "persona_journey": "journey_stages",
    },
}
_EVIDENCE_ALIASES = {
    "claim": "claim",
    "statement": "claim",
    "text": "claim",
    "record_ids": "record_ids",
    "record_id": "record_ids",
    "records": "record_ids",
    "source_record_ids": "record_ids",
    "source_ids": "record_ids",
    "field_path": "field_path",
    "field": "field_path",
    "path": "field_path",
    "target_field": "field_path",
    "persona_field": "field_path",
    "confidence": "confidence",
    "score": "confidence",
}
_JOURNEY_ALIASES = {
    "stage": "stage",
    "mindset": "mindset",
    "key_actions": "key_actions",
    "actions": "key_actions",
    "content_preferences": "content_preferences",
    "preferred_content": "content_preferences",
    "content_types": "content_preferences",
}
_UNKNOWN_SENTINEL = object()


class DigestError(ValueError):
    """Raised when provider output cannot be normalized safely."""

    def __init__(self, message: str, *, warnings: list[str] | None = None) -> None:
        super().__init__(message)
        self.warnings = warnings or []


def digest_provider_output(
    *,
    provider: str,
    model: str,
    raw_output: dict,
    schema_version: str = "1.0",
    strict: bool = False,
) -> dict:
    """Normalize provider-native output into PersonaV1-compatible shape."""
    if isinstance(raw_output, str):
        raw_output = _parse_json(raw_output)
    if not isinstance(raw_output, dict):
        raise DigestError(
            f"Expected provider output dict, got {type(raw_output).__name__}",
        )

    normalized_provider = normalize_provider(provider)
    payload = _unwrap_payload(copy.deepcopy(raw_output), strict=strict)
    warnings: list[str] = []
    canonical: dict[str, Any] = {"schema_version": schema_version}

    aliases = dict(_TOP_LEVEL_ALIASES)
    aliases.update(_PROVIDER_TOP_LEVEL_ALIASES.get(normalized_provider, {}))

    for key, value in payload.items():
        canonical_key = _normalize_key(key)
        alias_key = aliases.get(canonical_key, canonical_key)
        canonical_key = alias_key
        if canonical_key == "schema_version":
            canonical["schema_version"] = _normalize_schema_version(value, schema_version)
            continue
        if canonical_key == "demographics":
            canonical["demographics"] = _normalize_demographics(
                value,
                warnings,
                strict=strict,
            )
            continue
        if canonical_key == "firmographics":
            canonical["firmographics"] = _normalize_firmographics(
                value,
                warnings,
                strict=strict,
            )
            continue
        if canonical_key == "journey_stages":
            canonical["journey_stages"] = _normalize_journey_stages(
                value,
                warnings,
                strict=strict,
            )
            continue
        if canonical_key == "source_evidence":
            canonical["source_evidence"] = _normalize_source_evidence(
                value,
                warnings,
                strict=strict,
            )
            continue
        if canonical_key in _TOP_LEVEL_LIST_FIELDS:
            canonical[canonical_key] = _coerce_list(
                value,
                warnings,
                path=canonical_key,
                strict=strict,
            )
            continue
        if canonical_key in {
            "name",
            "summary",
        }:
            canonical[canonical_key] = value
            continue

        nested = _assign_nested_alias(
            canonical=canonical,
            key=canonical_key,
            value=value,
            warnings=warnings,
            strict=strict,
        )
        if nested:
            continue

        warnings.append(f"Dropped unknown top-level key '{key}'")

    _merge_top_level_into_nested(canonical, warnings, strict=strict)
    _log_warnings(provider=normalized_provider, model=model, warnings=warnings)
    return canonical


def _assign_nested_alias(
    *,
    canonical: dict[str, Any],
    key: str,
    value: Any,
    warnings: list[str],
    strict: bool,
) -> bool:
    if key in _DEMOGRAPHIC_ALIASES:
        demographics = _ensure_dict(canonical, "demographics")
        target = _DEMOGRAPHIC_ALIASES[key]
        if target in _DEMOGRAPHIC_LIST_FIELDS:
            demographics[target] = _coerce_list(
                value,
                warnings,
                path=f"demographics.{target}",
                strict=strict,
            )
        else:
            demographics[target] = value
        return True
    if key in _FIRMOGRAPHIC_ALIASES:
        firmographics = _ensure_dict(canonical, "firmographics")
        target = _FIRMOGRAPHIC_ALIASES[key]
        if target in _FIRMOGRAPHIC_LIST_FIELDS:
            firmographics[target] = _coerce_list(
                value,
                warnings,
                path=f"firmographics.{target}",
                strict=strict,
            )
        else:
            firmographics[target] = value
        return True
    return False


def _merge_top_level_into_nested(
    canonical: dict[str, Any],
    warnings: list[str],
    *,
    strict: bool,
) -> None:
    demographics = canonical.get("demographics")
    if isinstance(demographics, dict):
        for source, target in _DEMOGRAPHIC_ALIASES.items():
            if source in demographics and target not in demographics:
                demographics[target] = demographics.pop(source)
        if "location_signals" in demographics:
            demographics["location_signals"] = _coerce_list(
                demographics["location_signals"],
                warnings,
                path="demographics.location_signals",
                strict=strict,
            )

    firmographics = canonical.get("firmographics")
    if isinstance(firmographics, dict):
        for source, target in _FIRMOGRAPHIC_ALIASES.items():
            if source in firmographics and target not in firmographics:
                firmographics[target] = firmographics.pop(source)
        for field_name in _FIRMOGRAPHIC_LIST_FIELDS:
            if field_name in firmographics:
                firmographics[field_name] = _coerce_list(
                    firmographics[field_name],
                    warnings,
                    path=f"firmographics.{field_name}",
                    strict=strict,
                )


def _normalize_demographics(
    value: Any,
    warnings: list[str],
    *,
    strict: bool,
) -> dict[str, Any]:
    value = _parse_embedded_json(value)
    if not isinstance(value, dict):
        if strict:
            raise DigestError("demographics must be an object", warnings=warnings)
        warnings.append("Expected demographics object; leaving as-is for validation")
        return {"age_range": value} if value is not None else {}
    value = _expand_embedded_json_fields(
        value,
        warnings,
        path="demographics",
    )

    normalized: dict[str, Any] = {}
    for key, raw_value in value.items():
        target = _DEMOGRAPHIC_ALIASES.get(_normalize_key(key))
        if not target:
            warnings.append(f"Dropped unknown demographics key '{key}'")
            continue
        if target in _DEMOGRAPHIC_LIST_FIELDS:
            normalized[target] = _coerce_list(
                raw_value,
                warnings,
                path=f"demographics.{target}",
                strict=strict,
            )
        else:
            normalized[target] = raw_value
    return normalized


def _normalize_firmographics(
    value: Any,
    warnings: list[str],
    *,
    strict: bool,
) -> dict[str, Any]:
    value = _parse_embedded_json(value)
    if not isinstance(value, dict):
        if strict:
            raise DigestError("firmographics must be an object", warnings=warnings)
        warnings.append("Expected firmographics object; leaving as-is for validation")
        return {}
    value = _expand_embedded_json_fields(
        value,
        warnings,
        path="firmographics",
    )

    normalized: dict[str, Any] = {}
    for key, raw_value in value.items():
        target = _FIRMOGRAPHIC_ALIASES.get(_normalize_key(key))
        if not target:
            warnings.append(f"Dropped unknown firmographics key '{key}'")
            continue
        if target in _FIRMOGRAPHIC_LIST_FIELDS:
            normalized[target] = _coerce_list(
                raw_value,
                warnings,
                path=f"firmographics.{target}",
                strict=strict,
            )
        else:
            normalized[target] = raw_value
    return normalized


def _normalize_journey_stages(
    value: Any,
    warnings: list[str],
    *,
    strict: bool,
) -> list[Any]:
    value = _parse_embedded_json(value)
    stages = _coerce_list(
        value,
        warnings,
        path="journey_stages",
        strict=strict,
    )
    normalized: list[Any] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            if strict:
                raise DigestError(
                    f"journey_stages[{index}] must be an object",
                    warnings=warnings,
                )
            normalized.append(stage)
            continue

        item: dict[str, Any] = {}
        for key, raw_value in stage.items():
            target = _JOURNEY_ALIASES.get(_normalize_key(key))
            if not target:
                warnings.append(f"Dropped unknown journey_stages[{index}] key '{key}'")
                continue
            if target in {"key_actions", "content_preferences"}:
                item[target] = _coerce_list(
                    raw_value,
                    warnings,
                    path=f"journey_stages[{index}].{target}",
                    strict=strict,
                )
            else:
                item[target] = raw_value
        normalized.append(item)
    return normalized


def _normalize_source_evidence(
    value: Any,
    warnings: list[str],
    *,
    strict: bool,
) -> list[Any]:
    value = _parse_embedded_json(value)
    evidence_list = _coerce_list(
        value,
        warnings,
        path="source_evidence",
        strict=strict,
    )
    normalized: list[Any] = []
    for index, item in enumerate(evidence_list):
        if not isinstance(item, dict):
            if strict:
                raise DigestError(
                    f"source_evidence[{index}] must be an object",
                    warnings=warnings,
                )
            normalized.append(item)
            continue

        evidence: dict[str, Any] = {}
        for key, raw_value in item.items():
            target = _EVIDENCE_ALIASES.get(_normalize_key(key))
            if not target:
                warnings.append(f"Dropped unknown source_evidence[{index}] key '{key}'")
                continue
            if target == "record_ids":
                evidence[target] = _normalize_record_ids(
                    raw_value,
                    warnings,
                    path=f"source_evidence[{index}].record_ids",
                    strict=strict,
                )
                continue
            if target == "field_path":
                evidence[target] = _normalize_field_path(raw_value)
                continue
            evidence[target] = raw_value

        if "record_ids" not in evidence:
            if strict:
                raise DigestError(
                    f"source_evidence[{index}] missing record_ids",
                    warnings=warnings,
                )
            warnings.append(
                f"source_evidence[{index}] missing record_ids; keeping empty list",
            )
            evidence["record_ids"] = []

        if strict and not isinstance(evidence.get("field_path"), str):
            raise DigestError(
                f"source_evidence[{index}] missing field_path",
                warnings=warnings,
            )

        normalized.append(evidence)
    return normalized


def _normalize_field_path(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    field_path = value.strip().replace("/", ".")
    field_path = re.sub(r"^persona\.", "", field_path)
    field_path = field_path.lstrip(".")
    field_path = re.sub(r"\[(\d+)\]", r".\1", field_path)
    field_path = re.sub(r"\.+", ".", field_path)
    return field_path


def _coerce_list(
    value: Any,
    warnings: list[str],
    *,
    path: str,
    strict: bool,
) -> list[Any]:
    if value is None:
        if strict:
            raise DigestError(f"{path} cannot be null", warnings=warnings)
        return []
    if isinstance(value, list):
        return _drop_blank_strings(value)
    if isinstance(value, tuple | set):
        warnings.append(f"Coerced {path} to list")
        return _drop_blank_strings(list(value))
    if isinstance(value, str):
        stripped = value.strip()
        warnings.append(f"Coerced scalar string at {path} to single-item list")
        return [stripped] if stripped else []
    if strict:
        raise DigestError(f"{path} must be a list", warnings=warnings)
    warnings.append(f"Coerced scalar at {path} to single-item list")
    return [value]


def _ensure_dict(canonical: dict[str, Any], key: str) -> dict[str, Any]:
    existing = canonical.get(key, _UNKNOWN_SENTINEL)
    if isinstance(existing, dict):
        return existing
    canonical[key] = {}
    return canonical[key]


def _log_warnings(*, provider: str, model: str, warnings: list[str]) -> None:
    if not warnings:
        return
    for warning in warnings:
        logger.warning(
            "Digest warning for provider=%s model=%s: %s",
            provider,
            model,
            warning,
        )


def _drop_blank_strings(values: list[Any]) -> list[Any]:
    cleaned: list[Any] = []
    for value in values:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                cleaned.append(stripped)
            continue
        cleaned.append(value)
    return cleaned


def _expand_embedded_json_fields(
    value: dict[str, Any],
    warnings: list[str],
    *,
    path: str,
) -> dict[str, Any]:
    expanded = dict(value)
    for key, raw_value in list(value.items()):
        parsed = _parse_embedded_json(raw_value)
        if not isinstance(parsed, dict):
            continue
        warnings.append(f"Expanded embedded JSON object at {path}.{key}")
        expanded.pop(key, None)
        for nested_key, nested_value in parsed.items():
            expanded.setdefault(nested_key, nested_value)
    return expanded


def _normalize_record_ids(
    value: Any,
    warnings: list[str],
    *,
    path: str,
    strict: bool,
) -> list[str]:
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[\n,]", value) if part.strip()]
        warnings.append(f"Normalized scalar record IDs at {path} to list")
        return list(dict.fromkeys(parts))
    values = _coerce_list(value, warnings, path=path, strict=strict)
    normalized = [str(item).strip() for item in values if str(item).strip()]
    return list(dict.fromkeys(normalized))


def _normalize_key(value: str) -> str:
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return snake.strip().lower()


def _normalize_schema_version(value: Any, fallback: str) -> str:
    normalized = str(value or fallback).strip().lower()
    if normalized in {"1", "1.0", "1.0.0", "v1"}:
        return "1.0"
    return str(value or fallback)


def _unwrap_payload(raw_output: dict[str, Any], *, strict: bool) -> dict[str, Any]:
    if raw_output.get("name") == "create_persona" and "arguments" in raw_output:
        arguments = raw_output["arguments"]
        if isinstance(arguments, str):
            parsed = _parse_json(arguments)
            if isinstance(parsed, dict):
                return parsed
        if isinstance(arguments, dict):
            return arguments

    candidates: list[dict[str, Any]] = []
    for key in ("persona", "output", "data", "input", "arguments"):
        candidate = raw_output.get(key)
        if isinstance(candidate, str):
            try:
                candidate = _parse_json(candidate)
            except DigestError:
                continue
        if isinstance(candidate, dict):
            candidates.append(candidate)

    function_block = raw_output.get("function")
    if isinstance(function_block, dict):
        arguments = function_block.get("arguments")
        if isinstance(arguments, str):
            parsed = _parse_json(arguments)
            if isinstance(parsed, dict):
                candidates.append(parsed)
        elif isinstance(arguments, dict):
            candidates.append(arguments)

    if len(candidates) > 1:
        raise DigestError("Ambiguous provider payload wrapper")
    if len(candidates) == 1:
        return candidates[0]
    return raw_output


def _parse_json(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise DigestError(f"Could not parse provider JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise DigestError(
            f"Expected parsed provider JSON object, got {type(parsed).__name__}",
        )
    return parsed


def _parse_embedded_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value
