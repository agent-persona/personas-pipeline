from __future__ import annotations

import pytest

from synthesis.engine.model_backend import LLMResult
from synthesis.engine.synthesizer import synthesize_for_schema_version, synthesize_v2
from synthesis.models import ClusterData, CohortBuilder, PersonaV1, PersonaV2


def _cluster() -> ClusterData:
    return ClusterData.model_validate(
        {
            "cluster_id": "cluster_growth_ops",
            "tenant": {
                "tenant_id": "tenant_acme_corp",
                "industry": "B2B SaaS",
                "product_description": "Project management tool for engineering teams",
            },
            "summary": {
                "cluster_size": 3,
                "top_behaviors": ["reads comparison pages", "books demos"],
            },
            "sample_records": [
                {"record_id": "rec_1", "source": "ga4", "payload": {"page": "/pricing"}},
                {"record_id": "rec_2", "source": "hubspot", "payload": {"title": "Growth Lead"}},
                {"record_id": "rec_3", "source": "intercom", "payload": {"message": "Need clearer ROI"}},
            ],
        }
    )


def _persona_payload() -> dict:
    cohort = CohortBuilder().build(birth_year=1988, eval_year=2026)
    return {
        "schema_version": "2.0",
        "name": "Avery Chen",
        "summary": "Growth lead at a mid-market B2B SaaS company.",
        "demographics": {
            "age_range": "30-44",
            "gender_distribution": "mixed",
            "location_signals": ["New York", "Remote"],
        },
        "firmographics": {
            "company_size": "50-200 employees",
            "industry": "B2B SaaS",
            "role_titles": ["Growth Lead"],
            "tech_stack_signals": ["HubSpot", "GA4"],
        },
        "goals": ["Increase qualified pipeline", "Reduce CAC"],
        "pains": ["Low signal attribution", "Manual reporting"],
        "motivations": ["Prove ROI", "Scale repeatable growth"],
        "objections": ["Long implementation cycle"],
        "channels": ["LinkedIn", "Communities"],
        "vocabulary": ["pipeline", "attribution", "intent"],
        "decision_triggers": ["Clear ROI model"],
        "sample_quotes": [
            "I need cleaner attribution before I add spend.",
            "If this saves analyst time, I can justify it.",
        ],
        "journey_stages": [
            {
                "stage": "awareness",
                "mindset": "Problem framing",
                "key_actions": ["Read benchmark content"],
                "content_preferences": ["Reports"],
            },
            {
                "stage": "decision",
                "mindset": "Vendor shortlist",
                "key_actions": ["Request demo"],
                "content_preferences": ["Case studies"],
            },
        ],
        "source_evidence": [
            {"claim": "Needs better attribution", "record_ids": ["rec_1"], "field_path": "goals.0", "confidence": 0.9},
            {"claim": "Wants lower CAC", "record_ids": ["rec_2"], "field_path": "goals.1", "confidence": 0.8},
            {"claim": "Low signal attribution", "record_ids": ["rec_1"], "field_path": "pains.0", "confidence": 0.9},
            {"claim": "Manual reporting pain", "record_ids": ["rec_3"], "field_path": "pains.1", "confidence": 0.8},
            {"claim": "Needs proof", "record_ids": ["rec_3"], "field_path": "motivations.0", "confidence": 0.8},
            {"claim": "Wants scale", "record_ids": ["rec_2"], "field_path": "motivations.1", "confidence": 0.7},
            {"claim": "Concerned about implementation", "record_ids": ["rec_2"], "field_path": "objections.0", "confidence": 0.8},
        ],
        "birth_year": 1988,
        "eval_year": 2026,
        "age": 38,
        "cohort_label": CohortBuilder.build_label(1988),
        "tech_familiarity_snapshot": cohort.tech_familiarity.model_dump(),
        "cohort": cohort.model_dump(),
        "contradictions": [
            {
                "axis": "curious-but-busy",
                "description": "Wants to test new tools but needs proof first.",
                "behavioral_manifestation": "Reads reviews before booking a demo.",
                "confidence": 0.8,
            }
        ],
    }


class FakeBackend:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def generate(self, system: str, messages: list[dict], tool: dict) -> LLMResult:
        self.calls.append({"system": system, "messages": messages, "tool": tool})
        return LLMResult(
            tool_input=self.payload,
            input_tokens=100,
            output_tokens=200,
            model="fake-model",
        )


@pytest.mark.asyncio
async def test_synthesize_v2_validates_persona_v2_and_injects_cohort_context() -> None:
    backend = FakeBackend(_persona_payload())

    result = await synthesize_v2(
        _cluster(),
        backend,
        birth_year=1988,
        eval_year=2026,
    )

    assert isinstance(result.persona, PersonaV2)
    assert result.persona.cohort_label == "millennial (1988-born)"
    assert result.groundedness.passed is True
    assert backend.calls
    assert "Historical Cohort Context" in backend.calls[0]["messages"][0]["content"]
    assert "birth_year" in backend.calls[0]["tool"]["input_schema"]["properties"]


@pytest.mark.asyncio
async def test_schema_version_router_dispatches_v1_and_v2() -> None:
    v1_backend = FakeBackend(
        {
            "schema_version": "1.0",
            "name": "Avery Chen",
            "summary": "Growth lead at a mid-market B2B SaaS company.",
            "demographics": {
                "age_range": "30-44",
                "gender_distribution": "mixed",
                "location_signals": ["New York", "Remote"],
            },
            "firmographics": {
                "company_size": "50-200 employees",
                "industry": "B2B SaaS",
                "role_titles": ["Growth Lead"],
                "tech_stack_signals": ["HubSpot", "GA4"],
            },
            "goals": ["Increase qualified pipeline", "Reduce CAC"],
            "pains": ["Low signal attribution", "Manual reporting"],
            "motivations": ["Prove ROI", "Scale repeatable growth"],
            "objections": ["Long implementation cycle"],
            "channels": ["LinkedIn", "Communities"],
            "vocabulary": ["pipeline", "attribution", "intent"],
            "decision_triggers": ["Clear ROI model"],
            "sample_quotes": [
                "I need cleaner attribution before I add spend.",
                "If this saves analyst time, I can justify it.",
            ],
            "journey_stages": [
                {
                    "stage": "awareness",
                    "mindset": "Problem framing",
                    "key_actions": ["Read benchmark content"],
                    "content_preferences": ["Reports"],
                },
                {
                    "stage": "decision",
                    "mindset": "Vendor shortlist",
                    "key_actions": ["Request demo"],
                    "content_preferences": ["Case studies"],
                },
            ],
            "source_evidence": [
                {"claim": "Needs better attribution", "record_ids": ["rec_1"], "field_path": "goals.0", "confidence": 0.9},
                {"claim": "Wants lower CAC", "record_ids": ["rec_2"], "field_path": "goals.1", "confidence": 0.8},
                {"claim": "Low signal attribution", "record_ids": ["rec_1"], "field_path": "pains.0", "confidence": 0.9},
                {"claim": "Manual reporting pain", "record_ids": ["rec_3"], "field_path": "pains.1", "confidence": 0.8},
                {"claim": "Needs proof", "record_ids": ["rec_3"], "field_path": "motivations.0", "confidence": 0.8},
                {"claim": "Wants scale", "record_ids": ["rec_2"], "field_path": "motivations.1", "confidence": 0.7},
                {"claim": "Concerned about implementation", "record_ids": ["rec_2"], "field_path": "objections.0", "confidence": 0.8},
            ],
        }
    )
    v2_backend = FakeBackend(_persona_payload())

    v1_result = await synthesize_for_schema_version(
        _cluster(),
        v1_backend,
        schema_version="v1",
    )
    v2_result = await synthesize_for_schema_version(
        _cluster(),
        v2_backend,
        schema_version="v2",
        birth_year=1988,
        eval_year=2026,
    )

    assert isinstance(v1_result.persona, PersonaV1)
    assert isinstance(v2_result.persona, PersonaV2)
