import copy
import json
import logging

import pytest

from synthesis.engine.digest import DigestError, digest_provider_output


def _valid_persona_payload() -> dict:
    return {
        "schema_version": "1.0",
        "name": "Avery Chen",
        "summary": "Growth lead at a mid-market B2B SaaS company.",
        "demographics": {
            "age_range": "30-44",
            "gender_distribution": "mixed",
            "location_signals": ["New York", "Remote"],
            "education_level": "Bachelor's",
            "income_bracket": "$100k-$150k",
        },
        "firmographics": {
            "company_size": "50-200 employees",
            "industry": "B2B SaaS",
            "role_titles": ["Growth Lead", "Demand Gen Manager"],
            "tech_stack_signals": ["HubSpot", "GA4"],
        },
        "goals": ["Increase qualified pipeline", "Reduce CAC"],
        "pains": ["Low signal attribution", "Manual reporting"],
        "motivations": ["Prove ROI", "Scale repeatable growth"],
        "objections": ["Long implementation cycle"],
        "channels": ["LinkedIn", "Communities"],
        "vocabulary": ["pipeline", "attribution", "intent", "conversion"],
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
            {
                "claim": "Needs better attribution",
                "record_ids": ["rec_1"],
                "field_path": "goals.0",
                "confidence": 0.9,
            },
            {
                "claim": "Hates manual reporting",
                "record_ids": ["rec_2"],
                "field_path": "pains.1",
                "confidence": 0.8,
            },
            {
                "claim": "Wants ROI proof",
                "record_ids": ["rec_3"],
                "field_path": "decision_triggers.0",
                "confidence": 0.7,
            },
        ],
    }


def test_digest_leaves_valid_anthropic_payload_unchanged():
    raw = _valid_persona_payload()
    raw_copy = copy.deepcopy(raw)

    digested = digest_provider_output(
        provider="anthropic",
        model="claude-sonnet-4-6",
        raw_output=raw,
    )

    assert digested == raw_copy
    assert raw == raw_copy


def test_digest_normalizes_minimax_payload_into_persona_shape():
    raw = {
        "name": "Avery Chen",
        "summary": "Growth lead at a mid-market B2B SaaS company.",
        "demographics": json.dumps({
            "age_range": "30-44",
            "gender_distribution": "mixed",
            "location_signals": "New York",
        }),
        "firmographic": json.dumps({
            "company_size": "50-200 employees",
            "industry": "B2B SaaS",
            "role_title": "Growth Lead",
            "tech_stack": "HubSpot",
        }),
        "goals": "Increase qualified pipeline",
        "pains": ["Low signal attribution", "Manual reporting"],
        "motivations": "Prove ROI",
        "objections": "Long implementation cycle",
        "channels": "LinkedIn",
        "vocabulary": ["pipeline", "attribution", "intent"],
        "decision_triggers": "Clear ROI model",
        "sample_quotes": [
            "I need cleaner attribution before I add spend.",
            "If this saves analyst time, I can justify it.",
        ],
        "journey": json.dumps({
            "stage": "decision",
            "mindset": "Vendor shortlist",
            "actions": "Request demo",
            "preferred_content": "Case studies",
        }),
        "evidence": json.dumps([
            {
                "statement": "Needs better attribution",
                "record_id": "rec_1",
                "path": "goals[0]",
                "score": 0.9,
            },
            {
                "statement": "Hates manual reporting",
                "record_ids": ["rec_2"],
                "field": "pains[1]",
                "confidence": 0.8,
            },
            {
                "statement": "Wants ROI proof",
                "records": "rec_3",
                "target_field": "decision_triggers[0]",
                "confidence": 0.7,
            },
        ]),
    }

    digested = digest_provider_output(
        provider="minimax",
        model="MiniMax-M2.7",
        raw_output=raw,
    )

    assert digested["schema_version"] == "1.0"
    assert digested["demographics"]["age_range"] == "30-44"
    assert digested["demographics"]["location_signals"] == ["New York"]
    assert digested["firmographics"]["role_titles"] == ["Growth Lead"]
    assert digested["firmographics"]["tech_stack_signals"] == ["HubSpot"]
    assert digested["goals"] == ["Increase qualified pipeline"]
    assert digested["motivations"] == ["Prove ROI"]
    assert digested["channels"] == ["LinkedIn"]
    assert digested["journey_stages"][0]["key_actions"] == ["Request demo"]
    assert digested["journey_stages"][0]["content_preferences"] == ["Case studies"]
    assert digested["source_evidence"][0]["claim"] == "Needs better attribution"
    assert digested["source_evidence"][0]["record_ids"] == ["rec_1"]
    assert digested["source_evidence"][0]["field_path"] == "goals.0"


def test_digest_rejects_malformed_evidence_in_strict_mode():
    raw = _valid_persona_payload()
    raw["source_evidence"] = [{"claim": "Needs better attribution"}]

    with pytest.raises(DigestError, match="missing record_ids"):
        digest_provider_output(
            provider="minimax",
            model="MiniMax-M2.7",
            raw_output=raw,
            strict=True,
        )


def test_digest_expands_embedded_json_object_fields_for_minimax():
    raw = _valid_persona_payload()
    raw["demographics"] = {
        "age_range": (
            '{"age_range":"30-44","gender_distribution":"mixed",'
            '"location_signals":["New York","Remote"],'
            '"education_level":"Bachelor\'s","income_bracket":"$100k-$150k"}'
        ),
    }

    digested = digest_provider_output(
        provider="minimax",
        model="MiniMax-M2.7",
        raw_output=raw,
    )

    assert digested["demographics"] == {
        "age_range": "30-44",
        "gender_distribution": "mixed",
        "location_signals": ["New York", "Remote"],
        "education_level": "Bachelor's",
        "income_bracket": "$100k-$150k",
    }


def test_digest_drops_unknown_keys_with_warning(caplog: pytest.LogCaptureFixture):
    raw = _valid_persona_payload()
    raw["surprise"] = "junk"
    raw["firmographics"]["mystery"] = "noise"

    with caplog.at_level(logging.WARNING):
        digested = digest_provider_output(
            provider="anthropic",
            model="claude-sonnet-4-6",
            raw_output=raw,
        )

    assert "surprise" not in digested
    assert "mystery" not in digested["firmographics"]
    assert "Dropped unknown top-level key 'surprise'" in caplog.text
    assert "Dropped unknown firmographics key 'mystery'" in caplog.text
