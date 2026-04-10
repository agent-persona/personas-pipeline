from __future__ import annotations

from synthesis.models import CohortBuilder, PersonaV2


def _persona_v1_payload() -> dict:
    return {
        "schema_version": "2.0",
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


def test_default_decade_table_loads_and_covers_expected_periods():
    builder = CohortBuilder()

    assert builder.decades[0].decade == "1920s"
    assert builder.decades[-1].decade == "2020s"
    assert len(builder.decades) == 11


def test_cohort_builder_assembles_windows_and_familiarity():
    cohort = CohortBuilder().build(birth_year=1988, eval_year=2026)

    assert cohort.birth_year == 1988
    assert cohort.childhood_window.start_year == 1988
    assert cohort.childhood_window.end_year == 2000
    assert cohort.adolescence_window.start_year == 2001
    assert cohort.adolescence_window.end_year == 2005
    assert cohort.early_adulthood_window.start_year == 2006
    assert cohort.early_adulthood_window.end_year == 2013
    assert cohort.tech_familiarity.grew_up_with
    assert cohort.tech_familiarity.adopted_as_adult
    assert cohort.cultural_references.music
    assert cohort.cultural_references.tv_and_film
    assert cohort.economic_assumptions.career_model in {
        "company_man",
        "ladder_climb",
        "job_hopping",
        "portfolio_career",
    }
    assert cohort.slang_compatibility.active_slang


def test_persona_v2_accepts_cohort_snapshot_and_exports_cleanly():
    builder = CohortBuilder()
    cohort = builder.build(birth_year=1988, eval_year=2026)
    persona = PersonaV2(
        **_persona_v1_payload(),
        birth_year=1988,
        eval_year=2026,
        age=38,
        cohort_label=builder.build_label(1988),
        tech_familiarity_snapshot=cohort.tech_familiarity,
        cohort=cohort,
        contradictions=[
            {
                "axis": "curious-but-busy",
                "description": "Wants to test new tools but needs proof first.",
                "behavioral_manifestation": "Reads reviews before booking a demo.",
                "confidence": 0.8,
            }
        ],
    )

    dumped = persona.model_dump(mode="json")

    assert dumped["schema_version"] == "2.0"
    assert dumped["birth_year"] == 1988
    assert dumped["cohort_label"] == "millennial (1988-born)"
    assert dumped["tech_familiarity_snapshot"]["grew_up_with"]
    assert dumped["cohort"]["birth_year"] == 1988
    assert dumped["contradictions"][0]["axis"] == "curious-but-busy"
