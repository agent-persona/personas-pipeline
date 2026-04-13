from __future__ import annotations

import pytest

from synthesis.adapters.eval_adapter import persona_v1_to_eval
from synthesis.models.evidence import SourceEvidence
from synthesis.models.persona import (
    CommunicationStyle,
    Demographics,
    EmotionalProfile,
    Firmographics,
    JourneyStage,
    MoralFramework,
    PersonaV1,
)


def _fully_populated_persona() -> PersonaV1:
    return PersonaV1(
        name="Alex the DevOps Engineer",
        summary="Senior DevOps engineer at a fintech SMB.",
        demographics=Demographics(
            age_range="28-38",
            gender_distribution="predominantly male",
            location_signals=["US or EU tech hub"],
            education_level="MS Computer Science",
            income_bracket="$120k-$180k",
        ),
        firmographics=Firmographics(
            company_size="50-200 employees",
            industry="Fintech",
            role_titles=["Senior DevOps Engineer", "SRE"],
            tech_stack_signals=["Terraform", "Webhooks", "GraphQL"],
        ),
        goals=["Automate state transitions", "Provision via Terraform"],
        pains=["GraphQL schema drift", "Webhook retry docs thin"],
        motivations=["Reduce toil", "Audit-ready infra"],
        objections=["GraphQL unstable"],
        channels=["GitHub", "Hacker News"],
        vocabulary=["idempotent", "IaC", "pipeline"],
        decision_triggers=["Stable GraphQL schema"],
        sample_quotes=[
            "If it's not in Terraform it doesn't exist",
            "GraphQL drift is killing our release velocity",
        ],
        journey_stages=[
            JourneyStage(stage="evaluation", mindset="skeptical", key_actions=["reads docs"], content_preferences=["API ref"]),
            JourneyStage(stage="activation", mindset="building", key_actions=["first integration"], content_preferences=["samples"]),
        ],
        communication_style=CommunicationStyle(
            tone="direct",
            formality="professional",
            vocabulary_level="advanced",
            preferred_channels=["Intercom", "GitHub"],
        ),
        emotional_profile=EmotionalProfile(
            baseline_mood="pragmatic",
            stress_triggers=["schema drift", "silent webhook drops"],
            coping_mechanisms=["writes automation", "files detailed tickets"],
        ),
        moral_framework=MoralFramework(
            core_values=["efficiency", "reliability"],
            ethical_stance="utilitarian",
            moral_foundations={"care": 0.4, "fairness": 0.7, "liberty": 0.8},
        ),
        source_evidence=[
            SourceEvidence(claim="c1", record_ids=["r1"], field_path="goals.0", confidence=0.9),
            SourceEvidence(claim="c2", record_ids=["r2"], field_path="pains.0", confidence=0.9),
            SourceEvidence(claim="c3", record_ids=["r1"], field_path="motivations.0", confidence=0.9),
        ],
    )


class TestAdapterIdentityAndDemographics:
    def test_id_and_name(self) -> None:
        out = persona_v1_to_eval(_fully_populated_persona(), persona_id="clust_abc")
        assert out.id == "clust_abc"
        assert out.name == "Alex the DevOps Engineer"

    def test_summary_becomes_bio(self) -> None:
        out = persona_v1_to_eval(_fully_populated_persona(), persona_id="clust_abc")
        assert out.bio == "Senior DevOps engineer at a fintech SMB."

    def test_age_range_midpoint(self) -> None:
        out = persona_v1_to_eval(_fully_populated_persona(), persona_id="clust_abc")
        assert out.age == 33
        assert out.extra["age_range"] == "28-38"

    def test_gender_strip_predominantly(self) -> None:
        out = persona_v1_to_eval(_fully_populated_persona(), persona_id="clust_abc")
        assert out.gender == "male"
        assert out.extra["gender_distribution"] == "predominantly male"

    def test_gender_mixed_stays_empty(self) -> None:
        p = _fully_populated_persona()
        p.demographics.gender_distribution = "mixed"
        out = persona_v1_to_eval(p, persona_id="clust_abc")
        assert out.gender == ""

    def test_location_takes_first_signal(self) -> None:
        out = persona_v1_to_eval(_fully_populated_persona(), persona_id="clust_abc")
        assert out.location == "US or EU tech hub"

    def test_firmographic_flatten(self) -> None:
        out = persona_v1_to_eval(_fully_populated_persona(), persona_id="clust_abc")
        assert out.occupation == "Senior DevOps Engineer"
        assert out.industry == "Fintech"
        assert out.extra["company_size"] == "50-200 employees"
        assert "Terraform" in out.knowledge_domains

    def test_education_and_income(self) -> None:
        out = persona_v1_to_eval(_fully_populated_persona(), persona_id="clust_abc")
        assert out.education == "MS Computer Science"
        assert out.income_bracket == "$120k-$180k"

    def test_age_range_unparsable_fallback(self) -> None:
        p = _fully_populated_persona()
        p.demographics.age_range = "unknown"
        out = persona_v1_to_eval(p, persona_id="clust_abc")
        assert out.age is None
        assert out.extra["age_range"] == "unknown"

    def test_age_range_plus_notation_falls_back(self) -> None:
        """'65+' is preserved in extra but age is None — pins behavior so an LLM shift gets caught."""
        p = _fully_populated_persona()
        p.demographics.age_range = "65+"
        out = persona_v1_to_eval(p, persona_id="clust_abc")
        assert out.age is None
        assert out.extra["age_range"] == "65+"

    def test_gender_strip_primarily(self) -> None:
        p = _fully_populated_persona()
        p.demographics.gender_distribution = "primarily female"
        out = persona_v1_to_eval(p, persona_id="clust_abc")
        assert out.gender == "female"

    def test_gender_non_binary_passes_through(self) -> None:
        """No prefix, not in the 'mixed' set — pass through to eval.Persona as-is."""
        p = _fully_populated_persona()
        p.demographics.gender_distribution = "non-binary"
        out = persona_v1_to_eval(p, persona_id="clust_abc")
        assert out.gender == "non-binary"
        assert out.extra["gender_distribution"] == "non-binary"


class TestAdapterBehavioralLists:
    def test_goals_direct_copy(self) -> None:
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="c1")
        assert out.goals == p.goals

    def test_motivations_direct_copy(self) -> None:
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="c1")
        assert out.motivations == p.motivations

    def test_pains_become_pain_points(self) -> None:
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="c1")
        assert out.pain_points == p.pains

    def test_objections_become_frustrations(self) -> None:
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="c1")
        assert out.frustrations == p.objections


class TestAdapterPsychologicalCopy:
    def test_communication_style_round_trip(self) -> None:
        out = persona_v1_to_eval(_fully_populated_persona(), persona_id="c1")
        assert out.communication_style.tone == "direct"
        assert out.communication_style.formality == "professional"
        assert out.communication_style.vocabulary_level == "advanced"
        assert out.communication_style.preferred_channels == ["Intercom", "GitHub"]

    def test_emotional_profile_round_trip(self) -> None:
        out = persona_v1_to_eval(_fully_populated_persona(), persona_id="c1")
        assert out.emotional_profile.baseline_mood == "pragmatic"
        assert out.emotional_profile.stress_triggers == ["schema drift", "silent webhook drops"]
        assert out.emotional_profile.coping_mechanisms == ["writes automation", "files detailed tickets"]

    def test_moral_framework_round_trip(self) -> None:
        out = persona_v1_to_eval(_fully_populated_persona(), persona_id="c1")
        assert out.moral_framework.core_values == ["efficiency", "reliability"]
        assert out.moral_framework.ethical_stance == "utilitarian"
        assert out.moral_framework.moral_foundations == {"care": 0.4, "fairness": 0.7, "liberty": 0.8}

    def test_core_values_mirrored_into_values(self) -> None:
        out = persona_v1_to_eval(_fully_populated_persona(), persona_id="c1")
        assert out.values == ["efficiency", "reliability"]


class TestAdapterEvidenceAndExtras:
    def test_source_ids_from_evidence_deduped(self) -> None:
        from synthesis.models.evidence import SourceEvidence
        p = _fully_populated_persona()
        p.source_evidence.append(
            SourceEvidence(claim="extra", record_ids=["r1", "r2"], field_path="goals.1", confidence=0.8)
        )
        out = persona_v1_to_eval(p, persona_id="c1")
        assert out.source_ids == ["r1", "r2"]  # deduped, order preserved

    def test_journey_stages_in_extra(self) -> None:
        out = persona_v1_to_eval(_fully_populated_persona(), persona_id="c1")
        stages = out.extra["journey_stages"]
        assert isinstance(stages, list)
        assert stages[0]["stage"] == "evaluation"

    def test_sample_quotes_in_extra(self) -> None:
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="c1")
        assert out.extra["sample_quotes"] == p.sample_quotes

    def test_decision_triggers_in_extra(self) -> None:
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="c1")
        assert out.extra["decision_triggers"] == p.decision_triggers

    def test_channels_in_extra(self) -> None:
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="c1")
        assert out.extra["channels"] == p.channels

    def test_vocabulary_in_extra(self) -> None:
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="c1")
        assert out.extra["vocabulary"] == p.vocabulary

    def test_source_evidence_preserved_in_extra(self) -> None:
        out = persona_v1_to_eval(_fully_populated_persona(), persona_id="c1")
        dumped = out.extra["source_evidence"]
        assert isinstance(dumped, list)
        assert dumped[0]["claim"] == "c1"

    def test_empty_moral_foundations_round_trip(self) -> None:
        """Empty moral_foundations is valid per 'omit rather than guess' rule."""
        p = _fully_populated_persona()
        p.moral_framework.moral_foundations = {}
        out = persona_v1_to_eval(p, persona_id="c1")
        assert out.moral_framework.moral_foundations == {}

    def test_source_ids_dedup_within_single_evidence_entry(self) -> None:
        """Dedup must also collapse repeats *within* a single evidence entry's record_ids."""
        from synthesis.models.evidence import SourceEvidence
        p = _fully_populated_persona()
        p.source_evidence.append(
            SourceEvidence(
                claim="dup-within",
                record_ids=["r9", "r9", "r10"],
                field_path="goals.1",
                confidence=0.8,
            )
        )
        out = persona_v1_to_eval(p, persona_id="c1")
        assert out.source_ids.count("r9") == 1
        assert out.source_ids.count("r10") == 1

    def test_extra_journey_stages_is_independent_copy(self) -> None:
        """Mutating out.extra shouldn't mutate the source PersonaV1 — aliasing defence."""
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="c1")
        out.extra["journey_stages"][0]["stage"] = "MUTATED"
        assert p.journey_stages[0].stage == "evaluation"
