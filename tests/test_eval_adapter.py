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
