from __future__ import annotations

from synthesis.engine.groundedness import check_groundedness
from synthesis.models.cluster import (
    ClusterData,
    ClusterSummary,
    SampleRecord,
    TenantContext,
)
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


def _cluster(record_ids: list[str]) -> ClusterData:
    return ClusterData(
        cluster_id="c1",
        tenant=TenantContext(tenant_id="t1"),
        summary=ClusterSummary(cluster_size=len(record_ids)),
        sample_records=[
            SampleRecord(record_id=rid, source="ga4") for rid in record_ids
        ],
    )


def _persona(evidence: list[SourceEvidence]) -> PersonaV1:
    return PersonaV1(
        name="Tester",
        summary="A persona for groundedness tests.",
        demographics=Demographics(
            age_range="25-34",
            gender_distribution="mixed",
            location_signals=["US"],
        ),
        firmographics=Firmographics(),
        goals=["g0", "g1"],
        pains=["p0", "p1"],
        motivations=["m0", "m1"],
        objections=["o0"],
        not_this=["n0", "n1"],
        channels=["Slack"],
        vocabulary=["a", "b", "c"],
        decision_triggers=["t0"],
        sample_quotes=["q0", "q1"],
        journey_stages=[
            JourneyStage(stage="evaluation", mindset="x", key_actions=["a"], content_preferences=["c"]),
            JourneyStage(stage="activation", mindset="y", key_actions=["a"], content_preferences=["c"]),
        ],
        communication_style=CommunicationStyle(
            tone="direct", formality="professional", vocabulary_level="advanced",
            preferred_channels=["Slack"],
        ),
        emotional_profile=EmotionalProfile(
            baseline_mood="calm", stress_triggers=["outages"], coping_mechanisms=["automation"],
        ),
        moral_framework=MoralFramework(
            core_values=["efficiency", "fairness"],
            ethical_stance="utilitarian",
            moral_foundations={"care": 0.5},
        ),
        source_evidence=evidence,
    )


def _full_evidence() -> list[SourceEvidence]:
    return [
        SourceEvidence(claim="goal 0", record_ids=["r1"], field_path="goals.0", confidence=0.9),
        SourceEvidence(claim="goal 1", record_ids=["r1"], field_path="goals.1", confidence=0.9),
        SourceEvidence(claim="pain 0", record_ids=["r1"], field_path="pains.0", confidence=0.9),
        SourceEvidence(claim="pain 1", record_ids=["r1"], field_path="pains.1", confidence=0.9),
        SourceEvidence(claim="mot 0", record_ids=["r1"], field_path="motivations.0", confidence=0.9),
        SourceEvidence(claim="mot 1", record_ids=["r1"], field_path="motivations.1", confidence=0.9),
        SourceEvidence(claim="obj 0", record_ids=["r1"], field_path="objections.0", confidence=0.9),
        SourceEvidence(claim="tone evidence", record_ids=["r1"], field_path="communication_style.tone", confidence=0.8),
        SourceEvidence(claim="mood evidence", record_ids=["r1"], field_path="emotional_profile.baseline_mood", confidence=0.8),
        SourceEvidence(claim="values evidence", record_ids=["r1"], field_path="moral_framework.core_values.0", confidence=0.8),
    ]


class TestPsychologicalGroundedness:
    def test_full_evidence_passes(self) -> None:
        cluster = _cluster(["r1"])
        persona = _persona(_full_evidence())
        report = check_groundedness(persona, cluster)
        assert report.passed, f"Expected pass, got violations: {report.violations}"

    def test_missing_communication_style_evidence_tracked(self) -> None:
        cluster = _cluster(["r1"])
        ev = [e for e in _full_evidence() if not e.field_path.startswith("communication_style")]
        persona = _persona(ev)
        report = check_groundedness(persona, cluster)
        assert "communication_style" in report.missing_psychological_prefixes
        assert any("communication_style" in v for v in report.violations)

    def test_missing_emotional_profile_evidence_tracked(self) -> None:
        cluster = _cluster(["r1"])
        ev = [e for e in _full_evidence() if not e.field_path.startswith("emotional_profile")]
        persona = _persona(ev)
        report = check_groundedness(persona, cluster)
        assert "emotional_profile" in report.missing_psychological_prefixes
        assert any("emotional_profile" in v for v in report.violations)

    def test_missing_moral_framework_evidence_tracked(self) -> None:
        cluster = _cluster(["r1"])
        ev = [e for e in _full_evidence() if not e.field_path.startswith("moral_framework")]
        persona = _persona(ev)
        report = check_groundedness(persona, cluster)
        assert "moral_framework" in report.missing_psychological_prefixes
        assert any("moral_framework" in v for v in report.violations)

    def test_all_psych_prefixes_missing_fails_on_score(self) -> None:
        """Dropping all three psych prefixes pushes score below 0.9 → passed=False via threshold."""
        cluster = _cluster(["r1"])
        ev = [
            e for e in _full_evidence()
            if not any(e.field_path.startswith(p) for p in ("communication_style", "emotional_profile", "moral_framework"))
        ]
        persona = _persona(ev)
        report = check_groundedness(persona, cluster)
        assert not report.passed
        assert report.score < 0.9
