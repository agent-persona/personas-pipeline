import math

from evaluation.metrics import (
    capability_coherence,
    developmental_fit,
    field_interdependence,
    field_interdependence_breakdown,
    historical_fit,
    relational_realism,
    run_core_metrics,
    stability,
    stability_breakdown,
    summarize_metric_runs,
)


def test_stability_breakdown_matches_overall_and_surfaces_field_drift():
    persona_runs = [
        [
            {
                "name": "Alex Chen, the Platform Architect",
                "summary": "Senior platform engineer in fintech.",
                "goals": ["Reduce toil", "Improve reliability"],
                "pains": ["Flaky deploys", "Manual audits"],
                "motivations": ["Trust", "Scale"],
                "objections": ["Migration risk"],
            },
        ],
        [
            {
                "name": "Alex Chen, the Platform Architect",
                "summary": "Senior platform engineer in fintech.",
                "goals": ["Reduce toil", "Improve reliability"],
                "pains": ["Flaky deploys", "Manual audits"],
                "motivations": ["Trust", "Scale"],
                "objections": ["Migration risk"],
            },
        ],
        [
            {
                "name": "Priya Nair, the Platform Architect",
                "summary": "Infra leader in regulated SaaS.",
                "goals": ["Reduce toil", "Improve reliability"],
                "pains": ["Flaky deploys", "Manual audits"],
                "motivations": ["Trust", "Scale"],
                "objections": ["Migration risk"],
            },
        ],
    ]

    overall = stability(persona_runs)
    breakdown = stability_breakdown(persona_runs)

    assert breakdown["overall"] == overall
    assert breakdown["num_runs"] == 3
    assert breakdown["num_personas"] == 1
    assert breakdown["fields"]["goals"]["similarity"] == 1.0
    assert breakdown["fields"]["name"]["similarity"] < 1.0
    assert breakdown["fields"]["summary"]["similarity"] < 1.0


def test_stability_breakdown_handles_single_run():
    breakdown = stability_breakdown([[{"name": "Only Run"}]])

    assert breakdown["num_runs"] == 1
    assert breakdown["num_personas"] == 1
    assert breakdown["fields"] == {}
    assert breakdown["overall"] != breakdown["overall"]


def test_summarize_metric_runs_averages_numeric_values_and_ignores_missing():
    summary = summarize_metric_runs([
        {"groundedness": 1.0, "stability": 0.2},
        {"groundedness": 0.5, "stability": 0.4},
        {"groundedness": None, "stability": float("nan")},
    ])

    assert summary["num_runs"] == 3
    assert summary["counts"]["groundedness"] == 2
    assert summary["counts"]["stability"] == 2
    assert math.isclose(summary["means"]["groundedness"], 0.75)
    assert math.isclose(summary["means"]["stability"], 0.3)
    assert math.isclose(summary["stdevs"]["groundedness"], 0.25)


def test_field_interdependence_penalizes_cross_persona_field_swaps():
    aligned = {
        "summary": "DevOps engineer automating CI pipelines with Terraform and GitHub Actions.",
        "firmographics": {
            "industry": "B2B SaaS",
            "role_titles": ["DevOps Engineer"],
            "tech_stack_signals": ["Terraform", "GitHub Actions", "CI/CD"],
        },
        "goals": ["Automate CI pipelines", "Reduce deployment toil"],
        "pains": ["Manual release steps", "Fragile Terraform modules"],
        "motivations": ["Reliable automation", "Stronger platform tooling"],
        "objections": ["Weak API coverage"],
        "decision_triggers": ["Native Terraform support"],
        "channels": ["GitHub", "Terraform docs"],
        "vocabulary": ["terraform", "pipeline", "deployments", "automation"],
        "sample_quotes": ["We need better Terraform support for our deployment pipeline."],
        "journey_stages": [
            {
                "stage": "decision",
                "key_actions": ["terraform_setup", "api_review"],
                "content_preferences": ["API docs", "Terraform guides"],
            }
        ],
    }
    swapped = {
        **aligned,
        "goals": ["Find faster moodboards", "Win more logo clients"],
        "pains": ["Manual invoice followups", "Late client feedback"],
        "motivations": ["Creative flow", "More referrals"],
        "decision_triggers": ["Figma templates"],
        "channels": ["Instagram", "Dribbble"],
        "vocabulary": ["branding", "moodboard", "client revisions"],
        "sample_quotes": ["I need cleaner moodboards before I pitch this logo system."],
    }

    aligned_score = field_interdependence(aligned)
    swapped_score = field_interdependence(swapped)
    breakdown = field_interdependence_breakdown(aligned)

    assert breakdown["rules"]["role_goal_alignment"]["score"] is not None
    assert aligned_score > swapped_score


def _base_v2_persona() -> dict:
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
            "tech_stack_signals": ["HubSpot", "GA4", "email"],
        },
        "goals": ["Increase qualified pipeline", "Reduce CAC"],
        "pains": ["Low signal attribution", "Manual reporting"],
        "motivations": ["Prove ROI", "Scale repeatable growth"],
        "objections": ["Long implementation cycle"],
        "channels": ["LinkedIn", "Communities"],
        "vocabulary": ["pipeline", "attribution", "intent", "email"],
        "decision_triggers": ["Clear ROI model"],
        "sample_quotes": [
            "I need cleaner attribution before I add spend.",
            "If this saves analyst time, I can justify it.",
        ],
        "birth_year": 1988,
        "eval_year": 2026,
        "age": 38,
        "cohort_label": "millennial (1988-born)",
        "tech_familiarity_snapshot": {
            "grew_up_with": ["internet", "email"],
            "adopted_as_adult": ["ai assistants", "remote collaboration"],
            "never_used": ["telegraph"],
        },
        "cohort": {
            "birth_year": 1988,
            "eval_year": 2026,
            "tech_familiarity": {
                "grew_up_with": ["internet", "email"],
                "adopted_as_adult": ["ai assistants", "remote collaboration"],
                "never_used": ["telegraph"],
            },
            "slang_compatibility": {
                "active_slang": ["lit"],
                "recognized_slang": ["slay", "fam"],
                "unknown_slang": ["rizz"],
            },
            "major_events_lived_through": ["COVID-19", "global financial crisis"],
        },
        "capability_matrix": {
            "growth_marketing": {
                "factual_knowledge": 4.2,
                "procedural_skill": 4.0,
                "taste_judgment": 3.4,
                "creativity": 3.6,
                "speed": 3.8,
                "consistency": 4.1,
                "error_recovery": 3.5,
                "teaching_ability": 3.2,
                "tool_fluency": 4.3,
                "confidence_calibration": 3.9,
                "failure_mode": "over-indexes on attribution",
                "meta_learning_ability": 3.8,
                "social_proof_dependency": 1.5,
                "identity_salience": "core",
                "motivation": 4.7,
                "confidence_level": "grounded",
                "evidence": "Growth lead owns attribution and pipeline targets.",
                "experience": {
                    "years_exposed": 12,
                    "years_practiced": 8,
                    "deliberate_practice_intensity": 3.4,
                    "recency": "active",
                    "decay_rate": 0.15,
                    "success_rate": 0.78,
                    "mentorship_received": "formal",
                    "unconscious_competence_stage": 3,
                },
                "conditions": {
                    "stress_modifier": -0.6,
                    "fatigue_modifier": -0.8,
                    "context_switch_cost": "medium",
                    "peak_performance_context": "quiet dashboard review",
                    "ceiling": 4.8,
                    "floor": 2.0,
                },
            },
        },
        "relational_self": {
            "self_monitoring_level": 3.2,
            "baseline_warmth": 0.65,
            "baseline_dominance": 0.45,
            "trait_distributions": {
                "extraversion": {
                    "mean": 2.8,
                    "variance": 0.7,
                    "skew": -0.1,
                    "floor": 1.2,
                    "ceiling": 4.1,
                }
            },
            "relationship_profiles": {
                "close_friend": {
                    "warmth": 0.88,
                    "dominance": 0.42,
                    "disclosure_level": 0.86,
                    "self_monitoring": 0.18,
                    "spontaneity": 0.82,
                    "slang_level": 0.55,
                    "humor_frequency": 0.72,
                    "trust_level": 0.93,
                    "vulnerability": 0.78,
                    "reputation_concern": 0.14,
                },
                "boss": {
                    "warmth": 0.54,
                    "dominance": 0.28,
                    "disclosure_level": 0.26,
                    "self_monitoring": 0.84,
                    "spontaneity": 0.22,
                    "slang_level": 0.08,
                    "humor_frequency": 0.18,
                    "trust_level": 0.58,
                    "vulnerability": 0.18,
                    "reputation_concern": 0.83,
                },
                "stranger": {
                    "warmth": 0.38,
                    "dominance": 0.33,
                    "disclosure_level": 0.14,
                    "self_monitoring": 0.76,
                    "spontaneity": 0.26,
                    "slang_level": 0.05,
                    "humor_frequency": 0.12,
                    "trust_level": 0.24,
                    "vulnerability": 0.1,
                    "reputation_concern": 0.61,
                },
            },
            "if_then_signatures": [
                {
                    "condition": {"relationship": "boss", "stakes": "high"},
                    "behavior_modifiers": {"self_monitoring": 0.3, "disclosure_level": -0.2},
                    "strength": 0.9,
                    "evidence": "Observed in review meetings",
                }
            ],
            "group_profile": {
                "conformity_increase": 0.55,
                "signaling_increase": 0.33,
                "nuance_decrease": 0.28,
                "risk_taking_shift": -0.12,
                "in_group_warmth_boost": 0.46,
                "out_group_guardedness": 0.62,
                "audience_size_sensitivity": 0.51,
                "leadership_emergence": 0.41,
                "deference_to_consensus": 0.38,
            },
        },
    }


def test_developmental_fit_penalizes_child_executive_profile():
    adult = _base_v2_persona()
    child = _base_v2_persona()
    child["age"] = 9
    child["demographics"]["age_range"] = "6-12"
    child["firmographics"]["role_titles"] = ["Chief Revenue Officer"]
    child["summary"] = "Executive leading revenue strategy for a software company."
    child["sample_quotes"] = [
        "I need tighter procurement alignment before next quarter.",
        "The board expects revenue acceleration and cross-functional execution.",
    ]

    assert developmental_fit(adult) > developmental_fit(child)


def test_historical_fit_penalizes_unknown_slang_and_future_year_references():
    grounded = _base_v2_persona()
    drifted = _base_v2_persona()
    drifted["sample_quotes"] = [
        "This workflow has real rizz and will age well in 2035.",
        "The campaign tone feels rizz-heavy to me.",
    ]

    assert historical_fit(grounded) > historical_fit(drifted)


def test_capability_coherence_penalizes_impossible_experience_claims():
    coherent = _base_v2_persona()
    impossible = _base_v2_persona()
    impossible["age"] = 20
    impossible["demographics"]["age_range"] = "18-25"
    impossible["capability_matrix"]["growth_marketing"]["experience"]["years_exposed"] = 35
    impossible["capability_matrix"]["growth_marketing"]["experience"]["years_practiced"] = 30
    impossible["capability_matrix"]["growth_marketing"]["experience"]["unconscious_competence_stage"] = 4
    impossible["capability_matrix"]["growth_marketing"]["speed"] = 4.9
    impossible["capability_matrix"]["growth_marketing"]["confidence_level"] = "grounded"
    impossible["capability_matrix"]["growth_marketing"]["evidence"] = ""

    assert capability_coherence(coherent) > capability_coherence(impossible)


def test_relational_realism_prefers_contextual_differentiation_over_flat_profiles():
    nuanced = _base_v2_persona()
    flat = _base_v2_persona()
    flat["relational_self"]["relationship_profiles"] = {
        "close_friend": {
            "warmth": 0.5,
            "dominance": 0.5,
            "disclosure_level": 0.5,
            "self_monitoring": 0.5,
            "spontaneity": 0.5,
            "slang_level": 0.5,
            "humor_frequency": 0.5,
            "trust_level": 0.5,
            "vulnerability": 0.5,
            "reputation_concern": 0.5,
        },
        "boss": {
            "warmth": 0.5,
            "dominance": 0.5,
            "disclosure_level": 0.5,
            "self_monitoring": 0.5,
            "spontaneity": 0.5,
            "slang_level": 0.5,
            "humor_frequency": 0.5,
            "trust_level": 0.5,
            "vulnerability": 0.5,
            "reputation_concern": 0.5,
        },
    }
    flat["relational_self"]["if_then_signatures"] = []

    assert relational_realism(nuanced) > relational_realism(flat)


def test_run_core_metrics_surfaces_v2_metric_bundle():
    persona = _base_v2_persona()

    metrics = run_core_metrics(personas=[persona], schema_cls=None, groundedness_reports=None)

    assert metrics["developmental_fit"] > 0.0
    assert metrics["historical_fit"] > 0.0
    assert metrics["capability_coherence"] > 0.0
    assert metrics["relational_realism"] > 0.0
