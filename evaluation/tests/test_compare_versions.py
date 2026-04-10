import math

from scripts.compare_versions import compute_metric_deltas
from scripts.run_baseline import _persona_metric_bundle as baseline_metric_bundle
from scripts.run_temp_sweep import pick_best_temperature


def test_metric_deltas_prefer_rerun_summary_but_keep_stability_from_aggregate():
    baseline_a = {
        "aggregate": {"groundedness": 1.0, "stability": 0.2},
        "aggregate_summary": {"means": {"groundedness": 0.4}},
    }
    baseline_b = {
        "aggregate": {"groundedness": 0.0, "stability": 0.5},
        "aggregate_summary": {"means": {"groundedness": 0.8}},
    }

    deltas = compute_metric_deltas(baseline_a, baseline_b)
    by_metric = {row["metric"]: row for row in deltas}

    assert by_metric["groundedness"]["winner"] == "b"
    assert by_metric["groundedness"]["delta"] == 0.4
    assert by_metric["stability"]["winner"] == "b"
    assert by_metric["stability"]["delta"] == 0.3


def test_pick_best_temperature_uses_mean_metrics_when_available():
    results = [
        {
            "status": "ok",
            "temperature": 0.3,
            "aggregate": {"groundedness": 0.95, "stability": 0.2},
            "aggregate_summary": {"means": {"groundedness": 0.7}},
            "judge_overall_mean": 0.5,
            "total_cost_usd": 1.0,
        },
        {
            "status": "ok",
            "temperature": 0.7,
            "aggregate": {"groundedness": 0.8, "stability": 0.1},
            "aggregate_summary": {"means": {"groundedness": 0.9}},
            "judge_overall_mean": 0.4,
            "total_cost_usd": 1.2,
        },
    ]

    decision = pick_best_temperature(results, groundedness_tolerance=0.0)

    assert decision["best_temperature"] == 0.7
    assert decision["best_summary"]["groundedness"] == 0.9


def test_runner_metric_bundle_exposes_extended_schema_metrics_when_present():
    persona = {
        "schema_version": "2.0",
        "name": "Avery Chen",
        "demographics": {"age_range": "30-44"},
        "birth_year": 1988,
        "eval_year": 2026,
        "age": 38,
        "cohort_label": "millennial (1988-born)",
        "cohort": {
            "birth_year": 1988,
            "eval_year": 2026,
            "major_events_lived_through": ["COVID-19"],
        },
        "tech_familiarity_snapshot": {
            "grew_up_with": ["internet", "email"],
            "adopted_as_adult": ["ai"],
            "never_used": ["pager"],
        },
        "capability_matrix": {
            "coding": {
                "factual_knowledge": 4.5,
                "procedural_skill": 4.0,
                "taste_judgment": 3.5,
                "creativity": 3.0,
                "speed": 4.2,
                "consistency": 4.1,
                "error_recovery": 3.8,
                "teaching_ability": 2.5,
                "tool_fluency": 4.4,
                "confidence_calibration": 3.9,
                "identity_salience": "core",
                "experience": {
                    "years_exposed": 8,
                    "years_practiced": 5,
                    "recency": "active",
                },
            },
        },
        "relational_self": {
            "self_monitoring_level": 3.4,
            "baseline_warmth": 0.72,
            "baseline_dominance": 0.41,
            "relationship_profiles": {"boss": {"warmth": 0.4}},
            "trait_distributions": {"extraversion": {"mean": 2.8}},
            "if_then_signatures": [{"condition": {"relationship": "boss"}, "strength": 0.9}],
            "conflict_history": {"boss": {"safe": True}},
        },
    }

    metrics = baseline_metric_bundle(persona)

    assert set(metrics) == {
        "developmental_fit",
        "historical_fit",
        "capability_coherence",
        "relational_realism",
    }
    assert all(value is not None for value in metrics.values())


def test_runner_metric_bundle_falls_back_cleanly_without_extended_schema_fields():
    persona = {
        "name": "Alex Chen",
        "demographics": {"age_range": "30-44"},
    }

    metrics = baseline_metric_bundle(persona)

    assert metrics["developmental_fit"] is not None
    assert metrics["historical_fit"] is None
    assert metrics["capability_coherence"] is None
    assert metrics["relational_realism"] is None


def test_metric_deltas_include_extended_schema_metrics_when_present():
    baseline_a = {
        "aggregate": {
            "developmental_fit": 0.7,
            "historical_fit": 0.2,
            "capability_coherence": 0.4,
            "relational_realism": 0.3,
        },
        "aggregate_summary": {
            "means": {
                "developmental_fit": 0.7,
                "historical_fit": 0.2,
                "capability_coherence": 0.4,
                "relational_realism": 0.3,
            }
        },
    }
    baseline_b = {
        "aggregate": {
            "developmental_fit": 0.8,
            "historical_fit": 0.6,
            "capability_coherence": 0.5,
            "relational_realism": 0.9,
        },
        "aggregate_summary": {
            "means": {
                "developmental_fit": 0.8,
                "historical_fit": 0.6,
                "capability_coherence": 0.5,
                "relational_realism": 0.9,
            }
        },
    }

    deltas = compute_metric_deltas(baseline_a, baseline_b)
    by_metric = {row["metric"]: row for row in deltas}

    assert by_metric["historical_fit"]["winner"] == "b"
    assert math.isclose(by_metric["historical_fit"]["delta"], 0.4)
    assert by_metric["capability_coherence"]["winner"] == "b"
    assert by_metric["relational_realism"]["winner"] == "b"
