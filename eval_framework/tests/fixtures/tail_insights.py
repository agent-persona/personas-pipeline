"""Fixture: sample tail insight benchmark data for testing D-NEW scorer.

Each scenario has:
- research_question: the prompt context
- benchmark_insights: annotated insights with novelty scores
- majority_response: what a typical/expected persona would say

These are synthetic examples for testing. Real benchmark data requires
expert annotation per the design doc at docs/plans/2026-04-09-tail-insight-detection-design.md.
"""

from __future__ import annotations

SAMPLE_BENCHMARK = [
    {
        "insight_id": "ti_001",
        "insight_text": (
            "Users with chronic conditions prefer voice interfaces not for "
            "convenience but because screen fatigue from medical portals "
            "causes them to skip medication refills"
        ),
        "novelty_score": 5,
        "prevalence": 0.06,
        "domain": "healthcare_ux",
        "semantic_threshold": 0.65,
    },
    {
        "insight_id": "ti_002",
        "insight_text": (
            "First-generation college students avoid office hours not from "
            "shyness but because they interpret professor availability as "
            "a sign the course is too hard for them"
        ),
        "novelty_score": 4,
        "prevalence": 0.09,
        "domain": "education",
        "semantic_threshold": 0.65,
    },
    {
        "insight_id": "ti_003",
        "insight_text": (
            "Budget-conscious families buy premium organic baby food not for "
            "health beliefs but as guilt compensation for not being able to "
            "afford other enrichment activities"
        ),
        "novelty_score": 5,
        "prevalence": 0.04,
        "domain": "consumer_behavior",
        "semantic_threshold": 0.65,
    },
    {
        "insight_id": "ti_004",
        "insight_text": (
            "Remote workers in rural areas report higher job satisfaction "
            "than urban remote workers despite worse internet because the "
            "social status of a tech salary in a small town changes their "
            "entire community standing"
        ),
        "novelty_score": 3,
        "prevalence": 0.12,
        "domain": "workplace",
        "semantic_threshold": 0.65,
    },
    {
        "insight_id": "ti_005",
        "insight_text": (
            "Elderly users prefer physical bank branches not for technophobia "
            "but because the branch visit is their primary social interaction "
            "of the week"
        ),
        "novelty_score": 3,
        "prevalence": 0.11,
        "domain": "fintech",
        "semantic_threshold": 0.65,
    },
]

MAJORITY_RESPONSES = [
    "I prefer digital banking because it's convenient and saves time.",
    "Voice assistants are useful for hands-free convenience.",
    "I choose organic food because I care about my family's health.",
    "Remote work is great because of the flexibility and no commute.",
    "I think office hours are important for getting help with coursework.",
]
