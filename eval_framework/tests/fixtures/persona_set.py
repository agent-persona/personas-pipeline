"""Fixture: generate sets of Persona objects for distributional tests."""

from __future__ import annotations

import random

from persona_eval.schemas import Persona, CommunicationStyle, EmotionalProfile

OCCUPATIONS = [
    "Junior Developer", "Senior Product Manager", "Marketing Director",
    "Data Analyst", "UX Designer", "Sales Representative", "CTO",
    "Customer Support Lead", "HR Manager", "Financial Analyst",
    "DevOps Engineer", "Content Strategist", "Operations Manager",
    "Research Scientist", "Account Executive",
]
INDUSTRIES = ["SaaS", "Fintech", "Healthcare", "E-commerce", "Education", "Manufacturing"]
EDUCATION_LEVELS = ["High school", "Associate's", "Bachelor's degree", "Master's degree", "PhD"]
INCOME_BRACKETS = ["low", "lower-middle", "middle", "upper-middle", "high"]
LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Austin, TX", "Seattle, WA",
    "Chicago, IL", "Denver, CO", "Miami, FL", "Portland, OR",
    "Atlanta, GA", "Boston, MA", "Rural Iowa", "Bangalore, India",
    "London, UK", "Berlin, Germany", "Tokyo, Japan",
]
GENDERS = ["male", "female", "non-binary"]
MARITAL_STATUSES = ["single", "married", "divorced", "widowed", "partnered"]
ETHNICITIES = [
    "White", "Black", "Hispanic/Latino", "Asian American",
    "South Asian", "Mixed", "Middle Eastern",
]
TRAITS = [
    "analytical", "creative", "pragmatic", "ambitious", "cautious",
    "empathetic", "competitive", "collaborative", "independent", "detail-oriented",
]
TONES = ["formal", "casual", "professional but warm", "direct", "diplomatic"]
FORMALITIES = ["very_informal", "informal", "semi-formal", "formal", "very_formal"]
VOCAB_LEVELS = ["basic", "intermediate", "advanced", "technical"]
MOODS = ["optimistic", "neutral", "anxious", "driven", "calm"]
LIFESTYLES = ["remote-first", "urban commuter", "suburban", "digital nomad", "rural"]
VALUES = ["innovation", "stability", "growth", "efficiency", "quality", "speed"]


def generate_test_persona_set(n: int = 50, seed: int = 42) -> list[Persona]:
    """Generate a diverse set of n Persona objects for distributional testing."""
    rng = random.Random(seed)
    personas = []

    for i in range(n):
        age = rng.randint(22, 65)
        max_exp = age - 18
        exp = rng.randint(0, max(0, max_exp))

        persona = Persona(
            id=f"test-persona-{i:03d}",
            name=f"Test Person {i}",
            age=age,
            gender=rng.choice(GENDERS),
            location=rng.choice(LOCATIONS),
            education=rng.choice(EDUCATION_LEVELS),
            occupation=rng.choice(OCCUPATIONS),
            industry=rng.choice(INDUSTRIES),
            experience_years=exp,
            income_bracket=rng.choice(INCOME_BRACKETS),
            ethnicity=rng.choice(ETHNICITIES),
            marital_status=rng.choice(MARITAL_STATUSES),
            personality_traits=rng.sample(TRAITS, k=3),
            values=rng.sample(VALUES, k=rng.randint(1, 3)),
            lifestyle=rng.choice(LIFESTYLES),
            goals=[f"Goal {j}" for j in range(rng.randint(1, 3))],
            pain_points=[f"Pain point {j}" for j in range(rng.randint(1, 3))],
            knowledge_domains=[f"Domain {j}" for j in range(rng.randint(1, 3))],
            behaviors=[f"Behavior {j}" for j in range(rng.randint(1, 3))],
            communication_style=CommunicationStyle(
                tone=rng.choice(TONES),
                formality=rng.choice(FORMALITIES),
                vocabulary_level=rng.choice(VOCAB_LEVELS),
                preferred_channels=["email"],
            ),
            emotional_profile=EmotionalProfile(
                baseline_mood=rng.choice(MOODS),
                stress_triggers=["deadlines"],
                coping_mechanisms=["exercise"],
            ),
            bio=f"Test Person {i} is a {rng.choice(OCCUPATIONS)} with {exp} years of experience in {rng.choice(INDUSTRIES)}. They bring a unique perspective to their work.",
        )
        personas.append(persona)

    return personas


def generate_homogeneous_set(n: int = 50) -> list[Persona]:
    """Generate a homogeneous persona set — should FAIL diversity tests."""
    return [
        Persona(
            id=f"clone-{i:03d}",
            name=f"Clone {i}",
            age=35,
            gender="male",
            location="San Francisco, CA",
            education="Master's degree",
            occupation="Senior Product Manager",
            industry="SaaS",
            experience_years=10,
            income_bracket="upper-middle",
            ethnicity="White",
            marital_status="married",
            personality_traits=["analytical", "collaborative", "pragmatic"],
            values=["innovation"],
            lifestyle="urban commuter",
            goals=["Ship product"],
            pain_points=["Too many tools"],
            knowledge_domains=["Product management"],
            behaviors=["Reads analytics daily"],
            communication_style=CommunicationStyle(
                tone="professional but warm",
                formality="semi-formal",
                vocabulary_level="advanced",
                preferred_channels=["email"],
            ),
            emotional_profile=EmotionalProfile(
                baseline_mood="optimistic",
                stress_triggers=["deadlines"],
                coping_mechanisms=["exercise"],
            ),
            bio="Clone is a senior product manager at a mid-stage SaaS startup with 10 years of experience.",
        )
        for i in range(n)
    ]
