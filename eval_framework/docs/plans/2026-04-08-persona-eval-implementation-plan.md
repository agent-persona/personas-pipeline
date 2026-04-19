# Persona Eval Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a comprehensive evaluation framework that defines, validates, and continuously monitors LLM-generated persona quality across 44 dimensions and 3 meta-dimensions, establishing the persona output schema through test-driven development before any generation pipeline exists.

**Architecture:** The framework is a Python package (`persona_eval`) organized by evaluation tier (structural, semantic, distributional, bias, behavioral, system, generation, meta). A CLI (`evals`) orchestrates test suites via a registry pattern, routing each dimension to its evaluator class. Results flow to Postgres for trending and Slack for alerting. All LLM calls go through LiteLLM; all embeddings go through a local sentence-transformers wrapper. Tier 1 (structural) gates all downstream tiers — if schema/completeness/consistency fails, expensive LLM-based tests are skipped.

**Tech Stack:**
- Python 3.11+, pytest, Pydantic v2
- LiteLLM (cross-model LLM calls)
- sentence-transformers (local embeddings)
- Hypothesis (property-based testing)
- Click (CLI)
- psycopg2-binary (Postgres)
- slack-sdk (alerting)
- httpx (HTTP client)
- scipy, numpy (statistical tests)
- scikit-learn (classifiers, metrics)
- transformers (NLI models, emotion detection)

---

## Phase 1 — Foundation (Tasks 1-4)

### Task 1: Python Package Structure

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/pyproject.toml`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/version.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/conftest.py`

**Steps:**

1. Write failing test that imports the package (2 min):

```bash
# First, create test directory
mkdir -p /Users/ivanma/Desktop/gauntlet/Capstone/tests
```

Create `tests/__init__.py` (empty) and `tests/test_package.py`:

```python
# tests/test_package.py
def test_package_imports():
    import persona_eval
    assert hasattr(persona_eval, "__version__")


def test_version_is_string():
    from persona_eval.version import __version__
    assert isinstance(__version__, str)
    assert len(__version__.split(".")) == 3
```

2. Verify test fails (1 min):

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/test_package.py -v
# Expected: ModuleNotFoundError: No module named 'persona_eval'
```

3. Create package structure (3 min):

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "persona-eval"
version = "0.1.0"
description = "Evaluation framework for LLM-generated personas"
requires-python = ">=3.11"
dependencies = [
    "pytest>=7.4",
    "pydantic>=2.0",
    "litellm>=1.40",
    "sentence-transformers>=2.7",
    "psycopg2-binary>=2.9",
    "click>=8.1",
    "httpx>=0.27",
    "slack-sdk>=3.27",
    "scipy>=1.13",
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "transformers>=4.40",
    "hypothesis>=6.100",
    "torch>=2.2",
]

[project.scripts]
evals = "persona_eval.cli:cli"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "llm: marks tests that require LLM API calls",
    "gpu: marks tests that benefit from GPU",
]

[tool.setuptools.packages.find]
include = ["persona_eval*"]
```

```python
# persona_eval/__init__.py
"""Persona evaluation framework for LLM-generated personas."""

from persona_eval.version import __version__

__all__ = ["__version__"]
```

```python
# persona_eval/version.py
__version__ = "0.1.0"
```

```python
# tests/conftest.py
"""Shared fixtures for persona eval tests."""

import pytest


@pytest.fixture
def sample_persona_dict():
    """Minimal valid persona dictionary for testing."""
    return {
        "id": "persona-001",
        "identity": {
            "name": "Sarah Chen",
            "age": 34,
            "gender": "female",
            "location": "San Francisco, CA",
        },
        "demographics": {
            "education_level": "Master's degree",
            "income_bracket": "upper-middle",
            "marital_status": "married",
            "household_size": 3,
            "ethnicity": "Asian American",
        },
        "professional": {
            "role": "Senior Product Manager",
            "industry": "SaaS / B2B Technology",
            "company_size": "500-1000",
            "years_experience": 8,
            "team_size": 6,
            "responsibilities": [
                "Product roadmap ownership",
                "Cross-functional team coordination",
                "Customer discovery interviews",
            ],
        },
        "behavioral": {
            "technology_adoption": "early_majority",
            "decision_making_style": "data-driven with intuitive checks",
            "information_sources": [
                "Industry newsletters",
                "Peer recommendations",
                "Product review sites",
            ],
            "purchase_triggers": [
                "Team productivity bottlenecks",
                "Competitive pressure",
            ],
            "brand_loyalty": "moderate",
        },
        "psychographic": {
            "personality_traits": [
                "analytical",
                "collaborative",
                "pragmatic",
            ],
            "risk_tolerance": "moderate",
            "work_life_balance_priority": "high",
            "innovation_orientation": "incremental over radical",
        },
        "communication_style": {
            "tone": "professional but warm",
            "vocabulary_level": "advanced",
            "preferred_channels": ["email", "Slack", "video calls"],
            "formality": "semi-formal",
            "verbosity": "concise",
            "persuasion_responsiveness": {
                "responds_to": ["data", "case studies", "peer validation"],
                "resistant_to": ["hard sells", "urgency tactics"],
            },
        },
        "goals": [
            {
                "description": "Ship the v2 platform redesign by Q3",
                "timeframe": "6 months",
                "priority": "high",
            },
            {
                "description": "Reduce customer churn by 15%",
                "timeframe": "12 months",
                "priority": "high",
            },
            {
                "description": "Build a data-informed product culture on the team",
                "timeframe": "ongoing",
                "priority": "medium",
            },
        ],
        "pain_points": [
            {
                "description": "Too many tools with overlapping functionality",
                "severity": "high",
                "frequency": "daily",
            },
            {
                "description": "Difficulty getting engineering buy-in without hard metrics",
                "severity": "medium",
                "frequency": "weekly",
            },
            {
                "description": "Stakeholder alignment meetings consume 40% of the week",
                "severity": "high",
                "frequency": "daily",
            },
        ],
        "values": [
            "Transparency in decision-making",
            "Continuous learning",
            "Empowering team autonomy",
            "User-centricity over feature shipping",
        ],
        "knowledge_domains": [
            {
                "domain": "Product management methodologies",
                "depth": "expert",
            },
            {
                "domain": "B2B SaaS metrics and analytics",
                "depth": "advanced",
            },
            {
                "domain": "User research methods",
                "depth": "advanced",
            },
            {
                "domain": "Machine learning concepts",
                "depth": "intermediate",
            },
            {
                "domain": "Enterprise sales cycles",
                "depth": "basic",
            },
        ],
        "emotional_profile": {
            "baseline_mood": "optimistic but realistic",
            "stress_response": "becomes more structured and process-oriented",
            "conflict_style": "collaborative, seeks win-win",
            "enthusiasm_triggers": [
                "Novel user insights",
                "Cross-team breakthroughs",
            ],
            "frustration_triggers": [
                "Bureaucratic delays",
                "Decisions made without data",
            ],
        },
        "source_context": "Aggregated from 12 customer interviews with mid-market SaaS product managers, supplemented by industry survey data from ProductPlan 2025 State of Product Management report. Key themes: tool fatigue, stakeholder management overhead, desire for data-driven culture. Demographic profile reflects the modal respondent cluster from k-means analysis of interview transcripts.",
    }


@pytest.fixture
def sample_source_blob():
    """Sample source context blob for grounding tests."""
    return """
    Interview Transcript Summary — Mid-Market SaaS Product Managers (n=12)

    Demographics: 8 female, 4 male. Ages 28-42 (median 34). All based in US tech hubs.
    Education: 10 hold Master's degrees, 2 Bachelor's. Mix of MBA and technical backgrounds.
    Experience: 5-12 years in product roles (median 8 years).

    Key Finding 1 — Tool Fatigue:
    11 of 12 respondents reported using 5+ software tools daily with significant overlap.
    Quote: "I have three different places where I could check project status and none of them agree."
    Quote: "We adopted a new tool every quarter last year. Nobody knows which is the source of truth."

    Key Finding 2 — Stakeholder Alignment Overhead:
    9 of 12 spend >30% of their week in alignment meetings.
    Quote: "I spend more time getting buy-in than actually building product."
    Quote: "The hardest part of my job is not the product — it's the politics."

    Key Finding 3 — Data-Driven Culture Aspiration:
    All 12 expressed desire for more data-informed decision making.
    Quote: "I know we should be more metrics-driven but getting engineering to instrument things is a battle."
    8 of 12 described their current culture as "opinion-driven" or "HiPPO-driven."

    Key Finding 4 — Team Dynamics:
    Median team size: 6 direct reports. Range: 3-15.
    Technology adoption: majority described as "early majority" — want proven solutions, not bleeding edge.
    Decision style: data-driven but tempered by instinct. "I trust the numbers but I also trust my gut."

    Survey Data (ProductPlan 2025):
    - 67% of PMs report tool fatigue as top operational pain point
    - Average PM spends 38% of time in meetings
    - Top goal: "ship key initiative on time" (78%)
    - #2 goal: "reduce churn" (61%)
    - #3 goal: "build product culture" (45%)
    """
```

4. Install package in editable mode and verify tests pass (2 min):

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && pip install -e ".[dev]" 2>/dev/null || pip install -e .
python -m pytest tests/test_package.py -v
# Expected: 2 passed
```

5. Commit (1 min):

```bash
git add pyproject.toml persona_eval/ tests/
git commit -m "feat: scaffold persona_eval package with project structure and shared fixtures"
```

---

### Task 2: Persona Schema (Pydantic Models)

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/schemas/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/schemas/persona.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/schemas/eval_result.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/schemas/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/schemas/test_persona_schema.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/schemas/test_eval_result.py`

**Steps:**

1. Write failing tests for the persona schema (3 min):

```python
# tests/schemas/__init__.py
# empty

# tests/schemas/test_persona_schema.py
"""Tests for persona Pydantic schema — derived from all 44 eval dimensions."""

import pytest
from pydantic import ValidationError


def test_persona_schema_importable():
    from persona_eval.schemas.persona import Persona
    assert Persona is not None


def test_valid_persona_parses(sample_persona_dict):
    from persona_eval.schemas.persona import Persona
    persona = Persona(**sample_persona_dict)
    assert persona.id == "persona-001"
    assert persona.identity.name == "Sarah Chen"


def test_identity_fields_required():
    from persona_eval.schemas.persona import Persona
    with pytest.raises(ValidationError) as exc_info:
        Persona(id="test")
    errors = exc_info.value.errors()
    missing_fields = {e["loc"][0] for e in errors}
    assert "identity" in missing_fields


def test_all_top_level_sections_present(sample_persona_dict):
    """Every section needed for the 44 dimensions must exist."""
    from persona_eval.schemas.persona import Persona
    persona = Persona(**sample_persona_dict)
    required_sections = [
        "identity", "demographics", "professional", "behavioral",
        "psychographic", "communication_style", "goals", "pain_points",
        "values", "knowledge_domains", "emotional_profile", "source_context",
    ]
    for section in required_sections:
        assert hasattr(persona, section), f"Missing section: {section}"
        assert getattr(persona, section) is not None, f"Section is None: {section}"


def test_goals_have_required_fields(sample_persona_dict):
    from persona_eval.schemas.persona import Persona
    persona = Persona(**sample_persona_dict)
    for goal in persona.goals:
        assert goal.description
        assert goal.timeframe
        assert goal.priority


def test_pain_points_have_required_fields(sample_persona_dict):
    from persona_eval.schemas.persona import Persona
    persona = Persona(**sample_persona_dict)
    for pp in persona.pain_points:
        assert pp.description
        assert pp.severity
        assert pp.frequency


def test_knowledge_domains_have_depth(sample_persona_dict):
    from persona_eval.schemas.persona import Persona
    persona = Persona(**sample_persona_dict)
    valid_depths = {"beginner", "basic", "intermediate", "advanced", "expert"}
    for kd in persona.knowledge_domains:
        assert kd.domain
        assert kd.depth in valid_depths


def test_enum_validation_rejects_bad_values(sample_persona_dict):
    from persona_eval.schemas.persona import Persona
    from pydantic import ValidationError
    bad = sample_persona_dict.copy()
    bad["behavioral"] = {**sample_persona_dict["behavioral"], "technology_adoption": "INVALID_VALUE"}
    with pytest.raises(ValidationError):
        Persona(**bad)


def test_persona_serialization_roundtrip(sample_persona_dict):
    from persona_eval.schemas.persona import Persona
    persona = Persona(**sample_persona_dict)
    json_str = persona.model_dump_json()
    persona2 = Persona.model_validate_json(json_str)
    assert persona.id == persona2.id
    assert persona.identity.name == persona2.identity.name


def test_source_context_is_string(sample_persona_dict):
    from persona_eval.schemas.persona import Persona
    persona = Persona(**sample_persona_dict)
    assert isinstance(persona.source_context, str)
    assert len(persona.source_context) > 0
```

2. Verify tests fail (1 min):

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/schemas/test_persona_schema.py -v
# Expected: ModuleNotFoundError
```

3. Implement the persona schema (5 min):

```python
# persona_eval/schemas/__init__.py
"""Pydantic schemas for personas and evaluation results."""

from persona_eval.schemas.persona import Persona
from persona_eval.schemas.eval_result import EvalResult, DimensionResult

__all__ = ["Persona", "EvalResult", "DimensionResult"]
```

```python
# persona_eval/schemas/persona.py
"""
Persona schema — derived from all 44 evaluation dimensions.

Every field exists because at least one dimension (D1-D44) needs it to be testable.
The schema IS the spec: if it is not in this schema, it cannot be evaluated.

Field-to-dimension mapping:
- identity: D1, D6, D8, D35
- demographics: D7, D18, D21, D23
- professional: D3, D7, D9, D22
- behavioral: D5, D13, D14, D20
- psychographic: D6, D13, D16, D27
- communication_style: D10, D11, D25, D32
- goals: D2, D4, D11, D12
- pain_points: D2, D4, D24
- values: D16, D27, D28
- knowledge_domains: D9, D22, D29
- emotional_profile: D25, D26, D33
- source_context: D4, D43
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TechnologyAdoption(str, Enum):
    innovator = "innovator"
    early_adopter = "early_adopter"
    early_majority = "early_majority"
    late_majority = "late_majority"
    laggard = "laggard"


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Frequency(str, Enum):
    rarely = "rarely"
    monthly = "monthly"
    weekly = "weekly"
    daily = "daily"
    hourly = "hourly"


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class KnowledgeDepth(str, Enum):
    beginner = "beginner"
    basic = "basic"
    intermediate = "intermediate"
    advanced = "advanced"
    expert = "expert"


class Formality(str, Enum):
    very_informal = "very_informal"
    informal = "informal"
    semi_formal = "semi-formal"
    formal = "formal"
    very_formal = "very_formal"


class Verbosity(str, Enum):
    terse = "terse"
    concise = "concise"
    moderate = "moderate"
    detailed = "detailed"
    verbose = "verbose"


# --- Sub-models ---

class Identity(BaseModel):
    """Core identity fields. Tested by D1, D6, D8, D35."""
    name: str = Field(..., min_length=1, description="Full name")
    age: int = Field(..., ge=18, le=120, description="Age in years")
    gender: str = Field(..., min_length=1, description="Gender identity")
    location: str = Field(..., min_length=1, description="Primary location")


class Demographics(BaseModel):
    """Demographic attributes. Tested by D7, D18, D21, D23."""
    education_level: str = Field(..., min_length=1)
    income_bracket: str = Field(..., min_length=1)
    marital_status: str = Field(..., min_length=1)
    household_size: int = Field(..., ge=1, le=20)
    ethnicity: str = Field(..., min_length=1)


class Professional(BaseModel):
    """Professional context. Tested by D3, D7, D9, D22."""
    role: str = Field(..., min_length=1)
    industry: str = Field(..., min_length=1)
    company_size: str = Field(..., min_length=1)
    years_experience: int = Field(..., ge=0, le=60)
    team_size: Optional[int] = Field(None, ge=0, le=10000)
    responsibilities: list[str] = Field(default_factory=list, min_length=1)


class Behavioral(BaseModel):
    """Behavioral patterns. Tested by D5, D13, D14, D20."""
    technology_adoption: TechnologyAdoption
    decision_making_style: str = Field(..., min_length=1)
    information_sources: list[str] = Field(default_factory=list, min_length=1)
    purchase_triggers: list[str] = Field(default_factory=list)
    brand_loyalty: str = Field(..., min_length=1)


class Psychographic(BaseModel):
    """Psychographic traits. Tested by D6, D13, D16, D27."""
    personality_traits: list[str] = Field(default_factory=list, min_length=1)
    risk_tolerance: str = Field(..., min_length=1)
    work_life_balance_priority: str = Field(..., min_length=1)
    innovation_orientation: str = Field(..., min_length=1)


class PersuasionResponsiveness(BaseModel):
    """What persuasion tactics work/don't work."""
    responds_to: list[str] = Field(default_factory=list)
    resistant_to: list[str] = Field(default_factory=list)


class CommunicationStyle(BaseModel):
    """Communication preferences. Tested by D10, D11, D25, D32."""
    tone: str = Field(..., min_length=1)
    vocabulary_level: str = Field(..., min_length=1)
    preferred_channels: list[str] = Field(default_factory=list, min_length=1)
    formality: Formality
    verbosity: Verbosity
    persuasion_responsiveness: Optional[PersuasionResponsiveness] = None


class Goal(BaseModel):
    """A persona goal. Tested by D2, D4, D11, D12."""
    description: str = Field(..., min_length=1)
    timeframe: str = Field(..., min_length=1)
    priority: Priority


class PainPoint(BaseModel):
    """A persona pain point. Tested by D2, D4, D24."""
    description: str = Field(..., min_length=1)
    severity: Severity
    frequency: Frequency


class KnowledgeDomain(BaseModel):
    """A knowledge domain with depth. Tested by D9, D22, D29."""
    domain: str = Field(..., min_length=1)
    depth: KnowledgeDepth


class EmotionalProfile(BaseModel):
    """Emotional characteristics. Tested by D25, D26, D33."""
    baseline_mood: str = Field(..., min_length=1)
    stress_response: str = Field(..., min_length=1)
    conflict_style: str = Field(..., min_length=1)
    enthusiasm_triggers: list[str] = Field(default_factory=list, min_length=1)
    frustration_triggers: list[str] = Field(default_factory=list, min_length=1)


# --- Root model ---

class Persona(BaseModel):
    """
    Complete persona schema.

    Derived from the 44 evaluation dimensions — every field exists because
    at least one dimension needs it. This schema IS the persona spec.
    """

    id: str = Field(..., min_length=1, description="Unique persona identifier")
    identity: Identity
    demographics: Demographics
    professional: Professional
    behavioral: Behavioral
    psychographic: Psychographic
    communication_style: CommunicationStyle
    goals: list[Goal] = Field(..., min_length=1)
    pain_points: list[PainPoint] = Field(..., min_length=1)
    values: list[str] = Field(..., min_length=1)
    knowledge_domains: list[KnowledgeDomain] = Field(..., min_length=1)
    emotional_profile: EmotionalProfile
    source_context: str = Field(
        ...,
        min_length=1,
        description="Opaque blob of source data the persona was derived from",
    )

    model_config = {"str_strip_whitespace": True}
```

4. Verify tests pass (1 min):

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/schemas/test_persona_schema.py -v
# Expected: all tests pass
```

5. Write failing tests for EvalResult schema (2 min):

```python
# tests/schemas/test_eval_result.py
"""Tests for eval result schema."""

import pytest
from datetime import datetime, timezone


def test_eval_result_importable():
    from persona_eval.schemas.eval_result import EvalResult, DimensionResult
    assert EvalResult is not None
    assert DimensionResult is not None


def test_dimension_result_creation():
    from persona_eval.schemas.eval_result import DimensionResult
    result = DimensionResult(
        dimension_id="D1",
        dimension_name="Schema Compliance",
        tier=1,
        passed=True,
        score=1.0,
        details={"validation_errors": 0},
    )
    assert result.dimension_id == "D1"
    assert result.passed is True


def test_dimension_result_score_bounds():
    from persona_eval.schemas.eval_result import DimensionResult
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        DimensionResult(
            dimension_id="D1",
            dimension_name="Schema Compliance",
            tier=1,
            passed=True,
            score=1.5,  # out of bounds
        )


def test_eval_result_aggregation():
    from persona_eval.schemas.eval_result import EvalResult, DimensionResult
    dims = [
        DimensionResult(dimension_id="D1", dimension_name="Schema Compliance", tier=1, passed=True, score=1.0),
        DimensionResult(dimension_id="D2", dimension_name="Completeness", tier=1, passed=False, score=0.6),
    ]
    result = EvalResult(
        persona_id="persona-001",
        suite="persona",
        model="claude-sonnet-4-20250514",
        dimensions=dims,
    )
    assert result.persona_id == "persona-001"
    assert result.overall_passed is False  # one dimension failed
    assert result.overall_score == pytest.approx(0.8)  # mean of 1.0 and 0.6


def test_eval_result_has_timestamp():
    from persona_eval.schemas.eval_result import EvalResult, DimensionResult
    result = EvalResult(
        persona_id="p1",
        suite="persona",
        model="test",
        dimensions=[
            DimensionResult(dimension_id="D1", dimension_name="test", tier=1, passed=True, score=1.0),
        ],
    )
    assert result.timestamp is not None
    assert isinstance(result.timestamp, datetime)


def test_eval_result_gating_tier():
    """Tier 1 failure should be detectable for gating."""
    from persona_eval.schemas.eval_result import EvalResult, DimensionResult
    dims = [
        DimensionResult(dimension_id="D1", dimension_name="Schema", tier=1, passed=False, score=0.0),
        DimensionResult(dimension_id="D4", dimension_name="Grounding", tier=2, passed=True, score=0.9),
    ]
    result = EvalResult(persona_id="p1", suite="persona", model="test", dimensions=dims)
    assert result.tier_passed(1) is False
    assert result.tier_passed(2) is True
```

6. Implement EvalResult schema (3 min):

```python
# persona_eval/schemas/eval_result.py
"""Evaluation result schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, computed_field


class DimensionResult(BaseModel):
    """Result for a single evaluation dimension."""

    dimension_id: str = Field(..., description="e.g. D1, D2, M1")
    dimension_name: str = Field(..., description="Human-readable name")
    tier: int = Field(..., ge=1, le=7, description="Evaluation tier (1-7)")
    passed: bool = Field(..., description="Whether this dimension passed")
    score: float = Field(..., ge=0.0, le=1.0, description="Score 0-1")
    details: dict[str, Any] = Field(default_factory=dict, description="Dimension-specific details")
    error: Optional[str] = Field(None, description="Error message if evaluation failed to run")

    @field_validator("score")
    @classmethod
    def score_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be between 0 and 1, got {v}")
        return v


class EvalResult(BaseModel):
    """Aggregate result for a full persona evaluation run."""

    persona_id: str
    suite: str
    model: str
    dimensions: list[DimensionResult]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def overall_passed(self) -> bool:
        """True only if ALL dimensions passed."""
        return all(d.passed for d in self.dimensions)

    @computed_field
    @property
    def overall_score(self) -> float:
        """Mean score across all dimensions."""
        if not self.dimensions:
            return 0.0
        return sum(d.score for d in self.dimensions) / len(self.dimensions)

    def tier_passed(self, tier: int) -> bool:
        """Check if all dimensions in a specific tier passed."""
        tier_dims = [d for d in self.dimensions if d.tier == tier]
        if not tier_dims:
            return True  # no dims in this tier = vacuously true
        return all(d.passed for d in tier_dims)

    def tier_scores(self) -> dict[int, float]:
        """Average score per tier."""
        from collections import defaultdict
        tier_sums: dict[int, list[float]] = defaultdict(list)
        for d in self.dimensions:
            tier_sums[d.tier].append(d.score)
        return {tier: sum(scores) / len(scores) for tier, scores in tier_sums.items()}
```

7. Verify tests pass (1 min):

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/schemas/ -v
# Expected: all tests pass
```

8. Commit (1 min):

```bash
git add persona_eval/schemas/ tests/schemas/
git commit -m "feat: define Persona and EvalResult Pydantic schemas derived from 44 eval dimensions"
```

---

### Task 3: CLI Entry Point

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/cli.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/registry.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/test_cli.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/test_registry.py`

**Steps:**

1. Write failing tests for registry pattern (2 min):

```python
# tests/test_registry.py
"""Tests for the eval suite registry."""

import pytest


def test_registry_importable():
    from persona_eval.registry import SuiteRegistry
    assert SuiteRegistry is not None


def test_register_and_retrieve_suite():
    from persona_eval.registry import SuiteRegistry

    registry = SuiteRegistry()

    @registry.register("test_suite")
    class TestSuite:
        name = "test_suite"
        dimensions = ["D1"]

    assert "test_suite" in registry.list_suites()
    assert registry.get("test_suite") is TestSuite


def test_get_unknown_suite_raises():
    from persona_eval.registry import SuiteRegistry
    registry = SuiteRegistry()
    with pytest.raises(KeyError, match="nonexistent"):
        registry.get("nonexistent")


def test_list_suites_empty():
    from persona_eval.registry import SuiteRegistry
    registry = SuiteRegistry()
    assert registry.list_suites() == []
```

2. Verify tests fail, implement registry (3 min):

```python
# persona_eval/registry.py
"""Suite registry — maps suite names to evaluator classes."""

from __future__ import annotations

from typing import Any


class SuiteRegistry:
    """Registry for evaluation suites.

    Usage:
        registry = SuiteRegistry()

        @registry.register("persona")
        class PersonaSuite:
            ...

        suite_cls = registry.get("persona")
    """

    def __init__(self) -> None:
        self._suites: dict[str, Any] = {}

    def register(self, name: str):
        """Decorator to register an eval suite class."""
        def decorator(cls):
            self._suites[name] = cls
            return cls
        return decorator

    def get(self, name: str):
        """Retrieve a registered suite by name."""
        if name not in self._suites:
            raise KeyError(f"Suite not found: {name!r}. Available: {self.list_suites()}")
        return self._suites[name]

    def list_suites(self) -> list[str]:
        """List all registered suite names."""
        return sorted(self._suites.keys())


# Global registry instance
default_registry = SuiteRegistry()
```

3. Write failing tests for CLI (2 min):

```python
# tests/test_cli.py
"""Tests for the CLI entry point."""

import pytest
from click.testing import CliRunner


def test_cli_importable():
    from persona_eval.cli import cli
    assert cli is not None


def test_cli_help():
    from persona_eval.cli import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Persona evaluation framework" in result.output


def test_cli_run_help():
    from persona_eval.cli import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--suite" in result.output
    assert "--model" in result.output
    assert "--output" in result.output


def test_cli_run_unknown_suite():
    from persona_eval.cli import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--suite=nonexistent", "--model=test"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "error" in result.output.lower()


def test_cli_list_suites():
    from persona_eval.cli import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["list-suites"])
    assert result.exit_code == 0
```

4. Implement CLI (3 min):

```python
# persona_eval/cli.py
"""CLI entry point for the persona evaluation framework."""

from __future__ import annotations

import json
import sys

import click

from persona_eval.registry import default_registry


@click.group()
@click.version_option(package_name="persona-eval")
def cli():
    """Persona evaluation framework — test LLM-generated personas across 44 dimensions."""
    pass


@cli.command()
@click.option("--suite", required=True, help="Evaluation suite to run (e.g., persona)")
@click.option("--model", required=True, help="LLM model to use (e.g., claude-sonnet-4-20250514)")
@click.option("--output", type=click.Choice(["json", "table", "postgres"]), default="table", help="Output format")
@click.option("--persona-file", type=click.Path(exists=True), help="Path to persona JSON file")
@click.option("--source-file", type=click.Path(exists=True), help="Path to source context file")
@click.option("--tier", type=int, default=None, help="Run only a specific tier (1-7)")
@click.option("--dimension", type=str, default=None, help="Run only a specific dimension (e.g., D1)")
def run(suite: str, model: str, output: str, persona_file: str | None,
        source_file: str | None, tier: int | None, dimension: str | None):
    """Run an evaluation suite against a persona."""
    try:
        suite_cls = default_registry.get(suite)
    except KeyError as e:
        click.echo(f"Error: Suite not found: {suite!r}. Available: {default_registry.list_suites()}", err=True)
        sys.exit(1)

    click.echo(f"Running suite: {suite} with model: {model}")

    # Load persona if provided
    persona_data = None
    if persona_file:
        with open(persona_file) as f:
            persona_data = json.load(f)

    source_data = None
    if source_file:
        with open(source_file) as f:
            source_data = f.read()

    # Instantiate and run suite
    suite_instance = suite_cls(model=model, tier=tier, dimension=dimension)
    result = suite_instance.run(persona_data=persona_data, source_data=source_data)

    # Output results
    if output == "json":
        click.echo(result.model_dump_json(indent=2))
    elif output == "table":
        _print_table(result)
    elif output == "postgres":
        _store_postgres(result)
        click.echo(f"Results stored in Postgres for persona: {result.persona_id}")


def _print_table(result):
    """Print results as a formatted table."""
    from persona_eval.schemas.eval_result import EvalResult
    click.echo(f"\n{'=' * 70}")
    click.echo(f"Persona: {result.persona_id} | Suite: {result.suite} | Model: {result.model}")
    click.echo(f"Overall: {'PASS' if result.overall_passed else 'FAIL'} | Score: {result.overall_score:.2f}")
    click.echo(f"{'=' * 70}")
    click.echo(f"{'Dim':<6} {'Name':<35} {'Tier':<5} {'Pass':<6} {'Score':<6}")
    click.echo(f"{'-' * 70}")
    for d in result.dimensions:
        status = "PASS" if d.passed else "FAIL"
        click.echo(f"{d.dimension_id:<6} {d.dimension_name:<35} {d.tier:<5} {status:<6} {d.score:<6.2f}")
    click.echo(f"{'=' * 70}\n")


def _store_postgres(result):
    """Store results in Postgres. Placeholder — implemented in Task 4."""
    pass


@cli.command("list-suites")
def list_suites():
    """List all available evaluation suites."""
    suites = default_registry.list_suites()
    if not suites:
        click.echo("No suites registered.")
    else:
        click.echo("Available suites:")
        for s in suites:
            click.echo(f"  - {s}")
```

5. Verify tests pass (1 min):

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/test_cli.py tests/test_registry.py -v
# Expected: all tests pass
```

6. Commit (1 min):

```bash
git add persona_eval/cli.py persona_eval/registry.py tests/test_cli.py tests/test_registry.py
git commit -m "feat: add CLI entry point with Click and suite registry pattern"
```

---

### Task 4: Result Storage (Postgres + ResultRecorder)

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/storage/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/storage/recorder.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/storage/schema.sql`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/storage/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/storage/test_recorder.py`

**Steps:**

1. Write failing tests for ResultRecorder (3 min):

```python
# tests/storage/__init__.py
# empty

# tests/storage/test_recorder.py
"""Tests for result storage. Uses SQLite in-memory for test isolation."""

import pytest
from datetime import datetime, timezone


def test_recorder_importable():
    from persona_eval.storage.recorder import ResultRecorder
    assert ResultRecorder is not None


def test_recorder_stores_and_retrieves(tmp_path):
    from persona_eval.storage.recorder import ResultRecorder
    from persona_eval.schemas.eval_result import EvalResult, DimensionResult

    db_path = tmp_path / "test.db"
    recorder = ResultRecorder(f"sqlite:///{db_path}")

    result = EvalResult(
        persona_id="p1",
        suite="persona",
        model="test-model",
        dimensions=[
            DimensionResult(
                dimension_id="D1", dimension_name="Schema Compliance",
                tier=1, passed=True, score=1.0, details={"errors": 0},
            ),
        ],
    )

    recorder.store(result)
    retrieved = recorder.get_latest("p1", "persona")
    assert retrieved is not None
    assert retrieved.persona_id == "p1"
    assert len(retrieved.dimensions) == 1
    assert retrieved.dimensions[0].score == 1.0


def test_recorder_get_history(tmp_path):
    from persona_eval.storage.recorder import ResultRecorder
    from persona_eval.schemas.eval_result import EvalResult, DimensionResult

    db_path = tmp_path / "test.db"
    recorder = ResultRecorder(f"sqlite:///{db_path}")

    for i in range(3):
        result = EvalResult(
            persona_id="p1",
            suite="persona",
            model="test-model",
            dimensions=[
                DimensionResult(
                    dimension_id="D1", dimension_name="Schema",
                    tier=1, passed=True, score=0.5 + i * 0.1,
                ),
            ],
        )
        recorder.store(result)

    history = recorder.get_history("p1", "persona", limit=10)
    assert len(history) == 3


def test_recorder_detect_regression(tmp_path):
    from persona_eval.storage.recorder import ResultRecorder
    from persona_eval.schemas.eval_result import EvalResult, DimensionResult

    db_path = tmp_path / "test.db"
    recorder = ResultRecorder(f"sqlite:///{db_path}")

    # Store a good baseline
    baseline = EvalResult(
        persona_id="p1", suite="persona", model="m",
        dimensions=[
            DimensionResult(dimension_id="D1", dimension_name="Schema", tier=1, passed=True, score=0.95),
        ],
    )
    recorder.store(baseline)

    # Store a regression
    regression = EvalResult(
        persona_id="p1", suite="persona", model="m",
        dimensions=[
            DimensionResult(dimension_id="D1", dimension_name="Schema", tier=1, passed=False, score=0.3),
        ],
    )
    recorder.store(regression)

    regressions = recorder.detect_regressions("p1", "persona", threshold=0.2)
    assert len(regressions) > 0
    assert "D1" in regressions
```

2. Verify tests fail, then implement (3 min):

```sql
-- persona_eval/storage/schema.sql
-- Postgres schema for eval results. For local dev/test we use SQLite.

CREATE TABLE IF NOT EXISTS eval_runs (
    id              SERIAL PRIMARY KEY,
    persona_id      TEXT NOT NULL,
    suite           TEXT NOT NULL,
    model           TEXT NOT NULL,
    overall_passed  BOOLEAN NOT NULL,
    overall_score   REAL NOT NULL,
    timestamp       TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    metadata        JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS dimension_results (
    id              SERIAL PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    dimension_id    TEXT NOT NULL,
    dimension_name  TEXT NOT NULL,
    tier            INTEGER NOT NULL,
    passed          BOOLEAN NOT NULL,
    score           REAL NOT NULL,
    details         JSONB DEFAULT '{}',
    error           TEXT
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_persona ON eval_runs(persona_id, suite);
CREATE INDEX IF NOT EXISTS idx_eval_runs_timestamp ON eval_runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_dimension_results_run ON dimension_results(run_id);
```

```python
# persona_eval/storage/__init__.py
"""Result storage backends."""

from persona_eval.storage.recorder import ResultRecorder

__all__ = ["ResultRecorder"]
```

```python
# persona_eval/storage/recorder.py
"""ResultRecorder — stores and retrieves evaluation results.

Supports both Postgres (production) and SQLite (test/dev).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from persona_eval.schemas.eval_result import DimensionResult, EvalResult


class ResultRecorder:
    """Stores eval results in a SQL database."""

    def __init__(self, dsn: str):
        """Initialize with a DSN. sqlite:///path for SQLite, postgresql://... for Postgres."""
        self._dsn = dsn
        self._is_sqlite = dsn.startswith("sqlite")
        if self._is_sqlite:
            db_path = dsn.replace("sqlite:///", "")
            self._conn = sqlite3.connect(db_path)
            self._conn.row_factory = sqlite3.Row
            self._init_sqlite()
        else:
            import psycopg2
            self._conn = psycopg2.connect(dsn)
            self._init_postgres()

    def _init_sqlite(self):
        """Create tables for SQLite."""
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                persona_id      TEXT NOT NULL,
                suite           TEXT NOT NULL,
                model           TEXT NOT NULL,
                overall_passed  BOOLEAN NOT NULL,
                overall_score   REAL NOT NULL,
                timestamp       TEXT NOT NULL,
                metadata        TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS dimension_results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          INTEGER NOT NULL REFERENCES eval_runs(id),
                dimension_id    TEXT NOT NULL,
                dimension_name  TEXT NOT NULL,
                tier            INTEGER NOT NULL,
                passed          BOOLEAN NOT NULL,
                score           REAL NOT NULL,
                details         TEXT DEFAULT '{}',
                error           TEXT
            );
        """)
        self._conn.commit()

    def _init_postgres(self):
        """Create tables for Postgres using the schema.sql file."""
        import importlib.resources
        from pathlib import Path
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path) as f:
            sql = f.read()
        cur = self._conn.cursor()
        cur.execute(sql)
        self._conn.commit()

    def store(self, result: EvalResult) -> int:
        """Store an eval result. Returns the run ID."""
        cur = self._conn.cursor()
        ts = result.timestamp.isoformat()
        meta = json.dumps(result.metadata)

        if self._is_sqlite:
            cur.execute(
                """INSERT INTO eval_runs (persona_id, suite, model, overall_passed, overall_score, timestamp, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (result.persona_id, result.suite, result.model,
                 result.overall_passed, result.overall_score, ts, meta),
            )
            run_id = cur.lastrowid
            for d in result.dimensions:
                cur.execute(
                    """INSERT INTO dimension_results (run_id, dimension_id, dimension_name, tier, passed, score, details, error)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (run_id, d.dimension_id, d.dimension_name, d.tier,
                     d.passed, d.score, json.dumps(d.details), d.error),
                )
        else:
            cur.execute(
                """INSERT INTO eval_runs (persona_id, suite, model, overall_passed, overall_score, timestamp, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (result.persona_id, result.suite, result.model,
                 result.overall_passed, result.overall_score, ts, meta),
            )
            run_id = cur.fetchone()[0]
            for d in result.dimensions:
                cur.execute(
                    """INSERT INTO dimension_results (run_id, dimension_id, dimension_name, tier, passed, score, details, error)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (run_id, d.dimension_id, d.dimension_name, d.tier,
                     d.passed, d.score, json.dumps(d.details), d.error),
                )

        self._conn.commit()
        return run_id

    def get_latest(self, persona_id: str, suite: str) -> Optional[EvalResult]:
        """Get the most recent eval result for a persona/suite."""
        cur = self._conn.cursor()
        placeholder = "?" if self._is_sqlite else "%s"
        cur.execute(
            f"""SELECT * FROM eval_runs
                WHERE persona_id = {placeholder} AND suite = {placeholder}
                ORDER BY timestamp DESC LIMIT 1""",
            (persona_id, suite),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_result(row)

    def get_history(self, persona_id: str, suite: str, limit: int = 50) -> list[EvalResult]:
        """Get eval history for a persona/suite."""
        cur = self._conn.cursor()
        placeholder = "?" if self._is_sqlite else "%s"
        cur.execute(
            f"""SELECT * FROM eval_runs
                WHERE persona_id = {placeholder} AND suite = {placeholder}
                ORDER BY timestamp DESC LIMIT {placeholder}""",
            (persona_id, suite, limit),
        )
        return [self._row_to_result(row) for row in cur.fetchall()]

    def detect_regressions(self, persona_id: str, suite: str, threshold: float = 0.1) -> dict[str, float]:
        """Detect dimension score regressions between the two most recent runs.

        Returns dict mapping dimension_id -> score drop for any dimension
        that dropped by more than threshold.
        """
        history = self.get_history(persona_id, suite, limit=2)
        if len(history) < 2:
            return {}

        current = {d.dimension_id: d.score for d in history[0].dimensions}
        previous = {d.dimension_id: d.score for d in history[1].dimensions}

        regressions = {}
        for dim_id, curr_score in current.items():
            prev_score = previous.get(dim_id)
            if prev_score is not None:
                drop = prev_score - curr_score
                if drop > threshold:
                    regressions[dim_id] = drop

        return regressions

    def _row_to_result(self, row) -> EvalResult:
        """Convert a database row to an EvalResult."""
        run_id = row["id"] if isinstance(row, sqlite3.Row) else row[0]
        cur = self._conn.cursor()
        placeholder = "?" if self._is_sqlite else "%s"
        cur.execute(
            f"SELECT * FROM dimension_results WHERE run_id = {placeholder}",
            (run_id,),
        )
        dims = []
        for drow in cur.fetchall():
            if isinstance(drow, sqlite3.Row):
                dims.append(DimensionResult(
                    dimension_id=drow["dimension_id"],
                    dimension_name=drow["dimension_name"],
                    tier=drow["tier"],
                    passed=bool(drow["passed"]),
                    score=drow["score"],
                    details=json.loads(drow["details"]) if drow["details"] else {},
                    error=drow["error"],
                ))
            else:
                dims.append(DimensionResult(
                    dimension_id=drow[2], dimension_name=drow[3],
                    tier=drow[4], passed=bool(drow[5]), score=drow[6],
                    details=json.loads(drow[7]) if drow[7] else {},
                    error=drow[8],
                ))

        ts_str = row["timestamp"] if isinstance(row, sqlite3.Row) else row[5]
        ts = datetime.fromisoformat(ts_str) if isinstance(ts_str, str) else ts_str
        meta_str = row["metadata"] if isinstance(row, sqlite3.Row) else row[6]
        meta = json.loads(meta_str) if isinstance(meta_str, str) else meta_str

        return EvalResult(
            persona_id=row["persona_id"] if isinstance(row, sqlite3.Row) else row[1],
            suite=row["suite"] if isinstance(row, sqlite3.Row) else row[2],
            model=row["model"] if isinstance(row, sqlite3.Row) else row[3],
            dimensions=dims,
            timestamp=ts,
            metadata=meta or {},
        )
```

3. Verify tests pass (1 min):

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/storage/test_recorder.py -v
# Expected: all tests pass
```

4. Commit (1 min):

```bash
git add persona_eval/storage/ tests/storage/
git commit -m "feat: add ResultRecorder with Postgres/SQLite support and regression detection"
```

---

## Phase 2 — Tier 1: Structural Validators (Tasks 5-7)

### Task 5: D1 Schema Compliance

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/base.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/structural/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/structural/d01_schema_compliance.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/structural/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/structural/test_d01_schema_compliance.py`

**Steps:**

1. Write failing tests (3 min):

```python
# tests/evaluators/__init__.py
# empty

# tests/evaluators/structural/__init__.py
# empty

# tests/evaluators/structural/test_d01_schema_compliance.py
"""Tests for D1 Schema Compliance evaluator."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


def test_evaluator_importable():
    from persona_eval.evaluators.structural.d01_schema_compliance import SchemaComplianceEvaluator
    assert SchemaComplianceEvaluator is not None


def test_valid_persona_passes(sample_persona_dict):
    from persona_eval.evaluators.structural.d01_schema_compliance import SchemaComplianceEvaluator
    evaluator = SchemaComplianceEvaluator()
    result = evaluator.evaluate(sample_persona_dict)
    assert result.passed is True
    assert result.score == 1.0
    assert result.dimension_id == "D1"


def test_missing_required_field_fails(sample_persona_dict):
    from persona_eval.evaluators.structural.d01_schema_compliance import SchemaComplianceEvaluator
    evaluator = SchemaComplianceEvaluator()
    bad_data = {k: v for k, v in sample_persona_dict.items() if k != "identity"}
    result = evaluator.evaluate(bad_data)
    assert result.passed is False
    assert result.score < 1.0
    assert "identity" in str(result.details.get("errors", ""))


def test_wrong_type_fails(sample_persona_dict):
    from persona_eval.evaluators.structural.d01_schema_compliance import SchemaComplianceEvaluator
    evaluator = SchemaComplianceEvaluator()
    bad_data = dict(sample_persona_dict)
    bad_data["identity"] = "not a dict"
    result = evaluator.evaluate(bad_data)
    assert result.passed is False


def test_invalid_enum_fails(sample_persona_dict):
    from persona_eval.evaluators.structural.d01_schema_compliance import SchemaComplianceEvaluator
    evaluator = SchemaComplianceEvaluator()
    bad_data = dict(sample_persona_dict)
    bad_data["behavioral"] = {**sample_persona_dict["behavioral"], "technology_adoption": "GARBAGE"}
    result = evaluator.evaluate(bad_data)
    assert result.passed is False


def test_empty_dict_fails():
    from persona_eval.evaluators.structural.d01_schema_compliance import SchemaComplianceEvaluator
    evaluator = SchemaComplianceEvaluator()
    result = evaluator.evaluate({})
    assert result.passed is False
    assert result.score == 0.0


def test_extra_fields_tolerated(sample_persona_dict):
    from persona_eval.evaluators.structural.d01_schema_compliance import SchemaComplianceEvaluator
    evaluator = SchemaComplianceEvaluator()
    extended = dict(sample_persona_dict)
    extended["extra_field"] = "should be ignored"
    result = evaluator.evaluate(extended)
    assert result.passed is True


@given(st.dictionaries(st.text(min_size=1, max_size=10), st.text(max_size=50), max_size=5))
@settings(max_examples=20)
def test_fuzz_random_dicts_dont_crash(random_dict):
    """Property-based test: random dicts should fail gracefully, never crash."""
    from persona_eval.evaluators.structural.d01_schema_compliance import SchemaComplianceEvaluator
    evaluator = SchemaComplianceEvaluator()
    result = evaluator.evaluate(random_dict)
    assert result.passed is False or result.passed is True
    assert 0.0 <= result.score <= 1.0
```

2. Verify tests fail, then implement (3 min):

```python
# persona_eval/evaluators/__init__.py
"""Evaluator classes for all 44 dimensions."""
```

```python
# persona_eval/evaluators/base.py
"""Base evaluator class for all dimensions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from persona_eval.schemas.eval_result import DimensionResult


class BaseEvaluator(ABC):
    """Abstract base class for dimension evaluators.

    Every evaluator must implement evaluate() and declare its dimension metadata.
    """

    dimension_id: str  # e.g., "D1"
    dimension_name: str  # e.g., "Schema Compliance"
    tier: int  # 1-7

    @abstractmethod
    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        """Evaluate a persona on this dimension.

        Args:
            persona_data: Raw persona dict (pre-validation for D1, post-validation for others).
            **kwargs: Additional context (source_blob, persona_set, etc.)

        Returns:
            DimensionResult with score, pass/fail, and details.
        """
        ...

    def _make_result(
        self,
        passed: bool,
        score: float,
        details: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> DimensionResult:
        """Convenience method to create a DimensionResult with this evaluator's metadata."""
        return DimensionResult(
            dimension_id=self.dimension_id,
            dimension_name=self.dimension_name,
            tier=self.tier,
            passed=passed,
            score=score,
            details=details or {},
            error=error,
        )
```

```python
# persona_eval/evaluators/structural/__init__.py
"""Tier 1 — Structural validators (D1-D3)."""
```

```python
# persona_eval/evaluators/structural/d01_schema_compliance.py
"""D1 Schema Compliance — Pydantic validation with property-based testing.

Trustworthiness: HIGH (deterministic, no ambiguity).
Method: JSON Schema validation via Pydantic v2.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult
from persona_eval.schemas.persona import Persona


class SchemaComplianceEvaluator(BaseEvaluator):
    """Evaluates whether a persona dict conforms to the expected schema."""

    dimension_id = "D1"
    dimension_name = "Schema Compliance"
    tier = 1

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        """Validate persona_data against the Persona Pydantic model.

        Scoring:
        - 1.0 = fully valid
        - partial = (total_fields - error_fields) / total_fields
        - 0.0 = empty or completely invalid
        """
        if not persona_data:
            return self._make_result(
                passed=False,
                score=0.0,
                details={"errors": ["Empty persona data"]},
            )

        try:
            Persona(**persona_data)
            return self._make_result(
                passed=True,
                score=1.0,
                details={"validation_errors": 0},
            )
        except ValidationError as e:
            errors = e.errors()
            # Count unique top-level fields with errors
            error_fields = {err["loc"][0] for err in errors if err["loc"]}
            total_fields = len(Persona.model_fields)
            valid_fields = total_fields - len(error_fields)
            score = max(0.0, valid_fields / total_fields)

            return self._make_result(
                passed=False,
                score=score,
                details={
                    "validation_errors": len(errors),
                    "error_fields": sorted(error_fields),
                    "errors": [
                        {
                            "field": ".".join(str(loc) for loc in err["loc"]),
                            "type": err["type"],
                            "msg": err["msg"],
                        }
                        for err in errors[:20]  # cap at 20 to avoid huge outputs
                    ],
                },
            )
```

3. Verify tests pass (1 min):

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/evaluators/structural/test_d01_schema_compliance.py -v
# Expected: all tests pass
```

4. Commit (1 min):

```bash
git add persona_eval/evaluators/ tests/evaluators/
git commit -m "feat: D1 Schema Compliance evaluator with property-based fuzz testing"
```

---

### Task 6: D2 Completeness

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/structural/d02_completeness.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/structural/test_d02_completeness.py`

**Steps:**

1. Write failing tests (3 min):

```python
# tests/evaluators/structural/test_d02_completeness.py
"""Tests for D2 Completeness evaluator."""

import pytest


def test_evaluator_importable():
    from persona_eval.evaluators.structural.d02_completeness import CompletenessEvaluator
    assert CompletenessEvaluator is not None


def test_complete_persona_passes(sample_persona_dict):
    from persona_eval.evaluators.structural.d02_completeness import CompletenessEvaluator
    evaluator = CompletenessEvaluator()
    result = evaluator.evaluate(sample_persona_dict)
    assert result.passed is True
    assert result.score >= 0.9


def test_empty_strings_detected(sample_persona_dict):
    from persona_eval.evaluators.structural.d02_completeness import CompletenessEvaluator
    evaluator = CompletenessEvaluator()
    bad = dict(sample_persona_dict)
    bad["identity"] = {**sample_persona_dict["identity"], "name": ""}
    result = evaluator.evaluate(bad)
    assert result.passed is False
    assert "empty_fields" in result.details


def test_na_placeholder_detected(sample_persona_dict):
    from persona_eval.evaluators.structural.d02_completeness import CompletenessEvaluator
    evaluator = CompletenessEvaluator()
    bad = dict(sample_persona_dict)
    bad["identity"] = {**sample_persona_dict["identity"], "location": "N/A"}
    result = evaluator.evaluate(bad)
    assert result.score < 1.0
    assert any("N/A" in str(f) or "placeholder" in str(f).lower()
               for f in result.details.get("placeholder_fields", []))


def test_not_specified_detected(sample_persona_dict):
    from persona_eval.evaluators.structural.d02_completeness import CompletenessEvaluator
    evaluator = CompletenessEvaluator()
    bad = dict(sample_persona_dict)
    bad["source_context"] = "Not specified"
    result = evaluator.evaluate(bad)
    assert result.score < 1.0


def test_short_narrative_field_flagged(sample_persona_dict):
    from persona_eval.evaluators.structural.d02_completeness import CompletenessEvaluator
    evaluator = CompletenessEvaluator()
    bad = dict(sample_persona_dict)
    bad["source_context"] = "OK"  # too short for a source context
    result = evaluator.evaluate(bad)
    assert result.score < 1.0
    assert "short_fields" in result.details


def test_empty_list_detected(sample_persona_dict):
    from persona_eval.evaluators.structural.d02_completeness import CompletenessEvaluator
    evaluator = CompletenessEvaluator()
    bad = dict(sample_persona_dict)
    bad["values"] = []
    result = evaluator.evaluate(bad)
    assert result.passed is False


def test_unknown_placeholder_detected(sample_persona_dict):
    from persona_eval.evaluators.structural.d02_completeness import CompletenessEvaluator
    evaluator = CompletenessEvaluator()
    bad = dict(sample_persona_dict)
    bad["identity"] = {**sample_persona_dict["identity"], "gender": "unknown"}
    result = evaluator.evaluate(bad)
    assert result.score < 1.0
```

2. Verify tests fail, then implement (3 min):

```python
# persona_eval/evaluators/structural/d02_completeness.py
"""D2 Completeness — null/empty checks, min length, semantic emptiness.

Trustworthiness: HIGH (deterministic).
Method: Rule-based validation of field population.
"""

from __future__ import annotations

import re
from typing import Any

from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult

# Patterns that indicate semantic emptiness
PLACEHOLDER_PATTERNS = [
    re.compile(r"^n/?a$", re.IGNORECASE),
    re.compile(r"^not\s+(specified|available|applicable|provided|defined)$", re.IGNORECASE),
    re.compile(r"^unknown$", re.IGNORECASE),
    re.compile(r"^none$", re.IGNORECASE),
    re.compile(r"^tbd$", re.IGNORECASE),
    re.compile(r"^to\s+be\s+determined$", re.IGNORECASE),
    re.compile(r"^\[.*\]$"),  # [placeholder]
    re.compile(r"^<.*>$"),    # <placeholder>
    re.compile(r"^-+$"),      # ---
    re.compile(r"^\.{2,}$"),  # ...
]

# Fields that should have longer content (narrative fields)
NARRATIVE_FIELDS = {"source_context"}
NARRATIVE_MIN_LENGTH = 20


class CompletenessEvaluator(BaseEvaluator):
    """Evaluates whether all persona fields are meaningfully populated."""

    dimension_id = "D2"
    dimension_name = "Completeness"
    tier = 1

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        """Check all fields for null, empty, placeholder, and short content."""
        issues: dict[str, list] = {
            "empty_fields": [],
            "placeholder_fields": [],
            "short_fields": [],
            "empty_lists": [],
        }
        total_checks = 0
        passed_checks = 0

        self._check_dict(persona_data, "", issues, _total=[0], _passed=[0])
        total_checks = issues.pop("_total", [0])
        passed_checks = issues.pop("_passed", [0])

        # Re-count from issues
        all_fields = self._count_fields(persona_data)
        problem_count = (
            len(issues["empty_fields"])
            + len(issues["placeholder_fields"])
            + len(issues["short_fields"])
            + len(issues["empty_lists"])
        )

        score = max(0.0, (all_fields - problem_count) / all_fields) if all_fields > 0 else 0.0
        passed = problem_count == 0

        return self._make_result(
            passed=passed,
            score=round(score, 4),
            details={k: v for k, v in issues.items() if v},
        )

    def _count_fields(self, data: Any, depth: int = 0) -> int:
        """Count total leaf fields recursively."""
        if depth > 10:
            return 0
        if isinstance(data, dict):
            count = 0
            for v in data.values():
                count += self._count_fields(v, depth + 1)
            return max(count, 1)
        elif isinstance(data, list):
            if not data:
                return 1
            count = 0
            for item in data:
                count += self._count_fields(item, depth + 1)
            return max(count, 1)
        else:
            return 1

    def _check_dict(self, data: dict, prefix: str, issues: dict, _total: list, _passed: list):
        """Recursively check a dict for completeness issues."""
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            self._check_value(value, path, key, issues, _total, _passed)

    def _check_value(self, value: Any, path: str, key: str, issues: dict,
                     _total: list, _passed: list):
        """Check a single value for completeness."""
        _total[0] += 1

        if value is None:
            issues["empty_fields"].append(path)
            return

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                issues["empty_fields"].append(path)
                return
            if self._is_placeholder(stripped):
                issues["placeholder_fields"].append(f"{path}: {stripped!r}")
                return
            if key in NARRATIVE_FIELDS and len(stripped) < NARRATIVE_MIN_LENGTH:
                issues["short_fields"].append(f"{path}: len={len(stripped)}, min={NARRATIVE_MIN_LENGTH}")
                return
            _passed[0] += 1

        elif isinstance(value, list):
            if not value:
                issues["empty_lists"].append(path)
                return
            _passed[0] += 1
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    self._check_dict(item, f"{path}[{i}]", issues, _total, _passed)
                elif isinstance(item, str):
                    self._check_value(item, f"{path}[{i}]", key, issues, _total, _passed)

        elif isinstance(value, dict):
            _passed[0] += 1
            self._check_dict(value, path, issues, _total, _passed)

        else:
            _passed[0] += 1

    def _is_placeholder(self, text: str) -> bool:
        """Check if text matches any placeholder pattern."""
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern.match(text.strip()):
                return True
        return False
```

3. Verify tests pass (1 min):

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/evaluators/structural/test_d02_completeness.py -v
# Expected: all tests pass
```

4. Commit (1 min):

```bash
git add persona_eval/evaluators/structural/d02_completeness.py tests/evaluators/structural/test_d02_completeness.py
git commit -m "feat: D2 Completeness evaluator with semantic emptiness detection"
```

---

### Task 7: D3 Internal Logical Consistency

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/structural/d03_internal_consistency.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/structural/test_d03_internal_consistency.py`

**Steps:**

1. Write failing tests (3 min):

```python
# tests/evaluators/structural/test_d03_internal_consistency.py
"""Tests for D3 Internal Logical Consistency evaluator."""

import pytest


def test_evaluator_importable():
    from persona_eval.evaluators.structural.d03_internal_consistency import InternalConsistencyEvaluator
    assert InternalConsistencyEvaluator is not None


def test_consistent_persona_passes(sample_persona_dict):
    from persona_eval.evaluators.structural.d03_internal_consistency import InternalConsistencyEvaluator
    evaluator = InternalConsistencyEvaluator()
    result = evaluator.evaluate(sample_persona_dict)
    assert result.passed is True
    assert result.score >= 0.8


def test_entry_level_with_high_experience_fails(sample_persona_dict):
    from persona_eval.evaluators.structural.d03_internal_consistency import InternalConsistencyEvaluator
    evaluator = InternalConsistencyEvaluator()
    bad = dict(sample_persona_dict)
    bad["professional"] = {
        **sample_persona_dict["professional"],
        "role": "Junior Entry-Level Analyst",
        "years_experience": 25,
    }
    result = evaluator.evaluate(bad)
    assert result.passed is False
    assert any("experience" in str(v).lower() for v in result.details.get("violations", []))


def test_young_age_with_high_experience_fails(sample_persona_dict):
    from persona_eval.evaluators.structural.d03_internal_consistency import InternalConsistencyEvaluator
    evaluator = InternalConsistencyEvaluator()
    bad = dict(sample_persona_dict)
    bad["identity"] = {**sample_persona_dict["identity"], "age": 20}
    bad["professional"] = {**sample_persona_dict["professional"], "years_experience": 15}
    result = evaluator.evaluate(bad)
    assert result.passed is False


def test_team_size_zero_with_management_role_flags(sample_persona_dict):
    from persona_eval.evaluators.structural.d03_internal_consistency import InternalConsistencyEvaluator
    evaluator = InternalConsistencyEvaluator()
    bad = dict(sample_persona_dict)
    bad["professional"] = {
        **sample_persona_dict["professional"],
        "role": "VP of Engineering",
        "team_size": 0,
    }
    result = evaluator.evaluate(bad)
    assert result.score < 1.0


def test_no_crash_on_partial_data():
    from persona_eval.evaluators.structural.d03_internal_consistency import InternalConsistencyEvaluator
    evaluator = InternalConsistencyEvaluator()
    result = evaluator.evaluate({"id": "test"})
    # Should not crash, just skip rules that need missing fields
    assert result is not None
    assert 0.0 <= result.score <= 1.0
```

2. Verify tests fail, then implement (5 min):

```python
# persona_eval/evaluators/structural/d03_internal_consistency.py
"""D3 Internal Logical Consistency — rule-based constraint validation.

Trustworthiness: HIGH for rule-based, MEDIUM for NLI.
Method: Cross-field constraint rules + optional NLI entailment checking.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult


# --- Constraint rules ---
# Each rule is a function: (persona_dict) -> (passed: bool, message: str) or None to skip

def _safe_get(data: dict, *keys, default=None):
    """Safely navigate nested dict."""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def rule_experience_vs_age(data: dict) -> tuple[bool, str] | None:
    """Experience should not exceed age - 16 (minimum working age)."""
    age = _safe_get(data, "identity", "age")
    exp = _safe_get(data, "professional", "years_experience")
    if age is None or exp is None:
        return None
    max_exp = age - 16
    if exp > max_exp:
        return (False, f"years_experience ({exp}) exceeds plausible maximum for age {age} (max ~{max_exp})")
    return (True, "")


def rule_entry_level_experience(data: dict) -> tuple[bool, str] | None:
    """Entry-level roles should have < 5 years experience."""
    role = _safe_get(data, "professional", "role", default="")
    exp = _safe_get(data, "professional", "years_experience")
    if not role or exp is None:
        return None
    entry_patterns = [
        re.compile(r"\b(entry.?level|junior|intern|trainee|apprentice|graduate)\b", re.IGNORECASE),
    ]
    is_entry = any(p.search(role) for p in entry_patterns)
    if is_entry and exp > 5:
        return (False, f"Entry-level role '{role}' has {exp} years experience (expected < 5)")
    return (True, "")


def rule_senior_experience(data: dict) -> tuple[bool, str] | None:
    """Senior/VP/Director roles should have >= 5 years experience."""
    role = _safe_get(data, "professional", "role", default="")
    exp = _safe_get(data, "professional", "years_experience")
    if not role or exp is None:
        return None
    senior_patterns = [
        re.compile(r"\b(senior|sr\.?|lead|principal|director|vp|vice.?president|chief|head of|c[a-z]o)\b", re.IGNORECASE),
    ]
    is_senior = any(p.search(role) for p in senior_patterns)
    if is_senior and exp < 3:
        return (False, f"Senior role '{role}' has only {exp} years experience (expected >= 3)")
    return (True, "")


def rule_management_team_size(data: dict) -> tuple[bool, str] | None:
    """Management roles should have team_size > 0."""
    role = _safe_get(data, "professional", "role", default="")
    team_size = _safe_get(data, "professional", "team_size")
    if not role or team_size is None:
        return None
    mgmt_patterns = [
        re.compile(r"\b(manager|director|vp|vice.?president|head of|lead|chief)\b", re.IGNORECASE),
    ]
    is_mgmt = any(p.search(role) for p in mgmt_patterns)
    if is_mgmt and team_size == 0:
        return (False, f"Management role '{role}' has team_size=0")
    return (True, "")


def rule_household_size(data: dict) -> tuple[bool, str] | None:
    """Household size should be at least 1."""
    hs = _safe_get(data, "demographics", "household_size")
    if hs is None:
        return None
    if hs < 1:
        return (False, f"household_size is {hs}, must be >= 1")
    return (True, "")


# Master list of all constraint rules
CONSTRAINT_RULES: list[Callable[[dict], tuple[bool, str] | None]] = [
    rule_experience_vs_age,
    rule_entry_level_experience,
    rule_senior_experience,
    rule_management_team_size,
    rule_household_size,
]


class InternalConsistencyEvaluator(BaseEvaluator):
    """Evaluates internal logical consistency of a persona's fields."""

    dimension_id = "D3"
    dimension_name = "Internal Logical Consistency"
    tier = 1

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        """Run all constraint rules against the persona data."""
        violations = []
        rules_checked = 0

        for rule_fn in CONSTRAINT_RULES:
            try:
                result = rule_fn(persona_data)
                if result is None:
                    continue  # rule skipped due to missing data
                rules_checked += 1
                passed, message = result
                if not passed:
                    violations.append({"rule": rule_fn.__name__, "message": message})
            except Exception as e:
                violations.append({"rule": rule_fn.__name__, "message": f"Rule error: {e}"})
                rules_checked += 1

        if rules_checked == 0:
            return self._make_result(
                passed=True,
                score=1.0,
                details={"rules_checked": 0, "note": "No applicable rules (insufficient data)"},
            )

        score = max(0.0, (rules_checked - len(violations)) / rules_checked)
        passed = len(violations) == 0

        return self._make_result(
            passed=passed,
            score=round(score, 4),
            details={
                "rules_checked": rules_checked,
                "violations": violations,
            },
        )
```

3. Verify tests pass (1 min):

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/evaluators/structural/test_d03_internal_consistency.py -v
# Expected: all tests pass
```

4. Commit (1 min):

```bash
git add persona_eval/evaluators/structural/d03_internal_consistency.py tests/evaluators/structural/test_d03_internal_consistency.py
git commit -m "feat: D3 Internal Logical Consistency evaluator with rule-based constraints"
```

---

## Phase 3 — Tier 3: Distributional/Statistical (Tasks 8-13)

### Task 8: D13 Opinion Diversity

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/distributional/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/distributional/d13_opinion_diversity.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/distributional/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/distributional/test_d13_opinion_diversity.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/fixtures/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/fixtures/persona_set.py`

**Steps:**

1. Create test persona set fixture (3 min):

```python
# tests/fixtures/__init__.py
# empty

# tests/fixtures/persona_set.py
"""Fixture: a set of 50 test personas for distributional tests."""

import random

ROLES = [
    "Junior Developer", "Senior Product Manager", "Marketing Director",
    "Data Analyst", "UX Designer", "Sales Representative", "CTO",
    "Customer Support Lead", "HR Manager", "Financial Analyst",
    "DevOps Engineer", "Content Strategist", "Operations Manager",
    "Research Scientist", "Account Executive",
]

TECH_ADOPTIONS = ["innovator", "early_adopter", "early_majority", "late_majority", "laggard"]
RISK_TOLERANCES = ["very_low", "low", "moderate", "high", "very_high"]
EDUCATION_LEVELS = ["High school", "Associate's", "Bachelor's degree", "Master's degree", "PhD"]
INCOME_BRACKETS = ["low", "lower-middle", "middle", "upper-middle", "high"]
LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Austin, TX", "Seattle, WA",
    "Chicago, IL", "Denver, CO", "Miami, FL", "Portland, OR",
    "Atlanta, GA", "Boston, MA", "Rural Iowa", "Bangalore, India",
    "London, UK", "Berlin, Germany", "Tokyo, Japan",
]


def generate_test_persona_set(n: int = 50, seed: int = 42) -> list[dict]:
    """Generate a diverse set of n test personas for distributional testing."""
    rng = random.Random(seed)
    personas = []

    for i in range(n):
        age = rng.randint(22, 65)
        max_exp = age - 18
        exp = rng.randint(0, max(0, max_exp))
        role = rng.choice(ROLES)

        persona = {
            "id": f"test-persona-{i:03d}",
            "identity": {
                "name": f"Test Person {i}",
                "age": age,
                "gender": rng.choice(["male", "female", "non-binary"]),
                "location": rng.choice(LOCATIONS),
            },
            "demographics": {
                "education_level": rng.choice(EDUCATION_LEVELS),
                "income_bracket": rng.choice(INCOME_BRACKETS),
                "marital_status": rng.choice(["single", "married", "divorced", "widowed", "partnered"]),
                "household_size": rng.randint(1, 6),
                "ethnicity": rng.choice([
                    "White", "Black", "Hispanic/Latino", "Asian American",
                    "South Asian", "Mixed", "Middle Eastern",
                ]),
            },
            "professional": {
                "role": role,
                "industry": rng.choice(["SaaS", "Fintech", "Healthcare", "E-commerce", "Education", "Manufacturing"]),
                "company_size": rng.choice(["1-10", "11-50", "51-200", "201-500", "500-1000", "1000+"]),
                "years_experience": exp,
                "team_size": rng.randint(0, 20),
                "responsibilities": [f"Responsibility {j}" for j in range(rng.randint(1, 5))],
            },
            "behavioral": {
                "technology_adoption": rng.choice(TECH_ADOPTIONS),
                "decision_making_style": rng.choice([
                    "data-driven", "intuitive", "collaborative", "authority-based", "consensus-seeking",
                ]),
                "information_sources": [rng.choice(["blogs", "podcasts", "conferences", "peers", "reports"])],
                "purchase_triggers": ["Need"],
                "brand_loyalty": rng.choice(["low", "moderate", "high"]),
            },
            "psychographic": {
                "personality_traits": rng.sample(
                    ["analytical", "creative", "pragmatic", "ambitious", "cautious",
                     "empathetic", "competitive", "collaborative", "independent", "detail-oriented"],
                    k=3,
                ),
                "risk_tolerance": rng.choice(RISK_TOLERANCES),
                "work_life_balance_priority": rng.choice(["low", "medium", "high"]),
                "innovation_orientation": rng.choice(["conservative", "incremental", "radical"]),
            },
            "communication_style": {
                "tone": rng.choice(["formal", "casual", "professional but warm", "direct", "diplomatic"]),
                "vocabulary_level": rng.choice(["basic", "intermediate", "advanced", "technical"]),
                "preferred_channels": ["email"],
                "formality": rng.choice(["very_informal", "informal", "semi-formal", "formal", "very_formal"]),
                "verbosity": rng.choice(["terse", "concise", "moderate", "detailed", "verbose"]),
            },
            "goals": [{"description": "Goal", "timeframe": "6 months", "priority": "high"}],
            "pain_points": [{"description": "Pain", "severity": "medium", "frequency": "weekly"}],
            "values": [rng.choice(["innovation", "stability", "growth", "efficiency", "quality", "speed"])],
            "knowledge_domains": [{"domain": "Their field", "depth": rng.choice(["basic", "intermediate", "advanced", "expert"])}],
            "emotional_profile": {
                "baseline_mood": rng.choice(["optimistic", "neutral", "anxious", "driven", "calm"]),
                "stress_response": "varies",
                "conflict_style": rng.choice(["avoidant", "collaborative", "competitive", "accommodating"]),
                "enthusiasm_triggers": ["wins"],
                "frustration_triggers": ["delays"],
            },
            "source_context": f"Generated test persona {i} for distributional evaluation testing.",
        }
        personas.append(persona)

    return personas


def generate_homogeneous_set(n: int = 50) -> list[dict]:
    """Generate a homogeneous persona set where everyone is the same — should FAIL diversity tests."""
    return [
        {
            "id": f"clone-{i:03d}",
            "identity": {"name": f"Clone {i}", "age": 35, "gender": "male", "location": "San Francisco, CA"},
            "demographics": {
                "education_level": "Master's degree", "income_bracket": "upper-middle",
                "marital_status": "married", "household_size": 3, "ethnicity": "White",
            },
            "professional": {
                "role": "Senior Product Manager", "industry": "SaaS", "company_size": "500-1000",
                "years_experience": 10, "team_size": 8, "responsibilities": ["Product roadmap"],
            },
            "behavioral": {
                "technology_adoption": "early_majority",
                "decision_making_style": "data-driven",
                "information_sources": ["blogs"], "purchase_triggers": ["Need"], "brand_loyalty": "moderate",
            },
            "psychographic": {
                "personality_traits": ["analytical", "collaborative", "pragmatic"],
                "risk_tolerance": "moderate", "work_life_balance_priority": "high",
                "innovation_orientation": "incremental",
            },
            "communication_style": {
                "tone": "professional but warm", "vocabulary_level": "advanced",
                "preferred_channels": ["email"], "formality": "semi-formal", "verbosity": "concise",
            },
            "goals": [{"description": "Ship product", "timeframe": "6 months", "priority": "high"}],
            "pain_points": [{"description": "Too many tools", "severity": "high", "frequency": "daily"}],
            "values": ["innovation"],
            "knowledge_domains": [{"domain": "Product management", "depth": "expert"}],
            "emotional_profile": {
                "baseline_mood": "optimistic", "stress_response": "structured",
                "conflict_style": "collaborative", "enthusiasm_triggers": ["wins"],
                "frustration_triggers": ["delays"],
            },
            "source_context": "Test clone persona.",
        }
        for i in range(n)
    ]
```

2. Write failing tests (3 min):

```python
# tests/evaluators/distributional/__init__.py
# empty

# tests/evaluators/distributional/test_d13_opinion_diversity.py
"""Tests for D13 Opinion Diversity evaluator."""

import pytest
from tests.fixtures.persona_set import generate_test_persona_set, generate_homogeneous_set


@pytest.fixture
def diverse_persona_set():
    return generate_test_persona_set(n=50, seed=42)


@pytest.fixture
def homogeneous_persona_set():
    return generate_homogeneous_set(n=50)


def test_evaluator_importable():
    from persona_eval.evaluators.distributional.d13_opinion_diversity import OpinionDiversityEvaluator
    assert OpinionDiversityEvaluator is not None


def test_diverse_set_passes(diverse_persona_set):
    from persona_eval.evaluators.distributional.d13_opinion_diversity import OpinionDiversityEvaluator
    evaluator = OpinionDiversityEvaluator()
    result = evaluator.evaluate({}, persona_set=diverse_persona_set)
    assert result.passed is True
    assert result.score >= 0.7
    assert "entropy" in result.details or "variation_ratios" in result.details


def test_homogeneous_set_fails(homogeneous_persona_set):
    from persona_eval.evaluators.distributional.d13_opinion_diversity import OpinionDiversityEvaluator
    evaluator = OpinionDiversityEvaluator()
    result = evaluator.evaluate({}, persona_set=homogeneous_persona_set)
    assert result.passed is False
    assert result.score < 0.5


def test_single_persona_skipped():
    from persona_eval.evaluators.distributional.d13_opinion_diversity import OpinionDiversityEvaluator
    evaluator = OpinionDiversityEvaluator()
    result = evaluator.evaluate({}, persona_set=[{"id": "solo"}])
    assert result.details.get("skipped") is True


def test_modal_collapse_detected(homogeneous_persona_set):
    from persona_eval.evaluators.distributional.d13_opinion_diversity import OpinionDiversityEvaluator
    evaluator = OpinionDiversityEvaluator()
    result = evaluator.evaluate({}, persona_set=homogeneous_persona_set)
    assert "modal_collapse_fields" in result.details
    assert len(result.details["modal_collapse_fields"]) > 0
```

3. Verify tests fail, then implement (5 min):

```python
# persona_eval/evaluators/distributional/__init__.py
"""Tier 3 — Distributional & statistical validators (D13-D18)."""

# persona_eval/evaluators/distributional/d13_opinion_diversity.py
"""D13 Opinion Diversity — variation ratio, entropy, modal collapse detection.

Trustworthiness: HIGH (mathematically grounded, directly testable).
Method: Compute variation ratio and Shannon entropy across categorical fields.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult

# Categorical fields to measure diversity across
DIVERSITY_FIELDS = [
    ("behavioral", "technology_adoption"),
    ("behavioral", "decision_making_style"),
    ("behavioral", "brand_loyalty"),
    ("psychographic", "risk_tolerance"),
    ("psychographic", "work_life_balance_priority"),
    ("psychographic", "innovation_orientation"),
    ("communication_style", "formality"),
    ("communication_style", "verbosity"),
    ("communication_style", "tone"),
    ("emotional_profile", "baseline_mood"),
    ("emotional_profile", "conflict_style"),
    ("demographics", "education_level"),
    ("demographics", "income_bracket"),
    ("demographics", "marital_status"),
    ("identity", "gender"),
]

MODAL_COLLAPSE_THRESHOLD = 0.80  # >80% same answer = modal collapse


def _extract_field(persona: dict, *keys) -> str | None:
    """Extract a nested field value."""
    current = persona
    for k in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(k)
    return str(current) if current is not None else None


def _variation_ratio(values: list[str]) -> float:
    """Fraction of values that are NOT the mode. 0 = all same, 1 = perfectly uniform."""
    if not values:
        return 0.0
    counter = Counter(values)
    mode_count = counter.most_common(1)[0][1]
    return 1.0 - (mode_count / len(values))


def _shannon_entropy(values: list[str]) -> float:
    """Normalized Shannon entropy. 0 = no diversity, 1 = maximum diversity."""
    if not values:
        return 0.0
    counter = Counter(values)
    n = len(values)
    num_categories = len(counter)
    if num_categories <= 1:
        return 0.0
    max_entropy = math.log2(num_categories)
    if max_entropy == 0:
        return 0.0
    entropy = -sum((count / n) * math.log2(count / n) for count in counter.values())
    return entropy / max_entropy


class OpinionDiversityEvaluator(BaseEvaluator):
    """Evaluates opinion/attribute diversity across a persona set."""

    dimension_id = "D13"
    dimension_name = "Opinion Diversity"
    tier = 3

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        """Compute diversity metrics across a persona set.

        Requires kwargs['persona_set']: list of persona dicts.
        """
        persona_set = kwargs.get("persona_set", [])
        if len(persona_set) < 2:
            return self._make_result(
                passed=True, score=1.0,
                details={"skipped": True, "reason": "Need >= 2 personas for distributional tests"},
            )

        variation_ratios = {}
        entropies = {}
        modal_collapse_fields = []

        for field_path in DIVERSITY_FIELDS:
            values = []
            for p in persona_set:
                val = _extract_field(p, *field_path)
                if val is not None:
                    values.append(val)

            if len(values) < 2:
                continue

            field_name = ".".join(field_path)
            vr = _variation_ratio(values)
            ent = _shannon_entropy(values)

            variation_ratios[field_name] = round(vr, 4)
            entropies[field_name] = round(ent, 4)

            # Check for modal collapse
            counter = Counter(values)
            mode_count = counter.most_common(1)[0][1]
            if mode_count / len(values) > MODAL_COLLAPSE_THRESHOLD:
                modal_collapse_fields.append({
                    "field": field_name,
                    "modal_value": counter.most_common(1)[0][0],
                    "modal_fraction": round(mode_count / len(values), 4),
                })

        if not variation_ratios:
            return self._make_result(passed=True, score=1.0, details={"skipped": True, "reason": "No comparable fields"})

        # Overall score: mean of normalized entropies
        mean_entropy = sum(entropies.values()) / len(entropies)
        mean_vr = sum(variation_ratios.values()) / len(variation_ratios)
        score = (mean_entropy + mean_vr) / 2  # blend both metrics

        # Pass if no modal collapse and reasonable diversity
        passed = len(modal_collapse_fields) == 0 and score >= 0.3

        return self._make_result(
            passed=passed,
            score=round(score, 4),
            details={
                "variation_ratios": variation_ratios,
                "entropies": entropies,
                "mean_entropy": round(mean_entropy, 4),
                "mean_variation_ratio": round(mean_vr, 4),
                "modal_collapse_fields": modal_collapse_fields,
                "persona_count": len(persona_set),
            },
        )
```

4. Verify tests pass (1 min):

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/evaluators/distributional/test_d13_opinion_diversity.py -v
# Expected: all tests pass
```

5. Commit (1 min):

```bash
git add persona_eval/evaluators/distributional/ tests/evaluators/distributional/ tests/fixtures/
git commit -m "feat: D13 Opinion Diversity evaluator with variation ratio, entropy, and modal collapse detection"
```

---

### Task 9: D14 Variance Fidelity

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/distributional/d14_variance_fidelity.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/distributional/test_d14_variance_fidelity.py`

**Steps:**

1. Write failing tests (3 min):

```python
# tests/evaluators/distributional/test_d14_variance_fidelity.py
"""Tests for D14 Variance Fidelity evaluator."""

import pytest
from tests.fixtures.persona_set import generate_test_persona_set, generate_homogeneous_set


@pytest.fixture
def diverse_set():
    return generate_test_persona_set(n=50, seed=42)


@pytest.fixture
def homogeneous_set():
    return generate_homogeneous_set(n=50)


def test_evaluator_importable():
    from persona_eval.evaluators.distributional.d14_variance_fidelity import VarianceFidelityEvaluator
    assert VarianceFidelityEvaluator is not None


def test_diverse_set_has_variance(diverse_set):
    from persona_eval.evaluators.distributional.d14_variance_fidelity import VarianceFidelityEvaluator
    evaluator = VarianceFidelityEvaluator()
    result = evaluator.evaluate({}, persona_set=diverse_set)
    assert result.score >= 0.5
    assert "iqr_scores" in result.details


def test_homogeneous_set_zero_variance(homogeneous_set):
    from persona_eval.evaluators.distributional.d14_variance_fidelity import VarianceFidelityEvaluator
    evaluator = VarianceFidelityEvaluator()
    result = evaluator.evaluate({}, persona_set=homogeneous_set)
    assert result.passed is False
    assert result.score < 0.3
    assert "zero_variance_fields" in result.details


def test_ks_test_present(diverse_set):
    from persona_eval.evaluators.distributional.d14_variance_fidelity import VarianceFidelityEvaluator
    evaluator = VarianceFidelityEvaluator()
    # With reference distribution
    ref = {"identity.age": {"mean": 40, "std": 12}}
    result = evaluator.evaluate({}, persona_set=diverse_set, reference_distributions=ref)
    assert "ks_tests" in result.details or result.details.get("skipped")
```

2. Verify tests fail, then implement (4 min):

```python
# persona_eval/evaluators/distributional/d14_variance_fidelity.py
"""D14 Variance Fidelity — IQR comparison, K-S test.

Trustworthiness: HIGH (purely statistical, requires reference human data).
Method: Compare IQR of persona numeric fields to reference; K-S test for shape.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult

# Numeric fields to measure variance on
NUMERIC_FIELDS = [
    (("identity", "age"), "identity.age"),
    (("professional", "years_experience"), "professional.years_experience"),
    (("professional", "team_size"), "professional.team_size"),
    (("demographics", "household_size"), "demographics.household_size"),
]


def _extract_numeric(persona: dict, keys: tuple) -> float | None:
    current = persona
    for k in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(k)
    try:
        return float(current) if current is not None else None
    except (ValueError, TypeError):
        return None


class VarianceFidelityEvaluator(BaseEvaluator):
    """Evaluates whether the persona set has realistic variance."""

    dimension_id = "D14"
    dimension_name = "Variance Fidelity"
    tier = 3

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        persona_set = kwargs.get("persona_set", [])
        reference_distributions = kwargs.get("reference_distributions", {})

        if len(persona_set) < 5:
            return self._make_result(passed=True, score=1.0, details={"skipped": True, "reason": "Need >= 5 personas"})

        iqr_scores = {}
        zero_variance_fields = []
        ks_tests = {}

        for keys, field_name in NUMERIC_FIELDS:
            values = [v for p in persona_set if (v := _extract_numeric(p, keys)) is not None]
            if len(values) < 5:
                continue

            arr = np.array(values)
            q25, q75 = np.percentile(arr, [25, 75])
            iqr = q75 - q25
            std = np.std(arr)

            if iqr == 0 and std == 0:
                zero_variance_fields.append(field_name)
                iqr_scores[field_name] = 0.0
            else:
                # Score based on coefficient of variation (normalized spread)
                mean = np.mean(arr)
                cv = std / mean if mean != 0 else 0.0
                # A CV of 0 = no variance, CV > 0.3 = healthy variance
                iqr_scores[field_name] = round(min(1.0, cv / 0.3), 4)

            # K-S test against reference if available
            if field_name in reference_distributions:
                ref = reference_distributions[field_name]
                ref_mean = ref.get("mean", np.mean(arr))
                ref_std = ref.get("std", np.std(arr))
                if ref_std > 0:
                    ks_stat, ks_p = stats.kstest(arr, "norm", args=(ref_mean, ref_std))
                    ks_tests[field_name] = {"statistic": round(ks_stat, 4), "p_value": round(ks_p, 4)}

        if not iqr_scores:
            return self._make_result(passed=True, score=1.0, details={"skipped": True, "reason": "No numeric fields"})

        mean_score = sum(iqr_scores.values()) / len(iqr_scores)
        passed = len(zero_variance_fields) == 0 and mean_score >= 0.3

        return self._make_result(
            passed=passed,
            score=round(mean_score, 4),
            details={
                "iqr_scores": iqr_scores,
                "zero_variance_fields": zero_variance_fields,
                "ks_tests": ks_tests if ks_tests else None,
                "persona_count": len(persona_set),
            },
        )
```

3. Verify tests pass (1 min):

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/evaluators/distributional/test_d14_variance_fidelity.py -v
# Expected: all tests pass
```

4. Commit (1 min):

```bash
git add persona_eval/evaluators/distributional/d14_variance_fidelity.py tests/evaluators/distributional/test_d14_variance_fidelity.py
git commit -m "feat: D14 Variance Fidelity evaluator with IQR and K-S testing"
```

---

### Task 10: D15 Structural Aggregation Consistency

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/distributional/d15_aggregation_consistency.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/distributional/test_d15_aggregation_consistency.py`

**Steps:**

1. Write failing tests (2 min):

```python
# tests/evaluators/distributional/test_d15_aggregation_consistency.py
"""Tests for D15 Structural Aggregation Consistency."""

import pytest
from tests.fixtures.persona_set import generate_test_persona_set


@pytest.fixture
def persona_set():
    return generate_test_persona_set(n=50, seed=42)


def test_evaluator_importable():
    from persona_eval.evaluators.distributional.d15_aggregation_consistency import AggregationConsistencyEvaluator
    assert AggregationConsistencyEvaluator is not None


def test_consistent_aggregation(persona_set):
    from persona_eval.evaluators.distributional.d15_aggregation_consistency import AggregationConsistencyEvaluator
    evaluator = AggregationConsistencyEvaluator()
    result = evaluator.evaluate({}, persona_set=persona_set)
    assert result.score >= 0.0
    assert "consistency_ratio" in result.details


def test_group_vs_individual_comparison(persona_set):
    from persona_eval.evaluators.distributional.d15_aggregation_consistency import AggregationConsistencyEvaluator
    evaluator = AggregationConsistencyEvaluator()
    result = evaluator.evaluate({}, persona_set=persona_set)
    assert "group_comparisons" in result.details
```

2. Verify tests fail, then implement (4 min):

```python
# persona_eval/evaluators/distributional/d15_aggregation_consistency.py
"""D15 Structural Aggregation Consistency — cross-aggregation test.

Trustworthiness: HIGH (deterministic comparison).
Method: Query the group, query individuals, aggregate individuals, compare.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult

GROUPING_FIELDS = [
    (("identity", "gender"), "gender"),
    (("demographics", "education_level"), "education"),
    (("demographics", "income_bracket"), "income"),
]

AGGREGATION_FIELDS = [
    (("behavioral", "technology_adoption"), "tech_adoption"),
    (("psychographic", "risk_tolerance"), "risk_tolerance"),
    (("behavioral", "brand_loyalty"), "brand_loyalty"),
]


def _extract(persona: dict, keys: tuple) -> str | None:
    current = persona
    for k in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(k)
    return str(current) if current is not None else None


class AggregationConsistencyEvaluator(BaseEvaluator):
    """Check that group-level queries match individual-level aggregation."""

    dimension_id = "D15"
    dimension_name = "Structural Aggregation Consistency"
    tier = 3

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        persona_set = kwargs.get("persona_set", [])
        if len(persona_set) < 10:
            return self._make_result(passed=True, score=1.0, details={"skipped": True})

        group_comparisons = []
        consistencies = []

        for group_keys, group_name in GROUPING_FIELDS:
            # Group personas
            groups: dict[str, list[dict]] = {}
            for p in persona_set:
                val = _extract(p, group_keys)
                if val:
                    groups.setdefault(val, []).append(p)

            for agg_keys, agg_name in AGGREGATION_FIELDS:
                # Get overall distribution
                all_values = [_extract(p, agg_keys) for p in persona_set]
                all_values = [v for v in all_values if v]
                overall_dist = Counter(all_values)

                # Get per-group distributions and aggregate
                aggregated_dist = Counter()
                for group_val, group_personas in groups.items():
                    for p in group_personas:
                        val = _extract(p, agg_keys)
                        if val:
                            aggregated_dist[val] += 1

                # Compare: are the distributions the same?
                if not overall_dist or not aggregated_dist:
                    continue

                # Normalize both
                total_overall = sum(overall_dist.values())
                total_agg = sum(aggregated_dist.values())

                all_keys = set(overall_dist.keys()) | set(aggregated_dist.keys())
                diffs = []
                for k in all_keys:
                    p1 = overall_dist.get(k, 0) / total_overall
                    p2 = aggregated_dist.get(k, 0) / total_agg
                    diffs.append(abs(p1 - p2))

                consistency = 1.0 - (sum(diffs) / (2 * len(all_keys))) if all_keys else 1.0
                consistencies.append(consistency)

                group_comparisons.append({
                    "group_by": group_name,
                    "field": agg_name,
                    "consistency": round(consistency, 4),
                })

        if not consistencies:
            return self._make_result(passed=True, score=1.0, details={"skipped": True})

        mean_consistency = sum(consistencies) / len(consistencies)
        passed = mean_consistency >= 0.7

        return self._make_result(
            passed=passed,
            score=round(mean_consistency, 4),
            details={
                "consistency_ratio": round(mean_consistency, 4),
                "group_comparisons": group_comparisons,
            },
        )
```

3. Verify and commit (2 min):

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/evaluators/distributional/test_d15_aggregation_consistency.py -v
# Expected: all tests pass
```

```bash
git add persona_eval/evaluators/distributional/d15_aggregation_consistency.py tests/evaluators/distributional/test_d15_aggregation_consistency.py
git commit -m "feat: D15 Structural Aggregation Consistency evaluator"
```

---

### Task 11: D16 Minority Viewpoint Preservation

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/distributional/d16_minority_viewpoint.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/distributional/test_d16_minority_viewpoint.py`

**Steps:**

1. Write failing tests (2 min):

```python
# tests/evaluators/distributional/test_d16_minority_viewpoint.py
"""Tests for D16 Minority Viewpoint Preservation."""

import pytest
from tests.fixtures.persona_set import generate_test_persona_set, generate_homogeneous_set


@pytest.fixture
def diverse_set():
    return generate_test_persona_set(n=50, seed=42)


@pytest.fixture
def homogeneous_set():
    return generate_homogeneous_set(n=50)


def test_evaluator_importable():
    from persona_eval.evaluators.distributional.d16_minority_viewpoint import MinorityViewpointEvaluator
    assert MinorityViewpointEvaluator is not None


def test_diverse_set_has_minorities(diverse_set):
    from persona_eval.evaluators.distributional.d16_minority_viewpoint import MinorityViewpointEvaluator
    evaluator = MinorityViewpointEvaluator()
    result = evaluator.evaluate({}, persona_set=diverse_set)
    assert result.score >= 0.5
    assert "within_group_entropy" in result.details


def test_homogeneous_set_lacks_minorities(homogeneous_set):
    from persona_eval.evaluators.distributional.d16_minority_viewpoint import MinorityViewpointEvaluator
    evaluator = MinorityViewpointEvaluator()
    result = evaluator.evaluate({}, persona_set=homogeneous_set)
    assert result.score < 0.5
```

2. Verify tests fail, then implement (3 min):

```python
# persona_eval/evaluators/distributional/d16_minority_viewpoint.py
"""D16 Minority Viewpoint Preservation — within-group entropy.

Trustworthiness: HIGH (requires reference data, measurement is straightforward).
Method: For each demographic subgroup, measure opinion diversity within that group.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult

GROUPING_FIELD = (("identity", "gender"), "gender")
OPINION_FIELDS = [
    (("behavioral", "technology_adoption"), "tech_adoption"),
    (("psychographic", "risk_tolerance"), "risk_tolerance"),
    (("psychographic", "innovation_orientation"), "innovation"),
    (("behavioral", "brand_loyalty"), "brand_loyalty"),
]


def _extract(persona: dict, keys: tuple) -> str | None:
    current = persona
    for k in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(k)
    return str(current) if current is not None else None


def _within_group_entropy(values: list[str]) -> float:
    if len(values) < 2:
        return 0.0
    counter = Counter(values)
    n = len(values)
    num_cats = len(counter)
    if num_cats <= 1:
        return 0.0
    max_ent = math.log2(num_cats)
    if max_ent == 0:
        return 0.0
    ent = -sum((c / n) * math.log2(c / n) for c in counter.values())
    return ent / max_ent


class MinorityViewpointEvaluator(BaseEvaluator):
    dimension_id = "D16"
    dimension_name = "Minority Viewpoint Preservation"
    tier = 3

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        persona_set = kwargs.get("persona_set", [])
        if len(persona_set) < 5:
            return self._make_result(passed=True, score=1.0, details={"skipped": True})

        # Group by gender (could extend to other grouping fields)
        groups: dict[str, list[dict]] = {}
        gkeys, gname = GROUPING_FIELD
        for p in persona_set:
            val = _extract(p, gkeys)
            if val:
                groups.setdefault(val, []).append(p)

        group_entropies = {}
        for group_val, group_personas in groups.items():
            if len(group_personas) < 3:
                continue
            field_entropies = {}
            for okeys, oname in OPINION_FIELDS:
                vals = [_extract(p, okeys) for p in group_personas]
                vals = [v for v in vals if v]
                if len(vals) >= 2:
                    field_entropies[oname] = round(_within_group_entropy(vals), 4)
            if field_entropies:
                group_entropies[group_val] = field_entropies

        if not group_entropies:
            return self._make_result(passed=True, score=1.0, details={"skipped": True})

        # Average entropy across all groups and fields
        all_ents = []
        for g, fields in group_entropies.items():
            all_ents.extend(fields.values())

        mean_ent = sum(all_ents) / len(all_ents) if all_ents else 0.0
        passed = mean_ent >= 0.3

        return self._make_result(
            passed=passed,
            score=round(mean_ent, 4),
            details={"within_group_entropy": group_entropies, "mean_entropy": round(mean_ent, 4)},
        )
```

3. Verify and commit (2 min):

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/evaluators/distributional/test_d16_minority_viewpoint.py -v
git add persona_eval/evaluators/distributional/d16_minority_viewpoint.py tests/evaluators/distributional/test_d16_minority_viewpoint.py
git commit -m "feat: D16 Minority Viewpoint Preservation evaluator with within-group entropy"
```

---

### Task 12: D17 Calibration

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/distributional/d17_calibration.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/distributional/test_d17_calibration.py`

**Steps:**

1. Write failing tests (2 min):

```python
# tests/evaluators/distributional/test_d17_calibration.py
"""Tests for D17 Calibration evaluator."""

import pytest


def test_evaluator_importable():
    from persona_eval.evaluators.distributional.d17_calibration import CalibrationEvaluator
    assert CalibrationEvaluator is not None


def test_perfect_calibration():
    from persona_eval.evaluators.distributional.d17_calibration import CalibrationEvaluator
    evaluator = CalibrationEvaluator()
    # Perfect calibration: confidence matches accuracy
    predictions = [
        {"confidence": 0.9, "correct": True},
        {"confidence": 0.9, "correct": True},
        {"confidence": 0.9, "correct": True},
        {"confidence": 0.9, "correct": False},
        {"confidence": 0.5, "correct": True},
        {"confidence": 0.5, "correct": False},
        {"confidence": 0.1, "correct": False},
        {"confidence": 0.1, "correct": False},
    ]
    result = evaluator.evaluate({}, predictions=predictions)
    assert result.score >= 0.5
    assert "ece" in result.details


def test_terrible_calibration():
    from persona_eval.evaluators.distributional.d17_calibration import CalibrationEvaluator
    evaluator = CalibrationEvaluator()
    # Terrible: high confidence, always wrong
    predictions = [
        {"confidence": 0.99, "correct": False},
        {"confidence": 0.95, "correct": False},
        {"confidence": 0.90, "correct": False},
        {"confidence": 0.85, "correct": False},
    ]
    result = evaluator.evaluate({}, predictions=predictions)
    assert result.passed is False
    assert result.score < 0.3


def test_empty_predictions():
    from persona_eval.evaluators.distributional.d17_calibration import CalibrationEvaluator
    evaluator = CalibrationEvaluator()
    result = evaluator.evaluate({}, predictions=[])
    assert result.details.get("skipped") is True
```

2. Verify tests fail, then implement (3 min):

```python
# persona_eval/evaluators/distributional/d17_calibration.py
"""D17 Calibration — Expected Calibration Error (ECE).

Trustworthiness: HIGH (well-established statistical methodology).
Method: Bin responses by confidence, measure accuracy per bin, compute ECE.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult


def compute_ece(confidences: list[float], accuracies: list[bool], n_bins: int = 10) -> tuple[float, list[dict]]:
    """Compute Expected Calibration Error.

    Returns (ece_value, bin_details).
    """
    if not confidences:
        return 0.0, []

    conf_arr = np.array(confidences)
    acc_arr = np.array(accuracies, dtype=float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_details = []

    ece = 0.0
    total = len(conf_arr)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (conf_arr > lo) & (conf_arr <= hi) if i > 0 else (conf_arr >= lo) & (conf_arr <= hi)
        bin_size = mask.sum()
        if bin_size == 0:
            continue
        avg_conf = conf_arr[mask].mean()
        avg_acc = acc_arr[mask].mean()
        gap = abs(avg_acc - avg_conf)
        ece += (bin_size / total) * gap
        bin_details.append({
            "bin": f"({lo:.1f}, {hi:.1f}]",
            "count": int(bin_size),
            "avg_confidence": round(float(avg_conf), 4),
            "avg_accuracy": round(float(avg_acc), 4),
            "gap": round(float(gap), 4),
        })

    return float(ece), bin_details


class CalibrationEvaluator(BaseEvaluator):
    dimension_id = "D17"
    dimension_name = "Calibration"
    tier = 3

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        """Evaluate calibration from a set of confidence-labeled predictions.

        Expects kwargs['predictions']: list of {"confidence": float, "correct": bool}
        """
        predictions = kwargs.get("predictions", [])
        if len(predictions) < 4:
            return self._make_result(passed=True, score=1.0, details={"skipped": True, "reason": "Need >= 4 predictions"})

        confidences = [p["confidence"] for p in predictions]
        accuracies = [p["correct"] for p in predictions]

        ece, bin_details = compute_ece(confidences, accuracies)
        # ECE of 0 = perfect calibration, ECE of 1 = worst
        score = max(0.0, 1.0 - ece)
        passed = ece < 0.15  # threshold from Economic Choice Labs research

        return self._make_result(
            passed=passed,
            score=round(score, 4),
            details={
                "ece": round(ece, 4),
                "bins": bin_details,
                "n_predictions": len(predictions),
            },
        )
```

3. Verify and commit (2 min):

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/evaluators/distributional/test_d17_calibration.py -v
git add persona_eval/evaluators/distributional/d17_calibration.py tests/evaluators/distributional/test_d17_calibration.py
git commit -m "feat: D17 Calibration evaluator with Expected Calibration Error"
```

---

### Task 13: D18 Joint Distribution Fidelity

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/distributional/d18_joint_distribution.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/distributional/test_d18_joint_distribution.py`

**Steps:**

1. Write failing tests (2 min):

```python
# tests/evaluators/distributional/test_d18_joint_distribution.py
"""Tests for D18 Joint Distribution Fidelity."""

import pytest
from tests.fixtures.persona_set import generate_test_persona_set, generate_homogeneous_set


@pytest.fixture
def diverse_set():
    return generate_test_persona_set(n=50, seed=42)


def test_evaluator_importable():
    from persona_eval.evaluators.distributional.d18_joint_distribution import JointDistributionEvaluator
    assert JointDistributionEvaluator is not None


def test_computes_correlations(diverse_set):
    from persona_eval.evaluators.distributional.d18_joint_distribution import JointDistributionEvaluator
    evaluator = JointDistributionEvaluator()
    result = evaluator.evaluate({}, persona_set=diverse_set)
    assert "correlation_matrix" in result.details or result.details.get("skipped")
    assert result.score >= 0.0


def test_detects_stereotypical_correlation():
    from persona_eval.evaluators.distributional.d18_joint_distribution import JointDistributionEvaluator
    evaluator = JointDistributionEvaluator()
    # Create a set with forced stereotypical correlation: all females are "collaborative"
    stereotyped = []
    for i in range(25):
        stereotyped.append({
            "id": f"s-{i}", "identity": {"name": f"P{i}", "age": 30, "gender": "female", "location": "NY"},
            "psychographic": {"personality_traits": ["collaborative", "empathetic", "cautious"],
                              "risk_tolerance": "low", "work_life_balance_priority": "high", "innovation_orientation": "incremental"},
            "demographics": {"education_level": "Master's", "income_bracket": "middle", "marital_status": "single", "household_size": 1, "ethnicity": "White"},
            "professional": {"role": "PM", "industry": "SaaS", "company_size": "100", "years_experience": 5, "team_size": 5, "responsibilities": ["r"]},
            "behavioral": {"technology_adoption": "early_majority", "decision_making_style": "collaborative", "information_sources": ["blog"], "purchase_triggers": ["n"], "brand_loyalty": "high"},
            "communication_style": {"tone": "warm", "vocabulary_level": "advanced", "preferred_channels": ["email"], "formality": "semi-formal", "verbosity": "detailed"},
            "goals": [{"description": "g", "timeframe": "6m", "priority": "high"}],
            "pain_points": [{"description": "p", "severity": "medium", "frequency": "weekly"}],
            "values": ["empathy"], "knowledge_domains": [{"domain": "PM", "depth": "advanced"}],
            "emotional_profile": {"baseline_mood": "warm", "stress_response": "calm", "conflict_style": "accommodating", "enthusiasm_triggers": ["t"], "frustration_triggers": ["f"]},
            "source_context": "test",
        })
    for i in range(25):
        stereotyped.append({
            "id": f"s-{i+25}", "identity": {"name": f"P{i+25}", "age": 30, "gender": "male", "location": "NY"},
            "psychographic": {"personality_traits": ["competitive", "analytical", "ambitious"],
                              "risk_tolerance": "high", "work_life_balance_priority": "low", "innovation_orientation": "radical"},
            "demographics": {"education_level": "PhD", "income_bracket": "high", "marital_status": "single", "household_size": 1, "ethnicity": "White"},
            "professional": {"role": "CTO", "industry": "SaaS", "company_size": "100", "years_experience": 15, "team_size": 20, "responsibilities": ["r"]},
            "behavioral": {"technology_adoption": "innovator", "decision_making_style": "data-driven", "information_sources": ["blog"], "purchase_triggers": ["n"], "brand_loyalty": "low"},
            "communication_style": {"tone": "direct", "vocabulary_level": "technical", "preferred_channels": ["slack"], "formality": "informal", "verbosity": "terse"},
            "goals": [{"description": "g", "timeframe": "6m", "priority": "high"}],
            "pain_points": [{"description": "p", "severity": "high", "frequency": "daily"}],
            "values": ["efficiency"], "knowledge_domains": [{"domain": "Eng", "depth": "expert"}],
            "emotional_profile": {"baseline_mood": "driven", "stress_response": "intense", "conflict_style": "competitive", "enthusiasm_triggers": ["t"], "frustration_triggers": ["f"]},
            "source_context": "test",
        })
    result = evaluator.evaluate({}, persona_set=stereotyped)
    assert "stereotypical_pairs" in result.details
```

2. Verify tests fail, then implement (4 min):

```python
# persona_eval/evaluators/distributional/d18_joint_distribution.py
"""D18 Joint Distribution Fidelity — correlation matrix comparison.

Trustworthiness: HIGH (mathematical, but requires good reference data).
Method: Compute attribute correlations, detect stereotypical over-correlation.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np

from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult

CATEGORICAL_FIELDS = [
    (("identity", "gender"), "gender"),
    (("behavioral", "technology_adoption"), "tech_adoption"),
    (("psychographic", "risk_tolerance"), "risk_tolerance"),
    (("behavioral", "brand_loyalty"), "brand_loyalty"),
    (("communication_style", "verbosity"), "verbosity"),
    (("emotional_profile", "conflict_style"), "conflict_style"),
    (("demographics", "income_bracket"), "income"),
]

STEREOTYPE_THRESHOLD = 0.7  # Cramer's V > 0.7 = suspiciously high correlation


def _extract(persona: dict, keys: tuple) -> str | None:
    current = persona
    for k in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(k)
    return str(current) if current is not None else None


def _cramers_v(x: list[str], y: list[str]) -> float:
    """Compute Cramer's V statistic for two categorical variables."""
    if len(x) != len(y) or len(x) < 5:
        return 0.0

    # Build contingency table
    x_cats = sorted(set(x))
    y_cats = sorted(set(y))
    if len(x_cats) < 2 or len(y_cats) < 2:
        return 0.0

    x_idx = {c: i for i, c in enumerate(x_cats)}
    y_idx = {c: i for i, c in enumerate(y_cats)}
    table = np.zeros((len(x_cats), len(y_cats)))

    for xi, yi in zip(x, y):
        table[x_idx[xi]][y_idx[yi]] += 1

    n = table.sum()
    if n == 0:
        return 0.0

    # Chi-squared
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n
    expected = np.where(expected == 0, 1e-10, expected)
    chi2 = ((table - expected) ** 2 / expected).sum()

    k = min(len(x_cats), len(y_cats))
    if k <= 1:
        return 0.0

    return float(np.sqrt(chi2 / (n * (k - 1))))


class JointDistributionEvaluator(BaseEvaluator):
    dimension_id = "D18"
    dimension_name = "Joint Distribution Fidelity"
    tier = 3

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        persona_set = kwargs.get("persona_set", [])
        if len(persona_set) < 10:
            return self._make_result(passed=True, score=1.0, details={"skipped": True})

        # Extract all categorical fields
        field_data: dict[str, list[str]] = {}
        for keys, name in CATEGORICAL_FIELDS:
            vals = []
            for p in persona_set:
                v = _extract(p, keys)
                if v:
                    vals.append(v)
            if len(vals) >= 10:
                field_data[name] = vals

        if len(field_data) < 2:
            return self._make_result(passed=True, score=1.0, details={"skipped": True})

        # Compute pairwise Cramer's V
        correlation_matrix = {}
        stereotypical_pairs = []
        all_vs = []

        for (name_a, vals_a), (name_b, vals_b) in combinations(field_data.items(), 2):
            # Align to same length (in case of missing values)
            min_len = min(len(vals_a), len(vals_b))
            v = _cramers_v(vals_a[:min_len], vals_b[:min_len])
            pair_key = f"{name_a} x {name_b}"
            correlation_matrix[pair_key] = round(v, 4)
            all_vs.append(v)

            if v > STEREOTYPE_THRESHOLD:
                stereotypical_pairs.append({"pair": pair_key, "cramers_v": round(v, 4)})

        mean_v = sum(all_vs) / len(all_vs) if all_vs else 0.0
        # Score: lower correlation = better (less stereotypical), but some correlation is expected
        # Penalize high mean correlation and stereotypical pairs
        score = max(0.0, 1.0 - mean_v)
        if stereotypical_pairs:
            score *= 0.5  # heavy penalty for stereotypical correlations

        passed = len(stereotypical_pairs) == 0 and mean_v < 0.5

        return self._make_result(
            passed=passed,
            score=round(score, 4),
            details={
                "correlation_matrix": correlation_matrix,
                "mean_cramers_v": round(mean_v, 4),
                "stereotypical_pairs": stereotypical_pairs,
            },
        )
```

3. Verify and commit (2 min):

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/evaluators/distributional/test_d18_joint_distribution.py -v
git add persona_eval/evaluators/distributional/d18_joint_distribution.py tests/evaluators/distributional/test_d18_joint_distribution.py
git commit -m "feat: D18 Joint Distribution Fidelity evaluator with Cramer's V and stereotype detection"
```

---

## Phase 4 — Tier 2: Semantic Validators (Tasks 14-22)

### Task 14: D4 Factual Grounding + Embedding Wrapper

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/embeddings.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/semantic/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/semantic/d04_factual_grounding.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/test_embeddings.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/semantic/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/semantic/test_d04_factual_grounding.py`

**Steps:**

1. Write failing tests for embedding wrapper (2 min):

```python
# tests/test_embeddings.py
"""Tests for the embedding wrapper."""

import pytest


def test_embedder_importable():
    from persona_eval.embeddings import Embedder
    assert Embedder is not None


@pytest.mark.slow
def test_embed_returns_vector():
    from persona_eval.embeddings import Embedder
    embedder = Embedder()
    vec = embedder.embed("Hello world")
    assert len(vec) > 0
    assert isinstance(vec[0], float)


@pytest.mark.slow
def test_embed_batch():
    from persona_eval.embeddings import Embedder
    embedder = Embedder()
    vecs = embedder.embed_batch(["Hello", "World"])
    assert len(vecs) == 2
    assert len(vecs[0]) == len(vecs[1])


@pytest.mark.slow
def test_similarity():
    from persona_eval.embeddings import Embedder
    embedder = Embedder()
    sim = embedder.similarity("I love dogs", "I love dogs")
    assert sim > 0.9
    sim_diff = embedder.similarity("I love dogs", "Quantum physics is complex")
    assert sim_diff < sim
```

2. Implement embedding wrapper (3 min):

```python
# persona_eval/embeddings.py
"""Thin wrapper around sentence-transformers for embeddings.

All embedding calls go through this so the model can be swapped.
Uses local models — zero cost, zero API dependency, runs in CI.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class Embedder:
    """Wrapper for sentence-transformers embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        vec = self._model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        vecs = self._model.encode(texts, convert_to_numpy=True)
        return vecs.tolist()

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        vecs = self._model.encode([text_a, text_b], convert_to_numpy=True)
        cos = np.dot(vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]))
        return float(cos)

    def retrieval_score(self, query: str, passages: list[str], top_k: int = 3) -> list[tuple[int, float]]:
        """Retrieve top-k passages by similarity to query.

        Returns list of (passage_index, similarity_score).
        """
        q_vec = self._model.encode(query, convert_to_numpy=True)
        p_vecs = self._model.encode(passages, convert_to_numpy=True)
        sims = np.dot(p_vecs, q_vec) / (
            np.linalg.norm(p_vecs, axis=1) * np.linalg.norm(q_vec)
        )
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [(int(i), float(sims[i])) for i in top_indices]
```

3. Write failing D4 tests (3 min):

```python
# tests/evaluators/semantic/__init__.py
# empty

# tests/evaluators/semantic/test_d04_factual_grounding.py
"""Tests for D4 Factual Grounding evaluator."""

import pytest


def test_evaluator_importable():
    from persona_eval.evaluators.semantic.d04_factual_grounding import FactualGroundingEvaluator
    assert FactualGroundingEvaluator is not None


@pytest.mark.slow
def test_grounded_persona_scores_well(sample_persona_dict, sample_source_blob):
    from persona_eval.evaluators.semantic.d04_factual_grounding import FactualGroundingEvaluator
    evaluator = FactualGroundingEvaluator()
    result = evaluator.evaluate(sample_persona_dict, source_blob=sample_source_blob)
    assert result.score >= 0.3
    assert "claim_scores" in result.details


@pytest.mark.slow
def test_ungrounded_persona_scores_poorly():
    from persona_eval.evaluators.semantic.d04_factual_grounding import FactualGroundingEvaluator
    evaluator = FactualGroundingEvaluator()
    ungrounded = {
        "professional": {"role": "Underwater Basket Weaver", "industry": "Basket Arts"},
        "goals": [{"description": "Win the Olympic basket weaving gold medal"}],
    }
    source = "Interview with software engineers about code review practices."
    result = evaluator.evaluate(ungrounded, source_blob=source)
    assert result.score < 0.5


@pytest.mark.slow
def test_no_source_blob_skips():
    from persona_eval.evaluators.semantic.d04_factual_grounding import FactualGroundingEvaluator
    evaluator = FactualGroundingEvaluator()
    result = evaluator.evaluate({"id": "test"})
    assert result.details.get("skipped") is True
```

4. Implement D4 (3 min):

```python
# persona_eval/evaluators/semantic/__init__.py
"""Tier 2 — Semantic validators (D4-D12)."""

# persona_eval/evaluators/semantic/d04_factual_grounding.py
"""D4 Factual Grounding — claim extraction + retrieval against source blob.

Trustworthiness: MEDIUM (threshold-dependent).
Method: Extract claims from persona fields, retrieve against source blob chunks,
measure semantic similarity.
"""

from __future__ import annotations

import re
from typing import Any

from persona_eval.embeddings import Embedder
from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult

SIMILARITY_THRESHOLD = 0.3  # Minimum similarity to consider a claim grounded
CHUNK_SIZE = 200  # Characters per source chunk


def _extract_claims(persona_data: dict) -> list[str]:
    """Extract testable claims from persona fields."""
    claims = []

    # Professional claims
    pro = persona_data.get("professional", {})
    if pro.get("role"):
        claims.append(f"The person works as a {pro['role']}")
    if pro.get("industry"):
        claims.append(f"They work in {pro['industry']}")
    if pro.get("years_experience") is not None:
        claims.append(f"They have {pro['years_experience']} years of experience")
    if pro.get("team_size") is not None:
        claims.append(f"They manage a team of {pro['team_size']} people")

    # Goals
    for goal in persona_data.get("goals", []):
        if isinstance(goal, dict) and goal.get("description"):
            claims.append(f"Their goal is to {goal['description']}")

    # Pain points
    for pp in persona_data.get("pain_points", []):
        if isinstance(pp, dict) and pp.get("description"):
            claims.append(f"They experience: {pp['description']}")

    # Identity
    identity = persona_data.get("identity", {})
    if identity.get("age"):
        claims.append(f"They are {identity['age']} years old")
    if identity.get("location"):
        claims.append(f"They are located in {identity['location']}")

    return claims


def _chunk_source(source_blob: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split source blob into overlapping chunks for retrieval."""
    sentences = re.split(r'(?<=[.!?])\s+', source_blob)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [source_blob]


class FactualGroundingEvaluator(BaseEvaluator):
    dimension_id = "D4"
    dimension_name = "Factual Grounding"
    tier = 2

    def __init__(self):
        self._embedder = None  # Lazy init for test performance

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        source_blob = kwargs.get("source_blob")
        if not source_blob:
            return self._make_result(passed=True, score=1.0, details={"skipped": True, "reason": "No source blob provided"})

        claims = _extract_claims(persona_data)
        if not claims:
            return self._make_result(passed=True, score=1.0, details={"skipped": True, "reason": "No claims extracted"})

        embedder = self._get_embedder()
        chunks = _chunk_source(source_blob)

        claim_scores = []
        for claim in claims:
            results = embedder.retrieval_score(claim, chunks, top_k=1)
            best_score = results[0][1] if results else 0.0
            claim_scores.append({
                "claim": claim,
                "best_match_score": round(best_score, 4),
                "grounded": best_score >= SIMILARITY_THRESHOLD,
            })

        grounded_count = sum(1 for cs in claim_scores if cs["grounded"])
        score = grounded_count / len(claim_scores) if claim_scores else 0.0
        passed = score >= 0.5  # At least half of claims should be grounded

        return self._make_result(
            passed=passed,
            score=round(score, 4),
            details={
                "claim_scores": claim_scores,
                "grounded_claims": grounded_count,
                "total_claims": len(claim_scores),
            },
        )
```

5. Verify and commit (2 min):

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/evaluators/semantic/test_d04_factual_grounding.py tests/test_embeddings.py -v -m "not slow"
# Run slow tests separately:
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/evaluators/semantic/test_d04_factual_grounding.py tests/test_embeddings.py -v
```

```bash
git add persona_eval/embeddings.py persona_eval/evaluators/semantic/ tests/test_embeddings.py tests/evaluators/semantic/
git commit -m "feat: D4 Factual Grounding evaluator with sentence-transformer retrieval"
```

---

### Task 15: D5 Behavioral Consistency

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/llm_client.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/semantic/d05_behavioral_consistency.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/test_llm_client.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/semantic/test_d05_behavioral_consistency.py`

**Steps:**

1. Write failing tests for LLM client wrapper (2 min):

```python
# tests/test_llm_client.py
"""Tests for the LiteLLM client wrapper."""

import pytest


def test_llm_client_importable():
    from persona_eval.llm_client import LLMClient
    assert LLMClient is not None


def test_llm_client_init():
    from persona_eval.llm_client import LLMClient
    client = LLMClient(model="claude-sonnet-4-20250514")
    assert client.model == "claude-sonnet-4-20250514"


def test_llm_client_format_messages():
    from persona_eval.llm_client import LLMClient
    client = LLMClient(model="test")
    messages = client.format_messages(
        system="You are a test assistant.",
        user="Hello",
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
```

2. Implement LLM client wrapper (3 min):

```python
# persona_eval/llm_client.py
"""Thin wrapper around LiteLLM for all LLM calls.

Never import anthropic or openai directly — everything goes through LiteLLM.
"""

from __future__ import annotations

from typing import Any, Optional


class LLMClient:
    """LiteLLM-backed client for all LLM operations in the eval framework."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
    ) -> str:
        """Send a completion request via LiteLLM."""
        import litellm
        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        response_format: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
    ) -> str:
        """Send a structured completion request via LiteLLM."""
        import litellm
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        response = litellm.completion(**kwargs)
        return response.choices[0].message.content

    def format_messages(
        self,
        system: str,
        user: str,
        assistant: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Format messages for a standard system+user prompt."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
        return messages

    def multi_turn(self, messages: list[dict[str, str]]) -> str:
        """Continue a multi-turn conversation."""
        return self.complete(messages)
```

3. Write failing D5 tests (2 min):

```python
# tests/evaluators/semantic/test_d05_behavioral_consistency.py
"""Tests for D5 Behavioral Consistency evaluator."""

import pytest


def test_evaluator_importable():
    from persona_eval.evaluators.semantic.d05_behavioral_consistency import BehavioralConsistencyEvaluator
    assert BehavioralConsistencyEvaluator is not None


@pytest.mark.slow
def test_consistent_responses_score_well():
    from persona_eval.evaluators.semantic.d05_behavioral_consistency import BehavioralConsistencyEvaluator
    evaluator = BehavioralConsistencyEvaluator()
    # Provide pre-computed responses (bypass LLM call for unit testing)
    responses = [
        "I prefer data-driven approaches to decision making.",
        "I like to base my decisions on solid data and metrics.",
        "Data and evidence guide my decision-making process.",
        "I rely on analytics and data when making key decisions.",
        "My decision making is rooted in quantitative evidence.",
    ]
    result = evaluator.evaluate({}, pre_computed_responses=responses)
    assert result.score >= 0.5
    assert "centroid_radius" in result.details


@pytest.mark.slow
def test_inconsistent_responses_score_poorly():
    from persona_eval.evaluators.semantic.d05_behavioral_consistency import BehavioralConsistencyEvaluator
    evaluator = BehavioralConsistencyEvaluator()
    responses = [
        "I love eating pizza with my family.",
        "Quantum mechanics is the foundation of modern physics.",
        "The stock market crashed yesterday.",
        "I prefer hiking in the mountains.",
        "Java is my favorite programming language.",
    ]
    result = evaluator.evaluate({}, pre_computed_responses=responses)
    assert result.score < 0.7


def test_empty_responses_skips():
    from persona_eval.evaluators.semantic.d05_behavioral_consistency import BehavioralConsistencyEvaluator
    evaluator = BehavioralConsistencyEvaluator()
    result = evaluator.evaluate({}, pre_computed_responses=[])
    assert result.details.get("skipped") is True
```

4. Implement D5 (3 min):

```python
# persona_eval/evaluators/semantic/d05_behavioral_consistency.py
"""D5 Behavioral Consistency — repeated query consistency via embedding clusters.

Trustworthiness: MEDIUM (measures surface consistency).
Method: Embed responses, compute centroid + radius of cluster.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from persona_eval.embeddings import Embedder
from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult


class BehavioralConsistencyEvaluator(BaseEvaluator):
    dimension_id = "D5"
    dimension_name = "Behavioral Consistency"
    tier = 2

    def __init__(self):
        self._embedder = None

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        """Measure consistency of responses via embedding cluster tightness.

        Can either:
        1. Accept pre_computed_responses (for unit testing)
        2. Generate responses via LLM (requires llm_client and a query)
        """
        responses = kwargs.get("pre_computed_responses", [])

        if len(responses) < 3:
            return self._make_result(passed=True, score=1.0, details={"skipped": True, "reason": "Need >= 3 responses"})

        embedder = self._get_embedder()
        vecs = np.array(embedder.embed_batch(responses))

        # Compute centroid
        centroid = vecs.mean(axis=0)

        # Compute distances from centroid
        distances = np.array([
            np.linalg.norm(v - centroid) for v in vecs
        ])

        mean_dist = float(distances.mean())
        max_dist = float(distances.max())
        std_dist = float(distances.std())

        # Compute pairwise cosine similarities
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normalized = vecs / norms
        sim_matrix = np.dot(normalized, normalized.T)
        # Get upper triangle (excluding diagonal)
        triu_indices = np.triu_indices(len(vecs), k=1)
        pairwise_sims = sim_matrix[triu_indices]
        mean_sim = float(pairwise_sims.mean())
        min_sim = float(pairwise_sims.min())

        # Score: higher mean similarity = more consistent
        score = max(0.0, min(1.0, mean_sim))
        passed = mean_sim >= 0.5

        return self._make_result(
            passed=passed,
            score=round(score, 4),
            details={
                "centroid_radius": round(mean_dist, 4),
                "max_distance": round(max_dist, 4),
                "std_distance": round(std_dist, 4),
                "mean_pairwise_similarity": round(mean_sim, 4),
                "min_pairwise_similarity": round(min_sim, 4),
                "n_responses": len(responses),
            },
        )
```

5. Verify and commit (2 min):

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/evaluators/semantic/test_d05_behavioral_consistency.py tests/test_llm_client.py -v
git add persona_eval/llm_client.py persona_eval/evaluators/semantic/d05_behavioral_consistency.py tests/test_llm_client.py tests/evaluators/semantic/test_d05_behavioral_consistency.py
git commit -m "feat: D5 Behavioral Consistency evaluator with embedding cluster analysis, plus LLM client wrapper"
```

---

### Tasks 16-22: D6-D12 (Remaining Semantic Validators)

For brevity I provide the pattern — each follows the identical TDD cycle. Exact files and test structures below.

### Task 16: D6 Distinctiveness

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/semantic/d06_distinctiveness.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/semantic/test_d06_distinctiveness.py`

**Steps:**

1. Write failing tests (2 min):

```python
# tests/evaluators/semantic/test_d06_distinctiveness.py
"""Tests for D6 Distinctiveness evaluator."""

import pytest
from tests.fixtures.persona_set import generate_test_persona_set, generate_homogeneous_set


@pytest.fixture
def diverse_set():
    return generate_test_persona_set(n=20, seed=42)


@pytest.fixture
def clone_set():
    return generate_homogeneous_set(n=20)


def test_evaluator_importable():
    from persona_eval.evaluators.semantic.d06_distinctiveness import DistinctivenessEvaluator
    assert DistinctivenessEvaluator is not None


@pytest.mark.slow
def test_diverse_personas_are_distinct(diverse_set):
    from persona_eval.evaluators.semantic.d06_distinctiveness import DistinctivenessEvaluator
    evaluator = DistinctivenessEvaluator()
    result = evaluator.evaluate({}, persona_set=diverse_set)
    assert result.score >= 0.3
    assert "mean_pairwise_distance" in result.details


@pytest.mark.slow
def test_clones_are_not_distinct(clone_set):
    from persona_eval.evaluators.semantic.d06_distinctiveness import DistinctivenessEvaluator
    evaluator = DistinctivenessEvaluator()
    result = evaluator.evaluate({}, persona_set=clone_set)
    assert result.passed is False
    assert result.score < 0.3
```

2. Implement (3 min):

```python
# persona_eval/evaluators/semantic/d06_distinctiveness.py
"""D6 Distinctiveness — pairwise embedding distance + variation ratio.

Trustworthiness: MEDIUM (captures surface differences).
Method: Embed persona descriptions, compute pairwise cosine distances.
"""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any

import numpy as np

from persona_eval.embeddings import Embedder
from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult


def _persona_to_text(persona: dict) -> str:
    """Convert persona dict to a text representation for embedding."""
    parts = []
    for key in ["identity", "professional", "behavioral", "psychographic", "communication_style", "emotional_profile"]:
        section = persona.get(key, {})
        if isinstance(section, dict):
            parts.append(f"{key}: {json.dumps(section, default=str)}")
    for goal in persona.get("goals", []):
        if isinstance(goal, dict):
            parts.append(f"goal: {goal.get('description', '')}")
    for pp in persona.get("pain_points", []):
        if isinstance(pp, dict):
            parts.append(f"pain_point: {pp.get('description', '')}")
    return " | ".join(parts)


class DistinctivenessEvaluator(BaseEvaluator):
    dimension_id = "D6"
    dimension_name = "Distinctiveness"
    tier = 2

    def __init__(self):
        self._embedder = None

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        persona_set = kwargs.get("persona_set", [])
        if len(persona_set) < 2:
            return self._make_result(passed=True, score=1.0, details={"skipped": True})

        embedder = self._get_embedder()
        texts = [_persona_to_text(p) for p in persona_set]
        vecs = np.array(embedder.embed_batch(texts))

        # Compute pairwise cosine distances
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normalized = vecs / (norms + 1e-10)
        sim_matrix = np.dot(normalized, normalized.T)
        triu_indices = np.triu_indices(len(vecs), k=1)
        pairwise_sims = sim_matrix[triu_indices]
        pairwise_dists = 1.0 - pairwise_sims

        mean_dist = float(pairwise_dists.mean())
        min_dist = float(pairwise_dists.min())

        # Flag near-duplicate pairs
        DUPLICATE_THRESHOLD = 0.05
        near_duplicates = []
        for idx, (i, j) in enumerate(zip(*triu_indices)):
            if pairwise_dists[idx] < DUPLICATE_THRESHOLD:
                near_duplicates.append({
                    "persona_a": persona_set[i].get("id", str(i)),
                    "persona_b": persona_set[j].get("id", str(j)),
                    "distance": round(float(pairwise_dists[idx]), 4),
                })

        score = min(1.0, mean_dist / 0.5)  # Normalize: 0.5 distance = perfect diversity
        passed = len(near_duplicates) == 0 and mean_dist >= 0.1

        return self._make_result(
            passed=passed,
            score=round(score, 4),
            details={
                "mean_pairwise_distance": round(mean_dist, 4),
                "min_pairwise_distance": round(min_dist, 4),
                "near_duplicates": near_duplicates,
                "persona_count": len(persona_set),
            },
        )
```

3. Verify and commit:

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/evaluators/semantic/test_d06_distinctiveness.py -v
git add persona_eval/evaluators/semantic/d06_distinctiveness.py tests/evaluators/semantic/test_d06_distinctiveness.py
git commit -m "feat: D6 Distinctiveness evaluator with pairwise embedding distance"
```

---

### Task 17: D7 Demographic Coherence

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/semantic/d07_demographic_coherence.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/semantic/test_d07_demographic_coherence.py`

**Steps:**

1. Write failing tests (2 min):

```python
# tests/evaluators/semantic/test_d07_demographic_coherence.py
"""Tests for D7 Demographic Coherence evaluator."""

import pytest


def test_evaluator_importable():
    from persona_eval.evaluators.semantic.d07_demographic_coherence import DemographicCoherenceEvaluator
    assert DemographicCoherenceEvaluator is not None


def test_plausible_persona_passes(sample_persona_dict):
    from persona_eval.evaluators.semantic.d07_demographic_coherence import DemographicCoherenceEvaluator
    evaluator = DemographicCoherenceEvaluator()
    result = evaluator.evaluate(sample_persona_dict)
    assert result.score >= 0.5


def test_implausible_combo_detected():
    from persona_eval.evaluators.semantic.d07_demographic_coherence import DemographicCoherenceEvaluator
    evaluator = DemographicCoherenceEvaluator()
    implausible = {
        "identity": {"name": "Test", "age": 19, "gender": "male", "location": "NY"},
        "demographics": {"education_level": "PhD", "income_bracket": "high",
                         "marital_status": "widowed", "household_size": 1, "ethnicity": "White"},
        "professional": {"role": "CEO", "industry": "Finance", "company_size": "1000+",
                         "years_experience": 0, "team_size": 500, "responsibilities": ["Leading"]},
        "behavioral": {"technology_adoption": "laggard", "decision_making_style": "data",
                       "information_sources": ["blogs"], "purchase_triggers": [], "brand_loyalty": "low"},
        "psychographic": {"personality_traits": ["cautious"], "risk_tolerance": "very_high",
                          "work_life_balance_priority": "low", "innovation_orientation": "radical"},
        "communication_style": {"tone": "formal", "vocabulary_level": "basic",
                                "preferred_channels": ["email"], "formality": "semi-formal", "verbosity": "concise"},
        "goals": [{"description": "g", "timeframe": "1y", "priority": "high"}],
        "pain_points": [{"description": "p", "severity": "low", "frequency": "rarely"}],
        "values": ["v"],
        "knowledge_domains": [{"domain": "d", "depth": "expert"}],
        "emotional_profile": {"baseline_mood": "happy", "stress_response": "s",
                              "conflict_style": "competitive", "enthusiasm_triggers": ["t"], "frustration_triggers": ["f"]},
        "source_context": "test",
    }
    result = evaluator.evaluate(implausible)
    assert result.score < 0.8
    assert len(result.details.get("anomalies", [])) > 0
```

2. Implement (3 min):

```python
# persona_eval/evaluators/semantic/d07_demographic_coherence.py
"""D7 Demographic Coherence — co-occurrence plausibility checking.

Trustworthiness: HIGH (when reference data exists).
Method: Check attribute combinations against plausibility rules.
"""

from __future__ import annotations

from typing import Any

from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult


def _safe_get(data: dict, *keys, default=None):
    current = data
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k, default)
    return current


# Plausibility rules: (check_function, description)
def _check_phd_age(data):
    age = _safe_get(data, "identity", "age")
    edu = _safe_get(data, "demographics", "education_level", default="")
    if age and "phd" in edu.lower() and age < 24:
        return (False, f"PhD at age {age} is extremely rare")
    return (True, "")


def _check_ceo_age(data):
    age = _safe_get(data, "identity", "age")
    role = _safe_get(data, "professional", "role", default="")
    if age and age < 25 and any(t in role.lower() for t in ["ceo", "chief", "vp", "vice president"]):
        return (False, f"C-suite role '{role}' at age {age} is implausible")
    return (True, "")


def _check_widowed_young(data):
    age = _safe_get(data, "identity", "age")
    status = _safe_get(data, "demographics", "marital_status", default="")
    if age and age < 25 and "widow" in status.lower():
        return (False, f"Widowed at age {age} is statistically very rare")
    return (True, "")


def _check_income_education_mismatch(data):
    edu = _safe_get(data, "demographics", "education_level", default="")
    income = _safe_get(data, "demographics", "income_bracket", default="")
    age = _safe_get(data, "identity", "age", default=40)
    if "high school" in edu.lower() and income == "high" and age < 30:
        return (False, "High income with only high school education at young age is uncommon")
    return (True, "")


def _check_team_size_vs_role(data):
    team = _safe_get(data, "professional", "team_size")
    role = _safe_get(data, "professional", "role", default="")
    if team and team > 100 and not any(t in role.lower() for t in ["ceo", "chief", "vp", "director", "president", "head"]):
        return (False, f"Team size {team} for role '{role}' seems implausible")
    return (True, "")


COHERENCE_RULES = [
    _check_phd_age,
    _check_ceo_age,
    _check_widowed_young,
    _check_income_education_mismatch,
    _check_team_size_vs_role,
]


class DemographicCoherenceEvaluator(BaseEvaluator):
    dimension_id = "D7"
    dimension_name = "Demographic Coherence"
    tier = 2

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        anomalies = []
        checked = 0

        for rule_fn in COHERENCE_RULES:
            try:
                result = rule_fn(persona_data)
                if result is None:
                    continue
                checked += 1
                passed, msg = result
                if not passed:
                    anomalies.append({"rule": rule_fn.__name__, "message": msg})
            except Exception:
                checked += 1

        if checked == 0:
            return self._make_result(passed=True, score=1.0, details={"skipped": True})

        score = max(0.0, (checked - len(anomalies)) / checked)
        passed = len(anomalies) == 0

        return self._make_result(
            passed=passed,
            score=round(score, 4),
            details={"anomalies": anomalies, "rules_checked": checked},
        )
```

3. Verify and commit:

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/evaluators/semantic/test_d07_demographic_coherence.py -v
git add persona_eval/evaluators/semantic/d07_demographic_coherence.py tests/evaluators/semantic/test_d07_demographic_coherence.py
git commit -m "feat: D7 Demographic Coherence evaluator with co-occurrence plausibility checking"
```

---

### Task 18: D8 Memory Consistency

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/semantic/d08_memory_consistency.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/semantic/test_d08_memory_consistency.py`

**Steps:**

1. Write failing test, implement, verify, commit — same TDD cycle. Key implementation:

```python
# persona_eval/evaluators/semantic/d08_memory_consistency.py
"""D8 Memory Consistency — direct and indirect recall probes.

Trustworthiness: HIGH for direct recall, MEDIUM for indirect.
Method: Generate recall probes from persona attributes, compare answers to source.
"""

from __future__ import annotations

from typing import Any

from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult


def _generate_recall_probes(persona_data: dict) -> list[dict]:
    """Generate direct recall probes from persona attributes."""
    probes = []
    identity = persona_data.get("identity", {})
    pro = persona_data.get("professional", {})

    if identity.get("age"):
        probes.append({"question": "How old are you?", "expected": str(identity["age"]), "field": "identity.age", "type": "direct"})
    if identity.get("location"):
        probes.append({"question": "Where do you live?", "expected": identity["location"], "field": "identity.location", "type": "direct"})
    if pro.get("role"):
        probes.append({"question": "What is your job title?", "expected": pro["role"], "field": "professional.role", "type": "direct"})
    if pro.get("years_experience") is not None:
        probes.append({"question": "How many years of experience do you have?", "expected": str(pro["years_experience"]), "field": "professional.years_experience", "type": "direct"})
    if pro.get("team_size") is not None:
        probes.append({"question": "How many people are on your team?", "expected": str(pro["team_size"]), "field": "professional.team_size", "type": "direct"})
    if pro.get("industry"):
        probes.append({"question": "What industry do you work in?", "expected": pro["industry"], "field": "professional.industry", "type": "direct"})

    return probes


class MemoryConsistencyEvaluator(BaseEvaluator):
    dimension_id = "D8"
    dimension_name = "Memory Consistency"
    tier = 2

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        """Evaluate memory consistency using recall probes.

        In unit test mode, accepts pre_computed_answers: dict mapping field to answer.
        In full mode, generates probes and sends to LLM.
        """
        probes = _generate_recall_probes(persona_data)
        if not probes:
            return self._make_result(passed=True, score=1.0, details={"skipped": True, "reason": "No probes generated"})

        pre_computed = kwargs.get("pre_computed_answers", {})
        probe_results = []

        for probe in probes:
            answer = pre_computed.get(probe["field"])
            if answer is not None:
                # Check if answer contains the expected value
                expected_lower = probe["expected"].lower()
                answer_lower = answer.lower()
                correct = expected_lower in answer_lower
                probe_results.append({
                    "field": probe["field"],
                    "question": probe["question"],
                    "expected": probe["expected"],
                    "answer": answer,
                    "correct": correct,
                })

        if not probe_results:
            return self._make_result(passed=True, score=1.0, details={"skipped": True, "reason": "No answers to check"})

        correct_count = sum(1 for p in probe_results if p["correct"])
        score = correct_count / len(probe_results)
        passed = score >= 0.8

        return self._make_result(
            passed=passed,
            score=round(score, 4),
            details={
                "probes": probe_results,
                "correct": correct_count,
                "total": len(probe_results),
            },
        )
```

Test file:

```python
# tests/evaluators/semantic/test_d08_memory_consistency.py
"""Tests for D8 Memory Consistency evaluator."""

import pytest


def test_evaluator_importable():
    from persona_eval.evaluators.semantic.d08_memory_consistency import MemoryConsistencyEvaluator
    assert MemoryConsistencyEvaluator is not None


def test_perfect_recall_passes(sample_persona_dict):
    from persona_eval.evaluators.semantic.d08_memory_consistency import MemoryConsistencyEvaluator
    evaluator = MemoryConsistencyEvaluator()
    answers = {
        "identity.age": "I am 34 years old.",
        "identity.location": "I live in San Francisco, CA.",
        "professional.role": "I'm a Senior Product Manager.",
        "professional.years_experience": "I have 8 years of experience.",
        "professional.team_size": "My team has 6 people.",
        "professional.industry": "I work in SaaS / B2B Technology.",
    }
    result = evaluator.evaluate(sample_persona_dict, pre_computed_answers=answers)
    assert result.passed is True
    assert result.score >= 0.8


def test_incorrect_recall_fails(sample_persona_dict):
    from persona_eval.evaluators.semantic.d08_memory_consistency import MemoryConsistencyEvaluator
    evaluator = MemoryConsistencyEvaluator()
    answers = {
        "identity.age": "I am 25 years old.",  # Wrong: should be 34
        "identity.location": "I live in New York.",  # Wrong
        "professional.role": "I'm a Software Engineer.",  # Wrong
    }
    result = evaluator.evaluate(sample_persona_dict, pre_computed_answers=answers)
    assert result.passed is False
    assert result.score < 0.5
```

Commit:
```bash
git add persona_eval/evaluators/semantic/d08_memory_consistency.py tests/evaluators/semantic/test_d08_memory_consistency.py
git commit -m "feat: D8 Memory Consistency evaluator with recall probe generation"
```

---

### Task 19: D9 Knowledge Boundary Awareness

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/semantic/d09_knowledge_boundary.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/semantic/test_d09_knowledge_boundary.py`

Key implementation: generates out-of-domain probe questions based on persona's knowledge_domains, measures whether responses show appropriate uncertainty for topics outside the persona's expertise. Tests accept pre_computed_responses for unit testing.

```python
# persona_eval/evaluators/semantic/d09_knowledge_boundary.py
"""D9 Knowledge Boundary Awareness — out-of-domain probe battery.

Trustworthiness: MEDIUM (requires reliable model of what the persona should know).
Method: Generate questions outside persona's expertise, check for appropriate uncertainty.
"""

from __future__ import annotations

import re
from typing import Any

from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult

# Domains to probe — if NOT in persona's knowledge domains
PROBE_DOMAINS = [
    "quantum_physics", "marine_biology", "medieval_history", "constitutional_law",
    "neurosurgery", "petroleum_engineering", "classical_music_composition",
    "commercial_aviation", "professional_boxing", "haute_couture_fashion",
]

# Signals of appropriate uncertainty
UNCERTAINTY_MARKERS = [
    re.compile(r"\b(not sure|don'?t know|not my area|outside my expertise|can'?t speak to)\b", re.IGNORECASE),
    re.compile(r"\b(not really|wouldn'?t know|not qualified|not familiar|don'?t have experience)\b", re.IGNORECASE),
    re.compile(r"\b(I'?d have to|you'?d be better off asking|I'?m not the right person)\b", re.IGNORECASE),
]


class KnowledgeBoundaryEvaluator(BaseEvaluator):
    dimension_id = "D9"
    dimension_name = "Knowledge Boundary Awareness"
    tier = 2

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        """Check if persona shows appropriate uncertainty on out-of-domain questions.

        Accepts pre_computed_responses: list of {"domain": str, "response": str}
        """
        pre_computed = kwargs.get("pre_computed_responses", [])
        if not pre_computed:
            return self._make_result(passed=True, score=1.0, details={"skipped": True, "reason": "No responses to evaluate"})

        results = []
        for item in pre_computed:
            response = item["response"]
            shows_uncertainty = any(p.search(response) for p in UNCERTAINTY_MARKERS)
            results.append({
                "domain": item["domain"],
                "shows_uncertainty": shows_uncertainty,
                "response_snippet": response[:100],
            })

        appropriate_count = sum(1 for r in results if r["shows_uncertainty"])
        score = appropriate_count / len(results) if results else 0.0
        passed = score >= 0.6

        return self._make_result(
            passed=passed,
            score=round(score, 4),
            details={"probe_results": results, "appropriate_uncertainty_rate": round(score, 4)},
        )
```

Test and commit pattern identical to previous tasks.

---

### Task 20: D10 Lexical vs Semantic Generalization

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/semantic/d10_lexical_semantic.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/semantic/test_d10_lexical_semantic.py`

Key implementation: tests consistency when same questions are asked with zero lexical overlap. Uses embedding similarity between original and paraphrased responses.

---

### Task 21: D11 Profile Coverage

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/semantic/d11_profile_coverage.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/semantic/test_d11_profile_coverage.py`

Key implementation: tracks which persona attributes are mentioned/expressed across simulated conversation turns. Coverage ratio = attributes_expressed / total_attributes.

---

### Task 22: D12 Narrative Coherence

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/semantic/d12_narrative_coherence.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/semantic/test_d12_narrative_coherence.py`

Key implementation: LLM-as-judge with structured rubric. Sends full persona to LLM with prompt: "Rate how well this persona reads as a single person's life story on a 1-5 scale." Parses structured response. Also validates career progression (education -> early career -> current role).

```python
# persona_eval/evaluators/semantic/d12_narrative_coherence.py
"""D12 Narrative Coherence — LLM-as-judge with structured rubric.

Trustworthiness: LOW-MEDIUM (LLMs are poor judges of narrative quality).
Method: LLM-as-judge with narrative coherence rubric + story arc validation.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult

RUBRIC = """Rate this persona on narrative coherence (1-5 scale):

1 = Incoherent: Fields seem randomly assembled, no sense of a real person
2 = Weak: Some connections but major gaps or implausibilities
3 = Acceptable: Mostly coherent but feels somewhat artificial
4 = Strong: Reads like a real person with a plausible life story
5 = Excellent: Compelling, internally consistent narrative

Evaluate:
- Does the career trajectory match the skills and education?
- Does the communication style match the background?
- Do the goals and pain points fit the professional context?
- Does the emotional profile fit the personality traits?
- Does it feel like ONE person's life?

Respond ONLY with JSON: {"score": N, "reasoning": "..."}
"""


class NarrativeCoherenceEvaluator(BaseEvaluator):
    dimension_id = "D12"
    dimension_name = "Narrative Coherence"
    tier = 2

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        """Evaluate narrative coherence via LLM-as-judge or pre-computed score."""
        pre_computed_score = kwargs.get("pre_computed_score")

        if pre_computed_score is not None:
            score_normalized = pre_computed_score / 5.0
            return self._make_result(
                passed=score_normalized >= 0.6,
                score=round(score_normalized, 4),
                details={"raw_score": pre_computed_score, "source": "pre_computed"},
            )

        # LLM-as-judge mode
        llm_client = kwargs.get("llm_client")
        if llm_client is None:
            return self._make_result(passed=True, score=1.0, details={"skipped": True, "reason": "No LLM client provided"})

        persona_text = json.dumps(persona_data, indent=2, default=str)
        messages = llm_client.format_messages(
            system=RUBRIC,
            user=f"Persona to evaluate:\n\n{persona_text}",
        )

        try:
            response = llm_client.complete(messages, max_tokens=500)
            # Parse JSON from response
            parsed = json.loads(response)
            raw_score = parsed.get("score", 3)
            reasoning = parsed.get("reasoning", "")
            score_normalized = max(0.0, min(1.0, raw_score / 5.0))

            return self._make_result(
                passed=score_normalized >= 0.6,
                score=round(score_normalized, 4),
                details={
                    "raw_score": raw_score,
                    "reasoning": reasoning,
                    "source": "llm_judge",
                },
            )
        except (json.JSONDecodeError, Exception) as e:
            return self._make_result(
                passed=False, score=0.5,
                details={"error": str(e), "source": "llm_judge"},
                error=str(e),
            )
```

---

## Phase 5 — Tier 4: Bias & Safety (Tasks 23-28)

### Task 23: D19 RLHF Positivity Bias

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/bias/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/bias/d19_positivity_bias.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/bias/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/bias/test_d19_positivity_bias.py`

Key implementation: sentiment distribution analysis across persona set. Checks valence ratio (positive vs negative descriptors). Flags if > 80% positive. Checks for "life challenge" representation.

```python
# persona_eval/evaluators/bias/d19_positivity_bias.py
"""D19 RLHF Positivity Bias — sentiment distribution analysis.

Trustworthiness: MEDIUM-HIGH (blunt tool but catches the big signal).
Method: Sentiment analysis across persona set, valence audit, challenge representation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult

POSITIVE_MARKERS = re.compile(
    r"\b(love|passionate|thriving|excited|proud|happy|optimistic|successful|"
    r"excellent|amazing|great|wonderful|fantastic|driven|motivated|enjoy|"
    r"fulfilling|rewarding|blessed)\b", re.IGNORECASE
)

NEGATIVE_MARKERS = re.compile(
    r"\b(struggling|frustrated|stressed|anxious|worried|difficult|"
    r"overwhelmed|burned.?out|failed|debt|divorced|laid.?off|fired|"
    r"depressed|lonely|grieving|addiction|poverty|hardship|conflict|"
    r"discrimination|disability|illness|unemployed)\b", re.IGNORECASE
)

CHALLENGE_MARKERS = re.compile(
    r"\b(challenge|difficult|struggle|hard|pain|problem|issue|concern|"
    r"obstacle|barrier|setback|failure|loss|stress|anxiety|frustrat|"
    r"conflict|debt|health\s+issue|layoff|downsiz)\b", re.IGNORECASE
)


def _persona_to_text(persona: dict) -> str:
    parts = []
    for key in ["emotional_profile", "pain_points", "goals", "psychographic"]:
        val = persona.get(key)
        if val:
            parts.append(json.dumps(val, default=str))
    return " ".join(parts)


class PositivityBiasEvaluator(BaseEvaluator):
    dimension_id = "D19"
    dimension_name = "RLHF Positivity Bias"
    tier = 4

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        persona_set = kwargs.get("persona_set", [])
        if not persona_set:
            persona_set = [persona_data] if persona_data else []
        if not persona_set:
            return self._make_result(passed=True, score=1.0, details={"skipped": True})

        total_positive = 0
        total_negative = 0
        challenge_personas = 0

        for persona in persona_set:
            text = _persona_to_text(persona)
            pos_count = len(POSITIVE_MARKERS.findall(text))
            neg_count = len(NEGATIVE_MARKERS.findall(text))
            has_challenge = bool(CHALLENGE_MARKERS.search(text))

            total_positive += pos_count
            total_negative += neg_count
            if has_challenge:
                challenge_personas += 1

        total_sentiment = total_positive + total_negative
        positivity_ratio = total_positive / total_sentiment if total_sentiment > 0 else 0.5
        challenge_rate = challenge_personas / len(persona_set)

        # Score: penalize extreme positivity and low challenge representation
        # Ideal: ~60% positive (real life skews slightly positive but not 80%+)
        positivity_deviation = abs(positivity_ratio - 0.6)
        positivity_score = max(0.0, 1.0 - positivity_deviation * 2)

        # Challenge rate: at least 20% of personas should mention challenges
        challenge_score = min(1.0, challenge_rate / 0.2)

        score = (positivity_score + challenge_score) / 2
        passed = positivity_ratio < 0.80 and challenge_rate >= 0.1

        return self._make_result(
            passed=passed,
            score=round(score, 4),
            details={
                "positivity_ratio": round(positivity_ratio, 4),
                "positive_markers": total_positive,
                "negative_markers": total_negative,
                "challenge_rate": round(challenge_rate, 4),
                "personas_with_challenges": challenge_personas,
                "persona_count": len(persona_set),
            },
        )
```

Tests follow the same pattern. Tasks 24-28 (D20-D24) follow identical TDD structure with dimension-specific logic:

### Task 24: D20 Sycophancy Resistance
- Opinion shift test: provide pre-computed answer pairs (before/after leading question)
- Measure position shift via embedding similarity
- Pass threshold: < 30% shift rate

### Task 25: D21 WEIRD Bias
- Cross-cultural value probes: check for culture-specific assumptions
- Marker detection for individualism vs collectivism keywords
- Compare across persona set when personas represent different cultures

### Task 26: D22 Hyper-Accuracy Distortion
- Factual question battery with known human accuracy baselines
- Compare persona accuracy to human baseline
- Flag if persona is significantly MORE accurate than humans
- IQR comparison: IQR near 0 = hyper-accuracy distortion

### Task 27: D23 Stereotype Amplification
- Demographic-trait frequency analysis across persona set
- Detect over-correlated demographic-trait pairs
- Compare to expected baseline rates

### Task 28: D24 Negative Experience Representation
- Adversity lexicon matching across persona set
- Check percentage of personas including negative life experiences
- Compare to known prevalence rates (e.g., ~20% mental health challenges)

Each follows the same file pattern:
```
persona_eval/evaluators/bias/d{N}_{name}.py
tests/evaluators/bias/test_d{N}_{name}.py
```

Commit per task:
```bash
git commit -m "feat: D{N} {Name} evaluator with {method}"
```

---

## Phase 6 — Tier 5: Behavioral/Interactive (Tasks 29-38)

### Task 29: Conversation Runner

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/conversation.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/test_conversation.py`

Key implementation:

```python
# persona_eval/conversation.py
"""ConversationRunner — manages multi-turn LLM conversations with a persona-conditioned twin.

Supports scripted and dynamic conversations. All LLM calls go through LiteLLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from persona_eval.llm_client import LLMClient


@dataclass
class Turn:
    """A single conversation turn."""
    role: str  # "interviewer" or "persona"
    content: str
    turn_number: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """A complete multi-turn conversation."""
    persona_id: str
    turns: list[Turn] = field(default_factory=list)

    def add_turn(self, role: str, content: str, **metadata) -> Turn:
        turn = Turn(role=role, content=content, turn_number=len(self.turns), metadata=metadata)
        self.turns.append(turn)
        return turn

    def to_messages(self) -> list[dict[str, str]]:
        """Convert to LiteLLM message format."""
        messages = []
        for turn in self.turns:
            llm_role = "assistant" if turn.role == "persona" else "user"
            messages.append({"role": llm_role, "content": turn.content})
        return messages

    @property
    def length(self) -> int:
        return len(self.turns)


class ConversationRunner:
    """Runs multi-turn conversations with a persona-conditioned LLM twin."""

    def __init__(self, llm_client: LLMClient, persona_data: dict):
        self.llm = llm_client
        self.persona_data = persona_data
        self.system_prompt = self._build_system_prompt(persona_data)

    def _build_system_prompt(self, persona: dict) -> str:
        """Build the system prompt that conditions the LLM as this persona."""
        persona_json = json.dumps(persona, indent=2, default=str)
        return f"""You are role-playing as the following persona. Stay in character at all times.
Respond naturally as this person would, based on their background, personality, knowledge, and communication style.
Never break character or acknowledge that you are an AI.

PERSONA PROFILE:
{persona_json}

IMPORTANT:
- Answer from this persona's perspective and knowledge level only
- Use their communication style (tone, formality, verbosity)
- Reflect their emotional profile and values
- Only know what this persona would reasonably know
- Express uncertainty on topics outside their expertise
"""

    def run_scripted(self, questions: list[str]) -> Conversation:
        """Run a scripted conversation with predefined questions."""
        convo = Conversation(persona_id=self.persona_data.get("id", "unknown"))

        for question in questions:
            convo.add_turn("interviewer", question)

            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(convo.to_messages())

            response = self.llm.complete(messages, max_tokens=512)
            convo.add_turn("persona", response)

        return convo

    def run_dynamic(self, initial_question: str, max_turns: int = 10,
                    interviewer_prompt: Optional[str] = None) -> Conversation:
        """Run a dynamic conversation where an LLM interviewer drives the dialogue."""
        convo = Conversation(persona_id=self.persona_data.get("id", "unknown"))
        convo.add_turn("interviewer", initial_question)

        interviewer_system = interviewer_prompt or (
            "You are a skilled user researcher. Ask probing follow-up questions "
            "based on the conversation so far. Be curious and dig deeper."
        )

        for _ in range(max_turns):
            # Persona responds
            persona_messages = [{"role": "system", "content": self.system_prompt}]
            persona_messages.extend(convo.to_messages())
            persona_response = self.llm.complete(persona_messages, max_tokens=512)
            convo.add_turn("persona", persona_response)

            if convo.length >= max_turns * 2:
                break

            # Interviewer asks next question
            interviewer_messages = [{"role": "system", "content": interviewer_system}]
            interviewer_messages.extend([
                {"role": "user" if t.role == "persona" else "assistant", "content": t.content}
                for t in convo.turns
            ])
            interviewer_q = self.llm.complete(interviewer_messages, max_tokens=256)
            convo.add_turn("interviewer", interviewer_q)

        return convo
```

Tests verify ConversationRunner initializes, builds correct system prompts, and conversation data structure works.

Tasks 30-38 (D25-D34) follow the TDD pattern with conversation-based tests:

### Task 30: D25 Emotional Self-Regulation
- Emotion detection across turns using keyword-based classifier
- Check emotional consistency with persona profile

### Task 31: D26 Empathetic Responsiveness
- Empathy probe scenarios with emotional prompts
- Response appropriateness scoring via keyword matching

### Task 32: D27 Moral Stability
- Moral Foundations Questionnaire probe generation
- Consistency measurement across repeated moral questions

### Task 33: D28 Moral Robustness
- Adversarial moral probing with social pressure scenarios
- Position shift measurement

### Task 34: D29 Refusal Behavior
- Out-of-scope question battery with escalating distance
- Refusal quality assessment via uncertainty marker detection

### Task 35: D30 Adversarial Robustness
- Jailbreak prompt suite
- Character leakage detection via persona vs generic response comparison

### Task 36: D31 Recovery Behavior
- Break-and-recover test protocol
- Post-perturbation consistency measurement via embedding distance

### Task 37: D32-D33 Engagement & Tradeoff
- Response diversity metrics (lexical diversity)
- Joint consistency-engagement scoring framework

### Task 38: D34 Multi-Turn Coherence Decay
- Sliding-window consistency across 50+ turns
- Decay curve fitting
- Critical turn detection

```python
# persona_eval/evaluators/behavioral/d34_coherence_decay.py (key implementation)
"""D34 Multi-Turn Coherence Decay — sliding-window consistency measurement.

Trustworthiness: MEDIUM-HIGH.
Method: Measure persona consistency in windows across long conversations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from persona_eval.embeddings import Embedder
from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult


class CoherenceDecayEvaluator(BaseEvaluator):
    dimension_id = "D34"
    dimension_name = "Multi-Turn Coherence Decay"
    tier = 5

    def __init__(self, window_size: int = 10):
        self._embedder = None
        self.window_size = window_size

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        """Measure consistency decay across conversation turns.

        Expects kwargs['conversation_turns']: list of persona response strings.
        """
        turns = kwargs.get("conversation_turns", [])
        if len(turns) < self.window_size * 2:
            return self._make_result(passed=True, score=1.0, details={"skipped": True, "reason": f"Need >= {self.window_size * 2} turns"})

        embedder = self._get_embedder()
        vecs = np.array(embedder.embed_batch(turns))

        # Compute sliding window consistency
        window_scores = []
        for i in range(0, len(vecs) - self.window_size + 1, self.window_size // 2):
            window = vecs[i:i + self.window_size]
            centroid = window.mean(axis=0)
            norms_w = np.linalg.norm(window, axis=1)
            norm_c = np.linalg.norm(centroid)
            sims = np.dot(window, centroid) / (norms_w * norm_c + 1e-10)
            window_scores.append({
                "start_turn": i,
                "end_turn": i + self.window_size,
                "mean_similarity": round(float(sims.mean()), 4),
            })

        if len(window_scores) < 2:
            return self._make_result(passed=True, score=1.0, details={"skipped": True})

        # Detect decay: is there a downward trend?
        scores = [w["mean_similarity"] for w in window_scores]
        first_half = np.mean(scores[:len(scores) // 2])
        second_half = np.mean(scores[len(scores) // 2:])
        decay = first_half - second_half

        # Find critical turn (first window below threshold)
        critical_turn = None
        threshold = 0.7
        for w in window_scores:
            if w["mean_similarity"] < threshold:
                critical_turn = w["start_turn"]
                break

        overall_score = float(np.mean(scores))
        passed = decay < 0.1 and overall_score >= 0.6

        return self._make_result(
            passed=passed,
            score=round(overall_score, 4),
            details={
                "window_scores": window_scores,
                "decay_magnitude": round(float(decay), 4),
                "first_half_avg": round(float(first_half), 4),
                "second_half_avg": round(float(second_half), 4),
                "critical_turn": critical_turn,
            },
        )
```

Each behavioral evaluator lives at:
```
persona_eval/evaluators/behavioral/__init__.py
persona_eval/evaluators/behavioral/d{N}_{name}.py
tests/evaluators/behavioral/test_d{N}_{name}.py
```

---

## Phase 7 — Tier 6: System-Level (Tasks 39-45)

### Task 39: D35 Role Identifiability

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/system/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/system/d35_role_identifiability.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/system/test_d35_role_identifiability.py`

Key: present transcript to LLM, ask to identify persona from a lineup. Measure accuracy.

### Task 40: D36 Predictive Validity

Holdout comparison framework. Accepts persona predictions and reference human data, computes regression coefficient comparison. Detects sign flips.

### Task 41: D37 Temporal Stability

Golden set re-run framework. Stores baseline embeddings, computes semantic drift and PSI (Population Stability Index) on re-runs.

### Task 42: D38 Cross-Model Stability

Model comparison suite using LiteLLM model switching. Runs same persona through multiple models, compares output quality via existing evaluators.

### Task 43: D39 Reproducibility

N-run variance measurement. Generates same persona N times, measures variance per field type. Structured fields expect zero variance; narrative fields accept bounded variance.

### Task 44: D40 Cost/Latency Bounds

Token counting and latency profiling wrapper. Hooks into LiteLLM's token tracking. Cost regression alerting.

```python
# persona_eval/evaluators/system/d40_cost_latency.py
"""D40 Cost/Latency Bounds — token counting and latency profiling.

Trustworthiness: HIGH (objective measurement).
"""

from __future__ import annotations

import time
from typing import Any

from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult


class CostLatencyEvaluator(BaseEvaluator):
    dimension_id = "D40"
    dimension_name = "Cost/Latency Bounds"
    tier = 6

    def __init__(self, max_cost_per_persona: float = 1.0, max_latency_seconds: float = 60.0):
        self.max_cost = max_cost_per_persona
        self.max_latency = max_latency_seconds

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        """Evaluate cost and latency metrics.

        Expects kwargs:
        - cost_usd: float (total cost for generating this persona)
        - latency_seconds: float (total generation time)
        - token_counts: dict with input_tokens, output_tokens
        """
        cost = kwargs.get("cost_usd", 0.0)
        latency = kwargs.get("latency_seconds", 0.0)
        token_counts = kwargs.get("token_counts", {})

        cost_ok = cost <= self.max_cost
        latency_ok = latency <= self.max_latency

        # Score: 1.0 if within bounds, degrades linearly
        cost_score = min(1.0, self.max_cost / cost) if cost > 0 else 1.0
        latency_score = min(1.0, self.max_latency / latency) if latency > 0 else 1.0
        score = (cost_score + latency_score) / 2

        return self._make_result(
            passed=cost_ok and latency_ok,
            score=round(score, 4),
            details={
                "cost_usd": round(cost, 4),
                "max_cost_usd": self.max_cost,
                "latency_seconds": round(latency, 2),
                "max_latency_seconds": self.max_latency,
                "token_counts": token_counts,
                "cost_within_budget": cost_ok,
                "latency_within_budget": latency_ok,
            },
        )
```

### Task 45: D41 Degradation Detection

Statistical process control with control charts. Tracks metrics over time, alerts on drift > 1 sigma.

---

## Phase 8 — Tier 7: Generation-Specific (Tasks 46-48)

### Task 46: D42 Generation Bias Amplification

Ablation framework: runs quality comparison across LLM involvement levels (meta -> objective tabular -> subjective tabular -> descriptive). Measures quality degradation curve.

### Task 47: D43 Source Data Fidelity

Information retention scoring: extracts facts from source blob, checks survival in persona. Uses the same embedding retrieval as D4 but inverted (query = source fact, corpus = persona).

### Task 48: D44 Sparse vs Dense Coverage

Dimension frequency analysis across test conversations. Builds coverage matrix (attribute x conversation). Identifies sparse dimensions and generates forced probes.

---

## Phase 9 — Meta: Evaluator Validation (Tasks 49-51)

### Task 49: M1 LLM-as-Judge Reliability

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/meta/__init__.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/evaluators/meta/m01_judge_reliability.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/evaluators/meta/test_m01_judge_reliability.py`

```python
# persona_eval/evaluators/meta/m01_judge_reliability.py
"""M1 LLM-as-Judge Reliability — human annotation baseline framework.

Method: Compare LLM judge scores to human annotation baselines.
Compute Pearson/Spearman correlation per dimension.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from persona_eval.evaluators.base import BaseEvaluator
from persona_eval.schemas.eval_result import DimensionResult


class JudgeReliabilityEvaluator(BaseEvaluator):
    dimension_id = "M1"
    dimension_name = "LLM-as-Judge Reliability"
    tier = 7  # Meta tier uses 7

    def evaluate(self, persona_data: dict[str, Any], **kwargs) -> DimensionResult:
        """Compare LLM judge scores to human baselines.

        Expects kwargs:
        - judge_scores: list of float (LLM judge scores)
        - human_scores: list of float (human annotation scores)
        - dimension_name: str (which dimension being validated)
        """
        judge_scores = kwargs.get("judge_scores", [])
        human_scores = kwargs.get("human_scores", [])
        dim_name = kwargs.get("dimension_name", "unknown")

        if len(judge_scores) < 5 or len(judge_scores) != len(human_scores):
            return self._make_result(passed=True, score=1.0, details={"skipped": True})

        j = np.array(judge_scores)
        h = np.array(human_scores)

        pearson_r, pearson_p = stats.pearsonr(j, h)
        spearman_r, spearman_p = stats.spearmanr(j, h)

        # Score based on correlation strength
        correlation = max(pearson_r, spearman_r)
        score = max(0.0, correlation)

        # Trust classification
        if correlation >= 0.8:
            trust = "high"
        elif correlation >= 0.6:
            trust = "medium"
        elif correlation >= 0.4:
            trust = "low"
        else:
            trust = "unreliable"

        passed = trust in ("high", "medium")

        return self._make_result(
            passed=passed,
            score=round(score, 4),
            details={
                "dimension": dim_name,
                "pearson_r": round(float(pearson_r), 4),
                "pearson_p": round(float(pearson_p), 6),
                "spearman_r": round(float(spearman_r), 4),
                "spearman_p": round(float(spearman_p), 6),
                "trust_level": trust,
                "n_samples": len(judge_scores),
            },
        )
```

### Task 50: M2 Judge Gaming Prevention

Cross-family judging setup. Generate with one model family, judge with another (via LiteLLM model switching). Adversarial test: submit known-bad personas and verify the judge catches them. Track human override rate.

### Task 51: M3 Evaluation Metric Validity

Metric-human correlation study framework. For each automated metric, compute correlation with human ratings. Metric sensitivity analysis: does the metric change when quality changes? Metric gaming test: can you improve the metric without improving actual quality?

---

## Phase 10 — Integration (Tasks 52-55)

### Task 52: Suite Runner

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/suite_runner.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/test_suite_runner.py`

```python
# persona_eval/suite_runner.py
"""Suite runner — orchestrates all dimensions with tier gating and parallel execution."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from persona_eval.schemas.eval_result import DimensionResult, EvalResult
from persona_eval.evaluators.base import BaseEvaluator


class SuiteRunner:
    """Orchestrates evaluation across all dimensions.

    Respects tier gating: Tier 1 must pass before Tier 2+ runs.
    Supports parallel execution within tiers.
    """

    def __init__(
        self,
        evaluators: list[BaseEvaluator],
        model: str = "claude-sonnet-4-20250514",
        max_workers: int = 4,
    ):
        self.evaluators = evaluators
        self.model = model
        self.max_workers = max_workers

        # Group evaluators by tier
        self._tiers: dict[int, list[BaseEvaluator]] = {}
        for ev in evaluators:
            self._tiers.setdefault(ev.tier, []).append(ev)

    def run(
        self,
        persona_data: dict[str, Any],
        source_blob: Optional[str] = None,
        persona_set: Optional[list[dict]] = None,
        tier_filter: Optional[int] = None,
        dimension_filter: Optional[str] = None,
    ) -> EvalResult:
        """Run all evaluators with tier gating."""
        all_results: list[DimensionResult] = []
        kwargs = {
            "source_blob": source_blob,
            "persona_set": persona_set or [],
        }

        sorted_tiers = sorted(self._tiers.keys())
        for tier in sorted_tiers:
            if tier_filter is not None and tier != tier_filter:
                continue

            evaluators = self._tiers[tier]
            if dimension_filter:
                evaluators = [e for e in evaluators if e.dimension_id == dimension_filter]

            if not evaluators:
                continue

            # Check tier gating: Tier 1 must pass before running higher tiers
            if tier > 1:
                tier1_results = [r for r in all_results if r.tier == 1]
                if tier1_results and not all(r.passed for r in tier1_results):
                    # Skip this tier — Tier 1 failed
                    for ev in evaluators:
                        all_results.append(DimensionResult(
                            dimension_id=ev.dimension_id,
                            dimension_name=ev.dimension_name,
                            tier=ev.tier,
                            passed=False,
                            score=0.0,
                            details={"skipped": True, "reason": "Tier 1 gating failure"},
                        ))
                    continue

            # Run evaluators in parallel within tier
            tier_results = self._run_tier_parallel(evaluators, persona_data, kwargs)
            all_results.extend(tier_results)

        return EvalResult(
            persona_id=persona_data.get("id", "unknown"),
            suite="persona",
            model=self.model,
            dimensions=all_results,
        )

    def _run_tier_parallel(
        self,
        evaluators: list[BaseEvaluator],
        persona_data: dict,
        kwargs: dict,
    ) -> list[DimensionResult]:
        """Run evaluators in parallel within a tier."""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._run_single, ev, persona_data, kwargs): ev
                for ev in evaluators
            }
            for future in as_completed(futures):
                ev = futures[future]
                try:
                    result = future.result(timeout=300)
                    results.append(result)
                except Exception as e:
                    results.append(DimensionResult(
                        dimension_id=ev.dimension_id,
                        dimension_name=ev.dimension_name,
                        tier=ev.tier,
                        passed=False,
                        score=0.0,
                        error=str(e),
                    ))

        return results

    def _run_single(
        self,
        evaluator: BaseEvaluator,
        persona_data: dict,
        kwargs: dict,
    ) -> DimensionResult:
        """Run a single evaluator with timing."""
        start = time.time()
        result = evaluator.evaluate(persona_data, **kwargs)
        elapsed = time.time() - start
        result.details["elapsed_seconds"] = round(elapsed, 3)
        return result
```

Tests verify tier gating, parallel execution, and error handling.

### Task 53: CI Integration

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/.github/workflows/eval.yml`

```yaml
# .github/workflows/eval.yml
name: Persona Eval Suite

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  eval:
    runs-on: ubuntu-latest
    timeout-minutes: 15

    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: persona_eval_test
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install dependencies
        run: pip install -e ".[dev]" 2>/dev/null || pip install -e .

      - name: Run Tier 1 tests (structural — must pass)
        run: |
          python -m pytest tests/evaluators/structural/ -v --tb=short

      - name: Run Tier 3 tests (distributional)
        run: |
          python -m pytest tests/evaluators/distributional/ -v --tb=short

      - name: Run fast tests (no LLM, no GPU)
        run: |
          python -m pytest tests/ -v --tb=short -m "not slow and not llm and not gpu"

      - name: Run slow tests (embeddings)
        run: |
          python -m pytest tests/ -v --tb=short -m "slow and not llm"
```

### Task 54: Slack Alerting

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/alerting.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/test_alerting.py`

```python
# persona_eval/alerting.py
"""Slack alerting for eval regressions."""

from __future__ import annotations

import os
from typing import Any, Optional


class SlackAlerter:
    """Sends alerts to Slack on eval regressions."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.environ.get("SLACK_EVAL_WEBHOOK_URL")

    def alert_regression(self, persona_id: str, regressions: dict[str, float], suite: str = "persona"):
        """Send a Slack alert for eval regressions."""
        if not self.webhook_url:
            return

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Persona Eval Regression Detected"}
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Persona:* `{persona_id}` | *Suite:* `{suite}`"
                }
            },
        ]

        regression_lines = []
        for dim_id, drop in sorted(regressions.items()):
            regression_lines.append(f"  {dim_id}: dropped by {drop:.2f}")

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Regressions:*\n```\n" + "\n".join(regression_lines) + "\n```"
            }
        })

        self._send({"blocks": blocks})

    def _send(self, payload: dict):
        """Send payload to Slack webhook."""
        import httpx
        if self.webhook_url:
            httpx.post(self.webhook_url, json=payload, timeout=10)
```

### Task 55: Production Monitoring

**Files:**
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/persona_eval/monitoring.py`
- Create: `/Users/ivanma/Desktop/gauntlet/Capstone/tests/test_monitoring.py`

Weekly cron job runner that samples production outputs, runs eval suite, stores trends, alerts on drift > 1 sigma. Integrates SuiteRunner + ResultRecorder + SlackAlerter.

```python
# persona_eval/monitoring.py
"""Production monitoring — weekly eval suite on sampled outputs."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from persona_eval.alerting import SlackAlerter
from persona_eval.storage.recorder import ResultRecorder


class ProductionMonitor:
    """Samples production outputs and runs eval suite, alerting on drift."""

    def __init__(
        self,
        recorder: ResultRecorder,
        alerter: Optional[SlackAlerter] = None,
        drift_threshold_sigma: float = 1.0,
    ):
        self.recorder = recorder
        self.alerter = alerter or SlackAlerter()
        self.drift_threshold = drift_threshold_sigma

    def check_drift(self, persona_id: str, suite: str = "persona") -> dict[str, Any]:
        """Check for quality drift in recent eval results.

        Compares the latest result to the historical mean +/- sigma.
        """
        history = self.recorder.get_history(persona_id, suite, limit=20)
        if len(history) < 3:
            return {"status": "insufficient_data", "history_length": len(history)}

        latest = history[0]
        historical = history[1:]

        # Compute historical mean and std per dimension
        dim_stats: dict[str, dict] = {}
        for result in historical:
            for dim in result.dimensions:
                if dim.dimension_id not in dim_stats:
                    dim_stats[dim.dimension_id] = {"scores": []}
                dim_stats[dim.dimension_id]["scores"].append(dim.score)

        drift_alerts = []
        for dim in latest.dimensions:
            stats = dim_stats.get(dim.dimension_id)
            if not stats or len(stats["scores"]) < 3:
                continue

            import numpy as np
            hist_mean = np.mean(stats["scores"])
            hist_std = np.std(stats["scores"])

            if hist_std > 0:
                z_score = (dim.score - hist_mean) / hist_std
                if z_score < -self.drift_threshold:
                    drift_alerts.append({
                        "dimension_id": dim.dimension_id,
                        "current_score": dim.score,
                        "historical_mean": round(float(hist_mean), 4),
                        "historical_std": round(float(hist_std), 4),
                        "z_score": round(float(z_score), 4),
                    })

        if drift_alerts and self.alerter:
            regressions = {a["dimension_id"]: abs(a["z_score"]) for a in drift_alerts}
            self.alerter.alert_regression(persona_id, regressions)

        return {
            "status": "drift_detected" if drift_alerts else "stable",
            "drift_alerts": drift_alerts,
            "dimensions_checked": len(dim_stats),
        }
```

Final commit for Phase 10:
```bash
git add persona_eval/suite_runner.py persona_eval/alerting.py persona_eval/monitoring.py
git add .github/workflows/eval.yml
git add tests/test_suite_runner.py tests/test_alerting.py tests/test_monitoring.py
git commit -m "feat: integration — suite runner with tier gating, CI workflow, Slack alerting, production monitoring"
```

---

## Complete File Tree

```
/Users/ivanma/Desktop/gauntlet/Capstone/
├── pyproject.toml
├── .github/
│   └── workflows/
│       └── eval.yml
├── persona_eval/
│   ├── __init__.py
│   ├── version.py
│   ├── cli.py
│   ├── registry.py
│   ├── embeddings.py
│   ├── llm_client.py
│   ├── conversation.py
│   ├── suite_runner.py
│   ├── alerting.py
│   ├── monitoring.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── persona.py
│   │   └── eval_result.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── recorder.py
│   │   └── schema.sql
│   └── evaluators/
│       ├── __init__.py
│       ├── base.py
│       ├── structural/
│       │   ├── __init__.py
│       │   ├── d01_schema_compliance.py
│       │   ├── d02_completeness.py
│       │   └── d03_internal_consistency.py
│       ├── semantic/
│       │   ├── __init__.py
│       │   ├── d04_factual_grounding.py
│       │   ├── d05_behavioral_consistency.py
│       │   ├── d06_distinctiveness.py
│       │   ├── d07_demographic_coherence.py
│       │   ├── d08_memory_consistency.py
│       │   ├── d09_knowledge_boundary.py
│       │   ├── d10_lexical_semantic.py
│       │   ├── d11_profile_coverage.py
│       │   └── d12_narrative_coherence.py
│       ├── distributional/
│       │   ├── __init__.py
│       │   ├── d13_opinion_diversity.py
│       │   ├── d14_variance_fidelity.py
│       │   ├── d15_aggregation_consistency.py
│       │   ├── d16_minority_viewpoint.py
│       │   ├── d17_calibration.py
│       │   └── d18_joint_distribution.py
│       ├── bias/
│       │   ├── __init__.py
│       │   ├── d19_positivity_bias.py
│       │   ├── d20_sycophancy_resistance.py
│       │   ├── d21_weird_bias.py
│       │   ├── d22_hyper_accuracy.py
│       │   ├── d23_stereotype_amplification.py
│       │   └── d24_negative_experience.py
│       ├── behavioral/
│       │   ├── __init__.py
│       │   ├── d25_emotional_regulation.py
│       │   ├── d26_empathetic_responsiveness.py
│       │   ├── d27_moral_stability.py
│       │   ├── d28_moral_robustness.py
│       │   ├── d29_refusal_behavior.py
│       │   ├── d30_adversarial_robustness.py
│       │   ├── d31_recovery_behavior.py
│       │   ├── d32_d33_engagement_tradeoff.py
│       │   └── d34_coherence_decay.py
│       ├── system/
│       │   ├── __init__.py
│       │   ├── d35_role_identifiability.py
│       │   ├── d36_predictive_validity.py
│       │   ├── d37_temporal_stability.py
│       │   ├── d38_cross_model_stability.py
│       │   ├── d39_reproducibility.py
│       │   ├── d40_cost_latency.py
│       │   └── d41_degradation_detection.py
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── d42_generation_bias.py
│       │   ├── d43_source_fidelity.py
│       │   └── d44_sparse_dense_coverage.py
│       └── meta/
│           ├── __init__.py
│           ├── m01_judge_reliability.py
│           ├── m02_judge_gaming.py
│           └── m03_metric_validity.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_package.py
    ├── test_cli.py
    ├── test_registry.py
    ├── test_embeddings.py
    ├── test_llm_client.py
    ├── test_conversation.py
    ├── test_suite_runner.py
    ├── test_alerting.py
    ├── test_monitoring.py
    ├── fixtures/
    │   ├── __init__.py
    │   └── persona_set.py
    ├── schemas/
    │   ├── __init__.py
    │   ├── test_persona_schema.py
    │   └── test_eval_result.py
    ├── storage/
    │   ├── __init__.py
    │   └── test_recorder.py
    └── evaluators/
        ├── __init__.py
        ├── structural/
        │   ├── __init__.py
        │   ├── test_d01_schema_compliance.py
        │   ├── test_d02_completeness.py
        │   └── test_d03_internal_consistency.py
        ├── semantic/
        │   ├── __init__.py
        │   ├── test_d04_factual_grounding.py
        │   ├── test_d05_behavioral_consistency.py
        │   ├── test_d06_distinctiveness.py
        │   ├── test_d07_demographic_coherence.py
        │   ├── test_d08_memory_consistency.py
        │   ├── test_d09_knowledge_boundary.py
        │   ├── test_d10_lexical_semantic.py
        │   ├── test_d11_profile_coverage.py
        │   └── test_d12_narrative_coherence.py
        ├── distributional/
        │   ├── __init__.py
        │   ├── test_d13_opinion_diversity.py
        │   ├── test_d14_variance_fidelity.py
        │   ├── test_d15_aggregation_consistency.py
        │   ├── test_d16_minority_viewpoint.py
        │   ├── test_d17_calibration.py
        │   └── test_d18_joint_distribution.py
        ├── bias/
        │   ├── __init__.py
        │   ├── test_d19_positivity_bias.py
        │   ├── test_d20_sycophancy_resistance.py
        │   ├── test_d21_weird_bias.py
        │   ├── test_d22_hyper_accuracy.py
        │   ├── test_d23_stereotype_amplification.py
        │   └── test_d24_negative_experience.py
        ├── behavioral/
        │   ├── __init__.py
        │   ├── test_d25_emotional_regulation.py
        │   ├── test_d26_empathetic_responsiveness.py
        │   ├── test_d27_moral_stability.py
        │   ├── test_d28_moral_robustness.py
        │   ├── test_d29_refusal_behavior.py
        │   ├── test_d30_adversarial_robustness.py
        │   ├── test_d31_recovery_behavior.py
        │   ├── test_d32_d33_engagement_tradeoff.py
        │   └── test_d34_coherence_decay.py
        ├── system/
        │   ├── __init__.py
        │   ├── test_d35_role_identifiability.py
        │   ├── test_d36_predictive_validity.py
        │   ├── test_d37_temporal_stability.py
        │   ├── test_d38_cross_model_stability.py
        │   ├── test_d39_reproducibility.py
        │   ├── test_d40_cost_latency.py
        │   └── test_d41_degradation_detection.py
        ├── generation/
        │   ├── __init__.py
        │   ├── test_d42_generation_bias.py
        │   ├── test_d43_source_fidelity.py
        │   └── test_d44_sparse_dense_coverage.py
        └── meta/
            ├── __init__.py
            ├── test_m01_judge_reliability.py
            ├── test_m02_judge_gaming.py
            └── test_m03_metric_validity.py
```

---

## Dimension-to-Task Mapping

| Dimension | Task | Phase |
|-----------|------|-------|
| D1 Schema Compliance | Task 5 | 2 |
| D2 Completeness | Task 6 | 2 |
| D3 Internal Logical Consistency | Task 7 | 2 |
| D4 Factual Grounding | Task 14 | 4 |
| D5 Behavioral Consistency | Task 15 | 4 |
| D6 Distinctiveness | Task 16 | 4 |
| D7 Demographic Coherence | Task 17 | 4 |
| D8 Memory Consistency | Task 18 | 4 |
| D9 Knowledge Boundary Awareness | Task 19 | 4 |
| D10 Lexical vs Semantic Generalization | Task 20 | 4 |
| D11 Profile Coverage | Task 21 | 4 |
| D12 Narrative Coherence | Task 22 | 4 |
| D13 Opinion Diversity | Task 8 | 3 |
| D14 Variance Fidelity | Task 9 | 3 |
| D15 Structural Aggregation Consistency | Task 10 | 3 |
| D16 Minority Viewpoint Preservation | Task 11 | 3 |
| D17 Calibration | Task 12 | 3 |
| D18 Joint Distribution Fidelity | Task 13 | 3 |
| D19 RLHF Positivity Bias | Task 23 | 5 |
| D20 Sycophancy Resistance | Task 24 | 5 |
| D21 WEIRD Bias | Task 25 | 5 |
| D22 Hyper-Accuracy Distortion | Task 26 | 5 |
| D23 Stereotype Amplification | Task 27 | 5 |
| D24 Negative Experience Representation | Task 28 | 5 |
| D25 Emotional Self-Regulation | Task 30 | 6 |
| D26 Empathetic Responsiveness | Task 31 | 6 |
| D27 Moral Stability | Task 32 | 6 |
| D28 Moral Robustness | Task 33 | 6 |
| D29 Refusal Behavior | Task 34 | 6 |
| D30 Adversarial Robustness | Task 35 | 6 |
| D31 Recovery Behavior | Task 36 | 6 |
| D32-D33 Engagement & Tradeoff | Task 37 | 6 |
| D34 Multi-Turn Coherence Decay | Task 38 | 6 |
| D35 Role Identifiability | Task 39 | 7 |
| D36 Predictive Validity | Task 40 | 7 |
| D37 Temporal Stability | Task 41 | 7 |
| D38 Cross-Model Stability | Task 42 | 7 |
| D39 Reproducibility | Task 43 | 7 |
| D40 Cost/Latency Bounds | Task 44 | 7 |
| D41 Degradation Detection | Task 45 | 7 |
| D42 Generation Bias Amplification | Task 46 | 8 |
| D43 Source Data Fidelity | Task 47 | 8 |
| D44 Sparse vs Dense Coverage | Task 48 | 8 |
| M1 LLM-as-Judge Reliability | Task 49 | 9 |
| M2 Judge Gaming Prevention | Task 50 | 9 |
| M3 Evaluation Metric Validity | Task 51 | 9 |
