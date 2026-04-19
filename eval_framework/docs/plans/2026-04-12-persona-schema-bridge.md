# Persona Schema Bridge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend PersonaV1 (synthesis pipeline output) to capture psychological depth — `communication_style`, `emotional_profile`, `moral_framework` — grounded in source records, and write a trivial PersonaV1 → `persona_eval.Persona` adapter so synthesized personas feed directly into the behavioral eval suite.

**Architecture:**
The synthesis LLM already sees rich psychological signal in Intercom/GA4 records (verbatim frustration, values language, technical vs business register). Today it extracts `vocabulary` and `sample_quotes` from that signal but throws away the rest. We extend the synthesis schema and prompt to also extract three structured sub-schemas that mirror `persona_eval.Persona`'s psychological fields. Once PersonaV1 carries this data, the bridge to `persona_eval.Persona` becomes a pure field-mapping adapter — no second LLM call, no fabrication. Psychological fields are grounded via the existing `source_evidence` mechanism, so the groundedness checker enforces traceability.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, pytest-asyncio. No new dependencies.

**Scope constraints:**
- Do **not** change `persona_eval.Persona` schema — it is the stable target.
- Do **not** add a second LLM enrichment call — all psychological extraction happens in the existing synthesis call.
- Regenerate existing persona JSON fixtures (`output/persona_00.json`, `output/persona_01.json`) — they are mock outputs, not production data.
- Keep `schema_version = "1.0"`. This is additive within v1 since the project is pre-production.

**Key file anchors (read before starting):**
- `personas-pipeline/synthesis/synthesis/models/persona.py` — PersonaV1 schema
- `personas-pipeline/synthesis/synthesis/models/evidence.py` — SourceEvidence schema
- `personas-pipeline/synthesis/synthesis/engine/prompt_builder.py` — synthesis system prompt + user message builder
- `personas-pipeline/synthesis/synthesis/engine/groundedness.py` — evidence coverage checker
- `personas-pipeline/synthesis/synthesis/engine/synthesizer.py` — orchestrator (no changes needed, just read to understand flow)
- `persona_eval/schemas.py` — target schema (`Persona`, `CommunicationStyle`, `EmotionalProfile`, `MoralFramework`)
- `personas-pipeline/output/persona_00.json` — example v1 output (will be regenerated)
- `tests/golden_dataset.py` — reference for what fully-populated eval.Persona looks like

---

## Task 1: Add `CommunicationStyle`, `EmotionalProfile`, `MoralFramework` sub-schemas

**Files:**
- Modify: `personas-pipeline/synthesis/synthesis/models/persona.py`
- Create: `personas-pipeline/synthesis/tests/__init__.py` (empty)
- Create: `personas-pipeline/synthesis/tests/test_persona_psychological.py`

**Step 1: Write failing tests for the three new sub-schemas**

Create `personas-pipeline/synthesis/tests/test_persona_psychological.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from synthesis.models.persona import (
    CommunicationStyle,
    EmotionalProfile,
    MoralFramework,
)


class TestCommunicationStyle:
    def test_valid(self) -> None:
        cs = CommunicationStyle(
            tone="direct",
            formality="professional",
            vocabulary_level="advanced",
            preferred_channels=["Slack", "email"],
        )
        assert cs.tone == "direct"
        assert cs.vocabulary_level == "advanced"

    def test_requires_at_least_one_channel(self) -> None:
        with pytest.raises(ValidationError):
            CommunicationStyle(
                tone="direct",
                formality="professional",
                vocabulary_level="advanced",
                preferred_channels=[],
            )


class TestEmotionalProfile:
    def test_valid(self) -> None:
        ep = EmotionalProfile(
            baseline_mood="calm",
            stress_triggers=["production outages"],
            coping_mechanisms=["deep work blocks"],
        )
        assert ep.baseline_mood == "calm"

    def test_requires_at_least_one_stress_trigger(self) -> None:
        with pytest.raises(ValidationError):
            EmotionalProfile(
                baseline_mood="calm",
                stress_triggers=[],
                coping_mechanisms=["deep work blocks"],
            )

    def test_requires_at_least_one_coping_mechanism(self) -> None:
        with pytest.raises(ValidationError):
            EmotionalProfile(
                baseline_mood="calm",
                stress_triggers=["production outages"],
                coping_mechanisms=[],
            )


class TestMoralFramework:
    def test_valid(self) -> None:
        mf = MoralFramework(
            core_values=["fairness", "honesty"],
            ethical_stance="utilitarian",
            moral_foundations={"care": 0.7, "fairness": 0.9},
        )
        assert mf.ethical_stance == "utilitarian"

    def test_requires_at_least_two_core_values(self) -> None:
        with pytest.raises(ValidationError):
            MoralFramework(
                core_values=["fairness"],
                ethical_stance="utilitarian",
                moral_foundations={"care": 0.7},
            )

    def test_moral_foundations_weights_must_be_in_range(self) -> None:
        with pytest.raises(ValidationError):
            MoralFramework(
                core_values=["fairness", "honesty"],
                ethical_stance="utilitarian",
                moral_foundations={"care": 1.5},  # > 1.0
            )
```

**Step 2: Run tests to verify they fail**

Run: `cd personas-pipeline/synthesis && pytest tests/test_persona_psychological.py -v`
Expected: `ImportError: cannot import name 'CommunicationStyle'` (or similar — symbols don't exist yet).

**Step 3: Add the three sub-schemas to `persona.py`**

In `personas-pipeline/synthesis/synthesis/models/persona.py`, insert these classes **after `JourneyStage`** and **before `PersonaV1`**:

```python
class CommunicationStyle(BaseModel):
    """How this persona expresses themselves — matches persona_eval.CommunicationStyle."""

    tone: str = Field(
        description="Dominant emotional register, e.g. 'direct', 'warm', 'enthusiastic', 'analytical', 'skeptical'",
    )
    formality: str = Field(
        description="e.g. 'casual', 'professional', 'formal'",
    )
    vocabulary_level: str = Field(
        description="'basic', 'intermediate', or 'advanced' — based on technical sophistication of their language",
    )
    preferred_channels: list[str] = Field(
        min_length=1,
        description="Channels where this persona actually communicates (Slack, email, Intercom, forums, etc.)",
    )


class EmotionalProfile(BaseModel):
    """Emotional baseline + triggers + coping — matches persona_eval.EmotionalProfile."""

    baseline_mood: str = Field(
        description="Dominant emotional baseline — 'calm', 'anxious', 'optimistic', 'frustrated', 'enthusiastic', etc.",
    )
    stress_triggers: list[str] = Field(
        min_length=1,
        max_length=6,
        description="What reliably makes this persona stressed or frustrated, grounded in source records",
    )
    coping_mechanisms: list[str] = Field(
        min_length=1,
        max_length=6,
        description="How this persona handles frustration — 'files support ticket', 'writes automation', 'vents on Twitter', etc.",
    )


class MoralFramework(BaseModel):
    """Values and ethical stance — matches persona_eval.MoralFramework."""

    core_values: list[str] = Field(
        min_length=2,
        max_length=6,
        description="The values this persona treats as non-negotiable — 'fairness', 'autonomy', 'efficiency', etc.",
    )
    ethical_stance: str = Field(
        description="One of 'utilitarian', 'virtue ethics', 'deontological', 'principlist', 'care ethics'",
    )
    moral_foundations: dict[str, float] = Field(
        description=(
            "Moral Foundations Theory weights in [0.0, 1.0]. "
            "Keys: care, fairness, loyalty, authority, sanctity, liberty. "
            "Not all keys required — include only those with clear evidence."
        ),
    )
```

Add a `field_validator` at the bottom of `MoralFramework` (inside the class) to enforce the 0.0-1.0 range on dict values:

```python
    @field_validator("moral_foundations")
    @classmethod
    def _weights_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        for k, weight in v.items():
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"moral_foundations[{k}] = {weight} must be in [0.0, 1.0]")
        return v
```

Also add `field_validator` to the import at the top: `from pydantic import BaseModel, Field, field_validator`.

**Step 4: Run tests to verify they pass**

Run: `cd personas-pipeline/synthesis && pytest tests/test_persona_psychological.py -v`
Expected: 7 passed.

**Step 5: Commit**

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone
git add personas-pipeline/synthesis/synthesis/models/persona.py \
        personas-pipeline/synthesis/tests/__init__.py \
        personas-pipeline/synthesis/tests/test_persona_psychological.py
git commit -m "feat: add psychological sub-schemas to PersonaV1 models

CommunicationStyle, EmotionalProfile, MoralFramework mirror the shapes
in persona_eval.schemas so PersonaV1 can carry psychological depth
directly, bridging to the eval suite without LLM enrichment."
```

---

## Task 2: Add psychological fields to `PersonaV1`

**Files:**
- Modify: `personas-pipeline/synthesis/synthesis/models/persona.py`
- Create: `personas-pipeline/synthesis/tests/test_persona_v1.py`

**Step 1: Write failing test**

Create `personas-pipeline/synthesis/tests/test_persona_v1.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from synthesis.models.persona import (
    CommunicationStyle,
    Demographics,
    EmotionalProfile,
    Firmographics,
    JourneyStage,
    MoralFramework,
    PersonaV1,
)
from synthesis.models.evidence import SourceEvidence


def _minimal_persona_kwargs() -> dict:
    """Return kwargs sufficient to build a valid PersonaV1."""
    return dict(
        name="Test Persona",
        summary="A test persona for schema validation.",
        demographics=Demographics(
            age_range="25-34",
            gender_distribution="mixed",
            location_signals=["US"],
        ),
        firmographics=Firmographics(),
        goals=["goal one", "goal two"],
        pains=["pain one", "pain two"],
        motivations=["motivation one", "motivation two"],
        objections=["objection one"],
        channels=["Slack"],
        vocabulary=["alpha", "beta", "gamma"],
        decision_triggers=["trigger one"],
        sample_quotes=["quote one", "quote two"],
        journey_stages=[
            JourneyStage(
                stage="evaluation",
                mindset="skeptical",
                key_actions=["reads docs"],
                content_preferences=["API reference"],
            ),
            JourneyStage(
                stage="activation",
                mindset="building",
                key_actions=["first integration"],
                content_preferences=["code samples"],
            ),
        ],
        communication_style=CommunicationStyle(
            tone="direct",
            formality="professional",
            vocabulary_level="advanced",
            preferred_channels=["Slack"],
        ),
        emotional_profile=EmotionalProfile(
            baseline_mood="calm",
            stress_triggers=["outages"],
            coping_mechanisms=["automation"],
        ),
        moral_framework=MoralFramework(
            core_values=["efficiency", "fairness"],
            ethical_stance="utilitarian",
            moral_foundations={"care": 0.5, "fairness": 0.8},
        ),
        source_evidence=[
            SourceEvidence(claim="c1", record_ids=["r1"], field_path="goals.0", confidence=0.9),
            SourceEvidence(claim="c2", record_ids=["r1"], field_path="pains.0", confidence=0.9),
            SourceEvidence(claim="c3", record_ids=["r1"], field_path="motivations.0", confidence=0.9),
        ],
    )


class TestPersonaV1Psychological:
    def test_valid_with_psychological_fields(self) -> None:
        p = PersonaV1(**_minimal_persona_kwargs())
        assert p.communication_style.tone == "direct"
        assert p.emotional_profile.baseline_mood == "calm"
        assert p.moral_framework.ethical_stance == "utilitarian"

    def test_communication_style_is_required(self) -> None:
        kwargs = _minimal_persona_kwargs()
        del kwargs["communication_style"]
        with pytest.raises(ValidationError):
            PersonaV1(**kwargs)

    def test_emotional_profile_is_required(self) -> None:
        kwargs = _minimal_persona_kwargs()
        del kwargs["emotional_profile"]
        with pytest.raises(ValidationError):
            PersonaV1(**kwargs)

    def test_moral_framework_is_required(self) -> None:
        kwargs = _minimal_persona_kwargs()
        del kwargs["moral_framework"]
        with pytest.raises(ValidationError):
            PersonaV1(**kwargs)
```

**Step 2: Run test to verify it fails**

Run: `cd personas-pipeline/synthesis && pytest tests/test_persona_v1.py -v`
Expected: 4 failures — "missing field 'communication_style'" etc.

**Step 3: Add the three fields to `PersonaV1`**

In `personas-pipeline/synthesis/synthesis/models/persona.py`, add these three fields to `PersonaV1` **after `journey_stages`** and **before `source_evidence`**:

```python
    communication_style: CommunicationStyle = Field(
        description="How this persona speaks and writes. Derived from their verbatim messages in the source data.",
    )
    emotional_profile: EmotionalProfile = Field(
        description="Emotional baseline, stress triggers, and coping mechanisms. Grounded in support tickets, complaints, and behavioral signals.",
    )
    moral_framework: MoralFramework = Field(
        description="Core values and ethical stance. Inferred from what this persona cares about — language of fairness, autonomy, efficiency, etc.",
    )
```

**Step 4: Run test to verify it passes**

Run: `cd personas-pipeline/synthesis && pytest tests/test_persona_v1.py -v`
Expected: 4 passed.

**Step 5: Commit**

```bash
git add personas-pipeline/synthesis/synthesis/models/persona.py \
        personas-pipeline/synthesis/tests/test_persona_v1.py
git commit -m "feat: require communication_style, emotional_profile, moral_framework on PersonaV1

PersonaV1 now carries psychological depth directly. Existing JSON
fixtures under output/ will fail validation until regenerated in a
later step of this plan."
```

---

## Task 3: Update synthesis system prompt

**Files:**
- Modify: `personas-pipeline/synthesis/synthesis/engine/prompt_builder.py`
- Create: `personas-pipeline/synthesis/tests/test_prompt_builder.py`

**Step 1: Write failing test**

Create `personas-pipeline/synthesis/tests/test_prompt_builder.py`:

```python
from __future__ import annotations

from synthesis.engine.prompt_builder import SYSTEM_PROMPT


class TestSystemPrompt:
    def test_mentions_communication_style(self) -> None:
        assert "communication_style" in SYSTEM_PROMPT

    def test_mentions_emotional_profile(self) -> None:
        assert "emotional_profile" in SYSTEM_PROMPT

    def test_mentions_moral_framework(self) -> None:
        assert "moral_framework" in SYSTEM_PROMPT

    def test_requires_evidence_for_psychological_fields(self) -> None:
        # The prompt must instruct the LLM to cite evidence for psychological claims.
        assert "moral_framework.core_values" in SYSTEM_PROMPT or \
               "moral_framework.ethical_stance" in SYSTEM_PROMPT
        assert "emotional_profile.stress_triggers" in SYSTEM_PROMPT or \
               "emotional_profile.baseline_mood" in SYSTEM_PROMPT

    def test_warns_against_fabrication(self) -> None:
        # Prompt must explicitly say not to fabricate when evidence is thin.
        lower = SYSTEM_PROMPT.lower()
        assert "do not fabricate" in lower or "do not invent" in lower or "must be grounded" in lower
```

**Step 2: Run test to verify it fails**

Run: `cd personas-pipeline/synthesis && pytest tests/test_prompt_builder.py -v`
Expected: 5 failures.

**Step 3: Update `SYSTEM_PROMPT`**

In `personas-pipeline/synthesis/synthesis/engine/prompt_builder.py`, replace the current `SYSTEM_PROMPT` string with:

```python
SYSTEM_PROMPT = """\
You are a persona synthesis expert. Your job is to analyze behavioral data from a \
customer cluster and produce a single, richly detailed persona that a product marketer, \
a data scientist, AND a behavioral researcher would all trust.

Quality criteria:
- **Grounded**: Every claim must trace back to specific source records. Use the \
record IDs provided in the data to populate source_evidence entries. Do not fabricate \
psychological traits when evidence is thin — cite lower confidence instead.
- **Distinctive**: The persona should feel like a real individual, not a generic \
average. Use specific vocabulary, concrete quotes, and sharp motivations.
- **Actionable**: Goals, pains, and objections should be specific enough to inform \
product and marketing decisions.
- **Consistent**: Demographics, firmographics, vocabulary, quotes, communication \
style, emotional profile, and moral framework should all describe the same coherent person. \
A "calm, analytical" persona should not also have "panics under pressure" as a coping mechanism.

Psychological extraction rules:
- **communication_style** — Infer tone, formality, and vocabulary_level from verbatim \
support messages and the technical register of their language. An engineer writing \
"idempotent" and "schema drift" has advanced vocabulary_level; a user writing "the \
thing doesn't work" has basic. `preferred_channels` must reflect where this cluster \
actually communicates (Intercom messages → support channels; heavy forum signal → forums).
- **emotional_profile** — `baseline_mood` is the dominant tone across their messages \
(frustrated, optimistic, neutral, anxious). `stress_triggers` are the specific situations \
that produce complaint messages or long error-filled sessions. `coping_mechanisms` are \
the observable behaviors they take when stressed (file support tickets, write automation, \
switch tools, escalate to CSM).
- **moral_framework** — `core_values` are what they repeatedly advocate for in their own \
words (fairness, autonomy, efficiency, transparency, community). `ethical_stance` is your \
best-fit classification given their value language. `moral_foundations` weights the six \
MFT foundations (care, fairness, loyalty, authority, sanctity, liberty) in [0.0, 1.0] — \
only include foundations with clear evidence; omit rather than guess.

Evidence rules:
- Each entry in source_evidence must reference at least one record_id from the \
provided sample records.
- The field_path must use dot notation pointing to the persona field the evidence \
supports (e.g. "goals.0", "pains.2", "motivations.1", "communication_style.tone", \
"emotional_profile.stress_triggers.0", "moral_framework.core_values.1", \
"moral_framework.ethical_stance").
- Every item in goals, pains, motivations, and objections MUST have a corresponding \
source_evidence entry.
- Each of communication_style, emotional_profile, and moral_framework MUST have at \
least one source_evidence entry with a field_path rooted in that sub-object.
- Confidence should reflect how directly the data supports the claim (1.0 = verbatim \
from data, 0.5 = reasonable inference). For psychological claims, prefer lower \
confidence over false precision.

Example source_evidence entries:
{
  "claim": "Wants to reduce manual data entry by 50%",
  "record_ids": ["rec_0042", "rec_0087"],
  "field_path": "goals.0",
  "confidence": 0.85
}
{
  "claim": "Baseline mood is frustrated — multiple Intercom messages with complaint language",
  "record_ids": ["intercom_003", "intercom_007"],
  "field_path": "emotional_profile.baseline_mood",
  "confidence": 0.8
}
{
  "claim": "Core value of efficiency — repeatedly advocates for automation over manual UI",
  "record_ids": ["intercom_000", "ga4_003"],
  "field_path": "moral_framework.core_values.0",
  "confidence": 0.85
}
"""
```

**Step 4: Run test to verify it passes**

Run: `cd personas-pipeline/synthesis && pytest tests/test_prompt_builder.py -v`
Expected: 5 passed.

**Step 5: Commit**

```bash
git add personas-pipeline/synthesis/synthesis/engine/prompt_builder.py \
        personas-pipeline/synthesis/tests/test_prompt_builder.py
git commit -m "feat: extend synthesis prompt with psychological extraction rules

Teaches the LLM to extract communication_style, emotional_profile, and
moral_framework from source records, with evidence requirements and
examples. Includes explicit anti-fabrication guidance."
```

---

## Task 4: Update groundedness checker to cover psychological fields

**Files:**
- Modify: `personas-pipeline/synthesis/synthesis/engine/groundedness.py`
- Create: `personas-pipeline/synthesis/tests/test_groundedness.py`

**Context:** Today `check_groundedness` requires per-item evidence for `goals`, `pains`, `motivations`, `objections`. We add a group-level requirement: each of `communication_style`, `emotional_profile`, `moral_framework` must have at least one `source_evidence` entry whose `field_path` starts with that prefix.

**Step 1: Write failing test**

Create `personas-pipeline/synthesis/tests/test_groundedness.py`:

```python
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
    """Build evidence covering every required field + one per psych sub-schema."""
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

    def test_missing_communication_style_evidence_fails(self) -> None:
        cluster = _cluster(["r1"])
        ev = [e for e in _full_evidence() if not e.field_path.startswith("communication_style")]
        persona = _persona(ev)
        report = check_groundedness(persona, cluster)
        assert not report.passed
        assert any("communication_style" in v for v in report.violations)

    def test_missing_emotional_profile_evidence_fails(self) -> None:
        cluster = _cluster(["r1"])
        ev = [e for e in _full_evidence() if not e.field_path.startswith("emotional_profile")]
        persona = _persona(ev)
        report = check_groundedness(persona, cluster)
        assert not report.passed
        assert any("emotional_profile" in v for v in report.violations)

    def test_missing_moral_framework_evidence_fails(self) -> None:
        cluster = _cluster(["r1"])
        ev = [e for e in _full_evidence() if not e.field_path.startswith("moral_framework")]
        persona = _persona(ev)
        report = check_groundedness(persona, cluster)
        assert not report.passed
        assert any("moral_framework" in v for v in report.violations)
```

**Step 2: Run tests to verify they fail**

Run: `cd personas-pipeline/synthesis && pytest tests/test_groundedness.py -v`
Expected: The three "missing" tests FAIL (because current checker doesn't enforce psychological coverage). The "full_evidence_passes" test should PASS already.

**Step 3: Update `groundedness.py`**

In `personas-pipeline/synthesis/synthesis/engine/groundedness.py`:

Add after `EVIDENCE_REQUIRED_FIELDS`:

```python
# Sub-objects that require at least one evidence entry with a field_path rooted in them
PSYCHOLOGICAL_REQUIRED_PREFIXES = (
    "communication_style",
    "emotional_profile",
    "moral_framework",
)
```

Update `check_groundedness`: after the per-item coverage loop (after the `for field_name in EVIDENCE_REQUIRED_FIELDS:` block), before the score calculation, add:

```python
    # Check 3: Each psychological sub-object must have at least one valid evidence entry
    for prefix in PSYCHOLOGICAL_REQUIRED_PREFIXES:
        total_required += 1
        has_valid_evidence = any(
            path == prefix or path.startswith(f"{prefix}.")
            for path in valid_evidence_paths
        )
        if has_valid_evidence:
            covered += 1
        else:
            violations.append(
                f"No valid source_evidence entry rooted in {prefix!r} — "
                f"psychological fields must be grounded in source records"
            )
```

**Step 4: Run tests to verify they pass**

Run: `cd personas-pipeline/synthesis && pytest tests/test_groundedness.py -v`
Expected: 4 passed.

**Step 5: Commit**

```bash
git add personas-pipeline/synthesis/synthesis/engine/groundedness.py \
        personas-pipeline/synthesis/tests/test_groundedness.py
git commit -m "feat: require source evidence for psychological sub-schemas

Groundedness checker now rejects personas that lack at least one
evidence entry rooted in communication_style, emotional_profile, or
moral_framework. Prevents the LLM from fabricating psychological
depth without citing source records."
```

---

## Task 5: Create adapter package structure

**Files:**
- Create: `personas-pipeline/synthesis/synthesis/adapters/__init__.py` (empty)
- Create: `personas-pipeline/synthesis/synthesis/adapters/eval_adapter.py` (skeleton)
- Create: `personas-pipeline/synthesis/tests/test_eval_adapter.py` (empty scaffold)

**Step 1: Create the package skeleton**

Create `personas-pipeline/synthesis/synthesis/adapters/__init__.py` (empty file).

Create `personas-pipeline/synthesis/synthesis/adapters/eval_adapter.py`:

```python
"""Convert PersonaV1 (synthesis output) → persona_eval.Persona (eval input).

This is a pure field-mapping adapter — no LLM calls, no synthesis.
It assumes PersonaV1 already carries psychological depth (added in
the schema bridge plan, 2026-04-12).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from persona_eval.schemas import Persona as EvalPersona
    from synthesis.models.persona import PersonaV1


def persona_v1_to_eval(persona: "PersonaV1", persona_id: str) -> "EvalPersona":
    """Convert a synthesis PersonaV1 to a persona_eval.Persona.

    Args:
        persona: the synthesized PersonaV1.
        persona_id: stable id for the eval persona (e.g. cluster_id or slug).

    Returns:
        A populated persona_eval.Persona ready to feed into SuiteRunner.
    """
    raise NotImplementedError  # filled in by Tasks 6–8
```

**Step 2: Confirm persona_eval is importable from synthesis**

Run: `cd personas-pipeline/synthesis && python -c "from persona_eval.schemas import Persona; print(Persona.__name__)"`
Expected output: `Persona`.

If it fails (ModuleNotFoundError): the repo root needs to be on the path. Add an editable install or adjust `pyproject.toml`:
- Open `personas-pipeline/synthesis/pyproject.toml`
- Under `[project]` `dependencies`, add the repo-root editable dep by running from repo root: `pip install -e /Users/ivanma/Desktop/gauntlet/Capstone` (assumes `persona_eval` is installable from the repo root's `pyproject.toml`).
- If `persona_eval` has no pyproject, add this line to `personas-pipeline/synthesis/pyproject.toml` `[tool.setuptools.packages.find]` or similar, OR simply document the PYTHONPATH requirement and proceed.

**If import still fails**, stop and ask the user how they'd like to structure the cross-package import. Do not guess.

**Step 3: Commit scaffold**

```bash
git add personas-pipeline/synthesis/synthesis/adapters/
git commit -m "chore: scaffold PersonaV1 → eval.Persona adapter package"
```

---

## Task 6: Implement adapter — identity & demographic flattening

**Files:**
- Modify: `personas-pipeline/synthesis/synthesis/adapters/eval_adapter.py`
- Create: `personas-pipeline/synthesis/tests/test_eval_adapter.py`

**Context on flattening rules:**
- `age_range` "28-38" → `age` = 33 (midpoint, rounded). If unparsable, leave `None` and store raw in `extra["age_range"]`.
- `gender_distribution` "predominantly male" → `gender` = "male" (strip `predominantly `, `mostly `, `mainly `). If the result is "mixed" or "balanced", leave `gender=""`. Always preserve raw in `extra["gender_distribution"]`.
- `location_signals` `[...]` → `location` = first element (or `""`). Preserve full list in `extra["location_signals"]`.
- `education_level` → `education` (direct).
- `income_bracket` → `income_bracket` (direct).
- `firmographics.role_titles[0]` → `occupation` (or `""`).
- `firmographics.industry` → `industry`.
- `firmographics.company_size` → `extra["company_size"]`.
- `firmographics.tech_stack_signals` → `knowledge_domains`.

**Step 1: Write failing test**

Create `personas-pipeline/synthesis/tests/test_eval_adapter.py`:

```python
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
        sample_quotes=["If it's not in Terraform it doesn't exist", "I don't want to click anything"],
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
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="clust_abc")
        assert out.id == "clust_abc"
        assert out.name == "Alex the DevOps Engineer"

    def test_summary_becomes_bio(self) -> None:
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="clust_abc")
        assert out.bio == "Senior DevOps engineer at a fintech SMB."

    def test_age_range_midpoint(self) -> None:
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="clust_abc")
        assert out.age == 33  # midpoint of 28-38
        assert out.extra["age_range"] == "28-38"

    def test_gender_strip_predominantly(self) -> None:
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="clust_abc")
        assert out.gender == "male"
        assert out.extra["gender_distribution"] == "predominantly male"

    def test_gender_mixed_stays_empty(self) -> None:
        p = _fully_populated_persona()
        p.demographics.gender_distribution = "mixed"
        out = persona_v1_to_eval(p, persona_id="clust_abc")
        assert out.gender == ""
        assert out.extra["gender_distribution"] == "mixed"

    def test_location_takes_first_signal(self) -> None:
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="clust_abc")
        assert out.location == "US or EU tech hub"
        assert out.extra["location_signals"] == ["US or EU tech hub"]

    def test_firmographic_flatten(self) -> None:
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="clust_abc")
        assert out.occupation == "Senior DevOps Engineer"
        assert out.industry == "Fintech"
        assert out.extra["company_size"] == "50-200 employees"
        assert "Terraform" in out.knowledge_domains

    def test_education_and_income_direct_copy(self) -> None:
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="clust_abc")
        assert out.education == "MS Computer Science"
        assert out.income_bracket == "$120k-$180k"

    def test_age_range_unparsable_fallback(self) -> None:
        p = _fully_populated_persona()
        p.demographics.age_range = "unknown"
        out = persona_v1_to_eval(p, persona_id="clust_abc")
        assert out.age is None
        assert out.extra["age_range"] == "unknown"
```

**Step 2: Run test to verify it fails**

Run: `cd personas-pipeline/synthesis && pytest tests/test_eval_adapter.py -v`
Expected: NotImplementedError or similar — 9 fails.

**Step 3: Implement identity + demographics**

Replace `personas-pipeline/synthesis/synthesis/adapters/eval_adapter.py` with:

```python
"""Convert PersonaV1 (synthesis output) → persona_eval.Persona (eval input)."""
from __future__ import annotations

import re

from persona_eval.schemas import (
    CommunicationStyle as EvalCommunicationStyle,
    EmotionalProfile as EvalEmotionalProfile,
    MoralFramework as EvalMoralFramework,
    Persona as EvalPersona,
)

from synthesis.models.persona import PersonaV1

_AGE_RANGE_RE = re.compile(r"^\s*(\d{1,3})\s*[-–]\s*(\d{1,3})\s*$")
_GENDER_PREFIX_RE = re.compile(r"^(predominantly|mostly|mainly)\s+", re.IGNORECASE)
_GENDER_MIXED = {"mixed", "balanced", "diverse", "varied"}


def _age_midpoint(age_range: str) -> int | None:
    m = _AGE_RANGE_RE.match(age_range)
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    return round((lo + hi) / 2)


def _normalize_gender(gender_distribution: str) -> str:
    stripped = _GENDER_PREFIX_RE.sub("", gender_distribution).strip().lower()
    if stripped in _GENDER_MIXED or stripped == "":
        return ""
    return stripped


def persona_v1_to_eval(persona: PersonaV1, persona_id: str) -> EvalPersona:
    """Convert a synthesis PersonaV1 to a persona_eval.Persona."""
    demo = persona.demographics
    firm = persona.firmographics

    extra: dict = {
        "age_range": demo.age_range,
        "gender_distribution": demo.gender_distribution,
        "location_signals": list(demo.location_signals),
    }
    if firm.company_size is not None:
        extra["company_size"] = firm.company_size

    return EvalPersona(
        id=persona_id,
        name=persona.name,
        bio=persona.summary,
        age=_age_midpoint(demo.age_range),
        gender=_normalize_gender(demo.gender_distribution),
        location=demo.location_signals[0] if demo.location_signals else "",
        education=demo.education_level or "",
        income_bracket=demo.income_bracket or "",
        occupation=firm.role_titles[0] if firm.role_titles else "",
        industry=firm.industry or "",
        knowledge_domains=list(firm.tech_stack_signals),
        extra=extra,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd personas-pipeline/synthesis && pytest tests/test_eval_adapter.py::TestAdapterIdentityAndDemographics -v`
Expected: 9 passed.

**Step 5: Commit**

```bash
git add personas-pipeline/synthesis/synthesis/adapters/eval_adapter.py \
        personas-pipeline/synthesis/tests/test_eval_adapter.py
git commit -m "feat: adapter — identity + demographic flattening

Maps PersonaV1 identity, demographics, and firmographics to
eval.Persona flat fields. Preserves raw aggregates in extra dict."
```

---

## Task 7: Implement adapter — behavioral lists & psychological copy

**Context on mapping:**
- `goals` → `goals` (direct copy)
- `motivations` → `motivations` (direct copy)
- `pains` → `pain_points` (rename)
- `objections` → `frustrations` (best semantic match)
- `communication_style` → `communication_style` (direct copy — same shape)
- `emotional_profile` → `emotional_profile` (direct copy)
- `moral_framework` → `moral_framework` (direct copy)
- `moral_framework.core_values` → also mirror into `values` (eval schema has both)
- `vocabulary_level` informs no extra mapping (already inside communication_style)

**Step 1: Append failing tests to `test_eval_adapter.py`**

Append to `personas-pipeline/synthesis/tests/test_eval_adapter.py`:

```python
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
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="c1")
        assert out.communication_style.tone == "direct"
        assert out.communication_style.formality == "professional"
        assert out.communication_style.vocabulary_level == "advanced"
        assert out.communication_style.preferred_channels == ["Intercom", "GitHub"]

    def test_emotional_profile_round_trip(self) -> None:
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="c1")
        assert out.emotional_profile.baseline_mood == "pragmatic"
        assert out.emotional_profile.stress_triggers == ["schema drift", "silent webhook drops"]
        assert out.emotional_profile.coping_mechanisms == ["writes automation", "files detailed tickets"]

    def test_moral_framework_round_trip(self) -> None:
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="c1")
        assert out.moral_framework.core_values == ["efficiency", "reliability"]
        assert out.moral_framework.ethical_stance == "utilitarian"
        assert out.moral_framework.moral_foundations == {"care": 0.4, "fairness": 0.7, "liberty": 0.8}

    def test_core_values_mirrored_into_values(self) -> None:
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="c1")
        assert out.values == ["efficiency", "reliability"]
```

**Step 2: Run to verify fails**

Run: `cd personas-pipeline/synthesis && pytest tests/test_eval_adapter.py -v`
Expected: The 8 new tests fail; previous 9 still pass.

**Step 3: Extend adapter**

In `personas-pipeline/synthesis/synthesis/adapters/eval_adapter.py`, replace the final `EvalPersona(...)` return with this expanded version:

```python
    comm = persona.communication_style
    emo = persona.emotional_profile
    moral = persona.moral_framework

    return EvalPersona(
        id=persona_id,
        name=persona.name,
        bio=persona.summary,
        age=_age_midpoint(demo.age_range),
        gender=_normalize_gender(demo.gender_distribution),
        location=demo.location_signals[0] if demo.location_signals else "",
        education=demo.education_level or "",
        income_bracket=demo.income_bracket or "",
        occupation=firm.role_titles[0] if firm.role_titles else "",
        industry=firm.industry or "",
        knowledge_domains=list(firm.tech_stack_signals),
        goals=list(persona.goals),
        motivations=list(persona.motivations),
        pain_points=list(persona.pains),
        frustrations=list(persona.objections),
        values=list(moral.core_values),
        communication_style=EvalCommunicationStyle(
            tone=comm.tone,
            formality=comm.formality,
            vocabulary_level=comm.vocabulary_level,
            preferred_channels=list(comm.preferred_channels),
        ),
        emotional_profile=EvalEmotionalProfile(
            baseline_mood=emo.baseline_mood,
            stress_triggers=list(emo.stress_triggers),
            coping_mechanisms=list(emo.coping_mechanisms),
        ),
        moral_framework=EvalMoralFramework(
            core_values=list(moral.core_values),
            ethical_stance=moral.ethical_stance,
            moral_foundations=dict(moral.moral_foundations),
        ),
        extra=extra,
    )
```

**Step 4: Run to verify all pass**

Run: `cd personas-pipeline/synthesis && pytest tests/test_eval_adapter.py -v`
Expected: 17 passed (9 + 8).

**Step 5: Commit**

```bash
git add personas-pipeline/synthesis/synthesis/adapters/eval_adapter.py \
        personas-pipeline/synthesis/tests/test_eval_adapter.py
git commit -m "feat: adapter — behavioral lists + psychological sub-schema copy

goals/motivations direct copy, pains→pain_points, objections→
frustrations, and psychological sub-schemas round-trip into the
equivalent eval.Persona shapes. core_values also mirrored into
eval.Persona.values for scorer compatibility."
```

---

## Task 8: Implement adapter — evidence, extras, and source_ids

**Context:**
- `source_evidence[*].record_ids` → `source_ids` (deduplicated, order-preserving)
- `journey_stages` → `extra["journey_stages"]` (list of dicts via `.model_dump()`)
- `sample_quotes` → `extra["sample_quotes"]`
- `decision_triggers` → `extra["decision_triggers"]`
- `channels` → `extra["channels"]`
- `vocabulary` → `extra["vocabulary"]`
- `source_evidence` → `extra["source_evidence"]` (full dump for provenance)

**Step 1: Append failing tests**

Append to `test_eval_adapter.py`:

```python
class TestAdapterEvidenceAndExtras:
    def test_source_ids_from_evidence_deduped(self) -> None:
        p = _fully_populated_persona()
        p.source_evidence.append(
            # Duplicate record id across evidence entries
            __import__("synthesis.models.evidence", fromlist=["SourceEvidence"]).SourceEvidence(
                claim="extra", record_ids=["r1", "r2"], field_path="goals.1", confidence=0.8
            )
        )
        out = persona_v1_to_eval(p, persona_id="c1")
        assert out.source_ids == ["r1", "r2"]  # order preserved, deduped

    def test_journey_stages_in_extra(self) -> None:
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="c1")
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
        p = _fully_populated_persona()
        out = persona_v1_to_eval(p, persona_id="c1")
        dumped = out.extra["source_evidence"]
        assert isinstance(dumped, list)
        assert dumped[0]["claim"] == "c1"
```

**Step 2: Run to verify fails**

Run: `cd personas-pipeline/synthesis && pytest tests/test_eval_adapter.py::TestAdapterEvidenceAndExtras -v`
Expected: 7 fail.

**Step 3: Extend adapter**

In `eval_adapter.py`, update the `extra` dict construction (replace the earlier `extra = {...}` block) with:

```python
    # Dedupe source ids while preserving order
    seen_ids: set[str] = set()
    source_ids: list[str] = []
    for ev in persona.source_evidence:
        for rid in ev.record_ids:
            if rid not in seen_ids:
                seen_ids.add(rid)
                source_ids.append(rid)

    extra: dict = {
        "age_range": demo.age_range,
        "gender_distribution": demo.gender_distribution,
        "location_signals": list(demo.location_signals),
        "vocabulary": list(persona.vocabulary),
        "channels": list(persona.channels),
        "decision_triggers": list(persona.decision_triggers),
        "sample_quotes": list(persona.sample_quotes),
        "journey_stages": [js.model_dump() for js in persona.journey_stages],
        "source_evidence": [ev.model_dump() for ev in persona.source_evidence],
    }
    if firm.company_size is not None:
        extra["company_size"] = firm.company_size
```

Add `source_ids=source_ids,` to the `EvalPersona(...)` kwargs.

**Step 4: Run all adapter tests**

Run: `cd personas-pipeline/synthesis && pytest tests/test_eval_adapter.py -v`
Expected: 24 passed.

**Step 5: Commit**

```bash
git add personas-pipeline/synthesis/synthesis/adapters/eval_adapter.py \
        personas-pipeline/synthesis/tests/test_eval_adapter.py
git commit -m "feat: adapter — extras bucket + deduped source_ids

Journey stages, sample quotes, decision triggers, channels, vocabulary,
and full source_evidence now preserved in eval.Persona.extra for
provenance and downstream scorer access. source_ids deduped from
evidence record references."
```

---

## Task 9: Regenerate PersonaV1 fixtures + add end-to-end adapter smoke test

**Context:** `output/persona_00.json` and `output/persona_01.json` were generated under the old schema and will now fail PersonaV1 validation (missing psychological fields). They are mock fixtures, not production data — regenerate them by hand with psychological fields so the file round-trips.

**Files:**
- Modify: `personas-pipeline/output/persona_00.json`
- Modify: `personas-pipeline/output/persona_01.json`
- Create: `personas-pipeline/synthesis/tests/test_end_to_end_adapter.py`

**Step 1: Read the existing fixtures fully**

Run: `cat personas-pipeline/output/persona_00.json personas-pipeline/output/persona_01.json`
(via the Read tool — remember to use the tool, not shell cat)

**Step 2: Add psychological fields + evidence to each JSON**

For each fixture, add into the `persona` object (after `journey_stages`, before `source_evidence`):

```json
"communication_style": {
  "tone": "...",
  "formality": "...",
  "vocabulary_level": "...",
  "preferred_channels": ["..."]
},
"emotional_profile": {
  "baseline_mood": "...",
  "stress_triggers": ["..."],
  "coping_mechanisms": ["..."]
},
"moral_framework": {
  "core_values": ["...", "..."],
  "ethical_stance": "...",
  "moral_foundations": {"care": 0.0, "fairness": 0.0}
},
```

Derive values from existing persona content (vocabulary, sample_quotes, pains, motivations). For `persona_00.json` (DevOps engineer) example values:
- `tone`: "direct"
- `formality`: "professional"
- `vocabulary_level`: "advanced"
- `preferred_channels`: ["Intercom", "GitHub", "Slack"]
- `baseline_mood`: "pragmatic"
- `stress_triggers`: ["GraphQL schema drift", "undocumented webhook retries", "context-switching across docs"]
- `coping_mechanisms`: ["writes automation", "files detailed support tickets", "forks Terraform providers"]
- `core_values`: ["efficiency", "reliability", "infrastructure-as-code"]
- `ethical_stance`: "utilitarian"
- `moral_foundations`: {"care": 0.4, "fairness": 0.6, "liberty": 0.8}

For `persona_01.json` (freelance brand designer), derive analogous values from its content.

Then add new `source_evidence` entries (at least one per sub-schema) to satisfy groundedness. Use existing `record_ids` already in the fixture. Example:
```json
{"claim": "Direct technical tone — complains about GraphQL rough edges verbatim", "record_ids": ["intercom_000"], "field_path": "communication_style.tone", "confidence": 0.9},
{"claim": "Pragmatic baseline — complains but with specifics, not venting", "record_ids": ["intercom_000"], "field_path": "emotional_profile.baseline_mood", "confidence": 0.8},
{"claim": "Efficiency as core value — 'I don't want to click anything'", "record_ids": ["intercom_000", "ga4_000"], "field_path": "moral_framework.core_values.0", "confidence": 0.85}
```

**Step 3: Write the smoke test**

Create `personas-pipeline/synthesis/tests/test_end_to_end_adapter.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from persona_eval.schemas import Persona as EvalPersona

from synthesis.adapters.eval_adapter import persona_v1_to_eval
from synthesis.models.persona import PersonaV1

FIXTURES = [
    ("persona_00.json", "clust_1adb81b417c0"),
    ("persona_01.json", None),  # id discovered from file
]

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"


@pytest.mark.parametrize("filename,expected_cluster_id", FIXTURES)
def test_fixture_roundtrip(filename: str, expected_cluster_id: str | None) -> None:
    path = OUTPUT_DIR / filename
    data = json.loads(path.read_text())

    # 1. PersonaV1 validates (proves fixture was regenerated with psych fields)
    persona = PersonaV1.model_validate(data["persona"])

    # 2. Adapter produces a valid eval.Persona
    cluster_id = data["cluster_id"]
    eval_persona = persona_v1_to_eval(persona, persona_id=cluster_id)
    assert isinstance(eval_persona, EvalPersona)

    # 3. Round-trips through eval.Persona JSON (catches Pydantic shape drift)
    rt = EvalPersona.model_validate(eval_persona.model_dump())
    assert rt.id == cluster_id
    assert rt.name == persona.name
    assert rt.communication_style.tone == persona.communication_style.tone
    assert rt.emotional_profile.baseline_mood == persona.emotional_profile.baseline_mood
    assert rt.moral_framework.ethical_stance == persona.moral_framework.ethical_stance
```

**Step 4: Run the smoke test**

Run: `cd personas-pipeline/synthesis && pytest tests/test_end_to_end_adapter.py -v`
Expected: 2 passed.

If it fails at "PersonaV1.model_validate" — the fixture edits in Step 2 are incomplete. Read the validation error carefully and fill in the missing required fields.

**Step 5: Run full test suite**

Run: `cd personas-pipeline/synthesis && pytest tests/ -v`
Expected: all tests pass (previous tasks + this one).

**Step 6: Commit**

```bash
git add personas-pipeline/output/persona_00.json \
        personas-pipeline/output/persona_01.json \
        personas-pipeline/synthesis/tests/test_end_to_end_adapter.py
git commit -m "feat: regenerate persona fixtures with psychological depth + e2e adapter smoke test

Fixtures now validate against PersonaV1 with communication_style,
emotional_profile, and moral_framework fields, grounded in existing
source_evidence record IDs. End-to-end test proves PersonaV1 → eval.Persona
round-trips for both fixtures."
```

---

## Task 10: CLI converter script

**File:**
- Create: `personas-pipeline/synthesis/scripts/convert_to_eval_personas.py`

**Purpose:** Reads `personas-pipeline/output/persona_*.json`, produces `personas-pipeline/output/eval_personas/<cluster_id>.json` files validated as `persona_eval.Persona`.

**Step 1: Write the script**

Create `personas-pipeline/synthesis/scripts/convert_to_eval_personas.py`:

```python
"""Convert PersonaV1 JSON outputs to persona_eval.Persona JSON.

Usage:
    python -m scripts.convert_to_eval_personas [--input-dir DIR] [--output-dir DIR]

Defaults:
    --input-dir: personas-pipeline/output
    --output-dir: personas-pipeline/output/eval_personas
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from synthesis.adapters.eval_adapter import persona_v1_to_eval
from synthesis.models.persona import PersonaV1


def convert_file(src: Path, dst_dir: Path) -> Path:
    data = json.loads(src.read_text())
    persona = PersonaV1.model_validate(data["persona"])
    cluster_id = data.get("cluster_id") or src.stem
    eval_persona = persona_v1_to_eval(persona, persona_id=cluster_id)

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{cluster_id}.json"
    dst.write_text(json.dumps(eval_persona.model_dump(), indent=2))
    return dst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_in = Path(__file__).resolve().parents[2] / "output"
    parser.add_argument("--input-dir", type=Path, default=default_in)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_in / "eval_personas",
    )
    args = parser.parse_args(argv)

    src_files = sorted(args.input_dir.glob("persona_*.json"))
    if not src_files:
        print(f"No persona_*.json files found in {args.input_dir}", file=sys.stderr)
        return 1

    for src in src_files:
        dst = convert_file(src, args.output_dir)
        print(f"  {src.name} → {dst.relative_to(args.input_dir.parent)}")

    print(f"\nConverted {len(src_files)} persona(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Step 2: Run the script**

Run: `cd personas-pipeline/synthesis && python -m scripts.convert_to_eval_personas`
Expected output:
```
  persona_00.json → output/eval_personas/clust_1adb81b417c0.json
  persona_01.json → output/eval_personas/<its-cluster-id>.json

Converted 2 persona(s) to ...
```

**Step 3: Verify outputs validate as eval.Persona**

Run:
```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone && python -c "
import json, pathlib
from persona_eval.schemas import Persona
for p in pathlib.Path('personas-pipeline/output/eval_personas').glob('*.json'):
    obj = Persona.model_validate(json.loads(p.read_text()))
    print(f'{p.name}: {obj.name!r} ({obj.occupation})')
"
```
Expected: both files print without errors.

**Step 4: Commit**

```bash
git add personas-pipeline/synthesis/scripts/convert_to_eval_personas.py \
        personas-pipeline/output/eval_personas/
git commit -m "feat: CLI to convert synthesis outputs to eval.Persona JSON

Reads output/persona_*.json, writes output/eval_personas/<cluster_id>.json
validated as persona_eval.Persona. One-shot batch converter — the
adapter itself is the library API for programmatic use."
```

---

## Task 11: Wire converted personas into persona_eval suite (smoke test)

**File:**
- Create: `personas-pipeline/synthesis/tests/test_eval_integration.py`

**Purpose:** Prove that a converted persona actually loads into the eval framework. This is not a full scoring test — just validates that `SuiteRunner` (or its loader) accepts the output of our adapter.

**Step 1: Look up how persona_eval loads personas**

Read `persona_eval/cli.py` and `persona_eval/suite_runner.py` to find the entry point that takes a JSON file and returns/uses a Persona. Do not run the full eval — it's expensive. Just confirm the loading step succeeds.

**Step 2: Write the test**

Create `personas-pipeline/synthesis/tests/test_eval_integration.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from persona_eval.schemas import Persona as EvalPersona

from synthesis.adapters.eval_adapter import persona_v1_to_eval
from synthesis.models.persona import PersonaV1


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"


def test_converted_persona_loads_as_eval_persona() -> None:
    """Smoke test: a converted PersonaV1 must validate as persona_eval.Persona
    when re-loaded from JSON, with all eval-schema fields populated."""
    src = OUTPUT_DIR / "persona_00.json"
    data = json.loads(src.read_text())
    persona_v1 = PersonaV1.model_validate(data["persona"])

    eval_persona = persona_v1_to_eval(persona_v1, persona_id=data["cluster_id"])

    # Round-trip through JSON (simulates persona_eval CLI loading)
    rehydrated = EvalPersona.model_validate(
        json.loads(json.dumps(eval_persona.model_dump()))
    )

    # Fields the eval scorers actually read
    assert rehydrated.id
    assert rehydrated.name
    assert rehydrated.bio
    assert rehydrated.communication_style.tone
    assert rehydrated.communication_style.vocabulary_level
    assert rehydrated.emotional_profile.baseline_mood
    assert rehydrated.emotional_profile.stress_triggers
    assert rehydrated.moral_framework.core_values
    assert rehydrated.moral_framework.ethical_stance
    assert rehydrated.goals
    assert rehydrated.pain_points
    assert rehydrated.source_ids
```

**Step 3: Run the test**

Run: `cd personas-pipeline/synthesis && pytest tests/test_eval_integration.py -v`
Expected: 1 passed.

**Step 4: Commit**

```bash
git add personas-pipeline/synthesis/tests/test_eval_integration.py
git commit -m "test: smoke test that converted PersonaV1 loads as eval.Persona

Verifies every eval-facing field the scorers read is populated after
the PersonaV1 → eval.Persona round-trip through JSON. This proves the
schema bridge is end-to-end wired."
```

---

## Task 12: Code review loop

**Per the repo's CLAUDE.md, after finishing implementation:**

1. Invoke `superpowers:requesting-code-review` skill
2. Address every Critical and Important finding
3. Re-invoke until the reviewer returns zero Critical/Important issues
4. Then finalize and ship

**Final "ship" commit:**

After review passes, create one summary commit marking this as a shipped milestone:

```bash
git commit --allow-empty -m "feat: persona schema bridge — synthesis → eval with psychological depth [ship]

PersonaV1 now carries communication_style, emotional_profile, and
moral_framework grounded in source records. The synthesis prompt teaches
the LLM to extract these from Intercom and GA4 signals without fabrication,
and the groundedness checker enforces evidence coverage. A pure
field-mapping adapter converts PersonaV1 → persona_eval.Persona — no
second LLM call, no synthetic depth.

Realized the schema gap was real mid-conversation: the eval framework
wanted behavioral data the pipeline was throwing away. Fix wasn't a
post-hoc enrichment LLM — the raw Intercom messages had all the signal
already, the synthesis prompt just wasn't asking for it. One prompt
extension and three Pydantic classes later, the bridge is a 50-line
adapter."
```

---

## Summary of commits (predicted)

1. `feat: add psychological sub-schemas to PersonaV1 models`
2. `feat: require communication_style, emotional_profile, moral_framework on PersonaV1`
3. `feat: extend synthesis prompt with psychological extraction rules`
4. `feat: require source evidence for psychological sub-schemas`
5. `chore: scaffold PersonaV1 → eval.Persona adapter package`
6. `feat: adapter — identity + demographic flattening`
7. `feat: adapter — behavioral lists + psychological sub-schema copy`
8. `feat: adapter — extras bucket + deduped source_ids`
9. `feat: regenerate persona fixtures with psychological depth + e2e adapter smoke test`
10. `feat: CLI to convert synthesis outputs to eval.Persona JSON`
11. `test: smoke test that converted PersonaV1 loads as eval.Persona`
12. `feat: persona schema bridge — synthesis → eval with psychological depth [ship]`

---

## Open questions & risks for the executing engineer

1. **Cross-package import**: confirm `from persona_eval.schemas import Persona` works from inside the `synthesis` package at Task 5. If the repo root isn't installed/on `PYTHONPATH`, stop and ask the user before hacking a workaround.
2. **Schema version bump**: this plan keeps `schema_version = "1.0"` and treats the psychological fields as an additive breaking change within v1. If the user wants `schema_version = "2.0"` and a migration path for legacy v1 JSONs, raise this before Task 2.
3. **Groundedness threshold**: `passed` currently requires score ≥ 0.9. Adding 3 new required items into a cluster that previously had ~17 required items drops a hypothetical "passing with one miss" persona below 0.9. Verify the live synthesizer still converges on real fixture clusters before considering the plan done.
4. **Cost impact**: longer prompt + more output tokens → more per-synthesis cost. Run `run_exp_1_23.py` (or equivalent) once after the plan is implemented to confirm the end-to-end synthesis still succeeds within `COST_SAFETY_MULTIPLIER`. Raise with user if retries spike.
5. **Existing persona_01.json structure**: Task 9 assumes its shape matches persona_00.json. If it has unexpected fields, raise and ask before regenerating blindly.

---

## How to run the full test suite at any point

```bash
cd /Users/ivanma/Desktop/gauntlet/Capstone/personas-pipeline/synthesis && pytest tests/ -v
```

Expected after Task 11: all tests pass (psych schemas + PersonaV1 + prompt + groundedness + adapter × 3 categories + e2e + integration).
