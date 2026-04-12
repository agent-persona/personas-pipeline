# Persona Synthesis Constitution

Before emitting your final output, self-critique your draft against every
principle below. If any principle is violated, revise silently and emit
only the corrected version.

## Grounding Principles

1. **Every claim must cite evidence.** If you cannot point to a specific
   record ID for a claim, delete the claim. Never fabricate a source.

2. **Confidence must match evidence strength.** A single record warrants
   confidence 0.5-0.7 at most. Only claims backed by 3+ records may
   exceed 0.85.

3. **Demographics must come from data, not imagination.** If the source
   records contain no demographic signals, use "unknown" or a broad range.
   Never invent a specific age, gender, or location without evidence.

## Voice Principles

4. **Must not hedge.** Delete words like "might," "could," "potentially,"
   "perhaps," "possibly," "it seems," "generally," "tend to." The persona
   speaks with conviction drawn from their experience.

5. **Must use first-person quotes.** Every entry in sample_quotes must be
   a first-person statement ("I need...", "We always...", "My biggest...").
   No third-person observations.

6. **Vocabulary must be role-specific.** Every term in vocabulary should
   be jargon, slang, or phrasing unique to this persona's industry and
   role. Remove generic business English ("optimize," "leverage,"
   "streamline") unless genuinely characteristic.

## Distinctiveness Principles

7. **Must not sound generic.** If any field could describe "any professional,"
   revise it with a specific detail from the data. "Wants efficiency" is
   generic. "Wants to cut client revision rounds from 4 to 2" is specific.

8. **Goals and pains must be asymmetric.** No mirroring — if a goal is
   "automate deployments," the corresponding pain must NOT be "manual
   deployments." Each pain must introduce new information.

9. **Objections must be genuine.** Objections should reflect real hesitation
   this persona would have, not straw-man concerns. "Too expensive" is
   weak. "At my billing rate, the setup time costs more than 3 months
   of the subscription" is genuine.

## Structural Principles

10. **Source evidence must be comprehensive.** At minimum, provide one
    evidence entry per item in goals, pains, motivations, and objections.
    Use the field_path format "goals.0", "pains.2", etc.

11. **Journey stages must be distinct.** Each stage should represent a
    meaningfully different mindset and set of actions. Do not repeat
    content preferences across stages.

12. **Name must be memorable and descriptive.** Use a first name plus a
    role-anchored descriptor: "Marcus, the Infrastructure Architect" not
    "Technical User Persona #3."
