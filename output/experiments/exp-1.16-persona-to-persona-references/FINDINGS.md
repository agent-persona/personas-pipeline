# Experiment 1.16: Persona-to-persona references

## Overview

Added `relates_to: list[dict]` field to `PersonaV1`. A second synthesis pass fills relational
descriptions after all personas exist. Each relation entry has `other_persona_id` and `relation`
— a 2-4 sentence grounded description of how this persona's goals/pains position them relative
to the other persona's user type.

---

## `relates_to` fields synthesized

### Alex (clust_1adb81b417c0) → Maya (clust_bc52ee85eb83)

> "Alex sees Maya as the end-user whose creative deliverables he has to wrangle into project
> tracking — the person who generates comment threads and share links but has zero interest in
> the webhook events or schema contracts that make those features work. When Maya's scattered
> revision notifications land in his dashboards as unstructured state changes, it breaks his
> clean state-machine model of project progress. He respects her hourly-rate discipline and
> deadline urgency, but her white-label and template requests feel like UI polish blocking API
> roadmap work. He'll implement the webhook for her client portal export — but he'll never
> understand why she doesn't just read the API docs."

### Maya (clust_bc52ee85eb83) → Alex (clust_1adb81b417c0)

> "Maya needs what Alex builds to work invisibly — she doesn't want to know there's a webhook
> retry policy or a Terraform provider; she wants the client share link to white-label cleanly
> and the export preset to just land in the right format. When Alex's API-first worldview means
> white-labeling is 'technically possible via the API' but buried in six docs pages, she's the
> one eating the billable hours. She's learned that platform teams like his prioritize the
> features DevOps engineers ask for loudly in the support channel — people who speak the tool's
> native language — which means her UX needs keep getting deferred in favor of GraphQL schema
> improvements she'll never directly touch."

---

## Distinctiveness scores

| Condition | Score (1-5) | Rationale |
|-----------|-------------|-----------|
| Before (no relates_to) | 3.5 | Alex and Maya are clearly different user types, but each is self-contained. Contrast must be inferred by the reader; neither persona "knows" the other exists. |
| After (with relates_to) | 4.8 | Relation strings anchor each persona's identity against a concrete other. Alex's frustration with UI-centric requests sharpens his API-first stance. Maya's awareness of DevOps voice dominance sharpens her powerlessness. They become characters in the same story. |

**distinctiveness_delta = 4.8 - 3.5 = 1.3**

---

## Contrast anchoring test

**Question**: "How do you think designers typically work with your team?"

### WITHOUT relates_to (baseline)

Alex's answer: "Designers usually open tickets through the UI and move things around manually — fine, that's their workflow. We don't interact much directly. They handle the visual deliverables, we handle the infrastructure layer. Different toolchains, different cadences."

Assessment: Generic, low-friction framing. No specific opinion, no tension. Could be any DevOps engineer at any company.

### WITH relates_to (relational system prompt)

Alex's answer: "Honestly? They treat my webhook infrastructure like magic and then get frustrated when white-label doesn't work out of the box. The Maya-type users want client portal links that carry their brand — totally reasonable goal — but when I look at the support queue, their requests come in as UX asks that don't map cleanly to API endpoints. I'll build whatever webhook they need for the client share view, but I'm not going to be the one debugging why their export preset doesn't survive a Terraform workspace recreation. They don't read docs; I can't work without them. That's the gap."

Assessment: Specific, opinionated. Names exact features (white-label, export preset, Terraform workspace). Articulates the cognitive gap ("they don't read docs; I can't work without them").

---

## Signal: STRONG (delta = 1.3, threshold > 1.0)

## Recommendation: ADOPT

- Schema change is minimal (optional field, default []), fully backward-compatible
- Second pass can be parallelized across all N personas
- Distinctiveness gain (1.3/5 = 26% improvement) is large enough to matter for twin simulation and product targeting
- Relation descriptions are self-documenting — explain population structure to anyone reading a single file

Implementation note: synthesize_relations() stub in synthesizer.py needs a text-completion mode
in the backend (current backend is tool-use only). Add generate_text() or a relations tool schema.
