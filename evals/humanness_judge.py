"""Judge for scoring twin chat replies on humanness dimensions."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from anthropic import AsyncAnthropic


@dataclass
class HumannessScore:
    discourse_markers: float  # 1-5
    hedging: float  # 1-5
    specificity: float  # 1-5
    sentence_variety: float  # 1-5
    emotional_authenticity: float  # 1-5
    overall: float  # 1-5
    rationale: str = ""


_HUMANNESS_SYSTEM_PROMPT = """\
You are an expert linguist evaluating whether a chat reply reads like a real \
human wrote it or like an AI generated it. You score replies on five sub-dimensions.

Sub-dimensions (each 1-5):

1. **discourse_markers** — Does the reply use natural discourse markers like \
"That said", "Here's the thing", "Look", "I mean", "Honestly"? \
1 = none, robotic transitions. 5 = natural, varied discourse markers throughout.

2. **hedging** — Does the reply hedge appropriately with phrases like "I think", \
"probably", "not sure but", "in my experience"? \
1 = everything stated as absolute fact. 5 = natural hedging that mirrors real speech.

3. **specificity** — Does the reply reference specific anecdotes, numbers, \
experiences, or details rather than generic platitudes? \
1 = entirely generic ("I want better tools"). 5 = concrete and personal \
("Last Tuesday I spent 3 hours wrestling with the deploy pipeline").

4. **sentence_variety** — Does the reply mix short punchy sentences with longer \
ones? Does it use fragments, rhetorical questions, or asides? \
1 = uniform robotic sentence structure. 5 = varied, natural rhythm.

5. **emotional_authenticity** — Does the reply convey genuine emotion \
(frustration, excitement, resignation, humor) rather than flat affect? \
1 = emotionless corporate speak. 5 = emotionally textured and believable.

**overall** = holistic impression of whether this reads like a real person. \
Not a simple average — weight your judgment.

Respond with ONLY a JSON object (no markdown, no extra text):
{
  "discourse_markers": <1-5>,
  "hedging": <1-5>,
  "specificity": <1-5>,
  "sentence_variety": <1-5>,
  "emotional_authenticity": <1-5>,
  "overall": <1-5>,
  "rationale": "<brief justification>"
}
"""

_CALIBRATION_EXAMPLES = """\
=== CALIBRATION: Score 1 (Robotic) ===
Question: What's your biggest frustration with current tools?
Reply: "My biggest frustration with current tools is the lack of integration \
between platforms. Additionally, reporting capabilities are insufficient. \
Furthermore, the onboarding process is complex. I would recommend improvements \
in these areas."
SCORES: discourse_markers=1, hedging=1, specificity=1, sentence_variety=1, \
emotional_authenticity=1, overall=1
WHY: Reads like a bullet-point list reformatted as prose. No personality, \
no hedging, uniform sentence structure, zero emotion.

=== CALIBRATION: Score 3 (Decent but generic) ===
Question: What's your biggest frustration with current tools?
Reply: "Honestly, I think the reporting is the biggest pain point for me. \
It takes too long to pull the numbers I need, and I end up doing a lot of \
manual work in spreadsheets. It's not terrible, but it could be a lot better."
SCORES: discourse_markers=3, hedging=3, specificity=2, sentence_variety=3, \
emotional_authenticity=3, overall=3
WHY: Has some natural markers ("Honestly", "I think") and mild emotion, \
but the frustration is generic — could be anyone at any company.

=== CALIBRATION: Score 5 (Natural human voice) ===
Question: What's your biggest frustration with current tools?
Reply: "Oh god, where do I start. So every Monday I have to pull this \
pipeline report for the leadership sync, right? And it takes me like 45 \
minutes because I'm copy-pasting between three different dashboards. I've \
asked engineering twice to build an export and both times it got deprioritized. \
At this point I've just... accepted it? Which is probably worse."
SCORES: discourse_markers=5, hedging=5, specificity=5, sentence_variety=5, \
emotional_authenticity=5, overall=5
WHY: Discourse markers ("Oh god", "right?"), hedging ("probably"), specific \
detail (Monday, 45 minutes, three dashboards), varied rhythm (fragments, \
rhetorical question), genuine resignation and humor.
"""


class TwinReplyJudge:
    """Scores twin chat replies on humanness sub-dimensions."""

    def __init__(
        self,
        client,  # AsyncAnthropic or AsyncOpenAI
        model: str = "claude-sonnet-4-20250514",
        backend_type: str = "anthropic",
    ):
        self.client = client
        self.model = model
        self.backend_type = backend_type

    async def score_reply(
        self,
        reply: str,
        persona_name: str,
        question: str,
    ) -> HumannessScore:
        prompt = (
            f"{_CALIBRATION_EXAMPLES}\n\n"
            f"=== TARGET REPLY TO SCORE ===\n"
            f"Persona: {persona_name}\n"
            f"Question: {question}\n"
            f'Reply: "{reply}"\n\n'
            f"Score this reply on each sub-dimension (1-5)."
        )

        if self.backend_type == "openai":
            response = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": _HUMANNESS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content or ""
        else:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=_HUMANNESS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text

        return self._parse(text)

    def _parse(self, text: str) -> HumannessScore:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        data = None
        # Try direct parse
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try regex extraction of individual numeric fields
        if data is None:
            data = {}
            for key in ["discourse_markers", "hedging", "specificity",
                        "sentence_variety", "emotional_authenticity", "overall"]:
                match = re.search(rf'"{key}"\s*:\s*(\d+(?:\.\d+)?)', text)
                if match:
                    data[key] = float(match.group(1))
            # Extract rationale
            rat_match = re.search(r'"rationale"\s*:\s*"([^"]*)"', text)
            if rat_match:
                data["rationale"] = rat_match.group(1)

        if not data:
            return HumannessScore(
                discourse_markers=float("nan"),
                hedging=float("nan"),
                specificity=float("nan"),
                sentence_variety=float("nan"),
                emotional_authenticity=float("nan"),
                overall=float("nan"),
                rationale=f"Failed to parse: {text[:200]}",
            )

        def _get(key: str) -> float:
            v = data.get(key)
            return float(v) if v is not None else float("nan")

        return HumannessScore(
            discourse_markers=_get("discourse_markers"),
            hedging=_get("hedging"),
            specificity=_get("specificity"),
            sentence_variety=_get("sentence_variety"),
            emotional_authenticity=_get("emotional_authenticity"),
            overall=_get("overall"),
            rationale=data.get("rationale", ""),
        )
