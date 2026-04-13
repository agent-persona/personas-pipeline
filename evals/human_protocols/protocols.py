"""Canonical definitions of the three exp-5.06 human-labeling protocols.

Each protocol specifies:
  - name
  - columns to emit into the Prolific CSV
  - label format (what the rater clicks and what shape comes back)
  - agreement metric to apply when ingesting labels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AgreementMetric = Literal["cohen_kappa", "krippendorff_alpha_ordinal", "krippendorff_alpha_nominal"]


@dataclass
class Protocol:
    protocol_id: str
    name: str
    description: str
    csv_columns: list[str]
    response_column: str
    response_values: list[str]  # for nominal — used to drive encoding
    metric: AgreementMetric
    n_items_target: int = 60
    n_raters_per_item: int = 5
    instruction: str = ""


PROTOCOLS: dict[str, Protocol] = {
    "a_blind_matching": Protocol(
        protocol_id="a",
        name="Blind matching",
        description=(
            "Rater sees a 6-turn chat transcript and must pick which of 3 "
            "candidate personas produced the twin side of the transcript. "
            "The correct persona is randomized among the 3 positions per item."
        ),
        csv_columns=[
            "item_id",
            "transcript",
            "persona_a_description",
            "persona_b_description",
            "persona_c_description",
            "correct_position",  # hidden from rater — for eval
        ],
        response_column="picked_position",
        response_values=["a", "b", "c"],
        metric="cohen_kappa",
        instruction=(
            "Read the chat transcript below. One side is a real person (the user), "
            "the other side is a persona-driven chat bot. Read the three candidate "
            "personas and pick which one you think drove the bot side."
        ),
    ),
    "b_pairwise_preference": Protocol(
        protocol_id="b",
        name="Pairwise preference",
        description=(
            "Rater sees the same user prompt answered by two different twins "
            "(possibly driven by different personas or the same persona under "
            "different runtime settings). Picks which feels more like a real "
            "person fitting the described persona."
        ),
        csv_columns=[
            "item_id",
            "persona_description",
            "user_prompt",
            "reply_a",
            "reply_b",
        ],
        response_column="preference",
        response_values=["a_better", "b_better", "tie"],
        metric="krippendorff_alpha_ordinal",
        instruction=(
            "Read the persona description and the user prompt. Then read the two "
            "candidate replies. Pick which reply feels more like this persona "
            "would actually say it, or 'tie' if they are equivalent."
        ),
    ),
    "c_forced_choice_id": Protocol(
        protocol_id="c",
        name="Forced-choice persona ID",
        description=(
            "Rater sees a single twin reply to a single prompt, plus N candidate "
            "personas shown as compact cards. Picks which persona wrote the reply."
        ),
        csv_columns=[
            "item_id",
            "user_prompt",
            "reply",
            "persona_1_card",
            "persona_2_card",
            "persona_3_card",
            "persona_4_card",
            "correct_persona_idx",  # hidden from rater
        ],
        response_column="picked_persona_idx",
        response_values=["1", "2", "3", "4"],
        metric="cohen_kappa",
        instruction=(
            "Read the user prompt and the reply. Then read the four candidate "
            "persona cards below. Pick which persona you think wrote the reply."
        ),
    ),
}
