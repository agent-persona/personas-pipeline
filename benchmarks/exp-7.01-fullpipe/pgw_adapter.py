"""Lightweight Anthropic port of joongishin/persona-generation-workflow's
LLM-summarizing++ variant.

Upstream flow (DIS'24 paper, Shin et al.):
    1. Human pre-clusters users (we reuse our segmentation output to
       keep the comparison on equal clusters).
    2. LLM summarizes each cluster into a persona, with designer-
       specified qualities.

This module implements step 2 with a single Anthropic call per cluster,
same model as our pipeline (Haiku, temperature=0.0). Output fields
match the paper's PERSONA template (name, age, gender, occupation,
background, personality, plans, motivations). No source-evidence,
no structural groundedness, no retries — it's the simple LLM-as-
summarizer baseline.

The point is to reproduce the *methodology* (single-pass LLM
summarization of a clustered user group) under matched conditions, not
to copy their repo byte-for-byte.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from anthropic import AsyncAnthropic

PGW_SYSTEM = (
    "You are a UX researcher. Given records from one segment of a "
    "product's users, write a representative persona. Output a single "
    "JSON object with these fields: name (string), age (int, rough "
    "estimate), occupation (string), background (string, 2-3 "
    "sentences), personality (string, 2-3 sentences), plans (list of "
    "strings, 2-4 items), motivations (list of strings, 2-4 items). "
    "Use only information inferable from the records. Do not add "
    "information. Return valid JSON only, no markdown fences."
)


def _format_cluster(cluster: dict) -> str:
    tenant = cluster["tenant"]
    summary = cluster["summary"]
    sample = cluster.get("sample_records", [])
    parts = [
        f"Tenant: {tenant['tenant_id']} ({tenant.get('industry','?')} - {tenant.get('product_description','?')}).",
        f"Cluster size: {summary['cluster_size']} users.",
        f"Top behaviors: {', '.join(summary.get('top_behaviors', []))}.",
        f"Top pages: {', '.join(summary.get('top_pages', []))}.",
        f"Sample records ({len(sample)}):",
    ]
    for r in sample:
        parts.append(
            f"  - [{r.get('source')}] payload={r.get('payload')}"
        )
    return "\n".join(parts)


@dataclass
class PGWPersona:
    cluster_id: str
    persona: dict | None
    raw_text: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    error: str | None = None


# Haiku pricing $/M tokens: input 1.00, output 5.00
_HAIKU_INPUT = 1.00
_HAIKU_OUTPUT = 5.00


async def synthesize_pgw(
    cluster: dict,
    client: AsyncAnthropic,
    model: str = "claude-haiku-4-5-20251001",
) -> PGWPersona:
    """One-shot LLM summarization -> persona dict."""
    prompt = _format_cluster(cluster)
    resp = await client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.0,
        system=PGW_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    # Join all text blocks (handles the case where anthropic returns
    # multi-block content).
    txt = "".join(getattr(b, "text", "") for b in resp.content).strip()
    input_tok = resp.usage.input_tokens
    output_tok = resp.usage.output_tokens
    cost = (input_tok * _HAIKU_INPUT + output_tok * _HAIKU_OUTPUT) / 1_000_000

    # Strip common markdown fences and find the first JSON object.
    fenced = txt
    if "```" in fenced:
        # Grab the content between the first pair of fences.
        m = re.search(r"```(?:json)?\s*(.*?)```", fenced, re.DOTALL)
        if m:
            fenced = m.group(1).strip()
    obj_match = re.search(r"\{.*\}", fenced, re.DOTALL)
    candidate = obj_match.group(0) if obj_match else fenced

    try:
        persona = json.loads(candidate) if candidate else None
        if persona is None:
            raise ValueError("empty response text")
        return PGWPersona(
            cluster_id=cluster["cluster_id"],
            persona=persona,
            raw_text=txt,
            cost_usd=cost,
            input_tokens=input_tok,
            output_tokens=output_tok,
        )
    except Exception as e:
        return PGWPersona(
            cluster_id=cluster["cluster_id"],
            persona=None,
            raw_text=txt,
            cost_usd=cost,
            input_tokens=input_tok,
            output_tokens=output_tok,
            error=f"JSON parse failed: {e} | raw_start: {txt[:120]!r}",
        )
