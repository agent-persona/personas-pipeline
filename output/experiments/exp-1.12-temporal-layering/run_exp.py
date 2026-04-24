"""Experiment 1.12: Temporal Layering — synthesize 3 slices and run depth test."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from anthropic import AsyncAnthropic

# Add synthesis to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "synthesis"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "twin"))

from twin.chat import TwinChat, build_persona_system_prompt


WORKTREE = Path(__file__).parent.parent.parent.parent

DEPTH_QUESTION = (
    "How did you feel about setting up webhooks when you first started?"
)


def build_alex_temporal():
    """Hand-craft the three temporal slices for Alex the DevOps engineer."""

    past = {
        "year": 2024,
        "label": "past",
        "name": "Alex the API-First DevOps Engineer",
        "summary": (
            "Alex is a junior DevOps engineer at a 20-person startup, still cutting his "
            "teeth on IaC. In 2024 he's manually SSH-ing into servers, copying webhook URLs "
            "from browser tabs into config files, and spending entire afternoons debugging "
            "why payloads aren't arriving. He knows Terraform exists but runs it with "
            "copy-pasted HCL he barely understands. Every webhook feels like a magic trick "
            "he hasn't mastered yet."
        ),
        "demographics": {
            "age_range": "24-26",
            "gender_distribution": "predominantly male",
            "location_signals": ["US tech hub"],
            "education_level": "Bachelor's in Computer Science",
            "income_bracket": "$75,000-$95,000",
        },
        "firmographics": {
            "company_size": "10-30 employees",
            "industry": "SaaS startup",
            "role_titles": ["DevOps Engineer", "Junior SRE"],
            "tech_stack_signals": ["GitHub", "Bash", "Python scripts", "Basic Terraform"],
        },
        "goals": [
            "Get webhooks working without breaking production",
            "Stop manually SSH-ing into servers for every config change",
            "Learn enough Terraform to not feel like a fraud",
        ],
        "pains": [
            "Webhook payloads silently fail with no error feedback",
            "IaC feels overwhelming — too many moving parts",
            "Copying config values by hand leads to copy-paste errors",
            "No one on the team has set this up before so I'm alone",
        ],
        "motivations": [
            "Proving I can own infra work without hand-holding",
            "Stopping the 2am pages from broken manual processes",
        ],
        "objections": [
            "If the docs don't show a working example I'll waste a whole day",
            "I can't trust webhook delivery if there's no retry story",
        ],
        "channels": ["Stack Overflow", "YouTube tutorials", "GitHub README files", "Discord dev servers"],
        "vocabulary": ["webhook", "payload", "curl", "ngrok", "env vars", "secret", "POST request"],
        "decision_triggers": [
            "Finds a working copy-paste example in the docs",
            "Sees a response body that tells him what went wrong",
        ],
        "sample_quotes": [
            "I spent three hours on this webhook yesterday. Turns out I had an extra slash in the URL.",
            "I know I should be using Terraform for this but honestly the bash script is working so I'm not touching it.",
        ],
        "journey_stages": [
            {
                "stage": "learning",
                "mindset": "Overwhelmed but determined — Googling everything, afraid to break prod",
                "key_actions": ["Reading Stack Overflow", "Copying docs examples", "Testing with ngrok locally"],
                "content_preferences": ["Step-by-step tutorials", "Copy-paste examples", "Error message explanations"],
            }
        ],
        "source_evidence": [
            {"claim": "Junior DevOps at small startup", "record_ids": ["temporal_past"], "field_path": "firmographics", "confidence": 1.0},
        ],
    }

    present = {
        "year": 2026,
        "label": "present",
        "name": "Alex the API-First DevOps Engineer",
        "summary": (
            "Alex is a Senior DevOps Engineer at a 50-200 person fintech company who lives in "
            "the terminal and treats the project management tool as an integration platform first, "
            "a UI second. He spends long sessions deep in API docs, webhook configs, and Terraform "
            "setups — automating everything that can be automated. When something doesn't work as "
            "expected, he'll tell you directly in the support channel."
        ),
        "demographics": {
            "age_range": "28-38",
            "gender_distribution": "predominantly male",
            "location_signals": ["US or EU tech hub, fintech"],
            "education_level": "Bachelor's or Master's in Computer Science",
            "income_bracket": "$120,000-$180,000",
        },
        "firmographics": {
            "company_size": "50-200 employees",
            "industry": "Fintech",
            "role_titles": ["Senior DevOps Engineer", "Platform Engineer", "SRE"],
            "tech_stack_signals": ["GitHub", "Terraform", "Webhooks", "REST API", "GraphQL", "Slack"],
        },
        "goals": [
            "Automate all project state transitions via API so no engineer has to manually update tickets",
            "Integrate the project management tool deeply into the existing GitHub CI/CD pipeline",
            "Set up webhook-driven alerts that push status changes into Slack and internal dashboards",
            "Provision the entire workspace configuration through Terraform for reproducibility",
        ],
        "pains": [
            "GraphQL endpoint has rough edges and schema inconsistencies that break automation scripts",
            "Webhook configuration is time-consuming and lacks sufficient documentation for edge cases",
            "API setup sessions run very long because discovery is fragmented across docs pages",
        ],
        "motivations": [
            "Reducing toil — every manual step is a future incident waiting to happen",
            "Proving that infrastructure-as-code extends to the project management layer",
            "Staying ahead of audit requirements in fintech",
        ],
        "objections": [
            "If the GraphQL schema keeps changing without versioning, I can't depend on it in production pipelines",
            "Terraform provider support is immature — I shouldn't have to maintain my own fork",
            "Webhook retry logic isn't documented well enough for compliance-critical events",
        ],
        "channels": ["GitHub", "Hacker News", "Internal Slack engineering channels", "API docs", "Stack Overflow"],
        "vocabulary": ["idempotent", "webhook", "IaC", "pipeline", "terraform provider", "schema drift", "GraphQL introspection", "CI/CD", "event-driven", "observability", "gitops"],
        "decision_triggers": [
            "Finds a well-documented REST + GraphQL API with a stable schema",
            "Discovers a Terraform provider maintained by the vendor",
            "Gets a clear SLA on webhook delivery reliability and retry behavior",
        ],
        "sample_quotes": [
            "Your REST API is solid but the GraphQL endpoint has some rough edges. Plans to improve the schema?",
            "If I can't manage it in Terraform, it doesn't exist in our infrastructure.",
            "I don't want to click anything. Give me an API endpoint and I'll wire everything else up myself.",
            "Webhooks are great until they silently drop events. What's your retry and dead-letter story?",
        ],
        "journey_stages": [
            {
                "stage": "activation",
                "mindset": "Heads-down builder mode — connecting the tool to every existing system",
                "key_actions": ["api_setup", "webhook_config", "github_integration", "terraform_setup"],
                "content_preferences": ["Code samples", "Webhook event catalog", "Integration guides"],
            }
        ],
        "source_evidence": [
            {"claim": "Senior DevOps at fintech", "record_ids": ["hubspot_000"], "field_path": "firmographics", "confidence": 1.0},
        ],
    }

    future = {
        "year": 2028,
        "label": "future",
        "name": "Alex the API-First DevOps Engineer",
        "summary": (
            "Alex is now a Principal Platform Engineer building internal developer platforms (IDPs) "
            "at a 500-person fintech company. In 2028 he's less hands-on with individual webhooks "
            "and more focused on the abstractions that make webhooks invisible to his 40-engineer "
            "org. He designs self-service tooling, golden-path templates, and enforces API contracts "
            "through policy-as-code. He coaches juniors on the same IaC principles he once stumbled "
            "through, but now with strong opinions baked in."
        ),
        "demographics": {
            "age_range": "30-40",
            "gender_distribution": "predominantly male",
            "location_signals": ["US or EU, likely remote-first fintech"],
            "education_level": "Master's in CS or equivalent experience",
            "income_bracket": "$180,000-$240,000",
        },
        "firmographics": {
            "company_size": "200-1000 employees",
            "industry": "Fintech",
            "role_titles": ["Principal Platform Engineer", "Staff SRE", "Head of Developer Experience"],
            "tech_stack_signals": ["Backstage", "ArgoCD", "Crossplane", "Policy-as-code", "Service mesh", "OpenTelemetry"],
        },
        "goals": [
            "Build an internal developer platform that makes webhook configuration a one-click golden path",
            "Eliminate all manual infra configuration across the engineering org",
            "Enforce API contract compliance via automated policy gates in CI",
            "Mentor junior engineers on the IaC mindset, not just the tools",
        ],
        "pains": [
            "Platform adoption is a people problem, not a technical one",
            "Vendor API instability at scale breaks dozens of teams simultaneously",
            "Abstraction layers are hard to debug when something goes wrong three layers deep",
        ],
        "motivations": [
            "Multiplying his impact by building for 40 engineers instead of solving his own problems",
            "Leaving behind platforms that outlast him",
            "The craft of API design — now he cares about being the good vendor he used to wish existed",
        ],
        "objections": [
            "If the platform dependency has no SLA, I can't build on top of it at this scale",
            "Vendor lock-in in the platform layer is exponentially more painful than at the app layer",
        ],
        "channels": ["Internal RFCs", "PlatformCon", "SLOconf", "KubeCon", "Engineering blogs", "CNCF Slack"],
        "vocabulary": ["golden path", "platform engineering", "developer experience", "contract testing", "chaos engineering", "SLO", "error budget", "policy-as-code", "paved road"],
        "decision_triggers": [
            "Vendor provides an enterprise SLA with incident response commitments",
            "API surface area is small and stable enough to wrap in a platform abstraction",
            "Tooling has a public API that his IDP can call without UI interaction",
        ],
        "sample_quotes": [
            "We stopped configuring webhooks by hand two years ago. Now it's a Backstage plugin and engineers don't even know it's there.",
            "The best infra is invisible. If an engineer has to think about it, I haven't done my job.",
            "I tell juniors: every manual step you accept today is tech debt that will page you at 3am in 18 months.",
        ],
        "journey_stages": [
            {
                "stage": "platform scale",
                "mindset": "Systems thinker — optimizing for the org, not individual tasks",
                "key_actions": ["Designing platform APIs", "Running architecture reviews", "Writing RFCs", "Mentoring"],
                "content_preferences": ["Architecture papers", "Vendor roadmaps", "Case studies at scale", "SLO docs"],
            }
        ],
        "source_evidence": [
            {"claim": "Principal Platform Engineer at scale", "record_ids": ["temporal_future"], "field_path": "firmographics", "confidence": 1.0},
        ],
    }

    return {"past": past, "present": present, "future": future}


def build_temporal_system_prompt(slices: dict) -> str:
    """Build a system prompt that gives the twin all three temporal layers."""
    past = slices["past"]
    present = slices["present"]
    future = slices["future"]

    lines = [
        f"You are {present['name']}. Stay in character at all times.",
        "",
        "## Who you are NOW (2026)",
        present["summary"],
        "",
        "## Your past self (2024)",
        past["summary"],
        "",
        "Things you said then:",
        *(f'- "{q}"' for q in past["sample_quotes"]),
        "",
        "## Your future trajectory (2028)",
        future["summary"],
        "",
        "## How you talk (now)",
        f"You use words like: {', '.join(present['vocabulary'])}.",
        "",
        "Examples of things you say now:",
        *(f'- "{q}"' for q in present["sample_quotes"]),
        "",
        "## Rules",
        "- Answer in first person, in character as your PRESENT self (2026).",
        "- When asked about the past, draw on your PAST slice (2024) — be specific about what you knew and felt then.",
        "- When asked about the future, reflect on your FUTURE trajectory (2028) aspirations.",
        "- Use your vocabulary naturally — don't sound like a chatbot.",
        "- Keep responses under 5 sentences unless asked to elaborate.",
        "- Show the emotional arc between who you were and who you are now.",
        "- Do not break character to mention you are an AI.",
    ]
    return "\n".join(lines)


async def run_twin(system_prompt: str, question: str, client: AsyncAnthropic) -> str:
    """Run a single twin query with a custom system prompt."""
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    return next((b.text for b in response.content if b.type == "text"), "")


async def main():
    client = AsyncAnthropic()

    slices = build_alex_temporal()

    # Baseline: present-only system prompt
    baseline_prompt = build_persona_system_prompt(slices["present"])

    # Temporal: multi-slice system prompt
    temporal_prompt = build_temporal_system_prompt(slices)

    print("Running baseline twin query...")
    baseline_response = await run_twin(baseline_prompt, DEPTH_QUESTION, client)

    print("Running temporal twin query...")
    temporal_response = await run_twin(temporal_prompt, DEPTH_QUESTION, client)

    return {
        "slices": slices,
        "depth_question": DEPTH_QUESTION,
        "baseline_response": baseline_response,
        "temporal_response": temporal_response,
    }


if __name__ == "__main__":
    results = asyncio.run(main())
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out}")
    print(f"\n=== BASELINE ===\n{results['baseline_response']}")
    print(f"\n=== TEMPORAL ===\n{results['temporal_response']}")
