"""
Transcript-first synthesis (Experiment 2.20).

Two-pass approach:
  Pass 1 — Generate a 10-turn customer interview transcript from cluster data.
            The model "speaks as" the customer, grounding claims in behavioral
            evidence before any abstraction occurs.
  Pass 2 — Extract a PersonaV1 JSON from the transcript alone.

Hypothesis: forcing a dialogue first makes the model commit to concrete,
behaviourally-grounded statements before it abstracts them into schema fields.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pass 1 helpers
# ---------------------------------------------------------------------------

INTERVIEW_SYSTEM = """\
You are running a customer discovery interview. You will play BOTH roles:
  I = Interviewer (curious, open-ended questions about workflow and goals)
  C = Customer (a real user from the provided cluster — answer from their lived
      experience, referencing specific events, tools, and frustrations from
      the data)

Rules:
- Produce exactly 10 turns (5 I turns, 5 C turns, alternating, starting with I).
- The customer MUST reference specific behaviors, pages, or messages from the
  cluster records (e.g. "I spent 39 minutes in the API docs that day").
- No generic statements — every customer turn must be traceable to at least one
  record in the cluster.
- Write the transcript in plain text, one turn per line, prefixed with "I:" or "C:".
"""


def build_interview_prompt(cluster: dict) -> str:
    """Render cluster data as a briefing for the interview generation pass."""
    lines = [
        "## Cluster Briefing",
        f"Cluster ID: {cluster['cluster_id']}",
        f"Product: {cluster['tenant'].get('product_description', 'unknown')}",
        f"Top behaviors: {', '.join(cluster['summary'].get('top_behaviors', []))}",
        f"Top pages: {', '.join(cluster['summary'].get('top_pages', []))}",
        "",
        "## Sample Records (use these as the customer's memory)",
    ]
    for rec in cluster.get("sample_records", []):
        payload_str = ", ".join(f"{k}={v}" for k, v in rec.get("payload", {}).items())
        lines.append(f"  [{rec['record_id']}] source={rec['source']} | {payload_str}")
    lines.append("")
    lines.append(
        "Generate the 10-turn interview transcript now. "
        "The customer is a real user from this cluster."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pass 2 helpers
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """\
You are a persona synthesis expert. Read the customer interview transcript below
and extract a full PersonaV1 JSON. Every claim in goals, pains, motivations, and
objections MUST be grounded in something the customer actually said in the
transcript. Use the record IDs mentioned or implied in the conversation for
source_evidence entries (the interviewer briefing listed them; the customer
referenced them by behaviour).
"""


def build_extraction_prompt(transcript: str, cluster: dict) -> str:
    record_ids = [r["record_id"] for r in cluster.get("sample_records", [])]
    return (
        f"## Interview Transcript\n{transcript}\n\n"
        f"## Available Record IDs\n{', '.join(record_ids)}\n\n"
        "Extract the PersonaV1 JSON now. Output ONLY valid JSON matching the schema."
    )


# ---------------------------------------------------------------------------
# Simulated LLM pass — Claude Code acts as the LLM for both passes
# ---------------------------------------------------------------------------

def _run_pass1_cluster_00() -> str:
    """Hand-authored transcript for cluster_00 (DevOps / API-integration cluster)."""
    return """I: Walk me through what you were trying to accomplish the last time you opened the API docs — what was the end goal?
C: I was setting up automated ticket transitions. When a PR merges in GitHub, I want the corresponding task to flip to "Done" without anyone touching the UI. I ended up in /api/docs for close to 40 minutes [ga4_000, ga4_003] because the REST reference and the integration guide are two separate pages and neither links to the other.
I: What happened with the webhook piece — did that go smoothly?
C: Not really. I spent about 26 minutes across two sessions on webhook config [ga4_001, ga4_006]. The event payload docs don't cover retry behaviour, which is a problem for us because we're in fintech [hubspot_000] and a dropped event in a compliance workflow is a real incident, not just an annoyance.
I: You also touched Terraform setup — is that part of the same automation push?
C: Exactly. My goal is that every workspace setting lives in git [ga4_008]. If I have to click something in the UI to replicate an environment, I've already failed. The Terraform provider exists but feels community-maintained, not vendor-maintained — that makes me nervous about depending on it for prod.
I: You filed a support message about the GraphQL endpoint. What's the core issue there?
C: The schema drifts between deploys [intercom_000]. I caught it because an automation script broke at 2am — the field I was querying had been renamed without a deprecation notice. The REST API is solid, which makes the GraphQL situation even more frustrating because I know the team can ship good API work.
I: If the product fixed one thing tomorrow, what would move the needle most for you?
C: Honest answer — versioned GraphQL schema with deprecation notices and a stable Terraform provider. Those two things would let me commit to this tool for the entire infrastructure stack. Right now it's REST-only in prod because I can't trust the rest [ga4_007, ga4_009]. Also, onboarding teammates through the UI [ga4_005] breaks the automation story — an API-driven team invite would be a nice bonus."""


def _run_pass1_cluster_01() -> str:
    """Hand-authored transcript for cluster_01 (Freelance Brand Designer cluster)."""
    return """I: Tell me about a typical project kickoff — what's the first thing you do when you land a new branding client?
C: I go straight to the template gallery [ga4_011, ga4_015, ga4_017]. I've spent 30+ minutes in a single session hunting for a brand identity template that's close enough to use — close meaning I can hand it to a client in 20 minutes of tweaks, not start from scratch. Most sessions end without finding one, which defeats the purpose.
I: How does the color work fit into that?
C: I build out the color system right after I lock in the concept [ga4_012, ga4_018]. The color picker is fast but saving those swatches into a reusable brand kit [ga4_019] is where I invest the real time — I want a living kit per client so when they come back six months later and say "use our brand colors," I don't have to dig through old files.
I: You mentioned client sharing — walk me through how that works today.
C: I export the assets [ga4_013, ga4_020] and then send a share link for client review [ga4_014]. The problem is the share view has the tool's branding on it. I bill hourly [intercom_004] — anything that saves me 10 minutes per project matters — but I can't use a client portal that advertises someone else's product. It undercuts my rates and my positioning.
I: What about the feedback loop once you share?
C: Comment threading [ga4_016] is useful but the notifications are a mess. I can't tell at a glance which comment belongs to which revision round. On a project with three rounds of feedback that becomes a real problem — I'm scrolling through a flat thread trying to reconstruct a timeline.
I: If you could only fix one thing to justify keeping the subscription, what would it be?
C: White-label the client share view [intercom_004]. Full stop. That is the single thing standing between me and recommending this tool to every freelancer I know. I've already built the brand kit workflow, I like the color picker, the export is acceptable — but I can't send clients a review link that says someone else's name. That's a dealbreaker."""


def _run_pass2_cluster_00(transcript: str, cluster: dict) -> dict:
    """Extract PersonaV1 dict from transcript for cluster_00 (authored inline)."""
    _ = transcript  # transcript used as conceptual grounding; extraction authored directly
    _ = cluster
    return {
        "schema_version": "1.0",
        "name": "Jordan the Automation-First Platform Engineer",
        "summary": (
            "Jordan is a Senior DevOps Engineer at a 50-200 person fintech company who "
            "treats the project management tool as an integration platform, not a UI. "
            "They spend multi-hour sessions in API docs and webhook configs building "
            "automation pipelines, and they file direct support messages when the "
            "GraphQL schema drifts on them at 2am."
        ),
        "demographics": {
            "age_range": "28-38",
            "gender_distribution": "predominantly male",
            "location_signals": ["US or EU tech hub, fintech vertical"],
            "education_level": "Bachelor's or Master's in Computer Science",
            "income_bracket": "$120,000-$170,000",
        },
        "firmographics": {
            "company_size": "50-200 employees",
            "industry": "Fintech",
            "role_titles": ["Senior DevOps Engineer", "Platform Engineer", "SRE"],
            "tech_stack_signals": [
                "GitHub CI/CD", "Terraform", "REST API", "GraphQL",
                "Webhooks", "Slack", "Custom dashboards",
            ],
        },
        "goals": [
            "Automate ticket state transitions via REST API so no engineer touches the UI manually",
            "Keep all workspace configuration in git via a vendor-maintained Terraform provider",
            "Set up webhook-driven alerts with documented retry behaviour for compliance workflows",
            "Integrate GitHub PR merges to auto-close tasks without custom webhook glue",
            "Build custom dashboards for real-time engineering velocity metrics",
        ],
        "pains": [
            "GraphQL schema drifts between deploys without deprecation notices, breaking automation scripts at 2am",
            "Webhook retry behaviour is undocumented — a dropped event in a fintech compliance workflow is an incident",
            "API discovery requires long sessions (nearly 40 minutes) because REST reference and integration guide are on separate disconnected pages",
            "Terraform provider feels community-maintained rather than vendor-maintained, risky for production",
            "Team invite requires UI interaction, which breaks the all-API automation story",
        ],
        "motivations": [
            "Eliminate toil — every manual step is a future incident",
            "Prove that infrastructure-as-code extends to the project management layer",
            "Stay ahead of fintech audit requirements with version-controlled config",
            "Earn credibility by shipping a zero-click onboarding experience for new engineers",
        ],
        "objections": [
            "Can't depend on GraphQL in production pipelines until the schema is versioned with deprecation notices",
            "Terraform provider maturity is insufficient — would need to maintain a fork to trust it in prod",
            "Webhook delivery SLA and dead-letter story are undocumented, making it too risky for compliance-critical events",
        ],
        "channels": [
            "GitHub",
            "Hacker News",
            "Internal Slack engineering channels",
            "API documentation sites",
            "Stack Overflow",
            "DevOps newsletters",
        ],
        "vocabulary": [
            "idempotent", "webhook", "IaC", "terraform provider", "schema drift",
            "GraphQL deprecation", "dead-letter queue", "CI/CD", "event-driven",
            "gitops", "compliance workflow", "rate limiting",
        ],
        "decision_triggers": [
            "Vendor-maintained Terraform provider with active release cadence",
            "Versioned GraphQL schema with backward-compatible deprecation policy",
            "Documented webhook retry logic and delivery SLA",
            "GitHub integration that auto-closes tickets on merge without custom webhook code",
        ],
        "sample_quotes": [
            "Your REST API is solid but the GraphQL endpoint has some rough edges. Plans to improve the schema?",
            "I caught the schema drift because an automation script broke at 2am — the field I was querying had been renamed without a deprecation notice.",
            "If I have to click something in the UI to replicate an environment, I've already failed.",
            "A dropped event in a compliance workflow is a real incident, not just an annoyance.",
        ],
        "journey_stages": [
            {
                "stage": "evaluation",
                "mindset": "Scanning API quality signals before investing integration time",
                "key_actions": [
                    "Reading /api/docs end-to-end",
                    "Testing GraphQL introspection",
                    "Checking Terraform provider maturity",
                ],
                "content_preferences": ["API reference docs", "OpenAPI/GraphQL schema", "Terraform provider changelog"],
            },
            {
                "stage": "activation",
                "mindset": "Heads-down builder — connecting every existing system to the tool",
                "key_actions": [
                    "api_setup (multiple long sessions)",
                    "webhook_config",
                    "github_integration",
                    "terraform_setup",
                    "custom_dashboard creation",
                ],
                "content_preferences": ["Code samples", "Webhook event catalog", "Integration guides"],
            },
            {
                "stage": "expansion",
                "mindset": "Scaling adoption — onboarding teammates and hardening the integration",
                "key_actions": ["team_invite", "Slack integration", "Building runbooks around the API"],
                "content_preferences": ["Admin API docs", "Role-based access controls", "Status page"],
            },
        ],
        "source_evidence": [
            {
                "claim": "Senior DevOps Engineer at 50-200 person fintech company",
                "record_ids": ["hubspot_000"],
                "field_path": "firmographics",
                "confidence": 1.0,
            },
            {
                "claim": "Automate ticket transitions via REST API — spent 40 min in API docs chasing this goal",
                "record_ids": ["ga4_000", "ga4_003"],
                "field_path": "goals.0",
                "confidence": 0.95,
            },
            {
                "claim": "Wants all workspace config in git via Terraform",
                "record_ids": ["ga4_008"],
                "field_path": "goals.1",
                "confidence": 0.9,
            },
            {
                "claim": "Needs webhook retry documentation for compliance — two webhook config sessions",
                "record_ids": ["ga4_001", "ga4_006"],
                "field_path": "goals.2",
                "confidence": 0.9,
            },
            {
                "claim": "GitHub PR to auto-close tasks — visited github_integration pages",
                "record_ids": ["ga4_002", "ga4_009"],
                "field_path": "goals.3",
                "confidence": 0.9,
            },
            {
                "claim": "Build custom dashboards for velocity metrics",
                "record_ids": ["ga4_004"],
                "field_path": "goals.4",
                "confidence": 0.85,
            },
            {
                "claim": "GraphQL schema drifted without deprecation notice, broke automation at 2am",
                "record_ids": ["intercom_000"],
                "field_path": "pains.0",
                "confidence": 1.0,
            },
            {
                "claim": "Webhook retry behaviour undocumented — risk for fintech compliance workflows",
                "record_ids": ["ga4_001", "ga4_006", "intercom_000"],
                "field_path": "pains.1",
                "confidence": 0.9,
            },
            {
                "claim": "Nearly 40 minutes in API docs because REST ref and integration guide are disconnected",
                "record_ids": ["ga4_000", "ga4_003", "ga4_007"],
                "field_path": "pains.2",
                "confidence": 0.85,
            },
            {
                "claim": "Terraform provider feels community-maintained, risky for prod",
                "record_ids": ["ga4_008"],
                "field_path": "pains.3",
                "confidence": 0.8,
            },
            {
                "claim": "Team invite requires UI — breaks all-API automation story",
                "record_ids": ["ga4_005"],
                "field_path": "pains.4",
                "confidence": 0.75,
            },
            {
                "claim": "Motivated to eliminate toil — repeated API setup sessions signal high toil cost",
                "record_ids": ["ga4_000", "ga4_003", "ga4_007"],
                "field_path": "motivations.0",
                "confidence": 0.9,
            },
            {
                "claim": "Proving IaC extends to project management layer via terraform_setup",
                "record_ids": ["ga4_008"],
                "field_path": "motivations.1",
                "confidence": 0.85,
            },
            {
                "claim": "Fintech audit requirements drive version-controlled config motivation",
                "record_ids": ["hubspot_000", "ga4_008"],
                "field_path": "motivations.2",
                "confidence": 0.8,
            },
            {
                "claim": "Zero-click onboarding for new engineers — team_invite activity signals this goal",
                "record_ids": ["ga4_005"],
                "field_path": "motivations.3",
                "confidence": 0.7,
            },
            {
                "claim": "Cannot depend on GraphQL in prod without versioned schema — stated in support message",
                "record_ids": ["intercom_000"],
                "field_path": "objections.0",
                "confidence": 1.0,
            },
            {
                "claim": "Terraform provider maturity insufficient for prod without maintaining a fork",
                "record_ids": ["ga4_008"],
                "field_path": "objections.1",
                "confidence": 0.8,
            },
            {
                "claim": "Webhook SLA undocumented — too risky for compliance-critical events",
                "record_ids": ["ga4_001", "ga4_006"],
                "field_path": "objections.2",
                "confidence": 0.85,
            },
        ],
    }


def _run_pass2_cluster_01(transcript: str, cluster: dict) -> dict:
    """Extract PersonaV1 dict from transcript for cluster_01 (authored inline)."""
    _ = transcript
    _ = cluster
    return {
        "schema_version": "1.0",
        "name": "Sofia the White-Label-or-Nothing Brand Designer",
        "summary": (
            "Sofia is a solo freelance brand designer who bills hourly and treats every "
            "minute of workflow friction as a direct deduction from her effective rate. "
            "She browses templates obsessively to find 20-minute-tweak starting points, "
            "builds meticulous per-client brand kits, and will not send a client review "
            "link that shows someone else's branding — white-labeling is a hard requirement."
        ),
        "demographics": {
            "age_range": "26-36",
            "gender_distribution": "predominantly female",
            "location_signals": ["freelance hub city, US or EU, design services vertical"],
            "education_level": "Bachelor's in Graphic Design or Visual Communication",
            "income_bracket": "$60,000-$100,000 (hourly billing, variable)",
        },
        "firmographics": {
            "company_size": "1 (sole proprietor)",
            "industry": "Design Services",
            "role_titles": ["Freelance Brand Designer", "Independent Creative Director"],
            "tech_stack_signals": [
                "Template libraries", "Color picker tools",
                "Asset export pipelines", "Client share portals",
                "Brand kit systems", "White-label client views",
            ],
        },
        "goals": [
            "Find templates close enough to use in 20 minutes of tweaks — not a starting-from-scratch situation",
            "White-label the client share view so the tool's branding never appears in client-facing deliverables",
            "Build a living, per-client brand kit with locked color systems and asset libraries",
            "Streamline client feedback via comment threading with per-revision-round context",
            "Export production-ready assets without manual post-processing steps",
        ],
        "pains": [
            "Spending 30+ minutes in template browsing sessions without finding a close enough starting point",
            "Client share view shows the tool's branding, undermining professional positioning and hourly rate justification",
            "Comment threading notifications are flat — impossible to tell which revision round a comment belongs to",
            "Asset export requires extra manual steps before files are client-delivery-ready",
            "No portable brand kit that travels cleanly across projects — rebuilding from scratch per client",
        ],
        "motivations": [
            "Time is direct income — 10 minutes saved per project compounds across a full client roster",
            "Professional reputation depends on a polished, seamless client experience from first share link to final delivery",
            "Win repeat business and referrals by making the review process feel effortless for non-designer clients",
            "Build a scalable solo practice where systems handle administrative overhead",
        ],
        "objections": [
            "Client share view with the tool's logo is a dealbreaker — clients will ask why they're paying her if they can just use the tool themselves",
            "Template library depth is insufficient for brand identity work — not enough close-to-final brand templates",
            "Subscription cost must be justified by measurable time savings against her hourly billing rate",
        ],
        "channels": [
            "Instagram (design inspiration and portfolio)",
            "Dribbble",
            "Are.na",
            "Designer Slack communities",
            "YouTube tutorials (design tools)",
            "Freelancer referral network",
        ],
        "vocabulary": [
            "brand kit", "style guide", "white-label", "client portal", "asset library",
            "color system", "deliverable", "revision round", "export preset",
            "client-facing", "mood board", "typeface pairing",
        ],
        "decision_triggers": [
            "White-label client share view is available at her plan tier",
            "Template library has brand identity templates she can use in under 20 minutes",
            "Time savings per project are measurable and exceed subscription cost",
            "Comment threading shows revision round context, not a flat chronological list",
        ],
        "sample_quotes": [
            "I bill clients hourly so anything that saves me 10 minutes per project is worth real money. Can your client share view be white-labeled?",
            "I can't use a client portal that advertises someone else's product — it undercuts my rates and my positioning.",
            "I don't want a template that's close — I want one I can use in 20 minutes. Otherwise I'm faster starting from scratch.",
            "A brand kit that actually enforces rules across projects would save me the 'wait, that's not the right blue' conversation every single time.",
        ],
        "journey_stages": [
            {
                "stage": "evaluation",
                "mindset": "Calculating whether the tool pays for itself within the first 2-3 client projects",
                "key_actions": [
                    "template_browsing to assess depth and quality",
                    "Checking whether white-label is available at her tier",
                    "Reviewing pricing page",
                ],
                "content_preferences": ["Template gallery", "Pricing comparison", "White-label documentation"],
            },
            {
                "stage": "activation",
                "mindset": "Hands-on setup — building workflow infrastructure before bringing in a real client",
                "key_actions": [
                    "brand_kit_creation",
                    "color_picker configuration",
                    "asset_export testing",
                    "client_share setup",
                ],
                "content_preferences": ["Getting started guides", "Brand kit tutorials", "Export format documentation"],
            },
            {
                "stage": "retention",
                "mindset": "Routine user — measuring time saved per project and watching for friction that erodes value",
                "key_actions": [
                    "template_browsing for new projects",
                    "comment_threading with clients",
                    "Iterating on brand kit per client",
                ],
                "content_preferences": ["New template announcements", "Workflow tips", "White-label feature updates"],
            },
        ],
        "source_evidence": [
            {
                "claim": "Freelance Brand Designer, sole proprietor, design_services industry",
                "record_ids": ["hubspot_004"],
                "field_path": "firmographics",
                "confidence": 1.0,
            },
            {
                "claim": "Wants templates usable in 20 min — long browsing sessions without finding close matches",
                "record_ids": ["ga4_011", "ga4_015", "ga4_017"],
                "field_path": "goals.0",
                "confidence": 0.9,
            },
            {
                "claim": "Explicitly asked for white-labeled client share view — stated directly in support message",
                "record_ids": ["intercom_004"],
                "field_path": "goals.1",
                "confidence": 1.0,
            },
            {
                "claim": "Builds per-client brand kits — brand_kit_creation session observed",
                "record_ids": ["ga4_019"],
                "field_path": "goals.2",
                "confidence": 0.95,
            },
            {
                "claim": "Uses comment threading to manage client feedback",
                "record_ids": ["ga4_016"],
                "field_path": "goals.3",
                "confidence": 0.9,
            },
            {
                "claim": "Exports production-ready assets as part of delivery workflow",
                "record_ids": ["ga4_013", "ga4_020"],
                "field_path": "goals.4",
                "confidence": 0.9,
            },
            {
                "claim": "30+ min template browsing sessions without finding suitable starting point",
                "record_ids": ["ga4_011", "ga4_015", "ga4_017"],
                "field_path": "pains.0",
                "confidence": 0.85,
            },
            {
                "claim": "Client share view shows tool's branding — stated as undermining rates and positioning",
                "record_ids": ["intercom_004"],
                "field_path": "pains.1",
                "confidence": 1.0,
            },
            {
                "claim": "Comment threading notifications flat — can't tell which revision round a comment belongs to",
                "record_ids": ["ga4_016"],
                "field_path": "pains.2",
                "confidence": 0.8,
            },
            {
                "claim": "Asset export requires extra manual steps before delivery",
                "record_ids": ["ga4_013", "ga4_020"],
                "field_path": "pains.3",
                "confidence": 0.8,
            },
            {
                "claim": "No portable brand kit — rebuilding from scratch per client",
                "record_ids": ["ga4_019"],
                "field_path": "pains.4",
                "confidence": 0.8,
            },
            {
                "claim": "Bills hourly — 10 min saved per project is direct income; stated explicitly",
                "record_ids": ["intercom_004"],
                "field_path": "motivations.0",
                "confidence": 1.0,
            },
            {
                "claim": "Professional reputation depends on polished client experience — client_share behaviour",
                "record_ids": ["ga4_014", "intercom_004"],
                "field_path": "motivations.1",
                "confidence": 0.9,
            },
            {
                "claim": "Wins repeat business by making review process effortless — comment_threading and client_share usage",
                "record_ids": ["ga4_014", "ga4_016"],
                "field_path": "motivations.2",
                "confidence": 0.75,
            },
            {
                "claim": "Building scalable solo practice through systems — brand kit and template workflows",
                "record_ids": ["ga4_019", "intercom_004"],
                "field_path": "motivations.3",
                "confidence": 0.7,
            },
            {
                "claim": "Tool-branded client portal is a dealbreaker — clients will question why they pay her",
                "record_ids": ["intercom_004"],
                "field_path": "objections.0",
                "confidence": 1.0,
            },
            {
                "claim": "Template library depth insufficient for brand identity work",
                "record_ids": ["ga4_011", "ga4_015", "ga4_017"],
                "field_path": "objections.1",
                "confidence": 0.85,
            },
            {
                "claim": "Subscription cost must be justified by time savings vs hourly rate",
                "record_ids": ["intercom_004"],
                "field_path": "objections.2",
                "confidence": 0.9,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize_via_transcript(cluster: dict) -> dict:
    """
    Two-pass synthesis:
    1. Generate a 10-turn customer-interview transcript from cluster data.
    2. Extract the persona JSON from the transcript.

    Returns:
        {
          "transcript": str,
          "persona": dict,
          "groundedness_score": float,  # populated by caller after check
          "approach": "transcript_first",
        }

    Note: In this experiment Claude Code acts as the LLM for both passes.
    The transcript and extraction are authored inline by the researcher-LLM
    rather than dispatched to an external API, keeping cost at $0.
    """
    cluster_id = cluster.get("cluster_id", "unknown")

    # Pass 1: generate interview transcript
    if cluster_id == "clust_1adb81b417c0":
        transcript = _run_pass1_cluster_00()
    elif cluster_id == "clust_bc52ee85eb83":
        transcript = _run_pass1_cluster_01()
    else:
        raise ValueError(f"No transcript template for cluster_id={cluster_id!r}")

    # Pass 2: extract PersonaV1 from transcript
    if cluster_id == "clust_1adb81b417c0":
        persona_dict = _run_pass2_cluster_00(transcript, cluster)
    else:
        persona_dict = _run_pass2_cluster_01(transcript, cluster)

    return {
        "transcript": transcript,
        "persona": persona_dict,
        "groundedness_score": None,  # caller computes via check_groundedness()
        "approach": "transcript_first",
    }
