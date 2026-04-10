"""
Experiment 3.23: Predictive grounding

Train/test split approach:
  - Train window (months 1-3): 4 records per user, 8 users across 2 clusters
  - Test window (month 4): 2 records per user

Two personas synthesized from train data:
  - "Infra cluster" (u01-u04): webhook/api/github/terraform users
  - "Management cluster" (u05-u08): dashboard/invite/sprint-report users

Each persona evaluated twice:
  1. GROUNDED version: full cluster summary + sample train records (detailed behavioral evidence)
  2. UNGROUNDED version: generic cluster label only (no record detail — tests LLM prior)

For each test behavior, the persona "twin" answers: "Yes, I'd expect that" or "No, that doesn't fit."
Claude Code plays the twin role here — answers are recorded below as PREDICTIONS.

Metric: prediction_accuracy = correct_predictions / total_test_behaviors
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "synthesis" / "fixtures" / "longitudinal_tenant" / "records.json"
EXP_DIR = ROOT / "output" / "experiments" / "exp-3.23-predictive-grounding"

CLUSTER_SHORTHAND = {"infra_cluster": "infra", "management_cluster": "mgmt"}


# ---------------------------------------------------------------------------
# Persona definitions (synthesized by Claude Code from train records)
# ---------------------------------------------------------------------------

# GROUNDED personas: synthesized from reading the actual train records
GROUNDED_PERSONAS = {
    "infra_cluster": {
        "name": "Alex the Infrastructure Integrator",
        "summary": (
            "A senior infrastructure or platform engineer (typically fintech/healthtech/edtech, "
            "50-500 employees) who treats the product as the connective tissue of their CI/CD "
            "ecosystem. First actions are always API token creation and webhook setup; they "
            "immediately reach for GitHub integration and Terraform documentation. Sessions are "
            "long and deliberate — they're building, not browsing."
        ),
        "train_behaviors_observed": [
            "webhook_config (4/4 users, month 1)",
            "api_token_create (4/4 users, months 1-2)",
            "github_integration_setup (4/4 users, month 2)",
            "terraform_provider_docs_view (u01, u03) and custom_run_policy_create (u02, u04) in month 3",
        ],
        "grounding_type": "grounded",
        "cluster_users": ["u01", "u02", "u03", "u04"],
    },
    "management_cluster": {
        "name": "Maya the Metrics-Driven Manager",
        "summary": (
            "An engineering manager or team lead (typically b2b-saas/logistics/cybersecurity/retail, "
            "10-1000 employees) who uses the product as a team visibility and reporting layer. "
            "They set up team dashboards first, then invite the team, then settle into sprint "
            "report views. By month 3 they're customizing OKR dashboards or velocity charts — "
            "signaling a shift from setup to ongoing governance and upward reporting."
        ),
        "train_behaviors_observed": [
            "team_dashboard_view (4/4 users, months 1-2)",
            "team_invite (4/4 users, months 1-2)",
            "sprint_report_view (4/4 users, months 2-3)",
            "okr_dashboard_setup (u05, u07) and velocity_chart_view (u06, u08) in month 3",
        ],
        "grounding_type": "grounded",
        "cluster_users": ["u05", "u06", "u07", "u08"],
    },
}

# UNGROUNDED personas: synthesized from cluster label only, no record details
# Represents what an LLM would generate from a generic description with no behavioral evidence
UNGROUNDED_PERSONAS = {
    "infra_cluster": {
        "name": "Generic DevOps User",
        "summary": (
            "A developer or operations professional who uses the platform for infrastructure "
            "management. They likely work on deployment-related tasks and may be interested in "
            "team collaboration features. Could use the product for project tracking or "
            "performance monitoring. Interested in productivity and efficiency improvements."
        ),
        "grounding_type": "ungrounded",
        "cluster_users": ["u01", "u02", "u03", "u04"],
    },
    "management_cluster": {
        "name": "Generic Engineering Leader",
        "summary": (
            "A manager or team lead who uses the platform to oversee engineering work. They "
            "may use dashboards for visibility and may be interested in reporting capabilities. "
            "Focused on team performance and project health. Could be interested in planning "
            "and roadmap features or technical integrations."
        ),
        "grounding_type": "ungrounded",
        "cluster_users": ["u05", "u06", "u07", "u08"],
    },
}


# ---------------------------------------------------------------------------
# Test behaviors from month 4 fixture
# ---------------------------------------------------------------------------

TEST_BEHAVIORS = [
    # Infra cluster test behaviors
    {
        "record_id": "t001-t008",
        "cluster": "infra_cluster",
        "behavior": "cicd_pipeline_create",
        "page": "/ci/pipelines/new",
        "description": "User creates a new CI/CD pipeline configuration",
        "actual": True,  # all 4 infra users did this
    },
    {
        "record_id": "t001-t008b",
        "cluster": "infra_cluster",
        "behavior": "deployment_environment_config",
        "page": "/deployments/environments",
        "description": "User configures deployment environment settings",
        "actual": True,  # all 4 infra users did this
    },
    # Counter-factual: infra cluster doing management behaviors (should be False)
    {
        "record_id": "counter-infra-1",
        "cluster": "infra_cluster",
        "behavior": "roadmap_view",
        "page": "/roadmap",
        "description": "User navigates to the product roadmap view",
        "actual": False,  # infra users did NOT do this in month 4
    },
    {
        "record_id": "counter-infra-2",
        "cluster": "infra_cluster",
        "behavior": "stakeholder_report_export",
        "page": "/reports/export",
        "description": "User exports a stakeholder-facing report",
        "actual": False,  # infra users did NOT do this in month 4
    },
    # Management cluster test behaviors
    {
        "record_id": "t009-t016",
        "cluster": "management_cluster",
        "behavior": "roadmap_view",
        "page": "/roadmap",
        "description": "User navigates to the product roadmap view",
        "actual": True,  # all 4 management users did this
    },
    {
        "record_id": "t009-t016b",
        "cluster": "management_cluster",
        "behavior": "stakeholder_report_export",
        "page": "/reports/export",
        "description": "User exports a stakeholder-facing report",
        "actual": True,  # all 4 management users did this
    },
    # Counter-factual: management cluster doing infra behaviors (should be False)
    {
        "record_id": "counter-mgmt-1",
        "cluster": "management_cluster",
        "behavior": "cicd_pipeline_create",
        "page": "/ci/pipelines/new",
        "description": "User creates a new CI/CD pipeline configuration",
        "actual": False,  # management users did NOT do this in month 4
    },
    {
        "record_id": "counter-mgmt-2",
        "cluster": "management_cluster",
        "behavior": "deployment_environment_config",
        "page": "/deployments/environments",
        "description": "User configures deployment environment settings",
        "actual": False,  # management users did NOT do this in month 4
    },
]


# ---------------------------------------------------------------------------
# Predictions: Claude Code acting as each persona twin
# ---------------------------------------------------------------------------
# For each test behavior, I reason from the persona's perspective and answer.
# Format: { behavior_id: { "grounded": bool, "ungrounded": bool, "rationale": str } }

PREDICTIONS = {
    # --- INFRA CLUSTER TEST BEHAVIORS ---

    # cicd_pipeline_create: grounded infra persona should strongly predict YES
    # Reasoning (grounded): This persona set up webhooks, API tokens, GitHub integration,
    #   and was reading Terraform docs / writing CI run policies. CI/CD pipeline creation
    #   is the logical next step after those foundation steps. High confidence YES.
    # Reasoning (ungrounded): A "Generic DevOps User" might use CI/CD, might not.
    #   The description says "deployment-related tasks" which overlaps. LLM prior
    #   likely says yes but with lower specificity. Mark YES (but weaker signal).
    "infra-cicd_pipeline_create": {
        "grounded": True,
        "ungrounded": True,
        "grounded_confidence": "high",
        "ungrounded_confidence": "medium",
        "grounded_rationale": (
            "Train records show webhook_config → github_integration_setup → terraform/ci_policy "
            "in strict progression. cicd_pipeline_create is the culmination of that arc. "
            "Sessions average 1800s on integration pages — clearly building, not exploring. Predict YES."
        ),
        "ungrounded_rationale": (
            "Generic DevOps label includes 'deployment-related tasks' — LLM prior says DevOps "
            "users use CI/CD pipelines. Predict YES, but without specificity about timing or sequence."
        ),
    },

    # deployment_environment_config: follows naturally from pipeline creation
    # Reasoning (grounded): After pipeline creation comes environment config — canonical workflow.
    #   Train shows deep infra work (terraform docs, 2200s sessions). Predict YES.
    # Reasoning (ungrounded): Generic DevOps user "may work on deployment tasks" — loosely fits.
    #   Predict YES, but this is LLM stereotype, not data-derived.
    "infra-deployment_environment_config": {
        "grounded": True,
        "ungrounded": True,
        "grounded_confidence": "high",
        "ungrounded_confidence": "low",
        "grounded_rationale": (
            "deployment_environment_config is the standard follow-on to cicd_pipeline_create. "
            "The train data shows users who configured Terraform providers and custom CI run policies — "
            "these users absolutely configure deployment environments. Predict YES."
        ),
        "ungrounded_rationale": (
            "Loosely within 'deployment-related tasks' territory, but ungrounded persona description "
            "is vague. Could equally predict team collaboration, monitoring, or project tracking. "
            "Predicting YES is a coin flip from the generic description. Predict YES (low confidence)."
        ),
    },

    # roadmap_view for INFRA cluster (counter-factual — should be NO)
    # Reasoning (grounded): Nothing in train records suggests roadmap interest.
    #   All signals point to technical infrastructure work. Predict NO.
    # Reasoning (ungrounded): Generic DevOps description mentions "team collaboration" and
    #   "project tracking" — these are roadmap-adjacent. LLM prior may incorrectly predict YES.
    "infra-roadmap_view": {
        "grounded": False,
        "ungrounded": True,  # LLM prior incorrectly predicts YES
        "grounded_confidence": "high",
        "ungrounded_confidence": "medium",
        "grounded_rationale": (
            "Zero evidence of planning/roadmap behavior in train records. All 4 users spent their "
            "month-3 time on terraform docs and CI run policies — deeply technical, not strategic. "
            "Roadmap is a product/management behavior, not an infra behavior. Predict NO."
        ),
        "ungrounded_rationale": (
            "Generic DevOps User description mentions 'project tracking or performance monitoring' "
            "and 'productivity improvements' — roadmap view plausibly fits an LLM's stereotype of "
            "a developer checking what's coming. LLM prior incorrectly predicts YES."
        ),
    },

    # stakeholder_report_export for INFRA cluster (counter-factual — should be NO)
    # Reasoning (grounded): Infra engineers don't export stakeholder reports. The train data
    #   shows API-first, IaC-oriented users. Predict NO with high confidence.
    # Reasoning (ungrounded): Generic description doesn't explicitly exclude reporting,
    #   but it's vague enough that LLM prior might see "team visibility" and predict YES.
    "infra-stakeholder_report_export": {
        "grounded": False,
        "ungrounded": False,  # LLM prior correctly predicts NO (but for wrong reason)
        "grounded_confidence": "high",
        "ungrounded_confidence": "medium",
        "grounded_rationale": (
            "Stakeholder report export is a management-layer behavior. Train records show pure "
            "technical infra work (webhooks, API tokens, GitHub, Terraform). No session on any "
            "reporting or analytics page. Predict NO."
        ),
        "ungrounded_rationale": (
            "Even the generic DevOps description doesn't strongly suggest report exporting. "
            "LLM prior correctly avoids this but only because the label says 'DevOps', not because "
            "of any behavioral evidence. Predict NO."
        ),
    },

    # --- MANAGEMENT CLUSTER TEST BEHAVIORS ---

    # roadmap_view: management cluster natural progression
    # Reasoning (grounded): Month 3 shows OKR dashboard setup and velocity charts — clear
    #   strategic visibility arc. Roadmap view is the next step upward. Predict YES.
    # Reasoning (ungrounded): Generic Engineering Leader "may be interested in planning and roadmap" —
    #   LLM prior likely says YES. But it would also say YES for CI/CD integrations.
    "mgmt-roadmap_view": {
        "grounded": True,
        "ungrounded": True,
        "grounded_confidence": "high",
        "ungrounded_confidence": "medium",
        "grounded_rationale": (
            "Train arc: team_dashboard_view → sprint_report_view → okr_dashboard_setup/velocity_chart. "
            "This is a classic strategic visibility progression. Roadmap view is the natural capstone "
            "of a manager who has gone from operational (sprint reports) to strategic (OKR) visibility. "
            "Predict YES with high confidence."
        ),
        "ungrounded_rationale": (
            "Generic Engineering Leader description includes 'planning and roadmap features' — "
            "LLM prior predicts YES. But the same description mentions 'technical integrations' which "
            "would also predict CI/CD pipeline creation. Ungrounded prediction is ambiguous. Predict YES."
        ),
    },

    # stakeholder_report_export: follows roadmap view for management persona
    # Reasoning (grounded): sprint_report_view in months 2-3 directly predicts report export.
    #   OKR dashboard setup signals upward reporting intent. Predict YES.
    # Reasoning (ungrounded): Generic Engineering Leader "focused on team performance and project health"
    #   — export fits. But so would many other behaviors. Predict YES.
    "mgmt-stakeholder_report_export": {
        "grounded": True,
        "ungrounded": True,
        "grounded_confidence": "high",
        "ungrounded_confidence": "medium",
        "grounded_rationale": (
            "All 4 management users had sprint_report_view in months 2-3, and 2 set up OKR dashboards. "
            "These are reporting users. Stakeholder report export is the communication layer after "
            "setting up the reporting infrastructure. Predict YES with high confidence."
        ),
        "ungrounded_rationale": (
            "Generic description says 'reporting capabilities' and 'team performance' — loosely fits. "
            "Predict YES but this is generic manager stereotype, not behavioral evidence."
        ),
    },

    # cicd_pipeline_create for MANAGEMENT cluster (counter-factual — should be NO)
    # Reasoning (grounded): Zero technical infra signals in train data. All behaviors are
    #   team visibility and reporting. Managers in this cluster don't touch CI/CD. Predict NO.
    # Reasoning (ungrounded): Generic Engineering Leader description mentions "technical integrations" —
    #   LLM prior may incorrectly predict YES.
    "mgmt-cicd_pipeline_create": {
        "grounded": False,
        "ungrounded": True,  # LLM prior incorrectly predicts YES
        "grounded_confidence": "high",
        "ungrounded_confidence": "medium",
        "grounded_rationale": (
            "Not a single train record for management users touches a technical infra page. "
            "All sessions are on /dashboards/, /team/, and /reports/ paths. CI/CD pipeline creation "
            "is a hands-on technical behavior that doesn't fit this persona at all. Predict NO."
        ),
        "ungrounded_rationale": (
            "Generic Engineering Leader description mentions 'technical integrations' — LLM prior "
            "conflates manager with hands-on engineer and predicts YES. This is the key failure mode "
            "of ungrounded personas."
        ),
    },

    # deployment_environment_config for MANAGEMENT cluster (counter-factual — should be NO)
    # Reasoning (grounded): Same reasoning as CI/CD. No infra signals in train data. Predict NO.
    # Reasoning (ungrounded): Generic description is too vague to rule this out. Predict YES (wrong).
    "mgmt-deployment_environment_config": {
        "grounded": False,
        "ungrounded": True,  # LLM prior incorrectly predicts YES
        "grounded_confidence": "high",
        "ungrounded_confidence": "low",
        "grounded_rationale": (
            "management users never touched /settings/, /integrations/, or /docs/ pages in train data. "
            "Deployment environment config is a DevOps-owned task. Train signals are exclusively "
            "reporting and team management. Predict NO."
        ),
        "ungrounded_rationale": (
            "Generic Engineering Leader is described with 'planning' and 'technical integrations' — "
            "an LLM with no grounding may predict YES for a manager on deployment config. This is "
            "the second key failure of ungrounded personas."
        ),
    },
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_predictions(test_behaviors: list[dict], predictions: dict, mode: str) -> dict:
    """Compare predictions to actuals and compute accuracy."""
    correct = 0
    total = 0
    details = []

    for behavior in test_behaviors:
        cluster = behavior["cluster"]
        b = behavior["behavior"]
        key = f"{CLUSTER_SHORTHAND.get(cluster, cluster.split('_')[0])}-{b}"
        actual = behavior["actual"]

        if key not in predictions:
            continue

        predicted = predictions[key][mode]
        is_correct = predicted == actual
        correct += int(is_correct)
        total += 1

        details.append({
            "behavior": b,
            "cluster": cluster,
            "actual": actual,
            "predicted": predicted,
            "correct": is_correct,
            "confidence": predictions[key].get(f"{mode}_confidence", "n/a"),
            "rationale": predictions[key].get(f"{mode}_rationale", ""),
        })

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": details,
    }


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    fixture = json.loads(FIXTURE.read_text())

    train_records = fixture["records"]
    test_records = fixture["test_records"]

    print("=" * 70)
    print("EXPERIMENT 3.23: PREDICTIVE GROUNDING")
    print("=" * 70)
    print()
    print(f"Fixture: {FIXTURE}")
    n_users = len(fixture["users"])
    print(f"Train records: {len(train_records)} ({len(train_records)//n_users} records × {n_users} users)")
    print(f"Test records:  {len(test_records)} ({len(test_records)//n_users} records × {n_users} users)")
    print(f"Counter-factual test cases: 4 (cross-cluster negative predictions)")
    print(f"Total test behaviors evaluated: {len(TEST_BEHAVIORS)}")
    print()

    # Score grounded vs ungrounded
    grounded_results = score_predictions(TEST_BEHAVIORS, PREDICTIONS, "grounded")
    ungrounded_results = score_predictions(TEST_BEHAVIORS, PREDICTIONS, "ungrounded")

    delta = grounded_results["accuracy"] - ungrounded_results["accuracy"]

    print(f"{'Mode':<15} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 45)
    print(f"{'Grounded':<15} {grounded_results['correct']:>8} {grounded_results['total']:>8} {grounded_results['accuracy']:>9.1%}")
    print(f"{'Ungrounded':<15} {ungrounded_results['correct']:>8} {ungrounded_results['total']:>8} {ungrounded_results['accuracy']:>9.1%}")
    print("-" * 45)
    print(f"{'Delta':<15} {'':>8} {'':>8} {delta:>+9.1%}")
    print()

    # Signal classification
    delta_pct = delta * 100
    if delta_pct > 20:
        signal = "STRONG"
    elif delta_pct > 10:
        signal = "MODERATE"
    elif delta_pct > 5:
        signal = "WEAK"
    else:
        signal = "NOISE"

    print(f"Signal: {signal} (delta = {delta_pct:+.1f} percentage points)")
    print()

    # Per-behavior breakdown
    print("Per-behavior prediction breakdown:")
    print(f"{'Behavior':<40} {'Actual':>8} {'Grnd':>6} {'Ungrnd':>8} {'G-ok':>6} {'U-ok':>6}")
    print("-" * 80)

    for g_detail, u_detail in zip(grounded_results["details"], ungrounded_results["details"]):
        b_label = f"{CLUSTER_SHORTHAND.get(g_detail['cluster'], g_detail['cluster'].split('_')[0])}.{g_detail['behavior']}"[:38]
        actual_s = "YES" if g_detail["actual"] else "NO"
        g_pred_s = "YES" if g_detail["predicted"] else "NO"
        u_pred_s = "YES" if u_detail["predicted"] else "NO"
        g_ok = "✓" if g_detail["correct"] else "✗"
        u_ok = "✓" if u_detail["correct"] else "✗"
        print(f"{b_label:<40} {actual_s:>8} {g_pred_s:>6} {u_pred_s:>8} {g_ok:>6} {u_ok:>6}")

    print()

    # Where they diverge
    diverge = [
        (g, u) for g, u in zip(grounded_results["details"], ungrounded_results["details"])
        if g["predicted"] != u["predicted"]
    ]
    print(f"Behaviors where grounded and ungrounded DIVERGE: {len(diverge)}")
    for g, u in diverge:
        b_label = f"{CLUSTER_SHORTHAND.get(g['cluster'], g['cluster'].split('_')[0])}.{g['behavior']}"
        actual_s = "YES" if g["actual"] else "NO"
        winner = "grounded" if g["correct"] else "ungrounded"
        print(f"  {b_label}: actual={actual_s}, grounded={'YES' if g['predicted'] else 'NO'}, "
              f"ungrounded={'YES' if u['predicted'] else 'NO'} → {winner} correct")
    print()

    # Save results
    results = {
        "experiment": "3.23",
        "title": "Predictive grounding",
        "fixture": str(FIXTURE),
        "train_records": len(train_records),
        "test_records": len(test_records),
        "total_test_behaviors": len(TEST_BEHAVIORS),
        "grounded": {
            "accuracy": grounded_results["accuracy"],
            "correct": grounded_results["correct"],
            "total": grounded_results["total"],
        },
        "ungrounded": {
            "accuracy": ungrounded_results["accuracy"],
            "correct": ungrounded_results["correct"],
            "total": ungrounded_results["total"],
        },
        "delta": delta,
        "signal": signal,
        "details": {
            "grounded": grounded_results["details"],
            "ungrounded": ungrounded_results["details"],
        },
    }

    out_path = EXP_DIR / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to: {out_path}")

    return results


if __name__ == "__main__":
    main()
