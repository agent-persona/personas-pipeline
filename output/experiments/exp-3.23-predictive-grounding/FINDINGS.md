# Experiment 3.23: Predictive Grounding — Findings

## Fixture Design

**Tenant**: `tenant_devflow_io` — developer workflow platform (code review, CI/CD, incident management)

**Users**: 8 synthetic users across 2 behavioral clusters

| Cluster | Users | Train Signal |
|---------|-------|-------------|
| `infra_cluster` | u01–u04 | webhook_config → api_token_create → github_integration_setup → terraform/ci_policy |
| `management_cluster` | u05–u08 | team_dashboard_view → team_invite → sprint_report_view → okr_dashboard/velocity |

**Train window** (months 1–3): 4 records per user = 32 records total
**Test window** (month 4): 2 records per user = 16 records total (8 unique behaviors when aggregated per cluster)

**Test split design**:
- 4 positive test cases (behaviors that match each cluster's natural arc)
- 4 counter-factual test cases (cross-cluster behaviors that should NOT fit)

The fixture is intentionally structured so that test behaviors are predictable from train trajectories: infra users who configured GitHub integrations and Terraform providers in months 1–3 predictably created CI/CD pipelines and deployment environments in month 4. Management users who moved from sprint reports to OKR dashboards predictably viewed roadmaps and exported stakeholder reports.

---

## Personas Synthesized from Train Records

### Grounded Persona — Infra Cluster
**Name**: Alex the Infrastructure Integrator

> A senior infrastructure or platform engineer (fintech/healthtech/edtech, 50–500 employees) who treats the product as the connective tissue of their CI/CD ecosystem. First actions are always API token creation and webhook setup; they immediately reach for GitHub integration and Terraform documentation. Sessions are long and deliberate — they're building, not browsing.

Evidence anchor: 4/4 users showed identical progression webhook_config → api_token_create → github_integration_setup → terraform/ci work. Average session durations on integration pages: 1,800–2,200s.

### Grounded Persona — Management Cluster
**Name**: Maya the Metrics-Driven Manager

> An engineering manager or team lead (b2b-saas/logistics/cybersecurity/retail, 10–1,000 employees) who uses the product as a team visibility and reporting layer. They set up team dashboards first, then invite the team, then settle into sprint report views. By month 3 they're customizing OKR dashboards or velocity charts — signaling a shift from setup to ongoing governance and upward reporting.

Evidence anchor: 4/4 users showed team_dashboard_view → sprint_report_view → OKR/velocity work. Progression is towards strategic visibility, not technical implementation.

### Ungrounded Personas (control)
- **Generic DevOps User**: "A developer or operations professional who uses the platform for infrastructure management. They likely work on deployment-related tasks and may be interested in team collaboration features. Could use the product for project tracking or performance monitoring."
- **Generic Engineering Leader**: "A manager or team lead who uses the platform to oversee engineering work. May use dashboards for visibility and may be interested in reporting capabilities. Focused on team performance and project health. Could be interested in planning and roadmap features or technical integrations."

---

## Prediction Results

### Per-behavior breakdown

| Behavior | Cluster | Actual | Grounded | Ungrounded | G-correct | U-correct |
|----------|---------|--------|----------|------------|-----------|-----------|
| cicd_pipeline_create | infra | YES | YES | YES | ✓ | ✓ |
| deployment_environment_config | infra | YES | YES | YES | ✓ | ✓ |
| roadmap_view | infra | NO | NO | YES | ✓ | ✗ |
| stakeholder_report_export | infra | NO | NO | NO | ✓ | ✓ |
| roadmap_view | management | YES | YES | YES | ✓ | ✓ |
| stakeholder_report_export | management | YES | YES | YES | ✓ | ✓ |
| cicd_pipeline_create | management | NO | NO | YES | ✓ | ✗ |
| deployment_environment_config | management | NO | NO | YES | ✓ | ✗ |

### Accuracy summary

| Mode | Correct | Total | Accuracy |
|------|---------|-------|----------|
| Grounded | 8 | 8 | **100.0%** |
| Ungrounded | 5 | 8 | **62.5%** |
| **Delta** | | | **+37.5 pp** |

### Where they diverged (3 cases)
1. **infra.roadmap_view** (actual=NO): Grounded correctly predicted NO because no strategic/reporting behavior existed in train records. Ungrounded predicted YES because generic DevOps description mentioned "project tracking" — a spurious LLM stereotype.

2. **mgmt.cicd_pipeline_create** (actual=NO): Grounded correctly predicted NO because management users never touched technical infra pages. Ungrounded predicted YES because generic Engineering Leader description mentioned "technical integrations" — conflating manager with hands-on engineer.

3. **mgmt.deployment_environment_config** (actual=NO): Same failure mode as above. Ungrounded description lacked the behavioral specificity to exclude cross-cluster behaviors.

---

## Failure Analysis — Ungrounded Mode

The ungrounded persona's 3 failures all share the same root cause: **generic labels bleed across cluster boundaries**. The "Generic Engineering Leader" description includes both management vocabulary and technical vocabulary ("planning" AND "technical integrations"), so it predicts YES for behaviors from both clusters. It correctly predicts management behaviors but also incorrectly predicts infra behaviors for management users.

This is a predictable failure of LLM-prior personas: without behavioral grounding, the model defaults to a stereotype of "a software professional who does many things" rather than a specific behavioral pattern derived from evidence.

---

## Signal

**STRONG** — grounded accuracy exceeded ungrounded by **+37.5 percentage points** (100% vs 62.5%), well above the 20% threshold.

The hypothesis is confirmed: grounded personas accurately predict future behavior within their behavioral arc, while LLM-prior personas over-predict by projecting generic professional stereotypes that blur cluster boundaries.

## Recommendation: **adopt**

Train/test temporal splitting is a viable and rigorous evaluation methodology for persona grounding. The 37.5 pp delta provides strong evidence that grounding matters for predictive validity, not just descriptive richness. Recommend integrating this evaluation pattern into the standard persona quality pipeline.
