from __future__ import annotations
import json
import sys
from pathlib import Path
import click
from persona_eval.schemas import Persona
from persona_eval.source_context import SourceContext


@click.group()
def cli() -> None:
    """Persona evaluation CLI."""


@cli.command("run")
@click.option("--suite", default="persona", show_default=True, help="Suite name to run")
@click.option("--tier", default=None, type=int, help="Only run scorers in this tier")
@click.option(
    "--output",
    default="table",
    type=click.Choice(["table", "json"]),
    show_default=True,
    help="Output format",
)
@click.option("--persona-file", required=True, type=click.Path(exists=True), help="Path to persona JSON")
@click.option("--source-file", required=True, type=click.Path(exists=True), help="Path to source context JSON")
def run_suite(suite: str, tier: int | None, output: str, persona_file: str, source_file: str) -> None:
    """Run eval suite against a single persona + source context."""
    from persona_eval.scorers.all import get_all_scorers
    from persona_eval.suite_runner import SuiteRunner

    with open(persona_file) as f:
        persona = Persona.model_validate(json.load(f))
    with open(source_file) as f:
        source_context = SourceContext.model_validate(json.load(f))

    runner = SuiteRunner(scorers=get_all_scorers())
    results = runner.run(persona, source_context, tier_filter=tier)

    if output == "json":
        click.echo(json.dumps([r.model_dump() for r in results], indent=2))
    else:
        _print_table(results)


@cli.command("run-set")
@click.option("--tier", default=None, type=int, help="Only run scorers in this tier")
@click.option(
    "--output",
    default="table",
    type=click.Choice(["table", "json"]),
    show_default=True,
    help="Output format",
)
@click.option("--persona-dir", required=True, type=click.Path(exists=True), help="Dir of persona JSON files")
@click.option("--source-dir", required=True, type=click.Path(exists=True), help="Dir of source context JSON files")
def run_set(tier: int | None, output: str, persona_dir: str, source_dir: str) -> None:
    """Run full eval suite (per-persona + set-level) on a directory of personas."""
    from persona_eval.scorers.all import get_all_scorers
    from persona_eval.suite_runner import SuiteRunner

    persona_files = sorted(Path(persona_dir).glob("*.json"))
    source_files = sorted(Path(source_dir).glob("*.json"))

    if not persona_files:
        click.echo("Error: no JSON files found in persona-dir", err=True)
        sys.exit(1)

    personas = []
    for pf in persona_files:
        with open(pf) as f:
            personas.append(Persona.model_validate(json.load(f)))

    source_contexts = []
    for sf in source_files:
        with open(sf) as f:
            source_contexts.append(SourceContext.model_validate(json.load(f)))

    # Pad source contexts if fewer than personas
    while len(source_contexts) < len(personas):
        source_contexts.append(SourceContext(id="default", text=""))

    runner = SuiteRunner(scorers=get_all_scorers())
    results = runner.run_full(personas, source_contexts, tier_filter=tier)

    if output == "json":
        click.echo(json.dumps([r.model_dump() for r in results], indent=2))
    else:
        _print_table(results)


@cli.command("list")
def list_scorers() -> None:
    """List all available scorers."""
    from persona_eval.scorers.all import ALL_SCORERS
    click.echo(f"{'DIM':<8} {'NAME':<45} {'TIER':<5} {'SET':<4}")
    click.echo("-" * 65)
    for s in sorted(ALL_SCORERS, key=lambda x: (x.tier, x.dimension_id)):
        set_flag = "yes" if s.requires_set else ""
        click.echo(f"{s.dimension_id:<8} {s.dimension_name:<45} {s.tier:<5} {set_flag:<4}")
    click.echo(f"\n{len(ALL_SCORERS)} scorers total")


@cli.command("validate-proxies")
@click.option("--persona-dir", required=True, type=click.Path(exists=True))
@click.option("--source-dir", required=True, type=click.Path(exists=True))
@click.option(
    "--output",
    default="table",
    type=click.Choice(["table", "json"]),
    show_default=True,
)
@click.option(
    "--min-personas",
    default=5,
    type=int,
    show_default=True,
    help="Minimum non-skipped personas required to compute correlation",
)
def validate_proxies(persona_dir: str, source_dir: str, output: str, min_personas: int) -> None:
    """Compute Spearman correlation between proxy scorers and LLM judge scorers.

    Identifies which proxy dimensions reliably track judge quality signals.
    Skipped results (score=1.0, details.skipped=True) are excluded from correlation.
    Proxies with mean |ρ| < 0.3 are flagged NOISE; |ρ| > 0.6 flagged STRONG.
    """
    from persona_eval.scorers.structural.consistency import ConsistencyScorer
    from persona_eval.scorers.semantic.memory_consistency import MemoryConsistencyScorer
    from persona_eval.scorers.distributional.variance_fidelity import VarianceFidelityScorer
    from persona_eval.scorers.behavioral.refusal_behavior import RefusalBehaviorScorer
    from persona_eval.scorers.bias.hedge_inflation import HedgeInflationScorer
    from persona_eval.scorers.bias.balanced_opinion import BalancedOpinionScorer
    from persona_eval.scorers.judge.j1_behavioral_authenticity import BehavioralAuthenticityScorer
    from persona_eval.scorers.judge.j2_voice_consistency import VoiceConsistencyScorer
    from persona_eval.scorers.judge.j3_value_alignment import ValueAlignmentScorer
    from persona_eval.scorers.judge.j4_persona_depth import PersonaDepthScorer
    from persona_eval.scorers.judge.j5_contextual_adaptation import ContextualAdaptationScorer
    from persona_eval.stats import spearman_r

    proxy_scorers = [
        ConsistencyScorer(),
        MemoryConsistencyScorer(),
        VarianceFidelityScorer(),
        RefusalBehaviorScorer(),
        HedgeInflationScorer(),
        BalancedOpinionScorer(),
    ]
    judge_scorers = [
        BehavioralAuthenticityScorer(),
        VoiceConsistencyScorer(),
        ValueAlignmentScorer(),
        PersonaDepthScorer(),
        ContextualAdaptationScorer(),
    ]

    persona_files = sorted(Path(persona_dir).glob("*.json"))
    source_files = sorted(Path(source_dir).glob("*.json"))

    if not persona_files:
        click.echo("Error: no JSON files found in persona-dir", err=True)
        sys.exit(1)

    personas = []
    for pf in persona_files:
        with open(pf) as f:
            personas.append(Persona.model_validate(json.load(f)))

    source_contexts = []
    for sf in source_files:
        with open(sf) as f:
            source_contexts.append(SourceContext.model_validate(json.load(f)))

    while len(source_contexts) < len(personas):
        source_contexts.append(SourceContext(id="default", text=""))

    def _run_scorers(scorers, personas, contexts):
        """Return {dim_id: [score_or_None, ...]} — None for skipped results."""
        scores: dict[str, list[float | None]] = {s.dimension_id: [] for s in scorers}
        for scorer in scorers:
            for persona, ctx in zip(personas, contexts):
                try:
                    result = scorer.score(persona, ctx)
                    if result.details.get("skipped"):
                        scores[scorer.dimension_id].append(None)
                    else:
                        scores[scorer.dimension_id].append(result.score)
                except Exception:
                    scores[scorer.dimension_id].append(None)
        return scores

    proxy_scores = _run_scorers(proxy_scorers, personas, source_contexts)
    judge_scores = _run_scorers(judge_scorers, personas, source_contexts)

    # Compute Spearman ρ for each (proxy × judge) pair, excluding None values
    rows = []
    for proxy in proxy_scorers:
        pid = proxy.dimension_id
        row: dict = {"proxy": pid}
        rhos = []
        for judge in judge_scorers:
            jid = judge.dimension_id
            px = proxy_scores[pid]
            jx = judge_scores[jid]
            # Keep only paired non-None values
            pairs = [(p, j) for p, j in zip(px, jx) if p is not None and j is not None]
            if len(pairs) < min_personas:
                row[jid] = None
            else:
                xs, ys = zip(*pairs)
                row[jid] = round(spearman_r(list(xs), list(ys)), 3)
                rhos.append(abs(row[jid]))
        row["mean_rho"] = round(sum(rhos) / len(rhos), 3) if rhos else None
        if row["mean_rho"] is not None:
            row["flag"] = "STRONG" if row["mean_rho"] > 0.6 else ("NOISE" if row["mean_rho"] < 0.3 else "")
        else:
            row["flag"] = "N/A"
        rows.append(row)

    if output == "json":
        click.echo(json.dumps(rows, indent=2))
        return

    judge_ids = [j.dimension_id for j in judge_scorers]
    header = f"{'PROXY':<8}" + "".join(f"{jid:>8}" for jid in judge_ids) + f"{'MEAN_ρ':>9}  FLAG"
    click.echo(header)
    click.echo("-" * len(header))
    for row in rows:
        line = f"{row['proxy']:<8}"
        for jid in judge_ids:
            val = row.get(jid)
            line += f"{val:>8.3f}" if val is not None else f"{'N/A':>8}"
        mean = row["mean_rho"]
        line += f"{mean:>9.3f}" if mean is not None else f"{'N/A':>9}"
        line += f"  {row['flag']}"
        click.echo(line)


def _print_table(results: list) -> None:
    click.echo(f"{'DIM':<8} {'NAME':<40} {'PASS':<6} {'SCORE':<7} {'PERSONA':<12}")
    click.echo("-" * 75)
    for r in results:
        status = "PASS" if r.passed else "FAIL" if not r.details.get("skipped") else "SKIP"
        pid = r.persona_id[:10] if len(r.persona_id) > 10 else r.persona_id
        click.echo(f"{r.dimension_id:<8} {r.dimension_name:<40} {status:<6} {r.score:.3f}  {pid:<12}")
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed and not r.details.get("skipped"))
    skipped = sum(1 for r in results if r.details.get("skipped"))
    click.echo(f"\n{passed} passed, {failed} failed, {skipped} skipped ({len(results)} total)")
