# Ideal Customer Profiles — Persona Eval Framework

*Last updated: 2026-04-17. Built from digital watering hole research (Reddit, HN, arXiv, G2, product blogs). Revisit quarterly.*

---

## ICP 1: AI Eval Engineer at an LLM Product Company

**Profile**
- Title: ML Engineer, AI Eval Engineer, Applied AI Engineer
- Company: Series A–C startups or mid-size companies building agents, copilots, chatbots, or synthetic-data pipelines
- Team size: 3–15 person AI/ML team
- Reports to: Head of AI, VP Engineering, or CTO
- Already uses: LangSmith, Braintrust, Promptfoo, DeepEval, or homegrown eval scripts

**Primary Job to Be Done**
Ship reliable LLM-powered products where synthetic personas drive downstream behavior (testing, simulation, personalization) — and prove they aren't biased or shallow.

**Trigger Events**
- Shipped a feature that relied on synthetic personas and got bad results in production
- Built 200+ lines of ad-hoc persona scoring code and realized it's becoming a maintenance burden
- Peer review or compliance ask: "How do you know your synthetic data isn't biased?"
- Read the "Bias Runs Deep" paper (Allen AI) or "LLM Generated Persona is a Promise with a Catch" and realized their pipeline has no guard

**Top Pains**
1. "I don't actually know if my personas are good. I'm hoping nobody asks." — No structured way to evaluate persona quality beyond manual spot-checks
2. "My eval pipeline checks prompt quality and RAG faithfulness, but nothing checks the *persona itself*" — existing tools (LangSmith, Promptfoo, DeepEval) don't have persona-specific dimensions
3. "I built a quick eval script and it's already 500 lines. This is going to consume my quarter." — Homegrown solutions don't scale and lack methodology

**Desired Outcomes**
- Drop-in eval step in their CI/CD pipeline that catches broken personas before production
- Granular, per-dimension failure modes ("D45 register inflation: 0.3") not a generic score
- A methodology they can cite when asked "how do you validate personas?"

**Objections and Fears**
- "Will I have to rewrite my generation pipeline?" → No, pass JSON in, get JSON out
- "How do I know your scorers are right?" → Every metric in `details` is inspectable; pruning verdicts documented
- "We can build this ourselves." → Sure, but 52 scorers + 303 tests + bias methodology = a quarter of work

**Alternatives They Consider**
- Build their own scoring scripts (most common — and most regretted)
- Use general eval tools (LangSmith, Promptfoo, DeepEval) and add custom metrics
- Manual review / "looks good to me"
- Do nothing and hope for the best

**Key Vocabulary**
- "eval pipeline," "scoring dimensions," "bias detection," "synthetic data quality"
- "designed-to-fail," "distribution check," "persona grounding"
- "drop-in," "extensible," "per-dimension breakdown"

**How to Reach Them**
- Channels: r/MachineLearning, r/LocalLLaMA, Hacker News, GitHub (search for repos using promptfoo/langsmith/evals), ML Discord servers, Twitter/X ML community
- Content they consume: arXiv papers on LLM bias, eval framework comparison posts, "Show HN" launches
- Influencers: Hamel Husain, Jason Liu, Simon Willison, Shreya Shankar (eval-adjacent voices)

---

## ICP 2: UX/Product Researcher Using AI-Generated Personas

**Profile**
- Title: UX Researcher, Product Researcher, Design Lead, Product Manager
- Company: Product teams at mid-to-large companies, agencies, or consultancies using AI to generate user personas
- Team size: 1–5 person research team
- Reports to: Head of Product, VP Design, Chief Experience Officer
- Already uses: HubSpot Make My Persona, Synthetic Users, UXPressia, PersonaBuilder.co, or ChatGPT directly

**Primary Job to Be Done**
Validate that AI-generated user personas are trustworthy enough to drive product decisions — not just plausible-looking.

**Trigger Events**
- Stakeholder or client asks: "How do you know these personas represent real users?"
- Read the NNG article on synthetic users or the "Generative AI Personas Considered Harmful?" paper and got worried
- Generated personas that all sounded the same — suspiciously polished, no real contradictions
- Making a hiring, feature, or go-to-market decision based on AI-generated profiles and felt uneasy

**Top Pains**
1. "AI personas look plausible but I can't tell which ones are good and which are just confident hallucinations" — No quality signal beyond gut feel
2. "They're all too positive, too articulate, too balanced — where are the messy real humans?" — LLM-generated personas lack the contradictions and rough edges of real people
3. "I need something I can show my VP to justify that we did our homework" — No audit trail or defensible methodology

**Desired Outcomes**
- A report that shows per-dimension scores for each persona (schema, plausibility, depth, bias)
- Evidence they can point to: "We ran these through 52 evaluators and here's what passed"
- Confidence to use AI-generated personas for real decisions, or clear signal that they shouldn't

**Objections and Fears**
- "I'm not a developer — can I use this?" → CLI is `pip install` + one command, but this is a real friction point. Consider: hosted version, Colab notebook, or GUI wrapper
- "My personas aren't in your JSON format" → Conversion guidance needed; only `id` and `name` are required
- "52 scorers sounds like overkill for my use case" → Tier system lets you run only what matters

**Alternatives They Consider**
- Trust their gut / manual review
- Use persona-gen tools that claim "AI-validated" without methodology
- Don't evaluate at all — ship and hope
- Hire a research consultant

**Key Vocabulary**
- "user profiles," "persona validation," "AI-generated personas," "trustworthy"
- "bias check," "quality audit," "defensible research"
- NOT: "eval pipeline," "CI/CD," "scorer" — translate for this audience

**How to Reach Them**
- Channels: UX subreddits (r/userexperience, r/UXDesign), LinkedIn (product/UX community), UX Substack newsletters, Product Hunt
- Content they consume: NNG articles, UX Psychology Substack, IxDF, product management blogs
- Influencers: Jared Spool, Teresa Torres, NNG researchers

---

## ICP 3: Academic / Research Scientist Studying LLM Persona Behavior

**Profile**
- Title: PhD student, Postdoc, Research Scientist
- Org: University AI lab, industry research lab (Allen AI, DeepMind, Anthropic, Meta FAIR), policy think tank
- Team size: 1–4 researchers
- Already uses: Custom Python scripts, HuggingFace datasets, manual annotation

**Primary Job to Be Done**
Evaluate synthetic personas with a structured, reproducible, citable methodology for publication.

**Trigger Events**
- Writing a paper on LLM persona behavior and need an evaluation framework
- Peer reviewer asked: "What's your evaluation methodology for persona quality?"
- Want to compare persona generation across models (GPT-4 vs Claude vs Llama) and need consistent metrics
- Found the Allen AI "persona-bias" dataset and want a framework to operationalize evaluation

**Top Pains**
1. "Every lab builds its own evaluation scripts — there's no shared methodology" — Hard to compare across papers
2. "I need reproducible metrics, not vibes" — Ad-hoc evaluation doesn't survive peer review
3. "Bias evaluation especially is ad-hoc — everyone measures it differently" — No standard dimensions

**Desired Outcomes**
- A citable framework with named dimensions (D1–D50) they can reference in papers
- Reproducible scoring: same input → same output, documented methodology
- Extensible: add custom dimensions for their specific research question via BaseScorer subclass

**Objections and Fears**
- "Is this peer-reviewed?" → Research-backed methodology, but not a published paper (yet). The bias dimensions draw from "Bias Runs Deep" and adjacent work
- "Can I extend it for my specific research?" → Yes, BaseScorer subclass = new dimension in <50 lines
- "Will it work with my dataset format?" → Flexible schema, only `id` and `name` required

**Alternatives They Consider**
- Build custom evaluation from scratch (default)
- Use general NLP metrics (BLEU, embedding similarity) that don't capture persona-specific quality
- Manual annotation with inter-rater reliability
- Allen AI persona-bias dataset + custom analysis

**Key Vocabulary**
- "evaluation methodology," "dimensions," "reproducibility," "inter-rater reliability"
- "demographic bias," "distributional analysis," "persona grounding"
- "citable," "benchmark," "framework"

**How to Reach Them**
- Channels: arXiv, OpenReview, r/MachineLearning, Twitter/X ML academic community, conference poster sessions (AAAI AIES, ACL, CHI, EMNLP)
- Content they consume: Papers on persona bias (Allen AI, Cambridge, Zurich), HuggingFace datasets page
- Influencers: Authors of "Bias Runs Deep," "Whose Personae?," "LLM Generated Persona is a Promise with a Catch"

---

## ICP Priority for Outreach

| ICP | Volume | Ease of reach | Conversion signal | Priority |
|-----|--------|---------------|-------------------|----------|
| 1. AI Eval Engineer | Medium | High (GitHub, HN) | Stars + installs | **Primary** |
| 3. Academic Researcher | Medium | Medium (arXiv, Twitter) | Citations + stars | **Secondary** |
| 2. UX/Product Researcher | Large | Low (less technical, needs onboarding) | Installs, but needs smoother DX | **Tertiary (invest in DX first)** |

**For the 100-star goal:** Focus outreach on ICP 1 and ICP 3. They star repos, they share tools, and they're reachable via GitHub + HN + academic Twitter. ICP 2 is a bigger market but needs a lower-friction entry point (Colab notebook, web UI) before outreach converts.
