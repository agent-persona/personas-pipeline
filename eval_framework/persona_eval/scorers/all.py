"""Auto-registration of all scorers into the default 'persona' suite.

Import this module to populate the registry with all available scorers.
Usage: `from persona_eval.scorers import all`  # noqa: F401
"""

from persona_eval import registry

# --- Tier 1: Structural ---
from persona_eval.scorers.structural.schema_compliance import SchemaComplianceScorer
from persona_eval.scorers.structural.completeness import CompletenessScorer
from persona_eval.scorers.structural.consistency import ConsistencyScorer

# --- Tier 2: Semantic ---
from persona_eval.scorers.semantic.factual_grounding import FactualGroundingScorer
from persona_eval.scorers.semantic.behavioral_consistency import BehavioralConsistencyScorer
from persona_eval.scorers.semantic.distinctiveness import DistinctivenessScorer
from persona_eval.scorers.semantic.demographic_coherence import DemographicCoherenceScorer
from persona_eval.scorers.semantic.memory_consistency import MemoryConsistencyScorer
from persona_eval.scorers.semantic.knowledge_boundary import KnowledgeBoundaryScorer
from persona_eval.scorers.semantic.lexical_semantic import LexicalSemanticScorer
from persona_eval.scorers.semantic.profile_coverage import ProfileCoverageScorer
from persona_eval.scorers.semantic.narrative_coherence import NarrativeCoherenceScorer

# --- Tier 3: Distributional ---
from persona_eval.scorers.distributional.opinion_diversity import OpinionDiversityScorer
from persona_eval.scorers.distributional.variance_fidelity import VarianceFidelityScorer
from persona_eval.scorers.distributional.aggregation_consistency import AggregationConsistencyScorer
from persona_eval.scorers.distributional.minority_viewpoint import MinorityViewpointScorer
from persona_eval.scorers.distributional.calibration import CalibrationScorer
from persona_eval.scorers.distributional.joint_distribution import JointDistributionScorer
from persona_eval.scorers.distributional.tail_insight_detection import TailInsightDetectionScorer

# --- Tier 4: Bias ---
from persona_eval.scorers.bias.positivity_bias import PositivityBiasScorer
from persona_eval.scorers.bias.sycophancy_resistance import SycophancyResistanceScorer
from persona_eval.scorers.bias.weird_bias import WEIRDBiasScorer
from persona_eval.scorers.bias.hyper_accuracy import HyperAccuracyScorer
from persona_eval.scorers.bias.stereotype_amplification import StereotypeAmplificationScorer
from persona_eval.scorers.bias.negative_experience import NegativeExperienceScorer
from persona_eval.scorers.bias.detail_degradation import DetailDegradationScorer
from persona_eval.scorers.bias.register_inflation import RegisterInflationScorer
from persona_eval.scorers.bias.hedge_inflation import HedgeInflationScorer
from persona_eval.scorers.bias.balanced_opinion import BalancedOpinionScorer

# --- Tier 5: Behavioral ---
from persona_eval.scorers.behavioral.emotional_regulation import EmotionalRegulationScorer
from persona_eval.scorers.behavioral.empathetic_responsiveness import EmpatheticResponsivenessScorer
from persona_eval.scorers.behavioral.moral_stability import MoralStabilityScorer
from persona_eval.scorers.behavioral.moral_robustness import MoralRobustnessScorer
from persona_eval.scorers.behavioral.refusal_behavior import RefusalBehaviorScorer
from persona_eval.scorers.behavioral.adversarial_robustness import AdversarialRobustnessScorer
from persona_eval.scorers.behavioral.recovery_behavior import RecoveryBehaviorScorer
from persona_eval.scorers.behavioral.engagement import EngagementScorer
from persona_eval.scorers.behavioral.strategic_reasoning import StrategicReasoningScorer
from persona_eval.scorers.behavioral.coherence_decay import CoherenceDecayScorer

# --- Tier 6: System ---
from persona_eval.scorers.system.role_identifiability import RoleIdentifiabilityScorer
from persona_eval.scorers.system.predictive_validity import PredictiveValidityScorer
from persona_eval.scorers.system.temporal_stability import TemporalStabilityScorer
from persona_eval.scorers.system.cross_model_stability import CrossModelStabilityScorer
from persona_eval.scorers.system.reproducibility import ReproducibilityScorer
from persona_eval.scorers.system.cost_latency import CostLatencyScorer
from persona_eval.scorers.system.degradation_detection import DegradationDetectionScorer

# --- Tier 7: Generation ---
from persona_eval.scorers.generation.generation_bias import GenerationBiasAmplificationScorer
from persona_eval.scorers.generation.source_fidelity import SourceDataFidelityScorer
from persona_eval.scorers.generation.sparse_dense_coverage import SparseDenseCoverageScorer

# --- Tier 8: Meta ---
from persona_eval.scorers.meta.judge_reliability import JudgeReliabilityScorer
from persona_eval.scorers.meta.judge_gaming import JudgeGamingPreventionScorer
from persona_eval.scorers.meta.metric_validity import MetricValidityScorer

# --- Judge: LLM-as-Judge Rubric Scorers ---
from persona_eval.scorers.judge.j1_behavioral_authenticity import BehavioralAuthenticityScorer
from persona_eval.scorers.judge.j2_voice_consistency import VoiceConsistencyScorer
from persona_eval.scorers.judge.j3_value_alignment import ValueAlignmentScorer
from persona_eval.scorers.judge.j4_persona_depth import PersonaDepthScorer
from persona_eval.scorers.judge.j5_contextual_adaptation import ContextualAdaptationScorer


ALL_SCORERS = [
    # Tier 1
    SchemaComplianceScorer(),
    CompletenessScorer(),
    ConsistencyScorer(),
    # Tier 2
    FactualGroundingScorer(),
    BehavioralConsistencyScorer(),
    DistinctivenessScorer(),
    DemographicCoherenceScorer(),
    MemoryConsistencyScorer(),
    KnowledgeBoundaryScorer(),
    LexicalSemanticScorer(),
    ProfileCoverageScorer(),
    NarrativeCoherenceScorer(),
    # Tier 3
    OpinionDiversityScorer(),
    VarianceFidelityScorer(),
    AggregationConsistencyScorer(),
    MinorityViewpointScorer(),
    CalibrationScorer(),
    JointDistributionScorer(),
    TailInsightDetectionScorer(),
    # Tier 4
    PositivityBiasScorer(),
    SycophancyResistanceScorer(),
    WEIRDBiasScorer(),
    HyperAccuracyScorer(),
    StereotypeAmplificationScorer(),
    RegisterInflationScorer(),
    HedgeInflationScorer(),
    BalancedOpinionScorer(),
    NegativeExperienceScorer(),
    DetailDegradationScorer(),
    # Tier 5
    EmotionalRegulationScorer(),
    EmpatheticResponsivenessScorer(),
    MoralStabilityScorer(),
    MoralRobustnessScorer(),
    RefusalBehaviorScorer(),
    AdversarialRobustnessScorer(),
    RecoveryBehaviorScorer(),
    EngagementScorer(),
    StrategicReasoningScorer(),
    CoherenceDecayScorer(),
    # Tier 6
    RoleIdentifiabilityScorer(),
    PredictiveValidityScorer(),
    TemporalStabilityScorer(),
    CrossModelStabilityScorer(),
    ReproducibilityScorer(),
    CostLatencyScorer(),
    DegradationDetectionScorer(),
    # Tier 7
    GenerationBiasAmplificationScorer(),
    SourceDataFidelityScorer(),
    SparseDenseCoverageScorer(),
    # Tier 8
    JudgeReliabilityScorer(),
    JudgeGamingPreventionScorer(),
    MetricValidityScorer(),
    # Judge
    BehavioralAuthenticityScorer(),
    VoiceConsistencyScorer(),
    ValueAlignmentScorer(),
    PersonaDepthScorer(),
    ContextualAdaptationScorer(),
]


def register_all(suite: str = "persona") -> None:
    """Register all scorers into the named suite."""
    for scorer in ALL_SCORERS:
        registry.register(suite, scorer)


def get_all_scorers() -> list:
    """Return fresh instances of all scorers (for SuiteRunner)."""
    return list(ALL_SCORERS)
