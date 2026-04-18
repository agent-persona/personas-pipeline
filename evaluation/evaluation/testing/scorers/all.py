"""Auto-registration of all scorers into the default 'persona' suite.

Import this module to populate the registry with all available scorers.
Usage: `from evaluation.testing.scorers import all`  # noqa: F401
"""

from evaluation.testing import registry

# --- Tier 1: Structural ---
from evaluation.testing.scorers.structural.schema_compliance import SchemaComplianceScorer
from evaluation.testing.scorers.structural.completeness import CompletenessScorer
from evaluation.testing.scorers.structural.consistency import ConsistencyScorer

# --- Tier 2: Semantic ---
from evaluation.testing.scorers.semantic.factual_grounding import FactualGroundingScorer
from evaluation.testing.scorers.semantic.behavioral_consistency import BehavioralConsistencyScorer
from evaluation.testing.scorers.semantic.distinctiveness import DistinctivenessScorer
from evaluation.testing.scorers.semantic.demographic_coherence import DemographicCoherenceScorer
from evaluation.testing.scorers.semantic.memory_consistency import MemoryConsistencyScorer
from evaluation.testing.scorers.semantic.knowledge_boundary import KnowledgeBoundaryScorer
from evaluation.testing.scorers.semantic.lexical_semantic import LexicalSemanticScorer
from evaluation.testing.scorers.semantic.profile_coverage import ProfileCoverageScorer
from evaluation.testing.scorers.semantic.narrative_coherence import NarrativeCoherenceScorer

# --- Tier 3: Distributional ---
from evaluation.testing.scorers.distributional.opinion_diversity import OpinionDiversityScorer
from evaluation.testing.scorers.distributional.variance_fidelity import VarianceFidelityScorer
from evaluation.testing.scorers.distributional.aggregation_consistency import AggregationConsistencyScorer
from evaluation.testing.scorers.distributional.minority_viewpoint import MinorityViewpointScorer
from evaluation.testing.scorers.distributional.calibration import CalibrationScorer
from evaluation.testing.scorers.distributional.joint_distribution import JointDistributionScorer
from evaluation.testing.scorers.distributional.tail_insight_detection import TailInsightDetectionScorer

# --- Tier 4: Bias ---
from evaluation.testing.scorers.bias.positivity_bias import PositivityBiasScorer
from evaluation.testing.scorers.bias.sycophancy_resistance import SycophancyResistanceScorer
from evaluation.testing.scorers.bias.weird_bias import WEIRDBiasScorer
from evaluation.testing.scorers.bias.hyper_accuracy import HyperAccuracyScorer
from evaluation.testing.scorers.bias.stereotype_amplification import StereotypeAmplificationScorer
from evaluation.testing.scorers.bias.negative_experience import NegativeExperienceScorer
from evaluation.testing.scorers.bias.detail_degradation import DetailDegradationScorer
from evaluation.testing.scorers.bias.register_inflation import RegisterInflationScorer
from evaluation.testing.scorers.bias.hedge_inflation import HedgeInflationScorer
from evaluation.testing.scorers.bias.balanced_opinion import BalancedOpinionScorer

# --- Tier 5: Behavioral ---
from evaluation.testing.scorers.behavioral.emotional_regulation import EmotionalRegulationScorer
from evaluation.testing.scorers.behavioral.empathetic_responsiveness import EmpatheticResponsivenessScorer
from evaluation.testing.scorers.behavioral.moral_stability import MoralStabilityScorer
from evaluation.testing.scorers.behavioral.moral_robustness import MoralRobustnessScorer
from evaluation.testing.scorers.behavioral.refusal_behavior import RefusalBehaviorScorer
from evaluation.testing.scorers.behavioral.adversarial_robustness import AdversarialRobustnessScorer
from evaluation.testing.scorers.behavioral.recovery_behavior import RecoveryBehaviorScorer
from evaluation.testing.scorers.behavioral.engagement import EngagementScorer
from evaluation.testing.scorers.behavioral.strategic_reasoning import StrategicReasoningScorer
from evaluation.testing.scorers.behavioral.coherence_decay import CoherenceDecayScorer

# --- Tier 6: System ---
from evaluation.testing.scorers.system.role_identifiability import RoleIdentifiabilityScorer
from evaluation.testing.scorers.system.predictive_validity import PredictiveValidityScorer
from evaluation.testing.scorers.system.temporal_stability import TemporalStabilityScorer
from evaluation.testing.scorers.system.cross_model_stability import CrossModelStabilityScorer
from evaluation.testing.scorers.system.reproducibility import ReproducibilityScorer
from evaluation.testing.scorers.system.cost_latency import CostLatencyScorer
from evaluation.testing.scorers.system.degradation_detection import DegradationDetectionScorer

# --- Tier 7: Generation ---
from evaluation.testing.scorers.generation.generation_bias import GenerationBiasAmplificationScorer
from evaluation.testing.scorers.generation.source_fidelity import SourceDataFidelityScorer
from evaluation.testing.scorers.generation.sparse_dense_coverage import SparseDenseCoverageScorer

# --- Tier 8: Meta ---
from evaluation.testing.scorers.meta.judge_reliability import JudgeReliabilityScorer
from evaluation.testing.scorers.meta.judge_gaming import JudgeGamingPreventionScorer
from evaluation.testing.scorers.meta.metric_validity import MetricValidityScorer


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
]


def register_all(suite: str = "persona") -> None:
    """Register all scorers into the named suite."""
    for scorer in ALL_SCORERS:
        registry.register(suite, scorer)


def get_all_scorers() -> list:
    """Return fresh instances of all scorers (for SuiteRunner)."""
    return list(ALL_SCORERS)
