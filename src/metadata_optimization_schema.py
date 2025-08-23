"""
Metadata Schema Optimization for LLM Fine-tuning and Embeddings
Comprehensive metadata structure design for maximum training effectiveness
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from datetime import datetime, timezone
import uuid

class ContentType(Enum):
    """Content type classifications for training optimization"""
    PARAMETER_CONFIGURATION = "parameter_configuration"
    TROUBLESHOOTING = "troubleshooting"
    FEATURE_DESCRIPTION = "feature_description"
    COUNTER_ANALYSIS = "counter_analysis"
    NETWORK_OPTIMIZATION = "network_optimization"
    PROCEDURAL_GUIDE = "procedural_guide"
    DIAGNOSTIC_WORKFLOW = "diagnostic_workflow"
    CONCEPTUAL_EXPLANATION = "conceptual_explanation"

class DifficultyLevel(Enum):
    """Content difficulty levels for curriculum learning"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"
    EXPERT = "expert"

class TechnicalDomain(Enum):
    """Technical domain classifications"""
    RAN_CONFIGURATION = "ran_configuration"
    NETWORK_OPTIMIZATION = "network_optimization"
    PERFORMANCE_MONITORING = "performance_monitoring"
    TROUBLESHOOTING = "troubleshooting"
    PROTOCOL_ANALYSIS = "protocol_analysis"
    HARDWARE_MANAGEMENT = "hardware_management"
    SOFTWARE_CONFIGURATION = "software_configuration"

@dataclass
class TechnicalTermClassification:
    """Classification of technical terms for enhanced understanding"""
    term: str
    category: str  # protocol, hardware, software, metric, procedure
    frequency: int  # Usage frequency in dataset
    context_importance: float  # 0.0-1.0 importance score
    related_terms: List[str] = field(default_factory=list)
    definitions: Optional[str] = None

@dataclass
class ConversationStructure:
    """Analysis of conversation structure and flow"""
    turn_count: int
    avg_turn_length: int
    question_types: List[str]  # how_to, what_is, troubleshoot, etc.
    response_patterns: List[str]  # explanation, procedure, diagnostic
    coherence_score: float  # 0.0-1.0

@dataclass
class TrainingOptimization:
    """Optimization hints for training processes"""
    instruction_tuning_weight: float  # Relative weight for instruction tuning
    embedding_priority: float  # Priority for embedding generation
    context_window_requirement: int  # Minimum context window needed
    multi_turn_capability: bool  # Whether requires multi-turn understanding
    reasoning_complexity: str  # simple, moderate, complex
    domain_specificity: float  # 0.0-1.0, higher = more domain specific

@dataclass
class QualityAssurance:
    """Quality assurance and validation metadata"""
    validation_timestamp: str
    validation_version: str
    quality_checks_passed: List[str]
    quality_issues: List[str]
    human_reviewed: bool
    confidence_score: float
    technical_accuracy: float
    linguistic_quality: float

@dataclass
class SourceProvenance:
    """Detailed source tracking and provenance"""
    original_source: str
    source_document_id: Optional[str]
    extraction_method: str
    processing_pipeline: List[str]
    transformation_history: List[Dict[str, Any]]
    curator_notes: Optional[str]
    version_id: str

@dataclass
class OptimizedMetadata:
    """
    Comprehensive optimized metadata schema for LLM fine-tuning
    Designed for maximum training effectiveness and retrieval performance
    """
    
    # Core Identification
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dataset_version: str = "1.0"
    creation_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Content Classification
    content_type: ContentType = ContentType.CONCEPTUAL_EXPLANATION
    technical_domain: TechnicalDomain = TechnicalDomain.RAN_CONFIGURATION
    difficulty_level: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    
    # Technical Content Analysis
    feature_name: Optional[str] = None
    feature_category: Optional[str] = None
    product_family: Optional[str] = None  # e.g., "Radio System", "Core Network"
    technology_stack: List[str] = field(default_factory=list)  # ["LTE", "5G", "WCDMA"]
    
    # Technical Terms and Concepts
    technical_terms: List[TechnicalTermClassification] = field(default_factory=list)
    primary_concepts: List[str] = field(default_factory=list)
    secondary_concepts: List[str] = field(default_factory=list)
    concept_relationships: Dict[str, List[str]] = field(default_factory=dict)
    
    # Parameters and Configuration
    parameters_mentioned: List[Dict[str, Any]] = field(default_factory=list)
    # Structure: [{"name": "param", "mo_class": "class", "type": "config/counter", "importance": 0.8}]
    
    mo_classes_involved: List[str] = field(default_factory=list)
    configuration_context: Optional[str] = None
    
    # Performance and Monitoring
    counters_mentioned: List[Dict[str, Any]] = field(default_factory=list)
    # Structure: [{"name": "counter", "type": "pm/kpi", "unit": "percentage", "threshold": 90}]
    
    kpi_categories: List[str] = field(default_factory=list)
    performance_impact: Optional[str] = None
    
    # Conversation Analysis
    conversation_structure: Optional[ConversationStructure] = None
    question_classification: List[str] = field(default_factory=list)
    answer_completeness: float = 1.0  # 0.0-1.0
    
    # Training Optimization
    training_optimization: TrainingOptimization = field(
        default_factory=lambda: TrainingOptimization(
            instruction_tuning_weight=1.0,
            embedding_priority=0.8,
            context_window_requirement=512,
            multi_turn_capability=False,
            reasoning_complexity="moderate",
            domain_specificity=0.7
        )
    )
    
    # Quality and Validation
    quality_assurance: QualityAssurance = field(
        default_factory=lambda: QualityAssurance(
            validation_timestamp=datetime.now(timezone.utc).isoformat(),
            validation_version="1.0",
            quality_checks_passed=[],
            quality_issues=[],
            human_reviewed=False,
            confidence_score=0.8,
            technical_accuracy=0.8,
            linguistic_quality=0.8
        )
    )
    
    # Source and Provenance
    source_provenance: SourceProvenance = field(
        default_factory=lambda: SourceProvenance(
            original_source="unknown",
            source_document_id=None,
            extraction_method="automated",
            processing_pipeline=[],
            transformation_history=[],
            curator_notes=None,
            version_id="1.0"
        )
    )
    
    # Embedding and Retrieval Optimization
    embedding_tags: List[str] = field(default_factory=list)
    retrieval_keywords: List[str] = field(default_factory=list)
    semantic_clusters: List[str] = field(default_factory=list)
    similarity_groups: List[str] = field(default_factory=list)
    
    # Fine-tuning Specific
    instruction_following_score: float = 0.8
    response_quality_score: float = 0.8
    factual_consistency_score: float = 0.9
    relevance_score: float = 0.9
    
    # Curriculum Learning Support
    prerequisite_knowledge: List[str] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)
    skill_level_required: List[str] = field(default_factory=list)
    
    # Multi-modal Support (future extension)
    has_diagrams: bool = False
    has_code_examples: bool = False
    has_configuration_snippets: bool = False
    visual_elements: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizedMetadata':
        """Create from dictionary"""
        # Handle enum conversions
        if 'content_type' in data and isinstance(data['content_type'], str):
            data['content_type'] = ContentType(data['content_type'])
        if 'technical_domain' in data and isinstance(data['technical_domain'], str):
            data['technical_domain'] = TechnicalDomain(data['technical_domain'])
        if 'difficulty_level' in data and isinstance(data['difficulty_level'], str):
            data['difficulty_level'] = DifficultyLevel(data['difficulty_level'])
            
        return cls(**data)

class MetadataOptimizer:
    """Service for optimizing metadata for training effectiveness"""
    
    def __init__(self):
        self.technical_term_db = self._load_technical_terms()
        self.parameter_registry = self._load_parameter_registry()
        
    def _load_technical_terms(self) -> Dict[str, Dict]:
        """Load comprehensive technical terms database"""
        return {
            # RAN Technologies
            "LTE": {"category": "technology", "importance": 0.9, "related": ["eNodeB", "EPC"]},
            "5G": {"category": "technology", "importance": 0.95, "related": ["gNodeB", "SA", "NSA"]},
            "NR": {"category": "technology", "importance": 0.9, "related": ["5G", "gNodeB"]},
            
            # Network Elements
            "eNodeB": {"category": "hardware", "importance": 0.9, "related": ["LTE", "cell"]},
            "gNodeB": {"category": "hardware", "importance": 0.9, "related": ["5G", "NR"]},
            "RBS": {"category": "hardware", "importance": 0.8, "related": ["radio", "basestation"]},
            
            # Protocols
            "RRC": {"category": "protocol", "importance": 0.8, "related": ["connection", "control"]},
            "PDCP": {"category": "protocol", "importance": 0.7, "related": ["packet", "compression"]},
            "X2": {"category": "interface", "importance": 0.8, "related": ["handover", "eNodeB"]},
            
            # Measurements
            "RSRP": {"category": "measurement", "importance": 0.9, "related": ["signal", "strength"]},
            "RSRQ": {"category": "measurement", "importance": 0.8, "related": ["quality", "signal"]},
            "SINR": {"category": "measurement", "importance": 0.8, "related": ["interference", "ratio"]},
            
            # KPIs and Counters
            "KPI": {"category": "metric", "importance": 0.9, "related": ["performance", "indicator"]},
            "PM": {"category": "metric", "importance": 0.8, "related": ["performance", "measurement"]},
        }
        
    def _load_parameter_registry(self) -> Dict[str, Dict]:
        """Load parameter registry with MO class mappings"""
        return {
            "FeatureState.featureState": {
                "mo_class": "FeatureState",
                "type": "configuration",
                "importance": 0.9,
                "category": "feature_control"
            },
            "EUtranCellFDD.handoverMargin": {
                "mo_class": "EUtranCellFDD", 
                "type": "configuration",
                "importance": 0.8,
                "category": "mobility"
            },
            "pmPdcchCceUsed": {
                "mo_class": "EUtranCellFDD",
                "type": "counter",
                "importance": 0.7,
                "category": "performance_monitoring"
            }
        }
    
    def optimize_metadata(self, original_record: Dict) -> OptimizedMetadata:
        """
        Transform original metadata to optimized structure
        
        Args:
            original_record: Record with original metadata structure
            
        Returns:
            OptimizedMetadata: Optimized metadata instance
        """
        messages = original_record.get("messages", [])
        original_metadata = original_record.get("metadata", {})
        
        # Create optimized metadata
        optimized = OptimizedMetadata()
        
        # Basic mapping from original metadata
        optimized.feature_name = original_metadata.get("feature_name")
        optimized.feature_category = self._classify_feature_category(optimized.feature_name)
        
        # Analyze content for technical classification
        all_content = " ".join([msg.get("content", "") for msg in messages])
        optimized.technical_terms = self._extract_classified_terms(all_content)
        optimized.primary_concepts = self._extract_primary_concepts(all_content)
        
        # Content type classification
        optimized.content_type = self._classify_content_type(messages, original_metadata)
        optimized.technical_domain = self._classify_technical_domain(all_content)
        optimized.difficulty_level = self._assess_difficulty_level(all_content, original_metadata)
        
        # Parameter and configuration analysis
        optimized.parameters_mentioned = self._extract_parameters(all_content)
        optimized.mo_classes_involved = self._extract_mo_classes(all_content)
        optimized.counters_mentioned = self._extract_counters(all_content)
        
        # Conversation structure analysis
        optimized.conversation_structure = self._analyze_conversation_structure(messages)
        
        # Training optimization settings
        optimized.training_optimization = self._optimize_for_training(
            all_content, original_metadata
        )
        
        # Quality assessment
        optimized.quality_assurance = self._assess_quality(original_record)
        
        # Source provenance
        optimized.source_provenance = self._extract_provenance(original_metadata)
        
        # Embedding optimization
        optimized.embedding_tags = self._generate_embedding_tags(all_content)
        optimized.retrieval_keywords = self._generate_retrieval_keywords(all_content)
        
        return optimized
        
    def _classify_feature_category(self, feature_name: Optional[str]) -> Optional[str]:
        """Classify feature into high-level category"""
        if not feature_name:
            return None
            
        feature_lower = feature_name.lower()
        
        if any(term in feature_lower for term in ["handover", "mobility", "neighbor"]):
            return "mobility_management"
        elif any(term in feature_lower for term in ["power", "energy", "sleep"]):
            return "power_management"
        elif any(term in feature_lower for term in ["load", "balancing", "scheduling"]):
            return "resource_management"
        elif any(term in feature_lower for term in ["measurement", "monitoring", "pm"]):
            return "performance_monitoring"
        else:
            return "feature_operation"
            
    def _extract_classified_terms(self, content: str) -> List[TechnicalTermClassification]:
        """Extract and classify technical terms"""
        terms = []
        words = content.upper().split()
        
        for word in set(words):  # Remove duplicates
            if word in self.technical_term_db:
                term_info = self.technical_term_db[word]
                terms.append(TechnicalTermClassification(
                    term=word,
                    category=term_info["category"],
                    frequency=words.count(word),
                    context_importance=term_info["importance"],
                    related_terms=term_info.get("related", [])
                ))
                
        return terms
        
    def _extract_primary_concepts(self, content: str) -> List[str]:
        """Extract primary technical concepts"""
        concepts = []
        
        # Look for key concept indicators
        concept_patterns = [
            r"configure (\w+)",
            r"enable (\w+)", 
            r"(\w+) feature",
            r"(\w+) parameter",
            r"(\w+) optimization"
        ]
        
        import re
        for pattern in concept_patterns:
            matches = re.findall(pattern, content.lower())
            concepts.extend(matches)
            
        return list(set(concepts))  # Remove duplicates
        
    def _classify_content_type(self, messages: List[Dict], _metadata: Dict) -> ContentType:
        """Classify the type of content"""
        if not messages:
            return ContentType.CONCEPTUAL_EXPLANATION
            
        user_content = messages[0].get("content", "").lower()
        
        if any(word in user_content for word in ["configure", "set", "enable", "disable"]):
            return ContentType.PARAMETER_CONFIGURATION
        elif any(word in user_content for word in ["troubleshoot", "diagnose", "fix", "resolve"]):
            return ContentType.TROUBLESHOOTING
        elif any(word in user_content for word in ["counter", "kpi", "performance", "measurement"]):
            return ContentType.COUNTER_ANALYSIS
        elif any(word in user_content for word in ["how to", "procedure", "steps"]):
            return ContentType.PROCEDURAL_GUIDE
        elif any(word in user_content for word in ["what is", "explain", "describe"]):
            return ContentType.CONCEPTUAL_EXPLANATION
        else:
            return ContentType.FEATURE_DESCRIPTION
            
    def _classify_technical_domain(self, content: str) -> TechnicalDomain:
        """Classify technical domain"""
        content_lower = content.lower()
        
        if any(term in content_lower for term in ["configure", "parameter", "mo", "feature"]):
            return TechnicalDomain.RAN_CONFIGURATION
        elif any(term in content_lower for term in ["counter", "kpi", "pm", "measurement"]):
            return TechnicalDomain.PERFORMANCE_MONITORING
        elif any(term in content_lower for term in ["troubleshoot", "diagnose", "error", "failure"]):
            return TechnicalDomain.TROUBLESHOOTING
        elif any(term in content_lower for term in ["optimize", "improve", "enhance"]):
            return TechnicalDomain.NETWORK_OPTIMIZATION
        else:
            return TechnicalDomain.RAN_CONFIGURATION
            
    def _assess_difficulty_level(self, content: str, _metadata: Dict) -> DifficultyLevel:
        """Assess content difficulty level"""
        # Count technical terms
        technical_term_count = len(self._extract_classified_terms(content))
        
        # Check for complex concepts
        complex_indicators = ["algorithm", "optimization", "correlation", "analysis"]
        complex_count = sum(1 for indicator in complex_indicators if indicator in content.lower())
        
        if technical_term_count > 10 or complex_count > 2:
            return DifficultyLevel.EXPERT
        elif technical_term_count > 5 or complex_count > 1:
            return DifficultyLevel.ADVANCED
        elif technical_term_count > 2:
            return DifficultyLevel.INTERMEDIATE
        else:
            return DifficultyLevel.BASIC
            
    def _extract_parameters(self, content: str) -> List[Dict[str, Any]]:
        """Extract parameter mentions with classification"""
        parameters = []
        
        # Look for parameter patterns
        import re
        param_patterns = [
            r"(\w+\.\w+)",  # MO.parameter format
            r"(\w+Parameter)",  # ParameterName format
            r"set (\w+)",  # set parameter commands
        ]
        
        for pattern in param_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if match in self.parameter_registry:
                    param_info = self.parameter_registry[match]
                    parameters.append({
                        "name": match,
                        "mo_class": param_info["mo_class"],
                        "type": param_info["type"],
                        "importance": param_info["importance"]
                    })
                    
        return parameters
        
    def _extract_mo_classes(self, content: str) -> List[str]:
        """Extract MO class mentions"""
        mo_classes = []
        
        # Common MO class patterns
        mo_patterns = [
            "EUtranCellFDD", "EUtranCellTDD", "ENodeBFunction", "FeatureState",
            "CarrierAggregationFunction", "MobilityControlFunction", "LoadBalancingFunction"
        ]
        
        for mo_class in mo_patterns:
            if mo_class in content:
                mo_classes.append(mo_class)
                
        return mo_classes
        
    def _extract_counters(self, content: str) -> List[Dict[str, Any]]:
        """Extract counter mentions"""
        counters = []
        
        # Look for counter patterns
        import re
        counter_patterns = [
            r"pm(\w+)",  # PM counters
            r"(\w+Rate)",  # Rate counters
            r"(\w+Count)",  # Count counters
            r"(\w+Utilization)",  # Utilization counters
        ]
        
        for pattern in counter_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                counters.append({
                    "name": match,
                    "type": "pm",
                    "importance": 0.7  # Default importance
                })
                
        return counters
        
    def _analyze_conversation_structure(self, messages: List[Dict]) -> ConversationStructure:
        """Analyze conversation structure"""
        if not messages:
            return ConversationStructure(0, 0, [], [], 0.0)
            
        turn_count = len(messages)
        avg_turn_length = sum(len(msg.get("content", "")) for msg in messages) // turn_count
        
        # Analyze question types
        question_types = []
        response_patterns = []
        
        for msg in messages:
            content = msg.get("content", "").lower()
            if msg.get("role") == "user":
                if "how" in content:
                    question_types.append("how_to")
                elif "what" in content:
                    question_types.append("what_is")
                elif any(word in content for word in ["troubleshoot", "diagnose"]):
                    question_types.append("troubleshoot")
            elif msg.get("role") == "assistant":
                if any(word in content for word in ["configure", "set", "enable"]):
                    response_patterns.append("procedure")
                elif any(word in content for word in ["check", "verify", "validate"]):
                    response_patterns.append("diagnostic")
                else:
                    response_patterns.append("explanation")
                    
        # Simple coherence scoring based on content overlap
        coherence_score = 0.8  # Default value, could be improved with NLP
        
        return ConversationStructure(
            turn_count=turn_count,
            avg_turn_length=avg_turn_length,
            question_types=question_types,
            response_patterns=response_patterns,
            coherence_score=coherence_score
        )
        
    def _optimize_for_training(self, content: str, _metadata: Dict) -> TrainingOptimization:
        """Optimize settings for training effectiveness"""
        
        # Analyze content complexity
        technical_terms = len(self._extract_classified_terms(content))
        content_length = len(content)
        
        # Set instruction tuning weight based on content type
        if any(word in content.lower() for word in ["configure", "set", "how to"]):
            instruction_weight = 1.0
        else:
            instruction_weight = 0.8
            
        # Set embedding priority based on technical density
        embedding_priority = min(1.0, technical_terms / 10.0)
        
        # Determine context window requirement
        context_requirement = min(2048, max(512, content_length // 4))
        
        # Multi-turn capability
        multi_turn = "conversation" in content.lower() or "follow-up" in content.lower()
        
        # Reasoning complexity
        if technical_terms > 8:
            reasoning = "complex"
        elif technical_terms > 4:
            reasoning = "moderate"
        else:
            reasoning = "simple"
            
        # Domain specificity
        domain_specificity = min(1.0, technical_terms / 15.0)
        
        return TrainingOptimization(
            instruction_tuning_weight=instruction_weight,
            embedding_priority=embedding_priority,
            context_window_requirement=context_requirement,
            multi_turn_capability=multi_turn,
            reasoning_complexity=reasoning,
            domain_specificity=domain_specificity
        )
        
    def _assess_quality(self, record: Dict) -> QualityAssurance:
        """Assess record quality"""
        metadata = record.get("metadata", {})
        
        # Extract existing quality scores
        confidence = float(metadata.get("confidence", 0.8))
        quality_score = float(metadata.get("quality_score", 8.0)) / 10.0
        
        # Determine what quality checks were passed
        checks_passed = []
        if metadata.get("enhancement_applied"):
            checks_passed.append("enhancement_validation")
        if metadata.get("transformation_applied"):
            checks_passed.append("transformation_validation")
        if quality_score >= 0.8:
            checks_passed.append("quality_threshold")
            
        return QualityAssurance(
            validation_timestamp=datetime.now(timezone.utc).isoformat(),
            validation_version="1.0",
            quality_checks_passed=checks_passed,
            quality_issues=[],
            human_reviewed=False,
            confidence_score=confidence,
            technical_accuracy=quality_score,
            linguistic_quality=quality_score
        )
        
    def _extract_provenance(self, metadata: Dict) -> SourceProvenance:
        """Extract source provenance information"""
        return SourceProvenance(
            original_source=metadata.get("source_dataset", "unknown"),
            source_document_id=metadata.get("document_id"),
            extraction_method=metadata.get("processing_method", "automated"),
            processing_pipeline=metadata.get("transformer_stages", []),
            transformation_history=[],
            curator_notes=None,
            version_id="1.0"
        )
        
    def _generate_embedding_tags(self, content: str) -> List[str]:
        """Generate tags optimized for embedding retrieval"""
        tags = []
        
        # Add technical terms as tags
        terms = self._extract_classified_terms(content)
        tags.extend([term.term.lower() for term in terms])
        
        # Add conceptual tags
        if "configure" in content.lower():
            tags.append("configuration")
        if "troubleshoot" in content.lower():
            tags.append("troubleshooting")
        if "optimize" in content.lower():
            tags.append("optimization")
            
        return list(set(tags))  # Remove duplicates
        
    def _generate_retrieval_keywords(self, content: str) -> List[str]:
        """Generate keywords optimized for retrieval"""
        keywords = []
        
        # Extract important phrases
        import re
        phrases = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', content)  # Proper noun phrases
        keywords.extend(phrases)
        
        # Add technical terms
        terms = self._extract_classified_terms(content)
        keywords.extend([term.term for term in terms if term.context_importance > 0.7])
        
        return list(set(keywords))  # Remove duplicates

# Example usage
if __name__ == "__main__":
    # Test the metadata optimizer
    optimizer = MetadataOptimizer()
    
    # Sample input record
    sample_record = {
        "messages": [
            {
                "role": "user",
                "content": "How do I configure the EUtranCellFDD handoverMargin parameter for LTE optimization?"
            },
            {
                "role": "assistant",
                "content": "To configure the handoverMargin parameter in EUtranCellFDD, use the cmedit command: cmedit set EUtranCellFDD.handoverMargin=3. This parameter controls the RSRP offset for handover decisions."
            }
        ],
        "metadata": {
            "feature_name": "LTE Handover Optimization",
            "quality_score": 9.2,
            "technical_terms": ["LTE", "EUtranCellFDD", "handoverMargin", "RSRP"],
            "source_dataset": "enhanced_conversations",
            "enhancement_applied": True
        }
    }
    
    # Optimize metadata
    optimized = optimizer.optimize_metadata(sample_record)
    
    # Display results
    print("Optimized Metadata:")
    print(json.dumps(optimized.to_dict(), indent=2, default=str))