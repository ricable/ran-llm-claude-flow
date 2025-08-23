#!/usr/bin/env python3
"""
Semantic Quality Assessor for Multi-Model Pipeline

Advanced semantic quality assessment using embeddings, coherence analysis,
and technical domain validation. Complements Rust structural analysis with
semantic understanding for comprehensive quality scoring.

Features:
- Semantic coherence analysis with sentence transformers
- Technical domain relevance scoring
- Cross-document consistency validation  
- Quality prediction and trend analysis
- Performance-optimized for M3 Max processing

Author: Claude Code
Version: 2.0.0
"""

import asyncio
import logging
import time
import json
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from collections import defaultdict, Counter
import re

try:
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("Sentence transformers not available, using fallback methods")

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, limited analysis features")

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logging.info("Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available, using basic text processing")


class QualityDimension(Enum):
    """Quality assessment dimensions"""
    COHERENCE = "coherence"              # Logical flow and consistency
    RELEVANCE = "relevance"              # Technical domain relevance
    ACCURACY = "accuracy"                # Factual accuracy indicators
    COMPLETENESS = "completeness"        # Information completeness
    CLARITY = "clarity"                  # Text clarity and readability
    TECHNICAL_DEPTH = "technical_depth"  # Technical sophistication
    CONSISTENCY = "consistency"          # Internal consistency


class AssessmentLevel(Enum):
    """Assessment granularity levels"""
    DOCUMENT = "document"        # Individual document analysis
    SECTION = "section"          # Section-level analysis
    PARAGRAPH = "paragraph"      # Paragraph-level analysis
    SENTENCE = "sentence"        # Sentence-level analysis


@dataclass
class QualityScore:
    """Quality score with breakdown and metadata"""
    overall_score: float
    dimension_scores: Dict[QualityDimension, float]
    confidence: float
    sample_size: int
    assessment_time: float
    metadata: Dict[str, Any]


@dataclass
class SemanticAnalysis:
    """Detailed semantic analysis results"""
    coherence_score: float
    relevance_score: float
    technical_accuracy_score: float
    diversity_score: float
    completeness_score: float
    clarity_score: float
    
    # Detailed breakdowns
    sentence_coherence: List[float]
    topic_distribution: Dict[str, float]
    technical_term_relevance: Dict[str, float]
    concept_coverage: List[str]
    readability_metrics: Dict[str, float]
    
    # Quality indicators
    quality_indicators: List[str]
    quality_issues: List[str]
    improvement_suggestions: List[str]


@dataclass
class CrossDocumentAnalysis:
    """Cross-document consistency and similarity analysis"""
    consistency_score: float
    similarity_matrix: np.ndarray
    topic_overlap: Dict[str, float]
    terminology_consistency: float
    structural_similarity: float
    content_diversity: float
    duplicate_detection: List[Tuple[int, int, float]]  # (doc1_idx, doc2_idx, similarity)


@dataclass
class TechnicalDomainProfile:
    """Technical domain profile for relevance assessment"""
    domain_keywords: Set[str]
    technical_patterns: List[str]
    domain_embeddings: Optional[np.ndarray]
    concept_hierarchies: Dict[str, List[str]]
    domain_weights: Dict[str, float]


class SemanticQualityAssessor:
    """
    Advanced semantic quality assessor with ML-based analysis.
    
    Provides comprehensive semantic quality assessment including:
    - Embedding-based coherence analysis
    - Technical domain relevance scoring
    - Cross-document consistency validation
    - Quality prediction and optimization
    """
    
    def __init__(self, config_path: Optional[Path] = None, device: str = "auto"):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or Path("config/quality_config.yaml")
        
        # Device selection for Apple Silicon optimization
        if device == "auto":
            if torch and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            elif torch and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Initialize models
        self.embedding_model: Optional[SentenceTransformer] = None
        self.domain_profiles: Dict[str, TechnicalDomainProfile] = {}
        
        # Quality assessment cache
        self.assessment_cache: Dict[str, QualityScore] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        # Performance tracking
        self.assessment_history: List[Dict] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # Configuration
        self.config = self._load_config()
        
        # Initialize components
        asyncio.create_task(self._initialize_models())
    
    async def _initialize_models(self):
        """Initialize embedding models and domain profiles"""
        try:
            if EMBEDDINGS_AVAILABLE:
                # Load optimized sentence transformer model
                model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
                self.embedding_model = SentenceTransformer(model_name, device=self.device)
                
                # Optimize for Apple Silicon
                if self.device == "mps":
                    # Enable Metal Performance Shaders optimizations
                    self.embedding_model = self.embedding_model.to(self.device)
                
                self.logger.info(f"Initialized embedding model '{model_name}' on device '{self.device}'")
            else:
                self.logger.warning("Embedding model not available, using fallback methods")
            
            # Initialize domain profiles
            await self._initialize_domain_profiles()
            
            self.logger.info("Semantic quality assessor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            self.embedding_model = None
    
    async def _initialize_domain_profiles(self):
        """Initialize technical domain profiles for RAN/telecom"""
        # RAN/Telecom domain profile
        ran_keywords = {
            # Network elements
            "enodeb", "gnodeb", "ue", "mme", "sgw", "pgw", "hss", "pcrf",
            "amf", "smf", "upf", "nssf", "ausf", "udm", "pcf", "nrf",
            
            # Radio technologies
            "lte", "nr", "5g", "4g", "3g", "gsm", "umts", "wcdma",
            "ofdm", "mimo", "ca", "comp", "harq", "arq",
            
            # Protocol layers
            "rrc", "mac", "rlc", "pdcp", "phy", "nas", "x2", "s1", "ng",
            
            # Measurements and KPIs
            "rsrp", "rsrq", "sinr", "cqi", "pmi", "ri", "bler",
            "throughput", "latency", "jitter", "packet_loss",
            
            # Parameters and counters
            "parameter", "counter", "kpi", "threshold", "timer",
            "configuration", "setting", "optimization",
            
            # Procedures
            "handover", "attachment", "authentication", "registration",
            "paging", "scheduling", "mobility", "load_balancing"
        }
        
        ran_patterns = [
            r"\b[a-zA-Z]+(?:Id|Identifier)\b",  # Identifiers
            r"\b[a-zA-Z]+(?:Threshold|Level)\b",  # Thresholds
            r"\b[a-zA-Z]+(?:Timer|Timeout)\b",    # Timers
            r"\b[a-zA-Z]+(?:Count|Counter|Rate)\b",  # Counters
            r"\bdB[mW]?\b",  # Power units
            r"\b\d+\s*(?:MHz|GHz|kHz)\b",  # Frequency
            r"\b(?:QCI|ARP|GBR|MBR)\b",  # QoS parameters
        ]
        
        # Create embeddings for domain keywords if available
        domain_embeddings = None
        if self.embedding_model:
            try:
                keyword_list = list(ran_keywords)
                domain_embeddings = self.embedding_model.encode(keyword_list)
                self.logger.info(f"Generated embeddings for {len(keyword_list)} domain keywords")
            except Exception as e:
                self.logger.warning(f"Failed to generate domain embeddings: {e}")
        
        # Concept hierarchies for RAN domain
        concept_hierarchies = {
            "network_elements": ["enodeb", "gnodeb", "mme", "sgw", "pgw", "ue"],
            "radio_tech": ["lte", "nr", "5g", "4g", "mimo", "ca", "harq"],
            "measurements": ["rsrp", "rsrq", "sinr", "cqi", "bler"],
            "procedures": ["handover", "attachment", "paging", "scheduling"],
            "parameters": ["threshold", "timer", "counter", "configuration"]
        }
        
        # Domain-specific weights
        domain_weights = {
            "exact_parameters": 1.0,
            "network_elements": 0.9,
            "measurements": 0.8,
            "procedures": 0.7,
            "general_telecom": 0.6,
            "technical_general": 0.4
        }
        
        self.domain_profiles["ran_telecom"] = TechnicalDomainProfile(
            domain_keywords=ran_keywords,
            technical_patterns=ran_patterns,
            domain_embeddings=domain_embeddings,
            concept_hierarchies=concept_hierarchies,
            domain_weights=domain_weights
        )
        
        self.logger.info("Initialized RAN/Telecom domain profile")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load quality assessment configuration"""
        default_config = {
            "embedding_model": "all-MiniLM-L6-v2",
            "quality_weights": {
                QualityDimension.COHERENCE.value: 0.20,
                QualityDimension.RELEVANCE.value: 0.25,
                QualityDimension.ACCURACY.value: 0.20,
                QualityDimension.COMPLETENESS.value: 0.15,
                QualityDimension.CLARITY.value: 0.10,
                QualityDimension.TECHNICAL_DEPTH.value: 0.10
            },
            "coherence_threshold": 0.7,
            "relevance_threshold": 0.6,
            "diversity_threshold": 0.3,
            "enable_caching": True,
            "batch_size": 32,
            "max_sequence_length": 512
        }
        
        try:
            if self.config_path.exists():
                import yaml
                with open(self.config_path) as f:
                    loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
        except Exception as e:
            self.logger.warning(f"Failed to load config from {self.config_path}: {e}")
        
        return default_config
    
    async def assess_document_quality(
        self, 
        content: str, 
        metadata: Optional[Dict] = None,
        domain: str = "ran_telecom"
    ) -> QualityScore:
        """
        Comprehensive semantic quality assessment of a document.
        
        Args:
            content: Document content to assess
            metadata: Optional metadata for context
            domain: Technical domain for relevance assessment
            
        Returns:
            QualityScore with detailed breakdown
        """
        start_time = time.time()
        
        # Check cache
        content_hash = str(hash(content))
        if self.config.get("enable_caching", True) and content_hash in self.assessment_cache:
            cached_score = self.assessment_cache[content_hash]
            self.logger.debug(f"Using cached quality assessment (score: {cached_score.overall_score:.3f})")
            return cached_score
        
        self.logger.debug(f"Assessing semantic quality for document ({len(content)} chars)")
        
        # Perform detailed semantic analysis
        analysis = await self._perform_semantic_analysis(content, metadata, domain)
        
        # Calculate dimension scores
        dimension_scores = {
            QualityDimension.COHERENCE: analysis.coherence_score,
            QualityDimension.RELEVANCE: analysis.relevance_score,
            QualityDimension.ACCURACY: analysis.technical_accuracy_score,
            QualityDimension.COMPLETENESS: analysis.completeness_score,
            QualityDimension.CLARITY: analysis.clarity_score,
            QualityDimension.TECHNICAL_DEPTH: analysis.diversity_score,  # Using diversity as depth proxy
        }
        
        # Calculate weighted overall score
        weights = self.config.get("quality_weights", {})
        overall_score = sum(
            dimension_scores.get(dim, 0.0) * weights.get(dim.value, 1.0/len(dimension_scores))
            for dim in QualityDimension
        )
        overall_score = min(overall_score, 1.0)
        
        # Calculate confidence based on analysis completeness
        confidence = self._calculate_assessment_confidence(analysis, content)
        
        # Create quality score
        quality_score = QualityScore(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            confidence=confidence,
            sample_size=len(content.split()),
            assessment_time=time.time() - start_time,
            metadata={
                "analysis": asdict(analysis),
                "domain": domain,
                "content_length": len(content),
                "assessment_version": "2.0.0"
            }
        )
        
        # Cache result
        if self.config.get("enable_caching", True):
            self.assessment_cache[content_hash] = quality_score
        
        # Update performance tracking
        self._update_performance_metrics(quality_score)
        
        self.logger.info(
            f"Quality assessment complete: overall={overall_score:.3f}, "
            f"coherence={analysis.coherence_score:.3f}, "
            f"relevance={analysis.relevance_score:.3f}, "
            f"time={quality_score.assessment_time:.2f}s"
        )
        
        return quality_score
    
    async def _perform_semantic_analysis(
        self, 
        content: str, 
        metadata: Optional[Dict],
        domain: str
    ) -> SemanticAnalysis:
        """Perform comprehensive semantic analysis"""
        
        # Text preprocessing
        sentences = self._extract_sentences(content)
        paragraphs = self._extract_paragraphs(content)
        
        # Parallel analysis tasks
        coherence_task = self._analyze_coherence(sentences, content)
        relevance_task = self._analyze_domain_relevance(content, domain)
        accuracy_task = self._analyze_technical_accuracy(content, domain)
        diversity_task = self._analyze_content_diversity(sentences)
        completeness_task = self._analyze_completeness(content, metadata)
        clarity_task = self._analyze_clarity(content, sentences)
        
        # Execute analysis tasks
        results = await asyncio.gather(
            coherence_task,
            relevance_task, 
            accuracy_task,
            diversity_task,
            completeness_task,
            clarity_task,
            return_exceptions=True
        )
        
        # Extract results with error handling
        coherence_result = results[0] if not isinstance(results[0], Exception) else (0.5, [], {})
        relevance_result = results[1] if not isinstance(results[1], Exception) else (0.5, {}, {})
        accuracy_result = results[2] if not isinstance(results[2], Exception) else (0.5, {})
        diversity_result = results[3] if not isinstance(results[3], Exception) else (0.5, [], {})
        completeness_result = results[4] if not isinstance(results[4], Exception) else (0.5, [])
        clarity_result = results[5] if not isinstance(results[5], Exception) else (0.5, {})
        
        # Extract individual scores
        coherence_score, sentence_coherence, topic_dist = coherence_result
        relevance_score, term_relevance, domain_match = relevance_result
        accuracy_score, accuracy_indicators = accuracy_result
        diversity_score, concept_list, diversity_metrics = diversity_result
        completeness_score, completeness_indicators = completeness_result
        clarity_score, readability_metrics = clarity_result
        
        # Generate quality indicators and issues
        quality_indicators = []
        quality_issues = []
        improvement_suggestions = []
        
        # Quality indicators
        if coherence_score > 0.8:
            quality_indicators.append("High coherence and logical flow")
        if relevance_score > 0.8:
            quality_indicators.append("Strong domain relevance")
        if accuracy_score > 0.8:
            quality_indicators.append("Good technical accuracy indicators")
        if diversity_score > 0.7:
            quality_indicators.append("Rich content diversity")
        
        # Quality issues  
        if coherence_score < 0.6:
            quality_issues.append("Low coherence between sections")
            improvement_suggestions.append("Improve logical flow and transitions")
        if relevance_score < 0.5:
            quality_issues.append("Limited technical domain relevance")
            improvement_suggestions.append("Add more domain-specific technical content")
        if accuracy_score < 0.6:
            quality_issues.append("Weak technical accuracy indicators")
            improvement_suggestions.append("Include more specific technical details")
        if diversity_score < 0.4:
            quality_issues.append("Limited content diversity")
            improvement_suggestions.append("Expand topic coverage and examples")
        
        return SemanticAnalysis(
            coherence_score=coherence_score,
            relevance_score=relevance_score,
            technical_accuracy_score=accuracy_score,
            diversity_score=diversity_score,
            completeness_score=completeness_score,
            clarity_score=clarity_score,
            sentence_coherence=sentence_coherence,
            topic_distribution=topic_dist,
            technical_term_relevance=term_relevance,
            concept_coverage=concept_list,
            readability_metrics=readability_metrics,
            quality_indicators=quality_indicators,
            quality_issues=quality_issues,
            improvement_suggestions=improvement_suggestions
        )
    
    async def _analyze_coherence(
        self, 
        sentences: List[str], 
        content: str
    ) -> Tuple[float, List[float], Dict[str, float]]:
        """Analyze semantic coherence using embeddings"""
        if not sentences or len(sentences) < 2:
            return 0.5, [], {}
        
        if not self.embedding_model:
            # Fallback: simple lexical coherence
            return self._fallback_coherence_analysis(sentences, content)
        
        try:
            # Get sentence embeddings
            embeddings = await self._get_embeddings(sentences[:50])  # Limit for performance
            
            if embeddings is None or len(embeddings) < 2:
                return 0.5, [], {}
            
            # Calculate pairwise similarities
            similarities = []
            sentence_coherence = []
            
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                similarities.append(sim)
                sentence_coherence.append(sim)
            
            # Overall coherence score
            coherence_score = np.mean(similarities) if similarities else 0.5
            
            # Topic distribution analysis
            topic_distribution = {}
            if SKLEARN_AVAILABLE and len(embeddings) > 3:
                try:
                    # Simple clustering for topic identification
                    n_topics = min(5, len(embeddings) // 2)
                    kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
                    topic_labels = kmeans.fit_predict(embeddings)
                    
                    # Calculate topic distribution
                    topic_counts = Counter(topic_labels)
                    total_sentences = len(topic_labels)
                    
                    for topic_id, count in topic_counts.items():
                        topic_distribution[f"topic_{topic_id}"] = count / total_sentences
                        
                except Exception as e:
                    self.logger.debug(f"Topic clustering failed: {e}")
            
            return coherence_score, sentence_coherence, topic_distribution
            
        except Exception as e:
            self.logger.warning(f"Coherence analysis failed: {e}")
            return self._fallback_coherence_analysis(sentences, content)
    
    def _fallback_coherence_analysis(
        self, 
        sentences: List[str], 
        content: str
    ) -> Tuple[float, List[float], Dict[str, float]]:
        """Fallback coherence analysis without embeddings"""
        if not sentences:
            return 0.5, [], {}
        
        # Simple lexical overlap analysis
        sentence_coherence = []
        
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i + 1].lower().split())
            
            # Jaccard similarity
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            similarity = intersection / union if union > 0 else 0.0
            sentence_coherence.append(similarity)
        
        coherence_score = np.mean(sentence_coherence) if sentence_coherence else 0.5
        
        # Simple topic distribution based on word frequency
        words = content.lower().split()
        word_freq = Counter(words)
        
        # Get most common words as topic indicators
        common_words = word_freq.most_common(5)
        total_words = len(words)
        
        topic_distribution = {}
        for i, (word, count) in enumerate(common_words):
            topic_distribution[f"topic_{word}"] = count / total_words
        
        return coherence_score, sentence_coherence, topic_distribution
    
    async def _analyze_domain_relevance(
        self, 
        content: str, 
        domain: str
    ) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """Analyze technical domain relevance"""
        if domain not in self.domain_profiles:
            return 0.5, {}, {}
        
        profile = self.domain_profiles[domain]
        content_lower = content.lower()
        
        # Keyword matching analysis
        keyword_matches = {}
        total_keywords = len(profile.domain_keywords)
        matched_keywords = 0
        
        for keyword in profile.domain_keywords:
            if keyword in content_lower:
                # Count occurrences
                count = content_lower.count(keyword)
                keyword_matches[keyword] = count
                matched_keywords += 1
        
        # Keyword coverage score
        keyword_coverage = matched_keywords / total_keywords if total_keywords > 0 else 0.0
        
        # Pattern matching analysis
        pattern_matches = 0
        total_patterns = len(profile.technical_patterns)
        
        for pattern in profile.technical_patterns:
            try:
                if re.search(pattern, content, re.IGNORECASE):
                    pattern_matches += 1
            except re.error:
                continue
        
        pattern_coverage = pattern_matches / total_patterns if total_patterns > 0 else 0.0
        
        # Embedding-based similarity (if available)
        embedding_similarity = 0.0
        if self.embedding_model and profile.domain_embeddings is not None:
            try:
                content_embedding = await self._get_embeddings([content])
                if content_embedding is not None and len(content_embedding) > 0:
                    # Calculate similarity to domain embeddings
                    similarities = cosine_similarity(
                        content_embedding, 
                        profile.domain_embeddings
                    )
                    embedding_similarity = np.max(similarities)
            except Exception as e:
                self.logger.debug(f"Embedding similarity calculation failed: {e}")
        
        # Combine scores
        relevance_score = (
            keyword_coverage * 0.4 + 
            pattern_coverage * 0.3 + 
            embedding_similarity * 0.3
        )
        
        domain_analysis = {
            "keyword_coverage": keyword_coverage,
            "pattern_coverage": pattern_coverage,
            "embedding_similarity": embedding_similarity,
            "matched_keywords": len(keyword_matches),
            "total_keywords": total_keywords,
            "pattern_matches": pattern_matches
        }
        
        return relevance_score, keyword_matches, domain_analysis
    
    async def _analyze_technical_accuracy(
        self, 
        content: str, 
        domain: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Analyze technical accuracy indicators"""
        accuracy_indicators = {}
        
        # Technical depth indicators
        technical_terms = self._extract_technical_terms(content)
        accuracy_indicators["technical_term_count"] = len(technical_terms)
        
        # Specificity indicators
        specific_numbers = re.findall(r'\b\d+(?:\.\d+)?\s*(?:MHz|GHz|dBm?|ms|seconds?)\b', content, re.IGNORECASE)
        accuracy_indicators["specific_measurements"] = len(specific_numbers)
        
        # Parameter/configuration mentions
        param_patterns = [
            r'\b\w+(?:Id|Identifier|Parameter|Config|Setting|Threshold)\b',
            r'\b\w+\s*[:=]\s*\w+',
            r'\b(?:set|configure|enable|disable)\s+\w+'
        ]
        
        param_mentions = 0
        for pattern in param_patterns:
            param_mentions += len(re.findall(pattern, content, re.IGNORECASE))
        
        accuracy_indicators["parameter_mentions"] = param_mentions
        
        # Reference indicators (specs, standards, etc.)
        reference_patterns = [
            r'\b3GPP\s+\w+',
            r'\bTS\s+\d+\.\d+',
            r'\bRFC\s+\d+',
            r'\bsection\s+\d+(?:\.\d+)*',
            r'\bfigure\s+\d+'
        ]
        
        reference_count = 0
        for pattern in reference_patterns:
            reference_count += len(re.findall(pattern, content, re.IGNORECASE))
        
        accuracy_indicators["reference_count"] = reference_count
        
        # Calculate accuracy score
        term_score = min(len(technical_terms) / 20.0, 1.0) * 0.3
        measurement_score = min(len(specific_numbers) / 10.0, 1.0) * 0.25
        param_score = min(param_mentions / 15.0, 1.0) * 0.25
        reference_score = min(reference_count / 5.0, 1.0) * 0.2
        
        accuracy_score = term_score + measurement_score + param_score + reference_score
        
        accuracy_indicators["scores"] = {
            "technical_terms": term_score,
            "measurements": measurement_score,
            "parameters": param_score,
            "references": reference_score
        }
        
        return accuracy_score, accuracy_indicators
    
    async def _analyze_content_diversity(
        self, 
        sentences: List[str]
    ) -> Tuple[float, List[str], Dict[str, float]]:
        """Analyze content diversity and topic coverage"""
        if not sentences:
            return 0.0, [], {}
        
        # Extract concepts/topics from sentences
        concepts = set()
        for sentence in sentences:
            # Extract noun phrases and technical terms
            words = sentence.lower().split()
            
            # Simple noun phrase extraction (can be enhanced with NLP)
            for word in words:
                if len(word) > 3 and word.isalpha():
                    concepts.add(word)
        
        concept_list = list(concepts)
        
        # Calculate diversity metrics
        diversity_metrics = {}
        
        # Lexical diversity (unique words / total words)
        all_words = []
        for sentence in sentences:
            all_words.extend(sentence.lower().split())
        
        if all_words:
            unique_words = set(all_words)
            lexical_diversity = len(unique_words) / len(all_words)
        else:
            lexical_diversity = 0.0
        
        diversity_metrics["lexical_diversity"] = lexical_diversity
        
        # Sentence length diversity
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        if sentence_lengths:
            length_variance = np.var(sentence_lengths) / (np.mean(sentence_lengths) ** 2)
            diversity_metrics["length_diversity"] = min(length_variance, 1.0)
        else:
            diversity_metrics["length_diversity"] = 0.0
        
        # Topic diversity (concept coverage)
        concept_diversity = min(len(concepts) / 50.0, 1.0)  # Normalize to 50 concepts
        diversity_metrics["concept_diversity"] = concept_diversity
        
        # Overall diversity score
        diversity_score = (
            lexical_diversity * 0.4 + 
            diversity_metrics["length_diversity"] * 0.2 + 
            concept_diversity * 0.4
        )
        
        return diversity_score, concept_list[:20], diversity_metrics  # Limit concepts for performance
    
    async def _analyze_completeness(
        self, 
        content: str, 
        metadata: Optional[Dict]
    ) -> Tuple[float, List[str]]:
        """Analyze content completeness indicators"""
        completeness_indicators = []
        
        # Check for common documentation sections
        sections = [
            ("description", r"(?i)\b(?:description|overview|summary)\b"),
            ("parameters", r"(?i)\b(?:parameter|config|setting)\b"),
            ("examples", r"(?i)\b(?:example|sample|demo)\b"),
            ("usage", r"(?i)\b(?:usage|how\s+to|instruction)\b"),
            ("troubleshooting", r"(?i)\b(?:troubleshoot|problem|issue|error)\b"),
            ("references", r"(?i)\b(?:reference|see\s+also|related)\b")
        ]
        
        section_coverage = 0
        for section_name, pattern in sections:
            if re.search(pattern, content):
                completeness_indicators.append(f"Contains {section_name} section")
                section_coverage += 1
        
        # Check metadata completeness
        metadata_completeness = 0
        if metadata:
            expected_fields = ["title", "feature_name", "parameters", "counters"]
            for field in expected_fields:
                if field in metadata and metadata[field]:
                    metadata_completeness += 1
                    completeness_indicators.append(f"Has {field} metadata")
        
        # Content length appropriateness
        content_length = len(content)
        if content_length > 500:
            completeness_indicators.append("Adequate content length")
        if content_length > 2000:
            completeness_indicators.append("Comprehensive content length")
        
        # Calculate completeness score
        section_score = section_coverage / len(sections)
        metadata_score = metadata_completeness / 4.0 if metadata else 0.5
        length_score = min(content_length / 2000.0, 1.0)
        
        completeness_score = (section_score * 0.5 + metadata_score * 0.3 + length_score * 0.2)
        
        return completeness_score, completeness_indicators
    
    async def _analyze_clarity(
        self, 
        content: str, 
        sentences: List[str]
    ) -> Tuple[float, Dict[str, float]]:
        """Analyze text clarity and readability"""
        readability_metrics = {}
        
        if not sentences:
            return 0.5, readability_metrics
        
        # Average sentence length
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        readability_metrics["avg_sentence_length"] = avg_sentence_length
        
        # Sentence length variance (consistency)
        sentence_length_variance = np.var(sentence_lengths) if sentence_lengths else 0
        readability_metrics["sentence_length_variance"] = sentence_length_variance
        
        # Word complexity (average word length)
        all_words = content.split()
        if all_words:
            avg_word_length = np.mean([len(word) for word in all_words])
            readability_metrics["avg_word_length"] = avg_word_length
        else:
            avg_word_length = 0
            readability_metrics["avg_word_length"] = 0
        
        # Simple readability score (inverse of complexity)
        # Optimal sentence length: 15-20 words
        sentence_length_score = 1.0 - min(abs(avg_sentence_length - 17.5) / 17.5, 1.0)
        
        # Optimal word length: 4-6 characters for technical content
        word_length_score = 1.0 - min(abs(avg_word_length - 5) / 5, 1.0)
        
        # Sentence consistency (lower variance is better)
        consistency_score = 1.0 / (1.0 + sentence_length_variance / 100.0)
        
        clarity_score = (
            sentence_length_score * 0.4 + 
            word_length_score * 0.3 + 
            consistency_score * 0.3
        )
        
        readability_metrics["clarity_components"] = {
            "sentence_length_score": sentence_length_score,
            "word_length_score": word_length_score,
            "consistency_score": consistency_score
        }
        
        return clarity_score, readability_metrics
    
    async def assess_cross_document_consistency(
        self, 
        documents: List[Tuple[str, Optional[Dict]]]
    ) -> CrossDocumentAnalysis:
        """Analyze consistency and similarity across multiple documents"""
        if len(documents) < 2:
            return CrossDocumentAnalysis(
                consistency_score=1.0,
                similarity_matrix=np.array([[1.0]]),
                topic_overlap={},
                terminology_consistency=1.0,
                structural_similarity=1.0,
                content_diversity=0.0,
                duplicate_detection=[]
            )
        
        self.logger.info(f"Analyzing cross-document consistency for {len(documents)} documents")
        
        contents = [doc[0] for doc in documents]
        metadatas = [doc[1] for doc in documents]
        
        # Calculate similarity matrix
        similarity_matrix = await self._calculate_similarity_matrix(contents)
        
        # Analyze topic overlap
        topic_overlap = await self._analyze_topic_overlap(contents)
        
        # Analyze terminology consistency
        terminology_consistency = self._analyze_terminology_consistency(contents)
        
        # Analyze structural similarity
        structural_similarity = self._analyze_structural_similarity(contents)
        
        # Calculate content diversity
        content_diversity = self._calculate_content_diversity(similarity_matrix)
        
        # Detect potential duplicates
        duplicate_detection = self._detect_duplicates(similarity_matrix, threshold=0.85)
        
        # Calculate overall consistency score
        avg_similarity = np.mean(similarity_matrix[np.triu_indices(len(contents), k=1)])
        consistency_score = (
            avg_similarity * 0.4 +
            terminology_consistency * 0.3 +
            structural_similarity * 0.3
        )
        
        return CrossDocumentAnalysis(
            consistency_score=consistency_score,
            similarity_matrix=similarity_matrix,
            topic_overlap=topic_overlap,
            terminology_consistency=terminology_consistency,
            structural_similarity=structural_similarity,
            content_diversity=content_diversity,
            duplicate_detection=duplicate_detection
        )
    
    async def _calculate_similarity_matrix(self, contents: List[str]) -> np.ndarray:
        """Calculate pairwise similarity matrix for documents"""
        n_docs = len(contents)
        similarity_matrix = np.eye(n_docs)
        
        if self.embedding_model:
            try:
                # Get embeddings for all documents
                embeddings = await self._get_embeddings(contents)
                if embeddings is not None:
                    # Calculate cosine similarity matrix
                    similarity_matrix = cosine_similarity(embeddings)
                    return similarity_matrix
            except Exception as e:
                self.logger.warning(f"Failed to calculate embedding similarity: {e}")
        
        # Fallback: lexical similarity
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                words_i = set(contents[i].lower().split())
                words_j = set(contents[j].lower().split())
                
                # Jaccard similarity
                intersection = len(words_i & words_j)
                union = len(words_i | words_j)
                
                similarity = intersection / union if union > 0 else 0.0
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    async def _analyze_topic_overlap(self, contents: List[str]) -> Dict[str, float]:
        """Analyze topic overlap across documents"""
        # Extract topics from each document
        all_topics = []
        doc_topics = []
        
        for content in contents:
            topics = self._extract_technical_terms(content)
            doc_topics.append(set(topics))
            all_topics.extend(topics)
        
        # Calculate topic statistics
        topic_counts = Counter(all_topics)
        unique_topics = set(all_topics)
        
        # Calculate overlap metrics
        topic_overlap = {}
        
        if len(doc_topics) > 1:
            # Pairwise topic overlap
            overlaps = []
            for i in range(len(doc_topics)):
                for j in range(i + 1, len(doc_topics)):
                    intersection = len(doc_topics[i] & doc_topics[j])
                    union = len(doc_topics[i] | doc_topics[j])
                    overlap = intersection / union if union > 0 else 0.0
                    overlaps.append(overlap)
            
            topic_overlap["average_pairwise_overlap"] = np.mean(overlaps) if overlaps else 0.0
            topic_overlap["max_pairwise_overlap"] = np.max(overlaps) if overlaps else 0.0
            topic_overlap["min_pairwise_overlap"] = np.min(overlaps) if overlaps else 0.0
        
        # Global topic statistics
        topic_overlap["unique_topics_count"] = len(unique_topics)
        topic_overlap["total_topic_mentions"] = len(all_topics)
        topic_overlap["topic_diversity"] = len(unique_topics) / len(all_topics) if all_topics else 0.0
        
        return topic_overlap
    
    def _analyze_terminology_consistency(self, contents: List[str]) -> float:
        """Analyze consistency of terminology usage across documents"""
        # Extract technical terms from each document
        all_terms = []
        doc_term_sets = []
        
        for content in contents:
            terms = set(self._extract_technical_terms(content))
            doc_term_sets.append(terms)
            all_terms.extend(terms)
        
        if not all_terms:
            return 1.0
        
        # Count term usage across documents
        term_counts = Counter(all_terms)
        
        # Calculate consistency score
        # Terms used consistently across documents score higher
        consistency_scores = []
        
        for term, count in term_counts.items():
            # Consistency = usage frequency / number of documents
            consistency = count / len(contents)
            consistency_scores.append(consistency)
        
        # Weight by term frequency
        weighted_consistency = 0.0
        total_weight = 0.0
        
        for term, count in term_counts.items():
            consistency = count / len(contents)
            weight = count  # Terms used more often have higher weight
            weighted_consistency += consistency * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_consistency / total_weight
        else:
            return 1.0
    
    def _analyze_structural_similarity(self, contents: List[str]) -> float:
        """Analyze structural similarity across documents"""
        # Extract structural features
        structural_features = []
        
        for content in contents:
            features = {
                "sentence_count": len(self._extract_sentences(content)),
                "paragraph_count": len(self._extract_paragraphs(content)),
                "avg_sentence_length": np.mean([len(s.split()) for s in self._extract_sentences(content)]) if self._extract_sentences(content) else 0,
                "header_count": content.count('#'),
                "list_count": content.count('- ') + content.count('* '),
                "code_block_count": content.count('```'),
                "content_length": len(content)
            }
            structural_features.append(features)
        
        if len(structural_features) < 2:
            return 1.0
        
        # Calculate pairwise structural similarity
        similarities = []
        
        for i in range(len(structural_features)):
            for j in range(i + 1, len(structural_features)):
                feat_i = structural_features[i]
                feat_j = structural_features[j]
                
                # Calculate normalized differences for each feature
                feature_similarities = []
                
                for key in feat_i.keys():
                    val_i = feat_i[key]
                    val_j = feat_j[key]
                    
                    if val_i == 0 and val_j == 0:
                        similarity = 1.0
                    elif val_i == 0 or val_j == 0:
                        similarity = 0.0
                    else:
                        # Normalized similarity (1 - relative difference)
                        similarity = 1.0 - abs(val_i - val_j) / max(val_i, val_j)
                    
                    feature_similarities.append(similarity)
                
                # Average feature similarity
                overall_similarity = np.mean(feature_similarities)
                similarities.append(overall_similarity)
        
        return np.mean(similarities) if similarities else 1.0
    
    def _calculate_content_diversity(self, similarity_matrix: np.ndarray) -> float:
        """Calculate content diversity from similarity matrix"""
        if similarity_matrix.shape[0] < 2:
            return 0.0
        
        # Extract upper triangular values (excluding diagonal)
        upper_tri_indices = np.triu_indices(similarity_matrix.shape[0], k=1)
        similarities = similarity_matrix[upper_tri_indices]
        
        # Diversity is inverse of average similarity
        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity
        
        return max(0.0, diversity)
    
    def _detect_duplicates(
        self, 
        similarity_matrix: np.ndarray, 
        threshold: float = 0.85
    ) -> List[Tuple[int, int, float]]:
        """Detect potential duplicate documents"""
        duplicates = []
        
        for i in range(similarity_matrix.shape[0]):
            for j in range(i + 1, similarity_matrix.shape[1]):
                similarity = similarity_matrix[i, j]
                if similarity >= threshold:
                    duplicates.append((i, j, similarity))
        
        # Sort by similarity (highest first)
        duplicates.sort(key=lambda x: x[2], reverse=True)
        
        return duplicates
    
    def _extract_sentences(self, content: str) -> List[str]:
        """Extract sentences from content"""
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(content)
            except Exception:
                pass
        
        # Fallback: simple sentence splitting
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_paragraphs(self, content: str) -> List[str]:
        """Extract paragraphs from content"""
        paragraphs = content.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _extract_technical_terms(self, content: str) -> List[str]:
        """Extract technical terms from content"""
        # Combined pattern for technical terms
        patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b[a-zA-Z]+(?:Id|Parameter|Config|Counter|Timer|Threshold|Level)\b',  # Technical identifiers
            r'\b(?:eNodeB|gNodeB|MME|SGW|PGW|HSS|UE)\b',  # Network elements
            r'\b(?:LTE|NR|5G|4G|3G|GSM|UMTS)\b',  # Technologies
            r'\b(?:RSRP|RSRQ|SINR|CQI|PMI|RI|BLER)\b',  # Measurements
            r'\b\d+(?:\.\d+)?\s*(?:MHz|GHz|dBm?|ms)\b'  # Technical values
        ]
        
        technical_terms = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            technical_terms.extend(matches)
        
        # Remove duplicates and convert to lowercase
        return list(set(term.lower() for term in technical_terms))
    
    async def _get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Get embeddings for texts with caching"""
        if not self.embedding_model or not texts:
            return None
        
        # Check cache
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            text_hash = str(hash(text))
            if text_hash in self.embedding_cache:
                cached_embeddings.append((i, self.embedding_cache[text_hash]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Calculate embeddings for uncached texts
        if uncached_texts:
            try:
                batch_size = self.config.get("batch_size", 32)
                new_embeddings = []
                
                for i in range(0, len(uncached_texts), batch_size):
                    batch = uncached_texts[i:i + batch_size]
                    batch_embeddings = self.embedding_model.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                    new_embeddings.extend(batch_embeddings)
                
                # Cache new embeddings
                for text, embedding in zip(uncached_texts, new_embeddings):
                    text_hash = str(hash(text))
                    self.embedding_cache[text_hash] = embedding
                
                # Combine cached and new embeddings
                all_embeddings = [None] * len(texts)
                
                # Place cached embeddings
                for idx, embedding in cached_embeddings:
                    all_embeddings[idx] = embedding
                
                # Place new embeddings
                for i, embedding in enumerate(new_embeddings):
                    original_idx = uncached_indices[i]
                    all_embeddings[original_idx] = embedding
                
                return np.array(all_embeddings)
                
            except Exception as e:
                self.logger.error(f"Failed to calculate embeddings: {e}")
                return None
        else:
            # All embeddings were cached
            embeddings = [None] * len(texts)
            for idx, embedding in cached_embeddings:
                embeddings[idx] = embedding
            return np.array(embeddings)
    
    def _calculate_assessment_confidence(
        self, 
        analysis: SemanticAnalysis, 
        content: str
    ) -> float:
        """Calculate confidence in the quality assessment"""
        confidence_factors = []
        
        # Content length factor
        length_factor = min(len(content) / 1000.0, 1.0)  # Confidence increases with length up to 1000 chars
        confidence_factors.append(length_factor)
        
        # Analysis completeness factor
        completeness_factor = 1.0  # Start with full confidence
        
        # Reduce confidence if analysis components failed
        if not analysis.sentence_coherence:
            completeness_factor *= 0.8
        if not analysis.technical_term_relevance:
            completeness_factor *= 0.9
        if not analysis.concept_coverage:
            completeness_factor *= 0.9
        
        confidence_factors.append(completeness_factor)
        
        # Model availability factor
        model_factor = 1.0 if self.embedding_model else 0.7
        confidence_factors.append(model_factor)
        
        # Sample size factor (for coherence analysis)
        sample_size = len(analysis.sentence_coherence)
        sample_factor = min(sample_size / 10.0, 1.0)  # Full confidence with 10+ sentences
        confidence_factors.append(sample_factor)
        
        # Overall confidence is the geometric mean
        overall_confidence = np.prod(confidence_factors) ** (1.0 / len(confidence_factors))
        
        return min(overall_confidence, 1.0)
    
    def _update_performance_metrics(self, quality_score: QualityScore):
        """Update performance tracking metrics"""
        self.assessment_history.append({
            "timestamp": time.time(),
            "overall_score": quality_score.overall_score,
            "confidence": quality_score.confidence,
            "assessment_time": quality_score.assessment_time,
            "sample_size": quality_score.sample_size
        })
        
        # Keep only recent history (last 1000 assessments)
        if len(self.assessment_history) > 1000:
            self.assessment_history = self.assessment_history[-1000:]
        
        # Update rolling averages
        recent_assessments = self.assessment_history[-100:]  # Last 100 assessments
        
        if recent_assessments:
            self.performance_metrics.update({
                "avg_assessment_time": np.mean([a["assessment_time"] for a in recent_assessments]),
                "avg_overall_score": np.mean([a["overall_score"] for a in recent_assessments]),
                "avg_confidence": np.mean([a["confidence"] for a in recent_assessments]),
                "assessments_per_hour": 3600.0 / np.mean([a["assessment_time"] for a in recent_assessments]),
                "total_assessments": len(self.assessment_history)
            })
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            "performance_metrics": self.performance_metrics.copy(),
            "cache_stats": {
                "assessment_cache_size": len(self.assessment_cache),
                "embedding_cache_size": len(self.embedding_cache)
            },
            "model_info": {
                "embedding_model_available": self.embedding_model is not None,
                "device": self.device,
                "sklearn_available": SKLEARN_AVAILABLE,
                "nltk_available": NLTK_AVAILABLE
            },
            "domain_profiles": list(self.domain_profiles.keys()),
            "assessment_history_size": len(self.assessment_history)
        }
        
        return stats
    
    async def cleanup(self):
        """Cleanup resources and save state"""
        self.logger.info("Cleaning up semantic quality assessor")
        
        # Clear caches to free memory
        self.assessment_cache.clear()
        self.embedding_cache.clear()
        
        # Save performance metrics
        try:
            metrics_file = Path("semantic_quality_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump({
                    "performance_metrics": self.performance_metrics,
                    "assessment_history": self.assessment_history[-100:],  # Save recent history
                    "last_updated": time.time()
                }, f, indent=2)
            
            self.logger.info(f"Performance metrics saved to {metrics_file}")
        except Exception as e:
            self.logger.error(f"Failed to save performance metrics: {e}")
        
        self.logger.info("Semantic quality assessor cleanup complete")


# Factory function for easy instantiation
def create_quality_assessor(
    config_path: Optional[Path] = None,
    device: str = "auto"
) -> SemanticQualityAssessor:
    """Create and initialize semantic quality assessor"""
    return SemanticQualityAssessor(config_path=config_path, device=device)


# Example usage and testing
if __name__ == "__main__":
    async def test_quality_assessor():
        """Test the semantic quality assessor"""
        logging.basicConfig(level=logging.INFO)
        
        assessor = create_quality_assessor()
        await assessor._initialize_models()  # Ensure models are loaded
        
        # Test content
        test_content = """
        # LTE Handover Parameter Configuration
        
        This document describes the configuration of handover parameters in LTE networks.
        The handover procedure is critical for maintaining user connectivity during mobility.
        
        ## Key Parameters
        
        - **hysteresisA3**: Hysteresis value for A3 event (typically 2-6 dB)
        - **timeToTriggerA3**: Time to trigger for A3 measurement (40-5120 ms)
        - **rsrpThresholdA5**: RSRP threshold for A5 event (-140 to -44 dBm)
        
        ## Performance Counters
        
        - handoverSuccessRate: Percentage of successful handovers
        - handoverFailureRate: Percentage of failed handovers
        - avgHandoverTime: Average time for handover completion
        
        The optimal configuration depends on network topology and traffic patterns.
        Regular monitoring of KPIs is essential for maintaining service quality.
        """
        
        # Test single document assessment
        quality_score = await assessor.assess_document_quality(test_content)
        
        print("=== Quality Assessment Results ===")
        print(f"Overall Score: {quality_score.overall_score:.3f}")
        print(f"Confidence: {quality_score.confidence:.3f}")
        print(f"Assessment Time: {quality_score.assessment_time:.2f}s")
        
        print("\nDimension Scores:")
        for dimension, score in quality_score.dimension_scores.items():
            print(f"  {dimension.value}: {score:.3f}")
        
        # Test cross-document analysis
        test_documents = [
            (test_content, {"title": "LTE Handover Configuration"}),
            (test_content.replace("LTE", "5G NR").replace("handover", "mobility"), 
             {"title": "5G NR Mobility Configuration"})
        ]
        
        cross_analysis = await assessor.assess_cross_document_consistency(test_documents)
        
        print("\n=== Cross-Document Analysis ===")
        print(f"Consistency Score: {cross_analysis.consistency_score:.3f}")
        print(f"Content Diversity: {cross_analysis.content_diversity:.3f}")
        print(f"Terminology Consistency: {cross_analysis.terminology_consistency:.3f}")
        
        if cross_analysis.duplicate_detection:
            print("Potential Duplicates:")
            for i, j, similarity in cross_analysis.duplicate_detection:
                print(f"  Documents {i} and {j}: {similarity:.3f} similarity")
        
        # Performance statistics
        stats = assessor.get_performance_stats()
        print("\n=== Performance Statistics ===")
        for category, data in stats.items():
            print(f"{category}: {data}")
        
        await assessor.cleanup()
    
    asyncio.run(test_quality_assessor())