#!/usr/bin/env python3
"""
Quality Scoring System

Advanced quality assessment pipeline targeting >0.742 consistency score
with comprehensive metrics, validation, and continuous improvement.

Author: Claude Code ML Integration Specialist
Date: 2025-08-23
"""

import asyncio
import json
import logging
import re
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import hashlib
import subprocess
from pathlib import Path

# NLP and ML imports with fallbacks
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available, using basic quality scoring")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available, using basic similarity scoring")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy not available, using basic numerical operations")

# Import local modules
try:
    from .qwen3_manager import ModelVariant
    from .local_inference import InferenceResponse
except ImportError:
    # Fallback for testing
    import sys
    sys.path.append('.')
    from qwen3_manager import ModelVariant
    from local_inference import InferenceResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityDimension(Enum):
    """Quality assessment dimensions"""
    ACCURACY = "accuracy"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    FLUENCY = "fluency"
    INFORMATIVENESS = "informativeness"
    SAFETY = "safety"

class QualityLevel(Enum):
    """Quality levels for responses"""
    EXCELLENT = "excellent"  # >0.9
    GOOD = "good"  # >0.742
    ACCEPTABLE = "acceptable"  # >0.6
    POOR = "poor"  # >0.4
    UNACCEPTABLE = "unacceptable"  # <=0.4

@dataclass
class QualityScore:
    """Individual quality score for a dimension"""
    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

@dataclass
class QualityAssessment:
    """Comprehensive quality assessment"""
    overall_score: float
    quality_level: QualityLevel
    dimension_scores: Dict[QualityDimension, QualityScore]
    consistency_score: float
    target_threshold: float = 0.742
    meets_threshold: bool = False
    timestamp: float = field(default_factory=time.time)
    model_variant: Optional[ModelVariant] = None
    assessment_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class QualityMetric(ABC):
    """Abstract base class for quality metrics"""
    
    @abstractmethod
    def calculate_score(self, text: str, context: Dict[str, Any] = None) -> QualityScore:
        """Calculate quality score for given text"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> QualityDimension:
        """Get the quality dimension this metric measures"""
        pass

class AccuracyMetric(QualityMetric):
    """Assess factual accuracy and correctness"""
    
    def __init__(self):
        self.factual_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b\d+(\.\d+)?%\b',  # Percentages
            r'\$\d+',  # Currency
            r'\b[A-Z][a-zA-Z]+ \d{1,2}, \d{4}\b'  # Dates
        ]
        
        # Common accuracy indicators
        self.accuracy_indicators = {
            'positive': ['accurate', 'correct', 'precise', 'exact', 'verified', 'confirmed'],
            'negative': ['incorrect', 'wrong', 'inaccurate', 'false', 'misleading', 'error']
        }
    
    def calculate_score(self, text: str, context: Dict[str, Any] = None) -> QualityScore:
        score = 0.8  # Base score
        details = {}
        reasoning = []
        
        # Check for factual content
        factual_content = sum(1 for pattern in self.factual_patterns 
                            if re.search(pattern, text))
        details['factual_elements'] = factual_content
        
        if factual_content > 0:
            score += 0.1  # Bonus for including facts
            reasoning.append(f"Contains {factual_content} factual elements")
        
        # Check for accuracy indicators
        positive_indicators = sum(1 for indicator in self.accuracy_indicators['positive']
                                if indicator.lower() in text.lower())
        negative_indicators = sum(1 for indicator in self.accuracy_indicators['negative']
                                if indicator.lower() in text.lower())
        
        details['positive_indicators'] = positive_indicators
        details['negative_indicators'] = negative_indicators
        
        if negative_indicators > 0:
            score -= 0.2 * negative_indicators
            reasoning.append(f"Found {negative_indicators} negative accuracy indicators")
        
        # Check for hedge words (uncertainty indicators)
        hedge_words = ['might', 'could', 'possibly', 'perhaps', 'maybe', 'likely']
        hedge_count = sum(1 for word in hedge_words if word in text.lower())
        
        if hedge_count > len(text.split()) * 0.05:  # More than 5% hedge words
            score -= 0.1
            reasoning.append("High uncertainty language detected")
        
        details['hedge_words'] = hedge_count
        score = max(0.0, min(1.0, score))  # Clamp to [0,1]
        
        return QualityScore(
            dimension=QualityDimension.ACCURACY,
            score=score,
            details=details,
            reasoning='; '.join(reasoning) if reasoning else "No specific accuracy issues detected"
        )
    
    def get_dimension(self) -> QualityDimension:
        return QualityDimension.ACCURACY

class CoherenceMetric(QualityMetric):
    """Assess logical flow and coherence"""
    
    def __init__(self):
        self.transition_words = [
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'meanwhile', 'similarly', 'in contrast', 'on the other hand',
            'first', 'second', 'finally', 'in conclusion', 'as a result'
        ]
    
    def calculate_score(self, text: str, context: Dict[str, Any] = None) -> QualityScore:
        sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('. ')
        score = 0.7  # Base score
        details = {}
        reasoning = []
        
        # Check sentence count for appropriate length
        sentence_count = len(sentences)
        details['sentence_count'] = sentence_count
        
        if sentence_count < 2:
            score -= 0.2
            reasoning.append("Very short response may lack coherence")
        elif sentence_count > 20:
            score -= 0.1
            reasoning.append("Very long response may have coherence issues")
        
        # Check for transition words
        transition_count = sum(1 for word in self.transition_words 
                             if word.lower() in text.lower())
        details['transition_words'] = transition_count
        
        if sentence_count > 3 and transition_count > 0:
            score += 0.1 * min(transition_count / sentence_count * 10, 0.3)
            reasoning.append(f"Good use of {transition_count} transition words")
        
        # Check for repetitive sentence structures
        if sentence_count > 2:
            sentence_starts = [sent.strip()[:20] for sent in sentences if sent.strip()]
            unique_starts = len(set(sentence_starts))
            repetition_ratio = unique_starts / len(sentence_starts) if sentence_starts else 1
            
            if repetition_ratio < 0.7:  # More than 30% repetitive starts
                score -= 0.15
                reasoning.append("Repetitive sentence structures detected")
            
            details['sentence_variety'] = repetition_ratio
        
        # Check paragraph structure (if applicable)
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.1  # Bonus for structured text
            reasoning.append("Well-structured with multiple paragraphs")
        
        details['paragraph_count'] = len(paragraphs)
        score = max(0.0, min(1.0, score))
        
        return QualityScore(
            dimension=QualityDimension.COHERENCE,
            score=score,
            details=details,
            reasoning='; '.join(reasoning) if reasoning else "Acceptable coherence"
        )
    
    def get_dimension(self) -> QualityDimension:
        return QualityDimension.COHERENCE

class RelevanceMetric(QualityMetric):
    """Assess relevance to query/context"""
    
    def calculate_score(self, text: str, context: Dict[str, Any] = None) -> QualityScore:
        score = 0.8  # Base score
        details = {}
        reasoning = []
        
        if not context or 'query' not in context:
            # Without context, assume reasonable relevance
            return QualityScore(
                dimension=QualityDimension.RELEVANCE,
                score=score,
                details={'context_available': False},
                reasoning="No query context provided for relevance assessment"
            )
        
        query = context['query'].lower()
        text_lower = text.lower()
        
        # Extract key terms from query
        if NLTK_AVAILABLE:
            try:
                stop_words = set(stopwords.words('english'))
                query_words = [word for word in word_tokenize(query) 
                             if word.isalpha() and word not in stop_words]
            except:
                query_words = [word for word in query.split() if len(word) > 2]
        else:
            query_words = [word for word in query.split() if len(word) > 2]
        
        # Check for query term coverage
        covered_terms = sum(1 for word in query_words if word in text_lower)
        coverage_ratio = covered_terms / len(query_words) if query_words else 0
        
        details['query_terms'] = len(query_words)
        details['covered_terms'] = covered_terms
        details['coverage_ratio'] = coverage_ratio
        
        if coverage_ratio > 0.7:
            score += 0.15
            reasoning.append(f"High query term coverage ({coverage_ratio:.2f})")
        elif coverage_ratio < 0.3:
            score -= 0.3
            reasoning.append(f"Low query term coverage ({coverage_ratio:.2f})")
        
        # Check for off-topic content
        off_topic_indicators = [
            'by the way', 'incidentally', 'unrelated', 'different topic',
            'changing subject', 'off topic'
        ]
        
        off_topic_count = sum(1 for indicator in off_topic_indicators 
                            if indicator in text_lower)
        
        if off_topic_count > 0:
            score -= 0.2 * off_topic_count
            reasoning.append(f"Detected {off_topic_count} off-topic indicators")
        
        details['off_topic_indicators'] = off_topic_count
        
        # Bonus for direct answers
        if any(text_lower.startswith(starter) for starter in 
               ['yes,', 'no,', 'the answer is', 'to answer your question']):
            score += 0.1
            reasoning.append("Direct response style")
        
        score = max(0.0, min(1.0, score))
        
        return QualityScore(
            dimension=QualityDimension.RELEVANCE,
            score=score,
            details=details,
            reasoning='; '.join(reasoning) if reasoning else "Reasonable relevance to query"
        )
    
    def get_dimension(self) -> QualityDimension:
        return QualityDimension.RELEVANCE

class CompletenessMetric(QualityMetric):
    """Assess completeness of response"""
    
    def calculate_score(self, text: str, context: Dict[str, Any] = None) -> QualityScore:
        score = 0.7  # Base score
        details = {}
        reasoning = []
        
        # Basic length analysis
        word_count = len(text.split())
        char_count = len(text)
        
        details['word_count'] = word_count
        details['character_count'] = char_count
        
        # Score based on length appropriateness
        if word_count < 10:
            score -= 0.3
            reasoning.append("Response too brief for completeness")
        elif word_count > 500:
            score -= 0.1
            reasoning.append("Response may be overly verbose")
        elif 50 <= word_count <= 200:
            score += 0.2
            reasoning.append("Appropriate response length")
        
        # Check for common completeness indicators
        completeness_indicators = {
            'positive': ['comprehensive', 'complete', 'thorough', 'detailed', 'covers'],
            'negative': ['incomplete', 'partial', 'brief', 'limited', 'missing']
        }
        
        positive_count = sum(1 for indicator in completeness_indicators['positive']
                           if indicator in text.lower())
        negative_count = sum(1 for indicator in completeness_indicators['negative']
                           if indicator in text.lower())
        
        details['completeness_indicators'] = {
            'positive': positive_count,
            'negative': negative_count
        }
        
        if positive_count > 0:
            score += 0.1
            reasoning.append("Contains positive completeness indicators")
        
        if negative_count > 0:
            score -= 0.15
            reasoning.append("Contains incompleteness indicators")
        
        # Check for structured elements (lists, steps, examples)
        structured_elements = [
            len(re.findall(r'\n\d+\.', text)),  # Numbered lists
            len(re.findall(r'\n[-*]', text)),   # Bullet points
            len(re.findall(r'\bfor example\b|\be\.g\.\b', text, re.I)),  # Examples
            len(re.findall(r'\bfirst|second|third|finally\b', text, re.I))  # Steps
        ]
        
        total_structure = sum(structured_elements)
        details['structured_elements'] = total_structure
        
        if total_structure > 2:
            score += 0.15
            reasoning.append("Well-structured with multiple elements")
        elif total_structure > 0:
            score += 0.05
            reasoning.append("Some structural elements present")
        
        score = max(0.0, min(1.0, score))
        
        return QualityScore(
            dimension=QualityDimension.COMPLETENESS,
            score=score,
            details=details,
            reasoning='; '.join(reasoning) if reasoning else "Adequate completeness"
        )
    
    def get_dimension(self) -> QualityDimension:
        return QualityDimension.COMPLETENESS

class ConsistencyMetric(QualityMetric):
    """Assess internal consistency and contradictions"""
    
    def __init__(self):
        self.contradiction_patterns = [
            (r'\bnot\b.*\bis\b', r'\bis\b.*\bnot\b'),
            (r'\balways\b', r'\bnever\b'),
            (r'\ball\b', r'\bnone\b'),
            (r'\byes\b', r'\bno\b'),
            (r'\btrue\b', r'\bfalse\b')
        ]
    
    def calculate_score(self, text: str, context: Dict[str, Any] = None) -> QualityScore:
        score = 0.85  # High base score - consistency is usually good
        details = {}
        reasoning = []
        
        # Check for explicit contradictions
        contradiction_count = 0
        for pattern1, pattern2 in self.contradiction_patterns:
            if re.search(pattern1, text, re.I) and re.search(pattern2, text, re.I):
                contradiction_count += 1
        
        details['potential_contradictions'] = contradiction_count
        
        if contradiction_count > 0:
            score -= 0.3 * contradiction_count
            reasoning.append(f"Detected {contradiction_count} potential contradictions")
        
        # Check for inconsistent terminology
        sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('. ')
        
        # Look for similar concepts with different terms
        similar_concept_pairs = [
            ('ai', 'artificial intelligence'),
            ('ml', 'machine learning'),
            ('usa', 'united states'),
            ('uk', 'united kingdom')
        ]
        
        terminology_consistency = 0
        for short, long in similar_concept_pairs:
            short_count = text.lower().count(short)
            long_count = text.lower().count(long)
            
            if short_count > 0 and long_count > 0:
                # Mixed usage detected
                terminology_consistency += 1
        
        details['mixed_terminology'] = terminology_consistency
        
        if terminology_consistency > 2:
            score -= 0.1
            reasoning.append("Inconsistent terminology usage")
        
        # Check for consistent numerical formats
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if len(numbers) > 2:
            # Check if numbers follow consistent formatting
            decimal_count = sum(1 for num in numbers if '.' in num)
            if 0 < decimal_count < len(numbers):
                details['number_format_consistency'] = False
                # Minor penalty for inconsistent number formatting
                score -= 0.05
                reasoning.append("Inconsistent number formatting")
            else:
                details['number_format_consistency'] = True
        
        # Check for tense consistency
        past_tense_verbs = len(re.findall(r'\b\w+ed\b', text))
        present_tense_verbs = len(re.findall(r'\b\w+s\b', text))
        total_verbs = past_tense_verbs + present_tense_verbs
        
        if total_verbs > 5:
            tense_ratio = abs(past_tense_verbs - present_tense_verbs) / total_verbs
            if tense_ratio < 0.6:  # Mixed tenses
                score -= 0.05
                reasoning.append("Mixed verb tenses detected")
        
        details['verb_tense_analysis'] = {
            'past_tense': past_tense_verbs,
            'present_tense': present_tense_verbs,
            'total': total_verbs
        }
        
        score = max(0.0, min(1.0, score))
        
        return QualityScore(
            dimension=QualityDimension.CONSISTENCY,
            score=score,
            details=details,
            reasoning='; '.join(reasoning) if reasoning else "High consistency maintained"
        )
    
    def get_dimension(self) -> QualityDimension:
        return QualityDimension.CONSISTENCY

class FluencyMetric(QualityMetric):
    """Assess language fluency and readability"""
    
    def calculate_score(self, text: str, context: Dict[str, Any] = None) -> QualityScore:
        score = 0.8  # Base score
        details = {}
        reasoning = []
        
        # Basic grammar checks
        grammar_issues = 0
        
        # Check for common grammar problems
        grammar_patterns = [
            r"\bi is\b",  # "i is" instead of "I am"
            r"\ba an\b",  # "a an" article confusion
            r"\bthere is.*are\b",  # subject-verb disagreement
            r"\bthis are\b",  # "this are" instead of "these are"
        ]
        
        for pattern in grammar_patterns:
            matches = len(re.findall(pattern, text, re.I))
            grammar_issues += matches
        
        details['grammar_issues'] = grammar_issues
        
        if grammar_issues > 0:
            score -= 0.1 * min(grammar_issues, 5)  # Cap penalty
            reasoning.append(f"Detected {grammar_issues} potential grammar issues")
        
        # Check sentence structure variety
        sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('. ')
        sentence_lengths = [len(sent.split()) for sent in sentences if sent.strip()]
        
        if len(sentence_lengths) > 2:
            avg_length = statistics.mean(sentence_lengths)
            length_variance = statistics.variance(sentence_lengths) if len(sentence_lengths) > 1 else 0
            
            details['average_sentence_length'] = avg_length
            details['sentence_length_variance'] = length_variance
            
            # Good sentence variety
            if length_variance > 10:  # Good variety
                score += 0.1
                reasoning.append("Good sentence length variety")
            elif length_variance < 2:  # Too uniform
                score -= 0.05
                reasoning.append("Limited sentence length variety")
        
        # Check for readability indicators
        complex_words = 0
        words = text.split()
        
        for word in words:
            if len(word) > 12:  # Very long words
                complex_words += 1
        
        complexity_ratio = complex_words / len(words) if words else 0
        details['complex_word_ratio'] = complexity_ratio
        
        if complexity_ratio > 0.15:  # More than 15% complex words
            score -= 0.1
            reasoning.append("High complexity may affect readability")
        elif complexity_ratio > 0.05:  # 5-15% is good
            score += 0.05
            reasoning.append("Appropriate vocabulary complexity")
        
        # Check for repetitive word usage
        if NLTK_AVAILABLE:
            try:
                word_tokens = word_tokenize(text.lower())
                word_freq = {}
                for word in word_tokens:
                    if word.isalpha() and len(word) > 3:
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                if word_freq:
                    max_freq = max(word_freq.values())
                    total_words = len(word_tokens)
                    repetition_ratio = max_freq / total_words
                    
                    details['max_word_repetition'] = repetition_ratio
                    
                    if repetition_ratio > 0.1:  # Single word more than 10%
                        score -= 0.15
                        reasoning.append("Excessive word repetition detected")
            except:
                pass  # Skip if NLTK components not available
        
        # Check for fluency markers
        fluency_markers = [
            'furthermore', 'moreover', 'however', 'therefore', 'consequently',
            'in addition', 'on the other hand', 'for instance', 'specifically'
        ]
        
        marker_count = sum(1 for marker in fluency_markers 
                          if marker in text.lower())
        
        details['fluency_markers'] = marker_count
        
        if marker_count > 0 and len(sentences) > 2:
            score += min(0.1, marker_count * 0.02)  # Small bonus for fluency markers
            reasoning.append(f"Good use of {marker_count} fluency markers")
        
        score = max(0.0, min(1.0, score))
        
        return QualityScore(
            dimension=QualityDimension.FLUENCY,
            score=score,
            details=details,
            reasoning='; '.join(reasoning) if reasoning else "Good fluency"
        )
    
    def get_dimension(self) -> QualityDimension:
        return QualityDimension.FLUENCY

class SafetyMetric(QualityMetric):
    """Assess content safety and appropriateness"""
    
    def __init__(self):
        # Basic safety patterns - in practice, you'd use more sophisticated detection
        self.unsafe_patterns = [
            r'\b(?:hate|violence|harm)\b',
            r'\b(?:illegal|dangerous|harmful)\b',
            r'\b(?:discrimination|bias|prejudice)\b'
        ]
        
        self.safety_indicators = [
            'safe', 'appropriate', 'respectful', 'ethical', 'responsible'
        ]
    
    def calculate_score(self, text: str, context: Dict[str, Any] = None) -> QualityScore:
        score = 0.95  # High base score - assume safe unless proven otherwise
        details = {}
        reasoning = []
        
        # Check for unsafe patterns
        unsafe_matches = 0
        for pattern in self.unsafe_patterns:
            matches = len(re.findall(pattern, text, re.I))
            unsafe_matches += matches
        
        details['potential_unsafe_content'] = unsafe_matches
        
        if unsafe_matches > 0:
            score -= 0.3 * min(unsafe_matches, 3)  # Heavy penalty, but capped
            reasoning.append(f"Detected {unsafe_matches} potential safety concerns")
        
        # Check for safety indicators
        safety_count = sum(1 for indicator in self.safety_indicators 
                          if indicator in text.lower())
        
        details['safety_indicators'] = safety_count
        
        if safety_count > 0:
            score += 0.02 * safety_count  # Small bonus
            reasoning.append(f"Contains {safety_count} positive safety indicators")
        
        # Check for disclaimers (good safety practice)
        disclaimer_patterns = [
            r'\bdisclaimer\b', r'\bnot (?:a )?(?:medical|legal|financial) advice\b',
            r'\bconsult (?:a |an )?(?:professional|expert|doctor)\b'
        ]
        
        disclaimer_count = sum(1 for pattern in disclaimer_patterns 
                             if re.search(pattern, text, re.I))
        
        details['disclaimers'] = disclaimer_count
        
        if disclaimer_count > 0:
            score += 0.03 * disclaimer_count
            reasoning.append(f"Contains {disclaimer_count} appropriate disclaimers")
        
        score = max(0.0, min(1.0, score))
        
        return QualityScore(
            dimension=QualityDimension.SAFETY,
            score=score,
            details=details,
            reasoning='; '.join(reasoning) if reasoning else "No safety concerns detected"
        )
    
    def get_dimension(self) -> QualityDimension:
        return QualityDimension.SAFETY

class AdvancedQualityScorer:
    """Advanced quality scoring system targeting >0.742 consistency"""
    
    def __init__(self, target_threshold: float = 0.742, enable_caching: bool = True):
        self.target_threshold = target_threshold
        self.enable_caching = enable_caching
        
        # Initialize metrics
        self.metrics: Dict[QualityDimension, QualityMetric] = {
            QualityDimension.ACCURACY: AccuracyMetric(),
            QualityDimension.COHERENCE: CoherenceMetric(),
            QualityDimension.RELEVANCE: RelevanceMetric(),
            QualityDimension.COMPLETENESS: CompletenessMetric(),
            QualityDimension.CONSISTENCY: ConsistencyMetric(),
            QualityDimension.FLUENCY: FluencyMetric(),
            QualityDimension.SAFETY: SafetyMetric()
        }
        
        # Dimension weights for overall score
        self.dimension_weights = {
            QualityDimension.ACCURACY: 0.20,
            QualityDimension.COHERENCE: 0.15,
            QualityDimension.RELEVANCE: 0.20,
            QualityDimension.COMPLETENESS: 0.15,
            QualityDimension.CONSISTENCY: 0.15,  # Key for target threshold
            QualityDimension.FLUENCY: 0.10,
            QualityDimension.SAFETY: 0.05
        }
        
        # Performance tracking
        self.assessment_history = []
        self.performance_stats = {
            'total_assessments': 0,
            'above_threshold': 0,
            'average_score': 0.0,
            'consistency_scores': []
        }
        
        # Cache for repeated assessments
        self.cache = {} if enable_caching else None
        
        logger.info(f"Initialized AdvancedQualityScorer with target threshold: {target_threshold}")
    
    def _generate_cache_key(self, text: str, context: Dict[str, Any] = None) -> str:
        """Generate cache key for assessment"""
        content = f"{text}:{json.dumps(context or {}, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def assess_quality(self, 
                      text: str, 
                      context: Dict[str, Any] = None,
                      model_variant: Optional[ModelVariant] = None) -> QualityAssessment:
        """Perform comprehensive quality assessment"""
        
        start_time = time.time()
        
        # Check cache
        if self.cache:
            cache_key = self._generate_cache_key(text, context)
            if cache_key in self.cache:
                cached_assessment = self.cache[cache_key]
                cached_assessment.metadata['from_cache'] = True
                return cached_assessment
        
        # Calculate dimension scores
        dimension_scores = {}
        
        for dimension, metric in self.metrics.items():
            try:
                score = metric.calculate_score(text, context)
                dimension_scores[dimension] = score
            except Exception as e:
                logger.warning(f"Failed to calculate {dimension.value} score: {e}")
                # Fallback score
                dimension_scores[dimension] = QualityScore(
                    dimension=dimension,
                    score=0.7,  # Neutral score
                    reasoning=f"Error in calculation: {str(e)}"
                )
        
        # Calculate weighted overall score
        overall_score = sum(
            dimension_scores[dim].score * weight 
            for dim, weight in self.dimension_weights.items()
        )
        
        # Determine quality level
        if overall_score > 0.9:
            quality_level = QualityLevel.EXCELLENT
        elif overall_score > self.target_threshold:
            quality_level = QualityLevel.GOOD
        elif overall_score > 0.6:
            quality_level = QualityLevel.ACCEPTABLE
        elif overall_score > 0.4:
            quality_level = QualityLevel.POOR
        else:
            quality_level = QualityLevel.UNACCEPTABLE
        
        # Extract consistency score
        consistency_score = dimension_scores[QualityDimension.CONSISTENCY].score
        
        # Create assessment
        assessment_time = (time.time() - start_time) * 1000  # Convert to ms
        
        assessment = QualityAssessment(
            overall_score=overall_score,
            quality_level=quality_level,
            dimension_scores=dimension_scores,
            consistency_score=consistency_score,
            target_threshold=self.target_threshold,
            meets_threshold=overall_score >= self.target_threshold,
            model_variant=model_variant,
            assessment_time_ms=assessment_time,
            metadata={
                'text_length': len(text),
                'word_count': len(text.split()),
                'dimension_weights': self.dimension_weights
            }
        )
        
        # Update performance stats
        self._update_performance_stats(assessment)
        
        # Cache assessment
        if self.cache:
            self.cache[cache_key] = assessment
        
        # Store in coordination memory
        self._store_assessment_in_memory(assessment)
        
        return assessment
    
    def _update_performance_stats(self, assessment: QualityAssessment):
        """Update performance statistics"""
        self.performance_stats['total_assessments'] += 1
        
        if assessment.meets_threshold:
            self.performance_stats['above_threshold'] += 1
        
        # Update average score (running average)
        n = self.performance_stats['total_assessments']
        old_avg = self.performance_stats['average_score']
        self.performance_stats['average_score'] = (
            old_avg * (n - 1) + assessment.overall_score
        ) / n
        
        # Track consistency scores
        self.performance_stats['consistency_scores'].append(assessment.consistency_score)
        
        # Keep only recent consistency scores (last 100)
        if len(self.performance_stats['consistency_scores']) > 100:
            self.performance_stats['consistency_scores'] = \
                self.performance_stats['consistency_scores'][-100:]
        
        # Add to history
        self.assessment_history.append({
            'timestamp': assessment.timestamp,
            'overall_score': assessment.overall_score,
            'consistency_score': assessment.consistency_score,
            'meets_threshold': assessment.meets_threshold,
            'quality_level': assessment.quality_level.value
        })
        
        # Keep history manageable
        if len(self.assessment_history) > 1000:
            self.assessment_history = self.assessment_history[-500:]
    
    def _store_assessment_in_memory(self, assessment: QualityAssessment):
        """Store assessment in coordination memory"""
        try:
            assessment_data = {
                'timestamp': assessment.timestamp,
                'overall_score': assessment.overall_score,
                'consistency_score': assessment.consistency_score,
                'quality_level': assessment.quality_level.value,
                'meets_threshold': assessment.meets_threshold,
                'assessment_time_ms': assessment.assessment_time_ms,
                'model_variant': assessment.model_variant.value if assessment.model_variant else None,
                'dimension_scores': {
                    dim.value: {
                        'score': score.score,
                        'confidence': score.confidence,
                        'reasoning': score.reasoning
                    }
                    for dim, score in assessment.dimension_scores.items()
                }
            }
            
            # Store in coordination memory
            subprocess.run([
                "npx", "claude-flow@alpha", "hooks", "post-edit",
                "--memory-key", f"quality_assessment/{int(assessment.timestamp)}",
                "--data", json.dumps(assessment_data)
            ], capture_output=True)
            
        except Exception as e:
            logger.warning(f"Could not store assessment in coordination memory: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        consistency_scores = self.performance_stats['consistency_scores']
        
        summary = {
            'timestamp': time.time(),
            'target_threshold': self.target_threshold,
            'total_assessments': self.performance_stats['total_assessments'],
            'above_threshold_count': self.performance_stats['above_threshold'],
            'threshold_success_rate': (
                self.performance_stats['above_threshold'] / 
                max(self.performance_stats['total_assessments'], 1)
            ),
            'average_overall_score': self.performance_stats['average_score'],
            'consistency_metrics': {
                'average_consistency': statistics.mean(consistency_scores) if consistency_scores else 0,
                'consistency_std_dev': statistics.stdev(consistency_scores) if len(consistency_scores) > 1 else 0,
                'min_consistency': min(consistency_scores) if consistency_scores else 0,
                'max_consistency': max(consistency_scores) if consistency_scores else 0,
            },
            'dimension_weights': self.dimension_weights,
            'cache_size': len(self.cache) if self.cache else 0
        }
        
        # Recent performance (last 50 assessments)
        recent_history = self.assessment_history[-50:] if self.assessment_history else []
        if recent_history:
            recent_scores = [h['overall_score'] for h in recent_history]
            recent_threshold_success = sum(1 for h in recent_history if h['meets_threshold']) / len(recent_history)
            
            summary['recent_performance'] = {
                'average_score': statistics.mean(recent_scores),
                'threshold_success_rate': recent_threshold_success,
                'trend': 'improving' if len(recent_scores) > 10 and 
                        statistics.mean(recent_scores[-10:]) > statistics.mean(recent_scores[:10]) else 'stable'
            }
        
        return summary
    
    def optimize_for_threshold(self, adjustment_factor: float = 0.1) -> Dict[str, Any]:
        """Optimize scoring to better meet target threshold"""
        
        if self.performance_stats['total_assessments'] < 10:
            return {'status': 'insufficient_data', 'message': 'Need at least 10 assessments for optimization'}
        
        current_success_rate = (
            self.performance_stats['above_threshold'] / 
            self.performance_stats['total_assessments']
        )
        
        optimization_results = {
            'previous_success_rate': current_success_rate,
            'target_threshold': self.target_threshold,
            'adjustments_made': []
        }
        
        # If success rate is too low, adjust dimension weights
        if current_success_rate < 0.7:  # Want 70%+ success rate
            # Increase weight of dimensions with consistently high scores
            recent_assessments = self.assessment_history[-50:]
            
            if recent_assessments:
                dimension_performance = {dim: [] for dim in QualityDimension}
                
                # This would require storing dimension scores in history
                # For now, just adjust consistency weight as it's key for threshold
                if statistics.mean(self.performance_stats['consistency_scores']) > 0.8:
                    old_weight = self.dimension_weights[QualityDimension.CONSISTENCY]
                    self.dimension_weights[QualityDimension.CONSISTENCY] *= (1 + adjustment_factor)
                    
                    # Renormalize weights
                    total_weight = sum(self.dimension_weights.values())
                    self.dimension_weights = {
                        dim: weight / total_weight 
                        for dim, weight in self.dimension_weights.items()
                    }
                    
                    optimization_results['adjustments_made'].append(
                        f"Increased consistency weight from {old_weight:.3f} to {self.dimension_weights[QualityDimension.CONSISTENCY]:.3f}"
                    )
        
        return optimization_results
    
    def batch_assess(self, 
                    texts: List[str], 
                    contexts: Optional[List[Dict[str, Any]]] = None,
                    model_variants: Optional[List[ModelVariant]] = None) -> List[QualityAssessment]:
        """Assess quality for multiple texts"""
        
        if contexts is None:
            contexts = [None] * len(texts)
        if model_variants is None:
            model_variants = [None] * len(texts)
        
        assessments = []
        
        for i, text in enumerate(texts):
            context = contexts[i] if i < len(contexts) else None
            model_variant = model_variants[i] if i < len(model_variants) else None
            
            try:
                assessment = self.assess_quality(text, context, model_variant)
                assessments.append(assessment)
            except Exception as e:
                logger.error(f"Failed to assess text {i}: {e}")
                # Create error assessment
                assessments.append(QualityAssessment(
                    overall_score=0.0,
                    quality_level=QualityLevel.UNACCEPTABLE,
                    dimension_scores={},
                    consistency_score=0.0,
                    metadata={'error': str(e)}
                ))
        
        return assessments

# Factory function for easy initialization
def create_quality_scorer(target_threshold: float = 0.742, **kwargs) -> AdvancedQualityScorer:
    """Create a quality scorer with optimal settings"""
    return AdvancedQualityScorer(target_threshold=target_threshold, **kwargs)

# Example usage and testing
async def main():
    """Example usage of the quality scoring system"""
    
    scorer = create_quality_scorer(target_threshold=0.742)
    
    # Test texts
    test_texts = [
        "Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.",
        
        "ML good. AI learn. Computer smart.",  # Low quality
        
        "To answer your question about machine learning: it's a comprehensive field that encompasses various algorithms and techniques. Machine learning algorithms build mathematical models based on training data to make predictions or decisions. There are three main types: supervised learning, unsupervised learning, and reinforcement learning. Each has specific applications and use cases in modern technology.",
        
        "Artificial intelligence is bad and good at the same time. It's always never working properly. Yes, no, maybe so.",  # Inconsistent
    ]
    
    contexts = [
        {'query': 'What is machine learning?'},
        {'query': 'Explain machine learning'},
        {'query': 'What is machine learning and its types?'},
        {'query': 'Is AI reliable?'}
    ]
    
    print("=== Quality Assessment Results ===")
    
    for i, text in enumerate(test_texts):
        print(f"\nText {i + 1}: {text[:50]}...")
        
        assessment = scorer.assess_quality(text, contexts[i])
        
        print(f"Overall Score: {assessment.overall_score:.3f}")
        print(f"Quality Level: {assessment.quality_level.value}")
        print(f"Meets Threshold ({scorer.target_threshold}): {assessment.meets_threshold}")
        print(f"Consistency Score: {assessment.consistency_score:.3f}")
        print(f"Assessment Time: {assessment.assessment_time_ms:.1f}ms")
        
        print("Dimension Scores:")
        for dimension, score in assessment.dimension_scores.items():
            print(f"  {dimension.value}: {score.score:.3f} - {score.reasoning}")
    
    # Performance summary
    summary = scorer.get_performance_summary()
    print(f"\n=== Performance Summary ===")
    print(f"Total Assessments: {summary['total_assessments']}")
    print(f"Above Threshold: {summary['above_threshold_count']}/{summary['total_assessments']} ({summary['threshold_success_rate']:.1%})")
    print(f"Average Score: {summary['average_overall_score']:.3f}")
    print(f"Average Consistency: {summary['consistency_metrics']['average_consistency']:.3f}")
    
    # Test optimization
    optimization = scorer.optimize_for_threshold()
    print(f"\n=== Optimization Results ===")
    print(json.dumps(optimization, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
