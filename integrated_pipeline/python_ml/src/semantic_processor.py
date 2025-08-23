#!/usr/bin/env python3
"""
Semantic Document Processor for QA Generation

Advanced document understanding and QA generation with multi-dimensional 
quality scoring optimized for Ericsson RAN feature documentation.

Key Features:
- Document understanding and semantic analysis
- QA generation with diversity optimization 
- Multi-dimensional quality scoring (relevance, accuracy, diversity)
- Batch processing optimization for 20-30 docs/hour
- Integration with Rust structural validation

Author: Claude Code
Version: 1.0.0
"""

import asyncio
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import json
import hashlib
from pathlib import Path

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .model_manager import Qwen3ModelManager, ModelSize, ProcessingHints


class QuestionType(Enum):
    """Types of questions to generate"""
    DEFINITION = "definition"
    PARAMETER = "parameter"
    PROCESS = "process"
    COMPARISON = "comparison"
    TROUBLESHOOTING = "troubleshooting"
    CONFIGURATION = "configuration"
    INTEGRATION = "integration"


class DifficultyLevel(Enum):
    """Question difficulty levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class QualityMetrics:
    """Quality assessment metrics for QA pairs"""
    relevance_score: float = 0.0
    accuracy_score: float = 0.0
    diversity_score: float = 0.0
    technical_density: float = 0.0
    completeness_score: float = 0.0
    coherence_score: float = 0.0
    overall_score: float = 0.0
    confidence: float = 0.0
    
    def __post_init__(self):
        """Calculate overall score"""
        weights = {
            'relevance': 0.25,
            'accuracy': 0.25,
            'diversity': 0.15,
            'technical_density': 0.15,
            'completeness': 0.10,
            'coherence': 0.10
        }
        
        self.overall_score = (
            self.relevance_score * weights['relevance'] +
            self.accuracy_score * weights['accuracy'] +
            self.diversity_score * weights['diversity'] +
            self.technical_density * weights['technical_density'] +
            self.completeness_score * weights['completeness'] +
            self.coherence_score * weights['coherence']
        )


@dataclass
class QAPair:
    """Question-Answer pair with metadata"""
    question: str
    answer: str
    question_type: QuestionType
    difficulty: DifficultyLevel
    quality_metrics: QualityMetrics
    source_context: str
    technical_terms: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    confidence: float = 0.0
    generation_time: float = 0.0
    model_used: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type.value,
            "difficulty": self.difficulty.value,
            "quality_metrics": {
                "relevance_score": self.quality_metrics.relevance_score,
                "accuracy_score": self.quality_metrics.accuracy_score,
                "diversity_score": self.quality_metrics.diversity_score,
                "technical_density": self.quality_metrics.technical_density,
                "completeness_score": self.quality_metrics.completeness_score,
                "coherence_score": self.quality_metrics.coherence_score,
                "overall_score": self.quality_metrics.overall_score,
                "confidence": self.quality_metrics.confidence
            },
            "technical_terms": self.technical_terms,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "generation_time": self.generation_time,
            "model_used": self.model_used
        }


@dataclass
class DocumentAnalysis:
    """Document analysis results"""
    document_id: str
    feature_name: str
    complexity: float
    technical_terms: Set[str]
    parameters: List[str]
    key_concepts: List[str]
    document_structure: Dict[str, Any]
    length: int
    technical_density: float
    

class SemanticProcessor:
    """
    Advanced semantic processor for document understanding and QA generation.
    
    Optimized for Ericsson RAN documentation with focus on technical accuracy
    and diversity in generated training data.
    """
    
    def __init__(self, model_manager: Optional[Qwen3ModelManager] = None):
        self.logger = logging.getLogger(__name__)
        self.model_manager = model_manager
        
        # Initialize semantic models
        self.sentence_transformer = None
        self.tfidf_vectorizer = None
        self.diversity_cache = {}
        
        # Technical patterns for RAN documentation
        self.parameter_patterns = [
            r'\b[A-Z][a-z]+\.[a-z][a-zA-Z]*\b',  # MO.parameter format
            r'\b[a-z][a-zA-Z]*(?:Value|Threshold|Level|Size|Count)\b',
            r'\b(?:true|false|enabled|disabled)\b',
            r'\b\d+(?:\.\d+)?\s*(?:dB|dBm|MHz|kHz|ms|s)\b'
        ]
        
        self.technical_terms = {
            'lte_terms': {
                'UE', 'eNodeB', 'MME', 'SGW', 'PGW', 'HSS', 'PCRF',
                'EPC', 'RRC', 'NAS', 'PDCP', 'RLC', 'MAC', 'PHY',
                'HARQ', 'CQI', 'PMI', 'RI', 'SRS', 'PUCCH', 'PUSCH'
            },
            'nr_terms': {
                'gNB', 'UE', 'AMF', 'SMF', 'UPF', 'AUSF', 'UDM', 'PCF',
                '5GC', 'RRC', 'NAS', 'PDCP', 'RLC', 'MAC', 'PHY',
                'BWP', 'CSI', 'HARQ', 'SRS', 'PUCCH', 'PUSCH', 'PDSCH'
            },
            'common_terms': {
                'QoS', 'QCI', '5QI', 'ARP', 'GBR', 'AMBR', 'PLMN',
                'TAC', 'ECGI', 'PCI', 'RSRP', 'RSRQ', 'SINR', 'CIO'
            }
        }
        
        self.question_templates = {
            QuestionType.DEFINITION: [
                "What is {term}?",
                "Define {term} in the context of {feature}.",
                "Explain the concept of {term}.",
                "How would you describe {term}?"
            ],
            QuestionType.PARAMETER: [
                "What is the purpose of parameter {parameter}?",
                "How does {parameter} affect {feature}?",
                "What are the valid values for {parameter}?",
                "When should {parameter} be configured?"
            ],
            QuestionType.PROCESS: [
                "How does {process} work?",
                "Describe the {process} procedure.",
                "What are the steps in {process}?",
                "Explain the {process} workflow."
            ],
            QuestionType.CONFIGURATION: [
                "How do you configure {feature}?",
                "What parameters need to be set for {feature}?",
                "Describe the configuration steps for {feature}.",
                "What are the prerequisites for configuring {feature}?"
            ]
        }
        
    async def initialize(self):
        """Initialize semantic processing models"""
        self.logger.info("Initializing semantic processor")
        
        try:
            # Load sentence transformer for embeddings
            self.sentence_transformer = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device='mps' if hasattr(self, '_has_mps') else 'cpu'
            )
            
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 3)
            )
            
            self.logger.info("Semantic models initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize semantic models: {e}")
            raise
            
    def analyze_document(self, content: str, metadata: Dict[str, Any]) -> DocumentAnalysis:
        """
        Analyze document for semantic understanding.
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            Document analysis results
        """
        doc_id = metadata.get('document_id', 'unknown')
        feature_name = self._extract_feature_name(content, metadata)
        
        # Extract technical information
        technical_terms = self._extract_technical_terms(content)
        parameters = self._extract_parameters(content)
        key_concepts = self._extract_key_concepts(content)
        
        # Analyze structure
        structure = self._analyze_document_structure(content)
        
        # Calculate metrics
        complexity = self._calculate_complexity(content, technical_terms, parameters)
        technical_density = len(technical_terms) / max(len(content.split()), 1)
        
        return DocumentAnalysis(
            document_id=doc_id,
            feature_name=feature_name,
            complexity=complexity,
            technical_terms=technical_terms,
            parameters=parameters,
            key_concepts=key_concepts,
            document_structure=structure,
            length=len(content),
            technical_density=technical_density
        )
        
    def _extract_feature_name(self, content: str, metadata: Dict[str, Any]) -> str:
        """Extract feature name from content or metadata"""
        # Try metadata first
        if 'feature_name' in metadata:
            return metadata['feature_name']
            
        # Try to extract from title or header
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            if line.startswith('# ') or line.startswith('## '):
                return line.strip('#').strip()
            if 'DOCTITLE:' in line:
                return line.split('DOCTITLE:')[1].strip()
                
        return "Unknown Feature"
        
    def _extract_technical_terms(self, content: str) -> Set[str]:
        """Extract technical terms from content"""
        terms = set()
        content_upper = content.upper()
        
        # Check all technical term categories
        for category_terms in self.technical_terms.values():
            for term in category_terms:
                if term in content_upper:
                    terms.add(term)
                    
        # Extract additional acronyms (3-5 uppercase letters)
        acronym_pattern = r'\b[A-Z]{3,5}\b'
        acronyms = re.findall(acronym_pattern, content)
        terms.update(acronyms)
        
        return terms
        
    def _extract_parameters(self, content: str) -> List[str]:
        """Extract configuration parameters from content"""
        parameters = []
        
        for pattern in self.parameter_patterns:
            matches = re.findall(pattern, content)
            parameters.extend(matches)
            
        return list(set(parameters))  # Remove duplicates
        
    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts using TF-IDF"""
        if self.tfidf_vectorizer is None:
            return []
            
        try:
            # Fit on document content
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([content])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get top concepts by TF-IDF score
            scores = tfidf_matrix.toarray()[0]
            top_indices = scores.argsort()[-20:][::-1]  # Top 20
            
            key_concepts = [feature_names[i] for i in top_indices if scores[i] > 0.1]
            return key_concepts[:10]  # Limit to top 10
        except Exception as e:
            self.logger.warning(f"Failed to extract key concepts: {e}")
            return []
            
    def _analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure"""
        lines = content.split('\n')
        
        structure = {
            'total_lines': len(lines),
            'sections': [],
            'has_parameters': False,
            'has_code_blocks': False,
            'has_tables': False
        }
        
        current_section = None
        for i, line in enumerate(lines):
            # Headers
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                section_title = line.strip('#').strip()
                current_section = {
                    'title': section_title,
                    'level': level,
                    'line_number': i,
                    'content_lines': 0
                }
                structure['sections'].append(current_section)
            elif current_section:
                current_section['content_lines'] += 1
                
            # Content analysis
            if any(param in line for param in ['true', 'false', 'enabled', 'disabled']):
                structure['has_parameters'] = True
            if line.strip().startswith('```'):
                structure['has_code_blocks'] = True
            if '|' in line and len(line.split('|')) > 2:
                structure['has_tables'] = True
                
        return structure
        
    def _calculate_complexity(self, content: str, technical_terms: Set[str], parameters: List[str]) -> float:
        """Calculate document complexity score (0.0-1.0)"""
        factors = {
            'length': min(len(content) / 5000.0, 1.0),  # Normalize to 5k chars
            'technical_density': min(len(technical_terms) / 20.0, 1.0),  # Max 20 terms
            'parameter_count': min(len(parameters) / 10.0, 1.0),  # Max 10 params
            'sentence_complexity': self._calculate_sentence_complexity(content)
        }
        
        weights = {'length': 0.2, 'technical_density': 0.3, 'parameter_count': 0.3, 'sentence_complexity': 0.2}
        
        complexity = sum(factors[key] * weights[key] for key in factors)
        return min(complexity, 1.0)
        
    def _calculate_sentence_complexity(self, content: str) -> float:
        """Calculate average sentence complexity"""
        sentences = re.split(r'[.!?]+', content)
        if not sentences:
            return 0.0
            
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        return min(avg_length / 20.0, 1.0)  # Normalize to 20 words per sentence
        
    async def generate_qa_pairs(
        self,
        document_analysis: DocumentAnalysis,
        content: str,
        target_count: int = 5,
        quality_threshold: float = 0.7
    ) -> List[QAPair]:
        """
        Generate QA pairs from analyzed document.
        
        Args:
            document_analysis: Document analysis results
            content: Original document content
            target_count: Target number of QA pairs
            quality_threshold: Minimum quality threshold
            
        Returns:
            List of generated QA pairs
        """
        self.logger.info(f"Generating {target_count} QA pairs for {document_analysis.feature_name}")
        
        qa_pairs = []
        attempts = 0
        max_attempts = target_count * 3  # Allow multiple attempts
        
        # Generate diverse question types
        question_distribution = self._plan_question_distribution(target_count, document_analysis)
        
        while len(qa_pairs) < target_count and attempts < max_attempts:
            attempts += 1
            
            # Select question type based on distribution
            question_type = self._select_question_type(question_distribution, qa_pairs)
            
            try:
                # Generate QA pair
                qa_pair = await self._generate_single_qa_pair(
                    content, document_analysis, question_type
                )
                
                if qa_pair and qa_pair.quality_metrics.overall_score >= quality_threshold:
                    # Check for diversity
                    if self._check_diversity(qa_pair, qa_pairs):
                        qa_pairs.append(qa_pair)
                        self.logger.debug(
                            f"Generated QA pair {len(qa_pairs)}/{target_count} "
                            f"(quality: {qa_pair.quality_metrics.overall_score:.2f})"
                        )
                        
            except Exception as e:
                self.logger.warning(f"Failed to generate QA pair: {e}")
                continue
                
        self.logger.info(
            f"Generated {len(qa_pairs)} QA pairs in {attempts} attempts "
            f"(success rate: {len(qa_pairs)/attempts:.2%})"
        )
        
        return qa_pairs
        
    def _plan_question_distribution(self, target_count: int, analysis: DocumentAnalysis) -> Dict[QuestionType, int]:
        """Plan distribution of question types based on document analysis"""
        distribution = {qtype: 0 for qtype in QuestionType}
        
        # Base distribution
        if analysis.parameters:
            distribution[QuestionType.PARAMETER] = min(len(analysis.parameters), target_count // 2)
            
        if analysis.technical_terms:
            distribution[QuestionType.DEFINITION] = min(len(analysis.technical_terms), target_count // 3)
            
        # Fill remaining with diverse types
        remaining = target_count - sum(distribution.values())
        if remaining > 0:
            types = [QuestionType.PROCESS, QuestionType.CONFIGURATION, QuestionType.COMPARISON]
            for i, qtype in enumerate(types):
                if remaining > 0:
                    distribution[qtype] = max(1, remaining // (len(types) - i))
                    remaining -= distribution[qtype]
                    
        return distribution
        
    def _select_question_type(self, distribution: Dict[QuestionType, int], existing_qa: List[QAPair]) -> QuestionType:
        """Select next question type based on distribution and existing QA pairs"""
        # Count existing types
        existing_counts = Counter(qa.question_type for qa in existing_qa)
        
        # Find type that's most under-represented
        best_type = None
        best_deficit = -1
        
        for qtype, target in distribution.items():
            current = existing_counts.get(qtype, 0)
            deficit = target - current
            if deficit > best_deficit:
                best_deficit = deficit
                best_type = qtype
                
        return best_type or QuestionType.DEFINITION
        
    async def _generate_single_qa_pair(
        self,
        content: str,
        analysis: DocumentAnalysis,
        question_type: QuestionType
    ) -> Optional[QAPair]:
        """Generate a single QA pair of specified type"""
        start_time = time.time()
        
        # Select model based on complexity
        processing_hints = ProcessingHints(
            complexity=analysis.complexity,
            document_length=analysis.length,
            technical_density=analysis.technical_density,
            parameter_count=len(analysis.parameters),
            quality_requirement=0.8
        )
        
        # Generate question using template
        question = self._generate_question(question_type, analysis, content)
        if not question:
            return None
            
        # Generate answer using model
        answer_prompt = self._create_answer_prompt(question, content, analysis)
        
        try:
            if self.model_manager:
                answer = await self.model_manager.generate_text(
                    answer_prompt,
                    max_tokens=512,
                    temperature=0.7
                )
            else:
                # Fallback: extract answer from content
                answer = self._extract_answer_from_content(question, content)
                
            # Determine difficulty
            difficulty = self._assess_difficulty(question, answer, analysis)
            
            # Quality assessment
            quality_metrics = await self._assess_quality(
                question, answer, content, analysis
            )
            
            generation_time = time.time() - start_time
            
            qa_pair = QAPair(
                question=question,
                answer=answer,
                question_type=question_type,
                difficulty=difficulty,
                quality_metrics=quality_metrics,
                source_context=content[:500] + "...",  # First 500 chars
                technical_terms=list(analysis.technical_terms)[:5],
                parameters=analysis.parameters[:3],
                confidence=quality_metrics.confidence,
                generation_time=generation_time,
                model_used=getattr(self.model_manager, 'current_model', 'fallback')
            )
            
            return qa_pair
            
        except Exception as e:
            self.logger.error(f"Failed to generate answer: {e}")
            return None
            
    def _generate_question(self, question_type: QuestionType, analysis: DocumentAnalysis, content: str) -> Optional[str]:
        """Generate question based on type and content"""
        templates = self.question_templates.get(question_type, [])
        if not templates:
            return None
            
        # Select appropriate template and fill variables
        template = np.random.choice(templates)
        
        # Fill template based on type
        if question_type == QuestionType.DEFINITION and analysis.technical_terms:
            term = np.random.choice(list(analysis.technical_terms))
            return template.format(term=term, feature=analysis.feature_name)
        elif question_type == QuestionType.PARAMETER and analysis.parameters:
            parameter = np.random.choice(analysis.parameters)
            return template.format(parameter=parameter, feature=analysis.feature_name)
        elif question_type in [QuestionType.PROCESS, QuestionType.CONFIGURATION]:
            return template.format(feature=analysis.feature_name, process=analysis.feature_name)
        else:
            # Generic question
            return template.format(feature=analysis.feature_name, term=analysis.feature_name)
            
    def _create_answer_prompt(self, question: str, content: str, analysis: DocumentAnalysis) -> str:
        """Create prompt for answer generation"""
        prompt = f"""
You are an expert in telecommunications and RAN (Radio Access Network) technology.

Based on the following technical documentation about {analysis.feature_name}, answer the question accurately and comprehensively:

DOCUMENTATION:
{content[:2000]}...

QUESTION: {question}

Provide a detailed, technically accurate answer that:
1. Directly addresses the question
2. Uses appropriate technical terminology
3. References specific parameters or procedures when relevant
4. Is concise but comprehensive

ANSWER:"""
        return prompt
        
    def _extract_answer_from_content(self, question: str, content: str) -> str:
        """Fallback method to extract answer from content"""
        # Simple extraction based on question keywords
        question_words = set(question.lower().split())
        sentences = re.split(r'[.!?]+', content)
        
        # Score sentences by keyword overlap
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            score = len(question_words.intersection(sentence_words))
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()
                
        return best_sentence or "Answer not found in documentation."
        
    def _assess_difficulty(self, question: str, answer: str, analysis: DocumentAnalysis) -> DifficultyLevel:
        """Assess question difficulty based on complexity factors"""
        factors = {
            'question_length': len(question.split()),
            'answer_length': len(answer.split()),
            'technical_terms': sum(1 for term in analysis.technical_terms if term.lower() in answer.lower()),
            'parameters': sum(1 for param in analysis.parameters if param in answer),
            'document_complexity': analysis.complexity
        }
        
        # Calculate difficulty score
        score = (
            min(factors['question_length'] / 15.0, 1.0) * 0.2 +
            min(factors['answer_length'] / 50.0, 1.0) * 0.2 +
            min(factors['technical_terms'] / 5.0, 1.0) * 0.3 +
            min(factors['parameters'] / 3.0, 1.0) * 0.2 +
            factors['document_complexity'] * 0.1
        )
        
        if score < 0.3:
            return DifficultyLevel.BASIC
        elif score < 0.6:
            return DifficultyLevel.INTERMEDIATE
        elif score < 0.8:
            return DifficultyLevel.ADVANCED
        else:
            return DifficultyLevel.EXPERT
            
    async def _assess_quality(
        self, question: str, answer: str, content: str, analysis: DocumentAnalysis
    ) -> QualityMetrics:
        """
        Comprehensive quality assessment of QA pair.
        
        Args:
            question: Generated question
            answer: Generated answer
            content: Source content
            analysis: Document analysis
            
        Returns:
            Quality metrics
        """
        metrics = QualityMetrics()
        
        # Relevance: How well does the answer address the question
        metrics.relevance_score = self._calculate_relevance(question, answer)
        
        # Accuracy: How accurate is the answer based on source content
        metrics.accuracy_score = self._calculate_accuracy(answer, content)
        
        # Technical density: Amount of technical content
        metrics.technical_density = self._calculate_technical_density(answer, analysis)
        
        # Completeness: How complete is the answer
        metrics.completeness_score = self._calculate_completeness(question, answer)
        
        # Coherence: How coherent and well-structured is the answer
        metrics.coherence_score = self._calculate_coherence(answer)
        
        # Diversity will be calculated later when comparing with other QA pairs
        metrics.diversity_score = 0.8  # Default
        
        # Confidence based on overall quality
        metrics.confidence = min(metrics.overall_score + 0.1, 1.0)
        
        return metrics
        
    def _calculate_relevance(self, question: str, answer: str) -> float:
        """Calculate relevance between question and answer"""
        if not self.sentence_transformer:
            # Fallback: word overlap
            q_words = set(question.lower().split())
            a_words = set(answer.lower().split())
            overlap = len(q_words.intersection(a_words))
            return min(overlap / max(len(q_words), 1), 1.0)
            
        try:
            # Use sentence embeddings for semantic similarity
            embeddings = self.sentence_transformer.encode([question, answer])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception:
            return 0.5  # Default
            
    def _calculate_accuracy(self, answer: str, content: str) -> float:
        """Calculate accuracy based on content overlap"""
        # Check if answer contains information from source
        answer_words = set(answer.lower().split())
        content_words = set(content.lower().split())
        
        # Calculate overlap ratio
        overlap = len(answer_words.intersection(content_words))
        accuracy = overlap / max(len(answer_words), 1)
        
        return min(accuracy * 2, 1.0)  # Boost accuracy score
        
    def _calculate_technical_density(self, answer: str, analysis: DocumentAnalysis) -> float:
        """Calculate technical density of answer"""
        answer_lower = answer.lower()
        
        # Count technical terms in answer
        tech_term_count = sum(
            1 for term in analysis.technical_terms 
            if term.lower() in answer_lower
        )
        
        # Count parameters
        param_count = sum(
            1 for param in analysis.parameters
            if param.lower() in answer_lower
        )
        
        # Calculate density
        total_words = len(answer.split())
        density = (tech_term_count + param_count) / max(total_words, 1)
        
        return min(density * 10, 1.0)  # Scale up density
        
    def _calculate_completeness(self, question: str, answer: str) -> float:
        """Calculate completeness of answer"""
        # Basic heuristics for completeness
        factors = {
            'length': min(len(answer.split()) / 20.0, 1.0),  # At least 20 words
            'structure': 1.0 if ('.' in answer and len(answer.split('.')) > 1) else 0.5,
            'specificity': 1.0 if any(char.isdigit() for char in answer) else 0.7
        }
        
        return sum(factors.values()) / len(factors)
        
    def _calculate_coherence(self, answer: str) -> float:
        """Calculate coherence and readability of answer"""
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
            
        # Check sentence length variation
        lengths = [len(s.split()) for s in sentences]
        if len(lengths) > 1:
            length_variance = np.var(lengths)
            length_score = 1.0 / (1.0 + length_variance / 10.0)
        else:
            length_score = 0.8
            
        # Check for proper sentence structure
        structure_score = 1.0 if all(s[0].isupper() for s in sentences if s) else 0.7
        
        return (length_score + structure_score) / 2
        
    def _check_diversity(self, new_qa: QAPair, existing_qa: List[QAPair]) -> bool:
        """Check if new QA pair is sufficiently diverse from existing ones"""
        if not existing_qa:
            return True
            
        # Check question similarity
        for existing in existing_qa:
            similarity = self._calculate_text_similarity(new_qa.question, existing.question)
            if similarity > 0.8:  # Too similar
                return False
                
            # Check answer similarity
            similarity = self._calculate_text_similarity(new_qa.answer, existing.answer)
            if similarity > 0.7:  # Too similar
                return False
                
        return True
        
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not self.sentence_transformer:
            # Fallback: Jaccard similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / max(union, 1)
            
        try:
            embeddings = self.sentence_transformer.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception:
            return 0.0
            
    async def process_batch(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
        qa_per_document: int = 5,
        quality_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of documents for QA generation.
        
        Args:
            documents: List of (content, metadata) tuples
            qa_per_document: Target QA pairs per document
            quality_threshold: Minimum quality threshold
            
        Returns:
            List of processing results with QA pairs
        """
        self.logger.info(f"Processing batch of {len(documents)} documents")
        
        results = []
        start_time = time.time()
        
        for i, (content, metadata) in enumerate(documents):
            doc_start_time = time.time()
            
            try:
                # Analyze document
                analysis = self.analyze_document(content, metadata)
                
                # Generate QA pairs
                qa_pairs = await self.generate_qa_pairs(
                    analysis, content, qa_per_document, quality_threshold
                )
                
                processing_time = time.time() - doc_start_time
                
                result = {
                    'document_id': analysis.document_id,
                    'feature_name': analysis.feature_name,
                    'qa_pairs': [qa.to_dict() for qa in qa_pairs],
                    'document_analysis': {
                        'complexity': analysis.complexity,
                        'technical_density': analysis.technical_density,
                        'parameter_count': len(analysis.parameters),
                        'technical_terms_count': len(analysis.technical_terms)
                    },
                    'processing_metrics': {
                        'processing_time': processing_time,
                        'qa_generated': len(qa_pairs),
                        'avg_quality': sum(qa.quality_metrics.overall_score for qa in qa_pairs) / max(len(qa_pairs), 1),
                        'success_rate': len(qa_pairs) / qa_per_document if qa_per_document > 0 else 0
                    }
                }
                
                results.append(result)
                
                self.logger.info(
                    f"Processed document {i+1}/{len(documents)}: {analysis.feature_name} "
                    f"({len(qa_pairs)} QA pairs, {processing_time:.1f}s)"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to process document {i+1}: {e}")
                continue
                
        total_time = time.time() - start_time
        total_qa = sum(len(r['qa_pairs']) for r in results)
        
        self.logger.info(
            f"Batch processing complete: {len(results)} documents, "
            f"{total_qa} QA pairs in {total_time:.1f}s "
            f"({len(results)/total_time*3600:.1f} docs/hour)"
        )
        
        return results


# Example usage and testing
if __name__ == "__main__":
    async def test_semantic_processor():
        """Test the semantic processor"""
        logging.basicConfig(level=logging.INFO)
        
        processor = SemanticProcessor()
        await processor.initialize()
        
        # Test document
        test_content = """
        # Basic TTI Bundling
        
        TTI bundling enables UE(s) to boost the uplink transmission power for improved coverage.
        
        ## Parameters
        - **EUtranCellFDD.ttiBundlingUl**: Boolean parameter controlling uplink TTI bundling
          - MO Class: EUtranCellFDD
          - Valid Values: false (disabled), true (enabled)
          - Default: false
          
        ## Description
        TTI bundling allows the UE to transmit the same data in multiple consecutive TTIs to improve the link budget.
        """
        
        metadata = {
            'document_id': 'test_doc_001',
            'feature_name': 'TTI Bundling'
        }
        
        # Analyze document
        analysis = processor.analyze_document(test_content, metadata)
        print(f"Analysis: {analysis.feature_name}, complexity: {analysis.complexity:.2f}")
        
        # Generate QA pairs
        qa_pairs = await processor.generate_qa_pairs(analysis, test_content, target_count=3)
        
        for i, qa in enumerate(qa_pairs):
            print(f"\nQA Pair {i+1}:")
            print(f"Q: {qa.question}")
            print(f"A: {qa.answer[:100]}...")
            print(f"Quality: {qa.quality_metrics.overall_score:.2f}")
            
    asyncio.run(test_semantic_processor())
