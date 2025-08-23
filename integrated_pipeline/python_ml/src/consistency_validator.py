"""
Cross-Document Consistency Validation System
Advanced validation system for ensuring consistency across document processing results
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from datetime import datetime
from collections import defaultdict, Counter
import hashlib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import spacy
from sentence_transformers import SentenceTransformer
import networkx as nx
from difflib import SequenceMatcher

@dataclass
class ConsistencyRule:
    """Defines a consistency validation rule"""
    rule_id: str
    rule_type: str  # 'semantic', 'structural', 'factual', 'format', 'quality'
    description: str
    threshold: float
    severity: str  # 'info', 'warning', 'error', 'critical'
    enabled: bool = True
    
@dataclass
class ConsistencyViolation:
    """Represents a consistency validation violation"""
    violation_id: str
    rule_id: str
    severity: str
    document_ids: List[str]
    violation_type: str
    confidence: float
    evidence: Dict[str, Any]
    recommendation: str
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass 
class ValidationReport:
    """Comprehensive validation report"""
    report_id: str
    total_documents: int
    total_violations: int
    violations_by_severity: Dict[str, int]
    violations_by_type: Dict[str, int]
    overall_consistency_score: float
    processing_time_seconds: float
    recommendations: List[str]
    timestamp: datetime

@dataclass
class DocumentConsistencyProfile:
    """Profile of document for consistency checking"""
    doc_id: str
    content_hash: str
    semantic_embedding: np.ndarray
    structural_features: Dict[str, Any]
    extracted_facts: List[Dict[str, Any]]
    quality_metrics: Dict[str, float]
    format_features: Dict[str, Any]
    processing_metadata: Dict[str, Any]

class CrossDocumentConsistencyValidator:
    """Advanced cross-document consistency validation system"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # NLP components
        self.embedding_model = None
        self.nlp_model = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
        # Validation rules
        self.consistency_rules: Dict[str, ConsistencyRule] = {}
        self._initialize_default_rules()
        
        # Document profiles
        self.document_profiles: Dict[str, DocumentConsistencyProfile] = {}
        
        # Validation history
        self.validation_history: List[ValidationReport] = []
        
        # Knowledge graphs for fact checking
        self.fact_graph = nx.DiGraph()
        
        # Initialize components
        asyncio.create_task(self._initialize_components())
    
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load validation configuration"""
        default_config = {
            "validation": {
                "semantic_similarity_threshold": 0.85,
                "structural_similarity_threshold": 0.80,
                "factual_consistency_threshold": 0.90,
                "quality_consistency_threshold": 0.15,  # Max deviation
                "format_consistency_threshold": 0.90,
                "batch_size": 32,
                "enable_fact_extraction": True,
                "enable_semantic_clustering": True,
                "confidence_threshold": 0.75
            },
            "models": {
                "embedding_model": "all-MiniLM-L6-v2",
                "spacy_model": "en_core_web_sm",
                "fact_extraction_confidence": 0.8
            },
            "reporting": {
                "max_violations_per_report": 1000,
                "detailed_evidence": True,
                "include_recommendations": True
            }
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    async def _initialize_components(self):
        """Initialize NLP and ML components"""
        try:
            # Load embedding model
            self.logger.info(f"Loading embedding model: {self.config['models']['embedding_model']}")
            self.embedding_model = SentenceTransformer(self.config['models']['embedding_model'])
            
            # Load spaCy model for fact extraction
            try:
                self.nlp_model = spacy.load(self.config['models']['spacy_model'])
            except OSError:
                self.logger.warning("spaCy model not found, fact extraction will be limited")
                self.nlp_model = None
            
            self.logger.info("Consistency validator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize consistency validator: {e}")
            raise
    
    def _initialize_default_rules(self):
        """Initialize default consistency validation rules"""
        default_rules = [
            ConsistencyRule(
                rule_id="semantic_consistency",
                rule_type="semantic",
                description="Documents with similar content should have consistent processing results",
                threshold=0.85,
                severity="warning"
            ),
            ConsistencyRule(
                rule_id="quality_consistency", 
                rule_type="quality",
                description="Similar documents should have consistent quality scores",
                threshold=0.15,  # Max deviation
                severity="warning"
            ),
            ConsistencyRule(
                rule_id="format_consistency",
                rule_type="format",
                description="Documents with similar structure should be processed consistently",
                threshold=0.90,
                severity="info"
            ),
            ConsistencyRule(
                rule_id="factual_consistency",
                rule_type="factual", 
                description="Extracted facts should be consistent across related documents",
                threshold=0.90,
                severity="error"
            ),
            ConsistencyRule(
                rule_id="processing_consistency",
                rule_type="structural",
                description="Processing metadata should be consistent for similar document types",
                threshold=0.80,
                severity="info"
            ),
            ConsistencyRule(
                rule_id="entity_consistency",
                rule_type="factual",
                description="Named entities should be consistently recognized and processed",
                threshold=0.85,
                severity="warning"
            ),
            ConsistencyRule(
                rule_id="language_consistency",
                rule_type="structural",
                description="Language detection should be consistent for similar documents",
                threshold=0.95,
                severity="error"
            ),
            ConsistencyRule(
                rule_id="topic_consistency",
                rule_type="semantic",
                description="Topic classification should be consistent for related content",
                threshold=0.80,
                severity="warning"
            )
        ]
        
        for rule in default_rules:
            self.consistency_rules[rule.rule_id] = rule
    
    async def validate_document_batch(
        self, 
        processed_documents: List[Dict[str, Any]]
    ) -> ValidationReport:
        """Validate consistency across a batch of processed documents"""
        start_time = time.time()
        report_id = f"validation_{int(time.time())}"
        
        try:
            self.logger.info(f"Validating consistency for {len(processed_documents)} documents")
            
            # Create document profiles
            document_profiles = await self._create_document_profiles(processed_documents)
            
            # Update profile store
            for profile in document_profiles:
                self.document_profiles[profile.doc_id] = profile
            
            # Run validation checks
            all_violations = []
            
            # Semantic consistency validation
            semantic_violations = await self._validate_semantic_consistency(document_profiles)
            all_violations.extend(semantic_violations)
            
            # Quality consistency validation  
            quality_violations = await self._validate_quality_consistency(document_profiles)
            all_violations.extend(quality_violations)
            
            # Format consistency validation
            format_violations = await self._validate_format_consistency(document_profiles)
            all_violations.extend(format_violations)
            
            # Factual consistency validation
            if self.config['validation']['enable_fact_extraction']:
                factual_violations = await self._validate_factual_consistency(document_profiles)
                all_violations.extend(factual_violations)
            
            # Structural consistency validation
            structural_violations = await self._validate_structural_consistency(document_profiles)
            all_violations.extend(structural_violations)
            
            # Calculate overall consistency score
            consistency_score = self._calculate_overall_consistency_score(
                len(processed_documents), all_violations
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(all_violations)
            
            # Create report
            processing_time = time.time() - start_time
            violations_by_severity = Counter(v.severity for v in all_violations)
            violations_by_type = Counter(v.violation_type for v in all_violations)
            
            report = ValidationReport(
                report_id=report_id,
                total_documents=len(processed_documents),
                total_violations=len(all_violations),
                violations_by_severity=dict(violations_by_severity),
                violations_by_type=dict(violations_by_type),
                overall_consistency_score=consistency_score,
                processing_time_seconds=processing_time,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
            # Store report
            self.validation_history.append(report)
            
            # Log summary
            self.logger.info(
                f"Validation completed: {len(all_violations)} violations found, "
                f"consistency score: {consistency_score:.3f}"
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error in batch validation: {e}")
            raise
    
    async def _create_document_profiles(
        self, 
        processed_documents: List[Dict[str, Any]]
    ) -> List[DocumentConsistencyProfile]:
        """Create comprehensive profiles for consistency checking"""
        profiles = []
        
        # Prepare content for batch embedding
        contents = []
        for doc in processed_documents:
            content = doc.get('processed_content', doc.get('content', ''))
            contents.append(content)
        
        # Generate embeddings in batch
        embeddings = self.embedding_model.encode(contents, batch_size=self.config['validation']['batch_size'])
        
        for i, doc in enumerate(processed_documents):
            doc_id = doc.get('id', f"doc_{i}")
            content = doc.get('processed_content', doc.get('content', ''))
            
            # Create profile
            profile = DocumentConsistencyProfile(
                doc_id=doc_id,
                content_hash=hashlib.md5(content.encode()).hexdigest(),
                semantic_embedding=embeddings[i],
                structural_features=await self._extract_structural_features(doc),
                extracted_facts=await self._extract_facts(content),
                quality_metrics=self._extract_quality_metrics(doc),
                format_features=await self._extract_format_features(doc),
                processing_metadata=doc.get('processing_metadata', {})
            )
            
            profiles.append(profile)
        
        return profiles
    
    async def _extract_structural_features(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structural features from document"""
        content = doc.get('processed_content', doc.get('content', ''))
        
        features = {
            'word_count': len(content.split()),
            'sentence_count': len(content.split('.')),
            'paragraph_count': len(content.split('\n\n')),
            'avg_sentence_length': np.mean([len(sent.split()) for sent in content.split('.') if sent.strip()]) if content.split('.') else 0,
            'punctuation_density': sum(1 for c in content if not c.isalnum() and c != ' ') / max(len(content), 1),
            'caps_ratio': sum(1 for c in content if c.isupper()) / max(len(content), 1),
            'digit_ratio': sum(1 for c in content if c.isdigit()) / max(len(content), 1),
            'has_headers': bool(re.search(r'^#{1,6}\s', content, re.MULTILINE)),
            'has_lists': bool(re.search(r'^\s*[-*]\s', content, re.MULTILINE)),
            'has_numbered_lists': bool(re.search(r'^\d+\.\s', content, re.MULTILINE)),
            'has_code_blocks': bool(re.search(r'```|`[^`]+`', content)),
            'has_urls': bool(re.search(r'https?://', content)),
            'has_emails': bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content))
        }
        
        return features
    
    async def _extract_facts(self, content: str) -> List[Dict[str, Any]]:
        """Extract factual information from content"""
        facts = []
        
        if self.nlp_model and len(content) < 50000:  # Limit processing for performance
            try:
                doc = self.nlp_model(content[:10000])  # Process first 10k chars
                
                # Extract named entities
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY', 'QUANTITY']:
                        facts.append({
                            'type': 'entity',
                            'category': ent.label_,
                            'text': ent.text,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'confidence': 0.8  # spaCy doesn't provide confidence scores directly
                        })
                
                # Extract relationships (basic pattern matching)
                for sent in doc.sents:
                    # Simple pattern: PERSON VERB PERSON/ORG
                    entities_in_sent = [(ent.text, ent.label_) for ent in sent.ents]
                    if len(entities_in_sent) >= 2:
                        facts.append({
                            'type': 'relationship',
                            'entities': entities_in_sent,
                            'sentence': sent.text,
                            'confidence': 0.6
                        })
                        
            except Exception as e:
                self.logger.warning(f"Error extracting facts: {e}")
        
        # Extract simple patterns without spaCy
        # Numbers and dates
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', content)
        for num in numbers[:10]:  # Limit to first 10
            facts.append({
                'type': 'number',
                'value': num,
                'confidence': 0.9
            })
        
        # Email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
        for email in emails[:5]:
            facts.append({
                'type': 'contact',
                'category': 'email',
                'value': email,
                'confidence': 0.95
            })
        
        return facts
    
    def _extract_quality_metrics(self, doc: Dict[str, Any]) -> Dict[str, float]:
        """Extract quality-related metrics from document"""
        return {
            'quality_score': doc.get('quality_score', 0.5),
            'confidence_score': doc.get('confidence_score', 0.5),
            'processing_accuracy': doc.get('processing_accuracy', 0.5),
            'completeness_score': doc.get('completeness_score', 0.5)
        }
    
    async def _extract_format_features(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Extract format-related features"""
        features = {
            'source_format': doc.get('source_format', 'unknown'),
            'encoding': doc.get('encoding', 'unknown'),
            'language': doc.get('language', 'unknown'),
            'mime_type': doc.get('mime_type', 'unknown'),
            'file_size_bytes': doc.get('file_size_bytes', 0),
            'processing_method': doc.get('processing_method', 'unknown')
        }
        
        return features
    
    async def _validate_semantic_consistency(
        self, 
        profiles: List[DocumentConsistencyProfile]
    ) -> List[ConsistencyViolation]:
        """Validate semantic consistency across similar documents"""
        violations = []
        rule = self.consistency_rules.get('semantic_consistency')
        
        if not rule or not rule.enabled:
            return violations
        
        # Calculate similarity matrix
        embeddings = np.array([profile.semantic_embedding for profile in profiles])
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find similar document pairs
        threshold = rule.threshold
        similar_pairs = []
        
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                if similarity_matrix[i][j] >= threshold:
                    similar_pairs.append((i, j, similarity_matrix[i][j]))
        
        # Check consistency within similar pairs
        for i, j, similarity in similar_pairs:
            profile1, profile2 = profiles[i], profiles[j]
            
            # Check quality score consistency
            quality1 = profile1.quality_metrics.get('quality_score', 0.5)
            quality2 = profile2.quality_metrics.get('quality_score', 0.5)
            quality_diff = abs(quality1 - quality2)
            
            if quality_diff > 0.2:  # Significant quality difference
                violation = ConsistencyViolation(
                    violation_id=f"semantic_quality_{profile1.doc_id}_{profile2.doc_id}",
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    document_ids=[profile1.doc_id, profile2.doc_id],
                    violation_type="semantic_quality_inconsistency",
                    confidence=similarity * 0.8,
                    evidence={
                        'semantic_similarity': float(similarity),
                        'quality_scores': [quality1, quality2],
                        'quality_difference': quality_diff,
                        'threshold': threshold
                    },
                    recommendation=f"Review quality scoring for semantically similar documents",
                    timestamp=datetime.now()
                )
                violations.append(violation)
        
        return violations
    
    async def _validate_quality_consistency(
        self, 
        profiles: List[DocumentConsistencyProfile]
    ) -> List[ConsistencyViolation]:
        """Validate quality score consistency"""
        violations = []
        rule = self.consistency_rules.get('quality_consistency')
        
        if not rule or not rule.enabled:
            return violations
        
        # Group documents by similarity clusters
        embeddings = np.array([profile.semantic_embedding for profile in profiles])
        
        if len(profiles) >= 3:
            # Use clustering to group similar documents
            n_clusters = min(5, len(profiles) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Check quality consistency within each cluster
            clusters = defaultdict(list)
            for idx, label in enumerate(cluster_labels):
                clusters[label].append(profiles[idx])
            
            for cluster_id, cluster_profiles in clusters.items():
                if len(cluster_profiles) < 2:
                    continue
                
                # Calculate quality statistics
                quality_scores = [p.quality_metrics.get('quality_score', 0.5) for p in cluster_profiles]
                quality_std = np.std(quality_scores)
                quality_mean = np.mean(quality_scores)
                
                if quality_std > rule.threshold:
                    violation = ConsistencyViolation(
                        violation_id=f"quality_cluster_{cluster_id}_{int(time.time())}",
                        rule_id=rule.rule_id,
                        severity=rule.severity,
                        document_ids=[p.doc_id for p in cluster_profiles],
                        violation_type="quality_variance_inconsistency",
                        confidence=0.8,
                        evidence={
                            'cluster_id': cluster_id,
                            'quality_scores': quality_scores,
                            'quality_mean': quality_mean,
                            'quality_std': quality_std,
                            'threshold': rule.threshold
                        },
                        recommendation=f"Review quality scoring consistency for similar documents in cluster {cluster_id}",
                        timestamp=datetime.now()
                    )
                    violations.append(violation)
        
        return violations
    
    async def _validate_format_consistency(
        self, 
        profiles: List[DocumentConsistencyProfile]
    ) -> List[ConsistencyViolation]:
        """Validate format processing consistency"""
        violations = []
        rule = self.consistency_rules.get('format_consistency')
        
        if not rule or not rule.enabled:
            return violations
        
        # Group by source format
        format_groups = defaultdict(list)
        for profile in profiles:
            source_format = profile.format_features.get('source_format', 'unknown')
            format_groups[source_format].append(profile)
        
        # Check consistency within each format group
        for source_format, group_profiles in format_groups.items():
            if len(group_profiles) < 2 or source_format == 'unknown':
                continue
            
            # Check processing method consistency
            processing_methods = [p.processing_metadata.get('processing_method', 'unknown') for p in group_profiles]
            method_counts = Counter(processing_methods)
            
            if len(method_counts) > 1:  # Multiple processing methods for same format
                most_common_method = method_counts.most_common(1)[0][0]
                inconsistent_profiles = [p for p in group_profiles 
                                       if p.processing_metadata.get('processing_method') != most_common_method]
                
                if inconsistent_profiles:
                    violation = ConsistencyViolation(
                        violation_id=f"format_processing_{source_format}_{int(time.time())}",
                        rule_id=rule.rule_id,
                        severity=rule.severity,
                        document_ids=[p.doc_id for p in inconsistent_profiles],
                        violation_type="format_processing_inconsistency",
                        confidence=0.9,
                        evidence={
                            'source_format': source_format,
                            'processing_methods': dict(method_counts),
                            'expected_method': most_common_method,
                            'inconsistent_documents': [p.doc_id for p in inconsistent_profiles]
                        },
                        recommendation=f"Ensure consistent processing method for {source_format} format documents",
                        timestamp=datetime.now()
                    )
                    violations.append(violation)
        
        return violations
    
    async def _validate_factual_consistency(
        self, 
        profiles: List[DocumentConsistencyProfile]
    ) -> List[ConsistencyViolation]:
        """Validate factual consistency across documents"""
        violations = []
        rule = self.consistency_rules.get('factual_consistency')
        
        if not rule or not rule.enabled:
            return violations
        
        # Build fact index
        fact_index = defaultdict(list)  # fact_value -> [(doc_id, fact_info)]
        
        for profile in profiles:
            for fact in profile.extracted_facts:
                fact_key = self._normalize_fact_key(fact)
                if fact_key:
                    fact_index[fact_key].append((profile.doc_id, fact))
        
        # Check for contradictory facts
        for fact_key, fact_occurrences in fact_index.items():
            if len(fact_occurrences) < 2:
                continue
            
            # Check for contradictions
            contradictions = self._detect_fact_contradictions(fact_occurrences)
            
            if contradictions:
                violation = ConsistencyViolation(
                    violation_id=f"factual_contradiction_{hashlib.md5(fact_key.encode()).hexdigest()[:8]}",
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    document_ids=[doc_id for doc_id, _ in fact_occurrences],
                    violation_type="factual_contradiction",
                    confidence=0.7,
                    evidence={
                        'fact_key': fact_key,
                        'contradictions': contradictions,
                        'occurrences': [(doc_id, fact) for doc_id, fact in fact_occurrences]
                    },
                    recommendation=f"Review conflicting factual information: {fact_key}",
                    timestamp=datetime.now()
                )
                violations.append(violation)
        
        return violations
    
    def _normalize_fact_key(self, fact: Dict[str, Any]) -> Optional[str]:
        """Normalize fact for comparison"""
        if fact['type'] == 'entity':
            return f"entity_{fact['category']}_{fact['text'].lower()}"
        elif fact['type'] == 'number':
            return f"number_{fact['value']}"
        elif fact['type'] == 'contact':
            return f"contact_{fact['category']}_{fact['value'].lower()}"
        return None
    
    def _detect_fact_contradictions(self, fact_occurrences: List[Tuple[str, Dict]]) -> List[Dict]:
        """Detect contradictions in fact occurrences"""
        contradictions = []
        
        # For now, simple contradiction detection
        # In a real system, this would be much more sophisticated
        if len(fact_occurrences) > 1:
            first_fact = fact_occurrences[0][1]
            for doc_id, fact in fact_occurrences[1:]:
                # Check for different values for same fact type
                if (first_fact.get('value') != fact.get('value') or 
                    first_fact.get('text') != fact.get('text')):
                    contradictions.append({
                        'doc1': fact_occurrences[0][0],
                        'fact1': first_fact,
                        'doc2': doc_id,
                        'fact2': fact
                    })
        
        return contradictions
    
    async def _validate_structural_consistency(
        self, 
        profiles: List[DocumentConsistencyProfile]
    ) -> List[ConsistencyViolation]:
        """Validate structural processing consistency"""
        violations = []
        rule = self.consistency_rules.get('processing_consistency')
        
        if not rule or not rule.enabled:
            return violations
        
        # Group by document characteristics
        length_groups = defaultdict(list)
        for profile in profiles:
            word_count = profile.structural_features.get('word_count', 0)
            if word_count < 100:
                length_category = 'short'
            elif word_count < 1000:
                length_category = 'medium'
            elif word_count < 5000:
                length_category = 'long'
            else:
                length_category = 'very_long'
            length_groups[length_category].append(profile)
        
        # Check processing consistency within each group
        for category, group_profiles in length_groups.items():
            if len(group_profiles) < 2:
                continue
            
            # Check language detection consistency
            languages = [p.format_features.get('language', 'unknown') for p in group_profiles]
            language_counts = Counter(languages)
            
            if len(language_counts) > 1 and 'unknown' not in language_counts:
                violation = ConsistencyViolation(
                    violation_id=f"language_detection_{category}_{int(time.time())}",
                    rule_id="language_consistency",
                    severity="warning",
                    document_ids=[p.doc_id for p in group_profiles],
                    violation_type="language_detection_inconsistency",
                    confidence=0.8,
                    evidence={
                        'document_category': category,
                        'detected_languages': dict(language_counts),
                        'inconsistent_documents': [p.doc_id for p in group_profiles]
                    },
                    recommendation=f"Review language detection for {category} documents",
                    timestamp=datetime.now()
                )
                violations.append(violation)
        
        return violations
    
    def _calculate_overall_consistency_score(
        self, 
        total_documents: int, 
        violations: List[ConsistencyViolation]
    ) -> float:
        """Calculate overall consistency score"""
        if total_documents == 0:
            return 1.0
        
        # Weight violations by severity
        severity_weights = {'info': 0.1, 'warning': 0.3, 'error': 0.7, 'critical': 1.0}
        
        total_penalty = 0.0
        for violation in violations:
            weight = severity_weights.get(violation.severity, 0.5)
            confidence = violation.confidence
            penalty = weight * confidence
            total_penalty += penalty
        
        # Normalize by document count
        max_possible_penalty = total_documents * 1.0  # If every document had critical violation
        penalty_ratio = min(total_penalty / max_possible_penalty, 1.0)
        
        consistency_score = 1.0 - penalty_ratio
        return max(0.0, consistency_score)
    
    def _generate_recommendations(self, violations: List[ConsistencyViolation]) -> List[str]:
        """Generate actionable recommendations based on violations"""
        recommendations = []
        
        # Count violations by type
        violation_types = Counter(v.violation_type for v in violations)
        severity_counts = Counter(v.severity for v in violations)
        
        # General recommendations based on patterns
        if violation_types.get('semantic_quality_inconsistency', 0) > 5:
            recommendations.append(
                "Multiple semantic quality inconsistencies detected. "
                "Review quality scoring algorithm for similar documents."
            )
        
        if violation_types.get('format_processing_inconsistency', 0) > 3:
            recommendations.append(
                "Format processing inconsistencies found. "
                "Standardize processing pipelines for each document format."
            )
        
        if violation_types.get('factual_contradiction', 0) > 0:
            recommendations.append(
                "Factual contradictions detected. "
                "Implement fact verification and consistency checking."
            )
        
        if severity_counts.get('critical', 0) > 0:
            recommendations.append(
                "Critical consistency violations require immediate attention. "
                "Review and fix processing pipeline before production deployment."
            )
        
        if len(violations) > 20:
            recommendations.append(
                f"High number of consistency violations ({len(violations)}) detected. "
                "Consider comprehensive review of document processing pipeline."
            )
        
        # If no specific recommendations, provide general ones
        if not recommendations:
            if len(violations) > 0:
                recommendations.append(
                    "Minor consistency issues detected. "
                    "Monitor processing quality and consider fine-tuning parameters."
                )
            else:
                recommendations.append(
                    "Excellent consistency across processed documents. "
                    "Continue monitoring for consistency maintenance."
                )
        
        return recommendations
    
    async def generate_consistency_report(self, doc_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive consistency report"""
        if doc_ids:
            profiles = [self.document_profiles[doc_id] for doc_id in doc_ids if doc_id in self.document_profiles]
        else:
            profiles = list(self.document_profiles.values())
        
        if not profiles:
            return {"status": "no_data", "message": "No document profiles available"}
        
        # Run validation on selected profiles
        processed_docs = []
        for profile in profiles:
            processed_docs.append({
                'id': profile.doc_id,
                'content': '',  # Not needed for report
                'quality_score': profile.quality_metrics.get('quality_score', 0.5),
                'processing_metadata': profile.processing_metadata
            })
        
        report = await self.validate_document_batch(processed_docs)
        
        return {
            "report_id": report.report_id,
            "timestamp": report.timestamp.isoformat(),
            "summary": {
                "total_documents": report.total_documents,
                "total_violations": report.total_violations,
                "consistency_score": report.overall_consistency_score,
                "processing_time": report.processing_time_seconds
            },
            "violations_by_severity": report.violations_by_severity,
            "violations_by_type": report.violations_by_type,
            "recommendations": report.recommendations,
            "detailed_violations": [
                v.to_dict() for v in self.validation_history[-1].violations_by_type.keys()  # Last validation
            ] if self.validation_history else []
        }
    
    async def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics"""
        if not self.validation_history:
            return {"status": "no_data"}
        
        # Calculate statistics across all validation runs
        total_documents = sum(report.total_documents for report in self.validation_history)
        total_violations = sum(report.total_violations for report in self.validation_history)
        avg_consistency_score = np.mean([report.overall_consistency_score for report in self.validation_history])
        
        # Trend analysis
        recent_scores = [report.overall_consistency_score for report in self.validation_history[-10:]]
        score_trend = "improving" if len(recent_scores) > 5 and recent_scores[-1] > recent_scores[0] else \
                     "declining" if len(recent_scores) > 5 and recent_scores[-1] < recent_scores[0] else "stable"
        
        return {
            "total_validation_runs": len(self.validation_history),
            "total_documents_validated": total_documents,
            "total_violations_found": total_violations,
            "average_consistency_score": avg_consistency_score,
            "latest_consistency_score": self.validation_history[-1].overall_consistency_score,
            "consistency_trend": score_trend,
            "document_profiles_stored": len(self.document_profiles),
            "active_validation_rules": len([r for r in self.consistency_rules.values() if r.enabled]),
            "most_common_violation_types": dict(
                Counter([v_type for report in self.validation_history 
                        for v_type in report.violations_by_type.keys()]).most_common(5)
            )
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_consistency_validator():
        # Sample processed documents
        test_documents = [
            {
                "id": "doc1",
                "content": "This is a technical document about machine learning algorithms.",
                "processed_content": "Technical document covering machine learning algorithms and their applications.",
                "quality_score": 0.85,
                "processing_metadata": {"processing_method": "advanced_nlp", "model_used": "qwen3-7b"},
                "source_format": "pdf"
            },
            {
                "id": "doc2", 
                "content": "This document covers machine learning techniques and algorithms.",
                "processed_content": "Document discussing machine learning techniques and various algorithms.",
                "quality_score": 0.55,  # Inconsistent quality for similar content
                "processing_metadata": {"processing_method": "basic_nlp", "model_used": "qwen3-1.7b"},
                "source_format": "pdf"
            },
            {
                "id": "doc3",
                "content": "A cooking recipe for chocolate cake with detailed instructions.",
                "processed_content": "Recipe for chocolate cake with step-by-step instructions.",
                "quality_score": 0.90,
                "processing_metadata": {"processing_method": "advanced_nlp", "model_used": "qwen3-7b"},
                "source_format": "txt"
            }
        ]
        
        # Initialize validator
        validator = CrossDocumentConsistencyValidator()
        
        # Run validation
        report = await validator.validate_document_batch(test_documents)
        
        print(f"Validation Report:")
        print(f"Documents: {report.total_documents}")
        print(f"Violations: {report.total_violations}")
        print(f"Consistency Score: {report.overall_consistency_score:.3f}")
        print(f"Processing Time: {report.processing_time_seconds:.2f}s")
        print(f"Recommendations: {report.recommendations}")
        
        # Get statistics
        stats = await validator.get_validation_statistics()
        print(f"\nValidation Statistics: {stats}")
    
    # Run test
    asyncio.run(test_consistency_validator())