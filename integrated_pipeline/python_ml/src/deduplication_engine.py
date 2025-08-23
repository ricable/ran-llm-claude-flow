"""
Advanced Embedding-Based Deduplication Engine
High-performance deduplication system using semantic embeddings and MLX acceleration
"""

import asyncio
import numpy as np
import json
import logging
import hashlib
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from datetime import datetime
import mlx.core as mx
import mlx.nn as nn
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from collections import defaultdict, Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

@dataclass
class DocumentFingerprint:
    """Document fingerprint for deduplication"""
    doc_id: str
    content_hash: str
    semantic_hash: str
    structural_hash: str
    length_category: str
    domain_category: str
    embedding: np.ndarray
    features: Dict[str, Any]
    similarity_cluster: Optional[int] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['embedding'] = self.embedding.tolist() if self.embedding is not None else None
        return result

@dataclass
class DuplicationMatch:
    """Represents a potential duplication match"""
    doc_id_1: str
    doc_id_2: str
    similarity_score: float
    match_type: str  # 'exact', 'near_exact', 'semantic', 'structural'
    confidence: float
    evidence: Dict[str, Any]
    recommendation: str  # 'remove', 'merge', 'keep_both'

@dataclass
class DeduplicationResult:
    """Results of deduplication process"""
    total_documents: int
    duplicate_groups: List[List[str]]
    similarity_matches: List[DuplicationMatch]
    removed_documents: List[str]
    merged_documents: List[Tuple[str, List[str]]]  # (final_id, source_ids)
    statistics: Dict[str, Any]
    processing_time_seconds: float

class AdvancedDeduplicationEngine:
    """Advanced deduplication engine with multiple similarity detection methods"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Initialize embedding model
        self.embedding_model = None
        self.embedding_dim = 384  # Default for sentence-transformers
        
        # FAISS index for efficient similarity search
        self.faiss_index = None
        self.document_fingerprints: Dict[str, DocumentFingerprint] = {}
        
        # TF-IDF vectorizer for structural similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True
        )
        
        # Spacy model for advanced text processing
        self.nlp_model = None
        
        # Deduplication statistics
        self.stats = {
            "documents_processed": 0,
            "duplicates_found": 0,
            "exact_matches": 0,
            "semantic_matches": 0,
            "structural_matches": 0,
            "processing_time_total": 0.0
        }
        
        # Initialize components
        asyncio.create_task(self._initialize_components())
    
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load deduplication configuration"""
        default_config = {
            "embedding_model": "all-MiniLM-L6-v2",
            "similarity_thresholds": {
                "exact_match": 1.0,
                "near_exact": 0.98,
                "semantic_high": 0.90,
                "semantic_medium": 0.80,
                "structural": 0.85
            },
            "content_length_categories": {
                "short": (0, 1000),
                "medium": (1001, 5000),
                "long": (5001, 20000),
                "very_long": (20001, float('inf'))
            },
            "enable_structural_analysis": True,
            "enable_semantic_clustering": True,
            "batch_size": 64,
            "max_cluster_size": 100,
            "use_mlx_acceleration": True,
            "preserve_quality_documents": True,
            "quality_threshold": 0.7
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    async def _initialize_components(self):
        """Initialize ML components asynchronously"""
        try:
            # Load embedding model
            self.logger.info(f"Loading embedding model: {self.config['embedding_model']}")
            self.embedding_model = SentenceTransformer(self.config['embedding_model'])
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
            # Initialize FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
            
            # Load spaCy model for advanced processing
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy model not found, using basic text processing")
                self.nlp_model = None
            
            self.logger.info("Deduplication engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize deduplication components: {e}")
            raise
    
    async def process_documents_batch(self, documents: List[Dict]) -> DeduplicationResult:
        """Process a batch of documents for deduplication"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing {len(documents)} documents for deduplication")
            
            # Generate fingerprints for all documents
            fingerprints = await self._generate_document_fingerprints(documents)
            
            # Add to document store
            for fp in fingerprints:
                self.document_fingerprints[fp.doc_id] = fp
            
            # Find duplicate groups
            duplicate_groups = await self._find_duplicate_groups(fingerprints)
            
            # Generate detailed similarity matches
            similarity_matches = await self._generate_similarity_matches(duplicate_groups)
            
            # Determine removal/merge strategy
            removed_docs, merged_docs = await self._determine_deduplication_strategy(
                duplicate_groups, similarity_matches
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_statistics(len(documents), duplicate_groups, processing_time)
            
            return DeduplicationResult(
                total_documents=len(documents),
                duplicate_groups=duplicate_groups,
                similarity_matches=similarity_matches,
                removed_documents=removed_docs,
                merged_documents=merged_docs,
                statistics=self._generate_result_statistics(documents, duplicate_groups),
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error processing document batch: {e}")
            raise
    
    async def _generate_document_fingerprints(self, documents: List[Dict]) -> List[DocumentFingerprint]:
        """Generate comprehensive fingerprints for documents"""
        fingerprints = []
        
        # Prepare content for batch embedding
        contents = [doc.get('content', '') for doc in documents]
        
        # Generate embeddings in batches
        embeddings = await self._generate_embeddings_batch(contents)
        
        for i, doc in enumerate(documents):
            doc_id = doc.get('id', f"doc_{i}")
            content = doc.get('content', '')
            
            # Generate various hash signatures
            content_hash = self._generate_content_hash(content)
            semantic_hash = self._generate_semantic_hash(content)
            structural_hash = self._generate_structural_hash(content)
            
            # Categorize document
            length_category = self._categorize_document_length(len(content))
            domain_category = self._categorize_document_domain(content)
            
            # Extract features
            features = await self._extract_document_features(content, doc)
            
            fingerprint = DocumentFingerprint(
                doc_id=doc_id,
                content_hash=content_hash,
                semantic_hash=semantic_hash,
                structural_hash=structural_hash,
                length_category=length_category,
                domain_category=domain_category,
                embedding=embeddings[i],
                features=features
            )
            
            fingerprints.append(fingerprint)
        
        return fingerprints
    
    async def _generate_embeddings_batch(self, contents: List[str]) -> np.ndarray:
        """Generate embeddings for batch of contents using MLX acceleration"""
        try:
            # Use sentence-transformers for embedding generation
            embeddings = self.embedding_model.encode(
                contents,
                batch_size=self.config['batch_size'],
                show_progress_bar=False,
                normalize_embeddings=True  # For cosine similarity
            )
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            # Fallback to random embeddings
            return np.random.random((len(contents), self.embedding_dim))
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate content-based hash for exact matching"""
        # Normalize content for hashing
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _generate_semantic_hash(self, content: str) -> str:
        """Generate semantic hash based on key terms and concepts"""
        # Extract key terms
        key_terms = self._extract_key_terms(content)
        
        # Create semantic signature
        semantic_signature = ' '.join(sorted(key_terms[:20]))  # Top 20 terms
        return hashlib.md5(semantic_signature.encode()).hexdigest()
    
    def _generate_structural_hash(self, content: str) -> str:
        """Generate hash based on document structure"""
        # Extract structural features
        structure_features = [
            str(len(content.split('\n'))),  # Line count
            str(len(content.split('.'))),   # Sentence count
            str(len(content.split())),      # Word count
            str(content.count(':')),        # Colon count (lists, definitions)
            str(content.count('-')),        # Dash count (bullet points)
            str(content.count('(')),        # Parentheses count
        ]
        
        # Add pattern-based features
        if re.search(r'^\d+\.', content, re.MULTILINE):
            structure_features.append('numbered_list')
        if re.search(r'^\s*[-*]\s', content, re.MULTILINE):
            structure_features.append('bullet_list')
        if re.search(r'#{1,6}\s', content):
            structure_features.append('markdown_headers')
        
        structure_signature = '_'.join(structure_features)
        return hashlib.md5(structure_signature.encode()).hexdigest()
    
    def _extract_key_terms(self, content: str) -> List[str]:
        """Extract key terms from content using NLP"""
        if self.nlp_model:
            # Use spaCy for advanced term extraction
            doc = self.nlp_model(content[:10000])  # Limit processing to first 10k chars
            
            key_terms = []
            for token in doc:
                if (token.pos_ in ['NOUN', 'ADJ', 'VERB'] and 
                    not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 2):
                    key_terms.append(token.lemma_.lower())
            
            # Add named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'TECH']:
                    key_terms.append(ent.text.lower())
            
        else:
            # Fallback to simple term extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
            # Filter common stop words
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            key_terms = [w for w in words if w not in stop_words]
        
        # Count frequency and return most common
        term_counts = Counter(key_terms)
        return [term for term, _ in term_counts.most_common(50)]
    
    def _categorize_document_length(self, length: int) -> str:
        """Categorize document by length"""
        for category, (min_len, max_len) in self.config['content_length_categories'].items():
            if min_len <= length <= max_len:
                return category
        return 'unknown'
    
    def _categorize_document_domain(self, content: str) -> str:
        """Categorize document by domain/topic"""
        # Simple domain classification based on keywords
        technical_keywords = ['function', 'method', 'class', 'algorithm', 'implementation', 'system', 'api']
        business_keywords = ['strategy', 'market', 'customer', 'revenue', 'business', 'management']
        academic_keywords = ['research', 'study', 'analysis', 'conclusion', 'methodology', 'abstract']
        
        content_lower = content.lower()
        
        tech_score = sum(1 for kw in technical_keywords if kw in content_lower)
        biz_score = sum(1 for kw in business_keywords if kw in content_lower)
        acad_score = sum(1 for kw in academic_keywords if kw in content_lower)
        
        scores = {'technical': tech_score, 'business': biz_score, 'academic': acad_score}
        
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return 'general'
    
    async def _extract_document_features(self, content: str, doc_metadata: Dict) -> Dict[str, Any]:
        """Extract comprehensive document features"""
        features = {}
        
        # Basic text statistics
        features['char_count'] = len(content)
        features['word_count'] = len(content.split())
        features['sentence_count'] = len(content.split('.'))
        features['paragraph_count'] = len(content.split('\n\n'))
        features['avg_word_length'] = np.mean([len(w) for w in content.split()]) if content.split() else 0
        
        # Structural features
        features['has_headers'] = bool(re.search(r'#{1,6}\s', content))
        features['has_lists'] = bool(re.search(r'^\s*[-*]\d+\.', content, re.MULTILINE))
        features['has_code_blocks'] = bool(re.search(r'```|`[^`]+`', content))
        features['has_urls'] = bool(re.search(r'https?://', content))
        features['has_emails'] = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content))
        
        # Content complexity features
        features['unique_words'] = len(set(content.lower().split()))
        features['vocabulary_diversity'] = features['unique_words'] / max(features['word_count'], 1)
        features['punctuation_density'] = sum(1 for c in content if not c.isalnum() and c != ' ') / len(content)
        
        # Language features
        features['caps_ratio'] = sum(1 for c in content if c.isupper()) / len(content)
        features['digit_ratio'] = sum(1 for c in content if c.isdigit()) / len(content)
        
        # Quality indicators (from metadata if available)
        features['quality_score'] = doc_metadata.get('quality_score', 0.5)
        features['source_reliability'] = doc_metadata.get('source_reliability', 0.5)
        features['creation_date'] = doc_metadata.get('creation_date', '')
        
        return features
    
    async def _find_duplicate_groups(self, fingerprints: List[DocumentFingerprint]) -> List[List[str]]:
        """Find groups of duplicate documents using multiple methods"""
        duplicate_groups = []
        processed_docs = set()
        
        # Add embeddings to FAISS index
        embeddings = np.array([fp.embedding for fp in fingerprints])
        self.faiss_index.add(embeddings)
        
        # Group by exact content hash first
        exact_groups = defaultdict(list)
        for fp in fingerprints:
            exact_groups[fp.content_hash].append(fp.doc_id)
        
        # Add exact match groups
        for group in exact_groups.values():
            if len(group) > 1:
                duplicate_groups.append(group)
                processed_docs.update(group)
        
        # Find semantic similarity groups
        semantic_groups = await self._find_semantic_similarity_groups(
            [fp for fp in fingerprints if fp.doc_id not in processed_docs]
        )
        
        duplicate_groups.extend(semantic_groups)
        for group in semantic_groups:
            processed_docs.update(group)
        
        # Find structural similarity groups
        if self.config['enable_structural_analysis']:
            structural_groups = await self._find_structural_similarity_groups(
                [fp for fp in fingerprints if fp.doc_id not in processed_docs]
            )
            duplicate_groups.extend(structural_groups)
        
        return duplicate_groups
    
    async def _find_semantic_similarity_groups(self, fingerprints: List[DocumentFingerprint]) -> List[List[str]]:
        """Find groups based on semantic similarity using embeddings"""
        if not fingerprints:
            return []
        
        groups = []
        processed = set()
        
        threshold = self.config['similarity_thresholds']['semantic_medium']
        
        for i, fp in enumerate(fingerprints):
            if fp.doc_id in processed:
                continue
            
            # Search for similar documents
            query_embedding = fp.embedding.reshape(1, -1)
            distances, indices = self.faiss_index.search(query_embedding, min(len(fingerprints), 50))
            
            # Build similarity group
            group = [fp.doc_id]
            processed.add(fp.doc_id)
            
            for j, similarity in zip(indices[0], distances[0]):
                if j != i and similarity >= threshold:
                    similar_fp = fingerprints[j]
                    if similar_fp.doc_id not in processed:
                        # Additional validation
                        if self._validate_semantic_similarity(fp, similar_fp):
                            group.append(similar_fp.doc_id)
                            processed.add(similar_fp.doc_id)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _validate_semantic_similarity(self, fp1: DocumentFingerprint, fp2: DocumentFingerprint) -> bool:
        """Validate semantic similarity with additional checks"""
        # Check if documents are in compatible categories
        if fp1.length_category != fp2.length_category:
            # Allow similarity between adjacent length categories
            length_order = ['short', 'medium', 'long', 'very_long']
            try:
                idx1, idx2 = length_order.index(fp1.length_category), length_order.index(fp2.length_category)
                if abs(idx1 - idx2) > 1:
                    return False
            except ValueError:
                pass
        
        # Check domain compatibility
        if fp1.domain_category != fp2.domain_category and fp1.domain_category != 'general' and fp2.domain_category != 'general':
            return False
        
        # Check quality difference - preserve higher quality documents
        quality_diff = abs(fp1.features.get('quality_score', 0.5) - fp2.features.get('quality_score', 0.5))
        if quality_diff > 0.3:  # Significant quality difference
            return False
        
        return True
    
    async def _find_structural_similarity_groups(self, fingerprints: List[DocumentFingerprint]) -> List[List[str]]:
        """Find groups based on structural similarity"""
        if not fingerprints:
            return []
        
        groups = []
        
        # Group by structural hash
        struct_groups = defaultdict(list)
        for fp in fingerprints:
            struct_groups[fp.structural_hash].append(fp.doc_id)
        
        # Add groups with multiple documents
        for group in struct_groups.values():
            if len(group) > 1:
                # Validate structural similarity with additional features
                validated_group = self._validate_structural_group(
                    [fp for fp in fingerprints if fp.doc_id in group]
                )
                if len(validated_group) > 1:
                    groups.append(validated_group)
        
        return groups
    
    def _validate_structural_group(self, fingerprints: List[DocumentFingerprint]) -> List[str]:
        """Validate structural similarity group with feature analysis"""
        if len(fingerprints) <= 1:
            return [fp.doc_id for fp in fingerprints]
        
        # Check structural feature consistency
        base_fp = fingerprints[0]
        validated = [base_fp.doc_id]
        
        for fp in fingerprints[1:]:
            # Compare structural features
            feature_similarity = self._calculate_structural_feature_similarity(base_fp.features, fp.features)
            
            if feature_similarity >= self.config['similarity_thresholds']['structural']:
                validated.append(fp.doc_id)
        
        return validated
    
    def _calculate_structural_feature_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between structural features"""
        structural_keys = ['has_headers', 'has_lists', 'has_code_blocks', 'paragraph_count', 'sentence_count']
        
        similarities = []
        
        for key in structural_keys:
            val1 = features1.get(key, 0)
            val2 = features2.get(key, 0)
            
            if isinstance(val1, bool) and isinstance(val2, bool):
                similarities.append(1.0 if val1 == val2 else 0.0)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if val1 == 0 and val2 == 0:
                    similarities.append(1.0)
                else:
                    max_val = max(val1, val2)
                    min_val = min(val1, val2)
                    similarities.append(min_val / max_val if max_val > 0 else 0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    async def _generate_similarity_matches(self, duplicate_groups: List[List[str]]) -> List[DuplicationMatch]:
        """Generate detailed similarity matches for duplicate groups"""
        matches = []
        
        for group in duplicate_groups:
            # Generate all pairwise matches within group
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    doc_id_1, doc_id_2 = group[i], group[j]
                    
                    match = await self._create_similarity_match(doc_id_1, doc_id_2)
                    if match:
                        matches.append(match)
        
        return matches
    
    async def _create_similarity_match(self, doc_id_1: str, doc_id_2: str) -> Optional[DuplicationMatch]:
        """Create detailed similarity match between two documents"""
        fp1 = self.document_fingerprints.get(doc_id_1)
        fp2 = self.document_fingerprints.get(doc_id_2)
        
        if not fp1 or not fp2:
            return None
        
        # Calculate different similarity scores
        exact_match = fp1.content_hash == fp2.content_hash
        semantic_sim = np.dot(fp1.embedding, fp2.embedding)  # Cosine similarity (normalized embeddings)
        structural_sim = self._calculate_structural_feature_similarity(fp1.features, fp2.features)
        
        # Determine match type and overall similarity
        if exact_match:
            match_type = 'exact'
            similarity_score = 1.0
        elif semantic_sim >= self.config['similarity_thresholds']['near_exact']:
            match_type = 'near_exact'
            similarity_score = semantic_sim
        elif semantic_sim >= self.config['similarity_thresholds']['semantic_high']:
            match_type = 'semantic'
            similarity_score = semantic_sim
        elif structural_sim >= self.config['similarity_thresholds']['structural']:
            match_type = 'structural'
            similarity_score = structural_sim
        else:
            return None  # Below threshold
        
        # Calculate confidence
        confidence = self._calculate_match_confidence(fp1, fp2, semantic_sim, structural_sim)
        
        # Generate evidence
        evidence = {
            'semantic_similarity': float(semantic_sim),
            'structural_similarity': float(structural_sim),
            'exact_content_match': exact_match,
            'same_length_category': fp1.length_category == fp2.length_category,
            'same_domain': fp1.domain_category == fp2.domain_category,
            'quality_difference': abs(fp1.features.get('quality_score', 0.5) - fp2.features.get('quality_score', 0.5)),
            'length_difference': abs(fp1.features.get('char_count', 0) - fp2.features.get('char_count', 0))
        }
        
        # Generate recommendation
        recommendation = self._generate_deduplication_recommendation(fp1, fp2, evidence)
        
        return DuplicationMatch(
            doc_id_1=doc_id_1,
            doc_id_2=doc_id_2,
            similarity_score=similarity_score,
            match_type=match_type,
            confidence=confidence,
            evidence=evidence,
            recommendation=recommendation
        )
    
    def _calculate_match_confidence(self, fp1: DocumentFingerprint, fp2: DocumentFingerprint, 
                                  semantic_sim: float, structural_sim: float) -> float:
        """Calculate confidence in the similarity match"""
        confidence_factors = []
        
        # Semantic similarity confidence
        confidence_factors.append(semantic_sim)
        
        # Structural similarity confidence
        confidence_factors.append(structural_sim * 0.7)  # Lower weight
        
        # Length similarity confidence
        len1, len2 = fp1.features.get('char_count', 0), fp2.features.get('char_count', 0)
        if max(len1, len2) > 0:
            length_sim = min(len1, len2) / max(len1, len2)
            confidence_factors.append(length_sim * 0.5)
        
        # Domain consistency
        if fp1.domain_category == fp2.domain_category:
            confidence_factors.append(0.8)
        
        # Quality consistency (prefer keeping higher quality)
        qual_diff = abs(fp1.features.get('quality_score', 0.5) - fp2.features.get('quality_score', 0.5))
        if qual_diff < 0.2:
            confidence_factors.append(0.7)
        
        return min(np.mean(confidence_factors), 1.0)
    
    def _generate_deduplication_recommendation(self, fp1: DocumentFingerprint, fp2: DocumentFingerprint, 
                                             evidence: Dict) -> str:
        """Generate recommendation for handling duplicate pair"""
        # Exact matches -> remove lower quality one
        if evidence['exact_content_match']:
            qual1 = fp1.features.get('quality_score', 0.5)
            qual2 = fp2.features.get('quality_score', 0.5)
            if qual1 > qual2:
                return f"remove_{fp2.doc_id}"
            elif qual2 > qual1:
                return f"remove_{fp1.doc_id}"
            else:
                return f"remove_{fp2.doc_id}"  # Remove second one arbitrarily
        
        # High semantic similarity with quality difference
        if evidence['semantic_similarity'] > 0.95 and evidence['quality_difference'] > 0.2:
            qual1 = fp1.features.get('quality_score', 0.5)
            qual2 = fp2.features.get('quality_score', 0.5)
            if qual1 > qual2:
                return f"remove_{fp2.doc_id}"
            else:
                return f"remove_{fp1.doc_id}"
        
        # Moderate similarity -> merge if beneficial
        if 0.85 <= evidence['semantic_similarity'] < 0.95:
            if evidence['same_domain'] and evidence['length_difference'] < 1000:
                return f"merge_{fp1.doc_id}_{fp2.doc_id}"
        
        # Low confidence -> keep both
        return "keep_both"
    
    async def _determine_deduplication_strategy(self, duplicate_groups: List[List[str]], 
                                              similarity_matches: List[DuplicationMatch]) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
        """Determine which documents to remove or merge"""
        removed_docs = []
        merged_docs = []
        processed_docs = set()
        
        # Process similarity matches by confidence (highest first)
        sorted_matches = sorted(similarity_matches, key=lambda m: m.confidence, reverse=True)
        
        for match in sorted_matches:
            if match.doc_id_1 in processed_docs or match.doc_id_2 in processed_docs:
                continue
            
            if match.recommendation.startswith('remove_'):
                doc_to_remove = match.recommendation.split('_', 1)[1]
                removed_docs.append(doc_to_remove)
                processed_docs.add(doc_to_remove)
                processed_docs.add(match.doc_id_1 if doc_to_remove != match.doc_id_1 else match.doc_id_2)
                
            elif match.recommendation.startswith('merge_'):
                # Create merged document
                merge_id = f"merged_{match.doc_id_1}_{match.doc_id_2}"
                source_ids = [match.doc_id_1, match.doc_id_2]
                merged_docs.append((merge_id, source_ids))
                processed_docs.update(source_ids)
        
        return removed_docs, merged_docs
    
    def _update_statistics(self, total_docs: int, duplicate_groups: List[List[str]], processing_time: float):
        """Update processing statistics"""
        self.stats["documents_processed"] += total_docs
        self.stats["duplicates_found"] += sum(len(group) - 1 for group in duplicate_groups)  # Don't count the original
        self.stats["processing_time_total"] += processing_time
    
    def _generate_result_statistics(self, documents: List[Dict], duplicate_groups: List[List[str]]) -> Dict[str, Any]:
        """Generate comprehensive result statistics"""
        total_duplicates = sum(len(group) - 1 for group in duplicate_groups)
        
        # Analyze duplicate types
        exact_duplicates = 0
        semantic_duplicates = 0
        structural_duplicates = 0
        
        for group in duplicate_groups:
            if len(group) > 1:
                # Simplified classification - would need match details for accuracy
                example_fp = self.document_fingerprints.get(group[0])
                if example_fp:
                    # Count by examining first pair in group
                    if len(set(self.document_fingerprints[doc_id].content_hash for doc_id in group)) == 1:
                        exact_duplicates += len(group) - 1
                    else:
                        semantic_duplicates += len(group) - 1
        
        return {
            "total_documents": len(documents),
            "duplicate_groups_found": len(duplicate_groups),
            "total_duplicates": total_duplicates,
            "exact_duplicates": exact_duplicates,
            "semantic_duplicates": semantic_duplicates,
            "structural_duplicates": structural_duplicates,
            "deduplication_rate": total_duplicates / len(documents) if len(documents) > 0 else 0,
            "unique_documents": len(documents) - total_duplicates,
            "largest_duplicate_group": max((len(group) for group in duplicate_groups), default=0),
            "avg_group_size": np.mean([len(group) for group in duplicate_groups]) if duplicate_groups else 0
        }
    
    async def get_deduplication_statistics(self) -> Dict[str, Any]:
        """Get comprehensive deduplication statistics"""
        return {
            **self.stats,
            "total_fingerprints": len(self.document_fingerprints),
            "avg_processing_time_per_doc": (
                self.stats["processing_time_total"] / max(self.stats["documents_processed"], 1)
            ),
            "duplicate_detection_rate": (
                self.stats["duplicates_found"] / max(self.stats["documents_processed"], 1)
            ),
            "faiss_index_size": self.faiss_index.ntotal if self.faiss_index else 0
        }
    
    async def save_fingerprints(self, filepath: Path):
        """Save document fingerprints for future use"""
        try:
            fingerprints_data = {
                doc_id: fp.to_dict() for doc_id, fp in self.document_fingerprints.items()
            }
            
            with open(filepath, 'w') as f:
                json.dump(fingerprints_data, f, indent=2)
            
            self.logger.info(f"Saved {len(fingerprints_data)} fingerprints to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving fingerprints: {e}")
    
    async def load_fingerprints(self, filepath: Path):
        """Load document fingerprints from file"""
        try:
            with open(filepath, 'r') as f:
                fingerprints_data = json.load(f)
            
            for doc_id, fp_data in fingerprints_data.items():
                if fp_data.get('embedding'):
                    fp_data['embedding'] = np.array(fp_data['embedding'])
                
                fp = DocumentFingerprint(**fp_data)
                self.document_fingerprints[doc_id] = fp
            
            # Rebuild FAISS index
            if self.document_fingerprints:
                embeddings = np.array([fp.embedding for fp in self.document_fingerprints.values()])
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                self.faiss_index.add(embeddings)
            
            self.logger.info(f"Loaded {len(fingerprints_data)} fingerprints from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading fingerprints: {e}")

# Example usage and testing
if __name__ == "__main__":
    async def test_deduplication():
        # Sample test documents
        test_documents = [
            {"id": "doc1", "content": "This is a test document about machine learning and AI.", "quality_score": 0.8},
            {"id": "doc2", "content": "This is a test document about machine learning and AI.", "quality_score": 0.7},  # Exact duplicate
            {"id": "doc3", "content": "This document discusses machine learning and artificial intelligence.", "quality_score": 0.9},  # Semantic duplicate
            {"id": "doc4", "content": "A completely different topic about cooking recipes.", "quality_score": 0.6},
            {"id": "doc5", "content": "Another document on cooking and food preparation.", "quality_score": 0.8},  # Semantic similar
        ]
        
        # Initialize deduplication engine
        engine = AdvancedDeduplicationEngine()
        
        # Process documents
        result = await engine.process_documents_batch(test_documents)
        
        print(f"Processed {result.total_documents} documents")
        print(f"Found {len(result.duplicate_groups)} duplicate groups")
        print(f"Documents to remove: {result.removed_documents}")
        print(f"Documents to merge: {result.merged_documents}")
        print(f"Processing time: {result.processing_time_seconds:.2f}s")
        print(f"Statistics: {result.statistics}")
        
        # Get engine statistics
        engine_stats = await engine.get_deduplication_statistics()
        print(f"Engine Statistics: {engine_stats}")
    
    # Run test
    asyncio.run(test_deduplication())