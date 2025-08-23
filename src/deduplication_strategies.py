"""
Advanced Deduplication and Quality Filtering Strategies
Multi-level deduplication system for high-quality LLM training datasets
"""

import hashlib
import json
import re
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np
from difflib import SequenceMatcher
import logging
from pathlib import Path

@dataclass
class DuplicateCluster:
    """Represents a cluster of similar/duplicate records"""
    cluster_id: str
    records: List[Dict]
    similarity_scores: List[float]
    representative_record: Dict
    duplicate_type: str  # exact, near_exact, semantic, structural
    confidence: float

class ContentHasher:
    """Advanced content hashing for similarity detection"""
    
    def __init__(self):
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'through',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'this', 'that', 'these', 'those', 'what', 'which', 'who', 'when',
            'where', 'why', 'how', 'it', 'its', 'he', 'she', 'they', 'them', 'their'
        }
    
    def generate_content_hash(self, content: str, hash_type: str = "normalized") -> str:
        """Generate various types of content hashes"""
        
        if hash_type == "exact":
            return hashlib.md5(content.encode()).hexdigest()
        
        elif hash_type == "normalized":
            # Remove extra whitespace, normalize case
            normalized = re.sub(r'\s+', ' ', content.lower().strip())
            return hashlib.md5(normalized.encode()).hexdigest()
        
        elif hash_type == "semantic":
            # Focus on semantic content, remove stopwords
            words = re.findall(r'\b\w+\b', content.lower())
            meaningful_words = [w for w in words if w not in self.stopwords and len(w) > 2]
            semantic_content = ' '.join(sorted(meaningful_words))
            return hashlib.md5(semantic_content.encode()).hexdigest()
        
        elif hash_type == "technical":
            # Focus on technical terms and parameters
            technical_patterns = [
                r'\b[A-Z][a-z]+[A-Z][a-z]*\b',  # CamelCase (likely parameters)
                r'\b[A-Z]{2,}\b',  # Acronyms
                r'\b\w+\.\w+\b',  # MO.parameter format
                r'\b\d+(?:\.\d+)?\s*(?:dBm|MHz|Mbps|ms|%)\b'  # Values with units
            ]
            
            technical_content = []
            for pattern in technical_patterns:
                matches = re.findall(pattern, content)
                technical_content.extend(matches)
            
            tech_string = ' '.join(sorted(set(technical_content)))
            return hashlib.md5(tech_string.encode()).hexdigest()
        
        else:
            raise ValueError(f"Unknown hash type: {hash_type}")

class SimilarityCalculator:
    """Calculate various similarity metrics between records"""
    
    def __init__(self):
        self.hasher = ContentHasher()
    
    def calculate_content_similarity(self, content1: str, content2: str) -> Dict[str, float]:
        """Calculate multiple similarity metrics"""
        
        # Exact match
        exact_match = 1.0 if content1 == content2 else 0.0
        
        # Sequence similarity (character-based)
        sequence_similarity = SequenceMatcher(None, content1, content2).ratio()
        
        # Word-based Jaccard similarity
        words1 = set(re.findall(r'\b\w+\b', content1.lower()))
        words2 = set(re.findall(r'\b\w+\b', content2.lower()))
        
        if len(words1) == 0 and len(words2) == 0:
            jaccard_similarity = 1.0
        else:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Technical term similarity
        tech1 = self._extract_technical_terms(content1)
        tech2 = self._extract_technical_terms(content2)
        
        if len(tech1) == 0 and len(tech2) == 0:
            technical_similarity = 1.0
        else:
            tech_intersection = len(tech1.intersection(tech2))
            tech_union = len(tech1.union(tech2))
            technical_similarity = tech_intersection / tech_union if tech_union > 0 else 0.0
        
        # Structure similarity (sentence count, avg sentence length)
        struct_sim = self._calculate_structure_similarity(content1, content2)
        
        return {
            "exact": exact_match,
            "sequence": sequence_similarity,
            "jaccard": jaccard_similarity,
            "technical": technical_similarity,
            "structural": struct_sim
        }
    
    def _extract_technical_terms(self, content: str) -> Set[str]:
        """Extract technical terms for similarity comparison"""
        patterns = [
            r'\b[A-Z][a-z]+[A-Z][a-z]*\b',  # CamelCase
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+\.\w+\b',  # MO.parameter
            r'\bpm\w+\b',  # PM counters
            r'\b\w*[Mm]anager\b',  # Manager classes
            r'\b\w*[Ff]unction\b',  # Function classes
        ]
        
        technical_terms = set()
        for pattern in patterns:
            matches = re.findall(pattern, content)
            technical_terms.update(match.upper() for match in matches)
        
        return technical_terms
    
    def _calculate_structure_similarity(self, content1: str, content2: str) -> float:
        """Calculate structural similarity between contents"""
        
        # Sentence count similarity
        sentences1 = len(re.split(r'[.!?]', content1))
        sentences2 = len(re.split(r'[.!?]', content2))
        
        if sentences1 == 0 and sentences2 == 0:
            sent_sim = 1.0
        else:
            sent_sim = 1 - abs(sentences1 - sentences2) / max(sentences1, sentences2)
        
        # Average word length similarity
        words1 = re.findall(r'\b\w+\b', content1)
        words2 = re.findall(r'\b\w+\b', content2)
        
        if len(words1) == 0 and len(words2) == 0:
            word_sim = 1.0
        else:
            avg_len1 = np.mean([len(w) for w in words1]) if words1 else 0
            avg_len2 = np.mean([len(w) for w in words2]) if words2 else 0
            word_sim = 1 - abs(avg_len1 - avg_len2) / max(avg_len1, avg_len2, 1)
        
        return (sent_sim + word_sim) / 2

class MetadataSimilarityAnalyzer:
    """Analyze similarity in metadata structures"""
    
    def calculate_metadata_similarity(self, meta1: Dict, meta2: Dict) -> Dict[str, float]:
        """Calculate metadata-based similarity metrics"""
        
        # Feature name similarity
        feature_sim = self._compare_features(
            meta1.get("feature_name"), meta2.get("feature_name")
        )
        
        # Technical terms overlap
        terms1 = set(meta1.get("technical_terms", []))
        terms2 = set(meta2.get("technical_terms", []))
        
        if len(terms1) == 0 and len(terms2) == 0:
            terms_sim = 1.0
        else:
            terms_intersection = len(terms1.intersection(terms2))
            terms_union = len(terms1.union(terms2))
            terms_sim = terms_intersection / terms_union if terms_union > 0 else 0.0
        
        # Quality score similarity
        quality1 = float(meta1.get("quality_score", 0))
        quality2 = float(meta2.get("quality_score", 0))
        quality_sim = 1 - abs(quality1 - quality2) / 10.0  # Assuming 0-10 scale
        
        # Source dataset similarity
        source1 = meta1.get("source_dataset", "")
        source2 = meta2.get("source_dataset", "")
        source_sim = 1.0 if source1 == source2 else 0.0
        
        # Workflow type similarity (if present)
        workflow1 = meta1.get("workflow_type", "")
        workflow2 = meta2.get("workflow_type", "")
        workflow_sim = 1.0 if workflow1 == workflow2 else 0.0
        
        return {
            "feature": feature_sim,
            "technical_terms": terms_sim,
            "quality": quality_sim,
            "source": source_sim,
            "workflow": workflow_sim
        }
    
    def _compare_features(self, feature1: Optional[str], feature2: Optional[str]) -> float:
        """Compare feature names for similarity"""
        if feature1 is None and feature2 is None:
            return 1.0
        if feature1 is None or feature2 is None:
            return 0.0
        if feature1 == feature2:
            return 1.0
        
        # Use sequence similarity for partial matches
        return SequenceMatcher(None, feature1.lower(), feature2.lower()).ratio()

class DeduplicationEngine:
    """Main deduplication engine with multiple strategies"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.hasher = ContentHasher()
        self.similarity_calc = SimilarityCalculator()
        self.metadata_analyzer = MetadataSimilarityAnalyzer()
        self.logger = logging.getLogger(__name__)
        
    def _default_config(self) -> Dict:
        """Default deduplication configuration"""
        return {
            # Similarity thresholds
            "exact_threshold": 1.0,
            "near_exact_threshold": 0.95,
            "semantic_threshold": 0.85,
            "structural_threshold": 0.8,
            "metadata_threshold": 0.7,
            
            # Quality preferences for duplicate resolution
            "quality_weight": 0.4,
            "completeness_weight": 0.3,
            "enhancement_weight": 0.2,
            "source_preference_weight": 0.1,
            
            # Source preferences (higher is better)
            "source_preferences": {
                "pdf": 1.0,
                "enhanced_conversations": 0.9,
                "enhanced_grouped": 0.8,
                "diverse": 0.7,
                "unknown": 0.5
            },
            
            # Processing options
            "preserve_variants": True,  # Keep high-quality variants
            "max_cluster_size": 10,  # Maximum records per cluster
            "enable_cross_source": True  # Dedupe across different sources
        }
    
    def deduplicate_dataset(self, records: List[Dict]) -> Tuple[List[Dict], List[DuplicateCluster]]:
        """
        Main deduplication pipeline
        
        Returns:
            (deduplicated_records, duplicate_clusters)
        """
        self.logger.info(f"Starting deduplication of {len(records)} records")
        
        # Step 1: Exact duplicate detection
        exact_clusters = self._find_exact_duplicates(records)
        
        # Step 2: Near-exact duplicate detection
        near_exact_clusters = self._find_near_exact_duplicates(records, exact_clusters)
        
        # Step 3: Semantic duplicate detection
        semantic_clusters = self._find_semantic_duplicates(records, exact_clusters, near_exact_clusters)
        
        # Step 4: Structural similarity detection
        structural_clusters = self._find_structural_duplicates(records, 
                                                              exact_clusters, 
                                                              near_exact_clusters, 
                                                              semantic_clusters)
        
        # Combine all clusters
        all_clusters = exact_clusters + near_exact_clusters + semantic_clusters + structural_clusters
        
        # Step 5: Select best representatives from each cluster
        deduplicated_records = self._select_representatives(records, all_clusters)
        
        self.logger.info(f"Deduplication complete: {len(records)} -> {len(deduplicated_records)} records")
        self.logger.info(f"Found {len(all_clusters)} duplicate clusters")
        
        return deduplicated_records, all_clusters
    
    def _find_exact_duplicates(self, records: List[Dict]) -> List[DuplicateCluster]:
        """Find exact duplicate records"""
        clusters = []
        hash_to_records = defaultdict(list)
        
        for i, record in enumerate(records):
            # Create content hash from all messages
            content = self._extract_record_content(record)
            exact_hash = self.hasher.generate_content_hash(content, "exact")
            hash_to_records[exact_hash].append((i, record))
        
        # Create clusters for duplicates
        for hash_value, record_group in hash_to_records.items():
            if len(record_group) > 1:
                cluster_records = [r[1] for r in record_group]
                representative = self._select_best_record(cluster_records)
                
                cluster = DuplicateCluster(
                    cluster_id=f"exact_{hash_value[:8]}",
                    records=cluster_records,
                    similarity_scores=[1.0] * len(cluster_records),
                    representative_record=representative,
                    duplicate_type="exact",
                    confidence=1.0
                )
                clusters.append(cluster)
        
        return clusters
    
    def _find_near_exact_duplicates(self, records: List[Dict], 
                                  exclude_clusters: List[DuplicateCluster]) -> List[DuplicateCluster]:
        """Find near-exact duplicates (minor formatting differences)"""
        clusters = []
        excluded_records = set()
        
        # Build exclusion set
        for cluster in exclude_clusters:
            for record in cluster.records:
                excluded_records.add(id(record))
        
        # Group by normalized hash
        hash_to_records = defaultdict(list)
        
        for i, record in enumerate(records):
            if id(record) in excluded_records:
                continue
                
            content = self._extract_record_content(record)
            normalized_hash = self.hasher.generate_content_hash(content, "normalized")
            hash_to_records[normalized_hash].append((i, record))
        
        # Create clusters
        for hash_value, record_group in hash_to_records.items():
            if len(record_group) > 1:
                cluster_records = [r[1] for r in record_group]
                
                # Verify similarity within threshold
                similarities = self._calculate_group_similarities(cluster_records)
                if np.mean(similarities) >= self.config["near_exact_threshold"]:
                    representative = self._select_best_record(cluster_records)
                    
                    cluster = DuplicateCluster(
                        cluster_id=f"near_exact_{hash_value[:8]}",
                        records=cluster_records,
                        similarity_scores=similarities,
                        representative_record=representative,
                        duplicate_type="near_exact",
                        confidence=np.mean(similarities)
                    )
                    clusters.append(cluster)
        
        return clusters
    
    def _find_semantic_duplicates(self, records: List[Dict], 
                                *exclude_clusters: List[DuplicateCluster]) -> List[DuplicateCluster]:
        """Find semantically similar records"""
        clusters = []
        excluded_records = set()
        
        # Build exclusion set
        for cluster_group in exclude_clusters:
            for cluster in cluster_group:
                for record in cluster.records:
                    excluded_records.add(id(record))
        
        # Group by semantic hash
        hash_to_records = defaultdict(list)
        
        for i, record in enumerate(records):
            if id(record) in excluded_records:
                continue
                
            content = self._extract_record_content(record)
            semantic_hash = self.hasher.generate_content_hash(content, "semantic")
            hash_to_records[semantic_hash].append((i, record))
        
        # Create clusters for semantic similarities
        for hash_value, record_group in hash_to_records.items():
            if len(record_group) > 1:
                cluster_records = [r[1] for r in record_group]
                
                # Detailed similarity verification
                similarities = self._calculate_detailed_similarities(cluster_records)
                avg_similarity = np.mean([s["composite"] for s in similarities])
                
                if avg_similarity >= self.config["semantic_threshold"]:
                    representative = self._select_best_record(cluster_records)
                    
                    cluster = DuplicateCluster(
                        cluster_id=f"semantic_{hash_value[:8]}",
                        records=cluster_records,
                        similarity_scores=[s["composite"] for s in similarities],
                        representative_record=representative,
                        duplicate_type="semantic",
                        confidence=avg_similarity
                    )
                    clusters.append(cluster)
        
        return clusters
    
    def _find_structural_duplicates(self, records: List[Dict], 
                                  *exclude_clusters: List[DuplicateCluster]) -> List[DuplicateCluster]:
        """Find structurally similar records"""
        clusters = []
        excluded_records = set()
        
        # Build exclusion set
        for cluster_group in exclude_clusters:
            for cluster in cluster_group:
                for record in cluster.records:
                    excluded_records.add(id(record))
        
        remaining_records = [r for r in records if id(r) not in excluded_records]
        
        # Pairwise comparison for structural similarity
        # Note: This is O(n²) - for large datasets, consider sampling or LSH
        if len(remaining_records) > 1000:
            self.logger.warning(f"Large dataset ({len(remaining_records)} records) - "
                              f"structural deduplication may be slow")
        
        processed_indices = set()
        
        for i, record1 in enumerate(remaining_records):
            if i in processed_indices:
                continue
                
            cluster_candidates = [record1]
            cluster_indices = [i]
            
            for j, record2 in enumerate(remaining_records[i+1:], i+1):
                if j in processed_indices:
                    continue
                    
                # Calculate structural similarity
                content1 = self._extract_record_content(record1)
                content2 = self._extract_record_content(record2)
                
                similarity_scores = self.similarity_calc.calculate_content_similarity(content1, content2)
                structural_sim = similarity_scores["structural"]
                
                if structural_sim >= self.config["structural_threshold"]:
                    cluster_candidates.append(record2)
                    cluster_indices.append(j)
            
            # Create cluster if we have multiple candidates
            if len(cluster_candidates) > 1:
                processed_indices.update(cluster_indices)
                
                # Calculate all similarities within cluster
                similarities = self._calculate_group_similarities(cluster_candidates)
                representative = self._select_best_record(cluster_candidates)
                
                cluster = DuplicateCluster(
                    cluster_id=f"structural_{i}_{len(clusters)}",
                    records=cluster_candidates,
                    similarity_scores=similarities,
                    representative_record=representative,
                    duplicate_type="structural",
                    confidence=np.mean(similarities)
                )
                clusters.append(cluster)
        
        return clusters
    
    def _extract_record_content(self, record: Dict) -> str:
        """Extract content for hashing and comparison"""
        messages = record.get("messages", [])
        content_parts = []
        
        for message in messages:
            content = message.get("content", "")
            role = message.get("role", "")
            content_parts.append(f"{role}: {content}")
        
        return "\n".join(content_parts)
    
    def _calculate_group_similarities(self, records: List[Dict]) -> List[float]:
        """Calculate average similarity within a group of records"""
        if len(records) <= 1:
            return [1.0]
        
        similarities = []
        n = len(records)
        
        for i in range(n):
            record_similarities = []
            content_i = self._extract_record_content(records[i])
            
            for j in range(n):
                if i != j:
                    content_j = self._extract_record_content(records[j])
                    sim_scores = self.similarity_calc.calculate_content_similarity(content_i, content_j)
                    # Weighted composite similarity
                    composite = (
                        0.3 * sim_scores["sequence"] +
                        0.3 * sim_scores["jaccard"] +
                        0.2 * sim_scores["technical"] +
                        0.2 * sim_scores["structural"]
                    )
                    record_similarities.append(composite)
            
            similarities.append(np.mean(record_similarities))
        
        return similarities
    
    def _calculate_detailed_similarities(self, records: List[Dict]) -> List[Dict[str, float]]:
        """Calculate detailed similarity metrics for a group"""
        if len(records) <= 1:
            return [{"composite": 1.0, "content": 1.0, "metadata": 1.0}]
        
        detailed_similarities = []
        n = len(records)
        
        for i in range(n):
            content_sims = []
            metadata_sims = []
            
            for j in range(n):
                if i != j:
                    # Content similarity
                    content_i = self._extract_record_content(records[i])
                    content_j = self._extract_record_content(records[j])
                    content_sim_scores = self.similarity_calc.calculate_content_similarity(content_i, content_j)
                    
                    content_composite = (
                        0.3 * content_sim_scores["sequence"] +
                        0.3 * content_sim_scores["jaccard"] +
                        0.2 * content_sim_scores["technical"] +
                        0.2 * content_sim_scores["structural"]
                    )
                    content_sims.append(content_composite)
                    
                    # Metadata similarity
                    metadata_sim_scores = self.metadata_analyzer.calculate_metadata_similarity(
                        records[i].get("metadata", {}),
                        records[j].get("metadata", {})
                    )
                    
                    metadata_composite = (
                        0.3 * metadata_sim_scores["feature"] +
                        0.3 * metadata_sim_scores["technical_terms"] +
                        0.2 * metadata_sim_scores["quality"] +
                        0.1 * metadata_sim_scores["source"] +
                        0.1 * metadata_sim_scores["workflow"]
                    )
                    metadata_sims.append(metadata_composite)
            
            avg_content = np.mean(content_sims) if content_sims else 1.0
            avg_metadata = np.mean(metadata_sims) if metadata_sims else 1.0
            
            # Overall composite with content-heavy weighting
            composite = 0.7 * avg_content + 0.3 * avg_metadata
            
            detailed_similarities.append({
                "composite": composite,
                "content": avg_content,
                "metadata": avg_metadata
            })
        
        return detailed_similarities
    
    def _select_best_record(self, records: List[Dict]) -> Dict:
        """Select the best record from a group of duplicates"""
        if len(records) == 1:
            return records[0]
        
        best_record = records[0]
        best_score = 0.0
        
        for record in records:
            score = self._calculate_record_quality_score(record)
            if score > best_score:
                best_score = score
                best_record = record
        
        return best_record
    
    def _calculate_record_quality_score(self, record: Dict) -> float:
        """Calculate quality score for record selection"""
        metadata = record.get("metadata", {})
        
        # Base quality score
        base_quality = float(metadata.get("quality_score", 5.0)) / 10.0
        
        # Enhancement bonus
        enhancement_bonus = 0.0
        if metadata.get("enhancement_applied"):
            enhancement_bonus += 0.2
        if metadata.get("transformation_applied"):
            enhancement_bonus += 0.1
        if metadata.get("pdf_enhanced"):
            enhancement_bonus += 0.15
        
        # Completeness bonus
        completeness_bonus = 0.0
        metadata_fields = ["feature_name", "technical_terms", "source_dataset"]
        present_fields = sum(1 for field in metadata_fields if metadata.get(field))
        completeness_bonus = (present_fields / len(metadata_fields)) * 0.1
        
        # Source preference
        source = metadata.get("source_dataset", "unknown")
        source_bonus = self.config["source_preferences"].get(source, 0.5) * 0.1
        
        # Technical terms richness
        tech_terms = metadata.get("technical_terms", [])
        tech_bonus = min(0.1, len(tech_terms) / 10.0) if isinstance(tech_terms, list) else 0.0
        
        # Confidence score (if available)
        confidence = float(metadata.get("confidence", 0.8))
        confidence_bonus = confidence * 0.1
        
        # Calculate weighted composite
        composite_score = (
            self.config["quality_weight"] * base_quality +
            self.config["completeness_weight"] * completeness_bonus +
            self.config["enhancement_weight"] * enhancement_bonus +
            self.config["source_preference_weight"] * source_bonus +
            0.1 * tech_bonus +
            0.1 * confidence_bonus
        )
        
        return composite_score
    
    def _select_representatives(self, records: List[Dict], 
                             clusters: List[DuplicateCluster]) -> List[Dict]:
        """Select representative records from all clusters"""
        
        # Get all records that are part of clusters
        clustered_record_ids = set()
        representatives = []
        
        for cluster in clusters:
            # Add representative record
            representatives.append(cluster.representative_record)
            
            # Track all records in this cluster
            for record in cluster.records:
                clustered_record_ids.add(id(record))
            
            # If preserving variants, add high-quality alternatives
            if self.config.get("preserve_variants", False):
                sorted_records = sorted(
                    cluster.records, 
                    key=self._calculate_record_quality_score,
                    reverse=True
                )
                
                # Add up to 2 high-quality variants if significantly different
                for i, record in enumerate(sorted_records[1:], 1):
                    if i >= 2:  # Limit variants
                        break
                        
                    quality_score = self._calculate_record_quality_score(record)
                    rep_quality_score = self._calculate_record_quality_score(cluster.representative_record)
                    
                    # Add if quality is close to representative
                    if quality_score >= rep_quality_score * 0.9:
                        representatives.append(record)
        
        # Add all non-clustered records
        for record in records:
            if id(record) not in clustered_record_ids:
                representatives.append(record)
        
        return representatives

def analyze_duplication_report(clusters: List[DuplicateCluster]) -> Dict[str, Any]:
    """Generate detailed deduplication analysis report"""
    
    if not clusters:
        return {
            "total_clusters": 0,
            "total_duplicates_removed": 0,
            "duplication_types": {},
            "average_confidence": 0.0,
            "largest_cluster_size": 0
        }
    
    # Basic statistics
    total_clusters = len(clusters)
    total_duplicates_removed = sum(len(cluster.records) - 1 for cluster in clusters)
    
    # Duplication type breakdown
    type_counts = Counter(cluster.duplicate_type for cluster in clusters)
    
    # Confidence statistics
    confidences = [cluster.confidence for cluster in clusters]
    avg_confidence = np.mean(confidences)
    
    # Cluster size analysis
    cluster_sizes = [len(cluster.records) for cluster in clusters]
    largest_cluster = max(cluster_sizes)
    avg_cluster_size = np.mean(cluster_sizes)
    
    # Quality impact analysis
    quality_preservation = []
    for cluster in clusters:
        original_qualities = []
        for record in cluster.records:
            quality = record.get("metadata", {}).get("quality_score", 0)
            original_qualities.append(float(quality))
        
        if original_qualities:
            rep_quality = cluster.representative_record.get("metadata", {}).get("quality_score", 0)
            rep_quality = float(rep_quality)
            quality_preservation.append(rep_quality / max(original_qualities))
    
    report = {
        "total_clusters": total_clusters,
        "total_duplicates_removed": total_duplicates_removed,
        "duplication_types": dict(type_counts),
        "average_confidence": avg_confidence,
        "confidence_distribution": {
            "min": min(confidences),
            "max": max(confidences),
            "std": np.std(confidences)
        },
        "cluster_size_stats": {
            "largest_cluster": largest_cluster,
            "average_cluster_size": avg_cluster_size,
            "size_distribution": dict(Counter(cluster_sizes))
        },
        "quality_preservation": {
            "average_preservation_ratio": np.mean(quality_preservation) if quality_preservation else 1.0,
            "min_preservation": min(quality_preservation) if quality_preservation else 1.0
        }
    }
    
    return report

# Example usage and testing
if __name__ == "__main__":
    # Initialize deduplication engine
    deduplication_engine = DeduplicationEngine()
    
    # Sample test records
    test_records = [
        {
            "messages": [
                {"role": "user", "content": "How to configure LTE handover?"},
                {"role": "assistant", "content": "Configure LTE handover by setting handoverMargin parameter."}
            ],
            "metadata": {
                "feature_name": "LTE Handover",
                "quality_score": 9.0,
                "technical_terms": ["LTE", "handover"],
                "source_dataset": "enhanced"
            }
        },
        {
            "messages": [
                {"role": "user", "content": "How to configure LTE handover?"},
                {"role": "assistant", "content": "Configure LTE handover by setting the handoverMargin parameter."}
            ],
            "metadata": {
                "feature_name": "LTE Handover",
                "quality_score": 8.5,
                "technical_terms": ["LTE", "handover"],
                "source_dataset": "diverse"
            }
        }
    ]
    
    # Run deduplication
    deduplicated, clusters = deduplication_engine.deduplicate_dataset(test_records)
    
    # Generate report
    report = analyze_duplication_report(clusters)
    
    print(f"Original records: {len(test_records)}")
    print(f"After deduplication: {len(deduplicated)}")
    print(f"Duplicate clusters found: {len(clusters)}")
    print("Deduplication Report:", json.dumps(report, indent=2))