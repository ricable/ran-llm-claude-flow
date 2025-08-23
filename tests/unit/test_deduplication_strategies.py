"""
Unit tests for deduplication_strategies.py module
Tests advanced deduplication and quality filtering strategies
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import hashlib
import numpy as np
from collections import defaultdict

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from deduplication_strategies import (
    DuplicateCluster,
    ContentHasher,
    SimilarityCalculator,
    MetadataSimilarityAnalyzer,
    DeduplicationEngine,
    analyze_duplication_report
)


class TestDuplicateCluster(unittest.TestCase):
    """Test DuplicateCluster dataclass"""
    
    def test_duplicate_cluster_creation(self):
        """Test creating a DuplicateCluster"""
        records = [{"id": 1}, {"id": 2}]
        cluster = DuplicateCluster(
            cluster_id="test_cluster_001",
            records=records,
            similarity_scores=[0.95, 0.98],
            representative_record={"id": 1},
            duplicate_type="exact",
            confidence=0.96
        )
        
        self.assertEqual(cluster.cluster_id, "test_cluster_001")
        self.assertEqual(len(cluster.records), 2)
        self.assertEqual(cluster.similarity_scores, [0.95, 0.98])
        self.assertEqual(cluster.representative_record, {"id": 1})
        self.assertEqual(cluster.duplicate_type, "exact")
        self.assertEqual(cluster.confidence, 0.96)


class TestContentHasher(unittest.TestCase):
    """Test ContentHasher class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.hasher = ContentHasher()
    
    def test_initialization(self):
        """Test hasher initialization"""
        self.assertIsInstance(self.hasher.stopwords, set)
        self.assertIn("the", self.hasher.stopwords)
        self.assertIn("and", self.hasher.stopwords)
        self.assertTrue(len(self.hasher.stopwords) > 30)
    
    def test_generate_exact_hash(self):
        """Test exact content hashing"""
        content = "This is a test content"
        hash1 = self.hasher.generate_content_hash(content, "exact")
        hash2 = self.hasher.generate_content_hash(content, "exact")
        
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 32)  # MD5 hash length
        
        # Different content should produce different hash
        different_content = "This is different content"
        hash3 = self.hasher.generate_content_hash(different_content, "exact")
        self.assertNotEqual(hash1, hash3)
    
    def test_generate_normalized_hash(self):
        """Test normalized content hashing"""
        content1 = "This  is   a    test"
        content2 = "this is a test"
        content3 = "THIS IS A TEST"
        
        hash1 = self.hasher.generate_content_hash(content1, "normalized")
        hash2 = self.hasher.generate_content_hash(content2, "normalized")
        hash3 = self.hasher.generate_content_hash(content3, "normalized")
        
        # All should be the same after normalization
        self.assertEqual(hash1, hash2)
        self.assertEqual(hash2, hash3)
    
    def test_generate_semantic_hash(self):
        """Test semantic content hashing"""
        content1 = "The LTE eNodeB configuration is important"
        content2 = "LTE eNodeB configuration important"
        
        hash1 = self.hasher.generate_content_hash(content1, "semantic")
        hash2 = self.hasher.generate_content_hash(content2, "semantic")
        
        # Should be similar after removing stopwords
        self.assertEqual(hash1, hash2)
    
    def test_generate_technical_hash(self):
        """Test technical content hashing"""
        content = "Configure EUtranCellFDD.handoverMargin to 3 dBm for LTE optimization"
        hash_result = self.hasher.generate_content_hash(content, "technical")
        
        self.assertIsInstance(hash_result, str)
        self.assertEqual(len(hash_result), 32)
    
    def test_invalid_hash_type(self):
        """Test invalid hash type raises error"""
        with self.assertRaises(ValueError):
            self.hasher.generate_content_hash("test", "invalid_type")


class TestSimilarityCalculator(unittest.TestCase):
    """Test SimilarityCalculator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = SimilarityCalculator()
    
    def test_initialization(self):
        """Test calculator initialization"""
        self.assertIsInstance(self.calculator.hasher, ContentHasher)
    
    def test_calculate_content_similarity_exact_match(self):
        """Test content similarity for exact matches"""
        content1 = "This is exactly the same content"
        content2 = "This is exactly the same content"
        
        similarities = self.calculator.calculate_content_similarity(content1, content2)
        
        self.assertIn("exact", similarities)
        self.assertIn("sequence", similarities)
        self.assertIn("jaccard", similarities)
        self.assertIn("technical", similarities)
        self.assertIn("structural", similarities)
        
        self.assertEqual(similarities["exact"], 1.0)
        self.assertEqual(similarities["sequence"], 1.0)
        self.assertEqual(similarities["jaccard"], 1.0)
    
    def test_calculate_content_similarity_different_content(self):
        """Test content similarity for different content"""
        content1 = "Configure LTE handover parameters"
        content2 = "Setup 5G network optimization"
        
        similarities = self.calculator.calculate_content_similarity(content1, content2)
        
        self.assertEqual(similarities["exact"], 0.0)
        self.assertLess(similarities["sequence"], 1.0)
        self.assertLess(similarities["jaccard"], 1.0)
    
    def test_calculate_content_similarity_empty_content(self):
        """Test content similarity for empty content"""
        similarities = self.calculator.calculate_content_similarity("", "")
        
        self.assertEqual(similarities["exact"], 1.0)
        self.assertEqual(similarities["jaccard"], 1.0)
        self.assertEqual(similarities["technical"], 1.0)
    
    def test_extract_technical_terms(self):
        """Test technical term extraction"""
        content = "Configure EUtranCellFDD.handoverMargin using pmCounterA and LTE Manager"
        terms = self.calculator._extract_technical_terms(content)
        
        self.assertIsInstance(terms, set)
        # The actual implementation extracts the full parameter name
        self.assertIn("EUTRANCELLFDD.HANDOVERMARGIN", terms)
        self.assertIn("LTE", terms)
        self.assertIn("MANAGER", terms)
    
    def test_calculate_structure_similarity(self):
        """Test structural similarity calculation"""
        content1 = "First sentence. Second sentence. Third sentence."
        content2 = "Another sentence. Different sentence. Final sentence."
        
        similarity = self.calculator._calculate_structure_similarity(content1, content2)
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_calculate_structure_similarity_empty(self):
        """Test structural similarity for empty content"""
        similarity = self.calculator._calculate_structure_similarity("", "")
        self.assertEqual(similarity, 1.0)


class TestMetadataSimilarityAnalyzer(unittest.TestCase):
    """Test MetadataSimilarityAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = MetadataSimilarityAnalyzer()
    
    def test_calculate_metadata_similarity(self):
        """Test metadata similarity calculation"""
        meta1 = {
            "feature_name": "LTE Handover Optimization",
            "technical_terms": ["LTE", "eNodeB", "handover"],
            "quality_score": 8.5,
            "source_dataset": "pdf",
            "workflow_type": "optimization"
        }
        
        meta2 = {
            "feature_name": "LTE Handover Optimization",
            "technical_terms": ["LTE", "eNodeB", "mobility"],
            "quality_score": 8.2,
            "source_dataset": "pdf",
            "workflow_type": "optimization"
        }
        
        similarities = self.analyzer.calculate_metadata_similarity(meta1, meta2)
        
        self.assertIn("feature", similarities)
        self.assertIn("technical_terms", similarities)
        self.assertIn("quality", similarities)
        self.assertIn("source", similarities)
        self.assertIn("workflow", similarities)
        
        self.assertEqual(similarities["feature"], 1.0)  # Exact match
        self.assertEqual(similarities["source"], 1.0)   # Exact match
        self.assertEqual(similarities["workflow"], 1.0) # Exact match
        self.assertGreater(similarities["technical_terms"], 0.0)  # Partial overlap
    
    def test_calculate_metadata_similarity_empty(self):
        """Test metadata similarity with empty metadata"""
        similarities = self.analyzer.calculate_metadata_similarity({}, {})
        
        self.assertEqual(similarities["feature"], 1.0)  # Both None
        self.assertEqual(similarities["technical_terms"], 1.0)  # Both empty
        self.assertEqual(similarities["source"], 1.0)   # Both empty
    
    def test_compare_features(self):
        """Test feature comparison"""
        # Exact match
        similarity = self.analyzer._compare_features("Feature A", "Feature A")
        self.assertEqual(similarity, 1.0)
        
        # Both None
        similarity = self.analyzer._compare_features(None, None)
        self.assertEqual(similarity, 1.0)
        
        # One None
        similarity = self.analyzer._compare_features("Feature A", None)
        self.assertEqual(similarity, 0.0)
        
        # Partial match
        similarity = self.analyzer._compare_features("LTE Optimization", "LTE Configuration")
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)


class TestDeduplicationEngine(unittest.TestCase):
    """Test DeduplicationEngine class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = DeduplicationEngine()
        
        # Sample records for testing
        self.sample_records = [
            {
                "messages": [
                    {"role": "user", "content": "How to configure LTE handover?"},
                    {"role": "assistant", "content": "Use cmedit to set handoverMargin parameter."}
                ],
                "metadata": {
                    "feature_name": "LTE Handover",
                    "quality_score": 8.5,
                    "source_dataset": "pdf"
                }
            },
            {
                "messages": [
                    {"role": "user", "content": "How to configure LTE handover?"},
                    {"role": "assistant", "content": "Use cmedit to set handoverMargin parameter."}
                ],
                "metadata": {
                    "feature_name": "LTE Handover",
                    "quality_score": 8.5,
                    "source_dataset": "pdf"
                }
            },
            {
                "messages": [
                    {"role": "user", "content": "What is 5G optimization?"},
                    {"role": "assistant", "content": "5G optimization involves tuning network parameters."}
                ],
                "metadata": {
                    "feature_name": "5G Optimization",
                    "quality_score": 7.8,
                    "source_dataset": "enhanced"
                }
            }
        ]
    
    def test_initialization(self):
        """Test engine initialization"""
        self.assertIsInstance(self.engine.config, dict)
        self.assertIsInstance(self.engine.hasher, ContentHasher)
        self.assertIsInstance(self.engine.similarity_calc, SimilarityCalculator)
        self.assertIsInstance(self.engine.metadata_analyzer, MetadataSimilarityAnalyzer)
    
    def test_initialization_with_config(self):
        """Test engine initialization with custom config"""
        custom_config = {"exact_threshold": 0.99}
        engine = DeduplicationEngine(custom_config)
        self.assertEqual(engine.config["exact_threshold"], 0.99)
    
    def test_default_config(self):
        """Test default configuration"""
        config = self.engine._default_config()
        
        self.assertIn("exact_threshold", config)
        self.assertIn("near_exact_threshold", config)
        self.assertIn("semantic_threshold", config)
        self.assertIn("structural_threshold", config)
        self.assertIn("quality_weight", config)
        self.assertIn("source_preferences", config)
        
        self.assertEqual(config["exact_threshold"], 1.0)
        self.assertGreater(config["near_exact_threshold"], 0.9)
    
    @patch('deduplication_strategies.logging.getLogger')
    def test_deduplicate_dataset(self, mock_logger):
        """Test main deduplication pipeline"""
        mock_logger.return_value.info = Mock()
        
        deduplicated, clusters = self.engine.deduplicate_dataset(self.sample_records)
        
        self.assertIsInstance(deduplicated, list)
        self.assertIsInstance(clusters, list)
        self.assertLessEqual(len(deduplicated), len(self.sample_records))
        
        # Should find at least one duplicate cluster (first two records are identical)
        self.assertGreater(len(clusters), 0)
    
    def test_find_exact_duplicates(self):
        """Test exact duplicate detection"""
        clusters = self.engine._find_exact_duplicates(self.sample_records)
        
        # Should find one cluster with the first two identical records
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0].records), 2)
        self.assertEqual(clusters[0].duplicate_type, "exact")
        self.assertEqual(clusters[0].confidence, 1.0)
    
    def test_find_near_exact_duplicates(self):
        """Test near-exact duplicate detection"""
        # Create records with minor differences
        near_exact_records = [
            {
                "messages": [
                    {"role": "user", "content": "How to configure LTE handover?"},
                    {"role": "assistant", "content": "Use cmedit to set handoverMargin."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "How  to  configure  LTE  handover?"},  # Extra spaces
                    {"role": "assistant", "content": "Use cmedit to set handoverMargin."}
                ]
            }
        ]
        
        # First find exact duplicates (should be none)
        exact_clusters = self.engine._find_exact_duplicates(near_exact_records)
        
        # Then find near-exact duplicates
        near_exact_clusters = self.engine._find_near_exact_duplicates(near_exact_records, exact_clusters)
        
        self.assertIsInstance(near_exact_clusters, list)
    
    def test_extract_record_content(self):
        """Test record content extraction"""
        record = {
            "messages": [
                {"role": "user", "content": "Test question"},
                {"role": "assistant", "content": "Test answer"}
            ]
        }
        
        content = self.engine._extract_record_content(record)
        
        self.assertIn("user: Test question", content)
        self.assertIn("assistant: Test answer", content)
    
    def test_extract_record_content_empty(self):
        """Test record content extraction with empty messages"""
        record = {"messages": []}
        content = self.engine._extract_record_content(record)
        self.assertEqual(content, "")
        
        record = {}
        content = self.engine._extract_record_content(record)
        self.assertEqual(content, "")
    
    def test_calculate_group_similarities(self):
        """Test group similarity calculation"""
        records = self.sample_records[:2]  # Two identical records
        similarities = self.engine._calculate_group_similarities(records)
        
        self.assertIsInstance(similarities, list)
        self.assertEqual(len(similarities), len(records))
        self.assertTrue(all(0.0 <= s <= 1.0 for s in similarities))
    
    def test_select_best_record(self):
        """Test best record selection"""
        records = [
            {
                "metadata": {"quality_score": 7.0, "source_dataset": "enhanced"},
                "messages": [{"role": "user", "content": "Short"}]
            },
            {
                "metadata": {"quality_score": 9.0, "source_dataset": "pdf"},
                "messages": [{"role": "user", "content": "Much longer content with more detail"}]
            }
        ]
        
        best_record = self.engine._select_best_record(records)
        
        # Should select the record with higher quality score
        self.assertEqual(best_record["metadata"]["quality_score"], 9.0)
    
    def test_calculate_record_quality_score(self):
        """Test record quality score calculation"""
        record = {
            "metadata": {
                "quality_score": 8.5,
                "source_dataset": "pdf",
                "enhancement_applied": True
            },
            "messages": [
                {"role": "user", "content": "Detailed question with technical content"},
                {"role": "assistant", "content": "Comprehensive answer with examples"}
            ]
        }
        
        quality_score = self.engine._calculate_record_quality_score(record)
        
        self.assertIsInstance(quality_score, float)
        self.assertGreaterEqual(quality_score, 0.0)
    
    def test_select_representatives(self):
        """Test representative selection from clusters"""
        # Create a simple cluster
        cluster = DuplicateCluster(
            cluster_id="test",
            records=self.sample_records[:2],
            similarity_scores=[1.0, 1.0],
            representative_record=self.sample_records[0],
            duplicate_type="exact",
            confidence=1.0
        )
        
        representatives = self.engine._select_representatives(self.sample_records, [cluster])
        
        self.assertIsInstance(representatives, list)
        self.assertLessEqual(len(representatives), len(self.sample_records))


class TestDeduplicationEngineEdgeCases(unittest.TestCase):
    """Test edge cases for DeduplicationEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = DeduplicationEngine()
    
    def test_empty_dataset(self):
        """Test deduplication with empty dataset"""
        deduplicated, clusters = self.engine.deduplicate_dataset([])
        
        self.assertEqual(len(deduplicated), 0)
        self.assertEqual(len(clusters), 0)
    
    def test_single_record(self):
        """Test deduplication with single record"""
        single_record = [{
            "messages": [{"role": "user", "content": "Test"}],
            "metadata": {"quality_score": 8.0}
        }]
        
        deduplicated, clusters = self.engine.deduplicate_dataset(single_record)
        
        self.assertEqual(len(deduplicated), 1)
        self.assertEqual(len(clusters), 0)
    
    def test_malformed_records(self):
        """Test handling of malformed records"""
        malformed_records = [
            {},  # Empty record
            {"messages": [{"content": "No role"}]},  # Missing role
            {"messages": [{"role": "user"}]},  # Missing content
        ]
        
        # Should not raise exception
        deduplicated, clusters = self.engine.deduplicate_dataset(malformed_records)
        
        self.assertIsInstance(deduplicated, list)
        self.assertIsInstance(clusters, list)
    
    def test_large_cluster_handling(self):
        """Test handling of large clusters"""
        # Create many identical records
        large_cluster_records = []
        base_record = {
            "messages": [{"role": "user", "content": "Same content"}],
            "metadata": {"quality_score": 8.0}
        }
        
        for i in range(15):  # More than max_cluster_size
            large_cluster_records.append(base_record.copy())
        
        deduplicated, clusters = self.engine.deduplicate_dataset(large_cluster_records)
        
        self.assertIsInstance(deduplicated, list)
        self.assertIsInstance(clusters, list)
        self.assertGreater(len(clusters), 0)


class TestAnalyzeDuplicationReport(unittest.TestCase):
    """Test analyze_duplication_report function"""
    
    def test_analyze_duplication_report(self):
        """Test duplication report analysis"""
        clusters = [
            DuplicateCluster(
                cluster_id="exact_001",
                records=[
                    {"id": 1, "metadata": {"quality_score": 8.5}},
                    {"id": 2, "metadata": {"quality_score": 8.0}}
                ],
                similarity_scores=[1.0, 1.0],
                representative_record={"id": 1, "metadata": {"quality_score": 8.5}},
                duplicate_type="exact",
                confidence=1.0
            ),
            DuplicateCluster(
                cluster_id="semantic_001",
                records=[
                    {"id": 3, "metadata": {"quality_score": 7.5}},
                    {"id": 4, "metadata": {"quality_score": 7.0}},
                    {"id": 5, "metadata": {"quality_score": 6.5}}
                ],
                similarity_scores=[0.9, 0.85, 0.88],
                representative_record={"id": 3, "metadata": {"quality_score": 7.5}},
                duplicate_type="semantic",
                confidence=0.88
            )
        ]
        
        report = analyze_duplication_report(clusters)
        
        self.assertIsInstance(report, dict)
        self.assertIn("total_clusters", report)
        self.assertIn("total_duplicates_removed", report)
        self.assertIn("duplication_types", report)
        self.assertIn("average_confidence", report)
        self.assertIn("cluster_size_stats", report)
        
        self.assertEqual(report["total_clusters"], 2)
        self.assertEqual(report["total_duplicates_removed"], 3)  # 5 total records - 2 representatives
        self.assertEqual(report["cluster_size_stats"]["largest_cluster"], 3)
    
    def test_analyze_empty_report(self):
        """Test analysis of empty duplication report"""
        report = analyze_duplication_report([])
        
        self.assertEqual(report["total_clusters"], 0)
        self.assertEqual(report["total_duplicates_removed"], 0)
        self.assertEqual(len(report["duplication_types"]), 0)


class TestDeduplicationIntegration(unittest.TestCase):
    """Integration tests for deduplication system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = DeduplicationEngine()
    
    def test_end_to_end_deduplication(self):
        """Test complete deduplication workflow"""
        # Create a realistic dataset with various types of duplicates
        dataset = [
            # Exact duplicates
            {
                "messages": [
                    {"role": "user", "content": "How to configure LTE handover parameters?"},
                    {"role": "assistant", "content": "Use cmedit to set handoverMargin and a3Offset."}
                ],
                "metadata": {"feature_name": "LTE Handover", "quality_score": 8.5}
            },
            {
                "messages": [
                    {"role": "user", "content": "How to configure LTE handover parameters?"},
                    {"role": "assistant", "content": "Use cmedit to set handoverMargin and a3Offset."}
                ],
                "metadata": {"feature_name": "LTE Handover", "quality_score": 8.5}
            },
            # Near-exact duplicates (minor formatting differences)
            {
                "messages": [
                    {"role": "user", "content": "How  to  configure  LTE  handover  parameters?"},
                    {"role": "assistant", "content": "Use cmedit to set handoverMargin and a3Offset."}
                ],
                "metadata": {"feature_name": "LTE Handover", "quality_score": 8.2}
            },
            # Unique record
            {
                "messages": [
                    {"role": "user", "content": "What is 5G network slicing?"},
                    {"role": "assistant", "content": "Network slicing allows multiple virtual networks."}
                ],
                "metadata": {"feature_name": "5G Slicing", "quality_score": 9.0}
            }
        ]
        
        # Run deduplication
        deduplicated, clusters = self.engine.deduplicate_dataset(dataset)
        
        # Verify results
        self.assertLessEqual(len(deduplicated), len(dataset))  # Should not increase dataset size
        self.assertIsInstance(clusters, list)  # Should return clusters list
        
        # If duplicates were found, verify the report
        if len(clusters) > 0:
            report = analyze_duplication_report(clusters)
            self.assertGreater(report["total_duplicates_removed"], 0)
        
        # Verify that deduplicated records are valid
        for record in deduplicated:
            self.assertIsInstance(record, dict)
            content = self.engine._extract_record_content(record)
            self.assertIsInstance(content, str)


if __name__ == '__main__':
    unittest.main()