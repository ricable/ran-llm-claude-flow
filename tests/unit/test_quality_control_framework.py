"""
Unit tests for Quality Control Framework
TDD London School testing patterns with comprehensive mocking
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the module under test
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from quality_control_framework import (
    QualityMetrics,
    QualityController,
    DeduplicationEngine,
    validate_dataset_batch
)


class TestQualityMetrics:
    """Test quality metrics dataclass"""
    
    def test_quality_metrics_initialization(self):
        """Test quality metrics creation"""
        metrics = QualityMetrics(
            content_coherence=0.9,
            technical_accuracy=0.85,
            metadata_completeness=0.95,
            conversation_flow=0.88,
            terminology_consistency=0.92,
            overall_score=8.9
        )
        
        assert metrics.content_coherence == 0.9
        assert metrics.technical_accuracy == 0.85
        assert metrics.metadata_completeness == 0.95
        assert metrics.conversation_flow == 0.88
        assert metrics.terminology_consistency == 0.92
        assert metrics.overall_score == 8.9


class TestQualityController:
    """Test quality controller functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.controller = QualityController()
    
    def test_initialization_with_default_config(self):
        """Test controller initialization with default config"""
        assert self.controller.config["min_quality_score"] == 6.0
        assert self.controller.config["min_content_length"] == 30
        assert self.controller.config["max_content_length"] == 4096
        assert "feature_name" in self.controller.config["required_metadata_fields"]
        assert self.controller.config["technical_term_threshold"] == 2
        assert self.controller.config["similarity_threshold"] == 0.85
        assert self.controller.config["confidence_threshold"] == 0.7
    
    def test_initialization_with_custom_config(self):
        """Test controller initialization with custom config"""
        custom_config = {
            "min_quality_score": 7.5,
            "min_content_length": 100,
            "technical_term_threshold": 5
        }
        controller = QualityController(custom_config)
        
        assert controller.config["min_quality_score"] == 7.5
        assert controller.config["min_content_length"] == 100
        assert controller.config["technical_term_threshold"] == 5
    
    def test_technical_terms_database_loaded(self):
        """Test technical terms database is loaded"""
        assert len(self.controller.technical_terms_db) > 0
        assert "eNodeB" in self.controller.technical_terms_db
        assert "LTE" in self.controller.technical_terms_db
        assert "5G" in self.controller.technical_terms_db
        assert "RSRP" in self.controller.technical_terms_db
    
    def test_validate_record_valid_record(self):
        """Test validation of a valid record"""
        valid_record = {
            "messages": [
                {
                    "role": "user",
                    "content": "How do I configure the eNodeB for LTE handover optimization with RSRP threshold?"
                },
                {
                    "role": "assistant",
                    "content": "To configure LTE handover optimization, set the RSRP threshold to -110 dBm and configure the time-to-trigger to 320ms. This will optimize handover performance in your 5G network deployment."
                }
            ],
            "metadata": {
                "feature_name": "LTE Handover Optimization",
                "quality_score": 9.2,
                "technical_content": True
            }
        }
        
        is_valid, metrics, errors = self.controller.validate_record(valid_record)
        
        assert is_valid is True
        assert isinstance(metrics, QualityMetrics)
        assert metrics.overall_score >= 8.0
        assert len(errors) == 0
    
    def test_validate_record_missing_messages(self):
        """Test validation with missing messages field"""
        invalid_record = {
            "metadata": {
                "feature_name": "Test Feature",
                "quality_score": 8.5,
                "technical_content": True
            }
        }
        
        is_valid, metrics, errors = self.controller.validate_record(invalid_record)
        
        assert is_valid is False
        assert "Missing 'messages' field" in errors
        assert metrics.overall_score == 0
    
    def test_validate_record_missing_metadata(self):
        """Test validation with missing metadata field"""
        invalid_record = {
            "messages": [
                {"role": "user", "content": "Test question?"},
                {"role": "assistant", "content": "Test answer."}
            ]
        }
        
        is_valid, metrics, errors = self.controller.validate_record(invalid_record)
        
        assert is_valid is False
        assert "Missing 'metadata' field" in errors
        assert metrics.overall_score == 0
    
    def test_validate_record_insufficient_messages(self):
        """Test validation with insufficient messages"""
        invalid_record = {
            "messages": [
                {"role": "user", "content": "Only one message"}
            ],
            "metadata": {
                "feature_name": "Test Feature",
                "quality_score": 8.5,
                "technical_content": True
            }
        }
        
        is_valid, metrics, errors = self.controller.validate_record(invalid_record)
        
        assert is_valid is False
        assert any("need at least 2 messages" in error for error in errors)
    
    def test_validate_record_wrong_message_roles(self):
        """Test validation with wrong message roles"""
        invalid_record = {
            "messages": [
                {"role": "assistant", "content": "Wrong first role"},
                {"role": "user", "content": "Wrong second role"}
            ],
            "metadata": {
                "feature_name": "Test Feature",
                "quality_score": 8.5,
                "technical_content": True
            }
        }
        
        is_valid, metrics, errors = self.controller.validate_record(invalid_record)
        
        assert is_valid is False
        assert any("First message must be from user" in error for error in errors)
        assert any("Second message must be from assistant" in error for error in errors)
    
    def test_validate_content_too_short(self):
        """Test validation with content too short"""
        record_with_short_content = {
            "messages": [
                {"role": "user", "content": "Short"},  # Too short
                {"role": "assistant", "content": "Also very short reply"}  # Also too short
            ],
            "metadata": {
                "feature_name": "Test Feature",
                "quality_score": 8.5,
                "technical_content": True
            }
        }
        
        is_valid, metrics, errors = self.controller.validate_record(record_with_short_content)
        
        assert is_valid is False
        assert any("too short" in error for error in errors)
    
    def test_validate_content_too_long(self):
        """Test validation with content too long"""
        long_content = "x" * 5000  # Exceeds max_content_length
        
        record_with_long_content = {
            "messages": [
                {"role": "user", "content": "How do I configure eNodeB parameters?"},
                {"role": "assistant", "content": long_content}
            ],
            "metadata": {
                "feature_name": "Test Feature",
                "quality_score": 8.5,
                "technical_content": True
            }
        }
        
        is_valid, metrics, errors = self.controller.validate_record(record_with_long_content)
        
        assert is_valid is False
        assert any("too long" in error for error in errors)
    
    def test_validate_metadata_missing_required_fields(self):
        """Test metadata validation with missing required fields"""
        record_missing_metadata = {
            "messages": [
                {"role": "user", "content": "How do I configure eNodeB for LTE optimization?"},
                {"role": "assistant", "content": "Configure the RSRP threshold and time-to-trigger parameters."}
            ],
            "metadata": {
                # Missing required fields: feature_name, quality_score, technical_content
            }
        }
        
        is_valid, metrics, errors = self.controller.validate_record(record_missing_metadata)
        
        assert is_valid is False
        assert any("Missing metadata field: feature_name" in error for error in errors)
        assert any("Missing metadata field: quality_score" in error for error in errors)
        assert any("Missing metadata field: technical_content" in error for error in errors)
    
    def test_validate_metadata_invalid_quality_score(self):
        """Test metadata validation with invalid quality score"""
        record_invalid_score = {
            "messages": [
                {"role": "user", "content": "How do I configure eNodeB for LTE optimization?"},
                {"role": "assistant", "content": "Configure the RSRP threshold and time-to-trigger parameters."}
            ],
            "metadata": {
                "feature_name": "Test Feature",
                "quality_score": "invalid_score",  # Should be numeric
                "technical_content": True
            }
        }
        
        is_valid, metrics, errors = self.controller.validate_record(record_invalid_score)
        
        assert is_valid is False
        assert any("Quality score must be numeric" in error for error in errors)
    
    def test_validate_metadata_quality_score_out_of_range(self):
        """Test metadata validation with quality score out of range"""
        record_score_out_of_range = {
            "messages": [
                {"role": "user", "content": "How do I configure eNodeB for LTE optimization?"},
                {"role": "assistant", "content": "Configure the RSRP threshold and time-to-trigger parameters."}
            ],
            "metadata": {
                "feature_name": "Test Feature",
                "quality_score": 15.0,  # Out of 0-10 range
                "technical_content": True
            }
        }
        
        is_valid, metrics, errors = self.controller.validate_record(record_score_out_of_range)
        
        assert is_valid is False
        assert any("Quality score must be between 0.0 and 10.0" in error for error in errors)
    
    def test_extract_technical_terms(self):
        """Test technical term extraction"""
        content = "Configure the eNodeB for LTE handover with RSRP threshold and 5G NR parameters"
        
        terms = self.controller._extract_technical_terms(content)
        
        assert "eNodeB" in terms
        assert "LTE" in terms
        assert "RSRP" in terms
        assert "5G" in terms
        assert "NR" in terms
    
    def test_validate_technical_content_insufficient_terms(self):
        """Test technical content validation with insufficient terms"""
        record_insufficient_terms = {
            "messages": [
                {"role": "user", "content": "How do I do something?"},  # No technical terms
                {"role": "assistant", "content": "You can do it like this."}  # No technical terms
            ],
            "metadata": {
                "feature_name": "Test Feature",
                "quality_score": 8.5,
                "technical_content": True
            }
        }
        
        is_valid, metrics, errors = self.controller.validate_record(record_insufficient_terms)
        
        assert is_valid is False
        assert any("Insufficient technical terms" in error for error in errors)
    
    def test_score_content_quality_good_content(self):
        """Test content quality scoring for good content"""
        good_content = "How do I configure the eNodeB for LTE handover optimization? This involves setting RSRP thresholds."
        
        score = self.controller._score_content_quality(good_content)
        
        assert 0.8 <= score <= 1.0  # Should be high score
    
    def test_score_content_quality_poor_content(self):
        """Test content quality scoring for poor content"""
        poor_content = "short no tech terms and incomplete"  # No punctuation, no tech terms
        
        score = self.controller._score_content_quality(poor_content)
        
        assert score < 0.8  # Should be lower score
    
    def test_validate_conversation_flow_good_flow(self):
        """Test conversation flow validation for good flow"""
        record_good_flow = {
            "messages": [
                {"role": "user", "content": "How do I configure eNodeB handover parameters?"},
                {"role": "assistant", "content": "To configure eNodeB handover, set the parameters for RSRP threshold and time-to-trigger values."}
            ],
            "metadata": {
                "feature_name": "Test Feature",
                "quality_score": 8.5,
                "technical_content": True
            }
        }
        
        is_valid, metrics, errors = self.controller.validate_record(record_good_flow)
        
        assert is_valid is True
        assert metrics.conversation_flow > 0.5  # Should have good flow score
    
    def test_validate_conversation_flow_poor_flow(self):
        """Test conversation flow validation for poor flow"""
        record_poor_flow = {
            "messages": [
                {"role": "user", "content": "What is weather like today?"},
                {"role": "assistant", "content": "Configure the eNodeB parameters for LTE optimization."}  # Unrelated response
            ],
            "metadata": {
                "feature_name": "Test Feature",
                "quality_score": 8.5,
                "technical_content": True
            }
        }
        
        # This should still be valid structurally, but have lower conversation flow
        is_valid, metrics, errors = self.controller.validate_record(record_poor_flow)
        
        # May pass structural validation but have lower flow score
        assert metrics.conversation_flow < 0.8
    
    def test_calculate_composite_score(self):
        """Test composite score calculation"""
        # Test with good individual scores
        composite = self.controller._calculate_composite_score(
            content=0.9,
            metadata=0.95,
            technical=0.85,
            flow=0.88,
            terminology=0.92
        )
        
        assert 8.0 <= composite <= 10.0
        
        # Test with poor individual scores
        poor_composite = self.controller._calculate_composite_score(
            content=0.3,
            metadata=0.4,
            technical=0.2,
            flow=0.3,
            terminology=0.5
        )
        
        assert poor_composite < 5.0


class TestDeduplicationEngine:
    """Test deduplication functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.dedup_engine = DeduplicationEngine(similarity_threshold=0.85)
    
    def test_initialization(self):
        """Test deduplication engine initialization"""
        assert self.dedup_engine.similarity_threshold == 0.85
        assert len(self.dedup_engine.processed_hashes) == 0
    
    def test_generate_content_signature_same_content(self):
        """Test that same content generates same signature"""
        record1 = {
            "messages": [
                {"role": "user", "content": "How do I configure eNodeB?"},
                {"role": "assistant", "content": "Configure the parameters for optimal performance."}
            ]
        }
        
        record2 = {
            "messages": [
                {"role": "user", "content": "How do I configure eNodeB?"},
                {"role": "assistant", "content": "Configure the parameters for optimal performance."}
            ]
        }
        
        sig1 = self.dedup_engine._generate_content_signature(record1)
        sig2 = self.dedup_engine._generate_content_signature(record2)
        
        assert sig1 == sig2
        assert isinstance(sig1, str)
        assert len(sig1) == 32  # MD5 hash length
    
    def test_generate_content_signature_different_content(self):
        """Test that different content generates different signatures"""
        record1 = {
            "messages": [
                {"role": "user", "content": "How do I configure eNodeB?"},
                {"role": "assistant", "content": "Configure the parameters."}
            ]
        }
        
        record2 = {
            "messages": [
                {"role": "user", "content": "What is 5G NR?"},
                {"role": "assistant", "content": "5G NR is the new radio standard."}
            ]
        }
        
        sig1 = self.dedup_engine._generate_content_signature(record1)
        sig2 = self.dedup_engine._generate_content_signature(record2)
        
        assert sig1 != sig2
    
    def test_deduplicate_dataset_no_duplicates(self):
        """Test deduplication with no duplicates"""
        records = [
            {
                "messages": [
                    {"role": "user", "content": "Question 1?"},
                    {"role": "assistant", "content": "Answer 1."}
                ],
                "metadata": {"quality_score": 8.0}
            },
            {
                "messages": [
                    {"role": "user", "content": "Question 2?"},
                    {"role": "assistant", "content": "Answer 2."}
                ],
                "metadata": {"quality_score": 8.5}
            }
        ]
        
        unique_records = self.dedup_engine.deduplicate_dataset(records)
        
        assert len(unique_records) == 2
        assert unique_records == records
    
    def test_deduplicate_dataset_with_duplicates(self):
        """Test deduplication with duplicate content"""
        records = [
            {
                "messages": [
                    {"role": "user", "content": "How configure eNodeB?"},
                    {"role": "assistant", "content": "Configure parameters for performance."}
                ],
                "metadata": {"quality_score": 8.0}
            },
            {
                "messages": [
                    {"role": "user", "content": "How configure eNodeB?"},
                    {"role": "assistant", "content": "Configure parameters for performance."}
                ],
                "metadata": {"quality_score": 9.0}  # Higher quality
            },
            {
                "messages": [
                    {"role": "user", "content": "What is 5G?"},
                    {"role": "assistant", "content": "5G is next generation wireless."}
                ],
                "metadata": {"quality_score": 8.5}
            }
        ]
        
        unique_records = self.dedup_engine.deduplicate_dataset(records)
        
        assert len(unique_records) == 2  # One duplicate removed
        
        # Should keep the higher quality duplicate
        kept_duplicate = next(r for r in unique_records 
                            if "eNodeB" in r["messages"][0]["content"])
        assert kept_duplicate["metadata"]["quality_score"] == 9.0
    
    def test_select_best_record(self):
        """Test selection of best record from candidates"""
        candidates = [
            {
                "metadata": {
                    "quality_score": 7.5,
                    "enhancement_applied": False
                }
            },
            {
                "metadata": {
                    "quality_score": 8.0,
                    "enhancement_applied": True,  # Bonus
                    "technical_terms": ["LTE", "eNodeB", "5G", "RSRP"]  # More terms
                }
            },
            {
                "metadata": {
                    "quality_score": 8.5,
                    "enhancement_applied": False
                }
            }
        ]
        
        best_record = self.dedup_engine._select_best_record(candidates)
        
        # Should select the record with enhancement_applied and technical terms bonus
        assert best_record["metadata"]["enhancement_applied"] is True
    
    def test_score_record_quality_base_score(self):
        """Test record quality scoring with base score only"""
        record = {
            "metadata": {
                "quality_score": 8.5
            }
        }
        
        score = self.dedup_engine._score_record_quality(record)
        
        assert score == 8.5  # Just the base score
    
    def test_score_record_quality_with_bonuses(self):
        """Test record quality scoring with bonuses"""
        record = {
            "metadata": {
                "quality_score": 8.0,
                "enhancement_applied": True,  # +0.5 bonus
                "transformation_applied": True,  # +0.3 bonus
                "technical_terms": ["LTE", "eNodeB", "5G", "RSRP", "NR"]  # +0.2 bonus (>3 terms)
            }
        }
        
        score = self.dedup_engine._score_record_quality(record)
        
        assert score == 9.0  # 8.0 + 0.5 + 0.3 + 0.2


class TestValidateDatasetBatch:
    """Test batch validation functionality"""
    
    def test_validate_dataset_batch_all_valid(self):
        """Test batch validation with all valid records"""
        records = [
            {
                "messages": [
                    {"role": "user", "content": "How do I configure eNodeB for LTE handover optimization?"},
                    {"role": "assistant", "content": "Configure RSRP threshold to -110 dBm and time-to-trigger to 320ms for optimal LTE handover performance."}
                ],
                "metadata": {
                    "feature_name": "LTE Handover",
                    "quality_score": 9.0,
                    "technical_content": True
                }
            },
            {
                "messages": [
                    {"role": "user", "content": "What are the 5G NR beamforming parameters?"},
                    {"role": "assistant", "content": "5G NR beamforming uses CSI-RS for channel estimation and supports both analog and digital beamforming techniques."}
                ],
                "metadata": {
                    "feature_name": "5G Beamforming",
                    "quality_score": 8.8,
                    "technical_content": True
                }
            }
        ]
        
        controller = QualityController()
        valid_records, stats = validate_dataset_batch(records, controller)
        
        assert len(valid_records) == 2
        assert stats["total_processed"] == 2
        assert stats["valid_records"] == 2
        assert stats["invalid_records"] == 0
        assert stats["average_quality"] >= 7.9
        assert len(stats["validation_errors"]) == 0
    
    def test_validate_dataset_batch_mixed_validity(self):
        """Test batch validation with mixed valid/invalid records"""
        records = [
            {  # Valid record
                "messages": [
                    {"role": "user", "content": "How do I configure eNodeB for LTE handover optimization?"},
                    {"role": "assistant", "content": "Configure RSRP threshold to -110 dBm and time-to-trigger to 320ms."}
                ],
                "metadata": {
                    "feature_name": "LTE Handover",
                    "quality_score": 9.0,
                    "technical_content": True
                }
            },
            {  # Invalid record - missing messages
                "metadata": {
                    "feature_name": "Invalid Record",
                    "quality_score": 8.0,
                    "technical_content": True
                }
            },
            {  # Valid record
                "messages": [
                    {"role": "user", "content": "What is MIMO in 5G NR networks?"},
                    {"role": "assistant", "content": "MIMO (Multiple-Input Multiple-Output) in 5G NR uses multiple antennas with beamforming to improve throughput and reliability."}
                ],
                "metadata": {
                    "feature_name": "5G MIMO",
                    "quality_score": 8.5,
                    "technical_content": True
                }
            }
        ]
        
        controller = QualityController()
        valid_records, stats = validate_dataset_batch(records, controller)
        
        assert len(valid_records) == 2
        assert stats["total_processed"] == 3
        assert stats["valid_records"] == 2
        assert stats["invalid_records"] == 1
        assert len(stats["validation_errors"]) > 0
        assert any("Missing 'messages' field" in error for error in stats["validation_errors"])
    
    def test_validate_dataset_batch_all_invalid(self):
        """Test batch validation with all invalid records"""
        records = [
            {  # Missing messages
                "metadata": {"quality_score": 8.0}
            },
            {  # Missing metadata
                "messages": [
                    {"role": "user", "content": "Test"},
                    {"role": "assistant", "content": "Test"}
                ]
            }
        ]
        
        controller = QualityController()
        valid_records, stats = validate_dataset_batch(records, controller)
        
        assert len(valid_records) == 0
        assert stats["total_processed"] == 2
        assert stats["valid_records"] == 0
        assert stats["invalid_records"] == 2
        assert stats["average_quality"] == 0.0
        assert len(stats["validation_errors"]) > 0
    
    def test_validate_dataset_batch_empty(self):
        """Test batch validation with empty list"""
        records = []
        
        controller = QualityController()
        valid_records, stats = validate_dataset_batch(records, controller)
        
        assert len(valid_records) == 0
        assert stats["total_processed"] == 0
        assert stats["valid_records"] == 0
        assert stats["invalid_records"] == 0
        assert stats["average_quality"] == 0.0
        assert len(stats["validation_errors"]) == 0


@pytest.mark.integration
class TestQualityControllerIntegration:
    """Integration tests for quality controller"""
    
    def test_end_to_end_validation(self):
        """Test end-to-end validation workflow"""
        controller = QualityController()
        
        # Create a realistic dataset record
        record = {
            "messages": [
                {
                    "role": "user",
                    "content": "How do I configure the eNodeB handover parameters for LTE Advanced Pro networks to optimize RSRP thresholds and reduce handover failures?"
                },
                {
                    "role": "assistant", 
                    "content": "To configure eNodeB handover parameters for LTE Advanced Pro, you need to: 1) Set the RSRP threshold to -110 dBm for optimal coverage, 2) Configure time-to-trigger to 320ms to prevent ping-pong effects, 3) Set hysteresis to 2 dB for stable handover decisions, 4) Enable inter-frequency handover with proper priority settings, and 5) Configure measurement gaps for carrier aggregation scenarios."
                }
            ],
            "metadata": {
                "feature_name": "LTE Advanced Pro Handover Optimization",
                "quality_score": 9.2,
                "technical_content": True,
                "technical_terms": ["eNodeB", "LTE", "RSRP", "handover", "carrier aggregation"],
                "enhancement_applied": True,
                "transformation_applied": False
            }
        }
        
        is_valid, metrics, errors = controller.validate_record(record)
        
        assert is_valid is True
        assert isinstance(metrics, QualityMetrics)
        assert metrics.overall_score >= 8.0
        assert metrics.content_coherence > 0.8
        assert metrics.technical_accuracy >= 0.8
        assert metrics.metadata_completeness > 0.9
        assert metrics.conversation_flow > 0.7
        assert metrics.terminology_consistency >= 0.5
        assert len(errors) == 0
    
    def test_deduplication_integration(self):
        """Test deduplication integration"""
        dedup_engine = DeduplicationEngine()
        
        # Create dataset with near-duplicates
        records = [
            {
                "messages": [
                    {"role": "user", "content": "configure eNodeB handover"},
                    {"role": "assistant", "content": "set RSRP threshold parameters"}
                ],
                "metadata": {"quality_score": 8.0}
            },
            {
                "messages": [
                    {"role": "user", "content": "configure eNodeB handover"},  # Same content
                    {"role": "assistant", "content": "set RSRP threshold parameters"}  # Same content
                ],
                "metadata": {
                    "quality_score": 9.0,  # Higher quality
                    "enhancement_applied": True
                }
            },
            {
                "messages": [
                    {"role": "user", "content": "what is 5G beamforming"},
                    {"role": "assistant", "content": "5G beamforming focuses radio signals"}
                ],
                "metadata": {"quality_score": 8.5}
            }
        ]
        
        unique_records = dedup_engine.deduplicate_dataset(records)
        
        assert len(unique_records) == 2
        
        # Verify the higher quality duplicate was kept
        handover_record = next(r for r in unique_records 
                             if "handover" in r["messages"][0]["content"])
        assert handover_record["metadata"]["quality_score"] == 9.0