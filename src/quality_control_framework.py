"""
Quality Control and Validation Framework for LLM Dataset Processing
Comprehensive validation, deduplication, and quality assurance system
"""

import json
import hashlib
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path
import logging

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for dataset records"""
    content_coherence: float  # 0.0 - 1.0
    technical_accuracy: float  # 0.0 - 1.0
    metadata_completeness: float  # 0.0 - 1.0
    conversation_flow: float  # 0.0 - 1.0
    terminology_consistency: float  # 0.0 - 1.0
    overall_score: float  # Weighted composite score

class QualityController:
    """Main quality control engine for dataset processing"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.technical_terms_db = self._load_technical_terms()
        self.logger = logging.getLogger(__name__)
        
    def _default_config(self) -> Dict:
        return {
            "min_quality_score": 8.0,
            "min_content_length": 50,
            "max_content_length": 4096,
            "required_metadata_fields": [
                "feature_name", "quality_score", "technical_content"
            ],
            "technical_term_threshold": 3,  # Minimum technical terms per record
            "similarity_threshold": 0.85,  # For deduplication
            "confidence_threshold": 0.7
        }
    
    def _load_technical_terms(self) -> Set[str]:
        """Load Ericsson RAN technical terminology database"""
        return {
            # Radio Access Network Terms
            "eNodeB", "gNodeB", "RBS", "RNC", "BSC", "MSC", "MME", "SGW", "PGW",
            "LTE", "NR", "5G", "WCDMA", "GSM", "UMTS", "EN-DC", "NSA", "SA",
            
            # Protocol and Interface Terms  
            "X2", "S1", "S1-U", "S1-MME", "Xn", "F1", "E1", "RRC", "PDCP", "RLC", "MAC",
            "PHY", "SCTP", "GTP-U", "SIP", "IMS", "VoLTE", "VoNR", "CSFB",
            
            # Performance and Measurement
            "RSRP", "RSRQ", "RSSI", "SINR", "CQI", "RI", "PMI", "UE", "PRB", "MIMO",
            "CA", "CoMP", "SON", "ANR", "MLB", "MRO", "CCO", "PCI",
            
            # Quality and Configuration
            "KPI", "PM", "QoS", "QCI", "5QI", "ARP", "GBR", "MBR", "AMBR", "PDB", "PER",
            "BLER", "CRC", "HARQ", "ARQ", "TTI", "OFDM", "SC-FDMA", "BWP", "SCS"
        }

    def validate_record(self, record: Dict) -> Tuple[bool, QualityMetrics, List[str]]:
        """
        Comprehensive validation of a single dataset record
        
        Returns:
            (is_valid, quality_metrics, validation_errors)
        """
        errors = []
        
        # Structure validation
        if not self._validate_structure(record, errors):
            return False, QualityMetrics(0,0,0,0,0,0), errors
            
        # Content validation
        content_score = self._validate_content(record, errors)
        
        # Metadata validation  
        metadata_score = self._validate_metadata(record, errors)
        
        # Technical content validation
        tech_score = self._validate_technical_content(record, errors)
        
        # Conversation flow validation
        flow_score = self._validate_conversation_flow(record, errors)
        
        # Terminology consistency
        term_score = self._validate_terminology(record, errors)
        
        # Calculate composite quality score
        overall_score = self._calculate_composite_score(
            content_score, metadata_score, tech_score, flow_score, term_score
        )
        
        metrics = QualityMetrics(
            content_coherence=content_score,
            technical_accuracy=tech_score, 
            metadata_completeness=metadata_score,
            conversation_flow=flow_score,
            terminology_consistency=term_score,
            overall_score=overall_score
        )
        
        is_valid = overall_score >= self.config["min_quality_score"] and len(errors) == 0
        
        return is_valid, metrics, errors

    def _validate_structure(self, record: Dict, errors: List[str]) -> bool:
        """Validate basic record structure"""
        if "messages" not in record:
            errors.append("Missing 'messages' field")
            return False
            
        if "metadata" not in record:
            errors.append("Missing 'metadata' field")
            return False
            
        messages = record.get("messages", [])
        if not isinstance(messages, list) or len(messages) < 2:
            errors.append("Invalid messages structure - need at least 2 messages")
            return False
            
        # Validate message roles
        if messages[0].get("role") != "user":
            errors.append("First message must be from user")
            
        if messages[1].get("role") != "assistant":  
            errors.append("Second message must be from assistant")
            
        return len(errors) == 0

    def _validate_content(self, record: Dict, errors: List[str]) -> float:
        """Validate content quality and coherence"""
        messages = record.get("messages", [])
        scores = []
        
        for i, message in enumerate(messages):
            content = message.get("content", "")
            
            # Length validation
            if len(content) < self.config["min_content_length"]:
                errors.append(f"Message {i} too short: {len(content)} chars")
                scores.append(0.3)
                continue
                
            if len(content) > self.config["max_content_length"]:
                errors.append(f"Message {i} too long: {len(content)} chars")
                scores.append(0.7)
                continue
                
            # Content quality scoring
            score = self._score_content_quality(content)
            scores.append(score)
            
        return np.mean(scores) if scores else 0.0

    def _score_content_quality(self, content: str) -> float:
        """Score individual content quality"""
        score = 1.0
        
        # Check for incomplete sentences
        if content.count('.') == 0 and len(content) > 50:
            score -= 0.1
            
        # Check for technical coherence
        tech_terms = self._extract_technical_terms(content)
        if len(tech_terms) < 2 and len(content) > 100:
            score -= 0.2
            
        # Check for question-answer coherence (if applicable)
        if "?" in content and not any(indicator in content.lower() 
                                    for indicator in ["how", "what", "why", "when", "where"]):
            score -= 0.1
            
        return max(0.0, score)

    def _validate_metadata(self, record: Dict, errors: List[str]) -> float:
        """Validate metadata completeness and consistency"""
        metadata = record.get("metadata", {})
        required_fields = self.config["required_metadata_fields"]
        
        missing_fields = []
        for field in required_fields:
            if field not in metadata:
                missing_fields.append(field)
                
        if missing_fields:
            errors.extend([f"Missing metadata field: {field}" for field in missing_fields])
            
        # Calculate completeness score
        completeness = 1.0 - (len(missing_fields) / len(required_fields))
        
        # Validate field values
        quality_score = metadata.get("quality_score")
        if quality_score is not None:
            try:
                score_val = float(quality_score)
                if not 0.0 <= score_val <= 10.0:
                    errors.append("Quality score must be between 0.0 and 10.0")
            except (ValueError, TypeError):
                errors.append("Quality score must be numeric")
                
        return completeness

    def _validate_technical_content(self, record: Dict, errors: List[str]) -> float:
        """Validate technical accuracy and terminology usage"""
        messages = record.get("messages", [])
        all_content = " ".join([msg.get("content", "") for msg in messages])
        
        # Extract technical terms
        extracted_terms = self._extract_technical_terms(all_content)
        
        if len(extracted_terms) < self.config["technical_term_threshold"]:
            errors.append(f"Insufficient technical terms: {len(extracted_terms)}")
            
        # Validate term usage context
        term_score = self._score_technical_usage(all_content, extracted_terms)
        
        return term_score

    def _extract_technical_terms(self, content: str) -> Set[str]:
        """Extract technical terms from content"""
        words = re.findall(r'\b[A-Za-z0-9-]+\b', content.upper())
        return {word for word in words if word in self.technical_terms_db}

    def _score_technical_usage(self, content: str, terms: Set[str]) -> float:
        """Score technical term usage appropriateness"""
        if not terms:
            return 0.0
            
        # Check for proper technical context
        context_indicators = [
            "configure", "parameter", "feature", "enable", "disable",
            "measurement", "counter", "KPI", "threshold", "optimization"
        ]
        
        has_context = any(indicator in content.lower() for indicator in context_indicators)
        
        base_score = min(1.0, len(terms) / 5.0)  # Scale based on term count
        context_bonus = 0.2 if has_context else 0.0
        
        return min(1.0, base_score + context_bonus)

    def _validate_conversation_flow(self, record: Dict, errors: List[str]) -> float:
        """Validate logical flow of conversation"""
        messages = record.get("messages", [])
        if len(messages) < 2:
            return 0.0
            
        user_msg = messages[0].get("content", "").lower()
        assistant_msg = messages[1].get("content", "").lower()
        
        # Check if assistant response addresses user question
        user_terms = set(re.findall(r'\b\w+\b', user_msg))
        assistant_terms = set(re.findall(r'\b\w+\b', assistant_msg))
        
        # Calculate term overlap (indication of topical coherence)
        if user_terms:
            overlap = len(user_terms & assistant_terms) / len(user_terms)
        else:
            overlap = 0.0
            
        # Additional flow indicators
        flow_score = overlap
        
        # Bonus for proper question-answer pattern
        if "?" in user_msg and len(assistant_msg) > len(user_msg) * 0.5:
            flow_score += 0.2
            
        return min(1.0, flow_score)

    def _validate_terminology(self, record: Dict, errors: List[str]) -> float:
        """Validate terminology consistency across messages"""
        messages = record.get("messages", [])
        all_terms = []
        
        for msg in messages:
            content = msg.get("content", "")
            terms = self._extract_technical_terms(content)
            all_terms.extend(terms)
            
        if not all_terms:
            return 0.5  # Neutral score for non-technical content
            
        # Check for consistent terminology usage
        term_counts = Counter(all_terms)
        consistency_score = len(set(all_terms)) / len(all_terms) if all_terms else 0.0
        
        return consistency_score

    def _calculate_composite_score(self, content: float, metadata: float, 
                                 technical: float, flow: float, terminology: float) -> float:
        """Calculate weighted composite quality score"""
        weights = {
            "content": 0.25,
            "metadata": 0.15, 
            "technical": 0.30,
            "flow": 0.20,
            "terminology": 0.10
        }
        
        composite = (
            weights["content"] * content +
            weights["metadata"] * metadata + 
            weights["technical"] * technical +
            weights["flow"] * flow +
            weights["terminology"] * terminology
        )
        
        return composite * 10.0  # Scale to 0-10 range

class DeduplicationEngine:
    """Advanced deduplication system for dataset records"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.processed_hashes = set()
        
    def deduplicate_dataset(self, records: List[Dict]) -> List[Dict]:
        """Remove duplicate records while preserving highest quality versions"""
        unique_records = []
        content_signatures = defaultdict(list)
        
        # Group records by content similarity
        for i, record in enumerate(records):
            signature = self._generate_content_signature(record)
            content_signatures[signature].append((i, record))
            
        # Select best record from each group
        for signature_group in content_signatures.values():
            if len(signature_group) == 1:
                unique_records.append(signature_group[0][1])
            else:
                # Select highest quality record
                best_record = self._select_best_record([r[1] for r in signature_group])
                unique_records.append(best_record)
                
        return unique_records
        
    def _generate_content_signature(self, record: Dict) -> str:
        """Generate content-based signature for similarity detection"""
        messages = record.get("messages", [])
        content_parts = []
        
        for msg in messages:
            content = msg.get("content", "").lower()
            # Normalize content for comparison
            normalized = re.sub(r'\s+', ' ', content.strip())
            # Extract key phrases (words > 3 chars)
            key_words = [w for w in normalized.split() if len(w) > 3]
            content_parts.extend(key_words[:20])  # Limit to prevent huge signatures
            
        # Create hash from normalized content
        signature_text = " ".join(sorted(set(content_parts)))
        return hashlib.md5(signature_text.encode()).hexdigest()
        
    def _select_best_record(self, candidate_records: List[Dict]) -> Dict:
        """Select the best record from a group of similar records"""
        best_record = candidate_records[0]
        best_score = 0.0
        
        for record in candidate_records:
            # Score based on multiple factors
            score = self._score_record_quality(record)
            if score > best_score:
                best_score = score
                best_record = record
                
        return best_record
        
    def _score_record_quality(self, record: Dict) -> float:
        """Score record quality for deduplication selection"""
        metadata = record.get("metadata", {})
        
        # Base score from existing quality_score
        base_score = float(metadata.get("quality_score", 5.0))
        
        # Bonus factors
        bonus = 0.0
        
        # Prefer records with more complete metadata
        if "enhancement_applied" in metadata and metadata["enhancement_applied"]:
            bonus += 0.5
            
        if "transformation_applied" in metadata and metadata["transformation_applied"]:
            bonus += 0.3
            
        # Prefer records with technical terms
        technical_terms = metadata.get("technical_terms", [])
        if technical_terms and len(technical_terms) > 3:
            bonus += 0.2
            
        return base_score + bonus

def validate_dataset_batch(records: List[Dict], 
                         quality_controller: QualityController) -> Tuple[List[Dict], Dict]:
    """
    Process a batch of records through quality validation
    
    Returns:
        (valid_records, processing_stats)
    """
    valid_records = []
    stats = {
        "total_processed": len(records),
        "valid_records": 0,
        "invalid_records": 0,
        "average_quality": 0.0,
        "validation_errors": []
    }
    
    quality_scores = []
    
    for i, record in enumerate(records):
        is_valid, metrics, errors = quality_controller.validate_record(record)
        
        if is_valid:
            valid_records.append(record)
            stats["valid_records"] += 1
            quality_scores.append(metrics.overall_score)
        else:
            stats["invalid_records"] += 1
            stats["validation_errors"].extend([f"Record {i}: {err}" for err in errors])
            
    if quality_scores:
        stats["average_quality"] = np.mean(quality_scores)
        
    return valid_records, stats

# Example usage and testing
if __name__ == "__main__":
    # Initialize quality controller
    quality_controller = QualityController()
    
    # Sample record for testing
    test_record = {
        "messages": [
            {
                "role": "user",
                "content": "How do I configure the eNodeB for LTE handover optimization?"
            },
            {
                "role": "assistant", 
                "content": "To configure LTE handover optimization, set the handover parameters in the RRC configuration. Adjust the RSRP threshold to -110 dBm and configure the time-to-trigger to 320ms for optimal performance."
            }
        ],
        "metadata": {
            "feature_name": "LTE Handover Optimization",
            "quality_score": 9.5,
            "technical_content": True,
            "technical_terms": ["eNodeB", "LTE", "RRC", "RSRP", "handover"]
        }
    }
    
    # Validate the test record
    is_valid, metrics, errors = quality_controller.validate_record(test_record)
    
    print(f"Record Valid: {is_valid}")
    print(f"Quality Metrics: {metrics}")
    if errors:
        print(f"Errors: {errors}")