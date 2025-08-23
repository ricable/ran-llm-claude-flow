"""
Cross-Dataset Consistency Framework
Ensures consistency, harmonization, and integration across multiple datasets
"""

from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import json
import re
from pathlib import Path
import logging
from datetime import datetime, timezone

class ConsistencyLevel(Enum):
    """Levels of consistency checking"""
    BASIC = "basic"           # Basic structure and format
    STANDARD = "standard"     # Metadata fields and content quality
    ADVANCED = "advanced"     # Technical terminology and semantic consistency
    STRICT = "strict"         # Full cross-dataset validation

class InconsistencyType(Enum):
    """Types of inconsistencies that can be detected"""
    METADATA_SCHEMA = "metadata_schema"
    TECHNICAL_TERMS = "technical_terms"
    FEATURE_NAMING = "feature_naming"
    QUALITY_VARIANCE = "quality_variance"
    CONTENT_FORMAT = "content_format"
    PARAMETER_REFERENCES = "parameter_references"
    MO_CLASS_NAMING = "mo_class_naming"
    CONVERSATION_STRUCTURE = "conversation_structure"

@dataclass
class InconsistencyReport:
    """Report of detected inconsistencies"""
    type: InconsistencyType
    severity: str  # low, medium, high, critical
    description: str
    affected_datasets: List[str]
    affected_records: List[str]
    suggested_resolution: str
    confidence: float

@dataclass
class DatasetProfile:
    """Profile of a dataset for consistency analysis"""
    name: str
    record_count: int
    metadata_schema: Dict[str, Any]
    technical_terms: Set[str]
    feature_names: Set[str]
    parameter_names: Set[str]
    mo_classes: Set[str]
    quality_distribution: Dict[str, float]
    conversation_patterns: Dict[str, int]
    content_characteristics: Dict[str, Any]

@dataclass
class HarmonizationRule:
    """Rule for harmonizing inconsistencies"""
    rule_id: str
    name: str
    description: str
    target_field: str
    inconsistency_type: InconsistencyType
    transformation_function: str
    priority: int
    auto_apply: bool

class TechnicalTermNormalizer:
    """Normalizes technical terms across datasets"""
    
    def __init__(self):
        self.term_variations = self._initialize_term_variations()
        self.canonical_terms = self._initialize_canonical_terms()
        
    def _initialize_term_variations(self) -> Dict[str, List[str]]:
        """Initialize known variations of technical terms"""
        return {
            # Network elements
            "eNodeB": ["eNB", "enodeb", "enode-b", "e-NodeB", "E-UTRAN NodeB"],
            "gNodeB": ["gNB", "gnodeb", "gnode-b", "g-NodeB", "5G-NodeB"],
            "MME": ["mme", "Mobility Management Entity"],
            
            # Technologies
            "LTE": ["lte", "4G", "Long Term Evolution"],
            "5G": ["5g", "5G NR", "New Radio"],
            "NR": ["nr", "New Radio", "5G-NR"],
            "WCDMA": ["wcdma", "3G", "UMTS"],
            
            # Protocols
            "RRC": ["rrc", "Radio Resource Control"],
            "PDCP": ["pdcp", "Packet Data Convergence Protocol"],
            "X2": ["x2", "X2 interface"],
            "S1": ["s1", "S1 interface"],
            
            # Measurements
            "RSRP": ["rsrp", "Reference Signal Received Power"],
            "RSRQ": ["rsrq", "Reference Signal Received Quality"],
            "SINR": ["sinr", "Signal to Interference plus Noise Ratio"],
            
            # Parameters and Features
            "handoverMargin": ["handover_margin", "HO_margin", "handover-margin"],
            "FeatureState": ["feature_state", "Feature_State"],
            "EUtranCellFDD": ["EUtranCell", "E-UTRAN Cell FDD"],
        }
        
    def _initialize_canonical_terms(self) -> Dict[str, str]:
        """Map all variations to canonical forms"""
        canonical_mapping = {}
        
        for canonical, variations in self.term_variations.items():
            canonical_mapping[canonical] = canonical
            for variation in variations:
                canonical_mapping[variation.lower()] = canonical
                
        return canonical_mapping
        
    def normalize_term(self, term: str) -> str:
        """Normalize a technical term to its canonical form"""
        return self.canonical_terms.get(term.lower(), term)
        
    def normalize_term_list(self, terms: List[str]) -> List[str]:
        """Normalize a list of technical terms"""
        return [self.normalize_term(term) for term in terms]

class FeatureNameHarmonizer:
    """Harmonizes feature names across datasets"""
    
    def __init__(self):
        self.feature_patterns = self._initialize_feature_patterns()
        self.naming_rules = self._initialize_naming_rules()
        
    def _initialize_feature_patterns(self) -> Dict[str, str]:
        """Initialize patterns for feature name normalization"""
        return {
            # Pattern: canonical_name
            r".*handover.*optimization.*": "LTE Handover Optimization",
            r".*mimo.*sleep.*mode.*": "MIMO Sleep Mode",
            r".*carrier.*aggregation.*": "Carrier Aggregation",
            r".*load.*balancing.*": "Inter-Frequency Load Balancing",
            r".*mobility.*control.*": "Mobility Control",
            r".*power.*control.*": "Power Control",
            r".*prescheduling.*": "Prescheduling",
            r".*dynamic.*scheduling.*": "Dynamic Scheduling",
        }
        
    def _initialize_naming_rules(self) -> List[str]:
        """Initialize naming convention rules"""
        return [
            "Use title case for feature names",
            "Use full technology names (LTE, not lte)",
            "Use consistent abbreviations (MIMO, not Mimo)",
            "Remove unnecessary qualifiers unless they add specificity"
        ]
        
    def harmonize_feature_name(self, feature_name: str) -> str:
        """Harmonize a feature name according to conventions"""
        if not feature_name:
            return feature_name
            
        normalized = feature_name.lower().strip()
        
        # Check against patterns
        for pattern, canonical in self.feature_patterns.items():
            if re.match(pattern, normalized):
                return canonical
                
        # Apply basic normalization rules
        words = normalized.split()
        harmonized_words = []
        
        for word in words:
            if word.upper() in ["LTE", "NR", "5G", "MIMO", "CA", "SINR", "RSRP", "RSRQ"]:
                harmonized_words.append(word.upper())
            else:
                harmonized_words.append(word.capitalize())
                
        return " ".join(harmonized_words)

class ConsistencyAnalyzer:
    """Main analyzer for cross-dataset consistency"""
    
    def __init__(self, consistency_level: ConsistencyLevel = ConsistencyLevel.STANDARD):
        self.consistency_level = consistency_level
        self.term_normalizer = TechnicalTermNormalizer()
        self.feature_harmonizer = FeatureNameHarmonizer()
        self.logger = logging.getLogger(__name__)
        
    def analyze_datasets(self, datasets: Dict[str, List[Dict]]) -> Tuple[List[InconsistencyReport], Dict[str, DatasetProfile]]:
        """
        Analyze multiple datasets for consistency issues
        
        Args:
            datasets: Dictionary mapping dataset names to record lists
            
        Returns:
            Tuple of (inconsistency_reports, dataset_profiles)
        """
        
        self.logger.info(f"Analyzing {len(datasets)} datasets for consistency")
        
        # Generate profiles for each dataset
        profiles = {}
        for name, records in datasets.items():
            profiles[name] = self._generate_dataset_profile(name, records)
            
        # Detect inconsistencies
        inconsistencies = []
        
        if self.consistency_level.value in ["basic", "standard", "advanced", "strict"]:
            inconsistencies.extend(self._check_metadata_consistency(profiles))
            inconsistencies.extend(self._check_content_format_consistency(profiles))
            
        if self.consistency_level.value in ["standard", "advanced", "strict"]:
            inconsistencies.extend(self._check_quality_consistency(profiles))
            inconsistencies.extend(self._check_feature_naming_consistency(profiles))
            
        if self.consistency_level.value in ["advanced", "strict"]:
            inconsistencies.extend(self._check_technical_term_consistency(profiles))
            inconsistencies.extend(self._check_parameter_consistency(profiles))
            
        if self.consistency_level == ConsistencyLevel.STRICT:
            inconsistencies.extend(self._check_semantic_consistency(datasets, profiles))
            
        self.logger.info(f"Found {len(inconsistencies)} consistency issues")
        
        return inconsistencies, profiles
        
    def _generate_dataset_profile(self, name: str, records: List[Dict]) -> DatasetProfile:
        """Generate a comprehensive profile of a dataset"""
        
        metadata_fields = defaultdict(set)
        technical_terms = set()
        feature_names = set()
        parameter_names = set()
        mo_classes = set()
        quality_scores = []
        conversation_patterns = defaultdict(int)
        
        for record in records:
            metadata = record.get("metadata", {})
            messages = record.get("messages", [])
            
            # Collect metadata schema information
            for field, value in metadata.items():
                metadata_fields[field].add(type(value).__name__)
                
            # Extract technical terms
            terms = metadata.get("technical_terms", [])
            if isinstance(terms, list):
                technical_terms.update(term.upper() for term in terms)
                
            # Extract feature names
            feature_name = metadata.get("feature_name")
            if feature_name:
                feature_names.add(feature_name)
                
            # Extract parameters
            params = metadata.get("parameters_involved", [])
            if isinstance(params, list):
                parameter_names.update(params)
                
            # Extract MO classes
            mo_list = metadata.get("mo_classes", [])
            if isinstance(mo_list, list):
                mo_classes.update(mo_list)
                
            # Collect quality scores
            quality = metadata.get("quality_score")
            if quality is not None:
                try:
                    quality_scores.append(float(quality))
                except (ValueError, TypeError):
                    pass
                    
            # Analyze conversation patterns
            conversation_patterns[len(messages)] += 1
            
            for message in messages:
                role = message.get("role", "unknown")
                conversation_patterns[f"role_{role}"] += 1
                
        # Calculate quality distribution
        quality_distribution = {}
        if quality_scores:
            quality_distribution = {
                "mean": sum(quality_scores) / len(quality_scores),
                "min": min(quality_scores),
                "max": max(quality_scores),
                "count": len(quality_scores)
            }
            
        # Convert metadata fields to regular dict
        metadata_schema = {field: list(types) for field, types in metadata_fields.items()}
        
        # Calculate content characteristics
        content_chars = self._analyze_content_characteristics(records)
        
        return DatasetProfile(
            name=name,
            record_count=len(records),
            metadata_schema=metadata_schema,
            technical_terms=technical_terms,
            feature_names=feature_names,
            parameter_names=parameter_names,
            mo_classes=mo_classes,
            quality_distribution=quality_distribution,
            conversation_patterns=dict(conversation_patterns),
            content_characteristics=content_chars
        )
        
    def _analyze_content_characteristics(self, records: List[Dict]) -> Dict[str, Any]:
        """Analyze content characteristics of records"""
        
        total_length = 0
        question_types = Counter()
        content_patterns = Counter()
        
        for record in records:
            messages = record.get("messages", [])
            for message in messages:
                content = message.get("content", "")
                total_length += len(content)
                
                if message.get("role") == "user":
                    # Classify question types
                    content_lower = content.lower()
                    if any(word in content_lower for word in ["how", "configure", "set"]):
                        question_types["configuration"] += 1
                    elif any(word in content_lower for word in ["what", "explain", "describe"]):
                        question_types["explanation"] += 1
                    elif any(word in content_lower for word in ["troubleshoot", "diagnose", "fix"]):
                        question_types["troubleshooting"] += 1
                    elif any(word in content_lower for word in ["counter", "kpi", "measurement"]):
                        question_types["monitoring"] += 1
                        
                # Check for code/configuration patterns
                if "cmedit" in content:
                    content_patterns["cmedit_commands"] += 1
                if re.search(r'\w+\.\w+', content):
                    content_patterns["parameter_references"] += 1
                if "pm" in content.lower():
                    content_patterns["pm_counters"] += 1
                    
        avg_length = total_length / max(1, len(records))
        
        return {
            "average_content_length": avg_length,
            "question_type_distribution": dict(question_types),
            "content_pattern_distribution": dict(content_patterns)
        }
        
    def _check_metadata_consistency(self, profiles: Dict[str, DatasetProfile]) -> List[InconsistencyReport]:
        """Check for metadata schema inconsistencies"""
        
        inconsistencies = []
        
        # Compare metadata schemas across datasets
        all_fields = set()
        for profile in profiles.values():
            all_fields.update(profile.metadata_schema.keys())
            
        # Check for missing fields
        for field in all_fields:
            missing_datasets = []
            for name, profile in profiles.items():
                if field not in profile.metadata_schema:
                    missing_datasets.append(name)
                    
            if missing_datasets and len(missing_datasets) < len(profiles):
                inconsistencies.append(InconsistencyReport(
                    type=InconsistencyType.METADATA_SCHEMA,
                    severity="medium",
                    description=f"Field '{field}' missing from datasets: {', '.join(missing_datasets)}",
                    affected_datasets=missing_datasets,
                    affected_records=[],
                    suggested_resolution=f"Add '{field}' field to affected datasets or make it optional",
                    confidence=0.9
                ))
                
        # Check for field type inconsistencies
        for field in all_fields:
            field_types = {}
            for name, profile in profiles.items():
                if field in profile.metadata_schema:
                    field_types[name] = profile.metadata_schema[field]
                    
            if len(set(str(types) for types in field_types.values())) > 1:
                inconsistencies.append(InconsistencyReport(
                    type=InconsistencyType.METADATA_SCHEMA,
                    severity="high",
                    description=f"Field '{field}' has inconsistent types across datasets: {field_types}",
                    affected_datasets=list(field_types.keys()),
                    affected_records=[],
                    suggested_resolution=f"Standardize data type for field '{field}' across all datasets",
                    confidence=0.95
                ))
                
        return inconsistencies
        
    def _check_technical_term_consistency(self, profiles: Dict[str, DatasetProfile]) -> List[InconsistencyReport]:
        """Check for technical term inconsistencies"""
        
        inconsistencies = []
        
        # Collect all terms and their variations
        term_usage = defaultdict(set)
        
        for name, profile in profiles.items():
            for term in profile.technical_terms:
                canonical = self.term_normalizer.normalize_term(term)
                term_usage[canonical].add((term, name))
                
        # Find terms with multiple variations
        for canonical_term, variations in term_usage.items():
            if len(set(var[0] for var in variations)) > 1:
                variation_list = list(set(var[0] for var in variations))
                affected_datasets = list(set(var[1] for var in variations))
                
                inconsistencies.append(InconsistencyReport(
                    type=InconsistencyType.TECHNICAL_TERMS,
                    severity="medium",
                    description=f"Term '{canonical_term}' has variations: {variation_list}",
                    affected_datasets=affected_datasets,
                    affected_records=[],
                    suggested_resolution=f"Standardize to canonical form: '{canonical_term}'",
                    confidence=0.8
                ))
                
        return inconsistencies
        
    def _check_feature_naming_consistency(self, profiles: Dict[str, DatasetProfile]) -> List[InconsistencyReport]:
        """Check for feature naming inconsistencies"""
        
        inconsistencies = []
        
        # Collect all feature names and group similar ones
        feature_groups = defaultdict(list)
        
        for name, profile in profiles.items():
            for feature_name in profile.feature_names:
                normalized = self.feature_harmonizer.harmonize_feature_name(feature_name)
                feature_groups[normalized].append((feature_name, name))
                
        # Find feature names that should be harmonized
        for canonical_name, variations in feature_groups.items():
            if len(set(var[0] for var in variations)) > 1:
                variation_names = list(set(var[0] for var in variations))
                affected_datasets = list(set(var[1] for var in variations))
                
                inconsistencies.append(InconsistencyReport(
                    type=InconsistencyType.FEATURE_NAMING,
                    severity="low",
                    description=f"Feature name variations found: {variation_names}",
                    affected_datasets=affected_datasets,
                    affected_records=[],
                    suggested_resolution=f"Standardize to: '{canonical_name}'",
                    confidence=0.7
                ))
                
        return inconsistencies
        
    def _check_quality_consistency(self, profiles: Dict[str, DatasetProfile]) -> List[InconsistencyReport]:
        """Check for quality score inconsistencies"""
        
        inconsistencies = []
        
        # Analyze quality distributions
        quality_means = {}
        for name, profile in profiles.items():
            if profile.quality_distribution and "mean" in profile.quality_distribution:
                quality_means[name] = profile.quality_distribution["mean"]
                
        if len(quality_means) > 1:
            mean_values = list(quality_means.values())
            overall_mean = sum(mean_values) / len(mean_values)
            
            # Check for significant deviations
            for name, mean_quality in quality_means.items():
                deviation = abs(mean_quality - overall_mean)
                if deviation > 1.5:  # Significant deviation
                    severity = "high" if deviation > 2.5 else "medium"
                    
                    inconsistencies.append(InconsistencyReport(
                        type=InconsistencyType.QUALITY_VARIANCE,
                        severity=severity,
                        description=f"Dataset '{name}' has quality mean {mean_quality:.2f}, significantly different from overall mean {overall_mean:.2f}",
                        affected_datasets=[name],
                        affected_records=[],
                        suggested_resolution="Review quality scoring criteria and normalize if necessary",
                        confidence=0.8
                    ))
                    
        return inconsistencies
        
    def _check_content_format_consistency(self, profiles: Dict[str, DatasetProfile]) -> List[InconsistencyReport]:
        """Check for content format inconsistencies"""
        
        inconsistencies = []
        
        # Check conversation structure patterns
        conversation_structures = {}
        for name, profile in profiles.items():
            structures = profile.conversation_patterns
            conversation_structures[name] = structures
            
        # Find datasets with unusual conversation patterns
        all_patterns = set()
        for patterns in conversation_structures.values():
            all_patterns.update(patterns.keys())
            
        for pattern in all_patterns:
            pattern_counts = {}
            for name, patterns in conversation_structures.items():
                pattern_counts[name] = patterns.get(pattern, 0)
                
            # Check if some datasets are missing common patterns
            non_zero_counts = [count for count in pattern_counts.values() if count > 0]
            if len(non_zero_counts) < len(pattern_counts) and len(non_zero_counts) > 1:
                missing_datasets = [name for name, count in pattern_counts.items() if count == 0]
                
                inconsistencies.append(InconsistencyReport(
                    type=InconsistencyType.CONVERSATION_STRUCTURE,
                    severity="low",
                    description=f"Pattern '{pattern}' missing from datasets: {missing_datasets}",
                    affected_datasets=missing_datasets,
                    affected_records=[],
                    suggested_resolution="Review conversation structure consistency",
                    confidence=0.6
                ))
                
        return inconsistencies
        
    def _check_parameter_consistency(self, profiles: Dict[str, DatasetProfile]) -> List[InconsistencyReport]:
        """Check for parameter reference inconsistencies"""
        
        inconsistencies = []
        
        # Collect all parameter references
        all_parameters = set()
        for profile in profiles.values():
            all_parameters.update(profile.parameter_names)
            
        # Check for naming variations of the same parameter
        parameter_groups = defaultdict(list)
        for param in all_parameters:
            # Group parameters that might be the same
            base_name = param.split('.')[-1] if '.' in param else param
            parameter_groups[base_name.lower()].append(param)
            
        # Find potential inconsistencies
        for base_name, param_list in parameter_groups.items():
            if len(param_list) > 1:
                # Check if these are actually different variations
                unique_forms = set(param_list)
                if len(unique_forms) > 1:
                    # Find which datasets use which variations
                    dataset_usage = defaultdict(list)
                    for name, profile in profiles.items():
                        for param in param_list:
                            if param in profile.parameter_names:
                                dataset_usage[param].append(name)
                                
                    if len(dataset_usage) > 1:
                        inconsistencies.append(InconsistencyReport(
                            type=InconsistencyType.PARAMETER_REFERENCES,
                            severity="medium",
                            description=f"Parameter variations found for '{base_name}': {list(unique_forms)}",
                            affected_datasets=list(set(ds for ds_list in dataset_usage.values() for ds in ds_list)),
                            affected_records=[],
                            suggested_resolution="Standardize parameter naming conventions",
                            confidence=0.7
                        ))
                        
        return inconsistencies
        
    def _check_semantic_consistency(self, datasets: Dict[str, List[Dict]], 
                                  profiles: Dict[str, DatasetProfile]) -> List[InconsistencyReport]:
        """Check for semantic inconsistencies (advanced analysis)"""
        
        inconsistencies = []
        
        # This would involve more complex semantic analysis
        # For now, implement basic semantic checks
        
        # Check for conflicting information about the same features
        feature_content_map = defaultdict(list)
        
        for dataset_name, records in datasets.items():
            for record in records:
                metadata = record.get("metadata", {})
                feature_name = metadata.get("feature_name")
                if feature_name:
                    messages = record.get("messages", [])
                    content = " ".join([msg.get("content", "") for msg in messages])
                    feature_content_map[feature_name].append((dataset_name, content[:200]))
                    
        # Basic check for significantly different content for same feature
        for feature_name, content_list in feature_content_map.items():
            if len(content_list) > 1:
                datasets_with_feature = set(item[0] for item in content_list)
                if len(datasets_with_feature) > 1:
                    # This is a simplified check - in practice, would use more sophisticated NLP
                    inconsistencies.append(InconsistencyReport(
                        type=InconsistencyType.TECHNICAL_TERMS,
                        severity="low",
                        description=f"Feature '{feature_name}' appears in multiple datasets with potentially different descriptions",
                        affected_datasets=list(datasets_with_feature),
                        affected_records=[],
                        suggested_resolution="Review and harmonize feature descriptions across datasets",
                        confidence=0.5
                    ))
                    
        return inconsistencies

class DatasetHarmonizer:
    """Harmonizes datasets based on consistency analysis"""
    
    def __init__(self):
        self.term_normalizer = TechnicalTermNormalizer()
        self.feature_harmonizer = FeatureNameHarmonizer()
        self.harmonization_rules = self._initialize_harmonization_rules()
        
    def _initialize_harmonization_rules(self) -> List[HarmonizationRule]:
        """Initialize harmonization rules"""
        return [
            HarmonizationRule(
                rule_id="normalize_technical_terms",
                name="Normalize Technical Terms",
                description="Normalize technical terms to canonical forms",
                target_field="technical_terms",
                inconsistency_type=InconsistencyType.TECHNICAL_TERMS,
                transformation_function="normalize_terms",
                priority=1,
                auto_apply=True
            ),
            HarmonizationRule(
                rule_id="harmonize_feature_names",
                name="Harmonize Feature Names", 
                description="Standardize feature naming conventions",
                target_field="feature_name",
                inconsistency_type=InconsistencyType.FEATURE_NAMING,
                transformation_function="harmonize_feature_name",
                priority=2,
                auto_apply=True
            ),
            HarmonizationRule(
                rule_id="standardize_metadata_schema",
                name="Standardize Metadata Schema",
                description="Ensure consistent metadata fields across datasets",
                target_field="metadata",
                inconsistency_type=InconsistencyType.METADATA_SCHEMA,
                transformation_function="standardize_metadata",
                priority=3,
                auto_apply=False
            )
        ]
        
    def harmonize_datasets(self, datasets: Dict[str, List[Dict]], 
                          inconsistency_reports: List[InconsistencyReport]) -> Dict[str, List[Dict]]:
        """Apply harmonization rules to resolve inconsistencies"""
        
        harmonized_datasets = {}
        
        for dataset_name, records in datasets.items():
            harmonized_records = []
            
            for record in records:
                harmonized_record = self._harmonize_record(record, inconsistency_reports)
                harmonized_records.append(harmonized_record)
                
            harmonized_datasets[dataset_name] = harmonized_records
            
        return harmonized_datasets
        
    def _harmonize_record(self, record: Dict, inconsistency_reports: List[InconsistencyReport]) -> Dict:
        """Harmonize a single record"""
        
        harmonized_record = record.copy()
        metadata = harmonized_record.get("metadata", {}).copy()
        
        # Apply automatic harmonization rules
        for rule in self.harmonization_rules:
            if rule.auto_apply:
                if rule.transformation_function == "normalize_terms":
                    technical_terms = metadata.get("technical_terms", [])
                    if isinstance(technical_terms, list):
                        metadata["technical_terms"] = self.term_normalizer.normalize_term_list(technical_terms)
                        
                elif rule.transformation_function == "harmonize_feature_name":
                    feature_name = metadata.get("feature_name")
                    if feature_name:
                        metadata["feature_name"] = self.feature_harmonizer.harmonize_feature_name(feature_name)
                        
        harmonized_record["metadata"] = metadata
        
        return harmonized_record

def generate_consistency_report(inconsistency_reports: List[InconsistencyReport], 
                               profiles: Dict[str, DatasetProfile]) -> Dict[str, Any]:
    """Generate comprehensive consistency analysis report"""
    
    # Summary statistics
    severity_counts = Counter(report.severity for report in inconsistency_reports)
    type_counts = Counter(report.type.value for report in inconsistency_reports)
    
    # Affected datasets analysis
    affected_datasets = defaultdict(int)
    for report in inconsistency_reports:
        for dataset in report.affected_datasets:
            affected_datasets[dataset] += 1
            
    # Dataset comparison summary
    dataset_comparison = {}
    for name, profile in profiles.items():
        dataset_comparison[name] = {
            "record_count": profile.record_count,
            "unique_features": len(profile.feature_names),
            "unique_technical_terms": len(profile.technical_terms),
            "unique_parameters": len(profile.parameter_names),
            "average_quality": profile.quality_distribution.get("mean", 0) if profile.quality_distribution else 0
        }
    
    # Recommendations
    recommendations = []
    
    if severity_counts.get("high", 0) > 0:
        recommendations.append("Address high-severity inconsistencies immediately to ensure data quality")
        
    if type_counts.get("metadata_schema", 0) > 0:
        recommendations.append("Standardize metadata schema across all datasets")
        
    if type_counts.get("technical_terms", 0) > 0:
        recommendations.append("Implement technical term normalization")
        
    if type_counts.get("feature_naming", 0) > 0:
        recommendations.append("Establish and enforce feature naming conventions")
        
    report = {
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_inconsistencies": len(inconsistency_reports),
        "severity_breakdown": dict(severity_counts),
        "inconsistency_type_breakdown": dict(type_counts),
        "most_affected_datasets": dict(sorted(affected_datasets.items(), 
                                            key=lambda x: x[1], reverse=True)[:5]),
        "dataset_comparison": dataset_comparison,
        "detailed_reports": [
            {
                "type": report.type.value,
                "severity": report.severity,
                "description": report.description,
                "affected_datasets": report.affected_datasets,
                "resolution": report.suggested_resolution,
                "confidence": report.confidence
            }
            for report in inconsistency_reports
        ],
        "recommendations": recommendations
    }
    
    return report

# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ConsistencyAnalyzer(ConsistencyLevel.ADVANCED)
    
    # Sample datasets for testing
    sample_datasets = {
        "dataset_1": [
            {
                "messages": [
                    {"role": "user", "content": "How to configure LTE handover?"},
                    {"role": "assistant", "content": "Configure handoverMargin parameter."}
                ],
                "metadata": {
                    "feature_name": "LTE Handover Optimization",
                    "technical_terms": ["LTE", "handoverMargin"],
                    "quality_score": 9.0
                }
            }
        ],
        "dataset_2": [
            {
                "messages": [
                    {"role": "user", "content": "How to setup lte handover?"},
                    {"role": "assistant", "content": "Set handover_margin parameter."}
                ],
                "metadata": {
                    "feature_name": "lte handover optimization",
                    "technical_terms": ["lte", "handover_margin"],
                    "quality_score": 8.5
                }
            }
        ]
    }
    
    # Analyze consistency
    inconsistencies, profiles = analyzer.analyze_datasets(sample_datasets)
    
    # Generate report
    report = generate_consistency_report(inconsistencies, profiles)
    
    print("Consistency Analysis Results:")
    print(json.dumps(report, indent=2, default=str))