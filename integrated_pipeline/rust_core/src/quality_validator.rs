use crate::config::QualitySettings;
use crate::types::*;

use anyhow::Result;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};

/// Quality validator for document structural assessment
pub struct QualityValidator {
    config: QualitySettings,
    parameter_patterns: Vec<Regex>,
    counter_patterns: Vec<Regex>,
    technical_term_dictionary: TechnicalTermDictionary,
    completeness_assessor: CompletenessAssessor,
}

/// Technical term dictionary with tiered classification
pub struct TechnicalTermDictionary {
    pub tier1_exact_params: HashSet<String>,      // Exact parameter names
    pub tier2_config_terms: HashSet<String>,      // Configuration terms
    pub tier3_network_elements: HashSet<String>,  // Network elements
    pub tier4_rare_terms: HashSet<String>,        // Rare technical terms
    pub term_weights: HashMap<String, f64>,       // Term importance weights
}

/// Document completeness assessor
pub struct CompletenessAssessor {
    required_sections: Vec<String>,
    optional_sections: Vec<String>,
    section_patterns: HashMap<String, Regex>,
    metadata_fields: Vec<String>,
}

/// Structural quality metrics
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub parameter_extraction_score: f64,
    pub counter_extraction_score: f64,
    pub technical_density_score: f64,
    pub completeness_score: f64,
    pub content_coherence_score: f64,
    pub metadata_quality_score: f64,
}

/// Quality assessment results with detailed breakdown
#[derive(Debug, Clone)]
pub struct DetailedQualityAssessment {
    pub overall_score: f64,
    pub metrics: QualityMetrics,
    pub extracted_elements: ExtractedElements,
    pub quality_issues: Vec<QualityIssue>,
    pub recommendations: Vec<QualityRecommendation>,
}

/// Elements extracted during quality assessment
#[derive(Debug, Clone)]
pub struct ExtractedElements {
    pub parameters_found: usize,
    pub counters_found: usize,
    pub technical_terms_found: usize,
    pub sections_identified: Vec<String>,
    pub metadata_fields_populated: usize,
}

/// Quality issues identified during assessment
#[derive(Debug, Clone)]
pub struct QualityIssue {
    pub issue_type: QualityIssueType,
    pub severity: IssueSeverity,
    pub description: String,
    pub location: Option<String>,
}

/// Types of quality issues
#[derive(Debug, Clone)]
pub enum QualityIssueType {
    MissingParameters,
    MissingCounters,
    LowTechnicalDensity,
    IncompleteMetadata,
    PoorStructure,
    ContentTooShort,
    ContentTooLong,
}

/// Severity levels for quality issues
#[derive(Debug, Clone)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Quality improvement recommendations
#[derive(Debug, Clone)]
pub struct QualityRecommendation {
    pub recommendation_type: RecommendationType,
    pub description: String,
    pub expected_improvement: f64,
}

/// Types of quality recommendations
#[derive(Debug, Clone)]
pub enum RecommendationType {
    AddMoreParameters,
    ImproveStructure,
    EnhanceMetadata,
    BalanceContent,
    IncreaseTechnicalDepth,
}

impl QualityValidator {
    /// Create new quality validator with configuration
    pub fn new(config: &QualitySettings) -> Result<Self> {
        info!("Initializing Quality Validator");
        info!("Quality threshold: {:.2}", config.quality_threshold);
        info!("Minimum parameters: {}", config.min_parameters);
        info!("Minimum technical density: {:.2}", config.min_technical_density);
        
        // Initialize parameter detection patterns
        let parameter_patterns = Self::build_parameter_patterns()?;
        
        // Initialize counter detection patterns
        let counter_patterns = Self::build_counter_patterns()?;
        
        // Initialize technical term dictionary
        let technical_term_dictionary = TechnicalTermDictionary::new()?;
        
        // Initialize completeness assessor
        let completeness_assessor = CompletenessAssessor::new()?;
        
        info!("Quality Validator initialized with {} parameter patterns, {} counter patterns",
              parameter_patterns.len(), counter_patterns.len());
        
        Ok(Self {
            config: config.clone(),
            parameter_patterns,
            counter_patterns,
            technical_term_dictionary,
            completeness_assessor,
        })
    }
    
    /// Assess structural quality of a document
    pub fn assess_structural_quality(&self, document: &Document) -> Result<StructuralQuality> {
        debug!("Assessing structural quality for document: {}", document.id);
        
        let detailed_assessment = self.perform_detailed_assessment(document)?;
        
        // Convert to simplified StructuralQuality format
        let structural_quality = StructuralQuality {
            completeness_score: detailed_assessment.metrics.completeness_score,
            parameter_extraction_quality: detailed_assessment.metrics.parameter_extraction_score,
            counter_extraction_quality: detailed_assessment.metrics.counter_extraction_score,
            technical_density_score: detailed_assessment.metrics.technical_density_score,
            overall_score: detailed_assessment.overall_score,
        };
        
        debug!("Structural quality assessment completed: {:.2}", structural_quality.overall_score);
        
        // Log quality issues if score is below threshold
        if structural_quality.overall_score < self.config.quality_threshold {
            warn!("Document {} has quality score {:.2} below threshold {:.2}",
                  document.id, structural_quality.overall_score, self.config.quality_threshold);
            
            for issue in &detailed_assessment.quality_issues {
                match issue.severity {
                    IssueSeverity::Critical | IssueSeverity::High => {
                        warn!("Quality issue: {}", issue.description);
                    }
                    _ => {
                        debug!("Quality issue: {}", issue.description);
                    }
                }
            }
        }
        
        Ok(structural_quality)
    }
    
    /// Perform detailed quality assessment with full metrics
    pub fn perform_detailed_assessment(&self, document: &Document) -> Result<DetailedQualityAssessment> {
        let mut quality_issues = Vec::new();
        let mut recommendations = Vec::new();
        
        // 1. Parameter Extraction Assessment
        let parameter_score = self.assess_parameter_extraction(document, &mut quality_issues)?;
        
        // 2. Counter Extraction Assessment
        let counter_score = self.assess_counter_extraction(document, &mut quality_issues)?;
        
        // 3. Technical Density Assessment
        let technical_density_score = self.assess_technical_density(document, &mut quality_issues)?;
        
        // 4. Document Completeness Assessment
        let completeness_score = self.assess_completeness(document, &mut quality_issues)?;
        
        // 5. Content Coherence Assessment
        let coherence_score = self.assess_content_coherence(document, &mut quality_issues)?;
        
        // 6. Metadata Quality Assessment
        let metadata_score = self.assess_metadata_quality(document, &mut quality_issues)?;
        
        // Create quality metrics
        let metrics = QualityMetrics {
            parameter_extraction_score: parameter_score,
            counter_extraction_score: counter_score,
            technical_density_score,
            completeness_score,
            content_coherence_score: coherence_score,
            metadata_quality_score: metadata_score,
        };
        
        // Calculate overall score with weights
        let overall_score = self.calculate_weighted_score(&metrics);
        
        // Extract elements summary
        let extracted_elements = self.summarize_extracted_elements(document);
        
        // Generate recommendations
        self.generate_recommendations(&metrics, &quality_issues, &mut recommendations);
        
        Ok(DetailedQualityAssessment {
            overall_score,
            metrics,
            extracted_elements,
            quality_issues,
            recommendations,
        })
    }
    
    /// Assess parameter extraction quality
    fn assess_parameter_extraction(&self, document: &Document, issues: &mut Vec<QualityIssue>) -> Result<f64> {
        let parameters = &document.metadata.parameters;
        let parameter_count = parameters.len();
        
        // Base score from parameter count
        let count_score = if parameter_count >= self.config.min_parameters {
            (parameter_count as f64 / 10.0).min(1.0) // Normalize to max 10 parameters = 1.0
        } else {
            0.0
        };
        
        // Quality score from parameter completeness
        let mut completeness_score = 0.0;
        let mut complete_parameters = 0;
        
        for param in parameters {
            let mut param_completeness = 0.0;
            
            // Check for required fields
            if !param.name.is_empty() { param_completeness += 0.4; }
            if param.mo_class.is_some() { param_completeness += 0.2; }
            if param.valid_values.is_some() { param_completeness += 0.2; }
            if param.default_value.is_some() { param_completeness += 0.1; }
            if param.description.is_some() { param_completeness += 0.1; }
            
            if param_completeness >= 0.8 {
                complete_parameters += 1;
            }
            
            completeness_score += param_completeness;
        }
        
        if parameter_count > 0 {
            completeness_score /= parameter_count as f64;
        }
        
        // Combine scores
        let final_score = (count_score * 0.6 + completeness_score * 0.4);
        
        // Add issues if below threshold
        if parameter_count < self.config.min_parameters {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::MissingParameters,
                severity: IssueSeverity::High,
                description: format!("Document has only {} parameters, minimum required: {}", 
                                   parameter_count, self.config.min_parameters),
                location: Some("Parameters section".to_string()),
            });
        }
        
        if complete_parameters < parameter_count / 2 {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::MissingParameters,
                severity: IssueSeverity::Medium,
                description: "Many parameters are missing required metadata fields".to_string(),
                location: Some("Parameters section".to_string()),
            });
        }
        
        Ok(final_score)
    }
    
    /// Assess counter extraction quality
    fn assess_counter_extraction(&self, document: &Document, issues: &mut Vec<QualityIssue>) -> Result<f64> {
        let counters = &document.metadata.counters;
        let counter_count = counters.len();
        
        // Count score (counters are less common than parameters)
        let count_score = (counter_count as f64 / 5.0).min(1.0);
        
        // Completeness score
        let mut completeness_score = 0.0;
        for counter in counters {
            let mut counter_completeness = 0.0;
            
            if !counter.name.is_empty() { counter_completeness += 0.5; }
            if counter.description.is_some() { counter_completeness += 0.3; }
            if counter.mo_class.is_some() { counter_completeness += 0.1; }
            if counter.counter_type.is_some() { counter_completeness += 0.1; }
            
            completeness_score += counter_completeness;
        }
        
        if counter_count > 0 {
            completeness_score /= counter_count as f64;
        } else {
            // Don't penalize documents without counters too heavily
            completeness_score = 0.5;
        }
        
        let final_score = (count_score * 0.4 + completeness_score * 0.6);
        
        // Add issues for counter quality
        if counter_count == 0 && document.content.to_lowercase().contains("counter") {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::MissingCounters,
                severity: IssueSeverity::Medium,
                description: "Document mentions counters but none were extracted".to_string(),
                location: Some("Content analysis".to_string()),
            });
        }
        
        Ok(final_score)
    }
    
    /// Assess technical term density
    fn assess_technical_density(&self, document: &Document, issues: &mut Vec<QualityIssue>) -> Result<f64> {
        let technical_terms = &document.metadata.technical_terms;
        let content_length = document.content.len();
        
        // Calculate density (terms per 1000 characters)
        let density = if content_length > 0 {
            (technical_terms.len() as f64 * 1000.0) / content_length as f64
        } else {
            0.0
        };
        
        // Score based on density with optimal range
        let density_score = if density >= 5.0 {
            1.0 // Very high technical density
        } else if density >= 2.0 {
            0.8 + (density - 2.0) * 0.2 / 3.0 // Good density
        } else if density >= 1.0 {
            0.5 + (density - 1.0) * 0.3 // Acceptable density
        } else {
            density * 0.5 // Low density
        };
        
        // Assess term quality using weighted dictionary
        let mut weighted_term_score = 0.0;
        let mut total_weight = 0.0;
        
        for term in technical_terms {
            if let Some(weight) = self.technical_term_dictionary.term_weights.get(term) {
                weighted_term_score += weight;
                total_weight += 1.0;
            }
        }
        
        let term_quality_score = if total_weight > 0.0 {
            (weighted_term_score / total_weight).min(1.0)
        } else {
            0.0
        };
        
        let final_score = (density_score * 0.6 + term_quality_score * 0.4);
        
        // Add issues for low technical density
        if density < self.config.min_technical_density {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::LowTechnicalDensity,
                severity: IssueSeverity::Medium,
                description: format!("Technical term density {:.2} is below minimum {:.2}", 
                                   density, self.config.min_technical_density),
                location: Some("Content analysis".to_string()),
            });
        }
        
        if technical_terms.is_empty() {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::LowTechnicalDensity,
                severity: IssueSeverity::High,
                description: "No technical terms identified in document".to_string(),
                location: Some("Content analysis".to_string()),
            });
        }
        
        Ok(final_score)
    }
    
    /// Assess document completeness
    fn assess_completeness(&self, document: &Document, issues: &mut Vec<QualityIssue>) -> Result<f64> {
        let metadata = &document.metadata;
        
        // Metadata completeness score
        let mut metadata_score = 0.0;
        let mut metadata_fields_present = 0;
        
        if metadata.title.is_some() { metadata_score += 0.2; metadata_fields_present += 1; }
        if metadata.feature_name.is_some() { metadata_score += 0.25; metadata_fields_present += 1; }
        if metadata.product_info.is_some() { metadata_score += 0.15; metadata_fields_present += 1; }
        if metadata.feature_state.is_some() { metadata_score += 0.1; metadata_fields_present += 1; }
        if !metadata.parameters.is_empty() { metadata_score += 0.15; metadata_fields_present += 1; }
        if !metadata.counters.is_empty() { metadata_score += 0.1; metadata_fields_present += 1; }
        if !metadata.technical_terms.is_empty() { metadata_score += 0.05; metadata_fields_present += 1; }
        
        // Content structure completeness
        let structure_score = self.completeness_assessor.assess_structure_completeness(&document.content)?;
        
        // Content length appropriateness
        let length_score = self.assess_content_length_appropriateness(document.content.len());
        
        let final_score = (metadata_score * 0.4 + structure_score * 0.4 + length_score * 0.2);
        
        // Add issues for completeness
        if metadata_fields_present < 4 {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::IncompleteMetadata,
                severity: IssueSeverity::Medium,
                description: format!("Only {} of 7 metadata fields are populated", metadata_fields_present),
                location: Some("Document metadata".to_string()),
            });
        }
        
        if document.content.len() < 500 {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::ContentTooShort,
                severity: IssueSeverity::High,
                description: "Document content is very short (< 500 characters)".to_string(),
                location: Some("Content length".to_string()),
            });
        }
        
        Ok(final_score)
    }
    
    /// Assess content coherence
    fn assess_content_coherence(&self, document: &Document, issues: &mut Vec<QualityIssue>) -> Result<f64> {
        let content = &document.content;
        
        // Check for basic structural elements
        let has_sections = content.contains("##") || content.contains("# ");
        let has_lists = content.contains("- ") || content.contains("* ");
        let has_paragraphs = content.matches('\n').count() > 2;
        
        let mut structure_score = 0.0;
        if has_sections { structure_score += 0.4; }
        if has_lists { structure_score += 0.3; }
        if has_paragraphs { structure_score += 0.3; }
        
        // Check for logical flow indicators
        let flow_indicators = [
            "description", "parameters", "counters", "example", "usage",
            "configuration", "implementation", "troubleshooting"
        ];
        
        let mut flow_score = 0.0;
        for indicator in &flow_indicators {
            if content.to_lowercase().contains(indicator) {
                flow_score += 1.0 / flow_indicators.len() as f64;
            }
        }
        
        // Check for excessive repetition
        let repetition_penalty = self.assess_repetition_penalty(content);
        
        let final_score = ((structure_score + flow_score) / 2.0) * repetition_penalty;
        
        // Add issues for poor structure
        if !has_sections && content.len() > 1000 {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::PoorStructure,
                severity: IssueSeverity::Medium,
                description: "Long document lacks clear section structure".to_string(),
                location: Some("Document structure".to_string()),
            });
        }
        
        if repetition_penalty < 0.8 {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::PoorStructure,
                severity: IssueSeverity::Low,
                description: "Document contains excessive repetition".to_string(),
                location: Some("Content quality".to_string()),
            });
        }
        
        Ok(final_score)
    }
    
    /// Assess metadata quality
    fn assess_metadata_quality(&self, document: &Document, issues: &mut Vec<QualityIssue>) -> Result<f64> {
        let metadata = &document.metadata;
        let complexity_hints = &metadata.complexity_hints;
        
        // Check consistency between metadata and complexity hints
        let mut consistency_score = 0.0;
        
        // Parameter count consistency
        if complexity_hints.parameter_count == metadata.parameters.len() {
            consistency_score += 0.25;
        }
        
        // Counter count consistency
        if complexity_hints.counter_count == metadata.counters.len() {
            consistency_score += 0.25;
        }
        
        // Content length consistency
        if complexity_hints.content_length == document.content.len() {
            consistency_score += 0.25;
        }
        
        // Complexity assessment reasonableness
        let complexity_reasonable = match complexity_hints.estimated_complexity {
            ComplexityLevel::Fast => complexity_hints.parameter_count <= 2 && complexity_hints.technical_term_density < 2.0,
            ComplexityLevel::Balanced => complexity_hints.parameter_count <= 5 && complexity_hints.technical_term_density < 5.0,
            ComplexityLevel::Quality => complexity_hints.parameter_count > 3 || complexity_hints.technical_term_density >= 3.0,
        };
        
        if complexity_reasonable {
            consistency_score += 0.25;
        }
        
        // Check for metadata richness
        let richness_score = self.assess_metadata_richness(metadata);
        
        let final_score = (consistency_score * 0.6 + richness_score * 0.4);
        
        // Add issues for metadata quality
        if !complexity_reasonable {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::IncompleteMetadata,
                severity: IssueSeverity::Low,
                description: "Complexity level doesn't match document characteristics".to_string(),
                location: Some("Complexity assessment".to_string()),
            });
        }
        
        Ok(final_score)
    }
    
    /// Calculate weighted overall score
    fn calculate_weighted_score(&self, metrics: &QualityMetrics) -> f64 {
        // Weights optimized for RAN documentation
        let weights = [
            (metrics.parameter_extraction_score, 0.25),
            (metrics.counter_extraction_score, 0.15),
            (metrics.technical_density_score, 0.20),
            (metrics.completeness_score, 0.20),
            (metrics.content_coherence_score, 0.10),
            (metrics.metadata_quality_score, 0.10),
        ];
        
        weights.iter().map(|(score, weight)| score * weight).sum()
    }
    
    /// Summarize extracted elements
    fn summarize_extracted_elements(&self, document: &Document) -> ExtractedElements {
        let metadata = &document.metadata;
        
        // Count metadata fields that are populated
        let mut metadata_fields_populated = 0;
        if metadata.title.is_some() { metadata_fields_populated += 1; }
        if metadata.feature_name.is_some() { metadata_fields_populated += 1; }
        if metadata.product_info.is_some() { metadata_fields_populated += 1; }
        if metadata.feature_state.is_some() { metadata_fields_populated += 1; }
        if !metadata.parameters.is_empty() { metadata_fields_populated += 1; }
        if !metadata.counters.is_empty() { metadata_fields_populated += 1; }
        if !metadata.technical_terms.is_empty() { metadata_fields_populated += 1; }
        
        // Identify sections (simplified)
        let mut sections_identified = Vec::new();
        let content_lower = document.content.to_lowercase();
        
        for section in &["description", "parameters", "counters", "configuration", "examples"] {
            if content_lower.contains(section) {
                sections_identified.push(section.to_string());
            }
        }
        
        ExtractedElements {
            parameters_found: metadata.parameters.len(),
            counters_found: metadata.counters.len(),
            technical_terms_found: metadata.technical_terms.len(),
            sections_identified,
            metadata_fields_populated,
        }
    }
    
    /// Generate quality improvement recommendations
    fn generate_recommendations(
        &self,
        metrics: &QualityMetrics,
        issues: &[QualityIssue],
        recommendations: &mut Vec<QualityRecommendation>
    ) {
        // Parameter-related recommendations
        if metrics.parameter_extraction_score < 0.7 {
            recommendations.push(QualityRecommendation {
                recommendation_type: RecommendationType::AddMoreParameters,
                description: "Add more parameter definitions with complete metadata".to_string(),
                expected_improvement: 0.15,
            });
        }
        
        // Structure-related recommendations
        if metrics.content_coherence_score < 0.6 {
            recommendations.push(QualityRecommendation {
                recommendation_type: RecommendationType::ImproveStructure,
                description: "Improve document structure with clear sections and logical flow".to_string(),
                expected_improvement: 0.12,
            });
        }
        
        // Metadata-related recommendations
        if metrics.metadata_quality_score < 0.8 {
            recommendations.push(QualityRecommendation {
                recommendation_type: RecommendationType::EnhanceMetadata,
                description: "Complete missing metadata fields and ensure consistency".to_string(),
                expected_improvement: 0.08,
            });
        }
        
        // Technical depth recommendations
        if metrics.technical_density_score < 0.5 {
            recommendations.push(QualityRecommendation {
                recommendation_type: RecommendationType::IncreaseTechnicalDepth,
                description: "Add more technical terms and detailed explanations".to_string(),
                expected_improvement: 0.10,
            });
        }
        
        // Content balance recommendations
        for issue in issues {
            match issue.issue_type {
                QualityIssueType::ContentTooShort => {
                    recommendations.push(QualityRecommendation {
                        recommendation_type: RecommendationType::BalanceContent,
                        description: "Expand content with more detailed explanations and examples".to_string(),
                        expected_improvement: 0.20,
                    });
                }
                QualityIssueType::ContentTooLong => {
                    recommendations.push(QualityRecommendation {
                        recommendation_type: RecommendationType::BalanceContent,
                        description: "Consider splitting into multiple focused documents".to_string(),
                        expected_improvement: 0.05,
                    });
                }
                _ => {}
            }
        }
    }
    
    /// Helper functions for quality assessment
    
    fn assess_content_length_appropriateness(&self, length: usize) -> f64 {
        match length {
            0..=200 => 0.1,         // Too short
            201..=500 => 0.5,       // Short but acceptable
            501..=2000 => 1.0,      // Optimal length
            2001..=5000 => 0.9,     // Long but good
            5001..=10000 => 0.7,    // Very long
            _ => 0.5,               // Extremely long
        }
    }
    
    fn assess_repetition_penalty(&self, content: &str) -> f64 {
        // Simple repetition detection - in production would use more sophisticated algorithms
        let words: Vec<&str> = content.split_whitespace().collect();
        if words.is_empty() {
            return 1.0;
        }
        
        let mut word_counts = HashMap::new();
        for word in &words {
            let normalized = word.to_lowercase();
            *word_counts.entry(normalized).or_insert(0) += 1;
        }
        
        // Calculate repetition ratio
        let total_words = words.len();
        let unique_words = word_counts.len();
        let uniqueness_ratio = unique_words as f64 / total_words as f64;
        
        // Penalty for excessive repetition
        if uniqueness_ratio < 0.3 {
            0.5 // Heavy penalty
        } else if uniqueness_ratio < 0.5 {
            0.7 // Medium penalty
        } else if uniqueness_ratio < 0.7 {
            0.9 // Light penalty
        } else {
            1.0 // No penalty
        }
    }
    
    fn assess_metadata_richness(&self, metadata: &DocumentMetadata) -> f64 {
        let mut richness_score = 0.0;
        
        // Check parameter richness
        if !metadata.parameters.is_empty() {
            let avg_param_completeness = metadata.parameters.iter()
                .map(|p| {
                    let mut completeness = 0.0;
                    if !p.name.is_empty() { completeness += 0.2; }
                    if p.mo_class.is_some() { completeness += 0.2; }
                    if p.valid_values.is_some() { completeness += 0.2; }
                    if p.default_value.is_some() { completeness += 0.2; }
                    if p.description.is_some() { completeness += 0.2; }
                    completeness
                })
                .sum::<f64>() / metadata.parameters.len() as f64;
            richness_score += avg_param_completeness * 0.4;
        }
        
        // Check technical terms diversity
        let term_diversity = (metadata.technical_terms.len() as f64 / 20.0).min(1.0);
        richness_score += term_diversity * 0.3;
        
        // Check metadata field completeness
        let mut field_completeness = 0.0;
        if metadata.title.is_some() { field_completeness += 1.0/7.0; }
        if metadata.feature_name.is_some() { field_completeness += 1.0/7.0; }
        if metadata.product_info.is_some() { field_completeness += 1.0/7.0; }
        if metadata.feature_state.is_some() { field_completeness += 1.0/7.0; }
        if !metadata.parameters.is_empty() { field_completeness += 1.0/7.0; }
        if !metadata.counters.is_empty() { field_completeness += 1.0/7.0; }
        if !metadata.technical_terms.is_empty() { field_completeness += 1.0/7.0; }
        
        richness_score += field_completeness * 0.3;
        
        richness_score
    }
    
    /// Build parameter detection patterns
    fn build_parameter_patterns() -> Result<Vec<Regex>> {
        let patterns = vec![
            r"(?i)[-*]\s*\*\*([^*]+)\*\*[:\s]*([^\n]+)",
            r"(?i)parameter[s]?[:\s]+([A-Za-z0-9_.]+)",
            r"(?i)([A-Za-z0-9_.]+)[:\s]*parameter",
            r"(?i)MO[:\s]+([A-Za-z0-9_.]+)",
            r"(?i)([A-Za-z0-9_.]+)[:\s]*Boolean",
            r"(?i)([A-Za-z0-9_.]+)[:\s]*Integer",
        ];
        
        patterns.into_iter()
            .map(|pattern| Regex::new(pattern))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("Failed to compile parameter pattern: {}", e))
    }
    
    /// Build counter detection patterns
    fn build_counter_patterns() -> Result<Vec<Regex>> {
        let patterns = vec![
            r"(?i)counter[s]?[:\s]+([A-Za-z0-9_.]+)",
            r"(?i)([A-Za-z0-9_.]+)[:\s]*counter",
            r"(?i)KPI[:\s]+([A-Za-z0-9_.]+)",
            r"(?i)PM[:\s]+([A-Za-z0-9_.]+)",
            r"(?i)([A-Za-z0-9_.]+)[Cc]ount",
            r"(?i)([A-Za-z0-9_.]+)[Rr]ate",
        ];
        
        patterns.into_iter()
            .map(|pattern| Regex::new(pattern))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("Failed to compile counter pattern: {}", e))
    }
}

impl TechnicalTermDictionary {
    fn new() -> Result<Self> {
        // Initialize technical term tiers
        let tier1_exact_params = [
            "ttiBundlingUl", "cqiReportingMode", "pdcchAggregationLevel",
            "prachPreambleFormat", "srsConfigIndex", "drxInactivityTimer"
        ].iter().map(|s| s.to_string()).collect();
        
        let tier2_config_terms = [
            "configuration", "parameter", "setting", "option", "mode",
            "enable", "disable", "threshold", "limit", "timer"
        ].iter().map(|s| s.to_string()).collect();
        
        let tier3_network_elements = [
            "eNodeB", "gNodeB", "UE", "MME", "SGW", "PGW", "HSS",
            "PCRF", "EPC", "5GC", "AMF", "SMF", "UPF", "NSSF"
        ].iter().map(|s| s.to_string()).collect();
        
        let tier4_rare_terms = [
            "HARQ", "PDCCH", "PDSCH", "PUCCH", "PUSCH", "PRACH",
            "SRS", "CQI", "PMI", "RI", "MIMO", "CA", "CoMP"
        ].iter().map(|s| s.to_string()).collect();
        
        // Create term weights
        let mut term_weights = HashMap::new();
        
        // Tier 1: Highest weight (exact parameters)
        for term in &tier1_exact_params {
            term_weights.insert(term.clone(), 1.0);
        }
        
        // Tier 2: High weight (configuration terms)
        for term in &tier2_config_terms {
            term_weights.insert(term.clone(), 0.8);
        }
        
        // Tier 3: Medium weight (network elements)
        for term in &tier3_network_elements {
            term_weights.insert(term.clone(), 0.6);
        }
        
        // Tier 4: Lower weight (rare technical terms)
        for term in &tier4_rare_terms {
            term_weights.insert(term.clone(), 0.4);
        }
        
        Ok(Self {
            tier1_exact_params,
            tier2_config_terms,
            tier3_network_elements,
            tier4_rare_terms,
            term_weights,
        })
    }
}

impl CompletenessAssessor {
    fn new() -> Result<Self> {
        let required_sections = vec![
            "description".to_string(),
            "parameters".to_string(),
        ];
        
        let optional_sections = vec![
            "counters".to_string(),
            "examples".to_string(),
            "configuration".to_string(),
            "troubleshooting".to_string(),
        ];
        
        let mut section_patterns = HashMap::new();
        section_patterns.insert("description".to_string(), Regex::new(r"(?i)##?\s*description")?);
        section_patterns.insert("parameters".to_string(), Regex::new(r"(?i)##?\s*parameters?")?);
        section_patterns.insert("counters".to_string(), Regex::new(r"(?i)##?\s*counters?")?);
        section_patterns.insert("examples".to_string(), Regex::new(r"(?i)##?\s*examples?")?);
        
        let metadata_fields = vec![
            "title".to_string(),
            "feature_name".to_string(),
            "product_info".to_string(),
            "feature_state".to_string(),
        ];
        
        Ok(Self {
            required_sections,
            optional_sections,
            section_patterns,
            metadata_fields,
        })
    }
    
    fn assess_structure_completeness(&self, content: &str) -> Result<f64> {
        let mut score = 0.0;
        let mut sections_found = 0;
        let total_sections = self.required_sections.len() + self.optional_sections.len();
        
        // Check required sections
        for section in &self.required_sections {
            if let Some(pattern) = self.section_patterns.get(section) {
                if pattern.is_match(content) {
                    score += 0.4; // Required sections worth more
                    sections_found += 1;
                }
            }
        }
        
        // Check optional sections
        for section in &self.optional_sections {
            if let Some(pattern) = self.section_patterns.get(section) {
                if pattern.is_match(content) {
                    score += 0.15; // Optional sections worth less
                    sections_found += 1;
                }
            }
        }
        
        // Normalize score
        let normalized_score = (score).min(1.0);
        
        debug!("Structure completeness: {}/{} sections found, score: {:.2}",
               sections_found, total_sections, normalized_score);
        
        Ok(normalized_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DocumentMetadata;
    
    #[test]
    fn test_quality_validator_creation() {
        let config = QualitySettings {
            quality_threshold: 0.75,
            enable_structural_validation: true,
            min_parameters: 1,
            min_technical_density: 0.1,
            enable_early_filtering: true,
        };
        
        let validator = QualityValidator::new(&config).unwrap();
        assert!(validator.parameter_patterns.len() > 0);
        assert!(validator.counter_patterns.len() > 0);
    }
    
    #[test]
    fn test_technical_term_dictionary() {
        let dict = TechnicalTermDictionary::new().unwrap();
        assert!(dict.tier1_exact_params.len() > 0);
        assert!(dict.term_weights.len() > 0);
        
        // Test that tier 1 terms have highest weight
        for term in &dict.tier1_exact_params {
            assert_eq!(*dict.term_weights.get(term).unwrap(), 1.0);
        }
    }
    
    #[test]
    fn test_content_length_assessment() {
        let config = QualitySettings {
            quality_threshold: 0.75,
            enable_structural_validation: true,
            min_parameters: 1,
            min_technical_density: 0.1,
            enable_early_filtering: true,
        };
        
        let validator = QualityValidator::new(&config).unwrap();
        
        assert_eq!(validator.assess_content_length_appropriateness(100), 0.1);  // Too short
        assert_eq!(validator.assess_content_length_appropriateness(1000), 1.0); // Optimal
        assert_eq!(validator.assess_content_length_appropriateness(15000), 0.5); // Too long
    }
    
    #[test]
    fn test_repetition_penalty() {
        let config = QualitySettings {
            quality_threshold: 0.75,
            enable_structural_validation: true,
            min_parameters: 1,
            min_technical_density: 0.1,
            enable_early_filtering: true,
        };
        
        let validator = QualityValidator::new(&config).unwrap();
        
        let repetitive_content = "test test test test test";
        let diverse_content = "The quick brown fox jumps over the lazy dog";
        
        let repetitive_penalty = validator.assess_repetition_penalty(repetitive_content);
        let diverse_penalty = validator.assess_repetition_penalty(diverse_content);
        
        assert!(diverse_penalty > repetitive_penalty);
    }
}