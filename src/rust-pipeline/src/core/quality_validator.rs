/*!
# Quality Validator

Document quality validation with configurable thresholds.
*/

use crate::Result;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Quality validator for document processing results
#[derive(Debug)]
pub struct QualityValidator {
    validator_id: Uuid,
    config: QualityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConfig {
    pub min_content_length: usize,
    pub min_quality_score: f64,
    pub require_metadata: bool,
    pub check_encoding: bool,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            min_content_length: 100,
            min_quality_score: 0.7,
            require_metadata: false,
            check_encoding: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRequest {
    pub document_id: Uuid,
    pub content: String,
    pub metadata: std::collections::HashMap<String, String>,
    pub quality_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub document_id: Uuid,
    pub is_valid: bool,
    pub quality_score: f64,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}

impl QualityValidator {
    /// Create new quality validator
    pub fn new(config: QualityConfig) -> Self {
        Self {
            validator_id: Uuid::new_v4(),
            config,
        }
    }

    /// Validate document quality
    pub async fn validate(&self, request: ValidationRequest) -> Result<ValidationResult> {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        let mut is_valid = true;

        // Check content length
        if request.content.len() < self.config.min_content_length {
            issues.push(format!(
                "Content too short: {} characters (minimum: {})",
                request.content.len(),
                self.config.min_content_length
            ));
            is_valid = false;
        }

        // Check quality score
        if request.quality_score < self.config.min_quality_score {
            issues.push(format!(
                "Quality score too low: {:.2} (minimum: {:.2})",
                request.quality_score, self.config.min_quality_score
            ));
            is_valid = false;
        }

        // Check metadata requirements
        if self.config.require_metadata && request.metadata.is_empty() {
            issues.push("Missing required metadata".to_string());
            recommendations.push("Add document metadata for better processing".to_string());
        }

        // Check encoding
        if self.config.check_encoding && !request.content.is_ascii() {
            recommendations.push("Consider UTF-8 encoding validation".to_string());
        }

        Ok(ValidationResult {
            document_id: request.document_id,
            is_valid,
            quality_score: request.quality_score,
            issues,
            recommendations,
        })
    }
}

/// Initialize quality validator
pub async fn initialize() -> Result<()> {
    tracing::info!("Initializing quality validator");
    Ok(())
}
