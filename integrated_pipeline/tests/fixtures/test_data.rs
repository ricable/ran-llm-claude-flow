use std::path::PathBuf;
use std::collections::HashMap;
use uuid::Uuid;
use chrono::Utc;
use rust_core::types::*;
use serde_json::json;

/// Test data fixtures for comprehensive pipeline testing
pub struct TestDataFixtures;

impl TestDataFixtures {
    /// Creates a sample Ericsson 5G feature document for testing
    pub fn sample_ericsson_document() -> Document {
        Document {
            id: Uuid::new_v4(),
            path: PathBuf::from("test_data/ericsson_5g_feature.md"),
            format: DocumentFormat::Markdown,
            content: r#"# Radio Resource Partitioning

## Feature Overview
The Radio Resource Partitioning feature enables efficient resource allocation in 5G NR networks.

## Parameters
- **resourcePartitions.gbrPartitioning**: Controls GBR bearer partitioning (Boolean)
- **resourcePartitions.maxSharePercent**: Maximum share percentage (Integer, 0-100)
- **qosFlowMapping.priorityLevel**: QoS priority level (Integer, 1-255)

## Counters
- **pmRadioResourceUtilization**: Radio resource utilization percentage
- **pmGbrPartitionEfficiency**: GBR partition efficiency metric
- **pmQosFlowThroughput**: QoS flow throughput counter

## Technical Details
The feature implements advanced algorithms for:
- GBR bearer management
- QoS-based scheduling
- Resource allocation optimization
- Dynamic partition adjustment

Quality Assurance ensures optimal performance in high-load scenarios.
"#.to_string(),
            metadata: DocumentMetadata {
                title: Some("Radio Resource Partitioning".to_string()),
                feature_name: Some("Radio Resource Partitioning".to_string()),
                product_info: Some("Ericsson 5G RAN".to_string()),
                feature_state: Some("Active".to_string()),
                parameters: vec![
                    Parameter {
                        name: "resourcePartitions.gbrPartitioning".to_string(),
                        mo_class: Some("ResourcePartitions".to_string()),
                        data_type: Some("Boolean".to_string()),
                        valid_values: Some("true, false".to_string()),
                        default_value: Some("false".to_string()),
                        description: Some("Controls GBR bearer partitioning".to_string()),
                    },
                    Parameter {
                        name: "resourcePartitions.maxSharePercent".to_string(),
                        mo_class: Some("ResourcePartitions".to_string()),
                        data_type: Some("Integer".to_string()),
                        valid_values: Some("0-100".to_string()),
                        default_value: Some("50".to_string()),
                        description: Some("Maximum share percentage".to_string()),
                    },
                    Parameter {
                        name: "qosFlowMapping.priorityLevel".to_string(),
                        mo_class: Some("QosFlowMapping".to_string()),
                        data_type: Some("Integer".to_string()),
                        valid_values: Some("1-255".to_string()),
                        default_value: Some("10".to_string()),
                        description: Some("QoS priority level".to_string()),
                    },
                ],
                counters: vec![
                    Counter {
                        name: "pmRadioResourceUtilization".to_string(),
                        description: Some("Radio resource utilization percentage".to_string()),
                        mo_class: Some("RadioResource".to_string()),
                        counter_type: Some("Gauge".to_string()),
                    },
                    Counter {
                        name: "pmGbrPartitionEfficiency".to_string(),
                        description: Some("GBR partition efficiency metric".to_string()),
                        mo_class: Some("ResourcePartitions".to_string()),
                        counter_type: Some("Counter".to_string()),
                    },
                    Counter {
                        name: "pmQosFlowThroughput".to_string(),
                        description: Some("QoS flow throughput counter".to_string()),
                        mo_class: Some("QosFlow".to_string()),
                        counter_type: Some("Counter".to_string()),
                    },
                ],
                technical_terms: vec![
                    "GBR".to_string(),
                    "QoS".to_string(),
                    "5G NR".to_string(),
                    "Radio Resource".to_string(),
                    "Partitioning".to_string(),
                ],
                complexity_hints: ComplexityHints {
                    parameter_count: 3,
                    counter_count: 3,
                    technical_term_density: 0.15,
                    content_length: 500,
                    estimated_complexity: ComplexityLevel::Balanced,
                },
            },
            size_bytes: 500,
            created_at: Utc::now(),
        }
    }

    /// Creates a complex 3GPP specification document
    pub fn complex_3gpp_document() -> Document {
        Document {
            id: Uuid::new_v4(),
            path: PathBuf::from("test_data/3gpp_spec.pdf"),
            format: DocumentFormat::Gpp3,
            content: r#"# 3GPP TS 38.321 - MAC Protocol Specification

## Section 5.4: MAC PDU Processing

### 5.4.1 DL-SCH Data Reception
The MAC entity shall process received MAC PDUs according to the following procedures:

#### Parameters:
- **mac-CellGroupConfig**: MAC cell group configuration
- **drx-Config**: DRX configuration parameters  
- **schedulingRequestConfig**: SR configuration
- **bsr-Config**: Buffer Status Report configuration
- **phr-Config**: Power Headroom Report configuration

#### Counters:
- **macDlDataVolume**: Downlink data volume in bytes
- **macUlDataVolume**: Uplink data volume in bytes
- **macHarqRetransmissions**: HARQ retransmission count
- **macCqiReports**: CQI report count
- **macSrTransmissions**: SR transmission count

The MAC entity maintains multiple logical channels, each with specific priority handling and QoS requirements. Buffer status reporting is performed according to configured triggers and periodicity.

Quality assurance mechanisms ensure reliable data transmission through HARQ processes and adaptive modulation schemes.
"#.to_string(),
            metadata: DocumentMetadata {
                title: Some("3GPP TS 38.321 - MAC Protocol".to_string()),
                feature_name: Some("MAC Protocol Processing".to_string()),
                product_info: Some("3GPP Release 17".to_string()),
                feature_state: Some("Standardized".to_string()),
                parameters: vec![
                    Parameter {
                        name: "mac-CellGroupConfig".to_string(),
                        mo_class: Some("MAC".to_string()),
                        data_type: Some("Sequence".to_string()),
                        valid_values: Some("As per ASN.1".to_string()),
                        default_value: None,
                        description: Some("MAC cell group configuration".to_string()),
                    },
                    Parameter {
                        name: "drx-Config".to_string(),
                        mo_class: Some("DRX".to_string()),
                        data_type: Some("Choice".to_string()),
                        valid_values: Some("setup, release".to_string()),
                        default_value: Some("release".to_string()),
                        description: Some("DRX configuration parameters".to_string()),
                    },
                ],
                counters: vec![
                    Counter {
                        name: "macDlDataVolume".to_string(),
                        description: Some("Downlink data volume in bytes".to_string()),
                        mo_class: Some("MAC".to_string()),
                        counter_type: Some("Counter".to_string()),
                    },
                    Counter {
                        name: "macHarqRetransmissions".to_string(),
                        description: Some("HARQ retransmission count".to_string()),
                        mo_class: Some("HARQ".to_string()),
                        counter_type: Some("Counter".to_string()),
                    },
                ],
                technical_terms: vec![
                    "MAC".to_string(),
                    "PDU".to_string(),
                    "DL-SCH".to_string(),
                    "HARQ".to_string(),
                    "DRX".to_string(),
                    "BSR".to_string(),
                    "PHR".to_string(),
                    "CQI".to_string(),
                    "3GPP".to_string(),
                ],
                complexity_hints: ComplexityHints {
                    parameter_count: 5,
                    counter_count: 5,
                    technical_term_density: 0.25,
                    content_length: 800,
                    estimated_complexity: ComplexityLevel::Quality,
                },
            },
            size_bytes: 800,
            created_at: Utc::now(),
        }
    }

    /// Creates a simple CSV data document
    pub fn simple_csv_document() -> Document {
        Document {
            id: Uuid::new_v4(),
            path: PathBuf::from("test_data/kpi_data.csv"),
            format: DocumentFormat::Csv,
            content: r#"Parameter,Value,Unit,Description
cellDownlinkThroughput,150.5,Mbps,Average cell downlink throughput
cellUplinkThroughput,75.2,Mbps,Average cell uplink throughput
activeUsers,245,count,Number of active users
resourceUtilization,68.3,percent,Resource block utilization
"#.to_string(),
            metadata: DocumentMetadata {
                title: Some("KPI Performance Data".to_string()),
                feature_name: Some("Performance Monitoring".to_string()),
                product_info: Some("Ericsson RAN".to_string()),
                feature_state: Some("Active".to_string()),
                parameters: vec![
                    Parameter {
                        name: "cellDownlinkThroughput".to_string(),
                        mo_class: Some("Cell".to_string()),
                        data_type: Some("Float".to_string()),
                        valid_values: Some("0.0-1000.0".to_string()),
                        default_value: Some("0.0".to_string()),
                        description: Some("Average cell downlink throughput".to_string()),
                    },
                ],
                counters: vec![
                    Counter {
                        name: "activeUsers".to_string(),
                        description: Some("Number of active users".to_string()),
                        mo_class: Some("Cell".to_string()),
                        counter_type: Some("Gauge".to_string()),
                    },
                ],
                technical_terms: vec!["KPI".to_string(), "Throughput".to_string()],
                complexity_hints: ComplexityHints {
                    parameter_count: 1,
                    counter_count: 1,
                    technical_term_density: 0.05,
                    content_length: 200,
                    estimated_complexity: ComplexityLevel::Fast,
                },
            },
            size_bytes: 200,
            created_at: Utc::now(),
        }
    }

    /// Creates expected QA pairs for the sample Ericsson document
    pub fn expected_qa_pairs() -> Vec<QAPair> {
        vec![
            QAPair {
                id: Uuid::new_v4(),
                question: "How does the Radio Resource Partitioning feature handle GBR bearers?".to_string(),
                answer: "GBR bearers are partitioned only when the resourcePartitions.gbrPartitioning attribute is set to TRUE, enabling dedicated resource allocation for guaranteed bit rate services.".to_string(),
                context: Some("GBR bearer management within Radio Resource Partitioning".to_string()),
                confidence: 0.95,
                metadata: QAMetadata {
                    question_type: QuestionType::Factual,
                    technical_terms: vec!["GBR".to_string(), "resourcePartitions.gbrPartitioning".to_string()],
                    parameters_mentioned: vec!["resourcePartitions.gbrPartitioning".to_string()],
                    counters_mentioned: vec![],
                    complexity_level: ComplexityLevel::Balanced,
                },
            },
            QAPair {
                id: Uuid::new_v4(),
                question: "What are the key performance counters for monitoring resource partitioning efficiency?".to_string(),
                answer: "The key counters are pmRadioResourceUtilization for overall usage, pmGbrPartitionEfficiency for GBR partition performance, and pmQosFlowThroughput for QoS flow monitoring.".to_string(),
                context: Some("Performance monitoring in Radio Resource Partitioning".to_string()),
                confidence: 0.92,
                metadata: QAMetadata {
                    question_type: QuestionType::Analytical,
                    technical_terms: vec!["QoS".to_string(), "GBR".to_string()],
                    parameters_mentioned: vec![],
                    counters_mentioned: vec![
                        "pmRadioResourceUtilization".to_string(),
                        "pmGbrPartitionEfficiency".to_string(),
                        "pmQosFlowThroughput".to_string(),
                    ],
                    complexity_level: ComplexityLevel::Balanced,
                },
            },
            QAPair {
                id: Uuid::new_v4(),
                question: "What is the valid range for the resourcePartitions.maxSharePercent parameter?".to_string(),
                answer: "The resourcePartitions.maxSharePercent parameter accepts integer values from 0 to 100, with a default value of 50.".to_string(),
                context: Some("Parameter configuration for resource partitioning".to_string()),
                confidence: 0.98,
                metadata: QAMetadata {
                    question_type: QuestionType::Factual,
                    technical_terms: vec!["resourcePartitions.maxSharePercent".to_string()],
                    parameters_mentioned: vec!["resourcePartitions.maxSharePercent".to_string()],
                    counters_mentioned: vec![],
                    complexity_level: ComplexityLevel::Fast,
                },
            },
        ]
    }

    /// Creates ML processing options for testing
    pub fn test_ml_processing_options() -> MLProcessingOptions {
        MLProcessingOptions {
            model_preference: Some(ComplexityLevel::Balanced),
            max_qa_pairs: Some(5),
            quality_threshold: 0.75,
            enable_diversity_enhancement: true,
            batch_processing: false,
        }
    }

    /// Creates processing configuration for M3 Max testing
    pub fn m3_max_test_config() -> serde_json::Value {
        json!({
            "performance": {
                "max_concurrent_docs": 16,
                "memory_limit_gb": 60,
                "gpu_acceleration": true,
                "apple_silicon_optimization": true
            },
            "quality": {
                "min_qa_pairs": 3,
                "max_qa_pairs": 10,
                "quality_threshold": 0.75,
                "diversity_threshold": 0.05
            },
            "models": {
                "fast": "qwen3-1.7b",
                "balanced": "qwen3-7b", 
                "quality": "qwen3-30b"
            },
            "ipc": {
                "timeout_seconds": 30,
                "retry_count": 3,
                "batch_size": 8
            }
        })
    }

    /// Creates test dataset from raw JSONL files
    pub fn load_real_dataset_sample(limit: usize) -> Vec<serde_json::Value> {
        // This would load actual data from the raw datasets
        vec![
            json!({
                "messages": [
                    {"role": "user", "content": "How does carrier aggregation work in 5G NR?"},
                    {"role": "assistant", "content": "Carrier aggregation in 5G NR allows the UE to simultaneously use multiple component carriers to increase data rates and improve spectrum efficiency."}
                ],
                "metadata": {
                    "feature_name": "Carrier Aggregation",
                    "quality_score": 8.5,
                    "technical_terms": ["5G", "NR", "UE", "carrier aggregation"],
                    "source_dataset": "ericsson_5g_features"
                }
            }),
            json!({
                "messages": [
                    {"role": "user", "content": "What parameters control MIMO operation?"},
                    {"role": "assistant", "content": "MIMO operation is controlled by parameters such as antennaPortsCount, maxLayers, and codeBookSubset configuration."}
                ],
                "metadata": {
                    "feature_name": "MIMO Configuration",
                    "quality_score": 9.2,
                    "technical_terms": ["MIMO", "antennaPortsCount", "maxLayers"],
                    "source_dataset": "ericsson_5g_features"
                }
            }),
        ]
    }

    /// Performance test targets and thresholds
    pub fn performance_targets() -> HashMap<String, f64> {
        let mut targets = HashMap::new();
        targets.insert("throughput_docs_per_hour".to_string(), 25.0);
        targets.insert("memory_usage_gb_max".to_string(), 60.0);
        targets.insert("ipc_latency_seconds_max".to_string(), 3.0);
        targets.insert("quality_score_min".to_string(), 0.75);
        targets.insert("quality_variance_max".to_string(), 0.05);
        targets.insert("cpu_utilization_max".to_string(), 0.85);
        targets.insert("error_rate_max".to_string(), 0.01);
        targets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_document_creation() {
        let doc = TestDataFixtures::sample_ericsson_document();
        assert_eq!(doc.format, DocumentFormat::Markdown);
        assert!(doc.content.contains("Radio Resource Partitioning"));
        assert_eq!(doc.metadata.parameters.len(), 3);
        assert_eq!(doc.metadata.counters.len(), 3);
    }

    #[test]
    fn test_qa_pairs_generation() {
        let qa_pairs = TestDataFixtures::expected_qa_pairs();
        assert_eq!(qa_pairs.len(), 3);
        assert!(qa_pairs.iter().all(|qa| qa.confidence >= 0.9));
    }

    #[test]
    fn test_performance_targets() {
        let targets = TestDataFixtures::performance_targets();
        assert!(targets.contains_key("throughput_docs_per_hour"));
        assert!(targets.get("quality_score_min").unwrap() >= &0.7);
    }
}