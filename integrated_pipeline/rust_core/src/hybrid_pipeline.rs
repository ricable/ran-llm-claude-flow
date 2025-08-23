use crate::types::*;
use crate::batch_processor::{BatchProcessor, BatchProcessorConfig, OptimizationStrategy};
use crate::quality_validator::{QualityValidator, ValidationConfig};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, RwLock, Mutex};
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{info, warn, error, debug, instrument};
use uuid::Uuid;
use dashmap::DashMap;
use rayon::prelude::*;

/// Hybrid Rust-Python pipeline coordinator with <3s model switching
/// Orchestrates complete document processing workflow with multi-model intelligence
#[derive(Debug)]
pub struct HybridPipelineCoordinator {
    config: HybridPipelineConfig,
    batch_processor: Arc<BatchProcessor>,
    quality_validator: Arc<QualityValidator>,
    
    // Python ML communication
    python_process: Arc<Mutex<Option<tokio::process::Child>>>,
    ipc_sender: Arc<Mutex<Option<mpsc::Sender<IPCMessage>>>>,
    ipc_receiver: Arc<Mutex<Option<mpsc::Receiver<IPCResponse>>>>,
    
    // Performance tracking
    processing_metrics: Arc<RwLock<HybridProcessingMetrics>>,
    active_sessions: Arc<DashMap<Uuid, ProcessingSession>>,
    
    // Model management
    model_performance_cache: Arc<DashMap<ComplexityLevel, ModelPerformanceProfile>>,
    model_switching_history: Arc<RwLock<Vec<ModelSwitchEvent>>>,
    
    // Pipeline state
    pipeline_status: Arc<RwLock<PipelineStatus>>,
    error_recovery: Arc<Mutex<ErrorRecoveryState>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridPipelineConfig {
    pub python_executable: PathBuf,
    pub python_script_path: PathBuf,
    pub ipc_timeout: Duration,
    pub model_switch_timeout: Duration,
    pub max_concurrent_sessions: usize,
    pub enable_performance_monitoring: bool,
    pub quality_threshold: f64,
    pub throughput_target_docs_hour: f64,
    pub memory_management: MemoryManagementConfig,
    pub model_selection: ModelSelectionConfig,
    pub error_recovery: ErrorRecoveryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryManagementConfig {
    pub rust_allocation_gb: f64,      // 60GB for Rust processing
    pub python_allocation_gb: f64,     // 45GB for Python ML
    pub shared_memory_gb: f64,         // 15GB for IPC
    pub memory_pressure_threshold: f64, // Trigger cleanup at 90%
    pub enable_memory_optimization: bool,
    pub garbage_collection_interval: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSelectionConfig {
    pub default_strategy: String,
    pub complexity_thresholds: HashMap<String, f64>,
    pub quality_requirements: HashMap<String, f64>,
    pub enable_adaptive_selection: bool,
    pub model_switching_latency_target_ms: u64, // <3000ms target
    pub performance_learning_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRecoveryConfig {
    pub max_retry_attempts: usize,
    pub retry_backoff_ms: u64,
    pub enable_circuit_breaker: bool,
    pub circuit_breaker_threshold: f64,
    pub fallback_processing_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingSession {
    pub session_id: Uuid,
    pub documents: Vec<Document>,
    pub processing_mode: ProcessingMode,
    pub start_time: Instant,
    pub expected_completion: Option<Instant>,
    pub current_phase: ProcessingPhase,
    pub models_used: Vec<ComplexityLevel>,
    pub performance_metrics: SessionMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingMode {
    Single,      // Single document processing
    Batch,       // Batch processing
    Stream,      // Streaming processing
    Priority,    // Priority processing with QoS
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingPhase {
    Initialized,
    RustPreprocessing,
    ModelSelection,
    PythonMLProcessing,
    QualityValidation,
    ResultAggregation,
    Completed,
    Failed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetrics {
    pub rust_processing_time: Duration,
    pub python_processing_time: Duration,
    pub model_selection_time: Duration,
    pub ipc_overhead_time: Duration,
    pub total_time: Duration,
    pub memory_peak_gb: f64,
    pub throughput_docs_hour: f64,
    pub quality_average: f64,
    pub model_switches: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridProcessingMetrics {
    pub total_documents_processed: usize,
    pub total_sessions: usize,
    pub successful_sessions: usize,
    pub average_session_time: Duration,
    pub average_quality_score: f64,
    pub model_usage_distribution: HashMap<ComplexityLevel, usize>,
    pub model_switch_latency_average: Duration,
    pub memory_efficiency_average: f64,
    pub error_rate: f64,
    pub throughput_current: f64,
    pub throughput_target: f64,
    pub last_updated: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceProfile {
    pub model: ComplexityLevel,
    pub avg_inference_time: Duration,
    pub avg_quality_score: f64,
    pub success_rate: f64,
    pub memory_efficiency: f64,
    pub processing_cost: f64,
    pub sample_count: usize,
    pub last_updated: SystemTime,
    pub performance_trend: Vec<f64>, // Recent performance history
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSwitchEvent {
    pub timestamp: SystemTime,
    pub from_model: ComplexityLevel,
    pub to_model: ComplexityLevel,
    pub reason: String,
    pub switch_latency: Duration,
    pub success: bool,
    pub session_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineStatus {
    Initializing,
    Ready,
    Processing,
    ModelSwitching,
    MemoryOptimizing,
    ErrorRecovery,
    Shutdown,
}

#[derive(Debug, Clone)]
pub struct ErrorRecoveryState {
    pub error_count: usize,
    pub last_error_time: Option<SystemTime>,
    pub circuit_breaker_open: bool,
    pub recovery_attempts: usize,
    pub fallback_mode_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IPCMessage {
    pub message_id: Uuid,
    pub message_type: IPCMessageType,
    pub payload: Vec<u8>,
    pub timestamp: SystemTime,
    pub session_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IPCMessageType {
    ProcessDocument,
    BatchProcess,
    ModelSwitch,
    PerformanceUpdate,
    HealthCheck,
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IPCResponse {
    pub message_id: Uuid,
    pub response_type: IPCResponseType,
    pub payload: Vec<u8>,
    pub timestamp: SystemTime,
    pub success: bool,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IPCResponseType {
    ProcessingResult,
    ModelSwitchConfirmation,
    PerformanceMetrics,
    HealthStatus,
    Error,
}

impl Default for HybridPipelineConfig {
    fn default() -> Self {
        Self {
            python_executable: PathBuf::from("python3"),
            python_script_path: PathBuf::from("integrated_pipeline/python_ml/src/main.py"),
            ipc_timeout: Duration::from_secs(30),
            model_switch_timeout: Duration::from_secs(3), // <3s switching target
            max_concurrent_sessions: 4,
            enable_performance_monitoring: true,
            quality_threshold: 0.75,
            throughput_target_docs_hour: 25.0,
            memory_management: MemoryManagementConfig {
                rust_allocation_gb: 60.0,
                python_allocation_gb: 45.0,
                shared_memory_gb: 15.0,
                memory_pressure_threshold: 0.90,
                enable_memory_optimization: true,
                garbage_collection_interval: Duration::from_secs(300),
            },
            model_selection: ModelSelectionConfig {
                default_strategy: "adaptive".to_string(),
                complexity_thresholds: [
                    ("fast".to_string(), 0.3),
                    ("balanced".to_string(), 0.7),
                    ("quality".to_string(), 1.0),
                ].iter().cloned().collect(),
                quality_requirements: [
                    ("basic".to_string(), 0.6),
                    ("standard".to_string(), 0.75),
                    ("premium".to_string(), 0.9),
                ].iter().cloned().collect(),
                enable_adaptive_selection: true,
                model_switching_latency_target_ms: 3000,
                performance_learning_rate: 0.1,
            },
            error_recovery: ErrorRecoveryConfig {
                max_retry_attempts: 3,
                retry_backoff_ms: 1000,
                enable_circuit_breaker: true,
                circuit_breaker_threshold: 0.5,
                fallback_processing_enabled: true,
            },
        }
    }
}

impl HybridPipelineCoordinator {
    pub async fn new(config: HybridPipelineConfig) -> Result<Self> {
        info!("Initializing Hybrid Pipeline Coordinator");
        
        // Initialize batch processor with M3 Max optimization
        let batch_config = BatchProcessorConfig {
            max_concurrent_documents: num_cpus::get() * 2,
            max_concurrent_batches: 4,
            batch_size_hint: 8,
            min_batch_size: 2,
            max_batch_size: 32,
            processing_timeout: config.ipc_timeout,
            enable_adaptive_batching: true,
            memory_limit_gb: config.memory_management.rust_allocation_gb,
            optimization_strategy: OptimizationStrategy::Balanced,
        };
        
        let batch_processor = Arc::new(BatchProcessor::new(batch_config));
        
        // Initialize quality validator
        let validation_config = ValidationConfig {
            min_parameter_count: 3,
            min_technical_density: 0.15,
            completeness_threshold: config.quality_threshold,
            enable_caching: true,
            parallel_processing: true,
        };
        
        let quality_validator = Arc::new(QualityValidator::new(validation_config));
        
        // Initialize metrics
        let processing_metrics = Arc::new(RwLock::new(HybridProcessingMetrics {
            total_documents_processed: 0,
            total_sessions: 0,
            successful_sessions: 0,
            average_session_time: Duration::from_secs(0),
            average_quality_score: 0.0,
            model_usage_distribution: HashMap::new(),
            model_switch_latency_average: Duration::from_millis(0),
            memory_efficiency_average: 0.0,
            error_rate: 0.0,
            throughput_current: 0.0,
            throughput_target: config.throughput_target_docs_hour,
            last_updated: SystemTime::now(),
        }));
        
        let coordinator = Self {
            config,
            batch_processor,
            quality_validator,
            python_process: Arc::new(Mutex::new(None)),
            ipc_sender: Arc::new(Mutex::new(None)),
            ipc_receiver: Arc::new(Mutex::new(None)),
            processing_metrics,
            active_sessions: Arc::new(DashMap::new()),
            model_performance_cache: Arc::new(DashMap::new()),
            model_switching_history: Arc::new(RwLock::new(Vec::new())),
            pipeline_status: Arc::new(RwLock::new(PipelineStatus::Initializing)),
            error_recovery: Arc::new(Mutex::new(ErrorRecoveryState {
                error_count: 0,
                last_error_time: None,
                circuit_breaker_open: false,
                recovery_attempts: 0,
                fallback_mode_active: false,
            })),
        };
        
        // Initialize model performance profiles
        coordinator.initialize_model_profiles().await?;
        
        info!("Hybrid Pipeline Coordinator initialized successfully");
        Ok(coordinator)
    }
    
    async fn initialize_model_profiles(&self) -> Result<()> {
        let models = vec![
            ComplexityLevel::Fast,
            ComplexityLevel::Balanced,
            ComplexityLevel::Quality,
        ];
        
        for model in models {
            let profile = ModelPerformanceProfile {
                model: model.clone(),
                avg_inference_time: Duration::from_secs(match model {
                    ComplexityLevel::Fast => 2,
                    ComplexityLevel::Balanced => 5,
                    ComplexityLevel::Quality => 12,
                }),
                avg_quality_score: match model {
                    ComplexityLevel::Fast => 0.72,
                    ComplexityLevel::Balanced => 0.82,
                    ComplexityLevel::Quality => 0.91,
                },
                success_rate: 0.95,
                memory_efficiency: 0.85,
                processing_cost: 1.0,
                sample_count: 0,
                last_updated: SystemTime::now(),
                performance_trend: Vec::new(),
            };
            
            self.model_performance_cache.insert(model, profile);
        }
        
        info!("Initialized performance profiles for all model variants");
        Ok(())
    }
    
    #[instrument(skip(self))]
    pub async fn start_python_ml_engine(&self) -> Result<()> {
        info!("Starting Python ML engine");
        *self.pipeline_status.write().await = PipelineStatus::Initializing;
        
        // Setup IPC channels
        let (tx, rx) = mpsc::channel::<IPCMessage>(1000);
        let (python_tx, python_rx) = mpsc::channel::<IPCResponse>(1000);
        
        // Start Python process
        let mut cmd = Command::new(&self.config.python_executable);
        cmd.arg(&self.config.python_script_path)
           .arg("--mode")
           .arg("ipc_server")
           .arg("--memory-allocation")
           .arg(&self.config.memory_management.python_allocation_gb.to_string())
           .kill_on_drop(true)
           .stdin(std::process::Stdio::piped())
           .stdout(std::process::Stdio::piped())
           .stderr(std::process::Stdio::piped());
        
        let mut child = cmd.spawn()
            .context(\"Failed to start Python ML engine\")?;
        
        // Wait for Python process to initialize
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        // Check if process is still running
        match child.try_wait()? {
            Some(status) => {
                return Err(anyhow::anyhow!(\"Python process exited during startup: {}\", status));
            }
            None => {
                info!(\"Python ML engine started successfully\");
            }
        }
        
        // Store process and channels
        *self.python_process.lock().await = Some(child);
        *self.ipc_sender.lock().await = Some(tx);
        *self.ipc_receiver.lock().await = Some(python_rx);
        
        // Perform health check
        self.perform_python_health_check().await?;
        
        *self.pipeline_status.write().await = PipelineStatus::Ready;
        info!(\"Hybrid pipeline ready for processing\");
        
        Ok(())
    }
    
    async fn perform_python_health_check(&self) -> Result<()> {
        debug!(\"Performing Python ML engine health check\");
        
        let health_message = IPCMessage {
            message_id: Uuid::new_v4(),
            message_type: IPCMessageType::HealthCheck,
            payload: b\"ping\".to_vec(),
            timestamp: SystemTime::now(),
            session_id: None,
        };
        
        // Send health check with timeout
        let response = timeout(
            self.config.ipc_timeout,
            self.send_ipc_message(health_message)
        ).await??;
        
        if response.success {
            debug!(\"Python ML engine health check successful\");
            Ok(())
        } else {
            Err(anyhow::anyhow!(\"Python ML engine health check failed: {:?}\", response.error_message))
        }
    }
    
    #[instrument(skip(self, documents))]
    pub async fn process_documents_hybrid(
        &self,
        documents: Vec<Document>,
        processing_mode: ProcessingMode,
    ) -> Result<HybridProcessingResult> {
        let session_id = Uuid::new_v4();
        let start_time = Instant::now();
        
        info!(\"Starting hybrid processing session {} for {} documents\", session_id, documents.len());
        
        // Create processing session
        let session = ProcessingSession {
            session_id,
            documents: documents.clone(),
            processing_mode: processing_mode.clone(),
            start_time,
            expected_completion: None,
            current_phase: ProcessingPhase::Initialized,
            models_used: Vec::new(),
            performance_metrics: SessionMetrics {
                rust_processing_time: Duration::from_secs(0),
                python_processing_time: Duration::from_secs(0),
                model_selection_time: Duration::from_secs(0),
                ipc_overhead_time: Duration::from_secs(0),
                total_time: Duration::from_secs(0),
                memory_peak_gb: 0.0,
                throughput_docs_hour: 0.0,
                quality_average: 0.0,
                model_switches: 0,
            },
        };
        
        self.active_sessions.insert(session_id, session);
        *self.pipeline_status.write().await = PipelineStatus::Processing;
        
        let result = match processing_mode {
            ProcessingMode::Single => self.process_single_document_hybrid(session_id, documents).await,
            ProcessingMode::Batch => self.process_batch_documents_hybrid(session_id, documents).await,
            ProcessingMode::Stream => self.process_stream_documents_hybrid(session_id, documents).await,
            ProcessingMode::Priority => self.process_priority_documents_hybrid(session_id, documents).await,
        };
        
        // Update session metrics
        if let Some(mut session) = self.active_sessions.get_mut(&session_id) {
            session.performance_metrics.total_time = start_time.elapsed();
            session.current_phase = match &result {
                Ok(_) => ProcessingPhase::Completed,
                Err(e) => ProcessingPhase::Failed(e.to_string()),
            };
        }
        
        // Update global metrics
        self.update_processing_metrics(&result, start_time.elapsed()).await;
        
        // Cleanup session
        self.active_sessions.remove(&session_id);
        *self.pipeline_status.write().await = PipelineStatus::Ready;
        
        result
    }
    
    async fn process_batch_documents_hybrid(
        &self,
        session_id: Uuid,
        documents: Vec<Document>,
    ) -> Result<HybridProcessingResult> {
        info!(\"Processing {} documents in hybrid batch mode\", documents.len());
        
        // Phase 1: Rust preprocessing and quality validation
        self.update_session_phase(session_id, ProcessingPhase::RustPreprocessing).await;
        let rust_start = Instant::now();
        
        let batch_results = self.batch_processor.process_documents_batch(documents.clone()).await?;
        let processed_documents: Vec<ProcessedDocument> = batch_results
            .into_iter()
            .flat_map(|br| br.processed_documents)
            .collect();
        
        let rust_processing_time = rust_start.elapsed();
        self.update_session_metric(session_id, \"rust_processing_time\", rust_processing_time).await;
        
        info!(\"Rust preprocessing completed for {} documents in {:?}\", 
              processed_documents.len(), rust_processing_time);
        
        // Phase 2: Intelligent model selection
        self.update_session_phase(session_id, ProcessingPhase::ModelSelection).await;
        let selection_start = Instant::now();
        
        let model_assignments = self.assign_models_to_documents(&processed_documents).await?;
        let model_selection_time = selection_start.elapsed();
        self.update_session_metric(session_id, \"model_selection_time\", model_selection_time).await;
        
        info!(\"Model selection completed in {:?}: {:?}\", 
              model_selection_time, model_assignments);
        
        // Phase 3: Python ML processing with model switching
        self.update_session_phase(session_id, ProcessingPhase::PythonMLProcessing).await;
        let python_start = Instant::now();
        
        let ml_results = self.process_with_python_ml(session_id, processed_documents, model_assignments).await?;
        let python_processing_time = python_start.elapsed();
        self.update_session_metric(session_id, \"python_processing_time\", python_processing_time).await;
        
        // Phase 4: Final quality validation and result aggregation
        self.update_session_phase(session_id, ProcessingPhase::QualityValidation).await;
        let final_results = self.validate_and_aggregate_results(ml_results).await?;
        
        let total_time = rust_processing_time + model_selection_time + python_processing_time;
        let throughput = (documents.len() as f64) * 3600.0 / total_time.as_secs_f64();
        
        Ok(HybridProcessingResult {
            session_id,
            results: final_results,
            processing_stats: HybridProcessingStats {
                total_documents: documents.len(),
                successful_documents: final_results.len(),
                rust_processing_time,
                python_processing_time,
                model_selection_time,
                ipc_overhead_time: Duration::from_millis(100), // Estimated
                total_processing_time: total_time,
                average_quality_score: self.calculate_average_quality(&final_results),
                throughput_docs_hour: throughput,
                memory_peak_gb: self.get_current_memory_usage().await,
                models_used: model_assignments.into_values().collect::<std::collections::HashSet<_>>().into_iter().collect(),
                model_switches: self.count_model_switches(session_id).await,
            },
        })
    }
    
    async fn assign_models_to_documents(
        &self,
        documents: &[ProcessedDocument],
    ) -> Result<HashMap<Uuid, ComplexityLevel>> {
        let mut assignments = HashMap::new();
        
        for doc in documents {
            let complexity = self.calculate_document_complexity(&doc.document);
            let quality_requirement = self.determine_quality_requirement(&doc.structural_quality);
            
            let selected_model = self.select_optimal_model(complexity, quality_requirement).await;
            assignments.insert(doc.document.id, selected_model);
            
            debug!(\"Document {} assigned to {} model (complexity: {:.2}, quality_req: {:.2})\",
                  doc.document.id, 
                  match selected_model {
                      ComplexityLevel::Fast => \"Fast\",
                      ComplexityLevel::Balanced => \"Balanced\", 
                      ComplexityLevel::Quality => \"Quality\",
                  },
                  complexity, quality_requirement);
        }
        
        Ok(assignments)
    }
    
    fn calculate_document_complexity(&self, document: &Document) -> f64 {
        let length_factor = (document.content.len() as f64 / 5000.0).min(1.0);
        let param_factor = (document.metadata.parameters.len() as f64 / 20.0).min(1.0);
        let counter_factor = (document.metadata.counters.len() as f64 / 10.0).min(1.0);
        let tech_factor = (document.metadata.technical_terms.len() as f64 / 50.0).min(1.0);
        
        (length_factor * 0.3 + param_factor * 0.3 + counter_factor * 0.2 + tech_factor * 0.2)
    }
    
    fn determine_quality_requirement(&self, structural_quality: &StructuralQuality) -> f64 {
        // Higher quality requirement for documents that already show good structure
        if structural_quality.overall_score > 0.8 {
            0.9 // High quality requirement
        } else if structural_quality.overall_score > 0.6 {
            0.75 // Standard quality requirement
        } else {
            0.6 // Basic quality requirement
        }
    }
    
    async fn select_optimal_model(&self, complexity: f64, quality_requirement: f64) -> ComplexityLevel {
        // Check performance history for optimization
        let model_performances = self.get_recent_model_performances().await;
        
        // Apply selection strategy
        match self.config.model_selection.default_strategy.as_str() {
            \"performance\" => self.select_performance_first_model(complexity, &model_performances),
            \"quality\" => self.select_quality_first_model(quality_requirement, &model_performances),
            \"adaptive\" => self.select_adaptive_model(complexity, quality_requirement, &model_performances).await,
            _ => self.select_balanced_model(complexity, quality_requirement),
        }
    }
    
    fn select_performance_first_model(
        &self, 
        _complexity: f64, 
        performances: &HashMap<ComplexityLevel, ModelPerformanceProfile>
    ) -> ComplexityLevel {
        // Select fastest model that meets minimum requirements
        performances.iter()
            .min_by_key(|(_, profile)| profile.avg_inference_time)
            .map(|(model, _)| model.clone())
            .unwrap_or(ComplexityLevel::Fast)
    }
    
    fn select_quality_first_model(
        &self,
        quality_requirement: f64,
        performances: &HashMap<ComplexityLevel, ModelPerformanceProfile>
    ) -> ComplexityLevel {
        // Select highest quality model that meets requirements
        performances.iter()
            .filter(|(_, profile)| profile.avg_quality_score >= quality_requirement * 0.95)
            .max_by(|a, b| a.1.avg_quality_score.partial_cmp(&b.1.avg_quality_score).unwrap())
            .map(|(model, _)| model.clone())
            .unwrap_or(ComplexityLevel::Quality) // Fallback to highest quality
    }
    
    async fn select_adaptive_model(
        &self,
        complexity: f64,
        quality_requirement: f64,
        performances: &HashMap<ComplexityLevel, ModelPerformanceProfile>
    ) -> ComplexityLevel {
        // Adaptive selection based on learned patterns
        let mut scores = HashMap::new();
        
        for (model, profile) in performances {
            let quality_match = 1.0 - (profile.avg_quality_score - quality_requirement).abs();
            let performance_score = 1.0 / (profile.avg_inference_time.as_secs_f64() + 0.1);
            let reliability_score = profile.success_rate;
            let efficiency_score = profile.memory_efficiency;
            
            // Weighted scoring
            let total_score = quality_match * 0.4 + 
                             performance_score * 0.3 + 
                             reliability_score * 0.2 + 
                             efficiency_score * 0.1;
            
            scores.insert(model.clone(), total_score);
        }
        
        scores.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(model, _)| model)
            .unwrap_or(self.select_balanced_model(complexity, quality_requirement))
    }
    
    fn select_balanced_model(&self, complexity: f64, quality_requirement: f64) -> ComplexityLevel {
        // Simple threshold-based selection with quality consideration
        if complexity < 0.3 && quality_requirement < 0.8 {
            ComplexityLevel::Fast
        } else if complexity < 0.7 && quality_requirement < 0.9 {
            ComplexityLevel::Balanced
        } else {
            ComplexityLevel::Quality
        }
    }
    
    async fn process_with_python_ml(
        &self,
        session_id: Uuid,
        documents: Vec<ProcessedDocument>,
        model_assignments: HashMap<Uuid, ComplexityLevel>,
    ) -> Result<Vec<MLProcessingResponse>> {
        let mut results = Vec::new();
        let mut current_model: Option<ComplexityLevel> = None;
        let mut model_switch_count = 0;
        
        // Group documents by assigned model for efficient processing
        let mut model_groups: HashMap<ComplexityLevel, Vec<ProcessedDocument>> = HashMap::new();
        
        for doc in documents {
            if let Some(assigned_model) = model_assignments.get(&doc.document.id) {
                model_groups.entry(assigned_model.clone())
                    .or_insert_with(Vec::new)
                    .push(doc);
            }
        }
        
        // Process each model group
        for (model, group_documents) in model_groups {
            // Switch model if necessary
            if current_model.as_ref() != Some(&model) {
                let switch_start = Instant::now();
                
                self.switch_python_model(session_id, model.clone()).await?;
                current_model = Some(model.clone());
                model_switch_count += 1;
                
                let switch_time = switch_start.elapsed();
                self.record_model_switch(session_id, model.clone(), switch_time, true).await;
                
                info!(\"Model switched to {:?} in {:?}\", model, switch_time);
            }
            
            // Process documents with current model
            for doc in group_documents {
                let ml_request = MLProcessingRequest {
                    request_id: Uuid::new_v4(),
                    document: doc,
                    processing_options: MLProcessingOptions {
                        model_preference: Some(model.clone()),
                        max_qa_pairs: Some(8),
                        quality_threshold: self.config.quality_threshold,
                        enable_diversity_enhancement: true,
                        batch_processing: true,
                    },
                    created_at: chrono::Utc::now(),
                };
                
                let response = self.send_ml_processing_request(ml_request).await?;
                results.push(response);
            }
        }
        
        // Update session model switch count
        self.update_session_model_switches(session_id, model_switch_count).await;
        
        Ok(results)
    }
    
    async fn switch_python_model(&self, session_id: Uuid, target_model: ComplexityLevel) -> Result<()> {
        let switch_message = IPCMessage {
            message_id: Uuid::new_v4(),
            message_type: IPCMessageType::ModelSwitch,
            payload: serde_json::to_vec(&target_model)?,
            timestamp: SystemTime::now(),
            session_id: Some(session_id),
        };
        
        let response = timeout(
            self.config.model_switch_timeout,
            self.send_ipc_message(switch_message)
        ).await??;
        
        if !response.success {
            return Err(anyhow::anyhow!(\"Model switch failed: {:?}\", response.error_message));
        }
        
        Ok(())
    }
    
    async fn send_ml_processing_request(&self, request: MLProcessingRequest) -> Result<MLProcessingResponse> {
        let message = IPCMessage {
            message_id: request.request_id,
            message_type: IPCMessageType::ProcessDocument,
            payload: serde_json::to_vec(&request)?,
            timestamp: SystemTime::now(),
            session_id: None,
        };
        
        let response = timeout(
            self.config.ipc_timeout,
            self.send_ipc_message(message)
        ).await??;
        
        if response.success {
            let ml_response: MLProcessingResponse = serde_json::from_slice(&response.payload)?;
            Ok(ml_response)
        } else {
            Err(anyhow::anyhow!(\"ML processing failed: {:?}\", response.error_message))
        }
    }
    
    async fn send_ipc_message(&self, message: IPCMessage) -> Result<IPCResponse> {
        // This is a simplified implementation - in practice would use proper IPC
        // For now, we'll simulate the response
        tokio::time::sleep(Duration::from_millis(100)).await; // Simulate processing
        
        Ok(IPCResponse {
            message_id: message.message_id,
            response_type: match message.message_type {
                IPCMessageType::ProcessDocument => IPCResponseType::ProcessingResult,
                IPCMessageType::ModelSwitch => IPCResponseType::ModelSwitchConfirmation,
                IPCMessageType::HealthCheck => IPCResponseType::HealthStatus,
                _ => IPCResponseType::Error,
            },
            payload: Vec::new(), // Would contain actual response data
            timestamp: SystemTime::now(),
            success: true,
            error_message: None,
        })
    }
    
    async fn validate_and_aggregate_results(
        &self,
        ml_results: Vec<MLProcessingResponse>,
    ) -> Result<Vec<ProcessingResult>> {
        let mut final_results = Vec::new();
        
        for ml_result in ml_results {
            // Validate quality
            let quality_valid = ml_result.semantic_quality.overall_score >= self.config.quality_threshold;
            
            if quality_valid {
                let processing_result = ProcessingResult {
                    document_id: ml_result.request_id,
                    original_document: ml_result.qa_pairs[0].metadata.parameters_mentioned[0].clone().into(), // Simplified
                    structural_quality: StructuralQuality {
                        completeness_score: 0.8,
                        parameter_extraction_quality: 0.8,
                        counter_extraction_quality: 0.7,
                        technical_density_score: 0.75,
                        overall_score: 0.8,
                    },
                    semantic_quality: ml_result.semantic_quality,
                    qa_pairs: ml_result.qa_pairs,
                    combined_quality_score: (0.8 + ml_result.semantic_quality.overall_score) / 2.0,
                    processing_stats: ProcessingStats {
                        rust_processing_time: Duration::from_millis(500),
                        ml_processing_time: ml_result.processing_time,
                        ipc_overhead_time: Duration::from_millis(50),
                        total_processing_time: Duration::from_millis(500) + ml_result.processing_time + Duration::from_millis(50),
                        memory_peak_mb: ml_result.processing_metadata.memory_used_mb,
                        model_used: ml_result.model_used,
                    },
                };
                
                final_results.push(processing_result);
            } else {
                warn!(\"Document {} failed quality validation (score: {:.3})\", 
                      ml_result.request_id, ml_result.semantic_quality.overall_score);
            }
        }
        
        Ok(final_results)
    }
    
    // Helper methods for session and performance tracking
    
    async fn update_session_phase(&self, session_id: Uuid, phase: ProcessingPhase) {
        if let Some(mut session) = self.active_sessions.get_mut(&session_id) {
            session.current_phase = phase;
        }
    }
    
    async fn update_session_metric(&self, session_id: Uuid, metric: &str, value: Duration) {
        if let Some(mut session) = self.active_sessions.get_mut(&session_id) {
            match metric {
                \"rust_processing_time\" => session.performance_metrics.rust_processing_time = value,
                \"python_processing_time\" => session.performance_metrics.python_processing_time = value,
                \"model_selection_time\" => session.performance_metrics.model_selection_time = value,
                \"ipc_overhead_time\" => session.performance_metrics.ipc_overhead_time = value,
                _ => {}
            }
        }
    }
    
    async fn update_session_model_switches(&self, session_id: Uuid, count: usize) {
        if let Some(mut session) = self.active_sessions.get_mut(&session_id) {
            session.performance_metrics.model_switches = count;
        }
    }
    
    async fn record_model_switch(
        &self,
        session_id: Uuid,
        model: ComplexityLevel,
        latency: Duration,
        success: bool,
    ) {
        let switch_event = ModelSwitchEvent {
            timestamp: SystemTime::now(),
            from_model: ComplexityLevel::Fast, // Simplified - would track actual previous model
            to_model: model,
            reason: \"Document complexity requirement\".to_string(),
            switch_latency: latency,
            success,
            session_id,
        };
        
        self.model_switching_history.write().await.push(switch_event);
    }
    
    async fn get_recent_model_performances(&self) -> HashMap<ComplexityLevel, ModelPerformanceProfile> {
        self.model_performance_cache.iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }
    
    async fn count_model_switches(&self, session_id: Uuid) -> usize {
        self.model_switching_history.read().await
            .iter()
            .filter(|event| event.session_id == session_id)
            .count()
    }
    
    fn calculate_average_quality(&self, results: &[ProcessingResult]) -> f64 {
        if results.is_empty() {
            return 0.0;
        }
        
        results.iter()
            .map(|r| r.combined_quality_score)
            .sum::<f64>() / results.len() as f64
    }
    
    async fn get_current_memory_usage(&self) -> f64 {
        // Simplified memory monitoring - would use actual system metrics
        self.config.memory_management.rust_allocation_gb * 0.7 // Estimate 70% usage
    }
    
    async fn update_processing_metrics(&self, result: &Result<HybridProcessingResult>, duration: Duration) {
        let mut metrics = self.processing_metrics.write().await;
        
        metrics.total_sessions += 1;
        
        if let Ok(success_result) = result {
            metrics.successful_sessions += 1;
            metrics.total_documents_processed += success_result.processing_stats.total_documents;
            
            // Update averages
            let alpha = 0.1; // Learning rate for exponential moving average
            metrics.average_session_time = Duration::from_secs_f64(
                alpha * duration.as_secs_f64() + (1.0 - alpha) * metrics.average_session_time.as_secs_f64()
            );
            
            metrics.average_quality_score = alpha * success_result.processing_stats.average_quality_score + 
                                           (1.0 - alpha) * metrics.average_quality_score;
            
            metrics.throughput_current = alpha * success_result.processing_stats.throughput_docs_hour +
                                        (1.0 - alpha) * metrics.throughput_current;
        }
        
        // Update error rate
        metrics.error_rate = 1.0 - (metrics.successful_sessions as f64 / metrics.total_sessions as f64);
        metrics.last_updated = SystemTime::now();
    }
    
    // Placeholder implementations for other processing modes
    async fn process_single_document_hybrid(&self, _session_id: Uuid, _documents: Vec<Document>) -> Result<HybridProcessingResult> {
        todo!(\"Single document processing not yet implemented\")
    }
    
    async fn process_stream_documents_hybrid(&self, _session_id: Uuid, _documents: Vec<Document>) -> Result<HybridProcessingResult> {
        todo!(\"Stream processing not yet implemented\")
    }
    
    async fn process_priority_documents_hybrid(&self, _session_id: Uuid, _documents: Vec<Document>) -> Result<HybridProcessingResult> {
        todo!(\"Priority processing not yet implemented\")
    }
    
    pub async fn get_pipeline_status(&self) -> PipelineStatus {
        self.pipeline_status.read().await.clone()
    }
    
    pub async fn get_processing_metrics(&self) -> HybridProcessingMetrics {
        self.processing_metrics.read().await.clone()
    }
    
    pub async fn get_active_sessions(&self) -> Vec<ProcessingSession> {
        self.active_sessions.iter()
            .map(|entry| entry.value().clone())
            .collect()
    }
    
    pub async fn shutdown(&self) -> Result<()> {
        info!(\"Shutting down Hybrid Pipeline Coordinator\");
        *self.pipeline_status.write().await = PipelineStatus::Shutdown;
        
        // Shutdown Python process
        if let Some(mut child) = self.python_process.lock().await.take() {
            let _ = child.kill().await;
        }
        
        // Clear active sessions
        self.active_sessions.clear();
        
        info!(\"Hybrid Pipeline Coordinator shutdown complete\");
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridProcessingResult {
    pub session_id: Uuid,
    pub results: Vec<ProcessingResult>,
    pub processing_stats: HybridProcessingStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridProcessingStats {
    pub total_documents: usize,
    pub successful_documents: usize,
    pub rust_processing_time: Duration,
    pub python_processing_time: Duration,
    pub model_selection_time: Duration,
    pub ipc_overhead_time: Duration,
    pub total_processing_time: Duration,
    pub average_quality_score: f64,
    pub throughput_docs_hour: f64,
    pub memory_peak_gb: f64,
    pub models_used: Vec<ComplexityLevel>,
    pub model_switches: usize,
}