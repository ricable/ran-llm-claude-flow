pub mod config;
pub mod types;
pub mod simple_document_processor;

// Optional modules that may have compilation issues
#[cfg(feature = "full")]
pub mod document_processor;
#[cfg(feature = "full")]
pub mod ipc_manager;
#[cfg(feature = "full")]
pub mod quality_validator;
#[cfg(feature = "full")]
pub mod shared_memory;
#[cfg(feature = "full")]
pub mod performance_monitor;
#[cfg(feature = "full")]
pub mod numa_optimizer;
#[cfg(feature = "full")]
pub mod mcp_server;
#[cfg(feature = "full")]
pub mod mcp_rpc_impl;
#[cfg(feature = "full")]
pub mod ml_integration;

pub use config::ProcessingConfig;
pub use simple_document_processor::SimpleDocumentProcessor;
pub use types::*;

#[cfg(feature = "full")]
pub use document_processor::DocumentProcessor;
#[cfg(feature = "full")]
pub use ipc_manager::IpcManager;
#[cfg(feature = "full")]
pub use quality_validator::QualityValidator;
#[cfg(feature = "full")]
pub use mcp_server::{McpServer, McpServerConfig};
#[cfg(feature = "full")]
pub use mcp_rpc_impl::McpRpcImpl;
#[cfg(feature = "full")]
pub use ml_integration::{MLIntegrationManager, MLIntegrationConfig};