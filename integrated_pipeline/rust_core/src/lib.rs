pub mod config;
pub mod document_processor;
pub mod ipc_manager;
pub mod quality_validator;
pub mod types;
// MCP (Model Context Protocol) modules
pub mod mcp_server;
pub mod mcp_rpc_impl;

pub use config::ProcessingConfig;
pub use document_processor::DocumentProcessor;
pub use ipc_manager::IpcManager;
pub use quality_validator::QualityValidator;
pub use types::*;
// MCP exports
pub use mcp_server::{McpServer, McpServerConfig};
pub use mcp_rpc_impl::McpRpcImpl;