"""
MCP (Model Context Protocol) Implementation for Python Pipeline Coordination

This package provides a complete MCP implementation for coordinating Python
machine learning pipelines and agents.

Components:
- protocol: MCP protocol definitions and schemas
- server: MCP server for pipeline coordination
- client: MCP client for agent communication  
- host: MCP host for pipeline orchestration
- handlers: Request/response handlers
- monitoring: Real-time performance monitoring

Usage:
    from mcp import MCPServer, MCPClient, MCPHost
    from mcp.monitoring import MCPMonitoringService
    
    # Start MCP server
    server = MCPServer(port=8700)
    await server.start()
    
    # Connect MCP client
    client = MCPClient("agent-001", "My Agent", "worker")
    await client.connect()
    
    # Create MCP host for orchestration
    host = MCPHost()
    await host.start()
"""

from .protocol import (
    MCPMessage,
    MCPMessageType, 
    MCPMethods,
    MCPErrors,
    PipelineStatus,
    TaskStatus,
    PipelineInfo,
    TaskInfo,
    AgentInfo,
    ModelMetrics,
    create_request,
    create_response,
    create_notification,
    MCP_VERSION,
    PROTOCOL_NAME
)

from .server import MCPServer
from .client import MCPClient, MCPClientManager, ExampleAgent
from .host import MCPHost, OrchestrationStrategy, OrchestrationConfig
from .handlers import MCPRequestHandler, InMemoryStorage
from .monitoring import (
    MCPMonitoringService,
    SystemMetrics,
    PipelineMetrics,
    AgentMetrics,
    ModelPerformanceMetrics,
    SystemMetricsCollector,
    PipelineMetricsCollector,
    AgentMetricsCollector,
    ModelMetricsCollector,
    AlertSystem
)

__version__ = "1.0.0"
__author__ = "Claude Flow MCP Team"

# Default configuration
DEFAULT_CONFIG = {
    "server_port": 8700,
    "client_port": 8701,
    "host_port": 8702,
    "max_message_size": 1024 * 1024,  # 1MB
    "default_timeout": 30.0,
    "heartbeat_interval": 5.0,
    "max_retry_attempts": 3
}

# Export all main classes
__all__ = [
    # Protocol
    "MCPMessage",
    "MCPMessageType",
    "MCPMethods", 
    "MCPErrors",
    "PipelineStatus",
    "TaskStatus",
    "PipelineInfo",
    "TaskInfo",
    "AgentInfo",
    "ModelMetrics",
    "create_request",
    "create_response",
    "create_notification",
    "MCP_VERSION",
    "PROTOCOL_NAME",
    
    # Core components
    "MCPServer",
    "MCPClient", 
    "MCPClientManager",
    "MCPHost",
    "MCPRequestHandler",
    "InMemoryStorage",
    
    # Orchestration
    "OrchestrationStrategy",
    "OrchestrationConfig",
    "ExampleAgent",
    
    # Monitoring
    "MCPMonitoringService",
    "SystemMetrics",
    "PipelineMetrics", 
    "AgentMetrics",
    "ModelPerformanceMetrics",
    "SystemMetricsCollector",
    "PipelineMetricsCollector",
    "AgentMetricsCollector", 
    "ModelMetricsCollector",
    "AlertSystem",
    
    # Configuration
    "DEFAULT_CONFIG"
]