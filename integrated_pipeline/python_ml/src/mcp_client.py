#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Client Implementation

This module provides a Python client for communicating with the Rust MCP server,
integrating with the existing ML pipeline and maintaining high-performance
zero-copy data transfers where possible.

Key Features:
- JSON-RPC 2.0 based MCP protocol client
- WebSocket and HTTP transport support  
- Integration with existing IPC system
- Automatic reconnection and error recovery
- Resource discovery and management
- Tool execution and result handling
- Prompt template management
- Performance monitoring and metrics

Author: Claude Code
Version: 1.0.0
"""

import asyncio
import json
import logging
import time
import uuid
import websockets
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# Import existing IPC components
from .ipc_client import IPCClient, MessageType as IPCMessageType
from .model_manager import ModelManager
from .mlx_accelerator import MLXAccelerator


class McpTransport(Enum):
    """MCP transport protocols"""
    WEBSOCKET = "websocket"
    HTTP = "http"
    STDIO = "stdio"


@dataclass 
class McpClientConfig:
    """MCP client configuration"""
    server_uri: str = "ws://127.0.0.1:8000"
    transport: McpTransport = McpTransport.WEBSOCKET
    client_name: str = "Python ML Engine"
    client_version: str = "1.0.0"
    max_retries: int = 5
    retry_delay: float = 1.0
    request_timeout: float = 30.0
    enable_logging: bool = True
    log_level: str = "INFO"


@dataclass
class McpCapabilities:
    """Client capabilities for MCP handshake"""
    experimental: Optional[Dict[str, Any]] = None
    sampling: Optional[Dict[str, Any]] = None
    roots: Optional[Dict[str, Any]] = None


@dataclass
class McpRequest:
    """MCP JSON-RPC request structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": self.id,
            "method": self.method,
            "params": self.params
        }


@dataclass 
class McpResponse:
    """MCP JSON-RPC response structure"""
    id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'McpResponse':
        return cls(
            id=data.get("id", ""),
            result=data.get("result"),
            error=data.get("error")
        )


@dataclass
class McpResource:
    """MCP resource definition"""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None
    annotations: Optional[Dict[str, Any]] = None


@dataclass
class McpTool:
    """MCP tool definition"""
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class McpPrompt:
    """MCP prompt template"""
    name: str
    description: Optional[str] = None
    arguments: Optional[List[Dict[str, Any]]] = None


class McpClient:
    """
    MCP Client for communicating with Rust MCP server.
    
    Provides high-level interface for MCP protocol operations while
    integrating with existing ML pipeline components.
    """
    
    def __init__(self, config: McpClientConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Connection state
        self.connected = False
        self.session_id: Optional[str] = None
        self.server_capabilities: Optional[Dict[str, Any]] = None
        
        # Transport connections
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        # Request/response handling
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.request_counter = 0
        
        # Discovered resources, tools, and prompts
        self.resources: Dict[str, McpResource] = {}
        self.tools: Dict[str, McpTool] = {}
        self.prompts: Dict[str, McpPrompt] = {}
        
        # Performance metrics
        self.metrics = {
            'requests_sent': 0,
            'requests_successful': 0,
            'requests_failed': 0,
            'avg_response_time': 0.0,
            'total_bytes_sent': 0,
            'total_bytes_received': 0,
            'connections': 0,
            'reconnections': 0
        }
        
        # Background tasks
        self.message_handler_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        
        # Integration components
        self.ipc_client: Optional[IPCClient] = None
        self.model_manager: Optional[ModelManager] = None
        self.mlx_accelerator: Optional[MLXAccelerator] = None
        
        # Executor for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self) -> bool:
        """Initialize the MCP client and establish connection"""
        self.logger.info(f"Initializing MCP client for {self.config.server_uri}")
        
        try:
            # Initialize integration components
            if not await self._initialize_components():
                self.logger.error("Failed to initialize integration components")
                return False
                
            # Establish MCP connection
            if not await self._connect():
                self.logger.error("Failed to establish MCP connection")
                return False
                
            # Perform MCP handshake
            if not await self._handshake():
                self.logger.error("MCP handshake failed")
                return False
                
            # Discover server capabilities
            await self._discover_capabilities()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info("MCP client initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP client: {e}")
            return False
    
    async def _initialize_components(self) -> bool:
        """Initialize integration components"""
        try:
            # Initialize IPC client for integration
            self.ipc_client = IPCClient()
            if not await self.ipc_client.initialize():
                self.logger.warning("IPC client initialization failed - some features may be limited")
            
            # Initialize model manager
            self.model_manager = ModelManager()
            if not await self.model_manager.initialize():
                self.logger.warning("Model manager initialization failed")
            
            # Initialize MLX accelerator
            self.mlx_accelerator = MLXAccelerator()
            if not await self.mlx_accelerator.initialize():
                self.logger.warning("MLX accelerator initialization failed")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            return False
    
    async def _connect(self) -> bool:
        """Establish transport connection"""
        retry_count = 0
        
        while retry_count < self.config.max_retries:
            try:
                if self.config.transport == McpTransport.WEBSOCKET:
                    self.websocket = await websockets.connect(
                        self.config.server_uri,
                        ping_interval=30,
                        ping_timeout=10
                    )
                    self.connected = True
                    self.metrics['connections'] += 1
                    if retry_count > 0:
                        self.metrics['reconnections'] += 1
                    return True
                    
                elif self.config.transport == McpTransport.HTTP:
                    self.http_session = aiohttp.ClientSession()
                    # Test connection with a simple request
                    async with self.http_session.post(self.config.server_uri) as resp:
                        if resp.status == 200:
                            self.connected = True
                            return True
                            
            except Exception as e:
                retry_count += 1
                self.logger.warning(f"Connection attempt {retry_count} failed: {e}")
                
                if retry_count < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * retry_count)
                    
        return False
    
    async def _handshake(self) -> bool:
        """Perform MCP initialization handshake"""
        try:
            capabilities = McpCapabilities(
                experimental={
                    "mlx_acceleration": True,
                    "shared_memory_integration": True,
                    "m3_max_optimization": True
                },
                sampling={},
                roots={}
            )
            
            request = McpRequest(
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": capabilities.__dict__,
                    "clientInfo": {
                        "name": self.config.client_name,
                        "version": self.config.client_version
                    }
                }
            )
            
            response = await self._send_request(request)
            if response and not response.error:
                self.server_capabilities = response.result
                self.session_id = str(uuid.uuid4())
                self.logger.info("MCP handshake successful")
                return True
            else:
                self.logger.error(f"Handshake failed: {response.error if response else 'No response'}")
                return False
                
        except Exception as e:
            self.logger.error(f"Handshake exception: {e}")
            return False
    
    async def _discover_capabilities(self):
        """Discover server resources, tools, and prompts"""
        try:
            # Discover resources
            if self.server_capabilities and self.server_capabilities.get("capabilities", {}).get("resources"):
                await self._discover_resources()
            
            # Discover tools
            if self.server_capabilities and self.server_capabilities.get("capabilities", {}).get("tools"):
                await self._discover_tools()
            
            # Discover prompts
            if self.server_capabilities and self.server_capabilities.get("capabilities", {}).get("prompts"):
                await self._discover_prompts()
                
        except Exception as e:
            self.logger.error(f"Capability discovery failed: {e}")
    
    async def _discover_resources(self):
        """Discover available resources"""
        request = McpRequest(method="resources/list", params={})
        response = await self._send_request(request)
        
        if response and response.result:
            for resource_data in response.result.get("resources", []):
                resource = McpResource(
                    uri=resource_data["uri"],
                    name=resource_data["name"],
                    description=resource_data.get("description"),
                    mime_type=resource_data.get("mimeType"),
                    annotations=resource_data.get("annotations")
                )
                self.resources[resource.name] = resource
                
            self.logger.info(f"Discovered {len(self.resources)} resources")
    
    async def _discover_tools(self):
        """Discover available tools"""
        request = McpRequest(method="tools/list", params={})
        response = await self._send_request(request)
        
        if response and response.result:
            for tool_data in response.result.get("tools", []):
                tool = McpTool(
                    name=tool_data["name"],
                    description=tool_data.get("description"),
                    input_schema=tool_data.get("inputSchema", {})
                )
                self.tools[tool.name] = tool
                
            self.logger.info(f"Discovered {len(self.tools)} tools")
    
    async def _discover_prompts(self):
        """Discover available prompts"""
        request = McpRequest(method="prompts/list", params={})
        response = await self._send_request(request)
        
        if response and response.result:
            for prompt_data in response.result.get("prompts", []):
                prompt = McpPrompt(
                    name=prompt_data["name"],
                    description=prompt_data.get("description"),
                    arguments=prompt_data.get("arguments")
                )
                self.prompts[prompt.name] = prompt
                
            self.logger.info(f"Discovered {len(self.prompts)} prompts")
    
    async def _start_background_tasks(self):
        """Start background tasks for connection management"""
        if self.config.transport == McpTransport.WEBSOCKET:
            self.message_handler_task = asyncio.create_task(self._websocket_message_handler())
        
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def _websocket_message_handler(self):
        """Handle incoming WebSocket messages"""
        try:
            while self.connected and self.websocket:
                try:
                    message = await self.websocket.recv()
                    self.metrics['total_bytes_received'] += len(message)
                    
                    data = json.loads(message)
                    
                    # Handle response
                    if "id" in data and data["id"] in self.pending_requests:
                        response = McpResponse.from_dict(data)
                        future = self.pending_requests.pop(data["id"])
                        future.set_result(response)
                    
                    # Handle notifications/requests from server
                    elif "method" in data:
                        await self._handle_server_request(data)
                        
                except websockets.exceptions.ConnectionClosed:
                    self.logger.info("WebSocket connection closed")
                    self.connected = False
                    break
                except Exception as e:
                    self.logger.error(f"Message handling error: {e}")
                    
        except Exception as e:
            self.logger.error(f"WebSocket message handler failed: {e}")
            self.connected = False
    
    async def _handle_server_request(self, data: Dict[str, Any]):
        """Handle requests from the server"""
        method = data.get("method", "")
        
        if method == "notifications/initialized":
            self.logger.info("Server initialization notification received")
        elif method.startswith("logging/"):
            # Handle logging notifications
            self.logger.debug(f"Server log: {data}")
        else:
            self.logger.debug(f"Unhandled server request: {method}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to maintain connection"""
        while self.connected:
            try:
                if self.config.transport == McpTransport.WEBSOCKET and self.websocket:
                    await self.websocket.ping()
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
            except Exception as e:
                self.logger.error(f"Heartbeat failed: {e}")
                await asyncio.sleep(5)
    
    async def _send_request(self, request: McpRequest) -> Optional[McpResponse]:
        """Send MCP request and wait for response"""
        if not self.connected:
            self.logger.error("Cannot send request - not connected")
            return None
        
        start_time = time.time()
        
        try:
            message = json.dumps(request.to_dict())
            self.metrics['requests_sent'] += 1
            self.metrics['total_bytes_sent'] += len(message)
            
            if self.config.transport == McpTransport.WEBSOCKET:
                # Create future for response
                response_future = asyncio.Future()
                self.pending_requests[request.id] = response_future
                
                # Send request
                await self.websocket.send(message)
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(
                        response_future, 
                        timeout=self.config.request_timeout
                    )
                    
                    response_time = time.time() - start_time
                    alpha = 0.1
                    self.metrics['avg_response_time'] = (
                        alpha * response_time + 
                        (1 - alpha) * self.metrics['avg_response_time']
                    )
                    
                    if response.error:
                        self.metrics['requests_failed'] += 1
                    else:
                        self.metrics['requests_successful'] += 1
                    
                    return response
                    
                except asyncio.TimeoutError:
                    self.logger.error(f"Request timeout: {request.method}")
                    self.pending_requests.pop(request.id, None)
                    self.metrics['requests_failed'] += 1
                    return None
                    
            elif self.config.transport == McpTransport.HTTP:
                async with self.http_session.post(
                    self.config.server_uri,
                    json=request.to_dict(),
                    timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
                ) as resp:
                    if resp.status == 200:
                        response_data = await resp.json()
                        response_time = time.time() - start_time
                        
                        alpha = 0.1
                        self.metrics['avg_response_time'] = (
                            alpha * response_time + 
                            (1 - alpha) * self.metrics['avg_response_time']
                        )
                        
                        response = McpResponse.from_dict(response_data)
                        
                        if response.error:
                            self.metrics['requests_failed'] += 1
                        else:
                            self.metrics['requests_successful'] += 1
                            
                        return response
                    else:
                        self.logger.error(f"HTTP request failed: {resp.status}")
                        self.metrics['requests_failed'] += 1
                        return None
                        
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            self.metrics['requests_failed'] += 1
            return None
    
    async def read_resource(self, uri: str) -> Optional[Dict[str, Any]]:
        """Read resource content by URI"""
        request = McpRequest(
            method="resources/read",
            params={"uri": uri}
        )
        
        response = await self._send_request(request)
        if response and response.result:
            return response.result
        else:
            self.logger.error(f"Failed to read resource: {uri}")
            return None
    
    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Execute a tool with given arguments"""
        if name not in self.tools:
            self.logger.error(f"Tool not found: {name}")
            return None
        
        request = McpRequest(
            method="tools/call",
            params={
                "name": name,
                "arguments": arguments or {}
            }
        )
        
        response = await self._send_request(request)
        if response and response.result:
            return response.result
        else:
            self.logger.error(f"Tool execution failed: {name}")
            return None
    
    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get prompt template with arguments"""
        if name not in self.prompts:
            self.logger.error(f"Prompt not found: {name}")
            return None
        
        request = McpRequest(
            method="prompts/get",
            params={
                "name": name,
                "arguments": arguments or {}
            }
        )
        
        response = await self._send_request(request)
        if response and response.result:
            return response.result
        else:
            self.logger.error(f"Failed to get prompt: {name}")
            return None
    
    async def process_document_via_mcp(self, content: str, **options) -> Optional[Dict[str, Any]]:
        """Process document using MCP tool integration"""
        return await self.call_tool("process-document", {
            "content": content,
            "options": options
        })
    
    async def benchmark_performance(self, test_type: str, iterations: int = 10) -> Optional[Dict[str, Any]]:
        """Run performance benchmark via MCP tool"""
        return await self.call_tool("benchmark-performance", {
            "test_type": test_type,
            "iterations": iterations
        })
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        # Get MCP metrics
        resource_data = await self.read_resource("mcp://rust-core/performance-metrics")
        mcp_metrics = resource_data.get("contents", [{}])[0].get("text", {}) if resource_data else {}
        
        # Combine with local metrics
        return {
            "mcp_client": self.metrics,
            "rust_server": json.loads(mcp_metrics) if isinstance(mcp_metrics, str) else mcp_metrics,
            "connection_status": {
                "connected": self.connected,
                "session_id": self.session_id,
                "transport": self.config.transport.value,
                "server_uri": self.config.server_uri
            },
            "discovered_capabilities": {
                "resources": len(self.resources),
                "tools": len(self.tools), 
                "prompts": len(self.prompts)
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health = {
            "healthy": True,
            "issues": [],
            "warnings": []
        }
        
        # Check connection
        if not self.connected:
            health["healthy"] = False
            health["issues"].append("Not connected to MCP server")
        
        # Check pending requests
        if len(self.pending_requests) > 10:
            health["warnings"].append("High number of pending requests")
        
        # Check error rate
        total_requests = self.metrics["requests_sent"]
        if total_requests > 0:
            error_rate = self.metrics["requests_failed"] / total_requests
            if error_rate > 0.1:
                health["warnings"].append(f"High error rate: {error_rate:.1%}")
        
        # Test basic functionality
        try:
            # Try to list resources
            request = McpRequest(method="resources/list", params={})
            response = await self._send_request(request)
            if not response or response.error:
                health["issues"].append("Cannot communicate with server")
                health["healthy"] = False
        except Exception as e:
            health["issues"].append(f"Health check communication failed: {str(e)}")
            health["healthy"] = False
        
        return health
    
    async def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up MCP client")
        
        self.connected = False
        
        # Cancel background tasks
        for task in [self.message_handler_task, self.heartbeat_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close connections
        if self.websocket:
            await self.websocket.close()
        if self.http_session:
            await self.http_session.close()
        
        # Cleanup integration components
        if self.ipc_client:
            await self.ipc_client.cleanup()
        if self.model_manager:
            await self.model_manager.cleanup()
        if self.mlx_accelerator:
            await self.mlx_accelerator.cleanup()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("MCP client cleanup complete")


# Example usage and testing
if __name__ == "__main__":
    async def test_mcp_client():
        """Test the MCP client"""
        logging.basicConfig(level=logging.INFO)
        
        config = McpClientConfig(
            server_uri="ws://127.0.0.1:8000",
            transport=McpTransport.WEBSOCKET
        )
        
        client = McpClient(config)
        
        # Initialize client
        success = await client.initialize()
        print(f"Client initialized: {success}")
        
        if success:
            # Test resource access
            print("\nAvailable resources:")
            for name, resource in client.resources.items():
                print(f"  - {name}: {resource.description}")
            
            # Test tool execution
            print("\nAvailable tools:")
            for name, tool in client.tools.items():
                print(f"  - {name}: {tool.description}")
            
            # Test document processing
            if "process-document" in client.tools:
                result = await client.process_document_via_mcp(
                    "Test document for MCP integration testing",
                    quality_threshold=0.8,
                    model_preference="qwen3-7b"
                )
                print(f"\nDocument processing result: {result is not None}")
            
            # Performance metrics
            metrics = await client.get_performance_metrics()
            print(f"\nPerformance metrics: {metrics.get('mcp_client', {}).get('requests_successful', 0)} successful requests")
            
            # Health check
            health = await client.health_check()
            print(f"Health check: {'Healthy' if health['healthy'] else 'Issues detected'}")
        
        await client.cleanup()
    
    asyncio.run(test_mcp_client())