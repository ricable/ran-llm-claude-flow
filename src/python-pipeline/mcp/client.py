"""
MCP Client for Inter-Agent Communication

This module implements the Model Context Protocol client that enables agents
to communicate with the MCP server and coordinate pipeline operations.
"""

import asyncio
import json
import logging
import websockets
import uuid
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from .protocol import (
    MCPMessage, MCPMessageType, MCPMethods,
    create_request, create_notification,
    DEFAULT_CLIENT_PORT
)


class MCPClient:
    """MCP Client for agent communication"""
    
    def __init__(self, agent_id: str, agent_name: str, agent_type: str,
                 server_host: str = "localhost", server_port: int = 8700):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.server_host = server_host
        self.server_port = server_port
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.handlers: Dict[str, Callable] = {}
        self.capabilities: List[str] = []
        
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
    
    async def connect(self, capabilities: Optional[List[str]] = None):
        """Connect to MCP server and register agent"""
        self.capabilities = capabilities or []
        
        try:
            uri = f"ws://{self.server_host}:{self.server_port}"
            self.websocket = await websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=10
            )
            self.connected = True
            self.logger.info(f"Connected to MCP server at {uri}")
            
            # Start message handling task
            asyncio.create_task(self._handle_messages())
            
            # Register with server
            await self._register_agent()
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.connected and self.websocket:
            try:
                # Unregister from server
                await self._unregister_agent()
                
                await self.websocket.close()
                self.connected = False
                self.logger.info("Disconnected from MCP server")
            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")
    
    async def _handle_messages(self):
        """Handle incoming messages from server"""
        try:
            async for raw_message in self.websocket:
                try:
                    message = MCPMessage.from_json(raw_message)
                    await self._process_message(message)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON received: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
        except websockets.exceptions.ConnectionClosed:
            self.connected = False
            self.logger.info("Connection to server closed")
        except Exception as e:
            self.logger.error(f"Error in message handler: {e}")
    
    async def _process_message(self, message: MCPMessage):
        """Process incoming MCP message"""
        if message.type == MCPMessageType.RESPONSE:
            # Handle response to our request
            if message.id in self.pending_requests:
                future = self.pending_requests.pop(message.id)
                if message.error:
                    future.set_exception(Exception(f"MCP Error: {message.error}"))
                else:
                    future.set_result(message.result)
        
        elif message.type == MCPMessageType.NOTIFICATION:
            # Handle server notification
            await self._handle_notification(message)
        
        elif message.type == MCPMessageType.REQUEST:
            # Server is requesting something from us
            await self._handle_server_request(message)
    
    async def _handle_notification(self, message: MCPMessage):
        """Handle notification from server"""
        self.logger.debug(f"Received notification: {message.method}")
        
        # Call registered handler if available
        if message.method in self.handlers:
            try:
                await self.handlers[message.method](message.params or {})
            except Exception as e:
                self.logger.error(f"Error in notification handler: {e}")
    
    async def _handle_server_request(self, message: MCPMessage):
        """Handle request from server"""
        # This would typically be task assignments or status queries
        self.logger.info(f"Server request: {message.method}")
        
        # For now, just acknowledge
        response = MCPMessage(
            id=message.id,
            type=MCPMessageType.RESPONSE,
            method="",
            timestamp=datetime.now().isoformat(),
            result={"status": "acknowledged"}
        )
        
        if self.websocket:
            await self.websocket.send(response.to_json())
    
    async def _register_agent(self):
        """Register this agent with the server"""
        try:
            result = await self.send_request(MCPMethods.AGENT_REGISTER, {
                "id": self.agent_id,
                "name": self.agent_name,
                "type": self.agent_type,
                "capabilities": self.capabilities
            })
            self.logger.info(f"Agent registered: {result}")
        except Exception as e:
            self.logger.error(f"Failed to register agent: {e}")
            raise
    
    async def _unregister_agent(self):
        """Unregister this agent from the server"""
        try:
            await self.send_request(MCPMethods.AGENT_UNREGISTER, {
                "agent_id": self.agent_id
            })
            self.logger.info("Agent unregistered")
        except Exception as e:
            self.logger.error(f"Failed to unregister agent: {e}")
    
    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send request to server and wait for response"""
        if not self.connected or not self.websocket:
            raise Exception("Not connected to server")
        
        request = create_request(method, params)
        future = asyncio.Future()
        self.pending_requests[request.id] = future
        
        try:
            await self.websocket.send(request.to_json())
            result = await asyncio.wait_for(future, timeout=30.0)
            return result or {}
        except asyncio.TimeoutError:
            if request.id in self.pending_requests:
                del self.pending_requests[request.id]
            raise Exception(f"Request timeout for method {method}")
        except Exception as e:
            if request.id in self.pending_requests:
                del self.pending_requests[request.id]
            raise
    
    async def send_notification(self, method: str, params: Optional[Dict[str, Any]] = None):
        """Send notification to server (no response expected)"""
        if not self.connected or not self.websocket:
            raise Exception("Not connected to server")
        
        notification = create_notification(method, params)
        await self.websocket.send(notification.to_json())
    
    def register_handler(self, method: str, handler: Callable):
        """Register handler for server notifications"""
        self.handlers[method] = handler
    
    # Pipeline operations
    async def create_pipeline(self, name: str, config: Dict[str, Any]) -> str:
        """Create a new pipeline"""
        result = await self.send_request(MCPMethods.PIPELINE_CREATE, {
            "name": name,
            "config": config
        })
        return result.get("pipeline_id")
    
    async def start_pipeline(self, pipeline_id: str) -> bool:
        """Start a pipeline"""
        result = await self.send_request(MCPMethods.PIPELINE_START, {
            "pipeline_id": pipeline_id
        })
        return result.get("status") == "started"
    
    async def stop_pipeline(self, pipeline_id: str) -> bool:
        """Stop a pipeline"""
        result = await self.send_request(MCPMethods.PIPELINE_STOP, {
            "pipeline_id": pipeline_id
        })
        return result.get("status") == "stopped"
    
    async def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get pipeline status"""
        return await self.send_request(MCPMethods.PIPELINE_STATUS, {
            "pipeline_id": pipeline_id
        })
    
    async def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all pipelines"""
        result = await self.send_request(MCPMethods.PIPELINE_LIST)
        return result.get("pipelines", [])
    
    # Task operations
    async def create_task(self, pipeline_id: str, name: str, config: Dict[str, Any]) -> str:
        """Create a new task"""
        result = await self.send_request(MCPMethods.TASK_CREATE, {
            "pipeline_id": pipeline_id,
            "name": name,
            "config": config
        })
        return result.get("task_id")
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status"""
        return await self.send_request(MCPMethods.TASK_STATUS, {
            "task_id": task_id
        })
    
    async def get_task_result(self, task_id: str) -> Any:
        """Get task result"""
        result = await self.send_request(MCPMethods.TASK_RESULT, {
            "task_id": task_id
        })
        return result.get("result")
    
    async def list_tasks(self, pipeline_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List tasks"""
        params = {}
        if pipeline_id:
            params["pipeline_id"] = pipeline_id
        
        result = await self.send_request(MCPMethods.TASK_LIST, params)
        return result.get("tasks", [])
    
    # Agent operations
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents"""
        result = await self.send_request(MCPMethods.AGENT_LIST)
        return result.get("agents", [])
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get agent status"""
        return await self.send_request(MCPMethods.AGENT_STATUS, {
            "agent_id": agent_id
        })
    
    # Model monitoring
    async def get_model_metrics(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get model metrics"""
        params = {}
        if model_id:
            params["model_id"] = model_id
        
        return await self.send_request(MCPMethods.MODEL_METRICS, params)
    
    async def get_model_health(self) -> Dict[str, Any]:
        """Get model health status"""
        return await self.send_request(MCPMethods.MODEL_HEALTH)
    
    # Memory operations
    async def get_memory(self, key: str) -> Any:
        """Get value from shared memory"""
        result = await self.send_request(MCPMethods.MEMORY_GET, {"key": key})
        return result.get("value")
    
    async def set_memory(self, key: str, value: Any) -> bool:
        """Set value in shared memory"""
        result = await self.send_request(MCPMethods.MEMORY_SET, {
            "key": key,
            "value": value
        })
        return result.get("status") == "stored"
    
    async def delete_memory(self, key: str) -> bool:
        """Delete value from shared memory"""
        result = await self.send_request(MCPMethods.MEMORY_DELETE, {"key": key})
        return result.get("status") == "deleted"
    
    async def list_memory_keys(self) -> List[str]:
        """List memory keys"""
        result = await self.send_request(MCPMethods.MEMORY_LIST)
        return result.get("keys", [])
    
    # Error reporting
    async def report_error(self, error_details: Dict[str, Any]):
        """Report an error to the server"""
        await self.send_request(MCPMethods.ERROR_REPORT, error_details)
    
    # Task completion reporting
    async def report_task_completion(self, task_id: str, result: Any, success: bool = True):
        """Report task completion to server"""
        await self.send_notification("task/completed", {
            "task_id": task_id,
            "agent_id": self.agent_id,
            "result": result,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
    
    async def report_task_progress(self, task_id: str, progress: float, details: Optional[str] = None):
        """Report task progress to server"""
        await self.send_notification("task/progress", {
            "task_id": task_id,
            "agent_id": self.agent_id,
            "progress": progress,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
    
    async def report_metrics(self, metrics: Dict[str, Any]):
        """Report agent metrics to server"""
        await self.send_notification("agent/metrics", {
            "agent_id": self.agent_id,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        })


class MCPClientManager:
    """Manager for multiple MCP clients"""
    
    def __init__(self):
        self.clients: Dict[str, MCPClient] = {}
    
    def create_client(self, agent_id: str, agent_name: str, agent_type: str,
                     server_host: str = "localhost", server_port: int = 8700) -> MCPClient:
        """Create and register a new MCP client"""
        client = MCPClient(agent_id, agent_name, agent_type, server_host, server_port)
        self.clients[agent_id] = client
        return client
    
    async def connect_all(self):
        """Connect all registered clients"""
        tasks = [client.connect() for client in self.clients.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def disconnect_all(self):
        """Disconnect all clients"""
        tasks = [client.disconnect() for client in self.clients.values() if client.connected]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_client(self, agent_id: str) -> Optional[MCPClient]:
        """Get client by agent ID"""
        return self.clients.get(agent_id)
    
    def list_clients(self) -> List[str]:
        """List all client IDs"""
        return list(self.clients.keys())


# Example agent using MCP client
class ExampleAgent:
    """Example agent that uses MCP client for coordination"""
    
    def __init__(self, agent_id: str, name: str, agent_type: str):
        self.client = MCPClient(agent_id, name, agent_type)
        self.running = False
    
    async def start(self):
        """Start the agent"""
        await self.client.connect(capabilities=["data_processing", "model_training"])
        
        # Register handlers for server notifications
        self.client.register_handler("task/assigned", self._handle_task_assignment)
        self.client.register_handler("pipeline/started", self._handle_pipeline_start)
        
        self.running = True
        self.client.logger.info(f"Agent {self.client.agent_name} started")
    
    async def stop(self):
        """Stop the agent"""
        self.running = False
        await self.client.disconnect()
        self.client.logger.info(f"Agent {self.client.agent_name} stopped")
    
    async def _handle_task_assignment(self, params: Dict[str, Any]):
        """Handle task assignment from server"""
        task_id = params.get("task_id")
        self.client.logger.info(f"Assigned task: {task_id}")
        
        # Report progress
        await self.client.report_task_progress(task_id, 0.0, "Starting task")
        
        # Do some work...
        await asyncio.sleep(2)
        
        # Report completion
        await self.client.report_task_completion(task_id, {"result": "task completed"})
    
    async def _handle_pipeline_start(self, params: Dict[str, Any]):
        """Handle pipeline start notification"""
        pipeline_id = params.get("pipeline_id")
        self.client.logger.info(f"Pipeline started: {pipeline_id}")


if __name__ == "__main__":
    # Example usage
    async def main():
        agent = ExampleAgent("agent-001", "Example Agent", "worker")
        
        try:
            await agent.start()
            
            # Keep running until interrupted
            while agent.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("Stopping agent...")
        finally:
            await agent.stop()
    
    asyncio.run(main())