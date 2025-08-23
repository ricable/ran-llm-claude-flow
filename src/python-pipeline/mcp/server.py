"""
MCP Server for Python Pipeline Coordination

This module implements the Model Context Protocol server that acts as the central
coordination hub for Python machine learning pipelines and agent communication.
"""

import asyncio
import json
import logging
import websockets
import uuid
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import asdict

from .protocol import (
    MCPMessage, MCPMessageType, MCPMethods, MCPErrors,
    PipelineInfo, PipelineStatus, TaskInfo, TaskStatus,
    AgentInfo, ModelMetrics, create_response, create_notification,
    DEFAULT_SERVER_PORT
)


class MCPServer:
    """MCP Server for pipeline coordination"""
    
    def __init__(self, port: int = DEFAULT_SERVER_PORT, host: str = "localhost"):
        self.port = port
        self.host = host
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.pipelines: Dict[str, PipelineInfo] = {}
        self.tasks: Dict[str, TaskInfo] = {}
        self.agents: Dict[str, AgentInfo] = {}
        self.memory: Dict[str, Any] = {}
        self.metrics_cache: Dict[str, ModelMetrics] = {}
        self.running = False
        
        # Method handlers
        self.handlers: Dict[str, Callable] = {
            # Pipeline operations
            MCPMethods.PIPELINE_CREATE: self._handle_pipeline_create,
            MCPMethods.PIPELINE_START: self._handle_pipeline_start,
            MCPMethods.PIPELINE_STOP: self._handle_pipeline_stop,
            MCPMethods.PIPELINE_PAUSE: self._handle_pipeline_pause,
            MCPMethods.PIPELINE_RESUME: self._handle_pipeline_resume,
            MCPMethods.PIPELINE_STATUS: self._handle_pipeline_status,
            MCPMethods.PIPELINE_LIST: self._handle_pipeline_list,
            MCPMethods.PIPELINE_DELETE: self._handle_pipeline_delete,
            
            # Task operations
            MCPMethods.TASK_CREATE: self._handle_task_create,
            MCPMethods.TASK_ASSIGN: self._handle_task_assign,
            MCPMethods.TASK_STATUS: self._handle_task_status,
            MCPMethods.TASK_RESULT: self._handle_task_result,
            MCPMethods.TASK_CANCEL: self._handle_task_cancel,
            MCPMethods.TASK_LIST: self._handle_task_list,
            
            # Model monitoring
            MCPMethods.MODEL_METRICS: self._handle_model_metrics,
            MCPMethods.MODEL_PERFORMANCE: self._handle_model_performance,
            MCPMethods.MODEL_MEMORY: self._handle_model_memory,
            MCPMethods.MODEL_HEALTH: self._handle_model_health,
            
            # Agent coordination
            MCPMethods.AGENT_REGISTER: self._handle_agent_register,
            MCPMethods.AGENT_UNREGISTER: self._handle_agent_unregister,
            MCPMethods.AGENT_LIST: self._handle_agent_list,
            MCPMethods.AGENT_STATUS: self._handle_agent_status,
            MCPMethods.AGENT_ASSIGN: self._handle_agent_assign,
            
            # Memory operations
            MCPMethods.MEMORY_GET: self._handle_memory_get,
            MCPMethods.MEMORY_SET: self._handle_memory_set,
            MCPMethods.MEMORY_DELETE: self._handle_memory_delete,
            MCPMethods.MEMORY_LIST: self._handle_memory_list,
            
            # Error handling
            MCPMethods.ERROR_REPORT: self._handle_error_report,
            MCPMethods.ERROR_RECOVER: self._handle_error_recover,
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the MCP server"""
        self.running = True
        self.logger.info(f"Starting MCP server on {self.host}:{self.port}")
        
        # Start the WebSocket server
        server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        )
        
        # Start background tasks
        await asyncio.gather(
            self._heartbeat_task(),
            self._metrics_collection_task(),
            server.wait_closed()
        )
    
    async def stop(self):
        """Stop the MCP server"""
        self.running = False
        self.logger.info("Stopping MCP server")
        
        # Close all client connections
        for client in self.clients.values():
            await client.close()
        self.clients.clear()
    
    async def _handle_client(self, websocket, path):
        """Handle new client connection"""
        client_id = str(uuid.uuid4())
        self.clients[client_id] = websocket
        self.logger.info(f"Client {client_id} connected")
        
        try:
            async for raw_message in websocket:
                try:
                    message = MCPMessage.from_json(raw_message)
                    await self._process_message(client_id, message, websocket)
                except json.JSONDecodeError:
                    error_response = create_response(
                        "unknown", error=MCPErrors.INVALID_REQUEST
                    )
                    await websocket.send(error_response.to_json())
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    error_response = create_response(
                        "unknown", error=MCPErrors.INTERNAL_ERROR
                    )
                    await websocket.send(error_response.to_json())
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if client_id in self.clients:
                del self.clients[client_id]
            self.logger.info(f"Client {client_id} disconnected")
    
    async def _process_message(self, client_id: str, message: MCPMessage, websocket):
        """Process incoming MCP message"""
        if message.type == MCPMessageType.REQUEST:
            if message.method in self.handlers:
                try:
                    result = await self.handlers[message.method](message.params or {})
                    response = create_response(message.id, result=result)
                except Exception as e:
                    self.logger.error(f"Handler error for {message.method}: {e}")
                    response = create_response(
                        message.id, error={**MCPErrors.INTERNAL_ERROR, "data": str(e)}
                    )
            else:
                response = create_response(message.id, error=MCPErrors.METHOD_NOT_FOUND)
            
            await websocket.send(response.to_json())
        elif message.type == MCPMessageType.NOTIFICATION:
            # Handle notifications (no response required)
            await self._handle_notification(client_id, message)
    
    async def _handle_notification(self, client_id: str, message: MCPMessage):
        """Handle notification messages"""
        self.logger.info(f"Received notification from {client_id}: {message.method}")
        # Broadcast to interested clients or store for later processing
    
    async def _broadcast_notification(self, method: str, params: Dict[str, Any]):
        """Broadcast notification to all connected clients"""
        notification = create_notification(method, params)
        
        for client in list(self.clients.values()):
            try:
                await client.send(notification.to_json())
            except websockets.exceptions.ConnectionClosed:
                # Client disconnected, will be cleaned up later
                pass
    
    # Pipeline handlers
    async def _handle_pipeline_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new pipeline"""
        pipeline_id = str(uuid.uuid4())
        pipeline = PipelineInfo(
            id=pipeline_id,
            name=params.get("name", f"Pipeline-{pipeline_id[:8]}"),
            status=PipelineStatus.IDLE,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            config=params.get("config", {})
        )
        
        self.pipelines[pipeline_id] = pipeline
        
        await self._broadcast_notification("pipeline/created", {
            "pipeline_id": pipeline_id,
            "name": pipeline.name
        })
        
        return {"pipeline_id": pipeline_id, "status": "created"}
    
    async def _handle_pipeline_start(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start a pipeline"""
        pipeline_id = params.get("pipeline_id")
        if not pipeline_id or pipeline_id not in self.pipelines:
            raise Exception("Pipeline not found")
        
        pipeline = self.pipelines[pipeline_id]
        if pipeline.status == PipelineStatus.RUNNING:
            raise Exception("Pipeline already running")
        
        pipeline.status = PipelineStatus.RUNNING
        pipeline.updated_at = datetime.now().isoformat()
        
        await self._broadcast_notification("pipeline/started", {
            "pipeline_id": pipeline_id
        })
        
        return {"pipeline_id": pipeline_id, "status": "started"}
    
    async def _handle_pipeline_stop(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Stop a pipeline"""
        pipeline_id = params.get("pipeline_id")
        if not pipeline_id or pipeline_id not in self.pipelines:
            raise Exception("Pipeline not found")
        
        pipeline = self.pipelines[pipeline_id]
        pipeline.status = PipelineStatus.COMPLETED
        pipeline.updated_at = datetime.now().isoformat()
        
        await self._broadcast_notification("pipeline/stopped", {
            "pipeline_id": pipeline_id
        })
        
        return {"pipeline_id": pipeline_id, "status": "stopped"}
    
    async def _handle_pipeline_pause(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Pause a pipeline"""
        pipeline_id = params.get("pipeline_id")
        if not pipeline_id or pipeline_id not in self.pipelines:
            raise Exception("Pipeline not found")
        
        pipeline = self.pipelines[pipeline_id]
        pipeline.status = PipelineStatus.PAUSED
        pipeline.updated_at = datetime.now().isoformat()
        
        return {"pipeline_id": pipeline_id, "status": "paused"}
    
    async def _handle_pipeline_resume(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resume a pipeline"""
        pipeline_id = params.get("pipeline_id")
        if not pipeline_id or pipeline_id not in self.pipelines:
            raise Exception("Pipeline not found")
        
        pipeline = self.pipelines[pipeline_id]
        pipeline.status = PipelineStatus.RUNNING
        pipeline.updated_at = datetime.now().isoformat()
        
        return {"pipeline_id": pipeline_id, "status": "resumed"}
    
    async def _handle_pipeline_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get pipeline status"""
        pipeline_id = params.get("pipeline_id")
        if not pipeline_id or pipeline_id not in self.pipelines:
            raise Exception("Pipeline not found")
        
        pipeline = self.pipelines[pipeline_id]
        return asdict(pipeline)
    
    async def _handle_pipeline_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List all pipelines"""
        return {
            "pipelines": [asdict(p) for p in self.pipelines.values()]
        }
    
    async def _handle_pipeline_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a pipeline"""
        pipeline_id = params.get("pipeline_id")
        if not pipeline_id or pipeline_id not in self.pipelines:
            raise Exception("Pipeline not found")
        
        del self.pipelines[pipeline_id]
        return {"pipeline_id": pipeline_id, "status": "deleted"}
    
    # Task handlers
    async def _handle_task_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new task"""
        task_id = str(uuid.uuid4())
        task = TaskInfo(
            id=task_id,
            pipeline_id=params.get("pipeline_id", ""),
            name=params.get("name", f"Task-{task_id[:8]}"),
            status=TaskStatus.PENDING,
            assigned_agent=None,
            created_at=datetime.now().isoformat(),
            started_at=None,
            completed_at=None,
            config=params.get("config", {})
        )
        
        self.tasks[task_id] = task
        return {"task_id": task_id, "status": "created"}
    
    async def _handle_task_assign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Assign task to agent"""
        task_id = params.get("task_id")
        agent_id = params.get("agent_id")
        
        if not task_id or task_id not in self.tasks:
            raise Exception("Task not found")
        if not agent_id or agent_id not in self.agents:
            raise Exception("Agent not found")
        
        task = self.tasks[task_id]
        task.assigned_agent = agent_id
        task.status = TaskStatus.ASSIGNED
        
        return {"task_id": task_id, "agent_id": agent_id, "status": "assigned"}
    
    async def _handle_task_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get task status"""
        task_id = params.get("task_id")
        if not task_id or task_id not in self.tasks:
            raise Exception("Task not found")
        
        task = self.tasks[task_id]
        return asdict(task)
    
    async def _handle_task_result(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get task result"""
        task_id = params.get("task_id")
        if not task_id or task_id not in self.tasks:
            raise Exception("Task not found")
        
        task = self.tasks[task_id]
        return {"task_id": task_id, "result": task.result}
    
    async def _handle_task_cancel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel a task"""
        task_id = params.get("task_id")
        if not task_id or task_id not in self.tasks:
            raise Exception("Task not found")
        
        task = self.tasks[task_id]
        task.status = TaskStatus.CANCELLED
        
        return {"task_id": task_id, "status": "cancelled"}
    
    async def _handle_task_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List tasks"""
        pipeline_id = params.get("pipeline_id")
        tasks = list(self.tasks.values())
        
        if pipeline_id:
            tasks = [t for t in tasks if t.pipeline_id == pipeline_id]
        
        return {"tasks": [asdict(t) for t in tasks]}
    
    # Agent handlers
    async def _handle_agent_register(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new agent"""
        agent = AgentInfo(
            id=params.get("id", str(uuid.uuid4())),
            name=params.get("name", "Unknown Agent"),
            type=params.get("type", "generic"),
            status="active",
            capabilities=params.get("capabilities", []),
            current_task=None,
            performance_score=None,
            last_active=datetime.now().isoformat()
        )
        
        self.agents[agent.id] = agent
        return {"agent_id": agent.id, "status": "registered"}
    
    async def _handle_agent_unregister(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Unregister an agent"""
        agent_id = params.get("agent_id")
        if not agent_id or agent_id not in self.agents:
            raise Exception("Agent not found")
        
        del self.agents[agent_id]
        return {"agent_id": agent_id, "status": "unregistered"}
    
    async def _handle_agent_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List all agents"""
        return {"agents": [asdict(a) for a in self.agents.values()]}
    
    async def _handle_agent_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get agent status"""
        agent_id = params.get("agent_id")
        if not agent_id or agent_id not in self.agents:
            raise Exception("Agent not found")
        
        agent = self.agents[agent_id]
        return asdict(agent)
    
    async def _handle_agent_assign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Assign agent to task"""
        return await self._handle_task_assign(params)
    
    # Model monitoring handlers
    async def _handle_model_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get model metrics"""
        model_id = params.get("model_id")
        if model_id and model_id in self.metrics_cache:
            return asdict(self.metrics_cache[model_id])
        
        return {"metrics": list(asdict(m) for m in self.metrics_cache.values())}
    
    async def _handle_model_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get model performance data"""
        # Implementation would integrate with actual model monitoring
        return {"performance": "placeholder_data"}
    
    async def _handle_model_memory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get model memory usage"""
        # Implementation would integrate with actual memory monitoring
        return {"memory_usage": "placeholder_data"}
    
    async def _handle_model_health(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get model health status"""
        return {"health": "healthy", "timestamp": datetime.now().isoformat()}
    
    # Memory handlers
    async def _handle_memory_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get value from memory"""
        key = params.get("key")
        if not key:
            raise Exception("Key required")
        
        value = self.memory.get(key)
        return {"key": key, "value": value}
    
    async def _handle_memory_set(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set value in memory"""
        key = params.get("key")
        value = params.get("value")
        
        if not key:
            raise Exception("Key required")
        
        self.memory[key] = value
        return {"key": key, "status": "stored"}
    
    async def _handle_memory_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete value from memory"""
        key = params.get("key")
        if not key:
            raise Exception("Key required")
        
        if key in self.memory:
            del self.memory[key]
        
        return {"key": key, "status": "deleted"}
    
    async def _handle_memory_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List memory keys"""
        return {"keys": list(self.memory.keys())}
    
    # Error handlers
    async def _handle_error_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Report an error"""
        error_id = str(uuid.uuid4())
        self.logger.error(f"Error reported: {params}")
        return {"error_id": error_id, "status": "reported"}
    
    async def _handle_error_recover(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt error recovery"""
        # Implementation would handle specific recovery strategies
        return {"status": "recovery_attempted"}
    
    # Background tasks
    async def _heartbeat_task(self):
        """Send periodic heartbeat to clients"""
        while self.running:
            await self._broadcast_notification("server/heartbeat", {
                "timestamp": datetime.now().isoformat(),
                "server_status": "running"
            })
            await asyncio.sleep(30)
    
    async def _metrics_collection_task(self):
        """Collect system metrics periodically"""
        while self.running:
            # Collect metrics from connected agents and models
            await self._collect_system_metrics()
            await asyncio.sleep(10)
    
    async def _collect_system_metrics(self):
        """Collect and cache system metrics"""
        # Implementation would collect actual metrics
        timestamp = datetime.now().isoformat()
        
        # Update server metrics
        self.memory["server/metrics"] = {
            "connected_clients": len(self.clients),
            "active_pipelines": len([p for p in self.pipelines.values() 
                                   if p.status == PipelineStatus.RUNNING]),
            "active_tasks": len([t for t in self.tasks.values() 
                               if t.status == TaskStatus.RUNNING]),
            "registered_agents": len(self.agents),
            "timestamp": timestamp
        }


if __name__ == "__main__":
    # Example server startup
    import logging
    
    logging.basicConfig(level=logging.INFO)
    server = MCPServer()
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("Server stopped")