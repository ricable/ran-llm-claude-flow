"""
MCP Request/Response Handlers

This module implements comprehensive handlers for all MCP operations
including pipeline control, task orchestration, and monitoring.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import asdict

from .protocol import (
    MCPMessage, MCPMessageType, MCPMethods, MCPErrors,
    PipelineInfo, PipelineStatus, TaskInfo, TaskStatus,
    AgentInfo, ModelMetrics, create_response, create_notification
)


class MCPRequestHandler:
    """Base class for MCP request handlers"""
    
    def __init__(self, storage_backend=None):
        self.storage = storage_backend or InMemoryStorage()
        self.logger = logging.getLogger(__name__)
        self.handlers: Dict[str, Callable] = {}
        self._register_handlers()
    
    def _register_handlers(self):
        """Register all request handlers"""
        # Pipeline handlers
        self.handlers[MCPMethods.PIPELINE_CREATE] = self.handle_pipeline_create
        self.handlers[MCPMethods.PIPELINE_START] = self.handle_pipeline_start
        self.handlers[MCPMethods.PIPELINE_STOP] = self.handle_pipeline_stop
        self.handlers[MCPMethods.PIPELINE_PAUSE] = self.handle_pipeline_pause
        self.handlers[MCPMethods.PIPELINE_RESUME] = self.handle_pipeline_resume
        self.handlers[MCPMethods.PIPELINE_STATUS] = self.handle_pipeline_status
        self.handlers[MCPMethods.PIPELINE_LIST] = self.handle_pipeline_list
        self.handlers[MCPMethods.PIPELINE_DELETE] = self.handle_pipeline_delete
        
        # Task handlers
        self.handlers[MCPMethods.TASK_CREATE] = self.handle_task_create
        self.handlers[MCPMethods.TASK_ASSIGN] = self.handle_task_assign
        self.handlers[MCPMethods.TASK_STATUS] = self.handle_task_status
        self.handlers[MCPMethods.TASK_RESULT] = self.handle_task_result
        self.handlers[MCPMethods.TASK_CANCEL] = self.handle_task_cancel
        self.handlers[MCPMethods.TASK_LIST] = self.handle_task_list
        
        # Agent handlers
        self.handlers[MCPMethods.AGENT_REGISTER] = self.handle_agent_register
        self.handlers[MCPMethods.AGENT_UNREGISTER] = self.handle_agent_unregister
        self.handlers[MCPMethods.AGENT_LIST] = self.handle_agent_list
        self.handlers[MCPMethods.AGENT_STATUS] = self.handle_agent_status
        self.handlers[MCPMethods.AGENT_ASSIGN] = self.handle_agent_assign
        
        # Model monitoring handlers
        self.handlers[MCPMethods.MODEL_METRICS] = self.handle_model_metrics
        self.handlers[MCPMethods.MODEL_PERFORMANCE] = self.handle_model_performance
        self.handlers[MCPMethods.MODEL_MEMORY] = self.handle_model_memory
        self.handlers[MCPMethods.MODEL_HEALTH] = self.handle_model_health
        
        # Memory handlers
        self.handlers[MCPMethods.MEMORY_GET] = self.handle_memory_get
        self.handlers[MCPMethods.MEMORY_SET] = self.handle_memory_set
        self.handlers[MCPMethods.MEMORY_DELETE] = self.handle_memory_delete
        self.handlers[MCPMethods.MEMORY_LIST] = self.handle_memory_list
        
        # Error handlers
        self.handlers[MCPMethods.ERROR_REPORT] = self.handle_error_report
        self.handlers[MCPMethods.ERROR_RECOVER] = self.handle_error_recover
    
    async def handle_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP request"""
        if method not in self.handlers:
            raise ValueError(f"Unknown method: {method}")
        
        try:
            return await self.handlers[method](params)
        except Exception as e:
            self.logger.error(f"Handler error for {method}: {e}")
            raise
    
    # Pipeline handlers
    async def handle_pipeline_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new pipeline"""
        pipeline_id = str(uuid.uuid4())
        
        # Validate required parameters
        name = params.get("name")
        if not name:
            raise ValueError("Pipeline name is required")
        
        pipeline = PipelineInfo(
            id=pipeline_id,
            name=name,
            status=PipelineStatus.IDLE,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            config=params.get("config", {}),
            metrics=None,
            error_info=None
        )
        
        # Store pipeline
        await self.storage.store_pipeline(pipeline)
        
        # Log creation
        self.logger.info(f"Created pipeline: {name} ({pipeline_id})")
        
        return {
            "pipeline_id": pipeline_id,
            "name": name,
            "status": pipeline.status.value,
            "created_at": pipeline.created_at
        }
    
    async def handle_pipeline_start(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start a pipeline"""
        pipeline_id = params.get("pipeline_id")
        if not pipeline_id:
            raise ValueError("Pipeline ID is required")
        
        # Get pipeline
        pipeline = await self.storage.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError("Pipeline not found")
        
        # Check if already running
        if pipeline.status == PipelineStatus.RUNNING:
            raise ValueError("Pipeline is already running")
        
        # Update status
        pipeline.status = PipelineStatus.RUNNING
        pipeline.updated_at = datetime.now().isoformat()
        
        # Store updated pipeline
        await self.storage.store_pipeline(pipeline)
        
        # Initialize metrics
        pipeline.metrics = {
            "started_at": datetime.now().isoformat(),
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0
        }
        
        self.logger.info(f"Started pipeline: {pipeline.name}")
        
        return {
            "pipeline_id": pipeline_id,
            "status": pipeline.status.value,
            "started_at": pipeline.metrics["started_at"]
        }
    
    async def handle_pipeline_stop(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Stop a pipeline"""
        pipeline_id = params.get("pipeline_id")
        if not pipeline_id:
            raise ValueError("Pipeline ID is required")
        
        pipeline = await self.storage.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError("Pipeline not found")
        
        # Update status
        old_status = pipeline.status
        pipeline.status = PipelineStatus.COMPLETED
        pipeline.updated_at = datetime.now().isoformat()
        
        # Update metrics if pipeline was running
        if old_status == PipelineStatus.RUNNING and pipeline.metrics:
            pipeline.metrics["completed_at"] = datetime.now().isoformat()
            
            # Calculate total execution time
            started_at = datetime.fromisoformat(pipeline.metrics["started_at"])
            completed_at = datetime.fromisoformat(pipeline.metrics["completed_at"])
            pipeline.metrics["total_execution_time"] = (completed_at - started_at).total_seconds()
        
        await self.storage.store_pipeline(pipeline)
        
        self.logger.info(f"Stopped pipeline: {pipeline.name}")
        
        return {
            "pipeline_id": pipeline_id,
            "status": pipeline.status.value,
            "metrics": pipeline.metrics
        }
    
    async def handle_pipeline_pause(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Pause a pipeline"""
        pipeline_id = params.get("pipeline_id")
        if not pipeline_id:
            raise ValueError("Pipeline ID is required")
        
        pipeline = await self.storage.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError("Pipeline not found")
        
        if pipeline.status != PipelineStatus.RUNNING:
            raise ValueError("Pipeline is not running")
        
        pipeline.status = PipelineStatus.PAUSED
        pipeline.updated_at = datetime.now().isoformat()
        
        await self.storage.store_pipeline(pipeline)
        
        return {
            "pipeline_id": pipeline_id,
            "status": pipeline.status.value
        }
    
    async def handle_pipeline_resume(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resume a pipeline"""
        pipeline_id = params.get("pipeline_id")
        if not pipeline_id:
            raise ValueError("Pipeline ID is required")
        
        pipeline = await self.storage.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError("Pipeline not found")
        
        if pipeline.status != PipelineStatus.PAUSED:
            raise ValueError("Pipeline is not paused")
        
        pipeline.status = PipelineStatus.RUNNING
        pipeline.updated_at = datetime.now().isoformat()
        
        await self.storage.store_pipeline(pipeline)
        
        return {
            "pipeline_id": pipeline_id,
            "status": pipeline.status.value
        }
    
    async def handle_pipeline_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get pipeline status"""
        pipeline_id = params.get("pipeline_id")
        if not pipeline_id:
            raise ValueError("Pipeline ID is required")
        
        pipeline = await self.storage.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError("Pipeline not found")
        
        return asdict(pipeline)
    
    async def handle_pipeline_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List pipelines with optional filtering"""
        status_filter = params.get("status")
        limit = params.get("limit", 100)
        offset = params.get("offset", 0)
        
        pipelines = await self.storage.list_pipelines(
            status_filter=status_filter,
            limit=limit,
            offset=offset
        )
        
        return {
            "pipelines": [asdict(p) for p in pipelines],
            "total": len(pipelines),
            "limit": limit,
            "offset": offset
        }
    
    async def handle_pipeline_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a pipeline"""
        pipeline_id = params.get("pipeline_id")
        if not pipeline_id:
            raise ValueError("Pipeline ID is required")
        
        pipeline = await self.storage.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError("Pipeline not found")
        
        # Can only delete idle or completed pipelines
        if pipeline.status in [PipelineStatus.RUNNING, PipelineStatus.PAUSED]:
            raise ValueError("Cannot delete running or paused pipeline")
        
        # Delete associated tasks first
        tasks = await self.storage.list_tasks(pipeline_id=pipeline_id)
        for task in tasks:
            await self.storage.delete_task(task.id)
        
        # Delete pipeline
        await self.storage.delete_pipeline(pipeline_id)
        
        self.logger.info(f"Deleted pipeline: {pipeline.name}")
        
        return {
            "pipeline_id": pipeline_id,
            "status": "deleted"
        }
    
    # Task handlers
    async def handle_task_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new task"""
        task_id = str(uuid.uuid4())
        
        # Validate required parameters
        name = params.get("name")
        if not name:
            raise ValueError("Task name is required")
        
        pipeline_id = params.get("pipeline_id", "")
        
        # If pipeline_id provided, validate it exists
        if pipeline_id:
            pipeline = await self.storage.get_pipeline(pipeline_id)
            if not pipeline:
                raise ValueError("Pipeline not found")
        
        task = TaskInfo(
            id=task_id,
            pipeline_id=pipeline_id,
            name=name,
            status=TaskStatus.PENDING,
            assigned_agent=None,
            created_at=datetime.now().isoformat(),
            started_at=None,
            completed_at=None,
            config=params.get("config", {}),
            result=None,
            error_info=None
        )
        
        await self.storage.store_task(task)
        
        self.logger.info(f"Created task: {name} ({task_id})")
        
        return {
            "task_id": task_id,
            "name": name,
            "status": task.status.value,
            "pipeline_id": pipeline_id
        }
    
    async def handle_task_assign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Assign task to agent"""
        task_id = params.get("task_id")
        agent_id = params.get("agent_id")
        
        if not task_id:
            raise ValueError("Task ID is required")
        if not agent_id:
            raise ValueError("Agent ID is required")
        
        # Get task and agent
        task = await self.storage.get_task(task_id)
        if not task:
            raise ValueError("Task not found")
        
        agent = await self.storage.get_agent(agent_id)
        if not agent:
            raise ValueError("Agent not found")
        
        # Check if task is assignable
        if task.status != TaskStatus.PENDING:
            raise ValueError("Task is not in pending state")
        
        # Check if agent is available
        if agent.current_task:
            raise ValueError("Agent is already assigned to another task")
        
        # Check capabilities if specified
        required_capabilities = task.config.get("required_capabilities", [])
        if required_capabilities:
            missing_capabilities = set(required_capabilities) - set(agent.capabilities)
            if missing_capabilities:
                raise ValueError(f"Agent missing capabilities: {missing_capabilities}")
        
        # Assign task
        task.assigned_agent = agent_id
        task.status = TaskStatus.ASSIGNED
        
        # Update agent
        agent.current_task = task_id
        agent.last_active = datetime.now().isoformat()
        
        # Store updates
        await self.storage.store_task(task)
        await self.storage.store_agent(agent)
        
        self.logger.info(f"Assigned task {task.name} to agent {agent.name}")
        
        return {
            "task_id": task_id,
            "agent_id": agent_id,
            "status": task.status.value
        }
    
    async def handle_task_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get task status"""
        task_id = params.get("task_id")
        if not task_id:
            raise ValueError("Task ID is required")
        
        task = await self.storage.get_task(task_id)
        if not task:
            raise ValueError("Task not found")
        
        return asdict(task)
    
    async def handle_task_result(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get task result"""
        task_id = params.get("task_id")
        if not task_id:
            raise ValueError("Task ID is required")
        
        task = await self.storage.get_task(task_id)
        if not task:
            raise ValueError("Task not found")
        
        return {
            "task_id": task_id,
            "status": task.status.value,
            "result": task.result,
            "error_info": task.error_info
        }
    
    async def handle_task_cancel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel a task"""
        task_id = params.get("task_id")
        if not task_id:
            raise ValueError("Task ID is required")
        
        task = await self.storage.get_task(task_id)
        if not task:
            raise ValueError("Task not found")
        
        # Can only cancel pending or assigned tasks
        if task.status not in [TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.RUNNING]:
            raise ValueError("Task cannot be cancelled in current state")
        
        task.status = TaskStatus.CANCELLED
        
        # Free up agent if assigned
        if task.assigned_agent:
            agent = await self.storage.get_agent(task.assigned_agent)
            if agent and agent.current_task == task_id:
                agent.current_task = None
                await self.storage.store_agent(agent)
        
        await self.storage.store_task(task)
        
        return {
            "task_id": task_id,
            "status": task.status.value
        }
    
    async def handle_task_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List tasks with optional filtering"""
        pipeline_id = params.get("pipeline_id")
        agent_id = params.get("agent_id")
        status_filter = params.get("status")
        limit = params.get("limit", 100)
        offset = params.get("offset", 0)
        
        tasks = await self.storage.list_tasks(
            pipeline_id=pipeline_id,
            agent_id=agent_id,
            status_filter=status_filter,
            limit=limit,
            offset=offset
        )
        
        return {
            "tasks": [asdict(t) for t in tasks],
            "total": len(tasks),
            "limit": limit,
            "offset": offset
        }
    
    # Agent handlers
    async def handle_agent_register(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new agent"""
        agent_id = params.get("id") or str(uuid.uuid4())
        name = params.get("name", f"Agent-{agent_id[:8]}")
        agent_type = params.get("type", "generic")
        capabilities = params.get("capabilities", [])
        
        # Check if agent already exists
        existing_agent = await self.storage.get_agent(agent_id)
        if existing_agent:
            raise ValueError("Agent already registered")
        
        agent = AgentInfo(
            id=agent_id,
            name=name,
            type=agent_type,
            status="active",
            capabilities=capabilities,
            current_task=None,
            performance_score=1.0,
            last_active=datetime.now().isoformat()
        )
        
        await self.storage.store_agent(agent)
        
        self.logger.info(f"Registered agent: {name} ({agent_id})")
        
        return {
            "agent_id": agent_id,
            "name": name,
            "type": agent_type,
            "status": "registered"
        }
    
    async def handle_agent_unregister(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Unregister an agent"""
        agent_id = params.get("agent_id")
        if not agent_id:
            raise ValueError("Agent ID is required")
        
        agent = await self.storage.get_agent(agent_id)
        if not agent:
            raise ValueError("Agent not found")
        
        # Cancel any assigned task
        if agent.current_task:
            await self.handle_task_cancel({"task_id": agent.current_task})
        
        await self.storage.delete_agent(agent_id)
        
        self.logger.info(f"Unregistered agent: {agent.name}")
        
        return {
            "agent_id": agent_id,
            "status": "unregistered"
        }
    
    async def handle_agent_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List agents with optional filtering"""
        agent_type = params.get("type")
        status_filter = params.get("status")
        capability_filter = params.get("capability")
        limit = params.get("limit", 100)
        offset = params.get("offset", 0)
        
        agents = await self.storage.list_agents(
            agent_type=agent_type,
            status_filter=status_filter,
            capability_filter=capability_filter,
            limit=limit,
            offset=offset
        )
        
        return {
            "agents": [asdict(a) for a in agents],
            "total": len(agents),
            "limit": limit,
            "offset": offset
        }
    
    async def handle_agent_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get agent status"""
        agent_id = params.get("agent_id")
        if not agent_id:
            raise ValueError("Agent ID is required")
        
        agent = await self.storage.get_agent(agent_id)
        if not agent:
            raise ValueError("Agent not found")
        
        # Include additional status information
        status_info = asdict(agent)
        
        # Add current task details if available
        if agent.current_task:
            task = await self.storage.get_task(agent.current_task)
            if task:
                status_info["current_task_details"] = {
                    "name": task.name,
                    "status": task.status.value,
                    "started_at": task.started_at
                }
        
        return status_info
    
    async def handle_agent_assign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Assign agent to task (alias for task_assign)"""
        return await self.handle_task_assign(params)
    
    # Model monitoring handlers
    async def handle_model_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get model metrics"""
        model_id = params.get("model_id")
        
        if model_id:
            metrics = await self.storage.get_model_metrics(model_id)
            if not metrics:
                raise ValueError("Model not found")
            return asdict(metrics)
        else:
            # Return all model metrics
            all_metrics = await self.storage.list_model_metrics()
            return {
                "models": [asdict(m) for m in all_metrics]
            }
    
    async def handle_model_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get model performance data"""
        model_id = params.get("model_id")
        timeframe = params.get("timeframe", "1h")  # 1h, 1d, 7d, 30d
        
        # This would integrate with actual performance monitoring system
        performance_data = await self.storage.get_model_performance(model_id, timeframe)
        
        return {
            "model_id": model_id,
            "timeframe": timeframe,
            "performance": performance_data
        }
    
    async def handle_model_memory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get model memory usage"""
        model_id = params.get("model_id")
        
        memory_data = await self.storage.get_model_memory_usage(model_id)
        
        return {
            "model_id": model_id,
            "memory_usage": memory_data
        }
    
    async def handle_model_health(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get model health status"""
        health_data = await self.storage.get_system_health()
        
        return {
            "health": health_data,
            "timestamp": datetime.now().isoformat()
        }
    
    # Memory handlers
    async def handle_memory_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get value from memory"""
        key = params.get("key")
        if not key:
            raise ValueError("Key is required")
        
        value = await self.storage.get_memory(key)
        
        return {
            "key": key,
            "value": value,
            "exists": value is not None
        }
    
    async def handle_memory_set(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set value in memory"""
        key = params.get("key")
        value = params.get("value")
        
        if not key:
            raise ValueError("Key is required")
        
        await self.storage.set_memory(key, value)
        
        return {
            "key": key,
            "status": "stored"
        }
    
    async def handle_memory_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete value from memory"""
        key = params.get("key")
        if not key:
            raise ValueError("Key is required")
        
        existed = await self.storage.delete_memory(key)
        
        return {
            "key": key,
            "status": "deleted" if existed else "not_found"
        }
    
    async def handle_memory_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List memory keys"""
        prefix = params.get("prefix")
        limit = params.get("limit", 1000)
        
        keys = await self.storage.list_memory_keys(prefix=prefix, limit=limit)
        
        return {
            "keys": keys,
            "total": len(keys),
            "prefix": prefix
        }
    
    # Error handlers
    async def handle_error_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Report an error"""
        error_id = str(uuid.uuid4())
        
        error_data = {
            "error_id": error_id,
            "timestamp": datetime.now().isoformat(),
            "component": params.get("component"),
            "error_type": params.get("error_type"),
            "message": params.get("message"),
            "details": params.get("details"),
            "context": params.get("context")
        }
        
        await self.storage.store_error_report(error_data)
        
        self.logger.error(f"Error reported: {error_data}")
        
        return {
            "error_id": error_id,
            "status": "reported"
        }
    
    async def handle_error_recover(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt error recovery"""
        error_id = params.get("error_id")
        recovery_action = params.get("recovery_action", "retry")
        
        # This would implement actual recovery strategies
        recovery_result = await self.storage.attempt_error_recovery(error_id, recovery_action)
        
        return {
            "error_id": error_id,
            "recovery_action": recovery_action,
            "status": recovery_result.get("status", "attempted"),
            "details": recovery_result.get("details")
        }


class InMemoryStorage:
    """Simple in-memory storage backend for development/testing"""
    
    def __init__(self):
        self.pipelines: Dict[str, PipelineInfo] = {}
        self.tasks: Dict[str, TaskInfo] = {}
        self.agents: Dict[str, AgentInfo] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.memory: Dict[str, Any] = {}
        self.errors: Dict[str, Dict[str, Any]] = {}
    
    # Pipeline operations
    async def store_pipeline(self, pipeline: PipelineInfo):
        self.pipelines[pipeline.id] = pipeline
    
    async def get_pipeline(self, pipeline_id: str) -> Optional[PipelineInfo]:
        return self.pipelines.get(pipeline_id)
    
    async def delete_pipeline(self, pipeline_id: str) -> bool:
        if pipeline_id in self.pipelines:
            del self.pipelines[pipeline_id]
            return True
        return False
    
    async def list_pipelines(self, status_filter=None, limit=100, offset=0) -> List[PipelineInfo]:
        pipelines = list(self.pipelines.values())
        
        if status_filter:
            pipelines = [p for p in pipelines if p.status.value == status_filter]
        
        return pipelines[offset:offset + limit]
    
    # Task operations
    async def store_task(self, task: TaskInfo):
        self.tasks[task.id] = task
    
    async def get_task(self, task_id: str) -> Optional[TaskInfo]:
        return self.tasks.get(task_id)
    
    async def delete_task(self, task_id: str) -> bool:
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False
    
    async def list_tasks(self, pipeline_id=None, agent_id=None, status_filter=None, 
                        limit=100, offset=0) -> List[TaskInfo]:
        tasks = list(self.tasks.values())
        
        if pipeline_id:
            tasks = [t for t in tasks if t.pipeline_id == pipeline_id]
        
        if agent_id:
            tasks = [t for t in tasks if t.assigned_agent == agent_id]
        
        if status_filter:
            tasks = [t for t in tasks if t.status.value == status_filter]
        
        return tasks[offset:offset + limit]
    
    # Agent operations
    async def store_agent(self, agent: AgentInfo):
        self.agents[agent.id] = agent
    
    async def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        return self.agents.get(agent_id)
    
    async def delete_agent(self, agent_id: str) -> bool:
        if agent_id in self.agents:
            del self.agents[agent_id]
            return True
        return False
    
    async def list_agents(self, agent_type=None, status_filter=None, capability_filter=None,
                         limit=100, offset=0) -> List[AgentInfo]:
        agents = list(self.agents.values())
        
        if agent_type:
            agents = [a for a in agents if a.type == agent_type]
        
        if status_filter:
            agents = [a for a in agents if a.status == status_filter]
        
        if capability_filter:
            agents = [a for a in agents if capability_filter in a.capabilities]
        
        return agents[offset:offset + limit]
    
    # Model operations
    async def get_model_metrics(self, model_id: str) -> Optional[ModelMetrics]:
        return self.model_metrics.get(model_id)
    
    async def list_model_metrics(self) -> List[ModelMetrics]:
        return list(self.model_metrics.values())
    
    async def get_model_performance(self, model_id: str, timeframe: str) -> Dict[str, Any]:
        # Placeholder implementation
        return {"performance": "placeholder_data"}
    
    async def get_model_memory_usage(self, model_id: str) -> Dict[str, Any]:
        # Placeholder implementation
        return {"memory_usage": "placeholder_data"}
    
    async def get_system_health(self) -> Dict[str, Any]:
        return {"status": "healthy", "components": {}}
    
    # Memory operations
    async def get_memory(self, key: str) -> Any:
        return self.memory.get(key)
    
    async def set_memory(self, key: str, value: Any):
        self.memory[key] = value
    
    async def delete_memory(self, key: str) -> bool:
        if key in self.memory:
            del self.memory[key]
            return True
        return False
    
    async def list_memory_keys(self, prefix=None, limit=1000) -> List[str]:
        keys = list(self.memory.keys())
        
        if prefix:
            keys = [k for k in keys if k.startswith(prefix)]
        
        return keys[:limit]
    
    # Error operations
    async def store_error_report(self, error_data: Dict[str, Any]):
        self.errors[error_data["error_id"]] = error_data
    
    async def attempt_error_recovery(self, error_id: str, recovery_action: str) -> Dict[str, Any]:
        return {"status": "attempted", "details": f"Recovery action: {recovery_action}"}


# Example usage
async def main():
    """Example usage of MCP request handler"""
    handler = MCPRequestHandler()
    
    # Create a pipeline
    result = await handler.handle_request(MCPMethods.PIPELINE_CREATE, {
        "name": "Test Pipeline",
        "config": {"timeout": 300}
    })
    print(f"Created pipeline: {result}")
    
    # Register an agent
    result = await handler.handle_request(MCPMethods.AGENT_REGISTER, {
        "name": "Test Agent",
        "type": "worker",
        "capabilities": ["data_processing"]
    })
    print(f"Registered agent: {result}")
    
    # Create a task
    result = await handler.handle_request(MCPMethods.TASK_CREATE, {
        "name": "Test Task",
        "config": {"required_capabilities": ["data_processing"]}
    })
    print(f"Created task: {result}")


if __name__ == "__main__":
    asyncio.run(main())