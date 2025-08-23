"""
MCP Host for Pipeline Orchestration

This module implements the Model Context Protocol host that orchestrates
pipeline operations and coordinates between servers and clients.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from .protocol import (
    MCPMessage, MCPMessageType, MCPMethods,
    PipelineInfo, PipelineStatus, TaskInfo, TaskStatus,
    AgentInfo, ModelMetrics, DEFAULT_HOST_PORT
)
from .server import MCPServer
from .client import MCPClient


class OrchestrationStrategy(Enum):
    """Pipeline orchestration strategies"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PRIORITY_BASED = "priority_based"
    RESOURCE_AWARE = "resource_aware"
    ADAPTIVE = "adaptive"


@dataclass
class OrchestrationConfig:
    """Configuration for pipeline orchestration"""
    strategy: OrchestrationStrategy = OrchestrationStrategy.ADAPTIVE
    max_concurrent_tasks: int = 10
    task_timeout: float = 300.0  # 5 minutes
    retry_attempts: int = 3
    resource_limits: Dict[str, float] = None
    priority_weights: Dict[str, float] = None


@dataclass
class ResourceUsage:
    """Resource usage tracking"""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    disk_io_mb: float = 0.0
    network_io_mb: float = 0.0


class MCPHost:
    """MCP Host for pipeline orchestration"""
    
    def __init__(self, host_id: str = None, config: OrchestrationConfig = None):
        self.host_id = host_id or str(uuid.uuid4())
        self.config = config or OrchestrationConfig()
        
        # Core components
        self.server: Optional[MCPServer] = None
        self.clients: Dict[str, MCPClient] = {}
        
        # Orchestration state
        self.pipelines: Dict[str, PipelineInfo] = {}
        self.tasks: Dict[str, TaskInfo] = {}
        self.agents: Dict[str, AgentInfo] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # Resource management
        self.resource_usage: ResourceUsage = ResourceUsage()
        self.resource_limits: Dict[str, float] = self.config.resource_limits or {}
        
        # Performance tracking
        self.metrics: Dict[str, Any] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
        # Task scheduling
        self.scheduler_running = False
        self.monitor_running = False
        
        self.logger = logging.getLogger(f"{__name__}.{self.host_id}")
    
    async def start(self, server_port: int = 8700):
        """Start the MCP host with embedded server"""
        self.logger.info(f"Starting MCP Host {self.host_id}")
        
        # Start embedded MCP server
        self.server = MCPServer(port=server_port)
        server_task = asyncio.create_task(self.server.start())
        
        # Wait for server to be ready
        await asyncio.sleep(1)
        
        # Start orchestration components
        await asyncio.gather(
            self._start_task_scheduler(),
            self._start_resource_monitor(),
            self._start_performance_tracker(),
            server_task
        )
    
    async def stop(self):
        """Stop the MCP host"""
        self.logger.info("Stopping MCP Host")
        
        # Stop orchestration components
        self.scheduler_running = False
        self.monitor_running = False
        
        # Cancel running tasks
        for task in self.running_tasks.values():
            task.cancel()
        
        # Stop clients
        for client in self.clients.values():
            await client.disconnect()
        
        # Stop server
        if self.server:
            await self.server.stop()
    
    async def register_agent(self, agent_id: str, agent_name: str, agent_type: str,
                           capabilities: List[str]) -> MCPClient:
        """Register a new agent with the host"""
        client = MCPClient(agent_id, agent_name, agent_type)
        
        try:
            await client.connect(capabilities)
            self.clients[agent_id] = client
            
            # Store agent info
            agent = AgentInfo(
                id=agent_id,
                name=agent_name,
                type=agent_type,
                status="active",
                capabilities=capabilities,
                current_task=None,
                performance_score=1.0,
                last_active=datetime.now().isoformat()
            )
            self.agents[agent_id] = agent
            
            self.logger.info(f"Registered agent: {agent_name} ({agent_id})")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent_id}: {e}")
            raise
    
    async def create_pipeline(self, name: str, tasks_config: List[Dict[str, Any]],
                            strategy: OrchestrationStrategy = None) -> str:
        """Create a new pipeline with tasks"""
        pipeline_id = str(uuid.uuid4())
        
        pipeline = PipelineInfo(
            id=pipeline_id,
            name=name,
            status=PipelineStatus.IDLE,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            config={
                "strategy": (strategy or self.config.strategy).value,
                "tasks_count": len(tasks_config)
            }
        )
        
        self.pipelines[pipeline_id] = pipeline
        
        # Create tasks for this pipeline
        task_ids = []
        for i, task_config in enumerate(tasks_config):
            task_id = str(uuid.uuid4())
            task = TaskInfo(
                id=task_id,
                pipeline_id=pipeline_id,
                name=task_config.get("name", f"Task-{i+1}"),
                status=TaskStatus.PENDING,
                assigned_agent=None,
                created_at=datetime.now().isoformat(),
                started_at=None,
                completed_at=None,
                config=task_config
            )
            self.tasks[task_id] = task
            task_ids.append(task_id)
        
        self.logger.info(f"Created pipeline {name} with {len(task_ids)} tasks")
        return pipeline_id
    
    async def start_pipeline(self, pipeline_id: str) -> bool:
        """Start pipeline execution"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        pipeline = self.pipelines[pipeline_id]
        if pipeline.status == PipelineStatus.RUNNING:
            return False
        
        pipeline.status = PipelineStatus.RUNNING
        pipeline.updated_at = datetime.now().isoformat()
        
        # Queue tasks for execution
        pipeline_tasks = [t for t in self.tasks.values() if t.pipeline_id == pipeline_id]
        
        strategy = OrchestrationStrategy(pipeline.config.get("strategy", "adaptive"))
        
        if strategy == OrchestrationStrategy.SEQUENTIAL:
            await self._execute_sequential(pipeline_tasks)
        elif strategy == OrchestrationStrategy.PARALLEL:
            await self._execute_parallel(pipeline_tasks)
        elif strategy == OrchestrationStrategy.PRIORITY_BASED:
            await self._execute_priority_based(pipeline_tasks)
        elif strategy == OrchestrationStrategy.RESOURCE_AWARE:
            await self._execute_resource_aware(pipeline_tasks)
        else:  # ADAPTIVE
            await self._execute_adaptive(pipeline_tasks)
        
        self.logger.info(f"Started pipeline {pipeline.name}")
        return True
    
    async def stop_pipeline(self, pipeline_id: str) -> bool:
        """Stop pipeline execution"""
        if pipeline_id not in self.pipelines:
            return False
        
        pipeline = self.pipelines[pipeline_id]
        pipeline.status = PipelineStatus.COMPLETED
        pipeline.updated_at = datetime.now().isoformat()
        
        # Cancel running tasks for this pipeline
        pipeline_tasks = [t for t in self.tasks.values() if t.pipeline_id == pipeline_id]
        for task in pipeline_tasks:
            if task.id in self.running_tasks:
                self.running_tasks[task.id].cancel()
                del self.running_tasks[task.id]
                task.status = TaskStatus.CANCELLED
        
        self.logger.info(f"Stopped pipeline {pipeline.name}")
        return True
    
    async def _execute_sequential(self, tasks: List[TaskInfo]):
        """Execute tasks sequentially"""
        for task in sorted(tasks, key=lambda t: t.created_at):
            await self._queue_task(task)
    
    async def _execute_parallel(self, tasks: List[TaskInfo]):
        """Execute tasks in parallel"""
        for task in tasks:
            await self._queue_task(task)
    
    async def _execute_priority_based(self, tasks: List[TaskInfo]):
        """Execute tasks based on priority"""
        # Sort by priority (assuming priority in config)
        sorted_tasks = sorted(tasks, 
                            key=lambda t: t.config.get("priority", 0), 
                            reverse=True)
        
        for task in sorted_tasks:
            await self._queue_task(task)
    
    async def _execute_resource_aware(self, tasks: List[TaskInfo]):
        """Execute tasks based on resource requirements"""
        # Sort by resource requirements
        sorted_tasks = sorted(tasks, 
                            key=lambda t: t.config.get("resource_weight", 1.0))
        
        for task in sorted_tasks:
            # Check if we have enough resources
            if await self._can_execute_task(task):
                await self._queue_task(task)
            else:
                # Wait for resources to become available
                await asyncio.sleep(1)
                await self._queue_task(task)
    
    async def _execute_adaptive(self, tasks: List[TaskInfo]):
        """Adaptive execution based on current conditions"""
        # Analyze current system state and choose best strategy
        cpu_usage = self.resource_usage.cpu_percent
        memory_usage = self.resource_usage.memory_mb
        active_tasks = len(self.running_tasks)
        
        if cpu_usage > 80 or memory_usage > 1000 or active_tasks > 5:
            # High load - use sequential execution
            await self._execute_sequential(tasks)
        elif len(tasks) > 10:
            # Many tasks - use priority-based
            await self._execute_priority_based(tasks)
        else:
            # Default to parallel
            await self._execute_parallel(tasks)
    
    async def _queue_task(self, task: TaskInfo):
        """Queue a task for execution"""
        await self.task_queue.put(task)
    
    async def _can_execute_task(self, task: TaskInfo) -> bool:
        """Check if task can be executed given current resources"""
        required_cpu = task.config.get("cpu_requirement", 10.0)
        required_memory = task.config.get("memory_requirement", 100.0)
        
        available_cpu = 100.0 - self.resource_usage.cpu_percent
        available_memory = 2048.0 - self.resource_usage.memory_mb  # Assuming 2GB limit
        
        return (available_cpu >= required_cpu and 
                available_memory >= required_memory and
                len(self.running_tasks) < self.config.max_concurrent_tasks)
    
    async def _start_task_scheduler(self):
        """Start the task scheduler"""
        self.scheduler_running = True
        self.logger.info("Task scheduler started")
        
        while self.scheduler_running:
            try:
                # Get next task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Find best agent for task
                agent_id = await self._find_best_agent(task)
                
                if agent_id:
                    # Assign and execute task
                    await self._execute_task(task, agent_id)
                else:
                    # No suitable agent, requeue task
                    await asyncio.sleep(1)
                    await self.task_queue.put(task)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
    
    async def _find_best_agent(self, task: TaskInfo) -> Optional[str]:
        """Find the best agent for a task"""
        required_capabilities = task.config.get("required_capabilities", [])
        
        # Find agents with required capabilities
        suitable_agents = []
        for agent_id, agent in self.agents.items():
            if (agent.status == "active" and 
                agent.current_task is None and
                all(cap in agent.capabilities for cap in required_capabilities)):
                suitable_agents.append(agent)
        
        if not suitable_agents:
            return None
        
        # Choose best agent based on performance score
        best_agent = max(suitable_agents, key=lambda a: a.performance_score or 0)
        return best_agent.id
    
    async def _execute_task(self, task: TaskInfo, agent_id: str):
        """Execute a task on an agent"""
        # Update task status
        task.status = TaskStatus.ASSIGNED
        task.assigned_agent = agent_id
        task.started_at = datetime.now().isoformat()
        
        # Update agent status
        agent = self.agents[agent_id]
        agent.current_task = task.id
        
        # Create execution coroutine
        execution_task = asyncio.create_task(
            self._run_task(task, agent_id)
        )
        
        self.running_tasks[task.id] = execution_task
        
        self.logger.info(f"Executing task {task.name} on agent {agent_id}")
    
    async def _run_task(self, task: TaskInfo, agent_id: str):
        """Run a task and handle completion"""
        try:
            client = self.clients[agent_id]
            
            # Update task status
            task.status = TaskStatus.RUNNING
            
            # Simulate task execution (in real implementation, this would
            # communicate with the agent to execute the actual task)
            execution_time = task.config.get("execution_time", 5.0)
            await asyncio.sleep(execution_time)
            
            # Mark task as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            task.result = {"status": "success", "message": "Task completed"}
            
            # Update agent
            agent = self.agents[agent_id]
            agent.current_task = None
            agent.last_active = datetime.now().isoformat()
            
            # Update performance score
            if agent.performance_score:
                agent.performance_score = min(1.0, agent.performance_score + 0.01)
            
            # Report completion
            await client.report_task_completion(task.id, task.result)
            
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            agent = self.agents[agent_id]
            agent.current_task = None
            
        except Exception as e:
            self.logger.error(f"Task execution error: {e}")
            task.status = TaskStatus.FAILED
            task.error_info = str(e)
            
            # Update agent
            agent = self.agents[agent_id]
            agent.current_task = None
            
            # Decrease performance score
            if agent.performance_score:
                agent.performance_score = max(0.1, agent.performance_score - 0.05)
        
        finally:
            # Clean up
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
    
    async def _start_resource_monitor(self):
        """Start resource monitoring"""
        self.monitor_running = True
        self.logger.info("Resource monitor started")
        
        while self.monitor_running:
            try:
                # Simulate resource monitoring (in real implementation,
                # this would use actual system monitoring tools)
                import random
                
                self.resource_usage.cpu_percent = random.uniform(10, 90)
                self.resource_usage.memory_mb = random.uniform(100, 1500)
                self.resource_usage.gpu_percent = random.uniform(0, 50)
                
                # Check for resource alerts
                if self.resource_usage.cpu_percent > 90:
                    self.logger.warning("High CPU usage detected")
                
                if self.resource_usage.memory_mb > 1800:
                    self.logger.warning("High memory usage detected")
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
    
    async def _start_performance_tracker(self):
        """Start performance tracking"""
        while self.scheduler_running:
            try:
                # Collect performance metrics
                metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "active_pipelines": len([p for p in self.pipelines.values() 
                                          if p.status == PipelineStatus.RUNNING]),
                    "running_tasks": len(self.running_tasks),
                    "queued_tasks": self.task_queue.qsize(),
                    "active_agents": len([a for a in self.agents.values() 
                                        if a.status == "active"]),
                    "resource_usage": asdict(self.resource_usage)
                }
                
                self.performance_history.append(metrics)
                
                # Keep only recent history (last hour)
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.performance_history = [
                    m for m in self.performance_history 
                    if datetime.fromisoformat(m["timestamp"]) > cutoff_time
                ]
                
                await asyncio.sleep(10)  # Track every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Performance tracking error: {e}")
    
    # API methods for external access
    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline status"""
        if pipeline_id in self.pipelines:
            return asdict(self.pipelines[pipeline_id])
        return None
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        if task_id in self.tasks:
            return asdict(self.tasks[task_id])
        return None
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent status"""
        if agent_id in self.agents:
            return asdict(self.agents[agent_id])
        return None
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            "resource_usage": asdict(self.resource_usage),
            "pipeline_count": len(self.pipelines),
            "task_count": len(self.tasks),
            "agent_count": len(self.agents),
            "running_tasks": len(self.running_tasks),
            "queued_tasks": self.task_queue.qsize()
        }
    
    def get_performance_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get performance history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            m for m in self.performance_history 
            if datetime.fromisoformat(m["timestamp"]) > cutoff_time
        ]


# Example usage
async def main():
    """Example usage of MCP Host"""
    logging.basicConfig(level=logging.INFO)
    
    host = MCPHost("orchestrator-001")
    
    try:
        # Start host
        host_task = asyncio.create_task(host.start())
        await asyncio.sleep(2)  # Let server start
        
        # Register some agents
        agent1 = await host.register_agent(
            "agent-001", "Data Processor", "data_processor", 
            ["data_processing", "validation"]
        )
        
        agent2 = await host.register_agent(
            "agent-002", "Model Trainer", "trainer", 
            ["model_training", "evaluation"]
        )
        
        # Create a pipeline
        tasks_config = [
            {
                "name": "Load Data",
                "required_capabilities": ["data_processing"],
                "execution_time": 3.0
            },
            {
                "name": "Train Model", 
                "required_capabilities": ["model_training"],
                "execution_time": 10.0
            },
            {
                "name": "Validate Results",
                "required_capabilities": ["validation"],
                "execution_time": 2.0
            }
        ]
        
        pipeline_id = await host.create_pipeline(
            "ML Training Pipeline", 
            tasks_config, 
            OrchestrationStrategy.SEQUENTIAL
        )
        
        # Start pipeline
        await host.start_pipeline(pipeline_id)
        
        # Monitor for a while
        for i in range(30):
            metrics = host.get_system_metrics()
            print(f"Metrics: {metrics['running_tasks']} running, "
                  f"{metrics['queued_tasks']} queued")
            await asyncio.sleep(1)
        
    except KeyboardInterrupt:
        print("Stopping host...")
    finally:
        await host.stop()


if __name__ == "__main__":
    asyncio.run(main())