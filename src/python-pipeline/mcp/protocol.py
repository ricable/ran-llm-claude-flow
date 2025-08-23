"""
MCP Protocol Definitions for Python Pipeline Coordination

This module defines the Model Context Protocol schemas, types, and constants
for coordinating Python machine learning pipelines and agents.
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid
from datetime import datetime


class MCPMessageType(Enum):
    """MCP message types for pipeline coordination"""
    REQUEST = "request"
    RESPONSE = "response" 
    NOTIFICATION = "notification"
    ERROR = "error"


class PipelineStatus(Enum):
    """Pipeline execution statuses"""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"


class TaskStatus(Enum):
    """Individual task statuses"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class MCPMessage:
    """Base MCP message structure"""
    id: str
    type: MCPMessageType
    method: str
    timestamp: str
    params: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'MCPMessage':
        """Create message from JSON string"""
        data = json.loads(json_str)
        data['type'] = MCPMessageType(data['type'])
        return cls(**data)


@dataclass
class PipelineInfo:
    """Pipeline information structure"""
    id: str
    name: str
    status: PipelineStatus
    created_at: str
    updated_at: str
    config: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    error_info: Optional[str] = None


@dataclass
class TaskInfo:
    """Task information structure"""
    id: str
    pipeline_id: str
    name: str
    status: TaskStatus
    assigned_agent: Optional[str]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    config: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error_info: Optional[str] = None


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_id: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    loss: Optional[float] = None
    training_time: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    timestamp: str = ""


@dataclass
class AgentInfo:
    """Agent information structure"""
    id: str
    name: str
    type: str
    status: str
    capabilities: List[str]
    current_task: Optional[str]
    performance_score: Optional[float]
    last_active: str


class MCPMethods:
    """MCP method names for pipeline operations"""
    
    # Pipeline control
    PIPELINE_START = "pipeline/start"
    PIPELINE_STOP = "pipeline/stop"
    PIPELINE_PAUSE = "pipeline/pause"
    PIPELINE_RESUME = "pipeline/resume"
    PIPELINE_STATUS = "pipeline/status"
    PIPELINE_LIST = "pipeline/list"
    PIPELINE_CREATE = "pipeline/create"
    PIPELINE_DELETE = "pipeline/delete"
    
    # Task orchestration
    TASK_CREATE = "task/create"
    TASK_ASSIGN = "task/assign"
    TASK_STATUS = "task/status"
    TASK_RESULT = "task/result"
    TASK_CANCEL = "task/cancel"
    TASK_LIST = "task/list"
    
    # Model monitoring
    MODEL_METRICS = "model/metrics"
    MODEL_PERFORMANCE = "model/performance"
    MODEL_MEMORY = "model/memory"
    MODEL_HEALTH = "model/health"
    
    # Agent coordination
    AGENT_REGISTER = "agent/register"
    AGENT_UNREGISTER = "agent/unregister"
    AGENT_LIST = "agent/list"
    AGENT_STATUS = "agent/status"
    AGENT_ASSIGN = "agent/assign"
    
    # Error handling
    ERROR_REPORT = "error/report"
    ERROR_RECOVER = "error/recover"
    ERROR_LIST = "error/list"
    
    # Memory operations
    MEMORY_GET = "memory/get"
    MEMORY_SET = "memory/set"
    MEMORY_DELETE = "memory/delete"
    MEMORY_LIST = "memory/list"


class MCPErrors:
    """MCP error codes and messages"""
    
    INVALID_REQUEST = {"code": -32600, "message": "Invalid Request"}
    METHOD_NOT_FOUND = {"code": -32601, "message": "Method not found"}
    INVALID_PARAMS = {"code": -32602, "message": "Invalid params"}
    INTERNAL_ERROR = {"code": -32603, "message": "Internal error"}
    
    PIPELINE_NOT_FOUND = {"code": -40001, "message": "Pipeline not found"}
    PIPELINE_ALREADY_RUNNING = {"code": -40002, "message": "Pipeline already running"}
    PIPELINE_NOT_RUNNING = {"code": -40003, "message": "Pipeline not running"}
    
    TASK_NOT_FOUND = {"code": -40011, "message": "Task not found"}
    TASK_ALREADY_ASSIGNED = {"code": -40012, "message": "Task already assigned"}
    
    AGENT_NOT_FOUND = {"code": -40021, "message": "Agent not found"}
    AGENT_NOT_AVAILABLE = {"code": -40022, "message": "Agent not available"}
    
    MODEL_NOT_FOUND = {"code": -40031, "message": "Model not found"}
    MODEL_INVALID_STATE = {"code": -40032, "message": "Model in invalid state"}


def create_request(method: str, params: Optional[Dict[str, Any]] = None) -> MCPMessage:
    """Create a new MCP request message"""
    return MCPMessage(
        id=str(uuid.uuid4()),
        type=MCPMessageType.REQUEST,
        method=method,
        timestamp=datetime.now().isoformat(),
        params=params or {}
    )


def create_response(request_id: str, result: Optional[Dict[str, Any]] = None, 
                   error: Optional[Dict[str, Any]] = None) -> MCPMessage:
    """Create a new MCP response message"""
    return MCPMessage(
        id=request_id,
        type=MCPMessageType.RESPONSE,
        method="",
        timestamp=datetime.now().isoformat(),
        result=result,
        error=error
    )


def create_notification(method: str, params: Optional[Dict[str, Any]] = None) -> MCPMessage:
    """Create a new MCP notification message"""
    return MCPMessage(
        id=str(uuid.uuid4()),
        type=MCPMessageType.NOTIFICATION,
        method=method,
        timestamp=datetime.now().isoformat(),
        params=params or {}
    )


# Protocol configuration constants
MCP_VERSION = "2024-11-05"
PROTOCOL_NAME = "claude-flow-python-pipeline"
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB
DEFAULT_TIMEOUT = 30.0  # seconds
HEARTBEAT_INTERVAL = 5.0  # seconds
MAX_RETRY_ATTEMPTS = 3

# Default port configurations
DEFAULT_SERVER_PORT = 8700
DEFAULT_CLIENT_PORT = 8701
DEFAULT_HOST_PORT = 8702