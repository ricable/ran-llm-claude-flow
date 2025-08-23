# MCP Python Pipeline Coordination

Complete Model Context Protocol (MCP) implementation for Python machine learning pipeline coordination and agent communication.

## Features

### 🚀 Core Components

- **MCP Server**: Central coordination hub with WebSocket communication
- **MCP Client**: Agent communication interface with automatic reconnection
- **MCP Host**: Pipeline orchestration with multiple execution strategies
- **Request Handlers**: Comprehensive handlers for all MCP operations
- **Real-time Monitoring**: System, pipeline, agent, and model performance tracking

### 📊 Monitoring & Orchestration

- **System Metrics**: CPU, memory, disk, network monitoring with psutil
- **Pipeline Metrics**: Task throughput, error rates, execution times
- **Agent Performance**: Success rates, task durations, resource usage
- **Model Monitoring**: Inference times, accuracy, memory usage
- **Alert System**: Configurable thresholds with cooldown periods

### 🔧 Orchestration Strategies

- **Sequential**: Execute tasks one after another
- **Parallel**: Run tasks concurrently
- **Priority-based**: Execute by task priority
- **Resource-aware**: Consider resource requirements
- **Adaptive**: Choose strategy based on system conditions

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
import asyncio
from mcp import MCPServer, MCPClient, MCPHost
from mcp.monitoring import MCPMonitoringService

# Start MCP server
async def start_server():
    server = MCPServer(port=8700)
    await server.start()

# Connect agent client
async def create_agent():
    client = MCPClient("agent-001", "Data Processor", "worker")
    await client.connect(capabilities=["data_processing", "validation"])
    return client

# Create orchestration host
async def create_orchestrator():
    host = MCPHost()
    await host.start()
    return host

# Initialize monitoring
def setup_monitoring(server):
    monitoring = MCPMonitoringService(server)
    return monitoring
```

### Pipeline Creation

```python
# Create a pipeline with tasks
async def create_pipeline(host):
    tasks_config = [
        {
            "name": "Data Loading",
            "required_capabilities": ["data_processing"],
            "execution_time": 5.0
        },
        {
            "name": "Model Training", 
            "required_capabilities": ["model_training"],
            "execution_time": 30.0
        },
        {
            "name": "Validation",
            "required_capabilities": ["validation"],  
            "execution_time": 10.0
        }
    ]
    
    pipeline_id = await host.create_pipeline(
        "ML Training Pipeline",
        tasks_config,
        strategy=OrchestrationStrategy.ADAPTIVE
    )
    
    # Start pipeline
    await host.start_pipeline(pipeline_id)
    return pipeline_id
```

### Agent Implementation

```python
from mcp.client import MCPClient

class DataProcessingAgent:
    def __init__(self, agent_id: str):
        self.client = MCPClient(agent_id, "Data Processor", "data_processor")
        
    async def start(self):
        await self.client.connect(capabilities=["data_processing", "validation"])
        self.client.register_handler("task/assigned", self.handle_task)
        
    async def handle_task(self, params):
        task_id = params.get("task_id")
        
        # Report progress
        await self.client.report_task_progress(task_id, 0.0, "Starting")
        
        # Process data (your implementation here)
        result = await self.process_data(params)
        
        # Report completion
        await self.client.report_task_completion(task_id, result)
        
    async def process_data(self, params):
        # Your data processing logic
        import asyncio
        await asyncio.sleep(2)  # Simulate work
        return {"status": "completed", "processed_items": 1000}
```

## Architecture

### MCP Protocol Structure

```
┌─────────────────┐    WebSocket     ┌─────────────────┐
│   MCP Client    │◄────────────────►│   MCP Server    │
│   (Agents)      │                  │  (Coordinator)  │
└─────────────────┘                  └─────────────────┘
        │                                      │
        │                                      │
        ▼                                      ▼
┌─────────────────┐                  ┌─────────────────┐
│ Task Execution  │                  │ Pipeline        │
│ & Reporting     │                  │ Management      │
└─────────────────┘                  └─────────────────┘
```

### Data Flow

1. **Pipeline Creation**: Host creates pipelines with task configurations
2. **Agent Registration**: Agents register with server, declare capabilities
3. **Task Assignment**: Server matches tasks to agents based on capabilities
4. **Execution**: Agents execute tasks, report progress and results
5. **Monitoring**: Real-time metrics collection and alerting
6. **Completion**: Pipeline completion notification and cleanup

## API Reference

### MCP Methods

#### Pipeline Operations
- `pipeline/create` - Create new pipeline
- `pipeline/start` - Start pipeline execution
- `pipeline/stop` - Stop pipeline
- `pipeline/pause` - Pause execution
- `pipeline/resume` - Resume execution
- `pipeline/status` - Get pipeline status
- `pipeline/list` - List all pipelines
- `pipeline/delete` - Delete pipeline

#### Task Operations  
- `task/create` - Create new task
- `task/assign` - Assign task to agent
- `task/status` - Get task status
- `task/result` - Get task result
- `task/cancel` - Cancel task
- `task/list` - List tasks

#### Agent Operations
- `agent/register` - Register agent
- `agent/unregister` - Unregister agent  
- `agent/list` - List agents
- `agent/status` - Get agent status
- `agent/assign` - Assign agent to task

#### Model Monitoring
- `model/metrics` - Get model metrics
- `model/performance` - Get performance data
- `model/memory` - Get memory usage
- `model/health` - Get health status

### Message Format

```json
{
  "id": "uuid",
  "type": "request|response|notification|error",
  "method": "pipeline/create", 
  "timestamp": "2024-01-01T00:00:00Z",
  "params": {},
  "result": {},
  "error": {}
}
```

### Data Structures

#### Pipeline Info
```python
@dataclass
class PipelineInfo:
    id: str
    name: str
    status: PipelineStatus  # IDLE, RUNNING, PAUSED, COMPLETED, FAILED
    created_at: str
    updated_at: str
    config: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    error_info: Optional[str] = None
```

#### Agent Info
```python
@dataclass
class AgentInfo:
    id: str
    name: str
    type: str
    status: str
    capabilities: List[str]
    current_task: Optional[str]
    performance_score: Optional[float]
    last_active: str
```

## Monitoring

### Metrics Collection

The monitoring system collects various metrics:

- **System Metrics**: CPU, memory, disk usage via psutil
- **Pipeline Metrics**: Task throughput, error rates, execution times
- **Agent Metrics**: Success rates, task durations, resource usage  
- **Model Metrics**: Inference times, accuracy, memory usage

### Alert System

Configurable alerts with thresholds:

```python
monitoring.set_alert_threshold("cpu_usage", 90.0)
monitoring.set_alert_threshold("error_rate", 0.1) 
monitoring.set_alert_threshold("task_failure_rate", 0.2)
```

### Metrics API

```python
# Get current metrics
current = monitoring.get_current_metrics()

# Get historical data
history = monitoring.get_metrics_history("system", duration_minutes=60)

# Set custom thresholds
monitoring.set_alert_threshold("model_inference_time", 200.0)
```

## Integration with Claude Flow

This MCP implementation is designed to integrate seamlessly with the Claude Flow ecosystem:

### Memory Integration
```python
# Store pipeline state in claude-flow memory
await client.set_memory("pipeline/config", pipeline_config)
config = await client.get_memory("pipeline/config")
```

### Hook Integration  
```python
# Use claude-flow hooks for coordination
await client.send_notification("claude-flow/hook", {
    "type": "pre-task",
    "task_id": task_id,
    "agent_id": agent_id
})
```

## Configuration

### Environment Variables

```bash
# MCP Configuration
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8700
MCP_CLIENT_TIMEOUT=30
MCP_MAX_RETRIES=3

# Monitoring Configuration  
MONITORING_INTERVAL=10
ALERT_COOLDOWN=300
METRICS_RETENTION_HOURS=24

# Resource Limits
MAX_CONCURRENT_TASKS=10
MAX_MEMORY_MB=2048
MAX_CPU_PERCENT=90
```

### Configuration File

```yaml
# config/mcp.yaml
server:
  host: localhost
  port: 8700
  max_connections: 100

orchestration:
  strategy: adaptive
  max_concurrent_tasks: 10
  task_timeout: 300.0
  retry_attempts: 3

monitoring:
  system_interval: 5.0
  pipeline_interval: 10.0
  agent_interval: 10.0
  model_interval: 15.0
  
alerts:
  cpu_threshold: 90.0
  memory_threshold: 85.0
  error_rate_threshold: 0.1
  cooldown_seconds: 300
```

## Development

### Running Tests

```bash
pytest tests/ -v --asyncio-mode=auto
```

### Code Formatting

```bash
black src/
mypy src/
```

### Example Integration

See `examples/` directory for complete integration examples with different orchestration strategies and monitoring configurations.

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

- Documentation: This README and inline code documentation
- Issues: Submit issues via the repository issue tracker  
- Integration: Compatible with claude-flow MCP ecosystem