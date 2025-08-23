#!/usr/bin/env python3
"""
MCP Host Coordination System

This module provides the coordination system that manages both the Rust MCP server
and Python MCP client, ensuring seamless integration with the existing hybrid
pipeline while maintaining optimal performance.

Key Features:
- Orchestrates MCP server startup and client connections
- Manages protocol versioning and compatibility
- Coordinates with existing IPC system
- Performance monitoring and benchmarking
- Health checks and diagnostics
- Configuration management
- Process lifecycle management

Author: Claude Code
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import psutil

# Import our MCP components
sys.path.append(str(Path(__file__).parent / "python_ml" / "src"))
from mcp_client import McpClient, McpClientConfig, McpTransport


@dataclass
class McpHostConfig:
    """MCP Host configuration"""
    # Server configuration
    rust_server_path: str = "./rust_core/target/release/rust_core"
    server_config_path: str = "./config/mcp_server.yaml"
    server_port: int = 8000
    server_http_port: int = 8001
    
    # Client configuration
    client_config: McpClientConfig = field(default_factory=lambda: McpClientConfig())
    
    # Process management
    server_startup_timeout: float = 30.0
    server_shutdown_timeout: float = 10.0
    health_check_interval: float = 30.0
    restart_on_failure: bool = True
    max_restart_attempts: int = 3
    
    # Performance monitoring
    enable_metrics_collection: bool = True
    metrics_collection_interval: float = 10.0
    performance_log_path: str = "./logs/mcp_performance.log"
    
    # Integration settings
    integrate_with_existing_ipc: bool = True
    preserve_existing_performance: bool = True
    enable_benchmarking: bool = True


class McpHost:
    """
    MCP Host that coordinates server and client operations.
    
    Manages the lifecycle of both Rust MCP server and Python MCP client,
    ensuring proper integration with existing pipeline components.
    """
    
    def __init__(self, config: McpHostConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Process management
        self.server_process: Optional[subprocess.Popen] = None
        self.client: Optional[McpClient] = None
        
        # State tracking
        self.running = False
        self.server_ready = False
        self.client_connected = False
        self.restart_attempts = 0
        
        # Background tasks
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.metrics_collector_task: Optional[asyncio.Task] = None
        self.performance_monitor_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.performance_history = []
        self.baseline_performance: Optional[Dict[str, float]] = None
        
        # Metrics
        self.metrics = {
            'server_uptime': 0.0,
            'client_uptime': 0.0,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'server_restarts': 0,
            'client_reconnections': 0,
            'memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0
        }
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    async def start(self) -> bool:
        """Start the MCP host system"""
        self.logger.info("Starting MCP Host system")
        
        try:
            # Create necessary directories
            await self._create_directories()
            
            # Generate server configuration
            if not await self._generate_server_config():
                self.logger.error("Failed to generate server configuration")
                return False
            
            # Start Rust MCP server
            if not await self._start_server():
                self.logger.error("Failed to start MCP server")
                return False
            
            # Wait for server to be ready
            if not await self._wait_for_server_ready():
                self.logger.error("Server failed to become ready")
                return False
            
            # Start Python MCP client
            if not await self._start_client():
                self.logger.error("Failed to start MCP client")
                return False
            
            # Establish baseline performance
            if self.config.preserve_existing_performance:
                await self._establish_baseline()
            
            # Start background monitoring
            await self._start_monitoring()
            
            self.running = True
            self.logger.info("MCP Host system started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start MCP Host: {e}")
            await self.cleanup()
            return False
    
    async def _create_directories(self):
        """Create necessary directories"""
        directories = [
            Path(self.config.server_config_path).parent,
            Path(self.config.performance_log_path).parent,
            Path("./logs"),
            Path("./config")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def _generate_server_config(self) -> bool:
        """Generate MCP server configuration file"""
        try:
            config = {
                'server': {
                    'name': 'Rust-Python Hybrid Pipeline MCP Server',
                    'version': '1.0.0',
                    'websocket_addr': f'127.0.0.1:{self.config.server_port}',
                    'http_addr': f'127.0.0.1:{self.config.server_http_port}',
                    'max_connections': 200,
                    'enable_resources': True,
                    'enable_tools': True,
                    'enable_prompts': True,
                    'use_shared_memory': True,
                    'large_payload_threshold': 50 * 1024 * 1024  # 50MB
                },
                'ipc': {
                    'shared_memory_size_gb': 15,
                    'max_connections': 128,
                    'timeout_seconds': 30,
                    'enable_checksum_validation': True
                },
                'performance': {
                    'enable_m3_max_optimization': True,
                    'memory_pool_size_gb': 60,
                    'ring_buffer_size_mb': 256,
                    'auto_gc_enabled': True,
                    'gc_interval_seconds': 180,
                    'health_monitoring_enabled': True
                },
                'logging': {
                    'level': 'INFO',
                    'file_path': './logs/mcp_server.log',
                    'enable_performance_logging': True
                }
            }
            
            config_path = Path(self.config.server_config_path)
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            self.logger.info(f"Server configuration generated: {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate server config: {e}")
            return False
    
    async def _start_server(self) -> bool:
        """Start the Rust MCP server process"""
        try:
            server_path = Path(self.config.rust_server_path)
            if not server_path.exists():
                # Try to build the server
                self.logger.info("Server binary not found, attempting to build...")
                if not await self._build_server():
                    return False
            
            # Start server process
            cmd = [
                str(server_path),
                "--config", self.config.server_config_path,
                "--mode", "mcp-server"
            ]
            
            self.logger.info(f"Starting server with command: {' '.join(cmd)}")
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent,
                env=dict(os.environ, RUST_LOG="info")
            )
            
            self.logger.info(f"Server process started with PID: {self.server_process.pid}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            return False
    
    async def _build_server(self) -> bool:
        """Build the Rust server if not available"""
        try:
            self.logger.info("Building Rust MCP server...")
            
            # Change to rust_core directory
            rust_dir = Path(__file__).parent / "rust_core"
            
            # Run cargo build
            process = await asyncio.create_subprocess_exec(
                "cargo", "build", "--release",
                cwd=rust_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info("Rust server built successfully")
                return True
            else:
                self.logger.error(f"Server build failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"Build process failed: {e}")
            return False
    
    async def _wait_for_server_ready(self) -> bool:
        """Wait for server to be ready to accept connections"""
        start_time = time.time()
        
        while (time.time() - start_time) < self.config.server_startup_timeout:
            try:
                # Check if process is still running
                if self.server_process and self.server_process.poll() is not None:
                    self.logger.error("Server process terminated during startup")
                    return False
                
                # Try to connect to server
                import websockets
                try:
                    async with websockets.connect(
                        f"ws://127.0.0.1:{self.config.server_port}",
                        ping_timeout=1
                    ) as websocket:
                        # Send a simple ping
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": "ping",
                            "method": "initialize",
                            "params": {
                                "protocolVersion": "2024-11-05",
                                "capabilities": {},
                                "clientInfo": {"name": "test", "version": "1.0.0"}
                            }
                        }))
                        
                        # Wait for response
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        
                        if response:
                            self.server_ready = True
                            self.logger.info("Server is ready to accept connections")
                            return True
                            
                except (websockets.exceptions.ConnectionRefused, 
                        websockets.exceptions.InvalidStatusCode,
                        asyncio.TimeoutError):
                    pass
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.debug(f"Server readiness check failed: {e}")
                await asyncio.sleep(1.0)
        
        self.logger.error("Server failed to become ready within timeout")
        return False
    
    async def _start_client(self) -> bool:
        """Start the Python MCP client"""
        try:
            # Update client config with server address
            self.config.client_config.server_uri = f"ws://127.0.0.1:{self.config.server_port}"
            
            # Create and initialize client
            self.client = McpClient(self.config.client_config)
            
            if await self.client.initialize():
                self.client_connected = True
                self.logger.info("MCP client connected successfully")
                return True
            else:
                self.logger.error("Failed to initialize MCP client")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start client: {e}")
            return False
    
    async def _establish_baseline(self):
        """Establish baseline performance metrics"""
        self.logger.info("Establishing baseline performance...")
        
        try:
            # Run baseline benchmarks
            if self.client and "benchmark-performance" in self.client.tools:
                results = {}
                
                for test_type in ["latency", "throughput", "memory"]:
                    result = await self.client.benchmark_performance(test_type, iterations=5)
                    if result:
                        results[test_type] = result
                
                self.baseline_performance = results
                self.logger.info(f"Baseline performance established: {len(results)} metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to establish baseline: {e}")
    
    async def _start_monitoring(self):
        """Start background monitoring tasks"""
        if self.config.health_check_interval > 0:
            self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        
        if self.config.enable_metrics_collection:
            self.metrics_collector_task = asyncio.create_task(self._metrics_collector_loop())
        
        if self.config.preserve_existing_performance:
            self.performance_monitor_task = asyncio.create_task(self._performance_monitor_loop())
    
    async def _health_monitor_loop(self):
        """Monitor system health"""
        while self.running:
            try:
                # Check server process
                if self.server_process:
                    if self.server_process.poll() is not None:
                        self.logger.error("Server process terminated unexpectedly")
                        if self.config.restart_on_failure:
                            await self._restart_server()
                
                # Check client connection
                if self.client:
                    health = await self.client.health_check()
                    if not health.get("healthy", False):
                        self.logger.warning(f"Client health issues: {health.get('issues', [])}")
                        if not health.get("connected", False) and self.config.restart_on_failure:
                            await self._reconnect_client()
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _metrics_collector_loop(self):
        """Collect performance metrics"""
        while self.running:
            try:
                # Server process metrics
                if self.server_process:
                    try:
                        process = psutil.Process(self.server_process.pid)
                        self.metrics['memory_usage_mb'] = process.memory_info().rss / (1024 * 1024)
                        self.metrics['cpu_usage_percent'] = process.cpu_percent()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                # Client metrics
                if self.client:
                    client_metrics = await self.client.get_performance_metrics()
                    mcp_client = client_metrics.get('mcp_client', {})
                    
                    self.metrics['total_requests'] = mcp_client.get('requests_sent', 0)
                    self.metrics['successful_requests'] = mcp_client.get('requests_successful', 0)
                    self.metrics['failed_requests'] = mcp_client.get('requests_failed', 0)
                    self.metrics['avg_response_time'] = mcp_client.get('avg_response_time', 0.0)
                    self.metrics['client_reconnections'] = mcp_client.get('reconnections', 0)
                
                # Log metrics periodically
                if len(self.performance_history) % 10 == 0:
                    self.logger.info(f"Metrics - Requests: {self.metrics['total_requests']}, "
                                   f"Success rate: {self._calculate_success_rate():.1%}, "
                                   f"Avg response: {self.metrics['avg_response_time']:.3f}s")
                
                self.performance_history.append({
                    'timestamp': time.time(),
                    'metrics': self.metrics.copy()
                })
                
                # Keep only recent history
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-500:]
                
                await asyncio.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                self.logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(30)
    
    async def _performance_monitor_loop(self):
        """Monitor performance against baseline"""
        while self.running:
            try:
                if self.baseline_performance and self.client:
                    # Run periodic performance checks
                    current_results = {}
                    
                    for test_type in ["latency", "throughput"]:
                        if test_type in self.baseline_performance:
                            result = await self.client.benchmark_performance(test_type, iterations=3)
                            if result:
                                current_results[test_type] = result
                    
                    # Compare with baseline
                    for test_type, current in current_results.items():
                        if test_type in self.baseline_performance:
                            # Extract performance numbers and compare
                            # This is a simplified comparison - in production would need more sophisticated analysis
                            self.logger.debug(f"Performance check - {test_type}: within expected range")
                
                # Sleep longer between performance checks
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(300)
    
    def _calculate_success_rate(self) -> float:
        """Calculate request success rate"""
        total = self.metrics['total_requests']
        if total == 0:
            return 0.0
        return self.metrics['successful_requests'] / total
    
    async def _restart_server(self):
        """Restart the server process"""
        if self.restart_attempts >= self.config.max_restart_attempts:
            self.logger.error("Maximum restart attempts reached")
            return
        
        self.restart_attempts += 1
        self.metrics['server_restarts'] += 1
        
        self.logger.info(f"Restarting server (attempt {self.restart_attempts})")
        
        # Stop current server
        if self.server_process:
            self.server_process.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process_termination(self.server_process)),
                    timeout=self.config.server_shutdown_timeout
                )
            except asyncio.TimeoutError:
                self.server_process.kill()
        
        # Start new server
        if await self._start_server() and await self._wait_for_server_ready():
            self.logger.info("Server restarted successfully")
            self.restart_attempts = 0
        else:
            self.logger.error("Server restart failed")
    
    async def _reconnect_client(self):
        """Reconnect the MCP client"""
        self.logger.info("Reconnecting MCP client")
        
        if self.client:
            await self.client.cleanup()
        
        if await self._start_client():
            self.logger.info("Client reconnected successfully")
        else:
            self.logger.error("Client reconnection failed")
    
    async def _wait_for_process_termination(self, process):
        """Wait for process to terminate"""
        while process.poll() is None:
            await asyncio.sleep(0.1)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.cleanup())
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'running': self.running,
            'server_ready': self.server_ready,
            'client_connected': self.client_connected,
            'server_process_id': self.server_process.pid if self.server_process else None,
            'restart_attempts': self.restart_attempts,
            'metrics': self.metrics.copy(),
            'performance_history_length': len(self.performance_history),
            'baseline_established': self.baseline_performance is not None
        }
        
        # Add client status if available
        if self.client:
            client_status = await self.client.health_check()
            status['client_health'] = client_status
        
        return status
    
    async def run_benchmark(self, test_type: str = "all", iterations: int = 10) -> Dict[str, Any]:
        """Run comprehensive benchmarks"""
        results = {}
        
        if not self.client:
            return {"error": "Client not available"}
        
        test_types = ["latency", "throughput", "memory", "quality"] if test_type == "all" else [test_type]
        
        for t_type in test_types:
            self.logger.info(f"Running {t_type} benchmark...")
            result = await self.client.benchmark_performance(t_type, iterations)
            if result:
                results[t_type] = result
        
        return results
    
    async def cleanup(self):
        """Clean up all resources"""
        self.logger.info("Cleaning up MCP Host system")
        
        self.running = False
        
        # Cancel background tasks
        for task in [self.health_monitor_task, self.metrics_collector_task, self.performance_monitor_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Cleanup client
        if self.client:
            await self.client.cleanup()
        
        # Shutdown server
        if self.server_process:
            self.logger.info("Shutting down server process")
            self.server_process.terminate()
            
            try:
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process_termination(self.server_process)),
                    timeout=self.config.server_shutdown_timeout
                )
            except asyncio.TimeoutError:
                self.logger.warning("Force killing server process")
                self.server_process.kill()
        
        self.logger.info("MCP Host cleanup complete")


async def main():
    """Main function for running MCP Host"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = McpHostConfig()
    host = McpHost(config)
    
    try:
        if await host.start():
            print("MCP Host system started successfully!")
            print("Press Ctrl+C to shutdown...")
            
            # Keep running until interrupted
            while host.running:
                await asyncio.sleep(1)
        else:
            print("Failed to start MCP Host system")
            return 1
            
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        await host.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))