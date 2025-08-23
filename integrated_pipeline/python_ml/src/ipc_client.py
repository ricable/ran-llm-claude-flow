#!/usr/bin/env python3
"""
IPC Client for Rust-Python Communication

Handles Inter-Process Communication between Python ML engine and Rust core
using named pipes and shared memory for high-performance data transfer.

Key Features:
- Named pipe communication with Rust core
- Shared memory access for zero-copy operations
- Data serialization/deserialization optimization
- Error handling and connection recovery
- Health monitoring and automatic reconnection

Author: Claude Code
Version: 1.0.0
"""

import asyncio
import logging
import json
import time
import mmap
import struct
import hashlib
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
import pickle
import zlib
import os

# For shared memory (Python 3.8+)
try:
    from multiprocessing import shared_memory
    SHARED_MEMORY_AVAILABLE = True
except ImportError:
    SHARED_MEMORY_AVAILABLE = False
    logging.warning("Shared memory not available, falling back to file-based IPC")


class MessageType(Enum):
    """IPC message types"""
    HEARTBEAT = "heartbeat"
    PROCESS_DOCUMENT = "process_document"
    BATCH_PROCESS = "batch_process"
    QA_GENERATION = "qa_generation"
    QUALITY_ASSESSMENT = "quality_assessment"
    MODEL_STATUS = "model_status"
    PERFORMANCE_METRICS = "performance_metrics"
    ERROR = "error"
    SHUTDOWN = "shutdown"
    RESPONSE = "response"


class Priority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class IPCMessage:
    """IPC message structure"""
    message_type: MessageType
    message_id: str
    payload: Dict[str, Any]
    priority: Priority = Priority.NORMAL
    timestamp: float = field(default_factory=time.time)
    response_expected: bool = True
    timeout: float = 30.0
    checksum: str = ""
    
    def __post_init__(self):
        """Calculate message checksum"""
        if not self.checksum:
            payload_str = json.dumps(self.payload, sort_keys=True)
            self.checksum = hashlib.md5(payload_str.encode()).hexdigest()
            
    def to_bytes(self) -> bytes:
        """Serialize message to bytes"""
        data = {
            'type': self.message_type.value,
            'id': self.message_id,
            'payload': self.payload,
            'priority': self.priority.value,
            'timestamp': self.timestamp,
            'response_expected': self.response_expected,
            'timeout': self.timeout,
            'checksum': self.checksum
        }
        
        # Compress for efficiency
        json_bytes = json.dumps(data).encode('utf-8')
        compressed = zlib.compress(json_bytes)
        
        # Add length header (4 bytes)
        length_header = struct.pack('<I', len(compressed))
        return length_header + compressed
        
    @classmethod
    def from_bytes(cls, data: bytes) -> 'IPCMessage':
        """Deserialize message from bytes"""
        try:
            # Decompress
            decompressed = zlib.decompress(data)
            msg_data = json.loads(decompressed.decode('utf-8'))
            
            return cls(
                message_type=MessageType(msg_data['type']),
                message_id=msg_data['id'],
                payload=msg_data['payload'],
                priority=Priority(msg_data['priority']),
                timestamp=msg_data['timestamp'],
                response_expected=msg_data['response_expected'],
                timeout=msg_data['timeout'],
                checksum=msg_data['checksum']
            )
        except Exception as e:
            raise ValueError(f"Failed to deserialize message: {e}")


@dataclass
class DocumentProcessingRequest:
    """Document processing request structure"""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    processing_hints: Dict[str, Any]
    qa_target_count: int = 5
    quality_threshold: float = 0.7
    

@dataclass
class ProcessingResponse:
    """Processing response structure"""
    document_id: str
    success: bool
    qa_pairs: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    error_message: str = ""
    model_used: str = ""


class SharedMemoryManager:
    """
    Manages shared memory regions for zero-copy data transfer.
    
    Uses system shared memory for large data transfers between
    Rust and Python processes.
    """
    
    def __init__(self, region_size_mb: int = 1024):
        self.logger = logging.getLogger(__name__)
        self.region_size = region_size_mb * 1024 * 1024  # Convert to bytes
        self.regions = {}
        self.region_locks = {}
        self.allocation_map = {}  # Track allocations within regions
        
        if not SHARED_MEMORY_AVAILABLE:
            self.logger.warning("Shared memory not available, using file-based fallback")
            self.temp_dir = Path("/tmp/ipc_shared")
            self.temp_dir.mkdir(exist_ok=True)
            
    def create_region(self, region_name: str) -> bool:
        """Create a shared memory region"""
        try:
            if SHARED_MEMORY_AVAILABLE:
                # Create shared memory
                shm = shared_memory.SharedMemory(
                    name=region_name,
                    create=True,
                    size=self.region_size
                )
                self.regions[region_name] = shm
                self.region_locks[region_name] = threading.Lock()
                self.allocation_map[region_name] = {'allocated': 0, 'blocks': []}
                
                self.logger.info(
                    f"Created shared memory region '{region_name}' "
                    f"({self.region_size // 1024 // 1024}MB)"
                )
            else:
                # File-based fallback
                file_path = self.temp_dir / f"{region_name}.shm"
                with open(file_path, 'wb') as f:
                    f.write(b'\x00' * self.region_size)  # Initialize with zeros
                    
                self.regions[region_name] = file_path
                self.region_locks[region_name] = threading.Lock()
                self.allocation_map[region_name] = {'allocated': 0, 'blocks': []}
                
                self.logger.info(f"Created file-based shared region '{region_name}'")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create shared memory region '{region_name}': {e}")
            return False
            
    def connect_region(self, region_name: str) -> bool:
        """Connect to existing shared memory region"""
        try:
            if SHARED_MEMORY_AVAILABLE:
                shm = shared_memory.SharedMemory(name=region_name)
                self.regions[region_name] = shm
                self.region_locks[region_name] = threading.Lock()
                
                self.logger.info(f"Connected to shared memory region '{region_name}'")
            else:
                # File-based fallback
                file_path = self.temp_dir / f"{region_name}.shm"
                if file_path.exists():
                    self.regions[region_name] = file_path
                    self.region_locks[region_name] = threading.Lock()
                    
                    self.logger.info(f"Connected to file-based region '{region_name}'")
                else:
                    raise FileNotFoundError(f"Region file not found: {file_path}")
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to shared memory region '{region_name}': {e}")
            return False
            
    def write_data(self, region_name: str, data: bytes) -> Optional[int]:
        """Write data to shared memory region"""
        if region_name not in self.regions:
            self.logger.error(f"Region '{region_name}' not found")
            return None
            
        with self.region_locks[region_name]:
            try:
                if SHARED_MEMORY_AVAILABLE:
                    shm = self.regions[region_name]
                    
                    # Simple allocation: write at beginning with length header
                    if len(data) + 4 > shm.size:
                        self.logger.error(f"Data too large for region '{region_name}'")
                        return None
                        
                    # Write length header (4 bytes) + data
                    struct.pack_into('<I', shm.buf, 0, len(data))
                    shm.buf[4:4+len(data)] = data
                    
                    return 0  # Return offset
                else:
                    # File-based fallback
                    file_path = self.regions[region_name]
                    
                    with open(file_path, 'r+b') as f:
                        if len(data) + 4 > self.region_size:
                            self.logger.error(f"Data too large for region '{region_name}'")
                            return None
                            
                        # Write length header + data
                        f.write(struct.pack('<I', len(data)))
                        f.write(data)
                        f.flush()
                        
                    return 0
                    
            except Exception as e:
                self.logger.error(f"Failed to write to shared memory: {e}")
                return None
                
    def read_data(self, region_name: str, offset: int = 0) -> Optional[bytes]:
        """Read data from shared memory region"""
        if region_name not in self.regions:
            self.logger.error(f"Region '{region_name}' not found")
            return None
            
        with self.region_locks[region_name]:
            try:
                if SHARED_MEMORY_AVAILABLE:
                    shm = self.regions[region_name]
                    
                    # Read length header
                    data_length = struct.unpack('<I', shm.buf[offset:offset+4])[0]
                    
                    # Read data
                    data = bytes(shm.buf[offset+4:offset+4+data_length])
                    return data
                else:
                    # File-based fallback
                    file_path = self.regions[region_name]
                    
                    with open(file_path, 'rb') as f:
                        f.seek(offset)
                        
                        # Read length header
                        length_bytes = f.read(4)
                        if len(length_bytes) != 4:
                            return None
                            
                        data_length = struct.unpack('<I', length_bytes)[0]
                        
                        # Read data
                        data = f.read(data_length)
                        return data
                        
            except Exception as e:
                self.logger.error(f"Failed to read from shared memory: {e}")
                return None
                
    def cleanup_region(self, region_name: str):
        """Clean up shared memory region"""
        if region_name in self.regions:
            try:
                if SHARED_MEMORY_AVAILABLE:
                    shm = self.regions[region_name]
                    shm.close()
                    try:
                        shm.unlink()  # Remove from system
                    except FileNotFoundError:
                        pass  # Already removed
                else:
                    # Remove file
                    file_path = self.regions[region_name]
                    if file_path.exists():
                        file_path.unlink()
                        
                del self.regions[region_name]
                del self.region_locks[region_name]
                
                self.logger.info(f"Cleaned up shared memory region '{region_name}'")
            except Exception as e:
                self.logger.error(f"Failed to cleanup region '{region_name}': {e}")


class IPCClient:
    """
    IPC Client for communicating with Rust core process.
    
    Handles bidirectional communication via named pipes and shared memory,
    with automatic reconnection and error recovery.
    """
    
    def __init__(self, pipe_path: str = "/tmp/rust_python_ipc"):
        self.logger = logging.getLogger(__name__)
        self.pipe_path = Path(pipe_path)
        self.pipe_read_path = Path(f"{pipe_path}_read")
        self.pipe_write_path = Path(f"{pipe_path}_write")
        
        # Connection state
        self.connected = False
        self.connection_lock = threading.Lock()
        
        # Shared memory manager
        self.shared_memory = SharedMemoryManager()
        
        # Message handling
        self.message_counter = 0
        self.pending_responses = {}  # message_id -> Future
        self.message_handlers = {}  # MessageType -> handler function
        self.response_timeout = 30.0
        
        # Background tasks
        self.reader_task = None
        self.heartbeat_task = None
        self.cleanup_task = None
        self.running = False
        
        # Performance metrics
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0,
            'reconnections': 0,
            'avg_response_time': 0.0,
            'bytes_transferred': 0
        }
        
        # Error recovery
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0
        
        # Register default handlers
        self._register_default_handlers()
        
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self.message_handlers[MessageType.ERROR] = self._handle_error
        self.message_handlers[MessageType.RESPONSE] = self._handle_response
        
    async def initialize(self) -> bool:
        """Initialize IPC client and establish connection"""
        self.logger.info("Initializing IPC client")
        
        try:
            # Create shared memory regions
            success = self.shared_memory.create_region("document_data")
            if not success:
                self.logger.error("Failed to create shared memory region")
                return False
                
            # Connect to Rust process
            success = await self._connect()
            if not success:
                self.logger.error("Failed to establish IPC connection")
                return False
                
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info("IPC client initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize IPC client: {e}")
            return False
            
    async def _connect(self) -> bool:
        """Establish connection to Rust process"""
        attempts = 0
        
        while attempts < self.max_reconnect_attempts:
            try:
                with self.connection_lock:
                    # Check if pipes exist
                    if not (self.pipe_read_path.exists() and self.pipe_write_path.exists()):
                        self.logger.debug(f"Waiting for pipes to be created (attempt {attempts + 1})")
                        await asyncio.sleep(self.reconnect_delay)
                        attempts += 1
                        continue
                        
                    # Open pipes
                    self.read_pipe = open(self.pipe_read_path, 'rb')
                    self.write_pipe = open(self.pipe_write_path, 'wb')
                    
                    self.connected = True
                    
                    if attempts > 0:
                        self.metrics['reconnections'] += 1
                        
                    self.logger.info("IPC connection established")
                    return True
                    
            except Exception as e:
                self.logger.error(f"Connection attempt {attempts + 1} failed: {e}")
                attempts += 1
                await asyncio.sleep(self.reconnect_delay)
                
        self.logger.error("Failed to establish connection after all attempts")
        return False
        
    async def _start_background_tasks(self):
        """Start background tasks for message handling"""
        self.running = True
        
        # Message reader task
        self.reader_task = asyncio.create_task(self._message_reader_loop())
        
        # Heartbeat task
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Cleanup task for expired responses
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def _message_reader_loop(self):
        """Background task to read incoming messages"""
        while self.running:
            try:
                if not self.connected:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Read message length header
                length_bytes = await asyncio.to_thread(self.read_pipe.read, 4)
                if len(length_bytes) != 4:
                    self.logger.warning("Incomplete length header received")
                    continue
                    
                message_length = struct.unpack('<I', length_bytes)[0]
                
                # Read message data
                message_data = await asyncio.to_thread(self.read_pipe.read, message_length)
                if len(message_data) != message_length:
                    self.logger.warning("Incomplete message received")
                    continue
                    
                # Deserialize message
                try:
                    message = IPCMessage.from_bytes(message_data)
                    await self._handle_message(message)
                    self.metrics['messages_received'] += 1
                    self.metrics['bytes_transferred'] += len(message_data)
                except Exception as e:
                    self.logger.error(f"Failed to process message: {e}")
                    self.metrics['errors'] += 1
                    
            except Exception as e:
                self.logger.error(f"Message reader error: {e}")
                if self.connected:
                    self.connected = False
                    asyncio.create_task(self._reconnect())
                await asyncio.sleep(1.0)
                
    async def _handle_message(self, message: IPCMessage):
        """Handle incoming message"""
        self.logger.debug(f"Received message: {message.message_type.value} (ID: {message.message_id})")
        
        # Check if this is a response to a pending request
        if message.message_id in self.pending_responses:
            future = self.pending_responses[message.message_id]
            if not future.done():
                future.set_result(message)
            del self.pending_responses[message.message_id]
            return
            
        # Handle message with registered handler
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                self.logger.error(f"Message handler error: {e}")
        else:
            self.logger.warning(f"No handler for message type: {message.message_type.value}")
            
    async def _handle_heartbeat(self, message: IPCMessage):
        """Handle heartbeat message"""
        # Respond to heartbeat
        response = IPCMessage(
            message_type=MessageType.RESPONSE,
            message_id=message.message_id,
            payload={'status': 'alive'},
            response_expected=False
        )
        await self._send_message(response)
        
    async def _handle_error(self, message: IPCMessage):
        """Handle error message"""
        self.logger.error(f"Received error from Rust: {message.payload}")
        
    async def _handle_response(self, message: IPCMessage):
        """Handle generic response message"""
        # This is handled in _handle_message
        pass
        
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            try:
                if self.connected:
                    heartbeat = IPCMessage(
                        message_type=MessageType.HEARTBEAT,
                        message_id=self._generate_message_id(),
                        payload={'timestamp': time.time()},
                        response_expected=False
                    )
                    await self._send_message(heartbeat)
                    
                await asyncio.sleep(10.0)  # Heartbeat every 10 seconds
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5.0)
                
    async def _cleanup_loop(self):
        """Clean up expired pending responses"""
        while self.running:
            try:
                current_time = time.time()
                expired_ids = []
                
                for msg_id, future in self.pending_responses.items():
                    if current_time - future.created_time > self.response_timeout:
                        expired_ids.append(msg_id)
                        
                for msg_id in expired_ids:
                    future = self.pending_responses[msg_id]
                    if not future.done():
                        future.set_exception(TimeoutError(f"Response timeout for message {msg_id}"))
                    del self.pending_responses[msg_id]
                    
                if expired_ids:
                    self.logger.warning(f"Cleaned up {len(expired_ids)} expired responses")
                    
                await asyncio.sleep(5.0)  # Cleanup every 5 seconds
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(10.0)
                
    async def _reconnect(self):
        """Attempt to reconnect"""
        self.logger.info("Attempting to reconnect...")
        
        # Clean up current connection
        with self.connection_lock:
            if hasattr(self, 'read_pipe'):
                self.read_pipe.close()
            if hasattr(self, 'write_pipe'):
                self.write_pipe.close()
                
        # Attempt reconnection
        success = await self._connect()
        if success:
            self.logger.info("Reconnection successful")
        else:
            self.logger.error("Reconnection failed")
            
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        self.message_counter += 1
        return f"msg_{int(time.time())}_{self.message_counter}"
        
    async def _send_message(self, message: IPCMessage) -> bool:
        """Send message to Rust process"""
        if not self.connected:
            self.logger.error("Cannot send message - not connected")
            return False
            
        try:
            message_bytes = message.to_bytes()
            
            # Send message
            await asyncio.to_thread(self.write_pipe.write, message_bytes)
            await asyncio.to_thread(self.write_pipe.flush)
            
            self.metrics['messages_sent'] += 1
            self.metrics['bytes_transferred'] += len(message_bytes)
            
            self.logger.debug(f"Sent message: {message.message_type.value} (ID: {message.message_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            self.metrics['errors'] += 1
            self.connected = False
            asyncio.create_task(self._reconnect())
            return False
            
    async def send_document_processing_request(
        self,
        request: DocumentProcessingRequest
    ) -> ProcessingResponse:
        """
        Send document processing request to Rust core.
        
        Args:
            request: Document processing request
            
        Returns:
            Processing response from Rust
        """
        # Prepare large data for shared memory if needed
        shared_memory_offset = None
        if len(request.content) > 10000:  # Use shared memory for large content
            data_bytes = request.content.encode('utf-8')
            shared_memory_offset = self.shared_memory.write_data("document_data", data_bytes)
            
            # Replace content with reference
            payload_content = ""
        else:
            payload_content = request.content
            
        # Create message
        message = IPCMessage(
            message_type=MessageType.PROCESS_DOCUMENT,
            message_id=self._generate_message_id(),
            payload={
                'document_id': request.document_id,
                'content': payload_content,
                'metadata': request.metadata,
                'processing_hints': request.processing_hints,
                'qa_target_count': request.qa_target_count,
                'quality_threshold': request.quality_threshold,
                'shared_memory_offset': shared_memory_offset
            },
            priority=Priority.HIGH
        )
        
        # Send and wait for response
        response_message = await self._send_and_wait(message)
        
        if response_message:
            # Parse response
            payload = response_message.payload
            return ProcessingResponse(
                document_id=payload.get('document_id', request.document_id),
                success=payload.get('success', False),
                qa_pairs=payload.get('qa_pairs', []),
                quality_metrics=payload.get('quality_metrics', {}),
                processing_time=payload.get('processing_time', 0.0),
                error_message=payload.get('error_message', ''),
                model_used=payload.get('model_used', '')
            )
        else:
            return ProcessingResponse(
                document_id=request.document_id,
                success=False,
                error_message="No response received"
            )
            
    async def _send_and_wait(self, message: IPCMessage) -> Optional[IPCMessage]:
        """Send message and wait for response"""
        if not message.response_expected:
            # Just send the message
            success = await self._send_message(message)
            return None if not success else message
            
        # Create future for response
        response_future = asyncio.Future()
        response_future.created_time = time.time()
        self.pending_responses[message.message_id] = response_future
        
        # Send message
        success = await self._send_message(message)
        if not success:
            del self.pending_responses[message.message_id]
            return None
            
        try:
            # Wait for response
            start_time = time.time()
            response = await asyncio.wait_for(response_future, timeout=message.timeout)
            response_time = time.time() - start_time
            
            # Update metrics
            alpha = 0.1
            self.metrics['avg_response_time'] = (
                alpha * response_time + 
                (1 - alpha) * self.metrics['avg_response_time']
            )
            
            return response
            
        except asyncio.TimeoutError:
            self.logger.error(f"Response timeout for message {message.message_id}")
            if message.message_id in self.pending_responses:
                del self.pending_responses[message.message_id]
            return None
            
    async def get_model_status(self) -> Dict[str, Any]:
        """Get model status from Rust core"""
        message = IPCMessage(
            message_type=MessageType.MODEL_STATUS,
            message_id=self._generate_message_id(),
            payload={},
            priority=Priority.NORMAL
        )
        
        response = await self._send_and_wait(message)
        return response.payload if response else {}
        
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from Rust core"""
        message = IPCMessage(
            message_type=MessageType.PERFORMANCE_METRICS,
            message_id=self._generate_message_id(),
            payload={},
            priority=Priority.LOW
        )
        
        response = await self._send_and_wait(message)
        if response:
            return response.payload
        else:
            return self.get_local_metrics()
            
    def get_local_metrics(self) -> Dict[str, Any]:
        """Get local IPC metrics"""
        return {
            'ipc_metrics': dict(self.metrics),
            'connected': self.connected,
            'pending_responses': len(self.pending_responses),
            'shared_memory_regions': len(self.shared_memory.regions),
            'uptime': time.time() - getattr(self, 'start_time', time.time())
        }
        
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register custom message handler"""
        self.message_handlers[message_type] = handler
        self.logger.info(f"Registered handler for {message_type.value}")
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health = {
            'healthy': True,
            'issues': [],
            'warnings': []
        }
        
        # Check connection
        if not self.connected:
            health['healthy'] = False
            health['issues'].append('Not connected to Rust core')
            
        # Check pending responses
        if len(self.pending_responses) > 10:
            health['warnings'].append('High number of pending responses')
            
        # Check error rate
        if self.metrics['errors'] > 0:
            error_rate = self.metrics['errors'] / max(self.metrics['messages_sent'], 1)
            if error_rate > 0.1:
                health['warnings'].append(f'High error rate: {error_rate:.1%}')
                
        # Check shared memory
        memory_stats = {}
        for region_name in self.shared_memory.regions:
            try:
                # Test write/read
                test_data = b'health_check'
                offset = self.shared_memory.write_data(region_name, test_data)
                read_data = self.shared_memory.read_data(region_name, offset or 0)
                
                if read_data != test_data:
                    health['warnings'].append(f'Shared memory region {region_name} integrity issue')
            except Exception as e:
                health['issues'].append(f'Shared memory region {region_name} error: {str(e)}')
                
        health.update({
            'metrics': self.get_local_metrics(),
            'timestamp': time.time()
        })
        
        return health
        
    async def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up IPC client")
        
        self.running = False
        
        # Cancel background tasks
        for task in [self.reader_task, self.heartbeat_task, self.cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        # Close connections
        with self.connection_lock:
            if hasattr(self, 'read_pipe'):
                self.read_pipe.close()
            if hasattr(self, 'write_pipe'):
                self.write_pipe.close()
                
        # Clean up shared memory
        for region_name in list(self.shared_memory.regions.keys()):
            self.shared_memory.cleanup_region(region_name)
            
        self.logger.info("IPC client cleanup complete")


# Example usage and testing
if __name__ == "__main__":
    async def test_ipc_client():
        """Test the IPC client"""
        logging.basicConfig(level=logging.INFO)
        
        client = IPCClient()
        
        # Initialize client
        success = await client.initialize()
        print(f"Client initialized: {success}")
        
        if success:
            # Test document processing request
            request = DocumentProcessingRequest(
                document_id="test_001",
                content="Test document content about LTE handover procedures.",
                metadata={'feature': 'LTE Handover'},
                processing_hints={'complexity': 0.5},
                qa_target_count=3
            )
            
            try:
                response = await client.send_document_processing_request(request)
                print(f"Processing response: {response.success}")
                print(f"QA pairs generated: {len(response.qa_pairs)}")
            except Exception as e:
                print(f"Processing failed: {e}")
                
            # Health check
            health = await client.health_check()
            print(f"Health check: {health['healthy']}")
            
            # Get metrics
            metrics = client.get_local_metrics()
            print(f"Messages sent: {metrics['ipc_metrics']['messages_sent']}")
            
        await client.cleanup()
        
    asyncio.run(test_ipc_client())
