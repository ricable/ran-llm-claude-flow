#!/usr/bin/env python3
"""
Rust-Python IPC Bridge - M3 Max Optimized
High-performance inter-process communication between Rust core and Python ML engine
"""

import asyncio
import logging
import json
import struct
import mmap
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import signal
import os
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """IPC message types"""
    DOCUMENT_PROCESS = "document_process"
    ML_INFERENCE = "ml_inference"
    EMBEDDING_REQUEST = "embedding_request"
    QUALITY_ASSESSMENT = "quality_assessment"
    BATCH_PROCESS = "batch_process"
    HEALTH_CHECK = "health_check"
    SHUTDOWN = "shutdown"
    ERROR = "error"
    RESPONSE = "response"

class Priority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2  
    HIGH = 3
    URGENT = 4

@dataclass
class IPCMessage:
    """IPC message structure"""
    message_id: str
    message_type: MessageType
    priority: Priority
    payload: Dict[str, Any]
    timestamp: float
    sender: str
    response_expected: bool = True
    timeout_seconds: float = 30.0
    
    def to_bytes(self) -> bytes:
        """Serialize message to bytes"""
        data = asdict(self)
        # Convert enums to strings
        data['message_type'] = self.message_type.value
        data['priority'] = self.priority.value
        
        json_str = json.dumps(data)
        json_bytes = json_str.encode('utf-8')
        
        # Prepend length as 4-byte unsigned int
        length = len(json_bytes)
        return struct.pack('!I', length) + json_bytes
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'IPCMessage':
        """Deserialize message from bytes"""
        # Extract length
        if len(data) < 4:
            raise ValueError("Invalid message data")
        
        length = struct.unpack('!I', data[:4])[0]
        if len(data) < 4 + length:
            raise ValueError("Incomplete message data")
        
        json_bytes = data[4:4+length]
        json_str = json_bytes.decode('utf-8')
        data_dict = json.loads(json_str)
        
        # Convert string enums back
        data_dict['message_type'] = MessageType(data_dict['message_type'])
        data_dict['priority'] = Priority(data_dict['priority'])
        
        return cls(**data_dict)

@dataclass
class SharedMemoryRegion:
    """Shared memory region for zero-copy data transfer"""
    name: str
    size: int
    offset: int = 0
    
class MemoryPool:
    """Managed memory pool for IPC operations"""
    
    def __init__(self, total_size_gb: int = 15):
        self.total_size = total_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.regions = {}
        self.free_regions = []
        self.allocated_size = 0
        self.lock = threading.RLock()
        
        logger.info(f"🧠 Memory pool initialized: {total_size_gb}GB")
    
    def allocate(self, size: int, identifier: str) -> Optional[SharedMemoryRegion]:
        """Allocate memory region"""
        with self.lock:
            if self.allocated_size + size > self.total_size:
                logger.warning(f"⚠️ Memory pool exhausted: need {size}, available {self.total_size - self.allocated_size}")
                return None
            
            # Create temporary file for shared memory
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(b'\0' * size)
            temp_file.flush()
            
            region = SharedMemoryRegion(
                name=temp_file.name,
                size=size,
                offset=0
            )
            
            self.regions[identifier] = region
            self.allocated_size += size
            
            logger.debug(f"🧠 Allocated {size} bytes for {identifier}")
            return region
    
    def deallocate(self, identifier: str) -> bool:
        """Deallocate memory region"""
        with self.lock:
            if identifier not in self.regions:
                return False
            
            region = self.regions.pop(identifier)
            self.allocated_size -= region.size
            
            # Clean up temporary file
            try:
                os.unlink(region.name)
            except OSError:
                pass
            
            logger.debug(f"🧠 Deallocated memory for {identifier}")
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        with self.lock:
            return {
                'total_size_gb': self.total_size / (1024**3),
                'allocated_size_gb': self.allocated_size / (1024**3),
                'free_size_gb': (self.total_size - self.allocated_size) / (1024**3),
                'utilization_percent': (self.allocated_size / self.total_size) * 100,
                'active_regions': len(self.regions)
            }

class NamedPipeManager:
    """Named pipe manager for message passing"""
    
    def __init__(self, pipe_dir: str = "/tmp/ran_llm_pipes"):
        self.pipe_dir = Path(pipe_dir)
        self.pipe_dir.mkdir(exist_ok=True)
        self.pipes = {}
        self.lock = threading.Lock()
        
    def create_pipe(self, pipe_name: str) -> Path:
        """Create a named pipe"""
        pipe_path = self.pipe_dir / pipe_name
        
        with self.lock:
            if pipe_name in self.pipes:
                return self.pipes[pipe_name]
            
            # Remove existing pipe if it exists
            if pipe_path.exists():
                pipe_path.unlink()
            
            # Create named pipe
            os.mkfifo(str(pipe_path))
            self.pipes[pipe_name] = pipe_path
            
            logger.info(f"📡 Created named pipe: {pipe_path}")
            return pipe_path
    
    def cleanup_pipes(self):
        """Cleanup all created pipes"""
        with self.lock:
            for pipe_name, pipe_path in self.pipes.items():
                try:
                    if pipe_path.exists():
                        pipe_path.unlink()
                    logger.debug(f"🗑️ Cleaned up pipe: {pipe_path}")
                except OSError as e:
                    logger.warning(f"⚠️ Failed to cleanup pipe {pipe_path}: {e}")
            
            self.pipes.clear()

class PythonMLServer:
    """Python ML server for handling Rust requests"""
    
    def __init__(self, server_id: str = "python_ml_server"):
        self.server_id = server_id
        self.memory_pool = MemoryPool()
        self.pipe_manager = NamedPipeManager()
        self.message_handlers = {}
        self.request_queue = queue.PriorityQueue()
        self.response_futures = {}
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Performance metrics
        self.stats = {
            'messages_processed': 0,
            'total_processing_time': 0.0,
            'average_latency': 0.0,
            'error_count': 0,
            'start_time': time.time()
        }
        
        # Create server pipes
        self.request_pipe = self.pipe_manager.create_pipe(f"{server_id}_requests")
        self.response_pipe = self.pipe_manager.create_pipe(f"{server_id}_responses")
        
        logger.info(f"🚀 Python ML Server initialized: {server_id}")
    
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register message handler"""
        self.message_handlers[message_type] = handler
        logger.info(f"🔧 Registered handler for {message_type.value}")
    
    async def start(self):
        """Start the server"""
        logger.info("🚀 Starting Python ML Server...")
        self.running = True
        
        # Start message processing tasks
        tasks = [
            asyncio.create_task(self._message_receiver()),
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._response_sender())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"❌ Server error: {e}")
        finally:
            await self.shutdown()
    
    async def _message_receiver(self):
        """Receive messages from Rust via named pipe"""
        logger.info("👂 Message receiver started")
        
        while self.running:
            try:
                # Open pipe for reading (blocking)
                with open(self.request_pipe, 'rb') as pipe:
                    while self.running:
                        # Read message length first
                        length_data = pipe.read(4)
                        if not length_data or len(length_data) != 4:
                            continue
                        
                        message_length = struct.unpack('!I', length_data)[0]
                        
                        # Read full message
                        message_data = pipe.read(message_length)
                        if len(message_data) != message_length:
                            logger.warning("⚠️ Incomplete message received")
                            continue
                        
                        # Parse message
                        full_data = length_data + message_data
                        message = IPCMessage.from_bytes(full_data)
                        
                        # Queue message by priority
                        priority_value = message.priority.value
                        self.request_queue.put((priority_value, message))
                        
                        logger.debug(f"📨 Received {message.message_type.value} message: {message.message_id}")
                        
            except Exception as e:
                if self.running:
                    logger.error(f"❌ Message receiver error: {e}")
                    await asyncio.sleep(1)  # Brief pause before retry
    
    async def _message_processor(self):
        """Process messages from the queue"""
        logger.info("⚙️ Message processor started")
        
        while self.running:
            try:
                # Get next message (with timeout)
                try:
                    priority, message = self.request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                start_time = time.time()
                
                # Process message
                response = await self._handle_message(message)
                
                # Update statistics
                processing_time = time.time() - start_time
                self.stats['messages_processed'] += 1
                self.stats['total_processing_time'] += processing_time
                self.stats['average_latency'] = (
                    self.stats['total_processing_time'] / self.stats['messages_processed']
                )
                
                # Send response if expected
                if message.response_expected and response:
                    response_message = IPCMessage(
                        message_id=f"{message.message_id}_response",
                        message_type=MessageType.RESPONSE,
                        priority=message.priority,
                        payload=response,
                        timestamp=time.time(),
                        sender=self.server_id,
                        response_expected=False
                    )
                    
                    # Store response for sender
                    self.response_futures[message.message_id] = response_message
                
                self.request_queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ Message processing error: {e}")
                self.stats['error_count'] += 1
    
    async def _response_sender(self):
        """Send responses back to Rust"""
        logger.info("📤 Response sender started")
        
        while self.running:
            try:
                # Check for responses to send
                responses_to_send = []
                
                for message_id, response in list(self.response_futures.items()):
                    responses_to_send.append((message_id, response))
                    del self.response_futures[message_id]
                
                if responses_to_send:
                    # Open pipe for writing
                    with open(self.response_pipe, 'wb') as pipe:
                        for message_id, response in responses_to_send:
                            try:
                                data = response.to_bytes()
                                pipe.write(data)
                                pipe.flush()
                                
                                logger.debug(f"📤 Sent response for {message_id}")
                                
                            except Exception as e:
                                logger.error(f"❌ Failed to send response for {message_id}: {e}")
                
                await asyncio.sleep(0.1)  # Brief pause
                
            except Exception as e:
                if self.running:
                    logger.error(f"❌ Response sender error: {e}")
                    await asyncio.sleep(1)
    
    async def _handle_message(self, message: IPCMessage) -> Dict[str, Any]:
        """Handle incoming message"""
        message_type = message.message_type
        
        if message_type in self.message_handlers:
            try:
                handler = self.message_handlers[message_type]
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    handler,
                    message
                )
                
                return {
                    'status': 'success',
                    'result': result,
                    'processing_time': time.time() - message.timestamp
                }
                
            except Exception as e:
                logger.error(f"❌ Handler error for {message_type.value}: {e}")
                return {
                    'status': 'error', 
                    'error': str(e),
                    'message_type': message_type.value
                }
        else:
            logger.warning(f"⚠️ No handler for message type: {message_type.value}")
            return {
                'status': 'error',
                'error': f'No handler for {message_type.value}'
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get server system status"""
        memory_stats = self.memory_pool.get_stats()
        process = psutil.Process()
        
        return {
            'server_id': self.server_id,
            'running': self.running,
            'uptime_seconds': time.time() - self.stats['start_time'],
            'performance_stats': self.stats.copy(),
            'memory_pool': memory_stats,
            'system_resources': {
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / (1024 * 1024),
                'threads': process.num_threads()
            },
            'queue_size': self.request_queue.qsize(),
            'pending_responses': len(self.response_futures)
        }
    
    async def shutdown(self):
        """Shutdown server gracefully"""
        logger.info("🛑 Shutting down Python ML Server...")
        self.running = False
        
        # Wait for queue to empty
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self.request_queue.join),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("⚠️ Timeout waiting for request queue to empty")
        
        # Cleanup resources
        self.pipe_manager.cleanup_pipes()
        self.executor.shutdown(wait=True)
        
        logger.info("✅ Server shutdown complete")

class RustIPCClient:
    """Client for communicating with Rust process"""
    
    def __init__(self, rust_server_id: str = "rust_core_server"):
        self.rust_server_id = rust_server_id
        self.pipe_manager = NamedPipeManager()
        self.pending_requests = {}
        self.client_id = f"python_client_{int(time.time())}"
        
        # Create client pipes
        self.request_pipe = self.pipe_manager.create_pipe(f"{rust_server_id}_requests")
        self.response_pipe = self.pipe_manager.create_pipe(f"{rust_server_id}_responses") 
        
        logger.info(f"🔗 Rust IPC Client initialized: {self.client_id}")
    
    async def send_message(
        self,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Send message to Rust server and wait for response"""
        
        message_id = f"{self.client_id}_{int(time.time() * 1000000)}"
        
        message = IPCMessage(
            message_id=message_id,
            message_type=message_type,
            priority=priority,
            payload=payload,
            timestamp=time.time(),
            sender=self.client_id,
            response_expected=True,
            timeout_seconds=timeout
        )
        
        try:
            # Send message
            with open(self.request_pipe, 'wb') as pipe:
                data = message.to_bytes()
                pipe.write(data)
                pipe.flush()
            
            logger.debug(f"📤 Sent {message_type.value} message: {message_id}")
            
            # Wait for response
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    with open(self.response_pipe, 'rb') as pipe:
                        # Try to read response
                        pipe_fd = pipe.fileno()
                        
                        # Set non-blocking mode
                        import fcntl
                        flags = fcntl.fcntl(pipe_fd, fcntl.F_GETFL)
                        fcntl.fcntl(pipe_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                        
                        try:
                            length_data = pipe.read(4)
                            if length_data and len(length_data) == 4:
                                message_length = struct.unpack('!I', length_data)[0]
                                message_data = pipe.read(message_length)
                                
                                if len(message_data) == message_length:
                                    full_data = length_data + message_data
                                    response = IPCMessage.from_bytes(full_data)
                                    
                                    if response.message_id == f"{message_id}_response":
                                        logger.debug(f"📨 Received response for {message_id}")
                                        return response.payload
                        
                        except BlockingIOError:
                            # No data available, continue waiting
                            pass
                
                except Exception as e:
                    logger.debug(f"Response read attempt failed: {e}")
                
                await asyncio.sleep(0.1)
            
            # Timeout reached
            logger.error(f"⏱️ Timeout waiting for response to {message_id}")
            return {
                'status': 'error',
                'error': 'Response timeout',
                'timeout_seconds': timeout
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to send message {message_id}: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

# Example message handlers for common operations
def handle_document_process(message: IPCMessage) -> Dict[str, Any]:
    """Handle document processing request"""
    payload = message.payload
    
    # Extract document content from shared memory if provided
    if 'shared_memory_key' in payload:
        # Read from shared memory (implementation depends on specific memory layout)
        content = "Document content from shared memory"
    else:
        content = payload.get('content', '')
    
    # Simulate document processing
    result = {
        'processed_content': f"Processed: {content[:100]}...",
        'word_count': len(content.split()),
        'processing_timestamp': time.time()
    }
    
    logger.info(f"📄 Processed document: {len(content)} chars")
    return result

def handle_ml_inference(message: IPCMessage) -> Dict[str, Any]:
    """Handle ML inference request"""
    payload = message.payload
    
    # Extract inference parameters
    model_id = payload.get('model_id', 'default')
    input_text = payload.get('input_text', '')
    max_tokens = payload.get('max_tokens', 512)
    
    # Simulate ML inference
    result = {
        'generated_text': f"ML generated response for: {input_text[:50]}...",
        'model_used': model_id,
        'tokens_generated': max_tokens // 2,
        'confidence_score': 0.85
    }
    
    logger.info(f"🧠 ML inference: {model_id} -> {len(result['generated_text'])} chars")
    return result

def handle_embedding_request(message: IPCMessage) -> Dict[str, Any]:
    """Handle embedding generation request"""
    payload = message.payload
    
    texts = payload.get('texts', [])
    model_name = payload.get('model_name', 'default')
    
    # Simulate embedding generation
    embeddings = [
        [0.1, 0.2, 0.3, 0.4] * 192  # 768-dimensional embeddings
        for _ in texts
    ]
    
    result = {
        'embeddings': embeddings,
        'model_name': model_name,
        'embedding_dim': 768,
        'num_texts': len(texts)
    }
    
    logger.info(f"🎯 Generated embeddings: {len(texts)} texts -> {len(embeddings)} vectors")
    return result

def handle_health_check(message: IPCMessage) -> Dict[str, Any]:
    """Handle health check request"""
    return {
        'status': 'healthy',
        'timestamp': time.time(),
        'server_id': 'python_ml_server'
    }

# Example usage
async def main():
    """Example usage of Python ML Server"""
    
    # Create and configure server
    server = PythonMLServer("python_ml_server")
    
    # Register message handlers
    server.register_handler(MessageType.DOCUMENT_PROCESS, handle_document_process)
    server.register_handler(MessageType.ML_INFERENCE, handle_ml_inference)
    server.register_handler(MessageType.EMBEDDING_REQUEST, handle_embedding_request)
    server.register_handler(MessageType.HEALTH_CHECK, handle_health_check)
    
    # Handle shutdown gracefully
    def signal_handler(signum, frame):
        logger.info(f"📡 Received signal {signum}, initiating shutdown...")
        asyncio.create_task(server.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start server
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("👋 Server interrupted by user")
    
    logger.info("✅ Server finished")

if __name__ == "__main__":
    asyncio.run(main())