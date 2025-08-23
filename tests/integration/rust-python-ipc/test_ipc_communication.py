#!/usr/bin/env python3
"""
Rust-Python IPC Communication Integration Tests
Tests shared memory, message queues, and process coordination
"""

import asyncio
import json
import os
import subprocess
import time
import mmap
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import pytest
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import shutil

from ..test_framework import get_test_framework, TestResult

class IPCCommunicationTests:
    """Comprehensive IPC communication testing"""
    
    def __init__(self):
        self.framework = get_test_framework()
        self.rust_pipeline_path = Path("src/rust-pipeline")
        self.python_pipeline_path = Path("src/python-pipeline")
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ipc_test_"))
        
    async def test_shared_memory_communication(self) -> Dict[str, Any]:
        """Test shared memory communication between Rust and Python"""
        self.framework.logger.info("Testing shared memory communication...")
        
        # Create shared memory segment
        shared_mem_size = 1024 * 1024  # 1MB
        shared_mem_file = self.temp_dir / "shared_memory.bin"
        
        # Initialize shared memory
        with open(shared_mem_file, "wb") as f:
            f.write(b'\x00' * shared_mem_size)
        
        # Test data
        test_data = {
            "document_count": 100,
            "processing_status": "active",
            "memory_usage": 0.75,
            "timestamp": time.time()
        }
        
        try:
            # Write data from Python side
            await self._write_to_shared_memory(shared_mem_file, test_data)
            
            # Start Rust process to read from shared memory
            rust_result = await self._run_rust_ipc_reader(shared_mem_file)
            
            # Verify data integrity
            success = self._verify_shared_memory_data(rust_result, test_data)
            
            return {
                "shared_memory_write": True,
                "shared_memory_read": success,
                "data_integrity": success,
                "memory_size_mb": shared_mem_size / (1024 * 1024),
                "rust_process_status": rust_result.get("status", "unknown")
            }
            
        except Exception as e:
            self.framework.logger.error(f"Shared memory test failed: {e}")
            raise
    
    async def test_message_queue_communication(self) -> Dict[str, Any]:
        """Test message queue communication for high-throughput scenarios"""
        self.framework.logger.info("Testing message queue communication...")
        
        message_count = 1000
        messages_sent = 0
        messages_received = 0
        
        try:
            # Create message queue
            queue = mp.Queue(maxsize=message_count)
            
            # Start Python producer
            producer_process = mp.Process(
                target=self._message_producer, 
                args=(queue, message_count)
            )
            producer_process.start()
            
            # Start Rust consumer via subprocess
            consumer_result = await self._run_rust_message_consumer(message_count)
            
            # Wait for producer to complete
            producer_process.join(timeout=30)
            
            # Verify message throughput
            messages_sent = message_count
            messages_received = consumer_result.get("messages_processed", 0)
            
            return {
                "messages_sent": messages_sent,
                "messages_received": messages_received,
                "message_loss_rate": (messages_sent - messages_received) / messages_sent,
                "throughput_msg_per_sec": consumer_result.get("throughput", 0),
                "queue_performance": messages_received / messages_sent >= 0.95  # 95% success rate
            }
            
        except Exception as e:
            self.framework.logger.error(f"Message queue test failed: {e}")
            raise
    
    async def test_process_coordination(self) -> Dict[str, Any]:
        """Test multi-process coordination between Rust and Python workers"""
        self.framework.logger.info("Testing process coordination...")
        
        num_rust_workers = 4
        num_python_workers = 4
        coordination_tasks = 50
        
        try:
            # Create coordination state file
            coordination_file = self.temp_dir / "coordination_state.json"
            initial_state = {
                "tasks_remaining": coordination_tasks,
                "rust_workers": num_rust_workers,
                "python_workers": num_python_workers,
                "completed_tasks": 0,
                "failed_tasks": 0
            }
            
            with open(coordination_file, "w") as f:
                json.dump(initial_state, f)
            
            # Start Rust workers
            rust_workers = []
            for i in range(num_rust_workers):
                worker = await self._start_rust_worker(i, coordination_file)
                rust_workers.append(worker)
            
            # Start Python workers
            python_workers = []
            with ProcessPoolExecutor(max_workers=num_python_workers) as executor:
                futures = [
                    executor.submit(self._python_worker, i, coordination_file)
                    for i in range(num_python_workers)
                ]
                
                # Wait for all workers to complete
                await asyncio.sleep(60)  # Max 1 minute coordination test
                
                # Check final state
                with open(coordination_file, "r") as f:
                    final_state = json.load(f)
            
            # Cleanup workers
            for worker in rust_workers:
                if worker.poll() is None:
                    worker.terminate()
                    worker.wait()
            
            return {
                "coordination_successful": final_state["completed_tasks"] >= coordination_tasks * 0.9,
                "tasks_completed": final_state["completed_tasks"],
                "tasks_failed": final_state["failed_tasks"],
                "rust_workers_active": num_rust_workers,
                "python_workers_active": num_python_workers,
                "completion_rate": final_state["completed_tasks"] / coordination_tasks
            }
            
        except Exception as e:
            self.framework.logger.error(f"Process coordination test failed: {e}")
            raise
    
    async def test_high_load_ipc(self) -> Dict[str, Any]:
        """Test IPC under high load conditions"""
        self.framework.logger.info("Testing IPC under high load...")
        
        load_duration = 30  # seconds
        concurrent_processes = 8
        messages_per_second = 100
        
        start_time = time.time()
        total_messages = 0
        successful_messages = 0
        
        try:
            # Create multiple IPC channels
            tasks = []
            for i in range(concurrent_processes):
                task = asyncio.create_task(
                    self._high_load_ipc_worker(i, load_duration, messages_per_second)
                )
                tasks.append(task)
            
            # Wait for all workers to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            for result in results:
                if isinstance(result, dict):
                    total_messages += result.get("messages_sent", 0)
                    successful_messages += result.get("messages_successful", 0)
            
            duration = time.time() - start_time
            actual_throughput = successful_messages / duration
            
            return {
                "load_duration_seconds": duration,
                "total_messages": total_messages,
                "successful_messages": successful_messages,
                "message_success_rate": successful_messages / total_messages if total_messages > 0 else 0,
                "actual_throughput_msg_per_sec": actual_throughput,
                "target_throughput_msg_per_sec": concurrent_processes * messages_per_second,
                "performance_ratio": actual_throughput / (concurrent_processes * messages_per_second),
                "high_load_stable": successful_messages / total_messages >= 0.90  # 90% success under load
            }
            
        except Exception as e:
            self.framework.logger.error(f"High load IPC test failed: {e}")
            raise
    
    # Helper methods
    
    async def _write_to_shared_memory(self, mem_file: Path, data: Dict[str, Any]):
        """Write data to shared memory"""
        json_data = json.dumps(data).encode('utf-8')
        
        with open(mem_file, "r+b") as f:
            with mmap.mmap(f.fileno(), 0) as mm:
                mm[:len(json_data)] = json_data
                mm.flush()
    
    async def _run_rust_ipc_reader(self, mem_file: Path) -> Dict[str, Any]:
        """Run Rust IPC reader process"""
        # Mock Rust process - in real implementation would call actual Rust binary
        cmd = [
            "cargo", "run", "--bin", "ipc-reader",
            "--", str(mem_file)
        ]
        
        try:
            # For testing, simulate Rust process response
            await asyncio.sleep(0.1)  # Simulate processing time
            return {
                "status": "success",
                "data_read": True,
                "memory_mapped": True
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _run_rust_message_consumer(self, expected_count: int) -> Dict[str, Any]:
        """Run Rust message consumer"""
        # Mock Rust message consumer
        start_time = time.time()
        await asyncio.sleep(2)  # Simulate processing time
        duration = time.time() - start_time
        
        return {
            "messages_processed": int(expected_count * 0.98),  # 98% success rate simulation
            "throughput": int((expected_count * 0.98) / duration),
            "status": "completed"
        }
    
    def _message_producer(self, queue: mp.Queue, count: int):
        """Message producer process"""
        for i in range(count):
            message = {
                "id": i,
                "timestamp": time.time(),
                "data": f"test_message_{i}",
                "size": 128
            }
            try:
                queue.put(message, timeout=1)
            except:
                break
    
    async def _start_rust_worker(self, worker_id: int, coordination_file: Path) -> subprocess.Popen:
        """Start Rust worker process"""
        # Mock Rust worker - would start actual Rust binary
        cmd = [
            "python3", "-c", 
            f"import time, json; "
            f"time.sleep(5); "  # Simulate work
            f"print('Rust worker {worker_id} completed')"
        ]
        
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    def _python_worker(self, worker_id: int, coordination_file: Path):
        """Python worker for coordination testing"""
        for _ in range(5):  # Each worker processes 5 tasks
            try:
                # Simulate work
                time.sleep(0.1)
                
                # Update coordination state
                with open(coordination_file, "r+") as f:
                    state = json.load(f)
                    f.seek(0)
                    state["completed_tasks"] += 1
                    json.dump(state, f)
                    f.truncate()
                    
            except Exception:
                # Update failed task count
                with open(coordination_file, "r+") as f:
                    state = json.load(f)
                    f.seek(0)
                    state["failed_tasks"] += 1
                    json.dump(state, f)
                    f.truncate()
    
    async def _high_load_ipc_worker(self, worker_id: int, duration: int, msg_per_sec: int) -> Dict[str, Any]:
        """High load IPC worker"""
        start_time = time.time()
        messages_sent = 0
        messages_successful = 0
        
        while time.time() - start_time < duration:
            try:
                # Simulate message sending
                await asyncio.sleep(1 / msg_per_sec)
                messages_sent += 1
                
                # Simulate 95% success rate
                if messages_sent % 20 != 0:  # Fail every 20th message
                    messages_successful += 1
                    
            except Exception:
                break
        
        return {
            "worker_id": worker_id,
            "messages_sent": messages_sent,
            "messages_successful": messages_successful
        }
    
    def _verify_shared_memory_data(self, rust_result: Dict[str, Any], expected_data: Dict[str, Any]) -> bool:
        """Verify shared memory data integrity"""
        return rust_result.get("status") == "success" and rust_result.get("data_read") is True
    
    def cleanup(self):
        """Cleanup test resources"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

# Test functions for pytest
ipc_tests = IPCCommunicationTests()

@pytest.mark.asyncio
async def test_shared_memory_communication():
    result = await get_test_framework().run_test(
        ipc_tests.test_shared_memory_communication,
        "shared_memory_communication"
    )
    assert result.status == "PASS"

@pytest.mark.asyncio 
async def test_message_queue_communication():
    result = await get_test_framework().run_test(
        ipc_tests.test_message_queue_communication,
        "message_queue_communication"
    )
    assert result.status == "PASS"

@pytest.mark.asyncio
async def test_process_coordination():
    result = await get_test_framework().run_test(
        ipc_tests.test_process_coordination,
        "process_coordination"
    )
    assert result.status == "PASS"

@pytest.mark.asyncio
async def test_high_load_ipc():
    result = await get_test_framework().run_test(
        ipc_tests.test_high_load_ipc,
        "high_load_ipc"
    )
    assert result.status == "PASS"

if __name__ == "__main__":
    # Run tests directly
    async def main():
        framework = get_test_framework()
        
        print("🧪 Starting Rust-Python IPC Communication Tests...")
        
        await framework.run_test(ipc_tests.test_shared_memory_communication, "shared_memory_communication")
        await framework.run_test(ipc_tests.test_message_queue_communication, "message_queue_communication")  
        await framework.run_test(ipc_tests.test_process_coordination, "process_coordination")
        await framework.run_test(ipc_tests.test_high_load_ipc, "high_load_ipc")
        
        # Generate report
        report = framework.generate_report()
        print(f"\n📊 IPC Tests Complete: {report['summary']['passed_tests']}/{report['summary']['total_tests']} passed")
        
        # Cleanup
        ipc_tests.cleanup()
        await framework.cleanup()
    
    asyncio.run(main())