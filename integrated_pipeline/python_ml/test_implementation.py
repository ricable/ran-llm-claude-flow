#!/usr/bin/env python3
"""
Quick implementation test for Python ML Engine

Verifies that all components can be imported and initialized properly.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test imports
try:
    from model_manager import Qwen3ModelManager, ModelSize, ProcessingHints
    from semantic_processor import SemanticProcessor, QAPair, QuestionType
    from mlx_accelerator import MLXAccelerator, MemoryPool
    from ipc_client import IPCClient, MessageType
    from main import PythonMLEngine
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

async def test_basic_functionality():
    """Test basic functionality without external dependencies"""
    print("\n🧪 Testing basic functionality...")
    
    # Test model manager initialization
    try:
        model_manager = Qwen3ModelManager()
        print("✅ ModelManager created")
        
        # Test model selection logic
        hints = ProcessingHints(
            complexity=0.5,
            document_length=1000,
            technical_density=0.7,
            parameter_count=5,
            quality_requirement=0.8
        )
        
        selected_model = model_manager.select_optimal_model(hints)
        print(f"✅ Model selection working: {selected_model.value}")
        
    except Exception as e:
        print(f"❌ ModelManager test failed: {e}")
        
    # Test semantic processor
    try:
        semantic_processor = SemanticProcessor()
        print("✅ SemanticProcessor created")
        
        # Test document analysis
        test_content = """
        # LTE Handover
        
        LTE handover enables UE mobility between eNodeBs.
        
        ## Parameters
        - EUtranCellFDD.a3Offset: Offset for A3 events
        - EUtranCellFDD.hysteresisA3: Hysteresis value
        """
        
        metadata = {'document_id': 'test', 'feature_name': 'LTE Handover'}
        analysis = semantic_processor.analyze_document(test_content, metadata)
        
        print(f"✅ Document analysis working: complexity={analysis.complexity:.2f}")
        print(f"   Technical terms: {len(analysis.technical_terms)}")
        print(f"   Parameters: {len(analysis.parameters)}")
        
    except Exception as e:
        print(f"❌ SemanticProcessor test failed: {e}")
        
    # Test MLX accelerator (without actual MLX)
    try:
        accelerator = MLXAccelerator(memory_budget_gb=8.0)
        memory_stats = accelerator.memory_pool.get_stats()
        print("✅ MLXAccelerator created")
        print(f"   Memory budget: {memory_stats['total_budget_gb']}GB")
        
    except Exception as e:
        print(f"❌ MLXAccelerator test failed: {e}")
        
    # Test IPC client (without actual pipes)
    try:
        ipc_client = IPCClient("/tmp/test_ipc")
        metrics = ipc_client.get_local_metrics()
        print("✅ IPCClient created")
        print(f"   Connected: {metrics['connected']}")
        
    except Exception as e:
        print(f"❌ IPCClient test failed: {e}")
        
    # Test main engine
    try:
        engine = PythonMLEngine()
        config_loaded = engine.load_config()
        print(f"✅ PythonMLEngine created, config loaded: {config_loaded}")
        
    except Exception as e:
        print(f"❌ PythonMLEngine test failed: {e}")
        
    print("\n🎉 Basic functionality tests completed!")

def test_system_info():
    """Test system information detection"""
    print("\n💻 System Information:")
    
    try:
        import platform
        import psutil
        
        print(f"   Platform: {platform.system()} {platform.machine()}")
        print(f"   Python: {platform.python_version()}")
        print(f"   CPU cores: {psutil.cpu_count()}")
        print(f"   Memory: {psutil.virtual_memory().total // (1024**3)}GB")
        
        # Check for MLX
        try:
            import mlx.core as mx
            print("   MLX: ✅ Available")
        except ImportError:
            print("   MLX: ❌ Not available")
            
        # Check for other dependencies
        deps = ['torch', 'transformers', 'sentence_transformers', 'sklearn', 'numpy']
        for dep in deps:
            try:
                __import__(dep)
                print(f"   {dep}: ✅")
            except ImportError:
                print(f"   {dep}: ❌")
                
    except Exception as e:
        print(f"❌ System info test failed: {e}")

def main():
    """Main test function"""
    print("🚀 Python ML Engine Implementation Test")
    print("=" * 50)
    
    # Setup basic logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    # Test system info
    test_system_info()
    
    # Test basic functionality
    asyncio.run(test_basic_functionality())
    
    print("\n" + "=" * 50)
    print("🏁 Implementation test completed!")
    print("\n💡 Next steps:")
    print("   1. Install dependencies: pip install -e .")
    print("   2. Test with MLX: python -m src.main --test")
    print("   3. Run benchmark: python -m src.main --benchmark")
    print("   4. Start interactive: python -m src.main --interactive")

if __name__ == "__main__":
    main()
