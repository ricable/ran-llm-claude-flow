#!/usr/bin/env python3
"""
Code validation script - validates Python code structure and logic
without requiring external dependencies like aiohttp, mlx, etc.
"""

import sys
import ast
import inspect
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class CodeValidator:
    """Validates Python code structure and design patterns"""
    
    def __init__(self):
        self.results = {
            "files_checked": 0,
            "classes_found": 0,
            "functions_found": 0,
            "async_functions_found": 0,
            "syntax_errors": [],
            "design_issues": [],
            "coverage_analysis": {}
        }
    
    def validate_file_syntax(self, file_path: Path) -> bool:
        """Validate file syntax without importing"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse AST to validate syntax
            tree = ast.parse(source_code, filename=str(file_path))
            self.results["files_checked"] += 1
            
            # Analyze code structure
            self._analyze_ast(tree, file_path)
            
            return True
            
        except SyntaxError as e:
            error_msg = f"{file_path}: Syntax error at line {e.lineno}: {e.msg}"
            self.results["syntax_errors"].append(error_msg)
            print(f"❌ {error_msg}")
            return False
            
        except Exception as e:
            error_msg = f"{file_path}: Parse error: {e}"
            self.results["syntax_errors"].append(error_msg)
            print(f"❌ {error_msg}")
            return False
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path):
        """Analyze AST for code structure"""
        classes = []
        functions = []
        async_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
                self.results["classes_found"] += 1
                
                # Check for proper class structure
                if not node.body:
                    self.results["design_issues"].append(f"{file_path}: Empty class {node.name}")
                
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
                self.results["functions_found"] += 1
                
            elif isinstance(node, ast.AsyncFunctionDef):
                async_functions.append(node.name)
                self.results["async_functions_found"] += 1
        
        # Store analysis for this file
        self.results["coverage_analysis"][str(file_path)] = {
            "classes": classes,
            "functions": functions,
            "async_functions": async_functions,
            "total_definitions": len(classes) + len(functions) + len(async_functions)
        }
    
    def validate_test_coverage(self):
        """Analyze test coverage patterns"""
        test_files = list(Path("tests/python/unit").glob("test_*.py"))
        source_files = list(Path("docs/python/integration").glob("*.py"))
        
        print(f"📊 Found {len(test_files)} test files for {len(source_files)} source files")
        
        # Check if each source file has corresponding test file
        coverage_map = {}
        
        for source_file in source_files:
            source_name = source_file.stem
            expected_test_file = f"test_{source_name}.py"
            
            test_exists = any(test_file.name == expected_test_file for test_file in test_files)
            coverage_map[source_name] = test_exists
            
            if test_exists:
                print(f"✅ {source_name}: Has test file")
            else:
                print(f"⚠️  {source_name}: Missing test file")
        
        coverage_percentage = sum(coverage_map.values()) / len(coverage_map) * 100 if coverage_map else 0
        
        self.results["test_coverage"] = {
            "files_with_tests": sum(coverage_map.values()),
            "total_files": len(coverage_map),
            "coverage_percentage": coverage_percentage,
            "missing_tests": [name for name, has_test in coverage_map.items() if not has_test]
        }
        
        return coverage_percentage

def validate_dataclass_definitions():
    """Validate dataclass definitions in source files"""
    print("🔍 Validating dataclass definitions...")
    
    files_to_check = [
        "docs/python/integration/lmstudio_connector.py",
        "docs/python/integration/local_llm_orchestrator.py", 
        "docs/python/integration/mlx_accelerator.py",
        "docs/python/integration/ollama_optimizer.py"
    ]
    
    dataclass_patterns = [
        "@dataclass",
        "class.*Config",
        "class.*Request",
        "class.*Response",
        "class.*Spec"
    ]
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                content = f.read()
                
                dataclass_count = content.count("@dataclass")
                print(f"✅ {Path(file_path).name}: Found {dataclass_count} dataclasses")
        else:
            print(f"⚠️  {file_path}: File not found")

def validate_test_structure():
    """Validate test file structure and patterns"""
    print("🧪 Validating test structure...")
    
    test_files = [
        "tests/python/unit/test_lmstudio_connector.py",
        "tests/python/unit/test_local_llm_orchestrator.py",
        "tests/python/unit/test_mlx_accelerator.py", 
        "tests/python/unit/test_ollama_optimizer.py",
        "tests/python/integration/test_multi_framework_coordination.py"
    ]
    
    required_patterns = [
        "import pytest",
        "class Test",
        "def test_",
        "@pytest.fixture",
        "@pytest.mark.asyncio"
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            with open(test_file, 'r') as f:
                content = f.read()
                
                patterns_found = []
                for pattern in required_patterns:
                    if pattern in content:
                        patterns_found.append(pattern)
                
                test_functions = content.count("def test_")
                async_tests = content.count("@pytest.mark.asyncio")
                test_classes = content.count("class Test")
                
                print(f"✅ {Path(test_file).name}: {test_functions} tests, {async_tests} async, {test_classes} test classes")
                
                if len(patterns_found) >= 3:
                    print(f"   ✅ Good test structure")
                else:
                    print(f"   ⚠️  Missing patterns: {set(required_patterns) - set(patterns_found)}")
        else:
            print(f"❌ {test_file}: File not found")

def main():
    """Main validation function"""
    print("🚀 Starting Python code validation")
    print("="*60)
    
    validator = CodeValidator()
    
    # Validate source files
    print("1️⃣ Validating source file syntax...")
    source_files = list(Path("docs/python/integration").glob("*.py"))
    
    syntax_valid = True
    for source_file in source_files:
        if not validator.validate_file_syntax(source_file):
            syntax_valid = False
    
    if syntax_valid:
        print("✅ All source files have valid syntax")
    else:
        print(f"❌ Found {len(validator.results['syntax_errors'])} syntax errors")
    
    # Validate test files
    print("\n2️⃣ Validating test file syntax...")
    test_files = list(Path("tests/python").glob("**/*.py"))
    
    test_syntax_valid = True
    for test_file in test_files:
        if not validator.validate_file_syntax(test_file):
            test_syntax_valid = False
    
    if test_syntax_valid:
        print("✅ All test files have valid syntax")
    
    # Analyze test coverage
    print("\n3️⃣ Analyzing test coverage...")
    coverage_percentage = validator.validate_test_coverage()
    
    # Validate specific patterns
    print("\n4️⃣ Validating code patterns...")
    validate_dataclass_definitions()
    
    print("\n5️⃣ Validating test patterns...")
    validate_test_structure()
    
    # Summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    print(f"Files checked: {validator.results['files_checked']}")
    print(f"Classes found: {validator.results['classes_found']}")
    print(f"Functions found: {validator.results['functions_found']}")
    print(f"Async functions found: {validator.results['async_functions_found']}")
    print(f"Syntax errors: {len(validator.results['syntax_errors'])}")
    print(f"Design issues: {len(validator.results['design_issues'])}")
    print(f"Test coverage: {coverage_percentage:.1f}%")
    
    # Determine overall success
    overall_success = (
        syntax_valid and 
        test_syntax_valid and
        len(validator.results["syntax_errors"]) == 0 and
        coverage_percentage >= 80  # Adjusted for file-based coverage
    )
    
    if overall_success:
        print("\n🎉 All validations passed!")
        return 0
    else:
        print("\n❌ Some validations failed. Check details above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)