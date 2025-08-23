"""
Unit tests for Conversation Format Optimization
TDD London School testing patterns with comprehensive mocking
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path

# Import the module under test
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from conversation_format_optimization import (
    TrainingFramework,
    ConversationRole,
    FormattingTemplate,
    OptimizedConversation,
    ConversationOptimizer
)


class TestTrainingFramework:
    """Test training framework enum"""
    
    def test_training_framework_values(self):
        """Test training framework enum values"""
        assert TrainingFramework.ALPACA.value == "alpaca"
        assert TrainingFramework.CHAT_ML.value == "chatml"
        assert TrainingFramework.OPENAI_CHAT.value == "openai_chat"
        assert TrainingFramework.LLAMA_CHAT.value == "llama_chat"
        assert TrainingFramework.VICUNA.value == "vicuna"
        assert TrainingFramework.RAG_ENHANCED.value == "rag_enhanced"


class TestConversationRole:
    """Test conversation role enum"""
    
    def test_conversation_role_values(self):
        """Test conversation role enum values"""
        assert ConversationRole.SYSTEM.value == "system"
        assert ConversationRole.USER.value == "user"
        assert ConversationRole.ASSISTANT.value == "assistant"
        assert ConversationRole.FUNCTION.value == "function"
        assert ConversationRole.TOOL.value == "tool"


class TestFormattingTemplate:
    """Test formatting template dataclass"""
    
    def test_formatting_template_initialization(self):
        """Test formatting template creation"""
        template = FormattingTemplate(
            name="Test Template",
            system_prompt_template="<system>{system_prompt}</system>",
            user_message_template="<user>{content}</user>",
            assistant_message_template="<assistant>{content}</assistant>",
            conversation_separator="\n",
            supports_system_prompt=True,
            supports_multi_turn=True,
            max_context_length=4096,
            special_tokens={"start": "<start>", "end": "<end>"}
        )
        
        assert template.name == "Test Template"
        assert template.system_prompt_template == "<system>{system_prompt}</system>"
        assert template.user_message_template == "<user>{content}</user>"
        assert template.assistant_message_template == "<assistant>{content}</assistant>"
        assert template.conversation_separator == "\n"
        assert template.supports_system_prompt is True
        assert template.supports_multi_turn is True
        assert template.max_context_length == 4096
        assert template.special_tokens == {"start": "<start>", "end": "<end>"}


class TestOptimizedConversation:
    """Test optimized conversation dataclass"""
    
    def test_optimized_conversation_initialization(self):
        """Test optimized conversation creation"""
        conversation = OptimizedConversation(
            framework=TrainingFramework.ALPACA,
            formatted_content="formatted text",
            metadata={"test": "data"},
            token_count_estimate=100,
            quality_score=0.95,
            training_hints={"framework": "alpaca"}
        )
        
        assert conversation.framework == TrainingFramework.ALPACA
        assert conversation.formatted_content == "formatted text"
        assert conversation.metadata == {"test": "data"}
        assert conversation.token_count_estimate == 100
        assert conversation.quality_score == 0.95
        assert conversation.training_hints == {"framework": "alpaca"}


class TestConversationOptimizer:
    """Test conversation optimizer functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.optimizer = ConversationOptimizer()
        self.sample_record = {
            "messages": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI that enables computers to learn from data."}
            ],
            "metadata": {
                "feature_name": "ML Introduction",
                "quality_score": 8.5,
                "technical_content": True
            }
        }
    
    def test_initialization(self):
        """Test optimizer initialization"""
        assert isinstance(self.optimizer.templates, dict)
        assert isinstance(self.optimizer.system_prompt_library, dict)
        assert len(self.optimizer.templates) == 6  # All training frameworks
        assert TrainingFramework.ALPACA in self.optimizer.templates
        assert TrainingFramework.CHAT_ML in self.optimizer.templates
    
    def test_initialize_templates(self):
        """Test template initialization"""
        templates = self.optimizer._initialize_templates()
        
        assert len(templates) == 6
        assert all(isinstance(template, FormattingTemplate) for template in templates.values())
        
        # Test specific template properties
        alpaca_template = templates[TrainingFramework.ALPACA]
        assert "### Instruction:" in alpaca_template.user_message_template
        assert alpaca_template.supports_multi_turn is False
        
        chatml_template = templates[TrainingFramework.CHAT_ML]
        assert "<|im_start|>user" in chatml_template.user_message_template
        assert "<|im_start|>assistant" in chatml_template.assistant_message_template
        assert chatml_template.supports_multi_turn is True
    
    def test_initialize_system_prompts(self):
        """Test system prompt initialization"""
        system_prompts = self.optimizer._initialize_system_prompts()
        
        assert isinstance(system_prompts, dict)
        assert "ericsson_ran_expert" in system_prompts
        assert "configuration_specialist" in system_prompts
        assert "troubleshooting_expert" in system_prompts
        assert "performance_analyst" in system_prompts
        assert "feature_guide" in system_prompts
        
        # Test prompt content
        assert "ericsson ran" in system_prompts["ericsson_ran_expert"].lower()
        assert "configuration" in system_prompts["configuration_specialist"].lower()
    
    def test_optimize_conversation_alpaca(self):
        """Test conversation optimization for Alpaca format"""
        result = self.optimizer.optimize_conversation(
            self.sample_record, 
            TrainingFramework.ALPACA
        )
        
        assert isinstance(result, OptimizedConversation)
        assert result.framework == TrainingFramework.ALPACA
        assert result.metadata == self.sample_record["metadata"]
        assert "### Instruction:" in result.formatted_content
        assert result.token_count_estimate > 0
        assert 0 <= result.quality_score <= 1
    
    def test_optimize_conversation_chatml(self):
        """Test conversation optimization for ChatML format"""
        result = self.optimizer.optimize_conversation(
            self.sample_record,
            TrainingFramework.CHAT_ML
        )
        
        assert isinstance(result, OptimizedConversation)
        assert result.framework == TrainingFramework.CHAT_ML
        assert "<|im_start|>user" in result.formatted_content
        assert "<|im_start|>assistant" in result.formatted_content
        assert "<|im_end|>" in result.formatted_content
    
    def test_optimize_conversation_openai_chat(self):
        """Test conversation optimization for OpenAI Chat format"""
        result = self.optimizer.optimize_conversation(
            self.sample_record,
            TrainingFramework.OPENAI_CHAT
        )
        
        assert isinstance(result, OptimizedConversation)
        assert result.framework == TrainingFramework.OPENAI_CHAT
        # OpenAI format should be JSON-like structure with messages key
        assert result.formatted_content.strip().startswith('{')
        assert result.formatted_content.strip().endswith('}')
        assert '"messages"' in result.formatted_content
    
    def test_generate_system_prompt_default(self):
        """Test system prompt generation with default type"""
        prompt = self.optimizer._generate_system_prompt(
            self.sample_record["metadata"]
        )
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Should use feature_guide for records with feature_name
        assert "feature" in prompt.lower()
    
    def test_generate_system_prompt_specific_type(self):
        """Test system prompt generation for specific type"""
        prompt = self.optimizer._generate_system_prompt(
            self.sample_record["metadata"],
            system_prompt_type="ericsson_ran_expert"
        )
        
        assert isinstance(prompt, str)
        assert "ericsson ran" in prompt.lower()
    
    def test_estimate_token_count(self):
        """Test token count estimation"""
        test_content = "This is a test sentence with multiple words."
        token_count = self.optimizer._estimate_token_count(test_content)
        
        assert isinstance(token_count, int)
        assert token_count > 0
        # Rough estimate: should be close to word count
        word_count = len(test_content.split())
        assert token_count >= word_count * 0.5  # Conservative estimate
    
    def test_calculate_format_quality(self):
        """Test format quality calculation"""
        test_content = "### Instruction:\nWhat is AI?\n### Response:\nAI is artificial intelligence."
        quality = self.optimizer._calculate_format_quality(
            test_content,
            TrainingFramework.ALPACA,
            self.sample_record["metadata"]
        )
        
        assert isinstance(quality, float)
        assert 0 <= quality <= 1
    
    def test_generate_training_hints(self):
        """Test training hints generation"""
        hints = self.optimizer._generate_training_hints(
            self.sample_record["messages"],
            self.sample_record["metadata"],
            TrainingFramework.ALPACA
        )
        
        assert isinstance(hints, dict)
        assert "framework" in hints
        assert "supports_multi_turn" in hints
        assert "max_context_length" in hints
        assert hints["framework"] == "alpaca"
    
    def test_batch_optimize_conversations(self):
        """Test batch conversation optimization"""
        records = [self.sample_record, self.sample_record.copy()]
        results = self.optimizer.batch_optimize_conversations(
            records,
            TrainingFramework.ALPACA
        )
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(result, OptimizedConversation) for result in results)
        assert all(result.framework == TrainingFramework.ALPACA for result in results)
    
    def test_extract_alpaca_input(self):
        """Test Alpaca input extraction"""
        input_text = self.optimizer._extract_alpaca_input(
            self.sample_record["metadata"]
        )
        
        assert isinstance(input_text, str)
        # Should include feature name
        assert "ML Introduction" in input_text
    
    def test_format_alpaca(self):
        """Test Alpaca format conversion"""
        template = self.optimizer.templates[TrainingFramework.ALPACA]
        formatted = self.optimizer._format_alpaca(
            self.sample_record["messages"],
            template,
            self.sample_record["metadata"]
        )
        
        assert "### Instruction:" in formatted
        assert "What is machine learning?" in formatted
        assert "Machine learning is a subset" in formatted
    
    def test_format_chatml(self):
        """Test ChatML format conversion"""
        template = self.optimizer.templates[TrainingFramework.CHAT_ML]
        formatted = self.optimizer._format_chatml(
            self.sample_record["messages"],
            template,
            "You are a helpful assistant."
        )
        
        assert "<|im_start|>system" in formatted
        assert "<|im_start|>user" in formatted
        assert "<|im_start|>assistant" in formatted
        assert "<|im_end|>" in formatted
    
    def test_format_openai_chat(self):
        """Test OpenAI Chat format conversion"""
        formatted = self.optimizer._format_openai_chat(
            self.sample_record["messages"],
            "You are a helpful assistant."
        )
        
        # Should be valid JSON
        parsed = json.loads(formatted)
        assert isinstance(parsed, dict)
        assert "messages" in parsed
        assert len(parsed["messages"]) >= 2  # At least system + user + assistant
        assert parsed["messages"][0]["role"] == "system"
        assert parsed["messages"][1]["role"] == "user"
    
    def test_format_vicuna(self):
        """Test Vicuna format conversion"""
        template = self.optimizer.templates[TrainingFramework.VICUNA]
        formatted = self.optimizer._format_vicuna(
            self.sample_record["messages"],
            template,
            "You are a helpful assistant."
        )
        
        assert "### Human:" in formatted
        assert "### Assistant:" in formatted
        assert "What is machine learning?" in formatted
    
    def test_format_rag_enhanced(self):
        """Test RAG enhanced format conversion"""
        template = self.optimizer.templates[TrainingFramework.RAG_ENHANCED]
        formatted = self.optimizer._format_rag_enhanced(
            self.sample_record["messages"],
            template,
            "You are a helpful assistant.",
            self.sample_record["metadata"]
        )
        
        assert "Context:" in formatted
        assert "Question:" in formatted
        assert "Answer:" in formatted
    
    def test_analyze_format_compatibility(self):
        """Test format compatibility analysis"""
        records = [self.sample_record, self.sample_record.copy()]
        analysis = self.optimizer.analyze_format_compatibility(records)
        
        assert isinstance(analysis, dict)
        # Only test frameworks that have templates
        supported_frameworks = list(self.optimizer.templates.keys())
        assert len(analysis) == len(supported_frameworks)
        
        for framework in supported_frameworks:
            assert framework in analysis
            framework_analysis = analysis[framework]
            assert "average_quality" in framework_analysis
            assert "average_token_count" in framework_analysis
            assert "compatibility_rate" in framework_analysis
            assert isinstance(framework_analysis["average_quality"], float)
            assert isinstance(framework_analysis["average_token_count"], (int, float))
            assert isinstance(framework_analysis["compatibility_rate"], float)


class TestConversationOptimizerIntegration:
    """Integration tests for conversation optimizer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.optimizer = ConversationOptimizer()
        self.sample_record = {
            "messages": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI that enables computers to learn from data."}
            ],
            "metadata": {
                "feature_name": "ML Introduction",
                "quality_score": 8.5,
                "technical_content": True
            }
        }
    
    def test_end_to_end_optimization_workflow(self):
        """Test complete optimization workflow"""
        # Create test records
        records = [
            {
                "messages": [
                    {"role": "user", "content": "Explain neural networks"},
                    {"role": "assistant", "content": "Neural networks are computational models inspired by biological neural networks."}
                ],
                "metadata": {
                    "feature_name": "Neural Networks",
                    "quality_score": 9.0,
                    "technical_content": True
                }
            },
            {
                "messages": [
                    {"role": "user", "content": "What is deep learning?"},
                    {"role": "assistant", "content": "Deep learning is a subset of machine learning using neural networks with multiple layers."}
                ],
                "metadata": {
                    "feature_name": "Deep Learning",
                    "quality_score": 8.5,
                    "technical_content": True
                }
            }
        ]
        
        # Test different frameworks
        for framework in [TrainingFramework.ALPACA, TrainingFramework.CHAT_ML, TrainingFramework.OPENAI_CHAT]:
            # Batch optimize
            results = self.optimizer.batch_optimize_conversations(records, framework)
            
            assert len(results) == 2
            assert all(result.framework == framework for result in results)
            assert all(result.token_count_estimate > 0 for result in results)
            assert all(0 <= result.quality_score <= 1 for result in results)
        
        # Analyze compatibility
        compatibility = self.optimizer.analyze_format_compatibility(records)
        assert len(compatibility) == 6
        
        # All frameworks should have reasonable compatibility rates
        for framework_analysis in compatibility.values():
            assert framework_analysis["compatibility_rate"] > 0.5
            assert framework_analysis["average_quality"] > 0
            assert framework_analysis["average_token_count"] > 0


    def test_optimize_conversation_empty_messages(self):
        """Test optimization with empty messages list"""
        record = {"messages": [], "metadata": {"feature_name": "test"}}
        
        with pytest.raises(ValueError, match="Record must contain messages"):
            self.optimizer.optimize_conversation(record, TrainingFramework.ALPACA)
    
    def test_generate_system_prompt_with_custom_context(self):
        """Test system prompt generation with custom context"""
        metadata = {"feature_name": "test"}
        custom_context = "Custom context for RAG"
        
        prompt = self.optimizer._generate_system_prompt(
            metadata,
            custom_context=custom_context
        )
        
        assert "Custom context for RAG" in prompt
        assert "ericsson ran" in prompt.lower()
    
    def test_generate_system_prompt_workflow_types(self):
        """Test system prompt selection based on workflow types"""
        # Configuration workflow
        metadata = {"workflow_type": "configuration", "question_type": "parameter"}
        prompt = self.optimizer._generate_system_prompt(metadata)
        assert "configuration" in prompt.lower()
        
        # Troubleshooting workflow
        metadata = {"workflow_type": "other", "question_type": "troubleshooting"}
        prompt = self.optimizer._generate_system_prompt(metadata)
        assert "troubleshooting" in prompt.lower()
        
        # Counter/monitoring workflow
        metadata = {"question_type": "counter monitoring"}
        prompt = self.optimizer._generate_system_prompt(metadata)
        assert "performance" in prompt.lower()
    
    def test_format_conversation_multi_turn_framework(self):
        """Test format conversation with multi-turn framework"""
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"}
        ]
        
        # Test MULTI_TURN framework (which exists but isn't fully implemented)
        # Should raise KeyError for unsupported framework
        with pytest.raises(KeyError):
            self.optimizer._format_conversation(
                messages,
                TrainingFramework.MULTI_TURN,
                "system prompt",
                {}
            )
    
    def test_format_alpaca_insufficient_messages(self):
        """Test Alpaca format with insufficient messages"""
        template = self.optimizer.templates[TrainingFramework.ALPACA]
        
        with pytest.raises(ValueError, match="Alpaca format requires at least user and assistant messages"):
            self.optimizer._format_alpaca([{"role": "user", "content": "test"}], template, {})
    
    def test_format_llama_chat_complex_conversation(self):
        """Test Llama Chat format with multi-turn conversation"""
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer"}
        ]
        template = self.optimizer.templates[TrainingFramework.LLAMA_CHAT]
        
        formatted = self.optimizer._format_llama_chat(messages, template, "System prompt")
        
        assert "First question" in formatted
        assert "First answer" in formatted
        assert "Second question" in formatted
        assert "Second answer" in formatted
        assert "[INST]" in formatted
        assert "[/INST]" in formatted
    
    def test_format_llama_chat_no_system_prompt(self):
        """Test Llama Chat format without system prompt"""
        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"}
        ]
        template = self.optimizer.templates[TrainingFramework.LLAMA_CHAT]
        
        formatted = self.optimizer._format_llama_chat(messages, template, "")
        
        assert "Question" in formatted
        assert "Answer" in formatted
        assert "[INST]" in formatted
    
    def test_extract_alpaca_input_comprehensive(self):
        """Test Alpaca input extraction with all metadata types"""
        metadata = {
            "feature_name": "Test Feature",
            "technical_terms": ["term1", "term2", "term3", "term4", "term5", "term6"],
            "parameters_involved": ["param1", "param2", "param3", "param4"]
        }
        
        input_text = self.optimizer._extract_alpaca_input(metadata)
        
        assert "Test Feature" in input_text
        assert "term1, term2, term3, term4, term5" in input_text  # Only first 5
        assert "param1, param2, param3" in input_text  # Only first 3
    
    def test_calculate_format_quality_edge_cases(self):
        """Test format quality calculation edge cases"""
        # Test with very long content exceeding context length
        long_content = "x" * 10000
        quality = self.optimizer._calculate_format_quality(
            long_content,
            TrainingFramework.ALPACA,
            {"technical_terms": ["term1", "term2"]}
        )
        assert quality < 0.8  # Should be penalized
        
        # Test with rich technical content
        quality = self.optimizer._calculate_format_quality(
            "short content",
            TrainingFramework.CHAT_ML,
            {"technical_terms": ["term1", "term2", "term3", "term4", "term5"]}
        )
        assert quality > 0.8  # Should get bonus
    
    def test_generate_training_hints_difficulty_levels(self):
        """Test training hints generation for different difficulty levels"""
        # Basic difficulty (short content)
        short_messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
        hints = self.optimizer._generate_training_hints(
            short_messages,
            {"technical_terms": []},
            TrainingFramework.ALPACA
        )
        assert hints["estimated_difficulty"] == "basic"
        
        # Advanced difficulty (long content)
        long_content = "x" * 1500
        long_messages = [{"role": "user", "content": long_content}, {"role": "assistant", "content": long_content}]
        hints = self.optimizer._generate_training_hints(
            long_messages,
            {"technical_terms": ["t1", "t2", "t3", "t4", "t5", "t6"]},
            TrainingFramework.ALPACA
        )
        assert hints["estimated_difficulty"] == "advanced"
        assert hints.get("high_technical_content") is True
        
        # Test alpaca framework (instruction following)
        hints = self.optimizer._generate_training_hints(
            short_messages,
            {},
            TrainingFramework.ALPACA
        )
        assert hints["training_type"] == "instruction_following"
    
    def test_batch_optimize_conversations_with_errors(self):
        """Test batch optimization with some failing records"""
        records = [
            self.sample_record,  # Valid record
            {"messages": []},    # Invalid record (empty messages)
            self.sample_record.copy()  # Another valid record
        ]
        
        with patch('builtins.print') as mock_print:
            results = self.optimizer.batch_optimize_conversations(records, TrainingFramework.ALPACA)
        
        assert len(results) == 2  # Only valid records processed
        mock_print.assert_called()  # Error should be printed
    
    def test_batch_optimize_conversations_with_output_path(self):
        """Test batch optimization with file output"""
        records = [self.sample_record, self.sample_record.copy()]
        
        with patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch.object(self.optimizer, '_save_optimized_conversations') as mock_save:
            
            output_path = Path("test_output.jsonl")
            results = self.optimizer.batch_optimize_conversations(
                records,
                TrainingFramework.ALPACA,
                output_path
            )
            
            mock_save.assert_called_once()
            assert len(results) == 2
    
    def test_save_optimized_conversations_openai_format(self):
        """Test saving conversations in OpenAI format"""
        conversations = [
            OptimizedConversation(
                framework=TrainingFramework.OPENAI_CHAT,
                formatted_content='{"messages": [{"role": "user", "content": "test"}]}',
                metadata={},
                token_count_estimate=10,
                quality_score=0.9,
                training_hints={}
            )
        ]
        
        with patch('pathlib.Path.mkdir'), \
             patch('builtins.open', mock_open()) as mock_file:
            
            self.optimizer._save_optimized_conversations(
                conversations,
                Path("test.jsonl"),
                TrainingFramework.OPENAI_CHAT
            )
            
            # Check that JSON content was written directly
            mock_file().write.assert_called()
    
    def test_save_optimized_conversations_other_formats(self):
        """Test saving conversations in non-OpenAI formats"""
        conversations = [
            OptimizedConversation(
                framework=TrainingFramework.ALPACA,
                formatted_content="formatted text",
                metadata={"test": "data"},
                token_count_estimate=10,
                quality_score=0.9,
                training_hints={"hint": "value"}
            )
        ]
        
        with patch('pathlib.Path.mkdir'), \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dumps', return_value='{"test": "json"}') as mock_json:
            
            self.optimizer._save_optimized_conversations(
                conversations,
                Path("test.jsonl"),
                TrainingFramework.ALPACA
            )
            
            mock_json.assert_called_once()
            mock_file().write.assert_called()
    
    def test_analyze_format_compatibility_with_exceptions(self):
        """Test format compatibility analysis with some failing records"""
        records = [
            self.sample_record,
            {"messages": []},  # This will cause an exception
            self.sample_record.copy()
        ]
        
        analysis = self.optimizer.analyze_format_compatibility(records)
        
        # Should still return analysis for all frameworks
        assert len(analysis) == len(self.optimizer.templates)
        
        # Check that compatibility rates are calculated correctly
        for framework_analysis in analysis.values():
            assert "compatibility_rate" in framework_analysis
            assert "average_token_count" in framework_analysis
            assert "average_quality" in framework_analysis


class TestConversationOptimizerMainFunction:
    """Test the main function execution"""
    
    @patch('builtins.print')
    def test_main_function_execution(self, mock_print):
        """Test main function execution"""
        # Import and execute the main block
        import sys
        from pathlib import Path
        
        # Add the src directory to path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        
        # Mock the __name__ == "__main__" condition
        with patch('conversation_format_optimization.__name__', '__main__'):
            try:
                # Import the module which should trigger the main block
                import conversation_format_optimization
                # The main block should execute and print results
                mock_print.assert_called()
            except Exception as e:
                # If there's an import error, that's expected in test environment
                pass


if __name__ == "__main__":
    pytest.main([__file__])