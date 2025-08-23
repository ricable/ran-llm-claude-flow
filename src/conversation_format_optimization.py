"""
Conversation Format Optimization for LLM Fine-tuning
Advanced conversation formatting and optimization for various training frameworks
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re
from pathlib import Path

class TrainingFramework(Enum):
    """Supported training frameworks with specific format requirements"""
    ALPACA = "alpaca"
    CHAT_ML = "chatml"
    OPENAI_CHAT = "openai_chat"
    LLAMA_CHAT = "llama_chat"
    VICUNA = "vicuna"
    CUSTOM_INSTRUCTION = "custom_instruction"
    RAG_ENHANCED = "rag_enhanced"
    MULTI_TURN = "multi_turn"

class ConversationRole(Enum):
    """Standardized conversation roles"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"

@dataclass
class FormattingTemplate:
    """Template configuration for specific training frameworks"""
    name: str
    system_prompt_template: str
    user_message_template: str
    assistant_message_template: str
    conversation_separator: str
    supports_system_prompt: bool
    supports_multi_turn: bool
    max_context_length: int
    special_tokens: Dict[str, str]

@dataclass
class OptimizedConversation:
    """Optimized conversation ready for training"""
    framework: TrainingFramework
    formatted_content: str
    metadata: Dict[str, Any]
    token_count_estimate: int
    quality_score: float
    training_hints: Dict[str, Any]

class ConversationOptimizer:
    """Main class for optimizing conversations for different training frameworks"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.system_prompt_library = self._initialize_system_prompts()
        
    def _initialize_templates(self) -> Dict[TrainingFramework, FormattingTemplate]:
        """Initialize formatting templates for different frameworks"""
        return {
            TrainingFramework.ALPACA: FormattingTemplate(
                name="Alpaca",
                system_prompt_template="",  # Alpaca doesn't use system prompts typically
                user_message_template="### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
                assistant_message_template="{output}",
                conversation_separator="\n\n",
                supports_system_prompt=False,
                supports_multi_turn=False,
                max_context_length=2048,
                special_tokens={"eos": "</s>", "bos": "<s>"}
            ),
            
            TrainingFramework.CHAT_ML: FormattingTemplate(
                name="ChatML",
                system_prompt_template="<|im_start|>system\n{system_prompt}<|im_end|>\n",
                user_message_template="<|im_start|>user\n{content}<|im_end|>\n",
                assistant_message_template="<|im_start|>assistant\n{content}<|im_end|>\n",
                conversation_separator="",
                supports_system_prompt=True,
                supports_multi_turn=True,
                max_context_length=4096,
                special_tokens={"start": "<|im_start|>", "end": "<|im_end|>"}
            ),
            
            TrainingFramework.OPENAI_CHAT: FormattingTemplate(
                name="OpenAI Chat",
                system_prompt_template="",  # Handled in messages array
                user_message_template="",   # Handled in messages array
                assistant_message_template="",  # Handled in messages array
                conversation_separator="",
                supports_system_prompt=True,
                supports_multi_turn=True,
                max_context_length=4096,
                special_tokens={}
            ),
            
            TrainingFramework.LLAMA_CHAT: FormattingTemplate(
                name="Llama Chat",
                system_prompt_template="<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n",
                user_message_template="[INST] {content} [/INST] ",
                assistant_message_template="{content} </s><s>",
                conversation_separator="",
                supports_system_prompt=True,
                supports_multi_turn=True,
                max_context_length=4096,
                special_tokens={"bos": "<s>", "eos": "</s>", "inst_start": "[INST]", "inst_end": "[/INST]"}
            ),
            
            TrainingFramework.VICUNA: FormattingTemplate(
                name="Vicuna",
                system_prompt_template="### System:\n{system_prompt}\n\n",
                user_message_template="### Human: {content}\n### Assistant: ",
                assistant_message_template="{content}\n",
                conversation_separator="\n",
                supports_system_prompt=True,
                supports_multi_turn=True,
                max_context_length=2048,
                special_tokens={}
            ),
            
            TrainingFramework.RAG_ENHANCED: FormattingTemplate(
                name="RAG Enhanced",
                system_prompt_template="Context: {context}\n\nYou are an expert Ericsson RAN engineer. Use the provided context to answer technical questions accurately.\n\n",
                user_message_template="Question: {content}\n\nAnswer: ",
                assistant_message_template="{content}",
                conversation_separator="\n\n",
                supports_system_prompt=True,
                supports_multi_turn=False,
                max_context_length=8192,
                special_tokens={"context_start": "[CONTEXT]", "context_end": "[/CONTEXT]"}
            )
        }
    
    def _initialize_system_prompts(self) -> Dict[str, str]:
        """Initialize library of system prompts for different scenarios"""
        return {
            "ericsson_ran_expert": """You are an expert Ericsson RAN (Radio Access Network) engineer with deep knowledge of LTE, 5G NR, and network optimization. You provide accurate, technical responses about RAN configuration, troubleshooting, and performance optimization. Always include specific parameter names, MO classes, and counter references when applicable.""",
            
            "configuration_specialist": """You are a specialist in Ericsson RAN configuration management. You help with parameter configuration, feature activation, and MO (Managed Object) class operations. Provide step-by-step configuration procedures with exact cmedit commands when requested.""",
            
            "troubleshooting_expert": """You are an expert in Ericsson RAN troubleshooting and diagnostics. You analyze performance issues, investigate KPI degradations, and provide systematic diagnostic approaches. Always suggest specific counters to check and validation steps to confirm fixes.""",
            
            "performance_analyst": """You are a RAN performance analyst specializing in KPI analysis, counter interpretation, and network optimization. You help analyze performance data, identify bottlenecks, and recommend optimization strategies based on measurement results.""",
            
            "feature_guide": """You are a comprehensive guide for Ericsson RAN features. You explain feature functionality, activation procedures, dependencies, and best practices. Provide clear explanations of feature benefits and configuration recommendations."""
        }
    
    def optimize_conversation(self, record: Dict, framework: TrainingFramework, 
                            system_prompt_type: Optional[str] = None,
                            custom_context: Optional[str] = None) -> OptimizedConversation:
        """
        Optimize a conversation record for a specific training framework
        
        Args:
            record: Original conversation record
            framework: Target training framework
            system_prompt_type: Type of system prompt to use
            custom_context: Additional context for RAG-enhanced formats
            
        Returns:
            OptimizedConversation: Formatted conversation ready for training
        """
        
        messages = record.get("messages", [])
        metadata = record.get("metadata", {})
        
        if not messages:
            raise ValueError("Record must contain messages")
        
        # Generate appropriate system prompt
        system_prompt = self._generate_system_prompt(metadata, system_prompt_type, custom_context)
        
        # Format conversation based on framework
        formatted_content = self._format_conversation(messages, framework, system_prompt, metadata)
        
        # Estimate token count
        token_count = self._estimate_token_count(formatted_content)
        
        # Calculate quality score
        quality_score = self._calculate_format_quality(formatted_content, framework, metadata)
        
        # Generate training hints
        training_hints = self._generate_training_hints(messages, metadata, framework)
        
        return OptimizedConversation(
            framework=framework,
            formatted_content=formatted_content,
            metadata=metadata,
            token_count_estimate=token_count,
            quality_score=quality_score,
            training_hints=training_hints
        )
    
    def _generate_system_prompt(self, metadata: Dict, system_prompt_type: Optional[str] = None,
                               custom_context: Optional[str] = None) -> str:
        """Generate appropriate system prompt based on content and metadata"""
        
        # Use custom context if provided (for RAG)
        if custom_context:
            return f"Context: {custom_context}\n\n{self.system_prompt_library['ericsson_ran_expert']}"
        
        # Use specified system prompt type
        if system_prompt_type and system_prompt_type in self.system_prompt_library:
            return self.system_prompt_library[system_prompt_type]
        
        # Auto-select based on content analysis
        feature_name = metadata.get("feature_name", "")
        question_type = metadata.get("question_type", "")
        workflow_type = metadata.get("workflow_type", "")
        
        if "configuration" in workflow_type.lower() or "parameter" in question_type.lower():
            return self.system_prompt_library["configuration_specialist"]
        elif "troubleshooting" in question_type.lower() or "diagnose" in question_type.lower():
            return self.system_prompt_library["troubleshooting_expert"]
        elif "counter" in question_type.lower() or "monitoring" in question_type.lower():
            return self.system_prompt_library["performance_analyst"]
        elif feature_name:
            return self.system_prompt_library["feature_guide"]
        else:
            return self.system_prompt_library["ericsson_ran_expert"]
    
    def _format_conversation(self, messages: List[Dict], framework: TrainingFramework,
                           system_prompt: str, metadata: Dict) -> str:
        """Format conversation for specific framework"""
        
        template = self.templates[framework]
        
        if framework == TrainingFramework.ALPACA:
            return self._format_alpaca(messages, template, metadata)
        elif framework == TrainingFramework.CHAT_ML:
            return self._format_chatml(messages, template, system_prompt)
        elif framework == TrainingFramework.OPENAI_CHAT:
            return self._format_openai_chat(messages, system_prompt)
        elif framework == TrainingFramework.LLAMA_CHAT:
            return self._format_llama_chat(messages, template, system_prompt)
        elif framework == TrainingFramework.VICUNA:
            return self._format_vicuna(messages, template, system_prompt)
        elif framework == TrainingFramework.RAG_ENHANCED:
            return self._format_rag_enhanced(messages, template, system_prompt, metadata)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def _format_alpaca(self, messages: List[Dict], template: FormattingTemplate, 
                      metadata: Dict) -> str:
        """Format for Alpaca training"""
        
        if len(messages) < 2:
            raise ValueError("Alpaca format requires at least user and assistant messages")
        
        user_message = messages[0].get("content", "")
        assistant_message = messages[1].get("content", "")
        
        # Extract context from metadata for input field
        input_context = self._extract_alpaca_input(metadata)
        
        instruction = user_message
        formatted = template.user_message_template.format(
            instruction=instruction,
            input=input_context
        )
        formatted += template.assistant_message_template.format(output=assistant_message)
        
        return formatted
    
    def _format_chatml(self, messages: List[Dict], template: FormattingTemplate,
                      system_prompt: str) -> str:
        """Format for ChatML training"""
        
        formatted_parts = []
        
        # Add system prompt if available
        if system_prompt and template.supports_system_prompt:
            formatted_parts.append(template.system_prompt_template.format(system_prompt=system_prompt))
        
        # Format each message
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "user":
                formatted_parts.append(template.user_message_template.format(content=content))
            elif role == "assistant":
                formatted_parts.append(template.assistant_message_template.format(content=content))
        
        return "".join(formatted_parts)
    
    def _format_openai_chat(self, messages: List[Dict], system_prompt: str) -> str:
        """Format for OpenAI Chat API training"""
        
        formatted_messages = []
        
        # Add system message if available
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation messages
        formatted_messages.extend(messages)
        
        return json.dumps({"messages": formatted_messages})
    
    def _format_llama_chat(self, messages: List[Dict], template: FormattingTemplate,
                          system_prompt: str) -> str:
        """Format for Llama Chat training"""
        
        formatted_parts = []
        
        # Handle first user message with system prompt
        if messages and messages[0].get("role") == "user":
            first_user_content = messages[0].get("content", "")
            
            if system_prompt:
                formatted_parts.append(template.system_prompt_template.format(system_prompt=system_prompt))
                formatted_parts.append(f"{first_user_content} [/INST] ")
            else:
                formatted_parts.append(template.user_message_template.format(content=first_user_content))
            
            # Add first assistant response
            if len(messages) > 1 and messages[1].get("role") == "assistant":
                assistant_content = messages[1].get("content", "")
                formatted_parts.append(template.assistant_message_template.format(content=assistant_content))
            
            # Handle additional turns
            for i in range(2, len(messages), 2):
                if i < len(messages):
                    user_content = messages[i].get("content", "")
                    formatted_parts.append(template.user_message_template.format(content=user_content))
                
                if i + 1 < len(messages):
                    assistant_content = messages[i + 1].get("content", "")
                    formatted_parts.append(template.assistant_message_template.format(content=assistant_content))
        
        return "".join(formatted_parts)
    
    def _format_vicuna(self, messages: List[Dict], template: FormattingTemplate,
                      system_prompt: str) -> str:
        """Format for Vicuna training"""
        
        formatted_parts = []
        
        # Add system prompt
        if system_prompt:
            formatted_parts.append(template.system_prompt_template.format(system_prompt=system_prompt))
        
        # Format conversation
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "user":
                formatted_parts.append(template.user_message_template.format(content=content))
            elif role == "assistant":
                formatted_parts.append(template.assistant_message_template.format(content=content))
        
        return "".join(formatted_parts)
    
    def _format_rag_enhanced(self, messages: List[Dict], template: FormattingTemplate,
                            system_prompt: str, metadata: Dict) -> str:
        """Format for RAG-enhanced training"""
        
        # Extract context from metadata
        context_parts = []
        
        # Add technical context
        feature_name = metadata.get("feature_name")
        if feature_name:
            context_parts.append(f"Feature: {feature_name}")
        
        technical_terms = metadata.get("technical_terms", [])
        if technical_terms:
            context_parts.append(f"Technical Terms: {', '.join(technical_terms[:10])}")
        
        parameters = metadata.get("parameters_involved", [])
        if parameters:
            context_parts.append(f"Parameters: {', '.join(parameters[:5])}")
        
        mo_classes = metadata.get("mo_classes", [])
        if mo_classes:
            context_parts.append(f"MO Classes: {', '.join(mo_classes[:3])}")
        
        context = " | ".join(context_parts)
        
        # Format with context
        formatted_system = template.system_prompt_template.format(context=context)
        
        if len(messages) >= 2:
            user_content = messages[0].get("content", "")
            assistant_content = messages[1].get("content", "")
            
            formatted_user = template.user_message_template.format(content=user_content)
            formatted_assistant = template.assistant_message_template.format(content=assistant_content)
            
            return formatted_system + formatted_user + formatted_assistant
        
        return formatted_system
    
    def _extract_alpaca_input(self, metadata: Dict) -> str:
        """Extract input context for Alpaca format"""
        
        input_parts = []
        
        # Add feature context
        feature_name = metadata.get("feature_name")
        if feature_name:
            input_parts.append(f"Feature: {feature_name}")
        
        # Add technical terms for context
        technical_terms = metadata.get("technical_terms", [])
        if technical_terms and len(technical_terms) > 0:
            input_parts.append(f"Related terms: {', '.join(technical_terms[:5])}")
        
        # Add parameter context
        parameters = metadata.get("parameters_involved", [])
        if parameters:
            input_parts.append(f"Parameters: {', '.join(parameters[:3])}")
        
        return " | ".join(input_parts) if input_parts else ""
    
    def _estimate_token_count(self, formatted_content: str) -> int:
        """Estimate token count for formatted content"""
        
        # Rough estimation: ~4 characters per token for English text
        # Technical content tends to have longer tokens, so we use 3.5
        char_count = len(formatted_content)
        estimated_tokens = int(char_count / 3.5)
        
        # Add tokens for special formatting
        special_token_patterns = [
            r'<\|.*?\|>',  # ChatML tokens
            r'<s>|</s>',   # BOS/EOS tokens
            r'\[INST\]|\[/INST\]',  # Llama instruction tokens
            r'###\s+\w+:',  # Section headers
        ]
        
        special_token_count = 0
        for pattern in special_token_patterns:
            matches = re.findall(pattern, formatted_content)
            special_token_count += len(matches)
        
        return estimated_tokens + special_token_count
    
    def _calculate_format_quality(self, formatted_content: str, framework: TrainingFramework,
                                 metadata: Dict) -> float:
        """Calculate quality score for formatted conversation"""
        
        base_score = 0.8
        
        # Check for proper formatting
        template = self.templates[framework]
        
        # Verify special tokens are properly formatted
        if framework == TrainingFramework.CHAT_ML:
            if "<|im_start|>" in formatted_content and "<|im_end|>" in formatted_content:
                base_score += 0.1
        elif framework == TrainingFramework.LLAMA_CHAT:
            if "[INST]" in formatted_content and "[/INST]" in formatted_content:
                base_score += 0.1
        
        # Check length appropriateness
        token_count = self._estimate_token_count(formatted_content)
        if token_count <= template.max_context_length:
            base_score += 0.05
        else:
            base_score -= 0.1  # Penalize overly long content
        
        # Reward technical content richness
        technical_terms = metadata.get("technical_terms", [])
        if isinstance(technical_terms, list) and len(technical_terms) > 3:
            base_score += 0.05
        
        return min(1.0, max(0.0, base_score))
    
    def _generate_training_hints(self, messages: List[Dict], metadata: Dict,
                               framework: TrainingFramework) -> Dict[str, Any]:
        """Generate hints for training optimization"""
        
        hints = {
            "framework": framework.value,
            "supports_multi_turn": self.templates[framework].supports_multi_turn,
            "max_context_length": self.templates[framework].max_context_length,
            "estimated_difficulty": "intermediate"  # Default
        }
        
        # Analyze conversation complexity
        total_length = sum(len(msg.get("content", "")) for msg in messages)
        
        if total_length > 1000:
            hints["estimated_difficulty"] = "advanced"
        elif total_length < 200:
            hints["estimated_difficulty"] = "basic"
        
        # Add technical complexity indicators
        technical_terms = metadata.get("technical_terms", [])
        if isinstance(technical_terms, list) and len(technical_terms) > 5:
            hints["high_technical_content"] = True
        
        # Add training recommendations
        if framework in [TrainingFramework.ALPACA, TrainingFramework.CUSTOM_INSTRUCTION]:
            hints["training_type"] = "instruction_following"
        else:
            hints["training_type"] = "conversational"
        
        return hints
    
    def batch_optimize_conversations(self, records: List[Dict], framework: TrainingFramework,
                                   output_path: Optional[Path] = None) -> List[OptimizedConversation]:
        """Optimize multiple conversations in batch"""
        
        optimized_conversations = []
        
        for i, record in enumerate(records):
            try:
                optimized = self.optimize_conversation(record, framework)
                optimized_conversations.append(optimized)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(records)} conversations")
                    
            except Exception as e:
                print(f"Error processing record {i}: {e}")
                continue
        
        # Save to file if path provided
        if output_path:
            self._save_optimized_conversations(optimized_conversations, output_path, framework)
        
        return optimized_conversations
    
    def _save_optimized_conversations(self, conversations: List[OptimizedConversation],
                                    output_path: Path, framework: TrainingFramework):
        """Save optimized conversations to file"""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                if framework == TrainingFramework.OPENAI_CHAT:
                    # OpenAI format is already JSON
                    f.write(conv.formatted_content + '\n')
                else:
                    # Create training record
                    record = {
                        "text": conv.formatted_content,
                        "metadata": conv.metadata,
                        "framework": framework.value,
                        "token_count": conv.token_count_estimate,
                        "quality_score": conv.quality_score,
                        "training_hints": conv.training_hints
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    def analyze_format_compatibility(self, records: List[Dict]) -> Dict[TrainingFramework, Dict[str, Any]]:
        """Analyze dataset compatibility with different training frameworks"""
        
        compatibility_report = {}
        
        for framework in TrainingFramework:
            template = self.templates.get(framework)
            if not template:
                continue
                
            compatible_count = 0
            avg_token_count = 0
            quality_scores = []
            
            for record in records[:100]:  # Sample for analysis
                try:
                    optimized = self.optimize_conversation(record, framework)
                    
                    # Check compatibility
                    if optimized.token_count_estimate <= template.max_context_length:
                        compatible_count += 1
                    
                    avg_token_count += optimized.token_count_estimate
                    quality_scores.append(optimized.quality_score)
                    
                except Exception:
                    continue
            
            compatibility_report[framework] = {
                "compatibility_rate": compatible_count / len(records[:100]),
                "average_token_count": avg_token_count / len(records[:100]) if records else 0,
                "average_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                "supports_multi_turn": template.supports_multi_turn,
                "max_context_length": template.max_context_length
            }
        
        return compatibility_report

# Example usage and testing
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = ConversationOptimizer()
    
    # Sample record
    sample_record = {
        "messages": [
            {
                "role": "user",
                "content": "How do I configure the handoverMargin parameter in EUtranCellFDD?"
            },
            {
                "role": "assistant",
                "content": "To configure the handoverMargin parameter in EUtranCellFDD, use: cmedit set EUtranCellFDD=1 handoverMargin=3. This parameter controls the RSRP offset for handover decisions in dB."
            }
        ],
        "metadata": {
            "feature_name": "LTE Handover Configuration",
            "technical_terms": ["EUtranCellFDD", "handoverMargin", "RSRP"],
            "parameters_involved": ["handoverMargin"],
            "mo_classes": ["EUtranCellFDD"],
            "quality_score": 9.2
        }
    }
    
    # Test different frameworks
    frameworks_to_test = [
        TrainingFramework.ALPACA,
        TrainingFramework.CHAT_ML,
        TrainingFramework.LLAMA_CHAT,
        TrainingFramework.RAG_ENHANCED
    ]
    
    for framework in frameworks_to_test:
        print(f"\n=== {framework.value.upper()} FORMAT ===")
        try:
            optimized = optimizer.optimize_conversation(sample_record, framework)
            print(f"Token count: {optimized.token_count_estimate}")
            print(f"Quality score: {optimized.quality_score:.2f}")
            print(f"Content preview:\n{optimized.formatted_content[:200]}...")
        except Exception as e:
            print(f"Error: {e}")