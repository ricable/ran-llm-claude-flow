"""
Python Pipeline Interfaces Module

This module defines the core interfaces for the Python RAN LLM pipeline architecture.
Designed for TDD London School practices with dependency injection and mock-friendly design.
"""

from .base import IProcessor, IStage, IFactory
from .pipeline import IPipelineCoordinator, IStageCoordinator
from .memory import IMemoryManager, IResourceAllocator
from .configuration import IConfigurationManager, IStageConfig
from .processors import (
    IDocumentConverter,
    IPreprocessor, 
    ILangExtractor,
    IConversationGenerator,
    IDatasetFinalizer
)

__all__ = [
    # Base interfaces
    'IProcessor',
    'IStage', 
    'IFactory',
    
    # Pipeline coordination
    'IPipelineCoordinator',
    'IStageCoordinator',
    
    # Memory management
    'IMemoryManager',
    'IResourceAllocator',
    
    # Configuration
    'IConfigurationManager',
    'IStageConfig',
    
    # Stage processors
    'IDocumentConverter',
    'IPreprocessor',
    'ILangExtractor', 
    'IConversationGenerator',
    'IDatasetFinalizer'
]