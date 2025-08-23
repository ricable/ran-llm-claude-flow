"""
Configuration management interfaces.
Defines contracts for pipeline and stage configuration management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ConfigScope(Enum):
    """Configuration scope levels."""
    GLOBAL = "global"
    PIPELINE = "pipeline" 
    STAGE = "stage"
    PROCESSOR = "processor"


class ConfigFormat(Enum):
    """Supported configuration formats."""
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    PYTHON = "python"


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.suggestions is None:
            self.suggestions = []


@dataclass
class ConfigMetadata:
    """Metadata about configuration."""
    version: str
    created_at: str
    modified_at: str
    created_by: str
    description: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class IConfigurationSchema(ABC):
    """Interface for configuration schema definition and validation."""
    
    @property
    @abstractmethod
    def schema_version(self) -> str:
        """Configuration schema version."""
        pass
    
    @property
    @abstractmethod
    def supported_scopes(self) -> List[ConfigScope]:
        """Supported configuration scopes."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any], scope: ConfigScope) -> ConfigValidationResult:
        """Validate configuration against schema."""
        pass
    
    @abstractmethod
    def get_default_config(self, scope: ConfigScope) -> Dict[str, Any]:
        """Get default configuration for scope."""
        pass
    
    @abstractmethod
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration with override."""
        pass
    
    @abstractmethod
    def get_required_fields(self, scope: ConfigScope) -> List[str]:
        """Get required configuration fields for scope."""
        pass


class IConfigurationManager(ABC):
    """Interface for configuration management."""
    
    @property
    @abstractmethod
    def schema(self) -> IConfigurationSchema:
        """Configuration schema."""
        pass
    
    @abstractmethod
    async def load_config(
        self,
        config_path: Union[str, Path],
        config_format: Optional[ConfigFormat] = None,
        scope: ConfigScope = ConfigScope.GLOBAL
    ) -> Dict[str, Any]:
        """Load configuration from file."""
        pass
    
    @abstractmethod
    async def save_config(
        self,
        config: Dict[str, Any],
        config_path: Union[str, Path],
        config_format: ConfigFormat = ConfigFormat.YAML,
        metadata: Optional[ConfigMetadata] = None
    ) -> bool:
        """Save configuration to file."""
        pass
    
    @abstractmethod
    async def get_config(self, scope: ConfigScope, key: str) -> Any:
        """Get configuration value by scope and key."""
        pass
    
    @abstractmethod
    async def set_config(self, scope: ConfigScope, key: str, value: Any) -> bool:
        """Set configuration value."""
        pass
    
    @abstractmethod
    async def get_stage_config(self, stage_id: str) -> Dict[str, Any]:
        """Get complete configuration for stage."""
        pass
    
    @abstractmethod
    async def get_processor_config(self, stage_id: str, processor_id: str) -> Dict[str, Any]:
        """Get configuration for specific processor."""
        pass
    
    @abstractmethod
    async def create_config_template(self, template_type: str) -> Dict[str, Any]:
        """Create configuration template."""
        pass
    
    @abstractmethod
    async def validate_current_config(self) -> ConfigValidationResult:
        """Validate currently loaded configuration."""
        pass
    
    @abstractmethod
    async def reload_config(self) -> bool:
        """Reload configuration from sources."""
        pass


class IStageConfig(ABC):
    """Interface for stage-specific configuration."""
    
    @property
    @abstractmethod
    def stage_id(self) -> str:
        """Stage identifier."""
        pass
    
    @property
    @abstractmethod
    def stage_type(self) -> str:
        """Stage type identifier."""
        pass
    
    @abstractmethod
    def get_processor_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all processor configurations for this stage."""
        pass
    
    @abstractmethod
    def get_stage_parameters(self) -> Dict[str, Any]:
        """Get stage-level parameters."""
        pass
    
    @abstractmethod
    def get_resource_limits(self) -> Dict[str, Any]:
        """Get resource allocation limits."""
        pass
    
    @abstractmethod
    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance optimization settings."""
        pass
    
    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if stage is enabled."""
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Get list of stage dependencies."""
        pass


class IEnvironmentConfigManager(ABC):
    """Interface for environment-specific configuration management."""
    
    @abstractmethod
    async def detect_environment(self) -> str:
        """Detect current execution environment."""
        pass
    
    @abstractmethod
    async def load_environment_config(self, environment: str) -> Dict[str, Any]:
        """Load configuration for specific environment."""
        pass
    
    @abstractmethod
    async def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware-specific configuration."""
        pass
    
    @abstractmethod
    async def optimize_for_m3_max(self) -> Dict[str, Any]:
        """Get M3 Max optimized configuration."""
        pass
    
    @abstractmethod
    async def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get available model configurations."""
        pass
    
    @abstractmethod
    async def setup_local_models(self) -> bool:
        """Setup local model configurations."""
        pass


class IConfigurationWatcher(ABC):
    """Interface for configuration change monitoring."""
    
    @abstractmethod
    async def watch_config_changes(
        self,
        config_paths: List[Union[str, Path]],
        callback: callable
    ) -> str:
        """Watch for configuration file changes."""
        pass
    
    @abstractmethod
    async def stop_watching(self, watch_id: str) -> bool:
        """Stop watching configuration changes."""
        pass
    
    @abstractmethod
    async def get_config_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        pass
    
    @abstractmethod
    async def rollback_config(self, version_id: str) -> bool:
        """Rollback to previous configuration version."""
        pass