"""
Unified Pipeline Configuration System

Manages configuration loading, validation, and environment-specific settings
for the Python RAN LLM pipeline.
"""

import os
import yaml
import json
import toml
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from ..interfaces.configuration import (
    IConfigurationManager, IConfigurationSchema, IStageConfig, IEnvironmentConfigManager,
    ConfigScope, ConfigFormat, ConfigValidationResult, ConfigMetadata
)


@dataclass
class StageConfigImpl:
    """Implementation of stage configuration."""
    stage_id: str
    stage_type: str
    enabled: bool = True
    processors: Dict[str, Dict[str, Any]] = None
    stage_parameters: Dict[str, Any] = None
    resource_limits: Dict[str, Any] = None
    performance_settings: Dict[str, Any] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.processors is None:
            self.processors = {}
        if self.stage_parameters is None:
            self.stage_parameters = {}
        if self.resource_limits is None:
            self.resource_limits = {}
        if self.performance_settings is None:
            self.performance_settings = {}
        if self.dependencies is None:
            self.dependencies = []


class PipelineConfigurationSchema(IConfigurationSchema):
    """Schema validator for pipeline configuration."""
    
    def __init__(self):
        self._schema_version = "1.0.0"
        self._logger = logging.getLogger(__name__)
        
        # Define required fields for each scope
        self._required_fields = {
            ConfigScope.GLOBAL: ['pipeline', 'stages'],
            ConfigScope.PIPELINE: ['mode', 'max_parallel_stages'],
            ConfigScope.STAGE: ['stage_id', 'stage_type'],
            ConfigScope.PROCESSOR: ['type']
        }
        
        # Define default configurations
        self._defaults = {
            ConfigScope.GLOBAL: {
                'version': self._schema_version,
                'environment': 'development',
                'logging': {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                }
            },
            ConfigScope.PIPELINE: {
                'mode': 'adaptive',
                'max_parallel_stages': 4,
                'batch_size': 100,
                'timeout_seconds': 3600,
                'memory_limit_gb': 100.0,
                'enable_checkpointing': True
            },
            ConfigScope.STAGE: {
                'enabled': True,
                'max_workers': 8,
                'batch_size': 50,
                'memory_limit_gb': 20.0,
                'timeout_seconds': 600,
                'retry_attempts': 2
            },
            ConfigScope.PROCESSOR: {
                'enabled': True,
                'timeout_seconds': 300,
                'retry_attempts': 1
            }
        }
    
    @property
    def schema_version(self) -> str:
        return self._schema_version
    
    @property
    def supported_scopes(self) -> List[ConfigScope]:
        return list(ConfigScope)
    
    def validate_config(self, config: Dict[str, Any], scope: ConfigScope) -> ConfigValidationResult:
        """Validate configuration against schema."""
        errors = []
        warnings = []
        suggestions = []
        
        # Check required fields
        required_fields = self._required_fields.get(scope, [])
        for field in required_fields:
            if field not in config:
                errors.append(f"Required field '{field}' is missing")
        
        # Scope-specific validation
        if scope == ConfigScope.PIPELINE:
            errors.extend(self._validate_pipeline_config(config))
        elif scope == ConfigScope.STAGE:
            errors.extend(self._validate_stage_config(config))
        elif scope == ConfigScope.PROCESSOR:
            errors.extend(self._validate_processor_config(config))
        
        # Check for deprecated fields
        deprecated_fields = self._get_deprecated_fields(scope)
        for field in deprecated_fields:
            if field in config:
                warnings.append(f"Field '{field}' is deprecated")
                suggestions.append(f"Consider removing deprecated field '{field}'")
        
        # Memory and resource validation
        warnings.extend(self._validate_resource_limits(config, scope))
        
        return ConfigValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def get_default_config(self, scope: ConfigScope) -> Dict[str, Any]:
        """Get default configuration for scope."""
        return self._defaults.get(scope, {}).copy()
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration with override."""
        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return deep_merge(base_config, override_config)
    
    def get_required_fields(self, scope: ConfigScope) -> List[str]:
        """Get required configuration fields for scope."""
        return self._required_fields.get(scope, []).copy()
    
    def _validate_pipeline_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate pipeline-specific configuration."""
        errors = []
        
        # Validate mode
        valid_modes = ['sequential', 'parallel', 'adaptive', 'streaming']
        if 'mode' in config and config['mode'] not in valid_modes:
            errors.append(f"Invalid mode '{config['mode']}'. Must be one of: {valid_modes}")
        
        # Validate numeric fields
        numeric_fields = {
            'max_parallel_stages': (1, 16),
            'batch_size': (1, 10000),
            'timeout_seconds': (60, 86400),
            'memory_limit_gb': (1.0, 128.0)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in config:
                value = config[field]
                if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                    errors.append(f"Field '{field}' must be between {min_val} and {max_val}")
        
        return errors
    
    def _validate_stage_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate stage-specific configuration."""
        errors = []
        
        # Validate stage_type
        valid_stage_types = [
            'raw_input', 'document_conversion', 'preprocessing',
            'langextract', 'conversation_generation', 'dataset_finalization'
        ]
        if 'stage_type' in config and config['stage_type'] not in valid_stage_types:
            errors.append(f"Invalid stage_type '{config['stage_type']}'. Must be one of: {valid_stage_types}")
        
        # Validate stage_id format
        if 'stage_id' in config:
            stage_id = config['stage_id']
            if not isinstance(stage_id, str) or len(stage_id) < 3:
                errors.append("stage_id must be a string with at least 3 characters")
        
        return errors
    
    def _validate_processor_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate processor-specific configuration."""
        errors = []
        
        # Validate processor type
        if 'type' in config:
            processor_type = config['type']
            if not isinstance(processor_type, str) or len(processor_type) < 2:
                errors.append("Processor type must be a non-empty string")
        
        return errors
    
    def _get_deprecated_fields(self, scope: ConfigScope) -> List[str]:
        """Get list of deprecated fields for scope."""
        deprecated_fields = {
            ConfigScope.PIPELINE: ['legacy_mode', 'old_timeout'],
            ConfigScope.STAGE: ['legacy_workers'],
            ConfigScope.PROCESSOR: ['old_config']
        }
        return deprecated_fields.get(scope, [])
    
    def _validate_resource_limits(self, config: Dict[str, Any], scope: ConfigScope) -> List[str]:
        """Validate resource limits and provide warnings."""
        warnings = []
        
        # Memory warnings for M3 Max
        if scope == ConfigScope.PIPELINE:
            memory_limit = config.get('memory_limit_gb', 0)
            if memory_limit > 120:
                warnings.append("Memory limit exceeds recommended 120GB for M3 Max")
            elif memory_limit > 100:
                warnings.append("High memory limit may affect system stability")
        
        # CPU core warnings
        max_workers = config.get('max_workers', 0)
        if max_workers > 12:
            warnings.append("max_workers exceeds M3 Max core count (8P + 4E)")
        
        return warnings


class PipelineConfigurationManager(IConfigurationManager):
    """Main configuration manager for the pipeline."""
    
    def __init__(self, base_config_path: Optional[Union[str, Path]] = None):
        self._schema = PipelineConfigurationSchema()
        self._logger = logging.getLogger(__name__)
        self._config_cache: Dict[str, Any] = {}
        self._config_metadata: Dict[str, ConfigMetadata] = {}
        
        # Set base configuration path
        if base_config_path:
            self._base_config_path = Path(base_config_path)
        else:
            self._base_config_path = Path.cwd() / "config" / "pipeline_config.yaml"
        
        # Load base configuration if it exists
        if self._base_config_path.exists():
            try:
                self._load_base_config()
            except Exception as e:
                self._logger.error(f"Failed to load base configuration: {e}")
    
    @property
    def schema(self) -> IConfigurationSchema:
        return self._schema
    
    async def load_config(self,
                         config_path: Union[str, Path],
                         config_format: Optional[ConfigFormat] = None,
                         scope: ConfigScope = ConfigScope.GLOBAL) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine format if not provided
        if config_format is None:
            config_format = self._detect_config_format(config_path)
        
        # Load configuration based on format
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_format == ConfigFormat.YAML:
                    config = yaml.safe_load(f)
                elif config_format == ConfigFormat.JSON:
                    config = json.load(f)
                elif config_format == ConfigFormat.TOML:
                    config = toml.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_format}")
            
            # Validate configuration
            validation_result = self._schema.validate_config(config, scope)
            if not validation_result.is_valid:
                self._logger.error(f"Configuration validation failed: {validation_result.errors}")
                raise ValueError(f"Invalid configuration: {validation_result.errors}")
            
            # Log warnings if any
            for warning in validation_result.warnings:
                self._logger.warning(warning)
            
            # Cache configuration
            cache_key = f"{scope.value}_{config_path.stem}"
            self._config_cache[cache_key] = config
            
            # Store metadata
            stat = config_path.stat()
            self._config_metadata[cache_key] = ConfigMetadata(
                version=config.get('version', '1.0.0'),
                created_at=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                created_by=os.getenv('USER', 'unknown'),
                description=config.get('description', f"Configuration loaded from {config_path}")
            )
            
            self._logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            self._logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    async def save_config(self,
                         config: Dict[str, Any],
                         config_path: Union[str, Path],
                         config_format: ConfigFormat = ConfigFormat.YAML,
                         metadata: Optional[ConfigMetadata] = None) -> bool:
        """Save configuration to file."""
        config_path = Path(config_path)
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata to config if provided
        if metadata:
            config['_metadata'] = asdict(metadata)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_format == ConfigFormat.YAML:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                elif config_format == ConfigFormat.JSON:
                    json.dump(config, f, indent=2)
                elif config_format == ConfigFormat.TOML:
                    toml.dump(config, f)
                else:
                    raise ValueError(f"Unsupported config format: {config_format}")
            
            self._logger.info(f"Saved configuration to {config_path}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to save configuration to {config_path}: {e}")
            return False
    
    async def get_config(self, scope: ConfigScope, key: str) -> Any:
        """Get configuration value by scope and key."""
        cache_key = f"{scope.value}_config"
        
        if cache_key not in self._config_cache:
            # Load default configuration for scope
            self._config_cache[cache_key] = self._schema.get_default_config(scope)
        
        config = self._config_cache[cache_key]
        return self._get_nested_value(config, key)
    
    async def set_config(self, scope: ConfigScope, key: str, value: Any) -> bool:
        """Set configuration value."""
        cache_key = f"{scope.value}_config"
        
        if cache_key not in self._config_cache:
            self._config_cache[cache_key] = self._schema.get_default_config(scope)
        
        config = self._config_cache[cache_key]
        self._set_nested_value(config, key, value)
        
        return True
    
    async def get_stage_config(self, stage_id: str) -> Dict[str, Any]:
        """Get complete configuration for stage."""
        # Try to get from cache first
        cache_key = f"stage_{stage_id}"
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        # Load from global config or create default
        global_config = self._config_cache.get('global_config', {})
        stages_config = global_config.get('stages', [])
        
        # Find stage configuration
        stage_config = None
        for stage in stages_config:
            if stage.get('stage_id') == stage_id:
                stage_config = stage
                break
        
        if not stage_config:
            # Create default stage configuration
            stage_config = self._schema.get_default_config(ConfigScope.STAGE)
            stage_config['stage_id'] = stage_id
        
        self._config_cache[cache_key] = stage_config
        return stage_config
    
    async def get_processor_config(self, stage_id: str, processor_id: str) -> Dict[str, Any]:
        """Get configuration for specific processor."""
        stage_config = await self.get_stage_config(stage_id)
        processors = stage_config.get('processors', {})
        
        processor_config = processors.get(processor_id, {})
        if not processor_config:
            processor_config = self._schema.get_default_config(ConfigScope.PROCESSOR)
        
        return processor_config
    
    async def create_config_template(self, template_type: str) -> Dict[str, Any]:
        """Create configuration template."""
        templates = {
            'm3_max_optimized': self._create_m3_max_template(),
            'high_throughput': self._create_high_throughput_template(),
            'quality_focused': self._create_quality_focused_template(),
            'development': self._create_development_template()
        }
        
        template = templates.get(template_type)
        if not template:
            raise ValueError(f"Unknown template type: {template_type}")
        
        return template
    
    async def validate_current_config(self) -> ConfigValidationResult:
        """Validate currently loaded configuration."""
        global_config = self._config_cache.get('global_config', {})
        return self._schema.validate_config(global_config, ConfigScope.GLOBAL)
    
    async def reload_config(self) -> bool:
        """Reload configuration from sources."""
        try:
            # Clear cache
            self._config_cache.clear()
            self._config_metadata.clear()
            
            # Reload base configuration
            if self._base_config_path.exists():
                self._load_base_config()
            
            self._logger.info("Configuration reloaded successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def _load_base_config(self):
        """Load base configuration synchronously."""
        with open(self._base_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self._config_cache['global_config'] = config
        
        # Extract stage configurations
        for stage in config.get('stages', []):
            stage_id = stage.get('stage_id')
            if stage_id:
                self._config_cache[f"stage_{stage_id}"] = stage
    
    def _detect_config_format(self, config_path: Path) -> ConfigFormat:
        """Detect configuration format from file extension."""
        extension = config_path.suffix.lower()
        format_mapping = {
            '.yaml': ConfigFormat.YAML,
            '.yml': ConfigFormat.YAML,
            '.json': ConfigFormat.JSON,
            '.toml': ConfigFormat.TOML
        }
        
        return format_mapping.get(extension, ConfigFormat.YAML)
    
    def _get_nested_value(self, config: Dict[str, Any], key: str) -> Any:
        """Get nested configuration value using dot notation."""
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """Set nested configuration value using dot notation."""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _create_m3_max_template(self) -> Dict[str, Any]:
        """Create M3 Max optimized configuration template."""
        return {
            'version': '1.0.0',
            'environment': 'production',
            'description': 'M3 Max optimized pipeline configuration',
            'pipeline': {
                'mode': 'adaptive',
                'max_parallel_stages': 6,
                'batch_size': 100,
                'timeout_seconds': 3600,
                'memory_limit_gb': 100.0,
                'enable_checkpointing': True,
                'checkpoint_interval': 1000
            },
            'hardware': {
                'target_platform': 'm3_max',
                'enable_apple_silicon_acceleration': True,
                'enable_metal_performance_shaders': True,
                'enable_neural_engine': True,
                'unified_memory_gb': 128
            },
            'stages': [
                {
                    'stage_id': 'stage_1_raw_input',
                    'stage_type': 'raw_input',
                    'max_workers': 4,
                    'batch_size': 50,
                    'memory_limit_gb': 15.0
                },
                {
                    'stage_id': 'stage_2_conversion',
                    'stage_type': 'document_conversion',
                    'max_workers': 6,
                    'batch_size': 30,
                    'memory_limit_gb': 25.0
                },
                {
                    'stage_id': 'stage_4_langextract',
                    'stage_type': 'langextract',
                    'max_workers': 3,
                    'batch_size': 10,
                    'memory_limit_gb': 40.0,
                    'model_strategy': {
                        'primary_model': 'qwen3:7b',
                        'fallback_model': 'qwen3:1.7b',
                        'enable_circuit_breaker': True
                    }
                }
            ]
        }
    
    def _create_high_throughput_template(self) -> Dict[str, Any]:
        """Create high throughput configuration template."""
        return {
            'version': '1.0.0',
            'description': 'High throughput processing configuration',
            'pipeline': {
                'mode': 'parallel',
                'max_parallel_stages': 8,
                'batch_size': 200,
                'memory_limit_gb': 90.0
            }
        }
    
    def _create_quality_focused_template(self) -> Dict[str, Any]:
        """Create quality-focused configuration template."""
        return {
            'version': '1.0.0',
            'description': 'Quality-focused processing configuration',
            'pipeline': {
                'mode': 'sequential',
                'max_parallel_stages': 2,
                'batch_size': 25,
                'enable_quality_validation': True,
                'quality_threshold': 0.742
            }
        }
    
    def _create_development_template(self) -> Dict[str, Any]:
        """Create development configuration template."""
        return {
            'version': '1.0.0',
            'environment': 'development',
            'description': 'Development environment configuration',
            'pipeline': {
                'mode': 'sequential',
                'max_parallel_stages': 2,
                'batch_size': 10,
                'enable_debug': True
            },
            'logging': {
                'level': 'DEBUG',
                'enable_file_logging': True
            }
        }