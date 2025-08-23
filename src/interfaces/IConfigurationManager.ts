/**
 * Configuration Management Interface
 * 
 * Provides comprehensive configuration management capabilities for RAN nodes,
 * including parameter validation, template management, deployment, and compliance checking.
 */

import { NodeConfig, CellConfig, ValidationResult } from './IRANNodeFactory';
import { IRANNode } from './IRANNode';

/**
 * Configuration Manager Interface
 * 
 * Manages all aspects of RAN configuration including templates, validation,
 * deployment, versioning, and compliance monitoring.
 */
export interface IConfigurationManager {
  /**
   * Load configuration template by ID
   * 
   * @param templateId Template identifier
   * @returns Configuration template
   * @throws {TemplateNotFoundException} When template is not found
   */
  loadTemplate(templateId: string): Promise<ConfigTemplate>;

  /**
   * Create new configuration template
   * 
   * @param template Template definition
   * @returns Created template with assigned ID
   * @throws {TemplateCreationException} When template cannot be created
   */
  createTemplate(template: Omit<ConfigTemplate, 'templateId' | 'createdAt' | 'updatedAt'>): Promise<ConfigTemplate>;

  /**
   * Update existing configuration template
   * 
   * @param templateId Template identifier
   * @param updates Partial template updates
   * @returns Updated template
   * @throws {TemplateNotFoundException} When template is not found
   * @throws {TemplateUpdateException} When template cannot be updated
   */
  updateTemplate(templateId: string, updates: Partial<ConfigTemplate>): Promise<ConfigTemplate>;

  /**
   * Delete configuration template
   * 
   * @param templateId Template identifier
   * @throws {TemplateNotFoundException} When template is not found
   * @throws {TemplateInUseException} When template is currently in use
   */
  deleteTemplate(templateId: string): Promise<void>;

  /**
   * List available configuration templates
   * 
   * @param filters Optional filters
   * @returns Array of matching templates
   */
  listTemplates(filters?: TemplateFilters): Promise<ConfigTemplate[]>;

  /**
   * Validate configuration against schema and business rules
   * 
   * @param config Configuration to validate
   * @param context Validation context
   * @returns Validation result with errors and warnings
   */
  validateConfig(config: any, context?: ValidationContext): Promise<ValidationResult>;

  /**
   * Validate configuration template
   * 
   * @param template Template to validate
   * @returns Validation result
   */
  validateTemplate(template: ConfigTemplate): Promise<ValidationResult>;

  /**
   * Apply configuration to a RAN node
   * 
   * @param node Target RAN node
   * @param config Configuration to apply
   * @param options Deployment options
   * @returns Deployment result
   * @throws {ConfigurationDeploymentException} When deployment fails
   */
  applyConfig(
    node: IRANNode,
    config: NodeConfig | CellConfig,
    options?: DeploymentOptions
  ): Promise<DeploymentResult>;

  /**
   * Apply configuration to multiple nodes
   * 
   * @param nodes Target RAN nodes
   * @param configs Configurations to apply (one per node)
   * @param options Deployment options
   * @returns Array of deployment results
   */
  applyConfigBatch(
    nodes: IRANNode[],
    configs: (NodeConfig | CellConfig)[],
    options?: DeploymentOptions
  ): Promise<DeploymentResult[]>;

  /**
   * Get parameter definitions for a node type
   * 
   * @param nodeType Node type
   * @param technology RAN technology
   * @returns Array of parameter definitions
   */
  getParameterDefinitions(
    nodeType?: string,
    technology?: 'LTE' | 'NR' | 'LTE-NR'
  ): Promise<ParameterDefinition[]>;

  /**
   * Get specific parameter definition
   * 
   * @param parameterName Parameter name
   * @param nodeType Node type
   * @param technology RAN technology
   * @returns Parameter definition or null if not found
   */
  getParameterDefinition(
    parameterName: string,
    nodeType?: string,
    technology?: 'LTE' | 'NR' | 'LTE-NR'
  ): Promise<ParameterDefinition | null>;

  /**
   * Create configuration from template
   * 
   * @param templateId Template identifier
   * @param substitutions Parameter substitutions
   * @returns Generated configuration
   * @throws {TemplateNotFoundException} When template is not found
   * @throws {TemplateRenderingException} When template rendering fails
   */
  createConfigFromTemplate(
    templateId: string,
    substitutions: Record<string, any>
  ): Promise<NodeConfig | CellConfig>;

  /**
   * Compare two configurations
   * 
   * @param config1 First configuration
   * @param config2 Second configuration
   * @returns Configuration differences
   */
  compareConfigs(
    config1: NodeConfig | CellConfig,
    config2: NodeConfig | CellConfig
  ): Promise<ConfigurationDiff>;

  /**
   * Merge configurations
   * 
   * @param baseConfig Base configuration
   * @param overrideConfig Override configuration
   * @param strategy Merge strategy
   * @returns Merged configuration
   */
  mergeConfigs(
    baseConfig: NodeConfig | CellConfig,
    overrideConfig: Partial<NodeConfig | CellConfig>,
    strategy?: MergeStrategy
  ): Promise<NodeConfig | CellConfig>;

  /**
   * Get configuration version history
   * 
   * @param nodeId Node identifier
   * @returns Array of configuration versions
   */
  getConfigHistory(nodeId: string): Promise<ConfigVersion[]>;

  /**
   * Get specific configuration version
   * 
   * @param nodeId Node identifier
   * @param version Version identifier
   * @returns Configuration version or null if not found
   */
  getConfigVersion(nodeId: string, version: string): Promise<ConfigVersion | null>;

  /**
   * Save configuration version
   * 
   * @param nodeId Node identifier
   * @param config Configuration to save
   * @param comment Version comment
   * @returns Saved configuration version
   */
  saveConfigVersion(
    nodeId: string,
    config: NodeConfig | CellConfig,
    comment?: string
  ): Promise<ConfigVersion>;

  /**
   * Rollback to previous configuration version
   * 
   * @param nodeId Node identifier
   * @param version Version to rollback to
   * @param options Rollback options
   * @returns Rollback result
   */
  rollbackToVersion(
    nodeId: string,
    version: string,
    options?: RollbackOptions
  ): Promise<RollbackResult>;

  /**
   * Check configuration compliance
   * 
   * @param config Configuration to check
   * @param policies Compliance policies
   * @returns Compliance check result
   */
  checkCompliance(
    config: NodeConfig | CellConfig,
    policies?: CompliancePolicy[]
  ): Promise<ComplianceResult>;

  /**
   * Get default compliance policies
   * 
   * @param nodeType Node type
   * @param technology RAN technology
   * @returns Array of default compliance policies
   */
  getDefaultCompliancePolicies(
    nodeType?: string,
    technology?: 'LTE' | 'NR' | 'LTE-NR'
  ): Promise<CompliancePolicy[]>;

  /**
   * Create compliance policy
   * 
   * @param policy Policy definition
   * @returns Created policy
   */
  createCompliancePolicy(
    policy: Omit<CompliancePolicy, 'policyId' | 'createdAt' | 'updatedAt'>
  ): Promise<CompliancePolicy>;

  /**
   * Export configuration
   * 
   * @param nodeId Node identifier
   * @param format Export format
   * @returns Exported configuration data
   */
  exportConfig(nodeId: string, format: ExportFormat): Promise<ExportResult>;

  /**
   * Import configuration
   * 
   * @param data Configuration data to import
   * @param format Import format
   * @param options Import options
   * @returns Import result
   */
  importConfig(
    data: string | Buffer,
    format: ImportFormat,
    options?: ImportOptions
  ): Promise<ImportResult>;

  /**
   * Schedule configuration deployment
   * 
   * @param nodeId Node identifier
   * @param config Configuration to deploy
   * @param scheduledTime Scheduled deployment time
   * @param options Deployment options
   * @returns Scheduled deployment ID
   */
  scheduleConfigDeployment(
    nodeId: string,
    config: NodeConfig | CellConfig,
    scheduledTime: Date,
    options?: DeploymentOptions
  ): Promise<string>;

  /**
   * Cancel scheduled deployment
   * 
   * @param deploymentId Deployment identifier
   */
  cancelScheduledDeployment(deploymentId: string): Promise<void>;

  /**
   * Get deployment status
   * 
   * @param deploymentId Deployment identifier
   * @returns Deployment status
   */
  getDeploymentStatus(deploymentId: string): Promise<DeploymentStatus>;

  /**
   * Subscribe to configuration change events
   * 
   * @param callback Event callback
   * @param filters Optional event filters
   * @returns Subscription ID
   */
  subscribeToConfigChanges(
    callback: (event: ConfigChangeEvent) => void,
    filters?: ConfigChangeFilters
  ): string;

  /**
   * Unsubscribe from configuration change events
   * 
   * @param subscriptionId Subscription ID
   */
  unsubscribeFromConfigChanges(subscriptionId: string): void;
}

/**
 * Configuration template
 */
export interface ConfigTemplate {
  /** Template identifier */
  templateId: string;
  /** Template name */
  name: string;
  /** Template description */
  description: string;
  /** Template version */
  version: string;
  /** Node type this template applies to */
  nodeType: string;
  /** RAN technology */
  technology: 'LTE' | 'NR' | 'LTE-NR';
  /** Template category */
  category: TemplateCategory;
  /** Template content (with placeholders) */
  content: Record<string, any>;
  /** Template parameters */
  parameters: TemplateParameter[];
  /** Validation schema */
  validationSchema?: Record<string, any>;
  /** Template tags */
  tags: string[];
  /** Template author */
  author: string;
  /** Creation timestamp */
  createdAt: Date;
  /** Last update timestamp */
  updatedAt: Date;
}

/**
 * Template category enumeration
 */
export enum TemplateCategory {
  BASIC = 'BASIC',
  ADVANCED = 'ADVANCED',
  OPTIMIZATION = 'OPTIMIZATION',
  MAINTENANCE = 'MAINTENANCE',
  TESTING = 'TESTING',
  CUSTOM = 'CUSTOM'
}

/**
 * Template parameter
 */
export interface TemplateParameter {
  /** Parameter name */
  name: string;
  /** Parameter type */
  type: ParameterType;
  /** Parameter description */
  description: string;
  /** Default value */
  defaultValue?: any;
  /** Required parameter */
  required: boolean;
  /** Validation rules */
  validationRules?: ValidationRule[];
  /** Parameter group */
  group?: string;
}

/**
 * Parameter type enumeration
 */
export enum ParameterType {
  STRING = 'STRING',
  INTEGER = 'INTEGER',
  FLOAT = 'FLOAT',
  BOOLEAN = 'BOOLEAN',
  ENUM = 'ENUM',
  ARRAY = 'ARRAY',
  OBJECT = 'OBJECT',
  IP_ADDRESS = 'IP_ADDRESS',
  MAC_ADDRESS = 'MAC_ADDRESS',
  FREQUENCY = 'FREQUENCY',
  POWER = 'POWER',
  PERCENTAGE = 'PERCENTAGE'
}

/**
 * Template filters
 */
export interface TemplateFilters {
  /** Filter by node type */
  nodeType?: string;
  /** Filter by technology */
  technology?: 'LTE' | 'NR' | 'LTE-NR';
  /** Filter by category */
  category?: TemplateCategory;
  /** Filter by tags */
  tags?: string[];
  /** Filter by author */
  author?: string;
  /** Text search in name/description */
  search?: string;
}

/**
 * Validation context
 */
export interface ValidationContext {
  /** Node type being configured */
  nodeType?: string;
  /** RAN technology */
  technology?: 'LTE' | 'NR' | 'LTE-NR';
  /** Existing configuration (for updates) */
  existingConfig?: NodeConfig | CellConfig;
  /** Deployment environment */
  environment?: 'PRODUCTION' | 'STAGING' | 'DEVELOPMENT';
  /** Additional context data */
  contextData?: Record<string, any>;
}

/**
 * Parameter definition
 */
export interface ParameterDefinition {
  /** Parameter name */
  name: string;
  /** Display name */
  displayName: string;
  /** Parameter description */
  description: string;
  /** Parameter type */
  type: ParameterType;
  /** Data type for validation */
  dataType: string;
  /** Unit of measurement */
  unit?: string;
  /** Minimum value (for numeric types) */
  minValue?: number;
  /** Maximum value (for numeric types) */
  maxValue?: number;
  /** Allowed values (for enum types) */
  allowedValues?: any[];
  /** Default value */
  defaultValue?: any;
  /** Whether parameter is mandatory */
  mandatory: boolean;
  /** Whether parameter is read-only */
  readOnly: boolean;
  /** Parameter group/category */
  category: string;
  /** Validation rules */
  validationRules: ValidationRule[];
  /** Dependencies on other parameters */
  dependencies: ParameterDependency[];
  /** Impact level of parameter changes */
  impactLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  /** Supported node types */
  supportedNodeTypes: string[];
  /** Supported technologies */
  supportedTechnologies: ('LTE' | 'NR' | 'LTE-NR')[];
}

/**
 * Validation rule
 */
export interface ValidationRule {
  /** Rule type */
  type: ValidationRuleType;
  /** Rule parameters */
  parameters: Record<string, any>;
  /** Error message */
  errorMessage: string;
  /** Warning message */
  warningMessage?: string;
  /** Rule severity */
  severity: 'ERROR' | 'WARNING';
}

/**
 * Validation rule type enumeration
 */
export enum ValidationRuleType {
  REQUIRED = 'REQUIRED',
  RANGE = 'RANGE',
  PATTERN = 'PATTERN',
  LENGTH = 'LENGTH',
  UNIQUE = 'UNIQUE',
  DEPENDENCY = 'DEPENDENCY',
  CUSTOM = 'CUSTOM'
}

/**
 * Parameter dependency
 */
export interface ParameterDependency {
  /** Dependent parameter name */
  dependentParameter: string;
  /** Dependency type */
  dependencyType: 'REQUIRES' | 'CONFLICTS_WITH' | 'IMPLIES';
  /** Condition for dependency */
  condition?: DependencyCondition;
}

/**
 * Dependency condition
 */
export interface DependencyCondition {
  /** Operator */
  operator: 'EQUALS' | 'NOT_EQUALS' | 'GREATER_THAN' | 'LESS_THAN' | 'IN' | 'NOT_IN';
  /** Value to compare against */
  value: any;
}

/**
 * Deployment options
 */
export interface DeploymentOptions {
  /** Dry run (validate only, don't deploy) */
  dryRun?: boolean;
  /** Force deployment even if warnings exist */
  force?: boolean;
  /** Backup current configuration before deployment */
  backup?: boolean;
  /** Rollback timeout in minutes */
  rollbackTimeout?: number;
  /** Deployment strategy */
  strategy?: DeploymentStrategy;
  /** Notification settings */
  notifications?: NotificationSettings;
  /** Custom deployment scripts */
  customScripts?: DeploymentScript[];
}

/**
 * Deployment strategy enumeration
 */
export enum DeploymentStrategy {
  IMMEDIATE = 'IMMEDIATE',
  STAGED = 'STAGED',
  CANARY = 'CANARY',
  BLUE_GREEN = 'BLUE_GREEN'
}

/**
 * Notification settings
 */
export interface NotificationSettings {
  /** Enable email notifications */
  email?: boolean;
  /** Email recipients */
  emailRecipients?: string[];
  /** Enable SMS notifications */
  sms?: boolean;
  /** SMS recipients */
  smsRecipients?: string[];
  /** Webhook URLs */
  webhooks?: string[];
}

/**
 * Deployment script
 */
export interface DeploymentScript {
  /** Script name */
  name: string;
  /** Execution phase */
  phase: 'PRE_DEPLOYMENT' | 'POST_DEPLOYMENT' | 'ON_FAILURE';
  /** Script content */
  script: string;
  /** Script timeout in seconds */
  timeout: number;
}

/**
 * Deployment result
 */
export interface DeploymentResult {
  /** Success indicator */
  success: boolean;
  /** Deployment ID */
  deploymentId: string;
  /** Start time */
  startTime: Date;
  /** End time */
  endTime: Date;
  /** Deployed parameters count */
  deployedParameters: number;
  /** Failed parameters */
  failedParameters: ParameterDeploymentResult[];
  /** Configuration backup ID */
  backupId?: string;
  /** Rollback performed */
  rollbackPerformed: boolean;
  /** Deployment messages */
  messages: DeploymentMessage[];
}

/**
 * Parameter deployment result
 */
export interface ParameterDeploymentResult {
  /** Parameter name */
  parameterName: string;
  /** Success indicator */
  success: boolean;
  /** Old value */
  oldValue: any;
  /** New value */
  newValue: any;
  /** Error message */
  errorMessage?: string;
}

/**
 * Deployment message
 */
export interface DeploymentMessage {
  /** Message level */
  level: 'INFO' | 'WARNING' | 'ERROR';
  /** Message text */
  message: string;
  /** Message timestamp */
  timestamp: Date;
  /** Related parameter */
  parameter?: string;
}

/**
 * Configuration difference
 */
export interface ConfigurationDiff {
  /** Added parameters */
  added: ParameterChange[];
  /** Modified parameters */
  modified: ParameterChange[];
  /** Removed parameters */
  removed: ParameterChange[];
  /** Summary statistics */
  summary: DiffSummary;
}

/**
 * Parameter change
 */
export interface ParameterChange {
  /** Parameter path */
  path: string;
  /** Parameter name */
  name: string;
  /** Old value */
  oldValue: any;
  /** New value */
  newValue: any;
  /** Change type */
  changeType: 'ADDED' | 'MODIFIED' | 'REMOVED';
}

/**
 * Diff summary
 */
export interface DiffSummary {
  /** Total changes */
  totalChanges: number;
  /** Added count */
  addedCount: number;
  /** Modified count */
  modifiedCount: number;
  /** Removed count */
  removedCount: number;
}

/**
 * Merge strategy enumeration
 */
export enum MergeStrategy {
  OVERRIDE = 'OVERRIDE',
  MERGE_DEEP = 'MERGE_DEEP',
  MERGE_SHALLOW = 'MERGE_SHALLOW',
  ADDITIVE_ONLY = 'ADDITIVE_ONLY'
}

/**
 * Configuration version
 */
export interface ConfigVersion {
  /** Version identifier */
  versionId: string;
  /** Node identifier */
  nodeId: string;
  /** Version number */
  versionNumber: number;
  /** Configuration data */
  configuration: NodeConfig | CellConfig;
  /** Version comment */
  comment?: string;
  /** Creation timestamp */
  createdAt: Date;
  /** Created by user */
  createdBy: string;
  /** Configuration checksum */
  checksum: string;
  /** Tags */
  tags: string[];
}

/**
 * Rollback options
 */
export interface RollbackOptions {
  /** Force rollback even if target version has issues */
  force?: boolean;
  /** Backup current configuration before rollback */
  backup?: boolean;
  /** Rollback comment */
  comment?: string;
  /** Notification settings */
  notifications?: NotificationSettings;
}

/**
 * Rollback result
 */
export interface RollbackResult {
  /** Success indicator */
  success: boolean;
  /** Rollback ID */
  rollbackId: string;
  /** Source version */
  sourceVersion: string;
  /** Target version */
  targetVersion: string;
  /** Start time */
  startTime: Date;
  /** End time */
  endTime: Date;
  /** Rollback messages */
  messages: DeploymentMessage[];
}

/**
 * Compliance policy
 */
export interface CompliancePolicy {
  /** Policy identifier */
  policyId: string;
  /** Policy name */
  name: string;
  /** Policy description */
  description: string;
  /** Policy category */
  category: string;
  /** Policy rules */
  rules: ComplianceRule[];
  /** Severity level */
  severity: 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';
  /** Applicable node types */
  applicableNodeTypes: string[];
  /** Applicable technologies */
  applicableTechnologies: ('LTE' | 'NR' | 'LTE-NR')[];
  /** Policy author */
  author: string;
  /** Creation timestamp */
  createdAt: Date;
  /** Last update timestamp */
  updatedAt: Date;
}

/**
 * Compliance rule
 */
export interface ComplianceRule {
  /** Rule identifier */
  ruleId: string;
  /** Rule name */
  name: string;
  /** Rule expression */
  expression: string;
  /** Rule message */
  message: string;
  /** Rule parameters */
  parameters?: Record<string, any>;
}

/**
 * Compliance result
 */
export interface ComplianceResult {
  /** Overall compliance status */
  compliant: boolean;
  /** Compliance score (0-100) */
  score: number;
  /** Policy results */
  policyResults: PolicyResult[];
  /** Compliance summary */
  summary: ComplianceSummary;
  /** Check timestamp */
  checkedAt: Date;
}

/**
 * Policy result
 */
export interface PolicyResult {
  /** Policy ID */
  policyId: string;
  /** Policy name */
  policyName: string;
  /** Compliance status for this policy */
  compliant: boolean;
  /** Rule results */
  ruleResults: RuleResult[];
  /** Policy severity */
  severity: 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';
}

/**
 * Rule result
 */
export interface RuleResult {
  /** Rule ID */
  ruleId: string;
  /** Rule name */
  ruleName: string;
  /** Rule passed */
  passed: boolean;
  /** Failure message */
  failureMessage?: string;
  /** Affected parameters */
  affectedParameters: string[];
}

/**
 * Compliance summary
 */
export interface ComplianceSummary {
  /** Total policies checked */
  totalPolicies: number;
  /** Passed policies */
  passedPolicies: number;
  /** Failed policies */
  failedPolicies: number;
  /** Total rules checked */
  totalRules: number;
  /** Passed rules */
  passedRules: number;
  /** Failed rules */
  failedRules: number;
}

/**
 * Export format enumeration
 */
export enum ExportFormat {
  JSON = 'JSON',
  XML = 'XML',
  YAML = 'YAML',
  CSV = 'CSV',
  EXCEL = 'EXCEL'
}

/**
 * Import format enumeration
 */
export enum ImportFormat {
  JSON = 'JSON',
  XML = 'XML',
  YAML = 'YAML',
  CSV = 'CSV',
  EXCEL = 'EXCEL'
}

/**
 * Export result
 */
export interface ExportResult {
  /** Export format */
  format: ExportFormat;
  /** Exported data */
  data: string | Buffer;
  /** Data size in bytes */
  size: number;
  /** Export timestamp */
  timestamp: Date;
  /** Metadata */
  metadata: Record<string, any>;
}

/**
 * Import options
 */
export interface ImportOptions {
  /** Validate before import */
  validate?: boolean;
  /** Merge strategy for conflicts */
  mergeStrategy?: MergeStrategy;
  /** Backup current configuration */
  backup?: boolean;
  /** Import comment */
  comment?: string;
}

/**
 * Import result
 */
export interface ImportResult {
  /** Success indicator */
  success: boolean;
  /** Import ID */
  importId: string;
  /** Imported parameters count */
  importedParameters: number;
  /** Failed parameters */
  failedParameters: ParameterImportResult[];
  /** Validation errors */
  validationErrors: ValidationError[];
  /** Import messages */
  messages: ImportMessage[];
  /** Backup ID if created */
  backupId?: string;
}

/**
 * Parameter import result
 */
export interface ParameterImportResult {
  /** Parameter name */
  parameterName: string;
  /** Success indicator */
  success: boolean;
  /** Imported value */
  importedValue: any;
  /** Error message */
  errorMessage?: string;
}

/**
 * Import message
 */
export interface ImportMessage {
  /** Message level */
  level: 'INFO' | 'WARNING' | 'ERROR';
  /** Message text */
  message: string;
  /** Line number (for file-based imports) */
  lineNumber?: number;
  /** Column number (for file-based imports) */
  columnNumber?: number;
}

/**
 * Deployment status
 */
export interface DeploymentStatus {
  /** Deployment ID */
  deploymentId: string;
  /** Current status */
  status: DeploymentStatusType;
  /** Status description */
  statusDescription: string;
  /** Progress percentage */
  progress: number;
  /** Start time */
  startTime?: Date;
  /** Estimated completion time */
  estimatedCompletionTime?: Date;
  /** Current phase */
  currentPhase?: string;
  /** Status messages */
  messages: DeploymentMessage[];
}

/**
 * Deployment status type enumeration
 */
export enum DeploymentStatusType {
  SCHEDULED = 'SCHEDULED',
  IN_PROGRESS = 'IN_PROGRESS',
  COMPLETED = 'COMPLETED',
  FAILED = 'FAILED',
  CANCELLED = 'CANCELLED',
  ROLLED_BACK = 'ROLLED_BACK'
}

/**
 * Configuration change event
 */
export interface ConfigChangeEvent {
  /** Event ID */
  eventId: string;
  /** Event type */
  eventType: ConfigChangeEventType;
  /** Node ID */
  nodeId: string;
  /** Changed parameters */
  changedParameters: ParameterChange[];
  /** User who made the change */
  changedBy: string;
  /** Change timestamp */
  timestamp: Date;
  /** Change comment */
  comment?: string;
  /** Deployment ID */
  deploymentId?: string;
}

/**
 * Configuration change event type enumeration
 */
export enum ConfigChangeEventType {
  PARAMETER_CHANGED = 'PARAMETER_CHANGED',
  TEMPLATE_APPLIED = 'TEMPLATE_APPLIED',
  CONFIG_IMPORTED = 'CONFIG_IMPORTED',
  CONFIG_ROLLED_BACK = 'CONFIG_ROLLED_BACK',
  COMPLIANCE_VIOLATION = 'COMPLIANCE_VIOLATION'
}

/**
 * Configuration change filters
 */
export interface ConfigChangeFilters {
  /** Filter by node IDs */
  nodeIds?: string[];
  /** Filter by event types */
  eventTypes?: ConfigChangeEventType[];
  /** Filter by user */
  changedBy?: string;
  /** Filter by time range */
  timeRange?: {
    startTime: Date;
    endTime: Date;
  };
  /** Filter by parameter names */
  parameters?: string[];
}