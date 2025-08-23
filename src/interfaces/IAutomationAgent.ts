/**
 * Automation Agent Interface
 * 
 * Defines the contract for AI-powered automation agents that can execute
 * various network optimization and maintenance operations autonomously.
 */

/**
 * Automation Agent Interface
 * 
 * Represents an autonomous agent capable of executing network operations,
 * optimization tasks, and maintenance procedures with AI-driven decision making.
 */
export interface IAutomationAgent {
  /**
   * Get agent identifier
   */
  getId(): string;

  /**
   * Get agent name
   */
  getName(): string;

  /**
   * Get agent type
   */
  getType(): AgentType;

  /**
   * Get agent capabilities
   */
  getCapabilities(): AgentCapability[];

  /**
   * Get current agent status
   */
  getStatus(): Promise<AgentStatus>;

  /**
   * Execute an operation
   * 
   * @param operation Operation to execute
   * @returns Operation result
   * @throws {OperationExecutionException} When operation fails
   */
  execute(operation: Operation): Promise<OperationResult>;

  /**
   * Execute multiple operations in batch
   * 
   * @param operations Array of operations to execute
   * @param options Batch execution options
   * @returns Array of operation results
   */
  executeBatch(operations: Operation[], options?: BatchExecutionOptions): Promise<OperationResult[]>;

  /**
   * Schedule an operation for later execution
   * 
   * @param operation Operation to schedule
   * @param schedule Schedule specification
   * @returns Scheduled operation ID
   * @throws {SchedulingException} When scheduling fails
   */
  scheduleOperation(operation: Operation, schedule: Schedule): Promise<string>;

  /**
   * Cancel a scheduled operation
   * 
   * @param scheduledOperationId Scheduled operation ID
   */
  cancelScheduledOperation(scheduledOperationId: string): Promise<void>;

  /**
   * Get scheduled operations
   * 
   * @returns Array of scheduled operations
   */
  getScheduledOperations(): Promise<ScheduledOperation[]>;

  /**
   * Start the agent
   * 
   * @throws {AgentStartException} When agent cannot be started
   */
  start(): Promise<void>;

  /**
   * Stop the agent
   * 
   * @throws {AgentStopException} When agent cannot be stopped
   */
  stop(): Promise<void>;

  /**
   * Pause agent operations
   */
  pause(): Promise<void>;

  /**
   * Resume agent operations
   */
  resume(): Promise<void>;

  /**
   * Get agent configuration
   */
  getConfiguration(): Promise<AgentConfiguration>;

  /**
   * Update agent configuration
   * 
   * @param config New configuration
   * @throws {ConfigurationException} When configuration is invalid
   */
  updateConfiguration(config: Partial<AgentConfiguration>): Promise<void>;

  /**
   * Get agent performance metrics
   */
  getMetrics(): Promise<AgentMetrics>;

  /**
   * Get agent execution history
   * 
   * @param timeRange Time range for history
   * @param operationType Optional operation type filter
   */
  getExecutionHistory(
    timeRange: TimeRange,
    operationType?: OperationType
  ): Promise<ExecutionRecord[]>;

  /**
   * Get agent learning progress
   */
  getLearningProgress(): Promise<LearningProgress>;

  /**
   * Train the agent with new data
   * 
   * @param trainingData Training data
   * @param trainingOptions Training options
   */
  train(trainingData: TrainingData, trainingOptions?: TrainingOptions): Promise<TrainingResult>;

  /**
   * Subscribe to agent events
   * 
   * @param callback Event callback
   * @param eventTypes Optional event type filter
   * @returns Subscription ID
   */
  subscribeToEvents(
    callback: (event: AgentEvent) => void,
    eventTypes?: AgentEventType[]
  ): string;

  /**
   * Unsubscribe from agent events
   * 
   * @param subscriptionId Subscription ID
   */
  unsubscribeFromEvents(subscriptionId: string): void;

  /**
   * Get available operation types
   */
  getAvailableOperations(): OperationType[];

  /**
   * Validate operation before execution
   * 
   * @param operation Operation to validate
   * @returns Validation result
   */
  validateOperation(operation: Operation): Promise<OperationValidationResult>;

  /**
   * Get operation templates
   * 
   * @param operationType Optional operation type filter
   * @returns Array of operation templates
   */
  getOperationTemplates(operationType?: OperationType): Promise<OperationTemplate[]>;

  /**
   * Create operation from template
   * 
   * @param templateId Template ID
   * @param parameters Template parameters
   * @returns Created operation
   */
  createOperationFromTemplate(
    templateId: string,
    parameters: Record<string, any>
  ): Promise<Operation>;
}

/**
 * Agent type enumeration
 */
export enum AgentType {
  OPTIMIZATION_AGENT = 'OPTIMIZATION_AGENT',
  MAINTENANCE_AGENT = 'MAINTENANCE_AGENT',
  ANALYSIS_AGENT = 'ANALYSIS_AGENT',
  COMPLIANCE_AGENT = 'COMPLIANCE_AGENT',
  PERFORMANCE_AGENT = 'PERFORMANCE_AGENT',
  SELF_HEALING_AGENT = 'SELF_HEALING_AGENT',
  CAPACITY_PLANNING_AGENT = 'CAPACITY_PLANNING_AGENT',
  ANOMALY_DETECTION_AGENT = 'ANOMALY_DETECTION_AGENT'
}

/**
 * Agent capability
 */
export interface AgentCapability {
  /** Capability name */
  name: string;
  /** Capability description */
  description: string;
  /** Capability version */
  version: string;
  /** Supported operations */
  supportedOperations: OperationType[];
  /** Required permissions */
  requiredPermissions: Permission[];
  /** Capability parameters */
  parameters: CapabilityParameter[];
}

/**
 * Capability parameter
 */
export interface CapabilityParameter {
  /** Parameter name */
  name: string;
  /** Parameter type */
  type: string;
  /** Parameter description */
  description: string;
  /** Default value */
  defaultValue?: any;
  /** Required parameter */
  required: boolean;
}

/**
 * Permission
 */
export interface Permission {
  /** Resource type */
  resource: string;
  /** Actions allowed */
  actions: string[];
  /** Resource constraints */
  constraints?: Record<string, any>;
}

/**
 * Agent status
 */
export interface AgentStatus {
  /** Current state */
  state: AgentState;
  /** State description */
  stateDescription: string;
  /** Health status */
  health: 'HEALTHY' | 'DEGRADED' | 'UNHEALTHY';
  /** Current operation */
  currentOperation?: OperationSummary;
  /** Uptime in seconds */
  uptime: number;
  /** Resource utilization */
  resourceUtilization: ResourceUtilization;
  /** Last activity timestamp */
  lastActivity: Date;
  /** Error information if any */
  errors: AgentError[];
}

/**
 * Agent state enumeration
 */
export enum AgentState {
  STOPPED = 'STOPPED',
  STARTING = 'STARTING',
  RUNNING = 'RUNNING',
  PAUSED = 'PAUSED',
  STOPPING = 'STOPPING',
  ERROR = 'ERROR'
}

/**
 * Operation summary
 */
export interface OperationSummary {
  /** Operation ID */
  operationId: string;
  /** Operation type */
  operationType: OperationType;
  /** Operation description */
  description: string;
  /** Start time */
  startTime: Date;
  /** Progress percentage */
  progress: number;
  /** Estimated completion time */
  estimatedCompletion?: Date;
}

/**
 * Resource utilization
 */
export interface ResourceUtilization {
  /** CPU utilization percentage */
  cpu: number;
  /** Memory utilization percentage */
  memory: number;
  /** Network utilization percentage */
  network: number;
  /** Storage utilization percentage */
  storage: number;
}

/**
 * Agent error
 */
export interface AgentError {
  /** Error ID */
  errorId: string;
  /** Error code */
  errorCode: string;
  /** Error message */
  message: string;
  /** Error severity */
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  /** Error timestamp */
  timestamp: Date;
  /** Stack trace */
  stackTrace?: string;
  /** Error context */
  context?: Record<string, any>;
}

/**
 * Operation definition
 */
export interface Operation {
  /** Operation ID */
  operationId: string;
  /** Operation type */
  operationType: OperationType;
  /** Operation name */
  name: string;
  /** Operation description */
  description: string;
  /** Target resources */
  targets: OperationTarget[];
  /** Operation parameters */
  parameters: Record<string, any>;
  /** Operation constraints */
  constraints: OperationConstraint[];
  /** Expected duration in seconds */
  expectedDuration?: number;
  /** Operation priority */
  priority: OperationPriority;
  /** Retry policy */
  retryPolicy?: RetryPolicy;
  /** Rollback policy */
  rollbackPolicy?: RollbackPolicy;
}

/**
 * Operation type enumeration
 */
export enum OperationType {
  // Optimization operations
  PARAMETER_OPTIMIZATION = 'PARAMETER_OPTIMIZATION',
  LOAD_BALANCING = 'LOAD_BALANCING',
  INTERFERENCE_MITIGATION = 'INTERFERENCE_MITIGATION',
  COVERAGE_OPTIMIZATION = 'COVERAGE_OPTIMIZATION',
  CAPACITY_OPTIMIZATION = 'CAPACITY_OPTIMIZATION',
  ENERGY_OPTIMIZATION = 'ENERGY_OPTIMIZATION',
  
  // Maintenance operations
  PREVENTIVE_MAINTENANCE = 'PREVENTIVE_MAINTENANCE',
  CORRECTIVE_MAINTENANCE = 'CORRECTIVE_MAINTENANCE',
  SOFTWARE_UPDATE = 'SOFTWARE_UPDATE',
  CONFIGURATION_BACKUP = 'CONFIGURATION_BACKUP',
  LOG_ANALYSIS = 'LOG_ANALYSIS',
  HEALTH_CHECK = 'HEALTH_CHECK',
  
  // Analysis operations
  PERFORMANCE_ANALYSIS = 'PERFORMANCE_ANALYSIS',
  TRAFFIC_ANALYSIS = 'TRAFFIC_ANALYSIS',
  ANOMALY_DETECTION = 'ANOMALY_DETECTION',
  ROOT_CAUSE_ANALYSIS = 'ROOT_CAUSE_ANALYSIS',
  TREND_ANALYSIS = 'TREND_ANALYSIS',
  
  // Compliance operations
  COMPLIANCE_CHECK = 'COMPLIANCE_CHECK',
  POLICY_ENFORCEMENT = 'POLICY_ENFORCEMENT',
  AUDIT_EXECUTION = 'AUDIT_EXECUTION',
  
  // Self-healing operations
  FAULT_DETECTION = 'FAULT_DETECTION',
  AUTOMATIC_RECOVERY = 'AUTOMATIC_RECOVERY',
  SERVICE_RESTORATION = 'SERVICE_RESTORATION',
  
  // Custom operations
  CUSTOM_SCRIPT = 'CUSTOM_SCRIPT'
}

/**
 * Operation target
 */
export interface OperationTarget {
  /** Target type */
  type: TargetType;
  /** Target identifier */
  id: string;
  /** Target name */
  name: string;
  /** Target properties */
  properties: Record<string, any>;
}

/**
 * Target type enumeration
 */
export enum TargetType {
  NODE = 'NODE',
  CELL = 'CELL',
  NETWORK = 'NETWORK',
  REGION = 'REGION',
  SERVICE = 'SERVICE',
  PARAMETER = 'PARAMETER'
}

/**
 * Operation constraint
 */
export interface OperationConstraint {
  /** Constraint type */
  type: ConstraintType;
  /** Constraint parameters */
  parameters: Record<string, any>;
  /** Hard constraint (must be satisfied) or soft constraint (preferred) */
  hard: boolean;
}

/**
 * Constraint type enumeration
 */
export enum ConstraintType {
  TIME_WINDOW = 'TIME_WINDOW',
  RESOURCE_LIMIT = 'RESOURCE_LIMIT',
  PERFORMANCE_THRESHOLD = 'PERFORMANCE_THRESHOLD',
  SAFETY_LIMIT = 'SAFETY_LIMIT',
  BUSINESS_RULE = 'BUSINESS_RULE',
  DEPENDENCY = 'DEPENDENCY'
}

/**
 * Operation priority enumeration
 */
export enum OperationPriority {
  LOW = 'LOW',
  MEDIUM = 'MEDIUM',
  HIGH = 'HIGH',
  URGENT = 'URGENT',
  CRITICAL = 'CRITICAL'
}

/**
 * Retry policy
 */
export interface RetryPolicy {
  /** Maximum number of retries */
  maxRetries: number;
  /** Delay between retries in seconds */
  retryDelay: number;
  /** Exponential backoff factor */
  backoffFactor: number;
  /** Maximum delay between retries */
  maxDelay: number;
  /** Conditions for retry */
  retryConditions: RetryCondition[];
}

/**
 * Retry condition
 */
export interface RetryCondition {
  /** Error type to retry on */
  errorType: string;
  /** Error code to retry on */
  errorCode?: string;
  /** Condition description */
  description: string;
}

/**
 * Rollback policy
 */
export interface RollbackPolicy {
  /** Enable automatic rollback on failure */
  autoRollback: boolean;
  /** Rollback timeout in seconds */
  rollbackTimeout: number;
  /** Rollback strategy */
  rollbackStrategy: RollbackStrategy;
  /** Conditions that trigger rollback */
  rollbackTriggers: RollbackTrigger[];
}

/**
 * Rollback strategy enumeration
 */
export enum RollbackStrategy {
  IMMEDIATE = 'IMMEDIATE',
  STAGED = 'STAGED',
  ON_DEMAND = 'ON_DEMAND'
}

/**
 * Rollback trigger
 */
export interface RollbackTrigger {
  /** Trigger type */
  type: 'ERROR' | 'PERFORMANCE_DEGRADATION' | 'TIMEOUT' | 'MANUAL';
  /** Trigger parameters */
  parameters: Record<string, any>;
  /** Trigger description */
  description: string;
}

/**
 * Operation result
 */
export interface OperationResult {
  /** Operation ID */
  operationId: string;
  /** Success indicator */
  success: boolean;
  /** Result code */
  resultCode: string;
  /** Result message */
  message: string;
  /** Start time */
  startTime: Date;
  /** End time */
  endTime: Date;
  /** Execution duration in seconds */
  duration: number;
  /** Operation output data */
  outputData: Record<string, any>;
  /** Performance metrics */
  metrics: OperationMetrics;
  /** Warnings generated during execution */
  warnings: OperationWarning[];
  /** Errors if any */
  errors: OperationError[];
  /** Rollback information */
  rollbackInfo?: RollbackInfo;
}

/**
 * Operation metrics
 */
export interface OperationMetrics {
  /** Resource consumption */
  resourceConsumption: ResourceConsumption;
  /** Performance impact */
  performanceImpact: PerformanceImpact;
  /** Quality metrics */
  qualityMetrics: Record<string, number>;
}

/**
 * Resource consumption
 */
export interface ResourceConsumption {
  /** CPU time in seconds */
  cpuTime: number;
  /** Memory peak usage in MB */
  memoryPeak: number;
  /** Network data transferred in MB */
  networkData: number;
  /** Storage used in MB */
  storageUsed: number;
}

/**
 * Performance impact
 */
export interface PerformanceImpact {
  /** KPI changes */
  kpiChanges: KPIChange[];
  /** Service impact */
  serviceImpact: ServiceImpact;
  /** User experience impact */
  userExperienceImpact: number; // Score from -100 to +100
}

/**
 * KPI change
 */
export interface KPIChange {
  /** KPI name */
  kpiName: string;
  /** Previous value */
  previousValue: number;
  /** New value */
  newValue: number;
  /** Change percentage */
  changePercentage: number;
  /** Improvement indicator */
  improvement: boolean;
}

/**
 * Service impact
 */
export interface ServiceImpact {
  /** Affected services */
  affectedServices: string[];
  /** Downtime in seconds */
  downtime: number;
  /** Service degradation level */
  degradationLevel: 'NONE' | 'MINOR' | 'MODERATE' | 'SEVERE';
}

/**
 * Operation warning
 */
export interface OperationWarning {
  /** Warning code */
  code: string;
  /** Warning message */
  message: string;
  /** Warning severity */
  severity: 'LOW' | 'MEDIUM' | 'HIGH';
  /** Warning timestamp */
  timestamp: Date;
  /** Related target */
  target?: OperationTarget;
}

/**
 * Operation error
 */
export interface OperationError {
  /** Error code */
  code: string;
  /** Error message */
  message: string;
  /** Error severity */
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  /** Error timestamp */
  timestamp: Date;
  /** Stack trace */
  stackTrace?: string;
  /** Related target */
  target?: OperationTarget;
}

/**
 * Rollback information
 */
export interface RollbackInfo {
  /** Rollback performed */
  rollbackPerformed: boolean;
  /** Rollback success */
  rollbackSuccess: boolean;
  /** Rollback reason */
  rollbackReason: string;
  /** Rollback start time */
  rollbackStartTime?: Date;
  /** Rollback end time */
  rollbackEndTime?: Date;
  /** Rollback details */
  rollbackDetails: Record<string, any>;
}

/**
 * Batch execution options
 */
export interface BatchExecutionOptions {
  /** Execution mode */
  executionMode: 'SEQUENTIAL' | 'PARALLEL' | 'STAGED';
  /** Maximum concurrent operations (for parallel mode) */
  maxConcurrency?: number;
  /** Stop on first failure */
  stopOnFailure: boolean;
  /** Rollback strategy for batch */
  rollbackStrategy: 'NONE' | 'FAILED_ONLY' | 'ALL_ON_FAILURE';
}

/**
 * Schedule specification
 */
export interface Schedule {
  /** Schedule type */
  type: ScheduleType;
  /** Schedule parameters */
  parameters: ScheduleParameters;
  /** Time zone */
  timeZone: string;
  /** Schedule validity period */
  validityPeriod: TimePeriod;
}

/**
 * Schedule type enumeration
 */
export enum ScheduleType {
  ONE_TIME = 'ONE_TIME',
  RECURRING = 'RECURRING',
  CRON = 'CRON',
  EVENT_DRIVEN = 'EVENT_DRIVEN'
}

/**
 * Schedule parameters
 */
export interface ScheduleParameters {
  /** Start time (for one-time schedule) */
  startTime?: Date;
  /** Recurrence interval (for recurring schedule) */
  interval?: number;
  /** Interval unit (for recurring schedule) */
  intervalUnit?: 'SECONDS' | 'MINUTES' | 'HOURS' | 'DAYS' | 'WEEKS' | 'MONTHS';
  /** Cron expression (for cron schedule) */
  cronExpression?: string;
  /** Event trigger (for event-driven schedule) */
  eventTrigger?: EventTrigger;
}

/**
 * Event trigger
 */
export interface EventTrigger {
  /** Event type */
  eventType: string;
  /** Event source */
  eventSource: string;
  /** Event conditions */
  conditions: EventCondition[];
}

/**
 * Event condition
 */
export interface EventCondition {
  /** Field name */
  field: string;
  /** Operator */
  operator: 'EQUALS' | 'NOT_EQUALS' | 'GREATER_THAN' | 'LESS_THAN' | 'CONTAINS' | 'MATCHES';
  /** Value */
  value: any;
}

/**
 * Time period
 */
export interface TimePeriod {
  /** Start time */
  startTime: Date;
  /** End time */
  endTime: Date;
}

/**
 * Time range
 */
export interface TimeRange {
  /** Start time */
  startTime: Date;
  /** End time */
  endTime: Date;
}

/**
 * Scheduled operation
 */
export interface ScheduledOperation {
  /** Scheduled operation ID */
  scheduledOperationId: string;
  /** Operation definition */
  operation: Operation;
  /** Schedule specification */
  schedule: Schedule;
  /** Schedule status */
  status: ScheduleStatus;
  /** Next execution time */
  nextExecutionTime?: Date;
  /** Last execution time */
  lastExecutionTime?: Date;
  /** Last execution result */
  lastExecutionResult?: OperationResult;
  /** Created timestamp */
  createdAt: Date;
  /** Created by */
  createdBy: string;
}

/**
 * Schedule status enumeration
 */
export enum ScheduleStatus {
  ACTIVE = 'ACTIVE',
  PAUSED = 'PAUSED',
  COMPLETED = 'COMPLETED',
  CANCELLED = 'CANCELLED',
  EXPIRED = 'EXPIRED'
}

/**
 * Agent configuration
 */
export interface AgentConfiguration {
  /** Agent name */
  name: string;
  /** Agent description */
  description: string;
  /** Enabled capabilities */
  enabledCapabilities: string[];
  /** Configuration parameters */
  parameters: Record<string, any>;
  /** Resource limits */
  resourceLimits: ResourceLimits;
  /** Logging configuration */
  loggingConfig: LoggingConfig;
  /** Notification settings */
  notificationSettings: NotificationSettings;
  /** Learning settings */
  learningSettings: LearningSettings;
}

/**
 * Resource limits
 */
export interface ResourceLimits {
  /** Maximum CPU usage percentage */
  maxCpuUsage: number;
  /** Maximum memory usage in MB */
  maxMemoryUsage: number;
  /** Maximum concurrent operations */
  maxConcurrentOperations: number;
  /** Maximum operation duration in seconds */
  maxOperationDuration: number;
}

/**
 * Logging configuration
 */
export interface LoggingConfig {
  /** Log level */
  level: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR';
  /** Log retention days */
  retentionDays: number;
  /** Log to file */
  logToFile: boolean;
  /** Log file path */
  logFilePath?: string;
  /** Log to console */
  logToConsole: boolean;
  /** Structured logging */
  structuredLogging: boolean;
}

/**
 * Notification settings
 */
export interface NotificationSettings {
  /** Enable notifications */
  enabled: boolean;
  /** Notification channels */
  channels: NotificationChannel[];
  /** Notification filters */
  filters: NotificationFilter[];
}

/**
 * Notification channel
 */
export interface NotificationChannel {
  /** Channel type */
  type: 'EMAIL' | 'SMS' | 'WEBHOOK' | 'SLACK';
  /** Channel configuration */
  configuration: Record<string, any>;
  /** Enabled */
  enabled: boolean;
}

/**
 * Notification filter
 */
export interface NotificationFilter {
  /** Event types to notify */
  eventTypes: AgentEventType[];
  /** Minimum severity */
  minSeverity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  /** Target filter */
  targetFilter?: string;
}

/**
 * Learning settings
 */
export interface LearningSettings {
  /** Enable learning */
  enabled: boolean;
  /** Learning mode */
  mode: 'SUPERVISED' | 'UNSUPERVISED' | 'REINFORCEMENT';
  /** Learning rate */
  learningRate: number;
  /** Training frequency */
  trainingFrequency: 'DAILY' | 'WEEKLY' | 'MONTHLY';
  /** Model update strategy */
  modelUpdateStrategy: 'AUTOMATIC' | 'MANUAL';
}

/**
 * Agent metrics
 */
export interface AgentMetrics {
  /** Operations executed */
  operationsExecuted: number;
  /** Success rate percentage */
  successRate: number;
  /** Average execution time in seconds */
  averageExecutionTime: number;
  /** Resource utilization history */
  resourceUtilizationHistory: TimestampedMetric<ResourceUtilization>[];
  /** Performance metrics */
  performanceMetrics: Record<string, number>;
  /** Error statistics */
  errorStatistics: ErrorStatistics;
}

/**
 * Timestamped metric
 */
export interface TimestampedMetric<T> {
  /** Timestamp */
  timestamp: Date;
  /** Metric value */
  value: T;
}

/**
 * Error statistics
 */
export interface ErrorStatistics {
  /** Total errors */
  totalErrors: number;
  /** Errors by type */
  errorsByType: Record<string, number>;
  /** Errors by severity */
  errorsBySeverity: Record<string, number>;
  /** Recent errors */
  recentErrors: AgentError[];
}

/**
 * Execution record
 */
export interface ExecutionRecord {
  /** Record ID */
  recordId: string;
  /** Operation ID */
  operationId: string;
  /** Operation type */
  operationType: OperationType;
  /** Execution result */
  result: OperationResult;
  /** Execution context */
  context: ExecutionContext;
}

/**
 * Execution context
 */
export interface ExecutionContext {
  /** User who initiated the operation */
  initiatedBy: string;
  /** Execution environment */
  environment: 'PRODUCTION' | 'STAGING' | 'DEVELOPMENT';
  /** Execution source */
  source: 'MANUAL' | 'SCHEDULED' | 'EVENT_DRIVEN' | 'AUTOMATED';
  /** Additional context data */
  contextData: Record<string, any>;
}

/**
 * Learning progress
 */
export interface LearningProgress {
  /** Model version */
  modelVersion: string;
  /** Training iterations completed */
  trainingIterations: number;
  /** Model accuracy */
  modelAccuracy: number;
  /** Last training time */
  lastTrainingTime: Date;
  /** Next training time */
  nextTrainingTime?: Date;
  /** Training data points */
  trainingDataPoints: number;
  /** Learning metrics */
  learningMetrics: Record<string, number>;
}

/**
 * Training data
 */
export interface TrainingData {
  /** Data type */
  type: 'SUPERVISED' | 'UNSUPERVISED' | 'REINFORCEMENT';
  /** Training samples */
  samples: TrainingSample[];
  /** Data metadata */
  metadata: Record<string, any>;
}

/**
 * Training sample
 */
export interface TrainingSample {
  /** Input features */
  input: Record<string, any>;
  /** Expected output (for supervised learning) */
  output?: Record<string, any>;
  /** Reward signal (for reinforcement learning) */
  reward?: number;
  /** Sample weight */
  weight?: number;
  /** Sample metadata */
  metadata?: Record<string, any>;
}

/**
 * Training options
 */
export interface TrainingOptions {
  /** Training algorithm */
  algorithm?: string;
  /** Learning rate */
  learningRate?: number;
  /** Number of epochs */
  epochs?: number;
  /** Batch size */
  batchSize?: number;
  /** Validation split */
  validationSplit?: number;
  /** Early stopping */
  earlyStopping?: boolean;
}

/**
 * Training result
 */
export interface TrainingResult {
  /** Training success */
  success: boolean;
  /** New model version */
  modelVersion: string;
  /** Training metrics */
  trainingMetrics: Record<string, number>;
  /** Validation metrics */
  validationMetrics: Record<string, number>;
  /** Training duration */
  trainingDuration: number;
  /** Model improvement */
  modelImprovement: number;
}

/**
 * Agent event
 */
export interface AgentEvent {
  /** Event ID */
  eventId: string;
  /** Event type */
  eventType: AgentEventType;
  /** Event timestamp */
  timestamp: Date;
  /** Agent ID */
  agentId: string;
  /** Event data */
  data: Record<string, any>;
  /** Event severity */
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  /** Event message */
  message: string;
}

/**
 * Agent event type enumeration
 */
export enum AgentEventType {
  AGENT_STARTED = 'AGENT_STARTED',
  AGENT_STOPPED = 'AGENT_STOPPED',
  AGENT_PAUSED = 'AGENT_PAUSED',
  AGENT_RESUMED = 'AGENT_RESUMED',
  OPERATION_STARTED = 'OPERATION_STARTED',
  OPERATION_COMPLETED = 'OPERATION_COMPLETED',
  OPERATION_FAILED = 'OPERATION_FAILED',
  CONFIGURATION_UPDATED = 'CONFIGURATION_UPDATED',
  MODEL_TRAINED = 'MODEL_TRAINED',
  ERROR_OCCURRED = 'ERROR_OCCURRED',
  WARNING_ISSUED = 'WARNING_ISSUED'
}

/**
 * Operation validation result
 */
export interface OperationValidationResult {
  /** Validation success */
  valid: boolean;
  /** Validation errors */
  errors: ValidationError[];
  /** Validation warnings */
  warnings: ValidationWarning[];
  /** Estimated execution time */
  estimatedExecutionTime?: number;
  /** Required resources */
  requiredResources?: ResourceRequirement[];
}

/**
 * Validation error
 */
export interface ValidationError {
  /** Error code */
  code: string;
  /** Error message */
  message: string;
  /** Field that caused the error */
  field?: string;
  /** Error severity */
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
}

/**
 * Validation warning
 */
export interface ValidationWarning {
  /** Warning code */
  code: string;
  /** Warning message */
  message: string;
  /** Field that caused the warning */
  field?: string;
}

/**
 * Resource requirement
 */
export interface ResourceRequirement {
  /** Resource type */
  type: 'CPU' | 'MEMORY' | 'NETWORK' | 'STORAGE';
  /** Required amount */
  amount: number;
  /** Unit of measurement */
  unit: string;
  /** Duration of requirement */
  duration?: number;
}

/**
 * Operation template
 */
export interface OperationTemplate {
  /** Template ID */
  templateId: string;
  /** Template name */
  name: string;
  /** Template description */
  description: string;
  /** Operation type */
  operationType: OperationType;
  /** Template parameters */
  parameters: TemplateParameter[];
  /** Template constraints */
  constraints: OperationConstraint[];
  /** Template version */
  version: string;
  /** Template author */
  author: string;
  /** Creation timestamp */
  createdAt: Date;
}

/**
 * Template parameter
 */
export interface TemplateParameter {
  /** Parameter name */
  name: string;
  /** Parameter type */
  type: string;
  /** Parameter description */
  description: string;
  /** Default value */
  defaultValue?: any;
  /** Required parameter */
  required: boolean;
  /** Parameter constraints */
  constraints?: Record<string, any>;
}