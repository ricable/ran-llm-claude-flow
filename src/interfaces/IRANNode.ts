/**
 * Base RAN Node Interface
 * 
 * Defines the contract for all RAN nodes (eNodeB, gNodeB) with common operations
 * and lifecycle management capabilities.
 */

import { NodeConfig, CellConfig, NodeType, CellType } from './IRANNodeFactory';

/**
 * RAN Cell interface
 */
export interface IRANCell {
  /** Get cell identifier */
  getCellId(): string;
  /** Get cell name */
  getCellName(): string;
  /** Get cell type */
  getCellType(): CellType;
  /** Get current cell status */
  getStatus(): Promise<CellStatus>;
  /** Configure cell */
  configure(config: CellConfig): Promise<void>;
  /** Start cell operations */
  start(): Promise<void>;
  /** Stop cell operations */
  stop(): Promise<void>;
  /** Get cell KPIs */
  getKPIs(): Promise<CellKPIs>;
  /** Optimize cell parameters */
  optimize(criteria: OptimizationCriteria): Promise<OptimizationResult>;
}

/**
 * Main RAN Node interface
 * 
 * Represents a base station (eNodeB for 4G, gNodeB for 5G) with full
 * lifecycle management, configuration, monitoring, and automation capabilities.
 */
export interface IRANNode {
  /**
   * Get unique node identifier
   */
  getId(): string;

  /**
   * Get human-readable node name
   */
  getName(): string;

  /**
   * Get node type (eNodeB, gNodeB, etc.)
   */
  getType(): NodeType;

  /**
   * Get current node status
   */
  getStatus(): Promise<NodeStatus>;

  /**
   * Configure the node with new settings
   * 
   * @param config Node configuration
   * @throws {ConfigurationException} When configuration is invalid or cannot be applied
   */
  configure(config: Partial<NodeConfig>): Promise<void>;

  /**
   * Start node operations
   * 
   * @throws {NodeOperationException} When node cannot be started
   */
  start(): Promise<void>;

  /**
   * Stop node operations gracefully
   * 
   * @throws {NodeOperationException} When node cannot be stopped
   */
  stop(): Promise<void>;

  /**
   * Restart the node
   * 
   * @throws {NodeOperationException} When node cannot be restarted
   */
  restart(): Promise<void>;

  /**
   * Get all cells managed by this node
   */
  getCells(): Promise<IRANCell[]>;

  /**
   * Get a specific cell by ID
   * 
   * @param cellId Cell identifier
   * @returns Cell instance or null if not found
   */
  getCell(cellId: string): Promise<IRANCell | null>;

  /**
   * Add a new cell to the node
   * 
   * @param config Cell configuration
   * @returns Created cell instance
   * @throws {CellCreationException} When cell cannot be created
   */
  addCell(config: CellConfig): Promise<IRANCell>;

  /**
   * Remove a cell from the node
   * 
   * @param cellId Cell identifier
   * @throws {CellOperationException} When cell cannot be removed
   */
  removeCell(cellId: string): Promise<void>;

  /**
   * Get current node KPIs
   */
  getKPIs(): Promise<NodeKPIs>;

  /**
   * Get historical KPI data
   * 
   * @param timeRange Time range for historical data
   */
  getHistoricalKPIs(timeRange: TimeRange): Promise<HistoricalKPIs>;

  /**
   * Monitor node health
   */
  monitor(): Promise<HealthStatus>;

  /**
   * Optimize node performance
   * 
   * @param criteria Optimization criteria
   * @returns Optimization results
   */
  optimize(criteria: OptimizationCriteria): Promise<OptimizationResult>;

  /**
   * Execute maintenance tasks
   * 
   * @param tasks Array of maintenance tasks to execute
   */
  executeMaintenance(tasks: MaintenanceTask[]): Promise<MaintenanceResult[]>;

  /**
   * Get node capabilities
   */
  getCapabilities(): NodeCapabilities;

  /**
   * Get current configuration
   */
  getCurrentConfig(): Promise<NodeConfig>;

  /**
   * Backup current configuration
   * 
   * @param backupName Optional backup name
   * @returns Backup identifier
   */
  backupConfiguration(backupName?: string): Promise<string>;

  /**
   * Restore configuration from backup
   * 
   * @param backupId Backup identifier
   */
  restoreConfiguration(backupId: string): Promise<void>;

  /**
   * Get alarms for this node
   * 
   * @param severity Optional severity filter
   */
  getAlarms(severity?: AlarmSeverity): Promise<Alarm[]>;

  /**
   * Acknowledge an alarm
   * 
   * @param alarmId Alarm identifier
   */
  acknowledgeAlarm(alarmId: string): Promise<void>;

  /**
   * Get event log
   * 
   * @param timeRange Time range for events
   * @param eventType Optional event type filter
   */
  getEvents(timeRange: TimeRange, eventType?: EventType): Promise<NodeEvent[]>;

  /**
   * Subscribe to node events
   * 
   * @param callback Event callback function
   * @param eventTypes Optional array of event types to subscribe to
   * @returns Subscription ID for unsubscribing
   */
  subscribeToEvents(
    callback: (event: NodeEvent) => void,
    eventTypes?: EventType[]
  ): string;

  /**
   * Unsubscribe from node events
   * 
   * @param subscriptionId Subscription ID
   */
  unsubscribeFromEvents(subscriptionId: string): void;

  /**
   * Execute a custom command on the node
   * 
   * @param command Command to execute
   * @returns Command result
   */
  executeCommand(command: NodeCommand): Promise<CommandResult>;

  /**
   * Get software information
   */
  getSoftwareInfo(): Promise<SoftwareInfo>;

  /**
   * Get hardware information
   */
  getHardwareInfo(): Promise<HardwareInfo>;

  /**
   * Perform software upgrade
   * 
   * @param upgradePackage Upgrade package information
   * @param options Upgrade options
   */
  upgradeSoftware(
    upgradePackage: UpgradePackage,
    options?: UpgradeOptions
  ): Promise<UpgradeResult>;
}

/**
 * Node status information
 */
export interface NodeStatus {
  /** Current operational state */
  operationalState: OperationalState;
  /** Administrative state */
  administrativeState: AdministrativeState;
  /** Availability status */
  availabilityStatus: AvailabilityStatus;
  /** Last update timestamp */
  lastUpdate: Date;
  /** Additional status information */
  additionalInfo: Record<string, any>;
}

/**
 * Cell status information
 */
export interface CellStatus {
  /** Cell operational state */
  operationalState: OperationalState;
  /** Cell administrative state */
  administrativeState: AdministrativeState;
  /** Number of connected UEs */
  connectedUEs: number;
  /** Cell load percentage */
  loadPercentage: number;
  /** Last update timestamp */
  lastUpdate: Date;
}

/**
 * Node KPIs
 */
export interface NodeKPIs {
  /** Availability percentage */
  availability: number;
  /** CPU utilization percentage */
  cpuUtilization: number;
  /** Memory utilization percentage */
  memoryUtilization: number;
  /** Total throughput in bps */
  totalThroughput: number;
  /** Active UE count */
  activeUEs: number;
  /** Connection success rate percentage */
  connectionSuccessRate: number;
  /** Handover success rate percentage */
  handoverSuccessRate: number;
  /** Drop rate percentage */
  dropRate: number;
  /** Average latency in ms */
  averageLatency: number;
  /** Packet loss rate percentage */
  packetLossRate: number;
  /** Energy efficiency in bits/Joule */
  energyEfficiency: number;
  /** Timestamp of measurement */
  timestamp: Date;
  /** Technology-specific KPIs */
  technologySpecific: Record<string, number>;
}

/**
 * Cell KPIs
 */
export interface CellKPIs {
  /** Cell identifier */
  cellId: string;
  /** RSRP (Reference Signal Received Power) in dBm */
  rsrp: number;
  /** RSRQ (Reference Signal Received Quality) in dB */
  rsrq: number;
  /** SINR (Signal to Interference plus Noise Ratio) in dB */
  sinr: number;
  /** Throughput in bps */
  throughput: number;
  /** Number of connected UEs */
  connectedUEs: number;
  /** PRB (Physical Resource Block) utilization percentage */
  prbUtilization: number;
  /** CQI (Channel Quality Indicator) average */
  averageCqi: number;
  /** Interference level in dBm */
  interferenceLevel: number;
  /** Spectral efficiency in bps/Hz */
  spectralEfficiency: number;
  /** Timestamp of measurement */
  timestamp: Date;
}

/**
 * Historical KPI data
 */
export interface HistoricalKPIs {
  /** Time range of data */
  timeRange: TimeRange;
  /** Historical node KPI samples */
  nodeKpis: TimestampedKPI<NodeKPIs>[];
  /** Historical cell KPI samples */
  cellKpis: Record<string, TimestampedKPI<CellKPIs>[]>;
}

/**
 * Timestamped KPI sample
 */
export interface TimestampedKPI<T> {
  /** Timestamp */
  timestamp: Date;
  /** KPI data */
  data: T;
}

/**
 * Time range specification
 */
export interface TimeRange {
  /** Start time */
  startTime: Date;
  /** End time */
  endTime: Date;
}

/**
 * Health status
 */
export interface HealthStatus {
  /** Overall health score (0-100) */
  overallHealth: number;
  /** Component health details */
  componentHealth: ComponentHealth[];
  /** Active issues */
  activeIssues: HealthIssue[];
  /** Health recommendations */
  recommendations: HealthRecommendation[];
  /** Last health check timestamp */
  lastCheckTime: Date;
}

/**
 * Component health information
 */
export interface ComponentHealth {
  /** Component name */
  componentName: string;
  /** Health score (0-100) */
  healthScore: number;
  /** Status description */
  status: string;
  /** Last checked timestamp */
  lastChecked: Date;
}

/**
 * Health issue
 */
export interface HealthIssue {
  /** Issue ID */
  issueId: string;
  /** Severity level */
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  /** Issue description */
  description: string;
  /** Affected component */
  component: string;
  /** Detection timestamp */
  detectedAt: Date;
}

/**
 * Health recommendation
 */
export interface HealthRecommendation {
  /** Recommendation ID */
  recommendationId: string;
  /** Priority level */
  priority: 'LOW' | 'MEDIUM' | 'HIGH';
  /** Recommendation text */
  recommendation: string;
  /** Expected impact */
  expectedImpact: string;
}

/**
 * Optimization criteria
 */
export interface OptimizationCriteria {
  /** Optimization objective */
  objective: OptimizationObjective;
  /** Target KPIs */
  targetKpis: Record<string, number>;
  /** Constraints */
  constraints: OptimizationConstraint[];
  /** Time window for optimization */
  timeWindow: TimeRange;
  /** Optimization algorithm preference */
  algorithm?: 'GENETIC' | 'GRADIENT_DESCENT' | 'REINFORCEMENT_LEARNING' | 'HEURISTIC';
}

/**
 * Optimization objective
 */
export enum OptimizationObjective {
  MAXIMIZE_THROUGHPUT = 'MAXIMIZE_THROUGHPUT',
  MINIMIZE_LATENCY = 'MINIMIZE_LATENCY',
  MAXIMIZE_COVERAGE = 'MAXIMIZE_COVERAGE',
  MINIMIZE_INTERFERENCE = 'MINIMIZE_INTERFERENCE',
  MAXIMIZE_ENERGY_EFFICIENCY = 'MAXIMIZE_ENERGY_EFFICIENCY',
  MAXIMIZE_USER_EXPERIENCE = 'MAXIMIZE_USER_EXPERIENCE',
  BALANCE_LOAD = 'BALANCE_LOAD'
}

/**
 * Optimization constraint
 */
export interface OptimizationConstraint {
  /** Parameter name */
  parameter: string;
  /** Constraint type */
  type: 'MIN' | 'MAX' | 'RANGE' | 'FIXED';
  /** Constraint value(s) */
  value: number | [number, number];
  /** Hard constraint (true) or soft constraint (false) */
  hard: boolean;
}

/**
 * Optimization result
 */
export interface OptimizationResult {
  /** Success indicator */
  success: boolean;
  /** Applied parameter changes */
  appliedChanges: ParameterChange[];
  /** Performance improvement metrics */
  performanceImprovement: Record<string, number>;
  /** Optimization duration */
  duration: number;
  /** Algorithm used */
  algorithmUsed: string;
  /** Convergence information */
  convergenceInfo: ConvergenceInfo;
}

/**
 * Parameter change information
 */
export interface ParameterChange {
  /** Parameter name */
  parameterName: string;
  /** Old value */
  oldValue: any;
  /** New value */
  newValue: any;
  /** Change reason */
  reason: string;
}

/**
 * Convergence information
 */
export interface ConvergenceInfo {
  /** Whether optimization converged */
  converged: boolean;
  /** Number of iterations */
  iterations: number;
  /** Final objective value */
  finalObjectiveValue: number;
  /** Convergence time in seconds */
  convergenceTime: number;
}

/**
 * Maintenance task
 */
export interface MaintenanceTask {
  /** Task ID */
  taskId: string;
  /** Task type */
  type: MaintenanceTaskType;
  /** Task description */
  description: string;
  /** Task priority */
  priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'URGENT';
  /** Scheduled execution time */
  scheduledTime?: Date;
  /** Task parameters */
  parameters: Record<string, any>;
}

/**
 * Maintenance task type
 */
export enum MaintenanceTaskType {
  SOFTWARE_UPDATE = 'SOFTWARE_UPDATE',
  CONFIGURATION_BACKUP = 'CONFIGURATION_BACKUP',
  LOG_CLEANUP = 'LOG_CLEANUP',
  PERFORMANCE_TUNING = 'PERFORMANCE_TUNING',
  CELL_OPTIMIZATION = 'CELL_OPTIMIZATION',
  ANTENNA_ALIGNMENT = 'ANTENNA_ALIGNMENT',
  INTERFERENCE_MITIGATION = 'INTERFERENCE_MITIGATION',
  CAPACITY_PLANNING = 'CAPACITY_PLANNING'
}

/**
 * Maintenance result
 */
export interface MaintenanceResult {
  /** Task ID */
  taskId: string;
  /** Success indicator */
  success: boolean;
  /** Execution start time */
  startTime: Date;
  /** Execution end time */
  endTime: Date;
  /** Result message */
  message: string;
  /** Any errors encountered */
  errors: string[];
  /** Generated artifacts */
  artifacts: Artifact[];
}

/**
 * Generated artifact
 */
export interface Artifact {
  /** Artifact name */
  name: string;
  /** Artifact type */
  type: 'LOG' | 'CONFIG' | 'REPORT' | 'DATA';
  /** File path or URL */
  location: string;
  /** Size in bytes */
  size: number;
  /** Creation timestamp */
  createdAt: Date;
}

/**
 * Node capabilities
 */
export interface NodeCapabilities {
  /** Supported RAN technologies */
  supportedTechnologies: string[];
  /** Supported frequency bands */
  supportedBands: number[];
  /** Maximum number of cells */
  maxCells: number;
  /** Maximum throughput in bps */
  maxThroughput: number;
  /** Supported features */
  supportedFeatures: FeatureCapability[];
  /** Hardware capabilities */
  hardwareCapabilities: HardwareCapability[];
  /** Software capabilities */
  softwareCapabilities: SoftwareCapability[];
}

/**
 * Feature capability
 */
export interface FeatureCapability {
  /** Feature name */
  featureName: string;
  /** Feature version */
  version: string;
  /** Whether feature is enabled */
  enabled: boolean;
  /** License status */
  licensed: boolean;
  /** Configuration options */
  configOptions: ConfigOption[];
}

/**
 * Configuration option
 */
export interface ConfigOption {
  /** Option name */
  name: string;
  /** Option type */
  type: 'BOOLEAN' | 'INTEGER' | 'FLOAT' | 'STRING' | 'ENUM';
  /** Default value */
  defaultValue: any;
  /** Allowed values (for ENUM type) */
  allowedValues?: any[];
  /** Value range (for numeric types) */
  range?: [number, number];
  /** Description */
  description: string;
}

/**
 * Hardware capability
 */
export interface HardwareCapability {
  /** Component name */
  component: string;
  /** Component type */
  type: string;
  /** Capacity or performance metric */
  capacity: number;
  /** Unit of measurement */
  unit: string;
}

/**
 * Software capability
 */
export interface SoftwareCapability {
  /** Software component name */
  component: string;
  /** Version */
  version: string;
  /** Supported APIs */
  supportedApis: string[];
  /** License information */
  license: string;
}

/**
 * Alarm information
 */
export interface Alarm {
  /** Alarm ID */
  alarmId: string;
  /** Alarm severity */
  severity: AlarmSeverity;
  /** Alarm type */
  alarmType: string;
  /** Alarm text */
  alarmText: string;
  /** Probable cause */
  probableCause: string;
  /** Specific problem */
  specificProblem: string;
  /** Managed object */
  managedObject: string;
  /** Event time */
  eventTime: Date;
  /** Acknowledgment status */
  acknowledged: boolean;
  /** Acknowledgment time */
  acknowledgedTime?: Date;
  /** Acknowledgment user */
  acknowledgedBy?: string;
  /** Additional attributes */
  additionalAttributes: Record<string, any>;
}

/**
 * Node event
 */
export interface NodeEvent {
  /** Event ID */
  eventId: string;
  /** Event type */
  eventType: EventType;
  /** Event timestamp */
  timestamp: Date;
  /** Source component */
  source: string;
  /** Event description */
  description: string;
  /** Event data */
  data: Record<string, any>;
  /** Severity level */
  severity: 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';
}

/**
 * Node command
 */
export interface NodeCommand {
  /** Command name */
  command: string;
  /** Command parameters */
  parameters: Record<string, any>;
  /** Execution timeout in seconds */
  timeout?: number;
}

/**
 * Command result
 */
export interface CommandResult {
  /** Success indicator */
  success: boolean;
  /** Return code */
  returnCode: number;
  /** Output message */
  output: string;
  /** Error message if any */
  error?: string;
  /** Execution time in ms */
  executionTime: number;
}

/**
 * Software information
 */
export interface SoftwareInfo {
  /** Product name */
  productName: string;
  /** Product version */
  productVersion: string;
  /** Build number */
  buildNumber: string;
  /** Release date */
  releaseDate: Date;
  /** Installed packages */
  installedPackages: SoftwarePackage[];
  /** License information */
  licenses: LicenseInfo[];
}

/**
 * Software package
 */
export interface SoftwarePackage {
  /** Package name */
  name: string;
  /** Package version */
  version: string;
  /** Package description */
  description: string;
  /** Installation date */
  installedDate: Date;
  /** Package size in bytes */
  size: number;
}

/**
 * License information
 */
export interface LicenseInfo {
  /** Feature name */
  featureName: string;
  /** License key */
  licenseKey: string;
  /** Expiration date */
  expirationDate?: Date;
  /** Licensed capacity */
  capacity?: number;
  /** License status */
  status: 'ACTIVE' | 'EXPIRED' | 'INVALID';
}

/**
 * Hardware information
 */
export interface HardwareInfo {
  /** Chassis information */
  chassis: ChassisInfo;
  /** Board information */
  boards: BoardInfo[];
  /** Port information */
  ports: PortInfo[];
  /** Environmental sensors */
  sensors: SensorInfo[];
}

/**
 * Chassis information
 */
export interface ChassisInfo {
  /** Chassis type */
  chassisType: string;
  /** Serial number */
  serialNumber: string;
  /** Product number */
  productNumber: string;
  /** Hardware revision */
  hardwareRevision: string;
  /** Manufacturing date */
  manufacturingDate: Date;
}

/**
 * Board information
 */
export interface BoardInfo {
  /** Slot number */
  slotNumber: number;
  /** Board type */
  boardType: string;
  /** Serial number */
  serialNumber: string;
  /** Product number */
  productNumber: string;
  /** Operational state */
  operationalState: OperationalState;
  /** Administrative state */
  administrativeState: AdministrativeState;
}

/**
 * Port information
 */
export interface PortInfo {
  /** Port identifier */
  portId: string;
  /** Port type */
  portType: string;
  /** Port speed */
  speed: number;
  /** Port status */
  status: 'UP' | 'DOWN' | 'TESTING';
  /** Connected device */
  connectedDevice?: string;
}

/**
 * Sensor information
 */
export interface SensorInfo {
  /** Sensor name */
  sensorName: string;
  /** Sensor type */
  sensorType: 'TEMPERATURE' | 'VOLTAGE' | 'CURRENT' | 'POWER' | 'FAN_SPEED';
  /** Current reading */
  currentReading: number;
  /** Unit of measurement */
  unit: string;
  /** Normal range */
  normalRange: [number, number];
  /** Status */
  status: 'NORMAL' | 'WARNING' | 'CRITICAL';
}

/**
 * Upgrade package information
 */
export interface UpgradePackage {
  /** Package name */
  packageName: string;
  /** Source version */
  fromVersion: string;
  /** Target version */
  toVersion: string;
  /** Package location */
  packageLocation: string;
  /** Package size in bytes */
  packageSize: number;
  /** Checksum */
  checksum: string;
  /** Release notes */
  releaseNotes: string;
}

/**
 * Upgrade options
 */
export interface UpgradeOptions {
  /** Force upgrade even if preconditions fail */
  force?: boolean;
  /** Backup current configuration before upgrade */
  backupConfiguration?: boolean;
  /** Rollback timeout in minutes */
  rollbackTimeout?: number;
  /** Maintenance window */
  maintenanceWindow?: TimeRange;
  /** Pre-upgrade validation */
  preUpgradeValidation?: boolean;
  /** Post-upgrade validation */
  postUpgradeValidation?: boolean;
}

/**
 * Upgrade result
 */
export interface UpgradeResult {
  /** Success indicator */
  success: boolean;
  /** Start time */
  startTime: Date;
  /** End time */
  endTime: Date;
  /** Previous version */
  previousVersion: string;
  /** New version */
  newVersion: string;
  /** Upgrade phases completed */
  completedPhases: UpgradePhase[];
  /** Rollback performed */
  rollbackPerformed: boolean;
  /** Error messages */
  errors: string[];
  /** Configuration backup ID */
  configBackupId?: string;
}

/**
 * Upgrade phase
 */
export interface UpgradePhase {
  /** Phase name */
  phaseName: string;
  /** Start time */
  startTime: Date;
  /** End time */
  endTime: Date;
  /** Success indicator */
  success: boolean;
  /** Phase description */
  description: string;
  /** Error message if any */
  error?: string;
}

/**
 * Operational state enumeration
 */
export enum OperationalState {
  ENABLED = 'ENABLED',
  DISABLED = 'DISABLED'
}

/**
 * Administrative state enumeration
 */
export enum AdministrativeState {
  LOCKED = 'LOCKED',
  UNLOCKED = 'UNLOCKED',
  SHUTTINGDOWN = 'SHUTTINGDOWN'
}

/**
 * Availability status enumeration
 */
export enum AvailabilityStatus {
  IN_SERVICE = 'IN_SERVICE',
  OUT_OF_SERVICE = 'OUT_OF_SERVICE',
  DEGRADED = 'DEGRADED'
}

/**
 * Alarm severity enumeration
 */
export enum AlarmSeverity {
  CRITICAL = 'CRITICAL',
  MAJOR = 'MAJOR',
  MINOR = 'MINOR',
  WARNING = 'WARNING',
  INDETERMINATE = 'INDETERMINATE',
  CLEARED = 'CLEARED'
}

/**
 * Event type enumeration
 */
export enum EventType {
  CONFIGURATION_CHANGE = 'CONFIGURATION_CHANGE',
  STATE_CHANGE = 'STATE_CHANGE',
  ALARM = 'ALARM',
  PERFORMANCE = 'PERFORMANCE',
  SECURITY = 'SECURITY',
  MAINTENANCE = 'MAINTENANCE',
  USER_ACTION = 'USER_ACTION',
  SYSTEM_ACTION = 'SYSTEM_ACTION'
}