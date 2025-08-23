/**
 * Monitoring Service Interface
 * 
 * Provides comprehensive monitoring capabilities for RAN networks including
 * KPI collection, performance monitoring, alerting, anomaly detection, and reporting.
 */

/**
 * Monitoring Service Interface
 * 
 * Manages all aspects of RAN network monitoring including real-time KPI collection,
 * historical data analysis, alerting, anomaly detection, and comprehensive reporting.
 */
export interface IMonitoringService {
  /**
   * Collect KPIs for a specific node
   * 
   * @param nodeId Node identifier
   * @param kpiTypes Optional filter for specific KPI types
   * @returns Collected KPI data
   * @throws {KPICollectionException} When KPI collection fails
   */
  collectKPIs(nodeId: string, kpiTypes?: KPIType[]): Promise<KPICollection>;

  /**
   * Collect KPIs for multiple nodes
   * 
   * @param nodeIds Array of node identifiers
   * @param kpiTypes Optional filter for specific KPI types
   * @returns Array of collected KPI data
   */
  collectKPIsBatch(nodeIds: string[], kpiTypes?: KPIType[]): Promise<KPICollection[]>;

  /**
   * Get historical KPI data
   * 
   * @param nodeId Node identifier
   * @param timeRange Time range for historical data
   * @param kpiTypes Optional filter for specific KPI types
   * @param aggregation Data aggregation settings
   * @returns Historical KPI data
   */
  getHistoricalKPIs(
    nodeId: string,
    timeRange: TimeRange,
    kpiTypes?: KPIType[],
    aggregation?: AggregationSettings
  ): Promise<HistoricalKPIData>;

  /**
   * Get real-time KPI stream
   * 
   * @param nodeId Node identifier
   * @param kpiTypes Optional filter for specific KPI types
   * @param callback Callback function for KPI updates
   * @returns Stream subscription ID
   */
  getKPIStream(
    nodeId: string,
    kpiTypes: KPIType[],
    callback: (kpis: KPICollection) => void
  ): string;

  /**
   * Stop KPI stream
   * 
   * @param subscriptionId Stream subscription ID
   */
  stopKPIStream(subscriptionId: string): Promise<void>;

  /**
   * Generate performance report
   * 
   * @param criteria Report generation criteria
   * @returns Generated report
   * @throws {ReportGenerationException} When report generation fails
   */
  generateReport(criteria: ReportCriteria): Promise<Report>;

  /**
   * Schedule automatic report generation
   * 
   * @param criteria Report generation criteria
   * @param schedule Schedule specification
   * @returns Scheduled report ID
   */
  scheduleReport(criteria: ReportCriteria, schedule: ReportSchedule): Promise<string>;

  /**
   * Cancel scheduled report
   * 
   * @param scheduledReportId Scheduled report ID
   */
  cancelScheduledReport(scheduledReportId: string): Promise<void>;

  /**
   * Get available reports
   * 
   * @param filters Optional filters
   * @returns Array of available reports
   */
  getAvailableReports(filters?: ReportFilters): Promise<ReportSummary[]>;

  /**
   * Setup alert configuration
   * 
   * @param alertConfig Alert configuration
   * @returns Created alert ID
   * @throws {AlertConfigurationException} When alert setup fails
   */
  setupAlert(alertConfig: AlertConfig): Promise<string>;

  /**
   * Update alert configuration
   * 
   * @param alertId Alert identifier
   * @param updates Partial alert configuration updates
   * @throws {AlertNotFoundException} When alert is not found
   */
  updateAlert(alertId: string, updates: Partial<AlertConfig>): Promise<void>;

  /**
   * Delete alert
   * 
   * @param alertId Alert identifier
   * @throws {AlertNotFoundException} When alert is not found
   */
  deleteAlert(alertId: string): Promise<void>;

  /**
   * Get active alerts
   * 
   * @param filters Optional filters
   * @returns Array of active alerts
   */
  getActiveAlerts(filters?: AlertFilters): Promise<Alert[]>;

  /**
   * Get alert history
   * 
   * @param timeRange Time range for alert history
   * @param filters Optional filters
   * @returns Array of historical alerts
   */
  getAlertHistory(timeRange: TimeRange, filters?: AlertFilters): Promise<Alert[]>;

  /**
   * Acknowledge alert
   * 
   * @param alertId Alert identifier
   * @param acknowledgment Acknowledgment details
   */
  acknowledgeAlert(alertId: string, acknowledgment: AlertAcknowledgment): Promise<void>;

  /**
   * Clear alert
   * 
   * @param alertId Alert identifier
   * @param clearReason Reason for clearing
   */
  clearAlert(alertId: string, clearReason: string): Promise<void>;

  /**
   * Get current performance metrics for a node
   * 
   * @param nodeId Node identifier
   * @returns Current performance metrics
   */
  getPerformanceMetrics(nodeId: string): Promise<PerformanceMetrics>;

  /**
   * Get network-wide performance summary
   * 
   * @param networkId Optional network identifier
   * @returns Network performance summary
   */
  getNetworkPerformanceSummary(networkId?: string): Promise<NetworkPerformanceSummary>;

  /**
   * Perform anomaly detection
   * 
   * @param nodeId Node identifier
   * @param detectionSettings Anomaly detection settings
   * @returns Anomaly detection results
   */
  detectAnomalies(
    nodeId: string,
    detectionSettings?: AnomalyDetectionSettings
  ): Promise<AnomalyDetectionResult>;

  /**
   * Setup anomaly detection monitoring
   * 
   * @param nodeId Node identifier
   * @param settings Anomaly detection settings
   * @param callback Callback for anomaly notifications
   * @returns Monitoring subscription ID
   */
  setupAnomalyMonitoring(
    nodeId: string,
    settings: AnomalyDetectionSettings,
    callback: (anomalies: Anomaly[]) => void
  ): string;

  /**
   * Stop anomaly monitoring
   * 
   * @param subscriptionId Monitoring subscription ID
   */
  stopAnomalyMonitoring(subscriptionId: string): Promise<void>;

  /**
   * Get service level metrics
   * 
   * @param serviceId Service identifier
   * @param timeRange Time range for metrics
   * @returns Service level metrics
   */
  getServiceLevelMetrics(serviceId: string, timeRange: TimeRange): Promise<ServiceLevelMetrics>;

  /**
   * Setup SLA monitoring
   * 
   * @param slaConfig SLA configuration
   * @returns SLA monitor ID
   */
  setupSLAMonitoring(slaConfig: SLAConfig): Promise<string>;

  /**
   * Get SLA compliance status
   * 
   * @param slaMonitorId SLA monitor ID
   * @param timeRange Time range for compliance check
   * @returns SLA compliance status
   */
  getSLACompliance(slaMonitorId: string, timeRange: TimeRange): Promise<SLAComplianceStatus>;

  /**
   * Create custom dashboard
   * 
   * @param dashboardConfig Dashboard configuration
   * @returns Created dashboard ID
   */
  createDashboard(dashboardConfig: DashboardConfig): Promise<string>;

  /**
   * Update dashboard
   * 
   * @param dashboardId Dashboard identifier
   * @param updates Dashboard configuration updates
   */
  updateDashboard(dashboardId: string, updates: Partial<DashboardConfig>): Promise<void>;

  /**
   * Get dashboard data
   * 
   * @param dashboardId Dashboard identifier
   * @param timeRange Time range for dashboard data
   * @returns Dashboard data
   */
  getDashboardData(dashboardId: string, timeRange: TimeRange): Promise<DashboardData>;

  /**
   * Export monitoring data
   * 
   * @param exportRequest Export request specification
   * @returns Export result
   */
  exportData(exportRequest: DataExportRequest): Promise<DataExportResult>;

  /**
   * Import monitoring data
   * 
   * @param importRequest Import request specification
   * @returns Import result
   */
  importData(importRequest: DataImportRequest): Promise<DataImportResult>;

  /**
   * Get monitoring service status
   * 
   * @returns Service status information
   */
  getServiceStatus(): Promise<MonitoringServiceStatus>;

  /**
   * Configure data retention policies
   * 
   * @param policies Array of retention policies
   */
  configureRetentionPolicies(policies: RetentionPolicy[]): Promise<void>;

  /**
   * Get current retention policies
   * 
   * @returns Array of current retention policies
   */
  getRetentionPolicies(): Promise<RetentionPolicy[]>;

  /**
   * Subscribe to monitoring events
   * 
   * @param callback Event callback
   * @param eventTypes Optional event type filter
   * @returns Subscription ID
   */
  subscribeToEvents(
    callback: (event: MonitoringEvent) => void,
    eventTypes?: MonitoringEventType[]
  ): string;

  /**
   * Unsubscribe from monitoring events
   * 
   * @param subscriptionId Subscription ID
   */
  unsubscribeFromEvents(subscriptionId: string): void;
}

/**
 * KPI type enumeration
 */
export enum KPIType {
  // Availability KPIs
  AVAILABILITY = 'AVAILABILITY',
  UPTIME = 'UPTIME',
  DOWNTIME = 'DOWNTIME',
  RELIABILITY = 'RELIABILITY',
  
  // Performance KPIs
  THROUGHPUT = 'THROUGHPUT',
  LATENCY = 'LATENCY',
  PACKET_LOSS = 'PACKET_LOSS',
  JITTER = 'JITTER',
  RESPONSE_TIME = 'RESPONSE_TIME',
  
  // Quality KPIs
  RSRP = 'RSRP', // Reference Signal Received Power
  RSRQ = 'RSRQ', // Reference Signal Received Quality
  SINR = 'SINR', // Signal to Interference plus Noise Ratio
  CQI = 'CQI',   // Channel Quality Indicator
  BLER = 'BLER', // Block Error Rate
  
  // Traffic KPIs
  DATA_VOLUME = 'DATA_VOLUME',
  SESSION_COUNT = 'SESSION_COUNT',
  CONNECTION_COUNT = 'CONNECTION_COUNT',
  ACTIVE_USERS = 'ACTIVE_USERS',
  
  // Resource KPIs
  CPU_UTILIZATION = 'CPU_UTILIZATION',
  MEMORY_UTILIZATION = 'MEMORY_UTILIZATION',
  STORAGE_UTILIZATION = 'STORAGE_UTILIZATION',
  BANDWIDTH_UTILIZATION = 'BANDWIDTH_UTILIZATION',
  PRB_UTILIZATION = 'PRB_UTILIZATION', // Physical Resource Block
  
  // Energy KPIs
  POWER_CONSUMPTION = 'POWER_CONSUMPTION',
  ENERGY_EFFICIENCY = 'ENERGY_EFFICIENCY',
  
  // Service KPIs
  SERVICE_ACCESSIBILITY = 'SERVICE_ACCESSIBILITY',
  SERVICE_RETAINABILITY = 'SERVICE_RETAINABILITY',
  SERVICE_MOBILITY = 'SERVICE_MOBILITY',
  SERVICE_INTEGRITY = 'SERVICE_INTEGRITY',
  
  // Technology-specific KPIs
  HANDOVER_SUCCESS_RATE = 'HANDOVER_SUCCESS_RATE',
  CALL_DROP_RATE = 'CALL_DROP_RATE',
  CONNECTION_SETUP_SUCCESS_RATE = 'CONNECTION_SETUP_SUCCESS_RATE',
  PAGING_SUCCESS_RATE = 'PAGING_SUCCESS_RATE',
  
  // Custom KPIs
  CUSTOM = 'CUSTOM'
}

/**
 * KPI collection result
 */
export interface KPICollection {
  /** Node identifier */
  nodeId: string;
  /** Collection timestamp */
  timestamp: Date;
  /** Collected KPIs */
  kpis: KPIValue[];
  /** Collection metadata */
  metadata: CollectionMetadata;
}

/**
 * KPI value
 */
export interface KPIValue {
  /** KPI type */
  type: KPIType;
  /** KPI name */
  name: string;
  /** KPI value */
  value: number;
  /** Value unit */
  unit: string;
  /** Data quality indicator */
  quality: DataQuality;
  /** Additional attributes */
  attributes: Record<string, any>;
}

/**
 * Data quality enumeration
 */
export enum DataQuality {
  GOOD = 'GOOD',
  FAIR = 'FAIR',
  POOR = 'POOR',
  INVALID = 'INVALID'
}

/**
 * Collection metadata
 */
export interface CollectionMetadata {
  /** Collection method */
  collectionMethod: 'SNMP' | 'REST_API' | 'CLI' | 'FILE_TRANSFER' | 'STREAMING';
  /** Collection duration in milliseconds */
  collectionDuration: number;
  /** Source system */
  sourceSystem: string;
  /** Data completeness percentage */
  completeness: number;
  /** Collection errors */
  errors: CollectionError[];
}

/**
 * Collection error
 */
export interface CollectionError {
  /** Error code */
  code: string;
  /** Error message */
  message: string;
  /** Affected KPI types */
  affectedKPIs: KPIType[];
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
 * Aggregation settings
 */
export interface AggregationSettings {
  /** Aggregation method */
  method: AggregationMethod;
  /** Aggregation interval */
  interval: AggregationInterval;
  /** Fill missing values */
  fillMissing: boolean;
  /** Fill value for missing data */
  fillValue?: number;
}

/**
 * Aggregation method enumeration
 */
export enum AggregationMethod {
  AVERAGE = 'AVERAGE',
  SUM = 'SUM',
  MIN = 'MIN',
  MAX = 'MAX',
  COUNT = 'COUNT',
  MEDIAN = 'MEDIAN',
  PERCENTILE_95 = 'PERCENTILE_95',
  PERCENTILE_99 = 'PERCENTILE_99'
}

/**
 * Aggregation interval enumeration
 */
export enum AggregationInterval {
  MINUTE = 'MINUTE',
  FIVE_MINUTES = 'FIVE_MINUTES',
  FIFTEEN_MINUTES = 'FIFTEEN_MINUTES',
  HOUR = 'HOUR',
  DAY = 'DAY',
  WEEK = 'WEEK',
  MONTH = 'MONTH'
}

/**
 * Historical KPI data
 */
export interface HistoricalKPIData {
  /** Node identifier */
  nodeId: string;
  /** Time range */
  timeRange: TimeRange;
  /** Aggregation settings used */
  aggregationSettings: AggregationSettings;
  /** Historical data points */
  dataPoints: HistoricalDataPoint[];
  /** Data summary statistics */
  summary: DataSummary;
}

/**
 * Historical data point
 */
export interface HistoricalDataPoint {
  /** Timestamp */
  timestamp: Date;
  /** KPI values */
  kpis: KPIValue[];
}

/**
 * Data summary
 */
export interface DataSummary {
  /** Total data points */
  totalDataPoints: number;
  /** Missing data points */
  missingDataPoints: number;
  /** Data completeness percentage */
  completeness: number;
  /** Summary statistics by KPI */
  kpiSummaries: Record<string, KPISummary>;
}

/**
 * KPI summary statistics
 */
export interface KPISummary {
  /** KPI type */
  kpiType: KPIType;
  /** Minimum value */
  min: number;
  /** Maximum value */
  max: number;
  /** Average value */
  average: number;
  /** Median value */
  median: number;
  /** Standard deviation */
  stdDev: number;
  /** 95th percentile */
  percentile95: number;
  /** Trend direction */
  trend: 'INCREASING' | 'DECREASING' | 'STABLE';
}

/**
 * Report criteria
 */
export interface ReportCriteria {
  /** Report name */
  name: string;
  /** Report description */
  description: string;
  /** Report type */
  type: ReportType;
  /** Target nodes */
  nodeIds: string[];
  /** Time range */
  timeRange: TimeRange;
  /** KPI types to include */
  kpiTypes: KPIType[];
  /** Report format */
  format: ReportFormat;
  /** Template to use */
  templateId?: string;
  /** Additional parameters */
  parameters: Record<string, any>;
}

/**
 * Report type enumeration
 */
export enum ReportType {
  PERFORMANCE_SUMMARY = 'PERFORMANCE_SUMMARY',
  KPI_ANALYSIS = 'KPI_ANALYSIS',
  TRENDING_REPORT = 'TRENDING_REPORT',
  COMPARISON_REPORT = 'COMPARISON_REPORT',
  ANOMALY_REPORT = 'ANOMALY_REPORT',
  SLA_REPORT = 'SLA_REPORT',
  CAPACITY_REPORT = 'CAPACITY_REPORT',
  AVAILABILITY_REPORT = 'AVAILABILITY_REPORT',
  CUSTOM = 'CUSTOM'
}

/**
 * Report format enumeration
 */
export enum ReportFormat {
  PDF = 'PDF',
  HTML = 'HTML',
  EXCEL = 'EXCEL',
  CSV = 'CSV',
  JSON = 'JSON'
}

/**
 * Generated report
 */
export interface Report {
  /** Report ID */
  reportId: string;
  /** Report metadata */
  metadata: ReportMetadata;
  /** Report content */
  content: ReportContent;
  /** Generation statistics */
  statistics: ReportStatistics;
}

/**
 * Report metadata
 */
export interface ReportMetadata {
  /** Report name */
  name: string;
  /** Report description */
  description: string;
  /** Report type */
  type: ReportType;
  /** Report format */
  format: ReportFormat;
  /** Generation timestamp */
  generatedAt: Date;
  /** Generated by */
  generatedBy: string;
  /** Time range covered */
  timeRange: TimeRange;
  /** Version */
  version: string;
}

/**
 * Report content
 */
export interface ReportContent {
  /** Executive summary */
  executiveSummary: string;
  /** Report sections */
  sections: ReportSection[];
  /** Charts and visualizations */
  charts: Chart[];
  /** Raw data (if requested) */
  rawData?: any[];
  /** Recommendations */
  recommendations: Recommendation[];
}

/**
 * Report section
 */
export interface ReportSection {
  /** Section title */
  title: string;
  /** Section content */
  content: string;
  /** Section data */
  data: any[];
  /** Section charts */
  charts: Chart[];
  /** Sub-sections */
  subSections?: ReportSection[];
}

/**
 * Chart definition
 */
export interface Chart {
  /** Chart ID */
  chartId: string;
  /** Chart title */
  title: string;
  /** Chart type */
  type: ChartType;
  /** Chart data */
  data: ChartData;
  /** Chart configuration */
  configuration: ChartConfiguration;
}

/**
 * Chart type enumeration
 */
export enum ChartType {
  LINE = 'LINE',
  BAR = 'BAR',
  PIE = 'PIE',
  SCATTER = 'SCATTER',
  HEATMAP = 'HEATMAP',
  GAUGE = 'GAUGE',
  TABLE = 'TABLE'
}

/**
 * Chart data
 */
export interface ChartData {
  /** Data series */
  series: DataSeries[];
  /** Chart labels */
  labels: string[];
  /** Data categories */
  categories?: string[];
}

/**
 * Data series
 */
export interface DataSeries {
  /** Series name */
  name: string;
  /** Series data points */
  data: number[];
  /** Series color */
  color?: string;
  /** Series type (for mixed charts) */
  type?: ChartType;
}

/**
 * Chart configuration
 */
export interface ChartConfiguration {
  /** Chart width */
  width?: number;
  /** Chart height */
  height?: number;
  /** Show legend */
  showLegend: boolean;
  /** Show grid */
  showGrid: boolean;
  /** X-axis configuration */
  xAxis: AxisConfiguration;
  /** Y-axis configuration */
  yAxis: AxisConfiguration;
  /** Additional options */
  options: Record<string, any>;
}

/**
 * Axis configuration
 */
export interface AxisConfiguration {
  /** Axis title */
  title: string;
  /** Show axis */
  show: boolean;
  /** Axis range */
  range?: [number, number];
  /** Axis format */
  format?: string;
}

/**
 * Recommendation
 */
export interface Recommendation {
  /** Recommendation ID */
  recommendationId: string;
  /** Recommendation title */
  title: string;
  /** Recommendation description */
  description: string;
  /** Priority level */
  priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  /** Category */
  category: string;
  /** Impact assessment */
  impact: string;
  /** Implementation effort */
  effort: 'LOW' | 'MEDIUM' | 'HIGH';
  /** Affected resources */
  affectedResources: string[];
}

/**
 * Report statistics
 */
export interface ReportStatistics {
  /** Generation time in seconds */
  generationTime: number;
  /** Data points processed */
  dataPointsProcessed: number;
  /** Report size in bytes */
  reportSize: number;
  /** Number of charts generated */
  chartsGenerated: number;
  /** Data coverage percentage */
  dataCoverage: number;
}

/**
 * Report schedule
 */
export interface ReportSchedule {
  /** Schedule frequency */
  frequency: ScheduleFrequency;
  /** Schedule time */
  time: string; // HH:MM format
  /** Schedule days (for weekly/monthly) */
  days?: number[];
  /** Time zone */
  timeZone: string;
  /** Start date */
  startDate: Date;
  /** End date (optional) */
  endDate?: Date;
}

/**
 * Schedule frequency enumeration
 */
export enum ScheduleFrequency {
  DAILY = 'DAILY',
  WEEKLY = 'WEEKLY',
  MONTHLY = 'MONTHLY',
  QUARTERLY = 'QUARTERLY'
}

/**
 * Report filters
 */
export interface ReportFilters {
  /** Filter by report type */
  type?: ReportType;
  /** Filter by generated date range */
  generatedDateRange?: TimeRange;
  /** Filter by generator */
  generatedBy?: string;
  /** Text search in report name/description */
  search?: string;
}

/**
 * Report summary
 */
export interface ReportSummary {
  /** Report ID */
  reportId: string;
  /** Report name */
  name: string;
  /** Report type */
  type: ReportType;
  /** Report format */
  format: ReportFormat;
  /** Generation timestamp */
  generatedAt: Date;
  /** Generated by */
  generatedBy: string;
  /** Report size */
  size: number;
  /** Status */
  status: 'GENERATING' | 'COMPLETED' | 'FAILED';
}

/**
 * Alert configuration
 */
export interface AlertConfig {
  /** Alert name */
  name: string;
  /** Alert description */
  description: string;
  /** Target nodes */
  nodeIds: string[];
  /** KPI type to monitor */
  kpiType: KPIType;
  /** Alert conditions */
  conditions: AlertCondition[];
  /** Alert severity */
  severity: AlertSeverity;
  /** Notification settings */
  notifications: AlertNotification[];
  /** Alert enabled */
  enabled: boolean;
  /** Suppression settings */
  suppressionSettings?: AlertSuppression;
}

/**
 * Alert condition
 */
export interface AlertCondition {
  /** Condition type */
  type: ConditionType;
  /** Threshold value */
  threshold: number;
  /** Duration threshold must be exceeded */
  duration: number; // in seconds
  /** Comparison operator */
  operator: ComparisonOperator;
}

/**
 * Condition type enumeration
 */
export enum ConditionType {
  THRESHOLD = 'THRESHOLD',
  RATE_OF_CHANGE = 'RATE_OF_CHANGE',
  DEVIATION = 'DEVIATION',
  PATTERN_MATCH = 'PATTERN_MATCH'
}

/**
 * Comparison operator enumeration
 */
export enum ComparisonOperator {
  GREATER_THAN = 'GREATER_THAN',
  LESS_THAN = 'LESS_THAN',
  EQUAL_TO = 'EQUAL_TO',
  NOT_EQUAL_TO = 'NOT_EQUAL_TO',
  GREATER_THAN_OR_EQUAL = 'GREATER_THAN_OR_EQUAL',
  LESS_THAN_OR_EQUAL = 'LESS_THAN_OR_EQUAL'
}

/**
 * Alert severity enumeration
 */
export enum AlertSeverity {
  CRITICAL = 'CRITICAL',
  MAJOR = 'MAJOR',
  MINOR = 'MINOR',
  WARNING = 'WARNING',
  INFO = 'INFO'
}

/**
 * Alert notification
 */
export interface AlertNotification {
  /** Notification type */
  type: NotificationType;
  /** Notification target */
  target: string;
  /** Notification template */
  template?: string;
  /** Additional parameters */
  parameters: Record<string, any>;
}

/**
 * Notification type enumeration
 */
export enum NotificationType {
  EMAIL = 'EMAIL',
  SMS = 'SMS',
  WEBHOOK = 'WEBHOOK',
  SNMP_TRAP = 'SNMP_TRAP',
  SYSLOG = 'SYSLOG',
  SLACK = 'SLACK',
  TEAMS = 'TEAMS'
}

/**
 * Alert suppression settings
 */
export interface AlertSuppression {
  /** Enable suppression */
  enabled: boolean;
  /** Suppression duration in seconds */
  duration: number;
  /** Maximum alerts per duration */
  maxAlerts: number;
}

/**
 * Alert instance
 */
export interface Alert {
  /** Alert ID */
  alertId: string;
  /** Alert configuration ID */
  alertConfigId: string;
  /** Node ID */
  nodeId: string;
  /** Alert name */
  name: string;
  /** Alert description */
  description: string;
  /** KPI type */
  kpiType: KPIType;
  /** Current value that triggered the alert */
  currentValue: number;
  /** Threshold value */
  thresholdValue: number;
  /** Alert severity */
  severity: AlertSeverity;
  /** Alert status */
  status: AlertStatus;
  /** First occurrence time */
  firstOccurrence: Date;
  /** Last occurrence time */
  lastOccurrence: Date;
  /** Occurrence count */
  occurrenceCount: number;
  /** Acknowledgment details */
  acknowledgment?: AlertAcknowledgment;
  /** Clear details */
  clearDetails?: AlertClearDetails;
}

/**
 * Alert status enumeration
 */
export enum AlertStatus {
  ACTIVE = 'ACTIVE',
  ACKNOWLEDGED = 'ACKNOWLEDGED',
  CLEARED = 'CLEARED',
  SUPPRESSED = 'SUPPRESSED'
}

/**
 * Alert acknowledgment
 */
export interface AlertAcknowledgment {
  /** Acknowledged by */
  acknowledgedBy: string;
  /** Acknowledgment time */
  acknowledgedAt: Date;
  /** Acknowledgment comment */
  comment: string;
}

/**
 * Alert clear details
 */
export interface AlertClearDetails {
  /** Cleared by */
  clearedBy: string;
  /** Clear time */
  clearedAt: Date;
  /** Clear reason */
  reason: string;
  /** Auto-cleared indicator */
  autoCleared: boolean;
}

/**
 * Alert filters
 */
export interface AlertFilters {
  /** Filter by node IDs */
  nodeIds?: string[];
  /** Filter by KPI types */
  kpiTypes?: KPIType[];
  /** Filter by severity */
  severities?: AlertSeverity[];
  /** Filter by status */
  statuses?: AlertStatus[];
  /** Filter by time range */
  timeRange?: TimeRange;
}

/**
 * Performance metrics
 */
export interface PerformanceMetrics {
  /** Node identifier */
  nodeId: string;
  /** Metrics timestamp */
  timestamp: Date;
  /** Overall performance score */
  overallScore: number;
  /** Category scores */
  categoryScores: CategoryScore[];
  /** Key performance indicators */
  keyIndicators: KeyIndicator[];
  /** Performance trends */
  trends: PerformanceTrend[];
}

/**
 * Category score
 */
export interface CategoryScore {
  /** Category name */
  category: string;
  /** Score value (0-100) */
  score: number;
  /** Score trend */
  trend: 'IMPROVING' | 'DEGRADING' | 'STABLE';
  /** Contributing factors */
  factors: string[];
}

/**
 * Key indicator
 */
export interface KeyIndicator {
  /** Indicator name */
  name: string;
  /** Current value */
  value: number;
  /** Target value */
  target?: number;
  /** Unit */
  unit: string;
  /** Status */
  status: 'GOOD' | 'WARNING' | 'CRITICAL';
}

/**
 * Performance trend
 */
export interface PerformanceTrend {
  /** Metric name */
  metric: string;
  /** Trend direction */
  direction: 'UP' | 'DOWN' | 'STABLE';
  /** Trend strength */
  strength: 'WEAK' | 'MODERATE' | 'STRONG';
  /** Trend confidence */
  confidence: number; // 0-1
}

/**
 * Network performance summary
 */
export interface NetworkPerformanceSummary {
  /** Network identifier */
  networkId: string;
  /** Summary timestamp */
  timestamp: Date;
  /** Total nodes */
  totalNodes: number;
  /** Healthy nodes */
  healthyNodes: number;
  /** Overall network health */
  overallHealth: number;
  /** Top performing nodes */
  topPerformingNodes: NodePerformanceSummary[];
  /** Underperforming nodes */
  underperformingNodes: NodePerformanceSummary[];
  /** Network-wide KPIs */
  networkKPIs: KPIValue[];
  /** Performance distribution */
  performanceDistribution: PerformanceDistribution;
}

/**
 * Node performance summary
 */
export interface NodePerformanceSummary {
  /** Node identifier */
  nodeId: string;
  /** Node name */
  nodeName: string;
  /** Performance score */
  performanceScore: number;
  /** Key issues */
  keyIssues: string[];
}

/**
 * Performance distribution
 */
export interface PerformanceDistribution {
  /** Excellent performance (90-100) */
  excellent: number;
  /** Good performance (70-89) */
  good: number;
  /** Fair performance (50-69) */
  fair: number;
  /** Poor performance (0-49) */
  poor: number;
}

/**
 * Anomaly detection settings
 */
export interface AnomalyDetectionSettings {
  /** Detection algorithm */
  algorithm: AnomalyDetectionAlgorithm;
  /** Sensitivity level */
  sensitivity: 'LOW' | 'MEDIUM' | 'HIGH';
  /** Training period */
  trainingPeriod: number; // in hours
  /** KPI types to monitor */
  monitoredKPIs: KPIType[];
  /** Detection threshold */
  threshold: number;
  /** Minimum anomaly duration */
  minimumDuration: number; // in seconds
}

/**
 * Anomaly detection algorithm enumeration
 */
export enum AnomalyDetectionAlgorithm {
  STATISTICAL = 'STATISTICAL',
  MACHINE_LEARNING = 'MACHINE_LEARNING',
  ISOLATION_FOREST = 'ISOLATION_FOREST',
  AUTOENCODER = 'AUTOENCODER',
  TIME_SERIES = 'TIME_SERIES'
}

/**
 * Anomaly detection result
 */
export interface AnomalyDetectionResult {
  /** Detection timestamp */
  timestamp: Date;
  /** Detected anomalies */
  anomalies: Anomaly[];
  /** Detection metadata */
  metadata: AnomalyDetectionMetadata;
}

/**
 * Anomaly
 */
export interface Anomaly {
  /** Anomaly ID */
  anomalyId: string;
  /** Node identifier */
  nodeId: string;
  /** KPI type */
  kpiType: KPIType;
  /** Anomaly type */
  type: AnomalyType;
  /** Severity */
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  /** Start time */
  startTime: Date;
  /** End time */
  endTime?: Date;
  /** Expected value */
  expectedValue: number;
  /** Actual value */
  actualValue: number;
  /** Deviation score */
  deviationScore: number;
  /** Confidence score */
  confidence: number;
  /** Description */
  description: string;
  /** Root cause analysis */
  rootCause?: string[];
}

/**
 * Anomaly type enumeration
 */
export enum AnomalyType {
  POINT_ANOMALY = 'POINT_ANOMALY',
  CONTEXTUAL_ANOMALY = 'CONTEXTUAL_ANOMALY',
  COLLECTIVE_ANOMALY = 'COLLECTIVE_ANOMALY',
  TREND_ANOMALY = 'TREND_ANOMALY',
  SEASONAL_ANOMALY = 'SEASONAL_ANOMALY'
}

/**
 * Anomaly detection metadata
 */
export interface AnomalyDetectionMetadata {
  /** Algorithm used */
  algorithmUsed: AnomalyDetectionAlgorithm;
  /** Model version */
  modelVersion: string;
  /** Detection duration */
  detectionDuration: number;
  /** Training data size */
  trainingDataSize: number;
  /** Model accuracy */
  modelAccuracy?: number;
}

/**
 * Service level metrics
 */
export interface ServiceLevelMetrics {
  /** Service identifier */
  serviceId: string;
  /** Time range */
  timeRange: TimeRange;
  /** Availability percentage */
  availability: number;
  /** Mean time to repair (MTTR) */
  mttr: number;
  /** Mean time between failures (MTBF) */
  mtbf: number;
  /** Service level indicators */
  slis: ServiceLevelIndicator[];
  /** Service level objectives */
  slos: ServiceLevelObjective[];
}

/**
 * Service level indicator
 */
export interface ServiceLevelIndicator {
  /** SLI name */
  name: string;
  /** Current value */
  value: number;
  /** Unit */
  unit: string;
  /** Target value */
  target: number;
  /** Status */
  status: 'MET' | 'AT_RISK' | 'VIOLATED';
}

/**
 * Service level objective
 */
export interface ServiceLevelObjective {
  /** SLO name */
  name: string;
  /** Target percentage */
  target: number;
  /** Current achievement */
  achievement: number;
  /** Error budget remaining */
  errorBudgetRemaining: number;
  /** Status */
  status: 'HEALTHY' | 'WARNING' | 'CRITICAL';
}

/**
 * SLA configuration
 */
export interface SLAConfig {
  /** SLA name */
  name: string;
  /** Service identifier */
  serviceId: string;
  /** SLA targets */
  targets: SLATarget[];
  /** Monitoring period */
  monitoringPeriod: 'MONTHLY' | 'QUARTERLY' | 'YEARLY';
  /** Penalty clauses */
  penaltyClauses: PenaltyClause[];
}

/**
 * SLA target
 */
export interface SLATarget {
  /** Metric name */
  metric: string;
  /** Target value */
  target: number;
  /** Unit */
  unit: string;
  /** Measurement method */
  measurementMethod: string;
}

/**
 * Penalty clause
 */
export interface PenaltyClause {
  /** Condition */
  condition: string;
  /** Penalty amount */
  penalty: number;
  /** Penalty type */
  penaltyType: 'FIXED' | 'PERCENTAGE';
}

/**
 * SLA compliance status
 */
export interface SLAComplianceStatus {
  /** SLA monitor ID */
  slaMonitorId: string;
  /** Compliance period */
  period: TimeRange;
  /** Overall compliance */
  overallCompliance: number;
  /** Target compliance results */
  targetResults: SLATargetResult[];
  /** Penalty calculations */
  penaltyCalculations: PenaltyCalculation[];
}

/**
 * SLA target result
 */
export interface SLATargetResult {
  /** Target metric */
  metric: string;
  /** Target value */
  target: number;
  /** Actual value */
  actual: number;
  /** Compliance percentage */
  compliance: number;
  /** Status */
  status: 'MET' | 'VIOLATED';
}

/**
 * Penalty calculation
 */
export interface PenaltyCalculation {
  /** Clause description */
  clause: string;
  /** Penalty amount */
  amount: number;
  /** Applicable period */
  period: TimeRange;
}

/**
 * Dashboard configuration
 */
export interface DashboardConfig {
  /** Dashboard name */
  name: string;
  /** Dashboard description */
  description: string;
  /** Dashboard widgets */
  widgets: DashboardWidget[];
  /** Layout configuration */
  layout: DashboardLayout;
  /** Refresh interval */
  refreshInterval: number;
  /** Auto-refresh enabled */
  autoRefresh: boolean;
}

/**
 * Dashboard widget
 */
export interface DashboardWidget {
  /** Widget ID */
  widgetId: string;
  /** Widget title */
  title: string;
  /** Widget type */
  type: WidgetType;
  /** Data source configuration */
  dataSource: WidgetDataSource;
  /** Widget size */
  size: WidgetSize;
  /** Widget position */
  position: WidgetPosition;
  /** Widget configuration */
  configuration: Record<string, any>;
}

/**
 * Widget type enumeration
 */
export enum WidgetType {
  KPI_METRIC = 'KPI_METRIC',
  LINE_CHART = 'LINE_CHART',
  BAR_CHART = 'BAR_CHART',
  PIE_CHART = 'PIE_CHART',
  GAUGE = 'GAUGE',
  TABLE = 'TABLE',
  HEATMAP = 'HEATMAP',
  MAP = 'MAP',
  ALERT_LIST = 'ALERT_LIST',
  STATUS_INDICATOR = 'STATUS_INDICATOR'
}

/**
 * Widget data source
 */
export interface WidgetDataSource {
  /** Data source type */
  type: 'KPI' | 'ALERT' | 'REPORT' | 'CUSTOM';
  /** Source parameters */
  parameters: Record<string, any>;
  /** Refresh interval */
  refreshInterval: number;
}

/**
 * Widget size
 */
export interface WidgetSize {
  /** Width in grid units */
  width: number;
  /** Height in grid units */
  height: number;
}

/**
 * Widget position
 */
export interface WidgetPosition {
  /** X coordinate */
  x: number;
  /** Y coordinate */
  y: number;
}

/**
 * Dashboard layout
 */
export interface DashboardLayout {
  /** Grid columns */
  columns: number;
  /** Grid rows */
  rows: number;
  /** Row height */
  rowHeight: number;
  /** Margins */
  margins: [number, number];
}

/**
 * Dashboard data
 */
export interface DashboardData {
  /** Dashboard ID */
  dashboardId: string;
  /** Data timestamp */
  timestamp: Date;
  /** Widget data */
  widgetData: Record<string, WidgetData>;
}

/**
 * Widget data
 */
export interface WidgetData {
  /** Widget ID */
  widgetId: string;
  /** Data values */
  data: any;
  /** Last update time */
  lastUpdate: Date;
  /** Data status */
  status: 'LOADING' | 'SUCCESS' | 'ERROR';
  /** Error message if any */
  error?: string;
}

/**
 * Data export request
 */
export interface DataExportRequest {
  /** Export type */
  type: ExportType;
  /** Time range */
  timeRange: TimeRange;
  /** Node IDs */
  nodeIds: string[];
  /** KPI types */
  kpiTypes: KPIType[];
  /** Export format */
  format: ExportFormat;
  /** Additional parameters */
  parameters: Record<string, any>;
}

/**
 * Export type enumeration
 */
export enum ExportType {
  KPI_DATA = 'KPI_DATA',
  ALERT_DATA = 'ALERT_DATA',
  REPORT_DATA = 'REPORT_DATA',
  CONFIGURATION_DATA = 'CONFIGURATION_DATA'
}

/**
 * Export format enumeration
 */
export enum ExportFormat {
  CSV = 'CSV',
  JSON = 'JSON',
  XML = 'XML',
  PARQUET = 'PARQUET'
}

/**
 * Data export result
 */
export interface DataExportResult {
  /** Export ID */
  exportId: string;
  /** Export status */
  status: 'IN_PROGRESS' | 'COMPLETED' | 'FAILED';
  /** File location */
  fileLocation?: string;
  /** File size in bytes */
  fileSize?: number;
  /** Export metadata */
  metadata: ExportMetadata;
}

/**
 * Export metadata
 */
export interface ExportMetadata {
  /** Export start time */
  startTime: Date;
  /** Export end time */
  endTime?: Date;
  /** Records exported */
  recordsExported: number;
  /** Compression used */
  compression?: string;
  /** Checksum */
  checksum?: string;
}

/**
 * Data import request
 */
export interface DataImportRequest {
  /** Import type */
  type: ImportType;
  /** File location */
  fileLocation: string;
  /** Import format */
  format: ImportFormat;
  /** Import options */
  options: ImportOptions;
}

/**
 * Import type enumeration
 */
export enum ImportType {
  KPI_DATA = 'KPI_DATA',
  CONFIGURATION_DATA = 'CONFIGURATION_DATA',
  ALERT_RULES = 'ALERT_RULES'
}

/**
 * Import format enumeration
 */
export enum ImportFormat {
  CSV = 'CSV',
  JSON = 'JSON',
  XML = 'XML'
}

/**
 * Import options
 */
export interface ImportOptions {
  /** Skip validation */
  skipValidation: boolean;
  /** Merge strategy */
  mergeStrategy: 'REPLACE' | 'MERGE' | 'APPEND';
  /** Backup before import */
  backup: boolean;
}

/**
 * Data import result
 */
export interface DataImportResult {
  /** Import ID */
  importId: string;
  /** Import status */
  status: 'IN_PROGRESS' | 'COMPLETED' | 'FAILED';
  /** Records imported */
  recordsImported: number;
  /** Import errors */
  errors: ImportError[];
  /** Import metadata */
  metadata: ImportMetadata;
}

/**
 * Import error
 */
export interface ImportError {
  /** Error code */
  code: string;
  /** Error message */
  message: string;
  /** Line number */
  lineNumber?: number;
  /** Field name */
  field?: string;
}

/**
 * Import metadata
 */
export interface ImportMetadata {
  /** Import start time */
  startTime: Date;
  /** Import end time */
  endTime?: Date;
  /** Source file size */
  sourceFileSize: number;
  /** Validation results */
  validationResults: ValidationResults;
}

/**
 * Validation results
 */
export interface ValidationResults {
  /** Total records */
  totalRecords: number;
  /** Valid records */
  validRecords: number;
  /** Invalid records */
  invalidRecords: number;
  /** Validation errors */
  errors: ValidationError[];
}

/**
 * Validation error
 */
export interface ValidationError {
  /** Record number */
  recordNumber: number;
  /** Field name */
  field: string;
  /** Error message */
  message: string;
  /** Error code */
  code: string;
}

/**
 * Monitoring service status
 */
export interface MonitoringServiceStatus {
  /** Service health */
  health: 'HEALTHY' | 'DEGRADED' | 'UNHEALTHY';
  /** Service version */
  version: string;
  /** Uptime in seconds */
  uptime: number;
  /** Active connections */
  activeConnections: number;
  /** Data collection rate */
  dataCollectionRate: number;
  /** Storage utilization */
  storageUtilization: number;
  /** Component statuses */
  components: ComponentStatus[];
}

/**
 * Component status
 */
export interface ComponentStatus {
  /** Component name */
  name: string;
  /** Component status */
  status: 'UP' | 'DOWN' | 'DEGRADED';
  /** Status message */
  message: string;
  /** Last check time */
  lastCheckTime: Date;
}

/**
 * Retention policy
 */
export interface RetentionPolicy {
  /** Policy name */
  name: string;
  /** Data type */
  dataType: 'KPI' | 'ALERT' | 'EVENT' | 'LOG';
  /** Retention period in days */
  retentionPeriod: number;
  /** Aggregation rules */
  aggregationRules: AggregationRule[];
  /** Archive settings */
  archiveSettings?: ArchiveSettings;
}

/**
 * Aggregation rule
 */
export interface AggregationRule {
  /** Age threshold in days */
  ageThreshold: number;
  /** Aggregation method */
  method: AggregationMethod;
  /** Aggregation interval */
  interval: AggregationInterval;
}

/**
 * Archive settings
 */
export interface ArchiveSettings {
  /** Enable archiving */
  enabled: boolean;
  /** Archive location */
  location: string;
  /** Compression enabled */
  compression: boolean;
  /** Encryption enabled */
  encryption: boolean;
}

/**
 * Monitoring event
 */
export interface MonitoringEvent {
  /** Event ID */
  eventId: string;
  /** Event type */
  eventType: MonitoringEventType;
  /** Event timestamp */
  timestamp: Date;
  /** Event source */
  source: string;
  /** Event data */
  data: Record<string, any>;
  /** Event severity */
  severity: 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';
}

/**
 * Monitoring event type enumeration
 */
export enum MonitoringEventType {
  KPI_COLLECTED = 'KPI_COLLECTED',
  ALERT_TRIGGERED = 'ALERT_TRIGGERED',
  ALERT_CLEARED = 'ALERT_CLEARED',
  ANOMALY_DETECTED = 'ANOMALY_DETECTED',
  REPORT_GENERATED = 'REPORT_GENERATED',
  THRESHOLD_BREACH = 'THRESHOLD_BREACH',
  SERVICE_HEALTH_CHANGE = 'SERVICE_HEALTH_CHANGE',
  DATA_COLLECTION_FAILURE = 'DATA_COLLECTION_FAILURE'
}