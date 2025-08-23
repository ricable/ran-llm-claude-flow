import { KPIMetric, PerformanceData, Alarm, NodeConfiguration } from '../types';
import { ConfigurationManager } from '../core/ConfigurationManager';
import { CMEditClient } from '../core/CMEditClient';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';

/**
 * Performance monitoring system for RAN nodes
 * Collects KPIs, generates alarms, and provides analytics
 */
export class PerformanceMonitor {
  private logger: Logger;
  private configManager: ConfigurationManager;
  private cmEditClient: CMEditClient;
  private kpiCache: Map<string, KPIMetric[]> = new Map();
  private alarmCache: Map<string, Alarm[]> = new Map();
  private monitoringInterval: NodeJS.Timeout | null = null;
  private isMonitoring: boolean = false;

  constructor(
    configManager?: ConfigurationManager,
    cmEditClient?: CMEditClient
  ) {
    this.logger = createLogger({
      level: 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json()
      ),
      defaultMeta: { service: 'PerformanceMonitor' },
      transports: [
        new transports.Console({
          format: format.combine(
            format.colorize(),
            format.simple()
          )
        })
      ]
    });

    this.configManager = configManager || new ConfigurationManager();
    this.cmEditClient = cmEditClient || new CMEditClient();
  }

  /**
   * Start performance monitoring
   */
  public async startMonitoring(intervalMs: number = 60000): Promise<void> {
    if (this.isMonitoring) {
      this.logger.warn('Performance monitoring is already running');
      return;
    }

    this.logger.info('Starting performance monitoring', { intervalMs });
    this.isMonitoring = true;

    this.monitoringInterval = setInterval(async () => {
      try {
        await this.collectAllMetrics();
      } catch (error) {
        this.logger.error('Error during metric collection', { error });
      }
    }, intervalMs);
  }

  /**
   * Stop performance monitoring
   */
  public stopMonitoring(): void {
    if (!this.isMonitoring) {
      this.logger.warn('Performance monitoring is not running');
      return;
    }

    this.logger.info('Stopping performance monitoring');
    this.isMonitoring = false;

    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
  }

  /**
   * Collect performance data for a specific node
   */
  public async collectNodeMetrics(nodeId: string): Promise<PerformanceData> {
    this.logger.info('Collecting metrics for node', { nodeId });

    try {
      // Get node configuration to determine type
      const config = await this.configManager.loadConfiguration(nodeId) as NodeConfiguration;
      
      let metrics: KPIMetric[] = [];
      let alarms: Alarm[] = [];

      if (config.nodeType === 'eNodeB') {
        metrics = await this.collectLTEMetrics(nodeId);
      } else if (config.nodeType === 'gNodeB') {
        metrics = await this.collectNRMetrics(nodeId);
      }

      // Collect alarms
      alarms = await this.collectAlarms(nodeId);

      // Cache the metrics
      this.kpiCache.set(nodeId, metrics);
      this.alarmCache.set(nodeId, alarms);

      const performanceData: PerformanceData = {
        nodeId,
        timestamp: new Date(),
        metrics,
        alarms
      };

      // Analyze metrics for threshold violations
      await this.analyzeMetrics(performanceData);

      this.logger.info('Metrics collected successfully', { 
        nodeId, 
        metricCount: metrics.length,
        alarmCount: alarms.length
      });

      return performanceData;

    } catch (error) {
      this.logger.error('Failed to collect node metrics', { nodeId, error });
      throw error;
    }
  }

  /**
   * Collect LTE-specific metrics
   */
  private async collectLTEMetrics(nodeId: string): Promise<KPIMetric[]> {
    const metrics: KPIMetric[] = [];
    const timestamp = new Date();

    try {
      // Cell availability
      metrics.push({
        name: 'Cell_Availability',
        value: this.generateRandomValue(95, 100),
        unit: '%',
        timestamp,
        nodeId
      });

      // RSRP (Reference Signal Received Power)
      metrics.push({
        name: 'RSRP_Average',
        value: this.generateRandomValue(-110, -80),
        unit: 'dBm',
        timestamp,
        nodeId
      });

      // SINR (Signal to Interference plus Noise Ratio)
      metrics.push({
        name: 'SINR_Average',
        value: this.generateRandomValue(5, 25),
        unit: 'dB',
        timestamp,
        nodeId
      });

      // Throughput
      metrics.push({
        name: 'DL_Throughput_Average',
        value: this.generateRandomValue(20, 150),
        unit: 'Mbps',
        timestamp,
        nodeId
      });

      metrics.push({
        name: 'UL_Throughput_Average',
        value: this.generateRandomValue(10, 50),
        unit: 'Mbps',
        timestamp,
        nodeId
      });

      // Handover Success Rate
      metrics.push({
        name: 'Handover_Success_Rate',
        value: this.generateRandomValue(95, 99.5),
        unit: '%',
        timestamp,
        nodeId
      });

      // Call Drop Rate
      metrics.push({
        name: 'Call_Drop_Rate',
        value: this.generateRandomValue(0.1, 2.0),
        unit: '%',
        timestamp,
        nodeId
      });

      // Resource Block Utilization
      metrics.push({
        name: 'PRB_Utilization_DL',
        value: this.generateRandomValue(30, 85),
        unit: '%',
        timestamp,
        nodeId
      });

      metrics.push({
        name: 'PRB_Utilization_UL',
        value: this.generateRandomValue(20, 60),
        unit: '%',
        timestamp,
        nodeId
      });

      // Interference
      metrics.push({
        name: 'Interference_Level',
        value: this.generateRandomValue(-120, -100),
        unit: 'dBm',
        timestamp,
        nodeId
      });

      // Connected Users
      metrics.push({
        name: 'Active_Users',
        value: Math.floor(this.generateRandomValue(50, 200)),
        unit: 'count',
        timestamp,
        nodeId
      });

      // Latency
      metrics.push({
        name: 'User_Plane_Latency',
        value: this.generateRandomValue(15, 45),
        unit: 'ms',
        timestamp,
        nodeId
      });

    } catch (error) {
      this.logger.error('Failed to collect LTE metrics', { nodeId, error });
    }

    return metrics;
  }

  /**
   * Collect NR-specific metrics
   */
  private async collectNRMetrics(nodeId: string): Promise<KPIMetric[]> {
    const metrics: KPIMetric[] = [];
    const timestamp = new Date();

    try {
      // Cell availability
      metrics.push({
        name: 'Cell_Availability',
        value: this.generateRandomValue(97, 100),
        unit: '%',
        timestamp,
        nodeId
      });

      // RSRP (Reference Signal Received Power)
      metrics.push({
        name: 'SSB_RSRP_Average',
        value: this.generateRandomValue(-115, -85),
        unit: 'dBm',
        timestamp,
        nodeId
      });

      // SINR
      metrics.push({
        name: 'SSB_SINR_Average',
        value: this.generateRandomValue(8, 30),
        unit: 'dB',
        timestamp,
        nodeId
      });

      // Throughput (higher for NR)
      metrics.push({
        name: 'DL_Throughput_Average',
        value: this.generateRandomValue(100, 800),
        unit: 'Mbps',
        timestamp,
        nodeId
      });

      metrics.push({
        name: 'UL_Throughput_Average',
        value: this.generateRandomValue(50, 200),
        unit: 'Mbps',
        timestamp,
        nodeId
      });

      // Handover Success Rate
      metrics.push({
        name: 'Handover_Success_Rate',
        value: this.generateRandomValue(96, 99.8),
        unit: '%',
        timestamp,
        nodeId
      });

      // Session Drop Rate
      metrics.push({
        name: 'Session_Drop_Rate',
        value: this.generateRandomValue(0.05, 1.5),
        unit: '%',
        timestamp,
        nodeId
      });

      // Resource Block Utilization
      metrics.push({
        name: 'PRB_Utilization_DL',
        value: this.generateRandomValue(25, 75),
        unit: '%',
        timestamp,
        nodeId
      });

      metrics.push({
        name: 'PRB_Utilization_UL',
        value: this.generateRandomValue(15, 50),
        unit: '%',
        timestamp,
        nodeId
      });

      // Beam-related metrics
      metrics.push({
        name: 'Active_Beams',
        value: Math.floor(this.generateRandomValue(4, 16)),
        unit: 'count',
        timestamp,
        nodeId
      });

      metrics.push({
        name: 'Beam_RSRP_Average',
        value: this.generateRandomValue(-100, -70),
        unit: 'dBm',
        timestamp,
        nodeId
      });

      // NR-specific: Slice performance
      metrics.push({
        name: 'eMBB_Slice_Utilization',
        value: this.generateRandomValue(40, 80),
        unit: '%',
        timestamp,
        nodeId
      });

      metrics.push({
        name: 'URLLC_Slice_Utilization',
        value: this.generateRandomValue(5, 25),
        unit: '%',
        timestamp,
        nodeId
      });

      // Ultra-low latency
      metrics.push({
        name: 'User_Plane_Latency',
        value: this.generateRandomValue(1, 10),
        unit: 'ms',
        timestamp,
        nodeId
      });

      // Connected devices
      metrics.push({
        name: 'Active_UEs',
        value: Math.floor(this.generateRandomValue(100, 500)),
        unit: 'count',
        timestamp,
        nodeId
      });

      // Energy efficiency
      metrics.push({
        name: 'Energy_Efficiency',
        value: this.generateRandomValue(80, 95),
        unit: '%',
        timestamp,
        nodeId
      });

    } catch (error) {
      this.logger.error('Failed to collect NR metrics', { nodeId, error });
    }

    return metrics;
  }

  /**
   * Collect alarms for a node
   */
  private async collectAlarms(nodeId: string): Promise<Alarm[]> {
    const alarms: Alarm[] = [];

    try {
      // Simulate some alarms based on random conditions
      if (Math.random() < 0.1) { // 10% chance of high interference alarm
        alarms.push({
          id: this.generateAlarmId(),
          severity: 'MAJOR',
          type: 'High Interference',
          description: 'Interference level exceeds threshold',
          nodeId,
          timestamp: new Date(),
          acknowledged: false
        });
      }

      if (Math.random() < 0.05) { // 5% chance of equipment failure
        alarms.push({
          id: this.generateAlarmId(),
          severity: 'CRITICAL',
          type: 'Equipment Failure',
          description: 'RF unit failure detected',
          nodeId,
          timestamp: new Date(),
          acknowledged: false
        });
      }

      if (Math.random() < 0.15) { // 15% chance of high load
        alarms.push({
          id: this.generateAlarmId(),
          severity: 'MINOR',
          type: 'High Load',
          description: 'Resource utilization above 80%',
          nodeId,
          timestamp: new Date(),
          acknowledged: false
        });
      }

      if (Math.random() < 0.08) { // 8% chance of degraded performance
        alarms.push({
          id: this.generateAlarmId(),
          severity: 'WARNING',
          type: 'Performance Degradation',
          description: 'Throughput below expected levels',
          nodeId,
          timestamp: new Date(),
          acknowledged: false
        });
      }

    } catch (error) {
      this.logger.error('Failed to collect alarms', { nodeId, error });
    }

    return alarms;
  }

  /**
   * Collect metrics for all configured nodes
   */
  private async collectAllMetrics(): Promise<void> {
    try {
      const nodeIds = await this.configManager.listConfigurations();
      
      const promises = nodeIds.map(nodeId => 
        this.collectNodeMetrics(nodeId).catch(error => {
          this.logger.error('Failed to collect metrics for node', { nodeId, error });
          return null;
        })
      );

      await Promise.all(promises);
      
      this.logger.info('Completed metric collection cycle', { nodeCount: nodeIds.length });

    } catch (error) {
      this.logger.error('Failed to collect metrics for all nodes', { error });
    }
  }

  /**
   * Analyze metrics for threshold violations and anomalies
   */
  private async analyzeMetrics(performanceData: PerformanceData): Promise<void> {
    const { nodeId, metrics } = performanceData;
    const newAlarms: Alarm[] = [];

    for (const metric of metrics) {
      // Define thresholds for different metrics
      const thresholds = this.getMetricThresholds(metric.name);
      
      if (thresholds) {
        if (metric.value < thresholds.critical.min || metric.value > thresholds.critical.max) {
          newAlarms.push({
            id: this.generateAlarmId(),
            severity: 'CRITICAL',
            type: 'Threshold Violation',
            description: `${metric.name} is at critical level: ${metric.value}${metric.unit}`,
            nodeId,
            timestamp: new Date(),
            acknowledged: false
          });
        } else if (metric.value < thresholds.major.min || metric.value > thresholds.major.max) {
          newAlarms.push({
            id: this.generateAlarmId(),
            severity: 'MAJOR',
            type: 'Threshold Violation',
            description: `${metric.name} is at major level: ${metric.value}${metric.unit}`,
            nodeId,
            timestamp: new Date(),
            acknowledged: false
          });
        } else if (metric.value < thresholds.minor.min || metric.value > thresholds.minor.max) {
          newAlarms.push({
            id: this.generateAlarmId(),
            severity: 'MINOR',
            type: 'Threshold Violation',
            description: `${metric.name} is at minor level: ${metric.value}${metric.unit}`,
            nodeId,
            timestamp: new Date(),
            acknowledged: false
          });
        }
      }
    }

    // Add new alarms to cache
    if (newAlarms.length > 0) {
      const existingAlarms = this.alarmCache.get(nodeId) || [];
      this.alarmCache.set(nodeId, [...existingAlarms, ...newAlarms]);
      
      this.logger.warn('New alarms generated', { 
        nodeId, 
        alarmCount: newAlarms.length 
      });
    }
  }

  /**
   * Get threshold definitions for metrics
   */
  private getMetricThresholds(metricName: string): any {
    const thresholds: Record<string, any> = {
      'Cell_Availability': {
        critical: { min: 0, max: 90 },
        major: { min: 90, max: 95 },
        minor: { min: 95, max: 98 }
      },
      'RSRP_Average': {
        critical: { min: -130, max: -120 },
        major: { min: -120, max: -110 },
        minor: { min: -110, max: -100 }
      },
      'SINR_Average': {
        critical: { min: -5, max: 0 },
        major: { min: 0, max: 5 },
        minor: { min: 5, max: 10 }
      },
      'Call_Drop_Rate': {
        critical: { min: 5, max: 100 },
        major: { min: 3, max: 5 },
        minor: { min: 2, max: 3 }
      },
      'Session_Drop_Rate': {
        critical: { min: 3, max: 100 },
        major: { min: 2, max: 3 },
        minor: { min: 1, max: 2 }
      },
      'PRB_Utilization_DL': {
        critical: { min: 95, max: 100 },
        major: { min: 85, max: 95 },
        minor: { min: 80, max: 85 }
      },
      'User_Plane_Latency': {
        critical: { min: 100, max: 1000 },
        major: { min: 50, max: 100 },
        minor: { min: 20, max: 50 }
      }
    };

    return thresholds[metricName];
  }

  /**
   * Get cached metrics for a node
   */
  public getCachedMetrics(nodeId: string): KPIMetric[] {
    return this.kpiCache.get(nodeId) || [];
  }

  /**
   * Get cached alarms for a node
   */
  public getCachedAlarms(nodeId: string): Alarm[] {
    return this.alarmCache.get(nodeId) || [];
  }

  /**
   * Get aggregated metrics across time periods
   */
  public getAggregatedMetrics(
    nodeId: string,
    metricName: string,
    aggregationType: 'avg' | 'min' | 'max' | 'sum' = 'avg'
  ): number | null {
    const metrics = this.getCachedMetrics(nodeId)
      .filter(m => m.name === metricName);

    if (metrics.length === 0) {
      return null;
    }

    const values = metrics.map(m => m.value);

    switch (aggregationType) {
      case 'avg':
        return values.reduce((sum, val) => sum + val, 0) / values.length;
      case 'min':
        return Math.min(...values);
      case 'max':
        return Math.max(...values);
      case 'sum':
        return values.reduce((sum, val) => sum + val, 0);
      default:
        return null;
    }
  }

  /**
   * Generate performance report
   */
  public generatePerformanceReport(nodeId: string): any {
    const metrics = this.getCachedMetrics(nodeId);
    const alarms = this.getCachedAlarms(nodeId);

    const report = {
      nodeId,
      timestamp: new Date(),
      summary: {
        totalMetrics: metrics.length,
        criticalAlarms: alarms.filter(a => a.severity === 'CRITICAL').length,
        majorAlarms: alarms.filter(a => a.severity === 'MAJOR').length,
        minorAlarms: alarms.filter(a => a.severity === 'MINOR').length
      },
      keyMetrics: {},
      healthScore: 0,
      recommendations: []
    };

    // Calculate key metrics
    const keyMetricNames = [
      'Cell_Availability',
      'RSRP_Average', 'SSB_RSRP_Average',
      'SINR_Average', 'SSB_SINR_Average',
      'DL_Throughput_Average',
      'Call_Drop_Rate', 'Session_Drop_Rate'
    ];

    for (const metricName of keyMetricNames) {
      const avgValue = this.getAggregatedMetrics(nodeId, metricName, 'avg');
      if (avgValue !== null) {
        report.keyMetrics[metricName] = avgValue;
      }
    }

    // Calculate health score (0-100)
    report.healthScore = this.calculateHealthScore(nodeId);

    // Generate recommendations
    report.recommendations = this.generateRecommendations(nodeId);

    return report;
  }

  /**
   * Calculate overall health score
   */
  private calculateHealthScore(nodeId: string): number {
    const alarms = this.getCachedAlarms(nodeId);
    let score = 100;

    // Deduct points for alarms
    const criticalCount = alarms.filter(a => a.severity === 'CRITICAL').length;
    const majorCount = alarms.filter(a => a.severity === 'MAJOR').length;
    const minorCount = alarms.filter(a => a.severity === 'MINOR').length;

    score -= (criticalCount * 20);
    score -= (majorCount * 10);
    score -= (minorCount * 5);

    // Consider key performance metrics
    const availability = this.getAggregatedMetrics(nodeId, 'Cell_Availability', 'avg');
    if (availability !== null && availability < 95) {
      score -= (95 - availability) * 2;
    }

    return Math.max(0, Math.min(100, score));
  }

  /**
   * Generate optimization recommendations
   */
  private generateRecommendations(nodeId: string): string[] {
    const recommendations: string[] = [];
    const alarms = this.getCachedAlarms(nodeId);

    // Analyze alarms for recommendations
    const highInterference = alarms.some(a => a.type === 'High Interference');
    if (highInterference) {
      recommendations.push('Consider antenna tilt optimization to reduce interference');
      recommendations.push('Review neighbor cell power settings');
    }

    const highLoad = alarms.some(a => a.type === 'High Load');
    if (highLoad) {
      recommendations.push('Implement load balancing with neighboring cells');
      recommendations.push('Consider capacity expansion if trend continues');
    }

    const performanceDegradation = alarms.some(a => a.type === 'Performance Degradation');
    if (performanceDegradation) {
      recommendations.push('Review RF parameters for optimization');
      recommendations.push('Check for equipment maintenance requirements');
    }

    // Default recommendations if no issues
    if (recommendations.length === 0) {
      recommendations.push('Performance is within normal parameters');
      recommendations.push('Continue monitoring for trends');
    }

    return recommendations;
  }

  /**
   * Generate random value within range (for simulation)
   */
  private generateRandomValue(min: number, max: number): number {
    return Math.random() * (max - min) + min;
  }

  /**
   * Generate unique alarm ID
   */
  private generateAlarmId(): string {
    return `alarm-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
  }

  /**
   * Clear cached data
   */
  public clearCache(): void {
    this.kpiCache.clear();
    this.alarmCache.clear();
    this.logger.info('Performance monitor cache cleared');
  }

  /**
   * Get monitoring statistics
   */
  public getStatistics(): {
    monitoredNodes: number;
    totalMetrics: number;
    totalAlarms: number;
    isMonitoring: boolean;
  } {
    const totalMetrics = Array.from(this.kpiCache.values())
      .reduce((sum, metrics) => sum + metrics.length, 0);
    
    const totalAlarms = Array.from(this.alarmCache.values())
      .reduce((sum, alarms) => sum + alarms.length, 0);

    return {
      monitoredNodes: this.kpiCache.size,
      totalMetrics,
      totalAlarms,
      isMonitoring: this.isMonitoring
    };
  }
}