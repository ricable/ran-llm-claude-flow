/**
 * Mock Monitoring Service for TDD London School Testing
 * Simulates performance monitoring and alerting behavior
 */

export interface MockMonitoringService {
  startMonitoring(nodeId: string, metrics: string[]): Promise<{ sessionId: string }>;
  stopMonitoring(sessionId: string): Promise<boolean>;
  getMetrics(nodeId: string, timeRange?: { start: Date; end: Date }): Promise<any>;
  setThreshold(nodeId: string, metric: string, threshold: { warning: number; critical: number }): Promise<boolean>;
  subscribeToAlerts(callback: (alert: any) => void): Promise<{ subscriptionId: string }>;
  unsubscribeFromAlerts(subscriptionId: string): Promise<boolean>;
  getAlerts(nodeId?: string, severity?: string): Promise<any[]>;
  acknowledgeAlert(alertId: string): Promise<boolean>;
  getHealthStatus(nodeId: string): Promise<{ status: 'healthy' | 'warning' | 'critical'; details: any }>;
  generateReport(nodeId: string, reportType: string): Promise<{ reportId: string; data: any }>;
}

export const createMockMonitoringService = (): jest.Mocked<MockMonitoringService> => {
  const activeSessions = new Map<string, { nodeId: string; metrics: string[] }>();
  const subscriptions = new Map<string, (alert: any) => void>();
  const thresholds = new Map<string, any>();

  return {
    startMonitoring: jest.fn()
      .mockImplementation((nodeId: string, metrics: string[]) => {
        const sessionId = `monitoring_${nodeId}_${Date.now()}`;
        activeSessions.set(sessionId, { nodeId, metrics });
        
        // Behavior: Validates monitoring parameters
        const isValidNode = nodeId && typeof nodeId === 'string';
        const hasMetrics = metrics && metrics.length > 0;
        
        if (!isValidNode || !hasMetrics) {
          return Promise.reject(new Error('Invalid monitoring parameters'));
        }

        return Promise.resolve({ sessionId });
      }),

    stopMonitoring: jest.fn()
      .mockImplementation((sessionId: string) => {
        const exists = activeSessions.has(sessionId);
        if (exists) {
          activeSessions.delete(sessionId);
        }
        return Promise.resolve(exists);
      }),

    getMetrics: jest.fn()
      .mockImplementation((nodeId: string, timeRange?: { start: Date; end: Date }) => {
        // Behavior: Generates mock metrics based on time range
        const now = new Date();
        const start = timeRange?.start || new Date(now.getTime() - 3600000); // 1 hour ago
        const end = timeRange?.end || now;
        
        const mockMetrics = {
          nodeId,
          timeRange: { start, end },
          data: {
            cpu: Array.from({ length: 12 }, (_, i) => ({
              timestamp: new Date(start.getTime() + (i * 300000)),
              value: Math.random() * 100
            })),
            memory: Array.from({ length: 12 }, (_, i) => ({
              timestamp: new Date(start.getTime() + (i * 300000)),
              value: Math.random() * 100
            })),
            throughput: Array.from({ length: 12 }, (_, i) => ({
              timestamp: new Date(start.getTime() + (i * 300000)),
              value: Math.random() * 1000
            }))
          }
        };

        return Promise.resolve(mockMetrics);
      }),

    setThreshold: jest.fn()
      .mockImplementation((nodeId: string, metric: string, threshold: { warning: number; critical: number }) => {
        // Behavior: Validates threshold values
        const isValidThreshold = threshold.warning < threshold.critical;
        
        if (isValidThreshold) {
          const key = `${nodeId}-${metric}`;
          thresholds.set(key, threshold);
        }

        return Promise.resolve(isValidThreshold);
      }),

    subscribeToAlerts: jest.fn()
      .mockImplementation((callback: (alert: any) => void) => {
        const subscriptionId = `sub_${Date.now()}`;
        subscriptions.set(subscriptionId, callback);
        
        // Behavior: Simulates immediate alert for testing
        setTimeout(() => {
          callback({
            id: `alert_${Date.now()}`,
            nodeId: 'test-node',
            metric: 'cpu',
            severity: 'warning',
            value: 85,
            threshold: 80,
            timestamp: new Date()
          });
        }, 10);

        return Promise.resolve({ subscriptionId });
      }),

    unsubscribeFromAlerts: jest.fn()
      .mockImplementation((subscriptionId: string) => {
        const exists = subscriptions.has(subscriptionId);
        if (exists) {
          subscriptions.delete(subscriptionId);
        }
        return Promise.resolve(exists);
      }),

    getAlerts: jest.fn()
      .mockImplementation((nodeId?: string, severity?: string) => {
        // Behavior: Filters alerts based on parameters
        const mockAlerts = [
          {
            id: 'alert_001',
            nodeId: 'node-001',
            metric: 'cpu',
            severity: 'warning',
            value: 85,
            acknowledged: false,
            timestamp: new Date()
          },
          {
            id: 'alert_002',
            nodeId: 'node-002',
            metric: 'memory',
            severity: 'critical',
            value: 95,
            acknowledged: true,
            timestamp: new Date()
          }
        ];

        let filteredAlerts = mockAlerts;
        
        if (nodeId) {
          filteredAlerts = filteredAlerts.filter(alert => alert.nodeId === nodeId);
        }
        
        if (severity) {
          filteredAlerts = filteredAlerts.filter(alert => alert.severity === severity);
        }

        return Promise.resolve(filteredAlerts);
      }),

    acknowledgeAlert: jest.fn()
      .mockImplementation((alertId: string) => {
        // Behavior: Simulates alert acknowledgment
        const validAlertIds = ['alert_001', 'alert_002', 'alert_003'];
        return Promise.resolve(validAlertIds.includes(alertId));
      }),

    getHealthStatus: jest.fn()
      .mockImplementation((nodeId: string) => {
        // Behavior: Generates health status based on node ID
        const healthStates = ['healthy', 'warning', 'critical'] as const;
        const status = healthStates[nodeId.length % 3];
        
        const details = {
          cpu: Math.random() * 100,
          memory: Math.random() * 100,
          connectivity: Math.random() > 0.1,
          lastUpdate: new Date()
        };

        return Promise.resolve({ status, details });
      }),

    generateReport: jest.fn()
      .mockImplementation((nodeId: string, reportType: string) => {
        const reportId = `report_${nodeId}_${reportType}_${Date.now()}`;
        
        // Behavior: Generates different report types
        const reportData: Record<string, any> = {
          'performance': {
            averageCPU: 45.2,
            averageMemory: 62.8,
            uptime: '99.95%',
            alerts: 3
          },
          'capacity': {
            currentLoad: 75,
            maxCapacity: 1000,
            projectedGrowth: '12%'
          },
          'availability': {
            uptime: '99.95%',
            downtime: '0.05%',
            incidents: 2
          }
        };

        return Promise.resolve({
          reportId,
          data: reportData[reportType] || { message: `Unknown report type: ${reportType}` }
        });
      })
  };
};

// Behavior verification helpers for London School TDD
export const verifyMonitoringBehavior = {
  verifyMonitoringLifecycle: (service: jest.Mocked<MockMonitoringService>, nodeId: string) => {
    expect(service.startMonitoring).toHaveBeenCalledWith(nodeId, expect.any(Array));
    expect(service.getHealthStatus).toHaveBeenCalledWith(nodeId);
  },

  verifyAlertSubscriptionFlow: (service: jest.Mocked<MockMonitoringService>) => {
    expect(service.subscribeToAlerts).toHaveBeenCalledWith(expect.any(Function));
    expect(service.getAlerts).toHaveBeenCalled();
  },

  verifyThresholdConfiguration: (service: jest.Mocked<MockMonitoringService>, nodeId: string, metric: string) => {
    expect(service.setThreshold).toHaveBeenCalledWith(
      nodeId,
      metric,
      expect.objectContaining({
        warning: expect.any(Number),
        critical: expect.any(Number)
      })
    );
  },

  verifyReportGeneration: (service: jest.Mocked<MockMonitoringService>, nodeId: string, reportType: string) => {
    expect(service.generateReport).toHaveBeenCalledWith(nodeId, reportType);
  },

  verifyCleanupOnStop: (service: jest.Mocked<MockMonitoringService>) => {
    expect(service.stopMonitoring).toHaveBeenCalled();
    expect(service.unsubscribeFromAlerts).toHaveBeenCalled();
  }
};