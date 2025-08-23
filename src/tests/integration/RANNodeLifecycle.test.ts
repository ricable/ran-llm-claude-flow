/**
 * Integration Tests for RAN Node Lifecycle - TDD London School
 * Tests collaboration between multiple components with controlled mocking
 */

import { RANNodeFactory } from '../../factories/RANNodeFactory';
import { ConfigurationManager } from '../../core/ConfigurationManager';
import { AutomationAgent } from '../../automation/AutomationAgent';
import { PerformanceMonitor } from '../../monitoring/PerformanceMonitor';
import {
  createMockCMEditClient,
  createMockLogger,
  createMockEventEmitter,
  createMockMonitoringService
} from '../mocks';

describe('RAN Node Lifecycle Integration - London School TDD', () => {
  let factory: RANNodeFactory;
  let configManager: ConfigurationManager;
  let automationAgent: AutomationAgent;
  let performanceMonitor: PerformanceMonitor;
  
  // Shared mocks for integration
  let mockCMEditClient: any;
  let mockLogger: any;
  let mockEventEmitter: any;
  let mockMonitoringService: any;

  beforeEach(() => {
    // London School: Create shared mocks for integration testing
    mockCMEditClient = createMockCMEditClient();
    mockLogger = createMockLogger();
    mockEventEmitter = createMockEventEmitter();
    mockMonitoringService = createMockMonitoringService();

    // Wire up the integrated system with mocks
    configManager = new ConfigurationManager(mockCMEditClient, mockLogger, mockEventEmitter);
    factory = new RANNodeFactory(configManager, mockCMEditClient, mockLogger);
    automationAgent = new AutomationAgent(configManager, mockEventEmitter, mockLogger);
    performanceMonitor = new PerformanceMonitor(mockMonitoringService, mockEventEmitter, mockLogger);
  });

  describe('Complete Node Provisioning Workflow', () => {
    it('should orchestrate end-to-end eNodeB provisioning with all components', async () => {
      // Arrange - Setup for complete workflow
      const nodeConfig = {
        nodeId: 'Integration-eNodeB-001',
        type: 'LTE',
        cellCount: 3,
        location: { lat: 59.3293, lon: 18.0686 },
        radioConfig: {
          maxPower: 43,
          frequencyBand: 20,
          bandwidth: '20MHz'
        }
      };

      // Mock successful responses for workflow
      mockCMEditClient.connect.mockResolvedValueOnce(true);
      mockCMEditClient.beginTransaction.mockResolvedValue({ transactionId: 'integration-tx-001' });
      mockCMEditClient.createMO.mockResolvedValue({ success: true, dn: `eNodeB=${nodeConfig.nodeId}` });
      mockCMEditClient.commitTransaction.mockResolvedValueOnce(true);
      mockMonitoringService.startMonitoring.mockResolvedValueOnce({ sessionId: 'monitor-001' });

      // Act - Execute complete provisioning workflow
      const node = await factory.createENodeB(nodeConfig);
      
      // Start automation for the new node
      await automationAgent.enableAutomation(node.id, {
        autoOptimization: true,
        performanceThresholds: {
          cpu: 80,
          memory: 90
        }
      });

      // Start performance monitoring
      await performanceMonitor.startNodeMonitoring(node.id, [
        'cpu', 'memory', 'throughput', 'latency'
      ]);

      // Assert - Verify complete integration behavior
      
      // 1. Node creation coordination
      expect(mockCMEditClient.connect).toHaveBeenCalled();
      expect(mockCMEditClient.beginTransaction).toHaveBeenCalled();
      expect(mockCMEditClient.createMO).toHaveBeenCalledWith(
        'eNodeB',
        nodeConfig.nodeId,
        expect.objectContaining(nodeConfig)
      );
      expect(mockCMEditClient.commitTransaction).toHaveBeenCalled();

      // 2. Event coordination between components
      expect(mockEventEmitter.emit).toHaveBeenCalledWith(
        'nodeCreated',
        expect.objectContaining({ nodeId: nodeConfig.nodeId })
      );
      expect(mockEventEmitter.emit).toHaveBeenCalledWith(
        'automationEnabled',
        expect.objectContaining({ nodeId: nodeConfig.nodeId })
      );
      expect(mockEventEmitter.emit).toHaveBeenCalledWith(
        'monitoringStarted',
        expect.objectContaining({ nodeId: nodeConfig.nodeId })
      );

      // 3. Monitoring integration
      expect(mockMonitoringService.startMonitoring).toHaveBeenCalledWith(
        nodeConfig.nodeId,
        ['cpu', 'memory', 'throughput', 'latency']
      );

      // 4. Cross-component logging coordination
      expect(mockLogger.info).toHaveBeenCalledWith(
        `Complete eNodeB provisioning successful: ${nodeConfig.nodeId}`
      );

      expect(node).toBeDefined();
      expect(node.id).toBe(nodeConfig.nodeId);
    });

    it('should handle partial failure scenarios with proper rollback coordination', async () => {
      // Arrange - Setup for failure scenario
      const nodeConfig = {
        nodeId: 'FailureScenario-eNodeB-002',
        type: 'LTE',
        cellCount: 2
      };

      // Mock failure in monitoring service
      mockCMEditClient.connect.mockResolvedValueOnce(true);
      mockCMEditClient.beginTransaction.mockResolvedValue({ transactionId: 'fail-tx-001' });
      mockCMEditClient.createMO.mockResolvedValue({ success: true, dn: `eNodeB=${nodeConfig.nodeId}` });
      mockCMEditClient.commitTransaction.mockResolvedValueOnce(true);
      mockMonitoringService.startMonitoring.mockRejectedValueOnce(new Error('Monitoring service unavailable'));

      // Act & Assert
      const node = await factory.createENodeB(nodeConfig);
      
      // Node creation should succeed
      expect(node).toBeDefined();
      
      // But monitoring startup should fail and trigger cleanup
      await expect(
        performanceMonitor.startNodeMonitoring(node.id, ['cpu'])
      ).rejects.toThrow('Monitoring service unavailable');

      // Verify failure handling and event coordination
      expect(mockEventEmitter.emit).toHaveBeenCalledWith(
        'monitoringStartupFailed',
        expect.objectContaining({ 
          nodeId: nodeConfig.nodeId,
          error: 'Monitoring service unavailable'
        })
      );

      expect(mockLogger.error).toHaveBeenCalledWith(
        'Integration failure during node provisioning',
        expect.objectContaining({
          nodeId: nodeConfig.nodeId,
          phase: 'monitoring-startup'
        })
      );
    });
  });

  describe('Configuration Update Propagation', () => {
    it('should coordinate configuration changes across all integrated components', async () => {
      // Arrange
      const nodeId = 'ConfigSync-eNodeB-003';
      const parameterUpdates = {
        'Radio.MaxPower': 45,
        'Cell.1.Azimuth': 120,
        'Network.IP': '192.168.100.50'
      };

      // Mock successful configuration update
      mockCMEditClient.beginTransaction.mockResolvedValue({ transactionId: 'config-sync-tx' });
      mockCMEditClient.setMO.mockResolvedValue(true);
      mockCMEditClient.commitTransaction.mockResolvedValueOnce(true);

      // Act - Update configuration and verify propagation
      await configManager.bulkUpdate(nodeId, parameterUpdates);

      // Trigger automation response to config change
      mockEventEmitter.emit.mockImplementation((event, data) => {
        if (event === 'configurationUpdated') {
          // Simulate automation agent response
          automationAgent.handleConfigurationChange(data);
        }
      });

      // Assert - Verify cross-component coordination
      
      // 1. Configuration update coordination
      expect(mockCMEditClient.beginTransaction).toHaveBeenCalled();
      Object.entries(parameterUpdates).forEach(([param, value]) => {
        expect(mockCMEditClient.setMO).toHaveBeenCalledWith(
          `ManagedElement=${nodeId}`,
          { [param]: value }
        );
      });
      expect(mockCMEditClient.commitTransaction).toHaveBeenCalled();

      // 2. Event propagation verification
      expect(mockEventEmitter.emit).toHaveBeenCalledWith(
        'configurationUpdated',
        expect.objectContaining({
          nodeId,
          updates: parameterUpdates
        })
      );

      // 3. Automation response coordination
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Automation responding to configuration change',
        expect.objectContaining({ nodeId })
      );
    });
  });

  describe('Performance Monitoring Integration', () => {
    it('should coordinate performance thresholds with automation responses', async () => {
      // Arrange
      const nodeId = 'PerfMon-eNodeB-004';
      const thresholds = {
        cpu: { warning: 70, critical: 85 },
        memory: { warning: 80, critical: 95 }
      };

      let alertCallback: (alert: any) => void;
      
      // Mock monitoring service with alert callback
      mockMonitoringService.subscribeToAlerts.mockImplementation((callback) => {
        alertCallback = callback;
        return Promise.resolve({ subscriptionId: 'alert-sub-001' });
      });

      // Act - Setup integrated monitoring
      await performanceMonitor.startNodeMonitoring(nodeId, ['cpu', 'memory']);
      await performanceMonitor.configureThresholds(nodeId, thresholds);
      
      // Subscribe automation to performance alerts
      await automationAgent.subscribeToPerformanceAlerts(nodeId);

      // Simulate performance alert
      setTimeout(() => {
        alertCallback({
          id: 'alert-001',
          nodeId,
          metric: 'cpu',
          value: 88,
          threshold: 85,
          severity: 'critical'
        });
      }, 10);

      await new Promise(resolve => setTimeout(resolve, 50)); // Wait for alert processing

      // Assert - Verify integrated alert handling
      
      // 1. Monitoring setup coordination
      expect(mockMonitoringService.startMonitoring).toHaveBeenCalledWith(
        nodeId, 
        ['cpu', 'memory']
      );
      expect(mockMonitoringService.setThreshold).toHaveBeenCalledWith(
        nodeId,
        'cpu',
        thresholds.cpu
      );

      // 2. Alert subscription coordination
      expect(mockMonitoringService.subscribeToAlerts).toHaveBeenCalledWith(
        expect.any(Function)
      );

      // 3. Cross-component alert response
      expect(mockEventEmitter.emit).toHaveBeenCalledWith(
        'performanceAlert',
        expect.objectContaining({
          nodeId,
          metric: 'cpu',
          severity: 'critical'
        })
      );

      // 4. Automation response to alert
      expect(mockLogger.warn).toHaveBeenCalledWith(
        'Automation triggered by performance alert',
        expect.objectContaining({
          nodeId,
          metric: 'cpu',
          action: 'threshold-exceeded-response'
        })
      );
    });
  });

  describe('System-wide Event Coordination', () => {
    it('should demonstrate complete event-driven integration patterns', async () => {
      // Arrange - Setup for comprehensive event flow
      const nodeId = 'EventFlow-gNodeB-005';
      const config = { nodeId, type: '5G-SA', nrCellCount: 2 };

      // Track event sequence for verification
      const eventSequence: string[] = [];
      mockEventEmitter.emit.mockImplementation((event: string, data: any) => {
        eventSequence.push(event);
        return true;
      });

      // Act - Execute full lifecycle with event tracking
      const node = await factory.createGNodeB(config);
      await automationAgent.enableAutomation(node.id, { autoOptimization: true });
      await performanceMonitor.startNodeMonitoring(node.id, ['throughput']);
      await configManager.setParameter(node.id, 'NR.MaxPower', 46);

      // Assert - Verify event sequence coordination
      expect(eventSequence).toEqual([
        'nodeCreationStarted',
        'configurationValidated',
        'nodeCreated',
        'automationEnabled',
        'monitoringStarted',
        'configurationUpdated'
      ]);

      // Verify each component responded to relevant events
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Event-driven integration completed successfully',
        expect.objectContaining({ nodeId, eventCount: eventSequence.length })
      );
    });
  });

  describe('London School Integration Verification', () => {
    it('should verify complete component interaction patterns', async () => {
      // This test demonstrates London School focus on component conversations
      const nodeId = 'LondonSchool-Test-006';

      // Act - Perform minimal operations to trigger interactions
      const config = { nodeId, type: 'LTE' };
      await factory.createENodeB(config);

      // Assert - Verify the conversation between all mocked collaborators
      
      // CMEditClient conversation
      expect(mockCMEditClient.connect).toHaveBeenCalled();
      expect(mockCMEditClient.beginTransaction).toHaveBeenCalled();
      expect(mockCMEditClient.createMO).toHaveBeenCalled();
      expect(mockCMEditClient.commitTransaction).toHaveBeenCalled();

      // Event coordination conversation
      expect(mockEventEmitter.emit).toHaveBeenCalledWith('nodeCreationStarted', expect.any(Object));
      expect(mockEventEmitter.emit).toHaveBeenCalledWith('nodeCreated', expect.any(Object));

      // Logging conversation
      expect(mockLogger.info).toHaveBeenCalledWith(
        expect.stringContaining('Creating eNodeB'),
        expect.any(Object)
      );

      // This is the essence of London School TDD: 
      // We verify HOW objects talk to each other, not just WHAT they return
    });
  });
});