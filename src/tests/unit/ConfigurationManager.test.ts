/**
 * Unit Tests for ConfigurationManager - TDD London School Approach
 * Emphasizes mock interactions and behavior verification
 */

import { ConfigurationManager } from '../../core/ConfigurationManager';
import {
  createMockCMEditClient,
  createMockLogger,
  createMockEventEmitter,
  verifyCMEditBehavior
} from '../mocks';

describe('ConfigurationManager - London School TDD', () => {
  let configManager: ConfigurationManager;
  let mockCMEditClient: any;
  let mockLogger: any;
  let mockEventEmitter: any;

  beforeEach(() => {
    // London School: Mock all collaborators
    mockCMEditClient = createMockCMEditClient();
    mockLogger = createMockLogger();
    mockEventEmitter = createMockEventEmitter();

    configManager = new ConfigurationManager(
      mockCMEditClient,
      mockLogger,
      mockEventEmitter
    );
  });

  describe('Configuration Loading Behavior', () => {
    it('should coordinate with CM Edit client to load node configuration', async () => {
      // Arrange
      const nodeId = 'TestNode-001';
      const expectedConfig = {
        'System.Name': 'TestNode-001',
        'Radio.MaxPower': 40
      };

      mockCMEditClient.getMO.mockResolvedValueOnce({
        found: true,
        attributes: expectedConfig
      });

      // Act
      const config = await configManager.loadConfiguration(nodeId);

      // Assert - Verify collaboration with CM Edit client
      expect(mockCMEditClient.getMO).toHaveBeenCalledWith(
        `ManagedElement=${nodeId}`,
        expect.any(Array)
      );
      
      // Verify event emission behavior
      expect(mockEventEmitter.emit).toHaveBeenCalledWith(
        'configurationLoaded',
        { nodeId, config: expectedConfig }
      );
      
      // Verify logging behavior
      expect(mockLogger.info).toHaveBeenCalledWith(
        `Configuration loaded for node: ${nodeId}`,
        expect.objectContaining({ parametersCount: Object.keys(expectedConfig).length })
      );

      expect(config).toEqual(expectedConfig);
    });

    it('should handle CM Edit client connection failures gracefully', async () => {
      // Arrange
      const nodeId = 'FailNode-001';
      mockCMEditClient.getMO.mockRejectedValueOnce(new Error('Connection failed'));

      // Act & Assert
      await expect(configManager.loadConfiguration(nodeId)).rejects.toThrow('Failed to load configuration');
      
      // Verify error handling behavior
      expect(mockLogger.error).toHaveBeenCalledWith(
        'Configuration loading failed',
        expect.objectContaining({ 
          nodeId, 
          error: 'Connection failed' 
        })
      );
      
      // Verify error event emission
      expect(mockEventEmitter.emit).toHaveBeenCalledWith(
        'configurationLoadError',
        { nodeId, error: expect.any(Error) }
      );
    });

    it('should retry configuration loading on transient failures', async () => {
      // Arrange
      const nodeId = 'RetryNode-001';
      
      // Mock first call fails, second succeeds
      mockCMEditClient.getMO
        .mockRejectedValueOnce(new Error('Temporary failure'))
        .mockResolvedValueOnce({ 
          found: true, 
          attributes: { 'System.Name': nodeId } 
        });

      // Act
      const config = await configManager.loadConfigurationWithRetry(nodeId, { maxRetries: 2 });

      // Assert - Verify retry behavior
      expect(mockCMEditClient.getMO).toHaveBeenCalledTimes(2);
      expect(mockLogger.warn).toHaveBeenCalledWith(
        'Configuration loading failed, retrying...',
        expect.objectContaining({ attempt: 1 })
      );
      
      expect(config).toBeDefined();
    });
  });

  describe('Configuration Validation Behavior', () => {
    it('should coordinate parameter validation with business rules', async () => {
      // Arrange
      const config = {
        'Radio.MaxPower': 45, // Valid range: 10-46 dBm
        'Cell.Count': 3,      // Valid range: 1-6
        'System.Name': 'ValidNode'
      };

      // Act
      const result = await configManager.validateConfiguration(config);

      // Assert - Verify validation logic interaction
      expect(result.valid).toBe(true);
      expect(mockLogger.debug).toHaveBeenCalledWith(
        'Configuration validation completed',
        expect.objectContaining({ 
          parametersValidated: Object.keys(config).length,
          result: 'valid' 
        })
      );
    });

    it('should detect and report parameter constraint violations', async () => {
      // Arrange
      const invalidConfig = {
        'Radio.MaxPower': 50, // Exceeds maximum of 46 dBm
        'Cell.Count': 10,     // Exceeds maximum of 6
        'System.Name': ''     // Required field
      };

      // Act
      const result = await configManager.validateConfiguration(invalidConfig);

      // Assert - Verify error detection behavior
      expect(result.valid).toBe(false);
      expect(result.errors).toHaveLength(3);
      expect(result.errors).toContain('Radio.MaxPower exceeds maximum value of 46');
      expect(result.errors).toContain('Cell.Count exceeds maximum value of 6');
      expect(result.errors).toContain('System.Name is required');

      // Verify error logging
      expect(mockLogger.warn).toHaveBeenCalledWith(
        'Configuration validation failed',
        expect.objectContaining({ errors: result.errors })
      );
    });
  });

  describe('Parameter Update Behavior', () => {
    it('should coordinate transactional parameter updates with CM Edit', async () => {
      // Arrange
      const nodeId = 'UpdateNode-001';
      const parameterPath = 'Radio.MaxPower';
      const newValue = 42;

      mockCMEditClient.beginTransaction.mockResolvedValueOnce({ transactionId: 'tx-001' });
      mockCMEditClient.setMO.mockResolvedValueOnce(true);
      mockCMEditClient.commitTransaction.mockResolvedValueOnce(true);

      // Act
      const result = await configManager.setParameter(nodeId, parameterPath, newValue);

      // Assert - Verify transactional behavior
      expect(mockCMEditClient.beginTransaction).toHaveBeenCalled();
      expect(mockCMEditClient.setMO).toHaveBeenCalledWith(
        `ManagedElement=${nodeId}`,
        { [parameterPath]: newValue }
      );
      expect(mockCMEditClient.commitTransaction).toHaveBeenCalledWith('tx-001');

      // Verify transaction sequence
      expect(mockCMEditClient.beginTransaction).toHaveBeenCalledBefore(mockCMEditClient.setMO);
      expect(mockCMEditClient.setMO).toHaveBeenCalledBefore(mockCMEditClient.commitTransaction);

      expect(result).toBe(true);
    });

    it('should rollback transaction on parameter update failure', async () => {
      // Arrange
      const nodeId = 'FailUpdateNode-001';
      const parameterPath = 'Radio.MaxPower';
      const newValue = 999; // Invalid value

      mockCMEditClient.beginTransaction.mockResolvedValueOnce({ transactionId: 'tx-002' });
      mockCMEditClient.setMO.mockRejectedValueOnce(new Error('Invalid parameter value'));

      // Act
      const result = await configManager.setParameter(nodeId, parameterPath, newValue);

      // Assert - Verify rollback behavior
      expect(mockCMEditClient.rollbackTransaction).toHaveBeenCalledWith('tx-002');
      expect(mockLogger.error).toHaveBeenCalledWith(
        'Parameter update failed, transaction rolled back',
        expect.objectContaining({ 
          nodeId,
          parameterPath,
          transactionId: 'tx-002'
        })
      );

      expect(result).toBe(false);
    });
  });

  describe('Bulk Operations Behavior', () => {
    it('should coordinate bulk parameter updates with proper transaction handling', async () => {
      // Arrange
      const nodeId = 'BulkNode-001';
      const updates = {
        'Radio.MaxPower': 44,
        'Cell.Count': 4,
        'System.Description': 'Updated bulk'
      };

      mockCMEditClient.beginTransaction.mockResolvedValueOnce({ transactionId: 'bulk-tx-001' });
      mockCMEditClient.setMO.mockResolvedValue(true);
      mockCMEditClient.commitTransaction.mockResolvedValueOnce(true);

      // Act
      const result = await configManager.bulkUpdate(nodeId, updates);

      // Assert - Verify bulk operation coordination
      expect(mockCMEditClient.beginTransaction).toHaveBeenCalled();
      
      // Verify each parameter was updated in the transaction
      Object.entries(updates).forEach(([path, value]) => {
        expect(mockCMEditClient.setMO).toHaveBeenCalledWith(
          `ManagedElement=${nodeId}`,
          { [path]: value }
        );
      });

      expect(mockCMEditClient.commitTransaction).toHaveBeenCalledWith('bulk-tx-001');
      expect(result.success).toBe(true);
    });

    it('should handle partial failures in bulk updates gracefully', async () => {
      // Arrange
      const nodeId = 'PartialFailNode-001';
      const updates = {
        'Radio.MaxPower': 44,    // Should succeed
        'ReadOnly.Parameter': 5,  // Should fail
        'Cell.Count': 4          // Should succeed
      };

      mockCMEditClient.beginTransaction.mockResolvedValueOnce({ transactionId: 'partial-tx-001' });
      
      // Mock partial success pattern
      mockCMEditClient.setMO
        .mockResolvedValueOnce(true)  // Radio.MaxPower succeeds
        .mockRejectedValueOnce(new Error('Read-only parameter')) // ReadOnly.Parameter fails
        .mockResolvedValueOnce(true); // Cell.Count succeeds

      // Act
      const result = await configManager.bulkUpdate(nodeId, updates);

      // Assert - Verify partial failure handling
      expect(result.success).toBe(false);
      expect(result.failures).toContain('ReadOnly.Parameter');
      expect(mockCMEditClient.rollbackTransaction).toHaveBeenCalledWith('partial-tx-001');
      
      // Verify failure logging
      expect(mockLogger.warn).toHaveBeenCalledWith(
        'Bulk update partially failed',
        expect.objectContaining({ 
          nodeId,
          failures: ['ReadOnly.Parameter']
        })
      );
    });
  });

  describe('Backup and Restore Behavior', () => {
    it('should coordinate configuration backup with proper metadata', async () => {
      // Arrange
      const nodeId = 'BackupNode-001';
      const mockConfig = {
        'System.Name': nodeId,
        'Radio.MaxPower': 40,
        'Cell.Count': 3
      };

      mockCMEditClient.getMO.mockResolvedValueOnce({
        found: true,
        attributes: mockConfig
      });

      // Act
      const backup = await configManager.backup(nodeId);

      // Assert - Verify backup coordination
      expect(mockCMEditClient.getMO).toHaveBeenCalledWith(`ManagedElement=${nodeId}`);
      expect(backup.backupId).toMatch(new RegExp(`backup_${nodeId}_\\d+`));
      expect(backup.timestamp).toBeDefined();
      expect(backup.config).toEqual(mockConfig);

      // Verify backup event emission
      expect(mockEventEmitter.emit).toHaveBeenCalledWith(
        'configurationBackedUp',
        { nodeId, backupId: backup.backupId }
      );
    });

    it('should coordinate configuration restore with validation', async () => {
      // Arrange
      const nodeId = 'RestoreNode-001';
      const backupId = `backup_${nodeId}_123456789`;
      const backupConfig = {
        'System.Name': nodeId,
        'Radio.MaxPower': 38
      };

      // Mock backup retrieval and validation
      configManager.getBackup = jest.fn().mockResolvedValueOnce({
        backupId,
        config: backupConfig,
        timestamp: new Date()
      });

      mockCMEditClient.beginTransaction.mockResolvedValueOnce({ transactionId: 'restore-tx-001' });
      mockCMEditClient.setMO.mockResolvedValue(true);
      mockCMEditClient.commitTransaction.mockResolvedValueOnce(true);

      // Act
      const result = await configManager.restore(nodeId, backupId);

      // Assert - Verify restore coordination
      expect(configManager.getBackup).toHaveBeenCalledWith(backupId);
      expect(mockCMEditClient.beginTransaction).toHaveBeenCalled();
      
      // Verify configuration application
      Object.entries(backupConfig).forEach(([path, value]) => {
        expect(mockCMEditClient.setMO).toHaveBeenCalledWith(
          `ManagedElement=${nodeId}`,
          { [path]: value }
        );
      });

      expect(mockCMEditClient.commitTransaction).toHaveBeenCalledWith('restore-tx-001');
      expect(result).toBe(true);
    });
  });

  describe('London School Behavior Verification', () => {
    it('should demonstrate complete interaction verification patterns', async () => {
      // Arrange
      const nodeId = 'InteractionTest-001';
      const config = { 'System.Name': nodeId };

      // Act - Perform a complete configuration cycle
      await configManager.loadConfiguration(nodeId);
      await configManager.validateConfiguration(config);
      await configManager.setParameter(nodeId, 'Radio.MaxPower', 42);

      // Assert - Use custom behavior verification
      verifyCMEditBehavior.verifyConnectionSequence(mockCMEditClient);
      verifyCMEditBehavior.verifyTransactionBoundaries(mockCMEditClient);

      // Verify complete conversation flow (London School emphasis)
      expect(mockCMEditClient.getMO).toHaveBeenCalled();
      expect(mockEventEmitter.emit).toHaveBeenCalledWith('configurationLoaded', expect.any(Object));
      expect(mockLogger.debug).toHaveBeenCalledWith('Configuration validation completed', expect.any(Object));
      expect(mockCMEditClient.beginTransaction).toHaveBeenCalled();
      expect(mockCMEditClient.setMO).toHaveBeenCalled();
      expect(mockCMEditClient.commitTransaction).toHaveBeenCalled();
    });
  });
});