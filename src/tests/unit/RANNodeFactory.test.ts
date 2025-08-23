/**
 * Unit Tests for RANNodeFactory - TDD London School Approach
 * Focus on mock-driven development and behavior verification
 */

import { RANNodeFactory } from '../../factories/RANNodeFactory';
import { 
  createMockConfigurationManager,
  createMockCMEditClient,
  createMockLogger,
  verifyConfigurationManagerBehavior 
} from '../mocks';

describe('RANNodeFactory', () => {
  let factory: RANNodeFactory;
  let mockConfigManager: any;
  let mockCMEditClient: any;
  let mockLogger: any;

  beforeEach(() => {
    // London School: Create mocks for all collaborators
    mockConfigManager = createMockConfigurationManager();
    mockCMEditClient = createMockCMEditClient();
    mockLogger = createMockLogger();

    // Inject mocks into the system under test
    factory = new RANNodeFactory(mockConfigManager, mockCMEditClient, mockLogger);
  });

  describe('ENodeB Creation', () => {
    it('should coordinate with configuration manager during eNodeB creation', async () => {
      // Arrange - London School: Define expected behavior
      const config = {
        nodeId: 'eNodeB-001',
        type: 'LTE',
        cellCount: 3
      };

      // Act - Execute the behavior we're testing
      const eNodeB = await factory.createENodeB(config);

      // Assert - Verify the conversation between objects (London School focus)
      expect(mockConfigManager.validateConfiguration).toHaveBeenCalledWith(config);
      expect(mockConfigManager.loadConfiguration).toHaveBeenCalledWith(config.nodeId);
      expect(mockLogger.info).toHaveBeenCalledWith(
        `Creating eNodeB with ID: ${config.nodeId}`,
        expect.any(Object)
      );
      
      // Verify the created node has correct interface
      expect(eNodeB).toBeDefined();
      expect(eNodeB.id).toBe(config.nodeId);
      expect(eNodeB.type).toBe('eNodeB');
    });

    it('should handle configuration validation failures during eNodeB creation', async () => {
      // Arrange - Mock validation failure behavior
      const invalidConfig = { nodeId: '', type: 'LTE' };
      mockConfigManager.validateConfiguration.mockResolvedValueOnce({
        valid: false,
        errors: ['Node ID is required']
      });

      // Act & Assert - Verify error handling behavior
      await expect(factory.createENodeB(invalidConfig)).rejects.toThrow('Configuration validation failed');
      
      // Verify error logging behavior
      expect(mockLogger.error).toHaveBeenCalledWith(
        'eNodeB creation failed',
        expect.objectContaining({ errors: ['Node ID is required'] })
      );
    });

    it('should establish CM Edit connection before node configuration', async () => {
      // Arrange
      const config = { nodeId: 'eNodeB-002', type: 'LTE' };
      
      // Act
      await factory.createENodeB(config);
      
      // Assert - Verify interaction sequence (London School emphasis)
      expect(mockCMEditClient.connect).toHaveBeenCalledBefore(mockConfigManager.loadConfiguration);
      expect(mockConfigManager.loadConfiguration).toHaveBeenCalledBefore(mockConfigManager.validateConfiguration);
    });
  });

  describe('GNodeB Creation', () => {
    it('should coordinate 5G-specific validation during gNodeB creation', async () => {
      // Arrange
      const config = {
        nodeId: 'gNodeB-001',
        type: '5G-SA',
        nrCellCount: 2,
        frequencyBands: [78, 3500]
      };

      // Act
      const gNodeB = await factory.createGNodeB(config);

      // Assert - Verify 5G-specific behavior
      expect(mockConfigManager.validateConfiguration).toHaveBeenCalledWith(
        expect.objectContaining({ type: '5G-SA' })
      );
      expect(mockLogger.info).toHaveBeenCalledWith(
        `Creating gNodeB with ID: ${config.nodeId}`,
        expect.objectContaining({ nrCellCount: 2 })
      );
      
      expect(gNodeB).toBeDefined();
      expect(gNodeB.type).toBe('gNodeB');
    });

    it('should validate 5G frequency bands before node creation', async () => {
      // Arrange
      const config = {
        nodeId: 'gNodeB-002',
        type: '5G-NSA',
        frequencyBands: [99999] // Invalid band
      };

      mockConfigManager.validateConfiguration.mockResolvedValueOnce({
        valid: false,
        errors: ['Invalid frequency band: 99999']
      });

      // Act & Assert
      await expect(factory.createGNodeB(config)).rejects.toThrow();
      
      // Verify validation was called with frequency band data
      expect(mockConfigManager.validateConfiguration).toHaveBeenCalledWith(
        expect.objectContaining({ frequencyBands: [99999] })
      );
    });
  });

  describe('Cell Creation', () => {
    it('should coordinate parent node validation during cell creation', async () => {
      // Arrange
      const config = {
        cellId: 'Cell-001',
        parentNodeId: 'eNodeB-001',
        sector: 1,
        azimuth: 45
      };

      // Mock parent node existence
      mockConfigManager.getParameter.mockResolvedValueOnce('eNodeB'); // parent type

      // Act
      const cell = await factory.createCell(config);

      // Assert - Verify parent validation behavior
      expect(mockConfigManager.getParameter).toHaveBeenCalledWith(
        config.parentNodeId,
        'System.Type'
      );
      expect(mockLogger.info).toHaveBeenCalledWith(
        `Creating cell ${config.cellId} for parent node ${config.parentNodeId}`
      );
      
      expect(cell).toBeDefined();
      expect(cell.id).toBe(config.cellId);
    });

    it('should fail cell creation if parent node does not exist', async () => {
      // Arrange
      const config = {
        cellId: 'Cell-002',
        parentNodeId: 'NonExistent-001',
        sector: 1
      };

      // Mock parent node not found
      mockConfigManager.getParameter.mockRejectedValueOnce(new Error('Node not found'));

      // Act & Assert
      await expect(factory.createCell(config)).rejects.toThrow('Parent node validation failed');
      
      // Verify error handling
      expect(mockLogger.error).toHaveBeenCalledWith(
        'Cell creation failed - invalid parent node',
        expect.objectContaining({ parentNodeId: 'NonExistent-001' })
      );
    });
  });

  describe('Factory Behavior Coordination', () => {
    it('should properly clean up resources after failed node creation', async () => {
      // Arrange
      const config = { nodeId: 'FailNode-001', type: 'LTE' };
      
      // Mock CM Edit connection failure
      mockCMEditClient.connect.mockResolvedValueOnce(false);

      // Act & Assert
      await expect(factory.createENodeB(config)).rejects.toThrow();
      
      // Verify cleanup behavior
      expect(mockCMEditClient.disconnect).toHaveBeenCalled();
      expect(mockLogger.error).toHaveBeenCalledWith(
        'Factory cleanup after creation failure',
        expect.any(Object)
      );
    });

    it('should maintain transaction boundaries during bulk creation', async () => {
      // Arrange
      const configs = [
        { nodeId: 'Bulk-001', type: 'LTE' },
        { nodeId: 'Bulk-002', type: 'LTE' },
        { nodeId: 'Bulk-003', type: 'LTE' }
      ];

      // Mock transaction support
      mockCMEditClient.beginTransaction.mockResolvedValue({ transactionId: 'tx-123' });

      // Act
      await factory.bulkCreateENodeBs(configs);

      // Assert - Verify transaction behavior
      expect(mockCMEditClient.beginTransaction).toHaveBeenCalled();
      expect(mockCMEditClient.commitTransaction).toHaveBeenCalledWith('tx-123');
      
      // Verify all nodes were created within transaction
      configs.forEach(config => {
        expect(mockConfigManager.validateConfiguration).toHaveBeenCalledWith(config);
      });
    });
  });

  describe('Mock Behavior Verification', () => {
    it('should demonstrate London School interaction verification patterns', async () => {
      // Arrange
      const config = { nodeId: 'Test-001', type: 'LTE' };

      // Act
      await factory.createENodeB(config);

      // Assert - Use custom behavior verification helpers
      verifyConfigurationManagerBehavior.verifyValidationCalled(mockConfigManager);
      verifyConfigurationManagerBehavior.verifyUpdateSequence(mockConfigManager, config.nodeId);
      
      // Verify complete conversation flow
      const expectedCallSequence = [
        'connect',
        'validateConfiguration', 
        'loadConfiguration',
        'info'
      ];
      
      // This demonstrates how London School TDD focuses on the conversation between objects
      expect(mockCMEditClient.connect).toHaveBeenCalled();
      expect(mockConfigManager.validateConfiguration).toHaveBeenCalled();
      expect(mockConfigManager.loadConfiguration).toHaveBeenCalled();
      expect(mockLogger.info).toHaveBeenCalled();
    });
  });
});