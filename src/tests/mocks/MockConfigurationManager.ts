/**
 * Mock Configuration Manager for TDD London School Testing
 * Focuses on interaction verification and behavior testing
 */

export interface MockConfigurationManager {
  loadConfiguration(nodeId: string): Promise<any>;
  saveConfiguration(nodeId: string, config: any): Promise<boolean>;
  validateConfiguration(config: any): Promise<{ valid: boolean; errors?: string[] }>;
  getParameter(nodeId: string, parameterPath: string): Promise<any>;
  setParameter(nodeId: string, parameterPath: string, value: any): Promise<boolean>;
  bulkUpdate(nodeId: string, updates: Record<string, any>): Promise<{ success: boolean; failures?: string[] }>;
  backup(nodeId: string): Promise<{ backupId: string }>;
  restore(nodeId: string, backupId: string): Promise<boolean>;
  getHistory(nodeId: string, limit?: number): Promise<any[]>;
}

export const createMockConfigurationManager = (): jest.Mocked<MockConfigurationManager> => {
  return {
    loadConfiguration: jest.fn()
      .mockImplementation((nodeId: string) => {
        // Behavior: Always loads successfully with mock data
        return Promise.resolve({
          nodeId,
          timestamp: new Date().toISOString(),
          parameters: {
            'System.Name': `MockNode_${nodeId}`,
            'Radio.MaxPower': 40,
            'Cell.Count': 3
          }
        });
      }),

    saveConfiguration: jest.fn()
      .mockImplementation((nodeId: string, config: any) => {
        // Behavior: Simulates validation before saving
        const isValid = config && typeof config === 'object';
        return Promise.resolve(isValid);
      }),

    validateConfiguration: jest.fn()
      .mockImplementation((config: any) => {
        // Behavior: Mock validation logic
        const errors: string[] = [];
        if (!config) errors.push('Configuration is required');
        if (!config.nodeId) errors.push('Node ID is required');
        
        return Promise.resolve({
          valid: errors.length === 0,
          errors: errors.length > 0 ? errors : undefined
        });
      }),

    getParameter: jest.fn()
      .mockImplementation((nodeId: string, parameterPath: string) => {
        // Behavior: Returns different mock values based on parameter path
        const mockValues: Record<string, any> = {
          'System.Name': `MockNode_${nodeId}`,
          'Radio.MaxPower': 40,
          'Cell.Count': 3,
          'Network.IP': '192.168.1.100'
        };
        
        return Promise.resolve(mockValues[parameterPath] || `mock-value-${parameterPath}`);
      }),

    setParameter: jest.fn()
      .mockImplementation((nodeId: string, parameterPath: string, value: any) => {
        // Behavior: Validates parameter before setting
        const validPaths = ['System.Name', 'Radio.MaxPower', 'Cell.Count', 'Network.IP'];
        return Promise.resolve(validPaths.includes(parameterPath));
      }),

    bulkUpdate: jest.fn()
      .mockImplementation((nodeId: string, updates: Record<string, any>) => {
        // Behavior: Simulates partial success scenarios
        const failures: string[] = [];
        Object.keys(updates).forEach(key => {
          if (key.startsWith('ReadOnly.')) {
            failures.push(key);
          }
        });

        return Promise.resolve({
          success: failures.length === 0,
          failures: failures.length > 0 ? failures : undefined
        });
      }),

    backup: jest.fn()
      .mockImplementation((nodeId: string) => {
        return Promise.resolve({
          backupId: `backup_${nodeId}_${Date.now()}`
        });
      }),

    restore: jest.fn()
      .mockImplementation((nodeId: string, backupId: string) => {
        // Behavior: Validates backup ID format
        const isValidBackup = backupId.startsWith(`backup_${nodeId}_`);
        return Promise.resolve(isValidBackup);
      }),

    getHistory: jest.fn()
      .mockImplementation((nodeId: string, limit = 10) => {
        // Behavior: Returns mock history entries
        return Promise.resolve(
          Array.from({ length: Math.min(limit, 5) }, (_, i) => ({
            timestamp: new Date(Date.now() - i * 60000).toISOString(),
            action: ['set', 'get', 'validate'][i % 3],
            parameter: `mock.parameter.${i}`,
            value: `mock-value-${i}`
          }))
        );
      })
  };
};

// Behavior verification helpers for London School TDD
export const verifyConfigurationManagerBehavior = {
  verifyParameterAccess: (manager: jest.Mocked<MockConfigurationManager>, nodeId: string, parameters: string[]) => {
    parameters.forEach(param => {
      expect(manager.getParameter).toHaveBeenCalledWith(nodeId, param);
    });
  },

  verifyUpdateSequence: (manager: jest.Mocked<MockConfigurationManager>, nodeId: string) => {
    expect(manager.loadConfiguration).toHaveBeenCalledWith(nodeId);
    expect(manager.loadConfiguration).toHaveBeenCalledBefore(manager.setParameter);
    expect(manager.setParameter).toHaveBeenCalledBefore(manager.saveConfiguration);
  },

  verifyValidationCalled: (manager: jest.Mocked<MockConfigurationManager>) => {
    expect(manager.validateConfiguration).toHaveBeenCalled();
  },

  verifyBackupBeforeUpdate: (manager: jest.Mocked<MockConfigurationManager>, nodeId: string) => {
    expect(manager.backup).toHaveBeenCalledWith(nodeId);
    expect(manager.backup).toHaveBeenCalledBefore(manager.bulkUpdate);
  }
};