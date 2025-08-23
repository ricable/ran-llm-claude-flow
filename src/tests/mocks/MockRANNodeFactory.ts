/**
 * Mock RAN Node Factory for TDD London School Testing
 * Provides mock implementations for all RAN node creation patterns
 */

import { RANNode } from '../../core/RANNode';
import { ENodeB } from '../../nodes/ENodeB';
import { GNodeB } from '../../nodes/GNodeB';
import { Cell } from '../../nodes/Cell';

export interface MockRANNodeFactory {
  createENodeB(config: any): jest.Mocked<ENodeB>;
  createGNodeB(config: any): jest.Mocked<GNodeB>;
  createCell(config: any): jest.Mocked<Cell>;
  createGenericNode(type: string, config: any): jest.Mocked<RANNode>;
}

export const createMockRANNodeFactory = (): jest.Mocked<MockRANNodeFactory> => {
  const mockENodeB = {
    id: 'mock-enodeb-001',
    type: 'eNodeB',
    status: 'active',
    configure: jest.fn().mockResolvedValue({ success: true }),
    validate: jest.fn().mockResolvedValue({ valid: true }),
    getConfiguration: jest.fn().mockResolvedValue({ lte: true }),
    setParameter: jest.fn().mockResolvedValue({ updated: true }),
    getParameter: jest.fn().mockResolvedValue({ value: 'mock-value' }),
    connect: jest.fn().mockResolvedValue({ connected: true }),
    disconnect: jest.fn().mockResolvedValue({ disconnected: true })
  } as unknown as jest.Mocked<ENodeB>;

  const mockGNodeB = {
    id: 'mock-gnodeb-001',
    type: 'gNodeB',
    status: 'active',
    configure: jest.fn().mockResolvedValue({ success: true }),
    validate: jest.fn().mockResolvedValue({ valid: true }),
    getConfiguration: jest.fn().mockResolvedValue({ nr: true }),
    setParameter: jest.fn().mockResolvedValue({ updated: true }),
    getParameter: jest.fn().mockResolvedValue({ value: 'mock-value' }),
    connect: jest.fn().mockResolvedValue({ connected: true }),
    disconnect: jest.fn().mockResolvedValue({ disconnected: true })
  } as unknown as jest.Mocked<GNodeB>;

  const mockCell = {
    id: 'mock-cell-001',
    type: 'cell',
    status: 'active',
    configure: jest.fn().mockResolvedValue({ success: true }),
    validate: jest.fn().mockResolvedValue({ valid: true }),
    getConfiguration: jest.fn().mockResolvedValue({ active: true }),
    setParameter: jest.fn().mockResolvedValue({ updated: true }),
    getParameter: jest.fn().mockResolvedValue({ value: 'mock-value' }),
    activate: jest.fn().mockResolvedValue({ activated: true }),
    deactivate: jest.fn().mockResolvedValue({ deactivated: true })
  } as unknown as jest.Mocked<Cell>;

  const mockGenericNode = {
    id: 'mock-generic-001',
    type: 'generic',
    status: 'active',
    configure: jest.fn().mockResolvedValue({ success: true }),
    validate: jest.fn().mockResolvedValue({ valid: true }),
    getConfiguration: jest.fn().mockResolvedValue({ configured: true }),
    setParameter: jest.fn().mockResolvedValue({ updated: true }),
    getParameter: jest.fn().mockResolvedValue({ value: 'mock-value' })
  } as unknown as jest.Mocked<RANNode>;

  return {
    createENodeB: jest.fn().mockReturnValue(mockENodeB),
    createGNodeB: jest.fn().mockReturnValue(mockGNodeB),
    createCell: jest.fn().mockReturnValue(mockCell),
    createGenericNode: jest.fn().mockReturnValue(mockGenericNode)
  };
};

// Behavior verification helpers for London School TDD
export const verifyNodeFactoryBehavior = {
  verifyCreationSequence: (factory: jest.Mocked<MockRANNodeFactory>, expectedTypes: string[]) => {
    const calls = [
      ...factory.createENodeB.mock.calls.map(() => 'eNodeB'),
      ...factory.createGNodeB.mock.calls.map(() => 'gNodeB'),
      ...factory.createCell.mock.calls.map(() => 'cell'),
      ...factory.createGenericNode.mock.calls.map((call) => call[0])
    ];
    expect(calls).toEqual(expectedTypes);
  },

  verifyConfigurationPassed: (factory: jest.Mocked<MockRANNodeFactory>, expectedConfigs: any[]) => {
    const allCalls = [
      ...factory.createENodeB.mock.calls,
      ...factory.createGNodeB.mock.calls,
      ...factory.createCell.mock.calls,
      ...factory.createGenericNode.mock.calls.map(call => [call[1]]) // Extract config from generic calls
    ];
    
    allCalls.forEach((call, index) => {
      expect(call[0]).toEqual(expectedConfigs[index]);
    });
  }
};