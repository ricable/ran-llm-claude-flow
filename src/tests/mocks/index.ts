/**
 * Mock Infrastructure Index for TDD London School Testing
 * Central export point for all mock implementations
 */

export { createMockRANNodeFactory, verifyNodeFactoryBehavior } from './MockRANNodeFactory';
export { createMockConfigurationManager, verifyConfigurationManagerBehavior } from './MockConfigurationManager';
export { createMockCMEditClient, verifyCMEditBehavior } from './MockCMEditClient';
export { createMockMonitoringService, verifyMonitoringBehavior } from './MockMonitoringService';

// Additional mock utilities
export const createMockLogger = () => ({
  debug: jest.fn(),
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
  child: jest.fn().mockReturnThis()
});

export const createMockAxios = () => ({
  get: jest.fn(),
  post: jest.fn(),
  put: jest.fn(),
  delete: jest.fn(),
  patch: jest.fn(),
  request: jest.fn(),
  defaults: {
    headers: {},
    timeout: 5000
  },
  interceptors: {
    request: { use: jest.fn(), eject: jest.fn() },
    response: { use: jest.fn(), eject: jest.fn() }
  }
});

export const createMockEventEmitter = () => ({
  on: jest.fn(),
  off: jest.fn(),
  emit: jest.fn(),
  once: jest.fn(),
  removeListener: jest.fn(),
  removeAllListeners: jest.fn(),
  listenerCount: jest.fn(),
  listeners: jest.fn()
});

// Behavior verification utilities
export const verifyMockCleanup = (...mocks: jest.Mock[]) => {
  mocks.forEach(mock => {
    expect(mock).toHaveBeenCalled();
    mock.mockClear();
  });
};

export const verifyCallOrder = (firstMock: jest.Mock, secondMock: jest.Mock) => {
  expect(firstMock).toHaveBeenCalledBefore(secondMock);
};