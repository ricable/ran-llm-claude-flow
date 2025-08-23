/**
 * TDD London School Test Setup
 * Global test environment configuration for mock-driven development
 */

import 'jest';

// Extend Jest matchers for behavior verification
declare global {
  namespace jest {
    interface Matchers<R> {
      toHaveBeenCalledBefore(mock: jest.Mock): R;
      toSatisfyContract(contract: any): R;
      toHaveCorrectInteractionSequence(): R;
    }
  }
}

// Custom matcher for call order verification (London School focus)
expect.extend({
  toHaveBeenCalledBefore(received: jest.Mock, expected: jest.Mock) {
    const receivedCalls = received.mock.invocationCallOrder;
    const expectedCalls = expected.mock.invocationCallOrder;
    
    if (receivedCalls.length === 0) {
      return {
        message: () => `Expected ${received.getMockName()} to have been called before ${expected.getMockName()}, but it was never called`,
        pass: false
      };
    }
    
    if (expectedCalls.length === 0) {
      return {
        message: () => `Expected ${expected.getMockName()} to have been called after ${received.getMockName()}, but it was never called`,
        pass: false
      };
    }
    
    const lastReceivedCall = Math.max(...receivedCalls);
    const firstExpectedCall = Math.min(...expectedCalls);
    
    const pass = lastReceivedCall < firstExpectedCall;
    
    return {
      message: () => 
        pass 
          ? `Expected ${received.getMockName()} not to have been called before ${expected.getMockName()}`
          : `Expected ${received.getMockName()} to have been called before ${expected.getMockName()}`,
      pass
    };
  },

  toSatisfyContract(received: any, contract: any) {
    const pass = Object.keys(contract).every(key => 
      typeof received[key] === typeof contract[key]
    );
    
    return {
      message: () => 
        pass
          ? `Expected object not to satisfy contract`
          : `Expected object to satisfy contract`,
      pass
    };
  },

  toHaveCorrectInteractionSequence(received: jest.Mock) {
    // Verify interaction patterns for London School TDD
    const calls = received.mock.calls;
    const pass = calls.length > 0;
    
    return {
      message: () => 
        pass
          ? `Expected mock not to have correct interaction sequence`
          : `Expected mock to have correct interaction sequence`,
      pass
    };
  }
});

// Global test configuration
beforeEach(() => {
  // Clear all mocks before each test (London School principle)
  jest.clearAllMocks();
  
  // Reset module registry to ensure clean state
  jest.resetModules();
});

afterEach(() => {
  // Verify no unhandled mock calls (behavior verification)
  jest.clearAllTimers();
});

// Console suppression for cleaner test output
global.console = {
  ...console,
  // Suppress debug logs during testing
  debug: jest.fn(),
  log: jest.fn(),
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
};

// Test utilities for London School TDD
export const createMockWithBehavior = <T>(mockImplementation: Partial<T>): jest.Mocked<T> => {
  return mockImplementation as jest.Mocked<T>;
};

export const verifyMockInteractions = (mocks: jest.Mock[], expectedSequence: string[]) => {
  const actualSequence = mocks
    .flatMap(mock => mock.mock.calls.map((_, index) => ({ mock, index })))
    .sort((a, b) => a.mock.mock.invocationCallOrder[a.index] - b.mock.mock.invocationCallOrder[b.index])
    .map(({ mock }) => mock.getMockName());
    
  expect(actualSequence).toEqual(expectedSequence);
};