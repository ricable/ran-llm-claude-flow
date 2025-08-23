/**
 * Test Environment Configuration
 * Manages shared test resources and configuration
 */

export class TestEnvironmentConfig {
  private initialized = false;
  private resources: Map<string, any> = new Map();

  async initialize(): Promise<void> {
    if (this.initialized) return;

    // Initialize shared test resources
    this.resources.set('mockAxios', this.createMockAxios());
    this.resources.set('testLogger', this.createTestLogger());
    this.resources.set('mockTimers', this.setupMockTimers());

    this.initialized = true;
  }

  async cleanup(): Promise<void> {
    this.resources.clear();
    this.initialized = false;
  }

  getResource<T>(name: string): T {
    return this.resources.get(name) as T;
  }

  private createMockAxios() {
    return {
      get: jest.fn(),
      post: jest.fn(),
      put: jest.fn(),
      delete: jest.fn(),
      patch: jest.fn()
    };
  }

  private createTestLogger() {
    return {
      debug: jest.fn(),
      info: jest.fn(),
      warn: jest.fn(),
      error: jest.fn()
    };
  }

  private setupMockTimers() {
    jest.useFakeTimers();
    return {
      advanceTimers: (ms: number) => jest.advanceTimersByTime(ms),
      runAllTimers: () => jest.runAllTimers()
    };
  }
}