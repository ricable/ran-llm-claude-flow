/**
 * Global Test Teardown for TDD London School
 * Cleans up test environment and shared resources
 */

export default async () => {
  console.log('🧹 Cleaning up TDD Test Environment...');
  
  // Cleanup global test configuration
  const testConfig = (global as any).testConfig;
  if (testConfig) {
    await testConfig.cleanup();
    delete (global as any).testConfig;
  }
  
  // Clear all timers and intervals
  jest.clearAllTimers();
  jest.useRealTimers();
  
  console.log('✅ TDD Test Environment Cleaned');
};