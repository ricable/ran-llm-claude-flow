/**
 * Global Test Setup for TDD London School
 * Initializes test environment and shared resources
 */

import { TestEnvironmentConfig } from './fixtures/TestEnvironmentConfig';

export default async () => {
  console.log('🧪 Initializing TDD London School Test Environment...');
  
  // Initialize test environment configuration
  process.env.NODE_ENV = 'test';
  process.env.LOG_LEVEL = 'error';
  
  // Setup global test configuration
  const testConfig = new TestEnvironmentConfig();
  await testConfig.initialize();
  
  // Store config in global scope for tests
  (global as any).testConfig = testConfig;
  
  console.log('✅ TDD Test Environment Ready');
};