# TDD London School Test Structure

This test suite follows the **London School of TDD** (mockist approach), emphasizing behavior verification and mock-driven development.

## 🧪 Testing Philosophy

### London School Principles
- **Mock all collaborators** - Every external dependency is mocked
- **Verify interactions** - Focus on HOW objects collaborate, not just WHAT they return  
- **Outside-in development** - Start with acceptance tests, drive down to implementation
- **Behavior over state** - Test the conversations between objects
- **Contract-driven** - Use mocks to define clear interfaces

## 📁 Directory Structure

```
tests/
├── unit/                    # Unit tests with extensive mocking
│   ├── RANNodeFactory.test.ts
│   ├── ConfigurationManager.test.ts
│   ├── AutomationAgent.test.ts
│   └── PerformanceMonitor.test.ts
├── integration/             # Integration tests for component interactions
│   ├── RANNodeLifecycle.test.ts
│   ├── WorkflowOrchestration.test.ts
│   └── EventCoordination.test.ts
├── contracts/               # Contract tests for interface verification
│   ├── RANNodeInterface.contract.test.ts
│   ├── ConfigurationAPI.contract.test.ts
│   └── MonitoringAPI.contract.test.ts
├── mocks/                   # Mock implementations and factories
│   ├── MockRANNodeFactory.ts
│   ├── MockConfigurationManager.ts
│   ├── MockCMEditClient.ts
│   ├── MockMonitoringService.ts
│   └── index.ts
├── fixtures/                # Test data and configuration
│   ├── RANTestData.ts
│   ├── TestEnvironmentConfig.ts
│   └── MockResponses.ts
├── test-setup.ts           # Global test configuration
├── global-setup.ts         # Test environment initialization
└── global-teardown.ts      # Test environment cleanup
```

## 🚀 Running Tests

### All Tests
```bash
npm test
```

### Specific Test Categories
```bash
# Unit tests only
npm test -- tests/unit

# Integration tests only
npm test -- tests/integration

# Contract tests only
npm test -- tests/contracts

# Watch mode
npm test -- --watch

# Coverage report
npm test -- --coverage
```

### TDD Workflow
```bash
# London School TDD cycle
npm test -- --watch --verbose
```

## 🎯 Test Examples

### Unit Test Pattern (London School)
```typescript
describe('ConfigurationManager', () => {
  let configManager: ConfigurationManager;
  let mockCMEditClient: any;
  let mockLogger: any;

  beforeEach(() => {
    // London School: Mock all collaborators
    mockCMEditClient = createMockCMEditClient();
    mockLogger = createMockLogger();
    
    // Inject mocks
    configManager = new ConfigurationManager(mockCMEditClient, mockLogger);
  });

  it('should coordinate with CM Edit client during configuration load', async () => {
    // Arrange - Define expected behavior
    const nodeId = 'TEST-001';
    
    // Act
    await configManager.loadConfiguration(nodeId);
    
    // Assert - Verify the conversation (London School focus)
    expect(mockCMEditClient.connect).toHaveBeenCalled();
    expect(mockCMEditClient.getMO).toHaveBeenCalledWith(`ManagedElement=${nodeId}`);
    expect(mockLogger.info).toHaveBeenCalledWith(`Configuration loaded: ${nodeId}`);
  });
});
```

### Integration Test Pattern
```typescript
describe('RAN Node Lifecycle Integration', () => {
  it('should coordinate complete node provisioning workflow', async () => {
    // Test how multiple mocked components work together
    const node = await factory.createENodeB(config);
    await automationAgent.enableAutomation(node.id);
    await performanceMonitor.startNodeMonitoring(node.id);
    
    // Verify cross-component event coordination
    expect(mockEventEmitter.emit).toHaveBeenCalledWith('nodeCreated');
    expect(mockEventEmitter.emit).toHaveBeenCalledWith('automationEnabled');
    expect(mockEventEmitter.emit).toHaveBeenCalledWith('monitoringStarted');
  });
});
```

### Contract Test Pattern
```typescript
describe('RANNode Interface Contract', () => {
  it('should verify all implementations satisfy the contract', () => {
    const mockRANNode = createContractCompliantMock(RANNodeContract);
    
    // Verify contract compliance
    verifyContractCompliance(mockRANNode, RANNodeContract);
    
    // Verify behavior contracts
    await mockRANNode.configure({});
    expect(mockRANNode.configure).toHaveBeenCalledWith({});
  });
});
```

## 🔧 Mock Infrastructure

### Mock Creation Pattern
```typescript
// Create behavior-focused mocks
const mockService = createMockConfigurationManager();

// Verify interactions, not just return values
expect(mockService.loadConfiguration).toHaveBeenCalledWith(nodeId);
expect(mockService.validate).toHaveBeenCalledBefore(mockService.save);
```

### Behavior Verification Helpers
```typescript
// Use custom matchers for interaction verification
expect(mockFirstService.method).toHaveBeenCalledBefore(mockSecondService.method);
expect(mockImplementation).toSatisfyContract(expectedContract);
expect(mockService).toHaveCorrectInteractionSequence();
```

## 📊 Coverage Requirements

- **Branches**: 85%
- **Functions**: 90%
- **Lines**: 90%
- **Statements**: 90%

Focus on **behavior coverage** rather than just code coverage.

## 🎨 London School vs Chicago School

| Aspect | London School (Our Approach) | Chicago School |
|--------|-------------------------------|----------------|
| Mocking | Mock all collaborators | Mock only external dependencies |
| Focus | Object interactions/behavior | State verification |
| Tests | Verify the conversation | Verify the outcome |
| Design | Outside-in, mock-driven | Inside-out, state-driven |
| Feedback | Fast, isolated | Realistic, integrated |

## 🛠 Custom Test Utilities

### Mock Behavior Verification
```typescript
import { verifyConfigurationManagerBehavior } from '../mocks';

// Verify complex interaction patterns
verifyConfigurationManagerBehavior.verifyUpdateSequence(mockManager, nodeId);
verifyConfigurationManagerBehavior.verifyTransactionBoundaries(mockManager);
```

### Contract Testing
```typescript
import { verifyContractCompliance } from '../contracts/utils';

// Ensure implementations satisfy contracts
verifyContractCompliance(implementation, RANNodeContract);
```

### Test Data Management
```typescript
import { TestDataHelpers, testENodeBConfigs } from '../fixtures/RANTestData';

// Use consistent test data
const config = TestDataHelpers.getRandomENodeBConfig();
const metrics = TestDataHelpers.generateMetrics('node-001', 2);
```

## 🚦 Test Execution Flow

1. **Global Setup** - Initialize test environment
2. **Test Setup** - Create mocks and inject dependencies  
3. **Test Execution** - Verify behavior and interactions
4. **Cleanup** - Reset mocks and clear state
5. **Global Teardown** - Clean up test environment

## 📈 Benefits of London School TDD

- ✅ **Fast Feedback** - Tests run quickly with mocks
- ✅ **Design Focus** - Drives better object design
- ✅ **Behavior Verification** - Tests what objects DO, not what they ARE
- ✅ **Contract Definition** - Mocks define clear interfaces
- ✅ **Isolation** - Each test is completely independent
- ✅ **Refactoring Safety** - Tests protect against breaking changes in collaborations

## 🔍 Debugging Tests

### Verbose Output
```bash
npm test -- --verbose
```

### Mock Call Inspection
```typescript
console.log('Mock calls:', mockService.method.mock.calls);
console.log('Call order:', mockService.method.mock.invocationCallOrder);
```

### Behavior Debugging
```typescript
// Use test utilities for debugging
verifyMockCleanup(mockService1, mockService2);
verifyCallOrder(mockFirst, mockSecond);
```

---

**Remember**: In London School TDD, we test the **conversation between objects**, not just their individual behavior. Focus on **HOW** components collaborate, not just **WHAT** they return.