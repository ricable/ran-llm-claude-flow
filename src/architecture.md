# Ericsson RAN Automation SDK Architecture

## Overview

This document defines a refactored architecture for the Ericsson RAN Automation SDK, designed using factory patterns and DRY principles to create a clean, testable, and maintainable system that supports TDD London School practices.

## System Architecture Diagram

```
                                     ┌─────────────────────────────────────────────────┐
                                     │              RAN SDK Core                       │
                                     └─────────────────────────────────────────────────┘
                                                              │
                                     ┌─────────────────────────────────────────────────┐
                                     │           Factory Layer                         │
                                     │  ┌─────────────────┐ ┌─────────────────────┐   │
                                     │  │ RANNodeFactory  │ │ AutomationFactory   │   │
                                     │  │ - 4G/5G Nodes   │ │ - Agents & Workflows │   │
                                     │  └─────────────────┘ └─────────────────────┘   │
                                     │  ┌─────────────────┐ ┌─────────────────────┐   │
                                     │  │ ConfigFactory   │ │ MonitoringFactory   │   │
                                     │  │ - Configurations │ │ - KPI & Metrics     │   │
                                     │  └─────────────────┘ └─────────────────────┘   │
                                     └─────────────────────────────────────────────────┘
                                                              │
                    ┌─────────────────┬────────────────────────┼────────────────────┬─────────────────┐
                    │                 │                        │                    │                 │
       ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
       │   RAN Nodes     │  │ Configuration   │  │   Automation    │  │   Monitoring    │  │   Integration   │
       │    Module       │  │     Module      │  │     Module      │  │     Module      │  │     Module      │
       └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
                │                        │                        │                        │                        │
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   eNodeB (4G)   │    │   Parameters    │    │    AI Agents    │    │   KPI Metrics   │    │   External API  │
    │   gNodeB (5G)   │    │   Policies      │    │   Workflows     │    │   Performance   │    │   Data Sources  │
    │   EUtranCell    │    │   Templates     │    │   Operations    │    │   Alarms        │    │   Integrations  │
    │   NRCell        │    │   Validation    │    │   Optimization  │    │   Reports       │    │   Export/Import │
    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Design Principles

### 1. Factory Pattern Implementation
- **Centralized Object Creation**: All RAN components created through specialized factories
- **Type Safety**: Strong typing with TypeScript interfaces for all factory products
- **Extensibility**: Easy addition of new RAN technologies and node types
- **Configuration Driven**: Factory behavior controlled by configuration injection

### 2. DRY Principle Adherence
- **Base Classes**: Shared functionality in abstract base classes
- **Composition over Inheritance**: Prefer composition patterns for flexibility
- **Utility Libraries**: Common operations in reusable utility modules
- **Template Systems**: Configuration and operation templates

### 3. TDD London School Support
- **Mock-Friendly Design**: All dependencies injectable via interfaces
- **Isolated Units**: Each component testable in isolation
- **Behavior Verification**: Focus on interaction testing
- **Interface Segregation**: Small, focused interfaces

## Module Architecture

### RAN Nodes Module
```typescript
interface IRANNodeFactory {
  create4GNode(config: NodeConfig): IRANNode;
  create5GNode(config: NodeConfig): IRANNode;
  createCell(nodeType: NodeType, config: CellConfig): IRANCell;
}

interface IRANNode {
  getId(): string;
  getType(): NodeType;
  getCells(): IRANCell[];
  configure(config: NodeConfig): Promise<void>;
  monitor(): Promise<NodeStatus>;
  optimize(): Promise<OptimizationResult>;
}
```

**Responsibilities**:
- RAN node lifecycle management (eNodeB, gNodeB)
- Cell management (EUtranCell, NRCell)
- Node configuration and status monitoring
- Performance optimization operations

**Factory Products**:
- `EricssonENodeB`: 4G base station implementation
- `EricssonGNodeB`: 5G base station implementation
- `EUtranCell`: 4G cell implementation
- `NRCell`: 5G cell implementation

### Configuration Module
```typescript
interface IConfigurationManager {
  loadTemplate(templateId: string): ConfigTemplate;
  validateConfig(config: any): ValidationResult;
  applyConfig(node: IRANNode, config: any): Promise<void>;
  getParameterDefinitions(): ParameterDefinition[];
}
```

**Responsibilities**:
- Parameter management and validation
- Configuration templates and policies
- Configuration deployment and rollback
- Compliance checking

**Key Components**:
- Parameter registry with 4G/5G specific definitions
- Template engine for configuration generation
- Validation engine with business rules
- Configuration versioning and audit trails

### Automation Module
```typescript
interface IAutomationAgent {
  execute(operation: Operation): Promise<OperationResult>;
  scheduleOperation(operation: Operation, schedule: Schedule): Promise<void>;
  getCapabilities(): AgentCapability[];
  getStatus(): AgentStatus;
}
```

**Responsibilities**:
- AI-powered automation agents
- Workflow orchestration
- Optimization algorithms
- Self-healing operations

**Agent Types**:
- `OptimizationAgent`: Network performance optimization
- `MaintenanceAgent`: Automated maintenance tasks  
- `AnalysisAgent`: Performance and fault analysis
- `ComplianceAgent`: Configuration compliance monitoring

### Monitoring Module
```typescript
interface IMonitoringService {
  collectKPIs(nodeId: string): Promise<KPICollection>;
  generateReport(criteria: ReportCriteria): Promise<Report>;
  setupAlert(alertConfig: AlertConfig): Promise<void>;
  getPerformanceMetrics(): PerformanceMetrics;
}
```

**Responsibilities**:
- KPI collection and aggregation (4G/5G specific)
- Performance monitoring and alerting
- Report generation and dashboards
- Anomaly detection

## Factory Implementation Strategy

### 1. Abstract Factory Pattern
```typescript
abstract class RANNodeFactory implements IRANNodeFactory {
  protected configManager: IConfigurationManager;
  protected monitoringService: IMonitoringService;
  
  constructor(deps: FactoryDependencies) {
    this.configManager = deps.configManager;
    this.monitoringService = deps.monitoringService;
  }
  
  abstract create4GNode(config: NodeConfig): IRANNode;
  abstract create5GNode(config: NodeConfig): IRANNode;
}

class EricssonRANNodeFactory extends RANNodeFactory {
  create4GNode(config: NodeConfig): IRANNode {
    return new EricssonENodeB(config, this.configManager, this.monitoringService);
  }
  
  create5GNode(config: NodeConfig): IRANNode {
    return new EricssonGNodeB(config, this.configManager, this.monitoringService);
  }
}
```

### 2. Builder Pattern for Complex Objects
```typescript
class NodeConfigurationBuilder {
  private config: Partial<NodeConfig> = {};
  
  withBasicSettings(settings: BasicSettings): this {
    this.config.basic = settings;
    return this;
  }
  
  withRadioSettings(settings: RadioSettings): this {
    this.config.radio = settings;
    return this;
  }
  
  withQoSSettings(settings: QoSSettings): this {
    this.config.qos = settings;
    return this;
  }
  
  build(): NodeConfig {
    return { ...this.config } as NodeConfig;
  }
}
```

### 3. Registry Pattern for Dynamic Discovery
```typescript
class ComponentRegistry {
  private static factories: Map<string, Factory> = new Map();
  
  static register<T>(type: string, factory: Factory<T>): void {
    this.factories.set(type, factory);
  }
  
  static create<T>(type: string, config: any): T {
    const factory = this.factories.get(type);
    if (!factory) {
      throw new Error(`Factory not found for type: ${type}`);
    }
    return factory.create(config);
  }
}
```

## Dependency Injection Architecture

### 1. Container Configuration
```typescript
interface ServiceContainer {
  register<T>(token: symbol, implementation: Constructor<T>): void;
  resolve<T>(token: symbol): T;
  createScope(): ServiceContainer;
}

// Service tokens
const TOKENS = {
  RANNodeFactory: Symbol('RANNodeFactory'),
  ConfigurationManager: Symbol('ConfigurationManager'),
  MonitoringService: Symbol('MonitoringService'),
  AutomationAgent: Symbol('AutomationAgent')
};
```

### 2. Scoped Services
```typescript
class ScopedServiceProvider {
  constructor(private container: ServiceContainer) {}
  
  createNodeScope(nodeId: string): ServiceContainer {
    const scope = this.container.createScope();
    
    // Register node-specific services
    scope.register(TOKENS.NodeContext, new NodeContext(nodeId));
    scope.register(TOKENS.NodeLogger, new NodeLogger(nodeId));
    
    return scope;
  }
}
```

## Error Handling Strategy

### 1. Domain-Specific Exceptions
```typescript
abstract class RANException extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly nodeId?: string
  ) {
    super(message);
    this.name = this.constructor.name;
  }
}

class ConfigurationException extends RANException {}
class NodeCommunicationException extends RANException {}
class OptimizationException extends RANException {}
```

### 2. Circuit Breaker Pattern
```typescript
class CircuitBreaker {
  private failureCount = 0;
  private state: 'CLOSED' | 'OPEN' | 'HALF_OPEN' = 'CLOSED';
  
  async execute<T>(operation: () => Promise<T>): Promise<T> {
    if (this.state === 'OPEN') {
      throw new Error('Circuit breaker is OPEN');
    }
    
    try {
      const result = await operation();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }
}
```

## Logging and Observability

### 1. Structured Logging
```typescript
interface Logger {
  debug(message: string, context?: LogContext): void;
  info(message: string, context?: LogContext): void;
  warn(message: string, context?: LogContext): void;
  error(message: string, error?: Error, context?: LogContext): void;
}

interface LogContext {
  nodeId?: string;
  operationId?: string;
  userId?: string;
  correlationId?: string;
}
```

### 2. Metrics Collection
```typescript
interface MetricsCollector {
  incrementCounter(name: string, labels?: Record<string, string>): void;
  recordHistogram(name: string, value: number, labels?: Record<string, string>): void;
  setGauge(name: string, value: number, labels?: Record<string, string>): void;
}
```

## Testing Strategy

### 1. Unit Testing with Mocks
```typescript
// Test example using London School TDD
describe('EricssonENodeB', () => {
  let mockConfigManager: jest.Mocked<IConfigurationManager>;
  let mockMonitoringService: jest.Mocked<IMonitoringService>;
  let eNodeB: EricssonENodeB;
  
  beforeEach(() => {
    mockConfigManager = createMock<IConfigurationManager>();
    mockMonitoringService = createMock<IMonitoringService>();
    eNodeB = new EricssonENodeB(testConfig, mockConfigManager, mockMonitoringService);
  });
  
  it('should configure node successfully', async () => {
    // Given
    const config = createTestNodeConfig();
    mockConfigManager.validateConfig.mockResolvedValue({ isValid: true });
    
    // When
    await eNodeB.configure(config);
    
    // Then
    expect(mockConfigManager.validateConfig).toHaveBeenCalledWith(config);
    expect(mockConfigManager.applyConfig).toHaveBeenCalledWith(eNodeB, config);
  });
});
```

### 2. Integration Testing
```typescript
class TestRANNodeFactory extends RANNodeFactory {
  create4GNode(config: NodeConfig): IRANNode {
    return new TestENodeB(config, this.configManager, this.monitoringService);
  }
}
```

## Performance Considerations

### 1. Caching Strategy
```typescript
interface CacheManager {
  get<T>(key: string): Promise<T | null>;
  set<T>(key: string, value: T, ttl?: number): Promise<void>;
  invalidate(pattern: string): Promise<void>;
}
```

### 2. Connection Pooling
```typescript
class ConnectionPoolManager {
  private pools: Map<string, ConnectionPool> = new Map();
  
  getPool(nodeId: string): ConnectionPool {
    if (!this.pools.has(nodeId)) {
      this.pools.set(nodeId, new ConnectionPool(nodeId, poolConfig));
    }
    return this.pools.get(nodeId)!;
  }
}
```

## Migration Strategy

### 1. Incremental Refactoring
1. **Phase 1**: Create interface definitions
2. **Phase 2**: Implement base factories and abstract classes
3. **Phase 3**: Migrate existing node implementations
4. **Phase 4**: Add automation and monitoring layers
5. **Phase 5**: Complete integration and testing

### 2. Backward Compatibility
```typescript
class LegacyAdapter implements IRANNode {
  constructor(private legacyNode: LegacyNodeType) {}
  
  getId(): string {
    return this.legacyNode.nodeId;
  }
  
  // Adapter methods to bridge old and new interfaces
}
```

## Conclusion

This architecture provides:
- **Scalability**: Factory patterns enable easy extension for new RAN technologies
- **Testability**: Dependency injection and interfaces support comprehensive testing
- **Maintainability**: DRY principles reduce code duplication and improve consistency
- **Flexibility**: Modular design allows independent development and deployment
- **Quality**: Strong typing and validation ensure reliability

The factory-based approach centralizes object creation while maintaining type safety and enabling easy mocking for tests. The modular architecture separates concerns while providing clear interfaces for integration.