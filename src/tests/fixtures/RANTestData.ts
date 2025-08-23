/**
 * Test Fixtures for RAN SDK Testing
 * Provides consistent test data for TDD London School testing
 */

export interface TestNodeConfig {
  nodeId: string;
  type: string;
  location?: {
    lat: number;
    lon: number;
    address?: string;
  };
  radioConfig?: any;
  cellConfig?: any[];
}

export interface TestCellConfig {
  cellId: string;
  parentNodeId: string;
  sector: number;
  azimuth: number;
  tilt?: number;
  power?: number;
  neighbors?: string[];
}

export interface TestParameterSet {
  path: string;
  value: any;
  type: 'string' | 'number' | 'boolean' | 'object' | 'array';
  constraints?: {
    min?: number;
    max?: number;
    enum?: any[];
    required?: boolean;
  };
}

// Standard ENodeB test configurations
export const testENodeBConfigs: TestNodeConfig[] = [
  {
    nodeId: 'TEST_ENODEB_001',
    type: 'LTE',
    location: {
      lat: 59.3293,
      lon: 18.0686,
      address: 'Stockholm, Sweden'
    },
    radioConfig: {
      maxPower: 43,
      frequencyBand: 20,
      bandwidth: '20MHz',
      mimo: '4x4',
      carrierAggregation: true
    },
    cellConfig: [
      { sector: 1, azimuth: 0, tilt: 5 },
      { sector: 2, azimuth: 120, tilt: 5 },
      { sector: 3, azimuth: 240, tilt: 5 }
    ]
  },
  {
    nodeId: 'TEST_ENODEB_002',
    type: 'LTE',
    location: {
      lat: 55.6761,
      lon: 12.5683,
      address: 'Copenhagen, Denmark'
    },
    radioConfig: {
      maxPower: 40,
      frequencyBand: 3,
      bandwidth: '15MHz',
      mimo: '2x2',
      carrierAggregation: false
    }
  },
  {
    nodeId: 'TEST_ENODEB_INVALID',
    type: 'LTE',
    // Missing required fields for validation testing
    radioConfig: {
      maxPower: 60, // Exceeds maximum
      frequencyBand: 999, // Invalid band
      bandwidth: '50MHz' // Invalid bandwidth
    }
  }
];

// Standard GNodeB test configurations
export const testGNodeBConfigs: TestNodeConfig[] = [
  {
    nodeId: 'TEST_GNODEB_001',
    type: '5G-SA',
    location: {
      lat: 60.1699,
      lon: 24.9384,
      address: 'Helsinki, Finland'
    },
    radioConfig: {
      maxPower: 46,
      frequencyBands: [3500, 28000],
      bandwidth: '100MHz',
      beamforming: true,
      massiveMIMO: '64x64',
      networkSlicing: true
    },
    cellConfig: [
      { sector: 1, azimuth: 0, tilt: 8, beams: 16 },
      { sector: 2, azimuth: 120, tilt: 8, beams: 16 }
    ]
  },
  {
    nodeId: 'TEST_GNODEB_002',
    type: '5G-NSA',
    location: {
      lat: 63.4305,
      lon: 10.3951,
      address: 'Trondheim, Norway'
    },
    radioConfig: {
      maxPower: 43,
      frequencyBands: [78, 3500],
      bandwidth: '50MHz',
      anchorBand: 20, // For NSA deployment
      dualConnectivity: true
    }
  }
];

// Standard Cell test configurations
export const testCellConfigs: TestCellConfig[] = [
  {
    cellId: 'TEST_CELL_001',
    parentNodeId: 'TEST_ENODEB_001',
    sector: 1,
    azimuth: 0,
    tilt: 5,
    power: 40,
    neighbors: ['TEST_CELL_002', 'TEST_CELL_003', 'TEST_CELL_004']
  },
  {
    cellId: 'TEST_CELL_002',
    parentNodeId: 'TEST_ENODEB_001',
    sector: 2,
    azimuth: 120,
    tilt: 5,
    power: 40,
    neighbors: ['TEST_CELL_001', 'TEST_CELL_003', 'TEST_CELL_005']
  },
  {
    cellId: 'TEST_CELL_NR_001',
    parentNodeId: 'TEST_GNODEB_001',
    sector: 1,
    azimuth: 0,
    tilt: 8,
    power: 46,
    neighbors: ['TEST_CELL_NR_002']
  }
];

// Standard parameter sets for testing
export const testParameterSets: Record<string, TestParameterSet[]> = {
  lte: [
    {
      path: 'Radio.MaxPower',
      value: 43,
      type: 'number',
      constraints: { min: 10, max: 46, required: true }
    },
    {
      path: 'System.Name',
      value: 'TEST_ENODEB_001',
      type: 'string',
      constraints: { required: true }
    },
    {
      path: 'Cell.Count',
      value: 3,
      type: 'number',
      constraints: { min: 1, max: 6, required: true }
    },
    {
      path: 'Network.IP',
      value: '192.168.1.100',
      type: 'string',
      constraints: { required: true }
    },
    {
      path: 'Feature.CarrierAggregation',
      value: true,
      type: 'boolean'
    }
  ],
  nr: [
    {
      path: 'NR.MaxPower',
      value: 46,
      type: 'number',
      constraints: { min: 10, max: 50, required: true }
    },
    {
      path: 'NR.FrequencyBands',
      value: [3500, 28000],
      type: 'array',
      constraints: { required: true }
    },
    {
      path: 'NR.Beamforming',
      value: true,
      type: 'boolean'
    },
    {
      path: 'NR.NetworkSlicing',
      value: {
        enabled: true,
        slices: [
          { id: 'eMBB', priority: 1 },
          { id: 'URLLC', priority: 2 }
        ]
      },
      type: 'object'
    }
  ]
};

// Mock CM Edit responses for testing
export const mockCMEditResponses = {
  successfulConnection: {
    connected: true,
    sessionId: 'mock-session-001'
  },
  
  failedConnection: {
    connected: false,
    error: 'Authentication failed'
  },
  
  successfulTransaction: {
    transactionId: 'tx-mock-001',
    status: 'active'
  },
  
  nodeCreationSuccess: {
    success: true,
    dn: 'ManagedElement=TEST_NODE_001',
    objectId: 'obj-001'
  },
  
  nodeCreationFailure: {
    success: false,
    error: 'Invalid node configuration',
    details: ['Missing required parameter: System.Name']
  },
  
  parameterGetSuccess: {
    found: true,
    attributes: {
      'System.Name': 'TEST_NODE_001',
      'Radio.MaxPower': 43,
      'Cell.Count': 3
    }
  },
  
  parameterGetNotFound: {
    found: false,
    error: 'Managed object not found'
  },
  
  parameterSetSuccess: {
    success: true,
    updatedParameters: ['Radio.MaxPower']
  },
  
  parameterSetFailure: {
    success: false,
    error: 'Parameter validation failed',
    failures: ['Radio.MaxPower: Value exceeds maximum']
  }
};

// Performance monitoring test data
export const testMonitoringData = {
  metrics: {
    cpu: Array.from({ length: 12 }, (_, i) => ({
      timestamp: new Date(Date.now() - (11 - i) * 300000),
      value: 30 + Math.random() * 40
    })),
    
    memory: Array.from({ length: 12 }, (_, i) => ({
      timestamp: new Date(Date.now() - (11 - i) * 300000),
      value: 50 + Math.random() * 30
    })),
    
    throughput: Array.from({ length: 12 }, (_, i) => ({
      timestamp: new Date(Date.now() - (11 - i) * 300000),
      value: 100 + Math.random() * 800
    }))
  },
  
  alerts: [
    {
      id: 'alert-001',
      nodeId: 'TEST_ENODEB_001',
      metric: 'cpu',
      severity: 'warning',
      threshold: 80,
      value: 85,
      timestamp: new Date(),
      acknowledged: false
    },
    {
      id: 'alert-002',
      nodeId: 'TEST_GNODEB_001',
      metric: 'memory',
      severity: 'critical',
      threshold: 90,
      value: 95,
      timestamp: new Date(),
      acknowledged: true
    }
  ],
  
  thresholds: {
    cpu: { warning: 70, critical: 85 },
    memory: { warning: 80, critical: 90 },
    throughput: { warning: 500, critical: 200 }
  }
};

// Automation workflow test data
export const testWorkflowData = {
  optimizationWorkflow: {
    id: 'optimization-001',
    name: 'Performance Optimization',
    steps: [
      {
        id: 'step-1',
        name: 'Analyze Performance Metrics',
        type: 'analysis',
        config: { timeRange: '1h', metrics: ['cpu', 'memory', 'throughput'] }
      },
      {
        id: 'step-2',
        name: 'Identify Bottlenecks',
        type: 'evaluation',
        config: { thresholds: testMonitoringData.thresholds }
      },
      {
        id: 'step-3',
        name: 'Apply Optimizations',
        type: 'action',
        config: { parameters: ['Radio.MaxPower', 'Cell.Tilt'] }
      }
    ]
  },
  
  maintenanceWorkflow: {
    id: 'maintenance-001',
    name: 'Scheduled Maintenance',
    steps: [
      {
        id: 'step-1',
        name: 'Backup Configuration',
        type: 'backup',
        config: { includeHistory: true }
      },
      {
        id: 'step-2',
        name: 'Apply Updates',
        type: 'update',
        config: { updateType: 'software' }
      },
      {
        id: 'step-3',
        name: 'Validate Configuration',
        type: 'validation',
        config: { fullValidation: true }
      },
      {
        id: 'step-4',
        name: 'Restore if Failed',
        type: 'rollback',
        config: { condition: 'validationFailed' }
      }
    ]
  }
};

// Error scenarios for testing
export const testErrorScenarios = {
  connectionTimeout: {
    error: 'Connection timeout',
    code: 'TIMEOUT',
    retryable: true
  },
  
  authenticationFailure: {
    error: 'Authentication failed',
    code: 'AUTH_FAILED',
    retryable: false
  },
  
  parameterValidationError: {
    error: 'Parameter validation failed',
    code: 'VALIDATION_ERROR',
    details: ['Radio.MaxPower exceeds maximum value'],
    retryable: false
  },
  
  transactionRollback: {
    error: 'Transaction rolled back due to constraint violation',
    code: 'TRANSACTION_ROLLBACK',
    retryable: true
  },
  
  nodeNotFound: {
    error: 'Node not found',
    code: 'NODE_NOT_FOUND',
    retryable: false
  }
};

// Helper functions for test data
export const TestDataHelpers = {
  getRandomENodeBConfig: (): TestNodeConfig => {
    return testENodeBConfigs[Math.floor(Math.random() * testENodeBConfigs.length)];
  },
  
  getRandomGNodeBConfig: (): TestNodeConfig => {
    return testGNodeBConfigs[Math.floor(Math.random() * testGNodeBConfigs.length)];
  },
  
  createInvalidConfig: (baseConfig: TestNodeConfig): TestNodeConfig => {
    return {
      ...baseConfig,
      nodeId: '', // Invalid empty ID
      radioConfig: {
        ...baseConfig.radioConfig,
        maxPower: 999, // Invalid power level
        frequencyBand: -1 // Invalid frequency
      }
    };
  },
  
  generateMetrics: (nodeId: string, hours: number = 1) => {
    const points = hours * 12; // 5-minute intervals
    return {
      nodeId,
      timeRange: { 
        start: new Date(Date.now() - hours * 3600000),
        end: new Date()
      },
      data: {
        cpu: Array.from({ length: points }, (_, i) => ({
          timestamp: new Date(Date.now() - (points - 1 - i) * 300000),
          value: 30 + Math.random() * 40
        })),
        memory: Array.from({ length: points }, (_, i) => ({
          timestamp: new Date(Date.now() - (points - 1 - i) * 300000),
          value: 50 + Math.random() * 30
        }))
      }
    };
  }
};