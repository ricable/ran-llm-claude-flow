/**
 * Core type definitions for Ericsson RAN SDK
 */

export enum NodeType {
  ENODEB = 'eNodeB',
  GNODEB = 'gNodeB',
  CELL = 'Cell'
}

export enum CellType {
  LTE = 'LTE',
  NR = 'NR'
}

export enum ParameterType {
  RF = 'RF',
  MOBILITY = 'MOBILITY',
  HANDOVER = 'HANDOVER',
  POWER = 'POWER',
  ANTENNA = 'ANTENNA',
  CARRIER = 'CARRIER'
}

export interface NodeConfiguration {
  nodeId: string;
  nodeName: string;
  nodeType: NodeType;
  siteId: string;
  coordinates?: {
    latitude: number;
    longitude: number;
  };
  parameters: Record<string, any>;
}

export interface CellConfiguration {
  cellId: string;
  cellName: string;
  cellType: CellType;
  parentNodeId: string;
  sector: number;
  frequency: number;
  bandwidth: number;
  parameters: Record<string, any>;
}

export interface Parameter {
  name: string;
  type: ParameterType;
  value: any;
  unit?: string;
  range?: {
    min: number;
    max: number;
  };
  description?: string;
  validationRules?: ValidationRule[];
}

export interface ValidationRule {
  type: 'range' | 'enum' | 'pattern' | 'custom';
  rule: any;
  message: string;
}

export interface AutomationTask {
  id: string;
  name: string;
  description: string;
  nodeIds: string[];
  parameters: Parameter[];
  schedule?: CronSchedule;
  status: TaskStatus;
  createdAt: Date;
  updatedAt: Date;
}

export interface CronSchedule {
  expression: string;
  timezone?: string;
  enabled: boolean;
}

export enum TaskStatus {
  PENDING = 'PENDING',
  RUNNING = 'RUNNING',
  COMPLETED = 'COMPLETED',
  FAILED = 'FAILED',
  CANCELLED = 'CANCELLED'
}

export interface CMEditCommand {
  operation: 'create' | 'update' | 'delete' | 'get';
  mo: string; // Managed Object
  attributes?: Record<string, any>;
  filter?: string;
}

export interface CMEditResponse {
  success: boolean;
  data?: any;
  error?: string;
  commandId: string;
  timestamp: Date;
}

export interface KPIMetric {
  name: string;
  value: number;
  unit: string;
  timestamp: Date;
  nodeId: string;
  cellId?: string;
}

export interface PerformanceData {
  nodeId: string;
  timestamp: Date;
  metrics: KPIMetric[];
  alarms?: Alarm[];
}

export interface Alarm {
  id: string;
  severity: 'CRITICAL' | 'MAJOR' | 'MINOR' | 'WARNING';
  type: string;
  description: string;
  nodeId: string;
  timestamp: Date;
  acknowledged: boolean;
}