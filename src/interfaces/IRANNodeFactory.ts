/**
 * Factory Interface for RAN Node Creation
 * 
 * Provides a centralized way to create different types of RAN nodes (4G/5G)
 * with proper dependency injection and configuration management.
 */

import { IRANNode } from './IRANNode';
import { IRANCell } from './IRANCell';

/**
 * Configuration for creating RAN nodes
 */
export interface NodeConfig {
  /** Unique identifier for the node */
  nodeId: string;
  /** Node name for display purposes */
  nodeName: string;
  /** RAN technology version */
  technology: 'LTE' | 'NR' | 'LTE-NR';
  /** Vendor-specific configuration */
  vendorConfig: VendorConfig;
  /** Basic node settings */
  basicSettings: BasicNodeSettings;
  /** Radio configuration */
  radioConfig: RadioConfig;
  /** Transport configuration */
  transportConfig: TransportConfig;
  /** Initial cells to create */
  initialCells?: CellConfig[];
}

/**
 * Configuration for creating RAN cells
 */
export interface CellConfig {
  /** Cell identifier */
  cellId: string;
  /** Cell name */
  cellName: string;
  /** Cell type */
  cellType: 'EUtranCell' | 'NRCell';
  /** Physical cell identifier */
  pci: number;
  /** Frequency configuration */
  frequency: FrequencyConfig;
  /** Power settings */
  powerSettings: PowerConfig;
  /** QoS configuration */
  qosConfig: QoSConfig;
}

/**
 * Vendor-specific configuration
 */
export interface VendorConfig {
  /** Vendor name (e.g., 'Ericsson', 'Nokia', 'Huawei') */
  vendor: string;
  /** Software version */
  softwareVersion: string;
  /** Hardware platform */
  hardwarePlatform: string;
  /** Vendor-specific parameters */
  vendorParameters: Record<string, any>;
}

/**
 * Basic node settings
 */
export interface BasicNodeSettings {
  /** Site identifier */
  siteId: string;
  /** Geographic coordinates */
  coordinates: {
    latitude: number;
    longitude: number;
    altitude?: number;
  };
  /** Tracking area code (for LTE) or tracking area identifier (for NR) */
  trackingAreaCode: number;
  /** Administrative state */
  administrativeState: 'LOCKED' | 'UNLOCKED' | 'SHUTTINGDOWN';
}

/**
 * Radio configuration
 */
export interface RadioConfig {
  /** Supported frequency bands */
  supportedBands: FrequencyBand[];
  /** Antenna configuration */
  antennaConfig: AntennaConfig;
  /** MIMO configuration */
  mimoConfig: MimoConfig;
  /** Carrier aggregation settings */
  carrierAggregation: CarrierAggregationConfig;
}

/**
 * Transport configuration
 */
export interface TransportConfig {
  /** Backhaul configuration */
  backhaul: BackhaulConfig;
  /** IP configuration */
  ipConfig: IPConfig;
  /** QoS transport settings */
  transportQos: TransportQoSConfig;
}

/**
 * Frequency configuration
 */
export interface FrequencyConfig {
  /** Operating band */
  band: number;
  /** Center frequency in Hz */
  centerFrequency: number;
  /** Bandwidth in Hz */
  bandwidth: number;
  /** EARFCN (for LTE) or ARFCN (for NR) */
  channelNumber: number;
}

/**
 * Frequency band definition
 */
export interface FrequencyBand {
  /** Band number */
  bandNumber: number;
  /** Band name/designation */
  bandName: string;
  /** Uplink frequency range */
  uplinkRange: FrequencyRange;
  /** Downlink frequency range */
  downlinkRange: FrequencyRange;
  /** Supported bandwidths */
  supportedBandwidths: number[];
}

/**
 * Frequency range
 */
export interface FrequencyRange {
  /** Start frequency in Hz */
  startFreq: number;
  /** End frequency in Hz */
  endFreq: number;
}

/**
 * Power configuration
 */
export interface PowerConfig {
  /** Maximum transmit power in dBm */
  maxTransmitPower: number;
  /** Reference signal power in dBm */
  referenceSignalPower: number;
  /** Power control settings */
  powerControl: PowerControlConfig;
}

/**
 * Power control configuration
 */
export interface PowerControlConfig {
  /** Enable dynamic power control */
  dynamicPowerControl: boolean;
  /** Power control algorithm */
  algorithm: 'FIXED' | 'ADAPTIVE' | 'ML_BASED';
  /** Power adjustment step size in dB */
  stepSize: number;
  /** Target SINR threshold */
  targetSinr: number;
}

/**
 * Antenna configuration
 */
export interface AntennaConfig {
  /** Number of antenna elements */
  antennaElements: number;
  /** Antenna patterns */
  antennaPatterns: AntennaPattern[];
  /** Beamforming configuration */
  beamforming: BeamformingConfig;
}

/**
 * Antenna pattern
 */
export interface AntennaPattern {
  /** Pattern identifier */
  patternId: string;
  /** Azimuth pattern */
  azimuthPattern: number[];
  /** Elevation pattern */
  elevationPattern: number[];
  /** Gain in dBi */
  gain: number;
}

/**
 * Beamforming configuration
 */
export interface BeamformingConfig {
  /** Enable beamforming */
  enabled: boolean;
  /** Beamforming type */
  type: 'ANALOG' | 'DIGITAL' | 'HYBRID';
  /** Number of beams */
  numBeams: number;
  /** Beam patterns */
  beamPatterns: BeamPattern[];
}

/**
 * Beam pattern
 */
export interface BeamPattern {
  /** Beam identifier */
  beamId: string;
  /** Azimuth angle in degrees */
  azimuthAngle: number;
  /** Elevation angle in degrees */
  elevationAngle: number;
  /** Beam width in degrees */
  beamWidth: number;
  /** Beam gain in dB */
  gain: number;
}

/**
 * MIMO configuration
 */
export interface MimoConfig {
  /** Transmission mode */
  transmissionMode: TransmissionMode;
  /** Number of layers */
  numLayers: number;
  /** Spatial multiplexing settings */
  spatialMultiplexing: SpatialMultiplexingConfig;
}

/**
 * Transmission mode enumeration
 */
export enum TransmissionMode {
  SISO = 'SISO',
  SIMO = 'SIMO',
  MISO = 'MISO',
  MIMO = 'MIMO',
  MASSIVE_MIMO = 'MASSIVE_MIMO'
}

/**
 * Spatial multiplexing configuration
 */
export interface SpatialMultiplexingConfig {
  /** Enable spatial multiplexing */
  enabled: boolean;
  /** Number of spatial streams */
  numStreams: number;
  /** Precoding matrix selection */
  precodingMatrix: string;
}

/**
 * Carrier aggregation configuration
 */
export interface CarrierAggregationConfig {
  /** Enable carrier aggregation */
  enabled: boolean;
  /** Primary component carrier */
  primaryCarrier: ComponentCarrier;
  /** Secondary component carriers */
  secondaryCarriers: ComponentCarrier[];
  /** Aggregation strategy */
  aggregationStrategy: 'CONTIGUOUS' | 'NON_CONTIGUOUS' | 'INTER_BAND';
}

/**
 * Component carrier configuration
 */
export interface ComponentCarrier {
  /** Carrier identifier */
  carrierId: string;
  /** Frequency configuration */
  frequency: FrequencyConfig;
  /** Bandwidth allocation */
  bandwidth: number;
  /** Carrier type */
  carrierType: 'PCC' | 'SCC';
}

/**
 * QoS configuration
 */
export interface QoSConfig {
  /** QCI/5QI profiles */
  qciProfiles: QCIProfile[];
  /** ARP (Allocation and Retention Priority) settings */
  arpSettings: ARPSettings[];
  /** GBR (Guaranteed Bit Rate) settings */
  gbrSettings: GBRSettings;
}

/**
 * QCI/5QI profile
 */
export interface QCIProfile {
  /** QCI or 5QI identifier */
  identifier: number;
  /** Resource type */
  resourceType: 'GBR' | 'NON_GBR' | 'DELAY_CRITICAL';
  /** Priority level */
  priority: number;
  /** Packet delay budget in ms */
  packetDelayBudget: number;
  /** Packet error loss rate */
  packetErrorLossRate: number;
}

/**
 * ARP settings
 */
export interface ARPSettings {
  /** Priority level (1-15, 1 = highest) */
  priorityLevel: number;
  /** Preemption capability */
  preemptionCapability: 'SHALL_NOT_TRIGGER' | 'MAY_TRIGGER';
  /** Preemption vulnerability */
  preemptionVulnerability: 'NOT_PREEMPTABLE' | 'PREEMPTABLE';
}

/**
 * GBR settings
 */
export interface GBRSettings {
  /** Guaranteed bit rate uplink in bps */
  gbrUl: number;
  /** Guaranteed bit rate downlink in bps */
  gbrDl: number;
  /** Maximum bit rate uplink in bps */
  mbrUl: number;
  /** Maximum bit rate downlink in bps */
  mbrDl: number;
}

/**
 * Backhaul configuration
 */
export interface BackhaulConfig {
  /** Backhaul type */
  type: 'FIBER' | 'MICROWAVE' | 'SATELLITE' | 'COPPER';
  /** Capacity in bps */
  capacity: number;
  /** Latency in ms */
  latency: number;
  /** Reliability percentage */
  reliability: number;
}

/**
 * IP configuration
 */
export interface IPConfig {
  /** Management IP address */
  managementIp: string;
  /** Control plane IP */
  controlPlaneIp: string;
  /** User plane IP */
  userPlaneIp: string;
  /** VLAN configuration */
  vlanConfig: VLANConfig[];
}

/**
 * VLAN configuration
 */
export interface VLANConfig {
  /** VLAN ID */
  vlanId: number;
  /** VLAN name */
  vlanName: string;
  /** IP address */
  ipAddress: string;
  /** Subnet mask */
  subnetMask: string;
  /** Default gateway */
  defaultGateway: string;
}

/**
 * Transport QoS configuration
 */
export interface TransportQoSConfig {
  /** DSCP marking rules */
  dscpRules: DSCPRule[];
  /** Traffic shaping policies */
  trafficShaping: TrafficShapingPolicy[];
}

/**
 * DSCP marking rule
 */
export interface DSCPRule {
  /** Traffic class */
  trafficClass: string;
  /** DSCP value */
  dscpValue: number;
  /** Priority */
  priority: number;
}

/**
 * Traffic shaping policy
 */
export interface TrafficShapingPolicy {
  /** Policy name */
  policyName: string;
  /** Traffic class */
  trafficClass: string;
  /** Rate limit in bps */
  rateLimit: number;
  /** Burst size in bytes */
  burstSize: number;
}

/**
 * Node type enumeration
 */
export enum NodeType {
  ENODEB = 'eNodeB',
  GNODEB = 'gNodeB',
  EN_GNODEB = 'en-gNB'
}

/**
 * Cell type enumeration
 */
export enum CellType {
  EUTRAN_CELL = 'EUtranCell',
  NR_CELL = 'NRCell'
}

/**
 * Factory dependencies interface
 */
export interface FactoryDependencies {
  /** Configuration manager instance */
  configManager: any; // Will be properly typed when IConfigurationManager is defined
  /** Monitoring service instance */
  monitoringService: any; // Will be properly typed when IMonitoringService is defined
  /** Logger instance */
  logger: any; // Will be properly typed when Logger interface is defined
}

/**
 * Factory for creating RAN nodes and cells
 * 
 * Implements the Abstract Factory pattern to provide a consistent interface
 * for creating different types of RAN infrastructure components.
 */
export interface IRANNodeFactory {
  /**
   * Create a 4G (LTE) RAN node
   * 
   * @param config Node configuration
   * @returns Promise resolving to the created 4G node
   * @throws {ConfigurationException} When configuration is invalid
   * @throws {NodeCreationException} When node creation fails
   */
  create4GNode(config: NodeConfig): Promise<IRANNode>;

  /**
   * Create a 5G (NR) RAN node
   * 
   * @param config Node configuration
   * @returns Promise resolving to the created 5G node
   * @throws {ConfigurationException} When configuration is invalid
   * @throws {NodeCreationException} When node creation fails
   */
  create5GNode(config: NodeConfig): Promise<IRANNode>;

  /**
   * Create a dual-mode LTE/NR node
   * 
   * @param config Node configuration
   * @returns Promise resolving to the created dual-mode node
   * @throws {ConfigurationException} When configuration is invalid
   * @throws {NodeCreationException} When node creation fails
   */
  createDualModeNode(config: NodeConfig): Promise<IRANNode>;

  /**
   * Create a cell of the specified type
   * 
   * @param nodeType Type of the parent node
   * @param config Cell configuration
   * @returns Promise resolving to the created cell
   * @throws {ConfigurationException} When configuration is invalid
   * @throws {CellCreationException} When cell creation fails
   */
  createCell(nodeType: NodeType, config: CellConfig): Promise<IRANCell>;

  /**
   * Create multiple cells in batch
   * 
   * @param nodeType Type of the parent node
   * @param configs Array of cell configurations
   * @returns Promise resolving to array of created cells
   * @throws {ConfigurationException} When any configuration is invalid
   * @throws {CellCreationException} When any cell creation fails
   */
  createCells(nodeType: NodeType, configs: CellConfig[]): Promise<IRANCell[]>;

  /**
   * Validate node configuration
   * 
   * @param config Node configuration to validate
   * @returns Validation result
   */
  validateNodeConfig(config: NodeConfig): Promise<ValidationResult>;

  /**
   * Validate cell configuration
   * 
   * @param config Cell configuration to validate
   * @returns Validation result
   */
  validateCellConfig(config: CellConfig): Promise<ValidationResult>;

  /**
   * Get supported node types
   * 
   * @returns Array of supported node types
   */
  getSupportedNodeTypes(): NodeType[];

  /**
   * Get supported cell types for a given node type
   * 
   * @param nodeType Node type
   * @returns Array of supported cell types
   */
  getSupportedCellTypes(nodeType: NodeType): CellType[];

  /**
   * Get default configuration template for node type
   * 
   * @param nodeType Node type
   * @returns Default configuration template
   */
  getDefaultNodeConfig(nodeType: NodeType): Partial<NodeConfig>;

  /**
   * Get default configuration template for cell type
   * 
   * @param cellType Cell type
   * @returns Default configuration template
   */
  getDefaultCellConfig(cellType: CellType): Partial<CellConfig>;
}

/**
 * Validation result interface
 */
export interface ValidationResult {
  /** Whether the configuration is valid */
  isValid: boolean;
  /** Validation errors, if any */
  errors: ValidationError[];
  /** Validation warnings */
  warnings: ValidationWarning[];
}

/**
 * Validation error
 */
export interface ValidationError {
  /** Error code */
  code: string;
  /** Error message */
  message: string;
  /** Field path that caused the error */
  field: string;
  /** Current value that caused the error */
  value: any;
}

/**
 * Validation warning
 */
export interface ValidationWarning {
  /** Warning code */
  code: string;
  /** Warning message */
  message: string;
  /** Field path that caused the warning */
  field: string;
  /** Current value that caused the warning */
  value: any;
}