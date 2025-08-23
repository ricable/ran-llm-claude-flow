import { RANNode } from '../core/RANNode';
import { NodeConfiguration, ParameterType, CMEditCommand } from '../types';
import { ConfigurationManager } from '../core/ConfigurationManager';
import { CMEditClient } from '../core/CMEditClient';

/**
 * 5G gNodeB implementation extending RANNode base class
 * Provides gNodeB-specific functionality and advanced NR parameter management
 */
export class GNodeB extends RANNode {

  constructor(
    config: NodeConfiguration,
    configManager?: ConfigurationManager,
    cmEditClient?: CMEditClient
  ) {
    super(config, configManager, cmEditClient);
    this.logger.info('gNodeB instance created', { nodeId: config.nodeId });
  }

  /**
   * Build managed object path for gNodeB
   */
  protected buildManagedObjectPath(mo: string): string {
    const baseId = this.config.nodeId;
    
    if (mo.startsWith('ManagedElement')) {
      return mo.replace('ManagedElement', `ManagedElement=${baseId}`);
    }
    
    return `ManagedElement=${baseId},${mo}`;
  }

  /**
   * Get parameter type for gNodeB parameters
   */
  protected getParameterType(parameterName: string): ParameterType {
    const parameterTypeMap: Record<string, ParameterType> = {
      // RF parameters
      'nrarfcnDL': ParameterType.RF,
      'nrarfcnUL': ParameterType.RF,
      'channelBandwidth': ParameterType.RF,
      'ssbSubCarrierSpacing': ParameterType.RF,
      'ssbReferenceSignalPower': ParameterType.RF,
      'frequencyBand': ParameterType.RF,
      
      // Power parameters
      'maxTransmissionPower': ParameterType.POWER,
      'pdschPowerAllocation': ParameterType.POWER,
      'puschPowerControl': ParameterType.POWER,
      'ssbPower': ParameterType.POWER,
      
      // Antenna parameters
      'antennaAzimuth': ParameterType.ANTENNA,
      'antennaTilt': ParameterType.ANTENNA,
      'mechanicalTilt': ParameterType.ANTENNA,
      'antennaGain': ParameterType.ANTENNA,
      'antennaElements': ParameterType.ANTENNA,
      'beamCount': ParameterType.ANTENNA,
      'beamWidth': ParameterType.ANTENNA,
      
      // Mobility parameters
      'hysteresisA3': ParameterType.MOBILITY,
      'thresholdA3': ParameterType.MOBILITY,
      'timeToTriggerA3': ParameterType.MOBILITY,
      'qQualMin': ParameterType.MOBILITY,
      'qRxLevMin': ParameterType.MOBILITY,
      
      // Handover parameters
      'a3Offset': ParameterType.HANDOVER,
      'reportInterval': ParameterType.HANDOVER,
      'reportAmount': ParameterType.HANDOVER,
      'handoverOptimization': ParameterType.HANDOVER,
      
      // Carrier parameters
      'tac': ParameterType.CARRIER,
      'cellBarred': ParameterType.CARRIER,
      'intraFreqReselection': ParameterType.CARRIER,
      'physicalCellId': ParameterType.CARRIER
    };
    
    return parameterTypeMap[parameterName] || ParameterType.RF;
  }

  /**
   * Get managed object for specific parameter
   */
  protected getParameterManagedObject(parameterName: string): string {
    const parameterMOMap: Record<string, string> = {
      // NRCellDU parameters
      'nrarfcnDL': 'GNBDUFunction=1,NRCellDU=1',
      'nrarfcnUL': 'GNBDUFunction=1,NRCellDU=1',
      'channelBandwidth': 'GNBDUFunction=1,NRCellDU=1',
      'physicalCellId': 'GNBDUFunction=1,NRCellDU=1',
      'tac': 'GNBDUFunction=1,NRCellDU=1',
      'cellBarred': 'GNBDUFunction=1,NRCellDU=1',
      
      // SSB parameters
      'ssbSubCarrierSpacing': 'GNBDUFunction=1,NRCellDU=1,NRSectorCarrier=1',
      'ssbReferenceSignalPower': 'GNBDUFunction=1,NRCellDU=1,NRSectorCarrier=1',
      'ssbFrequency': 'GNBDUFunction=1,NRCellDU=1,NRSectorCarrier=1',
      
      // Antenna parameters
      'antennaAzimuth': 'Equipment=1,AntennaUnitGroup=1,AntennaUnit=1,AntennaSubunit=1',
      'antennaTilt': 'Equipment=1,AntennaUnitGroup=1,AntennaUnit=1,AntennaSubunit=1',
      'mechanicalTilt': 'Equipment=1,AntennaUnitGroup=1,AntennaUnit=1,AntennaSubunit=1',
      
      // Power parameters
      'maxTransmissionPower': 'Equipment=1,FieldReplaceableUnit=Radio-1',
      'pdschPowerAllocation': 'GNBDUFunction=1,NRCellDU=1',
      'puschPowerControl': 'GNBDUFunction=1,NRCellDU=1',
      
      // Mobility parameters
      'hysteresisA3': 'GNBDUFunction=1,NRCellDU=1,NRFreqRelation=1,NRCellRelation=1',
      'thresholdA3': 'GNBDUFunction=1,NRCellDU=1,NRFreqRelation=1,NRCellRelation=1',
      'timeToTriggerA3': 'GNBDUFunction=1,NRCellDU=1,NRFreqRelation=1,NRCellRelation=1'
    };
    
    return parameterMOMap[parameterName] || 'GNBDUFunction=1,NRCellDU=1';
  }

  /**
   * Set gNodeB ID
   */
  public async setGNodeBId(gNBId: number): Promise<void> {
    if (gNBId < 0 || gNBId > 4294967295) {
      throw new Error('gNBId must be between 0 and 4294967295');
    }
    
    await this.setParameter('gNBId', gNBId);
    this.logger.info('gNBId set successfully', { gNBId });
  }

  /**
   * Set Tracking Area Code
   */
  public async setTAC(tac: number): Promise<void> {
    if (tac < 1 || tac > 16777215) { // 24-bit for NR
      throw new Error('TAC must be between 1 and 16777215');
    }
    
    await this.setParameter('tac', tac);
    this.logger.info('TAC set successfully', { tac });
  }

  /**
   * Set NR-ARFCN for downlink
   */
  public async setNRARFCNDL(nrarfcn: number): Promise<void> {
    if (nrarfcn < 0 || nrarfcn > 3279165) {
      throw new Error('NR-ARFCN DL must be between 0 and 3279165');
    }
    
    await this.setParameter('nrarfcnDL', nrarfcn);
    this.logger.info('NR-ARFCN DL set successfully', { nrarfcn });
  }

  /**
   * Set NR-ARFCN for uplink
   */
  public async setNRARFCNUL(nrarfcn: number): Promise<void> {
    if (nrarfcn < 0 || nrarfcn > 3279165) {
      throw new Error('NR-ARFCN UL must be between 0 and 3279165');
    }
    
    await this.setParameter('nrarfcnUL', nrarfcn);
    this.logger.info('NR-ARFCN UL set successfully', { nrarfcn });
  }

  /**
   * Set channel bandwidth
   */
  public async setChannelBandwidth(bandwidth: number): Promise<void> {
    const validBandwidths = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400]; // MHz
    if (!validBandwidths.includes(bandwidth)) {
      throw new Error(`Invalid channel bandwidth. Must be one of: ${validBandwidths.join(', ')}`);
    }
    
    await this.setParameter('channelBandwidth', bandwidth);
    this.logger.info('Channel bandwidth set successfully', { bandwidth });
  }

  /**
   * Set SSB subcarrier spacing
   */
  public async setSSBSubCarrierSpacing(scs: number): Promise<void> {
    const validSCS = [15, 30, 120, 240]; // kHz
    if (!validSCS.includes(scs)) {
      throw new Error(`Invalid SSB subcarrier spacing. Must be one of: ${validSCS.join(', ')} kHz`);
    }
    
    await this.setParameter('ssbSubCarrierSpacing', scs);
    this.logger.info('SSB subcarrier spacing set successfully', { scs });
  }

  /**
   * Set Physical Cell ID (expanded range for NR)
   */
  public async setPhysicalCellId(pci: number): Promise<void> {
    if (pci < 0 || pci > 1007) {
      throw new Error('Physical Cell ID must be between 0 and 1007 for NR');
    }
    
    await this.setParameter('physicalCellId', pci);
    this.logger.info('Physical Cell ID set successfully', { pci });
  }

  /**
   * Configure SSB parameters
   */
  public async configureSSBParameters(params: {
    ssbFrequency?: number;
    ssbPeriodicity?: number;
    ssbPattern?: string;
    maxSSBperBurst?: number;
    ssbReferenceSignalPower?: number;
  }): Promise<void> {
    this.logger.info('Configuring SSB parameters', { params });

    const updates: Array<Promise<void>> = [];

    if (params.ssbFrequency !== undefined) {
      if (params.ssbFrequency < 410000 || params.ssbFrequency > 7125000) {
        throw new Error('SSB frequency must be between 410000 and 7125000 kHz');
      }
      updates.push(this.setParameter('ssbFrequency', params.ssbFrequency));
    }

    if (params.ssbPeriodicity !== undefined) {
      const validPeriodicities = [5, 10, 20, 40, 80, 160];
      if (!validPeriodicities.includes(params.ssbPeriodicity)) {
        throw new Error(`SSB periodicity must be one of: ${validPeriodicities.join(', ')} ms`);
      }
      updates.push(this.setParameter('ssbPeriodicity', params.ssbPeriodicity));
    }

    if (params.ssbPattern !== undefined) {
      const validPatterns = ['Case A', 'Case B', 'Case C', 'Case D', 'Case E'];
      if (!validPatterns.includes(params.ssbPattern)) {
        throw new Error(`SSB pattern must be one of: ${validPatterns.join(', ')}`);
      }
      updates.push(this.setParameter('ssbPattern', params.ssbPattern));
    }

    if (params.maxSSBperBurst !== undefined) {
      if (params.maxSSBperBurst < 1 || params.maxSSBperBurst > 64) {
        throw new Error('Max SSB per burst must be between 1 and 64');
      }
      updates.push(this.setParameter('maxSSBperBurst', params.maxSSBperBurst));
    }

    if (params.ssbReferenceSignalPower !== undefined) {
      if (params.ssbReferenceSignalPower < -60 || params.ssbReferenceSignalPower > 50) {
        throw new Error('SSB reference signal power must be between -60 and 50 dBm');
      }
      updates.push(this.setParameter('ssbReferenceSignalPower', params.ssbReferenceSignalPower));
    }

    await Promise.all(updates);
    this.logger.info('SSB parameters configured successfully');
  }

  /**
   * Configure beam management parameters
   */
  public async configureBeamParameters(params: {
    beamCount?: number;
    beamWidth?: number;
    beamSweeping?: boolean;
    beamTracking?: boolean;
    csiReporting?: string;
  }): Promise<void> {
    this.logger.info('Configuring beam parameters', { params });

    const updates: Array<Promise<void>> = [];

    if (params.beamCount !== undefined) {
      if (params.beamCount < 1 || params.beamCount > 256) {
        throw new Error('Beam count must be between 1 and 256');
      }
      updates.push(this.setParameter('beamCount', params.beamCount));
    }

    if (params.beamWidth !== undefined) {
      if (params.beamWidth < 5 || params.beamWidth > 360) {
        throw new Error('Beam width must be between 5 and 360 degrees');
      }
      updates.push(this.setParameter('beamWidth', params.beamWidth));
    }

    if (params.beamSweeping !== undefined) {
      updates.push(this.setParameter('beamSweeping', params.beamSweeping));
    }

    if (params.beamTracking !== undefined) {
      updates.push(this.setParameter('beamTracking', params.beamTracking));
    }

    if (params.csiReporting !== undefined) {
      const validReporting = ['periodic', 'aperiodic', 'semi-persistent'];
      if (!validReporting.includes(params.csiReporting)) {
        throw new Error(`CSI reporting must be one of: ${validReporting.join(', ')}`);
      }
      updates.push(this.setParameter('csiReporting', params.csiReporting));
    }

    await Promise.all(updates);
    this.logger.info('Beam parameters configured successfully');
  }

  /**
   * Configure network slicing
   */
  public async configureNetworkSlicing(slices: Array<{
    sst: number;
    sd?: string;
    qci?: number[];
    priority?: number;
  }>): Promise<void> {
    this.logger.info('Configuring network slicing', { slices });

    // Validate slice configurations
    for (const slice of slices) {
      if (slice.sst < 1 || slice.sst > 255) {
        throw new Error('SST must be between 1 and 255');
      }
      
      if (slice.sd && !/^[0-9A-Fa-f]{6}$/.test(slice.sd)) {
        throw new Error('SD must be a 6-digit hexadecimal string');
      }
    }

    await this.setParameter('networkSlicing', true);
    await this.setParameter('supportedSlices', slices);
    
    this.logger.info('Network slicing configured successfully', { sliceCount: slices.length });
  }

  /**
   * Configure dual connectivity (NSA mode)
   */
  public async configureDualConnectivity(params: {
    enable: boolean;
    anchorENodeBId?: string;
    splitBearer?: boolean;
    flowControl?: string;
  }): Promise<void> {
    this.logger.info('Configuring dual connectivity', { params });

    await this.setParameter('dualConnectivity', params.enable);
    
    if (params.enable) {
      if (params.anchorENodeBId) {
        await this.setParameter('anchorENodeBId', params.anchorENodeBId);
      }
      
      if (params.splitBearer !== undefined) {
        await this.setParameter('splitBearer', params.splitBearer);
      }
      
      if (params.flowControl) {
        const validFlowControl = ['RLC', 'PDCP', 'RRC'];
        if (!validFlowControl.includes(params.flowControl)) {
          throw new Error(`Flow control must be one of: ${validFlowControl.join(', ')}`);
        }
        await this.setParameter('flowControl', params.flowControl);
      }

      await this.setParameter('enDcCapable', true);
    } else {
      await this.setParameter('enDcCapable', false);
    }

    this.logger.info('Dual connectivity configured successfully');
  }

  /**
   * Configure bandwidth parts
   */
  public async configureBandwidthParts(bwParts: Array<{
    id: number;
    startRB: number;
    numRB: number;
    numerology: number;
    cyclicPrefix?: string;
  }>): Promise<void> {
    this.logger.info('Configuring bandwidth parts', { bwParts });

    // Validate bandwidth parts
    for (const bwp of bwParts) {
      if (bwp.id < 0 || bwp.id > 3) {
        throw new Error('Bandwidth part ID must be between 0 and 3');
      }
      
      if (bwp.startRB < 0 || bwp.numRB < 1) {
        throw new Error('Invalid resource block configuration');
      }
      
      if (bwp.numerology < 0 || bwp.numerology > 4) {
        throw new Error('Numerology must be between 0 and 4');
      }
      
      if (bwp.cyclicPrefix && !['normal', 'extended'].includes(bwp.cyclicPrefix)) {
        throw new Error('Cyclic prefix must be "normal" or "extended"');
      }
    }

    await this.setParameter('bandwidthParts', bwParts);
    this.logger.info('Bandwidth parts configured successfully', { count: bwParts.length });
  }

  /**
   * Get NR cell status and advanced KPIs
   */
  public async getCellStatus(): Promise<any> {
    this.logger.info('Getting NR cell status');

    try {
      const response = await this.executeCMEditCommand({
        operation: 'get',
        mo: 'GNBDUFunction=1,NRCellDU=1',
        attributes: {
          select: [
            'administrativeState',
            'operationalState',
            'availabilityStatus',
            'cellBarred',
            'tac',
            'nrarfcnDL',
            'nrarfcnUL',
            'physicalCellId',
            'channelBandwidth',
            'ssbSubCarrierSpacing'
          ]
        }
      });

      if (!response.success) {
        throw new Error(`Failed to get NR cell status: ${response.error}`);
      }

      this.logger.info('NR cell status retrieved successfully');
      return response.data;

    } catch (error) {
      this.logger.error('Failed to get NR cell status', { error });
      throw error;
    }
  }

  /**
   * Get beam status and metrics
   */
  public async getBeamStatus(): Promise<any> {
    this.logger.info('Getting beam status');

    try {
      const response = await this.executeCMEditCommand({
        operation: 'get',
        mo: 'GNBDUFunction=1,NRCellDU=1,BeamManagement=*'
      });

      if (!response.success) {
        throw new Error(`Failed to get beam status: ${response.error}`);
      }

      this.logger.info('Beam status retrieved successfully');
      return response.data;

    } catch (error) {
      this.logger.error('Failed to get beam status', { error });
      throw error;
    }
  }

  /**
   * Add 5G neighbor cell relation
   */
  public async add5GNeighborCell(
    neighborCellId: string, 
    nrarfcn: number,
    nci: string
  ): Promise<void> {
    this.logger.info('Adding 5G neighbor cell relation', { neighborCellId, nrarfcn, nci });

    try {
      // Create frequency relation
      await this.executeCMEditCommand({
        operation: 'create',
        mo: `GNBDUFunction=1,NRCellDU=1,NRFreqRelation=${nrarfcn}`,
        attributes: {
          arfcnValueNRDl: nrarfcn
        }
      });

      // Create cell relation
      await this.executeCMEditCommand({
        operation: 'create',
        mo: `GNBDUFunction=1,NRCellDU=1,NRFreqRelation=${nrarfcn},NRCellRelation=${neighborCellId}`,
        attributes: {
          neighborCellId: neighborCellId,
          nRCellIdentifier: nci,
          isHoAllowed: true
        }
      });

      this.logger.info('5G neighbor cell relation added successfully', { neighborCellId });

    } catch (error) {
      this.logger.error('Failed to add 5G neighbor cell relation', { neighborCellId, error });
      throw error;
    }
  }

  /**
   * Enable/disable ultra-low latency mode
   */
  public async setUltraLowLatencyMode(enabled: boolean): Promise<void> {
    this.logger.info('Setting ultra-low latency mode', { enabled });

    const updates: Array<Promise<void>> = [];

    updates.push(this.setParameter('ultraLowLatency', enabled));
    
    if (enabled) {
      // Configure for URLLC
      updates.push(this.setParameter('numerology', 3)); // Higher numerology for lower latency
      updates.push(this.setParameter('schedulingAlgorithm', 'earliest_deadline_first'));
      updates.push(this.setParameter('harqProcesses', 16)); // More HARQ processes
      updates.push(this.setParameter('minislotAggregation', true));
    }

    await Promise.all(updates);
    this.logger.info('Ultra-low latency mode configured successfully', { enabled });
  }

  /**
   * gNodeB-specific validation
   */
  protected async validateNodeSpecificConfiguration(config: NodeConfiguration): Promise<void> {
    const params = config.parameters;

    // Validate gNodeB ID
    if (params.gNBId !== undefined) {
      const gNBId = Number(params.gNBId);
      if (isNaN(gNBId) || gNBId < 0 || gNBId > 4294967295) {
        throw new Error('gNBId must be between 0 and 4294967295');
      }
    }

    // Validate NR-ARFCN
    if (params.nrarfcnDL !== undefined) {
      const nrarfcn = Number(params.nrarfcnDL);
      if (isNaN(nrarfcn) || nrarfcn < 0 || nrarfcn > 3279165) {
        throw new Error('NR-ARFCN must be between 0 and 3279165');
      }
    }

    // Validate SSB subcarrier spacing
    if (params.ssbSubCarrierSpacing !== undefined) {
      const validSCS = [15, 30, 120, 240];
      if (!validSCS.includes(params.ssbSubCarrierSpacing)) {
        throw new Error(`Invalid SSB subcarrier spacing. Must be one of: ${validSCS.join(', ')} kHz`);
      }
    }

    // Validate frequency band
    if (params.frequencyBand !== undefined) {
      const band = Number(params.frequencyBand);
      if (isNaN(band) || band < 1 || band > 1024) {
        throw new Error('Frequency band must be between 1 and 1024');
      }
    }

    this.logger.debug('gNodeB configuration validation passed');
  }

  /**
   * Get gNodeB type
   */
  public getGNodeBType(): string {
    return '5G gNodeB';
  }

  /**
   * Get supported features
   */
  public getSupportedFeatures(): string[] {
    return [
      '5G NR',
      'Massive MIMO',
      'Network Slicing',
      'Ultra-Low Latency',
      'Enhanced Mobile Broadband',
      'Machine Type Communications',
      'Beam Management',
      'Carrier Aggregation',
      'Dual Connectivity (EN-DC)',
      'Dynamic Spectrum Sharing',
      'Edge Computing Integration'
    ];
  }

  /**
   * Get deployment mode
   */
  public getDeploymentMode(): string {
    return this.config.parameters.deploymentMode || 'sa';
  }

  /**
   * Check if network slicing is enabled
   */
  public isNetworkSlicingEnabled(): boolean {
    return this.config.parameters.networkSlicing || false;
  }

  /**
   * Check if dual connectivity is enabled
   */
  public isDualConnectivityEnabled(): boolean {
    return this.config.parameters.dualConnectivity || false;
  }
}