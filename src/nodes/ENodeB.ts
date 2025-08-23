import { RANNode } from '../core/RANNode';
import { NodeConfiguration, ParameterType, CMEditCommand } from '../types';
import { ConfigurationManager } from '../core/ConfigurationManager';
import { CMEditClient } from '../core/CMEditClient';

/**
 * LTE eNodeB implementation extending RANNode base class
 * Provides eNodeB-specific functionality and parameter management
 */
export class ENodeB extends RANNode {

  constructor(
    config: NodeConfiguration,
    configManager?: ConfigurationManager,
    cmEditClient?: CMEditClient
  ) {
    super(config, configManager, cmEditClient);
    this.logger.info('eNodeB instance created', { nodeId: config.nodeId });
  }

  /**
   * Build managed object path for eNodeB
   */
  protected buildManagedObjectPath(mo: string): string {
    const baseId = this.config.nodeId;
    
    if (mo.startsWith('ManagedElement')) {
      return mo.replace('ManagedElement', `ManagedElement=${baseId}`);
    }
    
    return `ManagedElement=${baseId},${mo}`;
  }

  /**
   * Get parameter type for eNodeB parameters
   */
  protected getParameterType(parameterName: string): ParameterType {
    const parameterTypeMap: Record<string, ParameterType> = {
      // RF parameters
      'earfcnDL': ParameterType.RF,
      'earfcnUL': ParameterType.RF,
      'dlChannelBandwidth': ParameterType.RF,
      'ulChannelBandwidth': ParameterType.RF,
      'referenceSignalPower': ParameterType.RF,
      
      // Power parameters
      'maxTransmissionPower': ParameterType.POWER,
      'pB': ParameterType.POWER,
      'pA': ParameterType.POWER,
      
      // Antenna parameters
      'antennaAzimuth': ParameterType.ANTENNA,
      'antennaTilt': ParameterType.ANTENNA,
      'mechanicalTilt': ParameterType.ANTENNA,
      'antennaGain': ParameterType.ANTENNA,
      
      // Mobility parameters
      'hysteresisA1': ParameterType.MOBILITY,
      'hysteresisA2': ParameterType.MOBILITY,
      'hysteresisA3': ParameterType.MOBILITY,
      'thresholdA1RSRP': ParameterType.MOBILITY,
      'thresholdA2RSRP': ParameterType.MOBILITY,
      'timeToTriggerA1': ParameterType.MOBILITY,
      'timeToTriggerA2': ParameterType.MOBILITY,
      'timeToTriggerA3': ParameterType.MOBILITY,
      
      // Handover parameters
      'a3Offset': ParameterType.HANDOVER,
      'reportInterval': ParameterType.HANDOVER,
      'reportAmount': ParameterType.HANDOVER,
      
      // Carrier parameters
      'tac': ParameterType.CARRIER,
      'cellBarred': ParameterType.CARRIER,
      'intraFreqReselection': ParameterType.CARRIER
    };
    
    return parameterTypeMap[parameterName] || ParameterType.RF;
  }

  /**
   * Get managed object for specific parameter
   */
  protected getParameterManagedObject(parameterName: string): string {
    const parameterMOMap: Record<string, string> = {
      // EUtranCellFDD parameters
      'earfcnDL': 'ENodeBFunction=1,EUtranCellFDD=1',
      'earfcnUL': 'ENodeBFunction=1,EUtranCellFDD=1',
      'dlChannelBandwidth': 'ENodeBFunction=1,EUtranCellFDD=1',
      'ulChannelBandwidth': 'ENodeBFunction=1,EUtranCellFDD=1',
      'tac': 'ENodeBFunction=1,EUtranCellFDD=1',
      'cellBarred': 'ENodeBFunction=1,EUtranCellFDD=1',
      'intraFreqReselection': 'ENodeBFunction=1,EUtranCellFDD=1',
      
      // Antenna parameters
      'antennaAzimuth': 'Equipment=1,AntennaUnitGroup=1,AntennaUnit=1,AntennaSubunit=1',
      'antennaTilt': 'Equipment=1,AntennaUnitGroup=1,AntennaUnit=1,AntennaSubunit=1',
      'mechanicalTilt': 'Equipment=1,AntennaUnitGroup=1,AntennaUnit=1,AntennaSubunit=1',
      
      // Power parameters
      'maxTransmissionPower': 'Equipment=1,FieldReplaceableUnit=Radio-1',
      'referenceSignalPower': 'ENodeBFunction=1,EUtranCellFDD=1',
      
      // Mobility parameters
      'hysteresisA1': 'ENodeBFunction=1,EUtranCellFDD=1,EUtranFreqRelation=1,EUtranCellRelation=1',
      'hysteresisA2': 'ENodeBFunction=1,EUtranCellFDD=1,EUtranFreqRelation=1,EUtranCellRelation=1',
      'hysteresisA3': 'ENodeBFunction=1,EUtranCellFDD=1,EUtranFreqRelation=1,EUtranCellRelation=1'
    };
    
    return parameterMOMap[parameterName] || 'ENodeBFunction=1,EUtranCellFDD=1';
  }

  /**
   * Set eNodeB ID
   */
  public async setENodeBId(eNodeBId: number): Promise<void> {
    if (eNodeBId < 0 || eNodeBId > 1048575) {
      throw new Error('eNodeBId must be between 0 and 1048575');
    }
    
    await this.setParameter('eNodeBId', eNodeBId);
    this.logger.info('eNodeBId set successfully', { eNodeBId });
  }

  /**
   * Set Tracking Area Code
   */
  public async setTAC(tac: number): Promise<void> {
    if (tac < 1 || tac > 65535) {
      throw new Error('TAC must be between 1 and 65535');
    }
    
    await this.setParameter('tac', tac);
    this.logger.info('TAC set successfully', { tac });
  }

  /**
   * Set EARFCN for downlink
   */
  public async setEARFCNDL(earfcn: number): Promise<void> {
    if (earfcn < 0 || earfcn > 262143) {
      throw new Error('EARFCN DL must be between 0 and 262143');
    }
    
    await this.setParameter('earfcnDL', earfcn);
    this.logger.info('EARFCN DL set successfully', { earfcn });
  }

  /**
   * Set EARFCN for uplink
   */
  public async setEARFCNUL(earfcn: number): Promise<void> {
    if (earfcn < 0 || earfcn > 262143) {
      throw new Error('EARFCN UL must be between 0 and 262143');
    }
    
    await this.setParameter('earfcnUL', earfcn);
    this.logger.info('EARFCN UL set successfully', { earfcn });
  }

  /**
   * Set channel bandwidth for downlink
   */
  public async setDLChannelBandwidth(bandwidth: number): Promise<void> {
    const validBandwidths = [1400, 3000, 5000, 10000, 15000, 20000]; // kHz
    if (!validBandwidths.includes(bandwidth)) {
      throw new Error(`Invalid DL bandwidth. Must be one of: ${validBandwidths.join(', ')}`);
    }
    
    await this.setParameter('dlChannelBandwidth', bandwidth);
    this.logger.info('DL channel bandwidth set successfully', { bandwidth });
  }

  /**
   * Set reference signal power
   */
  public async setReferenceSignalPower(power: number): Promise<void> {
    if (power < -60 || power > 50) {
      throw new Error('Reference signal power must be between -60 and 50 dBm');
    }
    
    await this.setParameter('referenceSignalPower', power);
    this.logger.info('Reference signal power set successfully', { power });
  }

  /**
   * Configure handover parameters
   */
  public async configureHandoverParameters(params: {
    hysteresisA3?: number;
    thresholdA3?: number;
    timeToTriggerA3?: number;
    a3Offset?: number;
  }): Promise<void> {
    this.logger.info('Configuring handover parameters', { params });

    const updates: Array<Promise<void>> = [];

    if (params.hysteresisA3 !== undefined) {
      if (params.hysteresisA3 < 0 || params.hysteresisA3 > 150) {
        throw new Error('hysteresisA3 must be between 0 and 150 (0.0-15.0 dB)');
      }
      updates.push(this.setParameter('hysteresisA3', params.hysteresisA3));
    }

    if (params.thresholdA3 !== undefined) {
      if (params.thresholdA3 < -30 || params.thresholdA3 > 30) {
        throw new Error('thresholdA3 must be between -30 and 30 dB');
      }
      updates.push(this.setParameter('thresholdA3Offset', params.thresholdA3));
    }

    if (params.timeToTriggerA3 !== undefined) {
      const validTTT = [0, 40, 64, 80, 100, 128, 160, 256, 320, 480, 512, 640, 1024, 1280, 2560, 5120];
      if (!validTTT.includes(params.timeToTriggerA3)) {
        throw new Error(`timeToTriggerA3 must be one of: ${validTTT.join(', ')} ms`);
      }
      updates.push(this.setParameter('timeToTriggerA3', params.timeToTriggerA3));
    }

    if (params.a3Offset !== undefined) {
      if (params.a3Offset < -30 || params.a3Offset > 30) {
        throw new Error('a3Offset must be between -30 and 30 dB');
      }
      updates.push(this.setParameter('a3Offset', params.a3Offset));
    }

    await Promise.all(updates);
    this.logger.info('Handover parameters configured successfully');
  }

  /**
   * Configure antenna parameters
   */
  public async configureAntennaParameters(params: {
    azimuth?: number;
    tilt?: number;
    mechanicalTilt?: number;
    gain?: number;
  }): Promise<void> {
    this.logger.info('Configuring antenna parameters', { params });

    const updates: Array<Promise<void>> = [];

    if (params.azimuth !== undefined) {
      if (params.azimuth < 0 || params.azimuth >= 360) {
        throw new Error('Azimuth must be between 0 and 359 degrees');
      }
      updates.push(this.setParameter('antennaAzimuth', params.azimuth));
    }

    if (params.tilt !== undefined) {
      if (Math.abs(params.tilt) > 90) {
        throw new Error('Antenna tilt must be between -90 and 90 degrees');
      }
      updates.push(this.setParameter('antennaTilt', params.tilt));
    }

    if (params.mechanicalTilt !== undefined) {
      if (Math.abs(params.mechanicalTilt) > 45) {
        throw new Error('Mechanical tilt must be between -45 and 45 degrees');
      }
      updates.push(this.setParameter('mechanicalTilt', params.mechanicalTilt));
    }

    if (params.gain !== undefined) {
      if (params.gain < 0 || params.gain > 30) {
        throw new Error('Antenna gain must be between 0 and 30 dBi');
      }
      updates.push(this.setParameter('antennaGain', params.gain));
    }

    await Promise.all(updates);
    this.logger.info('Antenna parameters configured successfully');
  }

  /**
   * Enable/disable cell barring
   */
  public async setCellBarring(barred: boolean): Promise<void> {
    const value = barred ? 'barred' : 'notBarred';
    await this.setParameter('cellBarred', value);
    this.logger.info('Cell barring configured', { barred });
  }

  /**
   * Get cell status and KPIs
   */
  public async getCellStatus(): Promise<any> {
    this.logger.info('Getting cell status');

    try {
      const response = await this.executeCMEditCommand({
        operation: 'get',
        mo: 'ENodeBFunction=1,EUtranCellFDD=1',
        attributes: {
          select: [
            'administrativeState',
            'operationalState',
            'availabilityStatus',
            'cellBarred',
            'tac',
            'earfcnDL',
            'earfcnUL'
          ]
        }
      });

      if (!response.success) {
        throw new Error(`Failed to get cell status: ${response.error}`);
      }

      this.logger.info('Cell status retrieved successfully');
      return response.data;

    } catch (error) {
      this.logger.error('Failed to get cell status', { error });
      throw error;
    }
  }

  /**
   * Perform cell unlock
   */
  public async unlockCell(): Promise<void> {
    this.logger.info('Unlocking cell');

    try {
      await this.executeCMEditCommand({
        operation: 'update',
        mo: 'ENodeBFunction=1,EUtranCellFDD=1',
        attributes: {
          administrativeState: 'UNLOCKED'
        }
      });

      this.logger.info('Cell unlocked successfully');

    } catch (error) {
      this.logger.error('Failed to unlock cell', { error });
      throw error;
    }
  }

  /**
   * Perform cell lock
   */
  public async lockCell(): Promise<void> {
    this.logger.info('Locking cell');

    try {
      await this.executeCMEditCommand({
        operation: 'update',
        mo: 'ENodeBFunction=1,EUtranCellFDD=1',
        attributes: {
          administrativeState: 'LOCKED'
        }
      });

      this.logger.info('Cell locked successfully');

    } catch (error) {
      this.logger.error('Failed to lock cell', { error });
      throw error;
    }
  }

  /**
   * Get neighbor cell list
   */
  public async getNeighborCells(): Promise<any[]> {
    this.logger.info('Getting neighbor cell list');

    try {
      const response = await this.executeCMEditCommand({
        operation: 'get',
        mo: 'ENodeBFunction=1,EUtranCellFDD=1,EUtranFreqRelation=*,EUtranCellRelation=*'
      });

      if (!response.success) {
        throw new Error(`Failed to get neighbor cells: ${response.error}`);
      }

      this.logger.info('Neighbor cell list retrieved successfully');
      return response.data;

    } catch (error) {
      this.logger.error('Failed to get neighbor cells', { error });
      throw error;
    }
  }

  /**
   * Add neighbor cell relation
   */
  public async addNeighborCell(neighborCellId: string, earfcn: number): Promise<void> {
    this.logger.info('Adding neighbor cell relation', { neighborCellId, earfcn });

    try {
      // First create frequency relation if it doesn't exist
      await this.executeCMEditCommand({
        operation: 'create',
        mo: `ENodeBFunction=1,EUtranCellFDD=1,EUtranFreqRelation=${earfcn}`,
        attributes: {
          arfcnValueEUtranDl: earfcn
        }
      });

      // Then create cell relation
      await this.executeCMEditCommand({
        operation: 'create',
        mo: `ENodeBFunction=1,EUtranCellFDD=1,EUtranFreqRelation=${earfcn},EUtranCellRelation=${neighborCellId}`,
        attributes: {
          neighborCellId: neighborCellId,
          isHoAllowed: true
        }
      });

      this.logger.info('Neighbor cell relation added successfully', { neighborCellId });

    } catch (error) {
      this.logger.error('Failed to add neighbor cell relation', { neighborCellId, error });
      throw error;
    }
  }

  /**
   * eNodeB-specific validation
   */
  protected async validateNodeSpecificConfiguration(config: NodeConfiguration): Promise<void> {
    const params = config.parameters;

    // Validate eNodeB ID
    if (params.eNodeBId !== undefined) {
      const eNodeBId = Number(params.eNodeBId);
      if (isNaN(eNodeBId) || eNodeBId < 0 || eNodeBId > 1048575) {
        throw new Error('eNodeBId must be between 0 and 1048575');
      }
    }

    // Validate TAC
    if (params.tac !== undefined) {
      const tac = Number(params.tac);
      if (isNaN(tac) || tac < 1 || tac > 65535) {
        throw new Error('TAC must be between 1 and 65535');
      }
    }

    // Validate PLMN
    if (params.mcc && !/^[0-9]{3}$/.test(params.mcc)) {
      throw new Error('MCC must be a 3-digit number');
    }

    if (params.mnc && !/^[0-9]{2,3}$/.test(params.mnc)) {
      throw new Error('MNC must be a 2 or 3-digit number');
    }

    this.logger.debug('eNodeB configuration validation passed');
  }

  /**
   * Get eNodeB type
   */
  public getENodeBType(): string {
    return 'LTE eNodeB';
  }

  /**
   * Get supported features
   */
  public getSupportedFeatures(): string[] {
    return [
      'LTE FDD',
      'LTE TDD',
      'Carrier Aggregation',
      'MIMO 4x4',
      'SON (Self-Organizing Networks)',
      'eICIC (enhanced Inter-Cell Interference Coordination)',
      'VoLTE',
      'eMBMS (evolved Multimedia Broadcast Multicast Service)'
    ];
  }
}