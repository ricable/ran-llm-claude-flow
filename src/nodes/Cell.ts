import { CellConfiguration, CellType, CMEditCommand, CMEditResponse } from '../types';
import { ConfigurationManager } from '../core/ConfigurationManager';
import { ParameterValidator } from '../core/ParameterValidator';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';

/**
 * Generic Cell class supporting both LTE and NR cell types
 * Provides common cell functionality with type-specific implementations
 */
export class Cell {
  protected logger: Logger;
  protected configManager: ConfigurationManager;
  protected validator: ParameterValidator;

  constructor(
    protected config: CellConfiguration,
    configManager?: ConfigurationManager
  ) {
    this.logger = createLogger({
      level: 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json()
      ),
      defaultMeta: { 
        cellId: this.config.cellId, 
        cellType: this.config.cellType,
        service: 'Cell'
      },
      transports: [
        new transports.Console({
          format: format.combine(
            format.colorize(),
            format.simple()
          )
        })
      ]
    });

    this.configManager = configManager || new ConfigurationManager();
    this.validator = new ParameterValidator();
  }

  /**
   * Get cell configuration
   */
  public getConfiguration(): CellConfiguration {
    return { ...this.config };
  }

  /**
   * Update cell configuration
   */
  public async updateConfiguration(updates: Partial<CellConfiguration>): Promise<void> {
    this.logger.info('Updating cell configuration', { updates });
    
    const newConfig = { ...this.config, ...updates };
    await this.validateConfiguration(newConfig);
    
    this.config = newConfig;
    await this.configManager.saveConfiguration(this.config.cellId, this.config);
    
    this.logger.info('Cell configuration updated successfully');
  }

  /**
   * Set cell parameter
   */
  public async setParameter(parameterName: string, value: any): Promise<void> {
    this.logger.info('Setting cell parameter', { parameterName, value });
    
    // Update in configuration
    this.config.parameters[parameterName] = value;
    
    // Save configuration
    await this.configManager.saveConfiguration(this.config.cellId, this.config);
    
    this.logger.info('Cell parameter set successfully', { parameterName, value });
  }

  /**
   * Get cell parameter
   */
  public getParameter(parameterName: string): any {
    return this.config.parameters[parameterName];
  }

  /**
   * Get all cell parameters
   */
  public getParameters(): Record<string, any> {
    return { ...this.config.parameters };
  }

  /**
   * Get cell ID
   */
  public getCellId(): string {
    return this.config.cellId;
  }

  /**
   * Get cell name
   */
  public getCellName(): string {
    return this.config.cellName;
  }

  /**
   * Get cell type
   */
  public getCellType(): CellType {
    return this.config.cellType;
  }

  /**
   * Get parent node ID
   */
  public getParentNodeId(): string {
    return this.config.parentNodeId;
  }

  /**
   * Get sector number
   */
  public getSector(): number {
    return this.config.sector;
  }

  /**
   * Get frequency
   */
  public getFrequency(): number {
    return this.config.frequency;
  }

  /**
   * Get bandwidth
   */
  public getBandwidth(): number {
    return this.config.bandwidth;
  }

  /**
   * Set frequency
   */
  public async setFrequency(frequency: number): Promise<void> {
    if (frequency <= 0) {
      throw new Error('Frequency must be positive');
    }

    this.config.frequency = frequency;
    
    // Update type-specific frequency parameters
    if (this.config.cellType === CellType.LTE) {
      await this.setParameter('earfcnDL', this.frequencyToEARFCN(frequency));
    } else if (this.config.cellType === CellType.NR) {
      await this.setParameter('nrarfcnDL', this.frequencyToNRARFCN(frequency));
    }

    await this.configManager.saveConfiguration(this.config.cellId, this.config);
    this.logger.info('Cell frequency set successfully', { frequency });
  }

  /**
   * Set bandwidth
   */
  public async setBandwidth(bandwidth: number): Promise<void> {
    if (bandwidth <= 0) {
      throw new Error('Bandwidth must be positive');
    }

    // Validate bandwidth for cell type
    if (this.config.cellType === CellType.LTE) {
      const validLTEBW = [1.4, 3, 5, 10, 15, 20];
      if (!validLTEBW.includes(bandwidth)) {
        throw new Error(`Invalid LTE bandwidth. Must be one of: ${validLTEBW.join(', ')} MHz`);
      }
    } else if (this.config.cellType === CellType.NR) {
      const validNRBW = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400];
      if (!validNRBW.includes(bandwidth)) {
        throw new Error(`Invalid NR bandwidth. Must be one of: ${validNRBW.join(', ')} MHz`);
      }
    }

    this.config.bandwidth = bandwidth;
    
    // Update bandwidth-related parameters
    if (this.config.cellType === CellType.LTE) {
      await this.setParameter('dlChannelBandwidth', bandwidth * 1000); // Convert to kHz
      await this.setParameter('ulChannelBandwidth', bandwidth * 1000);
    } else if (this.config.cellType === CellType.NR) {
      await this.setParameter('channelBandwidth', bandwidth);
    }

    await this.configManager.saveConfiguration(this.config.cellId, this.config);
    this.logger.info('Cell bandwidth set successfully', { bandwidth });
  }

  /**
   * Set Physical Cell Identity
   */
  public async setPCI(pci: number): Promise<void> {
    let maxPCI = 503; // LTE default
    
    if (this.config.cellType === CellType.NR) {
      maxPCI = 1007; // NR expanded range
    }

    if (pci < 0 || pci > maxPCI) {
      throw new Error(`PCI must be between 0 and ${maxPCI} for ${this.config.cellType}`);
    }

    await this.setParameter('pci', pci);
    
    if (this.config.cellType === CellType.NR) {
      await this.setParameter('physicalCellId', pci);
    }

    this.logger.info('PCI set successfully', { pci });
  }

  /**
   * Configure antenna parameters
   */
  public async configureAntenna(params: {
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
   * Configure power parameters
   */
  public async configurePower(params: {
    maxTxPower?: number;
    referenceSignalPower?: number;
    powerOffset?: number;
  }): Promise<void> {
    this.logger.info('Configuring power parameters', { params });

    const updates: Array<Promise<void>> = [];

    if (params.maxTxPower !== undefined) {
      if (params.maxTxPower < 0 || params.maxTxPower > 50) {
        throw new Error('Max transmission power must be between 0 and 50 dBm');
      }
      updates.push(this.setParameter('maxTransmissionPower', params.maxTxPower));
    }

    if (params.referenceSignalPower !== undefined) {
      if (params.referenceSignalPower < -60 || params.referenceSignalPower > 50) {
        throw new Error('Reference signal power must be between -60 and 50 dBm');
      }
      
      if (this.config.cellType === CellType.LTE) {
        updates.push(this.setParameter('referenceSignalPower', params.referenceSignalPower));
      } else if (this.config.cellType === CellType.NR) {
        updates.push(this.setParameter('ssbReferenceSignalPower', params.referenceSignalPower));
      }
    }

    if (params.powerOffset !== undefined) {
      if (Math.abs(params.powerOffset) > 10) {
        throw new Error('Power offset must be between -10 and 10 dB');
      }
      updates.push(this.setParameter('powerOffset', params.powerOffset));
    }

    await Promise.all(updates);
    this.logger.info('Power parameters configured successfully');
  }

  /**
   * Configure mobility parameters
   */
  public async configureMobility(params: {
    qQualMin?: number;
    qRxLevMin?: number;
    hysteresisA3?: number;
    thresholdA3?: number;
    timeToTriggerA3?: number;
  }): Promise<void> {
    this.logger.info('Configuring mobility parameters', { params });

    const updates: Array<Promise<void>> = [];

    if (params.qQualMin !== undefined) {
      const minValue = this.config.cellType === CellType.LTE ? -34 : -43;
      if (params.qQualMin < minValue || params.qQualMin > -3) {
        throw new Error(`qQualMin must be between ${minValue} and -3`);
      }
      updates.push(this.setParameter('qQualMin', params.qQualMin));
    }

    if (params.qRxLevMin !== undefined) {
      if (params.qRxLevMin < -70 || params.qRxLevMin > -22) {
        throw new Error('qRxLevMin must be between -70 and -22');
      }
      updates.push(this.setParameter('qRxLevMin', params.qRxLevMin));
    }

    if (params.hysteresisA3 !== undefined) {
      const maxHysteresis = this.config.cellType === CellType.LTE ? 150 : 150;
      if (params.hysteresisA3 < 0 || params.hysteresisA3 > maxHysteresis) {
        throw new Error(`hysteresisA3 must be between 0 and ${maxHysteresis} (0.1 dB steps)`);
      }
      updates.push(this.setParameter('hysteresisA3', params.hysteresisA3));
    }

    if (params.thresholdA3 !== undefined) {
      if (params.thresholdA3 < -30 || params.thresholdA3 > 30) {
        throw new Error('thresholdA3 must be between -30 and 30 dB');
      }
      updates.push(this.setParameter('thresholdA3', params.thresholdA3));
    }

    if (params.timeToTriggerA3 !== undefined) {
      const validTTT = [0, 40, 64, 80, 100, 128, 160, 256, 320, 480, 512, 640, 1024, 1280, 2560, 5120];
      if (!validTTT.includes(params.timeToTriggerA3)) {
        throw new Error(`timeToTriggerA3 must be one of: ${validTTT.join(', ')} ms`);
      }
      updates.push(this.setParameter('timeToTriggerA3', params.timeToTriggerA3));
    }

    await Promise.all(updates);
    this.logger.info('Mobility parameters configured successfully');
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
   * Get cell coverage information
   */
  public getCoverageInfo(): {
    frequency: number;
    bandwidth: number;
    sector: number;
    range: number;
    azimuth: number;
  } {
    return {
      frequency: this.config.frequency,
      bandwidth: this.config.bandwidth,
      sector: this.config.sector,
      range: this.config.parameters.cellRange || 5000,
      azimuth: this.config.parameters.antennaAzimuth || (this.config.sector - 1) * 120
    };
  }

  /**
   * Get cell capacity information
   */
  public getCapacityInfo(): {
    maxUEs: number;
    loadBalancingWeight: number;
    admissionThreshold: number;
    schedulingAlgorithm: string;
  } {
    return {
      maxUEs: this.config.parameters.maxUEs || 150,
      loadBalancingWeight: this.config.parameters.loadBalancingWeight || 100,
      admissionThreshold: this.config.parameters.admissionThreshold || 80,
      schedulingAlgorithm: this.config.parameters.schedulingAlgorithm || 'proportionalFair'
    };
  }

  /**
   * Get cell RF information
   */
  public getRFInfo(): Record<string, any> {
    if (this.config.cellType === CellType.LTE) {
      return {
        earfcnDL: this.config.parameters.earfcnDL,
        earfcnUL: this.config.parameters.earfcnUL,
        dlChannelBandwidth: this.config.parameters.dlChannelBandwidth,
        ulChannelBandwidth: this.config.parameters.ulChannelBandwidth,
        referenceSignalPower: this.config.parameters.referenceSignalPower,
        pci: this.config.parameters.pci
      };
    } else {
      return {
        nrarfcnDL: this.config.parameters.nrarfcnDL,
        nrarfcnUL: this.config.parameters.nrarfcnUL,
        channelBandwidth: this.config.parameters.channelBandwidth,
        ssbSubCarrierSpacing: this.config.parameters.ssbSubCarrierSpacing,
        ssbReferenceSignalPower: this.config.parameters.ssbReferenceSignalPower,
        physicalCellId: this.config.parameters.physicalCellId || this.config.parameters.pci
      };
    }
  }

  /**
   * Calculate theoretical throughput
   */
  public calculateTheoreticalThroughput(): { downlink: number; uplink: number } {
    const bandwidth = this.config.bandwidth;
    const mimoLayers = this.config.parameters.maxMimoLayers || 2;
    
    if (this.config.cellType === CellType.LTE) {
      // LTE theoretical throughput calculation
      const specEfficiency = 5.0; // bits/s/Hz for 64QAM
      const overhead = 0.25; // 25% overhead
      
      const dlThroughput = bandwidth * 1e6 * specEfficiency * mimoLayers * (1 - overhead) / 1e6; // Mbps
      const ulThroughput = bandwidth * 1e6 * specEfficiency * 1 * (1 - overhead) / 1e6; // Mbps (SISO)
      
      return { downlink: dlThroughput, uplink: ulThroughput };
    } else {
      // NR theoretical throughput calculation
      const numerology = this.config.parameters.numerology || 1;
      const specEfficiency = 7.4; // bits/s/Hz for 256QAM
      const overhead = 0.15; // 15% overhead (lower than LTE)
      
      const dlThroughput = bandwidth * 1e6 * specEfficiency * mimoLayers * (1 - overhead) / 1e6; // Mbps
      const ulThroughput = bandwidth * 1e6 * specEfficiency * 2 * (1 - overhead) / 1e6; // Mbps (2 layers UL)
      
      return { downlink: dlThroughput, uplink: ulThroughput };
    }
  }

  /**
   * Get neighbor cells
   */
  public getNeighborCells(): string[] {
    return this.config.parameters.neighborCells || [];
  }

  /**
   * Add neighbor cell
   */
  public async addNeighborCell(cellId: string): Promise<void> {
    const neighbors = this.getNeighborCells();
    if (!neighbors.includes(cellId)) {
      neighbors.push(cellId);
      await this.setParameter('neighborCells', neighbors);
      this.logger.info('Neighbor cell added', { cellId });
    }
  }

  /**
   * Remove neighbor cell
   */
  public async removeNeighborCell(cellId: string): Promise<void> {
    const neighbors = this.getNeighborCells();
    const index = neighbors.indexOf(cellId);
    if (index > -1) {
      neighbors.splice(index, 1);
      await this.setParameter('neighborCells', neighbors);
      this.logger.info('Neighbor cell removed', { cellId });
    }
  }

  /**
   * Validate cell configuration
   */
  private async validateConfiguration(config: CellConfiguration): Promise<void> {
    if (!config.cellId || !config.cellName) {
      throw new Error('Cell ID and name are required');
    }

    if (!config.parentNodeId) {
      throw new Error('Parent node ID is required');
    }

    if (!Object.values(CellType).includes(config.cellType)) {
      throw new Error(`Invalid cell type: ${config.cellType}`);
    }

    if (config.sector <= 0) {
      throw new Error('Sector must be a positive number');
    }

    if (config.frequency <= 0) {
      throw new Error('Frequency must be positive');
    }

    if (config.bandwidth <= 0) {
      throw new Error('Bandwidth must be positive');
    }

    this.logger.debug('Cell configuration validated');
  }

  /**
   * Convert frequency to EARFCN (simplified)
   */
  private frequencyToEARFCN(frequencyMHz: number): number {
    // Simplified conversion for Band 3 (1800 MHz)
    if (frequencyMHz >= 1710 && frequencyMHz <= 1785) {
      return Math.round(1200 + (frequencyMHz - 1710) * 10);
    }
    // Default for other bands
    return Math.round(frequencyMHz * 10);
  }

  /**
   * Convert frequency to NR-ARFCN (simplified)
   */
  private frequencyToNRARFCN(frequencyMHz: number): number {
    // Simplified conversion for n78 band (3.5 GHz)
    if (frequencyMHz >= 3300 && frequencyMHz <= 4200) {
      return Math.round(620000 + (frequencyMHz - 3300) * 100);
    }
    // Default
    return Math.round(frequencyMHz * 100);
  }

  /**
   * Dispose cell resources
   */
  public async dispose(): Promise<void> {
    this.logger.info('Disposing cell resources', { cellId: this.config.cellId });
    
    // Cleanup any resources, connections, etc.
    
    this.logger.info('Cell disposed successfully');
  }
}