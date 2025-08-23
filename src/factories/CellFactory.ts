import { CellConfiguration, CellType } from '../types';
import { ConfigurationManager } from '../core/ConfigurationManager';
import { Cell } from '../nodes/Cell';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';

/**
 * Factory for creating cell configurations and instances
 * Supports both LTE and NR cell types with appropriate parameters
 */
export class CellFactory {
  private logger: Logger;
  private configManager: ConfigurationManager;
  private cellCache: Map<string, Cell> = new Map();

  constructor(configManager?: ConfigurationManager) {
    this.logger = createLogger({
      level: 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json()
      ),
      defaultMeta: { service: 'CellFactory' },
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
  }

  /**
   * Create a cell instance from configuration
   */
  public async createCell(config: CellConfiguration): Promise<Cell> {
    this.logger.info('Creating cell', { 
      cellId: config.cellId, 
      cellType: config.cellType,
      parentNodeId: config.parentNodeId
    });

    // Check cache first
    if (this.cellCache.has(config.cellId)) {
      this.logger.info('Returning cached cell instance', { cellId: config.cellId });
      return this.cellCache.get(config.cellId)!;
    }

    // Validate configuration
    await this.validateCellConfiguration(config);

    // Enrich configuration with type-specific defaults
    const enrichedConfig = await this.enrichConfiguration(config);

    // Create cell instance
    const cell = new Cell(enrichedConfig, this.configManager);
    
    // Cache the cell
    this.cellCache.set(config.cellId, cell);
    
    // Save configuration
    await this.configManager.saveConfiguration(config.cellId, enrichedConfig);

    this.logger.info('Cell created successfully', { 
      cellId: config.cellId,
      cellType: config.cellType
    });

    return cell;
  }

  /**
   * Create LTE cell with default parameters
   */
  public async createLTECell(
    cellId: string,
    cellName: string,
    parentNodeId: string,
    sector: number,
    overrides: Partial<CellConfiguration> = {}
  ): Promise<Cell> {
    const config: CellConfiguration = {
      cellId,
      cellName,
      cellType: CellType.LTE,
      parentNodeId,
      sector,
      frequency: 1800, // Default Band 3
      bandwidth: 20,   // 20 MHz
      parameters: this.getDefaultLTEParameters(),
      ...overrides
    };

    return this.createCell(config);
  }

  /**
   * Create NR cell with default parameters
   */
  public async createNRCell(
    cellId: string,
    cellName: string,
    parentNodeId: string,
    sector: number,
    band: 'n78' | 'n28' | 'n1' | 'n3' = 'n78',
    overrides: Partial<CellConfiguration> = {}
  ): Promise<Cell> {
    const config: CellConfiguration = {
      cellId,
      cellName,
      cellType: CellType.NR,
      parentNodeId,
      sector,
      frequency: this.getFrequencyForNRBand(band),
      bandwidth: this.getBandwidthForNRBand(band),
      parameters: this.getDefaultNRParameters(band),
      ...overrides
    };

    return this.createCell(config);
  }

  /**
   * Create multiple cells for a site
   */
  public async createSiteCells(
    siteId: string,
    parentNodeId: string,
    sectorCount: number,
    cellType: CellType,
    overrides: Partial<CellConfiguration> = {}
  ): Promise<Cell[]> {
    this.logger.info('Creating site cells', { 
      siteId,
      parentNodeId,
      sectorCount,
      cellType
    });

    const cells: Cell[] = [];
    
    for (let sector = 1; sector <= sectorCount; sector++) {
      const cellId = `${siteId}_${parentNodeId}_${sector}`;
      const cellName = `${siteId}_Cell_${sector}`;

      let cell: Cell;
      if (cellType === CellType.LTE) {
        cell = await this.createLTECell(cellId, cellName, parentNodeId, sector, overrides);
      } else {
        cell = await this.createNRCell(cellId, cellName, parentNodeId, sector, 'n78', overrides);
      }
      
      cells.push(cell);
    }

    this.logger.info('Site cells created successfully', { 
      siteId,
      cellCount: cells.length
    });

    return cells;
  }

  /**
   * Create cell from template
   */
  public async createFromTemplate(
    template: 'macro' | 'micro' | 'pico' | 'indoor',
    cellId: string,
    cellName: string,
    parentNodeId: string,
    sector: number,
    cellType: CellType,
    overrides: Partial<CellConfiguration> = {}
  ): Promise<Cell> {
    this.logger.info('Creating cell from template', { 
      template,
      cellId,
      cellType
    });

    const templateConfig = this.getTemplateConfiguration(template, cellType);
    const config: CellConfiguration = {
      cellId,
      cellName,
      cellType,
      parentNodeId,
      sector,
      frequency: templateConfig.frequency || 1800,
      bandwidth: templateConfig.bandwidth || 20,
      parameters: { ...templateConfig.parameters },
      ...overrides
    };

    return this.createCell(config);
  }

  /**
   * Get cached cell
   */
  public getCachedCell(cellId: string): Cell | undefined {
    return this.cellCache.get(cellId);
  }

  /**
   * Remove cell from cache
   */
  public removeCachedCell(cellId: string): boolean {
    const removed = this.cellCache.delete(cellId);
    if (removed) {
      this.logger.info('Cell removed from cache', { cellId });
    }
    return removed;
  }

  /**
   * Clear all cached cells
   */
  public clearCache(): void {
    this.logger.info('Clearing cell cache', { count: this.cellCache.size });
    this.cellCache.clear();
  }

  /**
   * Get default LTE cell parameters
   */
  private getDefaultLTEParameters(): Record<string, any> {
    return {
      // Physical Cell Identity
      pci: 0,
      
      // RF parameters
      earfcnDL: 1950,        // Band 3
      earfcnUL: 19950,
      dlChannelBandwidth: 20000, // kHz
      ulChannelBandwidth: 20000,
      
      // Power parameters
      referenceSignalPower: 15,  // dBm
      maxTransmissionPower: 46,
      pB: 0,
      pA: 0,
      
      // Antenna parameters
      antennaAzimuth: 0,
      antennaTilt: 5,
      mechanicalTilt: 2,
      antennaGain: 18,
      
      // Cell access parameters
      cellBarred: 'notBarred',
      intraFreqReselection: 'allowed',
      qQualMin: -34,
      qRxLevMin: -70,
      
      // Handover parameters
      hysteresisA1: 20,
      hysteresisA2: 20,
      hysteresisA3: 20,
      thresholdA1RSRP: -95,
      thresholdA2RSRP: -90,
      thresholdA3Offset: 30,
      timeToTriggerA1: 320,
      timeToTriggerA2: 320,
      timeToTriggerA3: 160,
      
      // Load balancing
      loadBalancingWeight: 100,
      admissionThreshold: 80,
      
      // Scheduling
      schedulingAlgorithm: 'proportionalFair',
      maxMimoLayers: 2,
      
      // QoS
      qci: [1, 2, 3, 4, 5, 6, 7, 8, 9],
      
      // Coverage
      cellRange: 5000,       // meters
      maxUEs: 150
    };
  }

  /**
   * Get default NR cell parameters
   */
  private getDefaultNRParameters(band: string = 'n78'): Record<string, any> {
    return {
      // Physical Cell Identity (NR range)
      pci: 0,                // 0-1007
      
      // RF parameters
      nrarfcnDL: this.getNRARFCNForBand(band),
      nrarfcnUL: this.getNRARFCNForBand(band),
      frequencyBand: this.getBandNumberForBand(band),
      ssbSubCarrierSpacing: this.getSSBSCSForBand(band),
      numerology: this.getNumerologyForBand(band),
      
      // SSB parameters
      ssbFrequency: this.getSSBFrequencyForBand(band),
      ssbPeriodicity: 20,    // ms
      ssbPattern: this.getSSBPatternForBand(band),
      maxSSBperBurst: 8,
      
      // Power parameters
      ssbReferenceSignalPower: 0,  // dBm
      maxTransmissionPower: 46,
      pdschPowerAllocation: 0,
      puschPowerControl: 0,
      
      // Antenna parameters (enhanced for NR)
      antennaAzimuth: 0,
      antennaTilt: 10,
      mechanicalTilt: 5,
      antennaGain: 20,
      antennaType: 'massive_mimo',
      antennaElements: 64,
      
      // Beam parameters
      beamCount: 8,
      beamWidth: 65,
      beamSweeping: true,
      beamTracking: true,
      
      // Cell access parameters
      cellBarred: 'notBarred',
      intraFreqReselection: 'allowed',
      qQualMin: -43,
      qRxLevMin: -70,
      
      // Handover parameters (NR)
      hysteresisA3: 30,      // 3.0 dB
      thresholdA3: 30,       // 3.0 dB
      timeToTriggerA3: 100,  // 100 ms
      
      // CSI-RS configuration
      csiRsConfig: {
        periodicity: 20,     // ms
        density: 3,
        powerOffset: 0
      },
      
      // DMRS configuration
      dmrsConfig: {
        type: 'A',
        additionalPosition: 1,
        maxLength: 'single'
      },
      
      // Bandwidth parts
      bandwidthParts: [
        {
          id: 0,
          startRB: 0,
          numRB: this.getResourceBlocksForBandwidth(this.getBandwidthForNRBand(band)),
          numerology: this.getNumerologyForBand(band)
        }
      ],
      
      // Frame structure (TDD)
      frameStructure: 'TDD',
      tddPattern: {
        periodicity: '5ms',
        dlSlots: 7,
        ulSlots: 2,
        specialSlot: 1
      },
      
      // Scheduling (enhanced)
      schedulingAlgorithm: 'proportionalFair',
      maxMimoLayers: 4,
      resourceBlockGroups: true,
      
      // QoS (5G)
      qci5G: [1, 2, 3, 4, 5, 6, 7, 8, 9, 80, 82, 83, 84, 85],
      
      // Network slicing
      networkSlicing: false,
      supportedSlices: [
        { sst: 1, sd: '000001' }, // eMBB
        { sst: 2, sd: '000002' }, // URLLC
        { sst: 3, sd: '000003' }  // mMTC
      ],
      
      // Coverage and capacity
      cellRange: this.getCellRangeForBand(band),
      maxUEs: 200,           // Higher for NR
      
      // Load balancing
      loadBalancingWeight: 100,
      admissionThreshold: 85,
      
      // Advanced NR features
      carrierAggregation: true,
      coordinatedMultipoint: false,
      ultraLowLatency: false
    };
  }

  /**
   * Get template configuration
   */
  private getTemplateConfiguration(template: string, cellType: CellType): Partial<CellConfiguration> {
    const lteTemplates = {
      macro: {
        frequency: 1800,
        bandwidth: 20,
        parameters: {
          maxTransmissionPower: 46,
          cellRange: 10000,
          antennaTilt: 5,
          antennaGain: 18
        }
      },
      micro: {
        frequency: 1800,
        bandwidth: 20,
        parameters: {
          maxTransmissionPower: 38,
          cellRange: 2000,
          antennaTilt: 8,
          antennaGain: 15
        }
      },
      pico: {
        frequency: 2600,
        bandwidth: 10,
        parameters: {
          maxTransmissionPower: 30,
          cellRange: 500,
          antennaTilt: 10,
          antennaGain: 12
        }
      },
      indoor: {
        frequency: 2600,
        bandwidth: 10,
        parameters: {
          maxTransmissionPower: 24,
          cellRange: 100,
          antennaTilt: 0,
          antennaGain: 8,
          deploymentType: 'indoor'
        }
      }
    };

    const nrTemplates = {
      macro: {
        frequency: 3500,
        bandwidth: 100,
        parameters: {
          maxTransmissionPower: 46,
          cellRange: 15000,
          antennaTilt: 8,
          antennaGain: 20,
          antennaElements: 64,
          beamCount: 16
        }
      },
      micro: {
        frequency: 3500,
        bandwidth: 80,
        parameters: {
          maxTransmissionPower: 38,
          cellRange: 3000,
          antennaTilt: 12,
          antennaGain: 18,
          antennaElements: 32,
          beamCount: 8
        }
      },
      pico: {
        frequency: 3500,
        bandwidth: 40,
        parameters: {
          maxTransmissionPower: 30,
          cellRange: 1000,
          antennaTilt: 15,
          antennaGain: 15,
          antennaElements: 16,
          beamCount: 4
        }
      },
      indoor: {
        frequency: 3500,
        bandwidth: 40,
        parameters: {
          maxTransmissionPower: 24,
          cellRange: 200,
          antennaTilt: 0,
          antennaGain: 12,
          antennaElements: 16,
          beamCount: 4,
          deploymentType: 'indoor'
        }
      }
    };

    const templates = cellType === CellType.LTE ? lteTemplates : nrTemplates;
    return templates[template as keyof typeof templates] || {};
  }

  /**
   * Enrich configuration with type-specific parameters
   */
  private async enrichConfiguration(config: CellConfiguration): Promise<CellConfiguration> {
    const enriched = { ...config };
    
    // Add sector-specific parameters
    if (config.sector && config.sector > 0) {
      enriched.parameters = {
        ...enriched.parameters,
        antennaAzimuth: (config.sector - 1) * 120, // 0°, 120°, 240°
        pci: (config.sector - 1) % (config.cellType === CellType.NR ? 1008 : 504)
      };
    }
    
    return enriched;
  }

  /**
   * Validate cell configuration
   */
  private async validateCellConfiguration(config: CellConfiguration): Promise<void> {
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

    // Type-specific validation
    if (config.cellType === CellType.LTE) {
      await this.validateLTEConfiguration(config);
    } else if (config.cellType === CellType.NR) {
      await this.validateNRConfiguration(config);
    }

    this.logger.debug('Cell configuration validated', { 
      cellId: config.cellId,
      cellType: config.cellType
    });
  }

  /**
   * Validate LTE specific configuration
   */
  private async validateLTEConfiguration(config: CellConfiguration): Promise<void> {
    const validBandwidths = [1.4, 3, 5, 10, 15, 20]; // MHz
    if (!validBandwidths.includes(config.bandwidth)) {
      throw new Error(`Invalid LTE bandwidth. Must be one of: ${validBandwidths.join(', ')} MHz`);
    }

    if (config.parameters?.pci !== undefined) {
      const pci = Number(config.parameters.pci);
      if (isNaN(pci) || pci < 0 || pci > 503) {
        throw new Error('LTE PCI must be between 0 and 503');
      }
    }
  }

  /**
   * Validate NR specific configuration
   */
  private async validateNRConfiguration(config: CellConfiguration): Promise<void> {
    const validBandwidths = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400]; // MHz
    if (!validBandwidths.includes(config.bandwidth)) {
      throw new Error(`Invalid NR bandwidth. Must be one of: ${validBandwidths.join(', ')} MHz`);
    }

    if (config.parameters?.pci !== undefined) {
      const pci = Number(config.parameters.pci);
      if (isNaN(pci) || pci < 0 || pci > 1007) {
        throw new Error('NR PCI must be between 0 and 1007');
      }
    }
  }

  // Helper methods for NR band parameters
  private getFrequencyForNRBand(band: string): number {
    const frequencies = {
      n78: 3500, n28: 700, n1: 2100, n3: 1800
    };
    return frequencies[band as keyof typeof frequencies] || 3500;
  }

  private getBandwidthForNRBand(band: string): number {
    const bandwidths = {
      n78: 100, n28: 20, n1: 20, n3: 20
    };
    return bandwidths[band as keyof typeof bandwidths] || 100;
  }

  private getNRARFCNForBand(band: string): number {
    const arfcns = {
      n78: 632000, n28: 151600, n1: 422000, n3: 361000
    };
    return arfcns[band as keyof typeof arfcns] || 632000;
  }

  private getBandNumberForBand(band: string): number {
    return Number(band.replace('n', ''));
  }

  private getSSBSCSForBand(band: string): number {
    const scs = {
      n78: 30, n28: 15, n1: 15, n3: 15
    };
    return scs[band as keyof typeof scs] || 30;
  }

  private getNumerologyForBand(band: string): number {
    const numerologies = {
      n78: 1, n28: 0, n1: 0, n3: 0
    };
    return numerologies[band as keyof typeof numerologies] || 1;
  }

  private getSSBFrequencyForBand(band: string): number {
    const frequencies = {
      n78: 3500000, n28: 700000, n1: 2100000, n3: 1800000
    };
    return frequencies[band as keyof typeof frequencies] || 3500000;
  }

  private getSSBPatternForBand(band: string): string {
    const patterns = {
      n78: 'Case C', n28: 'Case A', n1: 'Case A', n3: 'Case A'
    };
    return patterns[band as keyof typeof patterns] || 'Case C';
  }

  private getResourceBlocksForBandwidth(bandwidthMHz: number): number {
    const rbMap = {
      5: 25, 10: 52, 15: 79, 20: 106, 25: 133, 30: 160,
      40: 216, 50: 270, 60: 324, 70: 378, 80: 432, 90: 486,
      100: 540, 200: 1080, 400: 2160
    };
    return rbMap[bandwidthMHz as keyof typeof rbMap] || 540;
  }

  private getCellRangeForBand(band: string): number {
    const ranges = {
      n78: 5000, n28: 15000, n1: 8000, n3: 8000
    };
    return ranges[band as keyof typeof ranges] || 5000;
  }
}