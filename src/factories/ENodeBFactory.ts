import { ENodeB } from '../nodes/ENodeB';
import { NodeConfiguration, NodeType, CellConfiguration, CellType } from '../types';
import { ConfigurationManager } from '../core/ConfigurationManager';
import { CMEditClient } from '../core/CMEditClient';
import { CellFactory } from './CellFactory';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';

/**
 * Specialized factory for creating LTE eNodeB instances
 * Provides eNodeB-specific templates and validation
 */
export class ENodeBFactory {
  private logger: Logger;
  private configManager: ConfigurationManager;
  private cmEditClient: CMEditClient;
  private cellFactory: CellFactory;

  constructor(
    configManager?: ConfigurationManager,
    cmEditClient?: CMEditClient
  ) {
    this.logger = createLogger({
      level: 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json()
      ),
      defaultMeta: { service: 'ENodeBFactory' },
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
    this.cmEditClient = cmEditClient || new CMEditClient();
    this.cellFactory = new CellFactory(this.configManager);
  }

  /**
   * Create eNodeB with default LTE configuration
   */
  public async createENodeB(
    nodeId: string,
    nodeName: string,
    siteId: string,
    overrides: Partial<NodeConfiguration> = {}
  ): Promise<ENodeB> {
    this.logger.info('Creating eNodeB', { nodeId, nodeName, siteId });

    const config: NodeConfiguration = {
      nodeId,
      nodeName,
      nodeType: NodeType.ENODEB,
      siteId,
      parameters: this.getDefaultENodeBParameters(),
      ...overrides
    };

    await this.validateENodeBConfiguration(config);
    
    const eNodeB = new ENodeB(config, this.configManager, this.cmEditClient);
    
    // Save configuration
    await this.configManager.saveConfiguration(nodeId, config);
    
    this.logger.info('eNodeB created successfully', { nodeId });
    return eNodeB;
  }

  /**
   * Create eNodeB with multiple sectors (cells)
   */
  public async createENodeBWithCells(
    nodeId: string,
    nodeName: string,
    siteId: string,
    sectorCount: number = 3,
    overrides: Partial<NodeConfiguration> = {}
  ): Promise<{ eNodeB: ENodeB; cells: Array<any> }> {
    this.logger.info('Creating eNodeB with cells', { 
      nodeId, 
      nodeName, 
      siteId, 
      sectorCount 
    });

    // Create eNodeB
    const eNodeB = await this.createENodeB(nodeId, nodeName, siteId, overrides);
    
    // Create cells for each sector
    const cells: Array<any> = [];
    for (let sector = 1; sector <= sectorCount; sector++) {
      const cellId = `${nodeId}_${sector}`;
      const cellName = `${nodeName}_Cell_${sector}`;
      
      const cellConfig: CellConfiguration = {
        cellId,
        cellName,
        cellType: CellType.LTE,
        parentNodeId: nodeId,
        sector,
        frequency: this.getFrequencyForSector(sector),
        bandwidth: 20, // Default 20 MHz
        parameters: this.getDefaultCellParameters(sector)
      };
      
      const cell = await this.cellFactory.createCell(cellConfig);
      cells.push(cell);
    }

    this.logger.info('eNodeB with cells created successfully', { 
      nodeId, 
      cellCount: cells.length 
    });

    return { eNodeB, cells };
  }

  /**
   * Create eNodeB from template
   */
  public async createFromTemplate(
    template: 'macro' | 'micro' | 'pico' | 'femto',
    nodeId: string,
    nodeName: string,
    siteId: string,
    overrides: Partial<NodeConfiguration> = {}
  ): Promise<ENodeB> {
    this.logger.info('Creating eNodeB from template', { 
      template, 
      nodeId, 
      nodeName 
    });

    const templateConfig = this.getTemplateConfiguration(template);
    const mergedConfig = { ...templateConfig, ...overrides };
    
    return this.createENodeB(nodeId, nodeName, siteId, mergedConfig);
  }

  /**
   * Bulk create eNodeBs from site configuration
   */
  public async createBulkENodeBs(
    sites: Array<{
      nodeId: string;
      nodeName: string;
      siteId: string;
      template?: 'macro' | 'micro' | 'pico' | 'femto';
      overrides?: Partial<NodeConfiguration>;
    }>
  ): Promise<Array<{ eNodeB: ENodeB; error?: string }>> {
    this.logger.info('Creating bulk eNodeBs', { count: sites.length });

    const results: Array<{ eNodeB: ENodeB; error?: string }> = [];
    
    for (const site of sites) {
      try {
        let eNodeB: ENodeB;
        
        if (site.template) {
          eNodeB = await this.createFromTemplate(
            site.template,
            site.nodeId,
            site.nodeName,
            site.siteId,
            site.overrides
          );
        } else {
          eNodeB = await this.createENodeB(
            site.nodeId,
            site.nodeName,
            site.siteId,
            site.overrides
          );
        }
        
        results.push({ eNodeB });
        
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        results.push({ 
          eNodeB: null as any, 
          error: errorMessage 
        });
        
        this.logger.error('Failed to create eNodeB', { 
          nodeId: site.nodeId, 
          error: errorMessage 
        });
      }
    }

    const successful = results.filter(r => !r.error).length;
    const failed = results.filter(r => r.error).length;
    
    this.logger.info('Bulk eNodeB creation completed', { 
      total: sites.length,
      successful,
      failed
    });

    return results;
  }

  /**
   * Get default eNodeB parameters
   */
  private getDefaultENodeBParameters(): Record<string, any> {
    return {
      // Basic eNodeB parameters
      eNodeBId: 0,
      tac: 1,
      mcc: '001', // Mobile Country Code
      mnc: '01',  // Mobile Network Code
      
      // RF parameters
      earfcnDL: 1950,    // EARFCN Downlink (Band 3, 1800 MHz)
      earfcnUL: 19950,   // EARFCN Uplink
      dlChannelBandwidth: 20000, // 20 MHz
      ulChannelBandwidth: 20000,
      
      // Cell parameters
      cellBarred: 'notBarred',
      intraFreqReselection: 'allowed',
      qQualMin: -34,
      qRxLevMin: -70,
      
      // Handover parameters
      hysteresisA3: 20,   // 2.0 dB
      thresholdA3: 30,    // 3.0 dB
      timeToTriggerA3: 160, // 160 ms
      
      // Power parameters
      referenceSignalPower: 15,  // dBm
      pB: 0,              // Power boosting
      pA: 0,              // PA parameter
      
      // Physical parameters
      rootSequenceIndex: 0,
      zeroCorrelationIndex: 0,
      highSpeedFlag: false,
      prb: 100,           // Physical Resource Blocks
      
      // Mobility parameters
      s_Measure: 0,       // Signal strength measurement
      neighCellConfig: 0,
      t_Evaluation: 240,  // 240s
      t_HystNormal: 240,  // 240s
      
      // QoS parameters
      qci: [1, 2, 3, 4, 5, 6, 7, 8, 9], // QCI values
      arp: {
        priorityLevel: 15,
        preEmptionCapability: 'shall_not_trigger_pre_emption',
        preEmptionVulnerability: 'not_pre_emptable'
      }
    };
  }

  /**
   * Get default cell parameters for sector
   */
  private getDefaultCellParameters(sector: number): Record<string, any> {
    return {
      // Physical Cell Identity
      pci: sector - 1,    // PCI 0, 1, 2 for sectors 1, 2, 3
      
      // Antenna parameters
      antennaAzimuth: (sector - 1) * 120,  // 0°, 120°, 240°
      antennaTilt: 5,     // Electrical tilt
      mechanicalTilt: 2,  // Mechanical tilt
      antennaGain: 18,    // dBi
      
      // Power parameters
      maxTransmissionPower: 46,  // dBm
      referenceSignalPower: 15,  // dBm
      
      // Scheduling parameters
      schedulingAlgorithm: 'proportionalFair',
      maxMimoLayers: 2,
      
      // Load balancing
      loadBalancingWeight: 100,
      admissionThreshold: 80,
      
      // Coverage parameters
      cellRange: 5000,    // meters
      maxUEs: 150,        // Maximum UEs per cell
      
      // Frequency specific
      subframeAssignment: 2,    // TDD configuration
      specialSubframePatterns: 7
    };
  }

  /**
   * Get template configuration
   */
  private getTemplateConfiguration(template: string): Partial<NodeConfiguration> {
    const templates = {
      macro: {
        parameters: {
          maxTransmissionPower: 46,  // dBm
          cellRange: 10000,          // 10km
          antennaHeight: 30,         // 30m
          antennaTilt: 5,
          referenceSignalPower: 18
        }
      },
      micro: {
        parameters: {
          maxTransmissionPower: 38,  // dBm
          cellRange: 2000,           // 2km
          antennaHeight: 15,         // 15m
          antennaTilt: 8,
          referenceSignalPower: 15
        }
      },
      pico: {
        parameters: {
          maxTransmissionPower: 30,  // dBm
          cellRange: 500,            // 500m
          antennaHeight: 8,          // 8m
          antennaTilt: 10,
          referenceSignalPower: 12
        }
      },
      femto: {
        parameters: {
          maxTransmissionPower: 20,  // dBm
          cellRange: 50,             // 50m
          antennaHeight: 3,          // 3m
          antennaTilt: 0,
          referenceSignalPower: 10,
          csgIndication: true,       // Closed Subscriber Group
          accessMode: 'closedAccess'
        }
      }
    };
    
    return templates[template as keyof typeof templates] || {};
  }

  /**
   * Get frequency for sector (frequency reuse)
   */
  private getFrequencyForSector(sector: number): number {
    // Simple frequency reuse pattern
    const baseFreq = 1800; // MHz
    return baseFreq + ((sector - 1) * 5); // 5 MHz separation
  }

  /**
   * Validate eNodeB specific configuration
   */
  private async validateENodeBConfiguration(config: NodeConfiguration): Promise<void> {
    if (config.nodeType !== NodeType.ENODEB) {
      throw new Error('Configuration is not for eNodeB');
    }

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

    // Validate EARFCN
    if (params.earfcnDL !== undefined) {
      const earfcn = Number(params.earfcnDL);
      if (isNaN(earfcn) || earfcn < 0 || earfcn > 262143) {
        throw new Error('EARFCN DL must be between 0 and 262143');
      }
    }

    // Validate bandwidth
    if (params.dlChannelBandwidth !== undefined) {
      const validBandwidths = [1400, 3000, 5000, 10000, 15000, 20000]; // kHz
      if (!validBandwidths.includes(params.dlChannelBandwidth)) {
        throw new Error(`Invalid DL bandwidth. Must be one of: ${validBandwidths.join(', ')}`);
      }
    }

    // Validate power parameters
    if (params.referenceSignalPower !== undefined) {
      const power = Number(params.referenceSignalPower);
      if (isNaN(power) || power < -60 || power > 50) {
        throw new Error('Reference signal power must be between -60 and 50 dBm');
      }
    }

    this.logger.debug('eNodeB configuration validated', { nodeId: config.nodeId });
  }
}