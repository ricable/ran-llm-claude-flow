import { GNodeB } from '../nodes/GNodeB';
import { NodeConfiguration, NodeType, CellConfiguration, CellType } from '../types';
import { ConfigurationManager } from '../core/ConfigurationManager';
import { CMEditClient } from '../core/CMEditClient';
import { CellFactory } from './CellFactory';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';

/**
 * Specialized factory for creating 5G gNodeB instances
 * Provides gNodeB-specific templates and validation for NR
 */
export class GNodeBFactory {
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
      defaultMeta: { service: 'GNodeBFactory' },
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
   * Create gNodeB with default 5G NR configuration
   */
  public async createGNodeB(
    nodeId: string,
    nodeName: string,
    siteId: string,
    overrides: Partial<NodeConfiguration> = {}
  ): Promise<GNodeB> {
    this.logger.info('Creating gNodeB', { nodeId, nodeName, siteId });

    const config: NodeConfiguration = {
      nodeId,
      nodeName,
      nodeType: NodeType.GNODEB,
      siteId,
      parameters: this.getDefaultGNodeBParameters(),
      ...overrides
    };

    await this.validateGNodeBConfiguration(config);
    
    const gNodeB = new GNodeB(config, this.configManager, this.cmEditClient);
    
    // Save configuration
    await this.configManager.saveConfiguration(nodeId, config);
    
    this.logger.info('gNodeB created successfully', { nodeId });
    return gNodeB;
  }

  /**
   * Create gNodeB with multiple NR cells
   */
  public async createGNodeBWithCells(
    nodeId: string,
    nodeName: string,
    siteId: string,
    sectorCount: number = 3,
    bandConfiguration: 'n78' | 'n28' | 'n1' | 'n3' = 'n78',
    overrides: Partial<NodeConfiguration> = {}
  ): Promise<{ gNodeB: GNodeB; cells: Array<any> }> {
    this.logger.info('Creating gNodeB with NR cells', { 
      nodeId, 
      nodeName, 
      siteId, 
      sectorCount,
      bandConfiguration
    });

    // Create gNodeB
    const gNodeB = await this.createGNodeB(nodeId, nodeName, siteId, overrides);
    
    // Create NR cells for each sector
    const cells: Array<any> = [];
    for (let sector = 1; sector <= sectorCount; sector++) {
      const cellId = `${nodeId}_${sector}`;
      const cellName = `${nodeName}_NRCell_${sector}`;
      
      const cellConfig: CellConfiguration = {
        cellId,
        cellName,
        cellType: CellType.NR,
        parentNodeId: nodeId,
        sector,
        frequency: this.getFrequencyForBand(bandConfiguration, sector),
        bandwidth: this.getBandwidthForBand(bandConfiguration),
        parameters: this.getDefaultNRCellParameters(sector, bandConfiguration)
      };
      
      const cell = await this.cellFactory.createCell(cellConfig);
      cells.push(cell);
    }

    this.logger.info('gNodeB with NR cells created successfully', { 
      nodeId, 
      cellCount: cells.length 
    });

    return { gNodeB, cells };
  }

  /**
   * Create gNodeB from template
   */
  public async createFromTemplate(
    template: 'macro5g' | 'micro5g' | 'indoor5g' | 'mmwave',
    nodeId: string,
    nodeName: string,
    siteId: string,
    overrides: Partial<NodeConfiguration> = {}
  ): Promise<GNodeB> {
    this.logger.info('Creating gNodeB from template', { 
      template, 
      nodeId, 
      nodeName 
    });

    const templateConfig = this.getTemplateConfiguration(template);
    const mergedConfig = { ...templateConfig, ...overrides };
    
    return this.createGNodeB(nodeId, nodeName, siteId, mergedConfig);
  }

  /**
   * Create NSA (Non-Standalone) gNodeB
   */
  public async createNSAGNodeB(
    nodeId: string,
    nodeName: string,
    siteId: string,
    anchorENodeBId: string,
    overrides: Partial<NodeConfiguration> = {}
  ): Promise<GNodeB> {
    this.logger.info('Creating NSA gNodeB', { 
      nodeId, 
      nodeName, 
      anchorENodeBId 
    });

    const nsaConfig: Partial<NodeConfiguration> = {
      parameters: {
        ...this.getDefaultGNodeBParameters(),
        deploymentMode: 'nsa',
        anchorENodeBId,
        dualConnectivity: true,
        enDcCapable: true,
        ...overrides.parameters
      }
    };

    const mergedConfig = { ...overrides, ...nsaConfig };
    return this.createGNodeB(nodeId, nodeName, siteId, mergedConfig);
  }

  /**
   * Create SA (Standalone) gNodeB with 5G Core
   */
  public async createSAGNodeB(
    nodeId: string,
    nodeName: string,
    siteId: string,
    coreNetworkConfig: {
      amfAddress: string;
      upfAddress: string;
      plmnId: { mcc: string; mnc: string };
    },
    overrides: Partial<NodeConfiguration> = {}
  ): Promise<GNodeB> {
    this.logger.info('Creating SA gNodeB', { 
      nodeId, 
      nodeName, 
      coreNetworkConfig 
    });

    const saConfig: Partial<NodeConfiguration> = {
      parameters: {
        ...this.getDefaultGNodeBParameters(),
        deploymentMode: 'sa',
        amfAddress: coreNetworkConfig.amfAddress,
        upfAddress: coreNetworkConfig.upfAddress,
        plmnId: coreNetworkConfig.plmnId,
        networkSlicing: true,
        ...overrides.parameters
      }
    };

    const mergedConfig = { ...overrides, ...saConfig };
    return this.createGNodeB(nodeId, nodeName, siteId, mergedConfig);
  }

  /**
   * Get default gNodeB parameters
   */
  private getDefaultGNodeBParameters(): Record<string, any> {
    return {
      // Basic gNodeB parameters
      gNBId: 0,
      gNBIdLength: 24,      // bits
      tac: 1,
      mcc: '001',
      mnc: '01',
      
      // NR RF parameters
      nrarfcnDL: 632000,    // n78 band (3.5 GHz)
      nrarfcnUL: 632000,
      frequencyBand: 78,
      channelBandwidth: 100, // MHz
      ssbSubCarrierSpacing: 30, // kHz
      
      // SSB (Synchronization Signal Block) parameters
      ssbFrequency: 3500000, // kHz
      ssbPeriodicity: 20,    // ms
      ssbPattern: 'Case C',
      maxSSBperBurst: 8,
      
      // Cell parameters
      cellBarred: 'notBarred',
      intraFreqReselection: 'allowed',
      qQualMin: -43,
      qRxLevMin: -70,
      
      // Handover parameters
      hysteresisA3: 30,      // 3.0 dB
      thresholdA3: 30,       // 3.0 dB
      timeToTriggerA3: 100,  // 100 ms
      
      // Power parameters
      ssbReferenceSignalPower: 0, // dBm (SSB-RSRP)
      pdschPowerAllocation: 0,    // dB
      puschPowerControl: 0,       // dB
      
      // Beam management
      maxBeams: 64,
      beamSweeping: true,
      csiReporting: 'periodic',
      
      // Physical layer
      physicalCellId: 0,     // 0-1007
      rootSequenceIndex: 0,
      numerology: 1,         // μ=1 (30kHz SCS)
      
      // Scheduling and QoS
      schedulingAlgorithm: 'proportionalFair',
      maxMimoLayers: 4,
      qci5G: [1, 2, 3, 4, 5, 6, 7, 8, 9, 80, 82, 83, 84, 85],
      
      // Network slicing (SA mode)
      networkSlicing: false,
      sliceTypes: [
        { sst: 1, sd: '000001' }, // eMBB
        { sst: 2, sd: '000002' }, // URLLC
        { sst: 3, sd: '000003' }  // mMTC
      ],
      
      // Dual Connectivity (NSA mode)
      dualConnectivity: false,
      enDcCapable: false,
      
      // Coverage and capacity
      cellRange: 5000,       // meters
      maxUEs: 200,           // per cell
      
      // Advanced features
      carrierAggregation: true,
      coordinatedMultipoint: false,
      deviceToDevice: false,
      vehicularCommunication: false,
      
      // Energy efficiency
      energySaving: {
        cellSleepMode: false,
        adaptiveBeamForming: true,
        dynamicSpectrumSharing: false
      }
    };
  }

  /**
   * Get default NR cell parameters
   */
  private getDefaultNRCellParameters(sector: number, band: string): Record<string, any> {
    return {
      // Physical Cell Identity (expanded range for NR)
      pci: sector - 1,       // 0-1007 for NR
      
      // Antenna parameters (advanced for 5G)
      antennaAzimuth: (sector - 1) * 120,
      antennaTilt: 10,       // More aggressive tilt for 5G
      mechanicalTilt: 5,
      antennaGain: 20,       // Higher gain antennas
      antennaType: 'massive_mimo',
      antennaElements: 64,   // Massive MIMO
      
      // Beam parameters
      beamWidth: 65,         // degrees
      beamTilt: 10,
      beamCount: 8,
      beamWeight: [1, 0.8, 0.6, 0.4, 0.4, 0.6, 0.8, 1],
      
      // Power parameters
      maxTransmissionPower: this.getMaxPowerForBand(band),
      ssbPower: 30,          // dBm
      pdschPowerOffset: 0,
      puschPowerOffset: 0,
      
      // Numerology and frame structure
      numerology: this.getNumerologyForBand(band),
      cyclicPrefix: 'normal',
      frameStructure: 'TDD',
      tddPattern: {
        periodicity: '5ms',
        dlSlots: 7,
        ulSlots: 2,
        specialSlot: 1
      },
      
      // Scheduling parameters
      schedulingAlgorithm: 'proportionalFair',
      resourceBlockGroups: true,
      codewordSwapping: false,
      
      // CSI-RS (Channel State Information Reference Signal)
      csiRsConfig: {
        periodicity: 20,     // ms
        density: 3,          // ports per RB
        powerOffset: 0       // dB
      },
      
      // DMRS (Demodulation Reference Signal)
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
          numRB: this.getResourceBlocksForBandwidth(this.getBandwidthForBand(band)),
          numerology: this.getNumerologyForBand(band),
          cyclicPrefix: 'normal'
        }
      ],
      
      // Load balancing and admission control
      loadBalancingWeight: 100,
      admissionThreshold: 85,
      congestionControl: true,
      
      // Coverage and mobility
      cellRange: this.getCellRangeForBand(band),
      maxUEs: 150,
      handoverOptimization: true,
      
      // Band-specific parameters
      bandSpecific: this.getBandSpecificParameters(band)
    };
  }

  /**
   * Get template configuration for different 5G deployment types
   */
  private getTemplateConfiguration(template: string): Partial<NodeConfiguration> {
    const templates = {
      macro5g: {
        parameters: {
          maxTransmissionPower: 46,   // dBm
          cellRange: 15000,           // 15km
          antennaHeight: 40,          // 40m
          antennaTilt: 8,
          antennaElements: 64,        // Massive MIMO
          beamCount: 16,
          frequencyBand: 78,          // n78 (3.5 GHz)
          channelBandwidth: 100       // MHz
        }
      },
      micro5g: {
        parameters: {
          maxTransmissionPower: 38,   // dBm
          cellRange: 3000,            // 3km
          antennaHeight: 20,          // 20m
          antennaTilt: 12,
          antennaElements: 32,
          beamCount: 8,
          frequencyBand: 78,
          channelBandwidth: 80        // MHz
        }
      },
      indoor5g: {
        parameters: {
          maxTransmissionPower: 24,   // dBm
          cellRange: 200,             // 200m
          antennaHeight: 6,           // 6m
          antennaTilt: 0,
          antennaElements: 16,
          beamCount: 4,
          frequencyBand: 78,
          channelBandwidth: 40,       // MHz
          deploymentType: 'indoor',
          distributedAntenna: true
        }
      },
      mmwave: {
        parameters: {
          maxTransmissionPower: 35,   // dBm
          cellRange: 500,             // 500m
          antennaHeight: 15,          // 15m
          antennaTilt: 15,
          antennaElements: 256,       // High gain for mmWave
          beamCount: 64,              // High beam count
          frequencyBand: 260,         // n260 (39 GHz)
          channelBandwidth: 400,      // MHz
          beamTracking: true,
          blockageHandling: true,
          ultraLowLatency: true
        }
      }
    };
    
    return templates[template as keyof typeof templates] || {};
  }

  /**
   * Get frequency for NR band and sector
   */
  private getFrequencyForBand(band: string, sector: number): number {
    const frequencies = {
      n78: 3500 + ((sector - 1) * 10),    // 3.5 GHz band
      n28: 700 + ((sector - 1) * 5),      // 700 MHz band
      n1: 2100 + ((sector - 1) * 10),     // 2.1 GHz band
      n3: 1800 + ((sector - 1) * 10)      // 1.8 GHz band
    };
    
    return frequencies[band as keyof typeof frequencies] || 3500;
  }

  /**
   * Get bandwidth for NR band
   */
  private getBandwidthForBand(band: string): number {
    const bandwidths = {
      n78: 100,   // MHz
      n28: 20,    // MHz
      n1: 20,     // MHz
      n3: 20      // MHz
    };
    
    return bandwidths[band as keyof typeof bandwidths] || 100;
  }

  /**
   * Get maximum power for band
   */
  private getMaxPowerForBand(band: string): number {
    const powers = {
      n78: 46,    // dBm (3.5 GHz)
      n28: 46,    // dBm (700 MHz)
      n1: 43,     // dBm (2.1 GHz)
      n3: 43,     // dBm (1.8 GHz)
      n260: 35    // dBm (mmWave)
    };
    
    return powers[band as keyof typeof powers] || 46;
  }

  /**
   * Get numerology for band
   */
  private getNumerologyForBand(band: string): number {
    const numerologies = {
      n78: 1,     // μ=1 (30 kHz SCS)
      n28: 0,     // μ=0 (15 kHz SCS)
      n1: 0,      // μ=0 (15 kHz SCS)
      n3: 0,      // μ=0 (15 kHz SCS)
      n260: 3     // μ=3 (120 kHz SCS for mmWave)
    };
    
    return numerologies[band as keyof typeof numerologies] || 1;
  }

  /**
   * Get resource blocks for bandwidth
   */
  private getResourceBlocksForBandwidth(bandwidthMHz: number): number {
    const rbMap = {
      5: 25,
      10: 52,
      15: 79,
      20: 106,
      25: 133,
      30: 160,
      40: 216,
      50: 270,
      60: 324,
      70: 378,
      80: 432,
      90: 486,
      100: 540,
      200: 1080,
      400: 2160
    };
    
    return rbMap[bandwidthMHz as keyof typeof rbMap] || 540;
  }

  /**
   * Get cell range for band
   */
  private getCellRangeForBand(band: string): number {
    const ranges = {
      n78: 5000,   // 5km
      n28: 15000,  // 15km (low band, better propagation)
      n1: 8000,    // 8km
      n3: 8000,    // 8km
      n260: 500    // 500m (mmWave)
    };
    
    return ranges[band as keyof typeof ranges] || 5000;
  }

  /**
   * Get band-specific parameters
   */
  private getBandSpecificParameters(band: string): Record<string, any> {
    const bandParams = {
      n78: {
        propagationModel: 'UMa',
        interferenceHandling: 'advanced',
        beamforming: 'massive_mimo'
      },
      n28: {
        propagationModel: 'RMa',
        interferenceHandling: 'basic',
        beamforming: 'traditional'
      },
      n260: {
        propagationModel: 'UMi',
        interferenceHandling: 'ultra_advanced',
        beamforming: 'hybrid',
        blockageHandling: true,
        beamTracking: 'fast'
      }
    };
    
    return bandParams[band as keyof typeof bandParams] || bandParams.n78;
  }

  /**
   * Validate gNodeB specific configuration
   */
  private async validateGNodeBConfiguration(config: NodeConfiguration): Promise<void> {
    if (config.nodeType !== NodeType.GNODEB) {
      throw new Error('Configuration is not for gNodeB');
    }

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

    // Validate PCI (expanded range for NR)
    if (params.physicalCellId !== undefined) {
      const pci = Number(params.physicalCellId);
      if (isNaN(pci) || pci < 0 || pci > 1007) {
        throw new Error('Physical Cell ID must be between 0 and 1007 for NR');
      }
    }

    // Validate channel bandwidth
    if (params.channelBandwidth !== undefined) {
      const validBW = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400];
      if (!validBW.includes(params.channelBandwidth)) {
        throw new Error(`Invalid channel bandwidth. Must be one of: ${validBW.join(', ')} MHz`);
      }
    }

    this.logger.debug('gNodeB configuration validated', { nodeId: config.nodeId });
  }
}