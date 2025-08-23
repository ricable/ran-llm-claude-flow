import { RANNode } from '../core/RANNode';
import { ENodeB } from '../nodes/ENodeB';
import { GNodeB } from '../nodes/GNodeB';
import { NodeConfiguration, NodeType } from '../types';
import { ConfigurationManager } from '../core/ConfigurationManager';
import { CMEditClient } from '../core/CMEditClient';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';

/**
 * Main factory for creating RAN nodes using the Factory Method pattern
 * Provides centralized node creation with proper dependency injection
 */
export class RANNodeFactory {
  private logger: Logger;
  private configManager: ConfigurationManager;
  private cmEditClient: CMEditClient;
  private nodeCache: Map<string, RANNode> = new Map();

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
      defaultMeta: { service: 'RANNodeFactory' },
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
  }

  /**
   * Create a RAN node instance based on configuration
   */
  public async createNode(config: NodeConfiguration): Promise<RANNode> {
    this.logger.info('Creating RAN node', { 
      nodeId: config.nodeId, 
      nodeType: config.nodeType 
    });

    // Check cache first
    if (this.nodeCache.has(config.nodeId)) {
      this.logger.info('Returning cached node instance', { nodeId: config.nodeId });
      return this.nodeCache.get(config.nodeId)!;
    }

    // Validate configuration
    await this.validateNodeConfiguration(config);

    // Create node based on type
    let node: RANNode;
    
    switch (config.nodeType) {
      case NodeType.ENODEB:
        node = new ENodeB(config, this.configManager, this.cmEditClient);
        break;
        
      case NodeType.GNODEB:
        node = new GNodeB(config, this.configManager, this.cmEditClient);
        break;
        
      default:
        throw new Error(`Unsupported node type: ${config.nodeType}`);
    }

    // Cache the node
    this.nodeCache.set(config.nodeId, node);
    
    // Save configuration
    await this.configManager.saveConfiguration(config.nodeId, config);

    this.logger.info('RAN node created successfully', { 
      nodeId: config.nodeId, 
      nodeType: config.nodeType 
    });

    return node;
  }

  /**
   * Create node from saved configuration
   */
  public async createNodeFromConfig(nodeId: string): Promise<RANNode> {
    this.logger.info('Creating node from saved configuration', { nodeId });

    // Check cache first
    if (this.nodeCache.has(nodeId)) {
      return this.nodeCache.get(nodeId)!;
    }

    // Load configuration
    const config = await this.configManager.loadConfiguration(nodeId) as NodeConfiguration;
    
    return this.createNode(config);
  }

  /**
   * Create multiple nodes from configurations
   */
  public async createNodes(configs: NodeConfiguration[]): Promise<RANNode[]> {
    this.logger.info('Creating multiple RAN nodes', { count: configs.length });

    const nodes: RANNode[] = [];
    const errors: Array<{ nodeId: string; error: string }> = [];

    for (const config of configs) {
      try {
        const node = await this.createNode(config);
        nodes.push(node);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        errors.push({ nodeId: config.nodeId, error: errorMessage });
        this.logger.error('Failed to create node', { 
          nodeId: config.nodeId, 
          error: errorMessage 
        });
      }
    }

    if (errors.length > 0) {
      this.logger.warn('Some nodes failed to create', { 
        successful: nodes.length,
        failed: errors.length,
        errors
      });
    }

    this.logger.info('Node creation completed', { 
      total: configs.length,
      successful: nodes.length,
      failed: errors.length
    });

    return nodes;
  }

  /**
   * Get cached node instance
   */
  public getCachedNode(nodeId: string): RANNode | undefined {
    return this.nodeCache.get(nodeId);
  }

  /**
   * Remove node from cache
   */
  public removeCachedNode(nodeId: string): boolean {
    const removed = this.nodeCache.delete(nodeId);
    if (removed) {
      this.logger.info('Node removed from cache', { nodeId });
    }
    return removed;
  }

  /**
   * Clear all cached nodes
   */
  public clearCache(): void {
    this.logger.info('Clearing node cache', { count: this.nodeCache.size });
    this.nodeCache.clear();
  }

  /**
   * Get cache statistics
   */
  public getCacheStats(): { size: number; nodeIds: string[] } {
    return {
      size: this.nodeCache.size,
      nodeIds: Array.from(this.nodeCache.keys())
    };
  }

  /**
   * Create node template for specific type
   */
  public createNodeTemplate(nodeType: NodeType, overrides: Partial<NodeConfiguration> = {}): NodeConfiguration {
    const baseTemplate = this.getBaseTemplate();
    const typeTemplate = this.getTypeSpecificTemplate(nodeType);
    
    return {
      ...baseTemplate,
      ...typeTemplate,
      ...overrides,
      nodeType
    };
  }

  /**
   * Validate node configurations in bulk
   */
  public async validateConfigurations(configs: NodeConfiguration[]): Promise<{
    valid: NodeConfiguration[];
    invalid: Array<{ config: NodeConfiguration; errors: string[] }>;
  }> {
    const valid: NodeConfiguration[] = [];
    const invalid: Array<{ config: NodeConfiguration; errors: string[] }> = [];

    for (const config of configs) {
      try {
        await this.validateNodeConfiguration(config);
        valid.push(config);
      } catch (error) {
        const errors = error instanceof Error ? [error.message] : ['Validation failed'];
        invalid.push({ config, errors });
      }
    }

    return { valid, invalid };
  }

  /**
   * Dispose of factory and cleanup resources
   */
  public async dispose(): Promise<void> {
    this.logger.info('Disposing RANNodeFactory');

    // Dispose all cached nodes
    for (const [nodeId, node] of this.nodeCache) {
      try {
        await node.dispose();
      } catch (error) {
        this.logger.error('Error disposing node', { nodeId, error });
      }
    }

    this.clearCache();
    await this.cmEditClient.disconnect();
    
    this.logger.info('RANNodeFactory disposed');
  }

  /**
   * Validate node configuration
   */
  private async validateNodeConfiguration(config: NodeConfiguration): Promise<void> {
    if (!config.nodeId || !config.nodeName) {
      throw new Error('Node ID and name are required');
    }

    if (!Object.values(NodeType).includes(config.nodeType)) {
      throw new Error(`Invalid node type: ${config.nodeType}`);
    }

    if (!config.siteId) {
      throw new Error('Site ID is required');
    }

    // Validate coordinates if provided
    if (config.coordinates) {
      const { latitude, longitude } = config.coordinates;
      if (latitude < -90 || latitude > 90) {
        throw new Error('Latitude must be between -90 and 90');
      }
      if (longitude < -180 || longitude > 180) {
        throw new Error('Longitude must be between -180 and 180');
      }
    }

    // Type-specific validation
    switch (config.nodeType) {
      case NodeType.ENODEB:
        await this.validateENodeBConfiguration(config);
        break;
      case NodeType.GNODEB:
        await this.validateGNodeBConfiguration(config);
        break;
    }
  }

  /**
   * Validate eNodeB specific configuration
   */
  private async validateENodeBConfiguration(config: NodeConfiguration): Promise<void> {
    // eNodeB specific validation
    if (config.parameters.eNodeBId !== undefined) {
      const eNodeBId = Number(config.parameters.eNodeBId);
      if (isNaN(eNodeBId) || eNodeBId < 0 || eNodeBId > 1048575) {
        throw new Error('eNodeBId must be between 0 and 1048575');
      }
    }
  }

  /**
   * Validate gNodeB specific configuration
   */
  private async validateGNodeBConfiguration(config: NodeConfiguration): Promise<void> {
    // gNodeB specific validation
    if (config.parameters.gNBId !== undefined) {
      const gNBId = Number(config.parameters.gNBId);
      if (isNaN(gNBId) || gNBId < 0 || gNBId > 4294967295) {
        throw new Error('gNBId must be between 0 and 4294967295');
      }
    }
  }

  /**
   * Get base configuration template
   */
  private getBaseTemplate(): Partial<NodeConfiguration> {
    return {
      nodeId: '',
      nodeName: '',
      siteId: '',
      parameters: {}
    };
  }

  /**
   * Get type-specific configuration template
   */
  private getTypeSpecificTemplate(nodeType: NodeType): Partial<NodeConfiguration> {
    switch (nodeType) {
      case NodeType.ENODEB:
        return {
          parameters: {
            eNodeBId: 0,
            tac: 1,
            earfcnDL: 1950,
            earfcnUL: 19950,
            cellBarred: 'notBarred',
            intraFreqReselection: 'allowed'
          }
        };
        
      case NodeType.GNODEB:
        return {
          parameters: {
            gNBId: 0,
            tac: 1,
            nrarfcnDL: 632000,
            nrarfcnUL: 632000,
            cellBarred: 'notBarred',
            ssbSubCarrierSpacing: 30
          }
        };
        
      default:
        return {};
    }
  }
}