import { NodeConfiguration, NodeType, Parameter, CMEditCommand, CMEditResponse } from '../types';
import { ConfigurationManager } from './ConfigurationManager';
import { CMEditClient } from './CMEditClient';
import { ParameterValidator } from './ParameterValidator';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';

/**
 * Base class for all RAN nodes providing common functionality
 * Implements the Template Method pattern with hooks for subclasses
 */
export abstract class RANNode {
  protected logger: Logger;
  protected configManager: ConfigurationManager;
  protected cmEditClient: CMEditClient;
  protected validator: ParameterValidator;

  constructor(
    protected config: NodeConfiguration,
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
      defaultMeta: { nodeId: this.config.nodeId, nodeType: this.config.nodeType },
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
    this.validator = new ParameterValidator();
  }

  /**
   * Get node configuration
   */
  public getConfiguration(): NodeConfiguration {
    return { ...this.config };
  }

  /**
   * Update node configuration
   */
  public async updateConfiguration(updates: Partial<NodeConfiguration>): Promise<void> {
    this.logger.info('Updating node configuration', { updates });
    
    const newConfig = { ...this.config, ...updates };
    await this.validateConfiguration(newConfig);
    
    this.config = newConfig;
    await this.configManager.saveConfiguration(this.config.nodeId, this.config);
    
    this.logger.info('Node configuration updated successfully');
  }

  /**
   * Set parameter value with validation
   */
  public async setParameter(parameterName: string, value: any): Promise<void> {
    this.logger.info('Setting parameter', { parameterName, value });
    
    const parameter: Parameter = {
      name: parameterName,
      type: this.getParameterType(parameterName),
      value
    };

    await this.validator.validateParameter(parameter);
    
    // Update in configuration
    this.config.parameters[parameterName] = value;
    
    // Apply to network element via CMEDIT
    await this.applyParameterToNetwork(parameter);
    
    this.logger.info('Parameter set successfully', { parameterName, value });
  }

  /**
   * Get parameter value
   */
  public getParameter(parameterName: string): any {
    return this.config.parameters[parameterName];
  }

  /**
   * Get all parameters
   */
  public getParameters(): Record<string, any> {
    return { ...this.config.parameters };
  }

  /**
   * Execute CMEDIT command on this node
   */
  public async executeCMEditCommand(command: CMEditCommand): Promise<CMEditResponse> {
    this.logger.info('Executing CMEDIT command', { command });
    
    // Add node context to command
    const nodeCommand: CMEditCommand = {
      ...command,
      mo: this.buildManagedObjectPath(command.mo)
    };

    const response = await this.cmEditClient.execute(nodeCommand);
    
    this.logger.info('CMEDIT command executed', { commandId: response.commandId, success: response.success });
    
    return response;
  }

  /**
   * Validate node configuration
   * Template method - can be overridden by subclasses
   */
  protected async validateConfiguration(config: NodeConfiguration): Promise<void> {
    if (!config.nodeId || !config.nodeName) {
      throw new Error('Node ID and name are required');
    }

    if (!Object.values(NodeType).includes(config.nodeType)) {
      throw new Error(`Invalid node type: ${config.nodeType}`);
    }

    // Allow subclasses to add specific validation
    await this.validateNodeSpecificConfiguration(config);
  }

  /**
   * Hook for subclasses to add specific validation
   */
  protected async validateNodeSpecificConfiguration(config: NodeConfiguration): Promise<void> {
    // Default implementation - can be overridden
  }

  /**
   * Apply parameter to network element
   */
  protected async applyParameterToNetwork(parameter: Parameter): Promise<void> {
    const command: CMEditCommand = {
      operation: 'update',
      mo: this.getParameterManagedObject(parameter.name),
      attributes: {
        [parameter.name]: parameter.value
      }
    };

    await this.executeCMEditCommand(command);
  }

  /**
   * Build managed object path for this node
   */
  protected abstract buildManagedObjectPath(mo: string): string;

  /**
   * Get parameter type - to be implemented by subclasses
   */
  protected abstract getParameterType(parameterName: string): any;

  /**
   * Get managed object for parameter - to be implemented by subclasses
   */
  protected abstract getParameterManagedObject(parameterName: string): string;

  /**
   * Get node type
   */
  public getNodeType(): NodeType {
    return this.config.nodeType;
  }

  /**
   * Get node ID
   */
  public getNodeId(): string {
    return this.config.nodeId;
  }

  /**
   * Get node name
   */
  public getNodeName(): string {
    return this.config.nodeName;
  }

  /**
   * Check if node is healthy
   */
  public async isHealthy(): Promise<boolean> {
    try {
      // Basic connectivity check
      const response = await this.executeCMEditCommand({
        operation: 'get',
        mo: 'ManagedElement'
      });
      
      return response.success;
    } catch (error) {
      this.logger.error('Health check failed', { error });
      return false;
    }
  }

  /**
   * Cleanup resources
   */
  public async dispose(): Promise<void> {
    this.logger.info('Disposing node resources');
    
    // Close connections, cleanup resources
    await this.cmEditClient.disconnect();
    
    this.logger.info('Node disposed successfully');
  }
}