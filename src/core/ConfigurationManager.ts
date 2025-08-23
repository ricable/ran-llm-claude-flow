import { NodeConfiguration, CellConfiguration } from '../types';
import { promises as fs } from 'fs';
import * as path from 'path';
import * as yaml from 'yaml';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';

/**
 * Centralized configuration management with persistence
 * Supports JSON and YAML formats with validation
 */
export class ConfigurationManager {
  private logger: Logger;
  private configCache: Map<string, NodeConfiguration | CellConfiguration> = new Map();
  private configDir: string;
  private autoSave: boolean;

  constructor(configDir: string = './config', autoSave: boolean = true) {
    this.configDir = configDir;
    this.autoSave = autoSave;
    
    this.logger = createLogger({
      level: 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json()
      ),
      defaultMeta: { service: 'ConfigurationManager' },
      transports: [
        new transports.Console({
          format: format.combine(
            format.colorize(),
            format.simple()
          )
        })
      ]
    });

    this.ensureConfigDirectory();
  }

  /**
   * Save node configuration
   */
  public async saveConfiguration(
    id: string, 
    config: NodeConfiguration | CellConfiguration,
    format: 'json' | 'yaml' = 'json'
  ): Promise<void> {
    this.logger.info('Saving configuration', { id, format });
    
    try {
      // Update cache
      this.configCache.set(id, config);
      
      if (this.autoSave) {
        const fileName = `${id}.${format}`;
        const filePath = path.join(this.configDir, fileName);
        
        let content: string;
        if (format === 'yaml') {
          content = yaml.stringify(config);
        } else {
          content = JSON.stringify(config, null, 2);
        }
        
        await fs.writeFile(filePath, content, 'utf8');
        this.logger.info('Configuration saved to file', { filePath });
      }
      
      this.logger.info('Configuration saved successfully', { id });
    } catch (error) {
      this.logger.error('Failed to save configuration', { id, error });
      throw new Error(`Failed to save configuration for ${id}: ${error}`);
    }
  }

  /**
   * Load node configuration
   */
  public async loadConfiguration(
    id: string,
    format: 'json' | 'yaml' = 'json'
  ): Promise<NodeConfiguration | CellConfiguration> {
    this.logger.info('Loading configuration', { id, format });
    
    try {
      // Check cache first
      if (this.configCache.has(id)) {
        this.logger.info('Configuration loaded from cache', { id });
        return this.configCache.get(id)!;
      }
      
      // Load from file
      const fileName = `${id}.${format}`;
      const filePath = path.join(this.configDir, fileName);
      
      const content = await fs.readFile(filePath, 'utf8');
      
      let config: NodeConfiguration | CellConfiguration;
      if (format === 'yaml') {
        config = yaml.parse(content);
      } else {
        config = JSON.parse(content);
      }
      
      // Validate configuration
      await this.validateConfiguration(config);
      
      // Cache it
      this.configCache.set(id, config);
      
      this.logger.info('Configuration loaded successfully', { id });
      return config;
      
    } catch (error) {
      this.logger.error('Failed to load configuration', { id, error });
      throw new Error(`Failed to load configuration for ${id}: ${error}`);
    }
  }

  /**
   * Get configuration from cache
   */
  public getConfiguration(id: string): NodeConfiguration | CellConfiguration | undefined {
    return this.configCache.get(id);
  }

  /**
   * Update configuration
   */
  public async updateConfiguration(
    id: string,
    updates: Partial<NodeConfiguration | CellConfiguration>
  ): Promise<void> {
    this.logger.info('Updating configuration', { id, updates });
    
    const existing = this.configCache.get(id);
    if (!existing) {
      throw new Error(`Configuration not found for ${id}`);
    }
    
    const updated = { ...existing, ...updates };
    await this.validateConfiguration(updated);
    await this.saveConfiguration(id, updated);
    
    this.logger.info('Configuration updated successfully', { id });
  }

  /**
   * Delete configuration
   */
  public async deleteConfiguration(id: string): Promise<void> {
    this.logger.info('Deleting configuration', { id });
    
    try {
      // Remove from cache
      this.configCache.delete(id);
      
      // Remove files
      const jsonFile = path.join(this.configDir, `${id}.json`);
      const yamlFile = path.join(this.configDir, `${id}.yaml`);
      
      try {
        await fs.unlink(jsonFile);
      } catch (error) {
        // File might not exist
      }
      
      try {
        await fs.unlink(yamlFile);
      } catch (error) {
        // File might not exist
      }
      
      this.logger.info('Configuration deleted successfully', { id });
    } catch (error) {
      this.logger.error('Failed to delete configuration', { id, error });
      throw new Error(`Failed to delete configuration for ${id}: ${error}`);
    }
  }

  /**
   * List all configuration IDs
   */
  public async listConfigurations(): Promise<string[]> {
    try {
      const files = await fs.readdir(this.configDir);
      const configIds = new Set<string>();
      
      for (const file of files) {
        if (file.endsWith('.json') || file.endsWith('.yaml')) {
          const id = path.parse(file).name;
          configIds.add(id);
        }
      }
      
      return Array.from(configIds);
    } catch (error) {
      this.logger.error('Failed to list configurations', { error });
      throw new Error(`Failed to list configurations: ${error}`);
    }
  }

  /**
   * Export all configurations
   */
  public async exportConfigurations(
    outputPath: string,
    format: 'json' | 'yaml' = 'json'
  ): Promise<void> {
    this.logger.info('Exporting configurations', { outputPath, format });
    
    try {
      const configIds = await this.listConfigurations();
      const allConfigs: Record<string, any> = {};
      
      for (const id of configIds) {
        const config = await this.loadConfiguration(id);
        allConfigs[id] = config;
      }
      
      let content: string;
      if (format === 'yaml') {
        content = yaml.stringify(allConfigs);
      } else {
        content = JSON.stringify(allConfigs, null, 2);
      }
      
      await fs.writeFile(outputPath, content, 'utf8');
      
      this.logger.info('Configurations exported successfully', { 
        outputPath, 
        count: configIds.length 
      });
    } catch (error) {
      this.logger.error('Failed to export configurations', { outputPath, error });
      throw new Error(`Failed to export configurations: ${error}`);
    }
  }

  /**
   * Import configurations from file
   */
  public async importConfigurations(
    inputPath: string,
    format: 'json' | 'yaml' = 'json'
  ): Promise<void> {
    this.logger.info('Importing configurations', { inputPath, format });
    
    try {
      const content = await fs.readFile(inputPath, 'utf8');
      
      let allConfigs: Record<string, any>;
      if (format === 'yaml') {
        allConfigs = yaml.parse(content);
      } else {
        allConfigs = JSON.parse(content);
      }
      
      let importCount = 0;
      for (const [id, config] of Object.entries(allConfigs)) {
        await this.validateConfiguration(config as any);
        await this.saveConfiguration(id, config as any);
        importCount++;
      }
      
      this.logger.info('Configurations imported successfully', { 
        inputPath, 
        count: importCount 
      });
    } catch (error) {
      this.logger.error('Failed to import configurations', { inputPath, error });
      throw new Error(`Failed to import configurations: ${error}`);
    }
  }

  /**
   * Clear all cached configurations
   */
  public clearCache(): void {
    this.logger.info('Clearing configuration cache');
    this.configCache.clear();
  }

  /**
   * Get cache statistics
   */
  public getCacheStats(): { size: number; keys: string[] } {
    return {
      size: this.configCache.size,
      keys: Array.from(this.configCache.keys())
    };
  }

  /**
   * Validate configuration object
   */
  private async validateConfiguration(config: any): Promise<void> {
    if (!config) {
      throw new Error('Configuration cannot be null or undefined');
    }
    
    if (typeof config !== 'object') {
      throw new Error('Configuration must be an object');
    }
    
    // Basic validation for required fields
    if ('nodeId' in config && 'nodeType' in config) {
      // Node configuration
      const nodeConfig = config as NodeConfiguration;
      if (!nodeConfig.nodeId || !nodeConfig.nodeName) {
        throw new Error('Node configuration must have nodeId and nodeName');
      }
    } else if ('cellId' in config && 'cellType' in config) {
      // Cell configuration
      const cellConfig = config as CellConfiguration;
      if (!cellConfig.cellId || !cellConfig.cellName) {
        throw new Error('Cell configuration must have cellId and cellName');
      }
    } else {
      throw new Error('Unknown configuration type');
    }
  }

  /**
   * Ensure configuration directory exists
   */
  private async ensureConfigDirectory(): Promise<void> {
    try {
      await fs.access(this.configDir);
    } catch (error) {
      this.logger.info('Creating configuration directory', { configDir: this.configDir });
      await fs.mkdir(this.configDir, { recursive: true });
    }
  }
}